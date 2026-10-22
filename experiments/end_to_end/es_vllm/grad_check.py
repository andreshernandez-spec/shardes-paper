#!/usr/bin/env python
"""How much of an ES gradient estimate on Tulu is signal? One iteration's members, many prompts.

    python -m es_vllm.grad_check --out runs/grad-check/start.json          # on the GPU
    python -m es_vllm.grad_check --analyze runs/grad-check/start.json      # anywhere
    python -m es_vllm.grad_check --out ... --smoke                         # laptop wiring
    python -m es_vllm.grad_check --sampled --out runs/grad-check/start-sampled.json
    python -m es_vllm.grad_check --analyze runs/grad-check/start-sampled.json \
        --greedy runs/grad-check/start.json

An ES update is the sum of the members' noise weighted by their z-scored fitness, so for a
given seed the estimate is fully determined by those N weights. Whether it points
anywhere depends on whether the members' ranking is a property of the weights or of the
prompts that happened to be drawn. This decodes the start model and iteration 0's 16
members (seed 0, the longer arm's own) on the 768 prompts of RL steps 1 to 16, with
per-prompt rewards, at several sigmas along the same noise directions, and the mirrored
members (-eps) at the run's sigma. At sigma 5e-4 the members' means on the first 192
prompts must equal the arm's logged iteration-0 fitness.

`--analyze` reports, per sigma: the members' mean change from the start, the share of
member-prompt pairs whose reward differs from the start's, the variance of the members'
true effects against the prompt-level noise (two-way decomposition of reward minus the
start's reward), the reliability of the ranking at 192, 384 and 768 prompts per member
predicted from it and measured by splitting the prompts, the same for mirrored
differences, and the first-order gain per iteration that reliability implies.

`--sampled` asks the same of fitness measured by sampling, as the RL run does (temperature
1.0, 8 samples per prompt, on the arm's 192 iteration-0 prompts, sigma 5e-4): once with
common random numbers (every member samples prompt j with the same seed) and once with
independent seeds. Its analysis gives the ranking's reliability at 96, 192 and 384
rollouts per member for each way of splitting them into prompts x samples, beside greedy's.
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

PKG = Path(__file__).resolve().parent
E2E = PKG.parent
REPO = E2E.parent.parent
sys.path.insert(0, str(E2E))
sys.path.insert(0, str(E2E.parent))
import harness  # noqa: E402
import releases as R  # noqa: E402
from provenance import env_block  # noqa: E402

N, P_RUN, SEED, ALPHA = 16, 192, 0, 5e-4
SIGMAS = (5e-4, 1e-3, 2e-3)
MIRRORED = 5e-4
BATCHES = 4  # four iterations' prompts: RL steps 1 to 16


def collect(args) -> dict:
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from es_vllm.run_tulu import Verifier, load_prompts, render  # noqa: PLC0415
    from es_vllm.worker import ESWorker  # noqa: PLC0415

    def set_sigma(self, s):
        self._es_sigma = float(s)
    ESWorker.gc_set_sigma = set_sigma

    if args.smoke:
        repo, revision, n, sigmas, mirrored, batches, cap = (
            "Qwen/Qwen2.5-0.5B-Instruct", None, 4, (5e-4, 1e-3), 5e-4, 1, 64)
    else:
        repo, revision, n, sigmas, mirrored, batches, cap = (
            R.TULU31_START.repo, R.TULU31_START.commit, N, SIGMAS, MIRRORED, BATCHES, 2048)
    if args.sampled:
        batches = 1  # the arm's iteration-0 prompts
    rows = [r for b in load_prompts(E2E / "data" / "tulu31", batches, P_RUN) for r in b]
    if args.smoke:
        rows = rows[:16]
    llm = LLM(model=repo, revision=revision, dtype="bfloat16", seed=0,
              gpu_memory_utilization=0.4 if args.smoke else 0.5, max_model_len=4096,
              enable_prefix_caching=False, worker_extension_cls="es_vllm.worker.ESWorker")
    model_dir = snapshot_download(repo, revision=revision,
                                  allow_patterns=["*.safetensors", "*.json"])
    llm.collective_rpc("es_init", args=(model_dir, n, sigmas[0], ALPHA * sigmas[0], SEED))
    llm.collective_rpc("es_ask")  # generation 0: the arm's iteration-0 members
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    ids = [render(tok, r) for r in rows]
    greedy = SamplingParams(temperature=0.0, max_tokens=cap)
    verifier = Verifier(REPO / ".venv-verify" / "bin" / "python")

    def decode():
        # in the run's chunks of 192, one generate call each, as the arm decodes them:
        # vLLM's greedy outputs are not promised to be independent of the batch
        outs = []
        for k in range(0, len(ids), P_RUN):
            outs += llm.generate([{"prompt_token_ids": i} for i in ids[k:k + P_RUN]], greedy,
                                 use_tqdm=False)
        rewards = verifier.score([{"text": o.outputs[0].text, "ground_truth": r["ground_truth"],
                                   "dataset": r["dataset"],
                                   "stopped": o.outputs[0].finish_reason == "stop"}
                                  for o, r in zip(outs, rows)])
        if len(rewards) != len(rows):  # the work asked for is the work done
            raise SystemExit(f"{len(rewards)} rewards for {len(rows)} prompts")
        return [float(x) for x in rewards]

    if args.sampled:
        return collect_sampled(llm, ids, rows, verifier, n, cap, sigmas[0], args.smoke)
    t0 = time.perf_counter()
    rec = {"center": decode(), "members": {}, "mirrored": {}}
    print(f"center {np.mean(rec['center']):.3f} {time.perf_counter() - t0:.0f}s", flush=True)
    plan = [(s, 1) for s in sigmas] + [(mirrored, -1)]
    for s, sign in plan:
        llm.collective_rpc("gc_set_sigma", args=(sign * s,))
        out = []
        for m in range(n):
            llm.collective_rpc("es_perturb", args=(m,))
            if m == 0:
                check = llm.collective_rpc("es_check", args=(0,))[0]
                if not check["ok"]:
                    raise SystemExit(f"engine weights are not member 0 at sigma {sign * s}")
            out.append(decode())
            print(f"sigma {sign * s:+g} member {m}: {np.mean(out[-1]):.3f} "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
        (rec["members"] if sign > 0 else rec["mirrored"])[str(s)] = out
    llm.collective_rpc("es_restore")
    if not llm.collective_rpc("es_check", args=(None,))[0]["ok"]:
        raise SystemExit("engine weights are not the start after restore")
    verifier.close()
    rec.update({"n": n, "prompts": len(rows), "datasets": [r["dataset"] for r in rows],
                "seconds": time.perf_counter() - t0})
    return rec


K_SAMPLES, TEMPERATURE = 8, 1.0  # the RL run samples at temperature 1.0


def collect_sampled(llm, ids, rows, verifier, n, cap, sigma, smoke) -> dict:
    """Iteration 0's members and the start, each prompt sampled K times at temperature 1.0,
    twice: with common random numbers (every member draws with the same seed per prompt,
    so their samples are coupled) and with independent seeds per member."""
    from vllm import SamplingParams  # noqa: PLC0415

    k = 2 if smoke else K_SAMPLES

    def decode(seed_of):
        outs = []
        for c in range(0, len(ids), P_RUN):
            chunk = range(c, min(c + P_RUN, len(ids)))
            params = [SamplingParams(n=k, temperature=TEMPERATURE, top_p=1.0, max_tokens=cap,
                                     seed=seed_of(j)) for j in chunk]
            outs += llm.generate([{"prompt_token_ids": ids[j]} for j in chunk], params,
                                 use_tqdm=False)
        items = [{"text": o.text, "ground_truth": r["ground_truth"], "dataset": r["dataset"],
                  "stopped": o.finish_reason == "stop"}
                 for out, r in zip(outs, rows) for o in out.outputs]
        rewards = verifier.score(items)
        if len(rewards) != len(rows) * k:  # the work asked for is the work done
            raise SystemExit(f"{len(rewards)} rewards for {len(rows)} prompts x {k}")
        lens = [len(o.token_ids) for out in outs for o in out.outputs]
        return [[float(x) for x in rewards[j * k:(j + 1) * k]] for j in range(len(rows))], lens

    llm.collective_rpc("gc_set_sigma", args=(sigma,))
    t0 = time.perf_counter()
    rec = {"k": k, "temperature": TEMPERATURE, "sigma": sigma}
    schemes = {"crn": lambda m: (lambda j: 10_000 + j),
               "independent": lambda m: (lambda j: 10_000 + j + 1_000_003 * (m + 2))}
    for name, seeds in schemes.items():
        llm.collective_rpc("es_restore")
        center, lens = decode(seeds(-1))
        res = {"center": center, "members": [], "mean_len": [float(np.mean(lens))]}
        print(f"{name} center {np.mean(center):.3f} len {np.mean(lens):.0f} "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        for m in range(n):
            llm.collective_rpc("es_perturb", args=(m,))
            if m == 0 and not llm.collective_rpc("es_check", args=(0,))[0]["ok"]:
                raise SystemExit("engine weights are not member 0")
            r, lens = decode(seeds(m))
            res["members"].append(r)
            res["mean_len"].append(float(np.mean(lens)))
            print(f"{name} member {m}: {np.mean(r):.3f} len {np.mean(lens):.0f} "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
        rec[name] = res
    llm.collective_rpc("es_restore")
    if not llm.collective_rpc("es_check", args=(None,))[0]["ok"]:
        raise SystemExit("engine weights are not the start after restore")
    verifier.close()
    rec.update({"n": n, "prompts": len(rows), "datasets": [r["dataset"] for r in rows],
                "seconds": time.perf_counter() - t0})
    return {"sampled": rec}


def split_allocation(r: np.ndarray, p_, k_, rng, draws=1000) -> dict:
    """Members x prompts x samples. Over random pairs of disjoint prompt subsets of size p_,
    each with k_ of the samples: the correlation of the members' means between the two
    (the ranking's reliability at p_ * k_ rollouts per member) and their covariance (an
    estimate of the variance of the members' true effects)."""
    n, p, k = r.shape
    cors, covs = [], []
    for _ in range(draws):
        perm = rng.permutation(p)
        a, b = perm[:p_], perm[p_:2 * p_]
        ma = r[:, a][:, :, rng.permutation(k)[:k_]].mean(axis=(1, 2))
        mb = r[:, b][:, :, rng.permutation(k)[:k_]].mean(axis=(1, 2))
        covs.append(np.cov(ma, mb)[0, 1])
        if ma.std() > 0 and mb.std() > 0:
            cors.append(np.corrcoef(ma, mb)[0, 1])
    return {"reliability": float(np.mean(cors)) if cors else 0.0, "cov": float(np.mean(covs))}


ALLOCATIONS = {96: [(96, 1), (48, 2), (24, 4), (12, 8)],
               192: [(96, 2), (48, 4), (24, 8)],
               384: [(96, 4), (48, 8)]}


def analyze_sampled(rec: dict, greedy: dict | None) -> dict:
    """Ranking reliability per rollout budget for each way of spending it, sampled with and
    without common random numbers, and greedy from the 768-prompt record for reference."""
    rng = np.random.default_rng(0)
    s = rec["sampled"]
    out = {"k": s["k"], "temperature": s["temperature"], "sigma": s["sigma"], "schemes": {}}
    for name in ("crn", "independent"):
        r = np.asarray(s[name]["members"])           # members x prompts x samples
        c = np.asarray(s[name]["center"])            # prompts x samples
        res = {"center_mean": float(c.mean()), "member_mean_change": float(r.mean() - c.mean()),
               "mean_len": float(np.mean(s[name]["mean_len"])),
               "v_true": split_allocation(r, r.shape[1] // 2, s["k"], rng)["cov"],
               "budgets": {}}
        for budget, allocs in ALLOCATIONS.items():
            res["budgets"][budget] = {f"{p_}x{k_}": split_allocation(r, p_, k_, rng)["reliability"]
                                      for p_, k_ in allocs if 2 * p_ <= r.shape[1] and k_ <= s["k"]}
        out["schemes"][name] = res
    if greedy is not None:
        g = np.asarray(greedy["members"]["0.0005"])[:, :, None]   # members x 768 x 1
        out["greedy"] = {"v_true": split_allocation(g, g.shape[1] // 2, 1, rng)["cov"],
                         "budgets": {b: split_allocation(g, b, 1, rng)["reliability"]
                                     for b in (96, 192, 384)}}
    return out


def variance_parts(delta: np.ndarray) -> dict:
    """Two-way decomposition (members x prompts, one observation per cell) of reward minus
    the start's reward: the variance of the members' true mean effects and the residual
    per member-prompt."""
    n, p = delta.shape
    m = delta.mean(axis=1)  # each member's mean effect
    resid = delta - m[:, None] - delta.mean(axis=0)[None, :] + delta.mean()
    v_e = float((resid ** 2).sum() / ((n - 1) * (p - 1)))
    v_m = float(max(m.var(ddof=1) - v_e / p, 0.0))
    return {"v_member": v_m, "v_resid": v_e,
            "reliability": {str(q): v_m / (v_m + v_e / q) if v_m + v_e > 0 else 0.0
                            for q in (192, 384, 768)}}


def split_reliability(delta: np.ndarray, parts: int, rng, draws=2000) -> float:
    """Mean correlation of the members' means between disjoint prompt subsets of size p/parts."""
    n, p = delta.shape
    rs = []
    for _ in range(draws // parts):
        perm = rng.permutation(p).reshape(parts, -1)
        means = [delta[:, idx].mean(axis=1) for idx in perm]
        for a in range(parts):
            for b in range(a + 1, parts):
                if means[a].std() > 0 and means[b].std() > 0:
                    rs.append(np.corrcoef(means[a], means[b])[0, 1])
    return float(np.mean(rs)) if rs else 0.0


def bootstrap(delta: np.ndarray, rng, draws=1000) -> dict:
    """90% intervals for v_member and the reliability at 192 prompts, resampling members
    and prompts (16 members make these rough)."""
    n, p = delta.shape
    vs, rel = [], []
    for _ in range(draws):
        d = delta[rng.integers(0, n, n)][:, rng.integers(0, p, p)]
        v = variance_parts(d)
        vs.append(v["v_member"])
        rel.append(v["reliability"]["192"])
    q = lambda x: [float(np.percentile(x, 5)), float(np.percentile(x, 95))]  # noqa: E731
    return {"v_member_90": q(vs), "reliability_192_90": q(rel)}


def analyze(rec: dict) -> dict:
    rng = np.random.default_rng(0)
    c = np.asarray(rec["center"])
    datasets = np.asarray(rec["datasets"])
    out = {"center_mean": float(c.mean()), "sigmas": {}}
    for s, rows in rec["members"].items():
        r = np.asarray(rows)
        delta = r - c[None, :]
        parts = variance_parts(delta)
        res = {"member_mean_change": float(delta.mean()),
               "member_sd": float(r.mean(axis=1).std(ddof=1)),
               "pairs_changed": float((delta != 0).mean()),
               "pairs_up": float((delta > 0).mean()), "pairs_down": float((delta < 0).mean()),
               "bootstrap": bootstrap(delta, rng),
               "by_source": {k: {"prompts": int((datasets == k).sum()),
                                 **variance_parts(delta[:, datasets == k]),
                                 "pairs_changed": float((delta[:, datasets == k] != 0).mean())}
                             for k in sorted(set(datasets))},
               **parts,
               "split_reliability": {"192": split_reliability(delta, 4, rng),
                                     "384": split_reliability(delta, 2, rng)},
               # first order, z-scored weights: gain per iteration ~ (alpha / sigma) *
               # sqrt(v_member * reliability at the run's 192 prompts)
               "implied_gain_per_iteration": float(
                   ALPHA / float(s) * np.sqrt(parts["v_member"] * parts["reliability"]["192"]))}
        if s in rec.get("mirrored", {}):
            minus = np.asarray(rec["mirrored"][s])
            anti = (r - minus) / 2  # the antithetic estimate of each member's directional effect
            res["mirrored"] = {**variance_parts(anti),
                               "minus_mean_change": float((minus - c[None, :]).mean()),
                               "split_reliability": {"192": split_reliability(anti, 4, rng),
                                                     "384": split_reliability(anti, 2, rng)}}
        out["sigmas"][s] = res
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analyze", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sampled", action="store_true",
                    help="temperature-1.0 samples on the first 192 prompts instead of greedy")
    ap.add_argument("--greedy", type=Path, help="with --analyze on a sampled record: the "
                    "greedy record to compare against")
    args = ap.parse_args(argv)
    if args.analyze:
        rec = json.loads((E2E / args.analyze).read_text())
        if "sampled" in rec:
            greedy = json.loads((E2E / args.greedy).read_text()) if args.greedy else None
            print(json.dumps(analyze_sampled(rec, greedy), indent=2))
        else:
            print(json.dumps(analyze(rec), indent=2))
        return 0
    if args.out is None:
        ap.error("--out or --analyze")
    out = E2E / args.out
    if out.exists():
        raise SystemExit(f"{out} exists")
    rec = collect(args)
    harness.write_atomic(out, {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "env": env_block(E2E, ["runs"], ("vllm", "torch", "jax", "jaxlib")),
        "smoke": args.smoke, "seed": SEED, "alpha": ALPHA, **rec})
    if "sampled" in rec:
        print(json.dumps(analyze_sampled(rec, None), indent=1), flush=True)
    else:
        print(json.dumps(analyze(rec)["sigmas"], indent=1)[:2000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
