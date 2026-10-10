#!/usr/bin/env python
"""The reward along RL's own direction against ES's random ones, and where ES's redraws come from.

    python -m es_vllm.direction_check --out runs/direction-check/start.json   # on the GPU
    python -m es_vllm.direction_check --analyze runs/direction-check/start.json
    python -m es_vllm.direction_check --out ... --smoke                       # laptop wiring

The gradient probe (`grad_check.py`, runs/grad-check/) found that a random ES perturbation
redraws most uncertain outcomes and moves each prompt's success probability by about 8
points, nearly independently across prompts, so its effect on the mean reward is small.
This asks the same model two further questions, on the same 768 prompts:

1. Along the RL run's displacement (theta_0 + lambda (theta_RL120 - theta_0), and the
   step-480 model itself), how does the reward change with distance, and which way do
   outcomes flip? Against ES's random directions (the arm's iteration-0 members) at the
   same distances.
2. Where do ES's redraws come from? The start's greedy decode on the first 192 prompts
   with the gap between its top two tokens' log-probabilities at every position, and,
   for four members at the run's sigma and for a step a quarter of the way along RL's
   direction, the first token where each answer departs from the start's and the gap
   the start had there.
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

N, P_RUN, SEED, SIGMA = 16, 192, 0, 5e-4
LAMBDAS = (0.25, 0.5, 1.0, 2.0)
RANDOM_MULTIPLES = (0.5, 1.0, 2.0)   # random directions at these multiples of |theta_RL120 - theta_0|
DIVERGE_MEMBERS = 4


def worker_methods():
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415

    def leaf(path, name):
        with safe_open(str(path), framework="pt", device="cuda") as st:
            t = st.get_tensor(name)
        x = jnp.asarray(jax.dlpack.from_dlpack(t)).astype(jnp.float32)
        x.block_until_ready()  # JAX has read it before torch may reuse the memory
        del t
        torch.cuda.synchronize()
        return x

    def dc_index(self, model_dir):
        index = {}
        for f in sorted(Path(model_dir).glob("*.safetensors")):
            with safe_open(str(f), framework="pt", device="cpu") as st:
                for n in st.keys():
                    index[n] = str(f)
        missing = [n for n in self._es_names if n not in index]
        if missing:
            raise SystemExit(f"{len(missing)} leaves missing, e.g. {missing[:3]}")
        norm2 = 0.0
        for n in self._es_names:
            d = leaf(index[n], n) - self._es_master[n]
            norm2 += float(jnp.sum(d * d))
            del d
        self._dc_index = index
        return float(np.sqrt(norm2))

    def dc_along(self, lam):
        def leaves():
            for n in self._es_names:
                m = self._es_master[n]
                yield n, (m + float(lam) * (leaf(self._dc_index[n], n) - m)).astype(jnp.bfloat16)
        self._load(leaves())

    def dc_sigma(self, s):
        self._es_sigma = float(s)

    def dc_count(self):
        return int(sum(v.size for v in self._es_master.values()))

    return {"dc_index": dc_index, "dc_along": dc_along, "dc_sigma": dc_sigma,
            "dc_count": dc_count}


def collect(args) -> dict:
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from es_vllm.heldout import rl_revision  # noqa: PLC0415
    from es_vllm.run_tulu import Verifier, load_prompts, render  # noqa: PLC0415
    from es_vllm.worker import ESWorker  # noqa: PLC0415

    for k, fn in worker_methods().items():
        setattr(ESWorker, k, fn)
    get = lambda repo, rev: snapshot_download(repo, revision=rev,  # noqa: E731
                                              allow_patterns=["*.safetensors", "*.json"])
    if args.smoke:  # wiring only: a base and an instruct checkpoint of one architecture
        repo, revision, cap, batches = "Qwen/Qwen2.5-0.5B-Instruct", None, 64, 1
        rl_dirs = {"120": get("Qwen/Qwen2.5-0.5B", None), "480": get("Qwen/Qwen2.5-0.5B", None)}
    else:
        repo, revision, cap, batches = R.TULU31_START.repo, R.TULU31_START.commit, 2048, 4
        rl_dirs = {t: get(R.TULU31_RL.repo, rl_revision(f"step_{t}")) for t in ("120", "480")}
    rows = [r for b in load_prompts(E2E / "data" / "tulu31", batches, P_RUN) for r in b]
    if args.smoke:
        rows = rows[:16]
    llm = LLM(model=repo, revision=revision, dtype="bfloat16", seed=0,
              gpu_memory_utilization=0.4 if args.smoke else 0.5, max_model_len=4096,
              enable_prefix_caching=False, worker_extension_cls="es_vllm.worker.ESWorker")
    rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
    model_dir = get(repo, revision)
    rpc("es_init", model_dir, N, SIGMA, 5e-4 * SIGMA, SEED)
    rpc("es_ask")  # the arm's iteration-0 members
    d = rpc("dc_count")
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    ids = [render(tok, r) for r in rows]
    verifier = Verifier(REPO / ".venv-verify" / "bin" / "python")

    def decode(rows_, ids_, logprobs=False):
        params = SamplingParams(temperature=0.0, max_tokens=cap, logprobs=2 if logprobs else None)
        outs = []
        for c in range(0, len(ids_), P_RUN):
            outs += llm.generate([{"prompt_token_ids": i} for i in ids_[c:c + P_RUN]], params,
                                 use_tqdm=False)
        rewards = verifier.score([{"text": o.outputs[0].text, "ground_truth": r["ground_truth"],
                                   "dataset": r["dataset"],
                                   "stopped": o.outputs[0].finish_reason == "stop"}
                                  for o, r in zip(outs, rows_)])
        if len(rewards) != len(rows_):  # the work asked for is the work done
            raise SystemExit(f"{len(rewards)} rewards for {len(rows_)} prompts")
        return [float(x) for x in rewards], outs

    t0 = time.perf_counter()
    rec = {"points": [], "prompts": len(rows), "datasets": [r["dataset"] for r in rows]}
    center, _ = decode(rows, ids)
    rec["center"] = center
    norms = {t: rpc("dc_index", rl_dirs[t]) for t in ("480", "120")}  # leaves the 120 index set
    rec["rl_norm"] = norms
    rec["d"] = d
    print(f"center {np.mean(center):.3f}; |theta_RL - theta_0| {norms} {time.perf_counter() - t0:.0f}s",
          flush=True)

    def point(kind, **kw):
        r, _ = decode(rows, ids)
        rec["points"].append({"kind": kind, **kw, "rewards": r})
        print(f"{kind} {kw}: {np.mean(r):.3f} {time.perf_counter() - t0:.0f}s", flush=True)

    for lam in LAMBDAS:
        rpc("dc_along", lam)
        point("rl120", lam=lam, norm=lam * norms["120"])
    rpc("dc_index", rl_dirs["480"])
    rpc("dc_along", 1.0)
    point("rl480", lam=1.0, norm=norms["480"])
    rpc("es_restore")
    # random directions: the arm's members, scaled to a given norm (sigma = norm / sqrt(d))
    for mult in RANDOM_MULTIPLES:
        for m in (0, 1):
            s = mult * norms["120"] / np.sqrt(d)
            rpc("dc_sigma", s)
            rpc("es_perturb", m)
            point("random", member=m, norm=mult * norms["120"], sigma=s)
    rpc("dc_sigma", SIGMA)
    rpc("es_restore")

    # where the redraws come from: first departure from the start's greedy answer
    rows_d, ids_d = rows[:P_RUN], ids[:P_RUN]
    _, outs = decode(rows_d, ids_d, logprobs=True)
    base_tokens = [list(o.outputs[0].token_ids) for o in outs]
    margins = []
    for o in outs:
        mg = []
        for lp in o.outputs[0].logprobs:
            top = sorted((v.logprob for v in lp.values()), reverse=True)
            mg.append(float(top[0] - top[1]) if len(top) > 1 else float("inf"))
        margins.append(mg)
    rec["divergence"] = {"center_margins": margins, "runs": []}

    def departures(kind, **kw):
        _, o2 = decode(rows_d, ids_d)
        out = []
        for j, o in enumerate(o2):
            t = list(o.outputs[0].token_ids)
            first = next((k for k, (a, b) in enumerate(zip(t, base_tokens[j])) if a != b), None)
            if first is None and len(t) != len(base_tokens[j]):
                first = min(len(t), len(base_tokens[j]))
            out.append(first)
        rec["divergence"]["runs"].append({"kind": kind, **kw, "first": out})
        print(f"departures {kind} {kw}: {sum(x is not None for x in out)} of {len(out)}", flush=True)

    rpc("dc_sigma", SIGMA)
    for m in range(DIVERGE_MEMBERS):
        rpc("es_perturb", m)
        departures("member", member=m, norm=SIGMA * np.sqrt(d))
    rpc("dc_index", rl_dirs["120"])
    rpc("dc_along", 0.25)
    departures("rl120", lam=0.25, norm=0.25 * norms["120"])
    rpc("es_restore")
    if not rpc("es_check", None)["ok"]:
        raise SystemExit("engine weights are not the start after restore")
    verifier.close()
    rec["seconds"] = time.perf_counter() - t0
    return rec


def analyze(rec: dict) -> dict:
    c = np.asarray(rec["center"])
    ds = np.asarray(rec["datasets"])
    out = {"center": float(c.mean()), "rl_norm": rec["rl_norm"], "d": rec["d"], "points": []}
    for p in rec["points"]:
        r = np.asarray(p["rewards"])
        diff = r - c  # paired, prompt by prompt
        out["points"].append({k: p[k] for k in p if k != "rewards"} | {
            "reward": float(r.mean()), "change": float(diff.mean()),
            "change_se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
            "change_per_unit_norm": float(diff.mean() / p["norm"]),
            "up": float(((r > c)).mean()), "down": float(((r < c)).mean()),
            "by_source": {k: float(diff[ds == k].mean()) for k in sorted(set(ds))}})
    dv = rec["divergence"]
    allm = np.concatenate([np.asarray(m) for m in dv["center_margins"] if len(m)])
    out["margins"] = {"median": float(np.median(allm)), "exact_ties": float((allm == 0).mean()),
                      "share_below": {str(x): float((allm < x).mean()) for x in (0.1, 0.5, 1.0)}}
    out["departures"] = []
    for run in dv["runs"]:
        firsts = [(j, f) for j, f in enumerate(run["first"]) if f is not None]
        at = [dv["center_margins"][j][f] for j, f in firsts if f < len(dv["center_margins"][j])]
        lens = [len(m) for m in dv["center_margins"]]
        out["departures"].append({k: run[k] for k in run if k != "first"} | {
            "departed": len(firsts) / len(run["first"]),
            "median_position": float(np.median([f for _, f in firsts])) if firsts else None,
            "median_position_share": float(np.median([f / max(lens[j], 1) for j, f in firsts]))
            if firsts else None,
            "median_margin_at_departure": float(np.median(at)) if at else None,
            "share_departures_below_0.5": float(np.mean(np.asarray(at) < 0.5)) if at else None,
            # how much likelier a departure is at a near tie than at a position drawn at random
            "near_tie_enrichment": float(np.mean(np.asarray(at) < 0.5) / (allm < 0.5).mean())
            if at else None,
            "departures_at_exact_ties": float(np.mean(np.asarray(at) == 0)) if at else None})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analyze", type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.analyze:
        print(json.dumps(analyze(json.loads((E2E / args.analyze).read_text())), indent=2))
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
        "smoke": args.smoke, "seed": SEED, "sigma": SIGMA, **rec})
    print(json.dumps(analyze(rec), indent=1)[:3000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
