#!/usr/bin/env python
"""ES with RL's contrast as its fitness: noise or dimension? Andres's proposal, measured.

    python -m es_vllm.contrastive_check --out runs/contrastive-check/start.json   # on the GPU
    python -m es_vllm.contrastive_check --analyze runs/contrastive-check/start.json
    python -m es_vllm.contrastive_check --out ... --smoke                         # laptop wiring

ES ranked its members by greedy reward, which is mostly redraws and prompt-specific effects
(runs/grad-check/, runs/why/). The proposal: score each member by the contrast RL uses.
The start samples 16 answers per prompt at temperature 1.0 (the RL run's setting) on the
384 prompts of RL steps 1 to 8; the run's verifiers mark them; prompts with both right and
wrong answers are kept. A model's contrastive fitness is the mean over those prompts of

    mean log p(right answers) - mean log p(wrong answers)

(sequence log-probabilities, teacher-forced on the fixed answers; nothing is generated, so
nothing is redrawn). Its gradient at the start is, prompt by prompt, the RL run's
REINFORCE gradient with a group baseline on these samples, up to a weight per prompt
(K p(1-p) for a share p right). Scored: the start; the arm's 16 iteration-0 members at
sigma 5e-4; the RL run's direction at lambda 0.25, 0.5, 1 (as in `direction_check.py`);
two random member directions at nominal norms 0.476 and 0.951 (effective 0.28 and 0.73
after bf16 rounding, runs/rl-geometry/effective-norm.json); the start again after the
restore, for the scoring's own noise.

`--analyze`: the ranking's split-half reliability over disjoint contrast prompts, the
per-prompt effects and their cross-prompt correlation (as `why_analysis.py` does for the
reward), and the objective's change along RL's direction against random directions.
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

N, P_RUN, SEED, SIGMA, ALPHA, K = 16, 192, 0, 5e-4, 5e-4, 16
LAMBDAS = (0.25, 0.5, 1.0)
RANDOM_NORMS = (0.4756, 0.9513)   # nominal; effective 0.28 and 0.73
SAMPLE_SEED = 30_000


def collect(args) -> dict:
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from es_vllm.direction_check import worker_methods  # noqa: PLC0415
    from es_vllm.heldout import rl_revision  # noqa: PLC0415
    from es_vllm.run_tulu import Verifier, load_prompts, render  # noqa: PLC0415
    from es_vllm.worker import ESWorker  # noqa: PLC0415

    for k, fn in worker_methods().items():
        setattr(ESWorker, k, fn)
    get = lambda repo, rev: snapshot_download(repo, revision=rev,  # noqa: E731
                                              allow_patterns=["*.safetensors", "*.json"])
    if args.smoke:
        repo, revision, cap, batches, k, n = "Qwen/Qwen2.5-0.5B-Instruct", None, 64, 1, 4, 4
        rl_dir = get("Qwen/Qwen2.5-0.5B", None)
    else:
        repo, revision, cap, batches, k, n = (R.TULU31_START.repo, R.TULU31_START.commit, 2048,
                                              2, K, N)
        rl_dir = get(R.TULU31_RL.repo, rl_revision("step_120"))
    rows = [r for b in load_prompts(E2E / "data" / "tulu31", batches, P_RUN) for r in b]
    if args.smoke:
        rows = rows[:24]
    llm = LLM(model=repo, revision=revision, dtype="bfloat16", seed=0,
              gpu_memory_utilization=0.4 if args.smoke else 0.5, max_model_len=4096,
              enable_prefix_caching=False, worker_extension_cls="es_vllm.worker.ESWorker")
    rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
    rpc("es_init", get(repo, revision), n, SIGMA, ALPHA * SIGMA, SEED)
    rpc("es_ask")  # the arm's iteration-0 members
    d = rpc("dc_count")
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    ids = [render(tok, r) for r in rows]
    verifier = Verifier(REPO / ".venv-verify" / "bin" / "python")
    t0 = time.perf_counter()

    # 1. the start's samples, marked right or wrong, as the RL run's groups
    params = [SamplingParams(n=k, temperature=1.0, top_p=1.0, max_tokens=cap, seed=SAMPLE_SEED + j)
              for j in range(len(rows))]
    outs = llm.generate([{"prompt_token_ids": i} for i in ids], params, use_tqdm=False)
    items = [{"text": o.text, "ground_truth": r["ground_truth"], "dataset": r["dataset"],
              "stopped": o.finish_reason == "stop"} for out, r in zip(outs, rows) for o in out.outputs]
    rewards = verifier.score(items)
    if len(rewards) != len(rows) * k:  # the work asked for is the work done
        raise SystemExit(f"{len(rewards)} rewards for {len(rows)} x {k}")
    verifier.close()
    groups = []
    for j, out in enumerate(outs):
        r = rewards[j * k:(j + 1) * k]
        good = [list(o.token_ids) for o, x in zip(out.outputs, r) if x > 0]
        bad = [list(o.token_ids) for o, x in zip(out.outputs, r) if x <= 0]
        if good and bad:
            groups.append({"prompt": j, "good": good, "bad": bad})
        elif args.smoke:  # wiring only: a small model rarely has both
            toks = [list(o.token_ids) for o in out.outputs]
            groups.append({"prompt": j, "good": toks[: k // 2], "bad": toks[k // 2:]})
    seqs = [(g_i, lab, ids[g["prompt"]], resp) for g_i, g in enumerate(groups)
            for lab in ("good", "bad") for resp in g[lab]]
    print(f"{len(groups)} contrast prompts of {len(rows)}; {len(seqs)} answers to score; "
          f"{time.perf_counter() - t0:.0f}s", flush=True)

    # 2. a model's log-probability of every kept answer, teacher-forced
    score_params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)

    def score():
        res = llm.generate([{"prompt_token_ids": p + resp} for _, _, p, resp in seqs],
                           score_params, use_tqdm=False)
        logp = []
        for (_, _, p, resp), o in zip(seqs, res):
            plp = o.prompt_logprobs
            total = 0.0
            for pos in range(len(p), len(p) + len(resp)):
                total += plp[pos][(p + resp)[pos]].logprob
            logp.append(total)
        return logp

    rec = {"rows": len(rows), "k": k, "groups": [{"prompt": g["prompt"], "good": len(g["good"]),
                                                   "bad": len(g["bad"])} for g in groups],
           "datasets": [rows[g["prompt"]]["dataset"] for g in groups],
           "answer_lengths": [len(s[3]) for s in seqs], "labels": [s[1] for s in seqs],
           "group_of": [s[0] for s in seqs], "d": d, "points": []}

    def point(kind, **kw):
        rec["points"].append({"kind": kind, **kw, "logp": score()})
        print(f"{kind} {kw} {time.perf_counter() - t0:.0f}s", flush=True)

    point("start")
    rpc("dc_sigma", SIGMA)
    for m in range(n):
        rpc("es_perturb", m)
        if m == 0 and not rpc("es_check", 0)["ok"]:
            raise SystemExit("engine weights are not member 0")
        point("member", member=m, norm=SIGMA * np.sqrt(d))
    for norm in RANDOM_NORMS:
        for m in (0, 1):
            rpc("dc_sigma", norm / np.sqrt(d))
            rpc("es_perturb", m)
            point("random", member=m, norm=norm)
    rpc("dc_sigma", SIGMA)
    rec["rl_norm"] = rpc("dc_index", rl_dir)
    for lam in LAMBDAS:
        rpc("dc_along", lam)
        point("rl120", lam=lam, norm=lam * rec["rl_norm"])
    rpc("es_restore")
    if not rpc("es_check", None)["ok"]:
        raise SystemExit("engine weights are not the start after restore")
    point("start_again")  # the scoring's own noise: should match "start" to the bit
    rec["seconds"] = time.perf_counter() - t0
    return rec


def objective(rec, logp) -> np.ndarray:
    """Per contrast prompt: mean log p(right) - mean log p(wrong)."""
    lp, lab, grp = np.asarray(logp), np.asarray(rec["labels"]), np.asarray(rec["group_of"])
    out = np.zeros(len(rec["groups"]))
    for g in range(len(out)):
        sel = grp == g
        out[g] = lp[sel & (lab == "good")].mean() - lp[sel & (lab == "bad")].mean()
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    """As `group_relative` on (n, 1): population standard deviation, zero if none."""
    return (x - x.mean()) / x.std() if x.std() > 0 else 0 * x


def analyze(rec: dict, draws=2000) -> dict:
    rng = np.random.default_rng(0)
    pts = rec["points"]
    f0 = objective(rec, pts[0]["logp"])
    members = np.asarray([objective(rec, p["logp"]) for p in pts if p["kind"] == "member"])
    dm = members - f0[None]                       # members x contrast prompts
    n, P = dm.shape
    rel, cov, held = [], [], []
    for _ in range(draws):
        perm = rng.permutation(P)
        a, b = perm[: P // 2], perm[P // 2: 2 * (P // 2)]
        x, y = dm[:, a].mean(1), dm[:, b].mean(1)
        cov.append(np.cov(x, y)[0, 1])
        if x.std() > 0 and y.std() > 0:
            rel.append(np.corrcoef(x, y)[0, 1])
        held += [np.mean(zscore(x) * y), np.mean(zscore(y) * x)]
    # One ES update ranked by this fitness, judged on prompts it was not ranked on. The
    # update is (alpha/N) sum_m z_m eps_m, so to first order it moves the objective by
    # (alpha/sigma) mean_m z_m delta_m; sum z = 0 cancels what all members share. Its
    # random part is sqrt(N) times shorter than a member, so it pays the members' mean
    # change (the curvature) over N.
    gain = ALPHA / SIGMA * float(np.mean(held))
    cost, cost_se = float(dm.mean()) / n, float(dm.mean(1).std(ddof=1)) / np.sqrt(n) / n
    per_prompt = float(dm.var(axis=0, ddof=1).mean())   # deterministic: all of it is effect
    shared = float(np.mean(cov))
    again = [p for p in pts if p["kind"] == "start_again"]
    out = {"contrast_prompts": P, "start_objective": float(f0.mean()),
           "rescore_max_abs_logp_diff": float(np.abs(np.subtract(again[0]["logp"], pts[0]["logp"])).max())
           if again else None,
           "members": {"mean_change": float(dm.mean()), "member_sd": float(dm.mean(1).std(ddof=1)),
                       "split_half_reliability": float(np.mean(rel)) if rel else 0.0,
                       "per_prompt_effect_variance": per_prompt, "shared_variance": shared,
                       "cross_prompt_correlation": shared / per_prompt if per_prompt else None},
           "es_update": {"first_order_gain_held_out": gain,
                         "first_order_gain_in_sample": ALPHA / SIGMA * float(np.mean(
                             zscore(dm.mean(1)) * dm.mean(1))),
                         "curvature_cost": cost, "curvature_cost_se": cost_se,
                         "net_held_out": gain + cost},
           "directions": []}
    for p in pts:
        if p["kind"] in ("random", "rl120"):
            dp = objective(rec, p["logp"]) - f0
            out["directions"].append({k: p[k] for k in p if k != "logp"} | {
                "change": float(dp.mean()), "change_se": float(dp.std(ddof=1) / np.sqrt(len(dp))),
                "share_of_prompts_up": float((dp > 0).mean())})
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
