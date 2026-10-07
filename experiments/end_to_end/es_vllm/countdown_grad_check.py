#!/usr/bin/env python
"""The gradient probe on Countdown, where ES learns: are its members' effects shared across prompts?

    python -m es_vllm.countdown_grad_check --out runs/grad-check/countdown-start.json
    python -m es_vllm.countdown_grad_check --analyze runs/grad-check/countdown-start.json

On Tulu a random perturbation moves each uncertain prompt's success probability by about
8 points, but its effects on two prompts correlate 0.01 (`why_analysis.py`), so the mean
over prompts that ES ranks members by carries little. The explanation predicts that ES
learns where the correlation is high. On Countdown at 0.5B (T2) it learned fast; there
the start mostly fails on format, which every prompt shares. This measures the same
quantities there: T2's seed-1 iteration-0 members (N = 30, sigma 1e-3) and the start on
the 200 training prompts, greedy, and K samples each at temperature 1.0 with independent
seeds per member, with es-at-scale's grader (reward 0.1 format + answer).
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
import yaml  # noqa: E402

PKG = Path(__file__).resolve().parent
E2E = PKG.parent
sys.path.insert(0, str(E2E))
sys.path.insert(0, str(E2E.parent))
import harness  # noqa: E402
import releases as R  # noqa: E402
from provenance import env_block  # noqa: E402

K = 4


def collect(args) -> dict:
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from es_vllm.run_countdown import Grader, es_at_scale, rows  # noqa: PLC0415

    cfg = yaml.safe_load((PKG / "countdown-s1.yaml").read_text())
    clone = es_at_scale(args.es_at_scale)
    train = rows(clone, "train")[: 16 if args.smoke else None]
    grader = Grader(cfg["grader_timeout"])  # forks: before CUDA exists here
    n = 4 if args.smoke else cfg["population"]
    rel = getattr(R, cfg["model"])
    llm = LLM(model=rel.repo, revision=rel.commit, dtype="bfloat16", seed=0,
              gpu_memory_utilization=0.4 if args.smoke else cfg["gpu_memory_utilization"],
              max_model_len=cfg["max_model_len"], enable_prefix_caching=False,
              worker_extension_cls="es_vllm.worker.ESWorker")
    model_dir = snapshot_download(rel.repo, revision=rel.commit,
                                  allow_patterns=["*.safetensors", "*.json"])
    llm.collective_rpc("es_init", args=(model_dir, n, cfg["sigma"], cfg["alpha"] * cfg["sigma"],
                                        cfg["seed"]))
    llm.collective_rpc("es_ask")  # T2 seed 1's iteration-0 members
    cap = 64 if args.smoke else cfg["max_tokens"]
    prompts = [r["context"] for r in train]

    def decode(params):
        outs = llm.generate(prompts, params, use_tqdm=False)
        texts = [[o.text for o in out.outputs] for out in outs]
        flat = [(t, r) for ts, r in zip(texts, train) for t in ts]
        scores = grader.score([t for t, _ in flat], [r for _, r in flat])
        if len(scores) != len(flat):  # the work asked for is the work done
            raise SystemExit(f"{len(scores)} scores for {len(flat)} responses")
        k = len(texts[0])
        reward = [[scores[j * k + s][0] for s in range(k)] for j in range(len(train))]
        answer = [[scores[j * k + s][1] for s in range(k)] for j in range(len(train))]
        return reward, answer

    greedy = SamplingParams(temperature=0.0, max_tokens=cap)
    sampled = lambda m: [SamplingParams(n=K, temperature=1.0, top_p=1.0, max_tokens=cap,  # noqa: E731
                                        seed=20_000 + j + 1_000_003 * (m + 2))
                         for j in range(len(train))]
    t0 = time.perf_counter()
    rec: dict = {"greedy": {"members": [], "answers": []}, "sampled": {"members": [], "answers": []}}
    r, a = decode(greedy)
    rec["greedy"].update({"center": [x[0] for x in r], "center_answer": [x[0] for x in a]})
    r, a = decode(sampled(-1))
    rec["sampled"].update({"center": r, "center_answer": a})
    for m in range(n):
        llm.collective_rpc("es_perturb", args=(m,))
        if m == 0 and not llm.collective_rpc("es_check", args=(0,))[0]["ok"]:
            raise SystemExit("engine weights are not member 0")
        r, a = decode(greedy)
        rec["greedy"]["members"].append([x[0] for x in r])
        rec["greedy"]["answers"].append([x[0] for x in a])
        r, a = decode(sampled(m))
        rec["sampled"]["members"].append(r)
        rec["sampled"]["answers"].append(a)
        print(f"member {m}: greedy {np.mean(rec['greedy']['members'][-1]):.4f} "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
    llm.collective_rpc("es_restore")
    grader.close()
    rec.update({"n": n, "k": K, "sigma": cfg["sigma"], "seed": cfg["seed"], "prompts": len(train),
                "seconds": time.perf_counter() - t0})
    return rec


def analyze(rec: dict, draws=2000) -> dict:
    rng = np.random.default_rng(0)
    out = {}
    # per-prompt true effects and their cross-prompt correlation, from the independent samples
    s = rec["sampled"]
    r = np.concatenate([np.asarray(s["center"])[None], np.asarray(s["members"])])  # models x prompts x K
    cell = r.mean(axis=2)
    noise = r.var(axis=2, ddof=1).mean(axis=0) / r.shape[2]
    excess = cell.var(axis=0, ddof=1) - noise
    active = cell.std(axis=0) > 0
    d = cell[1:] - cell[0][None]
    cov = []
    for _ in range(draws):
        perm = rng.permutation(cell.shape[1])
        a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
        cov.append(np.cov(d[:, a].mean(1), d[:, b].mean(1))[0, 1])
    shared, per = float(np.mean(cov)), float(excess[active].mean())
    out["sampled"] = {"active_prompts": int(active.sum()), "per_prompt_true_variance": per,
                      "shared_variance": shared,
                      "cross_prompt_correlation": shared / per * (cell.shape[1] / active.sum()) ** 2,
                      "mean_reward_center": float(np.asarray(s["center"]).mean())}
    # greedy: the ranking's split-half reliability at half the prompts
    g = rec["greedy"]
    gm = np.asarray(g["members"])
    rel = []
    for _ in range(draws):
        perm = rng.permutation(gm.shape[1])
        a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
        x, y = gm[:, a].mean(1), gm[:, b].mean(1)
        if x.std() > 0 and y.std() > 0:
            rel.append(np.corrcoef(x, y)[0, 1])
    c = np.asarray(g["center"])
    out["greedy"] = {"center": float(c.mean()), "member_mean_change": float(gm.mean() - c.mean()),
                     "pairs_changed": float((gm != c[None]).mean()),
                     "split_half_reliability_100_prompts": float(np.mean(rel)) if rel else 0.0}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analyze", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--es-at-scale", type=Path,
                    default=Path.home() / "private" / "open-source" / "es-at-scale")
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
        "smoke": args.smoke, **rec})
    print(json.dumps(analyze(rec), indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
