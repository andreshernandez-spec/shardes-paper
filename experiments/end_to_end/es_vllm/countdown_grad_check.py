#!/usr/bin/env python
"""The gradient probe on Countdown, where ES learns: are its members' effects shared across prompts?

    python -m es_vllm.countdown_grad_check --setting qwen0.5b --out runs/grad-check/countdown-qwen0.5b.json
    python -m es_vllm.countdown_grad_check --setting tulu8b --out runs/grad-check/countdown-tulu8b.json
    python -m es_vllm.countdown_grad_check --analyze runs/grad-check/countdown-tulu8b.json \
        --tulu runs/grad-check/start-sampled.json
    python -m es_vllm.countdown_grad_check --analyze runs/grad-check/countdown-qwen0.5b.json \
        --t2 runs/countdown-s1/log.jsonl

On Tulu a random perturbation moves each uncertain prompt's success probability by about
8 points, but its effects on two prompts correlate 0.0095 (`why_analysis.py`), and the
greedy ranking of the 16 members over 192 prompts is 12% their own effect (`grad_check.py`).
The explanation offered: ES needs an improvement that many prompts share, and Tulu's
prompts after DPO offer little of it, while on Countdown the start mostly fails on a
format that every prompt shares. This measures the same quantities on Countdown's 200
training prompts, with es-at-scale's raw prompts (no chat template, as es-at-scale gives
them to every model) and grader (format and answer), greedy and K samples per prompt at
temperature 1.0 with independent seeds per member:

- `qwen0.5b`: T2's seed-1 iteration-0 members (N = 30, sigma 1e-3), where ES learned.
- `tulu8b`: the Tulu arm's iteration-0 members (seed 0, N = 16) at sigma 5e-4 and 1e-3,
  the start and members of `grad_check.py`. Only the task differs from the Tulu probes.

Prediction, committed before the run. If the explanation holds:
- qwen0.5b: the greedy ranking's reliability at 192 prompts is at least 0.5 (Tulu 0.12)
  and the cross-prompt correlation of the members' true effects at least 0.05 (Tulu 0.0095).
- tulu8b at sigma 5e-4: reliability at least 0.3 and correlation at least 0.03.
If tulu8b on Countdown looks like Tulu on Tulu, the explanation is wrong or holds only
for the small model, and the difference lies in the model or the regime, not the task.
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
SETTINGS = {  # release, members, sigmas (sampled at the first), seed
    "qwen0.5b": ("QWEN25_05B", 30, (1e-3,), 1),
    "tulu8b": ("TULU31_START", 16, (5e-4, 1e-3), 0),
}
ALPHA = 5e-4


def collect(args) -> dict:
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from es_vllm.direction_check import worker_methods  # noqa: PLC0415
    from es_vllm.run_countdown import Grader, es_at_scale, rows  # noqa: PLC0415
    from es_vllm.worker import ESWorker  # noqa: PLC0415

    for k, fn in worker_methods().items():
        setattr(ESWorker, k, fn)
    cfg = yaml.safe_load((PKG / "countdown-s1.yaml").read_text())
    release, n, sigmas, seed = SETTINGS[args.setting]
    clone = es_at_scale(args.es_at_scale)
    train = rows(clone, "train")[: 16 if args.smoke else None]
    grader = Grader(cfg["grader_timeout"])  # forks: before CUDA exists here
    n = 4 if args.smoke else n
    rel = getattr(R, release)
    llm = LLM(model=rel.repo, revision=rel.commit, dtype="bfloat16", seed=0,
              gpu_memory_utilization=0.4 if args.smoke else cfg["gpu_memory_utilization"],
              max_model_len=cfg["max_model_len"], enable_prefix_caching=False,
              worker_extension_cls="es_vllm.worker.ESWorker")
    model_dir = snapshot_download(rel.repo, revision=rel.commit,
                                  allow_patterns=["*.safetensors", "*.json"])
    rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
    rpc("es_init", model_dir, n, sigmas[0], ALPHA * sigmas[0], seed)
    rpc("es_ask")  # the run's iteration-0 members
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
        fmt = [[scores[j * k + s][2] for s in range(k)] for j in range(len(train))]
        return reward, answer, fmt

    greedy = SamplingParams(temperature=0.0, max_tokens=cap)
    sampled = lambda m: [SamplingParams(n=K, temperature=1.0, top_p=1.0, max_tokens=cap,  # noqa: E731
                                        seed=20_000 + j + 1_000_003 * (m + 2))
                         for j in range(len(train))]
    t0 = time.perf_counter()
    rec: dict = {"greedy": {"members": {}, "answers": {}, "formats": {}},
                 "sampled": {"sigma": sigmas[0], "members": [], "answers": []}}
    r, a, f = decode(greedy)
    rec["greedy"].update({"center": [x[0] for x in r], "center_answer": [x[0] for x in a],
                          "center_format": [x[0] for x in f]})
    r, a, _ = decode(sampled(-1))
    rec["sampled"].update({"center": r, "center_answer": a})
    for i, sigma in enumerate(sigmas):
        key = str(sigma)
        for part in ("members", "answers", "formats"):
            rec["greedy"][part][key] = []
        rpc("dc_sigma", sigma)
        for m in range(n):
            rpc("es_perturb", m)
            if i == 0 and m == 0 and not rpc("es_check", 0)["ok"]:
                raise SystemExit("engine weights are not member 0")
            r, a, f = decode(greedy)
            rec["greedy"]["members"][key].append([x[0] for x in r])
            rec["greedy"]["answers"][key].append([x[0] for x in a])
            rec["greedy"]["formats"][key].append([x[0] for x in f])
            if i == 0:
                r, a, _ = decode(sampled(m))
                rec["sampled"]["members"].append(r)
                rec["sampled"]["answers"].append(a)
            print(f"sigma {sigma} member {m}: greedy {np.mean(rec['greedy']['members'][key][-1]):.4f} "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    rpc("es_restore")
    if not rpc("es_check", None)["ok"]:
        raise SystemExit("engine weights are not the start after restore")
    grader.close()
    rec.update({"setting": args.setting, "model": release, "n": n, "k": K, "sigmas": list(sigmas),
                "seed": seed, "prompts": len(train), "max_tokens": cap,
                "seconds": time.perf_counter() - t0})
    return rec


def effects(center, members, scale, draws=2000) -> dict:
    """True per-prompt effects of the members and how much of them prompts share, from
    independent samples (`center` prompts x K, `members` n x prompts x K). Two normalizations:
    over the prompts whose success probability is between 0.05 and 0.95 (`why_analysis`'s,
    for Tulu), and over all prompts, which needs no threshold on a partial-credit reward."""
    r = np.concatenate([np.asarray(center)[None], np.asarray(members)]) / scale
    cell = r.mean(axis=2)
    p = cell.mean(axis=0)
    noise = r.var(axis=2, ddof=1).mean(axis=0) / r.shape[2]
    excess = cell.var(axis=0, ddof=1) - noise
    unc = (p > 0.05) & (p < 0.95)
    d = cell[1:] - cell[0][None]
    rng = np.random.default_rng(0)
    cov = []
    for _ in range(draws):
        perm = rng.permutation(cell.shape[1])
        a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
        cov.append(np.cov(d[:, a].mean(1), d[:, b].mean(1))[0, 1])
    shared = float(np.mean(cov))
    per_unc = float(excess[unc].mean()) if unc.any() else float("nan")
    return {"uncertain_prompts": int(unc.sum()), "per_prompt_true_variance_uncertain": per_unc,
            "per_prompt_true_variance_all": float(excess.mean()), "shared_variance": shared,
            "cross_prompt_correlation_uncertain": shared / per_unc * (cell.shape[1] / unc.sum()) ** 2
            if unc.sum() >= 10 else None,
            "cross_prompt_correlation_all": shared / float(excess.mean())}


def analyze(rec: dict, t2_log: list | None = None, tulu: dict | None = None, draws=2000) -> dict:
    """As `why_analysis.effects` and `grad_check.analyze` do for Tulu, so the numbers compare.
    `t2_log`: T2's seed-1 log, whose iteration-0 members are the qwen0.5b setting's. `tulu`:
    the Tulu probe's sampled record (`grad_check.py --sampled`), put through `effects`."""
    from es_vllm.grad_check import split_reliability, variance_parts  # noqa: PLC0415

    rng = np.random.default_rng(0)
    s = rec["sampled"]
    out = {"setting": rec["setting"], "prompts": rec["prompts"], "n": rec["n"],
           "sampled": {"sigma": s["sigma"], "center": float(np.asarray(s["center"]).mean()),
                       **effects(s["center"], s["members"], 1.1, draws)}}
    g = rec["greedy"]
    c = np.asarray(g["center"])
    out["greedy"] = {"center": float(c.mean()),
                     "center_correct": float(np.mean(np.asarray(g["center_answer"]) > 0)),
                     "center_full_format": float(np.mean(np.asarray(g["center_format"]) == 1.0)),
                     "sigmas": {}}
    for key, rows_ in g["members"].items():
        sigma, delta = float(key), np.asarray(rows_) - c[None]
        parts = variance_parts(delta)
        v_m, v_e = parts["v_member"], parts["v_resid"]
        rel = v_m / (v_m + v_e / delta.shape[1]) if v_m + v_e > 0 else 0.0
        # first order, z-scored weights, as `grad_check`; the update's random part is
        # alpha / (sigma sqrt(N)) of a member's length, so it pays that squared of a
        # member's mean change (`why_analysis.curvature`)
        gain = ALPHA / sigma * float(np.sqrt(v_m * rel))
        cost = float(delta.mean()) * ALPHA ** 2 / (sigma ** 2 * delta.shape[0])
        out["greedy"]["sigmas"][key] = {
            "member_mean_change": float(delta.mean()),
            "member_full_format": float(np.mean(np.asarray(g["formats"][key]) == 1.0)),
            "member_sd": float(delta.mean(1).std(ddof=1)),
            "pairs_changed": float((delta != 0).mean()),
            "pairs_up": float((delta > 0).mean()), "pairs_down": float((delta < 0).mean()),
            "true_effect_sd": float(np.sqrt(v_m)), "residual": v_e,
            "reliability_192": v_m / (v_m + v_e / 192) if v_m + v_e > 0 else 0.0,
            "reliability_all": rel, "split_halves": split_reliability(delta, 2, rng),
            "implied_gain_per_iteration": gain, "random_walk_cost_per_iteration": cost}
    if t2_log is not None:
        probe = np.asarray(g["members"][str(rec["sigmas"][0])]).mean(1)
        logged = np.asarray(t2_log[0]["fitness"])
        fit = np.asarray([r["mean_fitness"] for r in t2_log])
        out["t2"] = {"corr_probe_logged_fitness": float(np.corrcoef(probe, logged)[0, 1]),
                     "max_abs_diff": float(np.abs(probe - logged).max()),
                     "mean_fitness_slope": {f"0-{hi}": float(np.polyfit(np.arange(hi + 1), fit[: hi + 1], 1)[0])
                                            for hi in (20, 50, len(fit) - 1)}}
    if tulu is not None:
        ind = tulu["sampled"]["independent"]
        out["tulu_on_tulu"] = effects(ind["center"], ind["members"], 10.0, draws)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analyze", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--setting", choices=sorted(SETTINGS), default="qwen0.5b")
    ap.add_argument("--t2", type=Path, help="with --analyze: T2's seed-1 log.jsonl")
    ap.add_argument("--tulu", type=Path, help="with --analyze: the Tulu probe's start-sampled.json")
    ap.add_argument("--es-at-scale", type=Path,
                    default=Path.home() / "private" / "open-source" / "es-at-scale")
    args = ap.parse_args(argv)
    if args.analyze:
        t2 = [json.loads(x) for x in (E2E / args.t2).read_text().splitlines()] if args.t2 else None
        tulu = json.loads((E2E / args.tulu).read_text()) if args.tulu else None
        print(json.dumps(analyze(json.loads((E2E / args.analyze).read_text()), t2, tulu), indent=2))
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
