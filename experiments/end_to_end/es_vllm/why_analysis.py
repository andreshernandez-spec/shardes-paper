#!/usr/bin/env python
"""Why ES's gradient on Tulu is mostly noise: three mechanisms tested on the probes' records.

    python -m es_vllm.why_analysis          # from experiments/end_to_end; CPU, seconds

Reads runs/grad-check/start.json (greedy, 768 prompts, iteration 0's members at three
sigmas), runs/grad-check/start-sampled.json (temperature 1.0, 8 samples, the first 192
prompts) and the longer arm's and control's held-out records.

1. Redraws: a member's greedy outcome on a prompt behaves like a fresh draw from the
   start's own success probability p (estimated from the independent-seed samples). The
   resampling model: flip up with probability q p where the start fails, flip down with
   q (1 - p) where it succeeds; q fitted, and how well it predicts each prompt's flip rate.
2. Per-prompt and shared effects: from the 8 independent samples per member and prompt,
   the variance of the members' true success probability on each prompt (cell variance
   minus binomial noise), against the shared part (the covariance of the members' mean
   change between disjoint prompt halves). Their ratio is the average correlation of one
   perturbation's effects on two prompts, which for isotropic perturbations equals the
   average cosine between the prompts' gradients.
3. Curvature: the members' mean change against perturbation norm (sigma sqrt(d)), and the
   cost of the arm's random walk it predicts (step norm alpha sqrt(d / N) per update),
   against the control's and the arm's held-out change.
"""

import json
from pathlib import Path

import numpy as np

E2E = Path(__file__).resolve().parent.parent
D, N, ALPHA, P = 8030326784, 16, 5e-4, 192


def load(p):
    return json.loads((E2E / p).read_text())


def redraws(g, s) -> dict:
    c = np.asarray(g["center"][:P]) / 10
    m = np.asarray(g["members"]["0.0005"])[:, :P] / 10
    ind = s["independent"]
    pr = (np.asarray(ind["members"]).sum(axis=(0, 2)) / 10
          + np.asarray(ind["center"]).sum(axis=1) / 10) / ((len(ind["members"]) + 1) * s["k"])
    flip = m != c[None, :]
    pred = np.where(c == 0, pr, 1 - pr)
    q = float(flip.mean(axis=0).sum() / pred.sum())
    bins = [0, 0.001, 0.2, 0.4, 0.6, 0.8, 0.999, 1.001]
    table = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (pr >= lo) & (pr < hi)
        w, r = sel & (c == 0), sel & (c == 1)
        table.append({"p": [lo, hi], "start_wrong": int(w.sum()),
                      "up_rate": float(flip[:, w].mean()) if w.any() else None,
                      "start_right": int(r.sum()),
                      "down_rate": float(flip[:, r].mean()) if r.any() else None})
    return {"flips": float(flip.mean()), "up": float((flip & (c[None] == 0)).mean()),
            "down": float((flip & (c[None] == 1)).mean()), "q": q,
            "corr_prompt_flip_rate": float(np.corrcoef(flip.mean(axis=0), q * pred)[0, 1]),
            "flips_where_p_is_0": float(flip[:, pr == 0].mean()) if (pr == 0).any() else None,
            "flips_where_p_is_1": float(flip[:, pr == 1].mean()) if (pr == 1).any() else None,
            "by_p": table}


def effects(s, draws=2000) -> dict:
    ind = s["independent"]
    r = np.concatenate([np.asarray(ind["center"])[None], np.asarray(ind["members"])]) / 10
    cell = r.mean(axis=2)
    p = cell.mean(axis=0)
    noise = r.var(axis=2, ddof=1).mean(axis=0) / s["k"]
    excess = cell.var(axis=0, ddof=1) - noise
    unc = (p > 0.05) & (p < 0.95)
    rng = np.random.default_rng(0)
    d = cell[1:] - cell[0][None]
    cov = []
    for _ in range(draws):
        perm = rng.permutation(cell.shape[1])
        a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
        cov.append(np.cov(d[:, a].mean(1), d[:, b].mean(1))[0, 1])
    shared = float(np.mean(cov))
    per = float(excess[unc].mean())
    return {"uncertain_prompts": int(unc.sum()), "per_prompt_true_variance": per,
            "per_prompt_true_sd": float(np.sqrt(max(per, 0))),
            "binomial_variance_one_draw": float((p * (1 - p))[unc].mean()),
            "shared_variance": shared, "shared_sd": float(np.sqrt(max(shared, 0))),
            "cross_prompt_correlation": shared / per * (cell.shape[1] / unc.sum()) ** 2}


def curvature(g) -> dict:
    c = np.asarray(g["center"])
    cost = {s: float(np.asarray(m).mean() - c.mean()) for s, m in g["members"].items()}
    norm = {s: float(s) * np.sqrt(D) for s in cost}
    k = cost["0.0005"] / norm["0.0005"] ** 2
    step2 = ALPHA ** 2 * D / N
    h = {n: {r["iteration"]: r["reward"] for r in map(json.loads, (E2E / f"runs/{n}/heldout.jsonl")
                                                       .read_text().splitlines())}
         for n in ("tulu-long-random", "tulu-long-s5e-4")}
    rows = [{"updates": t, "displacement": float(np.sqrt(t * step2)), "predicted": k * t * step2,
             "control": h["tulu-long-random"][t] - h["tulu-long-random"][0],
             "arm": h["tulu-long-s5e-4"][t] - h["tulu-long-s5e-4"][0]} for t in (30, 60, 90, 120)]
    return {"cost_by_sigma": {s: {"norm": norm[s], "cost": cost[s], "per_norm2": cost[s] / norm[s] ** 2}
                              for s in cost},
            "step_norm": float(np.sqrt(step2)), "predicted_per_iteration": k * step2,
            "random_walk": rows}


def main() -> int:
    g, s = load("runs/grad-check/start.json"), load("runs/grad-check/start-sampled.json")["sampled"]
    res = {"redraws": redraws(g, s), "effects": effects(s), "curvature": curvature(g)}
    out = E2E / "runs" / "why" / "analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
