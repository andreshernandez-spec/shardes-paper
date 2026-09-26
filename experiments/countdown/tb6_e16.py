#!/usr/bin/env python
"""The alignment tables: E15 and E16 at equal population, and against the predictions.

    python tb6_e16.py            # print the grid
    python tb6_e16.py --latex    # also write paper/generated/tb6.tex and tb6a.tex

tb6 is the main-text table: medians and ranges, one row per (model, batch, N),
one column per mirrored variant. Predictions and unpaired sampling stay in tb6a. tb6a is the appendix audit: every cell
with its range, the reference and the frozen prediction, and both ratios.

Reads only committed artifacts: E1's update--gradient-alignment results (the fits), the
E15 and E16 per-cell JSONs, and the model configs. Two predictions per cell, both from
the committed E1 fits (sigma 1e-3, centered ranks), neither fit to these measurements:

- `reference`: the mirrored full-rank fit at N/P, the same curve for every arm. This is
  the post hoc reading: at equal population the rank barely matters, so one curve should
  serve every mirrored arm. Unpaired arms have N, not N/2, directions and should sit
  about sqrt(2) above it.
- printed only, `linear`: what a locally linear fitness gives with no fit at all. Centered
  ranks are about Phi(z) - 1/2 of the standardized projection z, so for N << P the cosine
  is corr(U, z) sqrt(N/P) = sqrt(3/pi) sqrt(N/P) unpaired, and 1/sqrt(2) of that for
  mirrored pairs (N/2 directions). That assumes z is close to Gaussian. Low-rank noise
  AB^T/sqrt(r) has identity covariance, so the rank does not enter as long as it is; z
  is a sum of products of Gaussians, close to Gaussian when the gradient is spread over
  many entries and not when it is concentrated on a few.
- `frozen`: what was committed before each run, the arm's own E1 fit at N/d_samp
  (e16-gate.yaml, e16-stage2.yaml), and for rank 1 at 1.5B that fit times the 0.53
  correction E15 measured at 0.5B (frozen in e16-stage2.yaml). E1 never measured r=16,
  so that arm has no frozen prediction.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp

from analysis_e15 import e1_curve, fit_predict

HERE = Path(__file__).resolve().parent
from shardes.dimensions import FULL, sampling_dimension  # noqa: E402
from shardes.problems import qwen2  # noqa: E402

CORRECTION = 0.53  # frozen in e16-stage2.yaml before the run

ROWS = [  # (results dir, label, model config, calibrate rank-1?)
    ("results-e15", "0.5B", "qwen25_05b", False),
    ("results-e16-gate", "0.5B, second batch", "qwen25_05b", False),
    ("results-e16-stage2", "1.5B", "qwen25_15b", True),
]
# strategy -> (label, E1 curve or None, rank)
ARMS = {"mirrored_seed": ("seed, mirrored", "mirrored_full", FULL),
        "mirrored_lr1": ("rank 1", "mirrored_lr1", 1),
        "mirrored_lr4": ("rank 4", "mirrored_lr4", 4),
        "mirrored_lr16": ("rank 16", None, 16),
        "lowrank_r1": ("rank 1, unpaired", "lowrank_r1", 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args(argv)

    law = e1_curve("mirrored_full", "centered_ranks")
    lines, p_of = [], {}
    for rdir, label, config, calibrate in ROWS:
        cfg = getattr(qwen2.Config, config)()
        tree = jax.eval_shape(lambda: qwen2.init(jax.random.key(0), cfg,
                                                 dtype=jnp.bfloat16))
        p = sampling_dimension(tree, FULL)
        p_of[label] = p
        # What a fit read at N/d_samp carries over from the block: P/d_samp per rank.
        print(f"{label}: P {p}, P/d_samp " + ", ".join(
            f"rank {r} {p / sampling_dimension(tree, r):.0f}" for r in (1, 4)))
        for strategy, (name, e1_name, rank) in ARMS.items():
            files = sorted((HERE / rdir).glob(f"s={strategy}__N=*.json"),
                           key=lambda f: json.loads(f.read_text())["config"]["population"])
            for f in files:
                d = json.loads(f.read_text())
                n = d["config"]["population"]
                frozen = None
                if e1_name is not None:
                    frozen = fit_predict(e1_curve(e1_name, "centered_ranks"),
                                         n / sampling_dimension(tree, rank))
                    if calibrate and rank == 1:
                        frozen *= CORRECTION
                lines.append((label, name, n, d["cosine_median"], sorted(d["cosines"]),
                              fit_predict(law, n / p), frozen))

    lines = [l + (p_of[l[0]],) for l in lines]
    w = max(len(l[0]) for l in lines)
    for label, name, n, med, cos, pl, pf, p in lines:
        linear = math.sqrt(3 / math.pi) * math.sqrt(n / p)
        if "unpaired" not in name:
            linear /= math.sqrt(2)
        s = (f"{label:{w}s}  {name:16s} N={n:<4d} measured {med:8.2e} "
             f"[{cos[0]:.1e}-{cos[-1]:.1e}]  law {pl:8.2e} ratio {med / pl:4.2f}x"
             f"  linear {linear:8.2e} ratio {med / linear:4.2f}x")
        if pf is not None:
            s += f"  frozen {pf:8.2e} ratio {med / pf:4.2f}x"
        print(s)
    full = {(l[0], l[2]): l[3] for l in lines if l[1] == "seed, mirrored"}
    for label, name, n, med, *_ in lines:
        if (label, n) in full and name != "seed, mirrored":
            print(f"at equal N: {label:{w}s} {name:16s} N={n:<4d} over full rank "
                  f"{med / full[(label, n)]:.2f}")
    r1 = {(l[0], l[2]): l[3] for l in lines if l[1] == "rank 1"}
    for label, name, n, med, *_ in lines:
        if name == "rank 1, unpaired" and (label, n) in r1:
            print(f"unpaired over mirrored rank 1: {label} N={n} {med / r1[(label, n)]:.2f}")

    if args.latex:
        gen = HERE.parent.parent / "paper" / "generated"
        # Main text: measured alignment and variation; prediction audits stay below.
        cols = ["seed, mirrored", "rank 1", "rank 4", "rank 16"]
        cell = {(l[0], l[2], l[1]): (l[3], l[4]) for l in lines}
        short = {"0.5B": "0.5B", "0.5B, second batch": "0.5B, batch 2", "1.5B": "1.5B"}
        rows = []
        for label, n in dict.fromkeys((l[0], l[2]) for l in lines):
            if label == "0.5B, second batch":
                continue
            vals = []
            for c in cols:
                entry = cell.get((label, n, c))
                if entry is None:
                    vals.append("--")
                else:
                    med, samples = entry
                    vals.append(f"{med * 1e4:.2f} [{min(samples) * 1e4:.1f}, {max(samples) * 1e4:.1f}]")
            rows.append(f"{short[label]} & {n} & " + " & ".join(vals) + r" \\")
        (gen / "tb6.tex").write_text("\n".join([
            "% generated by experiments/countdown/tb6_e16.py --latex; do not edit",
            r"\begin{table*}[t]", r"\centering", r"\small",
            r"\caption{Alignment with the next-token loss gradient: "
            r"$\cos(\Delta\theta,-\nabla L)\times10^4$, median [min, max] across five "
            r"perturbation seeds. All variants use mirrored sampling. Rank-1 medians "
            r"are close to full rank in each check; the ranges show substantial variation "
            r"at small populations. Dashes denote unmeasured comparisons.}",
            r"\label{tab:tb6}", r"\begin{tabular}{llrrrr}", r"\toprule",
            r"model & population $N$ & full rank & rank 1 & rank 4 & rank 16 \\",
            r"\midrule", *rows,
            r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]))
        # Appendix: every cell against both predictions.
        rows, prev = [], None
        for label, name, n, med, cos, pl, pf, _ in lines:
            if prev is not None and label != prev:
                rows.append("\\midrule")
            prev = label
            meas = f"${med * 1e4:.2f}$ [{cos[0] * 1e4:.1f}, {cos[-1] * 1e4:.1f}]"
            fz = (f"${pf * 1e4:.2f}$ & ${med / pf:.2f}$" if pf is not None else "-- & --")
            rows.append(f"{label} & {name} & {n} & {meas} & ${pl * 1e4:.2f}$ & "
                        f"${med / pl:.2f}$ & {fz} \\\\")
        (gen / "tb6a.tex").write_text("\n".join([
            "% generated by experiments/countdown/tb6_e16.py --latex; do not edit",
            "\\begin{table*}[t]", "\\centering", "\\small",
            r"\caption{Alignment measurements and prediction audit "
            r"($\times10^{-4}$; median [min, max] across five seeds). "
            r"\emph{Later reference} uses the full-rank synthetic fit at $N/P$, "
            r"selected after the runs. \emph{Original prediction} was fixed before "
            r"each run and uses the variant's fit at $N/d_{\mathrm{samp}}$ "
            r"(with the advance correction at 1.5B described in the text). "
            r"Ratios are measured / predicted. The second 0.5B batch reuses the same "
            r"perturbations and changes only the puzzle numbers in the prompts; it is "
            r"a sensitivity check, not an independent replication.}",
            "\\label{tab:tb6-audit}",
            "\\begin{tabular}{lllrrrrr}", "\\toprule",
            " & & & & \\multicolumn{2}{c}{later reference} & "
            "\\multicolumn{2}{c}{original prediction} \\\\",
            "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
            "model & variant & $N$ & measured & pred. & ratio & pred. & ratio \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]))
        print(f"wrote {gen / 'tb6.tex'} and {gen / 'tb6a.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
