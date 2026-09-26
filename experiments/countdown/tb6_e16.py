#!/usr/bin/env python
"""The alignment tables: E15 and E16 at equal population, and against the predictions.

    python tb6_e16.py            # print the grid
    python tb6_e16.py --latex    # also write paper/generated/tb6.tex and tb6a.tex

tb6 is the main-text table: medians only, one row per (model, batch, N), one column per
arm, and the synthetic reference once per row. tb6a is the appendix audit: every cell
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
        # Main text: medians at equal population, one column per arm.
        cols = ["seed, mirrored", "rank 1", "rank 4", "rank 16", "rank 1, unpaired"]
        cell = {(l[0], l[2], l[1]): l[3] for l in lines}
        ref = {(l[0], l[2]): l[5] for l in lines}
        short = {"0.5B": "0.5B", "0.5B, second batch": "0.5B, batch 2", "1.5B": "1.5B"}
        rows = []
        for label, n in dict.fromkeys((l[0], l[2]) for l in lines):
            vals = [f"${cell[(label, n, c)] * 1e4:.2f}$" if (label, n, c) in cell else "--"
                    for c in cols]
            rows.append(f"{short[label]} & {n} & " + " & ".join(vals)
                        + f" & ${ref[(label, n)] * 1e4:.2f}$ \\\\")
        (gen / "tb6.tex").write_text("\n".join([
            "% generated by experiments/countdown/tb6_e16.py --latex; do not edit",
            "\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{3.2pt}",
            "\\caption{Cosine between one ES update and the exact gradient of the",
            "  next-token loss ($\\times 10^{-4}$, median over 5 perturbation seeds), at equal",
            "  population; \\emph{1 unp.} is rank 1 without mirrored pairs.",
            "  \\emph{ref.}: the mirrored full-rank fit from the synthetic block",
            "  (Figure~\\ref{fig:f5}) at the same $N/P$, a post hoc reference; unpaired",
            "  sampling should sit about $\\sqrt2$ above it. Batch 2 repeats two",
            "  configurations on freshly drawn prompts. Ranges and the preregistered",
            "  predictions are in Table~\\ref{tab:tb6-audit}.}",
            "\\label{tab:tb6}",
            "\\begin{tabular}{lrrrrrrr}", "\\toprule",
            " & & \\multicolumn{5}{c}{rank} & \\\\",
            "\\cmidrule(lr){3-7}",
            "model & $N$ & full & 1 & 4 & 16 & 1 unp. & ref. \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))
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
            "\\caption{The alignment measurements against both predictions",
            "  ($\\times 10^{-4}$; median [min, max] over 5 perturbation seeds).",
            "  \\emph{synthetic reference}: the mirrored full-rank fit from the synthetic",
            "  block at $N/P$, the same curve for every arm, found after the runs.",
            "  \\emph{preregistered}: the prediction committed before each run, the arm's own",
            "  synthetic fit at $N/d_{\\mathrm{samp}}$, and for rank 1 at 1.5B that fit times",
            "  the 0.53 correction measured at 0.5B. The synthetic study has no rank-16 fit.",
            "  Ratios are measured over predicted.}",
            "\\label{tab:tb6-audit}",
            "\\begin{tabular}{lllrrrrr}", "\\toprule",
            " & & & & \\multicolumn{2}{c}{synthetic reference} & "
            "\\multicolumn{2}{c}{preregistered} \\\\",
            "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
            "model & arm & $N$ & measured & pred. & ratio & pred. & ratio \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]))
        print(f"wrote {gen / 'tb6.tex'} and {gen / 'tb6a.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
