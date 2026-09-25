#!/usr/bin/env python
"""The alignment table: E15 and E16 against the full-rank law and the frozen predictions.

    python tb6_e16.py            # print the grid
    python tb6_e16.py --latex    # also write paper/generated/tb6.tex

Reads only committed artifacts: E1's update--gradient-alignment results (the fits), the
E15 and E16 per-cell JSONs, and the model configs. Two predictions per cell, both from
the committed E1 fits (sigma 1e-3, centered ranks), neither fit to these measurements:

- `law`: the full-rank fit at N/P, the same curve for every arm. This is the post hoc
  reading: at equal population the rank barely matters, so one curve should serve all.
- printed only, `linear`: what a locally linear fitness gives with no fit at all. Centered
  ranks are about Phi(z) - 1/2 of the standardized projection z, so for N << P the cosine
  is corr(U, z) sqrt(N/P) = sqrt(3/pi) sqrt(N/P) unpaired, and 1/sqrt(2) of that for
  mirrored pairs (N/2 directions). Low-rank noise AB^T/sqrt(r) has identity covariance,
  so the rank does not enter.
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

    if args.latex:
        rows, prev = [], None
        for label, name, n, med, cos, pl, pf, _ in lines:
            if prev is not None and label != prev:
                rows.append("\\midrule")
            prev = label
            meas = f"${med * 1e4:.2f}$ [{cos[0] * 1e4:.1f}, {cos[-1] * 1e4:.1f}]"
            fz = (f"${pf * 1e4:.2f}$ & ${med / pf:.2f}$" if pf is not None else "-- & --")
            rows.append(f"{label} & {name} & {n} & {meas} & ${pl * 1e4:.2f}$ & "
                        f"${med / pl:.2f}$ & {fz} \\\\")
        out = HERE.parent.parent / "paper" / "generated" / "tb6.tex"
        out.write_text("\n".join([
            "% generated by experiments/countdown/tb6_e16.py --latex; do not edit",
            "\\begin{table*}[t]", "\\centering", "\\small",
            "\\caption{Cosine between one ES update and the exact teacher-forced-NLL",
            "  gradient ($\\times 10^{-4}$; median [min, max] over 5 perturbation seeds).",
            "  \\emph{law}: the full-rank fit from the synthetic block at $N/P$, the same",
            "  curve for every arm (post hoc). \\emph{frozen}: the prediction committed",
            "  before each run, the arm's own synthetic fit at $N/d_{\\mathrm{samp}}$, and",
            "  for rank 1 at 1.5B that fit times the 0.53 correction measured at 0.5B.",
            "  The second 0.5B batch repeats two cells on freshly drawn prompts. The",
            "  synthetic study has no rank-16 fit. Ratios are measured over predicted.}",
            "\\label{tab:tb6}",
            "\\begin{tabular}{lllrrrrr}", "\\toprule",
            " & & & & \\multicolumn{2}{c}{law} & \\multicolumn{2}{c}{frozen} \\\\",
            "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
            "model & arm & $N$ & measured & pred. & ratio & pred. & ratio \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
