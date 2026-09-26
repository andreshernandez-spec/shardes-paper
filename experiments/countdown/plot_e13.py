#!/usr/bin/env python
"""Regenerate F7 from results/e13-a100-2026-08-22-clean. No hand-edited numbers.

    python plot_e13.py        # figures/f7-e13-heldout.png

F7: the share of the 2000 held-out puzzles solved, against scored training
completions (left) and against cumulative update time (right). One curve per ES
arm, mean over seeds 0-2 with a min-max band, a marker at every evaluation, the
base model as a dashed floor. One ES update scores N * puzzles_per_gen = 240
completions.

The reward is 0.9 x solved + 0.1 x well formed. The formatting half is not drawn:
every arm formats 99% or more of its answers by the first evaluation, which this
prints, so the solve rate carries everything that differs between arms.

The time axis is each run's own per-update seconds, as each arm was configured
(the full-rank arm scores its members in chunks of five, the low-rank arms in one
batch). Updates 0 and 1 are left out of it: both compile, 322-635 s each against a
steady 2.6-4.4 s. Held-out decoding is left out too. The evaluation at generation g
runs before update g (run_es.py evaluates at the top of the loop), so its x is the
time of the updates before g; x per evaluation is the mean over the three seeds.

GRPO is not drawn. It runs in another framework with another decoder (its base
model scores 0.037 against the ES decoder's 0.054 on the same weights), and the
paper does not compare it; its evals stay in results/e13-a100-2026-08-17.

Prints the per-arm means at every evaluation, which is where the text's numbers
come from.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this has to work over ssh and in a notebook driver
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "e13-a100-2026-08-22-clean"
# The GRPO baseline was not part of the clean ES rerun; its evals stay with the
# 08-17 campaign (results/grpo-a100-2026-08-17 holds the trainer state).
GRPO = HERE / "results" / "e13-a100-2026-08-17"


def where(stem):
    return GRPO if stem == "grpo" else RESULTS
FIGURES = HERE / "figures"

EVALS_PER_UNIT = 240  # per ES generation and per GRPO step alike; see module docstring

# Colour per arm, fixed so the same arm is the same colour in every figure and in any
# talk that reuses them.
# Seed and rank 1 keep their hues from the placement figures (orange, aqua).
ARM_STYLE = {
    "es-mirrored-seed": ("#eb6834", "seed, mirrored (full rank)"),
    "es-mirrored-lr1": ("#1baf7a", "rank 1"),
    "es-mirrored-lr4": ("#4a3aa7", "rank 4"),
    "es-mirrored-lr16": ("#e87ba4", "rank 16"),
    "es-lr1-frozen-embed": ("#008300", "rank 1, embedding frozen"),
}


def curves(stem: str, xkey: str, field: str = "eval_reward"):
    """[(x_evals, [field per seed]) ...] over the arm's three seed files."""
    by_x = {}
    for s in (0, 1, 2):
        for row in map(json.loads, (where(stem) / f"{stem}-s{s}-eval.jsonl").open()):
            by_x.setdefault(row[xkey] * EVALS_PER_UNIT, []).append(row[field])
    return sorted(by_x.items())


def seconds_before(stem: str) -> dict:
    """generation -> [cumulative update seconds before it, per seed]; see docstring."""
    out: dict = {}
    for s in (0, 1, 2):
        cum, c = {}, 0.0
        for r in map(json.loads, (where(stem) / f"{stem}-s{s}-log.jsonl").open()):
            cum[r["generation"]] = c
            if r["generation"] > 1:  # updates 0 and 1 compile
                c += r["seconds"]
        for e in map(json.loads, (where(stem) / f"{stem}-s{s}-eval.jsonl").open()):
            # the last evaluation runs after the last update, at the total
            out.setdefault(e["generation"], []).append(cum.get(e["generation"], c))
    return out


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    floor = statistics.mean(curves("es-mirrored-seed", "generation", "eval_solved")[0][1])
    fmt0 = curves("es-mirrored-seed", "generation", "eval_format")[0][1]
    print(f"  base model: solved {100 * floor:.2f}%, well formed "
          f"{100 * statistics.mean(fmt0):.1f}%")
    for stem, (color, label) in ARM_STYLE.items():
        pts = curves(stem, "generation", "eval_solved")
        mean = [100 * statistics.mean(v) for _, v in pts]
        lo = [100 * min(v) for _, v in pts]
        hi = [100 * max(v) for _, v in pts]
        secs = seconds_before(stem)
        t = [statistics.mean(secs[x // EVALS_PER_UNIT]) for x, _ in pts]
        for ax, xs in ((axes[0], [x for x, _ in pts]), (axes[1], t)):
            ax.plot(xs, mean, color=color, lw=1.5, marker="o", ms=3, label=label, zorder=3)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0, zorder=2)
        print(f"  {label:28s} " + " ".join(
            f"{x // 1000}k/{s:.0f}s:{m:.2f}" for (x, _), s, m in zip(pts, t, mean)))
        print(f"    at {pts[1][0] // 1000}k: solved {lo[1]:.2f}-{hi[1]:.2f}%, final "
              f"{mean[-1]:.2f}% [{lo[-1]:.2f}, {hi[-1]:.2f}]")
        fmt = curves(stem, "generation", "eval_format")[1][1]
        print(f"    well formed at the first evaluation: {100 * min(fmt):.1f}% or more")
        steady = [r["seconds"] for s in (0, 1, 2)
                  for r in map(json.loads, (where(stem) / f"{stem}-s{s}-log.jsonl").open())
                  if r["generation"] > 1]
        print(f"    steady-state s/update (updates 2 on, all seeds): "
              f"{statistics.mean(steady):.2f}")
    for ax in axes:
        ax.axhline(100 * floor, color="#888888", lw=1.0, ls="--", zorder=1)
        ax.annotate("base model", (0.62, 100 * floor), xycoords=("axes fraction", "data"),
                    ha="left", va="bottom", fontsize=8, color="#52514e")
        ax.grid(True, color="#e6e6e3", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_xlabel("training sample evaluations")
    axes[0].set_ylabel("held-out puzzles solved (%)")
    axes[0].set_xlim(0, 500 * EVALS_PER_UNIT)
    axes[0].set_xticks([0, 24000, 48000, 72000, 96000, 120000])
    axes[0].set_xticklabels(["0", "24k", "48k", "72k", "96k", "120k"])
    axes[1].set_xlabel("cumulative update time (s)")
    axes[1].set_xlim(0, None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = FIGURES / "f7-e13-heldout.png"
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
