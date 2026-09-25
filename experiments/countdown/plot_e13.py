#!/usr/bin/env python
"""Regenerate F7 and F7b from results/e13-a100-2026-08-22-clean. No hand-edited numbers.

    python plot_e13.py        # figures/f7-e13-heldout.png, f7b-e13-wallclock.png

F7: what the held-out evaluations measure, split into its two parts, against
sample evaluations: the share of the 2000 held-out puzzles solved, and the share
of answers that are well formed. The reward is 0.9 x solved + 0.1 x formatted, so
the two panels carry all of it, and they show what the reward hides: every arm
learns the format within the first 50 updates, and the plateau is ~6% solved.
One curve per ES arm, mean over seeds 0-2 with a min-max band, a marker at every
evaluation, the base model as a dashed floor. One ES update scores
N * puzzles_per_gen = 240 completions.

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


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    panels = (("eval_solved", "held-out puzzles solved (%)"),
              ("eval_format", "answers well formed (%)"))
    for ax, (field, ylabel) in zip(axes, panels):
        floor = statistics.mean(curves("es-mirrored-seed", "generation", field)[0][1])
        ax.axhline(100 * floor, color="#888888", lw=1.0, ls="--", zorder=1)
        ax.annotate("base model", (0.62, 100 * floor), xycoords=("axes fraction", "data"),
                    ha="left", va="bottom", fontsize=8, color="#52514e")
        for stem, (color, label) in ARM_STYLE.items():
            pts = curves(stem, "generation", field)
            xs = [x for x, _ in pts]
            mean = [100 * statistics.mean(v) for _, v in pts]
            ax.plot(xs, mean, color=color, lw=1.5, marker="o", ms=3, label=label, zorder=3)
            ax.fill_between(xs, [100 * min(v) for _, v in pts], [100 * max(v) for _, v in pts],
                            color=color, alpha=0.15, lw=0, zorder=2)
            print(f"  {field:12s} {label:28s} " + " ".join(
                f"{x // 1000}k:{m:.2f}" for x, m in zip(xs, mean)))
            if field == "eval_solved":
                rew = curves(stem, "generation")[1]
                print(f"    at {rew[0] // 1000}k: reward {statistics.mean(rew[1]):.3f} "
                      f"[{min(rew[1]):.3f}, {max(rew[1]):.3f}], solved "
                      f"{100 * min(pts[1][1]):.2f}-{100 * max(pts[1][1]):.2f}%")
        ax.set_xlabel("training sample evaluations")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 500 * EVALS_PER_UNIT)
        ax.set_xticks([0, 24000, 48000, 72000, 96000, 120000])
        ax.set_xticklabels(["0", "24k", "48k", "72k", "96k", "120k"])
        ax.grid(True, color="#e6e6e3", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_ylim(0, 105)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = FIGURES / "f7-e13-heldout.png"
    fig.savefig(out, dpi=200)
    print(out)


def wallclock() -> None:
    """F7b: the same held-out curves against cumulative steady-state accelerator time.

    ES arms only: GRPO runs in a different framework with a different decoder, so
    its wall clock is not commensurate and is deliberately absent. Generations 0
    AND 1 are both excluded from the cumulative time: both are compilation-scale
    events in every arm and seed (335--671 s against steady-state 2.4--4.3 s
    per generation).
    Held-out evaluation decode is excluded on both axes as measurement overhead.
    The eval at generation g runs BEFORE update g (run_es.py evaluates at the
    top of the loop), so its x is the cumulative time of updates strictly
    before g. x per eval point is the mean cumulative time over the three
    seeds; the band is min--max of reward.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for stem, (color, label) in ARM_STYLE.items():
        if stem == "grpo":
            continue
        xs_by_gen: dict = {}
        rew_by_gen: dict = {}
        for s in (0, 1, 2):
            logs = [json.loads(l) for l in
                    (where(stem) / f"{stem}-s{s}-log.jsonl").open()]
            cum, c = {}, 0.0
            for r in logs:
                if r["generation"] <= 1:
                    continue  # compilation lives in g0 and g1; see docstring
                # eval at g precedes update g: charge it the time BEFORE g.
                cum[r["generation"]] = c
                c += r["seconds"]
            for e in map(json.loads, (where(stem) / f"{stem}-s{s}-eval.jsonl").open()):
                g = e["generation"]
                if g <= 1:
                    continue
                # the final eval runs after the last update, at a generation
                # one past the last logged one; it lands at the total time
                xs_by_gen.setdefault(g, []).append(cum.get(g, c))
                rew_by_gen.setdefault(g, []).append(e["eval_reward"])
        gens = sorted(xs_by_gen)
        xs = [statistics.mean(xs_by_gen[g]) for g in gens]
        ax.plot(xs, [statistics.mean(rew_by_gen[g]) for g in gens],
                color=color, lw=1.6, label=label, zorder=3)
        ax.fill_between(xs, [min(rew_by_gen[g]) for g in gens],
                        [max(rew_by_gen[g]) for g in gens],
                        color=color, alpha=0.18, lw=0, zorder=2)
    ax.set_xlabel("cumulative accelerator seconds (steady state)")
    ax.set_ylabel("held-out reward (2000 puzzles, greedy)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "f7b-e13-wallclock.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(FIGURES / "f7b-e13-wallclock.png")


if __name__ == "__main__":
    main()
    wallclock()
