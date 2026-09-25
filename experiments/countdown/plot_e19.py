#!/usr/bin/env python
"""F7c: E19's N=16 arms beside the E13 N=30 arms.

    python plot_e19.py [--e13 results/e13-a100-<date>]

Share of held-out puzzles solved against training sample evaluations for full
rank and rank 1 at N=30 (E13, solid) and N=16 (E19, dashed), mean over three
seeds with a min-max band. Units differ per arm and are applied per arm: an E13
update scores 30 x 8 = 240 completions, an E19 update 16 x 8 = 128, so at the
same 120,000 samples N=16 takes 937 updates to N=30's 500. E19 was frozen as a
positive control (27% lower alignment per update at N=16); on this axis it is not
one, since N=16 also gets 1.9x the updates. Prints, per arm, the final reward and
solve rate and the first evaluation whose mean reward reaches 0.15.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
E19 = HERE / "results" / "e19-a100"
ARMS = [  # (results dir, stem, evals per update, color, line style, label)
    (None, "es-mirrored-seed", 240, "#eb6834", "-", "full rank, N=30"),
    (None, "es-mirrored-lr1", 240, "#1baf7a", "-", "rank 1, N=30"),
    (E19, "es-n16-seed", 128, "#eb6834", "--", "full rank, N=16"),
    (E19, "es-n16-lr1", 128, "#1baf7a", "--", "rank 1, N=16"),
]


def curves(root: Path, stem: str, unit: int, field: str = "eval_reward"):
    by_x: dict = {}
    for s in (0, 1, 2):
        for row in map(json.loads, (root / f"{stem}-s{s}-eval.jsonl").open()):
            by_x.setdefault(row["generation"] * unit, []).append(row[field])
    return sorted(by_x.items())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e13", type=Path, default=HERE / "results" / "e13-a100-2026-08-22-clean")
    args = ap.parse_args(argv)
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for root, stem, unit, color, ls, label in ARMS:
        root = root or args.e13
        if not (root / f"{stem}-s0-eval.jsonl").exists():
            print(f"missing {stem} under {root}")
            continue
        pts = curves(root, stem, unit, "eval_solved")
        xs = [x for x, _ in pts]
        mean = [100 * statistics.mean(v) for _, v in pts]
        ax.plot(xs, mean, color=color, ls=ls, lw=1.5, marker="o", ms=3, label=label, zorder=3)
        ax.fill_between(xs, [100 * min(v) for _, v in pts], [100 * max(v) for _, v in pts],
                        color=color, alpha=0.12, lw=0, zorder=2)
        rew = curves(root, stem, unit)
        hit = next((x for x, v in rew if statistics.mean(v) >= 0.15), None)
        print(f"{label:16s} final reward {statistics.mean(rew[-1][1]):.3f} "
              f"[{min(rew[-1][1]):.3f}, {max(rew[-1][1]):.3f}]  solved {mean[-1]:.2f}%"
              f"  first mean reward>=0.15 at {hit} samples")
    ax.set_xlabel("training sample evaluations")
    ax.set_ylabel("held-out puzzles solved (%)")
    ax.set_xticks([0, 24000, 48000, 72000, 96000, 120000])
    ax.set_xticklabels(["0", "24k", "48k", "72k", "96k", "120k"])
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, color="#e6e6e3", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = HERE / "figures" / "f7c-e19-n16.png"
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
