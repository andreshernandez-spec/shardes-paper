#!/usr/bin/env python
"""F1 and F2b, the cross-platform figures, from the committed sweeps.

    python plot_paper.py      # figures/f1-scaling.png, figures/f2b-crossover-vs-d.png

Sources: `results-consistent` + `results-qiu` (8x A100) and `results-tpu-v5e8`
(v5e-8), the same directories M1-M3 draw from; nothing here is a new measurement,
only an assembly, so any number can be checked against the per-platform figures.

F2b is the paper's placement figure: t_B / t_A against device count, one line per
(strategy, d, N) cell. It replaced F2, a D=8 heatmap whose four cells per panel
left most of each panel empty. The main figure shows dense, seed and mirrored rank 1 on both platforms.
Pairing variants remain in the appendix scaling grids.

F1 is the opener, so it shows one representative cell per mode rather than
M1/M2's full grid: the largest cell both platforms ran (d=2048: strong
N=256, weak N/device=32). Strong shows seconds per generation, log-log, no
per-panel ideal line (M1's docstring: with several series a time-panel ideal
reads as a target the others miss); weak shows members/second with the dashed
linear ideal the figure table asks for, anchored per series family at its own
D=1 point via the same convention as M2. Full grids stay in the M1/M2 figures
the appendix cites.

Draft for Andres: the F1 cell choice (largest common) and the decision to show
time + throughput rather than efficiency in the opener are the two judgement
calls; both are one-line changes here.
"""

from __future__ import annotations

import collections
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot import HUES, INK, LABELS, MARKERS, MUTED, _style, load  # noqa: E402

PLATFORMS = [
    ("8x A100", [HERE / "results-consistent", HERE / "results-qiu"]),
    ("TPU v5e-8", [HERE / "results-tpu-v5e8"]),
]
#: Only what both platforms ran; mirrored_seed is GPU-only (results-qiu).
STRATEGIES = ["iid_gaussian", "seed_regenerated", "mirrored_lr1", "lowrank_r1"]


def f2b(platform_rows: list[tuple[str, list[dict]]], out: pathlib.Path) -> None:
    """Main variants by hardware and width, with per-configuration repeat ranges."""
    main_strategies = ("iid_gaussian", "seed_regenerated", "mirrored_lr1")
    names = {"iid_gaussian": "stored full rank (dense)",
             "seed_regenerated": "regenerated full rank (seed)", "mirrored_lr1": "rank 1"}
    fig, axes = plt.subplots(2, len(platform_rows), figsize=(8.6, 5.6), sharex=True, sharey=True)
    for j, (name, rows) in enumerate(platform_rows):
        cells = collections.defaultdict(dict)
        repeats = collections.defaultdict(dict)
        for row in rows:
            c = row["config"]
            if c["mode"] == "strong":
                key = (c["strategy"], c["d_model"], c["population"])
                cells[key][c["devices"], c["how"]] = row["seconds_median"]
                repeats[key][c["devices"], c["how"]] = row["seconds_all"]
        for i, width in enumerate((512, 2048)):
            ax = axes[i, j]
            pops = sorted({n for _, d, n in cells if d == width})
            for (strategy, d, n), by in sorted(cells.items()):
                if d != width or strategy not in main_strategies:
                    continue
                ds = [dev for dev in (2, 4, 8) if (dev, "A") in by and (dev, "B") in by]
                vals = [by[dev, "B"] / by[dev, "A"] for dev in ds]
                lows, highs = [], []
                for dev, val in zip(ds, vals):
                    a, b = (repeats[strategy, d, n][dev, h] for h in "AB")
                    lows.append(val - min(b) / max(a))
                    highs.append(max(b) / min(a) - val)
                ax.errorbar(ds, vals, yerr=[lows, highs], color=HUES[strategy],
                            marker="o" if n == pops[0] else "s", ms=4.5,
                            mfc="white" if n == pops[0] else HUES[strategy],
                            lw=1.5, elinewidth=0.7, capsize=2)
            ax.axhline(1, color=MUTED, lw=1)
            ax.set(xscale="log", yscale="log", ylim=(0.16, 2.05))
            ax.set_xticks([2, 4, 8], labels=["2", "4", "8"])
            ax.set_yticks([0.2, 0.5, 1, 2], labels=["0.2", "0.5", "1", "2"])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_title(f"{name}, width {width}\nN = {pops[0]} (circles), {pops[-1]} (squares)",
                         fontsize=10, loc="left")
            ax.text(0.03, 0.94, "replication faster", transform=ax.transAxes, fontsize=8,
                    color=MUTED, va="top")
            ax.text(0.03, 0.04, "splitting faster", transform=ax.transAxes, fontsize=8, color=MUTED)
            _style(ax)
            if i == 1:
                ax.set_xlabel("devices")
            if j == 0:
                ax.set_ylabel("split / replicated time")
        unresolved = collections.defaultdict(list)
        for (strategy, d, n), by in repeats.items():
            if strategy not in STRATEGIES or (8, "A") not in by or (8, "B") not in by:
                continue
            a, b = by[8, "A"], by[8, "B"]
            family = "rank 1" if strategy in ("mirrored_lr1", "lowrank_r1") else "full rank"
            unresolved[family, d].append(min(b) / max(a) <= 1 <= max(b) / min(a))
        for (family, d), flags in sorted(unresolved.items()):
            print(f"{name} D=8 {family} d={d}: {sum(flags)} of {len(flags)} unresolved")
    handles = [plt.Line2D([], [], color=HUES[s], lw=1.5) for s in main_strategies]
    fig.legend(handles, [names[s] for s in main_strategies], frameon=False, ncol=3,
               loc="lower center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "f2b-crossover-vs-d.png", dpi=200, metadata={"Date": None, "Software": None})
    plt.close(fig)


def f1(platform_rows: list[tuple[str, list[dict]]], out: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, len(platform_rows), figsize=(4.4 * len(platform_rows), 6.8),
                             squeeze=False)
    for j, (name, rows) in enumerate(platform_rows):
        ax_s, ax_w = axes[0][j], axes[1][j]

        strong: dict = collections.defaultdict(dict)
        weak: dict = collections.defaultdict(dict)
        for r in rows:
            c = r["config"]
            if c["strategy"] not in STRATEGIES or c["d_model"] != 2048:
                continue
            if c["mode"] == "strong" and c["population"] == 256:
                strong[(c["strategy"], c["how"])][c["devices"]] = r["seconds_median"]
            if c["mode"] == "weak" and c["population"] // c["devices"] == 32:
                weak[(c["strategy"], c["how"])][c["devices"]] = (
                    c["population"] / r["seconds_median"])

        for (s, how), by_d in sorted(strong.items()):
            ds = sorted(by_d)
            ax_s.plot(ds, [by_d[d] for d in ds], marker=MARKERS[how], ms=6, lw=1.8,
                      color=HUES[s], label=f"{s} / {how}")
        # Parallel efficiency, each series against ITS OWN smallest measured
        # device count: eff(D) = (T(D0) * D0) / (T(D) * D) in throughput terms.
        # The panel used to plot absolute members/second against one ideal
        # line anchored to the alphabetically first series, which made
        # cross-series comparisons against that line meaningless (a faster
        # series crossed it without any superlinear scaling). Efficiency
        # gives every series the same ideal (1.0) whatever its D0, which
        # also handles arms whose D=1 cell is out of memory.
        for (s, how), by_d in sorted(weak.items()):
            ds = sorted(by_d)
            d0 = ds[0]
            ax_w.plot(ds, [(by_d[d] / d) / (by_d[d0] / d0) for d in ds],
                      marker=MARKERS[how], ms=6, lw=1.8, color=HUES[s])
            print(f"  weak {name} {s}/{how}: D0={d0} "
                  + " ".join(f"eff(D={d})={(by_d[d] / d) / (by_d[d0] / d0):.2f}"
                             for d in ds))
        ax_w.axhline(1.0, ls="--", lw=1.4, color=MUTED,
                     label="ideal" if j == 0 else None)

        ticks = sorted({d for by_d in strong.values() for d in by_d})
        for ax, ylab, ylog in ((ax_s, "seconds / generation", True),
                               (ax_w, "weak-scaling efficiency", False)):
            ax.set(xscale="log")
            if ylog:
                ax.set_yscale("log")
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(d) for d in ticks])
            ax.xaxis.set_minor_locator(NullLocator())
            _style(ax)
            if j == 0:
                ax.set_ylabel(ylab)
        ax_w.set_ylim(0.0, 1.25)
        ax_s.set_title(f"{name}\nstrong: d=2048, N=256", color=INK, loc="left",
                       fontsize=10)
        ax_w.set_title("weak: d=2048, N/device=32", color=INK, loc="left", fontsize=10)
        ax_w.set_xlabel("devices")

    handles: dict = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), frameon=False, fontsize=8,
               labelcolor=MUTED, loc="center left", bbox_to_anchor=(0.99, 0.5))
    fig.suptitle("F1  scaling by platform", color=INK, x=0.02, ha="left", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "f1-scaling.png", dpi=200, bbox_inches="tight", metadata={"Date": None, "Software": None})
    plt.close(fig)
    print(out / "f1-scaling.png")


def main() -> int:
    platform_rows = [(name, load(dirs)) for name, dirs in PLATFORMS]
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    f2b(platform_rows, out)
    f1(platform_rows, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
