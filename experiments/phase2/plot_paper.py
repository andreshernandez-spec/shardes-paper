#!/usr/bin/env python
"""F1 and F2b, the cross-platform figures, from the committed sweeps.

    python plot_paper.py      # figures/f1-scaling.png, figures/f2b-crossover-vs-d.png

Sources: `results-consistent` + `results-qiu` (8x A100) and `results-tpu-v5e8`
(v5e-8), the same directories M1-M3 draw from; nothing here is a new measurement,
only an assembly, so any number can be checked against the per-platform figures.

F2b is the paper's placement figure: t_B / t_A against device count, one line per
(strategy, d, N) cell. It replaced F2, a D=8 heatmap whose four cells per panel
left most of each panel empty. Only the four strategies present on BOTH platforms
are drawn (`mirrored_seed` exists only in the GPU sweep, results-qiu).

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
    """The placement result on the block: t_B / t_A against device count, per platform.

    One line per (strategy, d, N) configuration of the strong-scaling sweep, solid at
    d=2048 and dashed at d=512, open markers for the smaller population of each width and
    filled for the larger, from D=2. Plotted as the ratio on a log axis, so 0.5 and 2 are
    the same distance from a tie. At D=1 the two placements do the same work, so the
    spread of their ratio there is drawn as a grey band: a difference inside it is not
    one the sweep can tell from a tie. Prints every plotted value and the band, which is
    where the text's ranges come from.
    """
    fig, axes = plt.subplots(1, len(platform_rows), figsize=(4.4 * len(platform_rows), 3.5),
                             sharey=True, squeeze=False)
    for j, (name, rows) in enumerate(platform_rows):
        ax = axes[0][j]
        cells: dict = collections.defaultdict(dict)
        for r in rows:
            c = r["config"]
            if c["mode"] != "strong" or c["strategy"] not in STRATEGIES:
                continue
            cells[(c["strategy"], c["d_model"], c["population"])][
                (c["devices"], c["how"])] = r["seconds_median"]
        larger = {d: max(n for _, dd, n in cells if dd == d) for _, d, _ in cells}
        at_one = [by[(1, "B")] / by[(1, "A")] for by in cells.values()
                  if (1, "A") in by and (1, "B") in by]
        band = (min(at_one), max(at_one))
        ax.axhspan(*band, color="#d6d5d0", lw=0, zorder=0)
        print(f"  {name:10s} D=1 band {band[0]:.3f}-{band[1]:.3f} over {len(at_one)}")
        for (s, d, n), by in sorted(cells.items()):
            pts = [(dev, by[(dev, "B")] / by[(dev, "A")])
                   for dev in sorted({dev for dev, _ in by})
                   if dev > 1 and (dev, "A") in by and (dev, "B") in by]
            if len(pts) < 2:
                continue
            filled = n == larger[d]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4.5, lw=1.4,
                    ls="-" if d == 2048 else "--", color=HUES[s], alpha=0.9,
                    mfc=HUES[s] if filled else "white", mew=1.2)
            print(f"  {name:10s} {LABELS[s]:17s} d={d:<5d} N={n:<5d} "
                  + " ".join(f"D={dev}:{v:.3f}" for dev, v in pts))
        ax.axhline(1.0, color=MUTED, lw=1.0)
        ax.text(2.05, 1.9, "replicated (A) faster", color=MUTED, fontsize=7, va="top")
        ax.text(2.05, 0.19, "all-reduce (B) faster", color=MUTED, fontsize=7, va="bottom")
        ax.set(xscale="log", yscale="log", xlabel="devices D", ylim=(0.17, 2.0))
        ax.set_xticks([2, 4, 8])
        ax.set_xticklabels(["2", "4", "8"])
        ax.set_yticks([0.2, 0.3, 0.5, 0.7, 1.0, 1.5])
        ax.set_yticklabels(["0.2", "0.3", "0.5", "0.7", "1", "1.5"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        _style(ax)
        ax.set_title(name, color=INK, fontsize=10, loc="left")
        if j == 0:
            ax.set_ylabel("$t_B / t_A$")
    # Two rows: what a line is (arm, width), then what a marker and the band are.
    lines = [plt.Line2D([], [], color=HUES[s], lw=1.4) for s in STRATEGIES]
    lines += [plt.Line2D([], [], color=MUTED, lw=1.4, ls=ls) for ls in ("-", "--")]
    fig.legend(lines, [LABELS[s] for s in STRATEGIES] + ["d = 2048", "d = 512"],
               frameon=False, fontsize=8, labelcolor=INK, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 0.03))
    marks = [plt.Line2D([], [], color=MUTED, lw=0, marker="o", ms=4.5, mew=1.2, mfc=mfc)
             for mfc in ("white", MUTED)]
    marks += [plt.Rectangle((0, 0), 1, 1, color="#d6d5d0", lw=0)]
    fig.legend(marks, ["smaller N (256 at d = 512, 128 at d = 2048)",
                       "larger N (1024, 256)", "spread at D = 1"],
               frameon=False, fontsize=8, labelcolor=INK, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    fig.savefig(out / "f2b-crossover-vs-d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(out / "f2b-crossover-vs-d.png")


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
    fig.savefig(out / "f1-scaling.png", dpi=200, bbox_inches="tight")
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
