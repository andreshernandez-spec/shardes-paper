#!/usr/bin/env python
"""F9: the real-model contraction crossover as a figure, from results-e17b.

    python plot_e17.py            # writes figures/f9-e17-crossover.png

One panel per perturbation arm, x = device count, y = t_B / t_A of median
generation time on a log axis (the block figure's axis), one line per population.
Below 1: the all-reduce placement (B) wins. A cell where either placement ran
out of memory has no point; it used to be a hollow marker at y=0, which read as a
tie. Prints every plotted value, and for each one the range of the ratio over
repeats, min(t_B)/max(t_A) to max(t_B)/min(t_A), flagged where it contains 1 (the
block figure's criterion). Falls back to results-e17 when results-e17b does not
exist yet.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results-e17b" if (HERE / "results-e17b").exists() else HERE / "results-e17"
FIGURES = HERE / "figures"
ARMS = [("mirrored_seed", "seed, mirrored"), ("mirrored_lr1", "rank 1"),
        ("mirrored_lr4", "rank 4"), ("mirrored_lr16", "rank 16")]
DEVICES = (1, 2, 4, 8)
#: Ordinal blue steps far enough apart to tell at print size, plus a marker each.
SHADES = {32: "#0d366b", 64: "#256abf", 128: "#5598e7", 240: "#9ec5f4"}
SHAPES = {32: "o", 64: "s", 128: "^", 240: "D"}


def cell(strategy, how, n, d):
    f = RESULTS / f"s={strategy}__how={how}__N={n}__D={d}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text()).get("seconds_median", "oom")


def repeat_range(strategy, n, d):
    """min(t_B)/max(t_A) to max(t_B)/min(t_A) over the recorded repeats, or None."""
    a, b = (json.loads((RESULTS / f"s={strategy}__how={h}__N={n}__D={d}.json").read_text())
            .get("seconds_all") for h in "AB")
    return (min(b) / max(a), max(b) / min(a)) if a and b else None


def main() -> None:
    pops = sorted({int(f.name.split("N=")[1].split("__")[0]) for f in RESULTS.glob("s=*.json")})
    arms = [(s, l) for s, l in ARMS if any(RESULTS.glob(f"s={s}__*.json"))]
    fig = plt.figure(figsize=(2.4 * len(arms), 3.8))
    grid = fig.add_gridspec(2, len(arms), height_ratios=(3.2, 1), hspace=0.42)
    axes = [[]]
    memory_axes = []
    for j in range(len(arms)):
        axes[0].append(fig.add_subplot(grid[0, j], sharey=axes[0][0] if j else None))
        memory_axes.append(fig.add_subplot(grid[1, j]))
    for ax, (strategy, label) in zip(axes[0], arms):
        for n in pops:
            xs, ys = [], []
            for d in DEVICES:
                a, b = cell(strategy, "A", n, d), cell(strategy, "B", n, d)
                if a is None or b is None or a == "oom" or b == "oom":
                    continue
                xs.append(d)
                ys.append(b / a)
            if xs:
                ax.plot(xs, ys, marker=SHAPES.get(n, "o"), ms=4.5, lw=1.5,
                        color=SHADES.get(n, "#444444"), label=f"N = {n}")
                print(f"  {label:15s} N={n:<4d} " + " ".join(
                    f"D={d}:{y:.3f}" for d, y in zip(xs, ys)))
                for d in xs:
                    rr = repeat_range(strategy, n, d)
                    if rr and d > 1:
                        print(f"    D={d} repeats {rr[0]:.3f}-{rr[1]:.3f}"
                              + ("  contains 1" if rr[0] <= 1 <= rr[1] else ""))
        ax.axhline(1.0, color="#52514e", lw=1.0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(0.6, 1.9)
        ax.set_yticks([0.7, 0.8, 1.0, 1.25, 1.5])
        ax.set_yticklabels(["0.7", "0.8", "1", "1.25", "1.5"])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xticks(DEVICES)
        ax.set_xticklabels([str(d) for d in DEVICES])
        ax.set_title(label, fontsize=11, loc="left")
        ax.set_xlabel("devices D")
        ax.grid(True, color="#e6e6e3", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    for mem, (strategy, _) in zip(memory_axes, arms):
        for i, n in enumerate(pops):
            for j, d in enumerate(DEVICES):
                a, b = cell(strategy, "A", n, d), cell(strategy, "B", n, d)
                missing = a is None or b is None
                oom = a == "oom" or b == "oom"
                mem.text(j, i, "?" if missing else "x" if oom else ".",
                         ha="center", va="center", color="#777777" if oom else SHADES[n],
                         fontsize=10 if oom else 15)
        mem.set(xlim=(-0.5, 3.5), ylim=(len(pops)-0.5, -0.5))
        mem.set_xticks(range(4), labels=DEVICES)
        mem.set_yticks(range(len(pops)), labels=[str(n) for n in pops])
        mem.tick_params(length=0, labelsize=8)
        mem.spines[["top", "right", "bottom", "left"]].set_visible(False)
        mem.set_xlabel("devices; x = OOM, dot = fits", fontsize=8)
    memory_axes[0].set_ylabel("population N", fontsize=8)
    axes[0][0].set_ylabel("split / replicated time")
    axes[0][0].text(1.05, 1.8, "replication faster", color="#52514e", fontsize=8, va="top")
    axes[0][0].text(1.05, 0.62, "splitting faster", color="#52514e", fontsize=8)
    handles = {}
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), frameon=False, fontsize=10,
               loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.2, wspace=0.28)
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "f9-e17-crossover.png"
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
