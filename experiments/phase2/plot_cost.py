#!/usr/bin/env python
"""F4: where the low-rank rewrite pays, per platform.

    python plot_cost.py                                 # GPU panels from results-cost
    python plot_cost.py --results results-cost results-cost-tpu-v5e8   # both platforms

One row of panels: per platform, one panel per strategy (seed and rank 1; ranks 4
and 16 are the ablation table's geometric means), each a heatmap over (N, d) of
log10(t_strategy / t_iid_gaussian) at the same shape and dtype. Blue where the
strategy beats the dense baseline, red where the baseline wins, grey at parity.

**Out-of-memory cells are hatched, not coloured.** They used to take the full win
colour, which made a row of "no comparison possible" look like a row of large
wins. Feasibility is a different axis from speed, so it gets a different mark:
diagonal hatching and "dense OOM" where only the baseline does not fit, the
opposite diagonal and "OOM" where the strategy does not, and "both OOM" alone.

Annotations are multipliers ("0.13x"), not log values, because a reader
checking a cell against the table in the paper should not need to exponentiate.
The colour is still log-scaled so 4x-faster and 4x-slower sit symmetrically.

Default dtype is bfloat16: E8 claims cost at the precision a practitioner runs,
and the tensor-core/MXU story C4 is about is bf16-shaped. --dtype float32
draws the other surface from the same records.

Draft for Andres: the two decisions worth a second opinion are (1) ratio vs
absolute per-member time (ratio was chosen because C4 is a claim about a
comparison, and the absolute surfaces are recoverable from the JSONs), and
(2) bf16 as the headline dtype with f32 relegated to a flag.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
INK, MUTED = "#0b0b0b", "#52514e"
OOM = "oom"  # sentinel: the cell was visited and recorded undersized

#: Same poles as plot.py's CROSSOVER and the same reading: blue = the named thing
#: wins, red = the baseline wins, grey = parity.
CMAP = LinearSegmentedColormap.from_list("pays", ["#2a78d6", "#f0efec", "#e34948"])

#: Qiu's storage fix first (it loses throughput at D=1), then EGGROLL's rank 1.
STRATEGIES = ["seed_regenerated", "mirrored_lr1"]
TITLES = {"seed_regenerated": "seed", "mirrored_lr1": "rank 1"}
PLATFORM = {"NVIDIA A100-SXM4-80GB": "one A100 (80 GB)", "TPU v5 lite": "one TPU v5e chip (16 GB)"}


def edges(centers: list[int]) -> np.ndarray:
    """Cell edges halfway between centers in log space, so cells are even on log axes."""
    lc = np.log(np.asarray(centers, float))
    mid = (lc[1:] + lc[:-1]) / 2
    return np.exp(np.concatenate([[lc[0] - (mid[0] - lc[0])], mid,
                                  [lc[-1] + (lc[-1] - mid[-1])]]))
BASELINE = "iid_gaussian"


def load(dirs: list[pathlib.Path]) -> dict:
    """{platform: {(d, N, strategy, dtype): seconds | OOM}}, one platform per dir.

    Undersized records are kept as the OOM sentinel rather than dropped: the
    feasibility boundary is half of what this figure shows. A directory mixing
    two device kinds raises, because a surface stitched from two machines is
    not a surface.
    """
    out: dict = {}
    for d in dirs:
        rows = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
        kinds = {r["env"]["device_kind"] for r in rows}
        if len(kinds) != 1:
            raise SystemExit(f"{d} mixes device kinds: {sorted(kinds)}")
        cells = {}
        for r in rows:
            c = r["config"]
            key = (c["d_model"], c["population"], c["strategy"], c["dtype"])
            cells[key] = OOM if r.get("undersized") else r["seconds_median"]
        out[kinds.pop()] = cells
    return out


def panel(ax, cells: dict, strategy: str, dtype: str, dims, pops, norm) -> None:
    z = np.full((len(dims), len(pops)), np.nan)
    labels = np.full((len(dims), len(pops)), "", dtype=object)
    hatch = np.full((len(dims), len(pops)), "", dtype=object)
    for i, d in enumerate(dims):
        for j, n in enumerate(pops):
            t = cells.get((d, n, strategy, dtype))
            base = cells.get((d, n, BASELINE, dtype))
            if t is None or base is None:
                continue
            if t is not OOM and base is not OOM:
                z[i, j] = np.log10(t / base)
                labels[i, j] = f"{t / base:.2f}x"
            elif t is not OOM:  # the baseline is the one that does not fit
                labels[i, j], hatch[i, j] = "dense\nOOM", "////"
            elif base is not OOM:
                labels[i, j], hatch[i, j] = "OOM", "\\\\\\\\"
            else:
                labels[i, j] = "both\nOOM"
    xe, ye = edges(pops), edges(dims)
    ax.pcolormesh(xe, ye, np.ma.masked_invalid(z), cmap=CMAP, norm=norm, shading="flat",
                  edgecolor="#fcfcfb", linewidth=1.0)
    for i in range(len(dims)):
        for j in range(len(pops)):
            if hatch[i, j]:
                ax.add_patch(plt.Rectangle((xe[j], ye[i]), xe[j + 1] - xe[j],
                                           ye[i + 1] - ye[i], facecolor="#f0efec",
                                           edgecolor="#b5b4ae", hatch=hatch[i, j], lw=0))
            if labels[i, j]:
                ax.text(pops[j], dims[i], labels[i, j], ha="center", va="center",
                        fontsize=7.5, color=INK if not hatch[i, j] else MUTED)
    ax.set(xscale="log", yscale="log", xlabel="population N")
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])
    ax.set_xticks(pops)
    ax.set_xticklabels([f"{n // 1024}k" if n >= 1024 else str(n) for n in pops],
                       fontsize=8)
    ax.set_yticks(dims)
    ax.set_yticklabels([str(d) for d in dims], fontsize=8)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(length=0, colors=MUTED)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def common_comparison(platforms: dict, dtype: str, out: pathlib.Path) -> None:
    """Matched feasible configurations, separated from the full memory grid."""
    if len(platforms) != 2:
        return
    kinds = sorted(platforms)
    comparable = []
    for kind in kinds:
        cs = platforms[kind]
        comparable.append({(d, n) for d, n, strategy, dt in cs
                           if strategy == BASELINE and dt == dtype
                           and all(cs.get((d, n, st, dt)) not in (None, OOM)
                                   for st in [BASELINE, *STRATEGIES])})
    configs = sorted(set.intersection(*comparable))
    if not configs:
        raise ValueError("no common feasible configurations")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, strategy in zip(axes, STRATEGIES):
        for kind, color, marker in zip(kinds, ("#256abf", "#eb6834"), ("o", "^")):
            cs = platforms[kind]
            vals = [cs[d, n, strategy, dtype] / cs[d, n, BASELINE, dtype] for d, n in configs]
            ax.plot(vals, range(len(configs)), linestyle="none", marker=marker, ms=7,
                    color=color, label="A100" if "A100" in kind else "TPU v5e")
        ax.axvline(1, color=MUTED, lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("generation time / stored full-rank time")
        ax.set_title("Regenerated full rank (seed)" if strategy == "seed_regenerated"
                     else "Rank-1 perturbations", fontsize=11, loc="left")
        ax.grid(axis="x", color="#e6e6e3")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(configs)), labels=[f"d={d}, N={n}" for d, n in configs])
    axes[0].invert_yaxis()
    axes[0].set_xticks([1, 2, 5, 10], labels=["1", "2", "5", "10"])
    axes[0].set_xlim(0.85, 10)
    axes[1].set_xticks([0.05, 0.1, 0.2, 0.5, 1], labels=["0.05", "0.1", "0.2", "0.5", "1"])
    axes[1].set_xlim(0.035, 1.15)
    for ax in axes:
        ax.xaxis.set_minor_locator(NullLocator())
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
               frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    dest = out / f"f4-cost-common-{dtype}.png"
    fig.savefig(dest, dpi=200, metadata={"Date": None, "Software": None})
    plt.close(fig)
    print(dest)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=pathlib.Path, nargs="+",
                    default=[HERE / "results-cost"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "figures")
    args = ap.parse_args(argv)

    platforms = load(args.results)
    args.out.mkdir(parents=True, exist_ok=True)

    dims = sorted({k[0] for cells in platforms.values() for k in cells})
    pops = sorted({k[1] for cells in platforms.values() for k in cells})

    # One shared symmetric scale across every panel and platform, so the same
    # shade means the same ratio everywhere; the comparison IS the figure.
    finite = [np.log10(t / cells.get((k[0], k[1], BASELINE, k[3])))
              for cells in platforms.values() for k, t in cells.items()
              if k[2] != BASELINE and k[3] == args.dtype and t is not OOM
              and cells.get((k[0], k[1], BASELINE, k[3])) not in (None, OOM)]
    if not finite:
        print(f"no comparable ({args.dtype}) cells in {list(args.results)}")
        return 1
    lim = float(np.max(np.abs(finite)))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    # One row, platform after platform: the paper prints it full width, and two rows
    # of two took twice the height for the same cells.
    panels = [(kind, cells, st) for kind, cells in sorted(platforms.items())
              for st in STRATEGIES]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.75 * len(panels), 3.0),
                             squeeze=False, sharey=True)
    for j, (kind, cells, st) in enumerate(panels):
        ax = axes[0][j]
        panel(ax, cells, st, args.dtype, dims, pops, norm)
        ax.set_title(f"{TITLES[st]}\n{PLATFORM.get(kind, kind)}", color=INK, fontsize=9)
        if j == 0:
            ax.set_ylabel("model dimension d", color=INK)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    bar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01,
                       label="time relative to dense")
    # Log-scaled colour, labelled in ratios, so nobody has to exponentiate.
    ticks = [t for t in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20) if abs(np.log10(t)) <= lim]
    bar.set_ticks([np.log10(t) for t in ticks])
    bar.set_ticklabels([f"{t:g}x" for t in ticks])
    out = args.out / f"f4-cost-{args.dtype}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", metadata={"Date": None, "Software": None})
    print(out)
    plt.close(fig)
    common_comparison(platforms, args.dtype, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
