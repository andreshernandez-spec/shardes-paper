#!/usr/bin/env python
"""TB3, the ablation table, assembled from committed results. No new measurement.

    python tb3.py            # markdown to stdout
    python tb3.py --latex    # also writes ../../paper/generated/tb3.tex

Every number a reviewer might ask "did you check X" about, computed from the
directory that already answers it, so the table is reproducible from a clean
checkout (CLAUDE.md ground rule 2). Rows whose source directory is absent print
PENDING with the session that fills them, rather than silently vanishing: a
table missing a row reads as "didn't check" when the truth is "not yet run".

Ratios are geometric means over the cells where BOTH sides were measured, since
cost ratios multiply and the grid spans two orders of magnitude in wall time.
The σ and estimator-quality axes live in phase 0 (E1, F5) and are cited, not
recomputed: this script owns the systems rows.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import statistics

HERE = pathlib.Path(__file__).resolve().parent
COUNTDOWN = HERE.parent / "countdown" / "results" / "e13-a100-2026-08-22-clean"

GPU = HERE / "results-cost"
TPU = HERE / "results-cost-tpu-v5e8"
TPU_HIGHEST = HERE / "results-cost-tpu-v5e8-highest"


def cells(d: pathlib.Path) -> dict:
    out = {}
    for p in sorted(d.glob("*.json")):
        r = json.loads(p.read_text())
        c = r["config"]
        key = (c["d_model"], c["population"], c["strategy"], c["dtype"])
        out[key] = None if r.get("undersized") else r["seconds_median"]
    return out


def geomean(ratios: list[float]) -> float:
    return math.exp(sum(map(math.log, ratios)) / len(ratios))


def ratio(cs: dict, num_sel, den_sel) -> tuple[float, int] | None:
    """Geometric-mean ratio over cells where both selector variants measured, and how many."""
    rs = []
    for k, t in cs.items():
        nk = num_sel(k)
        if nk is None or t is None:
            continue
        base = cs.get(den_sel(k) if den_sel else k)
        num = cs.get(nk)
        if num and base:
            rs.append(num / base)
    return (geomean(rs), len(rs)) if rs else None


def ratio_row(cs: dict, num_sel, den_sel) -> str:
    r = ratio(cs, num_sel, den_sel)
    return f"{r[0]:.2f}x (n={r[1]})" if r else "no overlapping cells"


def rank_cost(cs: dict, platform: str) -> list[str]:
    rows = []
    for r in (1, 4, 16):
        val = ratio_row(
            cs,
            lambda k, r=r: (k[0], k[1], f"mirrored_lr{r}", k[2 + 1])
            if k[2] == "iid_gaussian" and k[3] == "bfloat16" else None,
            None,
        )
        rows.append(f"| cost, rank {r} vs dense | {platform}, bf16 | {val} |")
    return rows


def common_rank_cost(gpu: dict, tpu: dict) -> tuple[int, float, float]:
    """Use the same feasible configurations as the main cost plot for both platforms."""
    ratios = []
    for cs in (gpu, tpu):
        ratios.append({(d, n): t / cs[(d, n, "iid_gaussian", dtype)]
                       for (d, n, strategy, dtype), t in cs.items()
                       if strategy == "mirrored_lr1" and dtype == "bfloat16" and t
                       and cs.get((d, n, "iid_gaussian", dtype))
                       and cs.get((d, n, "seed_regenerated", dtype))})
    common = sorted(ratios[0].keys() & ratios[1].keys())
    if not common:
        raise ValueError("no common rank-1 and dense configurations")
    return len(common), *(geomean([r[k] for k in common]) for r in ratios)


def dtype_ratios(cs: dict) -> dict[str, tuple[float, int]]:
    """bf16 over f32 time per variant."""
    out = {}
    for s in ("iid_gaussian", "seed_regenerated", "mirrored_lr1"):
        r = ratio(cs, lambda k, s=s: (k[0], k[1], s, "bfloat16")
                  if k[2] == s and k[3] == "float32" else None, None)
        if r:
            out[s] = r
    return out


def dtype_cost(cs: dict, platform: str) -> list[str]:
    return [f"| cost, bf16 vs f32 | {platform}, {s} | {g:.2f}x (n={n}) |"
            for s, (g, n) in dtype_ratios(cs).items()]


#: Table order for the precision rows: dense, the low ranks in order, then seed.
PRECISION_ORDER = ("iid_gaussian", "mirrored_lr1", "mirrored_lr4", "mirrored_lr16",
                   "seed_regenerated", "mirrored_seed")


def precision_ratios() -> dict[str, tuple[float, int]]:
    """Highest over default precision time per variant, on the v5e."""
    hi, lo = cells(TPU_HIGHEST), cells(TPU)
    out = {}
    for s in sorted({k[2] for k in hi}, key=PRECISION_ORDER.index):
        rs = [hi[k] / lo[k] for k in hi if k[2] == s and hi.get(k) and lo.get(k)]
        if rs:
            out[s] = (geomean(rs), len(rs))
    return out


def precision_cost() -> list[str]:
    if not TPU_HIGHEST.exists() or not any(TPU_HIGHEST.glob("*.json")):
        return ["| cost, `highest` vs `default` | TPU v5e | "
                "PENDING: T4 session, cost-precision-tpu.yaml |"]
    return [f"| cost, `highest` vs `default` | TPU v5e, {s} | {g:.2f}x (n={n}) |"
            for s, (g, n) in precision_ratios().items()]


def task_quality() -> list[str]:
    if not COUNTDOWN.exists():
        return ["| task quality vs rank | Countdown | PENDING: e13 results missing |"]
    rows = []
    for stem, label in (("es-mirrored-seed", "full rank"), ("es-mirrored-lr1", "rank 1"),
                        ("es-mirrored-lr4", "rank 4"), ("es-mirrored-lr16", "rank 16")):
        finals = []
        for s in (0, 1, 2):
            lines = (COUNTDOWN / f"{stem}-s{s}-eval.jsonl").read_text().splitlines()
            finals.append(json.loads(lines[-1])["eval_reward"])
        # Mean, matching F7 and the campaign README; the statistic is named
        # in the caption because an unnamed summary is unreadable.
        rows.append(f"| held-out reward, {label} | Countdown, 3 seeds | "
                    f"{statistics.mean(finals):.3f} "
                    f"[{min(finals):.3f}, {max(finals):.3f}] |")
    return rows


def main() -> int:
    header = ["| ablation | where | result |", "|---|---|---|"]
    # Systems rows: timing ratios only. The validation rows (quality,
    # guards, invariance) answer a different question ("was it checked")
    # and go to their own table in the appendix.
    systems = []
    for d, platform in ((GPU, "A100"), (TPU, "TPU v5e")):
        if d.exists():
            systems += rank_cost(cells(d), platform)
    for d, platform in ((GPU, "A100"), (TPU, "TPU v5e")):
        if d.exists():
            systems += dtype_cost(cells(d), platform)
    systems += precision_cost()
    common_n, common_gpu, common_tpu = common_rank_cost(cells(GPU), cells(TPU))
    print(f"Common grid ({common_n} configurations): rank-1 / dense time "
          f"A100 {common_gpu:.3f}, v5e {common_tpu:.3f}")
    validation = task_quality() + [
        "| update--gradient alignment vs rank, σ, N | phase 0 | E1 / F5: "
        "`experiments/phase0/` (measured, committed) |",
        "| sub-f32 fitness | guard | refused by `tell`; measured collapse "
        "256 losses -> 2 in bf16 (docs/proposal-bf16-policy.md) |",
        "| device-count invariance | tests | `tests/`: D=1 vs D=8 within "
        "rtol 1e-5 bf16 / 1e-12 f32; C6d multi-GPU run |",
    ]
    print("\n".join(header + systems))
    print()
    print("\n".join(header + validation))

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    if ap.parse_args().latex:
        # The markdown rows are the single source; convert rather than rebuild,
        # so the two outputs cannot drift.
        names = {"iid_gaussian": "dense", "seed_regenerated": "seed",
                 "mirrored_seed": "seed, mirrored", "mirrored_lr16": "rank 16",
                 "mirrored_lr4": "rank 4", "mirrored_lr1": "rank 1"}

        # Section 6.4 says seed is slower than dense wherever both ran, fits the whole cost grid,
        # and that dense and rank 1 do not; check it before writing anything.
        for platform, cs in (("A100", cells(GPU)), ("v5e", cells(TPU))):
            fit = lambda s: [k for k, v in cs.items() if k[2] == s and v is not None]  # noqa: E731
            seed = {k: v for k, v in cs.items() if k[2] == "seed_regenerated"}
            dense = {k[:2] + k[3:]: v for k, v in cs.items() if k[2] == "iid_gaussian"}
            both = [(v, dense[k[:2] + k[3:]]) for k, v in seed.items()
                    if v is not None and dense.get(k[:2] + k[3:]) is not None]
            if any(v is None for v in seed.values()):
                raise SystemExit(f"Section 6.4 says seed fits the whole cost grid; not on the {platform}")
            if not both or not all(s > d for s, d in both):
                raise SystemExit(f"Section 6.4 says seed is slower than dense wherever both ran; not on the {platform}")
            for s in ("iid_gaussian", "mirrored_lr1"):
                if len(fit(s)) == sum(1 for k in cs if k[2] == s):
                    raise SystemExit(f"Section 6.4 says {s} does not fit the whole cost grid; it does on the {platform}")

        # Macros, not a sentence: Section 6.4 and the introduction both state these ratios,
        # and prose stays editable when only the numbers are generated.
        sys.path.insert(0, str(HERE.parent))
        from protocol_constants import rounded, spelled
        # Appendix B: highest against default precision by rank, and storage datatype overall.
        precision = precision_ratios()
        low = [g for s, (g, _) in precision.items() if s.startswith("mirrored_lr")]
        full = [g for s, (g, _) in precision.items() if not s.startswith("mirrored_lr")]
        dtype = [g for d in (GPU, TPU) for g, _ in dtype_ratios(cells(d)).values()]
        span = lambda xs, places: f"{rounded(min(xs), places)}--{rounded(max(xs), places)}"  # noqa: E731
        (HERE.parent.parent / "paper" / "generated" / "cost-common.tex").write_text(
            "% generated by experiments/phase2/tb3.py; do not edit\n"
            f"\\newcommand{{\\CostCommonConfigs}}{{{spelled(common_n)}}}\n"
            f"\\newcommand{{\\RankOneOverDenseGPU}}{{{rounded(common_gpu, 2)}}}\n"
            f"\\newcommand{{\\RankOneOverDenseTPU}}{{{rounded(common_tpu, 2)}}}\n"
            f"\\newcommand{{\\PrecisionLowRank}}{{{span(low, 1)}}}\n"
            f"\\newcommand{{\\PrecisionFullRank}}{{{span(full, 2)}}}\n"
            f"\\newcommand{{\\DtypeRange}}{{{span(dtype, 2)}}}\n")

        def esc(s):
            for code, name in names.items():
                s = s.replace(code, name)
            return (s.replace("\\", "").replace("&", "\\&").replace("_", "\\_")
                    .replace("σ", "$\\sigma$").replace("->", "$\\to$")
                    .replace("`", "").replace("%", "\\%"))

        def emit(rows, caption, label, fname):
            body = []
            previous_group = None
            for ln in rows:
                cellsx = [c.strip() for c in ln.strip("|").split("|")]
                group = ("Perturbation rank" if "rank " in cellsx[0] else
                         "Storage datatype" if "bf16 vs f32" in cellsx[0] else
                         "Matrix-multiplication precision (TPU)")
                if group != previous_group:
                    if previous_group is not None:
                        body.append(r"\midrule")
                    body.append(r"\multicolumn{3}{l}{\emph{" + group + r"}} \\")
                    previous_group = group
                cellsx[0] = cellsx[0].replace("cost, ", "").replace("`highest` vs `default`",
                                                                            "highest vs default precision")
                body.append(" & ".join(esc(c) for c in cellsx) + " \\\\")
            tex = "\n".join([
                "% generated by experiments/phase2/tb3.py; do not edit by hand",
                "\\begin{table*}", "\\centering", "\\small",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                "\\begin{tabular}{p{0.30\\linewidth}p{0.22\\linewidth}p{0.40\\linewidth}}",
                "\\toprule", "comparison & where & ratio \\\\", "\\midrule",
                *body,
                "\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
            dest = HERE.parent.parent / "paper" / "generated" / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(tex)
            print(f"wrote {dest}")

        # The validation rows stay in the markdown only: they point at files and
        # tests, which is what a README is for, and the rewards are Figure 5's.
        emit(systems,
             "Time ratios on one device for three choices: perturbation rank, storage "
             "datatype and matrix-multiplication precision. Each ratio is the time with "
             "the first setting over the time with the second (below 1: the first is "
             "faster), as a geometric mean over the configurations where both were "
             "measured (n: how many).",
             "tab:tb3", "tb3.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
