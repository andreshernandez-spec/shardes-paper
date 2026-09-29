#!/usr/bin/env python
"""Read the shaping-barrier records properly, and check them against the ladder.

    python barrier_report.py
    python barrier_report.py --results results-barrier-tpu-v5e8

`barrier.py` times three programs per (N, D): shaping `none`, `centered` and
`centered_ranks`. The first write-up read the `none` row as the fitness gather
and reported it at 2.4-6 us, flat in N up to a 1 MiB payload. That reading is
wrong, and the records themselves say so.

Under `none` the step is

    weights = replicate(f)                  # the gather
    f       = reshard(f + 1e-6 * weights)   # elementwise, then back to P("pop")

so each device's next carry depends only on its own slice of the gathered
array. Nothing downstream ever needs the other devices' slices, and the
compiler is free to drop the gather or leave it off the critical path. The
chain's docstring guards against hoisting; it does not guard against this.
`centered` and `centered_ranks` are different: a mean, a standard deviation or
a rank needs every member, so their gather cannot be skipped.

The test is payload dependence. A real all-gather is latency-bound for small
messages and bandwidth-bound for large ones, so what eight devices add to a
program must grow between N=64 and N=2^18. It does for the two programs that
consume the gathered array and does not for `none`. The growth also matches
the dedicated instrument, `allreduce_ladder.py`, which times the same tiled
all-gather of the same payload on the same chip.

What survives: every statement about the sort, which is read at D=1 where
there is no communication to confuse it, and the in-context closure in
`results-barrier-context`. What does not: "the gather costs 2.4-6 us, flat in
N". The gather costs what the ladder says it costs.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
LADDERS = {"tpu": "results-ladder/ladder-tpu-v5-lite-D8.json",
           "gpu": "results-ladder/ladder-nvidia-a100-sxm4-80gb-D8.json"}


def load(results: pathlib.Path) -> tuple[dict, str]:
    cells = collections.defaultdict(dict)
    platform = "tpu"
    for path in results.glob("*.json"):
        rec = json.loads(path.read_text())
        if "seconds_median" not in rec:
            continue
        cfg = rec["config"]
        cells[(cfg["population"], cfg["devices"])][cfg["shaping"]] = rec["seconds_median"]
        platform = rec.get("env", {}).get("device_platform", platform)
    return cells, platform


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results-barrier-tpu-v5e8")
    ap.add_argument("--devices", type=int, default=8)
    ap.add_argument("--latex", action="store_true",
                    help="also write paper/generated/ranking.tex, the figures Appendix A.1 states")
    args = ap.parse_args(argv)

    cells, platform = load(HERE / args.results)
    d = args.devices
    pops = sorted({n for n, dev in cells if dev == d and (n, 1) in cells})
    ladder = {}
    ladder_file = HERE / LADDERS.get(platform, LADDERS["tpu"])
    if ladder_file.exists():
        ladder = {int(k): v["step_seconds"]
                  for k, v in json.loads(ladder_file.read_text())["allgather"].items()}

    def us(x):
        return f"{x * 1e6:.1f}"

    print(f"## {args.results}: what D={d} adds to each program over D=1, microseconds\n")
    print("| N | payload | none | centered | centered_ranks | ladder all-gather step |")
    print("|---|---|---|---|---|---|")
    added = collections.defaultdict(dict)
    for n in pops:
        one, many = cells[(n, 1)], cells[(n, d)]
        for s in ("none", "centered", "centered_ranks"):
            added[s][n] = many[s] - one[s]
        print(f"| {n} | {4 * n} B | {us(added['none'][n])} | {us(added['centered'][n])} | "
              f"{us(added['centered_ranks'][n])} | {us(ladder[n]) if n in ladder else ''} |")

    lo, hi = pops[0], pops[-1]
    print(f"\nGrowth from N={lo} to N={hi}, the payload-dependent part of a gather:")
    for s in ("none", "centered", "centered_ranks"):
        print(f"  {s:15s} {us(added[s][hi] - added[s][lo]):>7} us")
    if ladder:
        small = min(ladder)
        print(f"  {'ladder':15s} {us(ladder[max(ladder)] - ladder[small]):>7} us"
              f"   (N={small} to N={max(ladder)})")
    print("\n`none` does not grow with a 4096-fold payload, so it is not paying for one. "
          "Read the gather from\nthe ladder, or from the two programs that have to "
          "consume what they gathered.")

    print(f"\n## The sort, read at D=1 where nothing is communicated, and the whole "
          f"barrier at D={d}\n")
    print(f"| N | sort us (ranks - none, D=1) | ns per member | whole ranks iteration, "
          f"D={d}, us |")
    print("|---|---|---|---|")
    for n in pops:
        sort = cells[(n, 1)]["centered_ranks"] - cells[(n, 1)]["none"]
        print(f"| {n} | {us(sort)} | {sort / n * 1e9:.1f} | "
              f"{us(cells[(n, d)]['centered_ranks'])} |")
    big = pops[-1]
    a, b = cells[(big, 1)]["centered_ranks"], cells[(big, d)]["centered_ranks"]
    print(f"\nAt N={big} the ranks program takes {us(a)} us on one device and {us(b)} on "
          f"{d}: {100 * (b - a) / a:.2f}% apart,\nand the {us(b - a)} us between them is "
          "the gather and the repeat noise, not a second sort.")
    if args.latex:
        (HERE.parent.parent / "paper" / "generated" / "ranking.tex").write_text(
            ranking_macros(cells, added, ladder, pops, d))
    return 0


def ranking_macros(cells, added, ladder, pops, d) -> str:
    """Appendix A.1's figures, rounded once, with the claims it makes in words checked."""
    import sys
    sys.path.insert(0, str(HERE.parent))
    from protocol_constants import rounded

    def claim(ok, text):
        if not ok:
            raise SystemExit(f"Appendix A.1 says {text}")

    small = [n for n in pops if n <= 1024]
    big = pops[-1]
    sort_big = cells[(big, 1)]["centered_ranks"] - cells[(big, 1)]["none"]
    gather = ladder[big]
    share = gather / added["centered_ranks"][big]
    claim(0.4 <= share <= 0.6, "about half of what eight devices add at the largest N is the gather")
    claim(cells[(big, d)]["centered_ranks"] >= cells[(big, 1)]["centered_ranks"],
          "the sort does not get faster with more devices")
    ctx = context_added()
    return "\n".join([
        "% generated by experiments/phase2/barrier_report.py --latex; do not edit",
        f"\\newcommand{{\\RankStepSmallMax}}{{{rounded(max(cells[(n, d)]['centered_ranks'] for n in small) * 1e6, 0)}}}",
        f"\\newcommand{{\\RankStepMid}}{{{rounded(cells[(4096, d)]['centered_ranks'] * 1e3, 1)}}}",
        f"\\newcommand{{\\RankAddedSmall}}{{{rounded(min(added['centered_ranks'][n] for n in small) * 1e6, 0)}"
        f"--{rounded(max(added['centered_ranks'][n] for n in small) * 1e6, 0)}}}",
        f"\\newcommand{{\\RankAddedMax}}{{{rounded(added['centered_ranks'][big] * 1e6, 1)}}}",
        f"\\newcommand{{\\RankGatherIsolated}}{{{rounded(gather * 1e6, 0)}}}",
        f"\\newcommand{{\\RankSortMax}}{{{rounded(sort_big * 1e3, 1)}}}",
        f"\\newcommand{{\\RankContextSmall}}{{{rounded(ctx[512, 1024] * 1e6, 0)}}}",
        f"\\newcommand{{\\RankContextLarge}}{{{rounded(ctx[2048, 256] * 1e6, 0)}}}",
        ""])


def context_added() -> dict:
    """What turning rank shaping on adds to a complete generation, from
    results-barrier-context; Appendix A.1 says both differences sit within the repeats."""
    runs = {}
    for path in (HERE / "results-barrier-context").glob("*.json"):
        rec = json.loads(path.read_text())
        c = rec["config"]
        runs[c["d_model"], c["population"], c["shaping"]] = rec
    out = {}
    for d_model, n in ((512, 1024), (2048, 256)):
        on, off = runs[d_model, n, "centered_ranks"], runs[d_model, n, "none"]
        overlap = min(on["seconds_all"]) <= max(off["seconds_all"]) and min(off["seconds_all"]) <= max(on["seconds_all"])
        if not overlap:
            raise SystemExit(f"Appendix A.1 says rank shaping's cost at d={d_model}, N={n} is within the repeats")
        out[d_model, n] = on["seconds_median"] - off["seconds_median"]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
