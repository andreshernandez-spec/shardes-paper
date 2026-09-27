#!/usr/bin/env python
"""How much overlap could help all-reduce (B), from the measured contraction terms.

    python overlap_bound.py

Reads only `results-contraction/` (contraction_isolation.py: the replicated contraction
C including A's weight gather, B's local share C_local, and the added communication cost, per cell at D=8).

Overlapping B's all-reduce with the NEXT generation's evaluation is not free: that
evaluation needs the updated weights, so hiding the transfer behind it evaluates at
one-step-stale weights, a different algorithm. The overlap that keeps the algorithm is
within the generation, bucketing the all-reduce behind B's own local contraction, as
DDP does behind backward. That changes B's contraction-plus-collective time from
C_local + AR to at best max(C_local, AR). Evaluation is the same in both placements and
cancels. For each cell this prints t_B - t_A from the measured terms, without overlap
and with perfect in-generation overlap. A's recorded C already includes its additional gather, so it is subtracted only once.
It is a bound from measured terms, not a
measurement of an overlapped implementation.
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
LOW_RANK = ("lowrank_r1", "mirrored_lr1")
NAMES = {"lowrank_r1": "rank 1, unpaired", "mirrored_lr1": "rank 1"}


def main() -> int:
    rows = []
    for p in sorted((HERE / "results-contraction").glob("*.json")):
        r = json.loads(p.read_text())
        s = next((s for s in LOW_RANK if f"__s={s}__" in p.name), None)
        if s is None or r.get("contraction_seconds") is None:
            continue
        d = int(p.name.split("d=")[1].split("__")[0])
        n = int(p.name.split("N=")[1].split("__")[0])
        c, cl, ar = (r[k] * 1e3 for k in ("contraction_seconds", "contraction_local_seconds",
                                          "allreduce_insitu_seconds"))
        rows.append((r["device_kind"], NAMES[s], d, n, c, cl, ar, cl + ar - c,
                     max(cl, ar) - c))
    print("| device | arm | d | N | A including gather | C_local | added communication | t_B - t_A | with overlap |")
    print("|---|---|---|---|---|---|---|---|---|")
    for kind, name, d, n, c, cl, ar, plain, over in rows:
        print(f"| {kind} | {name} | {d} | {n} | {c:.2f} | {cl:.2f} | {ar:.2f} | "
              f"{plain:+.2f} | {over:+.2f} |")
    big = [r for r in rows if r[2] == 2048]
    print(f"\nd=2048, all {len(big)} low-rank cells (ms): t_B - t_A "
          f"{min(r[7] for r in big):+.2f} to {max(r[7] for r in big):+.2f} without overlap, "
          f"{min(r[8] for r in big):+.2f} to {max(r[8] for r in big):+.2f} with it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
