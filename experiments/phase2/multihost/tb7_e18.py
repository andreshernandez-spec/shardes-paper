#!/usr/bin/env python
"""Tables 7 and 9 from results-e18: the placement across a host boundary, measured and expected.

    python tb7_e18.py            # markdown to stdout, with the fabric figures
    python tb7_e18.py --latex    # also writes paper/generated/tb7.tex and tb7-audit.tex

Reads only committed files: the results-e18 cell JSONs and preflight ladders (2x
A100-SXM4-80GB, RunPod Instant Cluster, socket inter-node transport) and the inputs
`predict.py` uses (the D=8 sweep anchor and `results-contraction`). The records and the
cost model work in t_B - t_A of median generation time; the tables print t_A - t_B in ms,
the paper's one sign convention, where positive favors splitting.

`expected` is Eq. 2 evaluated with the all-reduce rate measured on these hosts, each
ladder's beta taken at the payload it really moved (`costmodel.ladder_alpha_beta`). The
paper reports only these values. (`results-e18/predictions.json`, written before the 2x8
cells ran, used the preflight's beta as recorded, which was D times too high because the
preflight all-reduced 1/D of each label; it is not used here.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import costmodel

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results-e18"
SWEEP = HERE.parent / "results-consistent"
# (label, strategy, d, N), in reading order
CELLS = [
    ("seed", "seed_regenerated", 2048, 128),
    ("seed", "seed_regenerated", 2048, 256),
    ("seed", "seed_regenerated", 512, 1024),
    ("rank 1", "mirrored_lr1", 2048, 128),
    ("rank 1", "mirrored_lr1", 2048, 256),
    ("rank 1", "mirrored_lr1", 512, 1024),
]
DEVICES = {"1x8": 8, "2x4": 8, "2x8": 16}


def med(strategy, how, d, n, topo):
    f = RESULTS / f"arm={strategy}__how={how}__d={d}__N={n}__topo={topo}.json"
    return json.loads(f.read_text())["seconds_median"] if f.exists() else None


def measured(strategy, d, n, topo):
    a, b = med(strategy, "A", d, n, topo), med(strategy, "B", d, n, topo)
    return None if a is None or b is None else b - a


def anchor(strategy, d, n):
    """delta at D=8 from the committed single-node sweep, as predict.py reads it."""
    stem = f"mode=strong__D=8__d={d}__N={n}__s={strategy}__how="
    ta = json.loads((SWEEP / f"{stem}A.json").read_text())["seconds_median"]
    tb = json.loads((SWEEP / f"{stem}B.json").read_text())["seconds_median"]
    return tb - ta


def table_rows():
    """(preflights, fabric, rows); each row is (label, d, N, values) with values, in
    t_B - t_A seconds: 1x8 measured, 2x4 measured, 2x4 expected, 2x8 measured, 2x8 expected."""
    pre = {t: json.loads((RESULTS / f"preflight-{t}.json").read_text()) for t in DEVICES}
    fabric = {t: costmodel.ladder_alpha_beta(pre[t], DEVICES[t]) for t in DEVICES}

    def bump(topo, p_bytes):
        return (costmodel.ladder_seconds(*fabric[topo], p_bytes)
                - costmodel.ladder_seconds(*fabric["1x8"], p_bytes))

    rows = []
    for label, strat, d, n in CELLS:
        p_bytes = costmodel.params_bytes(d)
        c = costmodel.measured_contraction(strat, d, n)
        delta_8 = anchor(strat, d, n)
        expected_16 = costmodel.predict_delta(delta_8, bump("2x8", p_bytes), 16, c)
        rows.append((label, d, n, [
            measured(strat, d, n, "1x8"),
            measured(strat, d, n, "2x4"),
            delta_8 + bump("2x4", p_bytes),
            measured(strat, d, n, "2x8"),
            expected_16["delta_predicted"],
        ]))
    return pre, fabric, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args(argv)

    pre, fabric, rows = table_rows()

    def ms(v):
        """t_A - t_B in ms, from a t_B - t_A value: the paper's one sign convention
        (Eq. 2, Table 2), where positive favors splitting. The measurements and the cost
        model work in t_B - t_A, so the sign flips here and nowhere else."""
        if v is None:
            return "--"
        return "0.0" if abs(v) < 5e-5 else f"{-v * 1e3:+.1f}"

    heads = ["1x8", "2x4", "2x4 expected", "2x8", "2x8 expected"]
    print("| arm | d | N | " + " | ".join(heads) + " |")
    print("|---|---|---|" + "---|" * len(heads))
    for label, d, n, vals in rows:
        print(f"| {label} | {d} | {n} | " + " | ".join(ms(v) for v in vals) + " |")
    print()
    for t in DEVICES:
        a, b = fabric[t]
        print(f"{t}: alpha {a * 1e6:.0f} us, beta {b / 2**30:.2f} GiB/s at the payload moved "
              f"(record says {pre[t]['beta_bytes_per_second'] / 2**30:.1f})")

    ar = costmodel.params_bytes(2048)
    print(f"a {ar / 2**20:.0f} MiB all-reduce at 12 and 50 GiB/s: "
          f"{ar / (12 * 2**30) * 1e3:.1f} and {ar / (50 * 2**30) * 1e3:.1f} ms")
    print("boundary penalty on B, 1x8 -> 2x4 (ms): " + ", ".join(
        f"{label} d={d} N={n} {(v[1] - v[0]) * 1e3:.1f}" for label, d, n, v in rows))

    if args.latex:
        out = HERE.parent.parent.parent / "paper" / "generated" / "tb7-audit.tex"
        body = [f"{label} & {d} & {n} & " + " & ".join(f"${ms(v)}$" for v in vals) + r" \\"
                for label, d, n, vals in rows]
        s24, s28 = (fabric[t][1] / 2**30 for t in ("2x4", "2x8"))
        out.write_text("\n".join([
            "% generated by experiments/phase2/multihost/tb7_e18.py --latex; do not edit",
            r"\begin{table*}[t]", r"\centering", r"\small",
            r"\caption{Host-boundary check of Eq.~\ref{eq:crossover}: replicated minus split "
            r"generation time, $t_A-t_B$, in ms; positive values favor splitting, as in "
            r"Table~\ref{tab:worked}. Columns give hosts $\times$ devices per host. "
            r"\emph{Expected} values are Eq.~\ref{eq:crossover} with the all-reduce rates "
            r"measured on these hosts: an effective cross-host rate of "
            f"{s24:.2f}\\,GiB/s at eight devices "
            f"and {s28:.2f}\\,GiB/s at sixteen. "
            r"They start from the earlier single-host sweep; the measured "
            r"\texttt{1x8} column is a new measurement on the cluster's first host.}",
            r"\label{tab:tb7-audit}",
            r"\begin{tabular}{llrrrrrr}", r"\toprule",
            r" & & & & \multicolumn{2}{c}{\texttt{2x4}} & \multicolumn{2}{c}{\texttt{2x8}} \\",
            r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
            r"variant & $d$ & $N$ & \texttt{1x8} & measured & expected & measured & expected \\",
            r"\midrule",
            *body,
            r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]))
        compact = out.with_name("tb7.tex")
        compact_body = [f"{label} & {d} & {n} & "
                        + " & ".join(f"${ms(vals[i])}$" for i in (0, 1, 3)) + r" \\"
                        for label, d, n, vals in rows]
        compact.write_text("\n".join([
            "% generated by experiments/phase2/multihost/tb7_e18.py; do not edit",
            r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
            r"\caption{Generation time under replication minus generation time under "
            r"splitting, $t_A-t_B$, in ms; positive values favor splitting, as in "
            r"Table~\ref{tab:worked}. Columns give hosts $\times$ devices per host; the "
            r"two hosts are connected by TCP sockets. Table~\ref{tab:tb7-audit} sets them "
            r"beside the values Eq.~\ref{eq:crossover} gives.}",
            r"\label{tab:tb7}", r"\begin{tabular}{llrrrr}", r"\toprule",
            r"variant & $d$ & $N$ & $1\times8$ & $2\times4$ & $2\times8$ \\",
            r"\midrule", *compact_body, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
