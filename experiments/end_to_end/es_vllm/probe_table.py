#!/usr/bin/env python
"""One row per probed setting, from `runs/probe/*/prediction.json`, as a markdown table.

    python -m es_vllm.probe_table runs/probe            # the table in runs/probe/README.md
    python -m es_vllm.probe_table runs/probe --json     # the same rows as JSON

Level 0 first, then level 1, then the rule. Rewards are normalized to [0, 1]; `G` is
the update's odd response per unit update and `C` its even response per unit squared
at the probe's 64 members; `G^2/4|C|` is the fitted net per update at that N. The
ratio is ES's rollouts over RL's for a tenth of the reward range at 2% damage.
"""

import argparse
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
E2E = PKG.parent
sys.path.insert(0, str(E2E))

COLUMNS = (("setting", "setting"), ("live", "live share"), ("share", "gradient sharedness"),
           ("c", "curvature per length^2"), ("rl", "RL first-step gain"), ("rel", "reliability"),
           ("corr", "cross-prompt corr"), ("G", "G"), ("C", "C"), ("net", "G^2/4|C|"),
           ("ratio", "ES/RL rollouts"), ("ratio90", "ratio 90%"), ("call", "call"),
           ("offtask", "off-task at target"), ("damaged", "damaged first (scan / update)"))


def fmt(x, digits=3):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if x == 0:
        return "0"
    if abs(x) >= 100:
        return f"{x:,.0f}"
    if abs(x) >= 0.01:
        return f"{x:.{digits}g}"
    return f"{x:.2e}"


def row(pred: dict) -> dict:
    l0, l1, u, r = pred["level0"], pred["level1"], pred["update"] or {}, pred["rule"]
    s = l1.get("sigma", {})
    b = r.get("bootstrap_over_prompts") or {}
    ratio90 = b.get("ratio_90")
    return {"setting": pred["setting"], "live": l0["live_share"], "share": l0["gradient_sharedness"],
            "c": l0["curvature_per_length2"], "rl": l0["rl_first_step_gain"],
            "rel": s.get("reliability_at_prompts"), "corr": s.get("cross_prompt_correlation"),
            "G": u.get("G"), "C": u.get("C"), "net": r["es"]["fitted_net_per_update"] if r["es"] else None,
            "ratio": r["ratio_es_over_rl_rollouts"],
            "ratio90": f"{fmt(ratio90[0])} to {fmt(ratio90[1])}" if ratio90 else None, "call": r["call"],
            "offtask": u.get("offtask_change_at_target"),
            "damaged": f"{l0['first_task_damaged_random']} / {u.get('first_task_damaged', '-')}"}


def rows(root: Path) -> list:
    out = []
    for p in sorted(root.glob("*/prediction.json")):
        pred = json.loads(p.read_text())
        pred.setdefault("setting", p.parent.name)
        out.append(row(pred))
    return out


def spearman(xs, ys) -> float:
    """Rank correlation of two equally long lists (no ties expected)."""
    import numpy as np  # noqa: PLC0415

    rx, ry = np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys))
    return float(np.corrcoef(rx, ry)[0, 1])


def level0_vs_level1(rs: list) -> dict:
    """Does the gradient sharedness order the settings as the members' cross-prompt
    correlation does (docs/end_to_end/10, the level-0 test)?"""
    ok = [r for r in rs if r["share"] is not None and r["corr"] is not None]
    return {"settings": len(ok), "spearman_sharedness_vs_cross_prompt_correlation":
            spearman([r["share"] for r in ok], [r["corr"] for r in ok]) if len(ok) > 2 else None}


def table(rs: list) -> str:
    head = "| " + " | ".join(name for _, name in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    body = ["| " + " | ".join(fmt(r[key]) for key, _ in COLUMNS) + " |" for r in rs]
    return "\n".join([head, sep, *body])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    root = args.root if args.root.is_absolute() else E2E / args.root
    rs = rows(root)
    print(json.dumps({"rows": rs, **level0_vs_level1(rs)}, indent=1) if args.json else table(rs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
