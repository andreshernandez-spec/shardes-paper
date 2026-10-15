#!/usr/bin/env python
"""One row per probed setting, from `runs/probe/*/prediction.json`, as a markdown table.

    python -m es_vllm.probe_table runs/probe            # the table in runs/probe/README.md
    python -m es_vllm.probe_table runs/probe --json     # the same rows as JSON
    python -m es_vllm.probe_table runs/probe2 --against runs/probe   # the second probe beside the first
    python -m es_vllm.probe_table runs/probe2 --offtask              # the members' off-task diagnostic

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
           ("offtask", "off-task at target"), ("damaged", "damaged first (scan / update)"),
           ("net_step", "net: step"), ("net_updates", "net: updates"), ("net_ratio", "net: ES/RL"), ("net_call", "net: call"))


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
    n = pred.get("rule_net") or {}
    n_es = n.get("es") or {}
    return {"setting": pred["setting"], "live": l0["live_share"], "share": l0["gradient_sharedness"],
            "c": l0["curvature_per_length2"], "rl": l0["rl_first_step_gain"],
            "rel": s.get("reliability_at_prompts"), "corr": s.get("cross_prompt_correlation"),
            "G": u.get("G"), "C": u.get("C"), "net": r["es"]["fitted_net_per_update"] if r["es"] else None,
            "ratio": r["ratio_es_over_rl_rollouts"],
            "ratio90": f"{fmt(ratio90[0])} to {fmt(ratio90[1])}" if ratio90 else None, "call": r["call"],
            "offtask": u.get("offtask_change_at_target"),
            "damaged": f"{l0['first_task_damaged_random']} / {u.get('first_task_damaged', '-')}",
            "net_step": n_es.get("step_multiple"), "net_updates": n_es.get("updates"),
            "net_ratio": n.get("ratio_es_over_rl_rollouts"), "net_call": n.get("call")}


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


def table(rs: list, columns=COLUMNS) -> str:
    head = "| " + " | ".join(name for _, name in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(fmt(r[key]) for key, _ in columns) + " |" for r in rs]
    return "\n".join([head, sep, *body])


def measured(pred: dict) -> dict:
    """A probe's measured inputs with their standard errors: the validation block's `G`
    and `C` per unit update, RL's slope along the sign direction, and the calls."""
    import numpy as np  # noqa: PLC0415

    l0, u, n = pred["level0"], pred["update"] or {}, pred.get("rule_net") or {}
    v = n.get("validation") or {}
    sgn = l0.get("sign_direction_responses") or {}
    used = [L for L, x in sgn.items() if x["usable"]] or list(sgn)
    rl = sgn[min(used, key=float)]["by_block"]["validation"] if sgn else {}
    return {"G": v.get("G", u.get("G")), "G_se": v.get("G_se", u.get("G_se")), "C": v.get("C"), "C_se": v.get("C_se"),
            "rl_slope": rl.get("slope_along_rl_direction"), "rl_slope_se": rl.get("slope_se"),
            "length_used": n.get("length_used"), "lam": (u.get("responses") or {}).get(n.get("length_used") or "", {}).get("lam"),
            "responses": u.get("responses") or {},
            "call": n.get("call"), "ratio": n.get("ratio_es_over_rl_rollouts"),
            "curve_call": (pred.get("rule_curve") or {}).get("call"),
            "hypot": lambda a, b: float(np.hypot(a, b))}


def at_length(m: dict, length) -> dict:
    """A probe's validation response at a scanned length (the key, or the nearest length
    within 1%), or {}."""
    if length is None:
        return {}
    r = m["responses"].get(length)
    if r is None:
        near = [k for k in m["responses"] if abs(float(k) - float(length)) <= 0.01 * float(length)]
        r = m["responses"][near[0]] if near else None
    return r["by_block"]["validation"] if r else {}


AGAINST = (("setting", "setting"), ("G1", "G, first"), ("G2", "G, second"), ("G_agree", "G within 2 SE"),
           ("G2m", "G, second at the first's length"), ("G_agree_m", "within 2 SE there"),
           ("C1", "C, first"), ("C2", "C, second"), ("C_agree", "C within 2 SE"),
           ("rl1", "RL slope, first"), ("rl2", "RL slope, second"), ("rl_agree", "RL slope within 2 SE"),
           ("lam2", "longest usable multiple, second"), ("call1", "call, first"), ("call2", "call, second"))


def against(root: Path, other: Path) -> list:
    """Each setting probed in both roots (`other` first, `root` second): the measured
    inputs side by side with their standard errors, and whether each pair agrees within
    two standard errors of the difference (docs/end_to_end/10, the second probe's
    expectation 1)."""
    out = []
    for p in sorted(root.glob("*/prediction.json")):
        q = other / p.parent.name / "prediction.json"
        if not q.exists():
            continue
        a, b = measured(json.loads(q.read_text())), measured(json.loads(p.read_text()))

        def pair(key):
            x, y = a[key], b[key]
            sx, sy = a[key + "_se"], b[key + "_se"]
            if x is None or y is None or sx is None or sy is None:
                return None, None, None
            agree = abs(x - y) <= 2 * a["hypot"](sx, sy)
            return f"{x:.3g} ± {sx:.2g}", f"{y:.3g} ± {sy:.2g}", "yes" if agree else "no"

        g1, g2, ga = pair("G")
        c1, c2, ca = pair("C")
        r1, r2, ra = pair("rl_slope")
        m = at_length(b, a["length_used"])                 # the second probe read where the first was
        g2m, gam = None, None
        if m and a["G"] is not None and a["G_se"] is not None:
            g2m = f"{m['G']:.3g} ± {m['G_se']:.2g}"
            gam = "yes" if abs(a["G"] - m["G"]) <= 2 * a["hypot"](a["G_se"], m["G_se"]) else "no"
        out.append({"setting": p.parent.name, "G1": g1, "G2": g2, "G_agree": ga, "G2m": g2m, "G_agree_m": gam,
                    "C1": c1, "C2": c2, "C_agree": ca,
                    "rl1": r1, "rl2": r2, "rl_agree": ra, "lam2": b["lam"], "call1": a["call"], "call2": b["call"]})
    return out


OFFTASK = (("setting", "setting"), ("block", "block"), ("G_members", "G from the members"), ("G_direct", "G from the steps scan"),
           ("corr", "corr. with the train odd effect"), ("corr90", "90% over pairs"), ("even", "even part at the member length"),
           ("per_gain", "off-task change per unit gain, members / direct"), ("at_target", "at the target gain, members / direct"))


def offtask(root: Path) -> list:
    """One row per block per setting whose record carries the members' off-task decodes
    (`level1_offtask`): the first-order change per unit update from the members beside
    the steps scan's, and the correlation across pairs with the train odd effect."""
    out = []
    for p in sorted(root.glob("*/prediction.json")):
        pred = json.loads(p.read_text())
        lo = pred.get("level1_offtask")
        if not lo:
            continue
        u = pred["update"] or {}
        direct = (u.get("responses") or {}).get(u.get("length_used") or "", {}).get("by_block", {})
        for b, v in sorted(lo["by_block"].items()):
            d = direct.get(b, {})
            c90 = v.get("correlation_90_pairs")
            first = b == sorted(lo["by_block"])[0]
            pg = lo.get("offtask_change_per_unit_gain")
            dg = (u["G_offtask"] / u["G"]) if u.get("G_offtask") is not None and u.get("G") else None
            out.append({"setting": p.parent.name, "block": b,
                        "G_members": f"{v['G_from_members']:.3g} ± {v['G_from_members_se']:.2g}",
                        "G_direct": f"{d['G']:.3g} ± {d['G_se']:.2g}" if d else None,
                        "corr": v.get("correlation_with_train_odd"),
                        "corr90": f"{fmt(c90[0])} to {fmt(c90[1])}" if c90 else None,
                        "even": f"{v['even_mean']:.3g} ± {v['even_se']:.2g}",
                        "per_gain": (f"{fmt(pg)} / {fmt(dg)}" if first else None),
                        "at_target": (f"{fmt(lo.get('offtask_change_at_target'))} / {fmt(u.get('offtask_change_at_target'))}" if first else None)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--against", type=Path, metavar="FIRST", help="the earlier probe's root: the measured inputs side by side")
    ap.add_argument("--offtask", action="store_true", help="the members' off-task diagnostic, one row per block")
    args = ap.parse_args(argv)
    root = args.root if args.root.is_absolute() else E2E / args.root
    if args.against:
        other = args.against if args.against.is_absolute() else E2E / args.against
        print(table(against(root, other), AGAINST))
        return 0
    if args.offtask:
        print(table(offtask(root), OFFTASK))
        return 0
    rs = rows(root)
    print(json.dumps({"rows": rs, **level0_vs_level1(rs)}, indent=1) if args.json else table(rs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
