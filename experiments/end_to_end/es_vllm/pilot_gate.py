#!/usr/bin/env python
"""Gate G3 of the Tulu 3.1 pilot, exactly as docs/end_to_end/03-tulu31-pilot.md fixes it.

    python -m es_vllm.pilot_gate          # from experiments/end_to_end

For each true-reward arm, d_g = center_reward(arm, g) - center_reward(random, g) over the
iterations both logged; the OLS slope of d on g with its standard error; the mean of d
over g = 20..29. Pass: slope > 2 SE and that mean > 0. Sigma: the passing arm with the
largest slope, the smaller sigma when two slopes are within one SE. Written before any
pilot result existed; the pass rule is not a parameter.
"""

import json
from pathlib import Path

import numpy as np

E2E = Path(__file__).resolve().parent.parent
ARMS = {"s5e-4": 5e-4, "s1e-3": 1e-3, "s2e-3": 2e-3}
CONTROL = "random"
LATE = range(20, 30)


def center(arm: str) -> dict:
    path = E2E / "runs" / f"tulu-pilot-{arm}" / "log.jsonl"
    if not path.exists():
        return {}
    return {r["iteration"]: r["center_reward"]
            for r in map(json.loads, path.read_text().splitlines())}


def ols(x, y):
    """Slope and its standard error."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    slope = (xc * (y - y.mean())).sum() / (xc ** 2).sum()
    resid = y - y.mean() - slope * xc
    se = np.sqrt((resid ** 2).sum() / (len(x) - 2) / (xc ** 2).sum())
    return float(slope), float(se)


def evaluate(arms: dict, control: dict) -> dict:
    rows = {}
    for arm, curve in arms.items():
        g = sorted(set(curve) & set(control))
        if len(g) < 3:
            rows[arm] = {"iterations": len(g), "status": "not enough iterations"}
            continue
        d = [curve[i] - control[i] for i in g]
        slope, se = ols(g, d)
        late = [curve[i] - control[i] for i in g if i in LATE]
        late_mean = float(np.mean(late)) if late else float("nan")
        rows[arm] = {"iterations": len(g), "slope": slope, "se": se,
                     "late_mean_d": late_mean,
                     "pass": bool(slope > 2 * se and late and late_mean > 0)}
    passing = [a for a, r in rows.items() if r.get("pass")]
    choice = None
    if passing:
        best = max(passing, key=lambda a: rows[a]["slope"])
        close = [a for a in passing
                 if rows[best]["slope"] - rows[a]["slope"] <= rows[best]["se"]]
        choice = min(close, key=lambda a: ARMS[a])
    return {"arms": rows, "passing": passing, "sigma_arm": choice}


def main() -> int:
    result = evaluate({a: center(a) for a in ARMS}, center(CONTROL))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
