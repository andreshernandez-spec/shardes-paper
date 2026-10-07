#!/usr/bin/env python
"""Gate G4 of the longer Tulu 3.1 arm, exactly as docs/end_to_end/07-tulu31-long.md fixes it.

    python -m es_vllm.long_gate          # from experiments/end_to_end

Every comparison is paired over the 1,920 held-out prompts (RL steps 481 to 520), which
the ES runs score in-run (`heldout.jsonl`) and `heldout.py` scores for the RL branches, in
the same order: d = mean over prompts of (reward_a - reward_b), SE = sd / sqrt(n).

- G4: ES arm minus control after 120 updates. Pass: d > 2 SE.
- Reported: ES after g updates against its own start, against the control, and against
  the RL branch at step 4g (matched in prompts and rollouts); ES at 0 against
  heldout.py's DPO start (the two measurements should agree); the slope of the in-run
  center-reward difference over the 120 iterations; and whether the arm's first 30
  iterations reproduce the pilot's s5e-4 log.

Written before any result of the run existed; the pass rule is not a parameter.
"""

import json
from pathlib import Path

import numpy as np

from es_vllm.pilot_gate import ols

E2E = Path(__file__).resolve().parent.parent
ARM, CONTROL, RL = "tulu-long-s5e-4", "tulu-long-random", "heldout-481-520"
POINTS = (0, 30, 60, 90, 120)
FINAL = 120


def paired(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"unpaired: {a.shape} vs {b.shape}")
    d = a - b
    return {"d": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))), "n": len(d)}


def heldout(run: Path) -> dict:
    path = run / "heldout.jsonl"
    if not path.exists():
        return {}
    return {r["iteration"]: r["per_prompt"] for r in map(json.loads, path.read_text().splitlines())}


def rl(d: Path) -> dict:
    return {p.stem: json.loads(p.read_text())["per_prompt"] for p in sorted(d.glob("*.json"))}


def center(run: Path) -> dict:
    path = run / "log.jsonl"
    if not path.exists():
        return {}
    return {r["iteration"]: r["center_reward"]
            for r in map(json.loads, path.read_text().splitlines())}


def reproduction(run: Path, pilot: Path) -> dict:
    """Do the arm's first iterations repeat the pilot's, fitness for fitness, digest for digest?"""
    if not (run / "log.jsonl").exists() or not (pilot / "log.jsonl").exists():
        return {}
    a = [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()]
    b = [json.loads(x) for x in (pilot / "log.jsonl").read_text().splitlines()]
    n = min(len(a), len(b))
    return {"iterations": n,
            "fitness_identical": sum(a[g]["fitness"] == b[g]["fitness"] for g in range(n)),
            "digests_compared": sum("digest" in a[g] and "digest" in b[g] for g in range(n)),
            "digests_identical": sum(a[g].get("digest") is not None
                                     and a[g].get("digest") == b[g].get("digest")
                                     for g in range(n))}


def evaluate(arm: dict, control: dict, rl_runs: dict, arm_center=None, control_center=None):
    res = {"g4": None, "status": "incomplete"}
    if FINAL in arm and FINAL in control:
        g4 = paired(arm[FINAL], control[FINAL])
        res["g4"] = {**g4, "pass": bool(g4["d"] > 2 * g4["se"])}
        res["status"] = "pass" if res["g4"]["pass"] else "fail"
    res["vs_start"] = {g: paired(arm[g], arm[0]) for g in POINTS if g in arm and 0 in arm}
    res["vs_control"] = {g: paired(arm[g], control[g]) for g in POINTS
                         if g in arm and g in control}
    res["rl_minus_es"] = {g: paired(rl_runs[f"rl-step{4 * g}"], arm[g]) for g in POINTS[1:]
                          if g in arm and f"rl-step{4 * g}" in rl_runs}
    if 0 in arm and "dpo-start" in rl_runs:
        res["measurement_check"] = {**paired(arm[0], rl_runs["dpo-start"]),
                                    "identical_prompts": int(sum(
                                        x == y for x, y in zip(arm[0], rl_runs["dpo-start"])))}
    res["means"] = {"arm": {g: float(np.mean(v)) for g, v in sorted(arm.items())},
                    "control": {g: float(np.mean(v)) for g, v in sorted(control.items())},
                    "rl": {k: float(np.mean(v)) for k, v in rl_runs.items()}}
    if arm_center and control_center:
        g = sorted(set(arm_center) & set(control_center))
        if len(g) >= 3:
            slope, se = ols(g, [arm_center[i] - control_center[i] for i in g])
            res["center_slope"] = {"slope": slope, "se": se, "iterations": len(g)}
    return res


def main() -> int:
    runs = E2E / "runs"
    res = evaluate(heldout(runs / ARM), heldout(runs / CONTROL), rl(runs / RL),
                   center(runs / ARM), center(runs / CONTROL))
    res["reproduces_pilot"] = reproduction(runs / ARM, runs / "tulu-pilot-s5e-4")
    (runs / "long-gate.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: res[k] for k in ("status", "g4", "reproduces_pilot",
                                          "measurement_check", "center_slope") if k in res},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
