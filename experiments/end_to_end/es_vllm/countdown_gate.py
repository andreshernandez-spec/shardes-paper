#!/usr/bin/env python
"""T2's gate, exactly as docs/end_to_end/06-countdown-check.md fixes it.

    python -m es_vllm.countdown_gate --compact runs/countdown-s1-ref   # on the pod
    python -m es_vllm.countdown_gate                                   # the gate

`--compact` reads es-at-scale's evaluation outputs (one JSON of 2,000 responses per
evaluation) and writes the `eval.jsonl` our runs write: per evaluation the mean reward,
the solve rate (answer reward > 0), the mean format score, the mean length, and those
per prompt, without the text. Its file for the evaluation before training is named
iteration1 and the others by their update count.

The gate: per run, the mean evaluation reward over the 20 evaluations after 5 to 100
updates; per implementation, the mean of its two seeds and their spread S. Inconclusive
if either reference run gains less than 0.03 from its start to the mean of its last four
evaluations. Pass if |ours - reference| <= 2 * max(S_ours, S_reference, 0.01). Written
before any T2 result existed; the rule is not a parameter.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

E2E = Path(__file__).resolve().parent.parent
SEEDS = (1, 2)
POINTS = tuple(range(5, 101, 5))
LATE = (85, 90, 95, 100)
MIN_GAIN, FLOOR = 0.03, 0.01


def compact(run: Path) -> Path:
    files = sorted((run / "run" / "eval-output").glob("model_eval_task*_iteration*.json"))
    if not files:
        raise SystemExit(f"no evaluation outputs under {run}")
    recs = []
    for f in files:
        k = int(re.search(r"_iteration(\d+)\.json$", f.name).group(1))
        updates = 0 if k == 1 else k
        items = json.loads(f.read_text())
        reward = [float(x["reward"]) for x in items]
        answer = [float(x["format"]["answer_reward"]) if x.get("format") else 0.0 for x in items]
        fmt = [float(x["format"]["format_reward"]) if x.get("format") else 0.0 for x in items]
        length = [int(x["response_length"]) for x in items]
        recs.append({"updates": updates, "reward": float(np.mean(reward)),
                     "solved": float(np.mean([a > 0 for a in answer])),
                     "format": float(np.mean(fmt)), "mean_len": float(np.mean(length)),
                     "per_prompt": {"reward": reward, "answer": answer, "format": fmt,
                                    "length": length}})
    recs.sort(key=lambda r: r["updates"])
    path = run / "eval.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return path


def curve(run: Path) -> dict:
    path = run / "eval.jsonl"
    if not path.exists():
        return {}
    return {r["updates"]: r["reward"] for r in map(json.loads, path.read_text().splitlines())}


def evaluate(ours: dict, ref: dict) -> dict:
    """`ours`, `ref`: seed -> {updates: eval reward}."""
    def run_stats(c):
        missing = [p for p in (0, *POINTS) if p not in c]
        if missing:
            return {"missing": missing}
        return {"auc": float(np.mean([c[p] for p in POINTS])), "start": c[0],
                "late": float(np.mean([c[p] for p in LATE]))}

    runs = {"ours": {s: run_stats(c) for s, c in ours.items()},
            "reference": {s: run_stats(c) for s, c in ref.items()}}
    complete = all(len(v) == len(SEEDS) and all("missing" not in r for r in v.values())
                   for v in runs.values())
    if not complete:
        return {"runs": runs, "status": "incomplete"}
    gains = {s: r["late"] - r["start"] for s, r in runs["reference"].items()}
    impl = {}
    for k, v in runs.items():
        aucs = [r["auc"] for r in v.values()]
        impl[k] = {"auc": float(np.mean(aucs)), "spread": float(abs(aucs[0] - aucs[1]))}
    diff = impl["ours"]["auc"] - impl["reference"]["auc"]
    bound = 2 * max(impl["ours"]["spread"], impl["reference"]["spread"], FLOOR)
    if min(gains.values()) < MIN_GAIN:
        status = "inconclusive"
    else:
        status = "pass" if abs(diff) <= bound else "fail"
    return {"runs": runs, "reference_gains": gains, "implementations": impl,
            "difference": diff, "bound": bound, "status": status}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact", type=Path)
    args = ap.parse_args(argv)
    if args.compact:
        print(compact(E2E / args.compact))
        return 0
    runs = E2E / "runs"
    ours = {s: curve(runs / f"countdown-s{s}") for s in SEEDS}
    ref = {s: curve(runs / f"countdown-s{s}-ref") for s in SEEDS}
    res = evaluate({s: c for s, c in ours.items() if c}, {s: c for s, c in ref.items() if c})
    (runs / "countdown-gate.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
