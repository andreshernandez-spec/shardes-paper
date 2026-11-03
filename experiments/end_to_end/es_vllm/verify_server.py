"""Reward server for the Tulu 3.1 verifiers, run in `.venv-verify` (the run's sympy and antlr).

    .venv-verify/bin/python -m es_vllm.verify_server      # from experiments/end_to_end

One JSON object per line on stdin: `{"text", "ground_truth", "dataset", "stopped"}`; one
line back per request: `{"reward": float}`. A response that never emitted eos
(`stopped` false) gets the run's `penalty_reward_value`, 0.0, whatever it says. On start
the server checks sympy's LaTeX parser and prints one line, `{"ready": {...}}`, or exits
with the error; a missing parser would otherwise zero MATH rewards without a word.
"""

import json
import sys

from verifiers import tulu31


def main() -> int:
    try:
        env = tulu31.check_environment()
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True)
        return 1
    print(json.dumps({"ready": env}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        if not req.get("stopped", True):
            r = tulu31.PENALTY_REWARD_VALUE
        else:
            r = tulu31.reward(req["text"], req["ground_truth"], req["dataset"])
        print(json.dumps({"reward": float(r)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
