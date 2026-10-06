"""T2's gate on synthetic curves: matching learners pass, a flat backend fails, a
reference that does not learn makes it inconclusive, a missing run makes it incomplete;
and es-at-scale's evaluation files compact to our eval.jsonl."""

import json
import pathlib
import sys

E2E = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "end_to_end"
sys.path.insert(0, str(E2E))

from es_vllm import countdown_gate as gate  # noqa: E402


def curve(start, end, rate, noise, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    return {u: start + (end - start) * (1 - np.exp(-rate * u)) + noise * rng.standard_normal()
            for u in range(0, 101, 5)}


def test_matching_learners_pass():
    ours = {1: curve(0.055, 0.15, 0.05, 0.003, 1), 2: curve(0.055, 0.15, 0.05, 0.003, 2)}
    ref = {1: curve(0.055, 0.15, 0.05, 0.003, 3), 2: curve(0.055, 0.145, 0.05, 0.003, 4)}
    assert gate.evaluate(ours, ref)["status"] == "pass"


def test_flat_backend_fails_low():
    ours = {1: curve(0.055, 0.056, 0.05, 0.003, 1), 2: curve(0.055, 0.056, 0.05, 0.003, 2)}
    ref = {1: curve(0.055, 0.15, 0.05, 0.003, 3), 2: curve(0.055, 0.15, 0.05, 0.003, 4)}
    r = gate.evaluate(ours, ref)
    assert r["status"] == "fail" and r["difference"] < 0


def test_reference_that_does_not_learn_is_inconclusive():
    ours = {1: curve(0.055, 0.15, 0.05, 0.003, 1), 2: curve(0.055, 0.15, 0.05, 0.003, 2)}
    ref = {1: curve(0.055, 0.07, 0.05, 0.003, 3), 2: curve(0.055, 0.15, 0.05, 0.003, 4)}
    assert gate.evaluate(ours, ref)["status"] == "inconclusive"


def test_missing_points_or_runs_are_incomplete():
    full = curve(0.055, 0.15, 0.05, 0.003, 1)
    short = {u: v for u, v in full.items() if u <= 50}
    assert gate.evaluate({1: full, 2: short}, {1: full, 2: full})["status"] == "incomplete"
    assert gate.evaluate({1: full}, {1: full, 2: full})["status"] == "incomplete"


def test_compact_reads_es_at_scale_outputs(tmp_path):
    out = tmp_path / "run" / "eval-output"
    out.mkdir(parents=True)
    item = {"reward": 1.1, "format": {"answer_reward": 1.0, "format_reward": 1.0},
            "response_length": 40}
    miss = {"reward": 0.06, "format": {"answer_reward": 0.0, "format_reward": 0.6},
            "response_length": 512}
    for k, items in ((1, [miss, miss]), (5, [item, miss]), (10, [item, item])):
        (out / f"model_eval_taskcountdown_eval_iteration{k}.json").write_text(json.dumps(items))
    recs = [json.loads(x) for x in gate.compact(tmp_path).read_text().splitlines()]
    assert [r["updates"] for r in recs] == [0, 5, 10]
    assert [r["solved"] for r in recs] == [0.0, 0.5, 1.0]
    assert abs(recs[1]["reward"] - 0.58) < 1e-12 and recs[0]["mean_len"] == 512
