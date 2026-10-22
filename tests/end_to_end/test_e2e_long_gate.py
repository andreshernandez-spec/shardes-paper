"""G4 on synthetic per-prompt rewards: a learning arm passes, a drifting one fails, the
pairing is what makes a small effect visible, and the comparisons line up by step."""

import pathlib
import sys

import numpy as np

E2E = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "end_to_end"
sys.path.insert(0, str(E2E))

from es_vllm import long_gate as gate  # noqa: E402

N = 1920


def prompts(seed):
    """Per-prompt rewards on the 0-10 scale: each prompt is hard or easy for every model."""
    rng = np.random.default_rng(seed)
    return rng.random(N) < 0.55


def model(base, flips, seed):
    """`base` with a fraction `flips` of the failed prompts now solved."""
    rng = np.random.default_rng(seed)
    out = base.copy()
    out[(~base) & (rng.random(N) < flips)] = True
    return (10.0 * out).tolist()


def test_learning_arm_passes_and_drift_fails():
    base = prompts(0)
    control = {g: model(base, 0.0, 1) for g in gate.POINTS}
    learner = {g: model(base, 0.002 * g, 2) for g in gate.POINTS}
    drift = {g: model(base, 0.0, 3) for g in gate.POINTS}
    assert gate.evaluate(learner, control, {})["status"] == "pass"
    assert gate.evaluate(drift, control, {})["status"] == "fail"


def test_pairing_sees_what_the_unpaired_error_hides():
    base = prompts(4)
    a, b = model(base, 0.03, 5), model(base, 0.0, 6)
    p = gate.paired(a, b)
    unpaired = np.sqrt(np.var(a, ddof=1) / N + np.var(b, ddof=1) / N)
    assert p["d"] > 2 * p["se"] and p["d"] < 2 * unpaired


def test_rl_branches_line_up_with_es_iterations_and_incomplete_is_reported():
    base = prompts(7)
    arm = {0: model(base, 0, 8), 30: model(base, 0.01, 9)}
    rl = {"dpo-start": arm[0], "rl-step120": model(base, 0.2, 10)}
    r = gate.evaluate(arm, {0: arm[0]}, rl)
    assert r["status"] == "incomplete" and list(r["rl_minus_es"]) == [30]
    assert r["rl_minus_es"][30]["d"] > 0
    assert r["measurement_check"]["identical_prompts"] == N and r["measurement_check"]["d"] == 0
