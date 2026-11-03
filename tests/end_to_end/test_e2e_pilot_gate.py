"""The pilot gate on synthetic curves: it passes a learning arm, fails a flat one, and
picks the smaller sigma when two slopes are within one standard error."""

import pathlib
import sys

E2E = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "end_to_end"
sys.path.insert(0, str(E2E))

from es_vllm import pilot_gate as gate  # noqa: E402


def curve(slope, noise, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    return {g: 7.0 + slope * g + noise * rng.standard_normal() for g in range(30)}


def test_learning_arm_passes_flat_arm_fails():
    control = curve(0.0, 0.2, 0)
    r = gate.evaluate({"s1e-3": curve(0.05, 0.2, 1), "s2e-3": curve(0.0, 0.2, 2)}, control)
    assert r["arms"]["s1e-3"]["pass"] and not r["arms"]["s2e-3"]["pass"]
    assert r["sigma_arm"] == "s1e-3"


def test_close_slopes_pick_the_smaller_sigma():
    control = curve(0.0, 0.05, 0)
    r = gate.evaluate({"s1e-3": curve(0.05, 0.05, 1), "s5e-4": curve(0.049, 0.05, 3)},
                      control)
    assert set(r["passing"]) == {"s1e-3", "s5e-4"}
    assert r["sigma_arm"] == "s5e-4"


def test_missing_iterations_do_not_pass():
    r = gate.evaluate({"s1e-3": {0: 1.0, 1: 2.0}}, {0: 1.0, 1: 1.0})
    assert r["passing"] == [] and r["sigma_arm"] is None
