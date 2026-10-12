"""es_vllm/probe_table.py: one row per prediction record, formatted."""

import json
import pathlib
import sys

E2E = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "end_to_end"
sys.path.insert(0, str(E2E))

from es_vllm import probe_table as pt  # noqa: E402


def prediction(setting, call, update=True):
    return {"setting": setting,
            "level0": {"live_share": 0.81, "gradient_sharedness": 0.031, "curvature_per_length2": -0.00212,
                       "rl_first_step_gain": 0.0012, "first_task_damaged_random": "MATH"},
            "level1": {"sigma": {"reliability_at_prompts": 0.42, "cross_prompt_correlation": 0.3}},
            "update": {"G": 0.0047, "C": -0.00008, "first_task_damaged": "MATH"} if update else None,
            "rule": {"es": {"fitted_net_per_update": 0.069} if update else None, "ratio_es_over_rl_rollouts": 31.5 if update else None,
                     "call": call}}


def test_rows_and_table(tmp_path):
    for name, call, update in (("a-x", "fails: rollouts", True), ("b-y", "fails: no odd response", False)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "prediction.json").write_text(json.dumps(prediction(name.replace("-", "/"), call, update)))
    rows = pt.rows(tmp_path)
    assert [r["setting"] for r in rows] == ["a/x", "b/y"]
    assert rows[0]["net"] == 0.069 and rows[0]["damaged"] == "MATH / MATH"
    assert rows[1]["G"] is None and rows[1]["net"] is None and rows[1]["damaged"] == "MATH / -"
    t = pt.table(rows).splitlines()
    assert len(t) == 4 and t[0].startswith("| setting |") and "| 31.5 | - | fails: rollouts | - |" in t[2]
    assert "| - | - | - | fails: no odd response | - |" in t[3]


def test_fmt():
    assert pt.fmt(None) == "-" and pt.fmt(0) == "0" and pt.fmt("works") == "works"
    assert pt.fmt(0.0047) == "4.70e-03" and pt.fmt(0.42) == "0.42" and pt.fmt(31.456) == "31.5"
    assert pt.fmt(12345.6) == "12,346"
