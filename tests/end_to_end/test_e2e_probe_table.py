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


def test_spearman_and_the_level0_test():
    assert pt.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert pt.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
    rs = [{"share": s, "corr": c} for s, c in ((0.02, 0.003), (0.009, 0.002), (0.007, -0.0007), (0.004, -0.0003))]
    out = pt.level0_vs_level1(rs)
    assert out["settings"] == 4 and abs(out["spearman_sharedness_vs_cross_prompt_correlation"] - 0.8) < 1e-9
    assert pt.level0_vs_level1(rs[:2])["spearman_sharedness_vs_cross_prompt_correlation"] is None


def test_fmt():
    assert pt.fmt(None) == "-" and pt.fmt(0) == "0" and pt.fmt("works") == "works"
    assert pt.fmt(0.0047) == "4.70e-03" and pt.fmt(0.42) == "0.42" and pt.fmt(31.456) == "31.5"
    assert pt.fmt(12345.6) == "12,346"


def full_prediction(G, G_se, C, C_se, slope, slope_se, call, offtask=None):
    """A record with the fields the replication and off-task tables read."""
    p = prediction("qwen3b/countdown", call)
    p["level0"]["sign_direction_responses"] = {"0.3": {"usable": True, "by_block": {"validation": {
        "slope_along_rl_direction": slope, "slope_se": slope_se}}}}
    p["update"].update({"G": G, "G_se": G_se, "G_offtask": -0.0005, "offtask_change_at_target": 0.1 * -0.0005 / G, "length_used": "7.5",
                        "responses": {"7.5": {"lam": 20.0, "by_block": {"validation": {"G": G, "G_se": G_se, "C": C, "C_se": C_se},
                                                                        "offtask": {"G": -0.0005, "G_se": 0.0004, "C": -1e-5, "C_se": 1e-5}}}}})
    p["rule_net"] = {"call": call, "ratio_es_over_rl_rollouts": 3.0, "length_used": "7.5",
                     "validation": {"G": G, "G_se": G_se, "C": C, "C_se": C_se}}
    p["level1_offtask"] = offtask
    return p


def test_against_compares_the_measured_inputs_within_two_standard_errors(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    for root, G, C, slope, call in ((first, 0.0023, -4.7e-5, 0.23, "undetermined: cost unresolved"),
                                     (second, 0.0019, -9e-5, 0.21, "works")):
        (root / "qwen3b-countdown").mkdir(parents=True)
        (root / "qwen3b-countdown" / "prediction.json").write_text(json.dumps(full_prediction(G, 0.0004, C, 1.5e-5, slope, 0.06, call)))
    (second / "other-setting").mkdir()                       # probed once only: not compared
    (second / "other-setting" / "prediction.json").write_text(json.dumps(full_prediction(0.001, 0.0004, -1e-5, 1e-5, 0.1, 0.1, "works")))
    rows = pt.against(second, first)
    assert len(rows) == 1 and rows[0]["setting"] == "qwen3b-countdown"
    assert rows[0]["G_agree"] == "yes" and rows[0]["rl_agree"] == "yes"       # |0.0004| < 2 * hypot(0.0004, 0.0004)
    assert rows[0]["C_agree"] == "no"                                        # |4.3e-5| > 2 * hypot(1.5e-5, 1.5e-5)
    assert rows[0]["call1"].startswith("undetermined") and rows[0]["call2"] == "works" and rows[0]["lam2"] == 20.0
    # the second probe read at the first probe's length (the same key here): 0.0019 against 0.0023
    assert rows[0]["G2m"] == "0.0019 ± 0.0004" and rows[0]["G_agree_m"] == "yes"
    t = pt.table(rows, pt.AGAINST).splitlines()
    assert len(t) == 3 and t[0].startswith("| setting | G, first |") and "| yes | " in t[2] and "| no | " in t[2]


def test_offtask_table_puts_the_members_beside_the_scan(tmp_path):
    lo = {"n": 64, "pairs": 32, "sigma": 5e-4, "prompts": {"validation": 600, "offtask": 600}, "train_odd_sd": 0.012, "train_odd_true_sd": 0.01,
          "by_block": {"validation": {"G_from_members": 0.0021, "G_from_members_se": 0.0006, "correlation_with_train_odd": 0.4,
                                      "correlation_90_pairs": [0.1, 0.6], "even_mean": -0.01, "even_se": 0.004, "odd_per_pair_sd": 0.009, "odd_per_pair_true_sd": 0.008},
                       "offtask": {"G_from_members": -0.0008, "G_from_members_se": 0.0005, "correlation_with_train_odd": -0.2,
                                   "correlation_90_pairs": [-0.45, 0.05], "even_mean": -0.004, "even_se": 0.003, "odd_per_pair_sd": 0.091, "odd_per_pair_true_sd": 0.09}},
          "offtask_change_per_unit_gain": -0.381, "offtask_change_at_target": -0.0381, "validation_resolved": True}
    (tmp_path / "qwen3b-countdown").mkdir()
    (tmp_path / "qwen3b-countdown" / "prediction.json").write_text(json.dumps(full_prediction(0.0023, 0.0004, -4.7e-5, 1.5e-5, 0.23, 0.06, "works", lo)))
    (tmp_path / "qwen3b-gsm8k").mkdir()                       # no off-task decodes: no rows
    (tmp_path / "qwen3b-gsm8k" / "prediction.json").write_text(json.dumps(full_prediction(0.001, 0.0004, -1e-5, 1e-5, 0.1, 0.1, "works")))
    rows = pt.offtask(tmp_path)
    assert [(r["setting"], r["block"]) for r in rows] == [("qwen3b-countdown", "offtask"), ("qwen3b-countdown", "validation")]
    assert rows[0]["G_members"] == "-0.0008 ± 0.0005" and rows[0]["G_direct"] == "-0.0005 ± 0.0004"
    assert rows[0]["corr90"] == "-0.45 to 0.05" and rows[0]["per_gain"].startswith("-0.381 / -0.217")
    assert rows[0]["odd_sd"] == 0.09 and abs(rows[0]["odd_sd_ratio"] - 9.0) < 1e-9 and abs(rows[1]["odd_sd_ratio"] - 0.8) < 1e-9
    assert rows[1]["per_gain"] is None and rows[1]["G_direct"] == "0.0023 ± 0.0004"
    t = pt.table(rows, pt.OFFTASK).splitlines()
    assert len(t) == 4 and t[0].startswith("| setting | block |")


def test_prescriptions_table_and_direct_comparison(tmp_path):
    for name, step in (("a-x", 3.0), ("b-y", 6.85)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "prediction.json").write_text(json.dumps(full_prediction(0.002, 0.0004, -4.7e-5, 1.5e-5, 0.23, 0.06, "works")))
        (tmp_path / name / "prescription.json").write_text(json.dumps({
            "method": "anchored", "step_multiple": step, "shrink": 0.1, "fallback": step < 5, "first_displacement_multiple": 0.9 * step,
            "response_at_first_displacement": {"odd": 0.0126, "odd_se": 0.0037, "even": 0.0088, "even_se": 0.0049, "within_scan": True},
            "P1_learns": "learns", "P2_collapses": "no call",
            "P2_spread_ratio": {"ratio": 0.65, "interval_90": [0.38, 2.3], "unresolved": False},
            "P3_pace": {"predicted_validation_change_after_5": 0.0708, "band": [0.0177, 0.0708]},
            "P4_rl_pace": {"predicted_step_to_target": 23.0, "band": [11.5, 46.0]} if step < 5 else None}))
    rows = pt.prescriptions(tmp_path)
    assert [r["setting"] for r in rows] == ["a-x", "b-y"] and rows[0]["fallback"] == "yes" and rows[1]["fallback"] == "no"
    assert rows[0]["odd"] == "+0.0126 ± 0.0037" and rows[0]["ratio"] == "0.65, 0.38 to 2.3" and rows[0]["P4"] == "23.0" and rows[1]["P4"] is None
    t = pt.table(rows, pt.PRESCRIPTIONS).splitlines()
    assert len(t) == 4 and t[0].startswith("| setting | step")
    # two setting directories compare directly, whatever their names
    rows = pt.against(tmp_path / "b-y", tmp_path / "a-x")
    assert len(rows) == 1 and rows[0]["setting"] == "b-y" and rows[0]["G_agree"] == "yes"
