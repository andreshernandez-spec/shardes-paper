"""es_vllm/probe_run.py: the analysis of a run's records and the prediction block, no GPU."""

import json
import pathlib
import sys

import numpy as np

E2E = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "end_to_end"
sys.path.insert(0, str(E2E))

from es_vllm import probe_run as pr  # noqa: E402


def write_prediction(path, G=0.002, C=-3e-5, G_off=-0.0003, step=15.0):
    path.mkdir(parents=True, exist_ok=True)
    (path / "prediction.json").write_text(json.dumps({
        "setting": "qwen3b/countdown",
        "update": {"G": G, "G_se": 0.0005, "C": C, "C_se": 2e-5, "G_offtask": G_off, "unit_norm": 3.3},
        "rule": {"call": "works", "ratio_es_over_rl_rollouts": 2.8,
                 "es": {"step_multiple": step, "updates": 2.9, "fitted_net_per_update": 0.033}}}))
    (path / "env.json").write_text(json.dumps({"block_digests": {"train": "a", "validation": "b", "test": "c", "offtask": "d"}}))


def test_stream_selection_excludes_protected_rows_and_repeats():
    """The stream passes over its rows more than once and the probe's blocks come from
    it: a batch holds distinct GSM8K rows, none of them protected, in stream order."""
    sys.path.insert(0, str(E2E))
    from tulu31_data import row_hash  # noqa: PLC0415

    def row(i, dataset="gsm8k"):
        return {"messages": [{"role": "user", "content": f"q{i}"}], "ground_truth": str(i), "dataset": dataset}

    stream = [row(i) for i in range(10)] + [row(3, "MATH")] + [row(i) for i in range(10)]   # a second pass
    protected = {row_hash(row(2)), row_hash(row(5))}
    batches = pr.select_stream(stream, protected, need=6, per=3)
    got = [r["ground_truth"] for b in batches for r in b]
    assert got == ["0", "1", "3", "4", "6", "7"]
    assert len(batches) == 2 and len(batches[0]) == 3
    try:
        pr.select_stream(stream, protected, need=9, per=3)
    except SystemExit as e:
        assert "needed" in str(e)
    else:
        raise AssertionError("eight distinct unprotected rows cannot serve nine")


def ev(it, test, val, off):
    return {"iteration": it, "by_block": {"validation": {"change": val}, "test": {"change": test}, "offtask": {"change": off}}}


def test_observed_classes_follow_the_frozen_definitions():
    # the first run: the target reached at 15 with GSM8K already 0.28 down, then a collapse
    first = [ev(0, 0, 0, 0), ev(5, 0.033, 0.091, -0.067), ev(10, 0.070, 0.057, -0.123), ev(15, 0.135, 0.115, -0.277),
             ev(20, 0.060, 0.112, -0.600), ev(25, -0.005, -0.012, -0.843), ev(30, -0.118, -0.114, -0.837)]
    o = pr.observed_class(first)
    assert o["class"] == "fails" and o["class_ignoring_retained"] == "works" and o["collapsed"]
    assert o["best_test_change"] == 0.135 and o["at_iteration_ignoring_retained"] == 15
    # the second: marginal at 15 with GSM8K up, the target at 30 with GSM8K 0.34 down, collapsed
    second = [ev(0, 0, 0, 0), ev(15, 0.059, 0.055, 0.143), ev(30, 0.104, 0.090, -0.340)]
    o = pr.observed_class(second)
    assert o["class"] == "marginal" and o["at_iteration"] == 15 and o["class_ignoring_retained"] == "works" and o["collapsed"]
    # a clean success and a plain failure
    assert pr.observed_class([ev(0, 0, 0, 0), ev(10, 0.11, 0.1, -0.01)])["class"] == "works"
    assert pr.observed_class([ev(0, 0, 0, 0), ev(10, 0.06, 0.05, 0.0)])["class"] == "marginal"
    o = pr.observed_class([ev(0, 0, 0, 0), ev(10, 0.02, 0.01, 0.0), ev(20, -0.03, 0.0, 0.0)])
    assert o["class"] == "fails" and o["collapsed"]


def test_prediction_block_carries_the_prescription(tmp_path):
    write_prediction(tmp_path / "probe")
    p = pr.prediction_block(tmp_path / "probe")
    assert p["step_multiple"] == 15.0 and p["updates_to_target"] == 2.9 and p["G"] == 0.002
    assert p["block_digests"]["test"] == "c" and p["call"] == "works"


def test_analysis_puts_observed_beside_predicted(tmp_path):
    """Rewards on two kinds (Countdown validation and test, GSM8K off-task) rising on task
    and falling off task: the normalized changes by block and task, and the rule's
    first-order prediction at each evaluated iteration."""
    rng = np.random.default_rng(0)
    write_prediction(tmp_path / "probe")
    out = tmp_path / "run"
    out.mkdir()
    pred = pr.prediction_block(tmp_path / "probe")
    (out / "run.json").write_text(json.dumps({"config": {"setting": "qwen3b/countdown", "seed": 0, "step": 15.0, "n": 64},
                                              "prediction": pred}))
    P = 2000
    kinds = ["countdown"] * P + ["gsm8k"] * (P // 2)
    block = ["validation"] * (P // 2) + ["test"] * (P // 2) + ["offtask"] * (P // 2)
    evs, logs = [], []
    for g in (0, 5, 10):
        on = (rng.random(P) < 0.14 + 0.01 * g) * 1.1          # Countdown on the 1.1 scale
        off = (rng.random(P // 2) < 0.8 - 0.02 * g) * 10.0    # GSM8K on the 10 scale
        evs.append({"iteration": g, "rewards": on.tolist() + off.tolist(), "kinds": kinds, "block": block, "mean_len": 300.0})
    for g in range(10):
        logs.append({"iteration": g, "fitness": [0.1] * 64, "mean_fitness": 0.1, "sd_fitness": 0.01, "rollouts": 64 * 192,
                     "update_norm": 3.3 * 15, "seconds": 300.0})
    (out / "eval.jsonl").write_text("\n".join(json.dumps(e) for e in evs) + "\n")
    (out / "log.jsonl").write_text("\n".join(json.dumps(r) for r in logs) + "\n")
    a = pr.analyze(out)
    assert a["iterations_done"] == 10 and a["rollouts_total"] == 10 * 64 * 192
    last = a["eval"][-1]
    assert abs(last["by_block"]["validation"]["change"] - 0.1) < 3 * last["by_block"]["validation"]["change_se"] + 0.01
    assert abs(last["by_block"]["offtask"]["change"] + 0.2) < 3 * last["by_block"]["offtask"]["change_se"] + 0.01
    assert abs(last["by_task"]["countdown"]["change"] - 0.1) < 0.03
    # the prediction after 10 updates at step 15: 10 * 15 * 0.002 gain, 10 * 225 * 3e-5 cost
    assert abs(last["predicted"]["gain_first_order"] - 0.3) < 1e-9
    assert abs(last["predicted"]["random_cost"] - 0.0675) < 1e-9
    assert abs(last["predicted"]["offtask_change"] + 0.045) < 1e-9
    assert a["at_end"]["updates_to_target_predicted"] == 2.9
    assert a["eval"][0]["by_block"]["validation"]["change"] == 0.0 and a["eval"][0]["by_block"]["validation"]["change_se"] == 0.0
    # the summary table: one row per run directory, the class from the frozen definitions
    rows = pr.summary(tmp_path)
    assert len(rows) == 1 and rows[0]["run"] == "run" and rows[0]["call"] == "works" and rows[0]["updates"] == 10
    assert rows[0]["class"] == "fails" and rows[0]["class_ignoring_retained"] == "works" and rows[0]["collapsed"]
    t = pr.table(rows).splitlines()
    assert len(t) == 3 and t[0].startswith("| run |") and "| fails | works | yes |" in t[2]
