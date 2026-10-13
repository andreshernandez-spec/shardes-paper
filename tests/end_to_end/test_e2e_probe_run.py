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
