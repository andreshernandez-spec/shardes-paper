"""es_vllm/probe_run.py: the analysis of a run's records and the prediction block, no GPU."""

import json
import pathlib
import sys

import numpy as np
import pytest

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
    (path / "env.json").write_text(json.dumps({"block_digests": {"train": "a", "validation": "b", "test": "c", "offtask": "d"},
                                               "blocks": {"train": 192, "validation": 300, "test": 300, "offtask": 300}}))


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


def test_observed_classes_select_on_validation():
    # the first run: GSM8K leaves the bound from update 5 on, so no checkpoint qualifies
    first = [ev(0, 0, 0, 0), ev(5, 0.033, 0.091, -0.067), ev(10, 0.070, 0.057, -0.123), ev(15, 0.135, 0.115, -0.277),
             ev(20, 0.060, 0.112, -0.600), ev(25, -0.005, -0.012, -0.843), ev(30, -0.118, -0.114, -0.837)]
    o = pr.observed_class(first)
    assert o["class"] == "fails" and o["selected_iteration"] is None and o["class_by_test_peak"] == "fails"
    assert o["class_ignoring_retained"] == "works" and o["best_test_change"] == 0.135 and o["collapsed"]
    # the second: the only checkpoint within the bound is 15, test +0.059 there: marginal
    second = [ev(0, 0, 0, 0), ev(15, 0.059, 0.055, 0.143), ev(30, 0.104, 0.090, -0.340)]
    o = pr.observed_class(second)
    assert o["class"] == "marginal" and o["selected_iteration"] == 15 and o["test_change_at_selected"] == 0.059
    assert o["class_by_test_peak"] == "marginal" and o["class_ignoring_retained"] == "works" and o["collapsed"]
    # selection on validation differs from the test peak: the peak at 10 is not the checkpoint
    o = pr.observed_class([ev(0, 0, 0, 0), ev(10, 0.12, 0.03, 0.0), ev(20, 0.04, 0.12, 0.0)])
    assert o["class"] == "fails" and o["selected_iteration"] == 20 and o["class_by_test_peak"] == "works"
    assert pr.observed_class([ev(0, 0, 0, 0), ev(10, 0.11, 0.1, -0.01)])["class"] == "works"
    assert pr.observed_class([ev(0, 0, 0, 0), ev(10, 0.06, 0.05, 0.0)])["class"] == "marginal"
    o = pr.observed_class([ev(0, 0, 0, 0), ev(10, 0.02, 0.01, 0.0), ev(20, -0.03, 0.0, 0.0)])
    assert o["class"] == "fails" and o["selected_iteration"] == 10 and o["collapsed"]


def test_loop_seconds_to_the_first_crossing():
    evals = [ev(0, 0, 0, 0), ev(5, 0.04, 0.03, 0.0), ev(10, 0.11, 0.09, -0.01), ev(15, 0.14, 0.12, -0.05)]
    es_log = [{"iteration": g, "seconds": 300.0, "rollouts": 12288} for g in range(15)]
    r = pr.loop_seconds_to(evals, es_log)
    assert r["iteration"] == 10 and r["loop_seconds"] == 3000.0 and r["rollouts"] == 122880 and r["retained_within_bound"]
    grpo_log = [{"iteration": g, "seconds": {"generate": 10.0, "train": 80.0}, "rollouts": 768} for g in range(15)]
    r = pr.loop_seconds_to(evals, grpo_log, threshold=0.05)
    assert r["iteration"] == 10 and r["loop_seconds"] == 900.0 and r["rollouts"] == 7680
    assert pr.loop_seconds_to(evals, es_log, threshold=0.5) is None
    t = pr.compare_table([{"run": "a-s0", "es_iteration": 5, "es_loop_seconds": 1598.0, "es_rollouts": 61440, "es_offtask_there": -0.003,
                           "es_retained": True, "grpo_arm": "a-lr5e-7-s0", "grpo_step": 15, "grpo_loop_seconds": 1376.0,
                           "grpo_rollouts": 11520, "grpo_offtask_there": -0.017, "seconds_ratio": 1.16, "rollouts_ratio": 5.33}]).splitlines()
    assert len(t) == 3 and "| 1.16 | 5.33 |" in t[2] and "| 61,440 |" in t[2] and "| yes |" in t[2]


def test_prediction_block_carries_the_prescription(tmp_path):
    write_prediction(tmp_path / "probe")
    p = pr.prediction_block(tmp_path / "probe")
    assert p["block_digests"]["test"] == "c" and p["call"] == "works" and p["blocks"]["validation"] == 300
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
        on = (rng.random(P) < 0.14 + 0.012 * g) * 1.1         # Countdown on the 1.1 scale
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
    assert abs(last["by_block"]["validation"]["change"] - 0.12) < 3 * last["by_block"]["validation"]["change_se"] + 0.01
    assert abs(last["by_block"]["offtask"]["change"] + 0.2) < 3 * last["by_block"]["offtask"]["change_se"] + 0.01
    assert abs(last["by_task"]["countdown"]["change"] - 0.12) < 0.03
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
    # with shrinkage 0.1 the prediction counts the geometrically weighted updates
    (out / "run.json").write_text(json.dumps({"config": {"setting": "qwen3b/countdown", "seed": 0, "step": 15.0, "n": 64,
                                                         "shrink": 0.1}, "prediction": pred}))
    p10 = pr.analyze(out)["eval"][-1]["predicted"]
    eff_g, eff_c = 0.9 * (1 - 0.9 ** 10) / 0.1, 0.81 * (1 - 0.9 ** 20) / (1 - 0.81)
    assert abs(p10["gain_first_order"] - eff_g * 15 * 0.002) < 1e-9 and abs(p10["random_cost"] - eff_c * 225 * 3e-5) < 1e-9
    assert pr.effective_updates(10, 0.0) == (10.0, 10.0) and abs(pr.effective_updates(1000, 0.1)[0] - 9.0) < 1e-9
    (out / "run.json").write_text(json.dumps({"config": {"setting": "qwen3b/countdown", "seed": 0, "step": 15.0, "n": 64},
                                              "prediction": pred}))
    # the curves printer: a heading per run and one row per evaluation, the prediction beside
    c = pr.curves(tmp_path).splitlines()
    assert c[0] == "**run** (step 15, 10 updates, 122,880 rollouts, 0.8 h)"
    assert c[2] == "| after | validation | test | off-task | answer length | predicted validation |"
    assert c[-1].startswith("| 10 | ") and c[-1].endswith(f"| 300 | {0.3 - 0.0675:+.3f} |")


def test_compare_pairs_each_run_with_the_reference_arm_by_config(tmp_path):
    """The GRPO reference is the arm at the recipe's learning rate and kl coefficient, seed
    0, found by its config; an ES run pairs with it through its setting, whatever the
    directory names say (S4a's arms carry a variant in theirs)."""
    P = 600
    kinds = ["countdown"] * P + ["gsm8k"] * (P // 2)
    block = ["validation"] * (P // 2) + ["test"] * (P // 2) + ["offtask"] * (P // 2)

    def evals(gain_per_step):                 # exact: the first share of each block's prompts solved
        rows = []
        for g in (0, 5, 10, 15):
            on = [1.1 if i % (P // 2) < (0.14 + gain_per_step * g) * (P // 2) else 0.0 for i in range(P)]
            off = [10.0 if i < 0.8 * (P // 2) else 0.0 for i in range(P // 2)]
            rows.append({"iteration": g, "rewards": on + off, "kinds": kinds, "block": block, "mean_len": 300.0})
        return rows

    def write(d, run, ev, log):
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps(run))
        (d / "eval.jsonl").write_text("\n".join(json.dumps(e) for e in ev) + "\n")
        (d / "log.jsonl").write_text("\n".join(json.dumps(r) for r in log) + "\n")

    es_log = [{"iteration": g, "fitness": [0.1] * 64, "mean_fitness": 0.1, "sd_fitness": 0.01, "rollouts": 64 * 192,
               "update_norm": 10.0, "seconds": 300.0} for g in range(15)]
    for name, extra in (("qwen3b-countdown-step3-s0", {"step": 3.0, "n": 64}),
                        ("qwen3b-countdown-n256-s0", {"step": 15.3, "n": 256}),
                        ("qwen3b-countdown-random-s0", {"step": 15.3, "n": 64, "ranking": "random", "shrink": None})):
        write(tmp_path / "s4a" / name, {"config": {"setting": "qwen3b/countdown", "seed": 0, **extra}}, evals(0.012), es_log)
    grpo_log = [{"iteration": g, "seconds": {"generate": 20.0, "train": 70.0}, "rollouts": 768, "mean_reward": 0.2,
                 "live_prompts": 30, "kl": [0.01], "grad_norm": [0.5], "mean_len": 300.0, "tokens": 100000} for g in range(15)]
    for name, cfg in (("qwen3b-countdown-lr5e-7-beta0-s0", {"lr": 5e-7, "beta": 0.0, "seed": 0}),
                      ("qwen3b-countdown-lr5e-7-s0", {"lr": 5e-7, "beta": 0.01, "seed": 0}),
                      ("qwen3b-countdown-lr5e-7-s1", {"lr": 5e-7, "beta": 0.01, "seed": 1}),
                      ("qwen3b-countdown-lr1e-6-s0", {"lr": 1e-6, "beta": 0.01, "seed": 0})):
        write(tmp_path / "grpo-run" / name, {"config": {"setting": "qwen3b/countdown", **cfg}}, evals(0.012), grpo_log)
    rows = pr.compare(tmp_path / "s4a", tmp_path / "grpo-run")
    assert [r["run"] for r in rows] == ["qwen3b-countdown-n256-s0", "qwen3b-countdown-random-s0", "qwen3b-countdown-step3-s0"]
    assert all(r["grpo_arm"] == "qwen3b-countdown-lr5e-7-s0" for r in rows)
    assert all(r["es_iteration"] == 10 and r["grpo_step"] == 10 for r in rows)
    assert all(abs(r["seconds_ratio"] - 3000 / 900) < 1e-9 for r in rows)
    # the summary names each run's variant
    s = {r["run"]: r["variant"] for r in pr.summary(tmp_path / "s4a")}
    assert s == {"qwen3b-countdown-step3-s0": None, "qwen3b-countdown-n256-s0": "N 256",
                 "qwen3b-countdown-random-s0": "random ranking"}
    assert "| variant |" in pr.table(pr.summary(tmp_path / "s4a")).splitlines()[0]
    # two reference arms for one setting is an error, not a silent choice
    write(tmp_path / "grpo-run" / "dup-lr5e-7-s0", {"config": {"setting": "qwen3b/countdown", "lr": 5e-7, "beta": 0.01, "seed": 0}},
          evals(0.012), grpo_log)
    with pytest.raises(SystemExit):
        pr.compare(tmp_path / "s4a", tmp_path / "grpo-run")


def fam(iteration, val, test, selection, reserved):
    """An evaluation row with per-family retained changes on the two off-task halves:
    `selection` and `reserved` map a family to (change, change_se)."""
    row = ev(iteration, val, test, 0.0)
    row["by_family_half"] = {f: {"selection": {"change": selection[f][0], "change_se": selection[f][1], "prompts": 150},
                                 "reserved": {"change": reserved[f][0], "change_se": reserved[f][1], "prompts": 150}}
                             for f in selection}
    return row


def test_retention_uses_a_fixed_margin_with_paired_intervals_per_family(tmp_path):
    ok = {"MATH": (-0.005, 0.01), "ifeval": (0.0, 0.01)}        # lower bound -0.025, above the margin
    lost_math = {"MATH": (-0.08, 0.01), "ifeval": (0.0, 0.01)}
    unresolved_math = {"MATH": (-0.03, 0.02), "ifeval": (0.0, 0.01)}
    evals = [ev(0, 0, 0, 0), fam(5, 0.06, 0.07, ok, ok), fam(10, 0.12, 0.11, unresolved_math, ok),
             fam(15, 0.15, 0.14, lost_math, lost_math)]
    r = pr.retention(evals)
    # 15 is ineligible (MATH lost on the selection half); 10 is eligible (unresolved does not
    # disqualify) and has the larger validation change
    assert r["selected_iteration"] == 10 and r["class"] == "works" and r["margin"] == 0.03
    assert r["per_evaluation"][2]["eligible"] is False and r["per_evaluation"][1]["selection"]["MATH"] == "unresolved"
    assert r["retained_at_selected"]["MATH"]["status"] == "within" and r["retained_within_margin"]
    # a gain in one family does not cancel a loss in another
    cancel = {"MATH": (-0.08, 0.01), "ifeval": (0.10, 0.01)}
    r2 = pr.retention([ev(0, 0, 0, 0), fam(5, 0.12, 0.11, cancel, cancel)])
    assert r2["class"] == "fails" and r2["selected_iteration"] is None
    # the old rows without family halves give no selection
    assert pr.retention([ev(0, 0, 0, 0), ev(5, 0.12, 0.11, 0.0)])["selected_iteration"] is None


def test_analysis_splits_the_retained_families_by_half_and_reads_a_prescription(tmp_path):
    write_prediction(tmp_path / "probe")
    (tmp_path / "probe" / "prescription.json").write_text(json.dumps({
        "method": "anchored", "step_multiple": 15.6, "shrink": 0.1, "fallback": False, "unit_norm": 3.3,
        "P1_learns": "learns", "P2_collapses": "no call", "P3_pace": None, "P4_rl_pace": None}))
    p = pr.prediction_block(tmp_path / "probe")
    assert p["step_multiple"] == 15.6 and p["shrink"] == 0.1 and p["call"] == "learns" and p["prescription"]["method"] == "anchored"
    out = tmp_path / "run"
    out.mkdir()
    (out / "run.json").write_text(json.dumps({"config": {"setting": "qwen3b/gsm8k", "seed": 0, "step": 15.6, "n": 64, "shrink": 0.1},
                                              "prediction": p}))
    P = 600                                   # the off-task block as the Task builds it: both families in each half
    kinds = ["gsm8k"] * P + ["MATH"] * 150 + ["ifeval"] * 150 + ["MATH"] * 150 + ["ifeval"] * 150
    block = ["validation"] * 300 + ["test"] * 300 + ["offtask"] * 600
    evs = []
    for g in (0, 5):
        on = [10.0 if i % 300 < 150 + 12 * g else 0.0 for i in range(P)]           # the task gains 0.04 per update
        math = [10.0 if i < 75 - 2 * g else 0.0 for i in range(150)]                # MATH loses 10 of 150 by update 5
        ife = [10.0] * 150
        evs.append({"iteration": g, "rewards": on + math + ife + math + ife, "kinds": kinds, "block": block, "mean_len": 300.0})
    (out / "eval.jsonl").write_text("\n".join(json.dumps(e) for e in evs) + "\n")
    (out / "log.jsonl").write_text("\n".join(json.dumps({"iteration": g, "fitness": [0.1] * 64, "mean_fitness": 0.1, "sd_fitness": 0.01,
                                                           "rollouts": 64 * 192, "update_norm": 10.0, "seconds": 300.0}) for g in range(5)) + "\n")
    (tmp_path / "probe" / "prescription.json").write_text(json.dumps({
        "method": "anchored", "step_multiple": 15.6, "shrink": 0.1, "fallback": False, "unit_norm": 3.3,
        "response_at_first_displacement": {"odd": 0.0126, "odd_se": 0.0037, "even": 0.0088, "even_se": 0.0049, "within_scan": True},
        "P1_learns": "learns", "P2_collapses": "no call", "P3_pace": None, "P4_rl_pace": None}))
    p = pr.prediction_block(tmp_path / "probe")
    (out / "run.json").write_text(json.dumps({"config": {"setting": "qwen3b/gsm8k", "seed": 0, "step": 15.6, "n": 64, "shrink": 0.1},
                                              "prediction": p}))
    a = pr.analyze(out)
    # the prediction of a prescribed run comes from the prescription's responses with the anchor's weights
    eff_g, eff_c = pr.effective_updates(5, 0.1)
    assert a["eval"][-1]["predicted"]["from"] == "prescription"
    assert abs(a["eval"][-1]["predicted"]["validation_change"] - (eff_g * 0.0126 + eff_c * 0.0088)) < 1e-9
    fh = a["eval"][-1]["by_family_half"]
    assert set(fh) == {"MATH", "ifeval"} and fh["MATH"]["selection"]["prompts"] == 150 and fh["MATH"]["reserved"]["prompts"] == 150
    assert abs(fh["ifeval"]["selection"]["change"]) < 1e-9 and fh["MATH"]["selection"]["change"] < 0
    assert a["retention"]["margin"] == 0.03 and a["retention"]["reserved_half_present"]
