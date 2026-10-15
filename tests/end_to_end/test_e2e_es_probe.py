"""es_vllm/probe.py: the rule and the analysis of a probe's records, without a GPU."""

import json
import pathlib
import sys

import numpy as np
import pytest

E2E = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "end_to_end"
sys.path.insert(0, str(E2E))

import releases as R  # noqa: E402
from es_vllm import probe  # noqa: E402


def test_settings_are_pinned():
    for name, (rel, template) in probe.MODELS.items():
        assert rel in R.ALL_REPOS, name
        assert len(rel.commit) == 40 and template in ("tulu", "chat")
    assert probe.setting("qwen3b-base/countdown", False)["repo"] == "Qwen/Qwen2.5-3B"


def test_rule_on_the_measured_settings():
    """Tulu 8B on Tulu (08, 09) and the 0.5B model on Countdown (06), in normalized units."""
    # Tulu at N = 1,024: gain 0.0047 and cost -0.00008 per unit on the 10 scale; the cost at
    # N = 64 is 16 times that. RL gains about 0.0105 per step (1.26 over 120 steps).
    tulu = probe.rule(G=0.00047, C=-0.000008 * 16, rl_gain_per_step=0.00105, P=192, N=64, live_share=0.6)
    assert tulu["call"] == "fails: rollouts"
    assert tulu["ratio_es_over_rl_rollouts"] > 25
    # the rollouts ES needs do not depend on N: the cost per unit scales as 1 / N
    at_1024 = probe.rule(G=0.00047, C=-0.000008, rl_gain_per_step=0.00105, P=192, N=1024, live_share=0.6)
    assert abs(at_1024["es"]["rollouts"] / tulu["es"]["rollouts"] - 1) < 1e-9
    # Countdown's gain over cost margin of about 600 against about 6 on Tulu
    cd = probe.rule(G=0.00047 * 10, C=-0.000008 * 16, rl_gain_per_step=0.00105, P=192, N=64, live_share=0.6)
    assert cd["call"] == "works"
    assert probe.rule(G=0.1, C=-0.001, rl_gain_per_step=0.001, P=192, N=64, live_share=0.05)["call"] == "fails: no signal"
    assert probe.rule(G=-0.1, C=-0.001, rl_gain_per_step=0.001, P=192, N=64, live_share=0.5)["call"] == "fails: no odd response"
    assert probe.rule(G=0.1, C=-0.001, rl_gain_per_step=0.0, P=192, N=64, live_share=0.5)["call"].startswith("undetermined")


def bernoulli(rng, p, P, scale):
    return ((rng.random(P) < np.clip(p, 0, 1)) * scale).tolist()


def test_prediction_recovers_constructed_responses(tmp_path):
    """Records with known gain, cost, sign slope and curvature; the analysis returns them
    and the rule's call follows."""
    rng = np.random.default_rng(0)
    P, scale, n, n2, prompts = 6000, 10.0, 64, 32, 192
    kinds = ["gsm8k"] * (P // 2) + ["MATH"] * (P // 4) + ["ifeval"] * (P // 4)
    block = ["validation"] * (P // 2) + ["offtask"] * (P // 2)
    start = bernoulli(rng, 0.5, P, scale)
    # level 0: sample and gradient records
    start_train = bernoulli(rng, 0.5, prompts, scale)
    (tmp_path / "sample.json").write_text(json.dumps({
        "prompts": prompts, "k": 16, "live_share": 0.6, "zero_advantage_share": 0.4, "answers_per_second": 50.0,
        "mean_sample_len": 300.0, "start_reward": {"train": float(np.mean(start_train)), "validation": 5.0, "offtask": 5.0},
        "start": {"train": {"rewards": start_train, "kinds": ["gsm8k"] * prompts}}}))
    (tmp_path / "gradient.json").write_text(json.dumps({"live_prompts": 115, "sharedness": 0.03, "diversity_yin": 0.3,
                                                        "batch_norm": 1.0}))
    # random directions: even response c L^2 with c = -0.002 per length^2 (normalized), twice
    # that on MATH, no odd part; lengths 4, 10, 25
    d_m, member_len = 4e8, 5e-4 * np.sqrt(4e8)
    scan = [{"kind": "start", "index": 0, "length": 0.0, "sign": 0, "kinds": kinds, "block": block, "rewards": start}]
    c_task = {"gsm8k": -0.002, "MATH": -0.004, "ifeval": -0.001}
    for j in range(4):
        for mult in probe.LENGTHS:
            L = member_len * mult
            for sign in (1, -1):
                p = [0.5 + c_task[t] * L ** 2 for t in kinds]
                scan.append({"kind": "random", "index": j, "length": L, "sign": sign, "kinds": kinds, "block": block,
                             "rewards": bernoulli(rng, np.asarray(p), P, scale)})
    (tmp_path / "scan.jsonl").write_text("\n".join(json.dumps(r) for r in scan) + "\n")
    # the sign direction: reward falls 0.3 per unit effective length along +sign(g), so RL's
    # slope is +0.3; effective lengths 0.1 and 0.2
    sign = [{"kind": "sign", "nominal": 0.0, "sign": 0, "effective_length": 0.0, "kinds": kinds, "block": block, "rewards": start}]
    for nominal, eff in ((0.12, 0.1), (0.24, 0.2)):
        for s in (1, -1):
            sign.append({"kind": "sign", "nominal": nominal, "sign": s, "effective_length": eff, "kinds": kinds, "block": block,
                         "rewards": bernoulli(rng, 0.5 - s * 0.3 * eff, P, scale)})
    (tmp_path / "sign.jsonl").write_text("\n".join(json.dumps(r) for r in sign) + "\n")
    # level 1: members with a true effect (sd 0.06 normalized; the Bernoulli noise of a
    # 192-prompt mean is 0.036) and the update's responses: G = 0.01 per unit, C = -0.0005
    # per unit squared, unit norm 5
    effect = rng.normal(0, 0.06, n)
    members = [bernoulli(rng, 0.5 + effect[m], prompts, scale) for m in range(n)]
    members2 = [bernoulli(rng, 0.5 + 2 * effect[m], prompts, scale) for m in range(n2)]
    (tmp_path / "members.json").write_text(json.dumps({"n": n, "n2": n2, "sigma": 5e-4, "sigma2": 1e-3, "rewards": members,
                                                       "rewards2": members2, "kinds": ["gsm8k"] * prompts}))
    unit, G, C = 5.0, 0.01, -0.0005
    steps = [{"length": 0.0, "sign": 0, "lam": 0.0, "kinds": kinds, "block": block, "rewards": start}]
    for mult in probe.LENGTHS:
        L = member_len * mult
        lam = L / unit
        for s in (1, -1):
            steps.append({"length": L, "sign": s, "lam": s * lam, "kinds": kinds, "block": block,
                          "rewards": bernoulli(rng, 0.5 + s * G * lam + C * lam ** 2, P, scale)})
    (tmp_path / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in steps) + "\n")
    (tmp_path / "steps-meta.json").write_text(json.dumps({"unit_norm": unit, "member_length": member_len,
                                                          "perturbed_params": d_m, "n": n}))

    pred = probe.predict(tmp_path)
    l0, u, r = pred["level0"], pred["update"], pred["rule"]
    # curvature by task from the random scan, and the task damaged first
    assert abs(l0["curvature_per_length2"] - np.mean([c_task[t] for t in kinds])) < 0.0005
    assert l0["first_task_damaged_random"] == "MATH"
    # the sign slope and RL's first-step gain: lr sqrt(d) times the slope
    assert abs(l0["rl_slope_along_direction"] - 0.3) < 3 * l0["sign_direction_responses"]["0.12"]["slope_se"]
    assert abs(l0["rl_first_step_gain"] - 5e-7 * np.sqrt(d_m) * l0["rl_slope_along_direction"]) < 1e-12
    # the update's gain and cost per unit, and the longest usable length was used
    assert abs(u["G"] - G) < 3 * u["G_se"] + 1e-9
    assert abs(u["C"] - C) < 3 * u["C_se"] + 1e-9
    assert u["length_used"] == f"{member_len * 2.5:.6g}"
    assert u["first_task_damaged"] in ("gsm8k", "MATH", "ifeval")
    # level 1: the ranking is reliable (0.0036 against 0.0013 of noise) and the members'
    # effects are shared across prompts (0.0036 of a per-prompt variance of 0.25)
    assert abs(pred["level1"]["sigma"]["reliability_at_prompts"] - 0.73) < 0.15
    assert abs(pred["level1"]["sigma"]["cross_prompt_correlation"] - 0.014) < 0.007
    assert pred["level1"]["sigma2"]["n"] == n2
    # the rule: ES's rollouts for the target against RL's, from these numbers
    es_rollouts = 0.1 ** 2 * -u["C"] * n * prompts / (0.02 * u["G"] ** 2)
    assert abs(r["es"]["rollouts"] - es_rollouts) < 1e-6 * es_rollouts
    assert abs(r["rl"]["rollouts"] - 0.1 / l0["rl_first_step_gain"] * 768) < 1e-6 * r["rl"]["rollouts"]
    assert r["call"] in ("works", "marginal", "fails: rollouts")
    assert r["call"] == ("works" if r["ratio_es_over_rl_rollouts"] <= 5 else
                         "marginal" if r["ratio_es_over_rl_rollouts"] <= 25 else "fails: rollouts")
    # the bootstrap over prompts brackets the point estimate and mostly agrees with the call
    b = r["bootstrap_over_prompts"]
    assert b["ratio_90"][0] < r["ratio_es_over_rl_rollouts"] < b["ratio_90"][1]
    assert b["call_shares"][r["call"]] > 0.5 and abs(sum(b["call_shares"].values()) - 1) < 1e-9
    # every member scoring alike leaves no update to scan: the record says so and the
    # rule fails the setting on the odd response (or on signal, if the live share is low)
    (tmp_path / "steps.jsonl").unlink()
    (tmp_path / "steps-meta.json").write_text(json.dumps({"unit_norm": 0.0, "member_length": member_len,
                                                          "perturbed_params": d_m, "n": n, "no_ranking_signal": True}))
    pred = probe.predict(tmp_path)
    assert pred["update"] is None and pred["rule"]["call"] == "fails: no odd response"
    assert pred["level0"]["curvature_per_length2"] == l0["curvature_per_length2"]


def test_rows_are_normalized_by_their_own_kind():
    """A Countdown setting's off-task block is on the 10 scale: a full score on either
    kind is 1 after normalization, and a mixed block's mean is the mean of the two."""
    kinds = ["countdown"] * 2 + ["gsm8k"] * 2
    assert probe.normalized([1.1, 0.1, 10.0, 0.0], kinds).tolist() == [1.0, 0.1 / 1.1, 1.0, 0.0]
    assert probe.scale_of(["MATH", "ifeval"]).tolist() == [10.0, 10.0]


def test_gain_is_the_validation_blocks_and_cost_pools_both(tmp_path):
    """An update that helps the task and hurts the off-task block: G is the task's, the
    off-task change at the target is reported and flagged, C pools both blocks."""
    rng = np.random.default_rng(2)
    P, scale, n, prompts = 4000, 10.0, 64, 192
    kinds = ["gsm8k"] * (P // 2) + ["MATH"] * (P // 2)
    block = ["validation"] * (P // 2) + ["offtask"] * (P // 2)
    start = bernoulli(rng, 0.5, P, scale)
    start_train = bernoulli(rng, 0.5, prompts, scale)
    (tmp_path / "sample.json").write_text(json.dumps({
        "prompts": prompts, "k": 16, "live_share": 0.6, "zero_advantage_share": 0.4, "answers_per_second": 50.0,
        "mean_sample_len": 300.0, "start": {"train": {"rewards": start_train, "kinds": ["gsm8k"] * prompts}}}))
    members = [bernoulli(rng, 0.5 + e, prompts, scale) for e in rng.normal(0, 0.06, n)]
    (tmp_path / "members.json").write_text(json.dumps({"n": n, "n2": 0, "sigma": 5e-4, "sigma2": 1e-3, "rewards": members,
                                                       "rewards2": [], "kinds": ["gsm8k"] * prompts}))
    unit, G_val, G_off, C = 5.0, 0.02, -0.03, -0.02
    steps = [{"length": 0.0, "sign": 0, "lam": 0.0, "kinds": kinds, "block": block, "rewards": start}]
    for L in (5.0, 10.0):
        lam = L / unit
        for s in (1, -1):
            p = [0.5 + s * (G_val if b == "validation" else G_off) * lam + C * lam ** 2 for b in block]
            steps.append({"length": L, "sign": s, "lam": s * lam, "kinds": kinds, "block": block,
                          "rewards": bernoulli(rng, np.asarray(p), P, scale)})
    (tmp_path / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in steps) + "\n")
    (tmp_path / "steps-meta.json").write_text(json.dumps({"unit_norm": unit, "member_length": 5.0, "perturbed_params": 4e8, "n": n}))
    pred = probe.predict(tmp_path)
    u, r = pred["update"], pred["rule"]
    assert abs(u["G"] - G_val) < 3 * u["G_se"] + 1e-9
    assert abs(u["G_offtask"] - G_off) < 0.01
    assert abs(u["G_pooled"] - (G_val + G_off) / 2) < 0.01
    assert abs(u["C"] - C) < 3 * u["C_se"] + 1e-9
    # at the target gain of 0.1 the off-task block moves 0.1 * G_off / G = -0.15: flagged
    assert abs(u["offtask_change_at_target"] + 0.15) < 0.05 and r["offtask_loss_at_target"]
    assert r["call"] == "undetermined: RL's gain unresolved"       # no sign scan in this record
    assert r["bootstrap_over_prompts"]["call_shares"]["undetermined: RL's gain unresolved"] > 0.9


def test_rule_net_targets_the_net_gain_and_bounds_each_retained_block():
    """The 3B instruct Countdown record's numbers: the net target needs more updates than
    the gross one, and a retained block's random loss can bind the step."""
    blocks = {"validation": {"G": 0.00227, "G_se": 0.0005, "C": -4.7e-5, "C_se": 1e-5},
              "offtask": {"G": -0.00025, "G_se": 0.0004, "C": -1.2e-5, "C_se": 1e-5}}
    r = probe.rule_net(blocks, rl_gain_per_step=0.0061, P=192, N=64, live_share=1.0, lam_max=20.0)
    es = r["es"]
    net = es["step_multiple"] * 0.00227 + es["step_multiple"] ** 2 * -4.7e-5
    assert abs(es["updates"] * net - 0.1) < 1e-6                     # the target is net
    assert es["feasible"] and es["retained_block_random_loss"]["offtask"] >= -0.02
    assert es["updates"] > 2.9                                         # more than the gross prescription
    assert r["call"] in ("works", "marginal")
    # a retained block whose random loss exceeds the bound at every step: infeasible, named
    bad = {**blocks, "offtask": {"G": 0.0, "G_se": 0.0004, "C": -0.05, "C_se": 0.001}}
    r2 = probe.rule_net(bad, 0.0061, 192, 64, 1.0, 20.0)
    assert not r2["es"]["feasible"] and r2["call"] == "fails: retained-task loss"
    # unresolved responses leave the call undetermined instead of deciding it
    weak = {**blocks, "validation": {"G": 0.0005, "G_se": 0.0005, "C": -4.7e-5, "C_se": 1e-5}}
    assert probe.rule_net(weak, 0.0061, 192, 64, 1.0, 20.0)["call"] == "undetermined: odd response unresolved"
    flat = {**blocks, "validation": {"G": 0.00227, "G_se": 0.0005, "C": -1e-5, "C_se": 1e-5}}
    assert probe.rule_net(flat, 0.0061, 192, 64, 1.0, 20.0)["call"] == "undetermined: cost unresolved"
    neg = {**blocks, "validation": {"G": -0.002, "G_se": 0.0005, "C": -4.7e-5, "C_se": 1e-5}}
    assert probe.rule_net(neg, 0.0061, 192, 64, 1.0, 20.0)["call"] == "fails: no odd response"


def test_scan_stops_when_the_cost_is_resolved_or_the_band_is_left():
    assert not probe.resolved({})
    noise = {"1": {"usable": True, "even": -0.001, "even_se": 0.01}}
    assert not probe.resolved(noise)
    assert probe.resolved({**noise, "2": {"usable": True, "even": -0.04, "even_se": 0.01}})
    assert probe.resolved({**noise, "2": {"usable": False, "even": -0.3, "even_se": 0.02}})
    assert not probe.resolved({**noise, "0.5": {"usable": True, "even": -0.05, "even_se": 0.01}})   # the longest decides


def test_unusable_lengths_fall_back_to_the_shortest():
    """When every length moved the reward past the usable band, the shortest is used and
    flagged."""
    rng = np.random.default_rng(1)
    P, scale = 2000, 10.0
    kinds, block = ["countdown"] * P, ["validation"] * P
    start = bernoulli(rng, 0.5, P, scale)
    steps = [{"length": 0.0, "sign": 0, "lam": 0.0, "kinds": kinds, "block": block, "rewards": start}]
    for L in (1.0, 2.0):
        for s in (1, -1):
            steps.append({"length": L, "sign": s, "lam": s * L, "kinds": kinds, "block": block,
                          "rewards": bernoulli(rng, 0.1, P, scale)})
    resp = probe.responses([p for p in steps if p["sign"] != 0], np.asarray(start) / scale, 1.0)
    assert not any(v["usable"] for v in resp.values())
    used = [L for L, v in resp.items() if v["usable"]] or list(resp)
    assert max(used, key=float) == "2"


def stub_data(monkeypatch, tmp_path):
    """Synthetic prompt pools in place of the data modules: Countdown's eval split and
    the Tulu stream's held-out rows, in the stream's order."""
    import types

    def heldout_rows(first, last):
        return [{"dataset": ("gsm8k", "MATH", "ifeval")[i % 3], "messages": [{"role": "user", "content": f"q{i}"}],
                 "ground_truth": str(i)} for i in range(first * 48, last * 48)]

    tulu = types.ModuleType("es_vllm.run_tulu")
    tulu.heldout_rows = heldout_rows
    tulu.load_prompts = lambda d, a, b: ([{"dataset": "tulu", "messages": [{"role": "user", "content": f"t{i}"}]} for i in range(b)], None)
    cd = types.ModuleType("es_vllm.run_countdown")
    cd.es_at_scale = lambda p: p
    cd.rows = lambda clone, split: [{"context": f"{split}{i}", "target": i} for i in range(2000 if split == "eval" else 200)]
    monkeypatch.setitem(sys.modules, "es_vllm.run_tulu", tulu)
    monkeypatch.setitem(sys.modules, "es_vllm.run_countdown", cd)


@pytest.mark.parametrize("task", ["countdown", "gsm8k", "tulu"])
def test_larger_blocks_keep_the_default_block_first_and_the_test_block_fixed(task, monkeypatch, tmp_path):
    stub_data(monkeypatch, tmp_path)
    base = probe.Task(task, "chat", False, tmp_path)
    big = probe.Task(task, "chat", False, tmp_path, block=600)
    assert {b: len(v) for b, v in base.blocks.items()} == {"train": 192, "validation": 300, "test": 300,
                                                          "offtask": 0 if task == "tulu" else 300}
    assert len(big.blocks["validation"]) == 600 and len(big.blocks["test"]) == 300 and len(big.blocks["train"]) == 192
    assert len(big.blocks["offtask"]) == (0 if task == "tulu" else 600)
    # the default blocks are the first rows of the larger ones; train and test are the same rows
    for b in ("validation", "offtask"):
        assert big.blocks[b][:300] == base.blocks[b]
    assert big.digests()["test"] == base.digests()["test"] and big.digests()["train"] == base.digests()["train"]
    assert big.prefix_digests() == {b: base.digests()[b] for b in ("validation", "offtask") if base.blocks[b]}
    assert base.prefix_digests() == {}
    # the added rows are new: none of them is in any default block
    seen = {probe.digest([r]) for rows in base.blocks.values() for r in rows}
    assert not any(probe.digest([r]) in seen for b in ("validation", "offtask") for r in big.blocks[b][300:])
    if task == "gsm8k":                                   # the off-task mix is kept, half and half
        kinds = [r["dataset"] for r in big.blocks["offtask"]]
        assert kinds.count("MATH") == 300 and kinds.count("ifeval") == 300
    with pytest.raises(SystemExit):
        probe.Task(task, "chat", False, tmp_path, block=100)
    smoke = probe.Task(task, "chat", True, tmp_path)
    assert len(smoke.blocks["train"]) == 16 and len(smoke.blocks["test"]) == 24


def test_members_offtask_diagnostic_recovers_constructed_effects(tmp_path):
    """Mirrored pairs whose directions help the train prompts, help validation less and
    hurt the off-task block, plus a common even loss off-task: the diagnostic recovers the
    signs, the ratio of the first-order changes and the even parts."""
    rng = np.random.default_rng(3)
    n, P, prompts, scale, sigma = 64, 600, 192, 10.0, 5e-4
    t = rng.normal(0, 0.1, n // 2)                       # each pair's slope on the train prompts
    v, o = 0.8 * t + rng.normal(0, 0.02, n // 2), -0.5 * t + rng.normal(0, 0.02, n // 2)
    members = []
    for m in range(n):
        s = 1 if m % 2 == 0 else -1
        members.append(bernoulli(rng, 0.5 + s * t[m // 2], prompts, scale))
    start_train = bernoulli(rng, 0.5, prompts, scale)
    mem = {"n": n, "n2": 0, "sigma": sigma, "sigma2": 1e-3, "seed": 0, "rank": 1, "rewards": members, "rewards2": [],
           "kinds": ["gsm8k"] * prompts}
    kinds = ["gsm8k"] * P + ["MATH"] * P
    block = ["validation"] * P + ["offtask"] * P
    rewards = []
    for m in range(n):
        s = 1 if m % 2 == 0 else -1
        p = [0.5 + s * v[m // 2]] * P + [0.5 + s * o[m // 2] - 0.03] * P
        rewards.append(bernoulli(rng, np.asarray(p), 2 * P, scale))
    off = {"n": n, "n2": 0, "sigma": sigma, "sigma2": 1e-3, "seed": 0, "rank": 1, "kinds": kinds, "block": block,
           "start": bernoulli(rng, 0.5, 2 * P, scale), "start_mean_len": {"validation": 300.0, "offtask": 400.0},
           "rewards": rewards, "rewards2": [], "mean_len": [{"validation": 310.0, "offtask": 420.0}] * n, "mean_len2": []}
    d = probe.level1_offtask(mem, off, np.asarray(start_train) / scale, draws=200)
    val, oft = d["by_block"]["validation"], d["by_block"]["offtask"]
    assert d["pairs"] == 32 and d["prompts"] == {"validation": P, "offtask": P}
    assert val["correlation_with_train_odd"] > 0.6 and oft["correlation_with_train_odd"] < -0.4
    assert val["correlation_90_pairs"][0] > 0.3 and oft["correlation_90_pairs"][1] < -0.1
    assert val["G_from_members"] > 3 * val["G_from_members_se"] and d["validation_resolved"]
    assert oft["G_from_members"] < -3 * oft["G_from_members_se"]
    # the first-order changes along the update scale as the constructed slopes: -0.5 / 0.8
    assert abs(d["offtask_change_per_unit_gain"] + 0.625) < 0.2
    assert abs(d["offtask_change_at_target"] - 0.1 * d["offtask_change_per_unit_gain"]) < 1e-12
    assert abs(oft["even_mean"] + 0.03) < 3 * oft["even_se"] + 0.005 and abs(val["even_mean"]) < 3 * val["even_se"] + 0.005
    assert d["validation_offtask_odd_correlation"] < -0.4
    assert 0 < val["top4_pairs_share_of_G"] < 1
    # the spread of the pairs' effects with the sampling part removed: the constructed
    # slopes' own spread, on each block
    assert abs(d["train_odd_true_sd"] - t.std(ddof=1)) < 0.02 and d["train_odd_noise_sd"] < d["train_odd_sd"]
    assert abs(val["odd_per_pair_true_sd"] - v.std(ddof=1)) < 0.02 and abs(oft["odd_per_pair_true_sd"] - o.std(ddof=1)) < 0.02
    # the summary, with and without the record
    pred = {"setting": "x", "level0": {k: None for k in ("live_share", "gradient_sharedness", "curvature_per_length2",
                                                         "rl_first_step_gain", "first_task_damaged_random")},
            "level1": {}, "update": None, "rule": {"call": "c", "ratio_es_over_rl_rollouts": None, "es": None, "rl": None},
            "level1_offtask": d}
    (tmp_path / "prediction.json").write_text(json.dumps(pred))
    a = probe.analyze(tmp_path)
    assert a["members_offtask"]["offtask_change_at_target"] == d["offtask_change_at_target"]
    assert a["members_offtask"]["G_from_members"]["offtask"][0] == oft["G_from_members"]
    pred["level1_offtask"] = None
    (tmp_path / "prediction.json").write_text(json.dumps(pred))
    assert probe.analyze(tmp_path)["members_offtask"] is None


def test_rule_curve_reads_the_measured_responses():
    """A saturating odd response (0.05 (1 - exp(-lam/5))) with a quadratic cost: the curve
    rule picks the step with the fewest updates to the net target from the responses
    themselves, bounds the retained block from its own even response, and leaves the call
    undetermined when the responses are noisy."""
    def resp(odd_se, even_se):
        out = {}
        for lam in (2.0, 5.0, 12.0, 30.0):
            odd, even, even_off = 0.05 * (1 - np.exp(-lam / 5)), -2e-4 * lam ** 2, -1.2e-4 * lam ** 2
            out[f"{lam * 3.3:.6g}"] = {"lam": lam, "usable_by_block": True, "by_block": {
                "validation": {"odd": odd, "even": even, "G": odd / lam, "G_se": odd_se / lam,
                               "C": even / lam ** 2, "C_se": even_se / lam ** 2},
                "offtask": {"odd": 0.0, "even": even_off, "G": 0.0, "G_se": odd_se / lam,
                            "C": even_off / lam ** 2, "C_se": even_se / lam ** 2}}}
        return out
    r = probe.rule_curve(resp(0.002, 0.002), rl_gain_per_step=0.0061, P=192, N=64)
    es = r["es"]
    assert 4.0 < es["step_multiple"] < 8.0                           # net 0.0266 at 5, negative by 30
    assert abs(es["net_per_update"] - 0.0266) < 0.002
    assert abs(es["updates"] * es["net_per_update"] - 0.1) < 1e-6    # the target is net
    assert es["feasible"] and es["retained_block_random_loss"]["offtask"] >= -0.02
    assert r["call"] == "works" and abs(r["rl"]["steps"] - 0.1 / 0.0061) < 1e-9 and r["lengths_used"] == [2.0, 5.0, 12.0, 30.0]
    # noisy responses: the net gain is not positive at its two-standard-error worst anywhere
    noisy = probe.rule_curve(resp(0.02, 0.02), 0.0061, 192, 64)
    assert noisy["call"] == "undetermined: net gain unresolved" and noisy["es"]["net_pessimistic"] <= 0
    # a retained block that loses more than the bound at every step: infeasible, named
    bad = resp(0.002, 0.002)
    for v in bad.values():
        v["by_block"]["offtask"]["even"] = -0.05 * v["lam"]
    assert probe.rule_curve(bad, 0.0061, 192, 64)["call"] == "fails: retained-task loss"
    assert probe.rule_curve({}, 0.0061, 192, 64)["call"].startswith("undetermined: no length")
