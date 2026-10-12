"""es_vllm/probe.py: the rule and the analysis of a probe's records, without a GPU."""

import json
import pathlib
import sys

import numpy as np

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
        "start": {"train": {"rewards": start_train}}}))
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
    (tmp_path / "members.json").write_text(json.dumps({"n": n, "n2": n2, "sigma": 5e-4, "sigma2": 1e-3,
                                                       "rewards": members, "rewards2": members2}))
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

    pred = probe.predict(tmp_path, scale)
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
    # every member scoring alike leaves no update to scan: the record says so and the
    # rule fails the setting on the odd response (or on signal, if the live share is low)
    (tmp_path / "steps.jsonl").unlink()
    (tmp_path / "steps-meta.json").write_text(json.dumps({"unit_norm": 0.0, "member_length": member_len,
                                                          "perturbed_params": d_m, "n": n, "no_ranking_signal": True}))
    pred = probe.predict(tmp_path, scale)
    assert pred["update"] is None and pred["rule"]["call"] == "fails: no odd response"
    assert pred["level0"]["curvature_per_length2"] == l0["curvature_per_length2"]


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
