"""es_vllm/block_check.py and the per-type pieces of es_vllm/lowrank.py, without a GPU."""

import json
import pathlib
import sys

import numpy as np

E2E = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "end_to_end"
sys.path.insert(0, str(E2E))

from es_vllm import block_check as bc  # noqa: E402
from es_vllm import lowrank  # noqa: E402


def test_kind_of_a_leaf():
    assert lowrank.kind("model.layers.3.self_attn.v_proj.weight") == "v_proj"
    assert lowrank.kind("model.layers.31.mlp.down_proj.weight") == "down_proj"
    assert lowrank.kind("model.embed_tokens.weight") is None
    assert set(lowrank.KINDS) == set(bc.GRADIENT_SHARES)


def test_analysis_recovers_gains_and_costs_by_type(tmp_path):
    """Two types with known slopes and curvatures: the per-type G and C, their sums against
    the full update, and the reweighted best net come out as constructed."""
    rng = np.random.default_rng(0)
    P = 20000
    start = (rng.random(P) < 0.5) * 10.0
    norms = {"full": 2.0, "v_proj": 1.0, "down_proj": np.sqrt(3.0)}
    for k in lowrank.KINDS:
        norms.setdefault(k, 0.0)
    params = {"full": 100, "v_proj": 25, "down_proj": 75}
    for k in lowrank.KINDS:
        params.setdefault(k, 0)
    # slopes small enough that each type's shorter length stays within a point of the start
    slope = {"v_proj": 0.02, "down_proj": 0.005, "full": None}
    curv = {"v_proj": -0.001, "down_proj": -0.0001, "full": None}   # v's longer point breaks
    # the full update: gain sum a_t |u_t|, cost sum b_t |u_t|^2, at its own length
    G = sum(slope[t] * norms[t] for t in ("v_proj", "down_proj"))
    C = sum(curv[t] * norms[t] ** 2 for t in ("v_proj", "down_proj"))
    slope["full"], curv["full"] = G / norms["full"], C / norms["full"] ** 2
    lengths = {"full": [20.0, 40.0], "v_proj": [20.0, 40.0], "down_proj": [20.0, 40.0]}
    for k in lowrank.KINDS:
        lengths.setdefault(k, [])
    (tmp_path / "norms.json").write_text(json.dumps({"n": 8, "norms": norms, "params": params, "lengths": lengths}))
    rows = [{"kind": "full", "length": 0.0, "sign": 0, "lam": 0.0, "rewards": start.tolist(), "mean_len": 300}]
    for t in ("full", "v_proj", "down_proj"):
        for length in lengths[t]:
            for sign in (1, -1):
                p = np.clip(0.5 + (sign * slope[t] * length + curv[t] * length ** 2) / 10, 0, 1)
                rows.append({"kind": t, "length": length, "sign": sign, "lam": sign * length / norms[t],
                             "rewards": ((rng.random(P) < p) * 10.0).tolist(), "mean_len": 300})
    (tmp_path / "points.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    a = bc.analyze(tmp_path)
    for t in ("v_proj", "down_proj"):
        want_g, want_c = slope[t] * norms[t], curv[t] * norms[t] ** 2
        assert abs(a["by_type"][t]["G"] - want_g) < 3 * a["by_type"][t]["G_se"] + 1e-9
        assert abs(a["by_type"][t]["C"] - want_c) < 3 * a["by_type"][t]["C_se"] + 1e-9
    assert abs(a["sums"]["G"] - a["sums"]["G_full"]) < 3 * (a["sums"]["G_full_se"] + 0.01)
    # by construction: uniform (G^2 / 4C) against sum_t G_t^2 / 4 C_t
    uniform = G ** 2 / (4 * -C)
    rew = sum((slope[t] * norms[t]) ** 2 / (4 * -curv[t] * norms[t] ** 2) for t in ("v_proj", "down_proj"))
    assert abs(a["best_net"]["ratio"] - rew / uniform) < 0.5 * rew / uniform
    assert a["by_type"]["v_proj"]["length_used"] == 20.0      # 40 moved the reward by over 2
    assert a["by_type"]["down_proj"]["length_used"] == 40.0
    # the multipliers: each type's G / C against the full update's, within a quarter
    full_ratio = G / -C
    for t in ("v_proj", "down_proj"):
        want = (slope[t] * norms[t] / (-curv[t] * norms[t] ** 2)) / full_ratio
        assert abs(a["multipliers"][t] - want) < 0.25 * want
