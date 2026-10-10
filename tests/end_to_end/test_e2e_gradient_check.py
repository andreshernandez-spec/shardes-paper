"""es_vllm/gradient_check.py: what needs no GPU, the analysis of a scan and the leaf kinds."""

import json
import pathlib
import sys

import numpy as np

E2E = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "end_to_end"
sys.path.insert(0, str(E2E))

from es_vllm import gradient_check as gc  # noqa: E402


def test_leaf_kinds():
    assert gc.leaf_kind("model.layers.3.mlp.down_proj.weight") == "down_proj"
    assert gc.leaf_kind("model.layers.0.self_attn.q_proj.weight") == "q_proj"
    assert gc.leaf_kind("model.embed_tokens.weight") == "embed_tokens"
    assert gc.leaf_kind("lm_head.weight") == "lm_head"
    assert gc.leaf_kind("model.layers.0.input_layernorm.weight") == "norm"


def test_scan_analysis_recovers_the_slope(tmp_path):
    """Reward that rises 0.2 points per unit effective length along +g and falls along -g,
    with a quadratic cost, gives D about 0.2 and a negative even part."""
    rng = np.random.default_rng(0)
    P = 4000
    start = (rng.random(P) < 0.5) * 10.0
    rows = [{"kind": "start", "lam": 0.0, "effective_length": 0.0, "rewards": start.tolist(), "mean_len": 300}]
    for lam, eff in ((1.0, 0.5), (4.0, 3.5)):
        for sgn in (1, -1):
            p = np.clip(0.5 + sgn * 0.02 * eff - 0.001 * eff ** 2, 0, 1)
            rows.append({"kind": "grad", "lam": sgn * lam, "effective_length": eff,
                         "rewards": ((rng.random(P) < p) * 10.0).tolist(), "mean_len": 300})
    (tmp_path / "scan.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    a = gc.analyze(tmp_path)["scan"]
    pts = {q["lam"]: q for q in a["points"]}
    assert abs(pts[1.0]["D_per_effective_length"] - 0.2) < 3 * pts[1.0]["D_se"]
    assert abs(pts[4.0]["D_per_effective_length"] - 0.2) < 3 * pts[4.0]["D_se"]
    assert pts[4.0]["even"] < 0
    # the reward rose along +g, so along RL's direction (-g) the slope is -0.2
    assert a["slopes"]["grad"]["at_effective_length"] == 0.5
    assert abs(a["slopes"]["grad"]["slope_along_rl_direction"] + 0.2) < 3 * a["slopes"]["grad"]["slope_se"]
    assert pts[1.0]["linear_regime"]
