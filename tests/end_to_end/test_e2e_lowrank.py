"""The low-rank members of es_vllm/lowrank.py: seeded, mirrored, and the update they imply.

The vLLM side (adapters served, updates merged) is checked on the GPU by
`lowrank_check.py`; this pins the arithmetic it relies on.
"""

import math
import pathlib
import sys

import numpy as np

E2E = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "end_to_end"
sys.path.insert(0, str(E2E))

from es_vllm import lowrank  # noqa: E402

SHAPES = [("model.layers.0.mlp.down_proj.weight", (6, 10)),
          ("model.layers.0.self_attn.q_proj.weight", (8, 6))]


def test_factors_are_seeded_and_distinct():
    a = lowrank.pair_factors(0, 3, SHAPES, 2)
    b = lowrank.pair_factors(0, 3, SHAPES, 2)
    for n in a:
        assert np.array_equal(a[n][0], b[n][0]) and np.array_equal(a[n][1], b[n][1])
    other = lowrank.pair_factors(0, 4, SHAPES, 2)
    assert not np.array_equal(a[SHAPES[0][0]][0], other[SHAPES[0][0]][0])
    assert not np.array_equal(a[SHAPES[0][0]][0][:4], a[SHAPES[1][0]][0][:4])  # leaves differ


def test_adapters_are_the_mirrored_perturbation():
    sigma, r = 5e-4, 2
    for m in (6, 7):
        t = lowrank.adapter_tensors(0, m, SHAPES, r, sigma)
        f = lowrank.pair_factors(0, m // 2, SHAPES, r)
        for name, (o, i) in SHAPES:
            mod = name[: -len(".weight")]
            lora_a = t[f"base_model.model.{mod}.lora_A.weight"].numpy()
            lora_b = t[f"base_model.model.{mod}.lora_B.weight"].numpy()
            assert lora_a.shape == (r, i) and lora_b.shape == (o, r)
            a, b = f[name]
            want = lowrank.sign(m) * sigma * a @ b.T / math.sqrt(r)
            np.testing.assert_allclose(lora_b @ lora_a, want, rtol=1e-5, atol=1e-12)  # f32


def test_coefficients_are_the_es_update():
    """sum_j k_j a_j b_j^T equals (alpha / N) sum_m z_m E_m over the members."""
    rng = np.random.default_rng(1)
    n, alpha, r = 8, 5e-4, 1
    fitness = rng.normal(size=n)
    k = lowrank.coefficients(fitness, n, alpha, r)
    z = (fitness - fitness.mean()) / fitness.std()
    for name, _ in SHAPES:
        update = sum(k[j] * np.outer(*[x[:, 0] for x in lowrank.pair_factors(0, j, SHAPES, r)[name]])
                     for j in range(n // 2))
        dense = alpha / n * sum(z[m] * lowrank.sign(m) * np.outer(
            *[x[:, 0] for x in lowrank.pair_factors(0, m // 2, SHAPES, r)[name]]) for m in range(n))
        np.testing.assert_allclose(update, dense, rtol=1e-6, atol=1e-12)


def test_coefficients_use_only_the_first_n_members():
    f = np.arange(16, dtype=float)
    assert len(lowrank.coefficients(f, 8, 1.0, 1)) == 4
    assert np.array_equal(lowrank.coefficients(f, 8, 1.0, 1), lowrank.coefficients(f[:8], 8, 1.0, 1))


def test_targets_are_the_attention_and_mlp_matrices():
    yes = ["model.layers.3.self_attn.q_proj.weight", "model.layers.31.mlp.down_proj.weight",
           "model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.gate_proj.weight"]
    no = ["model.embed_tokens.weight", "lm_head.weight", "model.layers.0.input_layernorm.weight",
          "model.layers.0.self_attn.q_proj.bias", "model.norm.weight"]
    assert all(lowrank.TARGETS.fullmatch(x) for x in yes)
    assert not any(lowrank.TARGETS.fullmatch(x) for x in no)


def test_generations_reuse_the_probe_seeds_first_then_differ():
    probe = lowrank.pair_factors(0, 2, SHAPES, 1)
    first = lowrank.pair_factors(0, 2, SHAPES, 1, lowrank.generation_of(0))
    later = lowrank.pair_factors(0, 2, SHAPES, 1, lowrank.generation_of(3))
    name = SHAPES[0][0]
    assert np.array_equal(probe[name][0], first[name][0])
    assert not np.array_equal(probe[name][0], later[name][0])
