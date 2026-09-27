"""The leaf-streamed ES in es_vllm/stream.py equals the library's, bit for bit.

This is the test that makes "the model is trained by shardes" true of the vLLM backend:
the backend never forms a member or an update any other way than these two functions,
and these two functions reproduce `SeedRegenerated.apply` and `ShardedES.tell`.
"""

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

E2E = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "end_to_end"
sys.path.insert(0, str(E2E))

from es_vllm import stream  # noqa: E402
from shardes import sharding  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.shaping import group_relative  # noqa: E402
from shardes.strategies.seed_regenerated import SeedPerturbation, SeedRegenerated  # noqa: E402

N, SIGMA, LR = 8, 1e-3, 5e-7


def small_master(seed=0):
    ks = jax.random.split(jax.random.key(seed), 4)
    return {  # the shapes of a transformer's leaves, in miniature, f32 master
        "model.embed_tokens.weight": jax.random.normal(ks[0], (64, 16)) * 0.02,
        "model.layers.0.mlp.down_proj.weight": jax.random.normal(ks[1], (16, 40)) * 0.02,
        "model.layers.0.self_attn.q_proj.weight": jax.random.normal(ks[2], (16, 16)) * 0.02,
        "model.norm.weight": jnp.ones((16,)) + jax.random.normal(ks[3], (16,)) * 1e-3,
    }


@pytest.fixture(scope="module")
def es():
    mesh = sharding.make_mesh(1)
    return ShardedES(SeedRegenerated(), n=N, sigma=SIGMA, lr=LR, mesh=mesh,
                     shaping=group_relative, compute_dtype=jnp.bfloat16)


def test_member_weights_match_the_library(es):
    state = es.init(jax.random.key(1), small_master())
    pert, state = es.ask(state)
    view = es._view(state.params)
    names = stream.leaf_names(state.params)
    ss = stream.streams(pert.base_key, len(names))
    for member in (0, 3, N - 1):
        one = SeedPerturbation(pert.base_key, jnp.array([member]), view)
        lib = es.strategy.apply(lambda p, x: p, view, one, SIGMA)(None)
        for k, name in enumerate(names):
            ours = stream.member_leaf(state.params[name], ss[k], member, SIGMA)
            assert ours.dtype == jnp.bfloat16
            np.testing.assert_array_equal(np.asarray(ours), np.asarray(lib[name][0]))


def test_update_matches_the_library(es):
    state = es.init(jax.random.key(2), small_master(3))
    pert, state = es.ask(state)
    reward = jax.random.uniform(jax.random.key(4), (N,))
    fitness = (-reward)[:, None].astype(jnp.float32)  # tell descends; (n, 1) for z-score
    lib = es.tell(state, pert, fitness).params
    weights = es.shaping(fitness)
    names = stream.leaf_names(state.params)
    ss = stream.streams(pert.base_key, len(names))
    for k, name in enumerate(names):
        ours = stream.updated_leaf(state.params[name], ss[k], pert.member_ids, weights,
                                   LR, SIGMA, N)
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(lib[name]))


def test_group_relative_on_one_column_is_qius_z_score():
    r = jnp.array([[0.1], [0.5], [0.9], [0.3]], dtype=jnp.float32)
    z = (r[:, 0] - r[:, 0].mean()) / r[:, 0].std()
    np.testing.assert_allclose(np.asarray(group_relative(r)), np.asarray(z), rtol=1e-5)
