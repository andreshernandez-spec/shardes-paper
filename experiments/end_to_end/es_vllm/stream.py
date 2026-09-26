"""shardes' full-rank ES, one parameter leaf at a time.

`SeedRegenerated.apply` forms a member's whole weight tree at once, and `ShardedES.tell`
accumulates a whole f32 tree. At 8B on one 80 GB GPU, beside vLLM's own weights and the
f32 master (32 GB), neither fits. This module does the same arithmetic leaf by leaf,
calling the library's own noise functions, so peak memory is one leaf:

- member weights: `view + asarray(sigma, view.dtype) * eps`, with `eps` from
  `coupling(leaf_streams(base_key, L)[k], member_id, size, dtype)`, exactly as
  `member_noise` builds each leaf and `SeedRegenerated.apply` combines them;
- update: `master - (lr / (n * sigma)) * sum_i w_i * eps_i`, the f32 sum taken over
  members in order with a `lax.scan`, as `SeedRegenerated.contract` does per leaf.

The view is the master cast to bf16 (`ShardedES._view`), recomputed per leaf rather than
stored. `tests/end_to_end/test_e2e_stream.py` holds both functions bit-identical to the
library on a small tree; that test is what lets the vLLM backend claim it trains with
shardes rather than with a copy of it.
"""

from functools import partial

import jax
import jax.numpy as jnp

from shardes.coupling import GAUSSIAN
from shardes.strategies._noise import leaf_streams

COMPUTE = jnp.bfloat16


def streams(base_key, n_leaves: int) -> list:
    return leaf_streams(base_key, n_leaves)


@partial(jax.jit, static_argnames=("coupling",))
def member_leaf(master_leaf, stream, member_id, sigma, coupling=GAUSSIAN):
    """One leaf of one member's weights, in the compute dtype."""
    view = master_leaf.astype(COMPUTE)
    eps = coupling(stream, member_id, view.size, view.dtype).reshape(view.shape)
    return view + jnp.asarray(sigma, view.dtype) * eps


@partial(jax.jit, static_argnames=("coupling",))
def contracted_leaf(master_leaf, stream, member_ids, weights, coupling=GAUSSIAN):
    """sum_i w_i * eps_i for one leaf, in f32, members in order (`SeedRegenerated.contract`)."""
    view = master_leaf.astype(COMPUTE)

    def step(acc, iw):
        i, wi = iw
        eps = coupling(stream, i, view.size, view.dtype).reshape(view.shape)
        return acc + wi * eps.astype(jnp.float32), None

    acc, _ = jax.lax.scan(step, jnp.zeros_like(view, dtype=jnp.float32),
                          (member_ids, weights.astype(jnp.float32)))
    return acc


def step_coefficient(lr: float, n: int, sigma: float):
    """`ShardedES.tell`'s `lr / (n * s)`, with `s` the state's f32 sigma. Computing it in
    Python floats instead moves the master by up to 16 f32 ulps a step (measured)."""
    return lr / (n * jnp.float32(sigma))


def updated_leaf(master_leaf, stream, member_ids, weights, lr, sigma, n):
    """One leaf of the master after `tell`: `p - (lr / (n * s)) * u`, the library's
    expression evaluated the way `tell` evaluates it, outside any jit."""
    u = contracted_leaf(master_leaf, stream, member_ids, weights)
    return master_leaf - step_coefficient(lr, n, sigma) * u


def leaf_names(master: dict) -> list:
    """Leaf order as `jax.tree.flatten` sees a dict: sorted keys. Streams follow it."""
    return [k for k, _ in sorted(master.items())]
