"""shardes' full-rank ES, one parameter leaf at a time.

`SeedRegenerated.apply` forms a member's whole weight tree at once, and `ShardedES.tell`
accumulates a whole f32 tree. At 8B on one 80 GB GPU, beside vLLM's own weights and the
f32 master (32 GB), neither fits. This module does the same arithmetic leaf by leaf,
calling the library's own noise functions, so peak memory is one leaf:

- member weights: `view + asarray(sigma, view.dtype) * eps`, with `eps` from
  `coupling(leaf_streams(base_key, L)[k], member_id, size, dtype)`, exactly as
  `member_noise` builds each leaf and `SeedRegenerated.apply` combines them;
- update: `master - (lr / (n * sigma)) * sum_i w_i * eps_i`, the f32 sum taken over
  members in order, as `SeedRegenerated.contract`'s scan does per leaf, one step per
  call (below).

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
def _member_term(acc, stream, member_id, weight, coupling=GAUSSIAN):
    """One step of `SeedRegenerated.contract`'s scan: `acc + w_i * eps_i`, in f32."""
    eps = coupling(stream, member_id, acc.size, COMPUTE).reshape(acc.shape)
    return acc + weight * eps.astype(jnp.float32)


def contracted_leaf(master_leaf, stream, member_ids, weights, coupling=GAUSSIAN):
    """sum_i w_i * eps_i for one leaf, in f32, members in order (`SeedRegenerated.contract`).

    The library's scan, one jitted step per member from Python. On the H200 the scan as a
    device-side while loop gave different bits from the same inputs in about 0.4% of calls,
    in some processes, in JAX alone; one step per call gave none in 6,000 and the rolled
    scan's correct bits (runs/update-check/README.md).
    """
    acc = jnp.zeros(master_leaf.shape, jnp.float32)
    w = weights.astype(jnp.float32)
    for k in range(member_ids.shape[0]):
        acc = _member_term(acc, stream, member_ids[k], w[k], coupling=coupling)
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
