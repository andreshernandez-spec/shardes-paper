"""The E18 preflight ladder all-reduces what its label says, and old records are corrected.

Until 2026-09-25 the ladder summed a (D, n/D + 1) array over its sharded axis, so every
point moved 1/D of its label and the recorded beta was D times too high. These pin the
fix and the correction applied to the records made before it.
"""

import re
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import NamedSharding, PartitionSpec

from shardes import sharding

MULTIHOST = Path(__file__).resolve().parents[1] / "experiments" / "phase2" / "multihost"
sys.path.insert(0, str(MULTIHOST))
import costmodel  # noqa: E402
import preflight  # noqa: E402


def allreduce_elements(fn, operand) -> int:
    hlo = fn.lower(operand).compile().as_text()
    sizes = [int(n) for n in re.findall(r"f32\[(\d+)\]\{0\} all-reduce\(", hlo)]
    assert len(sizes) == 1, sizes
    return sizes[0]


@pytest.mark.parametrize("size_bytes", [1024, 2**20])
def test_ladder_moves_its_label(size_bytes):
    mesh = sharding.make_mesh(len(jax.devices()))
    n_f32 = size_bytes // 4
    fn, y = preflight.ladder_program(mesh, n_f32)
    assert 4 * allreduce_elements(fn, y) == size_bytes


@pytest.mark.parametrize("size_bytes", [1024, 2**20])
def test_correction_matches_what_old_records_moved(size_bytes):
    """The operand every E18 record was measured with, and the payload costmodel assumes."""
    d = len(jax.devices())
    mesh = sharding.make_mesh(d)
    rep = NamedSharding(mesh, PartitionSpec())
    member = NamedSharding(mesh, PartitionSpec(sharding.POP))
    n_f32 = size_bytes // 4
    y = jax.device_put(jnp.ones((d, n_f32 // d + 1)), member)
    fn = jax.jit(lambda v: jax.lax.with_sharding_constraint(v.sum(axis=0), rep))
    assert 4 * allreduce_elements(fn, y) == costmodel.ladder_payload_bytes(size_bytes, d)


def test_alpha_beta_uses_the_recorded_payload_when_present():
    t = {"8": 1e-4, "104857600": 1e-4 + 0.01}
    new = {"allreduce_seconds_by_bytes": t,
           "ladder_payload_bytes": {"8": 8, "104857600": 104857600}}
    old = {"allreduce_seconds_by_bytes": t}
    _, beta_new = costmodel.ladder_alpha_beta(new, 8)
    _, beta_old = costmodel.ladder_alpha_beta(old, 8)
    assert beta_new == pytest.approx((104857600 - 8) / 0.01)
    assert beta_old == pytest.approx(beta_new / 8, rel=1e-5)
