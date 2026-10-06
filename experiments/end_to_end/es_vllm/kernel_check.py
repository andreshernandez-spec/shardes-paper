#!/usr/bin/env python
"""Which piece of the per-leaf update gives different bits from the same inputs?

    python -m es_vllm.kernel_check --repeats 200 --out runs/update-check/x-kernels.json

On the H200 the per-leaf update intermittently differs between two evaluations from the
same inputs, in JAX alone, and none of the XLA flags tried removes it
(runs/update-check/README.md). This repeats the update's pieces on one array the size of
a Llama-3.1-8B MLP matrix (14336 x 4096) and counts, for each, how often its output
differs from its first evaluation: the noise draw of one member (`jax.random.normal`,
bf16, as `Gaussian` draws it), the contraction over 16 members (`stream.contracted_leaf`),
the eager update arithmetic (`master - coef * u`), and plain f32 arithmetic of the same
size as a control. Exact checksums (sum of the bits as uint32); nothing is fetched until
the end, so the device queue stays as busy as in a replay.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import datetime  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

PKG = Path(__file__).resolve().parent
E2E = PKG.parent
sys.path.insert(0, str(E2E))
sys.path.insert(0, str(E2E.parent))
import harness  # noqa: E402
from provenance import env_block  # noqa: E402

SHAPE = (14336, 4096)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    out = E2E / args.out
    if out.exists():
        raise SystemExit(f"{out} exists")

    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from shardes.coupling import GAUSSIAN  # noqa: PLC0415

    from es_vllm import stream  # noqa: PLC0415

    n = 16
    s = stream.streams(jax.random.key(7), 4)[2]
    master = jax.random.normal(jax.random.key(1), SHAPE, jnp.float32) * 0.02
    w = jnp.asarray(np.random.default_rng(0).standard_normal(n), jnp.float32)
    ids = jnp.arange(n, dtype=jnp.int32)
    coef = stream.step_coefficient(5e-7, n, 1e-3)

    @jax.jit
    def checksum(a):
        bits = jax.lax.bitcast_convert_type(a, jnp.uint16 if a.dtype == jnp.bfloat16
                                            else jnp.uint32)
        return jnp.sum(bits.astype(jnp.uint32), dtype=jnp.uint32)

    draw = jax.jit(lambda s, i: GAUSSIAN(s, i, master.size, jnp.bfloat16))
    control = jax.jit(lambda m: (m * 1.0001 + 0.5) * 0.999)
    u0 = stream.contracted_leaf(master, s, ids, w)
    ops = {
        "draw": lambda: draw(s, jnp.int32(3)),
        "contract": lambda: stream.contracted_leaf(master, s, ids, w),
        "update": lambda: master - coef * u0,
        "control": lambda: control(master),
    }
    sums = {k: [] for k in ops}
    t0 = time.perf_counter()
    for _ in range(args.repeats):
        for k, op in ops.items():
            sums[k].append(checksum(op()))
    sums = {k: [int(x) for x in jax.device_get(v)] for k, v in sums.items()}
    seconds = time.perf_counter() - t0
    res = {k: {"differ": sum(x != v[0] for x in v), "distinct": len(set(v))}
           for k, v in sums.items()}
    print(f"{args.repeats} repeats in {seconds:.0f}s: "
          + "; ".join(f"{k} {r['differ']} differ ({r['distinct']} distinct)"
                      for k, r in res.items()), flush=True)
    harness.write_atomic(out, {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "env": env_block(E2E, ["runs"], ("jax", "jaxlib")),
        "xla_flags": os.environ.get("XLA_FLAGS", ""), "gpu": jax.devices()[0].device_kind,
        "shape": SHAPE, "repeats": args.repeats, "seconds": seconds, "result": res,
        "checksums": sums})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
