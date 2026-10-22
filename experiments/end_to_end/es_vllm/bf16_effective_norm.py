#!/usr/bin/env python
"""How much of a small perturbation survives bf16 rounding? The direction probe's, measured.

    python -m es_vllm.bf16_effective_norm --out runs/rl-geometry/effective-norm.json   # CPU

The engine holds bf16 weights. A perturbation is applied as round(theta_0 + delta), and
an element of delta below half the bf16 step at theta_0 rounds away. The direction probe
(`direction_check.py`) compared RL's direction with random ones at equal nominal norm;
this measures, on five tensors of the start (read by range request, as in
`rl_quantization.py`), what fraction of each perturbation's norm the rounding keeps:
random Gaussian directions at the probe's per-element scales (norm / sqrt(d), d the
model's parameter count) and at the ES members' sigma, and RL's step-120 displacement at
the probe's lambdas.
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
E2E = PKG.parent
sys.path.insert(0, str(E2E))
sys.path.insert(0, str(E2E.parent))
import harness  # noqa: E402
import releases as R  # noqa: E402
from provenance import env_block  # noqa: E402

D = 8030326784
RL_NORM_120 = 0.47563  # |theta_RL120 - theta_0|, runs/direction-check/start.json
RANDOM_NORMS = (0.2378, 0.4756, 0.9513, 5e-4 * np.sqrt(D))
LAMBDAS = (0.25, 0.5, 1.0, 2.0)


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 to the nearest bf16 (ties to even), returned as float32."""
    b = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    out = E2E / args.out
    if out.exists():
        raise SystemExit(f"{out} exists")
    from huggingface_hub import HfFileSystem  # noqa: PLC0415

    from es_vllm.heldout import rl_revision  # noqa: PLC0415
    from es_vllm.rl_quantization import TENSORS, read_tensor  # noqa: PLC0415

    fs = HfFileSystem()
    rev120 = rl_revision("step_120")
    rng = np.random.default_rng(0)
    kept = {f"random {n:.4g}": [0.0, 0.0] for n in RANDOM_NORMS}
    kept |= {f"rl120 lambda {lam}": [0.0, 0.0] for lam in LAMBDAS}
    for name in TENSORS:
        w0 = read_tensor(fs, R.TULU31_START.repo, R.TULU31_START.commit, name)
        dl = read_tensor(fs, R.TULU31_RL.repo, rev120, name) - w0
        eps = rng.standard_normal(w0.shape).astype(np.float32)
        for n in RANDOM_NORMS:
            p = (n / np.sqrt(D)) * eps
            eff = to_bf16(w0 + p) - w0
            kept[f"random {n:.4g}"][0] += float((eff.astype(np.float64) ** 2).sum())
            kept[f"random {n:.4g}"][1] += float((p.astype(np.float64) ** 2).sum())
        for lam in LAMBDAS:
            p = lam * dl
            eff = to_bf16(w0 + p) - w0
            kept[f"rl120 lambda {lam}"][0] += float((eff.astype(np.float64) ** 2).sum())
            kept[f"rl120 lambda {lam}"][1] += float((p.astype(np.float64) ** 2).sum())
        print(name, flush=True)
    res = {k: {"kept_fraction_of_norm": float(np.sqrt(a / b)) if b else None} for k, (a, b) in kept.items()}
    for k, v in res.items():
        nominal = float(k.split()[-1]) if k.startswith("random") else float(k.split()[-1]) * RL_NORM_120
        v["nominal_norm"] = nominal
        v["effective_norm"] = nominal * v["kept_fraction_of_norm"]
    harness.write_atomic(out, {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "env": env_block(E2E, ["runs"], ("numpy", "huggingface_hub")),
        "tensors": list(TENSORS), "start": R.TULU31_START.commit, "rl_step_120": rev120,
        "perturbations": res})
    for k, v in res.items():
        print(f"{k}: nominal {v['nominal_norm']:.3f}, kept {v['kept_fraction_of_norm']:.3f}, "
              f"effective {v['effective_norm']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
