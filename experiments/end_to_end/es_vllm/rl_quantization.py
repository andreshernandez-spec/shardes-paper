#!/usr/bin/env python
"""Is the RL run's checkpoint-to-checkpoint displacement a bf16 rounding pattern?

    python -m es_vllm.rl_quantization --out runs/rl-geometry/quantization.json   # CPU, ~750 MB

The released checkpoints are bf16. The RL run's typical change per weight after 120 steps
(about 1e-5) is below one bf16 step at a typical weight (about 6e-5), so the difference
between two checkpoints may be mostly weights whose update crossed a rounding boundary,
each moved by one step. If so, the share of weights "unchanged", the concentration of
the change and the effective rank (`rl_geometry.py`) partly describe the rounding grid and
the weights' magnitudes, not the update. This reads five tensors of the start and of the
RL branches step_120 and step_480 straight from the hub with range requests and measures
each change in units of the start weight's own bf16 step.
"""

import argparse
import datetime
import json
import struct
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

TENSORS = ("model.layers.0.self_attn.q_proj.weight", "model.layers.15.self_attn.q_proj.weight",
           "model.layers.31.self_attn.q_proj.weight", "model.layers.15.self_attn.v_proj.weight",
           "model.layers.15.mlp.down_proj.weight")


def read_tensor(fs, repo, rev, name) -> np.ndarray:
    """One bf16 tensor as float32, by byte range from its safetensors shard."""
    base = f"{repo}@{rev}"
    index = json.loads(fs.read_text(f"{base}/model.safetensors.index.json"))
    path = f"{base}/{index['weight_map'][name]}"
    with fs.open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        meta = header[name]
        if meta["dtype"] != "BF16":
            raise SystemExit(f"{name} is {meta['dtype']}")
        lo, hi = meta["data_offsets"]
        f.seek(8 + n + lo)
        raw = f.read(hi - lo)
    bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32).reshape(meta["shape"])


def bf16_step(x: np.ndarray) -> np.ndarray:
    """The spacing of bf16 numbers at |x| (normal range): 2^(exponent - 7)."""
    e = np.floor(np.log2(np.maximum(np.abs(x), 1e-30)))
    return np.exp2(e - 7)


def participation_ratio(m: np.ndarray) -> float:
    s2 = np.linalg.svd(m.astype(np.float64), compute_uv=False) ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    out = E2E / args.out
    if out.exists():
        raise SystemExit(f"{out} exists")
    from huggingface_hub import HfFileSystem  # noqa: PLC0415

    from es_vllm.heldout import rl_revision  # noqa: PLC0415

    fs = HfFileSystem()
    revs = {"120": rl_revision("step_120"), "480": rl_revision("step_480")}
    res = {}
    for name in TENSORS:
        w0 = read_tensor(fs, R.TULU31_START.repo, R.TULU31_START.commit, name)
        step = bf16_step(w0)
        r: dict = {"shape": list(w0.shape)}
        for t, rev in revs.items():
            d = read_tensor(fs, R.TULU31_RL.repo, rev, name) - w0
            ch = d != 0
            ulps = np.abs(d[ch]) / step[ch]
            q = np.abs(w0)
            edges = np.quantile(q, [0, 0.25, 0.5, 0.75, 1.0])
            r[t] = {"changed": float(ch.mean()),
                    "steps_1": float((np.round(ulps) == 1).mean()),
                    "steps_2": float((np.round(ulps) == 2).mean()),
                    "steps_3_plus": float((np.round(ulps) >= 3).mean()),
                    "median_steps": float(np.median(ulps)) if ulps.size else None,
                    "changed_by_weight_quartile": [float(ch[(q >= a) & (q <= b)].mean())
                                                   for a, b in zip(edges[:-1], edges[1:])],
                    "corr_abs_change_abs_weight": float(np.corrcoef(np.abs(d[ch]), q[ch])[0, 1])
                    if ch.sum() > 2 else None,
                    "participation_ratio": participation_ratio(d),
                    # the same rounding of a structureless update: one step, random sign,
                    # on a random set of weights of the same size
                    "participation_ratio_one_step_random": participation_ratio(
                        np.where(np.random.default_rng(0).random(w0.shape) < ch.mean(),
                                 step * np.sign(np.random.default_rng(1).standard_normal(w0.shape)),
                                 0.0))}
        res[name] = r
        print(name, json.dumps({t: {k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in r[t].items()} for t in revs}), flush=True)
    harness.write_atomic(out, {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "env": env_block(E2E, ["runs"], ("numpy", "huggingface_hub")),
        "start": R.TULU31_START.commit, "rl_revisions": revs, "tensors": res})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
