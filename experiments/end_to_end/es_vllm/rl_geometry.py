#!/usr/bin/env python
"""How far, and how, does the Tulu 3.1 RL run move the weights? Against ES's displacement.

    python -m es_vllm.rl_geometry --out runs/rl-geometry/geometry.json   # CPU, streams leaves

ES's displacement is known exactly: after T z-scored updates it is a sum of independent
Gaussian vectors, of norm alpha * sqrt(T d / N) (61 after 30 updates of the longer arm,
123 after 120), isotropic, dense, of full rank in every matrix. This measures the RL run's
from the released checkpoints (the DPO start and branches step_120, step_240, step_480, at
the pinned revisions), leaf by leaf on the CPU in f32: its norm, by module type; the share
of weights it leaves unchanged in bf16; how concentrated it is (share of the squared norm
in the largest 1% of changes; a Gaussian puts 9.3% there); how consistent its direction is
across steps; and, for a few matrices, its effective rank (participation ratio of the
singular values: (sum s^2)^2 / sum s^4), against the same for a Gaussian matrix.
"""

import argparse
import datetime
import json
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

STEPS = (120, 240, 480)
RANK_LEAVES = ("model.layers.0.self_attn.q_proj.weight", "model.layers.15.self_attn.q_proj.weight",
               "model.layers.31.self_attn.q_proj.weight", "model.layers.15.mlp.down_proj.weight",
               "model.layers.15.self_attn.v_proj.weight")


def kind(name: str) -> str:
    for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
              "embed_tokens", "lm_head", "norm"):
        if k in name:
            return k
    return "other"


def participation_ratio(m) -> float:
    import torch  # noqa: PLC0415
    s = torch.linalg.svdvals(m.double())
    s2 = s ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def tensors(model_dir: Path):
    """name -> (file, safe_open) lazily, over the checkpoint's shards."""
    from safetensors import safe_open  # noqa: PLC0415
    index = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as st:
            for n in st.keys():
                index[n] = f
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    out = E2E / args.out
    if out.exists():
        raise SystemExit(f"{out} exists")

    import torch  # noqa: PLC0415
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415

    from es_vllm.heldout import rl_revision  # noqa: PLC0415

    get = lambda repo, rev: Path(snapshot_download(  # noqa: E731
        repo, revision=rev, allow_patterns=["*.safetensors", "*.json"]))
    start = get(R.TULU31_START.repo, R.TULU31_START.commit)
    rl = {t: get(R.TULU31_RL.repo, rl_revision(f"step_{t}")) for t in STEPS}
    idx0 = tensors(start)
    idx = {t: tensors(d) for t, d in rl.items()}

    acc = {"theta0": 0.0, "d": 0}
    per = {t: {"norm2": 0.0, "unchanged": 0, "top1_share_num": 0.0, "by_kind": {}} for t in STEPS}
    dots = {"120.240": 0.0, "120.480": 0.0, "240.480": 0.0, "inc240.120": 0.0, "inc240": 0.0}
    ranks = {}
    rng = torch.Generator().manual_seed(0)
    for name in sorted(idx0):
        with safe_open(str(idx0[name]), framework="pt", device="cpu") as st:
            w0 = st.get_tensor(name).float()
        acc["theta0"] += float((w0 ** 2).sum())
        acc["d"] += w0.numel()
        deltas = {}
        for t in STEPS:
            with safe_open(str(idx[t][name]), framework="pt", device="cpu") as st:
                wt = st.get_tensor(name).float()
            if wt.shape != w0.shape:  # the RL run padded nothing here so far; refuse if it did
                raise SystemExit(f"{name}: {tuple(wt.shape)} vs {tuple(w0.shape)}")
            dt = wt - w0
            deltas[t] = dt
            n2 = float((dt ** 2).sum())
            p = per[t]
            p["norm2"] += n2
            p["unchanged"] += int((dt == 0).sum())
            flat = dt.abs().flatten()
            k = max(1, flat.numel() // 100)
            p["top1_share_num"] += float((torch.topk(flat, k).values ** 2).sum())
            b = p["by_kind"].setdefault(kind(name), {"norm2": 0.0, "theta0": 0.0, "n": 0})
            b["norm2"] += n2
            b["theta0"] += float((w0 ** 2).sum())
            b["n"] += w0.numel()
        dots["120.240"] += float((deltas[120] * deltas[240]).sum())
        dots["120.480"] += float((deltas[120] * deltas[480]).sum())
        dots["240.480"] += float((deltas[240] * deltas[480]).sum())
        inc = deltas[240] - deltas[120]
        dots["inc240.120"] += float((inc * deltas[120]).sum())
        dots["inc240"] += float((inc ** 2).sum())
        if name in RANK_LEAVES:
            g = torch.randn(w0.shape, generator=rng)
            ranks[name] = {"shape": list(w0.shape), "gaussian": participation_ratio(g),
                           **{str(t): participation_ratio(deltas[t]) for t in STEPS}}
            print(f"{name}: participation ratio {ranks[name]}", flush=True)
    n = {t: np.sqrt(per[t]["norm2"]) for t in STEPS}
    res = {
        "d": acc["d"], "theta0_norm": float(np.sqrt(acc["theta0"])),
        "rl": {str(t): {"norm": n[t], "relative": n[t] / np.sqrt(acc["theta0"]),
                        "unchanged_share": per[t]["unchanged"] / acc["d"],
                        "top1_share": per[t]["top1_share_num"] / per[t]["norm2"],
                        "by_kind": {k: {"norm": float(np.sqrt(v["norm2"])),
                                        "relative": float(np.sqrt(v["norm2"] / v["theta0"])),
                                        "share_of_norm2": v["norm2"] / per[t]["norm2"],
                                        "share_of_params": v["n"] / acc["d"]}
                                    for k, v in sorted(per[t]["by_kind"].items())}}
               for t in STEPS},
        "cosines": {"120,240": dots["120.240"] / (n[120] * n[240]),
                    "120,480": dots["120.480"] / (n[120] * n[480]),
                    "240,480": dots["240.480"] / (n[240] * n[480]),
                    "increment 120->240 vs 0->120": dots["inc240.120"] / (np.sqrt(dots["inc240"]) * n[120])},
        "participation_ratio": ranks,
        "gaussian_top1_share": 0.093,
        "es": {"alpha": 5e-4, "n": 16, "norm_after": {str(T): 5e-4 * float(np.sqrt(T * acc["d"] / 16))
                                                      for T in (30, 60, 120)}},
    }
    harness.write_atomic(out, {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "env": env_block(E2E, ["runs"], ("torch", "safetensors")),
        "start": {"repo": R.TULU31_START.repo, "revision": R.TULU31_START.commit},
        "rl_revisions": {str(t): rl_revision(f"step_{t}") for t in STEPS}, **res})
    print(json.dumps({k: v for k, v in res.items() if k != "participation_ratio"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
