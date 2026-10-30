#!/usr/bin/env python
"""What es-at-scale's ES did to the weights: its released MATH model against its base.

    python -m es_vllm.es_at_scale_ckpt --predict --out runs/es-at-scale-ckpt/prediction.json  # CPU, base only
    python -m es_vllm.es_at_scale_ckpt --measure --out runs/es-at-scale-ckpt/measured.json    # CPU, both
    python -m es_vllm.es_at_scale_ckpt --table runs/es-at-scale-ckpt

Hayes et al. (arXiv 2608.12679) release Qwen2.5-Math-7B after 500 iterations of es-at-scale
(32 members, sigma 1e-3, alpha 5e-4, 512 MATH prompts per iteration, 8 GPUs). Their trainer
(es-at-scale 574a9d1, `es_at_scale/utils/worker_extension.py`) does three things the paper
does not describe, and each should be visible in the weights:

- every tensor's noise comes from a generator reseeded with the member's seed, so tensors
  of one shape get the same draw: all 28 layers get the same perturbation and the same
  update, and so do the embedding and the output head (`06-countdown-check.md` noted the
  reseeding);
- members are applied and removed in place in bf16 (add the noise, then subtract it), and
  the update is summed in f32 but added in bf16, so an update below half a bf16 step at a
  weight rounds away unless the noise carries it across, and an add-then-subtract can
  leave a step behind where the noise crosses a power of two;
- the loop makes `n_iterations + 1` updates.

`--predict` reads only the base model and runs these mechanics on samples of its own
weights. It ignores the update's direction: per weight, the useful part of one update is
far below its noise, so what rounding does to the update is what it does to the noise.
Engine 0 evaluates 32 / 8 members per iteration and its weights are broadcast to the other
engines after the update, so 4 add-and-subtract round trips per iteration stay in the
weights (the paper trained it on 8 B200s, one engine each). `--measure` records the same
statistics from both checkpoints. The prediction is committed before the ES checkpoint is read.

The model card says 500 steps but not whether the release is the final model or the trainer's
best-by-eval save, which would have fewer updates. Where rounding is mild the rms change grows
as the square root of the number of updates, so the measurement shows which it is.
"""

import argparse
import datetime
import hashlib
import json
import math
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

N, SIGMA, ALPHA = 32, 1e-3, 5e-4
UPDATES = 501                        # 500 iterations, plus the loop's extra update
TRIPS = 4                            # engine 0's members per iteration: 32 members, 8 engines
EXACT_RMS = ALPHA / math.sqrt(N) * math.sqrt(UPDATES)   # per weight, with no rounding
LAYERS = 28
MATRICES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
BINADES = range(-14, 3)              # floor(log2 |w|); below and above are clipped into the ends
PREDICT_LAYERS = (0, 9, 18, 27)
PAIRS = ((13, 14), (0, 27))
PER_BINADE = 4000                    # sampled weights per binade per tensor
CORR_SAMPLE = 40000
CHUNK = 1 << 24
PACKAGES = ("numpy", "torch", "safetensors", "huggingface_hub")

# Gates, fixed with the prediction (before the ES checkpoint is read).
GATES = {
    "shared_draw": "for every matrix type, the measured correlation of the update between layers 13 "
                   "and 14 is at least 0.3 and at least half the simulated one (independent draws "
                   "per layer would give |r| < 0.01)",
    "rounding": "in every binade holding at least 1e5 weights of the layer matrices, the measured rms "
               "change is within a factor 1.5 of the simulated one and the measured share of weights "
               "changed within 0.15 of it",
    "head": "the embedding and the output head's changes correlate at least 0.3",
}


def pin(rel: R.Repo) -> dict:
    # A hub revision, not a `commit` key: the provenance audit reads those as git commits.
    return {"repo": rel.repo, "kind": rel.kind, "revision": rel.commit, "note": rel.note}


def matrix(kind: str, layer: int) -> str:
    part = "self_attn" if kind in MATRICES[:4] else "mlp"
    return f"model.layers.{layer}.{part}.{kind}.weight"


def group(name: str):
    """(group, kind, layer) of a parameter name."""
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return "head", name.split(".")[-2], None
    if name == "model.norm.weight":
        return "norm", "norm", None
    parts = name.split(".")
    layer = int(parts[2])
    kind = parts[-2]
    if kind in ("input_layernorm", "post_attention_layernorm"):
        return "norm", kind, layer
    if parts[-1] == "bias":
        return "bias", kind, layer
    return "matrix", kind, layer


class Checkpoint:
    """Tensors of a local snapshot by name, read one at a time (safetensors maps the file)."""

    def __init__(self, rel: R.Repo):
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        self.rel = rel
        self.dir = Path(snapshot_download(rel.repo, revision=rel.commit, local_files_only=True,
                                          allow_patterns=["*.json", "*.safetensors"]))
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self.where = index["weight_map"]

    def names(self):
        return sorted(self.where)

    def get(self, name: str):
        from safetensors import safe_open  # noqa: PLC0415
        # Copied out and closed: a file kept open leaves every page read mapped, and counted in RSS.
        with safe_open(str(self.dir / self.where[name]), framework="pt") as f:
            t = f.get_tensor(name).clone()
        if str(t.dtype) != "torch.bfloat16":
            raise SystemExit(f"{self.rel.repo} {name} is {t.dtype}")
        return t

    def hashes(self) -> dict:
        """sha256 of every shard, against the hub's LFS hash."""
        from huggingface_hub import HfApi  # noqa: PLC0415
        info = HfApi().model_info(self.rel.repo, revision=self.rel.commit, files_metadata=True)
        hub = {s.rfilename: s.lfs.sha256 for s in info.siblings or [] if s.lfs is not None}
        out = {}
        for f in sorted(set(self.where.values())):
            h = hashlib.sha256()
            with open(self.dir / f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            out[f] = {"sha256": h.hexdigest(), "matches_hub": h.hexdigest() == hub.get(f)}
            if not out[f]["matches_hub"]:
                raise SystemExit(f"{self.rel.repo} {f}: local hash differs from the hub's")
        return out


def binade(x):
    """floor(log2 |x|) as int, clipped into BINADES; zeros go to the lowest."""
    import torch  # noqa: PLC0415
    _, e = torch.frexp(x.float())
    return (e.to(torch.int16) - 1).clamp(BINADES[0], BINADES[-1])


def simulate(w, updates=UPDATES, trips=TRIPS, seed=0):
    """es-at-scale's mechanics on bf16 weights `w` (rows, S); the rows share every draw, as
    tensors of one shape do. Returns the change, f32."""
    import torch  # noqa: PLC0415
    g = torch.Generator().manual_seed(seed)
    p = w.clone()
    s = w.shape[-1]
    for _ in range(updates):
        for _ in range(trips):
            eps = torch.randn(s, generator=g).to(torch.bfloat16)
            p.add_(SIGMA * eps)          # perturb_self_weights: bf16 noise times sigma, added in bf16
            p.add_(-SIGMA * eps)         # restore_self_weights: the same noise subtracted
        u = torch.randn(s, generator=g) * (ALPHA / math.sqrt(N))
        p.add_(u.to(torch.bfloat16))     # update_weights_from_seeds: f32 sum cast to bf16, added
    return p.float() - w.float()


def corr(x, y) -> float:
    """Pearson correlation, summed in f64 chunks (the embedding has 545M weights)."""
    x, y = x.flatten(), y.flatten()
    s = [0.0] * 5
    for i in range(0, x.numel(), CHUNK):
        a, b = x[i:i + CHUNK].double(), y[i:i + CHUNK].double()
        s[0] += float(a.sum())
        s[1] += float(b.sum())
        s[2] += float((a * a).sum())
        s[3] += float((b * b).sum())
        s[4] += float((a * b).sum())
    n = x.numel()
    vx, vy = s[2] - s[0] ** 2 / n, s[3] - s[1] ** 2 / n
    return (s[4] - s[0] * s[1] / n) / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


def binade_stats(delta, w0) -> dict:
    """Per binade of |w0|: count, weights changed, sum of squared change, sum of |change| in
    bf16 steps of w0."""
    import torch  # noqa: PLC0415
    k = len(BINADES)
    acc = {key: torch.zeros(k, dtype=torch.float64) for key in ("count", "changed", "sumsq", "sum_steps")}
    dd, ww = delta.flatten(), w0.flatten()
    for i in range(0, dd.numel(), CHUNK):
        b = binade(ww[i:i + CHUNK]).long() - BINADES[0]
        d = dd[i:i + CHUNK].double()
        step = torch.exp2((b + BINADES[0] - 7).double())
        acc["count"] += torch.bincount(b, minlength=k).double()
        acc["changed"] += torch.bincount(b, weights=(d != 0).double(), minlength=k)
        acc["sumsq"] += torch.bincount(b, weights=d * d, minlength=k)
        acc["sum_steps"] += torch.bincount(b, weights=d.abs() / step, minlength=k)
    return {key: v.tolist() for key, v in acc.items()}


def add_stats(acc: dict, s: dict) -> None:
    for key, v in s.items():
        acc[key] = [a + b for a, b in zip(acc.get(key, [0.0] * len(v)), v)]


def summarize(acc: dict) -> list:
    rows = []
    for i, e in enumerate(BINADES):
        c = acc["count"][i]
        if c == 0:
            continue
        rows.append({"binade": e, "count": int(c), "changed": acc["changed"][i] / c,
                     "rms": math.sqrt(acc["sumsq"][i] / c), "rms_over_exact": math.sqrt(acc["sumsq"][i] / c) / EXACT_RMS,
                     "mean_steps": acc["sum_steps"][i] / c})
    return rows


def predict(args) -> int:
    import torch  # noqa: PLC0415
    torch.manual_seed(0)
    base = Checkpoint(R.QWEN25_MATH_7B)
    g = torch.Generator().manual_seed(1)

    # 1. Per binade: stratified samples of the layer matrices, every weight its own draw.
    pooled, hist = [], {}
    for kind in MATRICES:
        for layer in PREDICT_LAYERS:
            w = base.get(matrix(kind, layer)).flatten()
            b = binade(w)
            hist[f"{kind}.{layer}"] = {"count": torch.bincount(b.long() - BINADES[0], minlength=len(BINADES)).tolist(),
                                       "norm": float(w.float().norm())}
            for e in BINADES:
                idx = torch.nonzero(b == e).squeeze(1)
                if len(idx) > PER_BINADE:
                    idx = idx[torch.randperm(len(idx), generator=g)[:PER_BINADE]]
                pooled.append(w[idx])
    w = torch.cat(pooled)[None]
    by_binade = {}
    for trips in (TRIPS, 0):
        acc = {}
        add_stats(acc, binade_stats(simulate(w, args.updates, trips, seed=2), w))
        by_binade[f"trips_{trips}"] = summarize(acc)
        print(f"simulated {w.shape[1]} weights, {trips} round trips per update", flush=True)

    # Expected relative change of each sampled tensor from its binade histogram.
    rms = {r["binade"]: r["rms"] for r in by_binade[f"trips_{TRIPS}"]}
    rel = {}
    for key, h in hist.items():
        ss = sum(c * rms.get(e, 0.0) ** 2 for e, c in zip(BINADES, h["count"]))
        rel[key] = math.sqrt(ss) / h["norm"]

    # 2. Layers sharing one draw: correlation of the simulated changes.
    def pair_corr(wa, wb):
        idx = torch.randint(0, wa.numel(), (CORR_SAMPLE,), generator=g)
        rows = torch.stack([wa.flatten()[idx], wb.flatten()[idx]])
        d = simulate(rows, args.updates, seed=3)
        return corr(d[0], d[1])

    shared = {}
    for kind in MATRICES:
        shared[kind] = {f"{a}-{b}": pair_corr(base.get(matrix(kind, a)), base.get(matrix(kind, b))) for a, b in PAIRS}
    shared["embed-head"] = pair_corr(base.get("model.embed_tokens.weight"), base.get("lm_head.weight"))
    for kind in ("q_proj", "k_proj", "v_proj"):
        a, b = (base.get(f"model.layers.{layer}.self_attn.{kind}.bias") for layer in PAIRS[0])
        shared[f"{kind}.bias"] = {"13-14": pair_corr(a, b)}
    print("shared draws:", json.dumps(shared), flush=True)

    # 3. Norm weights, whole vectors: one draw for the two norms of a layer and the final norm.
    norms = torch.stack([base.get("model.layers.13.input_layernorm.weight"),
                         base.get("model.layers.13.post_attention_layernorm.weight"),
                         base.get("model.layers.14.input_layernorm.weight"),
                         base.get("model.norm.weight")])
    d = simulate(norms, args.updates, seed=4)
    norm_pred = {"changed": [float((d[i] != 0).float().mean()) for i in range(4)],
                 "rms": [float(d[i].pow(2).mean().sqrt()) for i in range(4)],
                 "abs_weight_median": [float(norms[i].float().abs().median()) for i in range(4)],
                 "corr_input_post_13": corr(d[0], d[1]), "corr_input_13_14": corr(d[0], d[2]),
                 "rows": ["layers.13.input_layernorm", "layers.13.post_attention_layernorm",
                          "layers.14.input_layernorm", "norm"]}

    out = E2E / args.out
    if out.exists():
        raise SystemExit(f"{out} exists")
    harness.write_atomic(out, {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "env": env_block(E2E, ["runs"], PACKAGES),
        "base": pin(R.QWEN25_MATH_7B), "es_at_scale": R.ES_AT_SCALE,
        "constants": {"members": N, "sigma": SIGMA, "alpha": ALPHA, "updates": args.updates, "round_trips": TRIPS,
                      "exact_rms": EXACT_RMS, "per_binade": PER_BINADE, "corr_sample": CORR_SAMPLE},
        "gates": GATES, "by_binade": by_binade, "relative_change": rel, "shared_draw": shared,
        "norms": norm_pred})
    print(f"wrote {out}", flush=True)
    return 0


def measure(args) -> int:
    base, es = Checkpoint(R.QWEN25_MATH_7B), Checkpoint(R.ES_MATH_7B)
    if base.names() != es.names():
        raise SystemExit(f"tensor names differ: {sorted(set(base.names()) ^ set(es.names()))}")
    hashes = {"base": base.hashes(), "es": es.hashes()}
    print("hashes match the hub", flush=True)

    tensors, groups, shared = {}, {}, {}
    prev, first = {}, {}
    for name in base.names():
        w0, w1 = base.get(name), es.get(name)
        if w0.shape != w1.shape:
            raise SystemExit(f"{name}: {tuple(w0.shape)} vs {tuple(w1.shape)}")
        delta = w1.float() - w0.float()
        grp, kind, layer = group(name)
        dn = float(delta.norm())
        tensors[name] = {"group": grp, "shape": list(w0.shape),
                         "changed": int(delta.count_nonzero()) / delta.numel(),
                         "rms": dn / math.sqrt(delta.numel()), "relative": dn / float(w0.float().norm())}
        add_stats(groups.setdefault(grp, {}), binade_stats(delta, w0))
        if name in ("model.embed_tokens.weight", "lm_head.weight"):
            first[name] = delta
        del w1
    shared["embed-head"] = corr(first["model.embed_tokens.weight"], first["lm_head.weight"])
    del first
    print("per tensor done", flush=True)

    # Layer to layer, one type at a time (two layers' changes in memory).
    for kind in MATRICES + ("q_proj.bias", "k_proj.bias", "v_proj.bias", "input_layernorm", "post_attention_layernorm"):
        def name_of(layer):
            if kind.endswith(".bias"):
                return f"model.layers.{layer}.self_attn.{kind[:-5]}.bias"
            if kind in MATRICES:
                return matrix(kind, layer)
            return f"model.layers.{layer}.{kind}.weight"
        r = {}
        prev = None
        d0 = None
        for layer in range(LAYERS):
            d = es.get(name_of(layer)).float() - base.get(name_of(layer)).float()
            if prev is not None:
                r[f"{layer - 1}-{layer}"] = corr(prev, d)
            if layer == 0:
                d0 = d
            prev = d
        r["0-27"] = corr(d0, prev)
        consecutive = [v for k, v in r.items() if k != "0-27"]
        shared[kind] = {"pairs": r, "min_consecutive": min(consecutive), "median_consecutive": float(np.median(consecutive))}
        print(kind, round(shared[kind]["median_consecutive"], 4), flush=True)
    d_in = es.get("model.layers.13.input_layernorm.weight").float() - base.get("model.layers.13.input_layernorm.weight").float()
    d_post = (es.get("model.layers.13.post_attention_layernorm.weight").float()
              - base.get("model.layers.13.post_attention_layernorm.weight").float())
    shared["norm_input_post_13"] = corr(d_in, d_post)

    out = E2E / args.out
    if out.exists():
        raise SystemExit(f"{out} exists")
    harness.write_atomic(out, {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "env": env_block(E2E, ["runs"], PACKAGES),
        "base": pin(R.QWEN25_MATH_7B), "es": pin(R.ES_MATH_7B), "hashes": hashes,
        "exact_rms": EXACT_RMS, "tensors": tensors,
        "by_binade": {g_: summarize(acc) for g_, acc in groups.items()}, "shared_draw": shared})
    print(f"wrote {out}", flush=True)
    return 0


def table(args) -> int:
    d = E2E / args.table
    p = json.loads((d / args.prediction).read_text())
    m = json.loads((d / "measured.json").read_text())
    print(f"prediction {args.prediction}, {p['constants']['updates']} updates simulated; exact-arithmetic rms change"
          f" per weight after {UPDATES} updates: {EXACT_RMS:.3g}\n")
    print("layer matrices by |w| binade: share changed and rms change (simulated / measured)")
    sim = {r["binade"]: r for r in p["by_binade"][f"trips_{TRIPS}"]}
    sim0 = {r["binade"]: r for r in p["by_binade"]["trips_0"]}
    ok_round = True
    for r in m["by_binade"]["matrix"]:
        s = sim.get(r["binade"])
        if s is None:
            continue
        big = r["count"] >= 1e5
        ok = abs(r["rms"] / s["rms"] - 1) <= 0.5 if s["rms"] > 0 else r["rms"] == 0
        ok = ok and abs(r["changed"] - s["changed"]) <= 0.15
        ok = ok or not big
        ok_round &= ok
        print(f"  2^{r['binade']:>3}  n={r['count']:>11,}  changed {s['changed']:.3f} / {r['changed']:.3f}"
              f"  rms {s['rms']:.2e} (no trips {sim0[r['binade']]['rms']:.2e}) / {r['rms']:.2e}"
              f"  {'' if big else '(small)'}{'' if ok else '  MISS'}")
    print("\nlayer-to-layer correlation of the change (simulated 13-14 / measured median consecutive, min, 13-14)")
    ok_shared = True
    for kind in MATRICES:
        s = p["shared_draw"][kind]["13-14"]
        mm = m["shared_draw"][kind]
        r1314 = mm["pairs"]["13-14"]
        ok = r1314 >= 0.3 and r1314 >= 0.5 * s
        ok_shared &= ok
        print(f"  {kind:10s} {s:.3f} / {mm['median_consecutive']:.3f}, {mm['min_consecutive']:.3f}, {r1314:.3f}"
              f"  (0-27: {p['shared_draw'][kind]['0-27']:.3f} / {mm['pairs']['0-27']:.3f}){'' if ok else '  MISS'}")
    eh = m["shared_draw"]["embed-head"]
    print(f"  embed-head {p['shared_draw']['embed-head']:.3f} / {eh:.3f}")
    for kind in ("q_proj.bias", "k_proj.bias", "v_proj.bias", "input_layernorm", "post_attention_layernorm"):
        print(f"  {kind:24s} measured median consecutive {m['shared_draw'][kind]['median_consecutive']:.3f}")
    print(f"\nnorms: simulated share changed {p['norms']['changed']}, measured by binade:")
    for r in m["by_binade"].get("norm", []):
        print(f"  2^{r['binade']:>3}  n={r['count']:>7,}  changed {r['changed']:.3f}  rms {r['rms']:.2e}")
    print("\nrelative change of the sampled tensors (simulated / measured)")
    for key, v in p["relative_change"].items():
        kind, layer = key.split(".")
        print(f"  {key:14s} {v:.2e} / {m['tensors'][matrix(kind, int(layer))]['relative']:.2e}")
    print(f"\ngates: shared_draw {'PASS' if ok_shared else 'FAIL'}, rounding {'PASS' if ok_round else 'FAIL'},"
          f" head {'PASS' if eh >= 0.3 else 'FAIL'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predict", action="store_true")
    mode.add_argument("--measure", action="store_true")
    mode.add_argument("--table", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--updates", type=int, default=UPDATES, help="--predict: updates to simulate")
    ap.add_argument("--prediction", default="prediction.json", help="--table: prediction file in the directory")
    args = ap.parse_args(argv)
    import torch  # noqa: PLC0415
    torch.set_num_threads(6)             # the laptop is shared
    if args.table:
        return table(args)
    if args.out is None:
        raise SystemExit("--out is required")
    return predict(args) if args.predict else measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
