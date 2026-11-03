#!/usr/bin/env python
"""Does replaying a pilot arm's fitness log reproduce the run, every time?

    python -m es_vllm.replay_check --arm s5e-4 --replays 4    # from experiments/end_to_end

Scoring the pilot's endpoints (runs/heldout-121-160/README.md), rebuilds of the
true-reward arms missed the log's final digest on some attempts and matched it on others,
with the same code, inputs and GPU type. This replays one arm several times in one engine
and records, per iteration, the shaping weights and, per leaf, an exact checksum of the
master (the uint32 sum of its bits) and the largest change the update made to it
(max |new - old|, exact). Two replays that part ways then show where, in which leaves,
and by how much. Each load of the master is also compared with the checkpoint. Replays
queue their updates without waiting, as `heldout.py` does; the only syncs are one per
`es_tell` (its return value) and the digest at the end. Nothing is generated.
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import datetime  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

PKG = Path(__file__).resolve().parent
E2E = PKG.parent
sys.path.insert(0, str(E2E))
sys.path.insert(0, str(E2E.parent))
import harness  # noqa: E402
import releases as R  # noqa: E402
from provenance import env_block  # noqa: E402

LEAF_STATS, SHAPING = [], []


def _instrument():
    """Record what every update did, without adding a sync."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from es_vllm import stream, worker  # noqa: PLC0415

    updated, shaping = stream.updated_leaf, worker.group_relative

    def updated_leaf(master_leaf, *args):
        new = updated(master_leaf, *args)
        LEAF_STATS.append((jnp.max(jnp.abs(new - master_leaf)),
                           jnp.sum(jax.lax.bitcast_convert_type(new, jnp.uint32),
                                   dtype=jnp.uint32)))
        return new

    def group_relative(x):
        w = shaping(x)
        SHAPING.append(w)
        return w

    stream.updated_leaf, worker.group_relative = updated_leaf, group_relative

    def leaf_hashes(self):
        out = {}
        for name in self._es_names:
            v = np.asarray(self._es_master[name].astype(jnp.bfloat16).view(jnp.uint16))
            out[name] = hashlib.sha256(v.tobytes()).hexdigest()
        return out

    def drop(self):
        self._es_master = None
        gc.collect()

    def names(self):
        return list(self._es_names)

    worker.ESWorker.rc_leaf_hashes = leaf_hashes
    worker.ESWorker.rc_drop = drop
    worker.ESWorker.rc_names = names


def checkpoint_hashes(model_dir: str) -> dict:
    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415

    ref = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as st:
            for n in st.keys():
                t = st.get_tensor(n).to(torch.bfloat16)
                ref[n] = hashlib.sha256(t.view(torch.uint16).numpy().tobytes()).hexdigest()
    return ref


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="s5e-4")
    ap.add_argument("--replays", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="the laptop smoke run's log")
    args = ap.parse_args(argv)

    import jax  # noqa: PLC0415
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from vllm import LLM  # noqa: PLC0415

    _instrument()
    name = "tulu-smoke" if args.smoke else f"tulu-pilot-{args.arm}"
    cfg = yaml.safe_load((PKG / f"{name}.yaml").read_text())
    log = [json.loads(x) for x in
           (E2E / "runs" / (name + "-smoke" * args.smoke) / "log.jsonl").read_text().splitlines()]
    rel = getattr(R, cfg["model"])
    repo, revision = ("Qwen/Qwen2.5-0.5B-Instruct", None) if args.smoke else (rel.repo, rel.commit)
    model_dir = snapshot_download(repo, revision=revision,
                                  allow_patterns=["*.safetensors", "*.json"])
    ref = checkpoint_hashes(model_dir)
    llm = LLM(model=repo, revision=revision, dtype="bfloat16", seed=0,
              gpu_memory_utilization=0.4 if args.smoke else 0.5, max_model_len=4096, enable_prefix_caching=False,
              worker_extension_cls="es_vllm.worker.ESWorker")
    rpc = llm.collective_rpc
    init = (model_dir, cfg["population"], cfg["sigma"], cfg["alpha"] * cfg["sigma"],
            cfg["seed"])

    replays = []
    for k in range(args.replays):
        rpc("rc_drop")
        rpc("es_init", args=init)
        h = rpc("rc_leaf_hashes")[0]
        load_bad = [n for n in h if h[n] != ref[n]]
        LEAF_STATS.clear()
        SHAPING.clear()
        for rec in log:
            rpc("es_ask")
            rpc("es_tell", args=(rec["fitness"],))
        digest = rpc("es_digest")[0]
        stats = jax.device_get(LEAF_STATS)
        L = len(stats) // len(log)
        rep = {
            "load_leaves_differing": load_bad,
            "digest": digest, "matches_log": digest == log[-1]["digest"],
            "shaping": [np.asarray(w).ravel().tolist() for w in jax.device_get(SHAPING)],
            "max_delta": [[float(stats[g * L + j][0]) for j in range(L)]
                          for g in range(len(log))],
            "checksum": [[int(stats[g * L + j][1]) for j in range(L)]
                         for g in range(len(log))],
        }
        replays.append(rep)
        print(f"replay {k}: load differs on {len(load_bad)} leaves, digest {digest[:12]}, "
              f"matches log {rep['matches_log']}", flush=True)

    names = rpc("rc_names")[0]
    # Compare with a replay that reached the log's digest, else with replay 0.
    good = next((r for r in replays if r["matches_log"]), replays[0])
    reference = replays.index(good)
    summary = []
    for k, rep in enumerate(replays):
        if rep is good:
            continue
        first_w = next((g for g in range(len(log))
                        if rep["shaping"][g] != good["shaping"][g]), None)
        first = next((g for g in range(len(log))
                      if rep["checksum"][g] != good["checksum"][g]), None)
        diff = [] if first is None else [
            j for j in range(len(names)) if rep["checksum"][first][j] != good["checksum"][first][j]]
        leaves = [{"leaf": names[j], "max_delta": rep["max_delta"][first][j],
                   "max_delta_good": good["max_delta"][first][j]} for j in diff]
        per_g = [sum(a != b for a, b in zip(rep["checksum"][g], good["checksum"][g]))
                 for g in range(len(log))]
        s = {"replay": k, "reference": reference, "first_shaping_difference": first_w, "first_iteration": first,
             "leaves_differing": len(diff), "leaves": leaves[:20],
             "leaves_differing_per_iteration": per_g}
        summary.append(s)
        print(f"replay {k} vs replay {reference}: shaping first differs at {first_w}; "
              f"master first differs at iteration {first} in {len(diff)} leaves "
              f"{[(x['leaf'], x['max_delta'], x['max_delta_good']) for x in leaves[:5]]}",
              flush=True)

    out = E2E / "runs" / ("replay-check" + "-smoke" * args.smoke)
    out.mkdir(parents=True, exist_ok=True)
    harness.write_atomic(out / f"{args.arm}.json", {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "env": env_block(E2E, ["runs"], ("vllm", "torch", "jax", "transformers")),
        "arm": args.arm, "log_digest": log[-1]["digest"], "leaves": names,
        "summary": summary,
        # per-leaf arrays are summarized above; keep the rest of each replay
        "replays": [{**{k: v for k, v in r.items() if k not in ("max_delta", "checksum")},
                     "max_delta_over_leaves": [max(x) for x in r["max_delta"]]}
                    for r in replays]})
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
