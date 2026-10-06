#!/usr/bin/env python
"""Is the per-leaf ES update deterministic? Computes it twice from the same inputs.

    python -m es_vllm.update_check --model qwen0.5b --out runs/update-check/x.json
    python -m es_vllm.update_check --model qwen0.5b --engine --out ...   # inside vLLM
    python -m es_vllm.update_check --compare runs/update-check           # summarize

Replays of the Tulu pilot's fitness logs did not reproduce its weights
(runs/heldout-121-160/README.md): from identical inputs the update sometimes gave
different bits. This probe runs the update on synthetic fitness and, for every leaf at
every iteration, evaluates `stream.contracted_leaf` twice on the same master leaf and
compares the two sums element by element (how many differ, the largest difference, the
largest value), then makes the production call (`stream.updated_leaf`) and checks it
against the master minus the coefficient times the first sum. The production result is
what the next iteration starts from, and its exact checksum (uint32 sum of the bits) is
recorded per leaf, so separate processes can be compared too.

Two ways to hold the master: in JAX alone, loaded from the safetensors on the CPU
(torch never touches the GPU), or `--engine`, inside a vLLM worker as `run_tulu.py`
holds it, with the engine resident. XLA_FLAGS is recorded. Nothing is generated.
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import datetime  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
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

MODELS = {  # name -> (repo, revision); revisions as the runs pinned them
    "qwen0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", None),
    "tulu8b": ("allenai/Llama-3.1-Tulu-3-8B-DPO", "a7beb67e33ffd01cc87ac3b46cadc1000985b8db"),
}
RECORDS = []


def _probe_methods():
    """Methods that run inside whatever holds the ES state (an ESWorker or a stand-in)."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from es_vllm import stream  # noqa: PLC0415
    from es_vllm.worker import group_relative  # noqa: PLC0415

    @jax.jit
    def diff(a, b):
        ne = jax.lax.bitcast_convert_type(a, jnp.uint32) != jax.lax.bitcast_convert_type(b, jnp.uint32)
        return (jnp.sum(ne, dtype=jnp.int32), jnp.max(jnp.where(ne, jnp.abs(a - b), 0.0)),
                jnp.max(jnp.abs(a)))

    @jax.jit
    def checksum(a):
        return jnp.sum(jax.lax.bitcast_convert_type(a, jnp.uint32), dtype=jnp.uint32)

    def uc_step(self, rewards):
        reward = jnp.asarray(rewards, dtype=jnp.float32)
        weights = group_relative((-reward)[:, None])  # as es_tell
        ids = jnp.arange(self._es_n, dtype=jnp.int32)
        coef = stream.step_coefficient(self._es_lr, self._es_n, self._es_sigma)
        rec = []
        for k, name in enumerate(self._es_names):
            m, s = self._es_master[name], self._es_streams[k]
            u1 = stream.contracted_leaf(m, s, ids, weights)
            u2 = stream.contracted_leaf(m, s, ids, weights)
            new = stream.updated_leaf(m, s, ids, weights, self._es_lr, self._es_sigma,
                                      self._es_n)
            rec.append((diff(u1, u2), diff(m - coef * u1, new), checksum(new)))
            self._es_master[name] = new
        RECORDS.append(rec)

    def uc_take(self):
        out = jax.device_get(RECORDS)
        RECORDS.clear()
        return out

    def uc_reset(self):
        self._es_master = None
        import gc  # noqa: PLC0415
        gc.collect()

    def uc_names(self):
        return list(self._es_names)

    return {"uc_step": uc_step, "uc_take": uc_take, "uc_reset": uc_reset,
            "uc_names": uc_names}


def cpu_master(model_dir: str) -> dict:
    """The checkpoint as f32 JAX arrays, read on the CPU: torch never uses the GPU."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415

    master = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as st:
            for name in st.keys():
                t = st.get_tensor(name)
                if t.dtype != torch.bfloat16:
                    raise SystemExit(f"{name} is {t.dtype}, expected bfloat16")
                bits = jnp.asarray(t.view(torch.int16).numpy())
                master[name] = jax.lax.bitcast_convert_type(bits, jnp.bfloat16).astype(jnp.float32)
    return master


def run(args) -> dict:
    import jax  # noqa: PLC0415
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    from es_vllm import stream  # noqa: PLC0415
    from es_vllm.worker import ESWorker  # noqa: PLC0415

    repo, revision = MODELS[args.model]
    model_dir = snapshot_download(repo, revision=revision,
                                  allow_patterns=["*.safetensors", "*.json"])
    lr = args.alpha * args.sigma
    for name, fn in _probe_methods().items():
        setattr(ESWorker, name, fn)

    if args.engine:
        from vllm import LLM  # noqa: PLC0415
        llm = LLM(model=repo, revision=revision, dtype="bfloat16", seed=0,
                  gpu_memory_utilization=args.util, max_model_len=4096,
                  enable_prefix_caching=False, worker_extension_cls="es_vllm.worker.ESWorker")

        def call(method, *a):
            return llm.collective_rpc(method, args=a)[0]

        def init():
            call("uc_reset")
            call("es_init", model_dir, args.n, args.sigma, lr, args.seed)
    else:
        state = ESWorker.__new__(ESWorker)

        def call(method, *a):
            return getattr(state, method)(*a)

        def init():
            call("uc_reset")
            state._es_master = cpu_master(model_dir)
            state._es_names = stream.leaf_names(state._es_master)
            state._es_n, state._es_sigma, state._es_lr = args.n, float(args.sigma), float(lr)
            state._es_key = jax.random.key(args.seed)
            state._es_generation = 0
            state._es_base_key = None

    replays = []
    for r in range(args.replays):
        init()
        rng = np.random.default_rng(args.fitness_seed)
        t0 = time.perf_counter()
        for _ in range(args.iterations):
            call("es_ask")
            call("uc_step", rng.uniform(size=args.n).tolist())
        recs = call("uc_take")
        digest = call("es_digest")
        names = call("uc_names")
        sums, eager, checks = [], [], []
        for g, rec in enumerate(recs):
            for j, ((n1, d1, a1), (n2, d2, _), _c) in enumerate(rec):
                if n1:
                    sums.append({"iteration": g, "leaf": names[j], "elements": int(n1),
                                 "max_diff": float(d1), "max_abs": float(a1)})
                if n2:
                    eager.append({"iteration": g, "leaf": names[j], "elements": int(n2),
                                  "max_diff": float(d2)})
            checks.append([int(c) for (_, _, c) in rec])
        rep = {"digest": digest, "seconds": time.perf_counter() - t0,
               "sum_differs": sums, "update_differs": eager, "checksums": checks}
        replays.append(rep)
        print(f"replay {r}: {len(sums)} leaf sums differ between two evaluations, "
              f"{len(eager)} updates differ from master - coef * sum; digest {digest[:12]}; "
              f"{rep['seconds']:.0f}s", flush=True)
    return {"gpu": jax.devices()[0].device_kind, "names": names, "replays": replays}


def write_record(out: Path, rec: dict) -> None:
    """The record as JSON, where the provenance audit reads it, and its per-leaf checksum
    arrays (most of its size) beside it in `<name>.checksums.json.gz`."""
    rec = json.loads(json.dumps(rec))
    arrays = {}
    if "checksums" in rec:  # kernel_check.py
        arrays["checksums"] = rec.pop("checksums")
    for i, rep in enumerate(rec.get("replays", [])):
        if "checksums" in rep:
            arrays[f"replay{i}"] = rep.pop("checksums")
    harness.write_atomic(out, rec)
    if arrays:
        side = out.with_name(out.stem + ".checksums.json.gz")
        side.write_bytes(gzip.compress(json.dumps(arrays).encode()))


def split(d: Path) -> None:
    """Re-encode whole gzipped records as write_record stores them. Lossless."""
    for p in sorted(d.glob("*.json.gz")):
        if p.name.endswith(".checksums.json.gz"):
            continue
        write_record(p.with_suffix(""), json.loads(gzip.decompress(p.read_bytes())))
        p.unlink()


def compare(d: Path) -> None:
    """Summaries of this probe's records and of kernel_check.py's, grouped by commit (the
    contraction changed at 154f63b), GPU, mode and XLA flags."""
    recs = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    kernels = {}
    for r in recs:
        if "config" not in r:  # kernel_check.py
            key = (r["env"]["commit"][:7], r["gpu"], r["xla_flags"])
            g = kernels.setdefault(key, {"processes": 0, "repeats": 0, "differ": {}})
            g["processes"] += 1
            g["repeats"] += r["repeats"]
            for op, v in r["result"].items():
                g["differ"][op] = g["differ"].get(op, 0) + v["differ"]
    for key, g in kernels.items():
        print(f"kernels {key[0]} {key[1]} flags='{key[2]}': {g['processes']} processes x "
              f"{g['repeats'] // g['processes']} repeats, differing: {g['differ']}")
    groups = {}
    for r in recs:
        if "config" not in r:
            continue
        c = r["config"]
        key = (c["model"], c["engine"], r["xla_flags"], c["n"], c["sigma"], c["alpha"],
               c["iterations"], c["seed"], c["fitness_seed"], r["gpu"], r["env"]["commit"][:7])
        groups.setdefault(key, []).append(r)
    for key, rs in groups.items():
        reps = [rep for r in rs for rep in r["replays"]]
        digests = sorted({rep["digest"][:12] for rep in reps})
        n_sum = sum(len(rep["sum_differs"]) for rep in reps)
        n_upd = sum(len(rep["update_differs"]) for rep in reps)
        worst = max((x["max_diff"] / max(x["max_abs"], 1e-30)
                     for rep in reps for x in rep["sum_differs"]), default=0.0)
        bad = sum(any(rep["sum_differs"] or rep["update_differs"] for rep in r["replays"])
                  for r in rs)
        print(f"{key[-1]} {key[0]} engine={key[1]} flags='{key[2]}' {key[-2]}: {len(rs)} "
              f"processes ({bad} with differences), "
              f"{len(reps)} replays, {n_sum} differing sums, {n_upd} differing updates, "
              f"largest relative difference {worst:.3g}, digests {digests}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="qwen0.5b")
    ap.add_argument("--engine", action="store_true", help="hold the master in a vLLM worker")
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--replays", type=int, default=2)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fitness-seed", type=int, default=0)
    ap.add_argument("--util", type=float, default=0.4)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", type=Path)
    ap.add_argument("--split", type=Path, help="re-encode gzipped records (one-off)")
    args = ap.parse_args(argv)
    if args.compare:
        compare(E2E / args.compare)
        return 0
    if args.split:
        split(E2E / args.split)
        return 0
    if args.out is None:
        ap.error("--out is required")
    out = E2E / args.out
    if out.exists():
        raise SystemExit(f"{out} exists")
    res = run(args)
    write_record(out, {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "env": env_block(E2E, ["runs"], ("vllm", "torch", "jax", "jaxlib")),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        **res})
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
