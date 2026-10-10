"""Diagnose why a replay of a pilot log does not reach the log's digests. Run from
experiments/end_to_end with the vllm venv. Not a result; a diagnosis."""
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, ".")
import releases as R  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "s5e-4"


def leaf_hashes(self):
    import jax.numpy as jnp
    import numpy as np
    out = {}
    for name in self._es_names:
        v = np.asarray(self._es_master[name].astype(jnp.bfloat16).view(jnp.uint16))
        out[name] = hashlib.sha256(v.tobytes()).hexdigest()
    return out


def drop(self):
    self._es_master = None
    gc.collect()


def fixed_load(self, model_dir):
    """load_master with the producer finished before torch frees the source."""
    import jax
    import jax.numpy as jnp
    from safetensors import safe_open
    master = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cuda") as st:
            for name in st.keys():
                t = st.get_tensor(name)
                m = jnp.asarray(jax.dlpack.from_dlpack(t)).astype(jnp.float32)
                m.block_until_ready()
                master[name] = m
                del t
    self._es_master = master
    torch.cuda.empty_cache()


def main():
    from huggingface_hub import snapshot_download
    from vllm import LLM
    rel = R.TULU31_START
    d = snapshot_download(rel.repo, revision=rel.commit,
                          allow_patterns=["*.safetensors", "*.json"])
    from safetensors import safe_open
    ref = {}
    for f in sorted(Path(d).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as st:
            for n in st.keys():
                t = st.get_tensor(n)
                if t.dtype != torch.bfloat16:
                    print("checkpoint dtype", n, t.dtype, flush=True)
                    t = t.to(torch.bfloat16)
                ref[n] = hashlib.sha256(t.view(torch.uint16).numpy().tobytes()).hexdigest()
    print(f"reference: {len(ref)} leaves", flush=True)

    cfg = yaml.safe_load(Path(f"es_vllm/tulu-pilot-{ARM}.yaml").read_text())
    log = [json.loads(x) for x in Path(f"runs/tulu-pilot-{ARM}/log.jsonl").read_text().splitlines()]
    llm = LLM(model=rel.repo, revision=rel.commit, dtype="bfloat16", seed=0,
              gpu_memory_utilization=0.5, max_model_len=4096, enable_prefix_caching=False,
              worker_extension_cls="es_vllm.worker.ESWorker")
    from es_vllm.worker import ESWorker
    ESWorker.diag_leaf_hashes = leaf_hashes
    ESWorker.diag_drop = drop
    ESWorker.diag_fixed_load = fixed_load
    rpc = llm.collective_rpc
    init_args = (d, cfg["population"], cfg["sigma"], cfg["alpha"] * cfg["sigma"], cfg["seed"])

    def vs_ref(tag):
        h = rpc("diag_leaf_hashes")[0]
        bad = [n for n in h if h[n] != ref[n]]
        print(f"{tag}: {len(bad)} of {len(h)} leaves differ from the checkpoint {bad[:5]}",
              flush=True)

    def replay(tag):
        first = None
        digests = []
        for rec in log:
            rpc("es_ask")
            rpc("es_tell", args=(rec["fitness"],))
            if "digest" in rec:
                dg = rpc("es_digest")[0]
                ok = dg == rec["digest"]
                digests.append(dg)
                if not ok and first is None:
                    first = rec["iteration"]
        print(f"{tag}: first mismatch at iteration {first}; digests "
              f"{[x[:8] for x in digests]} log {[r['digest'][:8] for r in log if 'digest' in r]}",
              flush=True)
        return digests

    for k in range(2):
        rpc("diag_drop")
        rpc("es_init", args=init_args)
        vs_ref(f"es_init #{k}")
        replay(f"replay #{k} (es_init)")
    for k in range(2):
        rpc("diag_drop")
        rpc("es_init", args=init_args)   # sets names, key, generation
        rpc("diag_drop")
        rpc("diag_fixed_load", args=(d,))
        vs_ref(f"fixed load #{k}")
        replay(f"replay #{k} (fixed load)")
    print("DIAG DONE", flush=True)


if __name__ == "__main__":
    main()
