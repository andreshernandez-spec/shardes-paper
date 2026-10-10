#!/usr/bin/env python
"""Does the engine hold exactly the view after `es_tell` then `es_restore`, every time?

    python -m es_vllm.race_check --rounds 20      # from experiments/end_to_end, one GPU

The Tulu pilot's center evaluations, which run right after a restore, collapsed at 4 to 7
of 30 iterations per arm while every member write was fine. The suspected cause is the
JAX to torch handoff: `es_tell` queues seconds of asynchronous JAX work, the restore
hands torch its results by DLPack, and torch copies before JAX has written them (or after
JAX has freed them). This loop provokes exactly that sequence with random fitness and
checks the engine bit for bit after every restore. Nothing is generated.
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--util", type=float, default=0.4)
    args = ap.parse_args(argv)

    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from vllm import LLM  # noqa: PLC0415

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.util,
              max_model_len=2048, enable_prefix_caching=False,
              worker_extension_cls="es_vllm.worker.ESWorker")
    model_dir = snapshot_download(args.model, allow_patterns=["*.safetensors", "*.json"])
    llm.collective_rpc("es_init", args=(model_dir, args.n, 1e-3, 5e-7, 0))
    rng = np.random.default_rng(0)
    bad = 0
    for r in range(args.rounds):
        llm.collective_rpc("es_ask")
        llm.collective_rpc("es_tell", args=(rng.uniform(size=args.n).tolist(),))
        llm.collective_rpc("es_restore")
        check = llm.collective_rpc("es_check", args=(None,))[0]
        member = None
        if r % 5 == 0:
            llm.collective_rpc("es_perturb", args=(3,))
            member = llm.collective_rpc("es_check", args=(3,))[0]["ok"]
        bad += not check["ok"]
        print(f"round {r}: restore ok {check['ok']} mismatched {len(check['mismatched'])}"
              + ("" if member is None else f" | member 3 ok {member}"), flush=True)
    print(f"RESULT {bad} of {args.rounds} restores wrong", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
