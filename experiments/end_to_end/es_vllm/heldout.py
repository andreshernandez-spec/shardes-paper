#!/usr/bin/env python
"""Score checkpoints on held-out prompts with the pilot's measurement.

    python -m es_vllm.heldout --config es_vllm/heldout-121-160.yaml --model rl-step120
    python -m es_vllm.heldout --config ... --model es-s5e-4 --smoke     # laptop wiring

One model per process: vLLM's engine runs in-process here (JAX shares it for the ES
rebuilds), and a second engine in the same process is not something to rely on.

Greedy decoding, the `tulu` template without bos, cap 2,048, the run's verifiers in their
pinned environment: the pilot's `center_reward`, on a fixed set of prompts none of the
models has trained on. Two kinds of model:

- `hf`: a released checkpoint at a pinned revision (the DPO start, RL branches);
- `es`: a pilot arm's final ES weights, rebuilt from its fitness log by replaying every
  `es_tell` from the start weights, checked bit for bit in the engine and against the
  log's last digest before anything is decoded.

One JSON per model, written once.
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import datetime  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

PKG = Path(__file__).resolve().parent
E2E = PKG.parent
REPO = E2E.parent.parent
sys.path.insert(0, str(E2E))
sys.path.insert(0, str(E2E.parent))
import harness  # noqa: E402
import releases as R  # noqa: E402
from es_vllm.run_tulu import Verifier, render  # noqa: E402
from provenance import env_block  # noqa: E402
from tulu31_data import row_hash  # noqa: E402


def heldout_rows(first: int, last: int) -> list:
    """The rows of stream steps [first, last), checked against the recorded hashes."""
    import pyarrow.parquet as pq  # noqa: PLC0415
    from huggingface_hub import HfFileSystem  # noqa: PLC0415

    data = E2E / "data" / "tulu31"
    tset = json.loads((data / "training-set.json").read_text())
    stream = json.loads((data / "prompt-stream.json").read_text())
    f = (f"datasets/{R.TULU31_DATA.repo}@{R.TULU31_DATA.commit}"
         "/data/train-00000-of-00001.parquet")
    rows = pq.read_table(f, filesystem=HfFileSystem()).to_pylist()
    out = []
    for s in range(first, last):
        for pos in stream["stream"][s]:
            row = rows[tset["kept_indices"][pos]]
            if row_hash(row) != tset["row_sha256"][pos]:
                raise SystemExit(f"row {tset['kept_indices'][pos]} does not match the record")
            out.append(row)
    return out


def rl_revision(branch: str) -> str:
    pins = sorted((E2E / "pins").glob("releases-*.json"))[-1]
    rec = json.loads(pins.read_text())["rl"][R.TULU31_RL.repo]["branches"]
    return rec[branch]["revision"]


SMOKE = ("Qwen/Qwen2.5-0.5B-Instruct", None)


def build(model: dict, smoke: bool = False):
    """An engine holding the model's weights, and a description of where they came from."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from vllm import LLM  # noqa: PLC0415

    if model["kind"] == "hf":
        rel = getattr(R, model["release"])
        repo, revision = (SMOKE if smoke else
                          (rel.repo, rl_revision(model["branch"]) if "branch" in model
                           else rel.commit))
        llm = LLM(model=repo, revision=revision, dtype="bfloat16", seed=0,
                  gpu_memory_utilization=0.5, max_model_len=4096,
                  enable_prefix_caching=False)
        return llm, {"repo": repo, "revision": revision, "branch": model.get("branch")}

    run = E2E / model["run"]
    cfg = yaml.safe_load((E2E / model["config"]).read_text())
    log = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    rel = getattr(R, cfg["model"])
    repo, revision = SMOKE if smoke else (rel.repo, rel.commit)
    llm = LLM(model=repo, revision=revision, dtype="bfloat16", seed=0,
              gpu_memory_utilization=0.4 if smoke else 0.5, max_model_len=4096,
              enable_prefix_caching=False, worker_extension_cls="es_vllm.worker.ESWorker")
    model_dir = snapshot_download(repo, revision=revision,
                                  allow_patterns=["*.safetensors", "*.json"])
    llm.collective_rpc("es_init", args=(model_dir, cfg["population"], cfg["sigma"],
                                        cfg["alpha"] * cfg["sigma"], cfg["seed"]))
    for rec in log:
        llm.collective_rpc("es_ask")
        llm.collective_rpc("es_tell", args=(rec["fitness"],))
    llm.collective_rpc("es_restore")
    check = llm.collective_rpc("es_check", args=(None,))[0]
    digest = llm.collective_rpc("es_digest")[0]
    if not check["ok"] or digest != log[-1]["digest"]:
        raise SystemExit(f"rebuild of {run.name} failed: check {check['ok']}, "
                         f"digest {digest[:12]} vs log {log[-1]['digest'][:12]}")
    return llm, {"run": model["run"], "iterations": len(log), "digest": digest,
                 "check": check["ok"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--model", required=True, help="one model name from the config")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load((E2E / args.config).read_text())
    if args.smoke:
        cfg["stream_steps"], cfg["max_tokens"] = [120, 121], 64
        cfg["models"] = [{**m, "run": "runs/tulu-smoke-smoke",
                          "config": "es_vllm/tulu-smoke.yaml"} if m["kind"] == "es" else m
                         for m in cfg["models"]]
    out = E2E / "runs" / (args.config.stem + ("-smoke" if args.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import SamplingParams  # noqa: PLC0415

    rows = heldout_rows(*cfg["stream_steps"])
    if args.smoke:
        rows = rows[:8]
    tok = AutoTokenizer.from_pretrained(R.TULU31_START.repo, revision=R.TULU31_START.commit)
    ids = [render(tok, r) for r in rows]
    greedy = SamplingParams(temperature=0.0, max_tokens=cfg["max_tokens"])
    verifier = Verifier(REPO / ".venv-verify" / "bin" / "python")
    env = env_block(E2E, ["runs"], ("vllm", "torch", "jax", "transformers"))

    for model in cfg["models"]:
        if model["name"] != args.model:
            continue
        path = out / f"{model['name']}.json"
        if path.exists():
            print(f"{model['name']}: exists, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        llm, source = build(model, args.smoke)
        outs = llm.generate([{"prompt_token_ids": i} for i in ids], greedy, use_tqdm=False)
        items = [{"text": o.outputs[0].text, "ground_truth": r["ground_truth"],
                  "dataset": r["dataset"], "stopped": o.outputs[0].finish_reason == "stop"}
                 for o, r in zip(outs, rows)]
        rewards = verifier.score(items)
        if len(rewards) != len(rows):  # the work asked for is the work done
            raise SystemExit(f"{len(rewards)} rewards for {len(rows)} prompts")
        by = {}
        for r, rw in zip(rows, rewards):
            by.setdefault(r["dataset"], []).append(rw)
        rec = {
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "env": env, "model": model, "source": source,
            "stream_steps": cfg["stream_steps"], "prompts": len(rows),
            "reward": float(np.mean(rewards)),
            "reward_se": float(np.std(rewards, ddof=1) / np.sqrt(len(rewards))),
            "by_source": {k: {"reward": float(np.mean(v)), "n": len(v)}
                          for k, v in sorted(by.items())},
            "mean_len": float(np.mean([len(o.outputs[0].token_ids) for o in outs])),
            "capped": int(sum(o.outputs[0].finish_reason == "length" for o in outs)),
            "seconds": time.perf_counter() - t0,
        }
        harness.write_atomic(path, rec)
        print(f"{model['name']}: reward {rec['reward']:.3f} +- {rec['reward_se']:.3f} "
              f"{ {k: round(v['reward'], 2) for k, v in rec['by_source'].items()} } "
              f"{rec['seconds']:.0f}s", flush=True)
        del llm
        gc.collect()
        import torch  # noqa: PLC0415
        torch.cuda.empty_cache()
    verifier.close()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
