#!/usr/bin/env python
"""Throughput probe: vLLM greedy decode of ES member batches, per GPU type.

    python probe_throughput.py --config probe.yaml --out results-<gpu>
    python probe_throughput.py --config probe.yaml --out results-smoke --smoke

One JSON per cell (setting x cell), written once; a rerun skips cells that exist, so a
killed pod resumes where it stopped. Each record holds, per member batch: prompt and
generated token counts per sequence, finish reasons, wall seconds for the batch, and the
time of one in-place rewrite of all weights through the worker extension.

`--smoke` swaps in a small cached model and tiny shapes so the whole path (data, template,
engine, worker extension, records) runs on the laptop. A smoke record is marked as one and
is never a result.

Run in a venv with `requirements-vllm.txt`. JAX is present only as the library's CPU
dependency; it is kept off the GPU so vLLM gets the memory the config assigns.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # before anything can import jax
# Greedy decoding never samples, and FlashInfer's sampler JIT-compiles at warmup, which
# needs nvcc; vLLM's own sampler does the same job with nothing to build.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

PROBE = Path(__file__).resolve().parent
E2E = PROBE.parent
sys.path.insert(0, str(E2E))
sys.path.insert(0, str(E2E.parent))
import harness  # noqa: E402
import releases as R  # noqa: E402
import templates  # noqa: E402
from provenance import env_block  # noqa: E402

SMOKE_MODEL = ("Qwen/Qwen2.5-0.5B-Instruct", None)


def summarize(lengths, finish) -> dict:
    """Length statistics for one cell. `finish` holds vLLM finish reasons."""
    a = np.asarray(lengths, dtype=float)
    return {
        "sequences": int(a.size),
        "mean": float(a.mean()), "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)),
        "max": int(a.max()),
        "cap_hits": int(sum(f == "length" for f in finish)),
    }


def cell_name(setting: str, cell: dict) -> str:
    return (f"{setting}-u{cell['gpu_memory_utilization']:.2f}"
            f"-p{cell['prompts_per_member']}-m{cell['members']}")


def prompt_messages(setting: str, spec: dict, needed: int) -> list:
    """`needed` prompts, each a message list, in the order members consume them."""
    if spec["prompts"] == "olmo3_if_stream":
        # Read from Dolci (8 MB) rather than the 4 GB source: the Phase 0 record proves
        # Dolci is the run's training set, prompt for prompt. Use it only on that proof.
        import pyarrow.parquet as pq  # noqa: PLC0415
        from huggingface_hub import HfFileSystem  # noqa: PLC0415

        tset = json.loads((E2E / "data/olmo3_if/training-set.json").read_text())
        stream = json.loads((E2E / "data/olmo3_if/prompt-stream.json").read_text())
        d = tset["dolci"]
        if (stream["order_keys_sha256"] != tset["order_keys_sha256"]
                or d["revision"] != R.OLMO3_IF_DOLCI.commit
                or d["common_keys"] != tset["training_rows"]
                or d["prompt_differences"]["count"] != 0):
            raise SystemExit("the Phase 0 record does not prove Dolci is the run's set")
        fs = HfFileSystem()
        base = f"datasets/{R.OLMO3_IF_DOLCI.repo}@{R.OLMO3_IF_DOLCI.commit}/data"
        raw = {r["key"]: r["prompt"]
               for f in sorted(fs.ls(base, detail=False)) if f.endswith(".parquet")
               for r in pq.read_table(f, filesystem=fs, columns=["key", "prompt"]).to_pylist()}
        keys = [tset["order_keys"][p] for step in stream["stream"] for p in step][:needed]
        out = []
        for k in keys:
            # Dolci's prompt is open-instruct's raw prompt, "user: <content>" for the
            # single-message rows; anything else is not what this probe expects.
            if not raw[k].startswith("user: "):
                raise SystemExit(f"{k}: not a single user message")
            out.append([{"role": "user", "content": raw[k][len("user: "):]}])
        return out
    if spec["prompts"] == "tulu31_sample":
        import pyarrow.parquet as pq  # noqa: PLC0415
        from huggingface_hub import HfFileSystem  # noqa: PLC0415

        fs = HfFileSystem()
        f = (f"datasets/{R.TULU31_DATA.repo}@{R.TULU31_DATA.commit}"
             "/data/train-00000-of-00001.parquet")
        rows = pq.read_table(f, filesystem=fs, columns=["messages"]).to_pylist()
        pick = np.random.default_rng(spec["prompt_seed"]).choice(len(rows), needed,
                                                                 replace=False)
        out = []
        for i in pick.tolist():
            m = rows[i]["messages"]
            m = eval_messages(m)
            out.append(m if len(m) == 1 else m[:-1])
        return out
    raise SystemExit(f"unknown prompt source {spec['prompts']}")


def eval_messages(m):
    import ast  # noqa: PLC0415
    return ast.literal_eval(m) if isinstance(m, str) else m


def render(tok, template: str, messages: list, add_bos: bool) -> list:
    """Token ids the way open-instruct built prompts: template text, then encode
    without special tokens (plus bos only if the run asked for it)."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   chat_template=template)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return ([tok.bos_token_id] if add_bos else []) + ids


def gpu_facts() -> dict:
    import torch  # noqa: PLC0415

    q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout
    return {"torch_device": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda, "nvidia_smi": q.strip()}


def run_setting(setting, spec, cells, out, cfg, smoke, env):
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    todo = [c for c in cells if not (out / f"{cell_name(setting, c)}.json").exists()]
    if not todo:
        print(f"{setting}: all cells exist, skipping")
        return
    rel = getattr(R, spec["model"])
    repo, revision = SMOKE_MODEL if smoke else (rel.repo, rel.commit)
    template = getattr(templates, spec["template"])
    needed = max(c["prompts_per_member"] * c["members"] for c in todo)
    messages = prompt_messages(setting, spec, needed)
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    ids = [render(tok, template, m, spec["add_bos"]) for m in messages]

    for util in sorted({c["gpu_memory_utilization"] for c in todo}, reverse=True):
        llm = LLM(model=repo, revision=revision, dtype="bfloat16", seed=cfg["vllm_seed"],
                  gpu_memory_utilization=util, max_model_len=spec["max_model_len"],
                  enable_prefix_caching=False,
                  worker_extension_cls="probe_worker.WeightRewriter")
        greedy = SamplingParams(temperature=0.0, max_tokens=spec["max_tokens"],
                                stop=spec["stop"] or None)
        w = cfg["warmup"]
        llm.generate([{"prompt_token_ids": i} for i in ids[: w["prompts"]]],
                     SamplingParams(temperature=0.0, max_tokens=w["max_tokens"]),
                     use_tqdm=False)
        for cell in [c for c in todo if c["gpu_memory_utilization"] == util]:
            P = cell["prompts_per_member"]
            members = []
            for m in range(cell["members"]):
                batch = ids[m * P:(m + 1) * P]
                rw = llm.collective_rpc("rewrite_all")[0]
                t0 = time.perf_counter()
                res = llm.generate([{"prompt_token_ids": i} for i in batch], greedy,
                                   use_tqdm=False)
                wall = time.perf_counter() - t0
                gen = [len(r.outputs[0].token_ids) for r in res]
                fin = [r.outputs[0].finish_reason for r in res]
                if len(res) != P or sum(gen) == 0:  # the work asked for is the work done
                    raise SystemExit(f"{setting} member {m}: {len(res)} results, "
                                     f"{sum(gen)} tokens")
                members.append({
                    "member": m, "wall_seconds": wall,
                    "prompt_tokens": [len(i) for i in batch],
                    "generated_tokens": gen, "finish_reasons": fin,
                    "generated_tokens_per_second": sum(gen) / wall,
                    "weight_rewrite": rw,
                })
                print(f"{setting} u{util} P{P} member {m}: {sum(gen)} tokens in "
                      f"{wall:.1f}s = {sum(gen) / wall:.0f} tok/s, cap hits "
                      f"{sum(f == 'length' for f in fin)}, rewrite "
                      f"{rw['seconds'] * 1e3:.1f} ms", flush=True)
            lens = [g for mm in members for g in mm["generated_tokens"]]
            fins = [f for mm in members for f in mm["finish_reasons"]]
            walls = sum(mm["wall_seconds"] for mm in members)
            harness.write_atomic(out / f"{cell_name(setting, cell)}.json", {
                "date": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"),
                "smoke": smoke, "env": env, "setting": setting,
                "model": {"repo": repo, "revision": revision},
                "spec": {k: v for k, v in spec.items() if k != "cells"}, "cell": cell,
                "members": members,
                "lengths": summarize(lens, fins),
                "generated_tokens_per_second": sum(lens) / walls,
                "prompt_tokens_total": sum(sum(mm["prompt_tokens"]) for mm in members),
            })
        del llm
        import gc  # noqa: PLC0415
        import torch  # noqa: PLC0415
        gc.collect()
        torch.cuda.empty_cache()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="directory under probe/")
    ap.add_argument("--only", default=None, help="run one setting")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load((PROBE / args.config).read_text())
    out = (PROBE / args.out).resolve()
    out.relative_to(PROBE)
    out.mkdir(parents=True, exist_ok=True)
    # The worker extension is imported inside vLLM's own processes.
    os.environ["PYTHONPATH"] = f"{PROBE}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

    env = env_block(PROBE, [str(out.relative_to(PROBE))],
                    ("vllm", "torch", "transformers", "numpy", "jax"))
    env.update(gpu_facts())
    env["vllm_env"] = {k: v for k, v in os.environ.items() if k.startswith("VLLM_")}
    for setting, spec in cfg["settings"].items():
        if args.only and setting != args.only:
            continue
        cells = spec["cells"]
        if args.smoke:
            spec = {**spec, "max_tokens": 64, "max_model_len": 4096}
            cells = [{"gpu_memory_utilization": 0.5, "prompts_per_member": 4,
                      "members": 2}]
        run_setting(setting, spec, cells, out, cfg, args.smoke, env)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
