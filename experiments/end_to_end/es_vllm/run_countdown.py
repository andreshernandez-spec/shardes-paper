#!/usr/bin/env python
"""ES on Countdown with the vLLM backend: the implementation check against es-at-scale.

    python -m es_vllm.run_countdown --config es_vllm/countdown-s1.yaml   # from experiments/end_to_end
    python -m es_vllm.run_countdown --config ... --smoke                 # laptop, tiny shapes

docs/end_to_end/06-countdown-check.md fixes what runs and the gate. The task is
es-at-scale's Countdown, read from a clone of it at the pinned commit and never copied
(its license is academic and copyleft): the 200 training prompts, the 2,000 evaluation
prompts and `countdown_reward_fn`, called in a process pool with es-at-scale's timeout
and its rule that a timed-out response scores 0.

One iteration: `es_ask`; for each member, write its weights into the engine, decode all
200 training prompts greedily, score them; `es_tell` on the members' mean rewards; write
the view back and check the engine holds it bit for bit. Before the first iteration,
every `eval_every` iterations and after the last, the view is decoded on the evaluation
prompts (`eval.jsonl`, with per-prompt reward, answer, format and length).

No resume: a killed run starts again. Replaying a fitness log does not reliably rebuild
the weights (runs/heldout-121-160/README.md).
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # engine, worker, JAX: one process
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")      # greedy; its JIT needs nvcc
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # JAX shares the GPU

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

PKG = Path(__file__).resolve().parent
E2E = PKG.parent
sys.path.insert(0, str(E2E))
sys.path.insert(0, str(E2E.parent))
import releases as R  # noqa: E402
from provenance import env_block  # noqa: E402

SMOKE = {"population": 4, "train_prompts": 8, "eval_prompts": 16, "iterations": 2,
         "max_tokens": 64, "eval_every": 1, "gpu_memory_utilization": 0.4}


def es_at_scale(path: Path) -> Path:
    """The clone, at the pinned commit with no local changes, on sys.path."""
    head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain",
                            "--untracked-files=no"], check=True, capture_output=True,
                           text=True).stdout.strip()
    if head != R.ES_AT_SCALE or dirty:
        raise SystemExit(f"{path} is at {head[:12]}{' with changes' if dirty else ''}, "
                         f"expected {R.ES_AT_SCALE[:12]}")
    sys.path.insert(0, str(path))
    return path


def rows(clone: Path, split: str) -> list:
    import pyarrow as pa  # noqa: PLC0415

    sub = {"train": "train/countdown/train",
           "eval": "evaluation_suite/countdown/countdown_eval"}[split]
    f = clone / "datasets" / sub / "data-00000-of-00001.arrow"
    with open(f, "rb") as fh:
        return pa.ipc.open_stream(fh).read_all().to_pylist()


class Grader:
    """es-at-scale's grading, as its trainer calls it: one response at a time through a
    pool of 8 processes, 0 on timeout. Built before CUDA exists in this process, since the
    pool forks."""

    def __init__(self, timeout: float):
        from es_at_scale.reward_function.countdown_grader import countdown_reward_fn  # noqa: PLC0415
        self.fn, self.timeout = countdown_reward_fn, timeout
        self.pool = multiprocessing.get_context("fork").Pool(8)

    def score(self, texts, batch):
        out = []
        for text, row in zip(texts, batch):
            res = self.pool.apply_async(
                self.fn, (text, {"numbers": row["numbers"], "target": row["target"]}))
            try:
                fmt, r = res.get(timeout=self.timeout)
                out.append((float(r), float(fmt["answer_reward"]), float(fmt["format_reward"])))
            except multiprocessing.TimeoutError:
                out.append((0.0, 0.0, 0.0))
        return out

    def close(self):
        self.pool.terminate()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--es-at-scale", type=Path,
                    default=Path.home() / "private" / "open-source" / "es-at-scale")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load((E2E / args.config).read_text())
    if args.smoke:
        cfg = {**cfg, **SMOKE}
    name = args.config.stem + ("-smoke" if args.smoke else "")
    out = E2E / "runs" / name
    if (out / "log.jsonl").exists():
        raise SystemExit(f"{out} has a log; a run is never resumed, move it aside first")
    out.mkdir(parents=True, exist_ok=True)

    clone = es_at_scale(args.es_at_scale)
    train = rows(clone, "train")[: cfg.get("train_prompts")]
    evals = rows(clone, "eval")[: cfg.get("eval_prompts")]
    grader = Grader(cfg["grader_timeout"])  # forks: before vLLM and JAX touch the GPU

    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    rel = getattr(R, cfg["model"])
    N, sigma, lr = cfg["population"], cfg["sigma"], cfg["alpha"] * cfg["sigma"]
    env = env_block(E2E, ["runs"], ("vllm", "torch", "jax", "jaxlib", "transformers", "numpy"))
    (out / "run.json").write_text(json.dumps(
        {"config": cfg, "config_file": str(args.config), "smoke": args.smoke,
         "model": {"repo": rel.repo, "revision": rel.commit}, "es_at_scale": R.ES_AT_SCALE,
         "train_prompts": len(train), "eval_prompts": len(evals), "env": env,
         "started": datetime.datetime.now(datetime.timezone.utc).isoformat()},
        indent=2, sort_keys=True))

    llm = LLM(model=rel.repo, revision=rel.commit, dtype="bfloat16", seed=0,
              gpu_memory_utilization=cfg["gpu_memory_utilization"],
              max_model_len=cfg["max_model_len"], enable_prefix_caching=False,
              worker_extension_cls="es_vllm.worker.ESWorker")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # unwind, so the engine goes too
    model_dir = snapshot_download(rel.repo, revision=rel.commit,
                                  allow_patterns=["*.safetensors", "*.json"])
    info = llm.collective_rpc("es_init", args=(model_dir, N, sigma, lr, cfg["seed"]))[0]
    print(f"es_init {info}", flush=True)
    greedy = SamplingParams(temperature=0.0, max_tokens=cfg["max_tokens"])

    def decode(batch):
        outs = llm.generate([r["context"] for r in batch], greedy, use_tqdm=False)
        scores = grader.score([o.outputs[0].text for o in outs], batch)
        if len(scores) != len(batch):  # the work asked for is the work done
            raise SystemExit(f"{len(scores)} scores for {len(batch)} prompts")
        lens = [len(o.outputs[0].token_ids) for o in outs]
        capped = sum(o.outputs[0].finish_reason == "length" for o in outs)
        return scores, lens, int(capped)

    def evaluate(updates):
        t0 = time.perf_counter()
        scores, lens, capped = decode(evals)
        s = np.asarray(scores)
        rec = {"updates": updates, "reward": float(s[:, 0].mean()),
               "solved": float((s[:, 1] > 0).mean()), "format": float(s[:, 2].mean()),
               "mean_len": float(np.mean(lens)), "capped": capped,
               "seconds": time.perf_counter() - t0,
               "per_prompt": {"reward": s[:, 0].tolist(), "answer": s[:, 1].tolist(),
                              "format": s[:, 2].tolist(), "length": lens}}
        with (out / "eval.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"eval after {updates} updates: reward {rec['reward']:.4f} solved "
              f"{rec['solved']:.4f} format {rec['format']:.3f} len {rec['mean_len']:.0f} "
              f"{rec['seconds']:.0f}s", flush=True)

    for g in range(cfg["iterations"]):
        if g % cfg["eval_every"] == 0:
            evaluate(g)
        t0 = time.perf_counter()
        llm.collective_rpc("es_ask")
        record: dict = {"iteration": g}
        fitness, lengths, capped, t_dec = [], [], 0, 0.0
        for m in range(N):
            llm.collective_rpc("es_perturb", args=(m,))
            if g % 10 == 0 and m in cfg.get("check_members", [0]):
                check = llm.collective_rpc("es_check", args=(m,))[0]
                record.setdefault("checks", []).append({"member": m, **check})
                if not check["ok"]:
                    raise SystemExit(f"engine weights are not member {m}: {check}")
            t1 = time.perf_counter()
            scores, lens, c = decode(train)
            t_dec += time.perf_counter() - t1
            fitness.append(float(np.mean([s[0] for s in scores])))
            lengths += lens
            capped += c
        t1 = time.perf_counter()
        tell = llm.collective_rpc("es_tell", args=(fitness,))[0]
        llm.collective_rpc("es_restore")
        check = llm.collective_rpc("es_check", args=(None,))[0]
        record.setdefault("checks", []).append({"member": None, **check})
        if not check["ok"]:
            raise SystemExit(f"engine weights are not the view after restore: {check}")
        t_upd = time.perf_counter() - t1
        record.update({"fitness": fitness, "mean_fitness": float(np.mean(fitness)),
                       "sd_fitness": float(np.std(fitness)), "mean_len": float(np.mean(lengths)),
                       "capped": capped, "tell": tell, "decode_seconds": t_dec,
                       "update_seconds": t_upd, "seconds": time.perf_counter() - t0})
        if (g + 1) % cfg["eval_every"] == 0 or g + 1 == cfg["iterations"]:
            record["digest"] = llm.collective_rpc("es_digest")[0]
        with (out / "log.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{g}] mean fitness {record['mean_fitness']:.4f} sd {record['sd_fitness']:.4f} "
              f"len {record['mean_len']:.0f} {record['seconds']:.0f}s "
              f"(decode {t_dec:.0f}, update {t_upd:.1f})", flush=True)
    evaluate(cfg["iterations"])
    grader.close()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
