#!/usr/bin/env python
"""Do the gains of low-rank ES updates accumulate? Eight updates at 512 members.

    python -m es_vllm.lowrank_run --out runs/lowrank-run          # on the GPU, resumable
    python -m es_vllm.lowrank_run --analyze runs/lowrank-run
    python -m es_vllm.lowrank_run --out ... --smoke --iterations 1   # wiring, then 2 to resume

One update from the start, at 512 low-rank mirrored members and its best step, gains
0.036 +- 0.010 held-out reward points net of its random part's cost (runs/lowrank-check/),
about RL's 0.040 per the same 192 prompts. That was measured once, from the start. This
runs the update eight times, at RL's prompt stream (iteration g on RL steps 4g+1 to 4g+4),
and scores the 3,840 held-out prompts of RL steps 121 to 160 and 481 to 520 after 0, 4
and 8 updates.

Members, as `lowrank_check.py`: rank-1 `sigma a b^T` (sigma 5e-4) on every attention and
MLP matrix, mirrored pairs, served as LoRA adapters built in memory from their seeds;
generation 0 uses `lowrank_check.py`'s seeds, so the first update is the one it measured
at 512 members. Each iteration: the 512 members decode the iteration's 192 prompts
greedily, the update `STEP (alpha / N) sum_m z_m E_m` is added to an f32 copy of the
perturbed leaves, and the engine's bf16 weights are written from it in place and checked
bit for bit. A restart replays the log (the factors regenerate from their seeds) and
checks the f32 copy's digest against the one logged.

Prediction, committed before the run: if each update keeps its first-update net, the held
out reward rises by about +0.14 after 4 updates and +0.29 after 8 (paired standard error
about 0.06 each). Below +0.10 after 8 updates, the gains do not accumulate at the one-update
rate. For scale, RL rose +0.44 on RL steps 121 to 160 by step 40 (runs/heldout-121-160/).
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import datetime  # noqa: E402
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
import releases as R  # noqa: E402
from es_vllm import lowrank  # noqa: E402
from es_vllm.lowrank_check import Verifiers, items  # noqa: E402
from provenance import env_block  # noqa: E402

SEED, SIGMA, ALPHA, RANK, P_RUN = 0, 5e-4, 5e-4, 1, 192
N, ITERATIONS, EVAL_EVERY = 512, 8, 4
# The best multiple of the unit update at 512 members: gain 0.0047 per unit, cost 0.000153
# per unit squared, lambda* = g / (2c) (runs/lowrank-check/, length 31.0).
STEP = 15.4
HELDOUT = ((120, 160), (480, 520))


def setting(smoke: bool) -> dict:
    if smoke:
        return {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "revision": None, "cap": 64, "n": 8,
                "eval_every": 1, "prompts": 16, "heldout": 24, "chunk": 4, "verifiers": 2,
                "gpu": 0.4}
    return {"repo": R.TULU31_START.repo, "revision": R.TULU31_START.commit, "cap": 2048,
            "n": N, "eval_every": EVAL_EVERY, "prompts": P_RUN, "heldout": None, "chunk": 64,
            "verifiers": 8, "gpu": 0.6}


def lines(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def append(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def run(args) -> int:
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from vllm.lora.request import LoRARequest  # noqa: PLC0415

    from es_vllm.run_tulu import heldout_rows, load_prompts, render  # noqa: PLC0415
    from es_vllm.worker import ESWorker  # noqa: PLC0415

    cfg = setting(args.smoke)
    out = E2E / args.out
    out.mkdir(parents=True, exist_ok=True)
    conf = {"seed": SEED, "sigma": SIGMA, "alpha": ALPHA, "rank": RANK, "step": STEP,
            "smoke": args.smoke, **{k: v for k, v in cfg.items() if k not in ("verifiers", "chunk")}}
    if (out / "run.json").exists():
        if json.loads((out / "run.json").read_text())["config"] != conf:
            raise SystemExit(f"{out} holds a run with another configuration")
    else:
        harness.write_atomic(out / "run.json", {
            "config": conf, "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "env": env_block(E2E, ["runs"], ("vllm", "torch"))})
    log, h_log = out / "log.jsonl", out / "heldout.jsonl"
    done, h_done = lines(log), {r["iteration"] for r in lines(h_log)}
    n, pairs = cfg["n"], cfg["n"] // 2

    model_dir = snapshot_download(cfg["repo"], revision=cfg["revision"],
                                  allow_patterns=["*.safetensors", "*.json"])
    shapes = lowrank.leaf_shapes(model_dir)
    template = lowrank.write_template(Path(args.workdir) / "adapter", RANK, cfg["repo"])
    lowrank.install_adapters(template, SEED, shapes, RANK, SIGMA, n=n)
    for k, fn in lowrank.worker_methods().items():
        setattr(ESWorker, k, fn)
    llm = LLM(model=cfg["repo"], revision=cfg["revision"], dtype="bfloat16", seed=0,
              enable_lora=True, max_lora_rank=RANK, max_loras=cfg["chunk"], max_cpu_loras=n,
              gpu_memory_utilization=cfg["gpu"], max_model_len=4096,
              enable_prefix_caching=False, max_num_seqs=512,
              worker_extension_cls="es_vllm.worker.ESWorker")
    rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
    rpc("lr_setup", model_dir, SEED, RANK, 0, shapes)
    t0 = time.perf_counter()
    for rec in done:  # replay: the factors regenerate from their seeds
        rpc("lr_factors", SEED, RANK, pairs, shapes, lowrank.generation_of(rec["iteration"]))
        rpc("lr_step", list(STEP * lowrank.coefficients(rec["fitness"], n, ALPHA, RANK)))
    if done:
        if rpc("lr_digest") != done[-1]["digest"]:
            raise SystemExit("the replayed weights are not the logged ones")
        print(f"replayed {len(done)} updates {time.perf_counter() - t0:.0f}s", flush=True)
    if not rpc("lr_check")["ok"]:
        raise SystemExit("engine weights are not the f32 copy")

    tok = AutoTokenizer.from_pretrained(cfg["repo"], revision=cfg["revision"])
    batches = load_prompts(E2E / "data" / "tulu31", args.iterations, P_RUN)
    if args.smoke:
        h_rows = [r for b in load_prompts(E2E / "data" / "tulu31", 8, P_RUN) for r in b][-cfg["heldout"]:]
    else:
        h_rows = [r for a, b in HELDOUT for r in heldout_rows(a, b)]
    h_ids = [render(tok, r) for r in h_rows]
    vs = Verifiers(cfg["verifiers"])
    greedy = SamplingParams(temperature=0.0, max_tokens=cfg["cap"])

    def heldout(g):
        if g in h_done or g % cfg["eval_every"]:
            return
        t1 = time.perf_counter()
        outs = llm.generate([{"prompt_token_ids": i} for i in h_ids], greedy, use_tqdm=False)
        r = vs.score(items(outs, h_rows))
        append(h_log, {"iteration": g, "prompts": len(r), "reward": float(np.mean(r)),
                       "mean_len": float(np.mean([len(o.outputs[0].token_ids) for o in outs])),
                       "per_prompt": r, "seconds": time.perf_counter() - t1})
        print(f"heldout after {g} updates: {np.mean(r):.3f}", flush=True)

    for g in range(len(done), args.iterations):
        heldout(g)
        t1 = time.perf_counter()
        rows = batches[g][: cfg["prompts"]]
        ids = [render(tok, r) for r in rows]
        rewards, lens = [], []
        for c0 in range(0, n, cfg["chunk"]):
            ms = list(range(c0, min(n, c0 + cfg["chunk"])))
            outs = llm.generate([{"prompt_token_ids": i} for _ in ms for i in ids], greedy,
                                lora_request=[LoRARequest(f"g{g}m{m}", g * n + m + 1, str(template))
                                              for m in ms for _ in ids], use_tqdm=False)
            r = vs.score(items(outs, rows * len(ms)))
            rewards += [r[k * len(ids):(k + 1) * len(ids)] for k in range(len(ms))]
            lens += [float(np.mean([len(o.outputs[0].token_ids) for o in outs[k * len(ids):(k + 1) * len(ids)]]))
                     for k in range(len(ms))]
        decode_s = time.perf_counter() - t1
        fitness = np.asarray(rewards).mean(1)
        k = list(STEP * lowrank.coefficients(fitness, n, ALPHA, RANK))
        rpc("lr_factors", SEED, RANK, pairs, shapes, lowrank.generation_of(g))
        norm = rpc("lr_norm", k)
        rpc("lr_step", k)
        check = rpc("lr_check")
        if not check["ok"]:
            raise SystemExit(f"engine weights are not the f32 copy after update {g + 1}")
        append(log, {"iteration": g, "fitness": fitness.tolist(), "mean_fitness": float(fitness.mean()),
                     "sd_fitness": float(fitness.std()), "mean_len": lens, "rewards": rewards,
                     "update_norm": norm, "digest": rpc("lr_digest"), "check": check["ok"],
                     "decode_seconds": decode_s, "seconds": time.perf_counter() - t1})
        print(f"[{g}] fitness {fitness.mean():.3f} sd {fitness.std():.3f} |update| {norm:.1f} "
              f"{time.perf_counter() - t1:.0f}s", flush=True)
    heldout(args.iterations)
    vs.close()
    return 0


def analyze(out: Path) -> dict:
    h = sorted(lines(out / "heldout.jsonl"), key=lambda r: r["iteration"])
    log = lines(out / "log.jsonl")
    base = np.asarray(h[0]["per_prompt"])
    res = {"heldout": [], "iterations": [{"iteration": r["iteration"], "mean_fitness": r["mean_fitness"],
                                         "update_norm": r["update_norm"], "seconds": r["seconds"]}
                                        for r in log]}
    for r in h:
        d = np.asarray(r["per_prompt"]) - base
        res["heldout"].append({"iteration": r["iteration"], "reward": r["reward"],
                               "change": float(d.mean()),
                               "change_se": float(d.std(ddof=1) / np.sqrt(d.size)) if r is not h[0] else 0.0,
                               "mean_len": r["mean_len"]})
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analyze", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/lowrank-run"))
    args = ap.parse_args(argv)
    if args.analyze:
        print(json.dumps(analyze(E2E / args.analyze), indent=2))
        return 0
    if args.out is None:
        ap.error("--out or --analyze")
    args.workdir.mkdir(parents=True, exist_ok=True)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
