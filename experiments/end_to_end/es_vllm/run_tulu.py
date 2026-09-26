#!/usr/bin/env python
"""ES on Tulu 3.1's RL data: shardes trains, vLLM generates (docs/end_to_end/01-tulu31-plan.md).

    python -m es_vllm.run_tulu --config es_vllm/tulu-pilot-s1e-3.yaml      # from experiments/end_to_end
    python -m es_vllm.run_tulu --config ... --smoke                          # laptop, small model

One iteration: `es_ask`; for each member, write its weights into the engine, decode the
iteration's prompts greedily, score each response with the run's verifiers; then
`es_tell` on the members' mean rewards and write the new view back. Every member sees
the same prompts: four of the RL run's steps (4 x 48), in the run's own order, so ES
iteration g covers RL steps 4g+1 to 4g+4 and the rollout count matches every 4 steps.

The log (`log.jsonl`, one line per iteration) holds each member's fitness, which is all a
rebuild needs: any checkpoint is the start weights plus the logged `es_tell`s. A killed
run resumes by replaying them, with no generation. `es_digest` every `digest_every`
iterations is what a rebuild is checked against.

`reward: random` is the control: members get seeded uniform fitness and nothing is
generated for them, so it costs only the updates.

With `eval_center`, each iteration first decodes the current weights (the view) on the
iteration's own prompts, before any member sees them. In the first epoch those prompts
are new to the run, so this is a held-out measurement of the weights so far, and the one
number the random-reward control and the true-reward arms share.
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # engine, worker, JAX: one process
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")      # greedy; its JIT needs nvcc
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # JAX shares the GPU

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
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
import releases as R  # noqa: E402
import templates  # noqa: E402
from olmo3_if_data import as_messages  # noqa: E402
from provenance import env_block  # noqa: E402
from tulu31_data import row_hash  # noqa: E402

STEPS_PER_ITERATION = 4
SMOKE = {"model": ("Qwen/Qwen2.5-0.5B-Instruct", None), "population": 4,
         "prompts_per_member": 48, "smoke_prompts": 8, "iterations": 2, "max_tokens": 64,
         "max_model_len": 4096, "gpu_memory_utilization": 0.4, "digest_every": 1}


def load_prompts(data_dir: Path, needed_steps: int, per_member: int):
    """The rows each iteration uses, checked against the recorded row hashes."""
    import pyarrow.parquet as pq  # noqa: PLC0415
    from huggingface_hub import HfFileSystem  # noqa: PLC0415

    tset = json.loads((data_dir / "training-set.json").read_text())
    stream = json.loads((data_dir / "prompt-stream.json").read_text())
    if stream["row_sha256_digest"] != tset["row_sha256_digest"]:
        raise SystemExit("prompt stream and training set disagree")
    fs = HfFileSystem()
    f = (f"datasets/{R.TULU31_DATA.repo}@{R.TULU31_DATA.commit}"
         "/data/train-00000-of-00001.parquet")
    rows = pq.read_table(f, filesystem=fs).to_pylist()
    kept = tset["kept_indices"]
    per_step = stream["prompts_per_step"]
    if per_member % per_step:
        raise SystemExit(f"prompts_per_member {per_member} is not a multiple of {per_step}")
    steps = per_member // per_step
    iterations = []
    for g in range(needed_steps):
        batch = []
        for s in range(g * steps, (g + 1) * steps):
            for pos in stream["stream"][s]:
                row = rows[kept[pos]]
                if row_hash(row) != tset["row_sha256"][pos]:
                    raise SystemExit(f"row {kept[pos]} does not match the record")
                batch.append(row)
        iterations.append(batch)
    return iterations


def render(tok, row) -> list:
    msgs = as_messages(row["messages"])
    prompt = msgs if len(msgs) == 1 else msgs[:-1]
    text = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True,
                                   chat_template=templates.TULU)
    return tok(text, add_special_tokens=False)["input_ids"]  # no bos, as the run


class Verifier:
    def __init__(self, python: Path):
        self.p = subprocess.Popen([str(python), "-m", "es_vllm.verify_server"], cwd=E2E,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        first = json.loads(self.p.stdout.readline())
        if "ready" not in first:
            raise SystemExit(f"verifier failed to start: {first}")
        self.env = first["ready"]

    def score(self, items) -> list:
        for it in items:
            self.p.stdin.write(json.dumps(it) + "\n")
        self.p.stdin.flush()
        return [json.loads(self.p.stdout.readline())["reward"] for _ in items]

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=30)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load((E2E / args.config).read_text())
    name = args.config.stem + ("-smoke" if args.smoke else "")
    if args.smoke:
        cfg = {**cfg, **{k: v for k, v in SMOKE.items() if k != "model"}}
    out = E2E / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"

    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    rel = getattr(R, cfg["model"])
    repo, revision = SMOKE["model"] if args.smoke else (rel.repo, rel.commit)
    N, P = cfg["population"], cfg["prompts_per_member"]
    sigma, lr = cfg["sigma"], cfg["alpha"] * cfg["sigma"]  # Qiu's step, library convention
    random_control = cfg.get("reward", "verifiers") == "random"

    done = [json.loads(line) for line in log_path.read_text().splitlines()] \
        if log_path.exists() else []
    # runs/ holds nothing but run outputs, so none of it makes a tree dirty: an arm chained
    # after another on the same pod must not see the first one's directory as a stray file.
    env = env_block(E2E, ["runs"],
                    ("vllm", "torch", "jax", "jaxlib", "transformers", "numpy"))
    if not done:
        (out / "run.json").write_text(json.dumps(
            {"config": cfg, "config_file": str(args.config), "smoke": args.smoke,
             "model": {"repo": repo, "revision": revision}, "env": env,
             "started": datetime.datetime.now(datetime.timezone.utc).isoformat()},
            indent=2, sort_keys=True))

    llm = LLM(model=repo, revision=revision, dtype="bfloat16", seed=0,
              gpu_memory_utilization=cfg["gpu_memory_utilization"],
              max_model_len=cfg["max_model_len"], enable_prefix_caching=False,
              worker_extension_cls="es_vllm.worker.ESWorker")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # unwind, so the engine goes too
    model_dir = snapshot_download(repo, revision=revision,
                                  allow_patterns=["*.safetensors", "*.json"])
    info = llm.collective_rpc("es_init", args=(model_dir, N, sigma, lr, cfg["seed"]))[0]
    print(f"es_init {info}", flush=True)

    # Resume: replay the logged updates. No generation, only the noise and the contraction.
    for rec in done:
        llm.collective_rpc("es_ask")
        llm.collective_rpc("es_tell", args=(rec["fitness"],))
    if done:
        llm.collective_rpc("es_restore")
        digest = llm.collective_rpc("es_digest")[0]
        last = next((r["digest"] for r in reversed(done) if "digest" in r), None)
        if last is not None and done[-1].get("digest") and digest != done[-1]["digest"]:
            raise SystemExit("replayed state does not match the log's last digest")
        print(f"resumed after {len(done)} iterations", flush=True)

    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    iterations = load_prompts(E2E / cfg["data"], cfg["iterations"], P)
    greedy = SamplingParams(temperature=0.0, max_tokens=cfg["max_tokens"])
    center = cfg.get("eval_center", False)
    verifier = None if random_control and not center else \
        Verifier(REPO / ".venv-verify" / "bin" / "python")
    rng = np.random.default_rng(cfg["seed"])
    for _ in range(len(done)):
        rng.uniform(size=N)  # keep the control's stream aligned across resumes

    for g in range(len(done), cfg["iterations"]):
        t0 = time.perf_counter()
        llm.collective_rpc("es_ask")
        rows = iterations[g][: cfg.get("smoke_prompts")] if args.smoke else iterations[g]
        record = {"iteration": g, "prompts": len(rows)}
        ids = [render(tok, r) for r in rows]

        def decode_and_score():
            outs = llm.generate([{"prompt_token_ids": i} for i in ids], greedy,
                                use_tqdm=False)
            items = [{"text": o.outputs[0].text, "ground_truth": r["ground_truth"],
                      "dataset": r["dataset"],
                      "stopped": o.outputs[0].finish_reason == "stop"}
                     for o, r in zip(outs, rows)]
            rewards = verifier.score(items)
            if len(rewards) != len(rows):  # the work asked for is the work done
                raise SystemExit(f"{len(rewards)} rewards for {len(rows)} prompts")
            return outs, rewards

        def by_source(rewards):
            acc = {}
            for r, rw in zip(rows, rewards):
                s = acc.setdefault(r["dataset"], [0.0, 0])
                s[0] += rw
                s[1] += 1
            return {k: v[0] / v[1] for k, v in acc.items()}

        if center:
            outs, rewards = decode_and_score()
            record.update({"center_reward": float(np.mean(rewards)),
                           "center_by_source": by_source(rewards),
                           "center_mean_len": float(np.mean(
                               [len(o.outputs[0].token_ids) for o in outs]))})
        if random_control:
            fitness = rng.uniform(size=N).tolist()
        else:
            fitness, lengths, capped, all_rewards = [], [], 0, []
            for m in range(N):
                llm.collective_rpc("es_perturb", args=(m,))
                if g == 0 and m in cfg.get("check_members", [0]):
                    check = llm.collective_rpc("es_check", args=(m,))[0]
                    record.setdefault("checks", []).append({"member": m, **check})
                    if not check["ok"]:
                        raise SystemExit(f"engine weights are not member {m}: {check}")
                outs, rewards = decode_and_score()
                fitness.append(float(np.mean(rewards)))
                all_rewards += rewards
                lengths += [len(o.outputs[0].token_ids) for o in outs]
                capped += sum(o.outputs[0].finish_reason == "length" for o in outs)
            record.update({
                "mean_len": float(np.mean(lengths)), "max_len": int(max(lengths)),
                "capped": int(capped),
                # all_rewards is member-major: member m's rewards for rows 0.., then m+1.
                "reward_by_source": {
                    k: float(np.mean([v for i, v in enumerate(all_rewards)
                                      if rows[i % len(rows)]["dataset"] == k]))
                    for k in sorted({r["dataset"] for r in rows})},
            })
        tell = llm.collective_rpc("es_tell", args=(fitness,))[0]
        llm.collective_rpc("es_restore")
        if g == 0 and not random_control:
            check = llm.collective_rpc("es_check", args=(None,))[0]
            record.setdefault("checks", []).append({"member": None, **check})
            if not check["ok"]:
                raise SystemExit(f"engine weights are not the view after restore: {check}")
        record.update({"fitness": fitness, "mean_fitness": float(np.mean(fitness)),
                       "tell": tell, "seconds": time.perf_counter() - t0})
        if (g + 1) % cfg.get("digest_every", 5) == 0 or g + 1 == cfg["iterations"]:
            record["digest"] = llm.collective_rpc("es_digest")[0]
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{g}] center {record.get('center_reward', float('nan')):.3f} mean fitness "
              f"{record['mean_fitness']:.3f} {record.get('reward_by_source', '')} "
              f"{record['seconds']:.0f}s", flush=True)

    if verifier is not None:
        verifier.close()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
