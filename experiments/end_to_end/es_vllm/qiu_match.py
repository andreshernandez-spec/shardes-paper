#!/usr/bin/env python
"""Qiu et al.'s Countdown ES setup, setting by setting, through our backend (Arm B of
docs/end_to_end/12-qiu-matched.md).

    python -m es_vllm.qiu_match --out runs/qiu-matched/qwen3b-s1 --ckpt /root/qiu-ckpt --seed 1
    python -m es_vllm.qiu_match --out ... --ckpt ... --smoke      # laptop: 0.5B, tiny shapes

The paper's numbers come from the archive/ scripts of es-at-scale (its README says so).
Everything below follows those, read from the pinned clone and never copied (the license
is academic and copyleft):

- problems: archive/countdown/data/countdown.json, the first 200 for training (each one,
  for every member, every update) and the last 2,000 for testing
  (es_fine-tuning_countdown.py --data_sample 200; eval_countdown.py --eval_offset -2000);
- prompts: the raw context through the slow tokenizer, no chat template;
- decoding: greedy, at most 1,024 new tokens, stopping at either of the model's end
  tokens, with the model's repetition penalty (1.05), which HF's generate applies even
  when greedy. HF also penalizes the left padding in every row shorter than the longest
  of its batch; `pad_penalty.PadPenalty` does the same for the rows his batches pad (all
  200 training prompts in one batch, the test prompts in batches of 500, in order);
- reward: his `reward_function` on the decoded prompt and answer together, with the
  numbers, target and response taken by his evaluator's helpers (a target that is not
  an integer earns no answer point);
- ES: 30 members, sigma 1e-3, alpha 5e-4, z-scored fitness, 500 updates.

What stays ours is the implementation: shardes' f32 master and independent noise per
tensor (his: bf16 weights changed in place, one generator reseeded for every tensor),
the update summed in f32 (his: bf16), vLLM instead of HF's generate, and the z-score
without his 1e-8 in the denominator.

The reported test numbers come from his eval_countdown.py on checkpoints written after
0, 100, ..., 500 updates (`<ckpt>/ckpt-<k>/`: the engine's bf16 weights, read back and
checked against the run's digest). Every `--eval-every` updates the 2,000 test prompts
are also decoded here by the same rule in vLLM (eval.jsonl), as a running curve. A
restart replays the log as probe_run --full-rank does.
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import shutil  # noqa: E402
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

N, SIGMA, ALPHA, ITERATIONS = 30, 1e-3, 5e-4, 500
TRAIN, TEST, MAX_NEW = 200, 2000, 1024
TEST_BATCH = 500                 # eval_countdown.py --batch_size, as his README
CKPT_EVERY, EVAL_EVERY = 100, 10
TIMEOUT, MAX_TIMEOUTS = 10.0, 4  # es-at-scale's grading timeout; a hung grade holds a pool worker
SMOKE = {"setting": "qwen0.5b/countdown", "train": 8, "test": 16, "test_batch": 8, "n": 4,
         "max_new": 256, "ckpt_every": 2, "eval_every": 1}
TOKENIZER_FILES = ("config.json", "generation_config.json", "tokenizer.json",
                   "tokenizer_config.json", "vocab.json", "merges.txt")


def lines(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def append(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def problems(clone: Path, train: int, test: int) -> tuple:
    """His training slice (the first `train`) and test slice (the last `test`)."""
    d = json.loads((clone / "archive" / "countdown" / "data" / "countdown.json").read_text())
    return d[:train], d[-test:]


def padded(lengths: list, batch: int) -> list:
    """Which rows HF's generate pads on the left: those shorter than the longest prompt
    of their batch, the rows taken in order in batches of `batch`."""
    out = []
    for i in range(0, len(lengths), batch):
        part = lengths[i:i + batch]
        out += [n < max(part) for n in part]
    return out


def accuracy(scores: list) -> float:
    """His accuracy: the share of answers with an answer point."""
    return float(np.mean([s[1] > 0 for s in scores]))


def penalize(x, penalty):
    """HF's RepetitionPenaltyLogitsProcessor on the selected logits (pad_penalty.py)."""
    import torch  # noqa: PLC0415

    return torch.where(x < 0, x * penalty, x / penalty)


def his_functions(clone: Path) -> dict:
    """His reward function and his evaluator's helpers for the numbers, target and response."""
    sys.path[:0] = [str(clone / "archive"), str(clone / "archive" / "countdown")]
    import eval_countdown as ev  # noqa: PLC0415
    from countdown_task import reward_function  # noqa: PLC0415

    return {"reward": reward_function, "numbers_target": ev.extract_numbers_and_target,
            "response": ev.extract_model_response}


_G: dict = {}  # filled before the pool forks: the slow tokenizer and his functions


def _grade(ids, context, target):
    """His trainer's grading of one answer: the whole sequence decoded, prompt included."""
    text = _G["tok"].decode(ids, skip_special_tokens=True)
    numbers, tgt = _G["numbers_target"](context, target)
    r = _G["reward"](_G["response"](text), numbers, tgt)
    return float(r["reward"]), float(r["reward_info"]["answer_reward"]), float(r["reward_info"]["format_reward"])


class Grader:
    """A pool of 8 processes, forked before CUDA exists in this process; 0 on timeout."""

    def __init__(self, clone: Path, tok, timeout: float = TIMEOUT):
        _G.update(tok=tok, **his_functions(clone))
        self.pool = multiprocessing.get_context("fork").Pool(8)
        self.timeout, self.timeouts = timeout, 0

    def score(self, seqs: list, rows: list) -> list:
        jobs = [self.pool.apply_async(_grade, (s, r["context"], r["target"])) for s, r in zip(seqs, rows)]
        out = []
        for j in jobs:
            try:
                out.append(j.get(timeout=self.timeout))
            except multiprocessing.TimeoutError:
                self.timeouts += 1
                out.append((0.0, 0.0, 0.0))
        if self.timeouts >= MAX_TIMEOUTS:
            raise SystemExit(f"{self.timeouts} grades timed out; restart to get the pool back")
        return out

    def close(self):
        self.pool.terminate()


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def run(args) -> int:
    from vllm import SamplingParams  # noqa: PLC0415
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    from es_vllm import probe  # noqa: PLC0415
    from es_vllm.pad_penalty import PadPenalty  # noqa: PLC0415
    from es_vllm.run_countdown import es_at_scale  # noqa: PLC0415

    s = SMOKE if args.smoke else {"setting": "qwen3b/countdown", "train": TRAIN, "test": TEST, "test_batch": TEST_BATCH,
                                  "n": N, "max_new": MAX_NEW, "ckpt_every": CKPT_EVERY, "eval_every": args.eval_every}
    cfg = probe.setting(s["setting"], args.smoke)
    clone = es_at_scale(args.es_at_scale)
    train, test = problems(clone, s["train"], s["test"])
    files = Path(snapshot_download(cfg["repo"], revision=cfg["revision"], allow_patterns=list(TOKENIZER_FILES)))
    gen = json.loads((files / "generation_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(str(files), use_fast=False)   # his trainer's tokenizer
    rep, eos, pad = float(gen["repetition_penalty"]), sorted(gen["eos_token_id"]), tok.pad_token_id
    if pad not in eos:
        raise SystemExit(f"pad {pad} is not an end token {eos}: decoding would not stop on it")
    ids = {"train": [tok(r["context"])["input_ids"] for r in train], "test": [tok(r["context"])["input_ids"] for r in test]}
    if any(pad in x for v in ids.values() for x in v):
        raise SystemExit("a prompt holds the pad token, which vLLM's own penalty would then also cover")
    pads = {"train": padded([len(x) for x in ids["train"]], len(ids["train"])),   # one batch, as his trainer
            "test": padded([len(x) for x in ids["test"]], s["test_batch"])}

    def params(block):
        return [SamplingParams(temperature=0.0, top_p=1.0, max_tokens=s["max_new"], repetition_penalty=rep,
                               stop_token_ids=eos, extra_args={"pad_penalty": [pad, rep]} if p else None)
                for p in pads[block]]

    out = E2E / args.out
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)
    lr = ALPHA * SIGMA                    # shardes steps lr / sigma; his alpha / N * sum z eps, the same
    conf = {"setting": s["setting"], "repo": cfg["repo"], "revision": cfg["revision"], "seed": args.seed,
            "n": s["n"], "sigma": SIGMA, "alpha": ALPHA, "train": s["train"], "test": s["test"],
            "max_new_tokens": s["max_new"], "repetition_penalty": rep, "stop_token_ids": eos, "pad_token": pad,
            "test_batch": s["test_batch"], "ckpt_every": s["ckpt_every"], "eval_every": s["eval_every"],
            "smoke": args.smoke}
    data = {"es_at_scale": harness.git(clone, "rev-parse", "HEAD"),
            "file": "archive/countdown/data/countdown.json",
            "file_sha256": hashlib.sha256((clone / "archive/countdown/data/countdown.json").read_bytes()).hexdigest(),
            "train_digest": digest(train), "test_digest": digest(test),
            "prompt_ids_digest": digest(ids), "padded": {b: int(sum(v)) for b, v in pads.items()},
            "non_integer_targets": {"train": sum(not str(r["target"]).isdigit() for r in train),
                                    "test": sum(not str(r["target"]).isdigit() for r in test)}}
    if (out / "run.json").exists():
        meta = json.loads((out / "run.json").read_text())
        if meta["config"] != conf or meta["data"] != data:
            raise SystemExit(f"{out} holds a run with another configuration or other data")
    else:
        harness.write_atomic(out / "run.json", {
            "config": conf, "data": data, "iterations": args.iterations,
            "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "env": env_block(E2E, ["runs"], ("vllm", "torch", "jax", "jaxlib", "transformers"))})
    log, e_log, c_log = out / "log.jsonl", out / "eval.jsonl", out / "ckpt.jsonl"
    done, e_done, c_done = lines(log), {r["iteration"] for r in lines(e_log)}, {r["iteration"] for r in lines(c_log)}

    grader = Grader(clone, tok)                         # before CUDA: the pool forks
    llm = probe.engine(cfg, args.gpu, worker_extension_cls="es_vllm.worker.ESWorker", logits_processors=[PadPenalty])
    rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
    print(f"es_init {rpc('es_init', probe.model_dir(cfg), s['n'], SIGMA, lr, args.seed)}", flush=True)
    if not rpc("es_check", None)["ok"]:
        raise SystemExit("engine weights are not the f32 master's view")
    t0 = time.perf_counter()
    for rec in done:                                    # replay: each update from its logged fitnesses
        rpc("es_ask")
        rpc("es_tell", rec["fitness"])
        if rpc("es_digest") != rec["digest"]:
            raise SystemExit(f"replayed update {rec['iteration'] + 1} is not the logged one")
    if done:
        rpc("es_restore")
        if not rpc("es_check", None)["ok"]:
            raise SystemExit("engine weights are not the replayed master's view")
        print(f"replayed {len(done)} updates, every digest the logged one, {time.perf_counter() - t0:.0f}s", flush=True)
        meta = json.loads((out / "run.json").read_text())
        meta.setdefault("resumes", []).append({
            "from": len(done), "iterations": args.iterations,
            "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "env": env_block(E2E, ["runs"], ("vllm", "torch", "jax", "jaxlib", "transformers"))})
        harness.write_atomic(out / "run.json", meta)

    def decode(block, rows):
        o = probe.decode(llm, ids[block], params(block))
        seqs = [p + list(x.outputs[0].token_ids) for p, x in zip(ids[block], o)]
        return grader.score(seqs, rows), [len(x.outputs[0].token_ids) for x in o], \
            [x.outputs[0].finish_reason for x in o]

    def evaluate(g):
        if g in e_done or (g % s["eval_every"] and g != args.iterations):
            return
        t1 = time.perf_counter()
        sc, n_tok, fin = decode("test", test)
        append(e_log, {"iteration": g, "accuracy": accuracy(sc), "reward": [x[0] for x in sc],
                       "answer": [x[1] for x in sc], "format": [x[2] for x in sc], "length": n_tok,
                       "stopped": [f == "stop" for f in fin], "seconds": time.perf_counter() - t1})
        print(f"test after {g} updates: accuracy {accuracy(sc):.4f} mean length {np.mean(n_tok):.0f}", flush=True)

    def checkpoint(g):
        if g in c_done or g % s["ckpt_every"]:
            return
        d = ckpt / f"ckpt-{g}"
        if d.exists():
            shutil.rmtree(d)                            # a save the run did not finish
        d.mkdir()
        for f in TOKENIZER_FILES:
            shutil.copy(files / f, d / f)
        saved = rpc("es_save", str(d))
        if not saved["ok"]:
            raise SystemExit(f"checkpoint {g} is not the view the run logged")
        rec = {"iteration": g, "dir": d.name, "digest": saved["file_digest"]}
        harness.write_atomic(d / "saved.json", rec)     # the evaluator watches for this
        append(c_log, rec)
        print(f"checkpoint after {g} updates: {d}", flush=True)

    checkpoint(len(done))
    for g in range(len(done), args.iterations):
        evaluate(g)
        t1 = time.perf_counter()
        rpc("es_ask")
        rewards, answers, lengths, checks = [], [], [], []
        for m in range(s["n"]):
            rpc("es_perturb", m)
            if m == 0:                                  # the JAX-to-torch handoff (worker.py)
                checks.append({"member": 0, **rpc("es_check", 0)})
                if not checks[0]["ok"]:
                    raise SystemExit(f"engine weights are not member 0 at update {g + 1}")
            sc, n_tok, _ = decode("train", train)
            rewards.append([x[0] for x in sc])
            answers.append(sum(x[1] > 0 for x in sc))
            lengths.append(float(np.mean(n_tok)))
        decode_s = time.perf_counter() - t1
        fitness = np.asarray(rewards).mean(1)           # his: the mean reward over the 200
        tell = rpc("es_tell", fitness.tolist())
        rpc("es_restore")
        check = rpc("es_check", None)
        checks.append({"member": None, **check})
        if not check["ok"]:
            raise SystemExit(f"engine weights are not the f32 master after update {g + 1}")
        append(log, {"iteration": g, "fitness": fitness.tolist(), "mean_fitness": float(fitness.mean()),
                     "sd_fitness": float(fitness.std()), "solved": answers, "mean_len": lengths,
                     "rewards": rewards, "rollouts": s["n"] * len(train), "digest": rpc("es_digest"),
                     "tell": tell, "checks": checks, "timeouts": grader.timeouts,
                     "decode_seconds": decode_s, "seconds": time.perf_counter() - t1})
        print(f"[{g}] fitness {fitness.mean():.4f} sd {fitness.std():.4f} solved {np.mean(answers) / len(train):.3f}"
              f" {time.perf_counter() - t1:.0f}s", flush=True)
        checkpoint(g + 1)
    evaluate(args.iterations)
    grader.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="the run's records, under experiments/end_to_end")
    ap.add_argument("--ckpt", type=Path, required=True, help="checkpoints, outside the repository (6 GB each)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--gpu", type=float, default=0.35, help="vLLM's share; his evaluator runs beside it")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--es-at-scale", type=Path, default=Path.home() / "private" / "open-source" / "es-at-scale")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
