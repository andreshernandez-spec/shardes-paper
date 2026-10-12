#!/usr/bin/env python
"""The a-priori probe of docs/end_to_end/10: level 0 and level 1 on one model and task.

    python -m es_vllm.probe --setting qwen3b/countdown --phase sample   --out runs/probe/qwen3b-countdown
    ... --phase gradient / scan / sign / members / steps (the last writes prediction.json)
    python -m es_vllm.probe --analyze runs/probe/qwen3b-countdown
    (--smoke: tiny blocks and populations, wiring only)

Each phase is its own process (plain engine, torch, engine with the worker extension,
engine with LoRA). Blocks per task: `train` (192 prompts, what one ES iteration or four
RL steps see), `validation` (scored by every scan), `offtask` (prompts from outside the
task, scored by the same scans) and `test` (reserved, never decoded here; its digest is
in env.json). Rewards stay in the task's units in the records and are normalized to
[0, 1] by the task's maximum in the analysis.

Level 0, nothing ranked:
- `sample`: 16 answers per train prompt at temperature 1.0 from the start, scored. The
  live share (prompts whose answers do not all score alike) and its complement, Cho and
  Han's zero-advantage share; the start's greedy reward on every block; answers per
  second.
- `gradient`: per live train prompt, the gradient of GRPO's loss (token mean over the
  prompt's answers, the run's advantages) over the attention and MLP matrices. The
  sharedness of the per-prompt gradients, `mean_{i != j} g_i . g_j / mean |g_j|^2`, the
  analogue of the cross-prompt correlation of member effects; Yin et al.'s diversity
  `sum |g_j|^2 / |sum g_j|^2`; the summed gradient saved for `sign`.
- `scan`: the validation and offtask blocks at `theta_0 +- L v` for four random mirrored
  rank-1 directions `v` of ES's shape and three lengths around a member's (sigma
  sqrt(d)): the even response per length squared is the curvature `c`, by task, and
  names the task a random step damages first.
- `sign`: the same blocks at `theta_0 +- L s`, `s` the unit sign vector of the summed
  gradient (Adam's first direction), at per-coordinate displacements matching the Tulu
  scan's: the odd response per unit effective length, negated, is the slope along RL's
  direction, and RL's first step (lr sqrt(d)) times it is RL's expected first-step gain.

Level 1, one ES iteration:
- `members`: 64 mirrored rank-1 members at sigma 5e-4, and 32 at 1e-3 with the same
  seeds, decode the train block greedily: reliability of the ranking, cross-prompt
  correlation of member effects, Liang et al.'s best-of-N improvement.
- `steps`: the update from the 64 members at `theta_0 +- lambda u` at the same three
  lengths, on the validation and offtask blocks: odd response per unit update (`G`), even
  per unit squared (`C`), by task. Then the rule's prediction, written to
  prediction.json before any training: ES's rollouts to gain a tenth of the reward range
  at 2% damage against RL's from the sign slope at the Tulu run's step, their ratio, and
  the call (works below 5, compute parity at the measured cost per answer; marginal
  below 25; fails otherwise, or before that if the live share is under 0.1), and the
  task damaged first.
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
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
import templates  # noqa: E402
from es_vllm import lowrank  # noqa: E402
from olmo3_if_data import as_messages  # noqa: E402
from provenance import env_block  # noqa: E402

SEED, SIGMA, SIGMA2, ALPHA, RANK = 0, 5e-4, 1e-3, 5e-4, 1
P_TRAIN, K, N_MEMBERS, N_MEMBERS2, BLOCK = 192, 16, 64, 32, 300
SAMPLE_SEED = 30_000                       # as gradient_check.py
LENGTHS = (0.4, 1.0, 2.5)                  # multiples of a member's length
SIGN_LENGTHS = (0.5, 1.0)                  # Tulu units; scaled by sqrt(d / d_tulu)
TULU_D_M = 6_979_321_856                   # runs/lowrank-check/members.json perturbed_params
RL_LR, RL_ROLLOUTS_PER_STEP = 5e-7, R.TULU31_ROLLOUTS_PER_STEP   # the Tulu run: 0.045 / sqrt(8.03e9)
G_TARGET, DAMAGE = 0.1, 0.02               # of the reward range
WORKS, MARGINAL, LIVE_MIN = 5.0, 25.0, 0.1  # runs/throughput-bench: 0.255 / 0.048 per answer
USABLE = 0.2                               # a scan point is usable if neither sign moved more
CAPS = {"countdown": 512, "tulu": 2048}    # es_vllm/countdown-s1.yaml; the Tulu run

MODELS = {
    "tulu8b": (R.TULU31_START, "tulu"),
    "qwen0.5b": (R.QWEN25_05B, "chat"),
    "qwen1.5b": (R.QWEN25_15B, "chat"),
    "qwen3b": (R.QWEN25_3B, "chat"),
    "qwen3b-base": (R.QWEN25_3B_BASE, "chat"),
    "qwen7b": (R.QWEN25_7B, "chat"),
}


def lines(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def append(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def setting(name: str, smoke: bool) -> dict:
    model, task = name.split("/")
    rel, template = MODELS[model]
    return {"name": name, "model": model, "repo": rel.repo, "revision": rel.commit,
            "template": template, "task": task, "smoke": smoke}


def model_dir(cfg) -> str:
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    return snapshot_download(cfg["repo"], revision=cfg["revision"], allow_patterns=["*.safetensors", "*.json"])


# ------------------------------------------------------------------ tasks

def is_countdown(row) -> bool:
    return "context" in row


def row_kind(row) -> str:
    """The task family a row belongs to, for the by-task responses."""
    return "countdown" if is_countdown(row) else row["dataset"]


class Task:
    """Prompts in blocks, a renderer and a scorer for one task."""

    def __init__(self, name: str, template: str, smoke: bool, es_at_scale: Path):
        from es_vllm.run_tulu import heldout_rows, load_prompts  # noqa: PLC0415

        self.name, self.template, self.smoke = name, template, smoke
        n_train, n_block = (16, 24) if smoke else (P_TRAIN, BLOCK)
        if name == "countdown":
            from es_vllm.run_countdown import es_at_scale as pin, rows  # noqa: PLC0415

            self.clone = pin(es_at_scale)
            train, ev = rows(self.clone, "train"), rows(self.clone, "eval")
            gsm = [r for r in heldout_rows(0, 40) if r["dataset"] == "gsm8k"]
            self.max_reward = 1.1
            self.blocks = {"train": train[:n_train], "validation": ev[:n_block],
                           "test": ev[n_block:2 * n_block], "offtask": gsm[:n_block]}
        elif name == "gsm8k":
            pool = heldout_rows(0, 160)
            gsm = [r for r in pool if r["dataset"] == "gsm8k"]
            math = [r for r in pool if r["dataset"] == "MATH"]
            ifeval = [r for r in pool if r["dataset"] == "ifeval"]
            self.max_reward = 10.0
            self.blocks = {"train": gsm[:n_train], "validation": gsm[n_train:n_train + n_block],
                           "test": gsm[n_train + n_block:n_train + 2 * n_block],
                           "offtask": math[:n_block // 2] + ifeval[:n_block // 2]}
        elif name == "tulu":
            self.max_reward = 10.0
            self.blocks = {"train": load_prompts(E2E / "data" / "tulu31", 1, P_TRAIN)[0][:n_train],
                           "validation": heldout_rows(120, 160)[:n_block],
                           "test": heldout_rows(480, 520)[:n_block], "offtask": []}
        else:
            raise SystemExit(f"unknown task {name}")
        for b, rows_ in self.blocks.items():
            want = {"train": n_train, "offtask": 0 if name == "tulu" else n_block}.get(b, n_block)
            if len(rows_) != want:
                raise SystemExit(f"{name} {b}: {len(rows_)} rows, wanted {want}")

    def digests(self) -> dict:
        """One hash per block, so a later run can show it used the same prompts."""
        return {b: hashlib.sha256("".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows_).encode()).hexdigest()
                for b, rows_ in self.blocks.items()}

    def render(self, tok, row) -> list:
        if is_countdown(row):                      # es-at-scale's raw prompt, as its trainer
            return tok(row["context"])["input_ids"]
        msgs = as_messages(row["messages"])
        msgs = msgs if len(msgs) == 1 else msgs[:-1]
        kw = {"chat_template": templates.TULU} if self.template == "tulu" else {}
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        return tok(text, add_special_tokens=False)["input_ids"]      # no bos, as the run

    def cap(self, row) -> int:
        if self.smoke:
            return 64 if is_countdown(row) else 48
        return CAPS["countdown" if is_countdown(row) else "tulu"]

    def params(self, rows, n=1, temperature=0.0, seed=None) -> list:
        from vllm import SamplingParams  # noqa: PLC0415

        return [SamplingParams(n=n, temperature=temperature, top_p=1.0, max_tokens=self.cap(r),
                               **({} if seed is None else {"seed": seed + j}))
                for j, r in enumerate(rows)]

    def scorer(self):
        """Scores answers of either kind of row; built before CUDA (the grader forks)."""
        from es_vllm.lowrank_check import Verifiers  # noqa: PLC0415

        grader = None
        if self.name == "countdown":
            from es_vllm.run_countdown import Grader  # noqa: PLC0415

            grader = Grader(10.0)                  # es-at-scale's reward-function timeout
        return Scorer(Verifiers(2 if self.smoke else 8), grader)


class Scorer:
    def __init__(self, verifiers, grader):
        self.v, self.g = verifiers, grader

    def score(self, texts, rows, stopped) -> list:
        out = [None] * len(rows)
        cd = [i for i, r in enumerate(rows) if is_countdown(r)]
        tu = [i for i, r in enumerate(rows) if not is_countdown(r)]
        if cd:
            if self.g is None:
                raise SystemExit("Countdown rows without the grader")
            for i, s in zip(cd, self.g.score([texts[i] for i in cd], [rows[i] for i in cd])):
                out[i] = float(s[0])
        if tu:
            r = self.v.score([{"text": texts[i], "ground_truth": rows[i]["ground_truth"],
                               "dataset": rows[i]["dataset"], "stopped": bool(stopped[i])} for i in tu])
            for i, s in zip(tu, r):
                out[i] = float(s)
        if any(x is None for x in out):  # the work asked for is the work done
            raise SystemExit(f"{sum(x is not None for x in out)} scores for {len(rows)} answers")
        return out

    def close(self):
        self.v.close()
        if self.g is not None:
            self.g.close()


def decode(llm, ids, params, lora=None):
    return llm.generate([{"prompt_token_ids": i} for i in ids], params, use_tqdm=False,
                        **({"lora_request": lora} if lora is not None else {}))


def score_outs(scorer, outs, rows) -> list:
    return scorer.score([o.outputs[0].text for o in outs], rows,
                        [o.outputs[0].finish_reason == "stop" for o in outs])


def scan_rows(task) -> tuple:
    """The validation and offtask rows together, with each row's task and block."""
    blocks = {b: task.blocks[b] for b in ("validation", "offtask") if task.blocks[b]}
    rows = [r for rs in blocks.values() for r in rs]
    return rows, [row_kind(r) for r in rows], [b for b, rs in blocks.items() for _ in rs]


def engine(cfg, gpu, **kw):
    from vllm import LLM  # noqa: PLC0415

    return LLM(model=cfg["repo"], revision=cfg["revision"], dtype="bfloat16", seed=0,
               gpu_memory_utilization=0.4 if cfg["smoke"] else gpu, max_model_len=4096,
               enable_prefix_caching=False, max_num_seqs=512, **kw)


def tokenizer(cfg):
    from transformers import AutoTokenizer  # noqa: PLC0415

    return AutoTokenizer.from_pretrained(cfg["repo"], revision=cfg["revision"])


# ------------------------------------------------------------------ level 0

def phase_sample(args, cfg, task) -> None:
    out = E2E / args.out
    scorer = task.scorer()
    tok = tokenizer(cfg)
    ids = {b: [task.render(tok, r) for r in rows_] for b, rows_ in task.blocks.items() if b != "test"}
    llm = engine(cfg, 0.6)
    rows = task.blocks["train"]
    k = 4 if cfg["smoke"] else K
    t0 = time.perf_counter()
    outs = decode(llm, ids["train"], task.params(rows, n=k, temperature=1.0, seed=SAMPLE_SEED))
    t_sample = time.perf_counter() - t0
    flat = [o for out_ in outs for o in out_.outputs]
    rewards = scorer.score([o.text for o in flat], [r for r in rows for _ in range(k)],
                           [o.finish_reason == "stop" for o in flat])
    rew = np.asarray(rewards).reshape(len(rows), k)
    sd = rew.std(axis=1, ddof=1)                                     # torch.std's correction
    adv = (rew - rew.mean(axis=1, keepdims=True)) / (sd[:, None] + 1e-8)
    synthetic = cfg["smoke"] and not (sd > 0).any()   # wiring only: a small model at a short cap scores nothing
    if synthetic:
        adv = np.random.default_rng(0).standard_normal(adv.shape)
    samples = [{"prompt": j, "tokens": list(o.token_ids), "reward": float(rew[j, s]), "advantage": float(adv[j, s])}
               for j, out_ in enumerate(outs) for s, o in enumerate(out_.outputs)]
    start = {}
    for b in ("train", "validation", "offtask"):
        if not ids[b]:
            continue
        t1 = time.perf_counter()
        o = decode(llm, ids[b], task.params(task.blocks[b]))
        start[b] = {"rewards": score_outs(scorer, o, task.blocks[b]), "seconds": time.perf_counter() - t1,
                    "mean_len": float(np.mean([len(x.outputs[0].token_ids) for x in o])),
                    "kinds": [row_kind(r) for r in task.blocks[b]]}
    scorer.close()
    rec = {"phase": "sample", "k": k, "prompts": len(rows), "prompt_tokens": ids["train"], "samples": samples,
           "synthetic_advantages": bool(synthetic),
           "live_share": float((sd > 0).mean()), "zero_advantage_share": float((sd == 0).mean()),
           "mean_sample_reward": float(rew.mean()),
           "mean_sample_len": float(np.mean([len(s["tokens"]) for s in samples])),
           "answers_per_second": len(samples) / t_sample, "start": start,
           "start_reward": {b: float(np.mean(v["rewards"])) for b, v in start.items()},
           "seconds": time.perf_counter() - t0}
    harness.write_atomic(out / "sample.json", rec)
    print(f"live share {rec['live_share']:.2f}, start {rec['start_reward']}, "
          f"{rec['answers_per_second']:.1f} answers/s, {rec['seconds']:.0f}s", flush=True)


def phase_gradient(args, cfg, task) -> None:
    import torch  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    out = E2E / args.out
    rec = json.loads((out / "sample.json").read_text())
    model = AutoModelForCausalLM.from_pretrained(cfg["repo"], revision=cfg["revision"],
                                                 dtype=torch.float32, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    shapes = lowrank.leaf_shapes(model_dir(cfg))
    names = {n for n, _ in shapes}
    params = {n: p for n, p in model.named_parameters() if n in names}
    if len(params) != len(names):
        raise SystemExit(f"{len(params)} of {len(names)} matrix leaves found in the model")
    total = {n: torch.zeros_like(p) for n, p in params.items()}
    by_prompt = {}
    for s in rec["samples"]:
        by_prompt.setdefault(s["prompt"], []).append(s)
    norm2s, kinds, t0, micro = {}, {}, time.perf_counter(), 4
    for j, group in sorted(by_prompt.items()):
        if all(abs(s["advantage"]) < 1e-12 for s in group):
            continue                                                  # a dead prompt: no gradient
        prompt = rec["prompt_tokens"][j]
        seqs = [prompt + s["tokens"] for s in group]
        n_resp = sum(len(s["tokens"]) for s in group)
        for b0 in range(0, len(seqs), micro):
            idx = range(b0, min(len(seqs), b0 + micro))
            L = max(len(seqs[i]) for i in idx)
            x = torch.zeros((len(idx), L), dtype=torch.long)
            mask = torch.zeros((len(idx), L), dtype=torch.long)
            weight = torch.zeros((len(idx), L))                       # A_i on response positions
            for r, i in enumerate(idx):
                x[r, :len(seqs[i])], mask[r, :len(seqs[i])] = torch.tensor(seqs[i]), 1
                weight[r, len(prompt):len(seqs[i])] = group[i]["advantage"]
            x, mask, weight = x.cuda(), mask.cuda(), weight.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=x, attention_mask=mask).logits
            logp = torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, x[:, 1:, None]).squeeze(-1)
            (-(weight[:, 1:] * logp).sum() / n_resp).backward()      # the group's token mean
            del logits, logp
        n2 = 0.0
        for n, p in params.items():
            s2 = float((p.grad.double() ** 2).sum())
            n2 += s2
            kinds[lowrank.kind(n)] = kinds.get(lowrank.kind(n), 0.0) + s2
            total[n] += p.grad
        norm2s[str(j)] = n2
        model.zero_grad(set_to_none=False)
        if len(norm2s) % 32 == 0:
            print(f"{len(norm2s)} live prompts {time.perf_counter() - t0:.0f}s", flush=True)
    M = len(norm2s)
    if M == 0:
        raise SystemExit("no live prompt: the gradient is zero")
    sum_norm2 = float(sum(norm2s.values()))
    batch_norm2 = float(sum((t.double() ** 2).sum() for t in total.values()))
    # the mean of g_i . g_j over i != j, as a fraction of the mean |g_j|^2
    shared = (batch_norm2 - sum_norm2) / (M * (M - 1)) / (sum_norm2 / M) if M > 1 else None
    save_file({n: t.contiguous().cpu() for n, t in total.items()}, str(args.workdir / "grad.safetensors"))
    harness.write_atomic(out / "gradient.json", {
        "phase": "gradient", "live_prompts": M, "per_prompt_norm2": norm2s,
        "perturbed_params": int(sum(o * i for _, (o, i) in shapes)),
        "sum_norm2": sum_norm2, "batch_norm2": batch_norm2, "batch_norm": float(np.sqrt(batch_norm2)),
        "mean_gradient_norm": float(np.sqrt(batch_norm2)) / M,
        "sharedness": shared, "diversity_yin": sum_norm2 / batch_norm2 if batch_norm2 else None,
        "share_of_norm2_by_kind": {k_: v / sum_norm2 for k_, v in kinds.items()},
        "seconds": time.perf_counter() - t0})
    print(f"{M} live prompts, |sum g| {np.sqrt(batch_norm2):.4g}, sharedness {shared}, "
          f"{time.perf_counter() - t0:.0f}s", flush=True)


def phase_scan(args, cfg, task) -> None:
    from es_vllm.worker import ESWorker  # noqa: PLC0415

    out = E2E / args.out
    scorer = task.scorer()
    for k_, fn in lowrank.worker_methods().items():
        setattr(ESWorker, k_, fn)
    mdir = model_dir(cfg)
    shapes = lowrank.leaf_shapes(mdir)
    d_m = sum(o * i for _, (o, i) in shapes)
    tok = tokenizer(cfg)
    rows, kinds, which = scan_rows(task)
    ids = [task.render(tok, r) for r in rows]
    llm = engine(cfg, 0.5, worker_extension_cls="es_vllm.worker.ESWorker")
    rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
    pairs = 4
    rpc("lr_setup", mdir, SEED, RANK, pairs, shapes)
    log = out / "scan.jsonl"
    done = {(r["kind"], r["index"], r["length"], r["sign"]) for r in lines(log)}
    member_len = SIGMA * np.sqrt(d_m)
    t0 = time.perf_counter()

    def point(kind, index, length, sign, k):
        if (kind, index, length, sign) in done:
            return
        rpc("lr_load", k)
        if not rpc("lr_check")["ok"]:
            raise SystemExit("engine weights are not the point")
        o = decode(llm, ids, task.params(rows))
        append(log, {"kind": kind, "index": index, "length": length, "sign": sign, "kinds": kinds,
                     "block": which, "rewards": score_outs(scorer, o, rows),
                     "mean_len": float(np.mean([len(x.outputs[0].token_ids) for x in o]))})
        print(f"{kind} {index} length {length:.3g} sign {sign:+}: {time.perf_counter() - t0:.0f}s", flush=True)

    point("start", 0, 0.0, 0, [0.0] * pairs)
    for j in range(pairs):                              # random directions of ES's shape
        one = [1.0 if i == j else 0.0 for i in range(pairs)]
        norm = rpc("lr_norm", one)
        for mult in LENGTHS:
            for sign in (1, -1):
                point("random", j, member_len * mult, sign, [sign * member_len * mult / norm * x for x in one])
    rpc("lr_load", [0.0] * pairs)
    scorer.close()
    harness.write_atomic(out / "scan-meta.json", {"perturbed_params": d_m, "member_length": member_len,
                                                  "pairs": pairs, "blocks": {b: which.count(b) for b in set(which)}})


def phase_sign(args, cfg, task) -> None:
    from es_vllm.gradient_check import worker_methods as grad_methods  # noqa: PLC0415
    from es_vllm.worker import ESWorker  # noqa: PLC0415

    out = E2E / args.out
    scorer = task.scorer()
    for k_, fn in grad_methods().items():
        setattr(ESWorker, k_, fn)
    mdir = model_dir(cfg)
    shapes = lowrank.leaf_shapes(mdir)
    d_m = sum(o * i for _, (o, i) in shapes)
    tok = tokenizer(cfg)
    rows, kinds, which = scan_rows(task)
    ids = [task.render(tok, r) for r in rows]
    llm = engine(cfg, 0.45, worker_extension_cls="es_vllm.worker.ESWorker")
    rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
    meta = rpc("gd_setup", mdir, str(args.workdir / "grad.safetensors"), shapes)
    log = out / "sign.jsonl"
    done = {(r["nominal"], r["sign"]) for r in lines(log)}
    t0 = time.perf_counter()

    def point(nominal, sign):
        if (nominal, sign) in done:
            return
        eff = rpc("gd_load", sign * nominal, "sign")
        o = decode(llm, ids, task.params(rows))
        append(log, {"kind": "sign", "nominal": nominal, "sign": sign, "effective_length": eff, "kinds": kinds,
                     "block": which, "rewards": score_outs(scorer, o, rows),
                     "mean_len": float(np.mean([len(x.outputs[0].token_ids) for x in o]))})
        print(f"sign {nominal:.3g} sign {sign:+}: effective {eff:.3g}, {time.perf_counter() - t0:.0f}s", flush=True)

    point(0.0, 0)
    for mult in SIGN_LENGTHS:                           # the same per-coordinate displacement as on Tulu
        for sign in (1, -1):
            point(mult * float(np.sqrt(d_m / TULU_D_M)), sign)
    rpc("gd_load", 0.0, "sign")
    scorer.close()
    harness.write_atomic(out / "sign-meta.json", {**meta, "perturbed_params": d_m,
                                                  "rl_step_length": RL_LR * float(np.sqrt(d_m))})


# ------------------------------------------------------------------ level 1

def phase_members(args, cfg, task) -> None:
    from vllm.lora.request import LoRARequest  # noqa: PLC0415

    out = E2E / args.out
    scorer = task.scorer()
    shapes = lowrank.leaf_shapes(model_dir(cfg))
    n, n2 = (8, 4) if cfg["smoke"] else (N_MEMBERS, N_MEMBERS2)
    template = lowrank.write_template(Path(args.workdir) / "adapter", RANK, cfg["repo"])
    # adapter ids 1..n are members 0..n-1 at sigma; n+1..n+n2 are members 0..n2-1 at sigma2
    lowrank.install_adapters(template, SEED, shapes, RANK, SIGMA,
                             resolve=lambda i: (i - 1, 0, SIGMA) if i - 1 < n else (i - 1 - n, 0, SIGMA2))
    tok = tokenizer(cfg)
    rows = task.blocks["train"]
    ids = [task.render(tok, r) for r in rows]
    chunk = 4 if cfg["smoke"] else 32
    llm = engine(cfg, 0.6, enable_lora=True, max_lora_rank=RANK, max_loras=chunk, max_cpu_loras=n + n2)
    t0 = time.perf_counter()
    rewards, lens = [], []
    for c0 in range(0, n + n2, chunk):
        ms = list(range(c0, min(n + n2, c0 + chunk)))
        o = decode(llm, [i for _ in ms for i in ids], task.params(rows * len(ms)),
                   [LoRARequest(f"m{m}", m + 1, str(template)) for m in ms for _ in ids])
        r = score_outs(scorer, o, rows * len(ms))
        rewards += [r[q * len(ids):(q + 1) * len(ids)] for q in range(len(ms))]
        lens += [float(np.mean([len(x.outputs[0].token_ids) for x in o[q * len(ids):(q + 1) * len(ids)]]))
                 for q in range(len(ms))]
        print(f"members {c0}-{ms[-1]} {time.perf_counter() - t0:.0f}s", flush=True)
    scorer.close()
    harness.write_atomic(out / "members.json", {
        "phase": "members", "n": n, "n2": n2, "sigma": SIGMA, "sigma2": SIGMA2, "seed": SEED, "rank": RANK,
        "rewards": rewards[:n], "rewards2": rewards[n:], "mean_len": lens[:n], "mean_len2": lens[n:],
        "kinds": [row_kind(r) for r in rows], "seconds": time.perf_counter() - t0})


def phase_steps(args, cfg, task) -> None:
    from es_vllm.worker import ESWorker  # noqa: PLC0415

    out = E2E / args.out
    scorer = task.scorer()
    for k_, fn in lowrank.worker_methods().items():
        setattr(ESWorker, k_, fn)
    mem = json.loads((out / "members.json").read_text())
    n = mem["n"]
    mdir = model_dir(cfg)
    shapes = lowrank.leaf_shapes(mdir)
    d_m = sum(o * i for _, (o, i) in shapes)
    member_len = SIGMA * np.sqrt(d_m)
    fitness = np.asarray(mem["rewards"]).mean(1)
    if fitness.std() == 0 and not cfg["smoke"]:
        # every member scored alike: there is no update to scan, and the rule says so
        harness.write_atomic(out / "steps-meta.json", {"unit_norm": 0.0, "member_length": member_len,
                                                       "perturbed_params": d_m, "n": n, "no_ranking_signal": True})
        pred = predict(out, task.max_reward)
        pred["setting"] = cfg["name"]
        harness.write_atomic(out / "prediction.json", pred)
        print(json.dumps(pred["rule"], indent=1), flush=True)
        return
    if fitness.std() == 0:
        fitness = np.random.default_rng(0).standard_normal(n)        # wiring only
    k = [float(x) for x in lowrank.coefficients(fitness, n, ALPHA, RANK)]
    tok = tokenizer(cfg)
    rows, kinds, which = scan_rows(task)
    ids = [task.render(tok, r) for r in rows]
    llm = engine(cfg, 0.5, worker_extension_cls="es_vllm.worker.ESWorker")
    rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
    rpc("lr_setup", mdir, SEED, RANK, n // 2, shapes)
    unit = rpc("lr_norm", k)
    log = out / "steps.jsonl"
    done = {(r["length"], r["sign"]) for r in lines(log)}
    t0 = time.perf_counter()

    def point(length, sign):
        if (length, sign) in done:
            return
        lam = sign * length / unit
        rpc("lr_load", [lam * x for x in k])
        if not rpc("lr_check")["ok"]:
            raise SystemExit("engine weights are not the update")
        o = decode(llm, ids, task.params(rows))
        append(log, {"length": length, "sign": sign, "lam": lam, "kinds": kinds, "block": which,
                     "rewards": score_outs(scorer, o, rows),
                     "mean_len": float(np.mean([len(x.outputs[0].token_ids) for x in o]))})
        print(f"update length {length:.3g} sign {sign:+}: {time.perf_counter() - t0:.0f}s", flush=True)

    point(0.0, 0)
    for mult in LENGTHS:
        for sign in (1, -1):
            point(member_len * mult, sign)
    rpc("lr_load", [0.0] * (n // 2))
    scorer.close()
    harness.write_atomic(out / "steps-meta.json", {"unit_norm": unit, "member_length": member_len,
                                                   "perturbed_params": d_m, "n": n,
                                                   "synthetic_ranking": bool(np.asarray(mem["rewards"]).mean(1).std() == 0)})
    pred = predict(out, task.max_reward)
    pred["setting"] = cfg["name"]
    harness.write_atomic(out / "prediction.json", pred)
    print(json.dumps(pred["rule"], indent=1), flush=True)


# ------------------------------------------------------------------ the analysis and the rule

def responses(points: list, start: np.ndarray, norm: float) -> dict:
    """Odd and even responses at each length, pooled, by task and by block, per unit of
    `norm` (a unit update's length, or 1 for responses per unit length)."""
    out = {}
    for L in sorted({p["length"] for p in points if p["sign"] != 0}):
        plus = next((p for p in points if p["length"] == L and p["sign"] == 1), None)
        minus = next((p for p in points if p["length"] == L and p["sign"] == -1), None)
        if plus is None or minus is None:
            continue
        p, m = np.asarray(plus["rewards"]), np.asarray(minus["rewards"])
        odd, even = (p - m) / 2, (p + m) / 2 - start
        lam = L / norm
        kinds, blocks = np.asarray(plus["kinds"]), np.asarray(plus["block"])
        out[f"{L:.6g}"] = {
            "lam": lam, "odd": float(odd.mean()), "odd_se": float(odd.std(ddof=1) / np.sqrt(odd.size)),
            "even": float(even.mean()), "even_se": float(even.std(ddof=1) / np.sqrt(even.size)),
            "G": float(odd.mean() / lam), "C": float(even.mean() / lam ** 2),
            "plus": float(p.mean()), "minus": float(m.mean()),
            "usable": bool(abs(p.mean() - start.mean()) < USABLE and abs(m.mean() - start.mean()) < USABLE),
            "by_task": {str(t): {"G": float(odd[kinds == t].mean() / lam), "C": float(even[kinds == t].mean() / lam ** 2),
                                 "odd": float(odd[kinds == t].mean()), "even": float(even[kinds == t].mean())}
                        for t in sorted(set(kinds))},
            "by_block": {str(b): {"G": float(odd[blocks == b].mean() / lam), "C": float(even[blocks == b].mean() / lam ** 2)}
                         for b in sorted(set(blocks))}}
    return out


def random_responses(scan: list, start: np.ndarray) -> dict:
    """The random directions' responses per unit length (odd) and length squared (even),
    averaged over directions, with the spread over directions, by task and block."""
    rnd = [p for p in scan if p["kind"] == "random"]
    out = {}
    for L in sorted({p["length"] for p in rnd}):
        per_dir = []
        for j in sorted({p["index"] for p in rnd if p["length"] == L}):
            r = responses([p for p in rnd if p["index"] == j and p["length"] == L], start, 1.0)
            if r:
                per_dir.append(r[f"{L:.6g}"])
        if not per_dir:
            continue
        mean = lambda key: float(np.mean([v[key] for v in per_dir]))  # noqa: E731
        out[f"{L:.6g}"] = {
            "directions": len(per_dir), "odd_per_length": mean("G"), "even_per_length2": mean("C"),
            "even_per_length2_sd_over_directions": float(np.std([v["C"] for v in per_dir])),
            "even": mean("even"), "even_se": float(np.sqrt(np.mean([v["even_se"] ** 2 for v in per_dir]) / len(per_dir))),
            "usable": all(v["usable"] for v in per_dir),
            "by_task": {t: {"even_per_length2": float(np.mean([v["by_task"][t]["C"] for v in per_dir])),
                            "odd_per_length": float(np.mean([v["by_task"][t]["G"] for v in per_dir]))}
                        for t in per_dir[0]["by_task"]},
            "by_block": {b: {"even_per_length2": float(np.mean([v["by_block"][b]["C"] for v in per_dir]))}
                         for b in per_dir[0]["by_block"]}}
    return out


def sign_responses(points: list, start: np.ndarray) -> dict:
    """Per nominal length: the odd response per unit effective length along +sign(g) (g the
    loss gradient, so RL's direction is its negative) and the even response."""
    out = {}
    for L in sorted({p["nominal"] for p in points if p["sign"] != 0}):
        plus = next((p for p in points if p["nominal"] == L and p["sign"] == 1), None)
        minus = next((p for p in points if p["nominal"] == L and p["sign"] == -1), None)
        if plus is None or minus is None:
            continue
        p, m = np.asarray(plus["rewards"]), np.asarray(minus["rewards"])
        eff = (plus["effective_length"] + minus["effective_length"]) / 2
        odd, even = (p - m) / 2, (p + m) / 2 - start
        kinds = np.asarray(plus["kinds"])
        out[f"{L:.6g}"] = {
            "effective_length": eff, "odd": float(odd.mean()), "odd_se": float(odd.std(ddof=1) / np.sqrt(odd.size)),
            "even": float(even.mean()), "even_se": float(even.std(ddof=1) / np.sqrt(even.size)),
            "slope_along_rl_direction": float(-odd.mean() / eff),
            "slope_se": float(odd.std(ddof=1) / np.sqrt(odd.size) / eff),
            "even_per_effective_length2": float(even.mean() / eff ** 2),
            "usable": bool(abs(p.mean() - start.mean()) < USABLE and abs(m.mean() - start.mean()) < USABLE),
            "by_task": {str(t): {"slope_along_rl_direction": float(-odd[kinds == t].mean() / eff)} for t in sorted(set(kinds))}}
    return out


def rule(G: float, C: float, rl_gain_per_step: float, P: int, N: int, live_share: float) -> dict:
    """docs/end_to_end/10's rule in normalized reward units: `G` and `C` are the unit
    update's odd response and even response per unit squared at `N` members on `P`
    prompts; `rl_gain_per_step` RL's expected first-step gain at the reference step."""
    es = rl = ratio = None
    if G > 0 and C < 0:
        # T updates at step lambda: T lambda G = G_target and T lambda^2 |C| = damage, so
        # T = G_target^2 |C| / (damage G^2); |C| is proportional to 1 / N, so N T is not
        lam = DAMAGE * G / (G_TARGET * -C)
        T = G_TARGET / (lam * G)
        es = {"step_multiple": lam, "updates": T, "rollouts": P * N * T, "fitted_net_per_update": G ** 2 / (4 * -C),
              "n_for_rl_step_gain": N * rl_gain_per_step * 4 * -C / G ** 2 if rl_gain_per_step > 0 else None}
    if rl_gain_per_step > 0:
        rl = {"steps": G_TARGET / rl_gain_per_step, "rollouts": G_TARGET / rl_gain_per_step * RL_ROLLOUTS_PER_STEP}
    if es and rl:
        ratio = es["rollouts"] / rl["rollouts"]
    if live_share < LIVE_MIN:
        call = "fails: no signal"
    elif G <= 0:
        call = "fails: no odd response"
    elif C >= 0:
        call = "undetermined: no resolved cost"
    elif ratio is None:
        call = "undetermined: RL's gain unresolved"
    elif ratio <= WORKS:
        call = "works"
    elif ratio <= MARGINAL:
        call = "marginal"
    else:
        call = "fails: rollouts"
    return {"es": es, "rl": rl, "ratio_es_over_rl_rollouts": ratio, "call": call,
            "constants": {"G_target": G_TARGET, "damage": DAMAGE, "works_below": WORKS, "marginal_below": MARGINAL,
                          "live_min": LIVE_MIN, "rl_lr": RL_LR, "rl_rollouts_per_step": RL_ROLLOUTS_PER_STEP}}


def level1(mem: dict, start_train: np.ndarray, max_reward: float) -> dict:
    """The ranking's reliability and the cross-prompt correlation of member effects, as
    grad_check.py and countdown_grad_check.py define them, from greedy decodes."""
    from es_vllm.grad_check import split_reliability, variance_parts  # noqa: PLC0415

    out = {}
    for key, rewards, sigma in (("sigma", mem["rewards"], mem["sigma"]), ("sigma2", mem["rewards2"], mem["sigma2"])):
        if not rewards:
            continue
        delta = np.asarray(rewards) / max_reward - start_train[None]
        parts = variance_parts(delta)
        rng = np.random.default_rng(0)
        cov = []
        for _ in range(1000):
            perm = rng.permutation(delta.shape[1])
            a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
            cov.append(np.cov(delta[:, a].mean(1), delta[:, b].mean(1))[0, 1])
        per_prompt = float(delta.var(axis=0, ddof=1).mean())
        fitness = delta.mean(1)
        out[key] = {"sigma": sigma, "n": delta.shape[0], "prompts": delta.shape[1],
                    "member_mean_change": float(delta.mean()), "member_sd": float(fitness.std(ddof=1)),
                    "pairs_changed": float((delta != 0).mean()),
                    "true_effect_sd": float(np.sqrt(parts["v_member"])), "residual": parts["v_resid"],
                    "reliability_at_prompts": parts["v_member"] / (parts["v_member"] + parts["v_resid"] / delta.shape[1])
                    if parts["v_member"] + parts["v_resid"] > 0 else 0.0,
                    "split_half_reliability": split_reliability(delta, 2, rng),
                    "shared_covariance": float(np.mean(cov)),
                    "cross_prompt_correlation": float(np.mean(cov) / per_prompt) if per_prompt > 0 else None,
                    "best_of_n_improvement": {str(q): float(np.sort(fitness[:q])[-1]) for q in (8, 16, 32, 64) if q <= len(fitness)}}
    return out


def predict(out: Path, max_reward: float) -> dict:
    sample = json.loads((out / "sample.json").read_text())
    grad = json.loads((out / "gradient.json").read_text()) if (out / "gradient.json").exists() else {}
    mem = json.loads((out / "members.json").read_text())
    smeta = json.loads((out / "steps-meta.json").read_text())
    norm = lambda rows: np.asarray(rows, dtype=float) / max_reward  # noqa: E731

    def normalized(points):
        return [{**p, "rewards": norm(p["rewards"]).tolist()} for p in points]

    scan, sign, steps = (normalized(lines(out / f)) for f in ("scan.jsonl", "sign.jsonl", "steps.jsonl"))
    d_m = smeta["perturbed_params"]
    rl_step = RL_LR * float(np.sqrt(d_m))

    # level 0 (the points are normalized already, their starts included)
    rnd = random_responses(scan, np.asarray(next(p for p in scan if p["kind"] == "start")["rewards"])) if scan else {}
    rnd_used = [L for L, v in rnd.items() if v["usable"]] or list(rnd)
    rnd_far = rnd[max(rnd_used, key=float)] if rnd else None
    sgn = sign_responses(sign, np.asarray(next(p for p in sign if p["sign"] == 0)["rewards"])) if sign else {}
    sgn_used = [L for L, v in sgn.items() if v["usable"]] or list(sgn)
    sgn_near = sgn[min(sgn_used, key=float)] if sgn else None
    rl_gain = rl_step * sgn_near["slope_along_rl_direction"] if sgn_near else 0.0
    level0 = {"live_share": sample["live_share"], "zero_advantage_share": sample["zero_advantage_share"],
              "start_reward_normalized": {b: v / max_reward for b, v in sample["start_reward"].items()},
              "answers_per_second": sample["answers_per_second"], "mean_sample_len": sample["mean_sample_len"],
              "live_prompts_in_gradient": grad.get("live_prompts"), "gradient_sharedness": grad.get("sharedness"),
              "gradient_diversity_yin": grad.get("diversity_yin"), "gradient_norm": grad.get("batch_norm"),
              "perturbed_params": d_m, "random_direction_responses": rnd,
              "curvature_per_length2": rnd_far["even_per_length2"] if rnd_far else None,
              "first_task_damaged_random": min(rnd_far["by_task"], key=lambda t: rnd_far["by_task"][t]["even_per_length2"])
              if rnd_far else None,
              "sign_direction_responses": sgn, "rl_step_length": rl_step,
              "rl_slope_along_direction": sgn_near["slope_along_rl_direction"] if sgn_near else None,
              "rl_first_step_gain": rl_gain}

    # level 1
    start_train = norm(sample["start"]["train"]["rewards"])
    lvl1 = level1(mem, start_train, max_reward)
    if steps and smeta["unit_norm"] > 0:
        start_steps = np.asarray(next(p for p in steps if p["sign"] == 0)["rewards"])
        resp = responses([p for p in steps if p["sign"] != 0], start_steps, smeta["unit_norm"])
        used = [L for L, v in resp.items() if v["usable"]] or list(resp)
        far = resp[max(used, key=float)]
        update = {"unit_norm": smeta["unit_norm"], "n": mem["n"], "responses": resp, "length_used": max(used, key=float),
                  "usable": far["usable"], "G": far["G"], "G_se": far["odd_se"] / far["lam"], "C": far["C"],
                  "C_se": far["even_se"] / far["lam"] ** 2,
                  "first_task_damaged": min(far["by_task"], key=lambda t: far["by_task"][t]["C"]),
                  "random_cost_at_this_N_per_length2": far["C"] / smeta["unit_norm"] ** 2}
        G, C = far["G"], far["C"]
    else:                                    # every member scored alike: no update to scan
        update, G, C = None, 0.0, 0.0
    verdict = rule(G, C, rl_gain, sample["prompts"], mem["n"], sample["live_share"])
    return {"setting": None, "level0": level0, "level1": lvl1, "update": update, "rule": verdict}


def analyze(out: Path) -> dict:
    pred = json.loads((out / "prediction.json").read_text())
    l0, u, r = pred["level0"], pred["update"] or {}, pred["rule"]
    return {"setting": pred["setting"], "call": r["call"], "ratio_es_over_rl_rollouts": r["ratio_es_over_rl_rollouts"],
            "live_share": l0["live_share"], "gradient_sharedness": l0["gradient_sharedness"],
            "curvature_per_length2": l0["curvature_per_length2"], "rl_first_step_gain": l0["rl_first_step_gain"],
            "first_task_damaged_random": l0["first_task_damaged_random"],
            "reliability": {k: v["reliability_at_prompts"] for k, v in pred["level1"].items()},
            "cross_prompt_correlation": {k: v["cross_prompt_correlation"] for k, v in pred["level1"].items()},
            "G": u.get("G"), "C": u.get("C"), "fitted_net_per_update": r["es"] and r["es"]["fitted_net_per_update"],
            "first_task_damaged": u.get("first_task_damaged"), "es": r["es"], "rl": r["rl"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", help="model/task, e.g. qwen3b/countdown")
    ap.add_argument("--phase", choices=("sample", "gradient", "scan", "sign", "members", "steps"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analyze", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/probe"))
    ap.add_argument("--es-at-scale", type=Path, default=Path.home() / "private" / "open-source" / "es-at-scale")
    args = ap.parse_args(argv)
    if args.analyze:
        print(json.dumps(analyze(E2E / args.analyze), indent=2))
        return 0
    if args.out is None or args.phase is None or args.setting is None:
        ap.error("--setting, --phase and --out, or --analyze")
    cfg = setting(args.setting, args.smoke)
    out = E2E / args.out
    out.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)
    task = Task(cfg["task"], cfg["template"], args.smoke, args.es_at_scale)
    env = out / "env.json"
    if not env.exists():
        env.write_text(json.dumps({
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "setting": args.setting, "repo": cfg["repo"], "revision": cfg["revision"], "smoke": args.smoke,
            "blocks": {b: len(v) for b, v in task.blocks.items()}, "block_digests": task.digests(),
            "constants": {"sigma": SIGMA, "sigma2": SIGMA2, "alpha": ALPHA, "rank": RANK, "k": K, "members": N_MEMBERS,
                          "members2": N_MEMBERS2, "lengths": LENGTHS, "sign_lengths": SIGN_LENGTHS, "rl_lr": RL_LR},
            "env": env_block(E2E, ["runs"], ("vllm", "torch", "transformers"))}, indent=2))
    {"sample": phase_sample, "gradient": phase_gradient, "scan": phase_scan, "sign": phase_sign,
     "members": phase_members, "steps": phase_steps}[args.phase](args, cfg, task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
