#!/usr/bin/env python
"""The a-priori probe of docs/end_to_end/10: level 0 and level 1 on one model and task.

    python -m es_vllm.probe --setting qwen3b/countdown --phase sample   --out runs/probe/qwen3b-countdown
    ... --phase gradient / scan / sign / members / steps (the last writes prediction.json)
    ... --phase members-offtask (after members; rewrites prediction.json if steps ran)
    python -m es_vllm.probe --analyze runs/probe/qwen3b-countdown
    (--smoke: tiny blocks and populations, wiring only; --seed: another member population;
    --block 600: validation and off-task blocks of 600 prompts, the default block first and
    the test block unchanged, so the records compare with a default probe's)

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
- `members-offtask`: the same members decode the validation and offtask blocks: per
  mirrored pair, the odd effect on each block beside the train one, so the update's
  first-order change on each block is decomposed over the pairs and the correlation
  across pairs between train and off-task effects says whether the directions that rank
  well are the ones that move the off-task block (`level1_offtask`).
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
MAX_LENGTH, RESOLVED = 40.0, 3.0           # the scans double the length past 2.5 until the even
                                           # response is this many standard errors from zero
SIGN_LENGTHS = (0.5, 1.0)                  # Tulu units; scaled by sqrt(d / d_tulu)
TULU_D_M = 6_979_321_856                   # runs/lowrank-check/members.json perturbed_params
RL_LR, RL_ROLLOUTS_PER_STEP = 5e-7, R.TULU31_ROLLOUTS_PER_STEP   # the Tulu run: 0.045 / sqrt(8.03e9)
G_TARGET, DAMAGE = 0.1, 0.02               # of the reward range
WORKS, MARGINAL, LIVE_MIN = 5.0, 25.0, 0.1  # runs/throughput-bench: 0.255 / 0.048 per answer
USABLE = 0.2                               # a scan point is usable if neither sign moved more
CAPS = {"countdown": 512, "tulu": 2048}    # es_vllm/countdown-s1.yaml; the Tulu run
SCALE = {"countdown": 1.1}                 # a row kind's maximum reward; the Tulu verifiers give 10


def scale_of(kinds) -> np.ndarray:
    return np.asarray([SCALE.get(k, 10.0) for k in kinds])


def normalized(rewards, kinds) -> np.ndarray:
    """Rewards in [0, 1], each row by its own kind's maximum: a Countdown setting's
    off-task block is scored by the Tulu verifiers on the 10 scale."""
    return np.asarray(rewards, dtype=float) / scale_of(kinds)

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
            "template": template, "task": task, "smoke": smoke, "seed": SEED, "block": None}


def model_dir(cfg) -> str:
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    return snapshot_download(cfg["repo"], revision=cfg["revision"], allow_patterns=["*.safetensors", "*.json"])


# ------------------------------------------------------------------ tasks

def is_countdown(row) -> bool:
    return "context" in row


def row_kind(row) -> str:
    """The task family a row belongs to, for the by-task responses."""
    return "countdown" if is_countdown(row) else row["dataset"]


def digest(rows) -> str:
    return hashlib.sha256("".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows).encode()).hexdigest()


class Task:
    """Prompts in blocks, a renderer and a scorer for one task. `block` is the size of
    the validation and off-task blocks (BLOCK by default); a larger block keeps the
    default block as its first rows and leaves the train and test blocks as they are, so
    its records compare with a default probe's and a run against it reserves the same
    test prompts."""

    def __init__(self, name: str, template: str, smoke: bool, es_at_scale: Path, block=None):
        from es_vllm.run_tulu import heldout_rows, load_prompts  # noqa: PLC0415

        self.name, self.template, self.smoke = name, template, smoke
        n_train, n_test = (16, 24) if smoke else (P_TRAIN, BLOCK)
        n_val = n_test if block is None else int(block)
        if n_val < n_test:
            raise SystemExit(f"block {n_val} is smaller than the test block ({n_test})")
        extra = n_val - n_test                 # rows after the test block, so the test block stays put
        if name == "countdown":
            from es_vllm.run_countdown import es_at_scale as pin, rows  # noqa: PLC0415

            self.clone = pin(es_at_scale)
            train, ev = rows(self.clone, "train"), rows(self.clone, "eval")
            gsm = [r for r in heldout_rows(0, 80) if r["dataset"] == "gsm8k"]
            self.blocks = {"train": train[:n_train], "validation": ev[:n_test] + ev[2 * n_test:2 * n_test + extra],
                           "test": ev[n_test:2 * n_test], "offtask": gsm[:n_val]}
        elif name == "gsm8k":
            pool = heldout_rows(0, 160)
            gsm = [r for r in pool if r["dataset"] == "gsm8k"]
            math = [r for r in pool if r["dataset"] == "MATH"]
            ifeval = [r for r in pool if r["dataset"] == "ifeval"]
            v0, t0 = n_train, n_train + n_test
            self.blocks = {"train": gsm[:n_train], "validation": gsm[v0:t0] + gsm[t0 + n_test:t0 + n_test + extra],
                           "test": gsm[t0:t0 + n_test],
                           "offtask": math[:n_test // 2] + ifeval[:n_test // 2]
                           + math[n_test // 2:n_val // 2] + ifeval[n_test // 2:n_val // 2]}
        elif name == "tulu":
            self.blocks = {"train": load_prompts(E2E / "data" / "tulu31", 1, P_TRAIN)[0][:n_train],
                           "validation": heldout_rows(120, 160)[:n_val],
                           "test": heldout_rows(480, 520)[:n_test], "offtask": []}
        else:
            raise SystemExit(f"unknown task {name}")
        self.n_test = n_test
        for b, rows_ in self.blocks.items():
            want = {"train": n_train, "test": n_test, "offtask": 0 if name == "tulu" else n_val}.get(b, n_val)
            if len(rows_) != want:
                raise SystemExit(f"{name} {b}: {len(rows_)} rows, wanted {want}")
            if len({digest([r]) for r in rows_}) != len(rows_):
                raise SystemExit(f"{name} {b}: repeated rows")

    def digests(self) -> dict:
        """One hash per block, so a later run can show it used the same prompts."""
        return {b: digest(rows_) for b, rows_ in self.blocks.items()}

    def prefix_digests(self) -> dict:
        """For a block larger than the default, the hash of its first default-sized
        rows: the block a default probe of the setting scored."""
        return {b: digest(rows_[:self.n_test]) for b, rows_ in self.blocks.items() if len(rows_) > self.n_test}

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
            weight = torch.zeros((len(idx), L))                       # A_i on response positions
            for r, i in enumerate(idx):
                x[r, :len(seqs[i])] = torch.tensor(seqs[i])
                weight[r, len(prompt):len(seqs[i])] = group[i]["advantage"]
            x, weight = x.cuda(), weight.cuda()
            # Right padding under causal attention and no padding mask: a real token never
            # sees a pad, and the pads' terms have weight 0. A padding mask leaves sdpa's
            # bf16 backward with fully masked rows, which gave NaN on 5 of 156 prompts.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=x).logits
            logp = torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, x[:, 1:, None]).squeeze(-1)
            loss = -(weight[:, 1:] * logp).sum() / n_resp                # the group's token mean
            if not torch.isfinite(loss):
                raise SystemExit(f"prompt {j}: the loss is not finite")
            loss.backward()
            del logits, logp
        n2 = 0.0
        for n, p in params.items():
            s2 = float((p.grad.double() ** 2).sum())
            n2 += s2
            kinds[lowrank.kind(n)] = kinds.get(lowrank.kind(n), 0.0) + s2
            total[n] += p.grad
        if not np.isfinite(n2):
            raise SystemExit(f"prompt {j}: the gradient is not finite")
        norm2s[str(j)] = n2
        model.zero_grad(set_to_none=False)
        if len(norm2s) % 32 == 0:
            print(f"{len(norm2s)} live prompts {time.perf_counter() - t0:.0f}s", flush=True)
    M = len(norm2s)
    if M == 0:
        raise SystemExit("no live prompt: the gradient is zero")
    sum_norm2 = float(sum(norm2s.values()))
    batch_norm2 = float(sum((t.double() ** 2).sum() for t in total.values()))
    if not np.isfinite(batch_norm2):
        raise SystemExit("the summed gradient is not finite")
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
    rpc("lr_setup", mdir, cfg["seed"], RANK, pairs, shapes)
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
    norms = {j: rpc("lr_norm", [1.0 if i == j else 0.0 for i in range(pairs)]) for j in range(pairs)}

    def length(mult):                                   # random directions of ES's shape
        for j in range(pairs):
            one = [1.0 if i == j else 0.0 for i in range(pairs)]
            for sign in (1, -1):
                point("random", j, member_len * mult, sign, [sign * member_len * mult / norms[j] * x for x in one])

    for mult in LENGTHS:
        length(mult)
    # past the fixed lengths, double until the even response is resolved (first batch: the
    # cost was within noise at 2.5 member lengths on four of six settings)
    mult = LENGTHS[-1] * 2
    while mult <= MAX_LENGTH:
        pts = lines(log)
        start = np.asarray(next(p for p in pts if p["kind"] == "start")["rewards"])
        scale = scale_of(pts[0]["kinds"])
        rnd = random_responses([{**p, "rewards": (np.asarray(p["rewards"]) / scale).tolist()} for p in pts], start / scale)
        if resolved(rnd):
            break
        length(mult)
        mult *= 2
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
    lowrank.install_adapters(template, cfg["seed"], shapes, RANK, SIGMA,
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
        "phase": "members", "n": n, "n2": n2, "sigma": SIGMA, "sigma2": SIGMA2, "seed": cfg["seed"], "rank": RANK,
        "rewards": rewards[:n], "rewards2": rewards[n:], "mean_len": lens[:n], "mean_len2": lens[n:],
        "kinds": [row_kind(r) for r in rows], "seconds": time.perf_counter() - t0})


def phase_members_offtask(args, cfg, task) -> None:
    """The `members` populations decoding the validation and off-task blocks greedily, and
    the start in the same process: per member, its effect on each block beside its train
    fitness, for the covariance diagnostic (docs/end_to_end/10, corrections item 2). Written
    a chunk at a time so a relaunch resumes; the record is assembled at the end and the
    prediction rewritten with the diagnostic."""
    from vllm.lora.request import LoRARequest  # noqa: PLC0415

    out = E2E / args.out
    mem = json.loads((out / "members.json").read_text())
    n, n2 = mem["n"], mem["n2"]
    if mem["seed"] != cfg["seed"] or mem["sigma"] != SIGMA or mem["sigma2"] != SIGMA2:
        raise SystemExit("members.json is another population")
    scorer = task.scorer()
    shapes = lowrank.leaf_shapes(model_dir(cfg))
    template = lowrank.write_template(Path(args.workdir) / "adapter", RANK, cfg["repo"])
    lowrank.install_adapters(template, cfg["seed"], shapes, RANK, SIGMA,
                             resolve=lambda i: (i - 1, 0, SIGMA) if i - 1 < n else (i - 1 - n, 0, SIGMA2))
    tok = tokenizer(cfg)
    rows, kinds, which = scan_rows(task)
    ids = [task.render(tok, r) for r in rows]
    chunk = 4 if cfg["smoke"] else 32
    llm = engine(cfg, 0.6, enable_lora=True, max_lora_rank=RANK, max_loras=chunk, max_cpu_loras=n + n2)
    log = out / "members-offtask.jsonl"
    done = {m for rec in lines(log) for m in rec["members"]}
    blocks = sorted(set(which))
    t0 = time.perf_counter()

    def per_block(outs):
        return {b: float(np.mean([len(x.outputs[0].token_ids) for x, w in zip(outs, which) if w == b])) for b in blocks}

    if -1 not in done:                                  # the start, member -1: no adapter
        o = decode(llm, ids, task.params(rows))
        append(log, {"members": [-1], "rewards": [score_outs(scorer, o, rows)], "mean_len": [per_block(o)],
                     "seconds": time.perf_counter() - t0})
        print(f"start {time.perf_counter() - t0:.0f}s", flush=True)
    for c0 in range(0, n + n2, chunk):
        ms = [m for m in range(c0, min(n + n2, c0 + chunk)) if m not in done]
        if not ms:
            continue
        t1 = time.perf_counter()
        o = decode(llm, [i for _ in ms for i in ids], task.params(rows * len(ms)),
                   [LoRARequest(f"m{m}", m + 1, str(template)) for m in ms for _ in ids])
        r = score_outs(scorer, o, rows * len(ms))
        append(log, {"members": ms, "rewards": [r[q * len(ids):(q + 1) * len(ids)] for q in range(len(ms))],
                     "mean_len": [per_block(o[q * len(ids):(q + 1) * len(ids)]) for q in range(len(ms))],
                     "seconds": time.perf_counter() - t1})
        print(f"members {ms[0]}-{ms[-1]} {time.perf_counter() - t0:.0f}s", flush=True)
    scorer.close()
    by = {m: (rw, ln) for rec in lines(log) for m, rw, ln in zip(rec["members"], rec["rewards"], rec["mean_len"])}
    if set(by) != set(range(-1, n + n2)):
        raise SystemExit(f"{len(by)} of {n + n2 + 1} decodes in {log}")
    harness.write_atomic(out / "members-offtask.json", {
        "phase": "members-offtask", "n": n, "n2": n2, "sigma": SIGMA, "sigma2": SIGMA2, "seed": cfg["seed"], "rank": RANK,
        "kinds": kinds, "block": which, "start": by[-1][0], "start_mean_len": by[-1][1],
        "rewards": [by[m][0] for m in range(n)], "rewards2": [by[m][0] for m in range(n, n + n2)],
        "mean_len": [by[m][1] for m in range(n)], "mean_len2": [by[m][1] for m in range(n, n + n2)],
        "seconds": sum(rec["seconds"] for rec in lines(log))})
    if (out / "steps-meta.json").exists():
        write_prediction(out, cfg)


def write_prediction(out: Path, cfg: dict) -> dict:
    pred = predict(out)
    pred["setting"] = cfg["name"]
    harness.write_atomic(out / "prediction.json", pred)
    print(json.dumps(pred["rule"], indent=1), flush=True)
    return pred


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
        write_prediction(out, cfg)
        return
    if fitness.std() == 0:
        fitness = np.random.default_rng(0).standard_normal(n)        # wiring only
    k = [float(x) for x in lowrank.coefficients(fitness, n, ALPHA, RANK)]
    tok = tokenizer(cfg)
    rows, kinds, which = scan_rows(task)
    ids = [task.render(tok, r) for r in rows]
    llm = engine(cfg, 0.5, worker_extension_cls="es_vllm.worker.ESWorker")
    rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
    rpc("lr_setup", mdir, cfg["seed"], RANK, n // 2, shapes)
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
    mult = LENGTHS[-1] * 2                              # as the random scan: until the cost is resolved
    while mult <= MAX_LENGTH:
        pts = lines(log)
        scale = scale_of(pts[0]["kinds"])
        norm_pts = [{**p, "rewards": (np.asarray(p["rewards"]) / scale).tolist()} for p in pts]
        resp = responses([p for p in norm_pts if p["sign"] != 0],
                         np.asarray(next(p for p in norm_pts if p["sign"] == 0)["rewards"]), unit)
        if resolved(resp):
            break
        for sign in (1, -1):
            point(member_len * mult, sign)
        mult *= 2
    # the prescribed step, evaluated directly (rule_net's, from the points so far)
    harness.write_atomic(out / "steps-meta.json", {"unit_norm": unit, "member_length": member_len,
                                                   "perturbed_params": d_m, "n": n})
    pre = (predict(out).get("rule_net") or {}).get("es")
    if pre:
        L = pre["step_multiple"] * unit
        if all(abs(p["length"] - L) > 0.1 * L for p in lines(log) if p["sign"] != 0):
            for sign in (1, -1):
                point(L, sign)
    rpc("lr_load", [0.0] * (n // 2))
    scorer.close()
    harness.write_atomic(out / "steps-meta.json", {"unit_norm": unit, "member_length": member_len,
                                                   "perturbed_params": d_m, "n": n,
                                                   "synthetic_ranking": bool(np.asarray(mem["rewards"]).mean(1).std() == 0)})
    write_prediction(out, cfg)


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
        within = lambda sel: bool(abs(p[sel].mean() - start[sel].mean()) < USABLE  # noqa: E731
                                  and abs(m[sel].mean() - start[sel].mean()) < USABLE)
        out[f"{L:.6g}"] = {
            "lam": lam, "odd": float(odd.mean()), "odd_se": float(odd.std(ddof=1) / np.sqrt(odd.size)),
            "even": float(even.mean()), "even_se": float(even.std(ddof=1) / np.sqrt(even.size)),
            "G": float(odd.mean() / lam), "C": float(even.mean() / lam ** 2),
            "plus": float(p.mean()), "minus": float(m.mean()),
            "usable": bool(abs(p.mean() - start.mean()) < USABLE and abs(m.mean() - start.mean()) < USABLE),
            # every block within the band on its own: pooled responses can hide opposite moves
            "usable_by_block": all(within(blocks == b) for b in sorted(set(blocks))),
            "by_task": {str(t): {"G": float(odd[kinds == t].mean() / lam), "C": float(even[kinds == t].mean() / lam ** 2),
                                 "odd": float(odd[kinds == t].mean()), "even": float(even[kinds == t].mean())}
                        for t in sorted(set(kinds))},
            "by_block": {str(b): {"G": float(odd[blocks == b].mean() / lam), "C": float(even[blocks == b].mean() / lam ** 2),
                                  "G_se": float(odd[blocks == b].std(ddof=1) / np.sqrt((blocks == b).sum()) / lam),
                                  "C_se": float(even[blocks == b].std(ddof=1) / np.sqrt((blocks == b).sum()) / lam ** 2),
                                  "odd": float(odd[blocks == b].mean()), "even": float(even[blocks == b].mean())}
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
            "usable_by_block": all(v["usable_by_block"] for v in per_dir),
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
        kinds, blocks = np.asarray(plus["kinds"]), np.asarray(plus["block"])
        slope = lambda sel: {"slope_along_rl_direction": float(-odd[sel].mean() / eff),  # noqa: E731
                             "slope_se": float(odd[sel].std(ddof=1) / np.sqrt(sel.sum()) / eff),
                             "even_per_effective_length2": float(even[sel].mean() / eff ** 2)}
        out[f"{L:.6g}"] = {
            "effective_length": eff, "odd": float(odd.mean()), "odd_se": float(odd.std(ddof=1) / np.sqrt(odd.size)),
            "even": float(even.mean()), "even_se": float(even.std(ddof=1) / np.sqrt(even.size)),
            "slope_along_rl_direction": float(-odd.mean() / eff),
            "slope_se": float(odd.std(ddof=1) / np.sqrt(odd.size) / eff),
            "even_per_effective_length2": float(even.mean() / eff ** 2),
            "usable": bool(abs(p.mean() - start.mean()) < USABLE and abs(m.mean() - start.mean()) < USABLE),
            "by_task": {str(t): slope(kinds == t) for t in sorted(set(kinds))},
            "by_block": {str(b): slope(blocks == b) for b in sorted(set(blocks))}}
    return out


def resolved(resp: dict) -> bool:
    """Whether a scan can stop: at the longest usable length the even response is RESOLVED
    standard errors from zero, or the longest length is already past the usable band."""
    if not resp:
        return False
    longest = resp[max(resp, key=float)]
    if not longest.get("usable_by_block", longest["usable"]):
        return True
    return abs(longest["even"]) >= RESOLVED * longest["even_se"]


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


RESOLVED_CALL = 2.0     # a response within this many standard errors of zero leaves a call undetermined


def rule_net(blocks: dict, rl_gain_per_step: float, P: int, N: int, live_share: float, lam_max: float) -> dict:
    """The corrected target (2026-10-13, after the first training run and the review of
    the project map): a net gain of G_TARGET on the validation block, each retained block
    losing at most DAMAGE from the random part, each block from its own odd and even
    responses per unit update (`blocks[b]` holds `G`, `G_se`, `C`, `C_se`). The step is
    searched up to `lam_max`, the longest usable multiple of the unit update scanned,
    rather than solved from the quadratic, and the update count is for the net gain.
    A retained block's odd response is a single draw of the update's random part and is
    reported, not used (see docs/end_to_end/10)."""
    v = blocks["validation"]
    G, C = v["G"], v["C"]
    others = {b: x for b, x in blocks.items() if b != "validation"}
    best = None
    for lam in np.linspace(lam_max / 400, lam_max, 400):
        net = lam * G + lam ** 2 * C
        if net <= 0:
            continue
        T = G_TARGET / net
        losses = {b: T * lam ** 2 * x["C"] for b, x in others.items()}
        cand = {"step_multiple": float(lam), "updates": float(T), "rollouts": float(P * N * T),
                "net_per_update": float(net), "retained_block_random_loss": losses,
                "retained_block_single_draw_odd_change": {b: T * lam * x["G"] for b, x in others.items()},
                "feasible": all(loss >= -DAMAGE for loss in losses.values())}
        if best is None or (cand["feasible"] and not best["feasible"]) or (cand["feasible"] == best["feasible"] and T < best["updates"]):
            best = cand
    rl = {"steps": G_TARGET / rl_gain_per_step, "rollouts": G_TARGET / rl_gain_per_step * RL_ROLLOUTS_PER_STEP} if rl_gain_per_step > 0 else None
    ratio = best["rollouts"] / rl["rollouts"] if best and rl else None
    if live_share < LIVE_MIN:
        call = "fails: no signal"
    elif G < RESOLVED_CALL * v["G_se"]:
        call = "fails: no odd response" if G <= -RESOLVED_CALL * v["G_se"] else "undetermined: odd response unresolved"
    elif -C < RESOLVED_CALL * v["C_se"]:
        call = "undetermined: cost unresolved"
    elif best is None:
        call = "fails: no net gain in the scanned range"
    elif not best["feasible"]:
        call = "fails: retained-task loss"
    elif rl is None:
        call = "undetermined: RL's gain unresolved"
    elif ratio <= WORKS:
        call = "works"
    elif ratio <= MARGINAL:
        call = "marginal"
    else:
        call = "fails: rollouts"
    return {"es": best, "rl": rl, "ratio_es_over_rl_rollouts": ratio, "call": call, "lam_max": lam_max,
            "constants": {"G_target_net": G_TARGET, "damage_per_retained_block": DAMAGE, "works_below": WORKS,
                          "marginal_below": MARGINAL, "live_min": LIVE_MIN, "resolved_call_se": RESOLVED_CALL,
                          "rl_lr": RL_LR, "rl_rollouts_per_step": RL_ROLLOUTS_PER_STEP}}


def rule_curve(resp: dict, rl_gain_per_step: float, P: int, N: int) -> dict:
    """Exploratory (2026-10-15, after the second probe of 3B instruct Countdown and S4a;
    frozen for S4b only after review): the prescription from the measured response curves
    rather than from `G` and `C` read at one length. The odd response is concave in the
    step from the smallest length scanned (3B instruct Countdown: +0.024, +0.032, +0.046
    at 3, 8 and 20 units), so a per-unit `G` depends on where it is read and a quadratic
    net gain overstates large steps. Here the odd and even responses of each block are
    interpolated linearly between the scanned lengths usable on every block (and from
    zero to the first), the net gain per update at a step is the validation block's odd
    plus even response there, the step is the one with the fewest updates to the net
    target among those whose retained blocks' even responses summed over the updates stay
    within the damage bound, and the call carries the resolution of the net gain at that
    step: it is undetermined unless the net gain is positive with both responses at their
    two-standard-error worst. RL's side is `rl_gain_per_step` as given (the preregistered
    reading, the shortest usable sign length)."""
    pts = sorted(((v["lam"], v) for v in resp.values() if v["usable_by_block"]), key=lambda x: x[0])
    if not pts:
        return {"call": "undetermined: no length usable on every block", "es": None, "rl": None, "ratio_es_over_rl_rollouts": None}
    blocks = sorted(pts[0][1]["by_block"])
    lams = np.asarray([0.0] + [lam for lam, _ in pts])

    def curve(block, key, scale=lambda lam: 1.0):
        return np.asarray([0.0] + [v["by_block"][block][key] * scale(lam) for lam, v in pts])

    at = lambda y, lam: float(np.interp(lam, lams, y))  # noqa: E731
    odd_v, even_v = curve("validation", "odd"), curve("validation", "even")
    odd_se_v = curve("validation", "G_se", lambda lam: lam)          # G_se is the odd response's error per unit
    even_se_v = curve("validation", "C_se", lambda lam: lam ** 2)
    others = [b for b in blocks if b != "validation"]
    best = None
    for lam in np.linspace(lams[1] / 10, lams[-1], 400):
        net = at(odd_v, lam) + at(even_v, lam)
        if net <= 0:
            continue
        T = G_TARGET / net
        losses = {b: T * at(curve(b, "even"), lam) for b in others}
        cand = {"step_multiple": float(lam), "updates": float(T), "rollouts": float(P * N * T), "net_per_update": float(net),
                "odd": at(odd_v, lam), "even": at(even_v, lam),
                "net_pessimistic": float(at(odd_v, lam) - 2 * at(odd_se_v, lam) + at(even_v, lam) - 2 * at(even_se_v, lam)),
                "retained_block_random_loss": losses, "feasible": all(loss >= -DAMAGE for loss in losses.values())}
        if best is None or (cand["feasible"] and not best["feasible"]) or (cand["feasible"] == best["feasible"] and T < best["updates"]):
            best = cand
    rl = {"steps": G_TARGET / rl_gain_per_step, "rollouts": G_TARGET / rl_gain_per_step * RL_ROLLOUTS_PER_STEP} if rl_gain_per_step > 0 else None
    ratio = best["rollouts"] / rl["rollouts"] if best and rl else None
    if best is None:
        call = "fails: no net gain in the scanned range"
    elif best["net_pessimistic"] <= 0:
        call = "undetermined: net gain unresolved"
    elif not best["feasible"]:
        call = "fails: retained-task loss"
    elif rl is None:
        call = "undetermined: RL's gain unresolved"
    elif ratio <= WORKS:
        call = "works"
    elif ratio <= MARGINAL:
        call = "marginal"
    else:
        call = "fails: rollouts"
    return {"es": best, "rl": rl, "ratio_es_over_rl_rollouts": ratio, "call": call, "lam_max": float(lams[-1]),
            "lengths_used": [float(lam) for lam, _ in pts], "exploratory": "2026-10-15"}


def bootstrap_rule(update: dict, sign: dict, rl_step: float, P: int, N: int, live_share: float, draws=1000) -> dict:
    """The rule's ratio and call under resampling of the scanned prompts within each block:
    `update` holds the plus, minus and start reward arrays and the block labels at the
    length read and the update's `lam`; `sign` the same for the sign direction with its
    effective length (or None). `G` and RL's slope come from the validation block, `C`
    from both blocks, as in `predict`."""
    rng = np.random.default_rng(0)
    blocks = np.asarray(update["block"])
    val = blocks == "validation"
    idx = {b: np.flatnonzero(blocks == b) for b in sorted(set(blocks))}
    joint = sign is not None and len(sign["block"]) == blocks.size and list(sign["block"]) == list(update["block"])
    ratios, calls = [], {}
    for _ in range(draws):
        i = np.concatenate([rng.choice(ix, ix.size) for ix in idx.values()])
        odd = (update["plus"][i] - update["minus"][i]) / 2
        even = (update["plus"][i] + update["minus"][i]) / 2 - update["start"][i]
        G, C = float(odd[val[i]].mean() / update["lam"]), float(even.mean() / update["lam"] ** 2)
        gain = 0.0
        if sign is not None:
            # the same prompts for ES and the RL proxy, so their covariance is kept
            j = i[val[i]] if joint else rng.choice(np.flatnonzero(np.asarray(sign["block"]) == "validation"), int(val.sum()))
            gain = rl_step * float(-((sign["plus"][j] - sign["minus"][j]) / 2).mean() / sign["effective_length"])
        r = rule(G, C, gain, P, N, live_share)
        calls[r["call"]] = calls.get(r["call"], 0) + 1
        if r["ratio_es_over_rl_rollouts"] is not None:
            ratios.append(r["ratio_es_over_rl_rollouts"])
    modal = max(calls, key=calls.get)
    return {"draws": draws, "joint_prompts": bool(joint),
            "ratio_90": [float(np.percentile(ratios, 5)), float(np.percentile(ratios, 95))] if ratios else None,
            "ratio_resolved_share": len(ratios) / draws, "call_shares": {k: v / draws for k, v in sorted(calls.items())},
            "modal_call": modal, "modal_share": calls[modal] / draws, "split": bool(calls[modal] / draws < 0.5)}


def level1(mem: dict, start_train: np.ndarray) -> dict:
    """The ranking's reliability and the cross-prompt correlation of member effects, as
    grad_check.py and countdown_grad_check.py define them, from greedy decodes."""
    from es_vllm.grad_check import split_reliability, variance_parts  # noqa: PLC0415

    out = {}
    for key, rewards, sigma in (("sigma", mem["rewards"], mem["sigma"]), ("sigma2", mem["rewards2"], mem["sigma2"])):
        if not rewards:
            continue
        delta = np.asarray(rewards) / scale_of(mem["kinds"])[None] - start_train[None]
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


def level1_offtask(mem: dict, off: dict, start_train: np.ndarray, draws=1000) -> dict:
    """The members' effects on the validation and off-task blocks beside their train
    fitness, by mirrored pair: the odd part of a pair's effect on a block is its direction's
    slope there, so the update's first-order change on each block is the sum of the
    pairs' odd effects weighted by the update's coefficients (the same quantity the
    `steps` scan measures directly, here decomposed over the members), and the
    correlation across pairs between the train odd effect and a block's says whether the
    directions that rank well are the ones that move the block. The even parts give the
    random-direction curvature per length squared on each block from n/2 directions at
    the member's length. Standard errors by bootstrap over the blocks' prompts; the
    correlations also by bootstrap over pairs."""
    n, r = mem["n"], mem["rank"]
    fit = np.asarray(mem["rewards"]) / scale_of(mem["kinds"])[None]
    f = fit.mean(1)                                                    # train fitness, as the update ranks it
    coef = lowrank.coefficients(f, n, ALPHA, r)                        # the update's coefficient per pair
    scale = scale_of(off["kinds"])
    R = np.asarray(off["rewards"]) / scale[None]
    start = np.asarray(off["start"]) / scale
    blocks = np.asarray(off["block"])
    train_odd = (f[0::2] - f[1::2]) / 2
    train_even = (f[0::2] + f[1::2]) / 2 - start_train.mean()
    sigma = mem["sigma"]
    rng = np.random.default_rng(0)
    # the spread of the pairs' odd effects has a sampling part: the bootstrap over prompts
    # gives each pair's noise, and the true spread is what is left
    boot_train = np.asarray([((fit[:, i].mean(1))[0::2] - (fit[:, i].mean(1))[1::2]) / 2
                             for i in (rng.choice(fit.shape[1], fit.shape[1]) for _ in range(draws))])
    train_noise = float(np.mean(boot_train.var(axis=0, ddof=1)))

    def per_pair(R_, start_, sel):
        D = R_[:, sel].mean(1) - start_[sel].mean()                   # each member's change on the block
        return (D[0::2] - D[1::2]) / 2, (D[0::2] + D[1::2]) / 2

    def along_update(odd):                                            # first-order change per unit update
        return float(np.sum(coef * odd) * np.sqrt(r) / sigma)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None

    out = {"n": n, "pairs": n // 2, "sigma": sigma, "prompts": {b: int((blocks == b).sum()) for b in sorted(set(blocks))},
           "train_odd_sd": float(train_odd.std(ddof=1)), "train_odd_noise_sd": float(np.sqrt(train_noise)),
           "train_odd_true_sd": float(np.sqrt(max(train_odd.var(ddof=1) - train_noise, 0.0))),
           "train_even_mean": float(train_even.mean()),
           "coefficient_norm": float(np.linalg.norm(coef)), "by_block": {}}
    for b in sorted(set(blocks)):
        sel = blocks == b
        odd, even = per_pair(R, start, sel)
        boot_g, boot_c, boot_e, boot_o = [], [], [], []
        for _ in range(draws):
            i = rng.choice(np.flatnonzero(sel), int(sel.sum()))
            o_, e_ = per_pair(R, start, i)
            boot_g.append(along_update(o_))
            boot_c.append(corr(train_odd, o_))
            boot_e.append(float(e_.mean()))
            boot_o.append(o_)
        noise = float(np.mean(np.asarray(boot_o).var(axis=0, ddof=1)))
        pairs_c = [corr(train_odd[j], odd[j]) for j in (rng.choice(n // 2, n // 2) for _ in range(draws))]
        pairs_c = [c for c in pairs_c if c is not None]
        top = np.argsort(-np.abs(coef))[:4]
        out["by_block"][b] = {
            "odd_per_pair_sd": float(odd.std(ddof=1)), "odd_per_pair_noise_sd": float(np.sqrt(noise)),
            "odd_per_pair_true_sd": float(np.sqrt(max(odd.var(ddof=1) - noise, 0.0))), "even_mean": float(even.mean()),
            "even_se": float(np.std(boot_e, ddof=1)),
            "G_from_members": along_update(odd), "G_from_members_se": float(np.std(boot_g, ddof=1)),
            "correlation_with_train_odd": corr(train_odd, odd),
            "correlation_se_prompts": float(np.std([c for c in boot_c if c is not None], ddof=1)) if any(c is not None for c in boot_c) else None,
            "correlation_90_pairs": [float(np.percentile(pairs_c, 5)), float(np.percentile(pairs_c, 95))] if pairs_c else None,
            "top4_pairs_share_of_G": float(np.sum(coef[top] * odd[top]) / np.sum(coef * odd)) if np.sum(coef * odd) != 0 else None,
            "mean_len": float(np.mean([m[b] for m in off["mean_len"]])), "start_mean_len": off["start_mean_len"][b]}
    v, o = out["by_block"].get("validation"), out["by_block"].get("offtask")
    if v and o:
        out["validation_offtask_odd_correlation"] = corr(*(per_pair(R, start, blocks == b)[0] for b in ("validation", "offtask")))
        g_v, g_o = v["G_from_members"], o["G_from_members"]
        out["offtask_change_per_unit_gain"] = g_o / g_v if g_v > 0 else None
        out["offtask_change_at_target"] = G_TARGET * g_o / g_v if g_v > 0 else None
        out["validation_resolved"] = bool(g_v > RESOLVED_CALL * v["G_from_members_se"])
    return out


def predict(out: Path) -> dict:
    """The rule's inputs and call from a setting's records. `G` and RL's slope are the
    validation block's (the task's gain); `C` pools the validation and off-task blocks
    (damage anywhere); the update's directional effect on the off-task block is reported
    beside them, scaled to the target gain."""
    sample = json.loads((out / "sample.json").read_text())
    grad = json.loads((out / "gradient.json").read_text()) if (out / "gradient.json").exists() else {}
    mem = json.loads((out / "members.json").read_text())
    smeta = json.loads((out / "steps-meta.json").read_text())

    def points(name):
        return [{**p, "rewards": normalized(p["rewards"], p["kinds"]).tolist()} for p in lines(out / name)]

    scan, sign, steps = (points(f) for f in ("scan.jsonl", "sign.jsonl", "steps.jsonl"))
    d_m = smeta["perturbed_params"]
    rl_step = RL_LR * float(np.sqrt(d_m))

    # level 0 (the points are normalized already, their starts included)
    rnd = random_responses(scan, np.asarray(next(p for p in scan if p["kind"] == "start")["rewards"])) if scan else {}
    rnd_used = [L for L, v in rnd.items() if v["usable"]] or list(rnd)
    rnd_far = rnd[max(rnd_used, key=float)] if rnd else None
    sgn = sign_responses(sign, np.asarray(next(p for p in sign if p["sign"] == 0)["rewards"])) if sign else {}
    sgn_used = [L for L, v in sgn.items() if v["usable"]] or list(sgn)
    sgn_near = sgn[min(sgn_used, key=float)] if sgn else None
    sgn_far = sgn[max(sgn_used, key=float)] if sgn else None
    rl_slope = sgn_near["by_block"]["validation"]["slope_along_rl_direction"] if sgn_near else None
    rl_gain = rl_step * rl_slope if sgn_near else 0.0
    # the slope at the longest usable sign length, reported beside the shortest (2026-10-15,
    # corrections item 10): resolved where the shortest is not, and lower where the
    # response along RL's direction is concave; the rule keeps the shortest
    rl_slope_far = sgn_far["by_block"]["validation"]["slope_along_rl_direction"] if sgn_far else None
    rl_gain_far = rl_step * rl_slope_far if sgn_far else 0.0
    level0 = {"live_share": sample["live_share"], "zero_advantage_share": sample["zero_advantage_share"],
              "start_reward_normalized": {b: float(normalized(v["rewards"], v["kinds"]).mean()) for b, v in sample["start"].items()},
              "answers_per_second": sample["answers_per_second"], "mean_sample_len": sample["mean_sample_len"],
              "live_prompts_in_gradient": grad.get("live_prompts"), "gradient_sharedness": grad.get("sharedness"),
              "gradient_diversity_yin": grad.get("diversity_yin"), "gradient_norm": grad.get("batch_norm"),
              "perturbed_params": d_m, "random_direction_responses": rnd,
              "curvature_per_length2": rnd_far["even_per_length2"] if rnd_far else None,
              "first_task_damaged_random": min(rnd_far["by_task"], key=lambda t: rnd_far["by_task"][t]["even_per_length2"])
              if rnd_far else None,
              "sign_direction_responses": sgn, "rl_step_length": rl_step,
              "rl_slope_along_direction": rl_slope,
              "rl_slope_offtask": sgn_near["by_block"].get("offtask", {}).get("slope_along_rl_direction") if sgn_near else None,
              "rl_first_step_gain": rl_gain,
              "rl_slope_at_longest_usable": rl_slope_far, "rl_slope_se_at_longest_usable":
              sgn_far["by_block"]["validation"].get("slope_se") if sgn_far else None,
              "rl_first_step_gain_at_longest_usable": rl_gain_far}

    # level 1
    start_train = normalized(sample["start"]["train"]["rewards"], sample["start"]["train"]["kinds"])
    lvl1 = level1(mem, start_train)
    off = (out / "members-offtask.json")
    lvl1_off = level1_offtask(mem, json.loads(off.read_text()), start_train) if off.exists() else None
    if steps and smeta["unit_norm"] > 0:
        start_steps = np.asarray(next(p for p in steps if p["sign"] == 0)["rewards"])
        resp = responses([p for p in steps if p["sign"] != 0], start_steps, smeta["unit_norm"])
        used = [L for L, v in resp.items() if v["usable"]] or list(resp)
        far = resp[max(used, key=float)]
        G, C = far["by_block"]["validation"]["G"], far["C"]
        G_off = far["by_block"].get("offtask", {}).get("G")
        update = {"unit_norm": smeta["unit_norm"], "n": mem["n"], "responses": resp, "length_used": max(used, key=float),
                  "usable": far["usable"], "G": G, "G_se": far["by_block"]["validation"]["G_se"], "C": C,
                  "C_se": far["even_se"] / far["lam"] ** 2, "G_pooled": far["G"], "G_offtask": G_off,
                  "offtask_change_at_target": G_TARGET * G_off / G if G_off is not None and G > 0 else None,
                  "first_task_damaged": min(far["by_task"], key=lambda t: far["by_task"][t]["C"]),
                  "random_cost_at_this_N_per_length2": C / smeta["unit_norm"] ** 2}
    else:                                    # every member scored alike: no update to scan
        update, G, C = None, 0.0, 0.0
    verdict = rule(G, C, rl_gain, sample["prompts"], mem["n"], sample["live_share"])
    if update is not None and update["offtask_change_at_target"] is not None:
        verdict["offtask_loss_at_target"] = bool(update["offtask_change_at_target"] < -DAMAGE)
    # the corrected target (2026-10-13): the longest length usable on every block, each
    # block's own responses, the step searched up to that length's multiple
    net = None
    if update is not None:
        ok = [L for L, v in resp.items() if v["usable_by_block"]]
        if ok:
            far_b = resp[max(ok, key=float)]
            net = rule_net(far_b["by_block"], rl_gain, sample["prompts"], mem["n"], sample["live_share"], far_b["lam"])
            net["length_used"] = max(ok, key=float)
            net["validation"] = far_b["by_block"]["validation"]
            if net["es"]:                 # the prescribed step scanned as its own point, if it was
                near = [L for L, v in resp.items() if abs(v["lam"] - net["es"]["step_multiple"]) <= 0.1 * net["es"]["step_multiple"]]
                net["prescribed_step_response"] = resp[near[0]] if near else None
            lengths = sorted(ok, key=float)
            if len(lengths) >= 2:         # the curvature at the two longest usable lengths
                a, b = resp[lengths[-1]]["by_block"]["validation"], resp[lengths[-2]]["by_block"]["validation"]
                net["curvature_stability"] = {"lengths": lengths[-2:], "C": [b["C"], a["C"]],
                                              "agree_within_2se": bool(abs(a["C"] - b["C"]) <= 2 * np.hypot(a["C_se"], b["C_se"]))}
        else:
            net = {"call": "undetermined: no length usable on every block", "es": None, "rl": None,
                   "ratio_es_over_rl_rollouts": None}
        update["rule_net_date"] = "2026-10-13"
    curve = rule_curve(resp, rl_gain, sample["prompts"], mem["n"]) if update is not None else None
    if update is not None:
        # the responses are keyed by the length's 6-digit form, so match points the same way
        arrays = lambda pts, key, val: {s: next(p for p in pts if f"{p[key]:.6g}" == val and p["sign"] == s)  # noqa: E731
                                        for s in (1, -1)}
        u = arrays(steps, "length", update["length_used"])
        upd = {"plus": np.asarray(u[1]["rewards"]), "minus": np.asarray(u[-1]["rewards"]), "start": start_steps,
               "block": u[1]["block"], "lam": u[1]["length"] / smeta["unit_norm"]}
        sgn_arrays = None
        if sgn_near is not None:
            s_ = arrays(sign, "nominal", min(sgn_used, key=float))
            sgn_arrays = {"plus": np.asarray(s_[1]["rewards"]), "minus": np.asarray(s_[-1]["rewards"]),
                          "block": s_[1]["block"], "effective_length": sgn_near["effective_length"]}
        verdict["bootstrap_over_prompts"] = bootstrap_rule(upd, sgn_arrays, rl_step, sample["prompts"], mem["n"],
                                                           sample["live_share"])
    return {"setting": None, "level0": level0, "level1": lvl1, "level1_offtask": lvl1_off, "update": update,
            "rule": verdict, "rule_net": net, "rule_curve": curve}


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
            "G_offtask": u.get("G_offtask"), "offtask_change_at_target": u.get("offtask_change_at_target"),
            "first_task_damaged": u.get("first_task_damaged"), "es": r["es"], "rl": r["rl"],
            "bootstrap_over_prompts": r.get("bootstrap_over_prompts"),
            "members_offtask": {"G_from_members": {b: (v["G_from_members"], v["G_from_members_se"]) for b, v in lo["by_block"].items()},
                                "correlation_with_train_odd": {b: v["correlation_with_train_odd"] for b, v in lo["by_block"].items()},
                                "offtask_change_at_target": lo.get("offtask_change_at_target"),
                                "validation_resolved": lo.get("validation_resolved")}
            if (lo := pred.get("level1_offtask")) else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", help="model/task, e.g. qwen3b/countdown")
    ap.add_argument("--phase", choices=("sample", "gradient", "scan", "sign", "members", "steps", "members-offtask"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analyze", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED, help="the member population (and the scan's directions)")
    ap.add_argument("--block", type=int, default=None,
                    help=f"validation and off-task prompts (default {BLOCK}; the test block stays at {BLOCK})")
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/probe"))
    ap.add_argument("--es-at-scale", type=Path, default=Path.home() / "private" / "open-source" / "es-at-scale")
    args = ap.parse_args(argv)
    if args.analyze:
        print(json.dumps(analyze(E2E / args.analyze), indent=2))
        return 0
    if args.out is None or args.phase is None or args.setting is None:
        ap.error("--setting, --phase and --out, or --analyze")
    cfg = setting(args.setting, args.smoke)
    cfg["seed"], cfg["block"] = args.seed, args.block
    out = E2E / args.out
    out.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)
    task = Task(cfg["task"], cfg["template"], args.smoke, args.es_at_scale, block=args.block)
    env = out / "env.json"
    if env.exists():
        prev = json.loads(env.read_text())
        if prev["block_digests"] != task.digests() or prev["constants"].get("seed", SEED) != args.seed:
            raise SystemExit(f"{out} holds a probe of other blocks or another population")
    else:
        env.write_text(json.dumps({
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "setting": args.setting, "repo": cfg["repo"], "revision": cfg["revision"], "smoke": args.smoke,
            "blocks": {b: len(v) for b, v in task.blocks.items()}, "block_digests": task.digests(),
            "block_prefix_digests": task.prefix_digests(),
            "constants": {"sigma": SIGMA, "sigma2": SIGMA2, "alpha": ALPHA, "rank": RANK, "k": K, "members": N_MEMBERS,
                          "members2": N_MEMBERS2, "lengths": LENGTHS, "sign_lengths": SIGN_LENGTHS, "rl_lr": RL_LR,
                          "seed": args.seed, "block": len(task.blocks["validation"])},
            "env": env_block(E2E, ["runs"], ("vllm", "torch", "transformers"))}, indent=2))
    {"sample": phase_sample, "gradient": phase_gradient, "scan": phase_scan, "sign": phase_sign,
     "members": phase_members, "steps": phase_steps, "members-offtask": phase_members_offtask}[args.phase](args, cfg, task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
