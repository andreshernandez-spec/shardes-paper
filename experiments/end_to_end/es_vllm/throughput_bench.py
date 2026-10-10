#!/usr/bin/env python
"""Where the time goes: decoding with and without LoRA members, answer lengths, and a gradient.

    python -m es_vllm.throughput_bench --part plain --seqs 512 --out runs/throughput-bench
    python -m es_vllm.throughput_bench --part plain --seqs 1024 --out runs/throughput-bench
    python -m es_vllm.throughput_bench --part lora --seqs 512 --out runs/throughput-bench
    python -m es_vllm.throughput_bench --part lora --seqs 1024 --out runs/throughput-bench
    python -m es_vllm.throughput_bench --part train --out runs/throughput-bench
    python -m es_vllm.throughput_bench --analyze runs/throughput-bench

ES's progress per hour is set by member-rollouts per second (runs/lowrank-check/: about
21 per second per H200 with 64 rank-1 adapters per batch and 512 sequences in flight).
This measures, on one H200 and the Tulu start, the 192 iteration-0 prompts decoded
greedily 32 times over (6,144 answers, cap 2,048):

- `plain`: the base model without LoRA, at `--seqs` sequences in flight;
- `lora`: a LoRA engine (rank 1, 64 adapters), the same requests without adapters and
  then as 32 members (`lowrank.py`, generation 0), with every answer's length and reward
  saved for the length analysis;
- `train`: what RL pays per token for its gradient, a forward and backward pass of the
  8B (gradient checkpointing, micro-batches of 2 as the RL run's
  `per_device_train_batch_size`) on the base's answers, and a forward without gradient
  (the reference model), on one GPU without optimizer or communication.
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

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
import releases as R  # noqa: E402
from es_vllm import lowrank  # noqa: E402
from provenance import env_block  # noqa: E402

COPIES, P_RUN, CAP, SEED, SIGMA, RANK = 32, 192, 2048, 0, 5e-4, 1


def append(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def setup(smoke):
    from transformers import AutoTokenizer  # noqa: PLC0415

    from es_vllm.run_tulu import load_prompts, render  # noqa: PLC0415

    repo, rev = ("Qwen/Qwen2.5-0.5B-Instruct", None) if smoke else (R.TULU31_START.repo, R.TULU31_START.commit)
    rows = load_prompts(E2E / "data" / "tulu31", 1, P_RUN)[0][: 8 if smoke else None]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    return repo, rev, rows, [render(tok, r) for r in rows]


def timed_generate(llm, ids, params, loras=None):
    import torch  # noqa: PLC0415

    torch.cuda.synchronize()
    t = time.perf_counter()
    kw = {"lora_request": loras} if loras is not None else {}
    outs = llm.generate([{"prompt_token_ids": i} for i in ids], params, use_tqdm=False, **kw)
    torch.cuda.synchronize()
    s = time.perf_counter() - t
    toks = sum(len(o.outputs[0].token_ids) for o in outs)
    return outs, {"seconds": s, "sequences": len(outs), "tokens": toks, "tokens_per_s": toks / s,
                  "sequences_per_s": len(outs) / s}


def part_decode(args, lora: bool) -> None:
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from vllm.lora.request import LoRARequest  # noqa: PLC0415

    from es_vllm.lowrank_check import Verifiers, items  # noqa: PLC0415

    repo, rev, rows, ids = setup(args.smoke)
    copies = 4 if args.smoke else args.copies
    chunk = min(copies, args.chunk)
    cap = 64 if args.smoke else CAP
    kw = {}
    if lora:
        model_dir = snapshot_download(repo, revision=rev, allow_patterns=["*.safetensors", "*.json"])
        shapes = lowrank.leaf_shapes(model_dir)
        template = lowrank.write_template(Path(args.workdir) / "adapter", RANK, repo)
        lowrank.install_adapters(template, SEED, shapes, RANK, SIGMA)
        kw = {"enable_lora": True, "max_lora_rank": RANK, "max_loras": chunk, "max_cpu_loras": copies}
    llm = LLM(model=repo, revision=rev, dtype="bfloat16", seed=0, max_model_len=4096,
              gpu_memory_utilization=0.4 if args.smoke else 0.85, enable_prefix_caching=False,
              max_num_seqs=args.seqs, **kw)
    params = SamplingParams(temperature=0.0, max_tokens=cap)
    all_ids = [i for _ in range(copies) for i in ids]
    out = E2E / args.out
    if lora and args.no_base:
        outs = None
    else:
        outs, base = timed_generate(llm, all_ids, params)
        append(out / "throughput.jsonl", {"part": "lora-engine-base" if lora else "plain",
                                          "seqs": args.seqs, "copies": copies, **base})
        print(json.dumps(base), flush=True)
    if not lora:
        return
    template = Path(args.workdir) / "adapter"
    # members in chunks of `chunk` adapters per generate call, as the run decodes them
    import torch  # noqa: PLC0415

    torch.cuda.synchronize()
    t = time.perf_counter()
    m_outs = []
    for c0 in range(0, copies, chunk):
        ms = range(c0, min(copies, c0 + chunk))
        m_outs += llm.generate([{"prompt_token_ids": i} for _ in ms for i in ids], params,
                               lora_request=[LoRARequest(f"member{m}", m + 1, str(template)) for m in ms for _ in ids],
                               use_tqdm=False)
    torch.cuda.synchronize()
    sec = time.perf_counter() - t
    toks = sum(len(o.outputs[0].token_ids) for o in m_outs)
    mem = {"seconds": sec, "sequences": len(m_outs), "tokens": toks, "tokens_per_s": toks / sec,
           "sequences_per_s": len(m_outs) / sec}
    append(out / "throughput.jsonl", {"part": "lora-members", "seqs": args.seqs, "copies": copies,
                                      "chunk": chunk, **mem})
    print(json.dumps(mem), flush=True)
    if (out / "answers.json").exists() or outs is None:
        return
    vs = Verifiers(2 if args.smoke else 8)
    base_one = outs[: len(ids)]
    rec = {"prompts": len(ids), "copies": copies, "cap": cap,
           "base": {"lengths": [len(o.outputs[0].token_ids) for o in base_one],
                    "stopped": [o.outputs[0].finish_reason == "stop" for o in base_one],
                    "rewards": vs.score(items(base_one, rows))},
           "members": {"lengths": [len(o.outputs[0].token_ids) for o in m_outs],
                       "stopped": [o.outputs[0].finish_reason == "stop" for o in m_outs],
                       "rewards": vs.score(items(m_outs, rows * copies))},
           "train_sequences": [list(i) + list(o.outputs[0].token_ids) for i, o in zip(ids, base_one)],
           "train_prompt_lengths": [len(i) for i in ids]}
    vs.close()
    (out / "answers.json").write_text(json.dumps(rec))


def part_train(args) -> None:
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    repo, rev, _, _ = setup(args.smoke)
    out = E2E / args.out
    rec = json.loads((out / "answers.json").read_text())
    seqs, plens = rec["train_sequences"], rec["train_prompt_lengths"]
    order = np.argsort([len(s) for s in seqs])        # micro-batches of similar length
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    def batch(idx):
        L = max(len(seqs[i]) for i in idx)
        x = torch.zeros((len(idx), L), dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        labels = torch.full((len(idx), L), -100, dtype=torch.long)
        for k, i in enumerate(idx):
            s = torch.tensor(seqs[i])
            x[k, :len(s)], mask[k, :len(s)] = s, 1
            labels[k, plens[i]:len(s)] = s[plens[i]:]
        return x.cuda(), mask.cuda(), labels.cuda(), sum(len(seqs[i]) for i in idx)

    res = {}
    for mode in ("forward_backward", "forward_no_grad"):
        model.train(mode == "forward_backward")
        steps, toks, secs = 0, 0, 0.0
        for b0 in range(0, len(order) - 1, 2):
            x, mask, labels, n_tok = batch(order[b0:b0 + 2])
            torch.cuda.synchronize()
            t = time.perf_counter()
            if mode == "forward_backward":
                model(input_ids=x, attention_mask=mask, labels=labels).loss.backward()
                model.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    model(input_ids=x, attention_mask=mask)
            torch.cuda.synchronize()
            if steps >= 2:  # warmup excluded
                toks, secs = toks + n_tok, secs + time.perf_counter() - t
            steps += 1
        res[mode] = {"micro_batches": steps - 2, "tokens": toks, "seconds": secs,
                     "tokens_per_s": toks / secs if secs else None}
    res["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
    append(out / "throughput.jsonl", {"part": "train", "micro_batch": 2, **res})
    print(json.dumps(res), flush=True)


def analyze(out: Path) -> dict:
    tp = [json.loads(x) for x in (out / "throughput.jsonl").read_text().splitlines()]
    a = json.loads((out / "answers.json").read_text())
    base_r = np.asarray(a["base"]["rewards"])
    lm, rm = np.asarray(a["members"]["lengths"]), np.asarray(a["members"]["rewards"])
    differs = rm != np.tile(base_r, a["copies"])
    bl = np.tile(np.asarray(a["base"]["lengths"]), a["copies"])
    caps = {}
    for t in (512, 768, 1024, 1536):
        long_ = (lm > t) | (bl > t)
        caps[str(t)] = {"answers_longer": float((lm > t).mean()),
                        "tokens_beyond": float(np.clip(lm - t, 0, None).sum() / lm.sum()),
                        "reward_changes_involving_a_longer_answer": float((differs & long_).sum() / max(differs.sum(), 1))}
    rate = lambda part, seqs=512: next(t["tokens_per_s"] for t in tp  # noqa: E731
                                       if t["part"] == part and t.get("seqs") == seqs)
    train = next(t for t in tp if t["part"] == "train")
    seq_tokens = train["forward_backward"]["tokens"] / (2 * train["forward_backward"]["micro_batches"])
    answer = float(np.mean(a["base"]["lengths"]))
    # GPU-seconds per 192 prompts at these single-GPU rates (no communication, no idling):
    # RL's four steps decode 3,072 answers and take a gradient and a reference forward on
    # each; one ES update at 512 members decodes 98,304. Gains per 192 prompts: RL 0.040
    # (runs/heldout-*: step 120 over 30), ES 0.0365 (runs/lowrank-check/, best net at
    # 512); the screen's rollout share 0.535 (lowrank_run.py --screen-check, 64 members).
    rl = {"decode": 3072 * answer / rate("plain"),
          "gradient": 3072 * seq_tokens / train["forward_backward"]["tokens_per_s"],
          "reference": 3072 * seq_tokens / train["forward_no_grad"]["tokens_per_s"]}
    es_tokens = 512 * 192 * float(lm.mean())
    es = {"lora": es_tokens / rate("lora-members"), "base_rate": es_tokens / rate("plain")}
    rl_total = sum(rl.values())
    per_gain = lambda gpu_s, gain: gpu_s / gain  # noqa: E731
    cmp = {k: per_gain(v, 0.0365) / per_gain(rl_total, 0.040) for k, v in
           {"es_lora": es["lora"], "es_lora_screen": 0.535 * es["lora"],
            "es_base_rate": es["base_rate"], "es_base_rate_screen": 0.535 * es["base_rate"]}.items()}
    return {"throughput": tp, "member_answers": int(lm.size), "member_mean_length": float(lm.mean()),
            "member_at_cap": float((lm >= a["cap"]).mean()),
            "member_reward_changes": float(differs.mean()), "length_caps": caps,
            "per_192_prompts_gpu_seconds": {"rl": rl, "rl_total": rl_total, "rl_sequence_tokens": seq_tokens,
                                            "es_512": es},
            "es_over_rl_gpu_seconds_per_gain": cmp}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("plain", "lora", "train"))
    ap.add_argument("--seqs", type=int, default=512, help="sequences in flight (max_num_seqs)")
    ap.add_argument("--copies", type=int, default=COPIES, help="members (and copies of the prompts)")
    ap.add_argument("--chunk", type=int, default=COPIES, help="lora: adapters per generate call")
    ap.add_argument("--no-base", action="store_true", help="lora: skip the base model's timing")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analyze", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/throughput-bench"))
    args = ap.parse_args(argv)
    if args.analyze:
        print(json.dumps(analyze(E2E / args.analyze), indent=2))
        return 0
    if args.out is None or args.part is None:
        ap.error("--part and --out, or --analyze")
    (E2E / args.out).mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)
    env = E2E / args.out / "env.json"
    if not env.exists():
        env.write_text(json.dumps({"date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                                   "env": env_block(E2E, ["runs"], ("vllm", "torch", "transformers"))}))
    if args.part == "train":
        part_train(args)
    else:
        part_decode(args, args.part == "lora")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
