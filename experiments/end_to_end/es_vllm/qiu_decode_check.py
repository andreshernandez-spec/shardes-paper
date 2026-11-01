#!/usr/bin/env python
"""Check 1 of docs/end_to_end/12: does our vLLM rule decode as Qiu's trainer does?

The 200 training prompts at the base weights, decoded by HF `generate` as his trainer
calls it (one left-padded batch, bf16, greedy, 1,024 new tokens, the model's generation
config, so its repetition penalty) and by vLLM three ways: the matched rule of
qiu_match.py, the same without the pad penalty, and without any repetition penalty.

    .venv-qiu-eval/bin/python experiments/end_to_end/es_vllm/qiu_decode_check.py --engine hf --out /root/dc-hf.json
    .venv-vllm/bin/python -m es_vllm.qiu_decode_check --engine vllm --hf /root/dc-hf.json \
        --record runs/qiu-matched/decode-check.json         # from experiments/end_to_end

The HF half runs in the evaluator's venv (no JAX), so it imports nothing of ours but
releases.py. The record says, for each vLLM variant, how many answers are token for
token HF's and the accuracy and mean reward each gives under his grading; the run goes
ahead only if the matched rule agrees with HF on at least as many answers as each
variant. In bf16 two implementations part ways within an answer through rounding alone,
so the record also has the floor: HF against itself with only the batching changed
(batches of 50, so other padding). `--fp32` runs both halves in float32, where what is
left of a difference is the rule rather than the rounding (a laptop check at 0.5B).
"""

import argparse
import json
import sys
from pathlib import Path

E2E = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(E2E))
import releases as R  # noqa: E402

VARIANTS = ("matched", "no_pad_penalty", "no_repetition_penalty")
SMOKE, SMOKE_NEW = 32, 256         # --smoke: 0.5B, 32 prompts, 256 tokens
REBATCH = 50                       # the floor: HF against itself in batches of 50 (smoke: 8)


def trim(ids: list, ends: set) -> list:
    """An answer up to its first end token, which is left out (HF pads after it)."""
    for i, t in enumerate(ids):
        if t in ends:
            return ids[:i]
    return ids


def agreement(answers: list, ref: list) -> dict:
    """How many answers are token for token the reference's, and where the others leave it."""
    same = [a == b for a, b in zip(answers, ref)]
    first = sorted(next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                   for a, b, s in zip(answers, ref, same) if not s)
    return {"identical": int(sum(same)), "first_difference_median": float(first[len(first) // 2]) if first else None}


def run_hf(args) -> dict:
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    torch.backends.cuda.matmul.allow_tf32 = True          # as his trainer
    m = R.QWEN25_05B if args.smoke else R.QWEN25_3B
    repo, rev = m.repo, m.commit
    rows = json.loads((args.es_at_scale / "archive/countdown/data/countdown.json").read_text())[:SMOKE if args.smoke else 200]
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, device_map={"": 0},
                                                 torch_dtype=torch.float32 if args.fp32 else torch.bfloat16)
    model.eval()
    tok = AutoTokenizer.from_pretrained(repo, revision=rev, use_fast=False)
    ends = set(model.generation_config.eos_token_id)

    def generate(texts):
        enc = tok(texts, return_tensors="pt", padding=True, padding_side="left")
        with torch.inference_mode():
            out = model.generate(enc["input_ids"].to("cuda"), attention_mask=enc["attention_mask"].to("cuda"),
                                 max_new_tokens=SMOKE_NEW if args.smoke else 1024, do_sample=False)
        return [trim(o[enc["input_ids"].shape[1]:].tolist(), ends) for o in out]

    texts = [r["context"] for r in rows]
    b = 8 if args.smoke else REBATCH
    return {"engine": "hf", "fp32": args.fp32, "transformers": transformers.__version__, "torch": torch.__version__,
            "repetition_penalty": model.generation_config.repetition_penalty, "rebatch": b,
            "answers": generate(texts),                                   # his trainer: one batch
            "answers_rebatched": [a for i in range(0, len(texts), b) for a in generate(texts[i:i + b])]}


def run_vllm(args) -> dict:
    import numpy as np  # noqa: PLC0415
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import SamplingParams  # noqa: PLC0415

    from es_vllm import probe, qiu_match as qm  # noqa: PLC0415
    from es_vllm.pad_penalty import PadPenalty  # noqa: PLC0415
    from es_vllm.run_countdown import es_at_scale  # noqa: PLC0415

    hf = json.loads(args.hf.read_text())
    clone = es_at_scale(args.es_at_scale)
    train, _ = qm.problems(clone, SMOKE if args.smoke else qm.TRAIN, qm.TEST)
    cfg = probe.setting("qwen0.5b/countdown" if args.smoke else "qwen3b/countdown", args.smoke)
    files = Path(snapshot_download(cfg["repo"], revision=cfg["revision"], allow_patterns=list(qm.TOKENIZER_FILES)))
    gen = json.loads((files / "generation_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(str(files), use_fast=False)
    rep, eos, pad = float(gen["repetition_penalty"]), sorted(gen["eos_token_id"]), tok.pad_token_id
    ids = [tok(r["context"])["input_ids"] for r in train]
    flags = qm.padded([len(x) for x in ids], len(ids))
    grader = qm.Grader(clone, tok)
    if args.fp32 != hf["fp32"]:
        raise SystemExit("the hf half ran at another precision")
    from vllm import LLM  # noqa: PLC0415

    llm = LLM(model=cfg["repo"], revision=cfg["revision"], dtype="float32" if args.fp32 else "bfloat16", seed=0,
              gpu_memory_utilization=0.4 if args.smoke else 0.35, max_model_len=4096, enable_prefix_caching=False,
              max_num_seqs=512, logits_processors=[PadPenalty])     # probe.engine's, with the precision free

    def params(variant):
        return [SamplingParams(temperature=0.0, top_p=1.0, max_tokens=SMOKE_NEW if args.smoke else qm.MAX_NEW, stop_token_ids=eos,
                               repetition_penalty=1.0 if variant == "no_repetition_penalty" else rep,
                               extra_args={"pad_penalty": [pad, rep]} if f and variant == "matched" else None)
                for f in flags]

    def graded(answers):
        sc = grader.score([p + a for p, a in zip(ids, answers)], train)
        return {"accuracy": qm.accuracy(sc), "mean_reward": float(np.mean([s[0] for s in sc]))}

    from provenance import env_block  # noqa: PLC0415

    rec = {"hf": {k: hf[k] for k in ("transformers", "torch", "repetition_penalty")} | graded(hf["answers"]),
           "hf_rebatched": {"batch": hf["rebatch"], **agreement(hf["answers_rebatched"], hf["answers"]),
                            **graded(hf["answers_rebatched"])},
           "fp32": args.fp32, "padded_rows": int(sum(flags)), "variants": {},
           "env": env_block(E2E, ["runs"], ("vllm", "torch", "jax", "jaxlib", "transformers"))}
    for v in VARIANTS:
        o = probe.decode(llm, ids, params(v))
        answers = [trim(list(x.outputs[0].token_ids), set(eos)) for x in o]
        rec["variants"][v] = {**agreement(answers, hf["answers"]), **graded(answers)}
        print(v, rec["variants"][v], flush=True)
    # the processor is live: a huge penalty on the token greedy picks first moves the pick
    one = SamplingParams(temperature=0.0, max_tokens=1)
    t0 = probe.decode(llm, ids[:1], [one])[0].outputs[0].token_ids[0]
    t1 = probe.decode(llm, ids[:1], [SamplingParams(temperature=0.0, max_tokens=1,
                                                    extra_args={"pad_penalty": [t0, 1e9]})])[0].outputs[0].token_ids[0]
    rec["processor_applied"] = bool(t1 != t0)
    print("hf_rebatched", rec["hf_rebatched"], flush=True)
    best_other = max(rec["variants"][v]["identical"] for v in VARIANTS[1:])
    rec["passes"] = rec["processor_applied"] and rec["variants"]["matched"]["identical"] >= best_other
    grader.close()
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("hf", "vllm"), required=True)
    ap.add_argument("--out", type=Path, help="hf: where its answers go")
    ap.add_argument("--hf", type=Path, help="vllm: the hf half's answers")
    ap.add_argument("--record", type=Path, help="vllm: the record, under experiments/end_to_end")
    ap.add_argument("--es-at-scale", type=Path, default=Path.home() / "private" / "open-source" / "es-at-scale")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="both halves in float32: what is left is the rule, not rounding")
    args = ap.parse_args(argv)
    if args.engine == "hf":
        args.out.write_text(json.dumps(run_hf(args)))
        return 0
    rec = run_vllm(args)
    path = E2E / args.record
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"decode check {'passes' if rec['passes'] else 'FAILS'}", flush=True)
    return 0 if rec["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
