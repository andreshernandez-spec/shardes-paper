#!/usr/bin/env python
"""Rebuild the training set and prompt stream of the Tulu 3.1 8B GRPO run.

    python tulu31_data.py --out data/tulu31

From open-instruct at 3f37c29 (`grpo_vllm_thread_ray_gtrl.py`, `dataset_transformation.py`)
and the model card's command; docs/end_to_end/02-tulu31-data-path.md walks through it:

1. The whole RLVR mix in file order (`"1.0"` gives `select(range(n))`).
2. The prompt rendered with the `tulu` template, `add_generation_prompt=True`, no bos
   (`add_bos` defaults to False and the template has none); a row is kept if its prompt
   and its full messages are at most 2,048 tokens each.
3. No shuffle: the script's `train_dataset.shuffle(seed=1)` discards its result.
4. `ShufflingIterator(np.arange(n), 48, seed=1)` gives each step's 48 prompts.

Unlike Olmo 3 there is no later release of the filtered set to check the length filter
against, so rows within a few tokens of either limit are recorded: a different tokenizer
version could count them differently, and one row more or fewer changes every
permutation after it.

The data has no key column, so rows are identified by their position in the file and a
sha256 of their messages and ground truth. Network (one 16.5 MB parquet, tokenizer
files), CPU only. Records refuse to overwrite.
"""

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import harness  # noqa: E402
import releases as R  # noqa: E402
import templates  # noqa: E402
from olmo3_if_data import as_messages, prompt_stream, sha256_lines  # noqa: E402
from provenance import env_block  # noqa: E402

ROWS = 29_946
PROMPTS_PER_STEP = 8 * 6          # local_rollout_batch_size x actor GPUs, card command
SAMPLES_PER_PROMPT = 16
SEED = 1
MAX_PROMPT = MAX_FULL = 2048      # --max_prompt_token_length, --max_token_length
STEPS = 2440                      # the last released branch
BORDER = 5                        # tokens from a limit that we flag


def row_hash(row) -> str:
    blob = json.dumps({"messages": as_messages(row["messages"]),
                       "ground_truth": row["ground_truth"]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def load_rows():
    import pyarrow.parquet as pq  # noqa: PLC0415
    from huggingface_hub import HfFileSystem  # noqa: PLC0415

    fs = HfFileSystem()
    f = (f"datasets/{R.TULU31_DATA.repo}@{R.TULU31_DATA.commit}"
         "/data/train-00000-of-00001.parquet")
    rows = pq.read_table(f, filesystem=fs).to_pylist()
    if len(rows) != ROWS:
        raise SystemExit(f"{len(rows)} rows, expected {ROWS}")
    return rows


def tokenizer():
    from transformers import AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(R.TULU31_START.repo, revision=R.TULU31_START.commit)
    tok.chat_template = templates.TULU
    return tok


def ids(tok, messages, generation_prompt: bool) -> list:
    """transformers 4.x `apply_chat_template(tokenize=True)`: render, then encode without
    special tokens. With add_bos False nothing is prepended."""
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=generation_prompt)
    return tok(text, add_special_tokens=False)["input_ids"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="directory, under this one")
    args = ap.parse_args(argv)
    out = (args.out if args.out.is_absolute() else HERE / args.out).resolve()
    out.relative_to(HERE)
    set_path, stream_path = out / "training-set.json", out / "prompt-stream.json"
    for p in (set_path, stream_path):
        if p.exists():
            raise SystemExit(f"{p} exists; records are never overwritten")

    rows, tok = load_rows(), tokenizer()
    kept, dropped, border, lens = [], [], [], []
    for pos, row in enumerate(rows):
        messages = as_messages(row["messages"])
        prompt = messages if len(messages) == 1 else messages[:-1]
        p_len = len(ids(tok, prompt, True))
        f_len = len(ids(tok, messages, False))
        entry = {"index": pos, "prompt_tokens": p_len, "full_tokens": f_len}
        if min(abs(p_len - MAX_PROMPT), abs(f_len - MAX_FULL)) <= BORDER:
            border.append(entry)
        if p_len <= MAX_PROMPT and f_len <= MAX_FULL:
            kept.append(pos)
            lens.append(p_len)
        else:
            dropped.append(entry)

    # No shuffle: the run's shuffle discarded its result, so file order stands.
    stream = prompt_stream(len(kept), PROMPTS_PER_STEP, SEED, STEPS)
    drawn = sum(len(b) for b in stream)
    if drawn != STEPS * PROMPTS_PER_STEP:
        raise SystemExit(f"stream drew {drawn} prompts, expected {STEPS * PROMPTS_PER_STEP}")

    hashes = [row_hash(rows[i]) for i in kept]
    lens = np.array(lens)
    sources = {}
    for i in kept:
        sources[rows[i]["dataset"]] = sources.get(rows[i]["dataset"], 0) + 1
    outputs = [str(set_path.relative_to(HERE)), str(stream_path.relative_to(HERE))]
    common = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "env": env_block(HERE, outputs, ("numpy", "transformers", "pyarrow",
                                        "huggingface_hub")),
        "data": {"repo": R.TULU31_DATA.repo, "revision": R.TULU31_DATA.commit,
                 "rows": len(rows)},
        "tokenizer": {"repo": R.TULU31_START.repo, "revision": R.TULU31_START.commit,
                      "chat_template_sha256": templates.TULU_SHA256, "add_bos": False},
        "open_instruct": R.TULU31_OPEN_INSTRUCT,
    }
    harness.write_atomic(set_path, {
        **common,
        "limits": {"prompt": MAX_PROMPT, "full": MAX_FULL},
        "training_rows": len(kept), "dropped": dropped,
        "near_a_limit": border,
        "order": "file order; the run's shuffle discarded its result",
        "kept_indices": kept, "row_sha256": hashes,
        "row_sha256_digest": sha256_lines(hashes),
        "kept_by_source": sources,
        "prompt_tokens": {"mean": float(lens.mean()), "median": float(np.median(lens)),
                          "p95": float(np.percentile(lens, 95)), "max": int(lens.max())},
    })
    harness.write_atomic(stream_path, {
        **common,
        "steps": STEPS, "prompts_per_step": PROMPTS_PER_STEP, "iterator_seed": SEED,
        "rollouts_per_step": PROMPTS_PER_STEP * SAMPLES_PER_PROMPT,
        "row_sha256_digest": sha256_lines(hashes),
        "stream": stream,
        "stream_sha256": sha256_lines(json.dumps(b) for b in stream),
    })
    print(f"{len(rows)} rows -> kept {len(kept)} (dropped {len(dropped)}, "
          f"{len(border)} within {BORDER} tokens of a limit); by source {sources}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
