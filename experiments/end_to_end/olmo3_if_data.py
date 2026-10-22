#!/usr/bin/env python
"""Rebuild the exact training set and prompt stream of the Olmo 3 RL-Zero IF run.

    python olmo3_if_data.py --out data/olmo3_if

"Same training data" is only true if we train on what the run trained on, and the
released Dolci-RL-Zero-IF-7B set came out a month after the run, from a different
config. So this reproduces the run's own data path, step by step, from open-instruct at
d928a7c and the run's logged config (W&B wn9zgjj3):

1. `get_cached_dataset_tulu` loads the source set (88,556 rows) and, for the logged
   count 13,314, keeps `RandomState(42).choice(88556, 13314, replace=False)` in that
   order (`DatasetConfig.select_samples`; 42 is the default, grpo_fast passes none).
2. `rlvr_tokenize_v1` (which is `rlvr_tokenize_v2`) renders the prompt with the
   `olmo_thinker` template and `add_generation_prompt=True`, and the full messages
   without it; `rlvr_filter_v1` keeps a row only if the prompt is at most 2,048 tokens
   and the full sequence at most 10,240.

   The filter is the one step we cannot replay exactly: it counts tokens with the
   run's transformers (4.57, Oct 2025), and 5.x counts one row 2,049 where the run
   kept it. The released Dolci-RL-Zero-IF-7B is that filtered set, dumped in the
   run's own pre-shuffle order (checked here: its rows are an order-preserving subset
   of step 1's sample). So the filtered set is taken from Dolci once that check
   passes, and every row where our count disagrees with the run's is recorded.
3. `setup_datasets` shuffles with `Dataset.shuffle(seed=1)`, which is
   `np.random.default_rng(1).permutation(n)` (checked against `datasets` below).
4. `ShufflingIterator(np.arange(n), 32, seed=1)` yields each step's 32 prompts: one
   shuffle, batches of 32, the remainder dropped, and a reshuffle from the same
   generator at each epoch boundary. `exclude_indices` exists but nothing calls it at
   d928a7c, and active sampling did not exist yet, so step k takes batch k.

Outputs, both refusing to overwrite:
- `training-set.json`: the ordered keys, what the filter dropped and why, token-length
  statistics, and the comparison with Dolci-RL-Zero-IF-7B;
- `prompt-stream.json`: for steps 1 to 2000 (the released checkpoint), the 32 row
  positions each step drew, as indices into the ordered keys.

Network (tokenizer files and a few columns of the source parquet), CPU only.
"""

import argparse
import ast
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
from provenance import env_block  # noqa: E402

COLUMNS = ["key", "messages", "ground_truth", "dataset", "constraint_type", "constraint"]
STEPS = 2000
SOURCE_ROWS = 88_556


class ShufflingIterator:
    """open-instruct grpo_fast.ShufflingIterator at d928a7c, minus checkpointing and the
    exclusion list nothing used. Kept line for line so the stream is theirs."""

    def __init__(self, data: np.ndarray, batch_size: int, seed: int):
        self.data = data.copy()
        self.batch_size = batch_size
        self.index = 0
        self.rng = np.random.default_rng(seed)
        self.rng.shuffle(self.data)
        self.effective_size = len(self.data) - (len(self.data) % self.batch_size)

    def __next__(self) -> list:
        if self.index >= self.effective_size:
            self.index = 0
            self.effective_size = len(self.data) - (len(self.data) % self.batch_size)
            self.rng.shuffle(self.data)
        end = self.index + self.batch_size
        batch = self.data[self.index:end].tolist()
        self.index = end
        return batch


def prompt_stream(n: int, batch: int, seed: int, steps: int) -> list:
    it = ShufflingIterator(np.arange(n), batch, seed)
    return [next(it) for _ in range(steps)]


def sample_indices(n_source: int, count: int, seed: int) -> np.ndarray:
    """`DatasetConfig.select_samples` for count <= n_source: no repeats, one choice."""
    if count > n_source:
        raise ValueError("upsampling is not what the logged run did")
    return np.random.RandomState(seed).choice(n_source, size=count, replace=False)


def filtered_from_release(sample: list, release_keys: list) -> list:
    """Source positions of the run's filtered set, taken from its release.

    `sample` is [(source position, key)] in the run's sample order. The release must be
    an order-preserving subset of it, which is what a dump of the filtered, not yet
    shuffled dataset is; anything else raises, since then it is not the run's set.
    """
    pos_of = dict((k, p) for p, k in sample)
    if len(pos_of) != len(sample):
        raise ValueError("duplicate keys in the sample; key-based matching is unsafe")
    wanted = set(release_keys)
    if not wanted <= set(pos_of):
        raise ValueError("the release has rows outside the run's sample")
    if list(release_keys) != [k for _, k in sample if k in wanted]:
        raise ValueError("the release is not the sample in the sample's order")
    return [pos_of[k] for k in release_keys]


def as_messages(value):
    """The source stores messages as a list of structs; older dumps as a repr string."""
    return ast.literal_eval(value) if isinstance(value, str) else value


def sha256_lines(items) -> str:
    return hashlib.sha256("".join(f"{x}\n" for x in items).encode()).hexdigest()


def load_source():
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415
    from huggingface_hub import HfFileSystem  # noqa: PLC0415

    fs = HfFileSystem()
    base = f"datasets/{R.OLMO3_IF_SOURCE.repo}@{R.OLMO3_IF_SOURCE.commit}/data"
    # `load_dataset` reads the train shards in name order; so do we.
    files = sorted(f for f in fs.ls(base, detail=False) if f.endswith(".parquet"))
    table = pa.concat_tables(pq.read_table(f, columns=COLUMNS, filesystem=fs) for f in files)
    if table.num_rows != SOURCE_ROWS:
        raise SystemExit(f"source has {table.num_rows} rows, the run saw {SOURCE_ROWS}")
    return table.to_pylist(), [f.rsplit("/", 1)[1] for f in files]


def load_dolci():
    import pyarrow.parquet as pq  # noqa: PLC0415
    from huggingface_hub import HfFileSystem  # noqa: PLC0415

    fs = HfFileSystem()
    base = f"datasets/{R.OLMO3_IF_DOLCI.repo}@{R.OLMO3_IF_DOLCI.commit}/data"
    files = sorted(f for f in fs.ls(base, detail=False) if f.endswith(".parquet"))
    rows = []
    for f in files:
        rows += pq.read_table(f, filesystem=fs).to_pylist()
    return rows


def tokenizer():
    from transformers import AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(R.OLMO3_BASE.repo, revision=R.OLMO3_BASE.commit)
    # open-instruct's get_tokenizer_tulu_v2_2 changes nothing when a distinct pad token
    # exists, and asserts pad != eos. Assert the same instead of re-deriving its branches.
    if tok.pad_token_id is None or tok.pad_token_id == tok.eos_token_id:
        raise SystemExit("tokenizer lacks a distinct pad token; open-instruct would alter it")
    tok.chat_template = templates.OLMO_THINKER
    return tok


def token_ids(tok, messages, generation_prompt: bool) -> list:
    """`apply_chat_template(tokenize=True)` as transformers 4.57 did it: render, then
    encode without special tokens. Pad ids are dropped, as rlvr_tokenize_v2 does."""
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=generation_prompt)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return [i for i in ids if i != tok.pad_token_id]


def compare_with_dolci(train_rows: list, dolci: list) -> dict:
    train = {r["key"]: r for r in train_rows}
    dol = {r["key"]: r for r in dolci}
    common = sorted(set(train) & set(dol))

    def raw_prompt(row):
        # rlvr_tokenize_v2's RAW_PROMPT_KEY: "role: content" per message, joined by "\n".
        # Exact, no stripping: prompts that start with whitespace are real.
        msgs = as_messages(row["messages"])
        prompt = msgs if len(msgs) == 1 else msgs[:-1]
        return "\n".join(f"{m['role']}: {m['content']}" for m in prompt)

    def prompt_text(row):
        return raw_prompt(row)

    differing = [k for k in common if raw_prompt(train[k]) != dol[k]["prompt"]]
    same_prompt = len(common) - len(differing)
    examples = [{"key": k, "ours": prompt_text(train[k])[-80:],
                 "dolci": dol[k]["prompt"][-80:]} for k in differing[:3]]
    return {
        "prompt_differences": {"count": len(differing), "keys": differing,
                               "examples_last_80_chars": examples},
        "training_rows": len(train_rows), "training_unique_keys": len(train),
        "dolci_rows": len(dolci), "dolci_unique_keys": len(dol),
        "common_keys": len(common),
        "only_in_training": sorted(set(train) - set(dol)),
        "only_in_dolci": sorted(set(dol) - set(train)),
        "common_with_identical_prompt": same_prompt,
    }


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

    logged = json.loads((HERE / "pins" / f"wandb-{R.OLMO3_IF_WANDB[2]}-config.json")
                        .read_text())["config"]
    for k, v in R.OLMO3_IF_LOGGED.items():
        if logged.get(k) != v:
            raise SystemExit(f"logged {k}={logged.get(k)!r}, releases.py says {v!r}")
    count = int(logged["dataset_mixer_list"][1])
    batch = logged["num_unique_prompts_rollout"]
    seed = logged["seed"]

    # Step 3's claim, checked rather than assumed: datasets' shuffle is this permutation.
    import datasets  # noqa: PLC0415
    probe = datasets.Dataset.from_dict({"i": list(range(1000))}).shuffle(seed=seed)["i"]
    if probe != np.random.default_rng(seed).permutation(1000).tolist():
        raise SystemExit("datasets.shuffle(seed) is no longer default_rng(seed).permutation")

    source, shard_names = load_source()
    sampled = sample_indices(len(source), count, R.OLMO3_IF_DATASET_CONFIG_SEED)

    tok = tokenizer()
    kept, dropped, prompt_lens = [], [], []
    for pos in sampled.tolist():
        row = source[pos]
        messages = as_messages(row["messages"])
        prompt = messages if len(messages) == 1 else messages[:-1]
        p_ids = token_ids(tok, prompt, generation_prompt=True)
        full = token_ids(tok, messages, generation_prompt=False)
        ok_p = len(p_ids) <= logged["max_prompt_token_length"]
        ok_f = len(full) <= logged["max_token_length"]
        if ok_p and ok_f:
            kept.append(pos)
            prompt_lens.append(len(p_ids))
        else:
            dropped.append({"source_index": pos, "key": row["key"],
                            "prompt_tokens": len(p_ids), "full_tokens": len(full)})

    # The run's filtered set, from Dolci, after proving Dolci is our sample, filtered,
    # in order. Anything else means the reconstruction is wrong and nothing is written.
    dolci_rows = load_dolci()
    filtered = filtered_from_release([(p, source[p]["key"]) for p in sampled.tolist()],
                                     [r["key"] for r in dolci_rows])
    ours, theirs = set(kept), set(filtered)
    disagreements = {
        "kept_by_run_dropped_by_us": [d for d in dropped if d["source_index"] in theirs],
        "dropped_by_run_kept_by_us": sorted(ours - theirs),
    }
    order = [filtered[i]
             for i in np.random.default_rng(seed).permutation(len(filtered)).tolist()]
    order_keys = [source[i]["key"] for i in order]
    stream = prompt_stream(len(order), batch, seed, STEPS)
    drawn = sum(len(b) for b in stream)
    if drawn != STEPS * batch:  # the work asked for is the work done
        raise SystemExit(f"stream drew {drawn} prompts, expected {STEPS * batch}")

    dolci = compare_with_dolci([source[i] for i in order], dolci_rows)
    lens = np.array(prompt_lens)
    outputs = [str(set_path.relative_to(HERE)), str(stream_path.relative_to(HERE))]
    env = env_block(HERE, outputs, ("numpy", "transformers", "datasets", "pyarrow",
                                    "huggingface_hub"))
    common = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "env": env,
        "source": {"repo": R.OLMO3_IF_SOURCE.repo, "revision": R.OLMO3_IF_SOURCE.commit,
                   "shards": shard_names, "rows": len(source)},
        "tokenizer": {"repo": R.OLMO3_BASE.repo, "revision": R.OLMO3_BASE.commit,
                      "chat_template_sha256": templates.OLMO_THINKER_SHA256},
        "open_instruct": R.OLMO3_IF_OPEN_INSTRUCT,
        "logged_run": "/".join(R.OLMO3_IF_WANDB),
    }
    harness.write_atomic(set_path, {
        **common,
        "sampled": count, "dataset_config_seed": R.OLMO3_IF_DATASET_CONFIG_SEED,
        "kept_by_our_filter": len(kept), "dropped_by_our_filter": dropped,
        "filter_disagreements_with_run": disagreements,
        "training_rows": len(order),
        "shuffle_seed": seed,
        "order_keys": order_keys, "order_keys_sha256": sha256_lines(order_keys),
        "order_source_indices": order,
        "prompt_tokens_our_tokenizer": {"mean": float(lens.mean()), "median": float(np.median(lens)),
                          "p95": float(np.percentile(lens, 95)), "max": int(lens.max())},
        "dolci": {"repo": R.OLMO3_IF_DOLCI.repo, "revision": R.OLMO3_IF_DOLCI.commit,
                  **dolci},
    })
    harness.write_atomic(stream_path, {
        **common,
        "steps": STEPS, "prompts_per_step": batch, "iterator_seed": seed,
        "rollouts_per_step": batch * logged["num_samples_per_prompt_rollout"],
        "order_keys_sha256": sha256_lines(order_keys),
        "stream": stream,
        "stream_sha256": sha256_lines(json.dumps(b) for b in stream),
    })
    print(f"source {len(source)} -> sampled {count} -> our filter keeps {len(kept)}, "
          f"the run kept {len(filtered)}; disagreements "
          f"{ {k: len(v) for k, v in disagreements.items()} }; prompt text differs "
          f"from Dolci on {dolci['prompt_differences']['count']} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
