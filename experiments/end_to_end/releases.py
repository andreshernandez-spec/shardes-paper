"""Every external artifact this paper compares against or trains from, pinned.

One place, so a driver never types a repo id or a revision itself. Hugging Face and
GitHub revisions are full commit hashes: a branch or tag can move, a commit cannot.
`pin_releases.py` checks these against the hub and records the weight hashes behind
them; it refuses to run if a pinned commit here no longer resolves.

Sources for the choices, all checked 2026-09-25 (docs/end_to_end/00-plan.md):
- Olmo 3 RL-Zero IF: open-instruct `scripts/train/olmo3/README.md` maps "7B RL Zero IF"
  to commit d928a7c and W&B run wn9zgjj3, whose logged config names the data,
  template, seed and lengths used below.
- Tulu 3.1 8B: the model card names open-instruct 3f37c29 and the exact command.
"""

from typing import NamedTuple


class Repo(NamedTuple):
    repo: str
    kind: str        # "model" or "dataset"
    commit: str      # the revision we use, as a full commit hash
    note: str = ""


# ---------------------------------------------------------------- Olmo 3 RL-Zero IF

OLMO3_BASE = Repo(
    "allenai/Olmo-3-1025-7B", "model", "a81bae42db3975be1671e27b9c9a56da1a9f980f",
    "main; its weights are byte-identical to branch stage3-step11921, the checkpoint "
    "the IF run logged as model_name_or_path (.../step11921-hf)",
)
OLMO3_BASE_LOGGED_BRANCH = "stage3-step11921"

OLMO3_IF_RL = Repo(
    "allenai/Olmo-3-7B-RL-Zero-IF", "model", "3cae007c5324a9a06ee6780f03b848bc7b8fd440",
    "main = step 2000; branches step_100 to step_1900 every 100",
)
OLMO3_IF_RL_BRANCHES = tuple(f"step_{s}" for s in range(100, 2000, 100))
OLMO3_IF_ROLLOUTS_PER_STEP = 32 * 8   # num_unique_prompts_rollout x samples, logged

# The run sampled from this set, not from the later Dolci release.
OLMO3_IF_SOURCE = Repo(
    "saurabh5/IF_multi_constraints_upto5_filtered_olmo_completions_filtered", "dataset",
    "3a5ace2c2a19f30cf7cd0e3e4c1d4310ea568d73",
    "created 2025-10-01, unmodified since; the run started 2025-10-15",
)
OLMO3_IF_DOLCI = Repo(
    "allenai/Dolci-RL-Zero-IF-7B", "dataset", "c96e4e424c0a5d416f725c488ec7c61b7f758d85",
    "the released IF set, uploaded 2025-11-17, after the run",
)

OLMO3_IF_OPEN_INSTRUCT = "d928a7cd29c2610af04300f817494c8e6dba977d"
OLMO3_IF_WANDB = ("ai2-llm", "Olmo-3-7B-RL-Zero", "wn9zgjj3")

# From the logged config of wn9zgjj3; olmo3_if_data.py asserts these against the
# recorded config rather than trusting this copy.
OLMO3_IF_LOGGED = {
    "dataset_mixer_list": ["saurabh5/IF_multi_constraints_upto5_filtered_olmo_completions_filtered", "13314"],
    "dataset_transform_fn": ["rlvr_tokenize_v1", "rlvr_filter_v1"],
    "chat_template_name": "olmo_thinker",
    "add_bos": False,
    "seed": 1,
    "max_prompt_token_length": 2048,
    "max_token_length": 10240,
    "response_length": 16384,
    "stop_strings": ["</answer>"],
    "num_unique_prompts_rollout": 32,
    "num_samples_per_prompt_rollout": 8,
}
# open-instruct's get_cached_dataset_tulu default at d928a7c; grpo_fast.py never passes
# its own, so the logged config has no key for it.
OLMO3_IF_DATASET_CONFIG_SEED = 42

# The harness gate runs on a checkpoint with published IFEval/IFBench numbers.
OLMO3_INSTRUCT = Repo(
    "allenai/Olmo-3-7B-Instruct", "model", "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
    "report Table 24: IFEval 85.8, IFBench 32.3",
)

# ---------------------------------------------------------------- Tulu 3.1 8B

TULU31_START = Repo(
    "allenai/Llama-3.1-Tulu-3-8B-DPO", "model", "a7beb67e33ffd01cc87ac3b46cadc1000985b8db",
)
TULU31_RL = Repo(
    "allenai/Llama-3.1-Tulu-3.1-8B", "model", "46239c2d07db76b412e1f1b0b4542f65b81fe01f",
    "main = step_1920; branches step_40 to step_2440 every 40",
)
TULU31_RL_BRANCHES = tuple(f"step_{s}" for s in range(40, 2441, 40))
TULU31_ROLLOUTS_PER_STEP = 48 * 16
TULU31_DATA = Repo(
    "allenai/RLVR-GSM-MATH-IF-Mixed-Constraints", "dataset",
    "7dbd180f5440c0b90f2944e6efea934b85437a95",
    "one data file, unchanged since 2024-11-18; later commits touch README and license",
)
TULU31_OPEN_INSTRUCT = "3f37c29ddc97d2c108a7658692d2d2c3708ef182"

ALL_REPOS = (
    OLMO3_BASE, OLMO3_IF_RL, OLMO3_IF_SOURCE, OLMO3_IF_DOLCI, OLMO3_INSTRUCT,
    TULU31_START, TULU31_RL, TULU31_DATA,
)
