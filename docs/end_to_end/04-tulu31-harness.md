# 04. Tulu 3.1 evaluation harness: what the card's numbers mean

T3 of `01-tulu31-plan.md`, desk part, 2026-09-27. Not run yet.

The card's GSM8K, MATH and IFEval come from AI2's OLMES suite `tulu_3_dev`. Read from
`allenai/olmes` (`oe_eval/configs/tasks.py`, `task_suites.py`, main at
`5a51f502d463b8cdc4a2dcad7d7096c41ff1197e`, to be pinned):

| card column | OLMES task | setup | primary metric |
|---|---|---|---|
| GSM8K | `gsm8k::tulu` | test split, 8-shot from `STD:GSM8k`, few-shot as multi-turn chat, chat format | `exact_match` |
| MATH | `minerva_math::tulu` (7 subject tasks, macro) | test split, 4-shot as multi-turn chat, greedy, 1,024 new tokens, no stop sequences | the suite's macro average |
| IFEval | `ifeval::tulu` | chat format, greedy, 2,048 new tokens | `prompt_level_loose_acc` |

So the IFEval column is **prompt-level loose**, which settles the card-versus-paper
label conflict (the paper says "strict").

## The gate (T3), to run

On the pilot's GPU type (H200, the amendment in `03-tulu31-pilot.md`), with OLMES at a
pinned commit and vLLM as its backend: score `Llama-3.1-Tulu-3-8B-DPO` and the released
`Llama-3.1-Tulu-3.1-8B` on the three tasks. Pass if each lands within a stated
tolerance of the card (DPO: 84.3 / 42.0 / 81.1; 3.1: 90.0 / 47.8 / 83.9); the tolerance
is fixed before the run from the tasks' sampling error (about 1 point on GSM8K's 1,319
problems, 0.7 on MATH's 5,000, 1.7 on IFEval's 541 prompts, at one standard error).
Then the same harness scores RL branches and ES checkpoints (rebuilt from their logs).
