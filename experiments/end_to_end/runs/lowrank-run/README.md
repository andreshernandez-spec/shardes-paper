# Low-rank ES for eight updates: the gains accumulate, at half the one-update rate, 2026-10-10

One update at 512 low-rank mirrored members and its best step gained 0.036 +- 0.010
held-out reward points net of its cost (`../lowrank-check/`). This runs eight of them
along RL's prompt stream. `es_vllm/lowrank_run.py` at 33b33b1, prediction in the docstring
committed before the run, on a Secure Cloud H200 (US-CO-1, driver 580.178.04, $5.29/h,
16:33 to 03:36 UTC), about $58.50. Records `log.jsonl` (every member's reward on every
prompt, the update's norm and a digest of the f32 weights), `heldout.jsonl`, `run.json`;
logs `pod-log*.txt.gz` (the smoke run resumed from its log once, with a matching digest).
Numbers:

    python -m es_vllm.lowrank_run --analyze runs/lowrank-run

Each iteration: 512 rank-1 members (`../lowrank-check/`'s seeds at iteration 0, new seeds
after) decoded the iteration's 192 prompts (RL steps 4g+1 to 4g+4) greedily, the update
at 15.4 times the unit step (length 28.7 to 30.6) was added to an f32 copy of the
perturbed matrices, and the engine's bf16 weights were written from it and checked bit
for bit. An iteration took 4,560 to 5,180 s.

## Held-out reward, 3,840 prompts of RL steps 121 to 160 and 481 to 520

| after | reward | change (SE) | prompts up / down | predicted |
|---|---|---|---|---|
| 0 updates | 5.453 | | | |
| 4 updates | 5.471 | +0.018 (0.055) | 227 / 220 | +0.14 |
| 8 updates | 5.594 | +0.141 (0.061) | 298 / 244 | +0.29 |

- The docstring's criterion was below +0.10 after 8 updates for "does not accumulate at
  the one-update rate". The run is above it, and below the +0.29 a full one-update rate
  would give: about +0.018 per update against 0.036. The two held-out sets agree (+0.13
  and +0.15).

## By task, against RL

| change after | MATH | GSM8K | IFEval | all |
|---|---|---|---|---|
| ES, 8 updates, both sets | -0.46 (0.15) | +0.10 (0.12) | +0.47 (0.07) | +0.14 (0.06) |
| ES, 8 updates, steps 121-160 | -0.38 | +0.04 | +0.42 | +0.13 |
| RL step 40, steps 121-160 (`../heldout-121-160/`) | -0.05 | +0.20 | +0.81 | +0.44 |

- Eight updates see the prompts of RL's first 32 steps. ES gains on IFEval about half of
  what RL has by step 40, and loses on MATH, where RL holds level.
- RL's 32 steps decode 24,576 answers; these 8 updates decoded 786,432, 32 times as many.
  At the per-token rates of `../throughput-bench/` (4,744 GPU-seconds per ES update, 784
  per four RL steps) and these gains (+0.018 per update against RL's +0.044 per four steps
  to step 40), ES used about 15 times RL's GPU-seconds per unit of held-out gain.

## What this does not show

- One seed, one step size, eight updates. Whether the MATH loss grows, stops or reverses
  with more updates is not measured; nor whether a smaller step would trade less MATH for
  less IFEval.
