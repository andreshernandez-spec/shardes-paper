# 07. The longer Tulu 3.1 arm: preregistration

Written and committed 2026-10-07, before any run. Decided by Andres the same day, after T2
(`06`): its high-side fail is read as "the backend works", so the pilot's null (`03`) is
not a broken backend, and the question becomes whether ES moves on Tulu at all with more
data. The gate below is fixed here and not changed after the data.

## What runs

All on H200 SXM (the pilot's GPU type), from commit-pinned configs, with the contraction
computed one member per call (154f63b; `experiments/end_to_end/runs/update-check/`):

- **Arm** (`es_vllm/tulu-long-s5e-4.yaml`): the pilot's s5e-4 arm, the one with the largest
  slope, run to 120 iterations instead of 30: full-rank `SeedRegenerated`, N = 16, 192
  prompts per member, sigma 5e-4, alpha 5e-4, greedy, cap 2,048, seed 0. Iteration g takes
  RL steps 4g+1 to 4g+4 in the run's order, so after g iterations it has seen the prompts
  and generated the rollouts of RL step 4g: 120 iterations match RL steps 1 to 480.
- **Control** (`es_vllm/tulu-long-random.yaml`): seeded uniform fitness, members never
  decoded, same sigma as the arm so the random walk takes the same step size (the pilot's
  control ran at 1e-3), 120 iterations.
- **RL side** (`es_vllm/heldout-481-520.yaml`, `es_vllm/heldout.py`): the DPO start and the
  released RL branches `step_120`, `step_240`, `step_360`, `step_480`.

## What is measured

**Held-out prompts**: the 1,920 prompts of RL steps 481 to 520 (stream steps [480, 520)),
which neither ES run trains on and none of the RL branches compared has seen. Each ES run
decodes its current weights on them after 0, 30, 60, 90 and 120 updates, in the training
process (`heldout.jsonl`); `heldout.py` scores the RL branches and the DPO start the same
way: greedy, `tulu` template, cap 2,048, the run's verifiers, reward on the 0-10 scale.
Rewards are kept per prompt, so every comparison is paired over the same 1,920 prompts:
`d = mean(reward_a - reward_b)`, `SE = sd / sqrt(1920)` (`es_vllm/long_gate.py`).

## Gate G4

`d` = arm minus control after 120 updates, paired.

- **Pass**: `d > 2 SE`. Reported as "ES learns on Tulu 3.1 within RL steps 1 to 480's data
  (prompts and rollouts)". A run matched to the released `step_1920` (about $155-185) is
  then worth deciding on.
- **Fail**: reported as "no detectable learning within RL steps 1 to 480's data at this
  configuration", with the RL gap below; no matched run is proposed without a change of
  design. The next step is decided with Andres either way.

Reported whatever the outcome:

- RL at step 4g minus ES after g updates, for g = 30, 60, 90, 120 (paired), and the means.
- ES after g updates against its start and against the control at each point (paired).
- The measurement check: ES after 0 updates is the DPO start in the training engine, and
  must agree with `heldout.py`'s DPO start on the same prompts (expected identical).
- Both runs' in-run center reward at every iteration (each iteration's own 192 prompts,
  new to the run), and the OLS slope of the arm-minus-control difference.
- Whether the arm's first 30 iterations reproduce the pilot's s5e-4 log, fitness for
  fitness and digest for digest. Expected: yes. Clean replays reach every digest of that
  log (`runs/heldout-121-160/`), so the pilot's arm was unaffected by the scan fault, and
  greedy decoding on H200 repeated the pilot's iteration 0 across hosts.
- Lengths, cap hits, engine checks, iteration times.

## Cost

Measured in the pilot: about 255 s per iteration on an H200 for the arm. 120 iterations
plus five held-out evaluations (about 3 minutes each) is about 9 hours on one H200 at
$4.59/h, about $42. A second H200 runs the RL scoring (5 models, about 3 minutes each)
and then the control (about 30 s per iteration, 1 to 1.5 hours), about $9. About $50 in
all. If the arm's measured iteration time projects the total past $60, it is stopped and
the plan revisited. Each run checkpoints by its fitness log, which replays exactly since
154f63b (the driver now checks every logged digest on a resume).
