# Training runs at the rule's prescription, first batch (docs/end_to_end/10, S3b)

Six runs on 2026-10-13 with `es_vllm/probe_run.py`: three probed settings, each at the
step the preregistered rule prescribed from its probe record and two seeds of members,
30 updates of 64 mirrored rank-1 members on the probe's 192 train prompts (the same
prompts every update), the validation, reserved test and off-task blocks scored every
5 updates by the current model. One Secure H200 (RunPod pod `mto163dn9so3ma`, US-NC-1,
$5.29/h), 08:40 to 22:03 UTC, about $71 by uptime including two smoke relaunches. The pod
ran commit 001c03e; every run's `run.json` stamps it, the prediction it was held
against, and the digests of the probe's blocks (the test block is the one the probe
reserved and never decoded).

| run | model | task | step (units of the update) | the probe's call |
|---|---|---|---|---|
| qwen3b-countdown-s0, -s1 | Qwen2.5-3B-Instruct | Countdown | 15.3 (two member lengths) | works, ES/RL rollouts 2.8 |
| qwen3b-base-countdown-s0, -s1 | Qwen2.5-3B (base) | Countdown | 3.0 | marginal, 17; GSM8K flagged |
| qwen0.5b-gsm8k-s0, -s1 | Qwen2.5-0.5B-Instruct | GSM8K | 1.1 | fails, 50 |

Files per run: `run.json` (configuration, the prediction block, block sizes and digests,
environment), `log.jsonl` (per update: every member's reward on every prompt, the
fitness, the update's norm, the digest of the f32 weights, the bit-for-bit check),
`eval.jsonl` (per evaluation: every prompt's reward on the three blocks with its kind and
block, the mean answer length). Rewards are in the task's units; the analysis normalizes
by each row's kind (Countdown 1.1, the Tulu verifiers 10).

## The table

`python -m es_vllm.probe_run --table runs/probe-run`. Changes are in normalized reward
units against the evaluation at update 0; "class" is the outcome class frozen in `10`
(works: the reserved test block reaches +0.10 at an evaluation where the off-task block
is within −0.02; marginal: +0.05 under the same condition; fails otherwise), "class, task
alone" the class the test block would earn without that condition, and "collapsed" is set
when any block falls more than 0.02 below its start at any evaluation.

| run | preregistered call | step | updates to target, predicted | updates | validation at end | test at end | off-task at end | peak test gain | at | class | class, task alone | collapsed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen0.5b-gsm8k-s0 | fails: rollouts | 1.1 | 63.3 | 30 | +0.053 | +0.030 | +0.013 | 0.0333 | 15 | fails | fails | yes |
| qwen0.5b-gsm8k-s1 | fails: rollouts | 1.1 | 63.3 | 30 | -0.003 | -0.027 | +0.003 | 0.00667 | 10 | fails | fails | yes |
| qwen3b-base-countdown-s0 | marginal | 3.01 | 21.8 | 30 | +0.090 | +0.104 | -0.340 | 0.104 | 30 | marginal | works | yes |
| qwen3b-base-countdown-s1 | marginal | 3.01 | 21.8 | 30 | +0.128 | +0.169 | -0.100 | 0.169 | 30 | works | works | yes |
| qwen3b-countdown-s0 | works | 15.3 | 2.89 | 30 | -0.114 | -0.118 | -0.837 | 0.135 | 15 | fails | works | yes |
| qwen3b-countdown-s1 | works | 15.3 | 2.89 | 30 | -0.070 | -0.055 | -0.843 | 0.164 | 15 | works | works | yes |

The collapse flag's 0.02 is below the 300-prompt standard error (about 0.025 on these
blocks), so a flag set by a single evaluation within that range (0.5B GSM8K: a −0.03 dip of the off-task
block at 20 on seed 0, a −0.027 reading of the test block at 30 on seed 1) says nothing;
the flags on the Countdown runs are set by drops of 0.1 to 0.84.

## The curves

Per-evaluation changes by block (`--analyze <run>` has the standard errors, about 0.02
to 0.03 per block, and the by-task changes):

**qwen0.5b-gsm8k-s0** (step 1.1, 30 updates, 368,640 rollouts, 1.9 h)

| after | validation | test | off-task | answer length | predicted validation |
|---|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 362 | +0.000 |
| 5 | +0.003 | -0.013 | +0.007 | 389 | +0.006 |
| 10 | +0.023 | +0.007 | +0.000 | 377 | +0.013 |
| 15 | +0.017 | +0.033 | +0.000 | 382 | +0.019 |
| 20 | +0.037 | +0.013 | -0.030 | 388 | +0.025 |
| 25 | +0.033 | +0.007 | -0.003 | 383 | +0.032 |
| 30 | +0.053 | +0.030 | +0.013 | 373 | +0.038 |

**qwen0.5b-gsm8k-s1** (step 1.1, 30 updates, 368,640 rollouts, 1.9 h)

| after | validation | test | off-task | answer length | predicted validation |
|---|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 362 | +0.000 |
| 5 | +0.010 | -0.017 | -0.007 | 372 | +0.006 |
| 10 | -0.007 | +0.007 | +0.003 | 379 | +0.013 |
| 15 | +0.003 | -0.020 | +0.003 | 382 | +0.019 |
| 20 | +0.010 | -0.013 | +0.003 | 383 | +0.025 |
| 25 | +0.017 | -0.003 | +0.003 | 387 | +0.032 |
| 30 | -0.003 | -0.027 | +0.003 | 408 | +0.038 |

**qwen3b-base-countdown-s0** (step 3.01, 30 updates, 368,640 rollouts, 2.0 h)

| after | validation | test | off-task | answer length | predicted validation |
|---|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 495 | +0.000 |
| 5 | +0.009 | +0.055 | +0.093 | 439 | +0.018 |
| 10 | +0.015 | +0.047 | +0.017 | 487 | +0.037 |
| 15 | +0.055 | +0.059 | +0.143 | 390 | +0.055 |
| 20 | +0.074 | +0.075 | +0.070 | 420 | +0.074 |
| 25 | +0.086 | +0.086 | -0.200 | 622 | +0.092 |
| 30 | +0.090 | +0.104 | -0.340 | 730 | +0.110 |

**qwen3b-base-countdown-s1** (step 3.01, 30 updates, 368,640 rollouts, 1.9 h)

| after | validation | test | off-task | answer length | predicted validation |
|---|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 495 | +0.000 |
| 5 | +0.055 | +0.042 | -0.060 | 542 | +0.018 |
| 10 | +0.083 | +0.086 | -0.077 | 542 | +0.037 |
| 15 | +0.107 | +0.122 | +0.033 | 479 | +0.055 |
| 20 | +0.126 | +0.127 | -0.060 | 518 | +0.074 |
| 25 | +0.126 | +0.143 | -0.097 | 544 | +0.092 |
| 30 | +0.128 | +0.169 | -0.100 | 582 | +0.110 |

**qwen3b-countdown-s0** (step 15.3, 30 updates, 368,640 rollouts, 2.7 h)

| after | validation | test | off-task | answer length | predicted validation |
|---|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 329 | +0.000 |
| 5 | +0.091 | +0.033 | -0.067 | 283 | +0.138 |
| 10 | +0.057 | +0.070 | -0.123 | 262 | +0.277 |
| 15 | +0.115 | +0.135 | -0.277 | 356 | +0.415 |
| 20 | +0.112 | +0.060 | -0.600 | 335 | +0.553 |
| 25 | -0.012 | -0.005 | -0.843 | 959 | +0.691 |
| 30 | -0.114 | -0.118 | -0.837 | 933 | +0.830 |

**qwen3b-countdown-s1** (step 15.3, 30 updates, 368,640 rollouts, 2.5 h)

| after | validation | test | off-task | answer length | predicted validation |
|---|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 329 | +0.000 |
| 5 | +0.147 | +0.111 | -0.003 | 287 | +0.138 |
| 10 | +0.116 | +0.128 | -0.057 | 316 | +0.277 |
| 15 | +0.161 | +0.164 | -0.090 | 340 | +0.415 |
| 20 | +0.076 | +0.090 | -0.420 | 354 | +0.553 |
| 25 | +0.045 | +0.051 | -0.737 | 544 | +0.691 |
| 30 | -0.070 | -0.055 | -0.843 | 963 | +0.830 |

## What the runs say about the rule

- **The calls in kind.** On the task alone every run did what its call said: the two
  Countdown settings learn (every seed reaches +0.10 on the reserved block, the instruct
  model at 5 to 15 updates, the base model at 15 to 30) and the 0.5B GSM8K setting does
  not (peaks of +0.033 and +0.007, within noise, while its train block rises 0.9 points
  on both seeds: it learns its 192 repeated prompts, not the task). The observed classes with the retained block
  counted are works / fails (instruct, by seed), works / marginal (base, by seed), fails
  / fails (0.5B).
- **The prescription's size.** The gross prescription said 2.9 updates for the instruct
  model and 22 for the base model; the net one (the corrected rule, `10`) 3.8 and 19.
  Observed: the instruct seeds reached +0.10 on the test block at 5 and at 10 to 15
  updates, the base seeds at 15 and 30. The first-order model is right to within a factor
  of two to four on the time to the target when the step is small (the base model's
  validation curve tracks the prediction through 30 updates, +0.090 observed against
  +0.110 predicted at seed 0) and wrong in shape when it is large: at two member lengths
  the instruct model gains fast, plateaus, and then unwinds.
- **The collapse.** Every Countdown run loses its off-task block late: GSM8K goes from
  0.84 to 0.00 (instruct, both seeds, between updates 15 and 30, with the task's own gain
  unwinding and the answers lengthening to the caps, 512 tokens on Countdown and 2,048
  on GSM8K) and from 0.40 to 0.06 or 0.30
  (base, seeds 0 and 1, after 20). The onset is tied to the task gain reaching about
  +0.08 to +0.10, not to the step: it comes ten updates later at a fifth of the step.
  Nothing in a one-update probe sees it; the probe's start-point off-task numbers were
  −0.01 (instruct) and −1.1 (base), and the base model's GSM8K first rose by 0.14. A
  retained-task bound therefore needs several updates of the actual run, and the rule's
  "works" is a statement about the task alone until that is measured.
- **Seeds.** At the large step the two instruct seeds class differently (works, fails)
  on the early trajectory and end identically; at the small step the base seeds both
  reach the target and differ in how much GSM8K they keep (−0.34 against −0.10 at 30).
  Resampling prompts does not stand for this; the probe's intervals do not include it.
- **The RL side** of every ratio is still the proxy (the Tulu run's step, a sign-direction
  slope, Tulu 8B's cost per answer): these runs test the ES side of the forecast only.

## Incidents

Two smoke failures before the batch (adapters per call above the smoke's population;
the iteration count inside the resume's configuration check), both fixed before the
first run. No run failed or resumed; every update's bit-for-bit check passed.
