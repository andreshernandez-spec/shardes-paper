# GRPO trained by us on probed settings: the measured RL reference (docs/end_to_end/10, S3c)

Three arms on 2026-10-14 with `es_vllm/grpo_run.py`, preregistered in `10` ("S3c"):
the Tulu 3.1 run's recipe (48 prompts and 16 samples per step at temperature 1.0,
group-normalized advantages, two mini-batches per batch, micro-batches of 2, the token
mean of the clipped policy term plus 0.01 times the kl3 estimator against the start,
AdamW at a constant learning rate, one epoch) on one Secure H200 (RunPod pod
`zj31w50qxykndl`, US-CO-1, $5.29/h), 05:30 to 08:41 UTC, about $17 by uptime. The pod ran
commit 8422e89; each arm's `run.json` stamps it, the probe record it is held against and
the digests of the probe's blocks. The policy is in f32 under gradient checkpointing, a
bf16 copy of the start is the reference, vLLM generates from the policy's weights
written into it after every step (170 engine parameters at 0.5B, 254 at 3B); the prompts
cycle through the probe's 192 train prompts in a seeded order, the prompts the ES runs
repeated; the validation, reserved test and off-task blocks are decoded greedily every
5 steps, as the ES runs were. The optimizer's betas (0.9, 0.95), eps 1e-8, no weight
decay and a gradient-norm clip of 1.0 are this driver's choices.

| arm | model | task | learning rate | the rule's RL-side prediction |
|---|---|---|---|---|
| qwen3b-countdown-lr5e-7-s0 | Qwen2.5-3B-Instruct | Countdown | 5e-7 (the reference) | +0.10 on the test block at step 16, off-task −0.006 |
| qwen3b-base-countdown-lr5e-7-s0 | Qwen2.5-3B (base) | Countdown | 5e-7 | step 20, off-task −0.028 |
| qwen3b-countdown-lr1e-6-s0 | Qwen2.5-3B-Instruct | Countdown | 1e-6 | twice the reference step: about 8 |

Files per arm: `run.json`, `log.jsonl` (per step: every sample's reward, the live
prompts, the losses, KL and gradient norms of both mini-batches, the clipped share, the
tokens, the seconds of each phase), `eval.jsonl` (as `probe-run`'s). Rewards are in the
task's units; the analysis normalizes by each row's kind.

## The table

`python -m es_vllm.grpo_run --table runs/grpo-run`. Changes are in normalized reward
units against the evaluation at step 0; the class is `10`'s as amended on 2026-10-14
(the checkpoint selected on the validation block, the class by the test block there;
`probe_run.observed_class`); "test +0.10 first at" is the first evaluation at or above the
target.

| arm | lr | steps | rollouts | validation at end | test at end | off-task at end | peak test gain | at | test +0.10 first at | validation-selected checkpoint | test there | class | collapsed | s per step | GPU-s per answer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3b-base-countdown-lr5e-7-s0 | 5e-07 | 40 | 30,720 | +0.121 | +0.135 | +0.143 | 0.156 | 35 | 30 | 40 | 0.135 | works | no | 87.3 | 0.114 |
| qwen3b-countdown-lr1e-6-s0 | 1e-06 | 40 | 30,720 | +0.159 | +0.213 | -0.003 | 0.213 | 40 | 15 | 20 | 0.122 | works | yes | 85.6 | 0.111 |
| qwen3b-countdown-lr5e-7-s0 | 5e-07 | 40 | 30,720 | +0.138 | +0.135 | -0.007 | 0.148 | 30 | 15 | 35 | 0.114 | works | no | 89.4 | 0.116 |

## The curves

**qwen3b-base-countdown-lr5e-7-s0** (lr 5e-07, 40 steps, 30,720 rollouts, 87 s per step)

| after | validation | test | off-task | answer length |
|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 495 |
| 5 | +0.014 | +0.002 | -0.010 | 484 |
| 10 | +0.011 | +0.023 | +0.043 | 435 |
| 15 | +0.020 | +0.057 | +0.053 | 410 |
| 20 | +0.055 | +0.094 | +0.097 | 382 |
| 25 | +0.080 | +0.098 | +0.087 | 373 |
| 30 | +0.088 | +0.115 | +0.093 | 346 |
| 35 | +0.113 | +0.156 | +0.140 | 305 |
| 40 | +0.121 | +0.135 | +0.143 | 318 |

**qwen3b-countdown-lr1e-6-s0** (lr 1e-06, 40 steps, 30,720 rollouts, 86 s per step)

| after | validation | test | off-task | answer length |
|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 329 |
| 5 | +0.039 | +0.065 | -0.007 | 269 |
| 10 | +0.099 | +0.073 | -0.013 | 225 |
| 15 | +0.134 | +0.132 | -0.017 | 205 |
| 20 | +0.159 | +0.122 | -0.013 | 218 |
| 25 | +0.153 | +0.127 | -0.003 | 232 |
| 30 | +0.141 | +0.195 | -0.017 | 234 |
| 35 | +0.162 | +0.176 | -0.023 | 241 |
| 40 | +0.159 | +0.213 | -0.003 | 240 |

**qwen3b-countdown-lr5e-7-s0** (lr 5e-07, 40 steps, 30,720 rollouts, 89 s per step)

| after | validation | test | off-task | answer length |
|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 329 |
| 5 | +0.025 | +0.017 | -0.013 | 312 |
| 10 | +0.065 | +0.052 | -0.020 | 266 |
| 15 | +0.073 | +0.111 | -0.017 | 248 |
| 20 | +0.103 | +0.093 | -0.020 | 237 |
| 25 | +0.109 | +0.075 | -0.007 | 226 |
| 30 | +0.125 | +0.148 | -0.017 | 221 |
| 35 | +0.171 | +0.114 | -0.007 | 222 |
| 40 | +0.138 | +0.135 | -0.007 | 223 |

## What the arms say

- **The RL side's gate.** The rule's first-step model put the target at step 16
  (instruct) and 20 (base); the reserved block reached it at 15 and between 25 and 30:
  both within the factor of two the preregistration asked for, so RL's side of the
  ratios is calibrated to the same accuracy as ES's. The sign-direction slope measured at
  one point extrapolates over 15 to 30 Adam steps with the KL at 0.04 to 0.05 nats per
  token by the end.
- **The off-task block.** GRPO kept GSM8K where every ES run on these settings lost it:
  −0.007 (instruct) and +0.143 (base) at the end against ES's −0.84 and −0.10 to −0.34.
  The answers shortened under GRPO (329 to 223 and 495 to 318 tokens) where ES's ran to
  the caps. The collapse the ES runs showed is therefore a property of ES's path on
  these tasks, not of the task or the prompts.
- **The measured ratios**, from the loops (`probe_run --compare`, the table in
  `runs/probe-run/README.md`): to the first evaluation at +0.10 on the reserved block, ES
  used 5.3 and 16 times GRPO's rollouts on the instruct model (seeds 1 and 0) and 8 and
  16 on the base model (the probes forecast 2.8 and 17), and 1.16 and 3.50 times GRPO's
  loop seconds on the instruct model, 1.40 and 2.67 on the base model, GRPO at 5e-7 in
  both. The per-answer costs say the same: 0.116 and 0.114 GPU-seconds per GRPO answer
  (generation 7, old log-probs 11, training 71 of 89 seconds per step) against 0.020 to
  0.026 per answer in the ES runs. The two seed-1 ES crossings came within twice GRPO's
  compute with GSM8K intact and deteriorated afterwards; the seed-0 crossings came with
  GSM8K already down. These are threshold crossings on 300-prompt blocks, noisy by a
  factor of about two from the evaluation grid alone. GRPO's KL term against the start
  (beta 0.01) is a confound for the retention difference: ES has no anchor at all.
- **The third arm**, the instruct model at twice the reference step (1e-6): the target at
  step 15 as well against a predicted 8 (the doubled first-step gain), inside the gate by
  a hair; +0.213 on the reserved block at 40, the largest Countdown gain of any run in
  the program, with GSM8K at −0.003 at the end. Its collapse flag is set by one reading
  of −0.023 at step 35, within the block's standard error; the answers shortened to 240
  tokens. At this step GRPO's KL against the start reaches 0.05 nats per token by step
  10 and stays there.

## Limitations

One seed per arm; GRPO's recipe was not tuned for these models (the Tulu run's learning
rate on a 3B model, and 40 steps where the published Countdown runs use hundreds); the
reference's KL floor is 6e-4 nats per token from the f32-under-autocast policy against
the bf16 reference, recorded in the smoke; the prompts are the same 192 the ES runs
used, so neither method saw a fresh prompt during training.
