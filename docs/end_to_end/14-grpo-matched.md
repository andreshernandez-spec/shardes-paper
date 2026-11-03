# 14. GRPO on Qiu et al.'s setup, beside S6's ES: preregistration

Written and committed 2026-11-03, before any run. Decided by Andres the same day, after
S6 (`12-qiu-matched.md`) and with the step-decay arm (`13-step-decay.md`).

## Why

S6 put our ES on Qiu et al.'s setup: their problems, token cap, decoding, reward and
evaluator. Our GRPO numbers so far come from our earlier setup (S5: other problems, 512
tokens, our grader), so ES and GRPO have not yet been compared on the same ground. This
run puts GRPO on S6's setup, so the two differ in the method alone, and compares them at
equal answers seen.

## What runs

`es_vllm/qiu_grpo.py` through `es_vllm/qiu_grpo_pod.sh <commit> 0`, one Secure H200, seed 0:

- **Setup, from S6** (`qiu_match.his_setup`): the first 200 problems of `countdown.json`
  for training, the last 2,000 for testing, the raw prompt through the slow tokenizer, his
  `reward_function` on the decoded prompt and answer (the reward ES's fitness averaged), at
  most 1,024 new tokens, Qwen2.5-3B-Instruct at `aa8e725`.
- **GRPO, S5's recipe** (`grpo_run`): temperature 1.0, advantages per prompt over its
  samples, two mini-batches, micro-batches of 2, the clipped surrogate (0.2) plus 0.01 times
  kl3 against the start, AdamW at a constant 1e-6 (the better of two rates on 3B Countdown
  in S5), gradient norm clip 1.0. The policy in f32, vLLM sampling from its bf16 copy.
- **One change from S5:** 50 prompts and 15 samples a step (S5: 48 and 16), 750 answers,
  so that ES's 6,000 answers an update are exactly 8 steps. Each step takes a quarter of a
  seeded permutation of the 200 problems.
- **480 steps**: 360,000 answers, ES's update 60.
- **Tests.** The running test (the 2,000 test problems, greedy by the matched rule in vLLM,
  as S6's running test) after 0, 80, ..., 480 steps, which are ES's updates 0, 10, ..., 60;
  his `eval_countdown.py`, unchanged, on the policy's bf16 weights after 240 and 480 steps,
  scored after training ends.

## Reading, fixed before the data

Primary statistic: D, the mean running test accuracy after 320, 400 and 480 steps minus
ES's after 40, 50 and 60 updates in S6 arm B (31.10, 36.40 and 40.00%, mean 35.83%), at
240,000, 300,000 and 360,000 answers.

- **D ≥ +3 points**: GRPO is ahead of ES at equal answers on Qiu et al.'s setup.
- **D ≤ −3 points**: ES is ahead.
- **Between**: neither is ahead by 3 points.

The threshold: the two evaluators disagree on 126 to 159 of the 2,000 problems on
identical weights, and arm B's running test scatters 0.50 points around its trend. One
seed per method.

Secondary, without a gate: the two curves on one answers axis from 0 to 360,000; his
evaluator on GRPO's 240 and 480 beside ES's running test there; GRPO's level at 360,000
answers against ES's 56.5% at 3,000,000 (his evaluator, S6); answer lengths and the share
cut at 1,024 tokens.

## Prediction

D ≥ +3. On our earlier setup GRPO was ahead of this ES recipe at every equal-sample
milestone, by 5.8 to 11.9 paired standard errors (`runs/s5-early/README.md`). In Qiu et
al.'s own runs their best GRPO baseline tied ES at 0.6M sample evaluations (36.0 against
35.6%, their Figure 7).

## Cost

S5's GRPO took 86 s a step with answers of 250 to 430 tokens; on this setup greedy answers
run 530 to 620 tokens and about a fifth reach the cap. At 130 to 170 s a step, 480 steps
and the tests are about 18 to 23 hours: $100 to 125 on a Secure H200 at $5.29/h, with a
200 GB disk for the 37 GB training state written every 80 steps. If the first 20 steps
project above $150, the run stops. Health checks every 30 minutes.
