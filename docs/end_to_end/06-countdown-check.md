# 06. Countdown implementation check (T2): preregistration

Written and committed 2026-10-06, before any run. T2 of `01-tulu31-plan.md`, kept by
Andres on 2026-09-26 and started after the Tulu pilot found no detectable learning
(`03`, `05`). The question is whether our ES backend (shardes trains, vLLM generates,
`es_vllm/`) learns a task that a reference implementation learns, under the same
hyperparameters. Without that, a null on Tulu cannot be told apart from a broken
backend. The gate below is fixed here and not changed after the data.

## What runs

Two implementations, two seeds each, four runs, all on one GPU type:

- **Reference**: es-at-scale (Qiu et al.'s library) at `574a9d1` (v1.0.0 plus #50,
  which fixes the evaluation schedule: at v1.0.0 the test `iteration+1 % eval_freq`
  never fires, so no evaluation runs during training). Its own `train.py --task
  countdown`, its own venv (vLLM 0.11.0, torch 2.8), one vLLM engine, seeds 1 and 2
  (`--seed 0` becomes 42 in its trainer).
- **Ours**: `es_vllm/run_countdown.py`, the Tulu backend with Countdown's data and
  grader (vLLM 0.30.0, jax 0.11.2, shardes pinned in `requirements.txt`), seeds 1 and 2.

Shared, from es-at-scale's README Countdown example and `train.py` defaults:
`Qwen/Qwen2.5-0.5B-Instruct` at `7ae5576`, population 30, sigma 1e-3, alpha 5e-4
(sigma / 2, its default; our `lr = alpha * sigma`, so both step by `alpha / N` times
the z-scored sum of noise), z-score shaping, all 200 training prompts of
`datasets/train/countdown` scored by every member every iteration, greedy, 512 new
tokens, reward `0.1 * format + answer` from its `countdown_grader.py` (read from the
pinned clone, not copied: the library is under an academic, copyleft license), graded
in a process pool with its 10-second timeout. Evaluation: the current weights on its
2,000 `countdown_eval` prompts, greedy, 512 tokens, before training and after every 5
updates, to 100 updates. (es-at-scale makes one more update after its last evaluation;
it is not scored.)

Qwen2.5-0.5B because shardes' own JAX implementation already learned this task at this
size (E13, `experiments/countdown/results/e13-a100-2026-08-22-clean/`: held-out reward
0.055 at start, about 0.14 after 50 updates, 0.15 at 100, three seeds within 0.01), so
the signal is known to be there within 100 updates. Most of it is format: solve rate
1% to 6%, format score 0.43 to 1.0.

## What differs by design

Each is a property of the implementation, not a setting, and is part of what the check
compares:

| | es-at-scale | ours |
|---|---|---|
| weights updated | vLLM's bf16 parameters, in place; the update is summed in f32 and added in bf16 | an f32 master; the engine gets its bf16 view |
| restore after a member | subtract the same noise (bf16 add then subtract leaves residue) | write the view again (exact) |
| noise | one torch generator seeded per member, reseeded for every tensor, so tensors of one shape get the same draw; drawn in vLLM's fused layout | an independent stream per HF leaf (`leaf_streams`), Gaussian, drawn in bf16 |
| z-score | population std (ddof 0), divided by std + 1e-8 | the same (shardes' `group_relative` on `(N, 1)`), with no epsilon and zeros when every member ties |
| vLLM | 0.11.0 | 0.30.0 |

## Measurement and gate

Primary statistic per run: the mean evaluation reward over the 20 evaluations after 5,
10, ..., 100 updates (the area under the learning curve, so it reflects both speed and
level). Per implementation: the mean of its two seeds, and its spread `S`, the absolute
difference between them.

- **Inconclusive** if either reference run gains less than 0.03 from its start to the
  mean of its last four evaluations (85 to 100 updates): the reference did not learn
  enough to compare against. Then nothing is concluded about our backend.
- **Pass** if `|ours - reference| <= 2 * max(S_ours, S_reference, 0.01)`. The 0.01 floor
  is about one standard error of the difference of two such means at 2,000 prompts
  (eval reward SD about 0.3 per prompt) and the seed spread E13 measured.
- **Fail** otherwise, reported with its sign. Ours below the reference means the backend
  learns worse than a known-good implementation, and no further Tulu spend happens until
  that is found. Ours above it fails the preregistered gate too; it is reported as such,
  with the differences above as the candidates, and the next step is decided with Andres.

Reported whatever the outcome: every run's curve (evaluation reward, solve rate, format
score, response length), the training fitness per iteration, iteration time, and for our
runs the bit-for-bit engine checks (after every restore, member 0 every ten iterations).

## Preflight on each pod

Before the runs, `es_vllm/update_check.py` on the pod at 0.5B, in JAX alone and inside
the engine (step 1 of the plan: the per-leaf update gave different bits from identical
inputs on the Tulu H200 runs). Its result is reported with T2; it does not gate T2.

## Cost

Estimate before measuring: about 4 to 5 hours per run on one A100 80GB, four runs, about
$25-35 at community A100 prices including setup. Two 2-GPU pods, one per seed, each
running both implementations side by side (`es_vllm/countdown_pod.sh`). Iteration time is
read from the first iterations on the pods; a run that projects past $15 is stopped and
the plan revisited. Both drivers were run end to end on the laptop at toy shapes first
(4 members, 2 iterations, 64 tokens); those runs are wiring checks, not results.

## Result, 2026-10-06

All four runs completed (details and curves: `experiments/end_to_end/runs/countdown-README.md`).
**Gate G2 fails, on the high side**: mean evaluation reward over updates 5 to 100 is 0.182
for ours and 0.159 for es-at-scale, a difference of +0.024 against a bound of 0.020; the
reference learned (gains 0.18), so the check is not inconclusive. Our backend learns the
task faster than the reference and reaches the same level by 100 updates (last four
evaluations: 0.234 and 0.239 against 0.229 and 0.220). The most direct candidate for the
speed difference is es-at-scale adding each update into bf16 weights, where an update of
about 1e-4 per element is comparable to one bf16 step; ours keeps an f32 master. As
preregistered, the next step is decided with Andres.


**Decision, Andres, 2026-10-07:** the high-side fail is read as "the backend works". The
longer Tulu arm (`07-tulu31-long.md`) runs next.
