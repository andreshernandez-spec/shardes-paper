# 10. When does ES work: a predictive account, preregistration

Written 2026-10-12, before any new run. Decided by Andres after the review of 2026-10-11
(`drafts/`, local): the program's goal moves from "does ES match RL on Tulu 3.1" to "when
can ES be expected to be effective and when is RL expected to dominate", stated so that
someone can tell a priori, without trial and error. This document fixes the rule to be
tested, the measurements that feed it, the settings it will be tested on, and what counts
as the rule being useful. Predictions for each setting are written into its record before
that setting is trained.

## The rule

Everything measured on Tulu 3.1 and Countdown (`08`, `09`, the review) fits one
expression for ES's progress per update, in the terms of Hoy et al. (2026, Section 6):

    progress per update ~ alpha rho D - c alpha^2 d / N,

where `alpha` is the step, `d` the number of perturbed parameters, `N` the members, `rho`
the correlation between a member's score and its first-order response (how informative
the scores are), `D` the reward's slope along the direction those scores identify, and `c`
the reward's curvature along random directions (how damaging a step is). It is a local,
quadratic, additive model: Section 3 of the draft states its assumptions, and the review
lists the ways it failed over eight updates at 1,024 members. The quantity to predict is
what it implies for the rollouts ES needs to gain `G` while accepting damage `Delta`:

    rollouts ~ P c d G^2 / (Delta rho^2 D^2),                       (R)

with `P` the prompts per member, independent of `N` and `alpha`. On the three settings
measured so far it retrodicts the outcomes: Tulu 8B on Tulu, about 100 times RL's
rollouts for RL's gain (ES did not learn); the 0.5B model on Countdown, a margin of gain
over cost of about 600 (it learned in 100 updates); Tulu 8B on Countdown, a margin of 6
to 29 (untested as a training run). Retrodictions are not predictions; the settings
below are.

ES is expected to be useful where the GPU-seconds (R) implies are within a factor of two
of RL's for the same gain, and RL to dominate where they are not. The two are compared in
both currencies, because the published claims hinge on which is used (`11`: Qiu matches
sample evaluations, EGGROLL wall clock at 11 times the generations, Hoy FLOPs at 15 times
the rollouts): rollouts, and GPU-seconds at the measured cost per answer, 0.048 GPU-s for
an ES answer and 0.255 for an RL answer carrying its share of the gradient and reference
passes (`runs/throughput-bench/`), so ES is compute-competitive up to about five times
RL's rollouts on this hardware. RL's need is estimated from the same probe: its first
step moves `lr sqrt(d)` along the sign of the gradient, so its gain per step is that
length times the slope `D_sign` along the sign direction (0.76 per unit length on Tulu),
and its rollouts per step are known.

## Two levels of "a priori"

**Level 0, no ES at all.** From the start model and the task, without ranking a member:

- *live share*: the fraction of prompts on which 16 sampled answers do not all score
  alike (the prompts either method can learn from);
- *gradient diversity*: the mean pairwise cosine of the per-prompt gradients of the RL
  loss (GRPO's, on the sampled answers), over 64 or more prompts, a proxy for `rho`'s
  shared part;
- *curvature by task*: the even response of the reward to random perturbations of the
  shape ES would use, at two lengths, both signs, split by task, a direct measurement of
  `c` and of which task a step damages first;
- *slope along the sign direction*: the odd response along the unit sign vector of the
  gradient at the length of one RL step, RL's expected first-step gain;
- *rollouts per second* of the inference engine for this model.

About $5 to 10 per setting at 8B on one H200 (the gradients are one backward pass per
prompt; the scans are a few decodes of a validation block).

**Level 1, one ES iteration.** 64 mirrored low-rank members on 192 prompts, their odd and
even responses on a validation block at two lengths, the reliability and cross-prompt
correlation of their scores: `rho`, `D` and `c` directly, by task. About $10 to 20 per
setting at 8B. This is what was run on Tulu and Countdown.

The first test is whether level 0 predicts level 1: gradient diversity against the
cross-prompt correlation of member effects (0.004 on Tulu, 0.30 on Countdown for the same
model), and curvature by task against the task-specific even response.

## Prior predictors to compare with

The published record (`11`) has three a-priori quantities with the same purpose, which
the probes compute alongside ours so the rule can be judged against them: Hoy et al.'s
gradient-aligned fraction `chi = (1 + (N+1) s) / (d + (N+1) s)` and their effective
dimension from the growth of the update's norm; Cho and Han's zero-advantage probability
`q`, which a probe of 200 perturbation pairs predicted to within 0.02; and Liang et al.'s
expected best-of-N improvement `Delta*_N` as an accessibility proxy. None of them, as far
as the record shows, has been tied to the sharedness of per-prompt gradients, which is
the quantity our rule puts first; that is the gap this stage fills or fails to.

## Settings

Each row gets a record directory, its level-0 and level-1 numbers, and the rule's
prediction (works / marginal / fails at a stated `N`, and the task damaged first) written
before any training on it. Training runs (30 to 100 updates at the predicted `N`, scored
on a fresh test block that no probe or run used) are made on a subset, chosen to cover
all three predicted outcomes. The published outcomes in `11` are external anchors: for
those settings the probe's prediction is checked against the paper's result without a run
of ours, and most of them are 0.5B to 4B models, where a probe costs a few dollars.

| setting | what it varies | status |
|---|---|---|
| Tulu 8B on the Tulu 3.1 stream | the anchor: mixed tasks, strong start | measured; ES fails |
| Qwen2.5-0.5B-Instruct on Countdown | the anchor: one shared failure, weak start | measured; ES works |
| Tulu 8B on Countdown | same model, one shared failure | probed (margin 6 to 29); training run to do |
| Tulu 8B on one IFEval constraint type | same model and scoring, sharedness raised | to probe |
| Tulu 8B on GSM8K alone | same model and scoring, one task family | to probe |
| Qwen2.5-0.5B-Instruct on Countdown after it writes the format | same task, the shared failure removed | to probe |
| Qwen2.5-3B-Instruct on Countdown | Qiu et al.'s matched-sample success | to probe; published outcome |
| Qwen2.5-3B base on Countdown | Palleti and Singh's collapse, same task as the row above | to probe; published outcome |
| Qwen2.5-1.5B-Instruct and Llama-3.2-1B-Instruct on Countdown, GSM8K, MATH | Abdi et al.: GRPO ahead on 7 of 8 at matched rollouts, ES far behind only on Llama Countdown | to probe; published outcomes |
| Qwen2.5-0.5B on GSM8K | Cho and Han: no progress at 0.5B, progress at 1.5B and 7B | to probe; published outcome |
| Qwen3-4B on Chemistry and on BoolQ | Hoy et al.: their slowest and fastest tasks | to probe; published outcomes |
| Qwen3-4B on DeepScaleR | EGGROLL's reported success at 2,048 members | to probe, level 0 and 1 |
| Qwen2.5-Instruct 0.5B, 1.5B, 3B, 7B on GSM8K | model size at a fixed task | to probe |
| Qwen2.5-1.5B-Instruct on GSM8K, 200 prompts repeated against a stream | prompt repetition | to probe and run, two seeds |
| dense against rank-1 perturbations on one setting | the perturbation's shape, for `c` | to probe |

## What counts as the rule being useful

- Over at least six settings with a training run or a published outcome, the rule's
  predicted progress per rollout orders them correctly: Spearman correlation at least 0.8
  with the observed order.
- Its three-way prediction (works / marginal / fails at the stated `N`) is right in at
  least five of six.
- Level 0 predicts level 1: gradient diversity and the cross-prompt correlation agree in
  order across settings, and curvature by task names the task damaged first.

Falsified by a setting with low gradient diversity where ES fails at the predicted `N`,
or one with high diversity where it works at small `N`.

## Adjustments after the published record (`11`, 2026-10-12)

- **A first gate on signal.** The live share and Cho and Han's zero-advantage probability
  are reported first; a setting below a threshold fixed per task family is predicted to
  fail before `rho`, `D` or `c` are read. This is the collapse on base models (Palleti and
  Singh: 15% against GRPO's 58% on Countdown) in the rule's terms: no odd response, the
  random part's cost intact.
- **Prompt repetition as an axis.** Every published ES success and our own Countdown run
  train on a fixed set of about 200 prompts passed over hundreds of times; the Tulu runs
  see a stream that never repeats a prompt. The sharedness contrast was measured on
  first-pass prompts in both cases and stands, but the learning dynamics may differ. One
  model and task (Qwen2.5-1.5B-Instruct on GSM8K) is run both ways, a 200-prompt set
  repeated against a stream of fresh prompts, at two seeds.
- **A model-size axis.** Ba et al. and Cho and Han report the population needed falling
  with model size, and Qiu's gains growing with it; the rule has `d` in its cost and
  must reproduce the trend from the measured terms. Qwen2.5-Instruct at 0.5B, 1.5B, 3B
  and 7B on one task (GSM8K) are probed at level 0 and 1.
- **Off-task damage.** The record's failure mode is loss on prior tasks (Schweighofer,
  Abdi, Hoy). The curvature scan and every training run score an off-task block of 500
  prompts from a benchmark outside the stream, and "the task damaged first" includes it.
- **Seeds.** Training runs on models below 7B use two or three seeds; the 8B runs one,
  stated as such.
- **The Countdown calibration run.** Tulu 8B on Countdown, which the probe called
  marginal, runs at Qiu et al.'s settings (30 members, sigma 1e-3, alpha 5e-4, 200 prompts,
  500 iterations, greedy), since they report the near-identical Llama-3.1-8B-Instruct at
  61.2 against GRPO's 50.2 at matched sample evaluations: a published outcome that fixes
  where "marginal" sits.
- **Sigma at two values in level 1**, since `rho` depends on it (0.12 at 5e-4 and 0.26
  at 2e-3 on Tulu, with members losing 1.45 points at the latter).

## Rules carried over

- A validation block sets steps and configurations; a separate test block, never used by
  a probe or a run, scores any trained model (the review's finding F6).
- Responses are called odd and even responses; optima from the quadratic fit are called
  fitted and reported beside an observed step at that length.
- Predictions are committed before data; records are never overwritten; every number in
  a document comes from a record and a script.

## Cost

Level 0 and level 1 on the 8B settings, about $10 to 20 each; on the 0.5B to 4B anchors,
about $2 to 5 each; the whole table about $120 to 180 with the off-task block. Training
runs of our own at the predicted `N`: the Countdown calibration run at 8B about $60 to 80,
the small-model runs with seeds about $10 to 30 each, about $150 to 250 in all. Stages can
stop after any setting. The published outcomes in `11` are the external check on the
rule and cost nothing beyond the probe.
