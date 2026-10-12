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

ES is expected to be useful where (R) is within a small factor of the rollouts RL needs
for the same gain, and RL to dominate where it is not. RL's need is estimated the same
way from the same probe: its first step moves `lr sqrt(d)` along the sign of the
gradient, so its gain per step is that length times the slope `D_sign` along the sign
direction (0.76 per unit length on Tulu), and its rollouts per step are known.

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

## Settings

Each row gets a record directory, its level-0 and level-1 numbers, and the rule's
prediction (works / marginal / fails at a stated `N`, and the task damaged first) written
before any training on it. Training runs (30 to 100 updates at the predicted `N`, scored
on a fresh test block that no probe or run used) are made on a subset, chosen to cover
all three predicted outcomes.

| setting | what it varies | status |
|---|---|---|
| Tulu 8B on the Tulu 3.1 stream | the anchor: mixed tasks, strong start | measured; ES fails |
| Qwen2.5-0.5B-Instruct on Countdown | the anchor: one shared failure, weak start | measured; ES works |
| Tulu 8B on Countdown | same model, one shared failure | probed (margin 6 to 29); training run to do |
| Tulu 8B on one IFEval constraint type | same model and scoring, sharedness raised | to probe |
| Tulu 8B on GSM8K alone | same model and scoring, one task family | to probe |
| Qwen2.5-0.5B-Instruct on Countdown after it writes the format | same task, the shared failure removed | to probe |
| Qwen2.5-3B-Instruct on Countdown | Qiu et al.'s reported success | to probe |
| Qwen3-4B on DeepScaleR | EGGROLL's reported success at 2,048 members | to probe, level 0 and 1 |
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

## Rules carried over

- A validation block sets steps and configurations; a separate test block, never used by
  a probe or a run, scores any trained model (the review's finding F6).
- Responses are called odd and even responses; optima from the quadratic fit are called
  fitted and reported beside an observed step at that length.
- Predictions are committed before data; records are never overwritten; every number in
  a document comes from a record and a script.

## Cost

Level 0 and level 1 on the six unprobed settings, about $100 to 150. Three training runs
at the predicted `N`, about $150 to 250. Stages can stop after any setting. The
literature's reported successes and failures (a search is under way) are the external
check on the rule and cost nothing to use.
