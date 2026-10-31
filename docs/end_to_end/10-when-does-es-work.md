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

## The probe (`es_vllm/probe.py`, 2026-10-12, before any record)

One tool runs both levels on a setting `model/task` in six phases, each its own process,
and writes `prediction.json` before any training on that setting:

- `sample`: 16 answers per train prompt at temperature 1.0 from the start, scored; the
  live share, Cho and Han's zero-advantage share, the start's greedy reward on the train,
  validation and off-task blocks, answers per second.
- `gradient`: per live train prompt, GRPO's token-mean gradient over the attention and
  MLP matrices (the run's advantages); the sharedness `mean over pairs of g_i . g_j`
  divided by `mean |g_j|^2`, the analogue of the cross-prompt correlation of member
  effects; Yin et al.'s diversity; the summed gradient kept for `sign`.
- `scan`: four random mirrored rank-1 directions of ES's shape at 0.4, 1 and 2.5 member
  lengths (`sigma sqrt(d)`), both signs, on the validation and off-task blocks: the even
  response per length squared is `c`, by task, and names the task a random step damages
  first.
- `sign`: the unit sign vector of the summed gradient (Adam's first direction) at the
  Tulu scan's per-coordinate displacement, both signs: the odd response per unit
  effective length, negated, is the slope along RL's direction; RL's first step
  `lr sqrt(d)` times it is RL's expected first-step gain.
- `members`: 64 mirrored rank-1 members at sigma 5e-4, and 32 at 1e-3 from the same
  seeds, decode the train block greedily: reliability of the ranking, cross-prompt
  correlation of member effects, Liang et al.'s best-of-N improvement.
- `steps`: the 64-member update at the same three lengths, both signs, on the validation
  and off-task blocks: `G` and `C` per unit update, by task.

Blocks: train 192, validation 300, test 300 (reserved, never decoded by the probe; its
digest is in `env.json`), off-task 300 (Countdown settings: GSM8K prompts from the Tulu
stream; GSM8K settings: 150 MATH and 150 IFEval). Rewards are normalized by the task's
maximum (1.1 on Countdown, 10 for the Tulu verifiers). The longest length at which
neither sign moved the pooled reward by more than 0.2 is the one read.

The rule's constants, fixed here: the target gain is a tenth of the reward range at 2%
damage, so ES needs `T = G_target^2 |C| / (Delta G^2)` updates at step
`lambda = Delta G / (G_target |C|)` and `P N T` rollouts, which does not depend on `N`
because `|C|` falls as `1 / N`. RL's reference is the Tulu run: learning rate 5e-7 (its
step of 0.045 over 8.03e9 parameters) and 768 rollouts per step, its gain per step the
first-step gain above. The call is "works" when ES's rollouts are within 5 times RL's
(compute parity at the measured 0.255 and 0.048 GPU-seconds per answer), "marginal"
within 25, "fails: rollouts" beyond; "fails: no signal" when the live share is below 0.1,
before anything else is read. The task damaged first is reported twice, from the random
scan (level 0) and from the update (level 1).

Changed after the first two records (2026-10-12, before any training run; the raw records
are untouched and every prediction is regenerated by the same code, commit noted in
`runs/probe/README.md`):

- Rows are normalized by their own kind's maximum. The analysis had divided a Countdown
  setting's off-task block, scored by the Tulu verifiers on the 10 scale, by Countdown's
  1.1, and that block then dominated every pooled response of the 3B base record.
- `G` and RL's slope are the validation block's, `C` pools the validation and off-task
  blocks. The 3B base record showed why: the update's direction gains on Countdown and
  loses heavily on GSM8K, a directional (odd) effect that pooling had turned into "no odd
  response". The off-task directional effect is now its own number, the off-task change at
  the target gain, `G_target G_offtask / G`, flagged when it exceeds the accepted damage;
  the same is reported for RL's direction.
- The ratio and the call carry a bootstrap over the scanned prompts (1,000 resamples
  within each block), since `G` on the first record was 1.6 standard errors from zero.

First batch, approved 2026-10-12 (about $40 on one H200): `qwen0.5b/gsm8k`,
`qwen3b-base/countdown`, `qwen3b/countdown`, `qwen1.5b/gsm8k`, `qwen3b/gsm8k`,
`qwen7b/gsm8k`, records in `runs/probe/<model>-<task>/`. Expectations written before
the records, from the published outcomes in `11`:

1. `qwen3b-base/countdown`: a live share below 0.1 or the smallest `G` of the batch, and
   a "fails" call (Palleti and Singh's collapse).
2. `qwen3b/countdown`: a "works" call with the largest `G^2 / |C|` of the batch (Qiu et
   al.'s success at 30 members).
3. The GSM8K size axis: the live share falls from 0.5B to 7B as the start's pass rate
   rises; Cho and Han's failure at 0.5B appears as the smallest `G^2 / |C|` on the axis
   and a "fails" call, with 1.5B and 7B at least "marginal".
4. Level 0 predicts level 1: the gradient sharedness orders the six settings as the
   cross-prompt correlation of member effects does (Spearman at least 0.8).
5. Where an off-task reasoning block is scored, it is the task damaged first.

## Results of the first batch (2026-10-12, `runs/probe/README.md`)

Six predictions, written before any training run (the training runs of stage 3 will
score the reserved test blocks):

| setting | live share | sharedness | reliability | G | C | ES/RL rollouts (90%) | call | off-task at target |
|---|---|---|---|---|---|---|---|---|
| qwen0.5b/gsm8k | 0.81 | 0.0037 | 0 | 0.0014 | -2.6e-4 | 50 (6.6 to 823) | fails: rollouts | +0.09 |
| qwen1.5b/gsm8k | 0.77 | 0.0026 | 0.12 | 0.0013 | -1.3e-5 | 2.5 (0.3 to 92) | works | +0.04 |
| qwen3b/gsm8k | 0.45 | 0.0016 | 0.02 | 0.0 | -2.6e-5 | (1.5 to 981) | fails: no odd response | - |
| qwen7b/gsm8k | 0.25 | 0.0071 | 0 | 0.0017 | -4.4e-5 | 13 (2.2 to 55) | marginal | +0.04 |
| qwen3b-base/countdown | 1.0 | 0.0196 | 0.38 | 0.0015 | -1.0e-4 | 17 (4.0 to 46) | marginal | -1.1, flagged |
| qwen3b/countdown | 1.0 | 0.0086 | 0.27 | 0.0023 | -3.0e-5 | 2.8 (0.25 to 12) | works | -0.01 |

## Corrections of 2026-10-13 (after the first training run and a review of the project map)

The review (`drafts/`, local) found the rule's arithmetic and its calls looser than the
record allowed. What changed, all without GPU time; the preregistered `rule` block in
each `prediction.json` is untouched and is what the running stage was prescribed from;
the corrected analysis sits beside it as `rule_net`, dated.

1. **The target is net, per block.** The rule solved for a gross gain of 0.10 against a
   damage of 0.02, so the net it promised was 0.08, and it took the gain from the
   validation block but the cost from the pooled blocks. `rule_net` asks for a net gain
   of 0.10 on the validation block from that block's own odd and even responses, bounds
   each retained block's loss from the random part at 0.02 from that block's even
   response, searches the step over the scanned range instead of solving the quadratic,
   and counts updates for the net gain. On the six records the corrected prescriptions
   are: 3B instruct Countdown step 20 units, 3.8 updates (was 15, 2.9); 3B base 4.0, 19
   (was 3.0, 22); 1.5B GSM8K 18, 5.7 (was 19, 4.1); 7B GSM8K 3.3, 18 (was 7.7, 7.8); 0.5B
   GSM8K 1.0, 90 (was 1.1, 63). The first run's "+0.1 at 10 to 15 updates against a
   predicted 3" compared the observed net on the test block with a gross prescription;
   against the net prescription the observed time to target is 3 to 4 times the
   predicted, not 4 to 5.
2. **A retained block's odd response is a single draw.** The odd response of a block the
   update was not ranked on is `grad R_off . u` with `u = E[u] + noise`: its expectation,
   `grad R_off . E[u]`, is the systematic effect of the training direction on that task,
   which can help or harm, and only the noise part averages away across draws. At 64
   members the noise part is of the systematic part's size, so one draw does not resolve
   the sign (the base model's probe put GSM8K at −0.25 after five updates; the run showed
   +0.09). The corrected rule's retained-block bound uses the even response only; that
   bounds the random part's damage and omits the systematic directional term, so it is
   not a retention bound. The level-1 predictor proposed for the next batch, the
   covariance across members between task fitness and off-task fitness (the members
   decoding the off-task block), is a local diagnostic whose power to predict delayed
   damage is unvalidated. (Corrected 2026-10-14 after a review: an earlier version of
   this item said the odd off-task response averages to zero across draws.)
3. **Calls carry their resolution.** An odd or even response within two standard errors
   of zero makes the call "undetermined", not "works" or "fails"; the bootstrap resamples
   the same prompts for the update and the RL proxy, keeping their covariance, and a
   call whose modal share is under a half is reported as split. Under this rule every one
   of the six settings is undetermined at 300 prompts (four on the cost, two on the odd
   response), which is the honest reading of the first batch: it can forecast ES's
   learning in kind, not call it. Resampling prompts does not stand for independent
   member populations; a second seed of members is the next batch's cost.
4. **The RL side is a proxy.** Every setting inherits the Tulu run's learning rate and
   rollout batch, RL's gain is extrapolated from a finite scan along the sign of a
   reconstructed gradient, and the compute comparison transfers Tulu 8B's cost per answer.
   The ratios are provisional forecasts until a GRPO run of our own, with its learning
   curve and measured cost, exists on at least one cheap setting (S3c, a decision for
   Andres, since the program's rule so far was that the RL side is only released
   checkpoints).
5. **The level-0 gate stays failed** (Spearman 0.54). Replacing the target with the
   update's odd response gives 0.83 on the same six records; that is exploratory and
   frozen here as the hypothesis for fresh settings, on which alone it counts. Published
   outcomes used to formulate the rule remain development evidence.
6. **Outcome classes for S3e, frozen now.** A training run's observed class: "works" if
   the reserved test block's net gain reaches 0.10 at an evaluation where the retained
   (off-task) block is within −0.02; "marginal" if it reaches 0.05 at such an evaluation;
   "fails" otherwise. The class the test block alone would earn is reported beside it,
   and "collapsed" is recorded when any block falls more than 0.02 below its start at any
   evaluation (`probe_run.observed_class`). Amended 2026-10-14, after a review: the
   checkpoint is selected on the validation block (the evaluation with the largest
   validation change among those whose retained block is within −0.02) and the class is
   the test block's change at that checkpoint; "works" means that one evaluated checkpoint
   qualifies, whatever happens later; the class by the test block's own peak is kept
   beside it. On the six runs of S3b the two definitions agree. A run shorter than the prescribed update count cannot measure the
   time to target; it reports the gain per update over the run against the prescription
   (the 0.5B GSM8K prescription is 90 updates against a 30-update run).
7. **The scans.** Extending a scan until the curvature is resolved can leave the region
   where the quadratic model holds: the extension now checks usability on every block
   separately, reports the curvature at the two longest usable lengths and whether they
   agree within two standard errors, and the update scan evaluates the prescribed step
   directly as its own point. The fixed-step runs are completed and reported as
   preregistered; a step schedule or a re-measurement during a run is a separate method,
   to be specified, tuned on validation data and tested on fresh prompts.
8. **The stream guard.** The repeated-against-stream control (S3c) draws fresh prompts
   from a stream that passes over its rows several times; the selector now excludes the
   probe's blocks by row hash and asserts distinctness. The current repeated-prompt runs
   are unaffected.
9. **The literature premise** "ES successes are instruct models with 200 repeated
   prompts" was wrong (EGGROLL's Qwen3-4B-Base on 40,000 DeepScaleR questions; Abdi's
   GRPO wins inside the instruct, single-task category); `11` is corrected. The Tulu 8B
   Countdown calibration run (S3d) is deferred: Tulu DPO and Qiu's Llama Instruct are
   different starts, so it would be a transfer experiment, not a calibration anchor.
10. **The responses are curves, not coefficients** (added 2026-10-15, after the second
    probe of 3B instruct Countdown and S4a, before the other five settings' records were
    read). The update's odd response on the validation block is concave in the step from
    the smallest length scanned: +0.024, +0.032, +0.046 at 3.2, 8.0 and 20 units in the
    first probe; +0.019, +0.016, +0.023, +0.040, +0.038 at 2.8, 4.5, 6.9, 17 and 35 in
    the second (another population, 600 prompts). So `G` per unit depends on where it is
    read: 0.00227 ± 0.00048 at 20 units in the first probe against 0.00110 ± 0.00017 at
    35 in the second, and 0.00228 ± 0.00036 at 17 in the second. The level-1 response
    replicates across populations at a matched length; the rule's reading of it at the
    longest usable length does not, and the quadratic net gain built from it favours
    large steps where the response has saturated, which S4a's step-3.0 and shrinkage arms
    contradict. The even response on the validation block stays within two standard
    errors of zero at 600 prompts and 35 units (−0.023 ± 0.011): one update's cost near
    the start is small and unresolved, and the cost that matters accumulates (S4a). RL's
    sign-direction slope at the shortest usable length is noise-dominated (0.231 ± 0.088
    on 300 prompts, 0.103 ± 0.057 on 600 with the same first 300), the longer length is
    resolved (0.266 ± 0.029, 0.238 ± 0.021) and agrees with it on this setting; on the
    base model the two differ (0.186 against 0.093), a concave response along RL's
    direction too, and GRPO's crossing there (25 to 30 steps) sat between the two
    readings' predictions (20 and 40). Both slopes are now reported
    (`rl_slope_at_longest_usable`); the rule keeps the preregistered reading. An
    exploratory rule from the curves (`rule_curve`: odd and even responses interpolated
    between the scanned lengths, the step with the fewest updates to the net target, the
    retained bound from each block's own even response, the call undetermined unless the
    net gain is positive with both responses at their two-standard-error worst) is
    written to every prediction record beside `rule_net`, labelled exploratory, and is
    frozen for S4b only after review. On the six first-probe records (regenerated from
    the same raw records) it prescribes 2.7 to 7 units on five settings and 21 on 1.5B
    GSM8K, and calls every setting undetermined: a one-update scan on 300 prompts does
    not resolve the net gain at that level. The second probe's expectation 1 is read at
    a matched length (`probe_table --against`, "at the first's length").
11. **Three corrections after S4b** (2026-10-21, from Andres's review of the stage; no
    record changed). The anchored forecast counted the first pull-back twice: the
    prescription reads the responses at the applied step's first displacement, 0.9 of the
    step, and `effective_updates` then weighted them from exponent one; for responses at
    that displacement the weights start at exponent zero (`effective_updates_from_first`:
    4.095 and 3.428 at five updates instead of 3.686 and 2.777; at one update the
    forecast is the measured change at the first displacement itself, the CPU check), and
    the review had mixed the two conventions in its own earlier advice. The frozen S4b
    prescriptions and the P3 bands scored against them are kept; the analysis reports the
    corrected forecast beside them, and new prescriptions use the corrected weights. The
    reserved half of the off-task block is reserved from checkpoint selection only: the
    probe's scans and off-task decodes use the whole block and the curve rule's retained
    bound reads its even response, so it is not an untouched final test; and the class
    admitted checkpoints with unresolved retention and called them "works" from the task
    alone, so the task class and the retention verdict are now separate fields and
    "works" alone means both. The next experiment reserves retained-task test prompts
    that neither the probe nor the selection touches (`Task(..., offtask_test=True)`).
    And the S4b write-up had said both of C's targets exceeded its headroom: with a start
    of 0.937 the +0.10 RL target was impossible and the +0.05 learning target was
    possible at about 79% of the remaining errors; corrected in `runs/s4b/README.md` and
    the results above.

## S3c, the measured RL reference: preregistration (2026-10-13, before any run)

Approved by Andres on 2026-10-13 after the review. `es_vllm/grpo_run.py` trains GRPO on
a probed setting with the Tulu 3.1 run's recipe (its model card: 48 prompts and 16
samples per step at temperature 1.0, group-normalized advantages, two mini-batches per
batch, micro-batches of 2, the token mean of the clipped policy term plus 0.01 times the
kl3 estimator against the start, AdamW at a constant 5e-7, one epoch), on one GPU with
the policy in f32, a bf16 copy of the start as the reference and vLLM generating from the
policy's weights after every step; the prompts cycle through the probe's 192 train
prompts; the validation, reserved test and off-task blocks are scored every 5 steps, as
the ES runs were. Each phase is timed, so the record carries RL's cost per step and per
answer on the same hardware, which the ratios so far took from the Tulu 8B bench.

Arms, in order, one seed each, 40 steps (30,720 rollouts, about two and a half ES
iterations' worth): 3B instruct on Countdown at 5e-7; 3B base on Countdown at 5e-7; 3B
instruct at 1e-6 if the balance allows. About $8 to 10 each.

Predictions from the probes' RL side (lr 5e-7, 768 rollouts per step, the sign-direction
slope on the validation block): the instruct model reaches +0.10 on the reserved block
after 16 steps, the base model after 20; their off-task change at that point −0.006 and
−0.028. The ES runs reached +0.10 at 5 to 15 updates (61k to 184k rollouts, instruct)
and 15 to 30 (184k to 369k, base), so the measured ratio of rollouts is those against
768 times RL's measured steps; the probes forecast 2.8 and 17. The compute currency uses
the measured seconds per answer of each method on this GPU.

What counts: RL's measured steps to +0.10 within a factor of two of the prediction on
each arm (the RL side's first-step model), and whether RL's off-task block holds where
ES's collapsed. Falsified if RL needs more than twice the predicted steps, or fewer than
half, on both arms.

## S4a, the collapse and the resolution: preregistration (2026-10-14, before any run)

Approved by Andres on 2026-10-14. Every Countdown ES run lost its off-task block late,
at either step and on every seed, while GRPO on the same prompts did not. S4a asks what
the collapse is, on the one checkpoint and task where it is best measured (Qwen2.5-3B-
Instruct on Countdown, the probe's blocks, 64 members at sigma 5e-4 on the 192 train
prompts, scored every 5 updates), by changing one thing at a time against the two
finished runs at step 15.3 (`runs/probe-run/qwen3b-countdown-s0`, `-s1`):

| arm | what changes | what it isolates |
|---|---|---|
| step 3.0 | the step, a fifth; population, sigma, prompts, horizon and scoring as before | the step itself (the instruct-against-base contrast changed the checkpoint too) |
| random-only, step 15.3 | the real fitnesses in a seeded random order: an update of the real norm and no direction | the random walk alone |
| 256 members, step 15.3, 20 updates | the random part's norm halved per unit, the useful part as before | the random part's size |
| shrinkage 0.1, step 15.3 | after each update the weights move a tenth of the way back toward the start | an anchor, ES's analogue of GRPO's KL term |
| GRPO 5e-7, seed 1 | a second population of samples | the reference's seed variance |
| GRPO 5e-7, beta 0 | the KL term removed | whether the anchor is what keeps GRPO's off-task block |

Each arm's `run.json` carries the instruct probe's prediction block; the ES arms' records
go to `runs/s4a/`, the GRPO arms to `runs/grpo-run/`. One seed each; about 19 hours on
one H200, about $100 by uptime (the ES arms about 2.7 hours each, the 256-member arm
about 8.5 hours, the GRPO arms about one hour each).

Predictions, from the instruct probe record (`G` 0.00227 per unit, `C` −3.0e-5 per unit
squared on the validation block, `C` −1.2e-5 on the off-task block) and the finished
runs:

1. Step 3.0: the first-order model holds as it did for the base model at this step:
   validation +0.20 at 30 (gain 0.204, random cost 0.008), the reserved block crossing
   +0.10 between 15 and 30. If the collapse follows the task gain, as both settings
   suggested (it began where the gain reached about +0.08 to +0.10), GSM8K starts down
   between 15 and 25 here too; if it follows the step, GSM8K stays within −0.05 at 30.
2. Random-only: no gain, validation −0.007 per update from the random cost (−0.2 at 30).
   If the collapse is the random walk, GSM8K falls below −0.20 by 20 to 30 as the real
   runs did; if it is the learned direction, GSM8K stays within the random part's own
   cost, about −0.08 at 30.
3. 256 members: the task gain per update as at 64 (+0.035 first order, the target by
   update 5) with a quarter of the random cost. If the collapse is the random walk, it is
   delayed past 20; if it is the direction, it comes at 15 to 20 as at 64 members.
4. Shrinkage 0.1: the displacement is bounded near ten updates' worth, the task gain
   saturates near +0.15; if the collapse is accumulated drift of either kind, GSM8K is
   held above −0.20 at 30 and the task gain is kept; if it is the direction's local
   effect, the anchor only delays it.
5. GRPO seed 1: the reserved block reaches +0.10 within a factor of two of 16 steps,
   GSM8K within −0.05 at 40. GRPO beta 0: the same learning; GSM8K within −0.05 at 40
   if the KL term is not what keeps it, below −0.20 if it is.

What counts: the collapse is called the random walk if arm 2 collapses and arm 3 is
delayed; the learned direction if arm 2 holds and arm 3 collapses on schedule; mixed
otherwise, and then the shrinkage arm says whether an anchor is enough. The outcome
classes and the compute comparison are read as in S3b and S3c.

### S4a results (2026-10-15, `runs/s4a/README.md`)

Six arms, 16 hours on one H200, about $85. The gate: the random-only arm collapsed
(GSM8K −0.083 at 10, −0.470 at 20, 0.000 from 25, the answers at the cap, with no
ranking signal at all) and the 256-member arm was delayed (GSM8K within −0.053 through
20, where both 64-member seeds had lost the block). **The collapse is the random walk**:
the update's random part, accumulated over updates, whose cost on the off-task block
grows as the model moves (the one-update curvature predicted −0.028 at 10 against
−0.083) and reaches a cliff. The step-3.0 arm (a fifth of the prescription) kept GSM8K
within −0.04 for 30 updates and reached +0.185 on the reserved block, with the
first-order model's +0.196 at 30 against +0.194 observed; the shrinkage arm kept GSM8K
within −0.07 at the full step and reached +0.235 at 25, the program's largest gain, for
1.15 times GRPO's loop seconds to the target. GRPO's second seed crossed at 20 (within
the window) and GRPO without its KL term kept GSM8K (−0.020 at 40) and learned more
(+0.191), so the KL anchor is not what separates GRPO's retention from un-anchored ES's.

Against the predictions: 1 held on its step-following branch (crossing earlier than
written, at 10); 2 held on the random-walk branch; 3 held (target by 5, delayed past 20),
with the task gain saturating near +0.2 from update 5 against the first-order +0.55 at
20; 4 held on retention and missed on the gain (+0.235, not a saturation near +0.15; the
anchored first-order model gives +0.27 at 30, the written figure was rougher); 5 held
on both GRPO arms. Compute to the target against the reference arm, from the loops:
3.4 times GRPO's seconds at step 3.0 (16 times the rollouts), 4.3 at 256 members (21),
1.15 with shrinkage (5.3).

Two corrections for what follows. The frozen class's retained-block bound of −0.02 is
below the off-task block's standard error (0.021), so it fails the 256-member and
shrinkage arms on readings between −0.03 and −0.07; those readings are not within two
standard errors of the start either (the shrinkage arm's −0.067 at 25 is 3.6 paired
standard errors below it, the 256-member arm's −0.053 at 10 is 3.1), so what the two arms
show is a loss held within 0.07 while the un-anchored runs lost everything, a trade-off,
not retention. The S4a classes stand as frozen; S4b separates the acceptable loss (a
fixed margin) from the measurement's precision (a paired interval), see its design.
And the rule's retained-block bound, derived from the even response of one update, is
not what limits a run: the random part's accumulated damage is, which the shrinkage,
step and member count control; a prescription has to carry one of them, and the second
probe says what a probe can predict about it.

The second half of S4a, GPU time permitting after these: the instruct and base members
decoding the off-task block for the covariance diagnostic (`probe --members-offtask`),
and a second member population with 600 validation prompts on the six probed settings,
so the corrected rule can call something at two standard errors.

### S4a, second half: the second probe (preregistered 2026-10-14, before any run)

The six settings probed again (`runs/probe2/<model>-<task>/`), every phase, with the
member population of seed 1 (the scan's four random directions are its first four pairs),
600 validation and 600 off-task prompts whose first 300 are the first probe's blocks and
the same 192 train and 300 test prompts (`env.json` carries the prefix digests), and the
scans extended until the even response is resolved (commit c945036, which the first
batch's records predate). On the two Countdown settings the members also decode the
validation and off-task blocks (`--phase members-offtask`, `level1_offtask` in
prediction.json): per mirrored pair, the odd part of the pair's change on a block is its
direction's slope there, so the update's first-order change on each block is the sum of
the pairs' odd effects weighted by the update's coefficients, the quantity the `steps`
scan measures directly, here decomposed over the members; the correlation across pairs
between the train odd effect and the off-task one says whether the directions that rank
well are the ones that move the off-task block. Run on the S4a pod after the six arms,
about 6 to 8 hours (the full probe at 600 prompts is about an hour per setting, the
off-task decodes about half an hour each), about $35 to 45 by uptime.

Expectations:

1. The second population's `G` on the validation block is within two standard errors of
   the first's on every setting (a population property of the setting, not of the draw),
   and the direct `G` from the `steps` scan agrees with `G_from_members` on the validation
   block within two standard errors of the latter (the decomposition is consistent).
2. The extended scans resolve the cost on the Countdown settings and on 7B GSM8K, so the
   corrected rule calls them: instruct Countdown works or marginal, base Countdown
   marginal, 0.5B GSM8K fails on rollouts, 7B GSM8K marginal. 3B GSM8K stays
   undetermined (no odd response at 300 prompts; 600 halves nothing that is zero), and
   RL's slope stays under two standard errors there.
3. The off-task diagnostic on the base model shows a resolved negative correlation
   between the train and off-task odd effects (its first probe put the update's single
   draw at −0.017 per unit on GSM8K, and its run found a direction that broke GSM8K).
   On the instruct model the correlation is unresolved at one update: the run's collapse
   came after the task gain reached +0.08, and a local diagnostic at the start does not
   see it. If the instruct correlation is resolved and negative, the learned direction
   carries the damage from the start, and S4a's arms 2 and 3 should say "direction".

What counts: the diagnostic is validated only where it agrees with S4a's gate on the
same checkpoint; a resolved negative correlation with a "random walk" verdict from the
arms, or an unresolved one with a "direction" verdict, leaves it a description of the
members and not a predictor. Expectation 1 failing on the odd response means the
level-1 numbers are draws, and the rule needs populations, not one.

#### The second probe's results (2026-10-15, `runs/probe2/README.md`)

Six settings, 5.6 hours on the S4a pod, about $30. Expectation 1 held at a matched
length on six of six settings (the level-1 odd response replicates across populations
where it is read at the same length; where the rule reads it, at the longest usable
length, two settings miss because their second scan ran into the saturated part of the
curve, item 10), and the members' decomposition agrees with the scan at the small-step
end of its curve. Expectation 2 failed: the extended scans resolve the random cost on
the GSM8K settings, where the odd response is now absent (0.5B, 1.5B and 7B at
reliability 0 on this population, within noise at every length; 3B GSM8K's negative,
−0.022 ± 0.009 at 37.5 units, so it is called "fails: no odd response"), and not on the
Countdown settings, where one update's cost on the validation block stays within two
standard errors of zero at 600 prompts (−0.023 ± 0.012 on instruct at 35 units; +0.015
± 0.009, a gain, on base at 17). Nothing moved from undetermined to a call except 3B
GSM8K. The first probe's GSM8K odd responses (0.0013 to 0.0017 per unit on 0.5B, 1.5B
and 7B) were draws at the high end: the training runs on 0.5B found no held-out gain,
and this population finds none at one update. RL's slope replicates within two standard
errors on every setting. The level-0 test fails again (Spearman 0.43).

Expectation 3 held for the instruct model (the train-versus-GSM8K correlation across
pairs 0.19, 90% −0.07 to 0.43) and failed for the base model (−0.05, −0.29 to 0.22,
not the resolved negative value expected): neither population's ranking carries GSM8K
damage, and the base model's first-probe "destructive direction" (−0.017 per unit) was
the seed-0 update's own random part (that update lowered GSM8K by 0.07, 0.18 and 0.29 at
2.7, 6.8 and 17 units; the seed-1 update by −0.03, 0.00 and +0.02). The diagnostic's
correlation is not a predictor of the collapse. What the decodes do give, read after
the fact and on two settings: the spread across directions of a random rank-1
perturbation's effect on the off-task block, with the sampling part removed, is 0.091 of
the range on the base model (twelve times its effect on the train block) and 0.003 on
the instruct model (a third). That is the random walk's lever on each checkpoint,
measured at the start with nothing ranked, and it agrees with S4a's verdict and with
the runs (the base model lost GSM8K at step 3.0, the instruct model did not). It enters
S4b's preregistration as a candidate predictor to test prospectively: an off-task block
whose per-direction spread exceeds the task's is predicted to collapse at the task's
prescribed step, and one below a third of it to hold.

The second half's own conclusion for the rule: with 64 members and 600 prompts, a
one-update probe resolves the odd response where it is large (Countdown) and the random
cost where it is large (GSM8K), never both on one setting, and never the net gain at
its two-standard-error worst (the exploratory curve rule calls one setting, the base
model, from a first-update response the run did not sustain). A call at two standard
errors needs either more rollouts in the probe than the rule allowed itself or a
different quantity: the response curve read at the step a run would take, and the
off-task block's per-direction spread for the collapse.

## S4b, fresh settings for the verdict: design (2026-10-15, decided after review; nothing run)

The stage is the prospective test of anchored ES and of the candidate predictors on
settings not yet probed. It is not a validation of the rollout-efficiency rule, whose
call is reported and not gated. The five choices were laid out with recommended defaults
on 2026-10-15 and reviewed by Andres the same day; his decisions are the design below,
with the earlier option noted where it was changed. Nothing is preregistered until the
settings' prescription records are written before their runs.

**1. Prescription.** The anchored curve-selected step: the curve rule's step `λ` (the
fewest updates to the net target on the interpolated responses) with shrinkage 0.1
after every update, on every setting. The anchor control, on A: a second arm at the same
step and seed with shrinkage 0, so the pair varies the anchor alone (the earlier
proposal, a small un-anchored step against the anchored large one, varied the step and
the anchor together; the small step stays an optional third policy and is not run in
this batch). The fallback, frozen now: a setting whose curve gives no positive net gain
or no feasible step runs at the smallest scanned multiple (0.4 member lengths, about 2.8
units) with shrinkage 0.1, so that its prediction, "does not learn" or "undetermined",
is tested at a defined step. The applied update of `λ` followed by the pull-back leaves a
first displacement of 0.9 `λ`; that is the point the probe measures (choice 5).

**2. Predictions.** Four, from the probe's record, written before the run.

| prediction | from | the call | scored as |
|---|---|---|---|
| P1 learns | the odd and even responses at the applied step's first displacement (0.9 `λ`) | learns: the odd response positive at two standard errors and the net (odd plus even) positive; does not learn: the odd response negative at two standard errors; undetermined otherwise | learns: the reserved block reaches +0.05 within 30 updates; does not: it stays under; undetermined: reported, not scored |
| P2 collapses (exploratory) | the off-task block's per-direction spread over the train block's, both noise-removed, with the ratio's interval from a bootstrap over pairs and prompts | collapses: the interval's lower bound above 1; holds: its upper bound below a third; abstain when either spread is unresolved or the interval straddles a threshold | reported with its coverage (settings with a call) and accuracy, separately from the gate: a new, method-dependent hypothesis whose thresholds came from un-anchored runs, tested here under the anchored policy (the instruct ratio 0.30 sits at the cutoff, and that model collapsed un-anchored at the large step despite it) |
| P3 pace | the responses at 0.9 `λ` with the anchor's geometric weights (`probe_run.effective_updates`: five anchored updates retain 3.69 updates' worth of first-order gain and 2.78 of the random part's variance) | the validation change after five updates: 3.69 odd + 2.78 even | within a quarter to once the prediction (S4a's anchored arm: observed +0.082 against +0.108; the un-anchored arms 0.4 to 0.6 of theirs) |
| P4 RL's pace | the sign-direction slope at the shortest usable length, where it is resolved at two standard errors (else no prediction) | GRPO reaches +0.10 on the reserved block at 0.1 over the first-step gain | within a factor of two of the predicted step; a target predicted within 20 steps and not reached by 40 is a miss, one predicted beyond 20 and not reached is undetermined; one count per setting (seeds and learning-rate arms are not counts); the denominator is the settings with a prediction |

Gates: P1 and P4 are gated, right on every setting with a call (A counted once); P2 and
P3 are reported. The stage reports measured cost, learning and retention on every
setting whether the predictors succeed or fail. Success on P1 supports the behavioural
prediction under the frozen training policy and does not establish the rollout-
efficiency rule. The decision on S4c is taken on these results and is not conditioned
on P2; three fresh settings are a prospective pilot, and a broader verdict needs more
independent settings.

**3. Settings.** The first batch is A, C and D: the cheap Countdown setting, an
instruction-following task and a reasoning task, with replication on A.

| setting | arms | about |
|---|---|---|
| A. 1.5B instruct Countdown | the probe twice (populations 1 and 0); anchored ES at seeds 0 and 1; the un-anchored control at seed 0; GRPO | $40 |
| C. 3B instruct, one IFEval constraint type | the probe; anchored ES; GRPO | $29 and the Task branch |
| D. 3B instruct, MATH alone | the probe; anchored ES; GRPO | $29 and the Task branch |

About $100, about $130 with setup and polling. C's constraint type is chosen before any
learning signal is inspected, by the count of unique examples: at least 1,092 rows of
the type in the stream (192 train, 600 validation, 300 test). **Feasibility, checked
2026-10-15: no type has them.** The stream's IFEval rows are spread over its constraint
types almost evenly, and the stream cycles through the training set, so the largest type
has 644 unique rows in all ("Multiple Sections"; "Quotation" 638, "Keyword Frequency"
638), far short of 1,092. The rows are base prompts (12,245 distinct across the IFEval
rows) with the constraint sentence attached and a single verifier record per type, so a
single-type setting of the required size can only be constructed (the type's sentence
attached to base prompts the stream pairs with other types, verified by the same
function), which is a generator of the kind E was deferred for, or the blocks shrunk to
what one type has (192, 300 and 150: 642 rows, within one of the largest type's count),
or the setting changed to IFEval with every type (the family, as the GSM8K settings were
a family), which no longer raises the sharedness the setting was for. Which of these,
or none for this batch, is a decision for the review; D needs no such choice (MATH alone
has 2,343 unique rows in the first 200 steps, and its `Task` branch exists).
**Decided 2026-10-16: construct it** (`quotation_rows` in `probe.py`, setting
`qwen3b/quotation`). A third of the stream's IFEval rows carry their constraint's
canonical sentence verbatim, appended (816 of 4,768 in the first 200 steps) or
prepended (791); the rest word the constraint differently and are not used. From the
appended rows the base prompt is the text before the sentence, kept with the separator
that row used, and the Quotation sentence ("Wrap your entire response with double
quotation marks.", whose verifier record has no per-row parameter) is appended; one row
per base prompt in the stream's order of first appearance, 1,376 rows from one pass over
the training set (600 steps; 400 give 1,091, one short). Where the stream itself has the
base prompt with the Quotation sentence appended, the constructed row reproduces it
byte for byte: 332 of 332. Off-task blocks GSM8K and MATH. The setting therefore tests
ES on a constructed single-constraint task, not on a slice of the stream, and its
records say so; the type was chosen before any learning signal was inspected, by the
rule that its verifier takes no per-row parameter and that it has the most stream rows
among such types. Deferred until their
feasibility is specified: B (7B instruct Countdown: the GRPO driver's f32 policy, its
gradients and Adam states, the bf16 reference and vLLM's share do not fit one H200, so
it needs a memory plan, such as a bf16 policy with an f32 master off the GPU or 8-bit
optimizer states, and revised timings before a cost can be quoted) and E (the Countdown
stream control: the source has 200 unique training problems and 30 fresh batches need
5,760, so it needs a generator with difficulty controls and disjointness checks against
every block, and GRPO on the same stream, before it is a control at all). F (Tulu 8B
Countdown) stays with S4c and is a transfer experiment, Tulu DPO not being Qiu et al.'s
start.

**4. Retention.** A fixed margin of 0.03 of the normalized reward, the practical limit
of acceptable loss, assessed on each retained task family separately (GSM8K, MATH and
IFEval, each where it is a retained block, so that a gain in one cannot cancel a loss in
another) with the paired confidence interval of the change (the per-prompt change
against the evaluation at update 0, as `probe_run.analyze` computes it): within the
margin if the interval's lower bound is above −0.03, lost if its upper bound is below
−0.03, unresolved otherwise. Precision decides whether a reading is resolved, not how
much loss is acceptable. The checkpoint is selected on the validation block among
evaluations within the margin on the retained-task validation half (the first 300
off-task prompts, the first probe's block); the final retention assessment uses the
reserved second half (the 300 prompts the second probe's design added), which selection
never touches. Historical classes are unchanged. The S4a text that said the anchored
arms kept GSM8K "within two standard errors" was wrong: the shrinkage arm's losses at
15, 20 and 25 (−0.063, −0.057, −0.067) are 3.2 to 3.7 paired standard errors below the
start and the 256-member arm's −0.053 at 10 is 3.1; those arms kept GSM8K within 0.07,
a trade-off, not retention within the margin (corrected in the S4a results above).

**5. Probe.** As the second probe (600 prompts, the population of seed 1, the scans
extended until resolved, the members decoding the off-task block on every setting),
with three additions: the steps scan evaluates the applied step's first displacement,
±0.9 `λ`, as a direct point; the noise-removed spreads carry an interval from a bootstrap
over both pairs and prompts, and an unresolved corrected variance is reported as such;
and A is probed with a second population (seed 0). The prescription is a frozen record
per setting, `prescription.json`, holding the method, the step, the shrinkage, whether
the fallback was used, and P1 to P4 with their values, written before the run and
stamped into the run's record; `probe_run` reads it (today's loader reads the legacy
`rule` block and has to change).

Implementation, done 2026-10-15 without a GPU (commit on the branch): the MATH `Task`
branch; the prescription record (`prescribe` in `probe.py`, written beside
prediction.json by the steps phase: the method, the step, the shrinkage, the fallback,
the responses at the first displacement and P1 to P4) and its loader in `probe_run`
(the prescription sets the step and the anchor; `--shrink 0` is the un-anchored
control); the ±0.9 `λ` point in the steps phase; the spread ratio's interval over pairs
and prompts; per-family retention with the paired interval on the selection and
reserved halves of the off-task block (`probe_run.retention`, beside the frozen class).
A dry run of the prescription on the second probe's records: 3B instruct Countdown
learns at a 17.3-unit step (first displacement 15.6; P2 no call, the spread ratio's
interval 0.07 to 0.94 straddling the cutoff; P3 +0.097 over five anchored updates, which
the S4a shrinkage arm's +0.082 sits inside; P4 none, the slope unresolved at 600
prompts); 3B base Countdown learns and collapses (ratio 5.7 to 18.4; P4 21 steps, GRPO
crossed at 25 to 30); 0.5B GSM8K falls back to a 3.2-unit step and is undetermined.
C's `Task` branch is the constructed Quotation setting above (2026-10-16). The probes
(A twice, C, D) ran on 2026-10-16 in one pod session (5.1 hours, about $27;
`runs/s4b-probe/README.md`); their prescription records are the frozen predictions for
the run session:

| setting | step | P1 learns | P2 collapses | P3: validation after 5 anchored updates | P4: GRPO step to +0.10 |
|---|---|---|---|---|---|
| A, 1.5B instruct Countdown (population 1) | 3.0 | learns | no call | +0.071 (band +0.018 to +0.071) | 23 (11.5 to 46) |
| C, 3B instruct Quotation | 6.85 | learns | no call | +0.048 (+0.012 to +0.048) | 9.7 (4.8 to 19.4) |
| D, 3B instruct MATH alone | 3.1, the fallback | undetermined | no call | +0.032, reported | none (RL's slope unresolved) |

Three things the probes showed before any run. A's two populations disagree in kind:
population 1's update helps only at small steps (its validation odd response peaks at 3
units and turns negative by 37) while population 0's helps up to 35 units (+0.031 at 15,
resolved at ten standard errors), so their prescriptions would be 3.0 and 15.8 units;
the second probe's replication at a matched length fails on this setting, the level-1
response being a property of the population there. The design's prescription
(population 1's) stands, and the seed-0 anchored arm, which runs population 0's members
at population 1's step, reports by its pace against P3 whether that step was too small
for it. P2 abstained on all three settings: the instruct checkpoints' off-task blocks
barely move under a random rank-1 direction of a member's length (spreads at or below
the sampling noise), so the exploratory predictor's coverage on this batch is zero, a
result about the predictor at this probe size. C's start already scores 0.90 on its
train block and 0.93 on validation, so its +0.05 threshold sits near the ceiling; the
prediction stands as written. Then the ES and GRPO arms in a second session.

#### The run session (2026-10-16 and 2026-10-21, `runs/s4b/README.md`)

Eight arms in two parts (the three anchored ES arms on the 16th, then a stop at
Andres's request; the un-anchored control, the second seed and the three GRPO arms on
the 21st), 16.8 pod hours, about $93 including a pod lost to slow downloads.

| | A, 1.5B instruct Countdown | C, 3B instruct Quotation (constructed) | D, 3B instruct MATH alone |
|---|---|---|---|
| ES anchored, reserved block at 30 | +0.148 and +0.147 (two seeds); the un-anchored control +0.180 | +0.027 (peak +0.037) | +0.017 |
| GRPO, reserved block at 40 | +0.192, +0.10 between steps 20 and 25 | +0.033 | +0.003 |
| P1 learns | held (three arms) | missed | unscored |
| P3 pace | 1.16 and 1.20 of the forecast (just above the band); the control 0.93 of the un-anchored one | below the band | reported |
| P4 RL's pace | held (step 23 predicted, band 11.5 to 46) | missed (9.7 predicted; never reached) | no prediction |
| P2 collapses | no call; GSM8K +0.025, +0.025, −0.025 | no call | no call |
| compute to +0.10, ES over GRPO | 1.1 to 1.2 in loop seconds, 6.4 in rollouts | neither reached it | neither reached it |

A is the clean case for the method: three ES arms learned at the 3.0-unit anchored
step, reached the target at 1.1 to 1.2 times GRPO's loop seconds, kept GSM8K within the
margin on one anchored seed (unresolved on the other and on the control), and GRPO
crossed the target where the sign slope said; the anchor cost about 0.03 of task gain
against the control and left GSM8K at +0.025 against the control's −0.025 at 30, within
noise either way. The forecast on A reads as follows after item 11 (the frozen
prescriptions counted the first pull-back twice): with the corrected weights the
forecast from one update's responses matches both seeds through update 10 (+0.082 and
+0.123 against +0.082 to +0.085 and +0.108 to +0.116) and overshoots afterwards, by 0.019
to 0.026 at 30 (+0.167 against +0.148 and +0.141); the claim written after the first arm,
agreement within 0.015 at every evaluation, rested on the double-counted pull-back and
is withdrawn. C is the ceiling case: the start scores 0.937, leaving 0.063 of headroom,
so the +0.10 RL target was impossible and the +0.05 learning target needed about 79% of
the remaining errors recovered; the probe's +0.023 response (3.3 standard errors) did not
continue for either method, and both gated predictions miss there as scored. D is the
no-signal case: an undetermined call, no RL prediction, and neither method moved the
task, which is not a successful negative prediction, since GRPO barely improved there
either. The gated predictions stand at P1 one of two settings with a call and P4 one of
two settings with a prediction, both misses on C; P2 made no call on any setting, so it
is untested, not failed. The quadratic rule's calls on the same records were wrong on A
("fails: no odd response", population 1) and C ("works"); the exploratory curve rule
prescribed the steps that worked on A and called C undetermined. The prescription itself
cost about 3,100 s of pod time on A (1,300 s without the off-task decodes that feed P2),
against GRPO's 1,635 s to the target, a cost reported beside the loop seconds and not
inside them. A's two populations prescribed 3.0 and 15.8 units; that one probe chooses
the step reliably is not shown, and it is the more immediate uncertainty.

What this batch says to the rule: a one-update probe read at the step a run will take
forecasts the first ten updates of an anchored ES run on a setting with a resolved
response and overstates the later ones; it says nothing useful where the task is at its
ceiling (C) or the response is absent (D). The next preregistration states the learning
targets as fractions of the start's headroom with a minimum detectable effect, reserves
a retained-task test block the probe never decodes, and needs a setting where the
off-task spread predictor makes a call before that predictor is anything.

#### After S4b: the order of work (Andres's review, 2026-10-21)

1. Correct and regenerate the analyses (done, item 11) and update the manuscript around
   the completed comparisons: ES's underperformance on Tulu, its learning on Countdown
   with the measured stabilization trade-off, and the compute comparisons, without a
   universal predictor as a condition for the paper. Done the same day:
   `papers/end_to_end/draft.md` carries the Countdown comparisons as Section 9 (Table 2,
   the collapse's mechanism, the 1.5B checkpoint with the corrected forecast, the two
   settings that bound what a one-update measurement says) and the measured compute in
   Section 8, every number from the READMEs of `probe-run`, `grpo-run`, `s4a`, `s4b` and
   `probe2`; the title and the abstract's wording are candidates for Andres.
2. A second GRPO seed on A, and the matched anchored and un-anchored ES pair on A
   extended from 30 to 60 updates with the horizon fixed before any new result is read:
   whether near-parity and the retention trade-off persist. About $20 to 30, more if the
   answers lengthen. If both arms stay stable the anchor was unnecessary on this
   setting; if only the anchored one does, the retention mechanism is clearer.
3. One fresh, non-saturated task with two probe populations and matched ES and GRPO
   arms: the transfer of the corrected prescription, with the headroom measured before
   the target is frozen and a minimum detectable effect stated.
4. S4c's published-recipe replication is a separate objective, deferred; if explaining
   Qiu et al.'s result becomes the question, a short faithful reproduction of their
   recipe comes before the full horizon.

Superseded in part the same day: with the paper's goal restated as the rule itself
(S5 below), items 2 and 3 fold into S5's case set.

## S5, the rule as a checklist on more cases: design for review (2026-10-21, nothing run)

Andres, 2026-10-21, after the manuscript update: the paper's goal is a rough rule, a
checklist, for deciding before the compute is spent when ES is more data-efficient than
GRPO; the title and the paper follow from it, not from the Tulu comparison alone. The
rule cannot be tested exhaustively, but it has to be tested on more than two cases, with
both outcomes represented, not one setting where ES does well and one where GRPO does.
This section inventories what the records support today, states the checklist to freeze,
proposes the cases, the gates and the cost, and lists the decisions. Nothing here is run.

### What the records support today

Cases with both methods measured, in the protocol of S3b to S4b (64 mirrored rank-1
members at sigma 5e-4, 192 prompts repeated, 30 updates; GRPO with the Tulu 3.1 recipe
at 5e-7, 40 steps of 768 rollouts; the target +0.10 on the reserved block), plus the
Tulu stream in its own protocol:

| case | ES | GRPO | rollouts, ES/GRPO | loop s, ES/GRPO | the rule's call before the run | right? |
|---|---|---|---|---|---|---|
| Tulu 8B DPO on the Tulu 3.1 stream (16 dense members; released GRPO) | fails (−0.25 after RL's data; +0.14 pooled at 512 low-rank members over 8 updates at 32 times RL's rollouts) | +1.87 | about 100 (retrodiction) | about 15 GPU-s per unit gain at 512 | fails (retrodiction, `10` "The rule") | yes, in kind |
| Qwen2.5-3B-Instruct Countdown | learns at 5 to 15 updates; GSM8K lost late unless anchored | learns at 15 to 20 steps, GSM8K kept | 5.3 and 16 | 1.16 and 3.50 (anchored 1.15) | works (ES/RL rollouts 2.8) | in kind; size off 2 to 6 |
| Qwen2.5-3B base Countdown | learns at 15 to 30; GSM8K lost late | learns at 30, GSM8K +0.14 | 8 and 16 | 1.40 and 2.67 | marginal (17) | in kind |
| Qwen2.5-1.5B-Instruct Countdown (anchored) | learns at 10, GSM8K kept | learns at 25 | 6.4 | 1.10 to 1.20 | learns (curve rule, population 1; the quadratic rule said "fails: no odd response", population 0's curve rule prescribed 15.8 units) | yes (curve rule); no (quadratic) |
| Qwen2.5-3B-Instruct Quotation (start 0.937) | +0.027 | +0.033 | neither reached the target | | learns (curve rule), works (quadratic) | no (a ceiling the targets ignored) |
| Qwen2.5-3B-Instruct MATH alone | +0.017 | +0.003 | neither | | undetermined (curve rule), fails (quadratic) | no call, or consistent |
| Qwen2.5-0.5B-Instruct GSM8K | fails (two seeds: train +0.09, held-out within noise) | not run | | | fails: rollouts (50); second probe: no odd response | yes on the ES side; the GRPO side unmeasured |

Three facts follow, and they set the design:

1. **In our own records, ES was never more data-efficient than GRPO.** On every case
   where both learned, GRPO reached the target in 5 to 16 times fewer samples, and in
   fewer loop seconds too (the best ES ratio is 1.10); on the Tulu stream, at equal
   samples, GRPO is ahead at every milestone. "Positive for ES" in our records means ES
   learns the task at 1.1 to 3.5 times GRPO's compute, with retention only when anchored;
   it has never meant fewer samples or less compute. The published record is different:
   Qiu et al. (2025) report ES ahead of GRPO on Countdown on all seven models at equal
   sample evaluations, which is a data-efficiency win in the paper's definition (the
   correction of 2026-10-22 below has the numbers and the comparator question);
   EGGROLL's Countdown win is at equal wall-clock with about 11 times the generations,
   so it is not one. Our records contain no positive instance of the paper's question
   for a rule to learn from; the published one is the case the next stage has to test
   against a comparator at least as strong as ours, over a horizon long enough to see
   it.
2. **No case in the shared protocol has GRPO learning where ES does not.** The one
   GRPO-only case is the Tulu stream, at 8B, with 16 dense members and a non-repeating
   stream, against released checkpoints. The two nulls (C, D) separate nothing. The
   0.5B GSM8K case is GRPO-only if GRPO learns it, which costs one arm to find out.
3. **The calls that held came from the response curve read at the applied step**, not
   from the quadratic model: the curve rule's "learns" was right on A and its abstention
   on D was consistent; the quadratic rule was wrong on A and C. The curve rule's miss
   on C is a target stated in absolute reward where 0.063 of headroom remained.

### The checklist to freeze

Read in order; each line names its measurement, its threshold and the case that set it.
Thresholds are frozen here and changed only in a dated correction before the next case.

0. **Headroom and signal.** The start's validation reward leaves at least 0.20 of the
   range (C had 0.063 and neither method reached a target) and the live share is at
   least 0.1 (`LIVE_MIN`). Targets are fractions of the headroom: the learning target
   is a quarter of it, the comparison target (where both methods are timed) the same,
   and a target below 2.5 standard errors of the reserved block (about 0.05 at 300
   prompts) makes the case "not measurable at this block size" rather than a call.
1. **ES learns at this population** if the one-update measurement (the update of 64
   mirrored rank-1 members scanned over lengths on the 600-prompt validation and off-task
   blocks, as `probe.py` does) has a step at which the validation block's net response,
   odd plus even, is positive at its two-standard-error worst (`rule_curve`'s resolution).
   The prescription is that step with the pull-back of 0.1 after each update, and the
   forecast carries the corrected weights (`effective_updates_from_first`). No such step:
   "ES does not learn at 64 members" (0.5B GSM8K, D, and the second probe's GSM8K
   settings). The ranking's reliability is reported beside the call (0.14 to 0.52 where
   ES learned, 0.19 or less where it did not), as a covariate and not a gate.
2. **GRPO's pace** from the sign-direction slope at the shortest usable length, resolved
   at two standard errors: steps to the target as in P4 (held on A, on 3B instruct and on
   the base model within a factor of two). Unresolved: "GRPO's pace unknown", and the
   case contributes a measured GRPO arm without a prediction.
3. **The comparison**, in two currencies, both reported and one chosen for the class
   (decision 1 below): rollouts, the prescription's updates times 12,288 over GRPO's
   steps times 768; and GPU-seconds, the same counts at the measured costs per answer
   (ES 0.015 to 0.027 and GRPO 0.083 to 0.116 on an H200 at 1.5B to 3B, so an ES
   answer costs 0.15 to 0.25 of a GRPO answer). Classes: in compute, "competitive" at a
   ratio of 2 or less, "marginal" to 5, "GRPO dominates" above 5; in rollouts, the
   frozen `WORKS` 5 and `MARGINAL` 25. Where two seeds exist the class is assigned from
   their mean. On the record, every case where ES learned is "marginal" in rollouts
   (5.3 to 16) and "competitive" or "marginal" in compute (1.1 to 3.5).
4. **Retention** is part of the prescription, not of the call: the pull-back of 0.1
   after each update, read on the untouched `offtask_test` block with the 0.03 margin
   and paired intervals (`probe_run.retention`). The off-task spread ratio (P2) stays
   exploratory and is reported with its coverage; the base model's 11.8 is the one
   "collapses" call it has made, and an anchored run on that checkpoint tests it.
5. **Pace.** The forecast with the corrected weights is scored at 5 and 10 updates
   (where it has matched on A) with the band of a quarter to once; the class is read at
   the end of the horizon (30 updates, 40 steps).

### The cases

Seven fresh cases in the shared protocol, chosen before any probe by the outcome the
record or the literature leads one to expect, so that both outcomes are represented at
least twice. Each case gets the probe at 600 prompts with the `offtask_test` block,
two member populations where the call is "learns" (A's two populations disagreed by a
factor of five in the step; the designated population is the first, the second's
prescription is recorded and not run), the anchored ES arm at the prescription, and a
GRPO arm; two ES seeds where ES is expected to learn, one where it is not. Costs are by
uptime at $5.29/h with a fifth added for setup and gaps.

| case | expected | why | what to run | cost |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct GSM8K | GRPO only | ES failed on two seeds (S3b); the second probe found no odd response; the start leaves 0.45 of the range | one GRPO arm | $4 |
| Qwen2.5-1.5B-Instruct GSM8K | GRPO only | the second probe's record: no odd response at 64 members, reliability 0; Cho and Han report progress at 1.5B under ES at larger populations, so the call is at this population | the probe's prescription (the fallback), one anchored ES seed, one GRPO arm | $18 |
| Llama-3.2-1B-Instruct Countdown | GRPO only | Abdi et al. at matched rollouts: ES 15.2 against GRPO 37.6, their one large gap; the same task as A on a different checkpoint | a new pin, probe, two ES seeds, GRPO | $24 |
| Qwen2.5-3B-Instruct Countdown, anchored at the curve rule's 17 units | ES competitive | the S4a anchored arm at 15.3 units was the program's best run; the second probe's population prescribes 17; the GRPO reference exists | one anchored ES arm | $16 |
| Qwen2.5-3B base Countdown, anchored at 17 units | ES learns; P2 says the off-task block collapses | the fragile checkpoint: a random direction of a member's length moves GSM8K 11.8 times as much as the task; the GRPO reference exists | one anchored ES arm | $13 |
| Qwen2.5-1.5B-Instruct on a string task with a generator (reasoning-gym `word_sorting` or `spell_backward`) | ES competitive | shared structure like Countdown's (a format the whole population shares), on a task family with no published ES result; the generator removes the 200-problem limit that deferred E | probe, two ES seeds, GRPO | $30 |
| Qwen2.5-1.5B-Instruct on an arithmetic or puzzle task with a generator (reasoning-gym `chain_sum` or `puzzle24`) | open | between GSM8K (no odd response) and Countdown (shared) in the structure the members share; the case the checklist is least sure of | probe, two ES seeds, GRPO | $30 |

About $135 by uptime, $160 with the fifth. Order: the cheapest discriminating cases
first (the 0.5B GRPO arm, the two anchored 3B arms, Llama 1B), then the generator tasks,
then 1.5B GSM8K. Priority 2 of "After S4b" (a second GRPO seed on A and the A pair to 60
updates, about $22 more) folds in at the end if the budget allows; it tests the
horizon, not the rule.

reasoning-gym (Apache-2.0, 112 procedural generators with verifiers, on PyPI) is the
proposed prompt source for the two generator cases: one task per kind, chosen by its
structure before any probe and named in the preregistration. Not in this stage: 7B
(GRPO in f32 does not fit one H200), the Tulu 8B stream in the 64-member protocol
(expensive, and its GRPO side is the released run), and S4c.

Implementation before the pod, no GPU: a `Task` branch backed by a generator (a fixed
seeded block of 192 train problems, repeated, for protocol parity; the validation,
reserved, off-task and `offtask_test` blocks from the same generator at other seeds;
the verifier through reasoning-gym's `score_answer`), the Llama-3.2-1B-Instruct pin with
the chat template and a smoke of the rank-1 adapters on its module names, headroom-
relative targets in `prescribe()`, and the checklist's class in the prescription record
beside the exploratory curve rule. One to two days.

### What counts as the rule being useful on this batch

- The class (ES competitive / marginal / GRPO dominates / neither / not measurable) is
  right on at least five of the cases that get a call, with both outcomes among the
  right ones at least twice each.
- The pace forecast is inside its band at 5 and 10 updates on every case where ES
  learns.
- The compute ratio measured from the loops is within a factor of two of the forecast
  on every case where both methods reach the target.
- The anchored arms keep the untouched off-task block within the margin wherever the
  checklist said "learns", or the trade-off is measured and reported as such.

Falsified by a case where ES reaches the target with the checklist saying "does not
learn at 64 members", or where GRPO reaches it at less than a fifth of ES's compute with
the checklist saying "competitive". Two such cases end the rule as a claim; the paper
then reports the measurement and its failures, which is the fallback Andres accepted on
2026-10-21 ("a universal predictor is not a condition for the paper").

### Decisions for Andres

1. **The currency.** Decided by Andres on 2026-10-22: samples seen. See the correction
   below, which supersedes the compute-based class proposed here and the comparison
   protocol of the cases above.
2. **The case set** as proposed, or a subset; whether reasoning-gym is acceptable as a
   source (a dependency of the experiment, pinned by version, not of the library).
3. **The thresholds**: competitive at 2, marginal at 5 in compute; a quarter of the
   headroom as the target.
4. **Seeds**: two where ES is expected to learn, one where it is not.
5. **Budget and order**: about $160 for the seven cases, stopping after any case; the
   order above.
6. **The paper's frame**: the title and the abstract's question follow the rule
   (candidates in the manuscript's header), and the Tulu and Countdown results become
   the first cases of its test rather than the paper's claim.

#### Correction of 2026-10-22: the currency is samples seen, and what that changes

Andres, 2026-10-22: data efficiency means the reward reached for the same number of
samples seen, one sample being one generated answer to one prompt, whichever method
generated it. Qiu et al.'s Countdown result ("higher final accuracy at equal sample
evaluations") is a data-efficiency claim in exactly this sense. Wall-clock matters as
much but depends on the implementation, so it is reported and not the criterion. The
paper's question is, and has been since 2026-10-12: when can ES be expected to have a
higher data efficiency than GRPO?

**What the record says at equal samples.** The one comparison at equal samples over a
long horizon is the Tulu stream: after 368,640 samples each, ES at 16 dense members is
0.25 below its start and the released GRPO checkpoint 1.87 above it, and the RL branch
is ahead at every intermediate milestone (`runs/tulu-long-README.md`). On Countdown the
GRPO arms stop at 30,720 samples (40 steps of 768), a twelfth of the ES arms' 368,640
(30 updates of 64 members on 192 prompts), so every Countdown comparison so far is at
ES's early horizon: the "rollouts to +0.10" ratios of 5.3 to 16 are the only
equal-samples readings, and they all favour GRPO. At GRPO's 30,720 samples no ES arm has
been scored (the first evaluation is at 61,440); at twice GRPO's samples the ES arms are
below GRPO's 40-step value on every setting (3B instruct: ES +0.03 to +0.11 against GRPO
+0.13 to +0.21; 1.5B: +0.05 to +0.09 against +0.19; the base model: +0.04 to +0.06
against +0.14, reserved block). Whether ES overtakes later, as Qiu et al. report at 3
million samples, is untested here: our GRPO arms were still rising at 40 steps.

**What the literature says at equal samples.** Four papers compare at matched samples or
FLOPs on LLM tasks (`11`): Qiu et al. find ES ahead on all seven Countdown models; Abdi
et al. find GRPO ahead on 7 of 8 cells; Hoy et al. find ES matching GRPO only at about
15 times the generations; Ba et al. find GRPO ahead on Pass@1 at matched FLOPs. On the
same model and task, Qwen2.5-1.5B-Instruct on Countdown, Qiu's GRPO baselines reach
14.8 to 31.0 and their ES 37.3, while Abdi's GRPO reaches 56.4 and their ES 53.0
(evaluation sets may differ between the papers). The published positive case therefore
rests on GRPO baselines that another group's GRPO beats by a wide margin on the same
setting, and the question for this paper is whether ES overtakes a GRPO run that is at
least as strong as ours at a horizon long enough to tell.

**The protocol that answers the question.** Both methods on the same setting to the
same sample budget `B`, scored on the untouched test block and the off-task block at
common sample milestones (every 61,440 samples, five ES updates or 80 GRPO steps, with
GRPO's finer grid every 20 steps kept), the difference paired over prompts with its
standard error. "ES more data-efficient at `B`" if ES's reward at `B` exceeds GRPO's by
more than two paired standard errors on the test block; the whole curve is reported, and
the sample count at which either method reaches a headroom-relative target beside it.
The GRPO side runs the published recipe at the better of its two learning rates on the
40-step arms (1e-6 on 3B instruct, +0.213 at 40 against +0.135 at 5e-7; the 1e-6 arm is
to be run on the other settings, about $4 each), stated as the comparator's tuning. The
ES side runs two arms: ours, anchored at the prescription (the existing 30-update arms
resume from their logs), and the published recipe (Qiu et al.: 30 dense members, sigma
1e-3, alpha 5e-4, 200 prompts, greedy decoding, no anchor), so that a miss cannot be
attributed to our estimator.

**Cost by sample budget**, at the measured rates (GRPO 89 s per step on 3B instruct, 87
on the base model, 64 on 1.5B; ES 300, 234 and 199 s per update), $5.29/h:

| budget `B` | GRPO steps | GRPO, 3B instruct | GRPO, 1.5B | our ES to `B`, 3B / 1.5B | note |
|---|---|---|---|---|---|
| 368,640 (the existing ES arms' count) | 480 | 12 h, $63 | 8.5 h, $45 | done | equal-samples curves on the existing ES arms at six milestones |
| 1,000,000 | 1,302 | 32 h, $171 | 23 h, $123 | +51 updates: 4.3 h, $23 / 2.8 h, $15 | a third of Qiu's horizon |
| 3,000,000 (Qiu's) | 3,906 | 97 h, $510 | 69 h, $367 | +193 updates: $85 / $57 | GRPO's training step (71 of 89 s) would need a cheaper implementation, which changes the comparator's numerics |

The published-recipe ES arm at 3 million samples is S4c's estimate, $200 to 600 with
its dense members at 3B; its rate is to be measured on a short run before it is costed.
The base model is costed as 3B instruct.

**The rule's part.** The one-update measurement gives each method's early gain per
sample (ES's net gain per update over 12,288; GRPO's first-step gain over 768) and their
ratio, which on every probe so far said GRPO, and the runs confirmed it within a factor
of two to six. It cannot see GRPO's plateau, and a crossing at a long horizon is a
statement about the plateau. The prediction written before each long run is therefore
the early ratio's direction and the claim that it persists to `B`; the paper's rule is
"ES is more data-efficient where the early ratio favours it" if no case overtakes, and
otherwise has to name what the overtaking cases share. Falsified by a case where ES
overtakes GRPO by two standard errors at `B` with the early ratio above two.

**The cases**, in place of the table above: the three published Countdown checkpoints
whose outcomes conflict across papers, Qwen2.5-3B-Instruct, Qwen2.5-1.5B-Instruct and
Llama-3.2-1B-Instruct (a new pin; Qiu 16.8 against 14.9, Abdi 15.2 against 37.6), as the
cases where ES might be more data-efficient; the GSM8K-family cases (the 0.5B GRPO arm,
1.5B GSM8K) and the Tulu stream as the cases where it is not; the base model's existing
arms extended if the budget allows. The generator tasks are dropped from this stage.

**Decisions for Andres, revised.** (1) The sample budget: 368,640 first on 3B instruct
and 1.5B ($108, plus the 1e-6 GRPO arm on 1.5B), then 1 million (about $330 more for
both), with Qiu's 3 million only if a cheaper GRPO step is accepted. (2) Whether the
published-recipe ES arm runs beside ours. (3) Llama-3.2-1B-Instruct as the third
Countdown case (about $40 at `B` = 368,640 including its probe and GRPO). (4) The
comparator's tuning policy: the better of two learning rates at 40 steps. (5) The
manuscript's question sentence, through the loop.

#### S5 stage 1: preregistration (2026-10-25, before any run)

Andres approved stage 1 on 2026-10-25 ("start on step 1"): GRPO run to the existing ES
arms' sample count on the two settings where the published Countdown claims sit, so
that the comparison is at equal samples seen over six milestones. Budget about $135.
The two candidate explanations of Qiu et al.'s result (their comparator is weak; a
long-horizon effect) are written down here so that the outcome is read against them.

**What runs.** On Qwen2.5-3B-Instruct Countdown, GRPO at 1e-6 (the better of the two
40-step arms on the validation block, +0.159 against +0.138) for 480 steps, 368,640
samples, the Tulu recipe otherwise unchanged, scored every 10 steps on the first
probe's blocks (`runs/probe/qwen3b-countdown`, the blocks every ES arm on this setting
used): `runs/s5/qwen3b-countdown-grpo-lr1e-6-s0`. On Qwen2.5-1.5B-Instruct Countdown,
first a 40-step arm at 1e-6 (`runs/grpo-run/qwen1.5b-countdown-lr1e-6-s0`), then 480
steps at the better of 5e-7 and 1e-6 on the validation block at step 40 (a tie goes to
the recipe's 5e-7), on the S4b probe's blocks (`runs/s4b-probe/qwen1.5b-countdown`):
`runs/s5/qwen1.5b-countdown-grpo-lr<chosen>-s0`. The driver saves a checkpoint (the
policy in f32, the optimizer and the sampler state) every 40 steps and at the end, on a
network volume, so a pod failure resumes and stage 2 continues the same run rather than
starting over; a resume is recorded in `run.json`. Two Secure H200 pods in US-NC-1 at
$5.29/h, about 12.5 and 10 hours.

**What is compared.** `probe_run --samples <grpo run> runs/probe-run runs/s4a` (and
`runs/s4b` for 1.5B): at every ES evaluation whose sample count a GRPO evaluation
matches (ES update 5m is 61,440m samples, GRPO step 80m; the 256-member arm's update 5
is 245,760, step 320), each method's change on the reserved test block and the
ES-minus-GRPO difference paired over the block's prompts, with the off-task block
beside. Both runs score the same prompts, checked by digest.

**Predictions.**

1. *The early ratio persists.* At every common milestone on both settings GRPO's test
   change exceeds every ES arm's, and at 368,640 samples by more than two paired standard
   errors against each of them (3B instruct: the two prescribed-step seeds, the step-3
   arm, the anchored arm, the shuffled-ranking arm, and the 256-member arm at 245,760;
   1.5B: the two anchored seeds and the un-anchored control).
2. *GRPO passes the best ES anywhere.* GRPO's test change at 480 steps exceeds the
   setting's best ES peak at any evaluation: +0.235 on 3B instruct (the anchored arm at
   update 25), +0.183 on 1.5B (the un-anchored arm at update 15).
3. *No decline.* GRPO's test change at 480 steps is not below its change at 240 steps by
   more than two standard errors. A decline, or a plateau below an ES arm, is the
   signature of the long-horizon candidate.
4. *Retention.* GSM8K stays within 0.03 of its start at every evaluation of both long
   arms (it did to 40 steps; the KL term is on).
5. *The comparator's tuning.* 1e-6 beats 5e-7 on the 1.5B validation block at step 40, as
   it did on 3B.

**What the outcomes mean.** 1 to 3 holding on both settings is the weak-comparator
candidate strengthened and the stop rule met: stage 2 (1 million samples, Qiu's recipe
as a second ES arm) runs on 3B instruct only. ES ahead at any milestone, or 3 failing,
keeps the long-horizon candidate alive and stage 2 runs on both settings. Prediction 2
failing while 1 holds means GRPO is the more data-efficient at every matched point but
the anchored ES arm's peak stands above GRPO's 480-step value, which is reported as
such. Prediction 4 failing is a retention finding about GRPO over a long horizon and
changes nothing in the stop rule.

**Not changed by this stage.** The ES arms (no new ES run); the frozen prescriptions;
the thresholds of S4b's classes; the Tulu case.

#### S5 stage 1 results (2026-10-26, `runs/s5/README.md`)

Both arms ran to 480 steps (368,640 samples) at 1e-6, the better learning rate on both
settings (prediction 5 held on 1.5B: +0.207 against +0.181 on validation at 40 steps).
About $169 by uptime, of which about $48 was idle time after both first attempts died at
their second checkpoint write on a full 80 GB volume and the grader's worker pool kept the
dead processes alive; both resumed from their step-40 checkpoints on the enlarged volume,
and a subagent checked progress every 30 minutes from then on.

At equal samples seen, GRPO is ahead of every ES arm at every one of the six common
milestones on both settings (`probe_run --samples`): on 3B instruct by 0.18 to 0.55 of
the range (6 to 20 paired standard errors), on 1.5B by 0.06 to 0.18 (2.2 to 7.5). At the
ES arms' full sample count GRPO's reserved block is +0.430 on 3B (peak +0.466 at 270
steps, 207,360 samples) against the best ES arm's +0.185 at that count and +0.235 at any
point, and +0.250 on 1.5B against +0.180 and +0.183. Predictions 1 to 3 held on both
settings; prediction 4 (GSM8K within 0.03 at every evaluation) held on 1.5B and at 48 of
49 evaluations on 3B (−0.033 at step 360, −0.017 at the end). GRPO's 1.5B curve
plateaus at about +0.25 from 46,080 samples on; its 3B curve is still at its plateau
level of +0.41 to +0.47 from 240 steps on.

**Reading.** The weak-comparator candidate is the one the data support: our GRPO on 3B,
with an untuned post-training recipe and a KL anchor, reaches in a fourteenth of Qiu et
al.'s sample budget a reserved-block accuracy (0.601 at the peak, 0.565 at the end)
above their reported ES end point (58.7 ± 1.5) and far above their RL's (44.4 ± 1.9),
with test sets of different size from the same source. On 1.5B our GRPO's plateau
(0.324) sits between their two RL variants and below their ES (37.7 ± 1.1), so that
setting is the one where a long-horizon ES overtaking is not excluded; whether the
plateau is the task's or the KL anchor's is the open question there (a beta-0 arm, about
$45). The stop rule is met: stage 2 (to 1 million samples, with Qiu's recipe as a second
ES arm) runs on 3B instruct only, and the 1.5B beta-0 arm is proposed beside it.

#### S5, the early history under the published recipe: preregistration (2026-10-30, before any run)

Andres approved on 2026-10-30 ("Yes, run those 20 iterations"). The question is whether
ES leads GRPO at any point early in training, even if GRPO wins later. Qiu et al.'s
Countdown claim (their appendix A.1, A.6, A.8) is made in the regime stage 1 already
used: 200 training problems reused at every iteration, the methods compared at equal
sample evaluations (one generated answer to one training problem) up to 3 million, RL
trained on the same problems, and ES "consistently outperformed RL ... throughout
training". Our GRPO cycled through the same 192 problems as the ES arms, so stage 1 is in
that regime, but every ES arm in it used our recipe (mirrored rank-1 members at sigma
5e-4, the rule's step, an anchor on some). The published recipe has not run on 3B.

**What runs.** `es_vllm/probe_run.py --full-rank`: es-at-scale's recipe on the first
probe's blocks (`runs/probe/qwen3b-countdown`), 30 full-rank Gaussian members on every
weight, sigma 1e-3, alpha 5e-4, the fitness z-scored, no anchor, every member decoding
the probe's 192 train prompts greedily at every update (5,760 samples per update), with
shardes in the worker as T2 ran it at 0.5B. Twenty updates, 115,200 samples, one seed,
the validation, test and off-task blocks scored after every update:

    python -m es_vllm.probe_run --setting qwen3b/countdown --prediction runs/probe/qwen3b-countdown \
        --full-rank --members 30 --iterations 20 --eval-every 1 --seed 0 \
        --out runs/s5-early/qwen3b-countdown-full-rank-s0

One H200, about 2.5 hours of uptime, about $10. The run stops early if a bit-for-bit
check fails (the driver exits) or if updates 1 and 2 project the run past 4 hours of
uptime, in which case it is stopped and reported before any more is spent.

**What is compared.** `probe_run --samples runs/s5/qwen3b-countdown-grpo-lr1e-6-s0
runs/s5-early`: ES update 4m and GRPO step 30m are both 23,040m samples, so the two
meet at five milestones, 23,040 to 115,200, paired over the test block's prompts as in
stage 1. The ES curve at every update is reported beside GRPO's, which is scored every
7,680 samples.

**Predictions.**

1. *GRPO ahead from the start.* At each of the five milestones GRPO's test change
   exceeds ES's by more than two paired standard errors. The basis: every rank-1 arm in
   stage 1 was behind at its first milestone (61,440 samples), and at 0.5B this recipe
   solved almost nothing in its first 15 updates (T2).
2. *No early window at all.* No ES evaluation in the 20 updates reaches GRPO's test
   change at its first evaluation, +0.113 after 7,680 samples.

**What the outcomes mean.** 1 holding means that on this setting, with the published
recipe in our code and the same problems, ES is behind GRPO from the first milestone:
Qiu et al.'s lead comes from their RL baseline, not from the regime, and starting with
ES before switching to GRPO has no early window to use here. ES ahead of GRPO by two
standard errors at any milestone is an early window: its length is then measured with
more seeds and the switch is the next test. A milestone within two standard errors is
reported as unresolved, with one seed. 2 failing while 1 holds says ES reaches GRPO's
first level within 115,200 samples but later than GRPO does.

**Not changed by this run.** The GRPO arm, the stage 1 ES arms, the blocks.

#### S5, the early history: results (2026-10-30, `runs/s5-early/README.md`)

The run went to 20 updates on one Secure H200, about $4.50 by uptime, every bit-for-bit
check passing. Prediction 1 held: GRPO is ahead at all five milestones by 0.14 to 0.29
of the range (5.3 to 10.1 paired standard errors). Prediction 2 failed by 0.0003: ES's
test change touched +0.1135 once, at update 19 (109,440 samples), against GRPO's +0.1132
at 7,680, and was +0.087 at the next evaluation. ES was at +0.021 after its first 5,760
samples. At equal loop seconds GRPO is also ahead (its +0.113 after 855 s; ES +0.059 at
841 s), although ES's decodes cost a sixth of GRPO's step per sample. GSM8K sat 0.02 to
0.04 below its start at 12 of 20 evaluations under ES, against 0 to 0.02 under GRPO.

**Reading.** No early window on 3B instruct Countdown: with the published recipe in our
code and the same problems, ES is behind GRPO from the first evaluation and needs 14
times GRPO's samples to reach GRPO's first level. Qiu et al.'s lead throughout training
comes from their RL baseline, not from the regime, and an ES-then-GRPO schedule has
nothing to gain here. One seed and 20 of their 500 iterations; a later overtaking is
not tested.

#### S5, the published recipe to 100 updates: preregistration (2026-10-31, before any run)

Andres approved on 2026-10-31 ("Yes, run Qiu's recipe to 100 updates"). The early-history
run stopped at 115,200 samples, before the first point of Qiu et al.'s training curves
(their Figure 7: every 0.6M sample evaluations, that is every 100 ES iterations). In
their measure, accuracy (the share of test problems answered correctly, format ignored;
doc 11 has their setup), our GRPO arm reaches 0.570 at 207,360 samples, above every RL
point they plot for Qwen-3B (at most 45.1 at 1.2M). Their ES is at 35.6 at 0.6M and 60.5
at 3.0M. Their own figure has Dr.GRPO-v level with ES at 0.6M and ahead at 1.2M, with ES
passing it near 1.5M. The open question is whether the published recipe in our code
follows their ES curve. If it does, ES on this setting is slow but keeps rising: behind
our GRPO for a long stretch, and possibly level with GRPO's best after about ten times
its samples.

**What runs.** The early-history command with 100 updates and member seed 1, so its
first 20 updates are also a second seed of the early history:

    python -m es_vllm.probe_run --setting qwen3b/countdown --prediction runs/probe/qwen3b-countdown \
        --full-rank --members 30 --iterations 100 --eval-every 1 --seed 1 \
        --out runs/s5-early/qwen3b-countdown-full-rank-s1

576,000 samples (their 0.6M is on 200 problems an iteration against our 192). One H200,
about 3.5 hours of uptime: about $13 on Community, $19 on Secure. The run stops early if
a bit-for-bit check fails (the driver exits) or if updates 1 and 2 project it past 5
hours of uptime.

**Measure.** Test accuracy from `probe_run --history`, whose accuracy column is added in
this commit: a Countdown reward of at least 1 is a correct answer, since format adds at
most 0.1. The 300 test problems come from es-at-scale's evaluation set, the pool of 2,000
Qiu et al. test on; the base model scores 0.083 here and 10.0% in their Table 1. The
standard error is about 0.028 near 0.35. Primary: the evaluation after update 100.
Secondary: the mean of the evaluations after updates 91 to 100, since the early run's
accuracy moved by up to 0.027 between consecutive updates.

**Predictions.**

1. *Our ES follows Qiu's curve.* Accuracy after update 100 is within two standard errors
   of their 0.356, so between 0.30 and 0.41. The basis: a least-squares line through the
   early run's accuracy column from update 4 gains 0.0023 per update and is at 0.35 at
   update 100 (0.0028 and 0.395 from update 1); their straight line from 10.0 to 35.6
   gains 0.0026 per update.
2. *Still below our GRPO.* Accuracy after update 100 is below 0.47: GRPO's last
   evaluation (0.527 after 368,640 samples) less two of its standard errors (0.029).
3. *The early history replicates.* `probe_run --samples` puts GRPO ahead by more than two
   paired standard errors at the early run's five milestones (23,040 to 115,200 samples).
   All 16 milestones up to GRPO's last step (368,640 samples, ES update 64) are reported.

**What the outcomes mean.** 1 holding means the published recipe in our code learns at
the rate their paper shows, so the difference between their results and ours is in the
RL arm; ES's level at 3M (60.5 in their run) is then the next test, 400 more updates for
about $55 to $85. Below 0.30 means our ES learns slower than theirs here, and the cause
(implementation, the 192-problem train block, the 300-problem test block, one seed) comes
before any longer run. Above 0.41 is reported as faster, with the same caveats. 2 failing
puts ES level with GRPO's last evaluation at 576,000 samples, which would change the
early run's reading. 3 failing at any milestone is an early window in one seed of two,
and gets a third seed before any conclusion.

**Not changed by this run.** The GRPO arm, the seed 0 run, the blocks.

#### S5, the published recipe to 100 updates: results (2026-10-31, `runs/s5-early/README.md`)

The run went to 100 updates on one Secure H200, about $17 by uptime, every bit-for-bit
check passing. Prediction 1 failed on the high side: accuracy after update 100 is 0.430
(standard error 0.029; the mean after updates 91 to 100 is 0.433) against the band of 0.30
to 0.41 around Qiu et al.'s 35.6% at 0.6M. Prediction 2 held: 0.430 is below 0.47, and
GRPO is ahead at all 16 equal-sample milestones up to its last step, by 5.8 to 11.9 paired
standard errors. Prediction 3 held: with seed 1, GRPO is ahead at the five early
milestones by 5.9 to 9.1 paired standard errors. Not preregistered: at equal loop seconds
the two are level within about one standard error through GRPO's step 110 (9,254 s), and
ES reached 0.43 first (7,269 s); GRPO then rises to 0.570 at step 270.

**Reading.** The published recipe in our code learns at least as fast as theirs did, so
the gap between Qiu et al.'s results and ours is in the RL arm: our GRPO's 0.570 at
207,360 samples is above every RL point they plot for this model. At equal samples ES
leads nowhere and needs about five times GRPO's samples to reach 0.43. At equal GPU time
on these two drivers it keeps pace over the 2.7 hours compared; GRPO's later 0.570 came
at 22,933 s, beyond this ES run. Whether ES keeps rising to their 60.5% at 3M, past our
GRPO's best, is not tested; one seed to 100 updates.

## S3c results (2026-10-14, `runs/grpo-run/README.md`)

Three GRPO arms, 40 steps of 768 rollouts each on one H200 (about $17):

| arm | lr | test +0.10 first at (predicted) | test at 40 | off-task at 40 | class | GPU-s per answer |
|---|---|---|---|---|---|---|
| 3B instruct Countdown | 5e-7 | 15 (16) | +0.135 | −0.007 | works | 0.116 |
| 3B base Countdown | 5e-7 | 30, between 25 and 30 (20) | +0.135 | +0.143 | works | 0.114 |
| 3B instruct Countdown | 1e-6 | 15 (8) | +0.213 | −0.003 | works | 0.111 |

The RL side's gate holds on both reference arms (15 against 16, 25 to 30 against 20)
and by a hair on the doubled step (15 against 8). GRPO kept the off-task block on every
arm where every ES run on these settings lost it, and its answers shortened where ES's
ran to the caps: the collapse is ES's, not the task's. The comparison, measured from the loops
(`probe_run --compare`, the table in `runs/probe-run/README.md`): to the first evaluation
at +0.10 on the reserved block, ES used 5.3 and 16 times GRPO's rollouts on the instruct
model (seeds 1 and 0) and 8 and 16 on the base model (the probes forecast 2.8 and 17),
and 1.16 and 3.50 times GRPO's loop seconds on the instruct model, 1.40 and 2.67 on the
base model, GRPO at 5e-7 in both (per answer: 0.11 to 0.12 GPU-seconds for GRPO, 0.020
to 0.026 in the ES runs). The two seed-1 ES runs reached the target within twice GRPO's
compute with GSM8K intact, the program's usefulness line, and deteriorated afterwards;
the seed-0 runs crossed with GSM8K already down. These are threshold crossings on
300-prompt blocks. One seed per arm, the Tulu run's recipe untuned for these models, 40
steps against the hundreds of the published Countdown runs, and GRPO's KL term against
the start (beta 0.01) where ES has no anchor at all, so the retention difference is not
attributable to the estimator alone.

## Results of the training runs (2026-10-13, `runs/probe-run/README.md`)

Six runs at the preregistered prescription, two seeds of three settings, 30 updates of
64 members each, scored on the reserved test blocks:

| run | call | step | peak test gain (update) | end: test, off-task | class / task alone | collapsed |
|---|---|---|---|---|---|---|
| 3B instruct Countdown, seeds 0 and 1 | works | 15.3 | +0.135 (15), +0.164 (15) | −0.118, −0.837; −0.055, −0.843 | fails / works; works / works | both |
| 3B base Countdown, seeds 0 and 1 | marginal | 3.0 | +0.104 (30), +0.169 (30) | +0.104, −0.340; +0.169, −0.100 | marginal / works; works / works | both |
| 0.5B GSM8K, seeds 0 and 1 | fails | 1.1 | +0.033 (15), +0.007 (10) | +0.030, +0.013; −0.027, +0.003 | fails; fails | within noise |

The runs establish initial learning on Countdown at every seed and no detectable
held-out gain on GSM8K at 0.5B; they do not establish the forecasts' sizes or sustained
retention. The prescription's size was off by two to four (the instruct model reached the target at 5 to 15 updates against 2.9
gross or 3.8 net; the base model at 15 to 30 against 22 or 19), with the first-order
model tracking the base model's validation curve through 30 updates and failing in shape
at the larger step (gain, plateau, unwinding). Every Countdown run lost its off-task
block late, once the task gain reached about +0.08 to +0.10, at either step and on every
seed (GSM8K to 0.00 on the instruct model, to 0.06 and 0.30 on the base model), with the
answers lengthening to the caps; nothing in the one-update probe sees this, so a "works"
call is a statement about the task until a retained-task bound is measured over several
updates of the run itself. The seed decides the frozen class where the early trajectory
is close to a threshold (works and fails on the instruct seeds, works and marginal on
the base seeds). The RL side of every ratio remains the proxy. Stage S3b is complete;
the rule's quantitative side is now calibrated against six curves, its off-task side is
not predictable from a probe, and the next experiment is the measured GRPO baseline.

Against the expectations: 2 held (the instruct Countdown model works with the largest
`G^2/|C|`), 3 held in its three parts except that 3B on GSM8K has no odd response, 5
held on four of six, 1 was wrong as written (the base model's collapse is not an absent
signal but a found direction that destroys GSM8K, which the off-task number now
carries), and 4 failed (Spearman 0.54: the Countdown settings lead both orders, the
four GSM8K settings are near zero in both measures). Two findings outside the
expectations: the curvature along random rank-1 directions falls three orders of
magnitude from 0.5B to 7B while `G` does not move, which is the published size trend in
the rule's terms; and a ranking of reliability 0 (7B) still gives an update whose odd
response is 3.6 standard errors from zero, as on Tulu. The cost term is unresolved on
four of six records at 300 prompts and 2.5 member lengths; the next batch extends the
scan until the even response is resolved.
