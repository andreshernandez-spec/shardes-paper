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
2. **A retained block's odd response is a single draw.** The update's directional effect
   on a block it was not ranked on is a directional derivative along the update's random
   part as much as along its useful part; across draws it averages to zero, within one
   draw it is large. The base model's probe put GSM8K at −0.25 after five updates; the
   run shows +0.09. The retained-block bound therefore uses the even response only, and
   the odd off-task response is reported as a single draw and not used. The level-1
   predictor of systematic off-task change would be the covariance across members between
   task fitness and off-task fitness, which needs the members to decode the off-task
   block; proposed for the next batch.
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
   evaluation (`probe_run.observed_class`). A run shorter than the prescribed update count cannot measure the
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

## Results of the training runs (2026-10-13, `runs/probe-run/README.md`)

Six runs at the preregistered prescription, two seeds of three settings, 30 updates of
64 members each, scored on the reserved test blocks:

| run | call | step | peak test gain (update) | end: test, off-task | class / task alone | collapsed |
|---|---|---|---|---|---|---|
| 3B instruct Countdown, seeds 0 and 1 | works | 15.3 | +0.135 (15), +0.164 (15) | −0.118, −0.837; −0.055, −0.843 | fails / works; works / works | both |
| 3B base Countdown, seeds 0 and 1 | marginal | 3.0 | +0.104 (30), +0.169 (30) | +0.104, −0.340; +0.169, −0.100 | marginal / works; works / works | both |
| 0.5B GSM8K, seeds 0 and 1 | fails | 1.1 | +0.033 (15), +0.007 (10) | +0.030, +0.013; −0.027, +0.003 | fails; fails | within noise |

On the task alone the calls held in kind on all six runs. The prescription's size was
off by two to four (the instruct model reached the target at 5 to 15 updates against 2.9
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
