# 08. Why RL learns Tulu 3.1 and ES does not: the measured chain

Written 2026-10-10. Not a preregistration: a synthesis of `03` to `07` and of the probes
run between 2026-10-07 and 10. Every number here is in a README under
`experiments/end_to_end/runs/`, with the script, the commit and the record it comes from;
the last section lists them. Decisions along the way were Andres's, recorded in the READMEs.

## Summary

On exactly the prompts and the rollout count of the Tulu 3.1 RL run, from the same start,
ES with the configuration of Qiu et al. (16 members, sigma 5e-4) does not learn: after
the data of 480 RL steps it ends 0.25 below its start on held-out prompts, while the RL
run is 1.9 above. The chain of measurements below says why, and the answer is not a bug,
a bad fitness or a bad step size.

An ES update is a zeroth-order estimate of the gradient. We measured, against the exact
GRPO gradient of the same batch, that the estimate is right: its component along RL's
gradient is 12% of the most a zeroth-order estimate can have, its expected direction is at
least as steep as RL's gradient direction, and a perfect estimate of RL's gradient would
gain 0.8 +- 0.15 of what ES's update already gains per unit. What ES lacks is length. The
useful part of an update is at most `alpha` = 5e-4 long, beside a random part of
`alpha sqrt(d / N)`: 11 at 16 members, 1.4 at 1,024. The random part costs reward in
proportion to its length squared, and at 16 members that cost equals the useful part's
gain. RL's step has no random part: it moves 0.045 along its direction every step, all of
it useful, because backpropagation tells it which weights produced each answer.

More members shrink the random part. At 1,024 low-rank members an update nets about twice
what RL gains on the same 192 prompts, and eight such updates accumulate at half that rate,
at 32 to 64 times RL's rollouts. On Countdown, where the improvement is one behaviour every
prompt shares, the same model's perturbations carry 75 times more signal per prompt, and
ES learns as its literature reports. On a realistic post-training stream, whose gains are
many small, prompt-specific behaviours, the zeroth-order variance is what separates the
two methods, and no fitness, optimizer or variance-reduction technique recovers it; only
rollouts do.

Terms used throughout: a *member* is one perturbed copy of the model; its *fitness* is
its mean greedy reward over the iteration's prompts; the *update* is the z-score-weighted
sum of the members' perturbations; *held-out* is the 3,840 prompts of RL steps 121 to 160
and 481 to 520, scored greedily with the run's verifiers on the 0-10 scale.

## 1. What was matched, and what the arms did

The RL run (open-instruct 3f37c29, 48 prompts x 16 samples per step) and the ES arm see
the same prompt stream: ES iteration g takes RL steps 4g+1 to 4g+4, so one iteration of 16
members x 192 prompts generates exactly the 3,072 answers of four RL steps. The RL side is
AllenAI's released intermediate checkpoints, pinned by commit and weight hash and scored
by us on prompts they had not trained on (`05`); the RL run itself was not rerun.

| | held-out reward (RL steps 121 to 160, 1,920 prompts) |
|---|---|
| DPO start | 5.43 |
| RL step 40 / 80 / 120 | 5.87 / 6.40 / 6.69 (+1.26 at step 120, mostly IFEval) |
| ES s5e-4 after 30 iterations (RL steps 1 to 120's data) | 5.42 |
| random-fitness control | 5.39 |

The pilot (`03`) passed no gate; the longer arm (`07`, 120 iterations, RL steps 1 to 480's
data) passed its gate against the control (+0.26 +- 0.09) while ending 0.25 +- 0.09 below
its start (5.255 against 5.505), with RL step 480 at 7.37. The backend reproduces the
library bit for bit and learns Countdown faster than es-at-scale, the reference
implementation (`06`), so the null is not an implementation fault.

## 2. The ranking of the members is mostly noise, and the noise is not sampling noise

`runs/grad-check/`: the arm's 16 iteration-0 members decoded on 768 prompts, with
per-prompt rewards. *Reliability* is the share of the variance of a member's fitness that
is a property of the member rather than of the prompts drawn.

- At the run's sigma a member changes 9.8% of the outcomes (4.6% up, 5.2% down) and the
  reliability of its fitness over 192 prompts is 0.12 (90% interval 0.01 to 0.47). The
  update is mostly a random direction. Larger sigma raises the signal but members lose
  1.45 points on average at 2e-3; smaller gives nothing to rank.
- Sampling at temperature 1.0 (8 samples per prompt, common random numbers) is no better
  per rollout than greedy decoding (reliability 0.11 against 0.13 at 192 rollouts), nor are
  mirrored pairs or any other split of the rollout budget.
- `runs/why/analysis.json`, `runs/direction-check/`: a member's greedy outcome is a redraw
  at the start's success probability (q = 0.75 of a fresh draw). Its answer departs from
  the start's within the first dozen tokens on 86 to 90% of prompts, at positions where the
  start's top two tokens are within a few bf16 steps of each other. A perturbation moves
  each uncertain prompt's success probability by about 8 points, but its effects on two
  prompts correlate at 0.0095 (0.004 over all prompts): the averaged fitness carries little.

Removing the decoding noise does not change this. `runs/contrastive-check/` scored each
member by the log-probability contrast of fixed right and wrong answers the start had
sampled, which is exact (a rescore of the start matched to the bit). The ranking over 186
prompts is then 36% common to prompts, half of that a length artifact (wrong answers are
2.5 times longer), and 14% per token, below reward's 21% from the same prompts. The
effects on two prompts correlate at 0.0024. A member moves the start's answers almost as
far as RL's 120 steps do (6.1 against 7.8 nats of KL per answer) at 90 times the distance
in weight space, in no shared direction.

The same measurement on Countdown separates the task from the method
(`runs/grad-check/README.md`, prediction committed before the run): with the same Tulu 8B
start and the same 16 noise directions, the members' effects on two Countdown prompts
correlate at 0.30 and the ranking is 42 to 68% real; for the 0.5B model of `06`, 1.06 and
0.94. There a perturbation changes two thirds of the outcomes and many of them the same
way (it writes the answer format or it does not).

## 3. RL's move is tiny and coherent

`runs/rl-geometry/`, `runs/direction-check/`: the released step-120 model is 0.476 from
the start in weight space (step 480: 1.134), against 61 for ES after the same data and 45
for a single ES member's perturbation. Along RL's direction the held-out reward rises +0.85
at effective length 0.32 and +1.02 at 0.48 (the engine's bf16 rounding shortens small
steps; lengths are given as what reached the model); random directions of the same lengths
change nothing (six changes within 1.3 SE of zero), and flips along RL's direction are
four to eight times as often up as down. The RL run keeps its direction (cosine 0.51
between its moves to step 120 and to 240); consecutive ES updates are independent random
directions. Claims about the sparsity or rank of the released difference were withdrawn:
most of it is one-step bf16 rounding on small weights.

## 4. One ES update gains what its random part costs

`runs/step-check/`: the iteration-0 update `u` built from four rankings of the same 16
members, applied as `theta_0 +- lambda u` on the 3,840 held-out prompts. Half the
difference of the two signs is the update's first-order gain (its *odd part*); their mean
minus the start is the cost of its random part (the *even part*), which is the same for any
ranking because it depends on the update's length, not its direction.

| per unit update (alpha = sigma = 5e-4) | predicted | measured |
|---|---|---|
| gain, best ranking (768 prompts) | +0.039 +- 0.029 | +0.004 +- 0.003 (at most 0.010) |
| cost of the random part, per unit squared | -0.0038 | -0.0042 +- 0.0005 |

- Random weightings of the same 16 directions give gains of +-0.011, so the ranked updates
  cannot be told from a random weighting. The arm's first update nets about nothing at the
  arm's step, as the arm did over 120 updates.
- The gradient probe had implied 0.021 per iteration from the members' fitness. The
  mirrored members show why it overstated: two thirds of what makes a member score well
  is what a perturbation of this size does whichever its sign, which a weighted sum of
  perturbations does not inherit.

## 5. More members: the gain per update stays, the cost falls as 1/N

`runs/lowrank-check/` (Andres's proposal: hundreds of members, low-rank): members are rank-1
perturbations of every attention and MLP matrix, in mirrored pairs, served as LoRA adapters
that vLLM builds in memory from their seeds, 64 per batch on one copy of the base weights
(adapters equal the merged weights at correlation 0.997). 1,024 members decoded the arm's
192 iteration-0 prompts; the update from the first N was measured as in section 4.

| members | gain per unit | cost per unit squared | best net per update (`gain^2 / 4 cost`) |
|---|---|---|---|
| 16, dense | +0.004 +- 0.003 | -0.0038 | 0.001 |
| 128 | +0.0037 +- 0.0011 | -0.00054 | 0.006 +- 0.004 |
| 512 | +0.0047 +- 0.0006 | -0.00015 | 0.036 +- 0.010 |
| 1,024 | +0.0047 +- 0.0004 | -0.000077 | 0.071 +- 0.016 |
| RL, per four steps (the same 192 prompts) | | | 0.040 |

The gain does not depend on N and is 11 standard errors from zero at 1,024 members: the
ranking carries a first-order signal that transfers to unseen prompts, hidden at 16 members
by the +-0.011 spread of random weightings. The cost falls as 1/N, 1.2 to 1.3 times a dense
update's at the same length. Per update ES matches RL per prompt at 512 members and exceeds
it at 1,024, at 32 and 64 times RL's 3,072 rollouts.

## 6. The gains accumulate, at half the one-update rate

`runs/lowrank-run/`: eight updates at 512 members and the best one-update step, along RL's
prompt stream (RL steps 1 to 32's data), held-out after 0, 4 and 8 updates.

| | held-out change | IFEval | MATH | GSM8K |
|---|---|---|---|---|
| ES after 4 updates | +0.018 +- 0.055 | | | |
| ES after 8 updates (predicted +0.29; no-accumulation bound +0.10) | +0.141 +- 0.061 | +0.47 (0.07) | -0.46 (0.15) | +0.10 (0.12) |
| RL at step 40, RL steps 121 to 160 | +0.44 | +0.81 | -0.05 | +0.20 |

ES moves in RL's direction, mostly IFEval, and damages MATH, which RL holds level. It
decoded 786,432 answers against RL's 24,576: about 15 times RL's GPU-seconds per unit of
held-out gain at the rates of section 8.

## 7. The estimate is a gradient estimate that works

`runs/gradient-check/`: GRPO's exact gradient `g` at the start on the RL run's batch for
the iteration-0 prompts (192 x 16 samples at temperature 1.0, the run's advantages and
loss; 91 of 192 groups have unequal rewards, the rest give RL's step no gradient, as ES's
dead prompts give its update none). Every measured update was projected onto it:
`rho_hat = u . g_hat / alpha` is the estimator's correlation with the gradient, at most 1.

| update | rho_hat toward reward (noise sd for a ranking with no signal) |
|---|---|
| low-rank, 1,024 members | +0.120 (0.031) |
| low-rank, 512 members, two independent decodes | +0.123, +0.116 (0.044) |
| dense, 16 members | unresolvable (0.25) |
| the run's updates 1 to 7, on the start's gradient | +0.02 +- 0.02: the gradient moves |

The held-out reward along the gradient (`theta_0 -+ lambda g_hat`, bf16 rounding included):

| direction | effective length | change toward reward | away | slope toward reward, per unit length |
|---|---|---|---|---|
| exact gradient | 0.022 | +0.14 | -0.19 | 7.3 +- 1.3 |
| exact gradient | 0.079 and beyond | model broken | | |
| Adam's sign direction (RL's actual first step) | 0.60 | +0.23 | -0.67 | 0.76 +- 0.06 |

- The exact gradient is concentrated: 993 of 7 billion elements carry 24% of its squared
  norm, so a step along it moves a few weights by a lot and breaks the model by length
  0.08. Adam's sign normalization is what makes RL's gradient usable: a tenth as steep,
  linear to length 0.6. ES's expected direction is a smoothed gradient, spread over all
  weights like the sign step; its first-order gain was linear to length 112.
- ES's useful component is at most `alpha` = 5e-4 long per unit and delivers 0.0047, so
  the slope along ES's expected direction is at least 9.4 per unit length. A perfect
  zeroth-order estimate of RL's gradient would gain `alpha` x 7.3 = 0.0037 +- 0.0007 per
  unit, 0.8 +- 0.15 of what ES's update already gains. No better fitness, control variate
  or smooth objective can raise the gain per unit by more than that.
- Neither can an optimizer. Adam's per-element normalization uses the update's own element
  magnitudes, which are 99.97% noise of the same size everywhere, so it reduces to one
  rescaling; momentum applies each random part once in total, spread over later steps;
  Muon's orthogonalization transforms the noise and the tiny useful part together. One
  ES iteration learns N numbers about an 8-billion-dimensional gradient, and no
  post-processing knows more than that. The one reshaping that could help uses prior
  knowledge of where the gradient lives: v_proj holds 27% of its squared norm in 2% of the
  parameters, so per-matrix-type step sizes could raise the best net per update by up to
  about 5 times if curvature is uniform across types, and by nothing if curvature tracks
  the gradient's density. Untested.
- RL's own displacement to step 120 has cosine 0.019 with this gradient: what 120 Adam
  steps of length 0.045 that are nearly orthogonal to one another give (`sqrt(120) x 0.045`
  = 0.49; the displacement is 0.476). A minibatch gradient is a small part of where RL
  goes too; it is the exactness of each step, not the gradient's constancy, that RL has.

## 8. Where the time goes

`runs/throughput-bench/`, one H200, per 192 prompts (four RL steps, one ES update at 512
members): RL decodes 3,072 answers (100 GPU-s), takes a gradient on them (538) and a
reference forward (146), 784 in all for +0.040; ES decodes 98,304 answers, 4,744 GPU-s at
7,000 tokens/s with LoRA members (10,600 without: a factor 1.5 for the kernels), for
+0.0365. Per unit of gain ES uses 6.6 times RL's GPU-seconds on one update and about 15
over eight; 3.5 with a screen that skips the prompts no member changes (54% of the
rollouts, 98.7% of the outcome changes kept; it also skips rare breakthroughs, so it should
spare prompts the model fails). A length cap does not pay: every cap gives up at least as
large a share of the outcome changes that rank the members as of the tokens it saves. These
are single-GPU rates; a real RL node adds communication and idling that ES's independent
members do not, by amounts not measured.

## 9. The mechanism

Both methods see the same prompts and generate the same answers, compared within each
prompt's group. RL turns each answer's relative reward into the exact weight change that
makes that answer more or less likely, so its step is entirely useful and it never
searches. ES learns one number per member, a mean over 192 prompts, and moves toward the
members that scored well: it infers the direction from how random perturbations of all 8
billion weights correlated with 16 to 1,024 scores. The inferred direction is right and
steep; the step along it is `alpha rho`, at most 5e-4, beside `alpha sqrt(d / N)` of random
movement whose cost in reward grows as its length squared and sets the step size.

Whether that trade is affordable depends on the task. On Countdown from a weak start, one
behaviour shared by every prompt gives a perturbation's score a large common part, and
the random part costs little because the model has little to lose: the margin of gain over
cost was about 600 for the 0.5B model, and ES learns. On Tulu after DPO, the remaining
gains are many small behaviours (IFEval's constraint types, MATH, GSM8K) that a random
perturbation touches one prompt at a time, so the common part of a score is small and the
random part damages a model that is already good: the margin was about 6 at 16 members,
and about 1 realized. That is a property of the regime, not of the dataset: Tulu is the
realistic post-training case, and the literature's ES successes are the other one. The
closest published comparisons agree in kind: ES's updates are orders of magnitude larger
than GRPO's and induce broader off-task drift at matched task accuracy on single tasks
(Hoy et al., COLM 2026, arXiv 2604.01499; Abdi et al. 2026, arXiv 2601.20861), and the
zeroth-order slowdown scales with the effective dimension of the problem (Malladi et al.
2023, arXiv 2305.17333). No published work runs ES on a mixed RLVR stream.

## 10. What ES would need, and what it would still lack

- **Rollouts.** The only lever on the random part is N. The best useful step grows in
  proportion to N: 0.0005 per update at 16 members, 0.015 at 1,024, RL's 0.045 at about
  3,000, if the cost keeps falling as 1/N. Low-rank members make that affordable in compute
  (`runs/lowrank-check/`), not in rollouts.
- **Throughput.** The dead-prompt screen (about 1.9x, with the caveat of section 8), LoRA
  kernels (at most 1.5x), per-matrix-type step sizes (1 to 5x, untested), speculative
  decoding from other members' answers (1.3 to 2.5x, untested). Together plausibly 3 to 6x,
  which would leave ES 2 to 5 times RL's compute per unit of gain.
- **What it would then have.** Inference-only memory (a 16 GB bf16 copy per GPU, no
  optimizer state, no reference model), communication of scalars only, any model that can
  be served (a quantized one through its own engine), and noise that does not grow with
  the answer length. Whether these pay for a 2 to 5x compute penalty depends on who is
  training; none was tested here.
- **What it would still lack.** Credit assignment. At matched data the useful step is
  shorter than RL's by the factor the rollout count has to make up, and it damaged MATH
  while gaining IFEval where RL did not.

## 11. Limitations

- One seed per arm and one step size; the long arm, the eight-update run and every probe
  are single runs with paired standard errors over prompts, not over seeds.
- The RL side is the released checkpoints, scored by us; the run was not reproduced, its
  wall clock is not measured, and the data matching assumes the card's command.
- Greedy decoding with the run's verifiers is the measure throughout; the held-out prompts
  are from the training distribution (later RL steps), not a separate benchmark.
- The engine holds bf16 weights, so small perturbations partly round away; every length is
  reported as what reached the model, and the slopes of section 7 are at the smallest
  lengths that allows.
- The exact gradient is one batch's, at the start, with the run's token-mean loss; RL's
  later steps carry Adam's state and clipping.
- First-order and quadratic models are fitted at the lengths measured and extrapolated no
  further than stated.

## 12. Records

All under `experiments/end_to_end/runs/`; each README cites its scripts and commits.

| record | question | cost |
|---|---|---|
| `pilot1-invalid/`, the pilot arms (`03`), `heldout-121-160/` (`05`) | does ES move within RL steps 1 to 120's data | about $80 |
| `countdown-s*/`, `update-check/` (`06`) | is the backend right | about $27 |
| `tulu-long-*/`, `heldout-481-520/` (`07`) | within RL steps 1 to 480's data | about $55 |
| `grad-check/` (with the sampled and Countdown records), `why/` | how much of the ranking is signal, and where the noise comes from | about $11 |
| `rl-geometry/`, `direction-check/` | how far and in what direction RL moves | about $50 (including an idle pod) |
| `contrastive-check/` | an exact, contrastive fitness | about $4 |
| `step-check/` | does one update deliver its implied gain | about $9 |
| `lowrank-check/` | gain and cost against N | about $18 |
| `throughput-bench/` | where the time goes | about $4 |
| `lowrank-run/` | do the gains accumulate | about $59 |
| `gradient-check/` | the estimate against the exact gradient | about $17 |

About $330 of the $350 budget by pod uptime; the rows are rounded.
