# S4b, the run session: eight arms on three fresh settings (docs/end_to_end/10, "S4b, fresh settings for the verdict: design")

The stage's run session, in two parts. On 2026-10-16 the three anchored ES arms ran and
the session was then stopped at Andres's request before anything else started (a
laptop shutdown); on 2026-10-21 the remaining five arms ran: A's un-anchored control
and second seed, and one GRPO arm per setting. Every ES arm is `es_vllm/probe_run.py` at
the setting's committed prescription (`runs/s4b-probe/<setting>/prescription.json`,
stamped into `run.json`): the curve-selected step with shrinkage 0.1 after every update
(0 on the control), 64 mirrored rank-1 members at sigma 5e-4 on the 192 train prompts,
30 updates, the 600-prompt validation and off-task blocks and the 300-prompt reserved
test block scored every 5 updates. Every GRPO arm is `es_vllm/grpo_run.py` with the
Tulu 3.1 recipe at 5e-7 and beta 0.01, 40 steps of 768 rollouts, scored on the same
blocks. Pods: `dogmmah7a9vbea` (Secure H200, US-NC-1, $5.29/h, 13:20 to 22:48 UTC on the
16th, 9.5 hours, about $50; a first pod in EUR-IS-4 was deleted after 50 minutes when its
PyPI downloads crawled, about $4) and `hynw9b81o04l2h` (the same class, 10:48 to 18:08
UTC on the 21st, 7.3 hours, about $39). The pods ran commits 38c14b9 and 0b5f183. The
analysis is from the commit after the review of 2026-10-21, which added the corrected
forecast weights and separated the retention verdict from the task class; the records
and the frozen prescriptions are unchanged.

| arm | record | step | shrinkage | prescription's P1 | hours |
|---|---|---|---|---|---|
| A anchored, seed 0 | `qwen1.5b-countdown-anchored-s0` | 3.03 | 0.1 | learns | 1.7 |
| A anchored, seed 1 | `qwen1.5b-countdown-anchored-s1` | 3.03 | 0.1 | learns | 1.6 |
| A un-anchored control, seed 0 | `qwen1.5b-countdown-unanchored-s0` | 3.03 | 0 | learns | 1.8 |
| C anchored | `qwen3b-quotation-anchored-s0` | 6.85 | 0.1 | learns | 2.0 |
| D anchored | `qwen3b-math-anchored-s0` | 3.13 (the fallback) | 0.1 | undetermined | 4.8 |
| GRPO A | `../grpo-run/qwen1.5b-countdown-lr5e-7-s0` | | | | 0.7 in the loop |
| GRPO C | `../grpo-run/qwen3b-quotation-lr5e-7-s0` | | | | 1.1 |
| GRPO D | `../grpo-run/qwen3b-math-lr5e-7-s0` | | | | 1.7 |

A is Qwen2.5-1.5B-Instruct on Countdown (off-task GSM8K), C is Qwen2.5-3B-Instruct on the
constructed single-constraint Quotation task (off-task GSM8K and MATH; `10` and
`probe.quotation_rows`), D is Qwen2.5-3B-Instruct on MATH alone (off-task GSM8K and
IFEval). Files per arm as in `runs/probe-run/` and `runs/grpo-run/`.

## Two conventions for the forecast

The prescription reads the probe's odd and even responses at the applied step's first
displacement, 0.9 of the step, which already carries one pull-back. The frozen S4b
prescriptions and the forecasts written before the runs weighted those responses with
`probe_run.effective_updates`, whose weights start at exponent one, so they counted the
first pull-back twice (the review of 2026-10-21): at five updates 3.686 for the gain and
2.777 for the random part's variance instead of 4.095 and 3.428. The curves below show
both: "predicted validation" is the frozen forecast the runs were held against, and
"forecast, corrected weights" is the same responses with `effective_updates_from_first`
(weights from exponent zero; at one update the forecast is the measured change at the
first displacement itself). The frozen P3 bands are kept as scored; the corrected
forecast is the one to read for how well one update's responses forecast a run.

## The tables

`python -m es_vllm.probe_run --table runs/s4b` (the frozen class of `10` item 6; S4b's
task class and retention verdict are below):

| run | preregistered call | step | variant | updates to target, predicted | updates | validation at end | test at end | off-task at end | peak test gain | at | validation-selected checkpoint | test there | class | class by test peak | class, task alone | collapsed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen1.5b-countdown-anchored-s0 | learns | 3.03 | shrink 0.1 | - | 30 | +0.148 | +0.148 | +0.025 | 0.149 | 20 | 30 | 0.148 | works | works | works | no |
| qwen1.5b-countdown-anchored-s1 | learns | 3.03 | shrink 0.1 | - | 30 | +0.141 | +0.147 | +0.025 | 0.147 | 30 | 25 | 0.137 | works | works | works | no |
| qwen1.5b-countdown-unanchored-s0 | learns | 3.03 | - | - | 30 | +0.185 | +0.180 | -0.025 | 0.183 | 15 | 20 | 0.171 | works | works | works | yes |
| qwen3b-math-anchored-s0 | undetermined | 3.13 | shrink 0.1 | - | 30 | +0.012 | +0.017 | -0.005 | 0.0367 | 25 | 15 | -0.00333 | fails | fails | fails | no |
| qwen3b-quotation-anchored-s0 | learns | 6.85 | shrink 0.1 | - | 30 | +0.023 | +0.027 | -0.018 | 0.0367 | 10 | 10 | 0.0367 | fails | fails | fails | no |

`python -m es_vllm.grpo_run --table runs/grpo-run`, the S4b arms:

| arm | lr | beta | seed | steps | rollouts | validation at end | test at end | off-task at end | peak test gain | at | test +0.10 first at | validation-selected checkpoint | test there | class | collapsed | s per step | GPU-s per answer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen1.5b-countdown-lr5e-7-s0 | 5e-07 | 0.01 | 0 | 40 | 30,720 | +0.181 | +0.192 | -0.002 | 0.192 | 40 | 25 | 40 | 0.192 | works | no | 64.1 | 0.0834 |
| qwen3b-math-lr5e-7-s0 | 5e-07 | 0.01 | 0 | 40 | 30,720 | +0.013 | +0.003 | -0.015 | 0.03 | 25 | - | 20 | 0.01 | fails | no | 155 | 0.202 |
| qwen3b-quotation-lr5e-7-s0 | 5e-07 | 0.01 | 0 | 40 | 30,720 | +0.042 | +0.033 | -0.008 | 0.04 | 35 | - | 40 | 0.0333 | fails | no | 98.9 | 0.129 |

## The curves

`python -m es_vllm.probe_run --curves runs/s4b` and `python -m es_vllm.grpo_run --curves
runs/grpo-run`. The two forecast columns are explained above; on the un-anchored control
both conventions coincide (no pull-back, weights equal to the update count).

**qwen1.5b-countdown-anchored-s0** (step 3.03, shrink 0.1, 30 updates, 368,640 rollouts, 1.7 h)

| after | validation | test | off-task | answer length | predicted validation | forecast, corrected weights |
|---|---|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 235 | +0.000 | +0.000 |
| 5 | +0.082 | +0.069 | -0.002 | 231 | +0.071 | +0.082 |
| 10 | +0.116 | +0.116 | +0.010 | 233 | +0.107 | +0.123 |
| 15 | +0.128 | +0.127 | -0.002 | 240 | +0.126 | +0.144 |
| 20 | +0.137 | +0.149 | +0.022 | 237 | +0.136 | +0.156 |
| 25 | +0.140 | +0.142 | +0.012 | 248 | +0.142 | +0.163 |
| 30 | +0.148 | +0.148 | +0.025 | 243 | +0.146 | +0.167 |

**qwen1.5b-countdown-anchored-s1** (step 3.03, shrink 0.1, 30 updates, 368,640 rollouts, 1.6 h)

| after | validation | test | off-task | answer length | predicted validation | forecast, corrected weights |
|---|---|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 235 | +0.000 | +0.000 |
| 5 | +0.085 | +0.052 | +0.003 | 230 | +0.071 | +0.082 |
| 10 | +0.108 | +0.119 | +0.007 | 221 | +0.107 | +0.123 |
| 15 | +0.131 | +0.133 | +0.013 | 226 | +0.126 | +0.144 |
| 20 | +0.132 | +0.133 | +0.013 | 235 | +0.136 | +0.156 |
| 25 | +0.145 | +0.137 | +0.020 | 243 | +0.142 | +0.163 |
| 30 | +0.141 | +0.147 | +0.025 | 245 | +0.146 | +0.167 |

**qwen1.5b-countdown-unanchored-s0** (step 3.03, 30 updates, 368,640 rollouts, 1.8 h)

| after | validation | test | off-task | answer length | predicted validation | forecast, corrected weights |
|---|---|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 235 | +0.000 | +0.000 |
| 5 | +0.100 | +0.085 | -0.010 | 241 | +0.107 | +0.107 |
| 10 | +0.146 | +0.145 | -0.002 | 252 | +0.214 | +0.214 |
| 15 | +0.162 | +0.183 | -0.005 | 247 | +0.321 | +0.321 |
| 20 | +0.179 | +0.171 | -0.002 | 249 | +0.428 | +0.428 |
| 25 | +0.177 | +0.170 | +0.012 | 251 | +0.535 | +0.535 |
| 30 | +0.185 | +0.180 | -0.025 | 243 | +0.642 | +0.642 |

**qwen3b-quotation-anchored-s0** (step 6.85, shrink 0.1, 30 updates, 368,640 rollouts, 2.0 h)

| after | validation | test | off-task | answer length | predicted validation | forecast, corrected weights |
|---|---|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 267 | +0.000 | +0.000 |
| 5 | -0.010 | +0.023 | -0.010 | 264 | +0.048 | +0.048 |
| 10 | +0.023 | +0.037 | -0.010 | 259 | +0.084 | +0.088 |
| 15 | +0.002 | +0.023 | -0.012 | 270 | +0.109 | +0.115 |
| 20 | +0.022 | +0.023 | -0.015 | 264 | +0.125 | +0.132 |
| 25 | +0.022 | +0.017 | -0.008 | 248 | +0.134 | +0.142 |
| 30 | +0.023 | +0.027 | -0.018 | 257 | +0.140 | +0.149 |

**qwen3b-math-anchored-s0** (step 3.13, shrink 0.1, 30 updates, 368,640 rollouts, 4.8 h)

| after | validation | test | off-task | answer length | predicted validation | forecast, corrected weights |
|---|---|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 492 | +0.000 | +0.000 |
| 5 | -0.013 | +0.023 | -0.003 | 477 | +0.032 | +0.035 |
| 10 | +0.008 | -0.020 | +0.013 | 478 | +0.052 | +0.057 |
| 15 | +0.012 | -0.003 | +0.000 | 483 | +0.065 | +0.071 |
| 20 | +0.003 | +0.003 | +0.000 | 489 | +0.072 | +0.079 |
| 25 | +0.008 | +0.037 | -0.020 | 489 | +0.076 | +0.084 |
| 30 | +0.012 | +0.017 | -0.005 | 486 | +0.079 | +0.087 |

**qwen1.5b-countdown-lr5e-7-s0** (lr 5e-07, beta 0.01, seed 0, 40 steps, 30,720 rollouts, 0.7 h in the loop)

| after | validation | test | off-task | answer length |
|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 235 |
| 5 | +0.021 | +0.029 | +0.013 | 148 |
| 10 | +0.032 | +0.033 | +0.010 | 149 |
| 15 | +0.042 | +0.030 | +0.005 | 134 |
| 20 | +0.047 | +0.040 | -0.002 | 121 |
| 25 | +0.149 | +0.150 | +0.013 | 119 |
| 30 | +0.177 | +0.177 | +0.003 | 118 |
| 35 | +0.178 | +0.186 | +0.012 | 116 |
| 40 | +0.181 | +0.192 | -0.002 | 116 |

**qwen3b-quotation-lr5e-7-s0** (lr 5e-07, beta 0.01, seed 0, 40 steps, 30,720 rollouts, 1.1 h in the loop)

| after | validation | test | off-task | answer length |
|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 267 |
| 5 | +0.002 | +0.003 | +0.003 | 264 |
| 10 | +0.005 | +0.013 | -0.007 | 265 |
| 15 | +0.033 | +0.030 | -0.017 | 255 |
| 20 | +0.027 | +0.030 | -0.003 | 262 |
| 25 | +0.033 | +0.030 | -0.015 | 256 |
| 30 | +0.032 | +0.033 | +0.000 | 254 |
| 35 | +0.032 | +0.040 | +0.002 | 257 |
| 40 | +0.042 | +0.033 | -0.008 | 258 |

**qwen3b-math-lr5e-7-s0** (lr 5e-07, beta 0.01, seed 0, 40 steps, 30,720 rollouts, 1.7 h in the loop)

| after | validation | test | off-task | answer length |
|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 492 |
| 5 | +0.007 | +0.013 | -0.002 | 491 |
| 10 | +0.003 | +0.003 | -0.017 | 480 |
| 15 | +0.005 | +0.010 | -0.008 | 484 |
| 20 | +0.017 | +0.010 | -0.018 | 488 |
| 25 | -0.007 | +0.030 | -0.013 | 490 |
| 30 | +0.007 | +0.010 | -0.010 | 483 |
| 35 | +0.008 | +0.023 | -0.002 | 480 |
| 40 | +0.013 | +0.003 | -0.015 | 490 |

## Compute to the target, and the cost of a prescription

`python -m es_vllm.probe_run --compare runs/s4b runs/grpo-run`: loop seconds and rollouts
through the first evaluation at +0.10 on the reserved test block, against the setting's
GRPO arm. Only A reached the target under either method.

| ES run | +0.10 at update | ES loop s | ES rollouts | off-task there | retained | GRPO +0.10 at step | GRPO loop s | GRPO rollouts | GRPO off-task there | seconds, ES/GRPO | rollouts, ES/GRPO |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen1.5b-countdown-anchored-s0 | 10 | 1,833 | 122,880 | +0.010 | yes | 25 | 1,635 | 19,200 | +0.013 | 1.12 | 6.40 |
| qwen1.5b-countdown-anchored-s1 | 10 | 1,800 | 122,880 | +0.007 | yes | 25 | 1,635 | 19,200 | +0.013 | 1.10 | 6.40 |
| qwen1.5b-countdown-unanchored-s0 | 10 | 1,955 | 122,880 | -0.002 | yes | 25 | 1,635 | 19,200 | +0.013 | 1.20 | 6.40 |
| qwen3b-math-anchored-s0 | - | - | - | - | - | - | - | - | - | - | - |
| qwen3b-quotation-anchored-s0 | - | - | - | - | - | - | - | - | - | - | - |

The crossings are at the first evaluation at or above the target (update 10 and step
25), upper bounds on the compute; GRPO's answers shortened from 235 to 116 tokens on A,
ES's did not. The loop seconds exclude what the prescription cost: on A the probe took
about 3,100 s of pod time (sample, gradient, scans, members and steps about 1,300 s; the
off-task decodes that feed the exploratory P2 another 1,788 s), against GRPO's 1,635 s
to the target. Training-loop efficiency and the cost of obtaining a prescription are
separate numbers, and a practical method that drops the P2 decodes is priced at the
smaller one.

## Task class and retention (S4b's class, `probe_run.retention`)

The task class is the reserved test block's change at the checkpoint selected on the
validation block among evaluations within the 0.03 margin on the selection half of the
off-task block. The retention verdict is read separately, at that checkpoint, on the
reserved half of the off-task block with the paired interval: "within" when the
interval's lower bound is above −0.03 on every family, "lost" when any family's upper
bound is below it, "unresolved" otherwise. "Works" as a joint verdict needs both; a
task class of "works" with unresolved retention is reported as such and not as
established joint success. The reserved half is reserved from checkpoint selection
only: the probe decoded it (the scans and the off-task member decodes use the whole
off-task block, and the curve rule's retained bound reads its even response), so it
influenced the step. The next preregistration reserves a retained-task test block the
probe never decodes (`Task(..., offtask_test=True)`, `scan_rows` leaves it out, the
runs score it, `retention` reads it when present).

| arm | task class | retention verdict (reserved half) | joint | selected | reserved block there | retained families at the selection |
|---|---|---|---|---|---|---|
| A anchored, seed 0 | works | within | works | 30 | +0.148 | GSM8K +0.037 ± 0.018 |
| A anchored, seed 1 | works | unresolved | works on the task; retention unresolved | 25 | +0.137 | GSM8K +0.003 ± 0.018 |
| A un-anchored | works | unresolved | works on the task; retention unresolved | 30 | +0.180 | GSM8K −0.040 ± 0.021 |
| C | fails | unresolved | fails | 10 | +0.037 | MATH −0.013 ± 0.027; GSM8K −0.020 ± 0.020 |
| D | fails | unresolved | fails | 15 | −0.003 | GSM8K −0.007 ± 0.020; IFEval −0.007 ± 0.022 |

The GRPO arms' off-task blocks stay within 0.02 of their starts on all three settings.

## Against the predictions

| | A (three ES arms, one GRPO) | C | D |
|---|---|---|---|
| P1 learns (reserved block +0.05 within 30) | learns: held on all three arms (+0.070 at 5 on the first, +0.148, +0.147 and +0.180 at 30) | learns: **missed**, peak +0.037 ± 0.013 at 10, +0.027 at 30 | undetermined, not scored; peak +0.037 ± 0.020 at 25 |
| P3 pace (validation after 5, band a quarter to once the frozen forecast of +0.071) | anchored: +0.082 and +0.085, 1.16 and 1.20 of the frozen forecast, just above its band; against the corrected forecast of +0.082, 1.00 and 1.04. The control +0.100 against the un-anchored +0.107 | −0.010 against +0.048: below the band | +0.032 forecast, −0.013 observed; reported |
| the whole forecast, corrected weights | matches both anchored arms through update 10 (+0.082 and +0.123 against +0.082 to +0.085 and +0.108 to +0.116) and then overshoots, by 0.019 to 0.026 at 30 (+0.167 against +0.148 and +0.141); the un-anchored forecast, linear in the update count, overshoots the control from update 10 on | overshoots from update 5 on (+0.149 at 30, +0.023 observed) | overshoots (+0.087 at 30, +0.012 observed) |
| P2 collapses | no call; GSM8K ended +0.025, +0.025 and −0.025 | no call; both families within 0.02 | no call; both families within 0.01 |
| P4 RL's pace (GRPO at +0.10 within a factor of two of the predicted step) | step 23 predicted, band 11.5 to 46: crossed between 20 and 25, held | step 9.7 predicted, band 4.8 to 19.4: never reached (+0.033 at 40): a miss | no prediction (RL's slope unresolved); never reached |

The gated predictions on this batch: P1 one of two settings with a call (A right, C
wrong; D unscored); P4 one of two settings with a prediction (A right, C wrong). Both
misses are on C. C starts at 0.937, which leaves 0.063 of headroom: the +0.10 RL target
was impossible there, and the +0.05 learning target was possible but needed about 79%
of the remaining errors recovered; the probe's response (+0.023 at the first
displacement, 3.3 standard errors) was real and did not continue past the first updates
for either method (ES peaked at +0.037, GRPO reached +0.033 at 40). Those misses stand
as scored; what they teach is that targets stated in absolute reward have to be checked
against the start's headroom before they are frozen, and that a headroom-relative target
still needs a minimum detectable effect so that a small target does not make a forecast
look precise that the evaluation cannot resolve. D is the no-signal case: neither
method moved MATH alone in this budget (ES +0.017, GRPO +0.003 on the reserved block);
the probe's undetermined call and the absent RL slope said as much, but an abstention
followed by little learning on both sides is not a successful negative prediction. P2
made no call on any setting, so its accuracy is untested, not failed.

A is the clean case for the method and the readable case for the forecast: three ES
arms learned at a 3.0-unit anchored step, reached the target at 1.1 to 1.2 times GRPO's
loop seconds (6.4 times its rollouts), kept GSM8K within the margin on one anchored seed
(unresolved on the other and on the control), and the corrected forecast from one
update's responses matched the first ten updates on both seeds before the runs
saturated below it. The anchor cost about 0.03 of task gain against the control and left
GSM8K at +0.025 against the control's −0.025 at 30, within noise either way. What A does
not show is that one probe reliably chooses that step: A's second population would have
prescribed 15.8 units, not 3.0 (`runs/s4b-probe/README.md`), and the replication of
learning at the chosen smaller step says nothing about that choice.

The quadratic rule's calls on the same probe records were "fails: no odd response" (A,
population 1), "works" (C, ratio 4.3) and "fails: no odd response" (D): wrong on A and
C; on D it agrees with an outcome neither method reached. The exploratory curve rule
prescribed the steps that worked on A and called C undetermined.

## Limitations

One seed on C and D and on the GRPO arms; one population per prescription, and A's two
populations disagree on the step. C is a constructed task near its ceiling, with targets
frozen before its start reward was known. The retention margin is unresolved on four of
five ES arms at this block size, and the reserved half it is read on was decoded by the
probe. The compute comparison exists on one setting and excludes the prescription's own
cost. Three fresh settings are a pilot, not a verdict.
