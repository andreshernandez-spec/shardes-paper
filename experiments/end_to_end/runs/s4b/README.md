# S4b, the run session: the first three arms (docs/end_to_end/10, "S4b, fresh settings for the verdict: design")

Three of the session's eight arms ran on 2026-10-16; the other five (A's un-anchored
control and second seed, and the three GRPO arms) did not start: Andres asked that
nothing new be started before a laptop shutdown, so the session script was stopped after
D had begun and D ran to its end alone. Each arm is `es_vllm/probe_run.py` at the
setting's committed prescription (`runs/s4b-probe/<setting>/prescription.json`, stamped
into `run.json`): the anchored curve-selected step with shrinkage 0.1 after every update,
64 mirrored rank-1 members at sigma 5e-4 on the 192 train prompts, 30 updates, the
600-prompt validation and off-task blocks and the 300-prompt reserved test block scored
every 5 updates. One Secure H200 (RunPod pod `dogmmah7a9vbea`, US-NC-1, $5.29/h), 13:20
to 22:48 UTC, 9.5 hours, about $50 by uptime (a first pod in EUR-IS-4, `h4zjdlby9ic7mv`,
was deleted after 50 minutes when its PyPI downloads ran at 1 to 2 MB/s, about $4). The
pod ran commit 38c14b9; the analysis is from 593485e (the prescribed run's prediction
read from its prescription).

| arm | setting | step | shrinkage | prescription's P1 | hours |
|---|---|---|---|---|---|
| `qwen1.5b-countdown-anchored-s0` | A, Qwen2.5-1.5B-Instruct on Countdown | 3.03 | 0.1 | learns | 1.7 |
| `qwen3b-quotation-anchored-s0` | C, Qwen2.5-3B-Instruct on the constructed Quotation task | 6.85 | 0.1 | learns | 2.0 |
| `qwen3b-math-anchored-s0` | D, Qwen2.5-3B-Instruct on MATH alone | 3.13 (the fallback) | 0.1 | undetermined | 4.8 |

Files per arm as in `runs/probe-run/`: `run.json` (configuration, the prescription,
block digests, environment), `log.jsonl` (per update), `eval.jsonl` (per evaluation,
every prompt's reward on the three blocks).

## The table

`python -m es_vllm.probe_run --table runs/s4b` (the frozen class of `10` item 6, kept
beside S4b's retention class below):

| run | preregistered call | step | variant | updates to target, predicted | updates | validation at end | test at end | off-task at end | peak test gain | at | validation-selected checkpoint | test there | class | class by test peak | class, task alone | collapsed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen1.5b-countdown-anchored-s0 | learns | 3.03 | shrink 0.1 | - | 30 | +0.148 | +0.148 | +0.025 | 0.149 | 20 | 30 | 0.148 | works | works | works | no |
| qwen3b-math-anchored-s0 | undetermined | 3.13 | shrink 0.1 | - | 30 | +0.012 | +0.017 | -0.005 | 0.0367 | 25 | 15 | -0.00333 | fails | fails | fails | no |
| qwen3b-quotation-anchored-s0 | learns | 6.85 | shrink 0.1 | - | 30 | +0.023 | +0.027 | -0.018 | 0.0367 | 10 | 10 | 0.0367 | fails | fails | fails | no |

## The curves

`python -m es_vllm.probe_run --curves runs/s4b`. "predicted validation" is the anchored
first-order forecast from the prescription: the odd and even responses at the applied
step's first displacement, weighted by the anchor's geometric update counts
(`probe_run.effective_updates`).

**qwen1.5b-countdown-anchored-s0** (step 3.03, shrink 0.1, 30 updates, 368,640 rollouts, 1.7 h)

| after | validation | test | off-task | answer length | predicted validation |
|---|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 235 | +0.000 |
| 5 | +0.082 | +0.069 | -0.002 | 231 | +0.071 |
| 10 | +0.116 | +0.116 | +0.010 | 233 | +0.107 |
| 15 | +0.128 | +0.127 | -0.002 | 240 | +0.126 |
| 20 | +0.137 | +0.149 | +0.022 | 237 | +0.136 |
| 25 | +0.140 | +0.142 | +0.012 | 248 | +0.142 |
| 30 | +0.148 | +0.148 | +0.025 | 243 | +0.146 |

**qwen3b-quotation-anchored-s0** (step 6.85, shrink 0.1, 30 updates, 368,640 rollouts, 2.0 h)

| after | validation | test | off-task | answer length | predicted validation |
|---|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 267 | +0.000 |
| 5 | -0.010 | +0.023 | -0.010 | 264 | +0.048 |
| 10 | +0.023 | +0.037 | -0.010 | 259 | +0.084 |
| 15 | +0.002 | +0.023 | -0.012 | 270 | +0.109 |
| 20 | +0.022 | +0.023 | -0.015 | 264 | +0.125 |
| 25 | +0.022 | +0.017 | -0.008 | 248 | +0.134 |
| 30 | +0.023 | +0.027 | -0.018 | 257 | +0.140 |

**qwen3b-math-anchored-s0** (step 3.13, shrink 0.1, 30 updates, 368,640 rollouts, 4.8 h)

| after | validation | test | off-task | answer length | predicted validation |
|---|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 492 | +0.000 |
| 5 | -0.013 | +0.023 | -0.003 | 477 | +0.032 |
| 10 | +0.008 | -0.020 | +0.013 | 478 | +0.052 |
| 15 | +0.012 | -0.003 | +0.000 | 483 | +0.065 |
| 20 | +0.003 | +0.003 | +0.000 | 489 | +0.072 |
| 25 | +0.008 | +0.037 | -0.020 | 489 | +0.076 |
| 30 | +0.012 | +0.017 | -0.005 | 486 | +0.079 |

## Retention (S4b's class, `probe_run.retention`)

The fixed margin of 0.03 on each retained family with the paired interval; the checkpoint
selected on the validation block among evaluations within the margin on the selection
half of the off-task block; the final reading on the reserved half.

| arm | class | selected | reserved block there | retained families on the reserved half at the selection |
|---|---|---|---|---|
| A | works | 30 | +0.148 | GSM8K +0.037 ± 0.018, within |
| C | fails | 10 | +0.037 | MATH −0.013 ± 0.027 unresolved; GSM8K −0.020 ± 0.020 unresolved |
| D | fails | 15 | −0.003 | GSM8K within; IFEval unresolved at most evaluations |

The 300-prompt halves resolve a 0.03 margin only on a block whose success rate is far
from one half (GSM8K at 0.75 on A, standard error 0.018); on C and D the retained
families' intervals (0.02 to 0.027) straddle the margin at most evaluations, so their
retention is unresolved rather than lost or kept.

## Against the predictions

| | A | C | D |
|---|---|---|---|
| P1 learns (reserved block +0.05 within 30) | learns: held, +0.070 at 5 and +0.148 at 30 | learns: **missed**, peak +0.037 ± 0.013 at 10, +0.027 at 30 | undetermined, not scored; peak +0.037 ± 0.020 at 25 |
| P3 pace (validation after 5, band a quarter to once the forecast) | +0.082 against +0.071: 1.15, just above the band | −0.010 against +0.048: below the band | +0.032 forecast, −0.013 observed; reported |
| the whole anchored forecast | tracks the curve to within 0.01 at every evaluation | overshoots from update 5 on (+0.140 forecast at 30, +0.023 observed) | overshoots (+0.079 at 30, +0.012 observed) |
| P2 collapses | no call | no call | no call |
| P4 RL's pace | the GRPO arm did not run | did not run | none |

A is the clean case: the probe's one-update responses at the applied step's first
displacement, carried through the anchor's geometric weights, reproduce 30 updates of
training on a setting the first probes never touched, and GSM8K is kept within the
margin. C and D are the ceiling and the no-signal cases: on C the start already scores
0.937 and the first update's +0.023 (3.3 standard errors) did not continue, so the
forecast's linear growth was wrong from the second evaluation on; on D the probe had no
resolved response and the fallback step moved nothing, as "undetermined" allows. The
gated prediction P1 is therefore one of two on this batch (A right, C wrong), with D not
scored; P4 has no reading because no GRPO arm ran. The quadratic rule had called C
"works" on its first-probe-style reading; the run says otherwise.

## Limitations

Three arms of eight; one seed each; no un-anchored control, no second population run,
no GRPO arm, so the stage's compute comparison and P4 are open. C is a constructed task
near its ceiling. The retention margin is unresolved on two of three arms at this block
size. The remaining five arms are the first thing a later session runs.
