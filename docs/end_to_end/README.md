# ES against released RL checkpoints: the plan, and where it stands

One page to review the whole program from. Each line says what was planned, whether it
is done, and where the details and the evidence are. Status words: **done**, **running**,
**open** (planned, not started), **deferred** (parked by decision), **dropped**. Dates
are when a stage was decided or finished; the decisions themselves are listed in
`review/01-goal-and-problems.md`. Updated 2026-10-13, after S3b.

## The question, as it has moved

| # | question | status | where |
|---|---|---|---|
| Q1 | Retrained with shardes' ES on exactly the data a released RL checkpoint was trained on, from the same start, does an open model learn what the RL run learned, at matched rollouts? | **done, answered no** on Tulu 3.1 8B: ES at RL's rollouts ends 0.25 below its start where RL gains 1.87 (a gap of 2.12) | `00-plan.md`, `01-tulu31-plan.md`, `07-tulu31-long.md`, `papers/end_to_end/draft.md` |
| Q2 | Why? What separates RL's learning from ES's on this data? | **done, a supported but provisional explanation**: the members' ranking is mostly prompt-specific effects, one update gains what its random part costs, the gain per unit is constant in N while the cost falls as 1/N; the estimate is a working gradient estimate, the useful step is too short beside the random part. Not isolated: why the quadratic prediction fails over many updates | `08-why-rl-learns-and-es-does-not.md`, `09-best-achievable.md`, `review/` |
| Q3 | Can it be told a priori, without trial and error, when ES will be effective and when RL will dominate? | **running**: a rule and two levels of predictors preregistered, six settings probed, the first training runs against the predictions on the pod now | `10-when-does-es-work.md`, `11-literature-es-vs-rl.md`, `runs/probe/README.md` |

The medium-term aim behind Q3 (stated 2026-10-10 and 12): find where ES has value against
RL, and if its data efficiency does not reach RL's, say where it is useful and where it
is not, with a rule that can be applied before spending the compute.

## The stages

Costs are by pod uptime. "Record" is a directory under `experiments/end_to_end/runs/`
unless another path is given; every number in a document comes from one.

| stage | what | status | result in one line | details | record | cost |
|---|---|---|---|---|---|---|
| S0 plan and data | settings, what is held equal, engine choice; Tulu 3.1's training set and per-step prompt stream rebuilt from open-instruct 3f37c29; every external release pinned by commit and weight hash | **done** (2026-09-26) | 29,689 rows, the stream reproduced exactly; Olmo 3 IF's set rebuilt too | `00-plan.md`, `01-tulu31-plan.md`, `02-tulu31-data-path.md` | `data/tulu31/`, `pins/` | $0 |
| S0 throughput probe | cost of a matched run on A100 and H100 | **done** | Tulu matched run $122 to 159; Olmo 3 base loops to the 16k cap | `00-plan.md` | `probe/` | $3.9 |
| S1 ES backend on vLLM | shardes trains, vLLM generates; weights bit-identical to the library's; restore by recomputation | **done** | smoke and bit-for-bit checks pass; two faults found and fixed (a JAX-to-torch race, an H200 nondeterminism in `lax.scan`) | `01-tulu31-plan.md` (T1), `review/01-goal-and-problems.md` (problems) | `es_vllm/`, `tests/end_to_end/`, `update-check/`, `replay-check/` | $9 |
| T2 Countdown check | our backend against es-at-scale's own trainer on its 0.5B Countdown example, two seeds each | **done** (2026-10-06) | ours learns faster (AUC 0.182 vs 0.159), same level by 100 updates; the two-sided gate fails on the high side, read as "the backend works" | `06-countdown-check.md` | `countdown-README.md`, `countdown-s1*`, `countdown-s2*` | $18 |
| T3 harness gate | OLMES scoring of a released checkpoint against its published numbers | **dropped** | superseded by in-run scoring of held-out stream prompts against the RL branches (S5) | `04-tulu31-harness.md` | | $0 |
| T4 pilot | 30 iterations at N=16 (RL's 16 samples per prompt), three sigmas and a random control | **done** (2026-09-27) | no arm passes the gate; best slope 1.7 SE; first run invalid (race), kept as evidence | `03-tulu31-pilot.md` | `tulu-pilot-*`, `pilot1-invalid/`, `pilot-gate.json` | $70 |
| T5 long arm | the pilot's s5e-4 arm to 120 iterations (RL steps 1 to 480 in prompts and rollouts), random control | **done** (2026-10-07) | gate passes against the control (+0.26 ± 0.09) but the arm ends 0.25 below its start; RL at step 480 is +1.87 above the start, 2.12 above the arm | `07-tulu31-long.md` | `tulu-long-README.md`, `tulu-long-*`, `long-gate.json` | $55 |
| T6 held-out context | the RL branches and the ES arms scored on the prompts of RL steps 121 to 160 and 481 to 520 | **done** | DPO 5.43, RL 5.87 / 6.40 / 6.69 at steps 40 / 80 / 120; ES 5.42 | `05-tulu31-heldout.md` | `heldout-121-160/`, `heldout-481-520/` | $7 |
| D diagnosis | why ES does not learn here: noise source, RL's geometry, contrastive fitness, Countdown contrast, the step probe, N scaling with low-rank members, eight updates at 512, throughput, the exact gradient, gain and cost by matrix type | **done** (2026-10-07 to 10) | synthesis in `08`; the experiments E7 to E18 with their numbers in `review/02` | `08-why-rl-learns-and-es-does-not.md`, `review/02-experiments-and-evidence.md` | `grad-check/`, `why/`, `rl-geometry/`, `direction-check/`, `contrastive-check/`, `step-check/`, `lowrank-check/`, `lowrank-run/`, `throughput-bench/`, `gradient-check/`, `block-check/` | $163 |
| S2 best achievable | 1,024 members, a step per matrix type, a dead-prompt screen; gate: ES GPU-seconds per unit held-out gain within 3x RL's | **done, gate failed** (2026-10-11) | held-out −0.32 ± 0.07 after eight updates: IFEval +0.65 at RL's pace, MATH −1.81; the pooled cost model over-steps the reasoning tasks | `09-best-achievable.md` (preregistration and result) | `block-check/`, `lowrank-run-1024/` | $94 |
| R review | a self-contained package for a fresh reviewer, the review, the revision | **done** (2026-10-11 to 12) | 11 findings accepted (the "ceiling" is not one; the model is Hoy et al.'s decomposition; validation data; numeric fixes); draft rewritten under the reviewer's title | `review/00-index.md` to `04-review-questions.md`; the review itself in `drafts/` (local, never committed) | `papers/end_to_end/draft.md` | $0 |
| L literature | ES failures, poor data efficiency and head-to-head comparisons with RL, with a-priori predictors | **done** (2026-10-12, agent compilation, provenance-labelled) | matched-budget GRPO wins in the LLM record (Abdi 7 of 8, Ba, Hoy at about 15x rollouts); ES wins at matched wall clock in EGGROLL (RWKV-7 Countdown; a Qwen3-4B base model on 40k DeepScaleR questions, so neither instruction tuning nor prompt repetition is an established condition for ES, both are hypotheses); three published predictors (Hoy's chi, Cho and Han's q, Liang's best-of-N) | `11-literature-es-vs-rl.md` | | $0 |
| S3a the rule and the probe | progress per update ≈ αρD − cα²d/N; level-0 (nothing ranked) and level-1 (one iteration) predictors; six settings probed, predictions written before any run | **done** (2026-10-12) | 3B instruct Countdown "works" (ES/RL rollouts 2.8), 3B base "marginal" with GSM8K collapsing along the update's direction, 0.5B GSM8K "fails", 1.5B "works" on an unresolved cost, 3B GSM8K no odd response, 7B "marginal"; of five expectations two held, two mostly, one wrong as written, the level-0 test failed (Spearman 0.54) | `10-when-does-es-work.md` (rule, levels, settings, gates, results) | `probe/README.md`, `probe/<setting>/` | $16 |
| S3b training runs against the predictions | 3B instruct Countdown, 3B base Countdown, 0.5B GSM8K at the rule's prescription, 30 updates at 64 members, scored on the reserved test blocks every 5, two seeds each | **done** (2026-10-13, 13.4 h on one H200) | the calls held in kind on all six runs on the task alone; the prescription's size was off by two to four; every Countdown run lost its off-task block late at either step and on every seed, which no one-update probe sees; the frozen class flips with the seed near a threshold | `10-when-does-es-work.md` ("Results of the training runs"), `es_vllm/probe_run.py` | `probe-run/README.md`, `probe-run/<setting>-s<seed>/` | $71 |
| S3c a measured RL baseline, then the controls | GRPO trained by us on one cheap setting (3B Countdown or 1.5B GSM8K) with its learning curve and measured cost, the reference the rule's ratios have lacked (they are forecasts against the Tulu run's step and costs until then); then the repeated-against-stream control with the stream guard; the remaining probes after that | **open**, the RL baseline needs a decision (it reverses the 2026-09-26 rule that the RL side is only released checkpoints) | | `10-when-does-es-work.md` | | $15 to 30 for the baseline; $120 to 180 for the probes |
| S3d calibration run | Tulu 8B on Countdown at Qiu et al.'s settings (30 members, σ 1e-3, 200 prompts, 500 iterations) | **deferred** (2026-10-13): Tulu DPO and Qiu's Llama Instruct are different starting checkpoints, so this is a transfer experiment, not a calibration anchor | | `10-when-does-es-work.md` | | $60 to 80 |
| S3e the rule's verdict | over at least six settings with a run or a published outcome: predicted order right (Spearman ≥ 0.8), the three-way call right in five of six, level 0 predicts level 1 | **open**: the outcome classes and the unfinished-target rule are frozen in `10` before any further setting; the six probed settings, concentrated in two task families, are a pilot; the revised level-0 hypothesis (the update's odd response as the target, Spearman 0.83 on the same six records, exploratory) is tested only on fresh settings | | `10-when-does-es-work.md` ("What counts as the rule being useful", "Corrections of 2026-10-13") | | |
| P paper | the manuscript | **open**: the Tulu comparison (the draft) is the defensible main result; the predictive stage enters as exploratory until the checks in S3c and S3e succeed | | `papers/end_to_end/draft.md`, `papers/end_to_end/README.md` | | |
| O Olmo 3 | the second setting of the original plan (7B RL-Zero IF) | **deferred** (2026-09-26): 2,900-token answers at temperature 1 and 16k-token loops under greedy decoding put a matched run at $2k to 6k | | `00-plan.md` | | |

## The evidence, by experiment

`review/02-experiments-and-evidence.md` describes E1 to E19 one by one with their numbers
and records; `review/03-metrics-explained.md` defines every metric from scratch. The
headline of each, with the two newer ones:

| experiment | headline | record |
|---|---|---|
| E1 Phase 0, throughput probe | data paths exact; matched run $122 to 159 | `probe/`, `data/tulu31/` |
| E2 pilot | no arm passes; the first run invalid (race) | `tulu-pilot-*`, `pilot1-invalid/` |
| E3 held-out context | RL +1.26 by step 120; ES at its start | `heldout-121-160/` |
| E4 H200 nondeterminism | 0.4% of `lax.scan` calls wrong; fixed by one jitted step per member | `update-check/`, `replay-check/` |
| E5 Countdown check | ours faster than es-at-scale, same level | `countdown-*` |
| E6 long arm | +0.26 vs control, −0.25 vs start, RL +1.87 | `tulu-long-*` |
| E7 gradient probe | 12% of the members' fitness variation is real (reliability 0.12) | `grad-check/` |
| E8 noise source | member outcomes are redraws at near-ties; cross-prompt correlation 0.004 | `why/` |
| E9 RL's geometry | RL's step-120 model 0.476 from the start; +0.85 along its direction at 0.32 | `rl-geometry/`, `direction-check/` |
| E10 contrastive fitness | exact scoring, ranking still 36% shared | `contrastive-check/` |
| E11 Countdown contrast | same model: cross-prompt correlation 0.30, reliability 0.42 to 0.68 | `grad-check/countdown-*` |
| E12 literature (first two searches) | nobody ran ES on a mixed RLVR stream; replications find ES updates far larger than GRPO's | `08`, section 9 |
| E13 step probe | second-order cost as predicted (−0.0038/unit²); gain +0.004 ± 0.003 per unit | `step-check/` |
| E14 N scaling, low-rank | gain per unit constant, cost ∝ 1/N; fitted net 0.071 ± 0.016 at 1,024 vs RL 0.040 | `lowrank-check/` |
| E15 eight updates at 512 | +0.141 ± 0.061 held-out, IFEval +0.47, MATH −0.46 | `lowrank-run/` |
| E16 throughput | ES 6.6x RL GPU-s per unit gain (one update), ~15x over eight | `throughput-bench/` |
| E17 exact gradient | ES's update projects 0.12 ± 0.03 onto GRPO's gradient; no bug | `gradient-check/` |
| E18 gain and cost by type | v_proj 17% of the gain at 2% of the parameters, but 11x the curvature | `block-check/` |
| E19 1,024 members, per-type steps | −0.32 ± 0.07; IFEval +0.65, MATH −1.81 | `lowrank-run-1024/` |
| E20 the probe, six settings | the table in `probe/README.md`; predictions before any run | `probe/` |
| E21 training runs at the prescription | running; first run above | `probe-run/` |

## Open items

Decisions for Andres:

- The measured GRPO baseline (S3c, $15 to 30), the one experiment the rule's RL side
  needs; RunPod holds about $55 after S3b.
- Whether the paper becomes the Tulu comparison alone (the draft) or the predictive
  account with the Tulu result as its anchor.
- Two seeds per run only for the cheap settings; the 8B settings run one seed.

Work in hand or queued (mine):

- A level-1 off-task predictor measured over several updates (the members decoding the
  off-task block, or a short run), since the one-update number cannot see the late
  collapse; a step schedule or re-measurement as a separately specified method, tuned on
  validation prompts and tested on fresh ones (`10`, corrections item 7).
- The probe's scan extension (commit c945036) applies to every later setting; the
  level-0 test has to be re-stated against the update's odd response rather than the
  ranking's reliability (the 7B record: reliability 0, update at 3.6 SE).
- Literature: read each source cited from `11` at the source before it enters the
  paper; the file is an agent compilation with provenance labels.
- Known limits carried from the review: one seed on the Tulu runs; RL's wall clock
  estimated from per-token rates, not measured; the task-specific cost measured on
  low-rank members only; the structural advantages of ES (inference-only memory, scalar
  communication, quantized models) untested.

## Rules that bind all of it

Preregistered predictions before data, records never overwritten, every number from a
committed script and record, configs committed before a run and cited by commit, a
validation block for choices and a reserved test block for results (`10`, "Rules carried
over"); the repository's rules in `CLAUDE.md` (pins, no rebase of cited branches,
generated tables never edited by hand); the tree's rules one level up (identity, no
push and no PR to upstream projects, drafts stay local). Reviews, outreach and anything
written for a particular reader stay in `drafts/`, which is never committed.

## Budget

Program budget $450 ($350 on 2026-09-26, a second $100 for S2 on 2026-10-10); about
$441 spent by uptime through S3a. S3b ran on the RunPod balance of $126 reported on
2026-10-13 and cost about $71 by uptime, leaving about $55. Every stage's cost above is by pod uptime, which runs
about 20% over the drivers' measured time.
