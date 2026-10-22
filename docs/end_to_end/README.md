# ES against released RL checkpoints: the plan, and where it stands

One page to review the whole program from. Each line says what was planned, whether it
is done, and where the details and the evidence are. Status words: **done**, **running**,
**open** (planned, not started), **deferred** (parked by decision), **dropped**. Dates
are when a stage was decided or finished; the decisions themselves are listed in
`review/01-goal-and-problems.md`. Updated 2026-10-14, after S3c and a second review, with the next stages proposed.

## The question, as it has moved

| # | question | status | where |
|---|---|---|---|
| Q1 | Retrained with shardes' ES on exactly the data a released RL checkpoint was trained on, from the same start, does an open model learn what the RL run learned, at matched rollouts? | **done, answered no** on Tulu 3.1 8B: ES at RL's rollouts ends 0.25 below its start where RL gains 1.87 (a gap of 2.12) | `00-plan.md`, `01-tulu31-plan.md`, `07-tulu31-long.md`, `papers/end_to_end/draft.md` |
| Q2 | Why? What separates RL's learning from ES's on this data? | **done, a supported but provisional explanation**: the members' ranking is mostly prompt-specific effects, one update gains what its random part costs, the gain per unit is constant in N while the cost falls as 1/N; the estimate is a working gradient estimate, the useful step is too short beside the random part. Not isolated: why the quadratic prediction fails over many updates | `08-why-rl-learns-and-es-does-not.md`, `09-best-achievable.md`, `review/` |
| Q3 | Can it be told a priori, without trial and error, when ES will be effective and when RL will dominate? | **running**: the rule's ES side calibrated on six runs and its RL side on three GRPO arms, both on two Countdown settings at 3B; the runs show initial learning on Countdown and none on 0.5B GSM8K, with the prescription's size off by two to four and no sustained retention: ES collapses off task late where GRPO does not, which no probe predicts; measured from the loops, ES reached +0.10 at 5 to 16 times GRPO's rollouts and 1.2 to 3.5 times its compute, twice within twice GRPO's compute with the other task intact before deteriorating; the verdict (S3e) waits for fresh settings | `10-when-does-es-work.md`, `11-literature-es-vs-rl.md`, `runs/probe/README.md` |

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
| S3c the measured RL reference | GRPO trained by us with the Tulu 3.1 recipe on 3B instruct and base Countdown (5e-7) and instruct at 1e-6, 40 steps each, scored on the probe's blocks, every phase timed | **done** (2026-10-14, $17) | the RL side's first-step model holds (target at 15 and 25 to 30 steps against 16 and 20 predicted); GRPO keeps the off-task block ES lost; measured from the loops to +0.10 on the reserved block, ES/GRPO rollouts 5.3 and 16 (instruct), 8 and 16 (base), loop seconds 1.16 and 3.50 (instruct), 1.40 and 2.67 (base); the two seed-1 ES crossings within twice GRPO's compute with GSM8K intact, then deteriorating; GRPO's KL term is a confound for the retention difference | `10-when-does-es-work.md` ("S3c"), `es_vllm/grpo_run.py` | `grpo-run/README.md`, `grpo-run/<arm>/` | $17 |
| S4a the collapse and the resolution | on the same 3B instruct Countdown checkpoint, holding population, sigma, prompts, horizon and scoring fixed: an ES run at a smaller fixed step (the direct step-size control the instruct-against-base contrast lacks), a random-only control (shuffled rankings, same update norm), a 256-member run, an ES run with shrinkage toward the start, to say whether the off-task collapse is the random walk or the learned direction; a second GRPO seed and a GRPO arm at beta 0 (the KL term is a confound for the retention difference); a level-1 off-task predictor from the members' task-versus-off-task fitness covariance, a local diagnostic whose power to predict delayed damage is unvalidated; a second member population and 600-prompt blocks on the six probed settings | **done** (2026-10-15: the six arms 16 h, about $85; the second probe, `probe --seed 1 --block 600` with `--phase members-offtask` on the Countdown settings, 5.6 h, about $30; pod deleted) | the collapse is the random walk: shuffled rankings at the prescribed step reproduce it with no direction (GSM8K to 0.000 by 25), 256 members delay it past 20, a fifth of the step keeps GSM8K through 30 (+0.185 on the reserved block, the first-order model exact at 30), and shrinkage 0.1 at the full step keeps GSM8K within −0.07 with the program's largest gain (+0.235 at 25) for 1.15 times GRPO's loop seconds; GRPO's second seed crosses within the window and GRPO without its KL term keeps GSM8K, so the anchor is not what separates GRPO's retention from ES's; the frozen −0.02 retained bound is below the block's standard error and fails two arms that held GSM8K within 0.07 (a loss of 3 to 4 paired standard errors, a trade-off rather than retention). Second probe: the level-1 response replicates across populations at a matched length on six of six settings but the odd response is concave in the step, so the rule's per-unit reading does not (item 10, an exploratory curve rule beside the preregistered one); the GSM8K settings have no odd response at 64 members on 600 prompts and only 3B GSM8K gets a call (fails); neither population's ranking carries GSM8K damage, the base model's first-probe destructive direction was one update's random part, and a random direction moves the base model's GSM8K twelve times more than its task and the instruct model's a third as much, a candidate collapse predictor for S4b | `10-when-does-es-work.md` ("S4a", "S4a results", "S4a, second half") | `s4a/`, `grpo-run/`, `probe2/` | $85 for the arms, $35 to 45 for the second probe |
| S4b fresh settings for the verdict | the prospective test of anchored ES and four predictions written before each run on three fresh settings: 1.5B instruct Countdown (A), 3B instruct on the constructed Quotation task (C), 3B instruct on MATH alone (D); every ES arm at the curve rule's step with shrinkage 0.1, an un-anchored control and a second seed on A, one GRPO arm per setting | **done** (probes 2026-10-16, 5.1 h, $27; eight arms 2026-10-16 and 21, 16.8 h, $93) | A: three ES arms learn to +0.15 to +0.18 with GSM8K kept within the margin on one seed, GRPO crosses where the sign slope said, ES at 1.1 to 1.2 times GRPO's loop seconds (the prescription's own cost, about 3,100 s, reported beside them); the corrected forecast from one update's responses matches the first ten updates on both seeds and overshoots by 0.02 at 30 (the frozen one had counted the first pull-back twice, item 11); C: at its ceiling (start 0.937, headroom 0.063), both gated predictions miss; D: no signal for either method; P2 abstains on all three, untested; A's two populations prescribe 3.0 and 15.8 units | `10-when-does-es-work.md` ("S4b, fresh settings for the verdict: design" and its results), `s4b-probe/README.md`, `s4b/README.md` | `s4b-probe/`, `s4b/`, `grpo-run/` | $120 |
| S5 the rule as a checklist on more cases | the paper's goal restated as the rule (Andres, 2026-10-21): a checklist frozen from what held (headroom and live share; a step with a resolved net response in the one-update scan; GRPO's pace from the sign slope; the class by the compute ratio; the anchor and an untouched off-task block), tested on seven fresh cases chosen for both outcomes: 0.5B and 1.5B GSM8K and Llama-3.2-1B Countdown expected GRPO-only, 3B instruct and base Countdown anchored at the curve rule's step, two generator tasks on 1.5B expected or open | **design for review** (2026-10-21, corrected 2026-10-22): data efficiency is the reward at equal samples seen (Andres); at equal samples the record favours GRPO everywhere measured (Tulu at 368,640; Countdown at ES's early horizon only, the GRPO arms stop at 30,720); the revised protocol runs both methods to a common sample budget on the three published Countdown checkpoints, GRPO at 480 steps first ($108 for two settings), 1 million samples next; the published ES recipe as a second ES arm | | `10-when-does-es-work.md` ("S5") | | about $160 |
| S4c the published conflict | Qiu et al.'s matched-evaluation comparison replicated on our instrumented 3B instruct Countdown: ES at their settings for 500 iterations against GRPO at matched evaluations (or a fraction with the curve) | **proposed**, only if S4b sustains the rule | | `11-literature-es-vs-rl.md`, `10-when-does-es-work.md` | | $200 to 600 |
| S3d calibration run | Tulu 8B on Countdown at Qiu et al.'s settings (30 members, σ 1e-3, 200 prompts, 500 iterations) | **deferred** (2026-10-13): Tulu DPO and Qiu's Llama Instruct are different starting checkpoints, so this is a transfer experiment, not a calibration anchor; its useful part (Tulu 8B on Countdown with our own GRPO arm) is in S4b | | `10-when-does-es-work.md` | | $60 to 80 |
| S3e the rule's verdict | over at least six settings with a run or a published outcome: predicted order right (Spearman ≥ 0.8), the three-way call right in five of six, level 0 predicts level 1 | **open**: the outcome classes and the unfinished-target rule are frozen in `10` before any further setting; the six probed settings, concentrated in two task families, are a pilot; the revised level-0 hypothesis (the update's odd response as the target, Spearman 0.83 on the same six records, exploratory) is tested only on fresh settings | | `10-when-does-es-work.md` ("What counts as the rule being useful", "Corrections of 2026-10-13") | | |
| P paper | the manuscript | **draft complete in markdown** (2026-10-21): the Tulu comparison is the main result and Section 9 adds the Countdown comparisons (learning, the collapse's mechanism, the stabilization trade-off, measured compute); the predictor is reported as a pilot, not a claim; open: the title and abstract wording (candidates offered), the LaTeX build and the number macros | | `papers/end_to_end/draft.md`, `papers/end_to_end/README.md` | | |
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
| E21 training runs at the prescription | initial learning on Countdown at every seed, none on 0.5B GSM8K; every Countdown run collapsed off task late | `probe-run/` |
| E22 GRPO trained by us | RL's first-step model holds; GRPO keeps the off-task block; to +0.10, ES at 1.2 to 3.5 times GRPO's loop seconds and 5 to 16 times its rollouts | `grpo-run/` |
| E23 the collapse's controls | the random walk: shuffled rankings collapse GSM8K with no direction, 256 members delay it, a fifth of the step or shrinkage 0.1 keeps it; GRPO's second seed and its beta-0 arm keep GSM8K | `s4a/`, `grpo-run/` |
| E24 the second probe | the level-1 response replicates at a matched length and is concave in the step; GSM8K settings have no odd response at 64 members; no ranking carries GSM8K damage; the per-direction off-task spread separates the fragile checkpoint from the robust one | `probe2/` |
| E25 S4b | on a fresh setting with a resolved response three anchored ES arms learn at a 3.0-unit step and reach the target at 1.1 to 1.2 times GRPO's loop seconds, and the corrected one-update forecast matches the first ten updates on two seeds and overshoots by 0.02 at 30; at a task's ceiling and without a response the predictions miss or abstain | `s4b-probe/`, `s4b/`, `grpo-run/` |

## Open items

Decisions for Andres:

- The budget was raised on 2026-10-14 (figure pending); S4a and its second probe spent
  about $115 of it. Next: S4b (~$150) on S4a's outcome and the second probe's, S4c ($200
  to 600) decided on S4b's results. The curve rule's step is frozen as S4b's
  prescription; its call is dropped.
- Whether the paper becomes the Tulu comparison alone (the draft) or the predictive
  account with the Tulu result as its anchor.
- Two seeds per run only for the cheap settings; the 8B settings run one seed.

Work in hand or queued (mine):

- Nothing running. The paper's question (Andres, 2026-10-21 and 22): when can ES be
  expected to have a higher data efficiency than GRPO, data efficiency being the reward
  reached at equal samples seen, with a rule that says so before the compute is spent
  and is tested on more than two cases with both outcomes. The design is `10` "S5" with
  its correction of 2026-10-22: equal-samples curves to a common budget on the published
  Countdown checkpoints (GRPO to 480 steps first, $108 for two settings), the GSM8K
  cases and Tulu for the other outcome; decisions for Andres listed there; nothing runs
  before them. The manuscript around the completed
  comparisons is done (`papers/end_to_end/draft.md` Section 9); the second GRPO seed on
  A and the A pair to 60 updates fold into S5's end; S4c stays deferred.
- A step schedule or re-measurement as a separately specified method, tuned on validation
  prompts and tested on fresh ones (`10`, corrections item 7), after S4a says what the
  collapse is.
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
$441 spent by uptime through S3a. S3b and S3c ran on the RunPod balance of $126
reported on 2026-10-13 and cost about $71 and $17 by uptime, leaving about $38. Every stage's cost above is by pod uptime, which runs
about 20% over the drivers' measured time.
