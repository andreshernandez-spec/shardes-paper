# 13. A shrinking step from update 200, on Qiu et al.'s setup: preregistration

Written and committed 2026-11-03, before any run. Decided by Andres the same day: the
first follow-up to S6 (`12-qiu-matched.md`) is the step-size schedule, before GRPO on the
matched setup.

## Why

On Qiu et al.'s setup our ES ends inside their band (56.5% at 500 updates) but gains
0.24 points per 10 updates from update 200 on, where their plotted run gains 0.64
(`runs/qiu-matched/README.md`). The update's size is fixed by the z-scored fitness
(alpha / N times a sum of 30 unit-variance terms), so the step stays the same while the
signal in the ranking falls. The rule of `10-when-does-es-work.md` (progress per update
about alpha rho D minus c alpha^2 d / N) has a gain linear in alpha and a cost of the
step's random part quadratic in alpha, growing with the number of directions d. A step
that shrinks once the signal has fallen keeps more of the gain than of the cost.

## What runs

`es_vllm/qiu_match.py --arm decay` through `es_vllm/qiu_match_pod.sh <commit> 1 decay`, one
Secure H200:

- **Branch.** Arm B's first 200 updates are replayed from its log
  (`runs/qiu-matched/qwen3b-s1/log.jsonl`), each against its logged digest; the weights
  written after update 200 have to be arm B's checkpoint 200 bit for bit, or the run
  stops.
- **Step.** Update g (counted from 0) uses alpha times 200/g from g = 200 on: 1 at the
  branch, 0.57 at the last update (349). Over the 150 updates the steps sum to 0.75 of
  arm B's and their squares to 0.57.
- **Same noise.** The seed is arm B's, so every update draws the noise of arm B's update
  of the same number. The two runs differ in the step and in what follows from it.
- **Everything else is S6's:** 30 members, sigma 1e-3, alpha 5e-4 before the branch, the
  200 training problems, 1,024 tokens, the matched decoding, his reward, the bit-for-bit
  engine checks at every update.
- **Tests.** The running test (the 2,000 test problems decoded by the matched rule in
  vLLM) after 200, 210, ..., 350 updates; his `eval_countdown.py`, unchanged, on
  checkpoints 300 and 350.

## Reading, fixed before the data

Primary statistic: D, the mean running test accuracy over updates 310, 320, ..., 350 in
this arm minus the same mean in arm B (53.32%).

- **D ≥ +2.0 points**: shrinking the step from update 200 helps.
- **D ≤ −2.0 points**: it hurts.
- **Between**: no difference found over 150 updates.

The threshold: arm B's running test scatters by 0.50 points (standard deviation around
its own straight line from update 200 to 500), and a mean of five evaluations scatters
less. Arm B's 0.24 points per 10 updates with their 0.64 in its place would put the mean
about 5 points higher. The chance difference between two runs is not measured (one
branch, no control seed); the shared noise draws make it smaller than between
independent seeds.

Secondary, without a gate: his evaluator on checkpoint 300 beside arm B's 53.70%, and on
checkpoint 350; the slope from 200 to 350; training accuracy and the number of training
problems that split the members, as in S6's table.

## Prediction

D between −2 and +2, most likely near +0.2. A toy version of that rule, with the
gain G and the cost C per 10 updates at the full step, fits both runs. Arm B gives
G − C = 0.24. Qiu's run, whose noise has 8.5 times fewer independent coordinates and so,
in the rule, an 8.5 times smaller cost, gives G − C/8.5 = 0.64. That makes G = 0.69
and C = 0.45. At a step s times the full one, the gain is s G − s² C, which is nearly
flat around s = 1: this schedule then adds 0.2 points to the mean. A D of 2 or more
would mean the cost is much larger than this toy version makes it, and that the shared
noise is not the whole of Qiu's later gain.

## Cost

Replaying 200 updates takes minutes. 150 updates at S6's 190 s each, with setup and
his evaluator, is about 9 hours: about $48 on a Secure H200 at $5.29/h. If the first 10
updates project above $65, the run stops. Health checks every 30 minutes, as in S6.

## Result, 2026-11-03

Details and the full curve: `runs/qiu-matched/README.md`, section S6b; numbers from
`python -m es_vllm.qiu_match_table runs/qiu-matched`.

**D = +0.96 points** (the arm 54.28%, arm B 53.32%): no difference found over 150
updates, as predicted. His evaluator gives the arm's checkpoint 300 53.80% (arm B 53.70%)
and checkpoint 350 54.40%. The lead of about a point opened in the first 10 updates,
while the step was still 0.95 to 1 of arm B's; from update 200 to 350 both arms rise 0.21
to 0.22 points per 10 updates. A step falling to 0.57 of arm B's leaves the late slope
unchanged, so the fixed step is not what holds it at a third of Qiu et al.'s. The replay
took 55 minutes, not minutes, and a first launch stopped at a replay mismatch at update 2
that the rerun did not repeat. Cost about $50.
