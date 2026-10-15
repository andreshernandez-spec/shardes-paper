# S4a: the collapse and the resolution (docs/end_to_end/10, "S4a")

Six arms on 2026-10-14 and 2026-10-15, all on Qwen2.5-3B-Instruct on Countdown with the
probe's blocks (`runs/probe/qwen3b-countdown`), each changing one thing against the two
finished runs at the prescribed step 15.3 (`runs/probe-run/qwen3b-countdown-s0`, `-s1`),
to say what the off-task collapse every Countdown ES run showed is made of. The four ES
arms (`es_vllm/probe_run.py`, 64 mirrored rank-1 members at sigma 5e-4 on the 192 train
prompts unless stated, scored every 5 updates) are here; the two GRPO arms
(`es_vllm/grpo_run.py`, the Tulu 3.1 recipe at 5e-7, 40 steps) are in `runs/grpo-run/`
beside the S3c arms. One Secure H200 (RunPod pod `4466roagwld1kw`, US-NC-1, $5.29/h),
12:41 UTC on the 14th to 04:42 UTC on the 15th for the six arms, 16.0 hours, about $85
by uptime (the pod went on to the second probe, `runs/probe2/`, costed there). The pod
ran commit ccc134c; the analysis below is from a1c1ce6 (the reference-arm pairing by
config, the curves printer, and the random-only and shrinkage predictions).

| arm | record | what changed | what it isolates |
|---|---|---|---|
| step 3.0 | `qwen3b-countdown-step3-s0` | the step, a fifth of 15.3 | the step itself |
| random only | `qwen3b-countdown-random-s0` | the real fitnesses in a seeded random order (`--ranking random`): an update of the real norm, no direction | the random walk alone |
| 256 members | `qwen3b-countdown-n256-s0` | four times the members at step 15.3, 20 updates | the random part's size |
| shrinkage 0.1 | `qwen3b-countdown-shrink0.1-s0` | after each update the weights move a tenth of the way back toward the start (`--shrink 0.1`) | an anchor |
| GRPO seed 1 | `../grpo-run/qwen3b-countdown-lr5e-7-s1` | a second population of samples | the reference's seed variance |
| GRPO beta 0 | `../grpo-run/qwen3b-countdown-lr5e-7-beta0-s0` | the kl3 term removed | whether the anchor keeps GRPO's off-task block |

Files per ES arm as in `runs/probe-run/`: `run.json` (configuration with the variant,
the prediction block, block digests, environment), `log.jsonl` (per update), `eval.jsonl`
(per evaluation, every prompt's reward on the validation, reserved test and off-task
blocks). Rewards are in the task's units; the analysis normalizes by each row's kind
(Countdown 1.1, the Tulu verifiers 10).

## The verdict

The preregistration's gate: the collapse is the random walk if the random-only arm
collapses and the 256-member arm is delayed; the learned direction if the random-only arm
holds and the 256-member arm collapses on schedule. The random-only arm took GSM8K to
0.000 by update 25 with no direction at all, and the 256-member arm ended 20 updates at
−0.037 where the 64-member runs were at −0.60 and −0.84. **The collapse is the random
walk**: the accumulated damage of the update's random part, which grows with the step
squared and shrinks with the members, and which the step-3.0 arm (GSM8K within −0.04
through 30 updates) and the anchored arm (within −0.07 at the full step) also avoided.
GRPO without its KL term kept GSM8K too (−0.020 at 40), so the anchor is not what
separates GRPO's retention from un-anchored ES's.

## The table

`python -m es_vllm.probe_run --table runs/s4a`. The class definitions are those of
`runs/probe-run/README.md` (frozen in `10`, item 6 as amended 2026-10-14): the checkpoint
is selected on the validation block among evaluations whose off-task block is within
−0.02 of its start, and the class is the test block's change there.

| run | preregistered call | step | variant | updates to target, predicted | updates | validation at end | test at end | off-task at end | peak test gain | at | validation-selected checkpoint | test there | class | class by test peak | class, task alone | collapsed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3b-countdown-n256-s0 | works | 15.3 | N 256 | 2.89 | 20 | +0.171 | +0.193 | -0.037 | 0.219 | 15 | - | - | fails | fails | works | yes |
| qwen3b-countdown-random-s0 | works | 15.3 | random ranking | 2.89 | 30 | -0.119 | -0.124 | -0.843 | 0 | - | - | - | fails | fails | fails | yes |
| qwen3b-countdown-shrink0.1-s0 | works | 15.3 | shrink 0.1 | 2.89 | 30 | +0.193 | +0.154 | -0.030 | 0.235 | 25 | - | - | fails | fails | works | yes |
| qwen3b-countdown-step3-s0 | works | 3 | - | 2.89 | 30 | +0.194 | +0.185 | -0.017 | 0.185 | 30 | 30 | 0.185 | works | works | works | yes |

The frozen class fails the 256-member and shrinkage arms although their reserved block
ended at +0.19 and +0.15 with GSM8K within −0.07 at every evaluation: no evaluation had
GSM8K within the −0.02 bound, which is below the block's standard error (0.021 at a
success rate of 0.84 on 300 prompts), so the bound distinguishes nothing between −0.02
and −0.05. The classes stand as frozen; the bound is the lesson (see `10`, S4a results).
The "collapsed" flag on the step-3.0 arm is one −0.040 reading at update 15 that
recovered; the flags on the other three arms are drops of 0.05 to 0.84.

The GRPO arms, `python -m es_vllm.grpo_run --table runs/grpo-run` (the S3c arms
included):

| arm | lr | beta | seed | steps | rollouts | validation at end | test at end | off-task at end | peak test gain | at | test +0.10 first at | validation-selected checkpoint | test there | class | collapsed | s per step | GPU-s per answer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3b-base-countdown-lr5e-7-s0 | 5e-07 | 0.01 | 0 | 40 | 30,720 | +0.121 | +0.135 | +0.143 | 0.156 | 35 | 30 | 40 | 0.135 | works | no | 87.3 | 0.114 |
| qwen3b-countdown-lr1e-6-s0 | 1e-06 | 0.01 | 0 | 40 | 30,720 | +0.159 | +0.213 | -0.003 | 0.213 | 40 | 15 | 20 | 0.122 | works | yes | 85.6 | 0.111 |
| qwen3b-countdown-lr5e-7-beta0-s0 | 5e-07 | 0 | 0 | 40 | 30,720 | +0.158 | +0.191 | -0.020 | 0.191 | 40 | 30 | 40 | 0.191 | works | no | 81.7 | 0.106 |
| qwen3b-countdown-lr5e-7-s0 | 5e-07 | 0.01 | 0 | 40 | 30,720 | +0.138 | +0.135 | -0.007 | 0.148 | 30 | 15 | 35 | 0.114 | works | no | 89.4 | 0.116 |
| qwen3b-countdown-lr5e-7-s1 | 5e-07 | 0.01 | 1 | 40 | 30,720 | +0.157 | +0.125 | -0.007 | 0.125 | 30 | 20 | 40 | 0.125 | works | no | 84.9 | 0.111 |

## The curves

`python -m es_vllm.probe_run --curves runs/s4a` and `python -m es_vllm.grpo_run --curves
runs/grpo-run`. Changes by block against the evaluation at update 0, with the first-order
prediction from the probe's `G` and `C` where it applies: the plain model for the
step-3.0 and 256-member arms, the random cost alone for the random-only arm, and the
geometrically weighted update count for the shrinkage arm (`probe_run.effective_updates`:
update i enters the displacement with weight 0.9 to the power g − i + 1, so the gain
counts 8.6 updates at 30 and the random part's variance 4.3).

**qwen3b-countdown-step3-s0** (step 3, 30 updates, 368,640 rollouts, 2.6 h)

| after | validation | test | off-task | answer length | predicted validation |
|---|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 329 | +0.000 |
| 5 | +0.051 | +0.065 | -0.023 | 299 | +0.033 |
| 10 | +0.113 | +0.100 | -0.007 | 300 | +0.065 |
| 15 | +0.126 | +0.125 | -0.040 | 317 | +0.098 |
| 20 | +0.143 | +0.179 | -0.037 | 317 | +0.131 |
| 25 | +0.170 | +0.177 | +0.003 | 323 | +0.163 |
| 30 | +0.194 | +0.185 | -0.017 | 321 | +0.196 |

**qwen3b-countdown-random-s0** (step 15.3, random ranking, 30 updates, 368,640 rollouts, 2.6 h)

| after | validation | test | off-task | answer length | predicted validation |
|---|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 329 | +0.000 |
| 5 | -0.040 | -0.034 | -0.027 | 331 | -0.035 |
| 10 | -0.072 | -0.069 | -0.083 | 364 | -0.069 |
| 15 | -0.092 | -0.100 | -0.120 | 374 | -0.104 |
| 20 | -0.084 | -0.089 | -0.470 | 502 | -0.139 |
| 25 | -0.116 | -0.120 | -0.843 | 1018 | -0.174 |
| 30 | -0.119 | -0.124 | -0.843 | 931 | -0.208 |

**qwen3b-countdown-n256-s0** (step 15.3, N 256, 20 updates, 983,040 rollouts, 6.0 h)

| after | validation | test | off-task | answer length | predicted validation |
|---|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 329 | +0.000 |
| 5 | +0.148 | +0.170 | -0.030 | 271 | +0.139 |
| 10 | +0.181 | +0.178 | -0.053 | 237 | +0.277 |
| 15 | +0.165 | +0.219 | -0.030 | 257 | +0.416 |
| 20 | +0.171 | +0.193 | -0.037 | 255 | +0.554 |

**qwen3b-countdown-shrink0.1-s0** (step 15.3, shrink 0.1, 30 updates, 368,640 rollouts, 2.5 h)

| after | validation | test | off-task | answer length | predicted validation |
|---|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 329 | +0.000 |
| 5 | +0.082 | +0.102 | -0.040 | 282 | +0.108 |
| 10 | +0.086 | +0.097 | -0.033 | 267 | +0.177 |
| 15 | +0.123 | +0.151 | -0.063 | 296 | +0.219 |
| 20 | +0.150 | +0.142 | -0.057 | 312 | +0.245 |
| 25 | +0.205 | +0.235 | -0.067 | 294 | +0.260 |
| 30 | +0.193 | +0.154 | -0.030 | 287 | +0.269 |

**qwen3b-countdown-lr5e-7-s1** (lr 5e-07, beta 0.01, seed 1, 40 steps, 30,720 rollouts, 0.9 h in the loop)

| after | validation | test | off-task | answer length |
|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 329 |
| 5 | +0.014 | +0.007 | -0.017 | 303 |
| 10 | +0.069 | +0.070 | -0.013 | 256 |
| 15 | +0.075 | +0.072 | -0.013 | 229 |
| 20 | +0.080 | +0.110 | +0.000 | 216 |
| 25 | +0.122 | +0.109 | -0.010 | 210 |
| 30 | +0.112 | +0.125 | -0.020 | 213 |
| 35 | +0.153 | +0.115 | -0.020 | 222 |
| 40 | +0.157 | +0.125 | -0.007 | 226 |

**qwen3b-countdown-lr5e-7-beta0-s0** (lr 5e-07, beta 0, seed 0, 40 steps, 30,720 rollouts, 0.9 h in the loop)

| after | validation | test | off-task | answer length |
|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 329 |
| 5 | +0.016 | +0.023 | -0.013 | 315 |
| 10 | +0.054 | +0.067 | -0.013 | 261 |
| 15 | +0.069 | +0.054 | -0.017 | 232 |
| 20 | +0.101 | +0.088 | -0.017 | 209 |
| 25 | +0.081 | +0.082 | -0.013 | 199 |
| 30 | +0.132 | +0.104 | -0.017 | 188 |
| 35 | +0.131 | +0.143 | -0.013 | 189 |
| 40 | +0.158 | +0.191 | -0.020 | 202 |

## Compute to the target

`python -m es_vllm.probe_run --compare runs/s4a runs/grpo-run`: loop seconds and rollouts
through the first evaluation at +0.10 on the reserved test block, against the reference
arm (5e-7, beta 0.01, seed 0, chosen by its configuration). A crossing at the first
evaluation (update 5) is an upper bound on the compute: the block is not scored between
updates.

| ES run | +0.10 at update | ES loop s | ES rollouts | off-task there | retained | GRPO +0.10 at step | GRPO loop s | GRPO rollouts | GRPO off-task there | seconds, ES/GRPO | rollouts, ES/GRPO |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3b-countdown-n256-s0 | 5 | 5,890 | 245,760 | -0.030 | no | 15 | 1,376 | 11,520 | -0.017 | 4.28 | 21.33 |
| qwen3b-countdown-random-s0 | - | - | - | - | - | 15 | 1,376 | 11,520 | -0.017 | - | - |
| qwen3b-countdown-shrink0.1-s0 | 5 | 1,581 | 61,440 | -0.040 | no | 15 | 1,376 | 11,520 | -0.017 | 1.15 | 5.33 |
| qwen3b-countdown-step3-s0 | 15 | 4,719 | 184,320 | -0.040 | no | 15 | 1,376 | 11,520 | -0.017 | 3.43 | 16.00 |

"retained" is the −0.02 bound again; every crossing here had GSM8K between −0.03 and
−0.04, within two standard errors of its start. The GRPO seed-1 arm crossed at step 20
(1,700 loop seconds, 15,360 rollouts), so against that seed the ratios are 0.93 to 3.5
in seconds.

## What the arms say

Against the preregistration's predictions:

1. **Step 3.0.** The first-order model's validation gain at 30 is +0.196 predicted,
   +0.194 observed, after running ahead of it through the middle (+0.113 against +0.065
   at 10). The reserved block crossed +0.10 at update 10, before the predicted 15 to 30.
   GSM8K stayed within −0.04 at every evaluation and ended at −0.017: the step-following
   branch. On this checkpoint the collapse was the step, not the task gain, and a fifth
   of the prescribed step gives a run that learns the task to +0.185 on the reserved
   block (the two step-15.3 seeds peaked at +0.135 and +0.164 before collapsing) and
   keeps GSM8K.
2. **Random only.** The task blocks fell at the predicted random cost through 15
   (−0.092 and −0.100 against −0.104) and then stopped at the floor; GSM8K fell to
   −0.083 at 10, −0.120 at 15, −0.470 at 20 and 0.000 from 25, with the answers running
   to the cap (1,018 tokens at 25), the real runs' collapse without any ranking signal.
   The random-walk branch; the direction branch (GSM8K within about −0.08 at 30) is
   excluded by a wide margin. The one-update even response under-predicts the off-task
   damage (−0.028 at 10 from the probe's curvature against −0.083): the random walk's
   cost on GSM8K grows as the model moves.
3. **256 members.** The target by update 5 as predicted (+0.170 on the reserved block);
   GSM8K −0.030, −0.053, −0.030, −0.037 at 5, 10, 15, 20, no collapse through 20 where
   both 64-member seeds had lost the block: delayed, the random-walk branch. The task
   gain saturated at about +0.18 to +0.22 from update 5 on, against the first-order
   model's +0.55 at 20; the 64-member seeds saturated near +0.14 to +0.16 at the same
   step before collapsing. The random cost at 256 members, a quarter of the 64-member
   one, keeps GSM8K; it does not raise the ceiling.
4. **Shrinkage 0.1.** GSM8K held between −0.03 and −0.07 for all 30 updates (the same
   step without the anchor: 0.000 by 25), the accumulated-drift branch. The gain did not
   saturate near +0.15 as written: +0.235 on the reserved block at 25, the largest of the
   program (GRPO at the doubled learning rate: +0.213 at 40), then +0.154 at 30 after a
   dip in the members' fitness at update 28. The anchored first-order model gives +0.27
   at 30; the preregistered +0.15 was a rougher figure.
5. **GRPO seed 1.** The reserved block crossed +0.10 at step 20 (seed 0: 15; the window
   allowed 32), ended at +0.125 with GSM8K −0.007. **GRPO beta 0.** Crossed at 30, ended
   at +0.191 (the best at this learning rate) with GSM8K −0.020: the KL term is not what
   keeps GRPO's off-task block, and GRPO's retention against un-anchored ES is the
   estimator's, not the anchor's.

What the collapse is, then: the random part of the ES update, accumulated over updates,
with a cost on the off-task block that the one-update probe under-states and that
reaches a cliff (GSM8K from −0.12 to −0.47 in five updates, with the answers lengthening
to the cap). Three ways out worked on this checkpoint, each at a price in the rule's
terms: a smaller step (3.0: 3.4 times GRPO's loop seconds to the target instead of 1.2
to 3.5), more members (256: 4.3 times, and 21 times the rollouts), or an anchor at the
full step (1.15 times GRPO's seconds, 5.3 times its rollouts, the cheapest, with the
largest gain). The rule's retained-block bound was derived from the random part's even
response per update and is not a retention bound (`10`, corrections item 2); these arms
say the random part's damage is the collapse, and that it needs an anchor, a smaller
step or more members to be kept below the cliff. What a probe can predict about it is
the second probe's question (`runs/probe2/`).

## Limitations

One seed per arm. The frozen classes and the "retained" column use a −0.02 bound that
is below the block's standard error. The 256-member arm ran 20 updates, not 30, and its
crossing at the first evaluation bounds its compute from above. The shrinkage arm's late
dip (update 28) is one reading. The GRPO arms use the Tulu run's recipe untuned for this
model and 40 steps. Everything here is on one checkpoint and one task; the base model's
collapse (`runs/probe-run/`, GSM8K rising to +0.14 then falling to −0.34) has not had
these controls.
