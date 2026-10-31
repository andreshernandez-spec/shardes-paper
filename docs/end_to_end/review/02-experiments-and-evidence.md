# The experiments and the evidence each collected

One entry per experiment, in the order they ran. Each names its record under
`experiments/end_to_end/runs/` (the README there has the command, the code version and
the pod), what question it answered, the key numbers, what it established, and the
caveats that travel with it. Rewards are on the 0 to 10 scale; "held-out" is RL steps 121
to 160 (1,920 prompts) or those plus steps 481 to 520 (3,840) as stated; errors are
standard errors over prompts, paired.

## E1. Phase 0 and the throughput probe (2026-09-26)

Records: `pins/`, `data/tulu31/`, `runs/probe*`. Question: can the released runs be pinned
and their data rebuilt, and what does a matched ES run cost? Result: every checkpoint
pinned by commit and weight hash; Tulu 3.1's 29,689-row training set and per-step prompt
stream rebuilt from open-instruct 3f37c29 (the script's shuffle discards its result, so
the iterator orders the prompts); verifiers vendored with their own environment (the MATH
verifier returns False silently without the LaTeX parser). A matched run was estimated at
$122 to 159. Established: the data matching. Caveat: the matching assumes the model
card's command.

## E2. The pilot (2026-09-27)

Records: `runs/pilot1-invalid/` (first run, invalid), the four `runs/tulu-pilot-*/`,
`runs/pilot-gate.json`. Design: 30 iterations at sigma 5e-4, 1e-3 and 2e-3 and a
random-fitness control, N = 16, 192 prompts per member, the data of RL steps 1 to 120.
Result: no arm passes the preregistered gate (slopes of held-out reward minus the
control's +0.0084, +0.0040, +0.0057 per iteration at 1.7, 0.9 and 0.9 SE). The first run
was invalid (a JAX-to-torch handoff race wrote wrong weights at some restores; kept as
evidence). Established: no detectable learning in 30 iterations at these settings.
Caveat: low power; a slope had to exceed about 0.01 per iteration to pass.

## E3. Held-out context (2026-09-27)

Record: `runs/heldout-121-160/`. Question: where is RL on prompts none of the models
trained on? Result: DPO start 5.43; RL steps 40, 80, 120: 5.87, 6.40, 6.69 (+1.26 at step
120, mostly IFEval 4.13 to 6.32, then MATH +0.47, GSM8K +0.23); the sigma-5e-4 ES weights
after 30 iterations 5.42; the control 5.39. Established: RL moves detectably by step 40,
a third of the pilot's data. Caveat: the 1e-3 and 2e-3 rebuilds were unscorable (E4).

## E4. The H200 update nondeterminism (2026-10-06)

Record: `runs/update-check/`. Question: why do rebuilds from the fitness logs not
reproduce the logged weights? Result: the jitted `lax.scan` summing the members' noise
returned wrong bits in about 0.4% of calls on the H200, in JAX alone; one jitted step per
member gives none in 6,000 and keeps the library's bits (fix 154f63b). Established: the
backend's updates are reproducible; the pilot's arms had about 0.4% corrupted leaf
updates. Caveat: the RTX 3080 and A100 never showed it; the cause in XLA was not found.

## E5. The Countdown check against es-at-scale (2026-10-06)

Records: `runs/countdown-s{1,2}/`, `runs/countdown-s{1,2}-ref/`, `runs/countdown-gate.json`.
Design: es-at-scale's own trainer and our backend on its Countdown example at 0.5B,
identical settings, two seeds each, 100 updates. Result: mean evaluation reward over
updates 5 to 100 is 0.182 for ours and 0.159 for the reference (bound 0.020): the
two-sided gate fails on the high side; both reach about 0.23 by update 100. Established:
the backend learns where ES is known to learn, faster than the reference (candidate
cause: the reference adds updates into bf16 weights). Decided by Andres as "the backend
works".

## E6. The longer run (2026-10-07)

Records: `runs/tulu-long-s5e-4/`, `runs/tulu-long-random/`, `runs/heldout-481-520/`,
`runs/long-gate.json`, `runs/tulu-long-README.md`. Design: the sigma-5e-4 configuration to
120 iterations (the data of RL steps 1 to 480) with a random control, held-out on RL
steps 481 to 520 after 0, 30, 60, 90, 120 updates; RL branches 120 to 480 scored the same
way. Result: the gate passes (arm minus control +0.260 +- 0.091 after 120 updates, 2.9
SE) but the arm ends 0.250 +- 0.087 below its start (5.255 against 5.505); the control
4.995; RL step 480 7.370. The first 30 iterations reproduce the pilot's arm bit for bit.
Established: ES does not learn the stream at the reference configuration; the "pass" is
the control degrading. Caveat: one seed.

## E7. The gradient probe (2026-10-07)

Records: `runs/grad-check/start.json`, `start-sampled.json`. Design: the arm's 16
iteration-0 members on 768 prompts with per-prompt rewards, at sigma 5e-4, 1e-3, 2e-3 and
mirrored; then 8 samples per prompt at temperature 1.0 with and without common random
numbers. Result: at 5e-4 a member changes 9.8% of outcomes (4.6% up, 5.2% down);
reliability of the fitness over 192 prompts 0.12 (90% interval 0.01 to 0.47); implied
first-order gain 0.021 per iteration against 0.007 realized by the arm; sigma 2e-3 has
reliability 0.26 but members lose 1.45 points; sampled fitness 0.11 against greedy 0.13
at 192 rollouts, common random numbers tripling the reliability against independent
seeds; mirrored pairs no better per rollout. Established: about nine tenths of the
ranking is noise, and neither decoding nor the budget split changes it.

## E8. Where the noise comes from (2026-10-07)

Records: `runs/why/analysis.json`, `runs/direction-check/`. Result: a member's greedy
outcome is a redraw at the start's success probability (q = 0.75, correlation 0.90 between
a prompt's flip rate and p(1 - p)); answers depart from the start's within the first
dozen tokens on 86 to 90% of prompts at near or exact bf16 ties; a perturbation moves an
uncertain prompt's p by about 8 points with cross-prompt correlation 0.0095 (0.004 over all
prompts); the random walk's curvature cost, -0.0038 per iteration from the members' cost
per unit length squared, matches the control's drift. Established: the noise is the
decoding's redraws plus prompt-specific effects, and the random walk's cost.

## E9. RL's geometry (2026-10-07)

Records: `runs/rl-geometry/` (`geometry.json`, `quantization.json`, `effective-norm.json`),
`runs/direction-check/`. Result: the step-120 checkpoint is 0.476 from the start (240:
0.751; 480: 1.134), ES 61.4 after the same data, one member 44.8; cosine 0.51 between RL's
moves to 120 and to 240; along RL's direction +0.85 at effective length 0.32 and +1.02 at
0.48, random directions at 0.28 to 0.73 within 1.3 SE of zero; flips along RL's direction
four to eight times as often up as down. The stored differences are mostly one-step bf16
rounding on small weights, so sparsity, concentration and rank claims were withdrawn;
effective lengths after rounding: random 0.238 to 0.107, 0.476 to 0.282, 0.951 to 0.734;
RL lambda 0.25 to 0.073. Established: RL's move is tiny and coherent; random directions of
its length do nothing. Caveat: the idle pod of this day cost about $46.

## E10. The contrastive fitness (2026-10-08)

Record: `runs/contrastive-check/`. Design (Andres's proposal): 16 answers per prompt
sampled by the start at temperature 1.0 on 384 prompts; 186 prompts with both right and
wrong answers; a member's fitness is mean log p(right) minus mean log p(wrong),
teacher-forced; the 16 members, random directions and RL's direction scored; the start
rescored after the restore. Result: the rescore matches to the bit; reliability 0.22
(halves) / 0.36 (all 186) per answer, 0.08 / 0.14 per token (wrong answers are 2.5 times
longer, so the summed version carries a length artifact); cross-prompt correlation
0.0024; a member moves the start's answers 6.1 nats of KL per answer against RL's 7.8 at
90 times the weight distance; RL's direction raises the contrast by +4.1 (81% of prompts
up), random directions by nothing; the held-out first-order ES gain at most 0.067 nats per
iteration against RL's 0.137. Established: removing the decoding noise does not make the
ranking reliable. Caveat: the first attempt deadlocked on the verifier pipe (fixed).

## E11. Countdown cross-prompt probe (2026-10-08)

Records: `runs/grad-check/countdown-qwen0.5b.json`, `countdown-tulu8b.json`; prediction
in the script, committed first. Design: the same measurement as E7 and E8 on Countdown's
200 prompts for T2's 0.5B members and for the Tulu 8B start with its own 16 members at
sigma 5e-4 and 1e-3. Result: Tulu 8B on Countdown: cross-prompt correlation 0.30 (0.49 on
uncertain prompts), reliability 0.42 / 0.68, outcomes changed 66 / 75%; 0.5B: 1.06 and
0.94. The prediction held. The implied gain overstates the realized gain on Countdown too
(T2 realized a third, the Tulu arm a tenth); what differs is the margin over the random
walk's cost (about 600 on T2, about 6 on Tulu). Established: the task, not the model or
the method, sets the shared signal.

## E12. The literature (2026-10-08 and 09, two searches)

No records; summaries in the conversation and in `08`. Verified at the source: Abdi et al.
2026 and Hoy et al. COLM 2026 (ES updates orders of magnitude larger than GRPO's, broader
drift, forgetting), the Tulu 3 paper and the Tulu 3.1 model card (hardware layout, no wall
clock). Reported by the agent and not independently checked: Ba et al. 2026, ESSA, the
Palleti and Singh replication, GRESO, SPEC-RL, LOREN, the K-FAC and multilevel Monte Carlo
references. Nobody has run ES on a mixed RLVR stream.

## E13. The step probe (2026-10-08)

Record: `runs/step-check/`; prediction in the script. Design: the iteration-0 update from
four rankings of the same 16 members, applied as `theta_0 +- lambda u` at lambda 4 and 10
on 3,840 held-out prompts (4,608 with the probe's 768); `theta_0 + 1 u` checked equal to
the arm's first update to the bit. Result: gain per unit +0.004 +- 0.003 (at most 0.010)
for the 768-prompt ranking, +0.005 for the contrastive, -0.010 for the arm's logged and
the control's; cost -0.0042 +- 0.0005 per unit squared for every ranking (predicted
-0.0038); random weightings give +-0.011; mirrored members put two thirds of the ranking
signal in the even (sign-independent) part. Established: one update gains what its random
part costs; the shortfall is in each update, not in how 120 add up.

## E14. N scaling with low-rank members (2026-10-09)

Record: `runs/lowrank-check/` (`members.json`, `steps.json`, `predictions.json`); backend
`es_vllm/lowrank.py`. Design: 1,024 rank-1 mirrored members as in-memory LoRA adapters
decode the 192 iteration-0 prompts; the update from the first 128, 512, 1,024 measured as
E13 at lengths 44.7 and 111.7. Result: gain per unit +0.0037 +- 0.0011, +0.0047 +- 0.0006,
+0.0047 +- 0.0004; cost per unit squared -0.00054, -0.00015, -0.000077 (1.2 to 1.3 times
a dense update's at the same length); fitted best net 0.006, 0.036 +- 0.010, 0.071 +-
0.016 against RL's 0.040 mean per four steps, the one step scored near the 1,024 optimum
+0.057 +- 0.053; the populations nested; adapters agree with merged weights at
correlation 0.997 and 0.995 on 32 answers;
members decode at about 21 answers per second. The numeric cost predictions written before
scoring had the wrong sign (the members' mean change was measured against one greedy
redraw of the start). Established: the gain does not depend on N and the cost falls as
1/N.

## E15. Eight updates at 512 members (2026-10-09 to 10)

Record: `runs/lowrank-run/`; prediction in the script. Design: eight updates at 512
members and the best one-update step (15.4 times the unit update, length about 30), along
the stream; held-out after 0, 4, 8; a resume checked by digest. Result: +0.018 +- 0.055
after 4, +0.141 +- 0.061 after 8 (predicted +0.29; no-accumulation bound +0.10); IFEval
+0.47 (0.07), MATH -0.46 (0.15), GSM8K +0.10 (0.12); RL at step 40 on RL steps 121 to 160:
+0.44, IFEval +0.81, MATH -0.05, GSM8K +0.20; 786,432 answers against RL's 24,576, about
15 times RL's GPU-seconds per unit of gain. Established: the gains accumulate at half the
one-update rate, in RL's direction, losing MATH.

## E16. Throughput (2026-10-09 and 10)

Record: `runs/throughput-bench/`. Result: base decoding 10,560 tokens/s on an H200, LoRA
members 6,970 (factor 1.5); more sequences in flight change nothing; 128 adapters per
call 21% slower than 64; forward and backward 4,465 tokens/s, forward alone 16,400; RL's
four steps per 192 prompts 784 GPU-s (decode 100, gradient 538, reference 146), one ES
update at 512 members 4,744 GPU-s: an estimated 6.6 times RL's GPU-seconds per unit of
fitted one-update gain, 3.5 with the dead-prompt screen, 2.3 at the base model's decode
rate as well (no optimizer, communication or pipeline idling timed on the RL side); a
length cap gives up at least as large a share of ranking signal as of tokens.
Established: where the time goes. Caveat: single-GPU rates; RL's pipeline overheads not
measured.

## E17. The exact gradient (2026-10-10)

Record: `runs/gradient-check/` (`samples.json`, `gradient.json`, `projections.json`,
`scan.jsonl`, `scan-meta.json`); prediction in the script. Design: the RL run's batch for
the 192 iteration-0 prompts (16 samples at temperature 1.0, the run's advantages), GRPO's
gradient at the start as open-instruct 3f37c29 forms it; every measured update projected
onto it; held-out reward along `-+ lambda g_hat` (matrices only) and along the sign
vector. Result: 91 of 192 groups live; |g| 0.549, matrices 96% of the squared norm (down
40%, v 27%, o 12%, up 10%); 993 elements carry 24% of the unit matrix gradient's norm;
projections toward reward 0.120 (0.031) at 1,024, 0.123 and 0.116 (0.044) at 512, 16
members unresolvable; later updates and RL's step-120 displacement have no measurable
component along the start's gradient (cosine 0.019, as 120 near-orthogonal Adam steps of
0.045 give); slope along the gradient 7.3 +- 1.3 at effective length 0.022, model broken
by 0.08; along the sign direction 0.76 +- 0.06 at 0.60; ES's slope along its own direction
at least 9.4; a perfect estimate of RL's gradient would gain 0.8 +- 0.15 of what ES's
update gains per unit. The first scan's lengths (1 to 8) all broke the model; the small
lengths were added. The docstring's sign was written for the reward's gradient while the
code computes the loss's. Established: the estimate is right; the useful step is short.

## E18. Gain and cost by matrix type (2026-10-10)

Record: `runs/block-check/`; prediction in the script; and the batching lines appended to
`runs/throughput-bench/`. Design: the 1,024-member update restricted to one matrix type
at a time, `theta_0 +- mu u_t / |u_t|` at two lengths per type on 3,840 held-out prompts,
with the full update as reference; a length counts when neither sign moved the reward by
more than two points. Result: v_proj 17% of the gain with 2% of the parameters but 23% of
the cost (curvature 11 times the average per unit length); up_proj 19% / 10%; q_proj no
gain; k and gate costs unresolved; sums match the full update within 1.6 and 0.3 SE;
reweighted best net 2.4 times the full update's own fit, 1.5 times the uniform optimum
from the same per-type sums, 1.0 over the resolved types on that baseline; multipliers q
0.25, k 4.0 (clipped), v 0.95, o 0.99, gate 2.3, up 2.5, down 1.17. The three predictions
held on the baseline they named. Established: a per-type step's fitted benefit is 1.5 on
the like-for-like baseline and rests on two unresolved types; v_proj's cost tracks its
gradient density.

## E19. Eight updates at 1,024 members with the screen and per-type steps (2026-10-11)

Record: `runs/lowrank-run-1024/`; preregistration and result in `09`. Design: 1,024
members, step 30.5 with the E18 multipliers (update length 80 to 86 against 43 uniform),
a 128-member screen, 64 adapters per call. Result: held-out -0.206 +- 0.063 after 4 and
-0.323 +- 0.072 after 8 (predicted +0.25 to +0.40); IFEval +0.65 (0.09), GSM8K -0.72
(0.14), MATH -1.81 (0.16); on RL steps 121 to 160 IFEval +0.73 against RL's +0.81 at step
40 (RL has 30,720 answers by then); the screen kept 97 to 135 prompts (62% of the
rollouts), skipping about half all-right and half all-wrong prompts, and the live set grew
as the model lost prompts it had been sure of; 979,712 answers, 17.7 per second of
iteration time; on IFEval alone about 7.8 times RL's estimated GPU-seconds per unit of
gain. The one-update record (E14) already showed the random part's cost at -1.07 on MATH,
-0.48 on GSM8K, -0.19 on IFEval per step of length 112 (and the dense record of E13 at
-0.78, -0.40, -0.25). Established: the odd response's task profile held (IFEval 0.081 per
update against 0.058 at 512); the reasoning tasks fell further than the additive
quadratic model allows, and which of its assumptions failed is not isolated; the gate
fails for this configuration. The evaluation prompts also set the configuration, so they
are validation data for it.
