# 09. ES's best achievable efficiency on Tulu 3.1: preregistration

Written and committed 2026-10-10, before any run. Decided by Andres after `08`: the
remaining question is how close ES can come to RL's compute per unit of gain on this
stream once the levers `08` identified are applied, since its structural advantages
(inference-only memory, scalar communication, any servable model) only matter if that
distance is small. Two stages; the second is configured from the first, by the rules
below, and not otherwise changed after the data.

## Stage A: two measurements, one pod

**A1. Per-matrix-type gain and cost** (`es_vllm/block_check.py`, `runs/block-check/`). The
N = 1,024 update of `runs/lowrank-check/` restricted to one matrix type at a time, applied
as `theta_0 +- mu u_t / |u_t|` at two lengths per type on the 3,840 held-out prompts, with
the full update at the same lengths on the same engine. Per type, the odd part gives the
slope along the type's component and the even part its curvature; the unit update's gain
`G_t` and cost `C_t` from that type follow, and their sums are checked against the full
update's. The best net per update with one step size is `(sum G_t)^2 / (4 sum C_t)`; with
a step size per type it is `sum_t G_t^2 / (4 C_t)`, reached at multipliers
`m_t = (G_t / C_t) / (sum G / sum C)`.

**A2. LoRA batching** (`es_vllm/throughput_bench.py --part lora`, `runs/throughput-bench/`):
members decoded in chunks of 64 adapters with 512 sequences in flight (the run's setting)
against 64 with 1,024 and 128 with 1,024.

## Stage B: eight updates at 1,024 members

`es_vllm/lowrank_run.py` at N = 1,024, eight updates along RL's prompt stream (RL steps 1
to 32's data), held-out after 0, 4 and 8 updates, as `runs/lowrank-run/` did at 512, with:

- the step at the best one-update multiple for 1,024 members, `lambda` = 30.5
  (`runs/lowrank-check/`: gain 0.0047 per unit, cost 0.000077 per unit squared);
- the dead-prompt screen with 128 screening members: the remaining 896 members decode
  only the prompts on which the 128 did not all score alike. On the iteration-0 record
  this keeps 99.7% of the outcome changes at 63% of the rollouts; the prompts it skips,
  with the start's reward on them, are logged every iteration so that what it cost can be
  reported (Andres's reservation: rare breakthroughs on prompts the model fails are among
  what a screen can miss);
- per-type step multipliers from A1, only if A1's reweighted best net is at least 1.3 times
  the uniform one and its sums check; multipliers are clipped to [0.25, 4] and applied as
  a scale on each type's part of the update;
- the batching of A2 if it is faster.

## Predictions

- **A1.** v_proj's share of the gain exceeds its 2% share of the parameters by at least
  five times. The reweighted best net is between 1.5 and 3 times the uniform one (5 if
  the curvature per parameter is uniform across types, 1 if it tracks the gradient's
  density). The sums of `G_t` and of `C_t` match the full update's gain and cost within
  two standard errors.
- **A2.** More adapters and sequences per call raise member throughput by less than 1.2x.
- **B.** Held-out change after 8 updates between +0.25 and +0.40 (the 512-member run gave
  +0.141 at half the one-update rate; the one-update best net at 1,024 is twice that at
  512, and the multipliers add what A1 says). RL's gain on these prompts after its 32nd
  step is about +0.35 (interpolated from +0.44 at step 40). MATH's loss smaller than the
  512-member run's -0.46, since the random part per update is half as long.

## Gate

GPU-seconds per unit of held-out gain, ES against RL, at the rates of
`runs/throughput-bench/` (RL 784 GPU-s per four steps for +0.040 on its first 120 steps;
ES's time measured in the run):

- **Within 3x**: reported as "ES reaches RL-like progress on a realistic post-training
  stream from inference infrastructure alone, at a compute penalty of under 3x", and the
  structural demonstration (`08`, step 3) is worth costing.
- **Above 3x**: reported as the complete negative result, with the best achieved factor.

Reported whatever the outcome: the held-out change at 4 and 8 updates by task, against
RL at steps 16 and 32 where branches exist (step 40 otherwise); the screen's skipped
prompts per iteration and the start's reward on them; the realized gain per update
against the one-update prediction; the per-type multipliers used; iteration times and
member throughput.

## Cost

Stage A about 3 hours on one H200 ($5.29/h), about $17. Stage B about 14 hours, about
$75. The program has spent about $330 of its first $350; this is the second budget.

## Result, 2026-10-11

**Stage A** (`experiments/end_to_end/runs/block-check/README.md`,
`runs/throughput-bench/README.md`): the three A1 predictions held (v_proj's gain share 8.8
times its parameter share; reweighted best net 2.4 times the uniform one, 1.5 over the
types whose cost is resolved; sums within 1.6 and 0.3 standard errors), but the mechanism
was not the one assumed: v_proj's curvature is 11 times the average per unit length, so
its multiplier is 0.95, and the reweighting came from up_proj (2.5), the removal of q_proj
(0.25) and two types with unresolved costs (gate 2.3, k clipped to 4). A2: 128 adapters per
call were 21% slower than 64; the batching stayed. By the rules above the multipliers
went into stage B.

**Stage B** (`runs/lowrank-run-1024/README.md`): held-out -0.206 +- 0.063 after 4 updates
and -0.323 +- 0.072 after 8, against the predicted +0.25 to +0.40. IFEval +0.65 (RL's own
pace on these prompts), GSM8K -0.72, MATH -1.81. **The gate fails**: the gain is negative,
so ES's compute per unit of gain is not within any multiple of RL's. The cause is in the
one-update record and was averaged away by the pooled cost model: the random part's cost
falls on MATH at five to six times the rate it falls on IFEval, so a step sized by the
pooled cost over-steps the reasoning tasks, and the damage accumulates faster than eight
single-step costs. The screen behaved as predicted (62% of the rollouts, the skipped
prompts half all-right and half all-wrong for the screening members) and the live set grew
as the model lost prompts it had been sure of.

Reported as `09` fixed: the complete negative result. The best factor achieved on the
pooled measure is none; on IFEval alone, which is not the measure, ES used 6.3 times RL's
GPU-seconds per unit of gain while losing the reasoning tasks. The structural
demonstration (`08`, step 3) is not worth costing on this evidence.
