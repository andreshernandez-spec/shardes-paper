# Eight low-rank ES updates at 1,024 members with the screen and per-type steps, 2026-10-11

Stage B of `docs/end_to_end/09`: the configuration its rules selected from stage A
(`../block-check/`, `../throughput-bench/`). `es_vllm/lowrank_run.py` at 4e1e807, on a
Secure Cloud H200 (US-CO-1, driver 580.178.04, $5.29/h, 22:26 to 14:08 UTC), about $83.
Records `log.jsonl` (every member's reward on every prompt it decoded, the prompts the
screen skipped with the screening members' common reward, the update's norm, a digest of
the f32 weights), `heldout.jsonl`, `run.json`; logs `pod-log*.txt.gz`. Numbers:

    python -m es_vllm.lowrank_run --analyze runs/lowrank-run-1024

1,024 rank-1 mirrored members per iteration (seeds as `../lowrank-check/` at iteration 0),
step 30.5 times the unit update with the multipliers per matrix type of `../block-check/`
(q 0.25, k 4.0, v 0.952, o 0.985, gate 2.336, up 2.522, down 1.167; update length 80 to 86
against 43 without them), a screen of 128 members (the other 896 decode only the prompts
on which the 128 did not all score alike), 64 adapters per call. RL's prompt stream,
iteration g on RL steps 4g+1 to 4g+4; held-out after 0, 4 and 8 updates on the 3,840
prompts of RL steps 121 to 160 and 481 to 520.

## Held-out reward

| after | reward | change (SE) | IFEval | GSM8K | MATH | prompts up / down |
|---|---|---|---|---|---|---|
| 0 updates | 5.453 | | | | | |
| 4 updates | 5.247 | -0.206 (0.063) | +0.30 (0.07) | -0.27 (0.13) | -1.13 (0.16) | 253 / 332 |
| 8 updates | 5.130 | -0.323 (0.072) | +0.65 (0.09) | -0.72 (0.14) | -1.81 (0.16) | 317 / 441 |

On RL steps 121 to 160 alone, after 8 updates: IFEval +0.73, GSM8K -0.91, MATH -1.61, all
-0.26; the RL run at step 40 on the same prompts (`../heldout-121-160/`): IFEval +0.81,
GSM8K +0.20, MATH -0.05, all +0.44. The `../lowrank-run/` 512-member run (step length
31, no screen, no multipliers) after 8: IFEval +0.47, GSM8K +0.10, MATH -0.46, all +0.14.

- `09` predicted +0.25 to +0.40 after 8 updates; the run gives -0.32 +- 0.07. The gate,
  ES's GPU-seconds per unit of held-out gain within 3 times RL's, fails: the gain is
  negative.
- The useful part of the updates does what the one-update measurements said. IFEval
  rises 0.08 per update, twice the 512-member run's 0.06 and about RL's own pace on these
  prompts (+0.81 by step 40, 0.08 per four steps).
- The random part destroys the reasoning tasks. Its cost is concentrated there: in the
  one-update record (`../lowrank-check/`, N = 1,024 at length 112) the even part is -1.07
  on MATH, -0.48 on GSM8K and -0.19 on IFEval. The pooled quadratic model that set the
  step and the multipliers averaged that away, so the step it chose (length 85 per
  update, 240 accumulated) is far past what MATH tolerates, and the accumulated damage
  is larger than eight single-step costs would add to.

## The screen

| iteration | prompts kept | skipped, all right for the 128 | skipped, all wrong | rollouts |
|---|---|---|---|---|
| 0 | 99 | 44 | 49 | 113,280 |
| 1 to 6 | 97 to 116 | 34 to 51 | 37 to 47 | 111,488 to 128,512 |
| 7 | 135 | 24 | 33 | 145,536 |

979,712 rollouts in all, 62% of the 1,572,864 without the screen; 55,308 s of decoding,
17.7 answers per second against 20.9 for the unscreened engine (two decoding phases per
iteration, and the live set grew as the model moved: by iteration 7 the model had become
uncertain on 36 prompts it had been sure of at iteration 0, mostly ones it used to get
right).

## Cost against RL

ES decoded 979,712 answers over 15.4 GPU-hours for a held-out change of -0.32; the RL run
reaches +0.44 on the RL steps 121 to 160 prompts by step 40 (24,576 answers, about 2.2
GPU-hours at the rates of `../throughput-bench/`). Counting IFEval alone, which is not
what `09` fixed as the measure: ES +0.73 for 55,308 GPU-seconds, RL +0.81 for about
7,800, 6.3 times RL's GPU-seconds per unit of IFEval gain, with the reasoning tasks lost.
