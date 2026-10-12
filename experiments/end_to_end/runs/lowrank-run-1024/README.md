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
- IFEval rises 0.081 per update over all held-out prompts, 1.4 times the 512-member run's
  0.058 (1.75 times on the first block); the RL checkpoint at step 40 has +0.81 on the
  first block, 0.08 per four steps, measured at 40 steps rather than at the 32 these
  eight updates match.
- The random part's cost is concentrated on the reasoning tasks: in the one-update record
  (`../lowrank-check/`, N = 1,024 at length 112) the even part is -1.07 on MATH, -0.48 on
  GSM8K and -0.19 on IFEval. That explains the trade between tasks, not the pooled loss:
  the additive quadratic model fitted to the same record predicted +0.15 per update, so
  at least one of its assumptions failed over eight updates (coefficients stable as the
  weights move, quadratic response at these lengths, additivity across matrix types, the
  screen's equivalence to the full ranking), and these records do not isolate which. The
  preregistration's expectation of less MATH damage than at 512 members also rested on an
  arithmetic error: at the uniform step the random part at 1,024 members is 43 long,
  sqrt(2) longer than that run's 31, not half, and the multipliers made it 85.

## The screen

| iteration | prompts kept | skipped, all right for the 128 | skipped, all wrong | rollouts |
|---|---|---|---|---|
| 0 | 99 | 44 | 49 | 113,280 |
| 1 to 6 | 97 to 116 | 34 to 51 | 37 to 47 | 111,488 to 128,512 |
| 7 | 135 | 24 | 33 | 145,536 |

979,712 rollouts in all, 62% of the 1,572,864 without the screen; 55,308 s of iteration
time (54,849 s decoding), 17.7 answers per second against 20.9 for the unscreened engine
(two decoding phases per iteration, and the live set grew as the model moved: by
iteration 7 the model had become uncertain on 36 prompts it had been sure of at iteration
0, mostly ones it used to get right). On the iteration-0 population of
`../lowrank-check/` the screened update's pair coefficients have cosine 0.997 with the
unscreened ones at 128 screening members and 0.992 at 64
(`python -m es_vllm.lowrank_run --screen-check runs/lowrank-check/members.json`); what the
screen does to later updates, whose live sets differ, was not measured against an
unscreened run.

## Cost against RL

ES decoded 979,712 answers in 15.4 hours of iteration time on one H200 for a held-out
change of -0.32; the RL checkpoint at step 40 has +0.44 on the first block (30,720
answers, about 2.2 GPU-hours estimated from the per-token rates of
`../throughput-bench/`, which time no optimizer, communication or pipeline idling).
Counting IFEval alone on the first block, which is not the measure `09` fixed: ES +0.73
in 55,308 s, RL +0.81 in about 7,845 s estimated, 7.8 times RL's GPU-seconds per unit of
IFEval gain, with the reasoning tasks lost.
