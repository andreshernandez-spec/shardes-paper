# More members: the gain per update stays, the cost falls as 1/N, 2026-10-09

With 16 members, one dense ES update on Tulu gains about what its random part costs
(`../step-check/`). Andres's proposal: hundreds of members per update, low-rank, as EGGROLL
does. `es_vllm/lowrank_check.py` and `es_vllm/lowrank.py` at a7a7e8b, prediction in the
docstring committed before the run, on a Secure Cloud H200 (US-CO-1, driver 580.178.04,
$5.29/h, 09:58 to 13:16 UTC), about $17.50. Records `members.json` and `steps.json`, the
numeric predictions written before any point was scored `predictions.json` (12:47:33 UTC;
the first point was scored after it), logs `pod-log-*.txt.gz`. Numbers:

    python -m es_vllm.lowrank_check --analyze runs/lowrank-check/steps.json

Members are rank-1 perturbations `sigma a b^T` (sigma 5e-4) of every attention and MLP
matrix, in mirrored pairs, each served as a LoRA adapter that vLLM builds in memory from
its seed, 64 adapters per batch on one copy of the base weights. 1,024 members decoded
the arm's 192 iteration-0 prompts (9,382 s, about 6,700 tokens/s). The update from the
first N members (N = 128, 512, 1,024, nested) was then merged into the weights at
`theta_0 +- lambda u` and decoded on the 3,840 held-out prompts of RL steps 121 to 160 and
481 to 520, as in `../step-check/`, at the same two lengths of `lambda u` (44.7, 111.7).

**Checks.** An adapter is the perturbation the update assumes: on the start's answers to
32 prompts, the change in log p under members 0 and 1 as adapters and as weights merged
into the model correlates 0.997 and 0.995 (slopes 1.01 and 0.94). Every merged point
matched the expected bf16 weights bit for bit. The members change 9.6% of the outcomes on
the 192 prompts (dense members 9.8%).

## Gain and cost per update

Per unit update (alpha = sigma = 5e-4, `u = (alpha / N) sum_m z_m E_m`), on the held-out
prompts, at length 111.7 (44.7 in parentheses):

| members | update length | gain per unit | cost per unit^2 | best net per update `g^2 / 4c` |
|---|---|---|---|---|
| 16, dense (`../step-check/`) | 11.17 | +0.004 +- 0.003 | -0.0038 | 0.001 |
| 128 | 3.92 | +0.0037 +- 0.0011 (+0.0074) | -0.00054 +- 0.00007 | 0.006 +- 0.004 |
| 512 | 2.01 | +0.0047 +- 0.0006 (+0.0049) | -0.00015 +- 0.00002 | 0.036 +- 0.010 |
| 1,024 | 1.41 | +0.0047 +- 0.0004 (+0.0056) | -0.000077 +- 0.000009 | 0.071 +- 0.016 |

- The cost falls as 1/N: by 7.0 times from 128 to 1,024 members. At each N it is 1.2 to
  1.3 times the dense update's cost at the same length (-0.00046, -0.00012, -0.00006
  scaled from `../step-check/`): a low-rank random part costs about what a dense one
  does per unit length.
- The gain does not depend on N, and it is about the dense one. At 1,024 members it is 11
  standard errors from zero: a ranking on 192 prompts carries a first-order signal that
  transfers to prompts it never saw. With 16 members it was hidden in the +-0.011 that a
  random weighting of 16 directions gives; with 1,024 it is about +-0.001.
- So the best net per update grows in proportion to N, as the docstring predicted (about
  0.07 at 1,024). The RL run gained 1.19 on these held-out prompts in its first 120
  steps, 0.040 per four steps, which see the same 192 prompts as one ES update. Per
  update, ES matches that at 512 members and exceeds it at 1,024: at 32 and 64 times RL's
  3,072 rollouts.
- A single step near the best length (1,024 members, lambda 31.8, length 44.7) changes the
  held-out reward by +0.057 +- 0.053; the paired design measures gain and cost more
  precisely than one step can.
- The numeric cost predictions in `predictions.json` have the wrong sign. They scale the
  members' mean change on the 192 prompts, +0.079, measured against one greedy decode of
  the start, which is itself one redraw (about +-0.2 on 192 prompts). The docstring's
  form, the dense cost scaled by 1/N, holds.
- Mirrored low-rank members show no sign-independent differences between pairs (true
  variance of the even part 0, against 0.0035 for dense members, `../step-check/`), and
  the pairs cancel it in the update in any case.

## What this does not show

- It is one update from the start. Whether gains accumulate at this step size is not
  measured: the best step at 1,024 members is 43 long, 90 times the RL run's whole move to
  step 120, and after 30 such updates the random part would be about 240 long, where the
  dense members' cost per unit length squared was 1.5 times the value at small steps
  (`../why/`).
- The price is in rollouts. 1,024 members on 192 prompts are 196,608 rollouts per update,
  64 times RL's; here 2.6 hours of one H200 per update.
