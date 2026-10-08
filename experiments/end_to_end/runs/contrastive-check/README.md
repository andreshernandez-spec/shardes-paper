# ES with RL's contrast as its fitness, 2026-10-08

Andres's proposal, after the gradient probe (`../grad-check/`) found nine tenths of the
members' ranking to be noise: score a member the way RL scores a sample, by contrasting
right and wrong answers to the same prompt. `es_vllm/contrastive_check.py` at fc02e93 on a
Secure Cloud H200 (US-NC-1, $4.59/h, 08:24 to 09:21 UTC, about $4.40). Record `start.json`,
logs `pod-log*.txt.gz`, status lines `pod-status.txt`. Numbers:

    python -m es_vllm.contrastive_check --analyze runs/contrastive-check/start.json \
        --reward runs/grad-check/start.json

The start sampled 16 answers at temperature 1.0 (the RL run's setting) to each of the 384
prompts of RL steps 1 to 8; the run's verifiers marked them. 186 prompts got both right
and wrong answers (2,976 answers: 1,698 right, 1,278 wrong). A model's fitness is the mean
over those prompts of mean log p(right answers) minus mean log p(wrong answers), sequence
log-probabilities teacher-forced on the fixed answers. Nothing is generated, so nothing is
redrawn. Scored: the start, the arm's 16 iteration-0 members (seed 0, sigma 5e-4, the
gradient probe's members), two random member directions at RL's length, the RL run's
direction to step 120 (as in `../direction-check/`), and the start again after the
restore.

The first full attempt (ed40f47) stalled: the verifier was handed 6,144 answers in one
batch, its reply pipe filled while the probe was still writing, and both processes
blocked (`pod-log-deadlock.txt.gz`). Fixed in 1b2f8fc; the restart at fc02e93 added the
rescore of the start.

## The scoring is exact; the ranking is still mostly prompt-specific

The start rescored after the restore gives the same 2,976 log-probabilities to the bit.
Every difference between members is therefore what each member does to each prompt.

| fitness | prompts drawn | prompts scored per member | reliability | cross-prompt correlation of member effects |
|---|---|---|---|---|
| greedy reward (`../grad-check/`) | 192 | 192 | 0.13 | 0.0095 (`../why/`, uncertain prompts) |
| greedy reward | 384 | 384 | 0.21 | |
| contrast, sequence log p | 192 | 93 | 0.22 | 0.0024 |
| contrast, sequence log p | 384 | 186 | 0.36 | |
| contrast, log p per token | 192 | 93 | 0.08 | 0.0005 |
| contrast, log p per token | 384 | 186 | 0.14 | |

Reliability is the share of the variance of a member's mean that is common to prompts,
measured as the correlation of the members' means between disjoint sets of prompts of
that size: for the contrast, halves of the 186 averaged over 2,000 splits, and the
Spearman-Brown value for all 186. Half the drawn prompts have both right and wrong answers
and only those count.

- A member moves a prompt's contrast by 2.9 nats (sd over members) and 0.24% of that is
  common to other prompts. Over 186 prompts, about a third of the variance of a member's
  fitness is common to prompts and the rest belongs to the particular prompts drawn.
- Part of the common part is not about right against wrong. Wrong answers are longer
  (median 384 tokens, right 150). Every member makes the start's answers less likely, by
  6.1 nats per answer on average, and a sum over more tokens falls further: the members
  raise the contrast by +0.59 on average, and the members that move the answers most
  score highest (correlation 0.40 with the drop below). Per token, the length effect is
  gone and 14% of the ranking is common to prompts, less than the reward's 21% from the
  same 384 prompts.
- The contrastive ranking does not predict the members' reward ranking from the gradient
  probe: correlation 0.08 with their reward change on the 384 prompts the contrast did
  not use, 0.20 on its own prompts. With 16 members, either value is within noise.

## A member and RL's step move the answers as far, in different directions

The start sampled these answers, so minus the mean change of their log-probability
estimates KL(start || model) per answer. Effective lengths after bf16 rounding are from
`../rl-geometry/effective-norm.json`.

| model | nominal length | effective length | KL per answer | log p right | log p wrong | contrast (SE) | prompts up |
|---|---|---|---|---|---|---|---|
| ES member, mean of 16 | 44.8 | 44.9 | 6.09 | -5.92 | -6.51 | +0.59 (member sd 0.26) | |
| random, two members | 0.476 | 0.282 | 0.04 | -0.05, -0.07 | -0.04, -0.06 | -0.01, -0.01 (0.03) | 52%, 49% |
| random, two members | 0.951 | 0.734 | 0.05 | -0.06, -0.06 | -0.05, -0.07 | -0.00, +0.00 (0.03) | 49%, 53% |
| RL, lambda 0.25 | 0.119 | 0.073 | 0.13 | -0.03 | -0.35 | +0.32 (0.07) | 66% |
| RL, lambda 0.5 | 0.238 | 0.315 | 2.25 | -1.55 | -3.42 | +1.87 (0.26) | 78% |
| RL, lambda 1 (the step-120 model) | 0.476 | 0.476 | 7.82 | -6.29 | -10.39 | +4.10 (0.47) | 81% |

- The step-120 model is 90 times closer to the start than a member and moves the start's
  answers further (7.8 nats per answer against 6.1). Random directions of RL's length
  move them by about 0.05 and leave the contrast where it was.
- RL's move is coherent: wrong answers fall 4.1 nats more than right ones, and the
  contrast rises on 81% of the prompts. A member's move changes each prompt's contrast in
  its own way.
- Along RL's direction the right answers become less likely too. The RL model moves away
  from the start's samples, the wrong ones faster.

## What one ES update ranked by this fitness would gain

The update is `(alpha/N) sum_m z_m eps_m`. To first order it changes the objective by
`(alpha/sigma) mean_m z_m delta_m`, which is measured without bias by ranking the members
on one half of the contrast prompts and taking their changes on the other half. Its
random part is sqrt(N) times shorter than a member, so its second-order change is the
members' mean change over N.

| | per iteration (192 prompts) |
|---|---|
| ES, first order, ranked on half, judged on the other half | +0.067 nats |
| ES, first order, in sample | +0.25 |
| ES, second order (the length effect above) | +0.037 (0.004) |
| RL to step 120, averaged over its 30 iterations' worth of prompts | +0.137 |

- The held-out gain is an upper bound: part of the ranking it rests on is the length
  effect, which favors members that move the answers more, not members that separate
  right from wrong.
- The RL figure is on prompts it trained on in its first 8 steps (with its own samples),
  which favors it slightly.

## Noise or dimension

- Removing the redraws does not make the ranking reliable. With nothing redrawn, a
  member's fitness over 186 prompts is still mostly specific to those prompts: 64% of its
  variance per answer, 86% per token. A random direction changes each prompt's contrast
  in its own way, and its effects on two prompts correlate at 0.0024.
- That prompt-specific share is not particular to ES. With an exact fitness, the members'
  z-scores are projections of the batch gradient of this objective on their noise, so to
  first order the update points, in expectation, along the gradient RL's step on these
  prompts would follow, and that gradient carries the same prompt-specific share. Both
  methods average it out over steps.
- What ES pays and RL does not is the rest of its step: a random part of length
  `alpha sqrt(d/N)` = 11.2 per iteration around a useful part of order `alpha`. A member
  moves the start's answers almost as far as 120 RL steps do (6.1 against 7.8 nats), and
  what it does to the contrast is specific to each prompt. The random part's cost grows
  as `alpha^2` and the useful gain as `alpha`, so the random part bounds the step size
  (`../why/`: 0.004 reward points per iteration at this `alpha`). At this step size the
  useful gain with this fitness is at most 0.07 nats per iteration, against RL's 0.14.
