# How much of an ES gradient estimate on Tulu is signal? 2026-10-07

Diagnosis asked for by Andres when the longer arm (`docs/end_to_end/07`) had barely moved
after 30 updates. An ES update is the members' noise weighted by their z-scored fitness;
for a given seed it is fixed by those 16 weights, so the question is whether the members'
ranking is a property of their weights or of the prompts drawn. `es_vllm/grad_check.py`
(7f83eb5) decoded the start model and the arm's own iteration-0 members (seed 0) on the
768 prompts of RL steps 1 to 16, in the run's chunks of 192, with per-prompt rewards, at
sigma 5e-4, 1e-3 and 2e-3 along the same noise directions, and the mirrored members at
5e-4. One Secure Cloud H200 (US-NC-1, $4.59/h), 14:39 to 15:47 UTC, about $5. Record
`start.json`; numbers below: `python -m es_vllm.grad_check --analyze runs/grad-check/start.json`.

**Exact check**: on the first 192 prompts the start scores 5.000 and every member's mean
equals the arm's logged iteration-0 fitness (16 of 16), so this is the run's gradient.

## The members' ranking is mostly prompt noise

Reward minus the start's reward, per member and prompt (0-10 scale), decomposed into the
variance of the members' true mean effects and the member-prompt residual:

| sigma | members vs start | outcomes changed (up / down) | true effect sd | residual | reliability at 192 prompts (90%) | split halves at 192 | at 384 | implied gain per iteration |
|---|---|---|---|---|---|---|---|---|
| 5e-4 (the run's) | -0.06 | 9.8% (4.6 / 5.2) | 0.062 | 5.6 | 0.12 (0.01-0.47) | 0.13 | 0.21 | 0.021 |
| 1e-3 | -0.22 | 13.2% (5.5 / 7.7) | 0.022 | 7.1 | 0.01 (0.00-0.42) | 0.02 | 0.03 | 0.001 |
| 2e-3 | -1.45 | 22.3% (3.9 / 18.4) | 0.131 | 9.2 | 0.26 (0.09-0.56) | 0.28 | 0.42 | 0.017 |

Reliability is the share of the variance of a member's mean over 192 prompts that is its
true effect, `v / (v + residual / 192)`; "split halves" measures the same thing directly
as the correlation of the members' means between disjoint prompt subsets, and agrees.
Implied gain is first order for z-scored weights: `(alpha / sigma) sqrt(v * reliability)`.

- At the run's sigma, about 12% of the member-to-member fitness differences are a property
  of the members; the rest is which of the roughly 10% of outcomes a perturbation happens
  to flip. The update is mostly a random direction, which is what the arm shows: it beats
  the random-walk control only by about as much as the control degrades.
- Larger sigma raises the signal (2e-3) but the perturbations themselves cost 1.45 points
  on average, with downs outnumbering ups 5 to 1: the ranking is then of how much each
  member is damaged. 1e-3 shows no more signal than 5e-4 here; with 16 members the
  intervals are too wide to rank the two.
- Mirrored members do not help per rollout: the antithetic difference of a pair is about
  as reliable as one member (0.11 at 192), at twice the decodes.
- By source at 5e-4, outcomes change on 17% of MATH, 8% of GSM8K and 7% of IFEval prompts.
  The per-source member effects are larger than the pooled one, and the pooled variance
  equals their share-weighted sum: effects on the three task families are about
  independent, so mixing them dilutes each one's signal rather than cancelling it.
- The implied gain at 5e-4, 0.021 per iteration, is first order and an upper bound; the
  arm's measured arm-minus-control center slope is 0.007 per iteration (`07`'s run so far).
- A reliability of 0.5 at this sigma would take about 1,400 prompts per member. At a fixed
  rollout budget that means fewer members; to first order, with the step size retuned, the
  gain per iteration does not depend on how the budget is split between N and P.

Caveat: 16 members make each sigma's estimate rough (the 90% intervals above, resampling
members and prompts). The conclusion that most of the ranking is noise holds across them.

## Sampled fitness is no better per rollout (`start-sampled.json`)

Approved by Andres the same day as the next diagnostic. The same members on the arm's
192 iteration-0 prompts, each prompt sampled 8 times at temperature 1.0 (the RL run's),
once with common random numbers (every member samples prompt j with the same seed) and
once with independent seeds per member (a44cd57; one Secure Cloud H200 in EU-FR-1,
driver 580.159.04, 16:41 to 17:25 UTC, about $3.50). Numbers:
`python -m es_vllm.grad_check --analyze runs/grad-check/start-sampled.json --greedy runs/grad-check/start.json`.
Reliability is the split-half correlation of the members' means at a given number of
rollouts per member, spent as prompts x samples:

| rollouts per member | greedy (prompts x 1) | sampled, common seeds | sampled, independent seeds |
|---|---|---|---|
| 96 | 0.074 | 0.054 (96x1), 0.054 (48x2), 0.063 (24x4), 0.045 (12x8) | 0.017, 0.014, 0.017, 0.021 |
| 192 (the run's) | 0.130 | 0.113 (96x2), 0.090 (48x4), 0.084 (24x8) | 0.037, 0.031, 0.022 |
| 384 | 0.241 | 0.192 (96x4), 0.152 (48x8) | 0.080, 0.065 |

- The members' true effects are the same size under both measures (sd 0.061 sampled
  with common seeds, 0.063 greedy, estimated as the covariance of their means between
  disjoint prompt halves). Sampling adds per-rollout noise without adding signal, so the
  best split is the most prompts with the fewest samples, which greedy already is.
- Common random numbers matter: they triple the reliability against independent seeds,
  which only brings sampling back to about greedy's level.
- Sampling at temperature 1.0 scores the start at 4.90 on these prompts (greedy 5.00),
  with responses of about 350 tokens.

At matched rollouts, then, neither the decoding nor mirrored pairs nor the split of the
budget changes the picture: on this task a perturbation of the size ES needs moves the
reward by little next to the outcome noise of a few hundred rollouts, and about nine
tenths of each ranking is noise.
