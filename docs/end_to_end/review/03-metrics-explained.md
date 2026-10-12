# The measurements, explained from scratch, with the numbers for both datasets

Written for a reader who does not know the metrics. Each is defined before its number;
the records behind the numbers are in `02-experiments-and-evidence.md`.

## 1. The score, and where it is measured

**Reward.** Every prompt has a checker. A math problem's checker compares the final
answer with the known one; an instruction-following prompt's checker tests the
constraint ("answer in exactly three bullet points"). The checker gives 10 for a pass and
0 for a fail, so a model's reward on a set of prompts is ten times its pass rate: 5.45
means it passes 54.5%. Countdown's checker gives partial credit for the answer format,
so its rewards are small fractions (0.04 means the model mostly fails even the format).

**Held-out.** Prompts the model is never trained on. On Tulu, 3,840 of them (two blocks of
the RL run's own stream, from later steps). Every claim about learning is a change in
reward on these, measured prompt by prompt, so a standard error of 0.06 means a change
below about 0.12 could be chance.

## 2. What each method does with the same data

Both methods see the same 192 prompts per round and generate the same number of answers,
3,072.

**RL (GRPO).** For each prompt, sample 16 answers, score them, and for each answer compute
how much better or worse it did than the other 15. Backpropagation then computes exactly
which weight changes make the good answers more likely and the bad ones less likely. That
is one step; the weights move 0.045 in length (the square root of the sum of squared
changes over all 8 billion weights).

**ES.** Draw 16 random changes to all 8 billion weights, each 45 long. Each changed copy
of the model is a **member**. Each member answers all 192 prompts; its **fitness** is its
average reward. Rank the fitnesses and move the weights toward the members above average
and away from those below. That is one **update**; its length is 11 at 16 members and 1.4
at 1,024.

So RL learns 3,072 directions per round, one per answer, each exactly attributed to
weights; ES learns 16 numbers.

## 3. Did it learn? (held-out change)

| | Tulu (8B after DPO) | Countdown |
|---|---|---|
| RL | +1.26 after 120 steps, +1.87 after 480 | no RL run |
| ES, 16 members, matched to RL's 480 steps | -0.25, below its start | learns: reward 0.04 to 0.23 in 100 updates (0.5B), the same as the reference implementation |

## 4. Is a member's score about the member? (reliability)

A member's fitness is a measurement with noise: given a second set of 192 prompts, would
it score about the same? **Reliability** is the fraction of the variation between members'
scores that would survive that re-test: 1 means the ranking is entirely a property of the
members, 0 that it is entirely which prompts they got.

- Tulu, 16 members, 192 prompts: **0.12**. Nine tenths of the ranking is noise. More
  samples per prompt and mirrored pairs did not raise it.
- Countdown, the same Tulu 8B model and the same 16 random changes: **0.42 to 0.68**. The
  0.5B model: **0.94**.

Where Tulu's noise comes from: on 86 to 90% of prompts a member's answer departs from the
original model's within the first dozen tokens, at a spot where the model was nearly
undecided between two tokens; the answer then continues as a fresh draw and passes or
fails largely at the model's usual rate. Scoring members on fixed answers instead (no
fresh draws, exact to the bit) still left the ranking mostly noise, which the next metric
explains.

## 5. Does a change help prompts together or separately? (cross-prompt correlation)

One random weight change moves each prompt's pass probability up or down. **Cross-prompt
correlation** asks whether a change that helps prompt A tends to help prompt B: 1 means
every prompt moves together, 0 that each moves on its own.

- Tulu: **0.004** (0.0095 among the prompts that can move at all).
- Countdown, the same model and changes: **0.30**. The 0.5B model: about **1**.

On Countdown the model mostly fails one thing, the answer format, and a change that fixes
it fixes every prompt at once. On Tulu the remaining gains are many separate behaviours,
and a random change touches one prompt at a time, so a 192-prompt average is mostly noise.
A 1,024-member ranking still carries a real signal, but it takes that many members.

## 6. What an update gains and what it costs (odd and even parts)

An update has a small piece along the helpful direction and a large random piece. Apply
the update both ways, theta + u and theta - u, and score held-out:
- **Odd part** = (score at +u minus score at -u) / 2. The helpful piece flips sign with the
  update and the random piece's cost does not, so this is the **gain**.
- **Even part** = the average of the two minus the start: the **cost** of the random
  piece, which grows with the square of the step length.

Tulu, per unit update:

| members | gain | cost | best net per update |
|---|---|---|---|
| 16 | +0.004 +- 0.003 | -0.0038 | 0.001 |
| 1,024 | +0.0047 +- 0.0004 | -0.00008 | 0.071 |
| RL, per 4 steps | | | 0.040 |

The gain does not change with members; the cost falls as 1/N. At 16 members they cancel,
which is why the run stayed flat. At 1,024 the fitted optimum exceeds RL's average
four-step gain, at 64 times the answers; the one step scored near that optimum gave
+0.057 +- 0.053, so the optimum is a fit. On Countdown the margin of gain over cost was
about 600 for the 0.5B model and 6 to 29 for the Tulu 8B start, against about 6 on Tulu.

By task, at 1,024 members and one step of length 112, the cost is -1.07 on MATH, -0.48 on
GSM8K and -0.19 on IFEval: the random piece damages reasoning five to six times more than
instruction following.

## 7. How far the weights move (lengths, and bf16 effective length)

Lengths are Euclidean distances over all weights. RL's whole 120-step move is 0.476 long;
one ES member's change is 45, and it moves the model's answers about as much as RL's 120
steps (6 nats of log-probability per answer against 8). Along RL's direction the held-out
reward rises +0.85 at length 0.32; along random directions of the same length, nothing.

**Effective length.** The engine stores weights in bf16, about three significant digits,
so a tiny change to a weight rounds away. A change's effective length is its length after
rounding, which is what the model experienced; it matters for RL's small steps, not for
ES's large ones.

## 8. Is ES's update pointing the right way? (the projection on the exact gradient)

RL's exact gradient was computed for the same batch and ES's update projected onto it.
The projection, as a fraction of the most a 1,024-number estimate could have, is
**0.12 +- 0.03**, toward higher reward, from three independent decodes. Two scans showed
that ES's useful piece is at least as steep as RL's gradient direction and that a perfect
estimate of RL's gradient would gain 0.8 of what ES already gains per unit. The ranking is
not the problem: the useful piece is at most 0.0005 long per update beside a random piece
of 1.4.

## 9. Compute per unit of gain (GPU-seconds)

On one H200, at per-token rates, RL's four steps are estimated at 784 GPU-seconds (100
decoding, 538 the gradient, 146 the reference model; no optimizer or communication) for
+0.040 on average; one ES update at 512 members costs 4,744 for a fitted +0.037. ES used
an estimated 6.6 times RL's compute per unit of gain on one update, about 15 over eight.
Skipping the gradient saves RL's largest cost; ES pays 32 times more answers.

## 10. The two datasets side by side

| | Tulu 8B on Tulu | Tulu 8B on Countdown | Qwen 0.5B on Countdown |
|---|---|---|---|
| start reward | 5.45 of 10 | 0.17 of 1.1 | 0.04 of 1.1 |
| what the model gets wrong | many separate things | mostly one thing (format) | one thing |
| outcomes a member changes | 10% | 66% | 66% |
| cross-prompt correlation | 0.004 | 0.30 | about 1 |
| reliability at about 200 prompts | 0.12 | 0.42 to 0.68 | 0.94 |
| gain over random cost at 16 to 30 members | about 6 | 6 to 29 | about 600 |
| does ES learn at the reference settings | no | not tested | yes |

## 11. What it adds up to

RL is told which weights produced each answer and moves them; ES finds that out by trial,
each trial a random change to all 8 billion weights scored by one number. When the model
fails one shared thing, a trial's score means something and ES works. When the model is
already good and its gains are many separate behaviours, a trial's score is mostly noise,
the random change costs more than the useful piece gains, and no fitness, optimizer or
step size changes that, only the number of trials. With a thousand trials per update ES
learns the shared skill (IFEval) as fast as RL, but the random part of those trials
destroys the fragile skills (MATH), which RL leaves untouched.
