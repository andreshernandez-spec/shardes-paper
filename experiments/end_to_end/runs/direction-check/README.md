# The reward along RL's direction and along ES's, 2026-10-07

Part of the question of why RL gets a signal on Tulu and ES mostly noise.
`es_vllm/direction_check.py` at 19e54d6, on a Secure Cloud H200 (US-CO-1, alongside
`../rl-geometry/`), record `start.json`, logs `pod-*-log.txt.gz`. Numbers:
`python -m es_vllm.direction_check --analyze runs/direction-check/start.json`; effective
lengths from `../rl-geometry/effective-norm.json`.

The start model on the gradient probe's 768 prompts (RL steps 1 to 16) scores 5.326, as it
did in the gradient probe, on another host. It was then moved along the RL run's
displacement to step 120 (`theta_0 + lambda (theta_RL120 - theta_0)`), to the step-480
model, and along two of the arm's random member directions, and scored again. Changes
are paired over the 768 prompts.

## Along RL's direction the reward rises; along random directions of the same length it does not

The engine holds bf16 weights, so an element of a perturbation below half a bf16 step
rounds away: the length that reaches the model (effective) can differ from the nominal one.

| direction | nominal length | effective length | change (SE) | flips up / down | MATH / GSM8K / IFEval |
|---|---|---|---|---|---|
| RL, lambda 0.25 | 0.119 | 0.073 | +0.09 (0.10) | 4.0% / 3.1% | -0.19 / +0.11 / +0.24 |
| RL, lambda 0.5 | 0.238 | 0.315 | +0.85 (0.13) | 11.3% / 2.9% | +0.38 / +0.32 / +1.37 |
| RL, lambda 1 (the step-120 model) | 0.476 | 0.476 | +1.02 (0.14) | 13.3% / 3.1% | +0.14 / +0.27 / +1.88 |
| RL, lambda 2 | 0.951 | 0.950 | +1.28 (0.16) | 16.5% / 3.8% | +0.38 / +0.16 / +2.33 |
| the step-480 model | 1.134 | 1.134 | +1.76 (0.16) | 20.2% / 2.6% | +0.57 / +0.22 / +3.19 |
| random, two members | 0.238 | 0.107 | -0.07, -0.03 (0.07) | about 2% / 2% | |
| random, two members | 0.476 | 0.282 | -0.01, -0.05 (0.07) | about 2% / 2% | |
| random, two members | 0.951 | 0.734 | +0.01, -0.09 (0.07) | about 2% / 2% | |
| an ES member, sigma 5e-4 (`../grad-check/`) | 44.8 | 44.9 | -0.06 (16 members, spread 0.11) | 4.6% / 5.2% | |

- At matched effective length, RL's direction gains +0.85 at 0.32 and +1.02 at 0.48;
  random directions at 0.28 and 0.73 gain nothing (all six random changes within 1.3
  standard errors of zero), with as many outcomes flipping down as up.
- Along RL's direction the flips are one-sided: four to eight times as many up as down,
  mostly on IFEval. That is what a direction that encodes a skill does: it helps many
  prompts at once.
- The jump between lambda 0.25 and 0.5 is mostly rounding: the effective length grows
  4.3 times between them, because half and quarter bf16 steps round differently.
- An ES member's perturbation is about 90 times longer than RL's whole move to step 120
  and changes the mean by about zero: what it changes is redraws (below), and the share
  of the change that is shared across prompts is small (`../why/analysis.json`).

## Where ES's redraws come from

The start's greedy decode on the first 192 prompts, with the gap between its top two
tokens' log-probabilities at every position (bf16 logits, so gaps come in steps of 0.125):
median gap 6.4; 7.6% of positions below 0.5; 1.2% exact ties.

| perturbation | answers that depart from the start's | median position of departure | median gap there | departures at gaps below 0.5 (enrichment) | at exact ties |
|---|---|---|---|---|---|
| ES members 0-3, sigma 5e-4 | 86-90% | token 9-16 | 0.25 | 70-80% (9.2-10.5x) | 13-18% |
| RL, lambda 0.25 | 66% | token 46 | 0.125 | 97% (12.7x) | 40% |

Greedy decoding passes through positions where the top two tokens are within a few bf16
steps of each other or tied; any perturbation, even RL's smallest, flips some of them,
and the answer continues from there as a different trajectory. An ES member flips one
within the first dozen or so tokens on nearly every prompt, which is why its outcomes
behave like fresh draws (`../why/analysis.json`: q = 0.75). The quarter step along RL's
direction departs on two thirds of the answers too, but its outcome changes are only
slightly one-sided (4.0% up, 3.1% down): at that length it mostly redraws as well, and
the systematic gain appears once the step is long enough to move the decisions RL's
direction encodes.
