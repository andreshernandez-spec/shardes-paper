# Does one ES update deliver the gain its ranking implies? 2026-10-08

The arm beat its random-walk control by 0.0022 reward points per update over 120 updates,
a tenth of the 0.021 the gradient probe implied (`../grad-check/`); on Countdown T2
realized a third. Whether ES on Tulu needs about RL's rollouts or about a hundred times
more turns on where that factor goes. `es_vllm/step_check.py` at 5ab60fc, prediction in
its docstring committed before the run, on a Secure Cloud H200 (US-NC-1, driver
595.91.07, 12:08 to 13:57 UTC; a first pod that never came up was deleted after 3
minutes), about $8.50. Record `start.json`, logs `pod-log*.txt.gz`. Numbers:

    python -m es_vllm.step_check --analyze runs/step-check/start.json
    python -m es_vllm.step_check --predict

The iteration-0 update `u` (seed 0, N = 16, alpha = sigma = 5e-4) was built from four
rankings of the same 16 members, the start moved to `theta_0 + lambda u` and
`theta_0 - lambda u` at lambda 4 and 10, and each decoded greedily on 4,608 prompts: the
gradient probe's 768 and RL steps 121 to 160 and 481 to 520 (3,840, held out from every
ranking). Half the difference of the two signs is the update's odd part, to first order
`lambda` times its gain; the cost of the random part is even and cancels. Their mean minus
the start is the even part, to second order `lambda^2` times that cost.

**Exact check:** `theta_0 + 1 u` from the arm's ranking equals the arm's first update
(`es_tell`) to the bit. The start scores 5.326 on the 768 prompts, as in `../grad-check/`
and `../direction-check/` on other hosts.

## The cost is as predicted; the gain is about the cost

Per unit update (lambda = 1) on the 3,840 held-out prompts, at lambda 10 (lambda 4 in
parentheses, about twice the standard error):

| update ranked by | gain predicted (all / odd part) | gain measured (odd part) | cost measured (even part) |
|---|---|---|---|
| the gradient probe's 768 prompts | +0.039 +- 0.029 / +0.014 +- 0.024 | +0.004 +- 0.003 (+0.006) | -0.0042 +- 0.0005 |
| the contrastive probe's fitness | | +0.005 +- 0.003 (+0.004) | -0.0032 |
| the arm's logged fitness (192 prompts) | +0.012 / +0.011 | -0.010 +- 0.003 (-0.004) | -0.0043 |
| the control's random fitness | -0.024 / -0.015 | -0.010 +- 0.003 (-0.008) | -0.0038 |

Predicted cost -0.0038 (the members' mean change over N). The predictions' errors are a
jackknife over the 16 members; the arm's and the control's rest on 16 members' changes
on 576 and 768 prompts and carry about +-0.02.

- The cost is what the second-order model says, for every ranking alike: it depends on
  the update's length, not its direction. At lambda 10 (length 112) it is 0.4 points.
- The best-ranked update gains 0.004 +- 0.003 per unit (at most about 0.010), on the
  first step. The gradient probe's 16 members implied 0.039 +- 0.029, so the probe could
  not tell a gain of a tenth from a full one; this measures it. Of the docstring's two
  outcomes at lambda 10, +0.3 if first order held and about +0.03 at a tenth, it is +0.04.
- It matches the arm: the arm beat its control by 0.0022 per update over 120 updates.
  The shortfall is in each update, not in how 120 of them add up.
- The update's gain is about the cost of its random part (0.0038 per unit squared). A
  step of `lambda` nets `0.004 lambda - 0.0038 lambda^2`, at most 0.001 per update at
  `lambda` = 0.5: about +0.13 over 120 updates (at the 0.010 bound, +0.8), against RL's
  +1.87 after the same data (`../heldout-481-520/`). The arm's step, `lambda` = 1, nets
  about nothing; the arm ended 0.25 below its start.
- On the 768 prompts the 768 ranking was made on, the update gains +0.003 +- 0.007
  (+0.018 +- 0.016 at lambda 4): not more than on new prompts.
- The four updates gain -0.010 to +0.005. Random weights of the same 16 directions give
  first-order gains of about +-0.011 (the odd part's true sd 0.044 over sqrt(16)), so
  the two ranked updates are not distinguishable from a random weighting.

## Why the ranking promises more than an update delivers

The gradient probe also decoded the mirrored members (`-eps`), so each member's effect
splits into an odd part, `(f(+eps) - f(-eps)) / 2`, which changes sign with the noise,
and an even part, `(f(+eps) + f(-eps)) / 2 - f`, which does not. An update is a weighted
sum of the noise vectors, so to first order it inherits only the odd part.

| part of a member's effect | variance of the members' true effects | what a 768 ranking implies per unit |
|---|---|---|
| as ES sees it, `f(+eps) - f` | 0.0039 | +0.039 +- 0.029 |
| odd | 0.0020 | +0.014 +- 0.024 |
| even | 0.0035 | +0.025 +- 0.016 |

The parts covary negatively. By the point estimates, two thirds of what makes a member
score well is what a perturbation of this size does whichever its sign, which an average
of 16 perturbations does not reproduce; the odd part left is consistent with the +0.004
an update delivered. Sixteen members pin none of these down well, which is why the
direct measurement was needed.

At sigma 5e-4 a member moves the start's answers as far as RL's 120 steps do
(`../contrastive-check/`); its effect on each prompt is mostly not linear in the noise.
