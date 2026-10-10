# ES's updates against GRPO's exact gradient, 2026-10-10

Andres's question: ES is meant to estimate the gradient; is the estimate working, or is
something reducing it? `es_vllm/gradient_check.py`: phases at 591341d, the small-step scan
and the analysis at a42605b, the prediction in the docstring committed at 4fa72c9 before
the run. One Secure Cloud H200 (US-CO-1, driver 580.178.04, $5.29/h, 13:16 to 16:28 UTC,
including a first attempt whose smoke stage failed on a zero gradient), about $17.
Records `samples.json` (the batch), `gradient.json`, `projections.json`, `scan.jsonl`,
`scan-meta.json`; logs `pod-log-*.txt.gz`, status `pod-status.txt`. Numbers:

    python -m es_vllm.gradient_check --analyze runs/gradient-check

Sign convention: the probe's `g` is the gradient of GRPO's loss, so RL descends it, and
every number below is given toward reward (along `-g`).

## The batch and the gradient

The RL run's batch for the 192 iteration-0 prompts (RL steps 1 to 4): 16 answers each at
temperature 1.0 (mean length 355, mean reward 4.83), the run's verifiers, GRPO's
advantages `(score - group mean) / (group sd + 1e-8)`. 91 of the 192 groups have unequal
rewards; 43 are all right and 58 all wrong, so 53% of the prompts give RL's first step no
gradient at all, as ES's dead prompts give its update none (`../lowrank-check/`).

The gradient at the start, as open-instruct 3f37c29 forms it: the token mean of
`-A_i log p(y_t)` over each micro-batch of 2 (random pairing, seed 0), averaged over the
1,536 micro-batches, f32 parameters under bf16 autocast, 2.39M tokens, 639 s. The ratio
is 1 and the KL term's gradient is 0 at the start, so this is the run's whole first-step
gradient. |g| = 0.549, of which the 224 attention and MLP matrices ES perturbs carry 96%
of the squared norm (down_proj 40%, v_proj 27%, o_proj 12%, up_proj 10%, gate_proj 5%;
embeddings 2.4%, lm_head 1.5%, norms 0.1%).

The gradient is concentrated. Of the unit matrix gradient's 7.0 billion elements, 993
above 0.01 carry 24% of its squared norm, 27,678 above 0.001 carry 43%, and 3.2 million
above 1e-4 carry 56%; the largest is 0.14. A Gaussian direction of this dimension has rms
1.2e-5 and nothing above 1e-4.

## ES's updates point along RL's gradient, at 12% of the maximum

`rho_hat = u . (-g_hat) / alpha`: for an update `u = (alpha / N) sum_m z_m E_m`, `E[u] =
alpha rho g_hat_ES`, so this is the estimator's correlation with its own gradient times
the cosine between that gradient and RL's. With a ranking that carries nothing it is
noise of sd `1 / sqrt(N)`; the control ranking is that case.

| update | members | rho_hat toward reward (noise sd) |
|---|---|---|
| `../lowrank-check/`, N = 1,024 | 1,024 | +0.120 (0.031) |
| `../lowrank-check/`, N = 512 | 512 | +0.123 (0.044) |
| `../lowrank-run/` iteration 0 (another decode of the same members) | 512 | +0.116 (0.044) |
| `../lowrank-check/`, N = 128 | 128 | +0.071 (0.088) |
| `../step-check/` dense: arm, probe768, contrast, control | 16 | +0.21, +0.22, +0.05, +0.36 (0.25 each; the four share their 16 directions) |
| `../lowrank-run/` iterations 1 to 7, on this gradient | 512 | +0.03, -0.02, +0.08, -0.09, -0.01, +0.08, +0.04 (0.044) |

- The three independent measurements at N >= 512 agree: ES's first update has a
  component along RL's gradient of 0.12 of the most a zeroth-order estimate can have, 3.9
  standard errors from zero and toward reward. The prediction (0.1 to 0.3) holds.
- Later updates, made at later weights, have no measurable component along the start's
  gradient (mean +0.02 +- 0.02). Nor does RL's own displacement to step 120: its cosine
  with this gradient is 0.019, what 120 Adam steps of length 0.045 (lr 5e-7 times
  sqrt(d)) that are nearly orthogonal to one another give (sqrt(120) x 0.045 = 0.49, and
  the displacement is 0.476). A minibatch gradient at the start is a small part of where
  RL goes.
- ES's displacement after eight updates (83 long) has cosine 1e-5 with RL's.

## Along RL's gradient the reward is steep for a short way

The held-out reward (3,840 prompts of RL steps 121 to 160 and 481 to 520; start 5.456)
at `theta_0 -+ lambda g_hat`, `g_hat` the unit gradient restricted to the matrices, and at
`theta_0 -+ lambda s_hat`, the unit sign vector (the direction of Adam's first step). The
engine rounds to bf16, so the effective length is what reached the model. Odd part
`(f(-lambda) - f(+lambda)) / 2`, the first-order gain along RL's direction; even part
`(f(-lambda) + f(+lambda)) / 2 - f(0)`.

| direction | nominal | effective length | toward reward | away | odd (SE) | even (SE) | slope toward reward, per unit length |
|---|---|---|---|---|---|---|---|
| gradient | 0.03 | 0.022 | 5.596 | 5.271 | +0.163 (0.029) | -0.022 (0.045) | 7.3 (1.3) |
| gradient | 0.1 | 0.079 | 2.151 | 0.474 | | -4.14 | broken |
| gradient | 0.3 to 8 | 0.26 to 8.05 | 0.04 to 0.00 | 0.01 to 0.00 | | | dead |
| sign | 1 | 0.598 | 5.688 | 4.784 | +0.452 (0.034) | -0.220 (0.049) | 0.76 (0.06) |
| sign | 4 | 4.17 | 2.500 | 0.945 | +0.78 | -3.73 | broken |

- Along RL's exact gradient the reward rises 7.3 +- 1.3 points per unit length at the
  smallest length the rounding allows (0.022), and the model is broken by length 0.08: a
  step along a direction this concentrated moves a few weights by a lot. The even part at
  0.022 is already 14% of the odd part, so the slope at zero is somewhat higher.
- Along Adam's sign direction, every weight moves alike: the slope is a tenth, 0.76, and
  it holds to length 0.6 and beyond. RL's actual step (length 0.045) would gain about
  0.034 to first order, and its first 40 steps gained 0.44 on these prompts, 0.011 per
  step.
- Both gains land where RL's do: at 0.022 along the gradient, IFEval +0.25, MATH +0.04,
  GSM8K +0.03 (234 prompts up, 180 down); along the sign direction, IFEval +0.45.
- The first scan (591341d) used lambda 1 to 8, sized for a Gaussian direction's rounding;
  every point broke the model. The small lambdas were added at a42605b.

## What this says about ES

ES's useful component per unit update is at most `alpha` = 5e-4 long (`E[u] = alpha rho
g_hat_ES`, rho <= 1). It delivered 0.0047 +- 0.0004 held-out points per unit at N = 1,024
(`../lowrank-check/`), so the slope along ES's expected direction is at least 9.4 per
unit length.

- A perfect zeroth-order estimate of RL's gradient, `rho` = 1 along `-g_hat`, would gain
  `alpha` x 7.3 = 0.0037 +- 0.0007 per unit, 0.8 +- 0.15 of what ES's update already
  gains. The estimator is not losing anything to its ranking that a better fitness could
  recover: ES's expected direction is at least as steep as RL's gradient direction, and it
  is spread over all weights like Adam's step rather than concentrated like the raw
  gradient, so it holds up at the lengths ES uses (the odd part was linear to length 112
  in `../lowrank-check/`).
- What separates the two is only the length of the useful step. RL moves 0.045 along its
  direction every step, all of it useful. ES moves at most 5e-4 along its direction per
  update, beside a random part of 1.4 at 1,024 members, whose cost falls as 1/N. The
  remaining lever is N, which is to say rollouts; the ceiling of a better fitness is
  about where ES is.
- `rho_hat` = 0.12 bounds the cosine between ES's expected direction and RL's gradient
  from below; `rho` itself, how well the ranking tracks ES's own gradient, is not
  separated from that cosine here. Either way the update is a gradient estimate that
  works, toward RL's gradient, with the variance zeroth-order estimation has.

## What this does not show

- One batch, one gradient, at the start. RL's step also carries Adam's history and
  clipping after the first step.
- The slopes are at the smallest lengths bf16 allows: 0.022 along the gradient, where the
  curvature is already visible, and 0.6 along the sign direction.
- The gradient is of the loss as the run forms it (token mean per micro-batch of 2, random
  pairing); a sequence-mean normalization was not computed.
