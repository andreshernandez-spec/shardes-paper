# S4b, the probes: the prescriptions and predictions before any run (docs/end_to_end/10, "S4b, fresh settings for the verdict: design")

Three settings probed on 2026-10-16 with `es_vllm/probe.py --seed 1 --block 600`, every
phase, the members decoding the off-task block on each, and A probed again with the
population of seed 0: A, Qwen2.5-1.5B-Instruct on Countdown (`qwen1.5b-countdown`,
`-pop0`); C, Qwen2.5-3B-Instruct on the constructed single-constraint IFEval setting
(`qwen3b-quotation`, the Quotation sentence on the stream's IFEval base prompts, see `10`
and `probe.quotation_rows`); D, Qwen2.5-3B-Instruct on MATH alone (`qwen3b-math`). Blocks
of 192 train, 600 validation, 300 reserved test and 600 off-task prompts (A: GSM8K; C:
GSM8K and MATH; D: GSM8K and IFEval), digests in each `env.json`. One Secure H200 (RunPod
pod `28fthaaa86vkn1`, US-NC-1, $5.29/h), 07:30 to 12:37 UTC, 5.1 hours, about $27 by
uptime, at commit 274d65b. The records are those the pod wrote; nothing was regenerated.

Each setting's `prescription.json`, written by the steps phase and rewritten with the
off-task decodes' spread by the members-offtask phase, is the frozen prescription the
run session is held against: the method, the step (the curve rule's, or the fallback of
0.4 member lengths where the curve gives no positive, feasible step), shrinkage 0.1, the
responses interpolated at the applied step's first displacement (0.9 of the step), and
the four predictions with their resolution rules (`probe.prescribe`). The run session
takes the step and the anchor from it and stamps it into every run's record.

## The prescriptions

`python -m es_vllm.probe_table runs/s4b-probe --prescriptions`:

| setting | step (units of the update) | first displacement | fallback | odd response there | even response there | P1 learns | P2 collapses | spread ratio, 90% | P3: validation after 5 | P4: GRPO step to +0.10 |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen1.5b-countdown | 3.03 | 2.72 | no | +0.0126 ± 0.0037 | +0.0088 ± 0.0049 | learns | no call | 0.65, 0.383 to 2.3 | +0.071 | 23.0 |
| qwen1.5b-countdown-pop0 | 15.8 | 14.2 | no | +0.0315 ± 0.0042 | +0.0187 ± 0.0060 | learns | no call | 0.187, 0.208 to 2.2 | +0.168 | 17.7 |
| qwen3b-math | 3.13 | 2.81 | yes | +0.0100 ± 0.0061 | -0.0017 ± 0.0101 | undetermined | no call | 1.56, 0.199 to 1.18 | +0.032 | - |
| qwen3b-quotation | 6.85 | 6.17 | no | +0.0226 ± 0.0069 | -0.0128 ± 0.0093 | learns | no call | 0, 0 to 0.797 (unresolved) | +0.048 | 9.7 |

What the run session is held against (the design's gates: P1 and P4 gated, P2 and P3
reported):

- **A, 1.5B Countdown**, the prescription of population 1 as the design designates: a
  3.0-unit anchored step. P1 learns (the reserved block reaches +0.05 within 30
  updates); P3 +0.071 on validation after five anchored updates, band +0.018 to +0.071;
  P4 GRPO at +0.10 on the reserved block at step 23, band 11.5 to 46; P2 no call. Three
  arms: anchored at seeds 0 and 1, un-anchored at seed 0.
- **C, 3B Quotation**: a 6.85-unit anchored step. P1 learns; P3 +0.048, band +0.012 to
  +0.048; P4 step 9.7, band 4.8 to 19.4; P2 no call. The start already scores 0.90 on the
  train block and 0.93 on the validation block (half the train prompts have all sixteen
  samples alike, live share 0.51), so the task has about 0.07 of headroom and P1's
  +0.05 sits near the ceiling; the prediction stands as written.
- **D, 3B MATH**: the fallback step of 3.1 units (the curve gives no feasible step; the
  odd response is within noise at every length, reliability 0.05). P1 undetermined and
  P4 none (RL's slope 0.100 is unresolved), so D enters neither gate and contributes
  measured learning and retention; P3 +0.032 is reported.

The quadratic rule's calls on the same records (`probe_table runs/s4b-probe`): A
population 1 "fails: no odd response" (its `G` at 37 units is negative), population 0
"fails: rollouts" (ratio 27), D "fails: no odd response", C "works" (ratio 4.3, 90% 2.6
to 6.6) with "marginal" under the net target: the first setting on which the rule made a
call.

## A's two populations

`python -m es_vllm.probe_table runs/s4b-probe/qwen1.5b-countdown-pop0 --against
runs/s4b-probe/qwen1.5b-countdown` (population 1 first, population 0 second):

| setting | G, first | G, second | G within 2 SE | G, second at the first's length | within 2 SE there | C, first | C, second | C within 2 SE | RL slope, first | RL slope, second | RL slope within 2 SE | longest usable multiple, second | call, first | call, second |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen1.5b-countdown-pop0 | -8.47e-05 ± 7.7e-05 | 0.000653 ± 6.7e-05 | no | 0.000653 ± 6.7e-05 | no | -1.14e-05 ± 3.6e-06 | -8.25e-06 ± 4e-06 | yes | 0.24 ± 0.051 | 0.313 ± 0.043 | yes | 35 | undetermined: odd response unresolved | fails: rollouts |

The validation odd response by scanned multiple of the unit update:

| population | 2.7 to 3 | 7 | 15 to 18 | 35 to 37 |
|---|---|---|---|---|
| 1 (seed 1) | +0.012, +0.016 | +0.011 | +0.001 | −0.003 |
| 0 (seed 0) | +0.013 | +0.019 | +0.032, +0.031 | +0.023 |

The two populations' updates differ in kind, not in noise: population 1's helps only at
small steps and turns against the task by 37 units; population 0's helps up to 35 units.
The replication at a matched length that held on all six settings of the second probe
fails here, and the two prescriptions (3.0 and 15.8 units) differ by five. The second
probe's expectation 1 said what this means: on this setting the level-1 response is a
property of the population, and a rule reading one population prescribes for that draw.
The run session keeps population 1's prescription as designated; the seed-0 anchored
arm, which runs population 0's members at population 1's step, says by its pace against
P3 whether that step was too small for it. Random-direction curvature, RL's slope, the
live share and the start rewards agree across the two populations, as they should.

## The members off task

`python -m es_vllm.probe_table runs/s4b-probe --offtask`:

| setting | block | odd effect per direction, SD (noise removed) | over the train block's | G from the members | G from the steps scan | corr. with the train odd effect | 90% over pairs | even part at the member length | off-task change per unit gain, members / direct | at the target gain, members / direct |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen1.5b-countdown | offtask | 4.16e-03 | 0.65 | 0.000385 ± 0.0011 | 0.000744 ± 0.00029 | 0.0589 | -0.2 to 0.309 | 0.00693 ± 0.0077 | 0.108 / -8.79 | 0.0108 / - |
| qwen1.5b-countdown | validation | 5.87e-03 | 0.916 | 0.00358 ± 0.00066 | -8.47e-05 ± 7.7e-05 | 0.702 | 0.495 to 0.828 | 0.00832 ± 0.0041 | - | - |
| qwen1.5b-countdown-pop0 | offtask | 1.09e-03 | 0.187 | -0.00134 ± 0.0012 | 0.0005 ± 0.00027 | -0.219 | -0.477 to 0.0655 | 0.00672 ± 0.008 | -0.297 / 0.766 | -0.0297 / 0.0766 |
| qwen1.5b-countdown-pop0 | validation | 5.61e-03 | 0.962 | 0.00452 ± 0.00057 | 0.000653 ± 6.7e-05 | 0.879 | 0.819 to 0.921 | 0.00847 ± 0.0041 | - | - |
| qwen3b-math | offtask | 3.39e-03 | 1.56 | -0.000405 ± 0.00074 | -0.000341 ± 0.00024 | -0.0781 | -0.376 to 0.241 | -0.0103 ± 0.0074 | -0.326 / 1.14 | -0.0326 / - |
| qwen3b-math | validation | 4.40e-03 | 2.03 | 0.00124 ± 0.00089 | -0.000299 ± 0.00025 | 0.199 | -0.0502 to 0.408 | -0.00388 ± 0.0096 | - | - |
| qwen3b-quotation | offtask | 0 | 0 | 0.000958 ± 0.00088 | -0.000195 ± 0.00043 | 0.211 | -0.117 to 0.479 | 0.00406 ± 0.0078 | 0.312 / -0.0388 | 0.0312 / -3.88e-03 |
| qwen3b-quotation | validation | 8.71e-03 | 0.776 | 0.00307 ± 0.001 | 0.00502 ± 0.00053 | 0.348 | 0.0638 to 0.581 | -0.0128 ± 0.0065 | - | - |

P2 abstains on every setting: the off-task per-direction spreads are at or below the
sampling noise (A 0.004 and 0.001 against a noise floor of 0.007 and 0.008; C 0 against
0.006; D 0.003 against 0.006), so the ratios' intervals straddle the thresholds or the
spread is unresolved. The coverage of the exploratory predictor on this batch is zero of
three settings, which is a result about the predictor at this probe size: on the four
instruct checkpoints measured so far (3B Countdown in the second probe and these three)
the off-task block barely moves under a random rank-1 direction of a member's length,
and only the base model's did. The ranking generalizes from the train prompts to the
validation block on A (0.70 and 0.88, both populations) and C (0.35), and not resolvedly
on D (0.20, −0.05 to 0.41).

## The rule's table

`python -m es_vllm.probe_table runs/s4b-probe`, the first batch's columns:

| setting | live share | gradient sharedness | curvature per length^2 | RL first-step gain | reliability | cross-prompt corr | G | C | G^2/4|C| | ES/RL rollouts | ratio 90% | call | off-task at target | damaged first (scan / update) | net: step | net: updates | net: ES/RL | net: call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen1.5b/countdown | 1 | 0.0197 | -3.81e-06 | 4.34e-03 | 0.519 | 5.62e-03 | -8.47e-05 | -3.65e-05 | - | - | 692 to 861,028 | fails: no odd response | - | gsm8k / gsm8k | - | - | - | undetermined: odd response unresolved |
| qwen1.5b/countdown (population 0) | 1 | 0.0197 | -5.09e-06 | 5.66e-03 | 0.418 | 3.77e-03 | 6.53e-04 | -2.56e-05 | 4.17e-03 | 27.2 | 13.5 to 46.6 | fails: rollouts | 0.0766 | gsm8k / gsm8k | 2.88 | 55.1 | 49.9 | fails: rollouts |
| qwen3b/math | 0.552 | 4.71e-03 | -4.83e-06 | 2.62e-03 | 0.0524 | 2.47e-04 | -2.99e-04 | -7.80e-05 | - | - | 102 to 18,487 | fails: no odd response | - | MATH / MATH | - | - | - | undetermined: odd response unresolved |
| qwen3b/quotation | 0.505 | 4.23e-03 | -6.22e-06 | 0.0103 | 0.343 | 2.72e-03 | 5.02e-03 | -1.30e-04 | 0.0486 | 4.25 | 2.56 to 6.56 | works | -3.88e-03 | ifeval / ifeval | 10.8 | 3.34 | 5.52 | marginal |

Start rewards (normalized): A train 0.081, validation 0.075, GSM8K 0.75; C train 0.901,
validation 0.930, off-task 0.737; D train 0.63, validation 0.677, off-task 0.642. Phase
times: the off-task decodes 1,788 s (A, each population), 2,972 s (C) and 3,903 s (D);
the members 257 to 853 s.

## Limitations

One population per prescription (two on A, which disagree), 600 prompts, one update's
responses; the predictions are the probe's at this size and the run session tests them.
C is a constructed task near its ceiling. The off-task spread predictor abstained
everywhere; its first prospective test on a setting where it makes a call is still to
come.
