# The second probe: another population, 600-prompt blocks, the members off task (docs/end_to_end/10, "S4a, second half")

The six settings of `runs/probe/` probed again on 2026-10-15 with `es_vllm/probe.py
--seed 1 --block 600`: the member population of seed 1 (the scan's four random directions
are its first four pairs), 600 validation and 600 off-task prompts whose first 300 are
the first probe's blocks (`env.json` carries the prefix digests, equal to the first
probe's block digests), the same 192 train and 300 test prompts, and the scans extended
until the even response is resolved (commit c945036, which the first batch predates). On
the two Countdown settings the members also decoded the validation and off-task blocks
(`--phase members-offtask`). One Secure H200 (RunPod pod `4466roagwld1kw`, the S4a pod,
US-NC-1, $5.29/h), 04:44 to 10:21 UTC, 5.6 hours, about $30 by uptime. The pod ran commit
a1c1ce6; `prediction.json` was regenerated locally at 5363b3c from the same raw records
(the exploratory curve rule, both RL slope readings and the noise-corrected spreads were
added; every key the pod wrote is unchanged).

Files per setting as in `runs/probe/`, plus `members-offtask.jsonl` (the decodes, a chunk
of members at a time) and `members-offtask.json` (the assembled record: every member's
reward on every validation and off-task prompt, the start's, and the answer lengths) on
the Countdown settings. Phase times: members 325 to 758 s; the off-task decodes 2,968 s
(instruct) and 4,868 s (base, whose GSM8K answers run to about 1,000 tokens).

## Against the first probe

`python -m es_vllm.probe_table runs/probe2 --against runs/probe`. `G` and `C` are the
validation block's odd and even responses per unit update and per unit squared at the
longest length usable on every block (the rule's reading); "at the first's length" reads
the second probe at the length the first probe read; RL's slope is along the sign of the
summed gradient at the shortest usable length (the rule's reading). "within 2 SE" is of
the difference.

| setting | G, first | G, second | G within 2 SE | G, second at the first's length | within 2 SE there | C, first | C, second | C within 2 SE | RL slope, first | RL slope, second | RL slope within 2 SE | longest usable multiple, second | call, first | call, second |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen0.5b-gsm8k | 0.00143 ± 0.0009 | -8.36e-05 ± 0.00056 | yes | -8.36e-05 ± 0.00056 | yes | -0.000279 ± 8.1e-05 | -0.000252 ± 4e-05 | yes | 0.525 ± 0.33 | 0.888 ± 0.23 | yes | 19.9 | undetermined: odd response unresolved | undetermined: odd response unresolved |
| qwen1.5b-gsm8k | 0.00126 ± 0.0006 | 0.000299 ± 0.00023 | yes | -0.000112 ± 0.00039 | yes | -1.49e-05 ± 4.2e-05 | -4.87e-05 ± 7.8e-06 | yes | 0.207 ± 0.15 | 0.0775 ± 0.11 | yes | 44.6 | undetermined: cost unresolved | undetermined: odd response unresolved |
| qwen3b-base-countdown | 0.00261 ± 0.00086 | 0.00311 ± 0.00036 | yes | 0.00418 ± 0.00073 | yes | -0.00033 ± 0.00021 | 5.01e-05 ± 2.9e-05 | yes | 0.186 ± 0.045 | 0.181 ± 0.03 | yes | 17.2 | undetermined: cost unresolved | undetermined: cost unresolved |
| qwen3b-countdown | 0.00227 ± 0.00048 | 0.0011 ± 0.00017 | no | 0.00228 ± 0.00036 | yes | -4.7e-05 ± 4.3e-05 | -1.91e-05 ± 9.6e-06 | yes | 0.231 ± 0.088 | 0.103 ± 0.057 | yes | 34.6 | undetermined: cost unresolved | undetermined: cost unresolved |
| qwen3b-gsm8k | 0 ± 0.00056 | -0.000578 ± 0.00025 | yes | -0.000311 ± 0.00035 | yes | -3.07e-05 ± 4.2e-05 | -4.86e-05 ± 9.1e-06 | yes | 0.0532 ± 0.062 | 0.126 ± 0.045 | yes | 37.5 | undetermined: odd response unresolved | fails: no odd response |
| qwen7b-gsm8k | 0.00167 ± 0.00046 | 0.000148 ± 0.00017 | no | 0.000806 ± 0.00028 | yes | 1.75e-05 ± 4.9e-05 | -4.85e-06 ± 6.7e-06 | yes | 0.252 ± 0.047 | 0.286 ± 0.035 | yes | 39.3 | undetermined: cost unresolved | undetermined: odd response unresolved |

The response curves behind the table, the validation block's odd and even responses at
each scanned multiple of the unit update (standard errors from the 600 prompts), and the
off-task block's:

| setting | multiple | validation odd | validation even | off-task odd | off-task even |
|---|---|---|---|---|---|
| 3B instruct Countdown | 2.8 | +0.019 ± 0.007 | +0.006 ± 0.010 | +0.005 | −0.008 |
| | 4.5 | +0.016 ± 0.006 | −0.000 ± 0.012 | −0.002 | −0.012 |
| | 6.9 | +0.023 ± 0.007 | −0.009 ± 0.012 | −0.001 | −0.003 |
| | 17.3 | +0.040 ± 0.006 | −0.015 ± 0.011 | +0.004 | −0.003 |
| | 34.6 | +0.038 ± 0.006 | −0.023 ± 0.012 | +0.008 | −0.053 |
| 3B base Countdown | 2.75 | +0.012 ± 0.004 | −0.011 ± 0.007 | −0.028 | +0.025 |
| | 6.9 | +0.029 ± 0.005 | −0.003 ± 0.008 | −0.004 | +0.036 |
| | 17.2 | +0.054 ± 0.006 | +0.015 ± 0.009 | +0.018 | +0.120 |
| 0.5B GSM8K | 3.2 | +0.005 ± 0.009 | +0.010 ± 0.012 | +0.005 | −0.012 |
| | 8.0 | −0.003 ± 0.010 | −0.019 ± 0.015 | −0.006 | −0.018 |
| | 19.9 | −0.002 ± 0.011 | −0.100 ± 0.016 | −0.014 | −0.076 |
| 1.5B GSM8K | 0.9 to 8.9 | 0.000 to +0.003 (± 0.006 to 0.008) | −0.005 to −0.007 | +0.008 to −0.007 | +0.013 to −0.007 |
| | 22.3 | −0.003 ± 0.009 | −0.023 ± 0.014 | −0.012 | −0.017 |
| | 44.6 | +0.013 ± 0.010 | −0.097 ± 0.015 | −0.038 | −0.103 |
| 3B GSM8K | 3.0 to 18.8 | −0.002 to −0.006 (± 0.005 to 0.007) | −0.002 to −0.001 | +0.012 to +0.008 | +0.007 to −0.021 |
| | 37.5 | −0.022 ± 0.009 | −0.068 ± 0.013 | +0.022 | −0.107 |
| 7B GSM8K | 0.8 to 7.9 | +0.001 to +0.008 (± 0.003 to 0.005) | +0.004 to +0.001 | +0.004 to −0.006 | +0.001 to +0.001 |
| | 19.7 | +0.016 ± 0.005 | +0.003 ± 0.007 | −0.006 | −0.016 |
| | 39.3 | +0.006 ± 0.007 | −0.008 ± 0.010 | +0.007 | −0.057 |

What replicates and what does not:

- **The level-1 response replicates at a matched length on every setting** (six of six
  within two standard errors at the first probe's reading length; four of six where the
  rule reads it, the two misses being the settings whose second scan ran to 35 and 39
  units where the odd response has saturated: 3B instruct Countdown +0.040 at 17 and
  +0.038 at 35, 7B GSM8K +0.016 at 20 and +0.006 at 39). The odd response is concave in
  the step on both Countdown settings from the smallest length scanned, so a per-unit `G`
  depends on where it is read (`10`, corrections item 10).
- **The GSM8K settings have no odd response at 64 members on 600 prompts.** 0.5B, 1.5B
  and 7B: the ranking's reliability is 0 on this population and the validation odd
  response is within noise at every length (0.5B: +0.005, −0.003, −0.002; 7B's +0.016 ±
  0.005 at 20 units is the one reading above two standard errors, back to +0.006 at
  39). 3B GSM8K's odd response is negative (−0.022 ± 0.009 at 37.5 units): the update
  hurts the task's own validation prompts, and the rule calls it "fails: no odd
  response". The first probe's odd responses on 0.5B (0.00143 ± 0.0009), 1.5B (0.00126
  ± 0.0006) and 7B (0.00167 ± 0.00046 per unit) were draws at the high end; the 0.5B
  training runs' absence of any held-out gain (`runs/probe-run/`) is what this
  population shows at one update.
- **The extended scans resolve the even response on GSM8K and not on Countdown.** The
  random cost per unit squared is now 4 to 6 standard errors from zero on 0.5B, 1.5B and
  3B GSM8K (−0.10 of the range at 20, 45 and 37.5 units), while on both Countdown
  settings one update's cost on the validation block stays within two standard errors of
  zero at 600 prompts (instruct −0.023 ± 0.012 at 35 units; base +0.015 ± 0.009 at 17,
  a gain). The rule's call on the Countdown settings is therefore still "cost
  unresolved"; the cost that limits those runs accumulates (`runs/s4a/`).
- **RL's slope replicates within two standard errors on every setting**, with the
  shortest-length reading noisy where the response is small (3B instruct Countdown 0.231
  ± 0.088 on 300 prompts and 0.103 ± 0.057 on 600 with the same first 300; the longer
  length gives 0.266 ± 0.029 and 0.238 ± 0.021). Both readings are in the records; the
  rule keeps the shortest.
- **The level-0 test fails again**: the gradient sharedness orders the six settings as
  the members' cross-prompt correlation does at Spearman 0.43 (first batch 0.54).

## The rule's table

`python -m es_vllm.probe_table runs/probe2`, the first batch's columns:

| setting | live share | gradient sharedness | curvature per length^2 | RL first-step gain | reliability | cross-prompt corr | G | C | G^2/4|C| | ES/RL rollouts | ratio 90% | call | off-task at target | damaged first (scan / update) | net: step | net: updates | net: ES/RL | net: call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen0.5b/gsm8k | 0.812 | 3.67e-03 | -1.58e-04 | 8.40e-03 | 0 | -3.35e-04 | -8.36e-05 | -2.21e-04 | - | - | 123 to 40,128 | fails: no odd response | - | MATH / MATH | - | - | - | undetermined: odd response unresolved |
| qwen1.5b/gsm8k | 0.766 | 2.64e-03 | -1.14e-05 | 1.40e-03 | 0 | -1.14e-03 | 2.99e-04 | -5.03e-05 | 4.44e-04 | 63.1 | 7.58 to 1,982 | fails: rollouts | -0.287 | gsm8k / MATH | 0.891 | 439 | 98.4 | undetermined: odd response unresolved |
| qwen3b-base/countdown | 1 | 0.0196 | 9.11e-07 | 4.77e-03 | 0.366 | 3.00e-03 | 3.11e-03 | 2.28e-04 | - | - | - | undetermined: no resolved cost | 0.0343 | countdown / countdown | 17.2 | 1.46 | 1.12 | undetermined: cost unresolved |
| qwen3b/countdown | 1 | 8.64e-03 | -2.28e-06 | 2.72e-03 | 0.139 | 8.79e-04 | 1.10e-03 | -3.18e-05 | 9.57e-03 | 5.68 | 1.26 to 15.3 | marginal | 0.0218 | gsm8k / gsm8k | 4.5 | 21.8 | 9.5 | undetermined: cost unresolved |
| qwen3b/gsm8k | 0.453 | 1.59e-03 | -2.09e-06 | 3.32e-03 | 0.189 | 1.20e-03 | -5.78e-04 | -6.22e-05 | - | - | 312 to 41,842 | fails: no odd response | - | MATH / MATH | - | - | - | fails: no odd response |
| qwen7b/gsm8k | 0.25 | 7.15e-03 | -7.71e-07 | 0.0116 | 0 | -3.06e-04 | 1.48e-04 | -2.08e-05 | 2.65e-04 | 872 | 88.5 to 13,069 | fails: rollouts | 0.114 | MATH / MATH | 0.786 | 880 | 1,627 | undetermined: odd response unresolved |

The exploratory curve rule (`rule_curve` in each record, `10` item 10): 3B base
Countdown works (a 17-unit step, 1.5 updates, ES/RL rollouts 1.1, from a first-update net
gain of +0.068 that the actual run at step 3 did not sustain: it gained about 0.003 per
update); 3B instruct Countdown undetermined (17 units, 4 updates, ratio 1.8); 0.5B and 7B
GSM8K undetermined (3.2 and 9.8 units, 7 and 10 updates, ratios 9 and 19); 1.5B and 3B
GSM8K fail with no net gain in the scanned range. It calls one setting, on a first-update
response the run contradicts in size.

## The members off task

`python -m es_vllm.probe_table runs/probe2 --offtask`: the 64 members at sigma 5e-4 and
the 32 at 1e-3 decoding the validation and off-task blocks. Per mirrored pair, the odd
part of the pair's change on a block is its direction's slope there; the update's
first-order change on a block is the sum of the pairs' odd effects weighted by the
update's coefficients ("G from the members", the quantity the steps scan measures
directly, here at the members' own length); the spread across pairs of the odd effect,
with the sampling part from a bootstrap over prompts removed, is how much a random rank-1
direction of a member's length moves that block.

| setting | block | odd effect per direction, SD (noise removed) | over the train block's | G from the members | G from the steps scan | corr. with the train odd effect | 90% over pairs | even part at the member length | off-task change per unit gain, members / direct | at the target gain, members / direct |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3b-base-countdown | offtask | 0.0913 | 11.8 | -0.000187 ± 0.0014 | 0.00107 ± 0.00074 | -0.051 | -0.294 to 0.22 | 0.0126 ± 0.013 | -0.0452 / 0.343 | -4.52e-03 / 0.0343 |
| qwen3b-base-countdown | validation | 7.41e-03 | 0.96 | 0.00415 ± 0.0006 | 0.00311 ± 0.00036 | 0.592 | 0.346 to 0.758 | -0.00846 ± 0.0059 | - | - |
| qwen3b-countdown | offtask | 2.75e-03 | 0.304 | 0.000881 ± 0.00084 | 0.000241 ± 0.00024 | 0.193 | -0.0733 to 0.429 | -0.0057 ± 0.0069 | 0.199 / 0.218 | 0.0199 / 0.0218 |
| qwen3b-countdown | validation | 0.0105 | 1.17 | 0.00442 ± 0.0011 | 0.0011 ± 0.00017 | 0.467 | 0.174 to 0.678 | 0.000894 ± 0.0092 | - | - |

- **The ranking generalizes from the train prompts to the validation block** on both
  settings (correlation 0.47 and 0.59 across pairs, resolved), and the members' estimate
  of the update's validation slope (0.0044 and 0.0042 per unit) agrees with the steps
  scan at the small-step end of its curve (0.0067 ± 0.0026 and 0.0036 ± 0.0014 at 2.8
  and 4.5 units on the instruct model; 0.0045 ± 0.0014 at 2.75 on the base model), as
  it should: the members sit at their own length, and the scan's per-unit value falls
  with the length.
- **Neither population's ranking carries GSM8K damage.** The correlation across pairs
  between the train and GSM8K odd effects is 0.19 (90% −0.07 to 0.43) on the instruct
  model and −0.05 (−0.29 to 0.22) on the base model; the update's first-order GSM8K
  change is zero within error on both. The preregistration expected a resolved negative
  correlation on the base model from its first probe's −0.017 per unit; that number was
  one population's draw (the seed-0 update's direction lowered GSM8K by 0.07, 0.18 and
  0.29 at 2.7, 6.8 and 17 units, resolved; the seed-1 update's by −0.03, 0.00 and
  +0.02), a property of that update's random part, not of the ranking.
- **Where the base model is fragile.** A random rank-1 direction of a member's length
  moves the base model's GSM8K by 0.091 of the range (twelve times what it moves the
  train block) and the instruct model's by 0.003 (a third of its train block's). That
  is the random walk's lever on each checkpoint, measured at the start with nothing
  ranked, and it matches the runs: the base model lost GSM8K even at step 3.0
  (`runs/probe-run/`), the instruct model did not (`runs/s4a/`). It is a post hoc
  reading of two settings and is stated in `10` as a candidate predictor for S4b to test
  prospectively, not as a result.

## Against the preregistration (`10`, "S4a, second half")

1. Held at a matched length on six of six settings, and the members' decomposition
   agrees with the scan at the small-step end; the reading at the longest usable length
   failed on two settings for the reason item 10 records.
2. Failed: the extended scans resolved the cost on the GSM8K settings, where the odd
   response is now absent, and not on the Countdown settings, where the cost at one
   update is too small to resolve at 600 prompts. No setting moved from undetermined to
   a call except 3B GSM8K (fails, from a negative odd response); 0.5B GSM8K's expected
   "fails on rollouts" is "no odd response" instead, the same verdict by another route.
3. Held for the instruct model (unresolved), failed for the base model (unresolved, not
   negative). The diagnostic's correlation is not the predictor; the per-direction
   spread may be.

The diagnostic is validated only where it agrees with S4a's gate: S4a said the random
walk, and the diagnostic finds no directional damage on the instruct model and a large
random-direction sensitivity on the base model, which is consistent, and is one
checkpoint each.

## Limitations

One population per probe (two in all), 600 prompts, one update's responses. The
Countdown costs are unresolved at this size; the GSM8K odd responses are unresolved or
absent. The off-task decodes cover the two Countdown settings only. The curve rule and
the per-direction spread were written after these records were read and are exploratory.
