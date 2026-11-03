# The a-priori probe, first batch (docs/end_to_end/10)

Six settings probed on 2026-10-12 with `es_vllm/probe.py`: level 0 (nothing ranked) and
level 1 (one ES iteration of 64 mirrored rank-1 members) on one model and task each,
and the rule's prediction for a training run, written before any such run. One Secure
H200 (RunPod pod `sk35qafr1gcr7x`, US-NC-1, $5.29/h), 14:00 to 17:02 UTC, about $16
including three relaunches. The pod ran commit 423f353 for every raw record except
`qwen0.5b-gsm8k`'s `sample.json` and `scan.jsonl`, written at 86ea0f9 (the same code
for those phases; `env.json` stamps it). Each `prediction.json` is regenerated from the
raw records by the analysis at the commit this README is committed in; the file the pod
wrote is kept beside it as `prediction-on-pod.json` (see "Analysis changes").

| setting | model | task | train / validation / test / off-task prompts |
|---|---|---|---|
| qwen0.5b/gsm8k | Qwen2.5-0.5B-Instruct | GSM8K from the Tulu stream | 192 / 300 / 300 / 150 MATH + 150 IFEval |
| qwen1.5b/gsm8k | Qwen2.5-1.5B-Instruct | same | same |
| qwen3b/gsm8k | Qwen2.5-3B-Instruct | same | same |
| qwen7b/gsm8k | Qwen2.5-7B-Instruct | same | same |
| qwen3b-base/countdown | Qwen2.5-3B (base) | Countdown, es-at-scale's prompts | 192 / 300 / 300 / 300 GSM8K |
| qwen3b/countdown | Qwen2.5-3B-Instruct | same | same |

The test block is reserved: no phase decodes it; its digest is in `env.json`. Models are
pinned in `releases.py` (`pins/releases-2026-10-12.json`).

## Files per setting

`sample.json` (16 answers per train prompt at temperature 1.0 with their rewards and
GRPO advantages; the start's greedy rewards on the train, validation and off-task blocks),
`gradient.json` (per-prompt gradient norms, the summed gradient's norm, sharedness, Yin's
diversity), `scan.jsonl` and `scan-meta.json` (four random rank-1 directions at three
lengths, both signs, on the validation and off-task blocks), `sign.jsonl` and
`sign-meta.json` (the sign of the summed gradient at two lengths, both signs, with
effective lengths after bf16 rounding), `members.json` (64 members at sigma 5e-4 and 32
at 1e-3 on the train block), `steps.jsonl` and `steps-meta.json` (the 64-member update
at three lengths, both signs), `prediction.json`.

## The table

`python -m es_vllm.probe_table runs/probe`. Rewards normalized to [0, 1] by each row's
kind (Countdown 1.1, the Tulu verifiers 10). `G`: the update's odd response per unit
update on the validation block; `C`: its even response per unit squared over both
blocks; `G^2/4|C|`: the fitted net per update at 64 members; ES/RL rollouts: for a tenth
of the reward range at 2% damage, RL at the Tulu run's step (lr 5e-7, 768 rollouts per
step); the 90% interval from 1,000 resamples of the scanned prompts within blocks;
off-task at target: the update's directional effect on the off-task block scaled to the
target gain.

| setting | live share | gradient sharedness | curvature per length^2 | RL first-step gain | reliability | cross-prompt corr | G | C | G^2/4|C| | ES/RL rollouts | ratio 90% | call | off-task at target | damaged first (scan / update) | net: step | net: updates | net: ES/RL | net: call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen0.5b/gsm8k | 0.812 | 3.67e-03 | -1.51e-04 | 4.96e-03 | 0 | -6.98e-04 | 1.43e-03 | -2.60e-04 | 1.97e-03 | 50.3 | 7.11 to 848 | fails: rollouts | 0.0933 | MATH / MATH | 0.959 | 89.5 | 71 | undetermined: odd response unresolved |
| qwen1.5b/gsm8k | 0.766 | 2.64e-03 | -2.85e-06 | 3.74e-03 | 0.121 | 7.16e-04 | 1.26e-03 | -1.30e-05 | 0.0305 | 2.45 | 0.372 to 76 | works | 0.0437 | MATH / MATH | 17.8 | 5.65 | 3.38 | undetermined: cost unresolved |
| qwen3b-base/countdown | 1 | 0.0196 | -1.23e-05 | 4.91e-03 | 0.38 | 3.17e-03 | 1.53e-03 | -1.02e-04 | 5.75e-03 | 17.1 | 3.72 to 49.5 | marginal | -1.11 | gsm8k / gsm8k | 3.95 | 19.4 | 15.2 | undetermined: cost unresolved |
| qwen3b/countdown | 1 | 8.64e-03 | -1.90e-06 | 6.09e-03 | 0.274 | 1.99e-03 | 2.27e-03 | -2.97e-05 | 0.0432 | 2.82 | 0.284 to 12.4 | works | -0.011 | countdown / countdown | 20.1 | 3.77 | 3.67 | undetermined: cost unresolved |
| qwen3b/gsm8k | 0.453 | 1.59e-03 | -3.36e-07 | 1.40e-03 | 0.0172 | 1.00e-04 | 0 | -2.56e-05 | - | - | 1.22 to 927 | fails: no odd response | - | gsm8k / ifeval | - | - | - | undetermined: odd response unresolved |
| qwen7b/gsm8k | 0.25 | 7.14e-03 | -2.86e-07 | 0.0102 | 0 | -2.51e-04 | 1.67e-03 | -4.37e-05 | 0.0161 | 12.7 | 1.98 to 52.2 | marginal | 0.0353 | MATH / MATH | 3.3 | 17.5 | 28.5 | undetermined: cost unresolved |


The four `net:` columns are the corrected analysis of 2026-10-13 (`rule_net`, docs/end_to_end/10,
"Corrections"): the step searched over the scanned range for a net gain of 0.10 on the
validation block with each retained block's random loss within 0.02, and a call that is
"undetermined" when the odd or even response is within two standard errors of zero. Under
it every setting of this batch is undetermined at 300 prompts; the preregistered columns
to its left are what the training runs were prescribed from and are unchanged (their
bootstrap now resamples the same prompts for ES and the RL proxy). The RL side of every
ratio is a proxy (the Tulu run's step and costs), so the ratios are provisional forecasts.

Standard errors and the by-task and by-block responses are in each `prediction.json`.
`G` is 1.6 (0.5B), 2.1 (1.5B), 5.1 (3B base), 4.7 (3B Countdown), 0.0 (3B GSM8K) and
3.6 (7B) standard errors from zero; `C` is resolved at 3 or more standard errors only
on the 0.5B record (1.9 on the 3B base, under 1.3 elsewhere), which is what the wide
intervals say. `python -m es_vllm.probe_table runs/probe --json` adds the Spearman
correlation between the gradient sharedness and the cross-prompt correlation of member
effects across the settings.

## What the records say

- **The Countdown pair.** The instruct model is the batch's clearest "works": the
  largest `G` (0.0023 +- 0.0005), a cost 3.4 times smaller than the base model's, and the
  largest fitted net per update (0.043). The base model gains on Countdown at the same
  rate but its update's direction loses heavily on GSM8K (odd response -0.29 at the
  length read, -1.1 of the reward range when scaled to the target gain on Countdown;
  RL's sign direction: -0.03), so its call is "marginal" on the task with the off-task
  flag set. This is Palleti and Singh's collapse located: not an absent signal (its
  ranking is the most reliable of the batch, 0.38) but a found direction that is
  destructive elsewhere.
- **The GSM8K size axis.** The live share falls with size (0.81, 0.77, 0.45, 0.25) as
  the start's pass rate rises (0.57, 0.74, 0.83, 0.86); the curvature along random
  rank-1 directions falls by three orders of magnitude (1.5e-4 to 2.9e-7 per length
  squared), so the random part's cost per unit update falls from 2.6e-4 to 4.4e-5 while
  `G` stays between 0.0013 and 0.0017 except at 3B: larger models are cheaper to
  perturb, which is the published
  "fewer members needed at larger sizes" in the rule's terms. The 0.5B model fails on
  rollouts (ratio 50, Cho and Han's "no progress at 0.5B"); 1.5B is "works" on the point
  estimate with an unresolved cost; 3B has no odd response at all (0.0 +- 0.0006, the
  ranking's reliability 0.017, the members lose 0.2 points on average); 7B is "marginal"
  (ratio 13) with a ranking of reliability 0 yet an update response of 3.6 standard
  errors, as on Tulu.
- **RL's side.** The slope along the sign direction on the task is 0.19 to 0.52 per unit
  effective length except on the saturated 3B GSM8K (0.05); RL's first-step gain grows
  with size with `lr sqrt(d)`, 0.0037 to 0.010 of the range per step (0.0014 on the 3B
  GSM8K).

## The expectations in `10`, checked

1. 3B base on Countdown "fails on signal or the smallest `G`": wrong as written. Its live
   share is 1.0 (Countdown's partial credit) and `G` is mid-batch; the collapse appears as
   the off-task directional loss, which the analysis now reports and flags.
2. 3B instruct on Countdown "works with the largest `G^2/|C|`": held (0.043, the largest).
3. The size axis: the live share falls with size, held; the 0.5B failure, held ("fails:
   rollouts"); 1.5B and 7B at least marginal, held; 3B's "no odd response" was not
   expected.
4. Level 0 predicts level 1 (Spearman at least 0.8 between gradient sharedness and the
   cross-prompt correlation of member effects): not held, 0.54 over six settings. The two
   Countdown settings lead both orders; the four GSM8K settings have near-zero values in
   both and their order is noise.
5. The off-task reasoning block damaged first: MATH on three of four GSM8K settings
   (not on 3B GSM8K, where every response is within noise); GSM8K on the base Countdown
   model, Countdown itself on the instruct one.

## Analysis changes made on the day (docs/end_to_end/10, "Changed after the first two records")

Rows are normalized by their own kind's maximum (the pod's analysis divided a Countdown
setting's off-task block, on the 10 scale, by 1.1); `G` and RL's slope come from the
validation block and `C` from both blocks; the off-task directional change at the target
gain is reported and flagged beyond 2% of the range; the ratio and call carry a bootstrap
over prompts. The pod's `prediction-on-pod.json` files predate these for every setting
and are wrong for the two Countdown settings; `prediction.json` is the prediction.

## Incidents

- The sdpa bf16 backward returned NaN gradients for 5 of 156 live prompts of the 0.5B
  GSM8K record when the micro-batch carried a padding mask (fully masked pad rows). The
  gradient phase now right-pads without a mask under causal attention; checked on the pod
  against single sequences (identical log-probabilities, summed gradient cosine 0.9994)
  before the gradient and sign phases of that setting were rerun (423f353). A non-finite
  loss or gradient stops the phase.
- The 7B gradient phase filled the 100 GB disk (28 GB per saved gradient; the earlier
  settings' files were still there). Freed, and the phase rerun; the pod script now
  deletes a setting's gradient file after its sign phase.
- Two smoke failures before the batch (no live prompt at a 48-token cap; a zero update
  when every member scores alike), both fixed in the code, both flagged in records.

## Limitations

300 validation prompts resolve `G` to about 0.0005 per unit and leave `C` unresolved
where the curvature is small (1.5B, 3B, 7B); the three lengths stop at 2.5 member
lengths, which is in the flat region for the larger models. The next batch should extend
the length until the even response is resolved, and score more prompts. One seed of
members per setting; RL's reference step is the Tulu run's, not each model's own recipe.
