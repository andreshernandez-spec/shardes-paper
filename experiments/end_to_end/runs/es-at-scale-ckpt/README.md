# What did es-at-scale's ES do to the weights of its released MATH 7B? 2026-10-30

Hayes et al. (arXiv 2608.12679) release `conorfhayes/Qwen2.5-Math-7B-ES-MATH`, trained from
`Qwen/Qwen2.5-Math-7B` with es-at-scale (32 members, sigma 1e-3, alpha 5e-4, 512 MATH
prompts per iteration, "500 steps" on its card and in the paper's Table 2). Both pinned in
`releases.py`. es-at-scale's trainer (574a9d1) reseeds the generator for every tensor and
applies members and updates in place in bf16. `es_vllm/es_at_scale_ckpt.py` simulated both
on the base model's own weights, ignoring the update's direction, and the prediction and
three gates were committed at a416908 before the ES weights were read. CPU only, on the
laptop.

    python -m es_vllm.es_at_scale_ckpt --predict --out runs/es-at-scale-ckpt/prediction.json
    python -m es_vllm.es_at_scale_ckpt --measure --out runs/es-at-scale-ckpt/measured.json
    python -m es_vllm.es_at_scale_ckpt --table runs/es-at-scale-ckpt
    python -m es_vllm.es_at_scale_ckpt --predict --updates 259 --out runs/es-at-scale-ckpt/posthoc-259-updates.json
    python -m es_vllm.es_at_scale_ckpt --table runs/es-at-scale-ckpt --prediction posthoc-259-updates.json

Every shard of both models matches the hub's sha256 (`measured.json`, `hashes`).

**All three gates pass.** The change in every layer is the same change, the embedding and
the output head share theirs, and bf16 rounding shapes it by weight magnitude as simulated.

## Every layer moved together

Correlation of the change between layers (simulated layers 13 and 14 / measured median
over the 27 consecutive pairs, layers 0 and 27 in parentheses):

| | simulated | measured | 0 and 27 (simulated / measured) |
|---|---|---|---|
| q_proj | 0.765 | 0.769 | 0.757 / 0.765 |
| k_proj | 0.760 | 0.769 | 0.729 / 0.734 |
| v_proj | 0.764 | 0.759 | 0.743 / 0.753 |
| o_proj | 0.766 | 0.769 | 0.773 / 0.780 |
| gate_proj | 0.754 | 0.761 | 0.775 / 0.782 |
| up_proj | 0.755 | 0.760 | 0.774 / 0.779 |
| down_proj | 0.754 | 0.761 | 0.777 / 0.783 |
| embedding and output head | 0.836 | 0.841 | |

Independent draws per layer would give |r| < 0.01. The correlation is below 1 only because
rounding differs between weights of different value. The q and k biases correlate 0.09
(simulated 0.09 and 0.10): they are large weights and most updates round away. The v bias
correlates 0.58 (simulated 0.69), the norm weights 0.38 and 0.41.

So the search was not over 7.6B independent coordinates. Each update is one draw per
tensor shape, applied to all 28 layers: one layer's seven matrices (233M weights), the
embedding (545M, shared with the head) and the vectors, about 0.78B. Qiu et al. (2025,
appendix) mention the correlation and report no significant difference from i.i.d. noise.

## The released model had about 260 updates, not 501

Measured change per weight is uniformly 0.72 of the 501-update prediction. In the
binades below 2^-11, where an update is over 20 bf16 steps and rounding is negligible, the rms change grows
as the square root of the number of updates, and the measured 1.42e-3 gives 259.

A second count comes from weights between 1/8 and 1/4. There an update (rms 8.8e-5) is
below half a bf16 step and rounds away; these weights move only by the step that adding
and then subtracting a member's noise leaves behind, so they count round trips (4 per
iteration, engine 0 of 8). Simulated at 259 updates with no round trips, the 2^-3 binade
changes rms 8.5e-5; measured 9.25e-4.

Simulated again at 259 updates (post hoc, `posthoc-259-updates.json`), every binade with at
least 1e5 weights matches within 2% in rms change and 0.002 in share changed, the two
residue-driven binades included:

| abs weight | weights | changed, simulated / measured | rms change, simulated / measured |
|---|---|---|---|
| 2^-9 | 451M | 0.996 / 0.996 | 1.43e-3 / 1.44e-3 |
| 2^-6 | 1,990M | 0.970 / 0.970 | 1.62e-3 / 1.63e-3 |
| 2^-5 | 1,234M | 0.946 / 0.947 | 1.81e-3 / 1.81e-3 |
| 2^-4 | 105M | 0.729 / 0.731 | 1.04e-3 / 1.06e-3 |
| 2^-3 | 0.85M | 0.467 / 0.466 | 9.43e-4 / 9.25e-4 |

A noise-only model explains the size of the change, measured two ways that depend on the
updates differently. The direction ES found does not show in these statistics.

Which checkpoint this is, the weights cannot say. es-at-scale evaluates every 5 iterations
by default and, with `--save-best-models`, saves at each new best mean pass@1; a save after
iteration 259 holds 260 updates. Its math evaluation suite is MATH500, Olympiad Bench,
Minerva, AIME and AMC, and the paper reports the first three. A run stopped near iteration
260 would look the same. At 260 iterations the model saw about 4.3M samples, not the
8.2M that 500 would give.

## Size of the change

The sampled layer matrices moved 4.3% to 9.7% of their norm (`--table`), the embedding
8.4% and the output head 8.8% (`measured.json`, `tensors`). The final norm (weights near
2.9) is unchanged except for one weight in 3,584, as predicted: there neither updates nor
round trips cross a bf16 step.
