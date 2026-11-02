# 12. Qiu et al.'s Countdown setup, matched: preregistration

Written and committed 2026-11-01, before any run. Decided by Andres the same day: the
comparison with Qiu et al. (2025) uses their setup exactly, training and evaluation, so
that only the implementation differs. Arm B (our ES on their setup) runs first; Arm A
(their code, run by us on the same hardware) is decided after Arm B's result.

## Why

S5 to 500 updates (`10-when-does-es-work.md`, `runs/s5-early/README.md`) used their ES
hyperparameters and nothing else of their setup, and ended at accuracy 0.496 against
their 60.5%. Their numbers come from the `archive/` scripts of es-at-scale (its README
says so), not from the es-at-scale library T2 compared against (`06-countdown-check.md`).
Read from the pinned clone (574a9d1), those scripts differ from S5 in seven settings:

| setting | their code | S5 | this run |
|---|---|---|---|
| training problems | the first 200 of `countdown.json`, each scored by every member every update | 192 of the same 200 | theirs |
| test problems | the last 2,000 of `countdown.json` | 300 of those 2,000 | theirs |
| new tokens, training and test | at most 1,024 | at most 512 | theirs |
| decoding | HF `generate`, greedy, with the model's repetition penalty 1.05, which `generate` applies even when greedy, padding included | vLLM greedy, no penalty | vLLM greedy with the penalty, padding included (below) |
| test precision | float16, `eval_countdown.py`'s default, flash-attention 2 | bf16 | theirs: their script scores our checkpoints |
| training reward | `reward_function` on the decoded prompt and answer together: the prompt's own tags give every answer 0.06 of format, and a target that is not an integer (58 of the 2,000 test problems, none of the 200) earns no answer point | es-at-scale's grader on the answer alone | theirs |
| test schedule | checkpoints after 100, 200, ..., 500 updates, scored by a separate script | every update, in the training engine | theirs, plus a running curve |

Already the same in S5: Qwen2.5-3B-Instruct at `aa8e725`, the raw prompt, 30 members,
sigma 1e-3, alpha 5e-4, z-scored fitness, greedy decoding, 500 updates, and accuracy as
the share of answers with the answer point. The data consumption is theirs: 200 problems
times 30 members times 500 updates, 3.0M answers.

## What stays different

The implementation, which is the question:

- **Weights.** Theirs are bf16, changed in place: each member adds its noise and subtracts
  it again, which leaves rounding residue. Ours are an f32 master; the engine gets its
  bf16 copy and each restore writes that copy again.
- **Update.** They sum `z_n eps_n` in bf16; shardes sums in f32.
- **Noise.** They reseed one generator with the member's seed for every tensor, so all
  tensors of one shape get the same draw (every layer's q_proj, for instance); the paper
  calls it partially correlated and reports no difference from i.i.d. noise. Shardes
  draws an independent stream per tensor.
- **Generation.** vLLM instead of HF `generate`, so different kernels and rounding.
  The decoding rule is matched: greedy, 1,024 tokens, both end tokens, the repetition
  penalty, and `pad_penalty.py` applies HF's penalty on the pad token to the rows their
  batches pad (shorter than the longest prompt of the batch; all 200 training prompts form
  one batch). The decode check below measures what is left.
- **z-score.** Theirs divides by `std + 1e-8`, ours by `std` (a relative difference of
  about 1e-6 at the spreads S5 saw).

## What runs

One seed (1), 500 updates, one H200: `es_vllm/qiu_match.py` through
`es_vllm/qiu_match_pod.sh`. Checkpoints after 0, 100, ..., 500 updates are the engine's
bf16 weights, read back and checked against the run's digest, and their
`eval_countdown.py` scores each one unchanged, with their README's command (2,000
problems, 1,024 tokens, batch 500), in its own venv (`requirements-qiu-eval.txt`: torch
2.8, transformers 4.56.2, flash-attn 2.8.3, the releases of September 2025, since their
requirements pin nothing). Every 10 updates the driver also decodes the 2,000 test
problems by the matched rule in vLLM, as a running curve. The usual checks run at every
update: member 0 and the restored weights bit for bit against the master, the weights'
digest logged for replay.

## Checks before the run counts

1. **Decode check** (`es_vllm/qiu_decode_check.py`), on the pod before training, at the
   base weights: the 200 training prompts decoded by HF `generate` as their trainer calls
   it (one batch) and by our vLLM rule, and by two variants of ours (no pad penalty; no
   repetition penalty), once in float32, where a remaining difference is the rule, and
   once in bf16, which the run uses. Two bf16 implementations part ways inside an answer
   through rounding alone, so the bf16 record also has the floor: HF against itself with
   only the batching changed (batches of 50). Reported: identical answers, where the
   others part, rewards and accuracy. In both precisions the matched rule has to agree
   with HF on at least as many answers as each variant, and a penalty put through the
   pad-penalty processor has to change a greedy pick; otherwise the run stops.
2. **Their evaluator on checkpoint 0** (the base weights) gives their Table 1 figure,
   10.0%, within 1.5 points. Otherwise the run stops.

## Reading, fixed before the data

Primary statistic: their evaluator's accuracy on checkpoint 500. Their reference is the
three-seed result of their Table 11 for Qwen2.5-3B, 58.67% with standard deviation 1.47,
so the band is the mean plus or minus two deviations, 55.7% to 61.6%.

- **55.7% or more**: shardes' ES reproduces theirs on their setup. S5's level came from
  the setup.
- **Below 55.7%**: on their setup our implementation ends below their reported runs. The
  cause is in the list above or in their result not reproducing on our hardware; Arm A
  separates the two.
- **Above 61.6%**: ours ends above their runs.

Secondary, reported without a gate: checkpoints 100 to 400 beside their Figure 7 (35.6,
41.2, 48.0, 54.2, read off the plot, good to about half a point), and the running curve
beside their evaluator at the six checkpoints (what the change of engine and precision
moves). One seed of ours against their three: our seed-to-seed spread is not measured,
and the band uses theirs.

## Cost

S5 took 96 s per update at 512 tokens on 192 prompts. At 1,024 tokens on 200, about 120
to 160 s, so 17 to 22 hours for 500 updates, with their evaluator beside it and setup:
$95 to 125 on a Secure H200 at $5.29/h. If the first 20 updates project above $150 the
run stops and the plan is revisited. Health checks every 30 minutes, as before.

## Amendment before update 200 (2026-11-01, written at update 140)

Andres, at update 138: if the curve is still flattening by update 200, stop the run and
find out why, running their code if useful. "Still flattening" is fixed here, before the
data after update 140: a least-squares line through the running test accuracies after
100, 110, ..., 200 updates (eleven evaluations of the 2,000 problems). If its value at 200
plus 300 times its slope is below 55.7% (the band's lower edge for checkpoint 500), the
run stops once checkpoint 200 is written and scored by their evaluator; otherwise it goes
on to 500. A straight line is generous to a curve that has been concave so far, so a stop
means even a straight continuation would miss the band.

If it stops, the diagnosis starts from one difference not yet weighed. Their trainer
reseeds one generator with the member's seed for every tensor (their i.i.d. variant only
shifts the seed by the tensor's index), so every tensor of one shape gets the same noise:
every layer's q_proj and o_proj share one draw, every gate_proj and up_proj another, and
so on. Their update is then one delta per shape, applied to all 36 layers: about 50M
independent directions in the transformer layers against our 2.8B. Doc 08 traced ES's
stall to the random part of the update, which grows with the directions searched, so this
is the first candidate. Their trainer runs unchanged for a few iterations on the same pod
with hooks that record the seeds, fitnesses and a few matrices (q_proj, o_proj, gate_proj,
down_proj and a layer norm at layers 0, 17 and 35), to show what their update does to the
weights: how much of the intended change survives bf16 rounding, what the perturb and
restore leave behind, and how the change is shared across layers. The same quantities
come from our checkpoints 0, 100 and 200.

**Applied at update 200 (2026-11-02, 02:10 UTC): the run continues.** The running tests
after 100 to 200 updates were 45.4, 45.6, 45.9, 48.1, 48.4, 47.7, 49.1, 48.2, 49.6, 49.3
and 50.6%; the line rises 0.48 points per 10 updates, stands at 50.4% at 200 and reaches
64.7% at 500, above 55.7%. Their evaluator gave checkpoint 200 49.9% (their Figure 7: 41.2%
at 1.2M sample evaluations).
