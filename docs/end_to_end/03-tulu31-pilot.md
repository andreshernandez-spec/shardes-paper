# 03. Tulu 3.1 pilot: preregistration

Written and committed 2026-09-27, before any pilot run. T4 of `01-tulu31-plan.md`. The
gate below is fixed here and not changed after the data; a different question after the
result gets a new document that says it came after.

## What runs

Four arms, each on its own H100 80GB SXM (the GPU type the matched run and its
evaluation will use), from `allenai/Llama-3.1-Tulu-3-8B-DPO`, driver
`experiments/end_to_end/es_vllm/run_tulu.py`, configs `es_vllm/tulu-pilot-*.yaml`:

| arm | sigma | alpha (lr = alpha x sigma) | reward |
|---|---|---|---|
| s5e-4 | 5e-4 | 5e-4 | the run's verifiers |
| s1e-3 | 1e-3 | 5e-4 | the run's verifiers |
| s2e-3 | 2e-3 | 5e-4 | the run's verifiers |
| random | 1e-3 | 5e-4 | seeded uniform (the control) |

Common: full-rank `SeedRegenerated`, N = 16 members, `group_relative` on `(N, 1)`
(Qiu's z-score), greedy decoding, cap 2,048, stop on eos, a response without eos
scores 0 (the run's `--non_stop_penalty`), 30 iterations, seed 0.

**N = 16 is what "same data, matched rollouts" requires.** The RL run drew 48 prompts
and 16 samples per prompt each step. An ES iteration here takes the 192 prompts of RL
steps 4g+1 to 4g+4, in the run's own order, and scores each with 16 members: 3,072
rollouts, exactly those four RL steps' 4 x 768. So after iteration g both methods have
seen the same prompts and generated the same number of rollouts, at every iteration.
Thirty iterations cover RL steps 1 to 120 (`step_120` is a released branch).

## What is measured

Each iteration first decodes the current weights (the view) on its own 192 prompts,
before any member sees them: `center_reward`, on the 0-10 scale, with per-source
means. In the first epoch those prompts are new to the run, so this is a held-out
measurement of the weights so far. All four arms see the same prompts at the same
iteration, so the arms are compared pairwise, iteration by iteration.

Also logged: every member's fitness (the rebuild record), response lengths, cap hits,
the member fitness spread, a digest every 5 iterations, and at iteration 0 a
bit-for-bit check that the engine held member 0's weights and, after the update, the
view.

## Gate G3

For each true-reward arm, `d_g = center_reward(arm, g) - center_reward(random, g)` for
g = 0..29, and the ordinary least squares slope of `d_g` on g with its standard error.

- **Pass** for an arm: slope > 2 standard errors above zero, and the mean of `d_g` over
  g = 20..29 above zero.
- **Sigma for the matched run**: the passing arm with the largest slope; if two slopes
  are within one standard error of each other, the smaller sigma.
- **Negative**: no arm passes. Reported as "no detectable learning within 30 iterations
  (RL steps 1 to 120's data)", with the diagnostics below, and the next step is decided
  with Andres; no configuration is added to rescue it.

Reported whatever the outcome: every arm's `center_reward` curve, the per-source curves,
the fitness spread per iteration (an arm whose members all score alike has no signal:
`group_relative` gives zero weights), lengths and cap hits.

## Cost

Measured by the probe: 17.8 s per 192-prompt batch on an H100. An iteration is 17
batches (center plus 16 members) plus the member writes and the update, about 5.5
minutes; 30 iterations about 2.75 h per arm. Three pods for about 3 h each, the random
arm chained after one of them (it decodes only the center, about 12 minutes): about
$25-35 at $2.69-3.49/h. All pods deleted after harvest.

## Amendment, 2026-09-27, before any pilot data

**GPU type: H200 SXM (141 GB) instead of H100 80GB, for every arm, and therefore for the
matched run and its evaluation.** The first arm, started on an H100, loaded the f32
master (8.03B parameters, 30 GiB) beside vLLM's half of the card and reached 80.7 GB,
surviving only because JAX's allocator retried with smaller blocks; the update's
temporaries would not fit. It then stopped at iteration 0 on a bug in our weight check
(vLLM pads the vocabulary from 128,264 to 128,320 rows; the check refused the padding,
fixed in the next commit). It had decoded and scored iteration 0's center before the
check stopped it, but the log line is written at the end of an iteration, so that value
was never logged or printed, and no member was decoded. No pilot data exists from that
pod, and the random arm it had started was stopped before
its first iteration, since the arms are compared pairwise and must share a GPU type.
Price: $3.59/h against the $3.49 the H100 has cost us on Secure Cloud; the H200 decodes
faster (4.8 against 3.35 TB/s), so per result it should cost no more. The configs are
unchanged: `gpu_memory_utilization` is a fraction of the card.

## Amendment 2, 2026-09-27: the first run is invalid; the pilot is rerun as preregistered

The first complete run (commit 2eb8506, all four arms on H200) is recorded in
`experiments/end_to_end/runs/pilot1-invalid/` and not used for the gate. Its primary
metric was corrupted: at 4 to 7 of 30 iterations per arm, the random control included,
`center_reward` collapsed to 0.3-0.8 with center responses of 900-1,600 tokens, then
recovered, while no member evaluation ever collapsed. Cause, reproduced on the laptop by
`es_vllm/race_check.py` (4 of 20 restores wrong): the JAX to torch weight handoff after
`es_tell` did not wait for JAX to finish each leaf or keep it alive until torch had
copied it. Fixed in `worker.py` (0 of 40 restores wrong after the fix), and the driver
now checks the engine bit for bit after every restore and one member every ten
iterations, aborting on a mismatch.

The gate applied to the invalid run as it stands (`gate-as-computed.json`: no arm
passes) is kept beside it. It is not the pilot's result, because the measurement was
wrong at known iterations; dropping those iterations after seeing them would be a rule
chosen from the data. The rerun uses the same configs, the same gate and the same GPU
type, from the commit that carries this amendment.

## Result, 2026-09-27

The rerun completed all four arms with every weight check passing. **Gate G3: no arm
passes** (slopes +0.0084, +0.0040 and +0.0057 per iteration at 1.7, 0.9 and 0.9 standard
errors; late mean differences +0.13, +0.10, +0.18). Preregistered outcome: no detectable
learning within 30 iterations of RL steps 1 to 120's data. The next step is decided with
Andres. Details and the other numbers: `experiments/end_to_end/runs/README.md`.
