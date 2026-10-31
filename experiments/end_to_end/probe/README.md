# Throughput probe, 2026-09-26

What one ES member batch costs to decode with vLLM 0.30.0, on one A100-SXM4-80GB and one
H100 80GB SXM, from each setting's start checkpoint with its own prompts, template,
stop strings and cap. Plan: `docs/end_to_end/00-plan.md`. Driver `probe_throughput.py`,
configs `probe.yaml` (committed at 7a2c969 and run at faf9b31) and `probe-sampling.yaml`
(run at ba7db25), cost table `cost.py`.

| directory | pod | commit | cells |
|---|---|---|---|
| `results-a100/` | community A100-SXM4-80GB, CUDA 13.1 host, $1.39/h | faf9b31 | olmo3_if greedy P32; tulu31 P48 and P192 |
| `results-h100/` | secure H100 80GB HBM3, AP-IN-1, CUDA 13.0 host, $3.49/h | faf9b31 | same |
| `results-a100-sampling/` | the same A100 pod, afterwards | ba7db25 | olmo3_if at temperature 1.0, P32 |

## Results

`python cost.py results-a100 results-h100 results-a100-sampling`, rows for the released
checkpoints (1.2x uptime overhead, list prices of 2026-09-25):

| gpu | setting | decode | matched to | P | s/member | mean len | cap hits | GPU-h | USD |
|---|---|---|---|---|---|---|---|---|---|
| A100 | olmo3_if | greedy | step 2000 | 32 | 642.2 | 9041 | 63/128 | 3424.8 | 4761-5446 |
| H100 | olmo3_if | greedy | step 2000 | 32 | 332.1 | 8571 | 59/128 | 1771.2 | 4764-6181 |
| A100 | olmo3_if | T=1 | step 2000 | 32 | 249.1 | 2873 | 3/128 | 1328.7 | 1847-2113 |
| A100 | tulu31 | greedy | step 1920 | 192 | 35.6 | 302 | 5/384 | 91.2 | 127-145 |
| H100 | tulu31 | greedy | step 1920 | 192 | 17.8 | 309 | 6/384 | 45.4 | 122-159 |
| A100 | tulu31 | greedy | step 1920 | 48 | 19.3 | 300 | 1/192 | 197.8 | 275-314 |
| H100 | tulu31 | greedy | step 1920 | 48 | 9.9 | 302 | 0/192 | 100.9 | 271-352 |

Reading:

- **Tulu 3.1 is affordable**: $122-159 for an ES run matched to the released checkpoint
  on either GPU type at 192 prompts per member, $41-53 to step 640. Four times the prompts
  per member cost 2.2x less per rollout, because a 48-prompt batch is bound by reading the
  weights.
- **Olmo 3 RL-Zero IF is not, as designed.** Under greedy decoding the base model runs to
  the 16,384-token cap on 46% (H100) and 49% (A100) of the IF prompts; median length 4,123
  and 15,589 tokens. At the run's own temperature 1.0 only 3 of 128 reach the cap, but the
  mean is still 2,873 tokens (median 1,630, p90 6,556), far above the RL run's per-step
  mean of ~340 over its training, and a full-rank member waits for its longest sample
  (155-301 s per 32-prompt batch on the A100).
- **Per result the GPU types cost the same**; the H100 takes half the wall clock.
- **Greedy decoding from this base model is not portable across GPU types**: on the same
  128 prompts the A100 and the H100 agree on the output length for 60 (most of them both
  at the cap). A run and its evaluation have to stay on one GPU type and engine setup.
- Rewriting all 14.6 GB of weights in place through the worker extension takes 12 ms on
  the H100 and 19 ms on the A100, and leaves them bit-identical: negligible per member.
- Prompts under `olmo_thinker` average 293 tokens.

## What was not run, and why

`probe.yaml` also lists olmo3_if at P128 (2 members) and at gpu_memory_utilization 0.55
(2 members). With half the greedy responses running to 16,384 tokens they would have
taken hours on the A100, so both drivers were stopped after the olmo3_if P32 cell was
written and relaunched with `--only tulu31`. `probe-sampling.yaml` was added after the
greedy result, committed before its run, and run on the A100 pod that was still up.

## Incidents

- Community H100 capacity on CUDA 13 hosts read "Medium" and failed twice at creation;
  the H100 is a Secure Cloud pod ($3.49/h instead of $2.69).
- Stopping the driver with `kill` orphaned vLLM's `EngineCore` process, which kept 69.5 GB
  of GPU memory until it was killed by pid. The ES driver must shut the engine down on
  SIGTERM.
- Found by the local smoke run before any pod existed: `pyarrow` missing from the vLLM
  venv, and FlashInfer's sampler JIT-compiling at warmup (needs nvcc; greedy decoding never
  uses it, so it is disabled with `VLLM_USE_FLASHINFER_SAMPLER=0`).
- vLLM 0.30.0's torch is built for CUDA 13.0; pods were created with `minCudaVersion 13.0`.

## Cost

Pod uptime: A100 11:35:37 to about 12:52 UTC (1.27 h, about $1.77); H100 11:35:58 to about
12:13 UTC (0.62 h, about $2.16). About $3.9 in total, from uptime; the billing API had not
caught up when this was written. Both pods deleted after harvest. `probe-log.txt.gz` in
the A100 and H100 directories is each pod's boot and probe log.
