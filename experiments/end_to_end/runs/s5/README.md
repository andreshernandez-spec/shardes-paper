# S5 stage 1: GRPO to the ES arms' sample count on two Countdown settings (docs/end_to_end/10, "S5 stage 1: preregistration")

Two GRPO arms on 2026-10-25 and 26 with `es_vllm/grpo_run.py`, each for 480 steps of 768
samples (368,640, the sample count of the 30-update ES arms on the same settings), scored
every 10 steps on the blocks those ES arms were scored on, so that data efficiency can be
read as the reward at equal samples seen (`10`, "S5", the correction of 2026-10-22). The
settings are the two with the most ES arms and the published Countdown claims:
Qwen2.5-3B-Instruct (the first probe's blocks, `runs/probe/qwen3b-countdown`, which the
S3b and S4a arms used) and Qwen2.5-1.5B-Instruct (the S4b probe's blocks,
`runs/s4b-probe/qwen1.5b-countdown`, which the S4b arms used). The recipe is the Tulu 3.1
run's as in `runs/grpo-run/` (48 prompts and 16 samples per step, beta 0.01, clip 0.2),
at the better of two learning rates on the validation block at step 40: 1e-6 on both
settings (3B: +0.159 against +0.138, `runs/grpo-run/`; 1.5B: +0.207 against +0.181, the
40-step arm `runs/grpo-run/qwen1.5b-countdown-lr1e-6-s0`, run first in this session).

Two Secure H200 pods in US-NC-1 at $5.29/h (`f0kcjdpdfutv9i` for 3B, `ipt6mek5i07yaq` for
1.5B), 21:44 UTC on the 25th to 15:12 and 12:12 UTC on the 26th: 17.5 and 14.5 hours,
about $169 by uptime, plus a 200 GB network volume (`dybguawvgn`, about $14 a month) that
holds each run's end checkpoint (the policy in f32, the optimizer and the sampler state,
37 GB and 18.5 GB) so that stage 2 can continue the same runs. The pods ran commit
72091fd; `run.json` stamps it.

**Incident.** Both first attempts died at their second checkpoint write (step 80) when the
volume, then 80 GB, filled: the 3B run's 37 GB checkpoint plus its 37 GB temporary file
plus the 1.5B run's 18.5 GB. The Countdown grader's worker pool kept each dead process
alive, so the pods sat idle for about 4.5 hours (about $48) before the processes were
killed at 04:17 UTC. The launch scripts then resumed both runs from their step-40
checkpoints (`run.json` lists the resume; the records at or after step 40 from the first
attempt were dropped and recomputed; the recomputed evaluation at step 40 equals the
first attempt's) with the volume enlarged to 200 GB. Every later checkpoint succeeded
(twelve per run). The driver fix for future runs, a non-fatal checkpoint failure and an
exit on any exception, is commit 998785f and was not deployed to the running arms. The
rest of the session was watched by a subagent every 30 minutes (progress, GPU, evals,
checkpoints, errors), which caught nothing further.

Files per arm as in `runs/grpo-run/`: `run.json` (with `resumed` and `checkpoints`),
`log.jsonl` (480 steps), `eval.jsonl` (49 evaluations).

## The table

`python -m es_vllm.grpo_run --table runs/s5`:

| arm | lr | beta | seed | steps | rollouts | validation at end | test at end | off-task at end | peak test gain | at | test +0.10 first at | validation-selected checkpoint | test there | class | collapsed | s per step | GPU-s per answer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen1.5b-countdown-grpo-lr1e-6-s0 | 1e-06 | 0.01 | 0 | 480 | 368,640 | +0.228 | +0.250 | -0.003 | 0.265 | 300 | 20 | 80 | 0.232 | works | no | 62.6 | 0.0816 |
| qwen3b-countdown-grpo-lr1e-6-s0 | 1e-06 | 0.01 | 0 | 480 | 368,640 | +0.432 | +0.430 | -0.017 | 0.466 | 270 | 10 | 300 | 0.442 | works | yes | 85.8 | 0.112 |

## Data efficiency: the ES arms against GRPO at equal samples seen

`python -m es_vllm.probe_run --samples <grpo run> <ES run directories>`: each ES
evaluation paired with the GRPO evaluation at the same sample count (ES update 5m is
61,440m samples, GRPO step 80m; the 256-member arm's update 5 is 245,760, step 320), the
change on the reserved test block under each method and the ES-minus-GRPO difference
paired over the block's 300 prompts, with the off-task (GSM8K) change beside.

3B instruct (`runs/probe-run` and `runs/s4a`):

| ES run | samples seen | ES update | GRPO step | ES test | GRPO test | ES minus GRPO | paired SE | ahead | ES off-task | GRPO off-task |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3b-countdown-s0 | 61,440 | 5 | 80 | +0.033 | +0.312 | -0.280 | +0.028 | GRPO | -0.067 | -0.003 |
| qwen3b-countdown-s0 | 122,880 | 10 | 160 | +0.070 | +0.373 | -0.303 | +0.029 | GRPO | -0.123 | -0.017 |
| qwen3b-countdown-s0 | 184,320 | 15 | 240 | +0.135 | +0.413 | -0.278 | +0.027 | GRPO | -0.277 | -0.013 |
| qwen3b-countdown-s0 | 245,760 | 20 | 320 | +0.060 | +0.433 | -0.373 | +0.026 | GRPO | -0.600 | +0.000 |
| qwen3b-countdown-s0 | 307,200 | 25 | 400 | -0.005 | +0.416 | -0.420 | +0.027 | GRPO | -0.843 | -0.017 |
| qwen3b-countdown-s0 | 368,640 | 30 | 480 | -0.118 | +0.430 | -0.548 | +0.027 | GRPO | -0.837 | -0.017 |
| qwen3b-countdown-s1 | 61,440 | 5 | 80 | +0.111 | +0.312 | -0.201 | +0.026 | GRPO | -0.003 | -0.003 |
| qwen3b-countdown-s1 | 122,880 | 10 | 160 | +0.128 | +0.373 | -0.245 | +0.026 | GRPO | -0.057 | -0.017 |
| qwen3b-countdown-s1 | 184,320 | 15 | 240 | +0.164 | +0.413 | -0.250 | +0.028 | GRPO | -0.090 | -0.013 |
| qwen3b-countdown-s1 | 245,760 | 20 | 320 | +0.090 | +0.433 | -0.344 | +0.028 | GRPO | -0.420 | +0.000 |
| qwen3b-countdown-s1 | 307,200 | 25 | 400 | +0.051 | +0.416 | -0.364 | +0.026 | GRPO | -0.737 | -0.017 |
| qwen3b-countdown-s1 | 368,640 | 30 | 480 | -0.055 | +0.430 | -0.484 | +0.027 | GRPO | -0.843 | -0.017 |
| qwen3b-countdown-n256-s0 | 245,760 | 5 | 320 | +0.170 | +0.433 | -0.263 | +0.027 | GRPO | -0.030 | +0.000 |
| qwen3b-countdown-random-s0 | 61,440 | 5 | 80 | -0.034 | +0.312 | -0.346 | +0.027 | GRPO | -0.027 | -0.003 |
| qwen3b-countdown-random-s0 | 122,880 | 10 | 160 | -0.069 | +0.373 | -0.442 | +0.030 | GRPO | -0.083 | -0.017 |
| qwen3b-countdown-random-s0 | 184,320 | 15 | 240 | -0.100 | +0.413 | -0.514 | +0.027 | GRPO | -0.120 | -0.013 |
| qwen3b-countdown-random-s0 | 245,760 | 20 | 320 | -0.089 | +0.433 | -0.522 | +0.027 | GRPO | -0.470 | +0.000 |
| qwen3b-countdown-random-s0 | 307,200 | 25 | 400 | -0.120 | +0.416 | -0.536 | +0.027 | GRPO | -0.843 | -0.017 |
| qwen3b-countdown-random-s0 | 368,640 | 30 | 480 | -0.124 | +0.430 | -0.553 | +0.027 | GRPO | -0.843 | -0.017 |
| qwen3b-countdown-shrink0.1-s0 | 61,440 | 5 | 80 | +0.102 | +0.312 | -0.211 | +0.029 | GRPO | -0.040 | -0.003 |
| qwen3b-countdown-shrink0.1-s0 | 122,880 | 10 | 160 | +0.097 | +0.373 | -0.276 | +0.028 | GRPO | -0.033 | -0.017 |
| qwen3b-countdown-shrink0.1-s0 | 184,320 | 15 | 240 | +0.151 | +0.413 | -0.263 | +0.028 | GRPO | -0.063 | -0.013 |
| qwen3b-countdown-shrink0.1-s0 | 245,760 | 20 | 320 | +0.142 | +0.433 | -0.292 | +0.027 | GRPO | -0.057 | +0.000 |
| qwen3b-countdown-shrink0.1-s0 | 307,200 | 25 | 400 | +0.235 | +0.416 | -0.181 | +0.026 | GRPO | -0.067 | -0.017 |
| qwen3b-countdown-shrink0.1-s0 | 368,640 | 30 | 480 | +0.154 | +0.430 | -0.276 | +0.029 | GRPO | -0.030 | -0.017 |
| qwen3b-countdown-step3-s0 | 61,440 | 5 | 80 | +0.065 | +0.312 | -0.248 | +0.028 | GRPO | -0.023 | -0.003 |
| qwen3b-countdown-step3-s0 | 122,880 | 10 | 160 | +0.100 | +0.373 | -0.274 | +0.028 | GRPO | -0.007 | -0.017 |
| qwen3b-countdown-step3-s0 | 184,320 | 15 | 240 | +0.125 | +0.413 | -0.289 | +0.026 | GRPO | -0.040 | -0.013 |
| qwen3b-countdown-step3-s0 | 245,760 | 20 | 320 | +0.179 | +0.433 | -0.254 | +0.028 | GRPO | -0.037 | +0.000 |
| qwen3b-countdown-step3-s0 | 307,200 | 25 | 400 | +0.177 | +0.416 | -0.239 | +0.027 | GRPO | +0.003 | -0.017 |
| qwen3b-countdown-step3-s0 | 368,640 | 30 | 480 | +0.185 | +0.430 | -0.244 | +0.025 | GRPO | -0.017 | -0.017 |

1.5B instruct (`runs/s4b`):

| ES run | samples seen | ES update | GRPO step | ES test | GRPO test | ES minus GRPO | paired SE | ahead | ES off-task | GRPO off-task |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen1.5b-countdown-anchored-s0 | 61,440 | 5 | 80 | +0.069 | +0.232 | -0.162 | +0.023 | GRPO | -0.002 | +0.012 |
| qwen1.5b-countdown-anchored-s0 | 122,880 | 10 | 160 | +0.116 | +0.253 | -0.137 | +0.025 | GRPO | +0.010 | -0.005 |
| qwen1.5b-countdown-anchored-s0 | 184,320 | 15 | 240 | +0.127 | +0.247 | -0.120 | +0.023 | GRPO | -0.002 | -0.007 |
| qwen1.5b-countdown-anchored-s0 | 245,760 | 20 | 320 | +0.149 | +0.262 | -0.113 | +0.027 | GRPO | +0.022 | +0.003 |
| qwen1.5b-countdown-anchored-s0 | 307,200 | 25 | 400 | +0.142 | +0.226 | -0.084 | +0.026 | GRPO | +0.012 | +0.015 |
| qwen1.5b-countdown-anchored-s0 | 368,640 | 30 | 480 | +0.148 | +0.250 | -0.102 | +0.026 | GRPO | +0.025 | -0.003 |
| qwen1.5b-countdown-anchored-s1 | 61,440 | 5 | 80 | +0.052 | +0.232 | -0.179 | +0.024 | GRPO | +0.003 | +0.012 |
| qwen1.5b-countdown-anchored-s1 | 122,880 | 10 | 160 | +0.119 | +0.253 | -0.134 | +0.023 | GRPO | +0.007 | -0.005 |
| qwen1.5b-countdown-anchored-s1 | 184,320 | 15 | 240 | +0.133 | +0.247 | -0.114 | +0.024 | GRPO | +0.013 | -0.007 |
| qwen1.5b-countdown-anchored-s1 | 245,760 | 20 | 320 | +0.133 | +0.262 | -0.129 | +0.027 | GRPO | +0.013 | +0.003 |
| qwen1.5b-countdown-anchored-s1 | 307,200 | 25 | 400 | +0.137 | +0.226 | -0.088 | +0.025 | GRPO | +0.020 | +0.015 |
| qwen1.5b-countdown-anchored-s1 | 368,640 | 30 | 480 | +0.147 | +0.250 | -0.103 | +0.026 | GRPO | +0.025 | -0.003 |
| qwen1.5b-countdown-unanchored-s0 | 61,440 | 5 | 80 | +0.085 | +0.232 | -0.146 | +0.023 | GRPO | -0.010 | +0.012 |
| qwen1.5b-countdown-unanchored-s0 | 122,880 | 10 | 160 | +0.145 | +0.253 | -0.108 | +0.025 | GRPO | -0.002 | -0.005 |
| qwen1.5b-countdown-unanchored-s0 | 184,320 | 15 | 240 | +0.183 | +0.247 | -0.064 | +0.023 | GRPO | -0.005 | -0.007 |
| qwen1.5b-countdown-unanchored-s0 | 245,760 | 20 | 320 | +0.171 | +0.262 | -0.091 | +0.027 | GRPO | -0.002 | +0.003 |
| qwen1.5b-countdown-unanchored-s0 | 307,200 | 25 | 400 | +0.170 | +0.226 | -0.055 | +0.025 | GRPO | +0.012 | +0.015 |
| qwen1.5b-countdown-unanchored-s0 | 368,640 | 30 | 480 | +0.180 | +0.250 | -0.070 | +0.026 | GRPO | -0.025 | -0.003 |

GRPO is ahead of every ES arm at every common milestone on both settings: on 3B by 0.18
to 0.55 (6 to 20 paired standard errors), on 1.5B by 0.06 to 0.18 (2.2 to 7.5). At the ES
arms' full sample count, 368,640, GRPO's reserved block is at +0.430 on 3B against the
ES arms' +0.185 (the step-3 arm), +0.154 (anchored) and below, and at +0.250 on 1.5B
against +0.148, +0.147 and +0.180.

## The GRPO curves

`python -m es_vllm.grpo_run --curves runs/s5`:

**qwen1.5b-countdown-grpo-lr1e-6-s0** (lr 1e-06, beta 0.01, seed 0, 480 steps, 368,640 rollouts, 8.4 h in the loop)

| after | validation | test | off-task | answer length |
|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 235 |
| 10 | +0.045 | +0.030 | +0.010 | 130 |
| 20 | +0.162 | +0.165 | +0.000 | 117 |
| 30 | +0.197 | +0.180 | +0.000 | 108 |
| 40 | +0.200 | +0.204 | +0.010 | 109 |
| 50 | +0.219 | +0.232 | -0.007 | 106 |
| 60 | +0.245 | +0.247 | +0.002 | 109 |
| 70 | +0.238 | +0.229 | -0.005 | 119 |
| 80 | +0.253 | +0.232 | +0.012 | 109 |
| 90 | +0.241 | +0.241 | -0.007 | 109 |
| 100 | +0.238 | +0.244 | +0.010 | 111 |
| 110 | +0.242 | +0.235 | -0.002 | 116 |
| 120 | +0.244 | +0.232 | +0.000 | 118 |
| 130 | +0.234 | +0.226 | +0.013 | 124 |
| 140 | +0.245 | +0.229 | +0.003 | 119 |
| 150 | +0.241 | +0.235 | +0.008 | 118 |
| 160 | +0.234 | +0.253 | -0.005 | 119 |
| 170 | +0.231 | +0.247 | +0.005 | 117 |
| 180 | +0.230 | +0.247 | -0.002 | 138 |
| 190 | +0.231 | +0.259 | +0.000 | 119 |
| 200 | +0.239 | +0.214 | +0.010 | 119 |
| 210 | +0.247 | +0.229 | +0.018 | 120 |
| 220 | +0.247 | +0.232 | +0.008 | 122 |
| 230 | +0.248 | +0.241 | +0.018 | 125 |
| 240 | +0.227 | +0.247 | -0.007 | 122 |
| 250 | +0.233 | +0.229 | +0.003 | 122 |
| 260 | +0.250 | +0.253 | +0.015 | 124 |
| 270 | +0.236 | +0.256 | +0.012 | 120 |
| 280 | +0.231 | +0.232 | +0.012 | 118 |
| 290 | +0.241 | +0.238 | -0.010 | 116 |
| 300 | +0.247 | +0.265 | +0.010 | 117 |
| 310 | +0.245 | +0.247 | +0.005 | 120 |
| 320 | +0.233 | +0.262 | +0.003 | 119 |
| 330 | +0.236 | +0.250 | +0.012 | 131 |
| 340 | +0.221 | +0.244 | +0.007 | 119 |
| 350 | +0.231 | +0.235 | +0.013 | 121 |
| 360 | +0.236 | +0.241 | +0.015 | 151 |
| 370 | +0.218 | +0.226 | +0.007 | 126 |
| 380 | +0.218 | +0.235 | +0.012 | 122 |
| 390 | +0.231 | +0.238 | -0.002 | 123 |
| 400 | +0.230 | +0.226 | +0.015 | 121 |
| 410 | +0.213 | +0.211 | +0.017 | 128 |
| 420 | +0.218 | +0.211 | +0.023 | 124 |
| 430 | +0.210 | +0.201 | +0.012 | 125 |
| 440 | +0.210 | +0.198 | +0.005 | 129 |
| 450 | +0.219 | +0.223 | +0.018 | 128 |
| 460 | +0.221 | +0.238 | +0.012 | 124 |
| 470 | +0.222 | +0.211 | +0.012 | 117 |
| 480 | +0.228 | +0.250 | -0.003 | 122 |

**qwen3b-countdown-grpo-lr1e-6-s0** (lr 1e-06, beta 0.01, seed 0, 480 steps, 368,640 rollouts, 11.4 h in the loop)

| after | validation | test | off-task | answer length |
|---|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 | 329 |
| 10 | +0.115 | +0.113 | -0.007 | 231 |
| 20 | +0.143 | +0.129 | -0.013 | 220 |
| 30 | +0.161 | +0.199 | -0.013 | 255 |
| 40 | +0.197 | +0.216 | -0.010 | 231 |
| 50 | +0.193 | +0.210 | -0.010 | 247 |
| 60 | +0.210 | +0.275 | +0.000 | 257 |
| 70 | +0.249 | +0.314 | -0.003 | 276 |
| 80 | +0.283 | +0.312 | -0.003 | 258 |
| 90 | +0.276 | +0.298 | -0.007 | 272 |
| 100 | +0.293 | +0.314 | -0.017 | 261 |
| 110 | +0.315 | +0.360 | -0.020 | 289 |
| 120 | +0.354 | +0.357 | -0.017 | 309 |
| 130 | +0.375 | +0.401 | -0.010 | 288 |
| 140 | +0.367 | +0.391 | -0.017 | 298 |
| 150 | +0.352 | +0.378 | -0.003 | 293 |
| 160 | +0.386 | +0.373 | -0.017 | 297 |
| 170 | +0.361 | +0.376 | -0.010 | 293 |
| 180 | +0.394 | +0.404 | -0.020 | 290 |
| 190 | +0.395 | +0.420 | -0.017 | 288 |
| 200 | +0.418 | +0.419 | +0.003 | 279 |
| 210 | +0.417 | +0.437 | -0.013 | 282 |
| 220 | +0.417 | +0.419 | -0.007 | 260 |
| 230 | +0.432 | +0.451 | -0.020 | 259 |
| 240 | +0.402 | +0.413 | -0.013 | 255 |
| 250 | +0.416 | +0.438 | -0.023 | 278 |
| 260 | +0.434 | +0.448 | -0.010 | 264 |
| 270 | +0.413 | +0.466 | -0.030 | 268 |
| 280 | +0.410 | +0.428 | -0.017 | 278 |
| 290 | +0.429 | +0.435 | -0.020 | 284 |
| 300 | +0.439 | +0.442 | -0.007 | 284 |
| 310 | +0.429 | +0.399 | -0.017 | 273 |
| 320 | +0.408 | +0.433 | +0.000 | 279 |
| 330 | +0.422 | +0.424 | -0.007 | 282 |
| 340 | +0.402 | +0.453 | -0.027 | 287 |
| 350 | +0.392 | +0.411 | -0.017 | 283 |
| 360 | +0.412 | +0.412 | -0.033 | 299 |
| 370 | +0.415 | +0.426 | -0.010 | 278 |
| 380 | +0.411 | +0.406 | -0.013 | 286 |
| 390 | +0.414 | +0.397 | -0.017 | 296 |
| 400 | +0.415 | +0.416 | -0.017 | 279 |
| 410 | +0.387 | +0.413 | -0.023 | 284 |
| 420 | +0.403 | +0.434 | -0.013 | 288 |
| 430 | +0.423 | +0.427 | -0.010 | 292 |
| 440 | +0.426 | +0.438 | +0.000 | 305 |
| 450 | +0.397 | +0.393 | -0.007 | 286 |
| 460 | +0.427 | +0.421 | -0.010 | 281 |
| 470 | +0.405 | +0.426 | -0.020 | 280 |
| 480 | +0.432 | +0.430 | -0.017 | 274 |

## Against the preregistration

1. *The early ratio persists*: held on both settings, at every milestone, by more than two
   paired standard errors against every arm (the smallest margin is the 1.5B un-anchored
   control at 307,200 samples, 2.2 standard errors; at 368,640 it is 2.7).
2. *GRPO passes the best ES anywhere*: held. 3B +0.430 at 480 steps against the anchored
   arm's +0.235 peak; 1.5B +0.250 against the un-anchored arm's +0.183.
3. *No decline*: held. 3B +0.413 at 240 steps and +0.430 at 480 (peak +0.466 at 270);
   1.5B +0.247 and +0.250 (peak +0.265 at 300).
4. *Retention within 0.03 at every evaluation*: held on 1.5B (lowest −0.010); on 3B at 48
   of 49 evaluations, the one miss −0.033 at step 360 (ending at −0.017). A loss of a few
   hundredths on GSM8K over 480 steps, against the ES arms' 0.84 at the prescribed step.
5. *1e-6 beats 5e-7 on 1.5B at 40 steps*: held (+0.207 against +0.181 on validation).

**What it says.** At equal samples seen, GRPO is the more data-efficient on both settings
at every point measured, and by the time the ES arms end GRPO has reached a level no ES
arm reached: in absolute terms the 3B reserved block is at 0.565 (peak 0.601) and the 1.5B
one at 0.324, against ES's best 0.370 and 0.257. The stop rule of the preregistration is
met: stage 2 runs on 3B instruct only.

**The comparator question.** Qiu et al. (2025) report, at 3 million samples and over
three seeds, ES 58.7 ± 1.5 and their best RL variant 44.4 ± 1.9 on Qwen2.5-3B-Instruct,
and ES 37.7 ± 1.1 against 33.0 ± 2.1 on 1.5B (their Appendix A.4; test sets differ from
ours, 2,000 problems against 300 from the same source). Our GRPO on 3B, with the Tulu
recipe and a KL anchor, is at 0.601 after 207,360 samples, a fourteenth of their budget,
above their ES's end point and their RL's. On 1.5B ours plateaus at 0.324 from about
46,000 samples on, between their two RL variants and below their ES. The 3B result says
their RL side is far below what the same model reaches under another recipe; the 1.5B
plateau is the one place where their ES's end point is above our GRPO's, and whether that
plateau is the task's or the KL anchor's (beta 0.01 to the start; the S4a arm without it
learned more at 40 steps on 3B) is untested. A 480-step arm at beta 0 on 1.5B costs about
$45.

## Limitations

One seed per arm. The comparator is one recipe at the better of two learning rates, not a
tuned one; its KL term may set the 1.5B plateau. The blocks are 300 prompts, the test
block never decoded before the runs; the off-task block was decoded by the probes. Both
runs were resumed once from a checkpoint; the resumed evaluation at step 40 reproduced the
first attempt's, but the step-by-step path after 40 is not the first attempt's. The
equal-samples readings are at six milestones of a 368,640-sample horizon, a hundredth of
the published 3-million-sample budget on the ES side.
