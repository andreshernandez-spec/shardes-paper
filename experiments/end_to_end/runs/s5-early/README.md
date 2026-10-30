# S5, the early history under the published recipe (docs/end_to_end/10, preregistered 2026-10-30)

One run on 2026-10-30 with `es_vllm/probe_run.py --full-rank`: es-at-scale's Countdown
recipe (Qiu et al. 2025: 30 full-rank Gaussian members on every weight, sigma 1e-3,
alpha 5e-4, the fitness z-scored, no anchor) on Qwen2.5-3B-Instruct, every member decoding
the first probe's 192 train prompts greedily at every update (5,760 samples per update),
20 updates (115,200 samples), the validation, test and off-task blocks scored after every
update. The train prompts and the blocks are those of the S5 GRPO arm
(`runs/s5/qwen3b-countdown-grpo-lr1e-6-s0`), checked by digest. The question: does ES lead
GRPO at any point early in training, at equal samples, on the same problems?

    python -m es_vllm.probe_run --setting qwen3b/countdown --prediction runs/probe/qwen3b-countdown \
        --full-rank --members 30 --iterations 20 --eval-every 1 --seed 0 \
        --out runs/s5-early/qwen3b-countdown-full-rank-s0

One Secure H200 (RunPod pod `kkq5pidam5sypf`, $5.29/h; Community had no stock), 16:30 to
17:22 UTC, about $4.50 by uptime: 10.5 minutes of bootstrap, then 40 minutes for the run.
The pod ran commit 7308118 with shardes bc39579 from the pin; `run.json` stamps both.
Member 0 and the restored view matched the f32 master bit for bit at every update. An
update took 102 to 116 s (82 to 91 s of it decoding), an evaluation about 11 s.

Files: `run.json` (configuration, block sizes and digests, environment; no prediction,
since the probe's is for rank-1 members), `log.jsonl` (per update: every member's reward
on every prompt, the fitness, the digest of the weights, the checks), `eval.jsonl` (per
evaluation: every prompt's reward on the three blocks).

## At equal samples seen

`python -m es_vllm.probe_run --samples runs/s5/qwen3b-countdown-grpo-lr1e-6-s0 runs/s5-early`.
The preregistered milestones are the multiples of 23,040 (ES update 4m, GRPO step 30m);
the rows at 97,920 and 109,440 are paired within the tool's 2% tolerance (GRPO at 99,840
and 107,520), as in stage 1.

| ES run | samples seen | ES update | GRPO step | ES test | GRPO test | ES minus GRPO | paired SE | ahead | ES off-task | GRPO off-task |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3b-countdown-full-rank-s0 | 23,040 | 4 | 30 | +0.062 | +0.199 | -0.138 | +0.026 | GRPO | -0.007 | -0.013 |
| qwen3b-countdown-full-rank-s0 | 46,080 | 8 | 60 | +0.059 | +0.275 | -0.216 | +0.027 | GRPO | -0.033 | +0.000 |
| qwen3b-countdown-full-rank-s0 | 69,120 | 12 | 90 | +0.079 | +0.298 | -0.219 | +0.025 | GRPO | -0.027 | -0.007 |
| qwen3b-countdown-full-rank-s0 | 92,160 | 16 | 120 | +0.072 | +0.357 | -0.284 | +0.028 | GRPO | -0.007 | -0.017 |
| qwen3b-countdown-full-rank-s0 | 97,920 | 17 | 130 | +0.076 | +0.401 | -0.325 | +0.028 | GRPO | -0.023 | -0.010 |
| qwen3b-countdown-full-rank-s0 | 109,440 | 19 | 140 | +0.113 | +0.391 | -0.277 | +0.030 | GRPO | -0.013 | -0.017 |
| qwen3b-countdown-full-rank-s0 | 115,200 | 20 | 150 | +0.087 | +0.378 | -0.291 | +0.029 | GRPO | -0.017 | -0.003 |

## The early history

`python -m es_vllm.probe_run --history runs/s5/qwen3b-countdown-grpo-lr1e-6-s0
runs/s5-early/qwen3b-countdown-full-rank-s0 --upto 115200`: every evaluation of both runs,
each against its own start, with the loop seconds before it (an ES update, a GRPO step's
phases; no evaluation or setup; both on H200s, on different pods).

| samples seen | loop seconds | run | update or step | test | SE | validation | off-task |
|---|---|---|---|---|---|---|---|
| 0 | 0 | qwen3b-countdown-full-rank-s0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | qwen3b-countdown-grpo-lr1e-6-s0 | 0 | 0 | 0 | 0 | 0 |
| 5,760 | 116 | qwen3b-countdown-full-rank-s0 | 1 | +0.021 | +0.019 | +0.024 | -0.017 |
| 7,680 | 855 | qwen3b-countdown-grpo-lr1e-6-s0 | 10 | +0.113 | +0.023 | +0.115 | -0.007 |
| 11,520 | 221 | qwen3b-countdown-full-rank-s0 | 2 | +0.042 | +0.018 | +0.014 | -0.023 |
| 15,360 | 1,672 | qwen3b-countdown-grpo-lr1e-6-s0 | 20 | +0.129 | +0.021 | +0.143 | -0.013 |
| 17,280 | 325 | qwen3b-countdown-full-rank-s0 | 3 | +0.039 | +0.020 | +0.055 | -0.027 |
| 23,040 | 429 | qwen3b-countdown-full-rank-s0 | 4 | +0.062 | +0.021 | +0.039 | -0.007 |
| 23,040 | 2,486 | qwen3b-countdown-grpo-lr1e-6-s0 | 30 | +0.199 | +0.025 | +0.161 | -0.013 |
| 28,800 | 533 | qwen3b-countdown-full-rank-s0 | 5 | +0.067 | +0.020 | +0.068 | -0.023 |
| 30,720 | 3,304 | qwen3b-countdown-grpo-lr1e-6-s0 | 40 | +0.216 | +0.025 | +0.197 | -0.010 |
| 34,560 | 636 | qwen3b-countdown-full-rank-s0 | 6 | +0.076 | +0.021 | +0.073 | -0.027 |
| 38,400 | 4,142 | qwen3b-countdown-grpo-lr1e-6-s0 | 50 | +0.210 | +0.024 | +0.193 | -0.010 |
| 40,320 | 739 | qwen3b-countdown-full-rank-s0 | 7 | +0.072 | +0.022 | +0.081 | -0.023 |
| 46,080 | 841 | qwen3b-countdown-full-rank-s0 | 8 | +0.059 | +0.021 | +0.076 | -0.033 |
| 46,080 | 4,991 | qwen3b-countdown-grpo-lr1e-6-s0 | 60 | +0.275 | +0.026 | +0.210 | 0 |
| 51,840 | 945 | qwen3b-countdown-full-rank-s0 | 9 | +0.054 | +0.023 | +0.072 | -0.027 |
| 53,760 | 5,842 | qwen3b-countdown-grpo-lr1e-6-s0 | 70 | +0.314 | +0.028 | +0.249 | -0.003 |
| 57,600 | 1,048 | qwen3b-countdown-full-rank-s0 | 10 | +0.067 | +0.022 | +0.089 | -0.037 |
| 61,440 | 6,692 | qwen3b-countdown-grpo-lr1e-6-s0 | 80 | +0.312 | +0.028 | +0.283 | -0.003 |
| 63,360 | 1,151 | qwen3b-countdown-full-rank-s0 | 11 | +0.055 | +0.021 | +0.101 | -0.027 |
| 69,120 | 1,254 | qwen3b-countdown-full-rank-s0 | 12 | +0.079 | +0.023 | +0.090 | -0.027 |
| 69,120 | 7,546 | qwen3b-countdown-grpo-lr1e-6-s0 | 90 | +0.298 | +0.027 | +0.276 | -0.007 |
| 74,880 | 1,357 | qwen3b-countdown-full-rank-s0 | 13 | +0.058 | +0.021 | +0.073 | -0.030 |
| 76,800 | 8,401 | qwen3b-countdown-grpo-lr1e-6-s0 | 100 | +0.314 | +0.027 | +0.293 | -0.017 |
| 80,640 | 1,459 | qwen3b-countdown-full-rank-s0 | 14 | +0.068 | +0.022 | +0.081 | -0.007 |
| 84,480 | 9,254 | qwen3b-countdown-grpo-lr1e-6-s0 | 110 | +0.360 | +0.028 | +0.315 | -0.020 |
| 86,400 | 1,561 | qwen3b-countdown-full-rank-s0 | 15 | +0.091 | +0.023 | +0.095 | -0.020 |
| 92,160 | 1,663 | qwen3b-countdown-full-rank-s0 | 16 | +0.072 | +0.022 | +0.081 | -0.007 |
| 92,160 | 10,106 | qwen3b-countdown-grpo-lr1e-6-s0 | 120 | +0.357 | +0.029 | +0.354 | -0.017 |
| 97,920 | 1,766 | qwen3b-countdown-full-rank-s0 | 17 | +0.076 | +0.022 | +0.096 | -0.023 |
| 99,840 | 10,964 | qwen3b-countdown-grpo-lr1e-6-s0 | 130 | +0.401 | +0.028 | +0.375 | -0.010 |
| 103,680 | 1,868 | qwen3b-countdown-full-rank-s0 | 18 | +0.099 | +0.023 | +0.117 | -0.003 |
| 107,520 | 11,816 | qwen3b-countdown-grpo-lr1e-6-s0 | 140 | +0.391 | +0.030 | +0.367 | -0.017 |
| 109,440 | 1,971 | qwen3b-countdown-full-rank-s0 | 19 | +0.113 | +0.024 | +0.097 | -0.013 |
| 115,200 | 2,074 | qwen3b-countdown-full-rank-s0 | 20 | +0.087 | +0.023 | +0.086 | -0.017 |
| 115,200 | 12,673 | qwen3b-countdown-grpo-lr1e-6-s0 | 150 | +0.378 | +0.028 | +0.352 | -0.003 |

`python -m es_vllm.probe_run --table runs/s5-early` (the classes of `10`). The collapse
flag is set by the off-task (GSM8K) block: 0.023 to 0.037 below its start at 12 of the 20
evaluations, 2 to 3 of its standard errors (about 0.012), and −0.017 at the end; GRPO's
stayed between 0 and −0.020 over the same samples.

| run | preregistered call | step | variant | updates to target, predicted | updates | validation at end | test at end | off-task at end | peak test gain | at | validation-selected checkpoint | test there | class | class by test peak | class, task alone | collapsed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3b-countdown-full-rank-s0 | - | 1 | full-rank, N 30 | - | 20 | +0.086 | +0.087 | -0.017 | 0.113 | 19 | 18 | 0.0991 | marginal | works | works | yes |

## Against the preregistration

1. *GRPO ahead from the start*: held. GRPO is ahead at all five milestones, by 0.14 to
   0.29 (5.3 to 10.1 paired standard errors), and at the two near-matches by 0.33 and 0.28
   (11.8 and 9.4).
2. *No early window at all*: failed, by 0.0003, far below the block's standard error.
   ES's test change reached +0.1135 at update 19 (109,440 samples) against GRPO's +0.1132
   at its first evaluation (7,680 samples); the next evaluation was back at +0.087. In the preregistered reading, ES reaches
   GRPO's first level within 115,200 samples, at 14 times GRPO's samples.

## Reading

With the published recipe in our code, on the same 192 problems, ES is behind GRPO at
every point of the first 115,200 samples, from the first evaluation on: +0.021 after
5,760 samples (about one standard error) against GRPO's +0.113 after 7,680. The recipe
learns (the members' mean training reward 0.142 to 0.230, the reserved block +0.054 to
+0.113 from update 4 on), but it took 14 times GRPO's samples to reach GRPO's first
level, and GRPO had passed +0.3 by then. At equal loop seconds GRPO is also ahead: ES's
member decodes are cheap (0.018 s per sample against GRPO's 0.110), but GRPO's first
evaluation (+0.113 after 855 s) is above every ES evaluation before update 19 (1,971 s),
and ES at 841 s (update 8) is at +0.059.

So there is no early window on this setting, and Qiu et al.'s "throughout training" lead
does not come from the regime (a few hundred problems reused, equal samples), which this
run and stage 1 reproduce. It comes from their RL baseline, which ended at 32.5 to 44.4
on this model after 3 million samples, against our GRPO's 0.565 (a change of +0.43) after
368,640, on test sets of different size from the same source.

## Limitations

One seed. 20 of Qiu et al.'s 500 iterations, 115,200 of their 3 million samples: whether
ES overtakes later is not tested here (stage 1's GRPO curve is at its plateau of +0.41 to
+0.47 from 240 steps, 184,320 samples). 192 training problems against their 200, and a
300-prompt test block against their 2,000 (the standard error of a change is about 0.02).
The loop seconds compare two unoptimized drivers on two pods; GRPO's step is mostly its
training phase (66 of 86 s).
