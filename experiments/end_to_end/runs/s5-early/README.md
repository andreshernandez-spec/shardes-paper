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
368,640, on test sets of different size from the same source. (Corrected 2026-10-31: 0.565
is our share of the maximum reward, which counts format; in their measure, accuracy, our
GRPO's last evaluation is 0.527. The seed 1 section below reads both runs in accuracy.)

## Limitations

One seed. 20 of Qiu et al.'s 500 iterations, 115,200 of their 3 million samples: whether
ES overtakes later is not tested here (stage 1's GRPO curve is at its plateau of +0.41 to
+0.47 from 240 steps, 184,320 samples). 192 training problems against their 200, and a
300-prompt test block against their 2,000 (the standard error of a change is about 0.02).
The loop seconds compare two unoptimized drivers on two pods; GRPO's step is mostly its
training phase (66 of 86 s).

# Seed 1, to 100 updates (docs/end_to_end/10, preregistered 2026-10-31 at 79fd8e2)

The same recipe and blocks with member seed 1, run to 100 updates (576,000 samples), so
that it reaches the first point of Qiu et al.'s Qwen-3B training curve (their Figure 7:
ES 35.6% accuracy at 0.6M sample evaluations, 100 iterations on 200 problems) and its
first 20 updates are a second seed of the early history.

    python -m es_vllm.probe_run --setting qwen3b/countdown --prediction runs/probe/qwen3b-countdown \
        --full-rank --members 30 --iterations 100 --eval-every 1 --seed 1 \
        --out runs/s5-early/qwen3b-countdown-full-rank-s1

One Secure H200 (RunPod pod `fsvvbextdnnzb9`, US-NC-1, $5.29/h; Community had no stock),
created 07:43:45 UTC and deleted at 10:56, about $17 by uptime: under 5 minutes to the
end of bootstrap (07:48:26), then 3 h 6 min for the run (finished 10:54:13). The pod ran
commit 79fd8e2 with shardes bc39579 from the pin; `run.json` stamps both, with a clean
worktree. Member 0 and the restored view matched the f32 master bit for bit at all 100
updates. An update took 96 to 109 s (82 to 91 s of it decoding), an evaluation 10 to
15 s. The harvested `run.json`, `log.jsonl` and `eval.jsonl` match the pod's by sha256.

The measure is test accuracy, the share of the 300 test problems answered correctly (a
reward of at least 1; format adds at most 0.1), Qiu et al.'s measure. The base model
scores 0.083 here and 10.0% on their 2,000 problems from the same pool.

## Accuracy by update

`python -m es_vllm.probe_run --history runs/s5-early/qwen3b-countdown-full-rank-s1`, every
tenth evaluation, the first at or above 0.43 (update 74) and the best (update 95). Test,
validation and off-task are changes in the share of the maximum reward, as above.

| samples seen | loop seconds | run | update or step | test | SE | test accuracy | accuracy SE | validation | off-task |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | qwen3b-countdown-full-rank-s1 | 0 | 0 | 0 | 0.083 | 0.016 | 0 | 0 |
| 57,600 | 1,001 | qwen3b-countdown-full-rank-s1 | 10 | +0.053 | +0.021 | 0.133 | 0.020 | +0.082 | -0.007 |
| 115,200 | 1,974 | qwen3b-countdown-full-rank-s1 | 20 | +0.135 | +0.025 | 0.220 | 0.024 | +0.123 | -0.013 |
| 172,800 | 2,943 | qwen3b-countdown-full-rank-s1 | 30 | +0.144 | +0.025 | 0.243 | 0.025 | +0.168 | -0.010 |
| 230,400 | 3,917 | qwen3b-countdown-full-rank-s1 | 40 | +0.164 | +0.026 | 0.270 | 0.026 | +0.142 | -0.020 |
| 288,000 | 4,893 | qwen3b-countdown-full-rank-s1 | 50 | +0.216 | +0.028 | 0.333 | 0.027 | +0.245 | -0.017 |
| 345,600 | 5,884 | qwen3b-countdown-full-rank-s1 | 60 | +0.249 | +0.028 | 0.373 | 0.028 | +0.237 | -0.020 |
| 403,200 | 6,873 | qwen3b-countdown-full-rank-s1 | 70 | +0.281 | +0.028 | 0.407 | 0.028 | +0.269 | -0.003 |
| 426,240 | 7,269 | qwen3b-countdown-full-rank-s1 | 74 | +0.302 | +0.028 | 0.430 | 0.029 | +0.264 | -0.003 |
| 460,800 | 7,860 | qwen3b-countdown-full-rank-s1 | 80 | +0.284 | +0.029 | 0.413 | 0.028 | +0.262 | -0.007 |
| 518,400 | 8,850 | qwen3b-countdown-full-rank-s1 | 90 | +0.294 | +0.028 | 0.423 | 0.029 | +0.278 | -0.003 |
| 547,200 | 9,344 | qwen3b-countdown-full-rank-s1 | 95 | +0.327 | +0.030 | 0.460 | 0.029 | +0.282 | -0.007 |
| 576,000 | 9,838 | qwen3b-countdown-full-rank-s1 | 100 | +0.299 | +0.029 | 0.430 | 0.029 | +0.302 | +0.003 |

Mean accuracy over ten evaluations at a time (`--history ... --window 1 10 --window 61 70`
and so on): 0.129 after updates 1 to 10, 0.396 after 61 to 70, 0.419 after 71 to 80,
0.423 after 81 to 90, 0.433 after 91 to 100. The curve rises about 0.045 per ten updates
to update 70, then about 0.012 (0.396 to 0.433 over the last 30 updates).

## At equal samples seen

`python -m es_vllm.probe_run --samples runs/s5/qwen3b-countdown-grpo-lr1e-6-s0 runs/s5-early`,
the 16 exact milestones of seed 1 (ES update 4m, GRPO step 30m) up to GRPO's last step;
the full output also pairs every other seed 1 evaluation within the tool's 2% tolerance,
all with GRPO ahead.

| ES run | samples seen | ES update | GRPO step | ES test | GRPO test | ES minus GRPO | paired SE | ahead | ES off-task | GRPO off-task |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3b-countdown-full-rank-s1 | 23,040 | 4 | 30 | +0.052 | +0.199 | -0.147 | +0.025 | GRPO | -0.007 | -0.013 |
| qwen3b-countdown-full-rank-s1 | 46,080 | 8 | 60 | +0.085 | +0.275 | -0.190 | +0.027 | GRPO | -0.017 | +0.000 |
| qwen3b-countdown-full-rank-s1 | 69,120 | 12 | 90 | +0.096 | +0.298 | -0.202 | +0.026 | GRPO | -0.003 | -0.007 |
| qwen3b-countdown-full-rank-s1 | 92,160 | 16 | 120 | +0.101 | +0.357 | -0.256 | +0.028 | GRPO | -0.003 | -0.017 |
| qwen3b-countdown-full-rank-s1 | 115,200 | 20 | 150 | +0.135 | +0.378 | -0.244 | +0.029 | GRPO | -0.013 | -0.003 |
| qwen3b-countdown-full-rank-s1 | 138,240 | 24 | 180 | +0.134 | +0.404 | -0.270 | +0.028 | GRPO | -0.003 | -0.020 |
| qwen3b-countdown-full-rank-s1 | 161,280 | 28 | 210 | +0.155 | +0.437 | -0.282 | +0.027 | GRPO | -0.010 | -0.013 |
| qwen3b-countdown-full-rank-s1 | 184,320 | 32 | 240 | +0.150 | +0.413 | -0.264 | +0.027 | GRPO | -0.007 | -0.013 |
| qwen3b-countdown-full-rank-s1 | 207,360 | 36 | 270 | +0.134 | +0.466 | -0.333 | +0.028 | GRPO | -0.020 | -0.030 |
| qwen3b-countdown-full-rank-s1 | 230,400 | 40 | 300 | +0.164 | +0.442 | -0.278 | +0.027 | GRPO | -0.020 | -0.007 |
| qwen3b-countdown-full-rank-s1 | 253,440 | 44 | 330 | +0.173 | +0.424 | -0.250 | +0.026 | GRPO | -0.007 | -0.007 |
| qwen3b-countdown-full-rank-s1 | 276,480 | 48 | 360 | +0.219 | +0.412 | -0.193 | +0.027 | GRPO | +0.000 | -0.033 |
| qwen3b-countdown-full-rank-s1 | 299,520 | 52 | 390 | +0.243 | +0.397 | -0.154 | +0.026 | GRPO | -0.017 | -0.017 |
| qwen3b-countdown-full-rank-s1 | 322,560 | 56 | 420 | +0.252 | +0.434 | -0.182 | +0.026 | GRPO | -0.023 | -0.013 |
| qwen3b-countdown-full-rank-s1 | 345,600 | 60 | 450 | +0.249 | +0.393 | -0.144 | +0.025 | GRPO | -0.020 | -0.007 |
| qwen3b-countdown-full-rank-s1 | 368,640 | 64 | 480 | +0.269 | +0.430 | -0.161 | +0.025 | GRPO | -0.013 | -0.017 |

## At equal loop seconds

Not preregistered. Loop seconds are the update or step time before an evaluation, with no
evaluation or setup, on two H200 pods. Rows from `--history` of each run (GRPO:
`--history runs/s5/qwen3b-countdown-grpo-lr1e-6-s0`), each GRPO evaluation beside the
seed 1 evaluation nearest in loop seconds:

| GRPO loop seconds | GRPO step | GRPO accuracy | ES loop seconds | ES update | ES accuracy |
|---|---|---|---|---|---|
| 855 | 10 | 0.180 | 903 | 9 | 0.147 |
| 1,672 | 20 | 0.197 | 1,682 | 17 | 0.203 |
| 2,486 | 30 | 0.287 | 2,459 | 25 | 0.243 |
| 3,304 | 40 | 0.300 | 3,330 | 34 | 0.267 |
| 4,142 | 50 | 0.293 | 4,112 | 42 | 0.283 |
| 4,991 | 60 | 0.367 | 4,991 | 51 | 0.337 |
| 5,842 | 70 | 0.417 | 5,884 | 60 | 0.373 |
| 6,692 | 80 | 0.410 | 6,674 | 68 | 0.410 |
| 7,546 | 90 | 0.397 | 7,565 | 77 | 0.410 |
| 8,401 | 100 | 0.410 | 8,353 | 85 | 0.427 |
| 9,254 | 110 | 0.470 | 9,245 | 94 | 0.443 |
| 10,106 | 120 | 0.483 | 9,838 | 100 | 0.430 |

ES first reached 0.43 at 7,269 s (update 74). GRPO's first evaluation at or above that
is step 110 at 9,254 s; step 100, at 8,401 s, was 0.410. GRPO went on to 0.570 at step 270
(22,933 s) and was at 0.527 at its last step (41,187 s).

## Classes

`python -m es_vllm.probe_run --table runs/s5-early`. The collapse flag of seed 1 is set by
the off-task (GSM8K) block, which fell more than 0.02 below its start at 10 of the 100
evaluations, by at most 0.037 (about three of its standard errors), and ended 0.003 above
it.

| run | preregistered call | step | variant | updates to target, predicted | updates | validation at end | test at end | off-task at end | peak test gain | at | validation-selected checkpoint | test there | class | class by test peak | class, task alone | collapsed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3b-countdown-full-rank-s0 | - | 1 | full-rank, N 30 | - | 20 | +0.086 | +0.087 | -0.017 | 0.113 | 19 | 18 | 0.0991 | marginal | works | works | yes |
| qwen3b-countdown-full-rank-s1 | - | 1 | full-rank, N 30 | - | 100 | +0.302 | +0.299 | +0.003 | 0.327 | 95 | 100 | 0.299 | works | works | works | yes |

## Against the preregistration

1. *Our ES follows Qiu's curve*: failed, on the high side. Accuracy after update 100 is
   0.430 (standard error 0.029), above the band of 0.30 to 0.41 and 0.074 above their
   0.356, 2.6 of our standard errors. The secondary measure, the mean after updates 91 to
   100, is 0.433. In the preregistered reading this is faster than their run, with the
   caveats listed there: their value is read off a figure; their 2,000 test problems come
   from the pool our 300 come from (base 10.0% there, 0.083 here); their training set has
   200 problems against our 192; and 576,000 samples against their 600,000. From each
   start, ours gained 0.347 and theirs 0.256.
2. *Still below our GRPO*: held. 0.430 is below 0.47. At equal samples GRPO is ahead at
   all 16 milestones, by 0.144 to 0.333 of the range (5.8 to 11.9 paired standard errors);
   in accuracy at 368,640 samples, ES is at 0.397 (update 64) and GRPO at 0.527.
3. *The early history replicates*: held. With seed 1, GRPO is ahead at the five early
   milestones by 0.147 to 0.256 (5.9 to 9.1 paired standard errors), against 0.138 to
   0.291 (5.3 to 10.1) with seed 0. Seed 1's accuracy after 20 updates is 0.220, seed 0's
   0.170.

## Reading

The published recipe in our code learns at least as fast as Qiu et al.'s run: 0.430 at
576,000 samples against their 35.6% at 600,000. The implementation is not what separates
their results from ours; the RL arm is. Our GRPO reaches 0.570 at 207,360 samples, above
every RL point they plot for this model at any budget (at most 45.1% at 1.2M), and above
their ES until somewhere between their 2.4M (54.2%) and 3.0M (60.5%) points, on test sets
of different size from the same pool.

At equal samples ES leads nowhere. It first reaches 0.43 after 426,240 samples; GRPO does
between 76,800 and 84,480, about five times fewer. The gap narrows after GRPO's plateau
(in the share of the maximum reward, from 0.333 at 207,360 samples to 0.161 at 368,640)
because GRPO stops rising and ES does not, although ES has slowed too, to about 0.012 per
ten updates over its last 30.

At equal loop seconds, which was not preregistered, the picture differs. ES's member
decodes are cheap enough that through GRPO's step 110 (9,254 s) the nearest ES evaluation
is between 0.044 below and 0.017 above GRPO's, within about one standard error of the
difference of two evaluations (0.04), and ES reached 0.43 first (7,269 s against GRPO's
9,254 s). With these two drivers the case for ES on this setting is GPU time, not
samples, and only over the 2.7 hours compared: GRPO's later 0.570 came at 22,933 s,
beyond this ES run.

## Limitations

One seed to 100 updates; seed 0 stopped at 20. Their curve point is read off a plot and
comes from one run. 100 of their 500 iterations: whether ES keeps rising to their 60.5%
at 3M, which would put it above our GRPO's best after about 14 times its samples, is not
tested. The loop seconds compare two unoptimized drivers on two pods. GRPO's step is
mostly its training phase, so a faster GRPO trainer or a slower ES decode would move that
comparison.
