# S6 arm B: our ES on Qiu et al.'s Countdown setup

Preregistered in `docs/end_to_end/12-qiu-matched.md` (f7e4012, the stop rule at 6ce8136).
Every number here is printed by `python -m es_vllm.qiu_match_table runs/qiu-matched`.

## What ran

`es_vllm/qiu_match.py` at f7e4012 (shardes bc39579), seed 1, 500 updates, through
`es_vllm/qiu_match_pod.sh` on one H200 (RunPod Secure, US-NC-1, pod `z5ut08ie7p48ry`),
2026-11-01 14:35 to 2026-11-02 18:36 UTC: 28.0 hours, about $148. Their setup, from
es-at-scale's `archive/` at 574a9d1: the first 200 problems of `countdown.json` for
training, the last 2,000 for testing, at most 1,024 new tokens, greedy decoding with the
model's repetition penalty (1.05) including HF's penalty on left padding, their reward on
the decoded prompt and answer, 30 members, sigma 1e-3, alpha 5e-4. Checkpoints after 0,
100, ..., 500 updates were scored by their `eval_countdown.py`, unchanged.

All 500 updates passed the bit-for-bit engine checks; no grade timed out; the median
update took 190 s. Andres approved finishing past the preregistered $150 cap (projected
$152 at update 40; the run came in under it).

Records: `qwen3b-s1/` (`run.json`, `log.jsonl` with every member's reward on every
training problem, `eval.jsonl` with every test answer's grade every 10 updates,
`ckpt.jsonl` with the checkpoint digests, `decode-check-{fp32,bf16}.json`, and
`qiu-eval/ckpt-<k>/` with their evaluator's summary and per-problem results).
`trace-his.json` and `trace-his-raw.json` are one traced iteration of their trainer.

## Checks before the run

**Decode check** (200 training prompts, base weights, HF `generate` as their trainer
calls it against our vLLM rule):

| | identical to HF | answers part at (median token) | accuracy |
|---|---|---|---|
| float32, our matched rule | 151 of 200 | 199 | 0.125 (HF 0.120) |
| float32, no pad penalty | 115 | 276 | 0.125 |
| float32, no repetition penalty (S5's decoding) | 0 | 9 | 0.105 |
| bf16, HF against itself in batches of 50 | 1 | | |
| bf16, our matched rule | 5 | 51 | 0.105 (HF 0.110) |
| bf16, no repetition penalty | 0 | 7 | 0.085 |

In float32 our rule agrees with theirs as often as their decoding agrees with itself in
other batches (153 of 200); in bf16 neither reproduces itself token for token. Both
checks passed.

**Their evaluator on checkpoint 0**: 9.55% (their Table 1: 10.0%; the gate was 8.5 to 11.5).

## Result

| updates | their evaluator | our running test | the two differ on | Qiu et al. (Fig. 7) |
|---|---|---|---|---|
| 0 | 9.55% | 8.35% | 246 of 2,000 | 10.0% (Table 1) |
| 100 | 45.05% | 45.45% | 126 | 35.6% |
| 200 | 49.90% | 50.60% | 158 | 41.2% |
| 300 | 53.70% | 52.80% | 148 | 48.0% |
| 400 | 54.90% | 55.45% | 159 | 54.2% |
| 500 | **56.50%** | 56.55% | 131 | **60.5%** |

Their Figure 7 values are read off the plot, to about half a point; their three seeds
end at 58.67% (standard deviation 1.47, Table 11).

**The preregistered reading: 56.5% is inside the band (55.7% to 61.6%), so our ES
reproduces theirs on their setup.** It ends 2.2 points below their three-seed mean and
4.0 below the run they plot, with one seed of ours. S5's 0.496 on our earlier setup came
from the setup: the same implementation with their problems, token cap, decoding, reward
and evaluator ends 6.9 points higher (on 2,000 test problems against S5's 300).

The shapes differ. Ours leads their curve by 9.5 points at 100 updates and 8.7 at 200;
the curves meet near 400, and theirs ends higher. The update-200 stop rule
(`docs/end_to_end/12-qiu-matched.md`) found the tests still rising (0.48 points per 10
updates, projected 64.7% at 500) and the run continued.

## Why ours slows

| updates | test, points per 10 updates | test answers gained, lost | training problems solved at the end |
|---|---|---|---|
| 0 to 100 | +3.62 | 783, 41 | 0.440 |
| 100 to 200 | +0.48 | 197, 94 | 0.539 |
| 200 to 300 | +0.24 | 147, 103 | 0.585 |
| 300 to 400 | +0.26 | 143, 90 | 0.617 |
| 400 to 500 | +0.20 | 117, 95 | 0.639 |

- About 7% of answers sit on a knife edge. On identical weights their evaluator (HF,
  float16) and ours (vLLM, bf16) differ on 126 to 159 of the 2,000 problems after training,
  in both directions about equally. After 100 updates, every 100 updates swap a similar
  number of test answers each way and net 1 to 5 points.
- The members are ranked by such flips. Each member's answers depart from the majority of
  the 30 on 15 to 23 of the 200 training problems; members differ by a net of 4 to 6
  problems (one standard deviation), and fitness follows that net with correlation 0.99
  to 1.00.

| update | answers off the majority (of 200) | sd of members' net | corr(fitness, net) | problems that split the members |
|---|---|---|---|---|
| 0 | 16.6 | 6.2 | 0.99 | 128 |
| 100 | 22.5 | 5.2 | 0.99 | 119 |
| 200 | 19.0 | 4.9 | 1.00 | 112 |
| 300 | 16.1 | 4.3 | 1.00 | 101 |
| 400 | 15.3 | 4.4 | 1.00 | 99 |
| 499 | 15.3 | 4.1 | 1.00 | 103 |

- The training problems are not used up: about 100 of the 200 still split the members at
  the end. Training accuracy rises 10.0 points from update 200 to 500 and test accuracy
  (their evaluator) 6.6, so the slowing is in what the updates learn, and a third of the late training gain
  does not reach the test problems.
- The update's size is fixed by the z-scored fitness (`alpha / N` times a sum of 30
  unit-variance terms, about 9.1e-5 per weight per update) whatever the fitness carries,
  so as the signal in the ranking falls the same step carries more random movement. The
  checkpoints showed this directly (each weight's change over 100 updates was the size of a
  random walk of that step, early and late), but that measurement was made on the pod
  and not kept as a record; the checkpoints are gone.

## Their trainer, traced

`es_vllm/qiu_trace.py` ran their `es_fine-tuning_countdown.py` unchanged for one iteration
from the base weights (one H200, four threads, 2,401 s with the hooks against our 190 s per
update) and recorded seeds, rewards and 15 matrices. Their update, recomputed from the
seeds and rewards, matches their weights bit for bit.

- Their noise is one draw per tensor shape: the update they meant is the same matrix in
  layers 0, 17 and 35 and in q_proj and o_proj (cosine 1.0000). Counted from the model's tensor shapes,
  their search has about 361M independent coordinates (50M in the transformer layers)
  against our 3,086M.
- In bf16, 35 to 51% of the matrices' elements do not move; the rest round to a whole
  bf16 step, so the update that lands is 1.03 to 1.05 times the meant one in size, at
  cosine 0.87 to 0.94 with it. Their step is not smaller than ours.
- Layer-norm weights never move (0.1% of elements at most).
- Adding and subtracting the members' noise leaves 22 to 29% of the matrices' elements
  changed, a residue 0.47 to 0.62 times the update's size per iteration.

Whether the shared noise is what keeps their curve rising was not tested (a branch from
update 300 with their noise in our code was proposed and not run). Their trainer at this
speed would take about 330 hours for 500 iterations, so arm A was dropped.

## Cost

Arm B $148; the trace $5.6 (two Community H200s with a failed GPU, $0.66, then a Secure
one).
