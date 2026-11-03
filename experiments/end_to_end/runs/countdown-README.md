# T2, the Countdown check against es-at-scale: results, 2026-10-06

Preregistration: `docs/end_to_end/06-countdown-check.md` (3b5e5ac, before any run). Runs
from 9bf6b96: our backend (`es_vllm/run_countdown.py`, vLLM 0.30.0, jax 0.11.2, shardes
bc39579) and es-at-scale's own trainer at 574a9d1 (`es_vllm/countdown_ref.py`, vLLM 0.11.0,
torch 2.8.0, ray 2.59.0), Qwen2.5-0.5B-Instruct at 7ae5576, two seeds each, settings from
one YAML (`es_vllm/countdown-s{1,2}.yaml`). Two Community Cloud pods with 2x A100 SXM
80GB each (.78/h, drivers 590.48.01 and 595.71.05), one per seed, ours on GPU 0 and
es-at-scale on GPU 1 side by side (`es_vllm/countdown_pod.sh`), 14:44 to about 18:01 UTC,
about 8 by uptime. Every run completed: ours 100 updates, es-at-scale 101 (it makes one
after its last evaluation), 21 evaluations each on the 2,000 `countdown_eval` prompts.

## Gate G2: fail, on the high side

`python -m es_vllm.countdown_gate` (`countdown-gate.json`): mean evaluation reward over
the 20 evaluations after 5 to 100 updates is 0.1824 for ours (seeds 0.1820 and 0.1829)
and 0.1587 for es-at-scale (0.1587 and 0.1586). The difference, +0.0238, exceeds the bound
of 0.0200 (twice the 0.01 floor; the seed spreads, 0.0009 and 0.0001, are below it). The
reference learned (gains 0.185 and 0.176 from its start to its last four evaluations), so
the check is not inconclusive. As preregistered, ours learning faster than the reference
fails the gate too, and the next step is decided with Andres.

What it shows either way: **our backend learns this task, faster than the reference
implementation and to the same level.** By the last four evaluations (85 to 100 updates)
the two are within 0.02 (ours 0.234 and 0.239, es-at-scale 0.229 and 0.220); the gap is
speed. Ours first reaches 0.20 eval reward at 45 and 50 updates, es-at-scale at 65 and 70.

Evaluation reward (`python -m es_vllm.countdown_gate --table reward`):

| updates | ours s1 | ours s2 | es-at-scale s1 | es-at-scale s2 |
|---|---|---|---|---|
| 0 | 0.0412 | 0.0412 | 0.0442 | 0.0442 |
| 5 | 0.0582 | 0.0491 | 0.0571 | 0.0571 |
| 10 | 0.0610 | 0.0591 | 0.0580 | 0.0592 |
| 15 | 0.0703 | 0.0605 | 0.0582 | 0.0683 |
| 20 | 0.0874 | 0.1006 | 0.0590 | 0.0819 |
| 25 | 0.1305 | 0.1089 | 0.0868 | 0.1037 |
| 30 | 0.1699 | 0.1496 | 0.1045 | 0.1224 |
| 35 | 0.1790 | 0.1524 | 0.1271 | 0.1380 |
| 40 | 0.1835 | 0.1973 | 0.1575 | 0.1667 |
| 45 | 0.2101 | 0.1988 | 0.1485 | 0.1701 |
| 50 | 0.2179 | 0.2235 | 0.1749 | 0.1741 |
| 55 | 0.2188 | 0.2201 | 0.1900 | 0.1693 |
| 60 | 0.2143 | 0.2306 | 0.1927 | 0.1750 |
| 65 | 0.2230 | 0.2337 | 0.2082 | 0.1849 |
| 70 | 0.2266 | 0.2391 | 0.2041 | 0.2022 |
| 75 | 0.2268 | 0.2388 | 0.2123 | 0.2107 |
| 80 | 0.2277 | 0.2394 | 0.2188 | 0.2098 |
| 85 | 0.2329 | 0.2305 | 0.2156 | 0.2203 |
| 90 | 0.2319 | 0.2355 | 0.2241 | 0.2208 |
| 95 | 0.2342 | 0.2458 | 0.2404 | 0.2168 |
| 100 | 0.2363 | 0.2442 | 0.2350 | 0.2230 |

Solve rate, answer reward above 0 (`--table solved`):

| updates | ours s1 | ours s2 | es-at-scale s1 | es-at-scale s2 |
|---|---|---|---|---|
| 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 5 | 0.0005 | 0.0010 | 0.0000 | 0.0005 |
| 10 | 0.0015 | 0.0010 | 0.0000 | 0.0010 |
| 15 | 0.0075 | 0.0030 | 0.0005 | 0.0080 |
| 20 | 0.0250 | 0.0410 | 0.0015 | 0.0100 |
| 25 | 0.0635 | 0.0460 | 0.0100 | 0.0350 |
| 30 | 0.1005 | 0.0870 | 0.0145 | 0.0490 |
| 35 | 0.1110 | 0.0765 | 0.0305 | 0.0670 |
| 40 | 0.1045 | 0.1145 | 0.0580 | 0.1005 |
| 45 | 0.1195 | 0.1120 | 0.0495 | 0.0940 |
| 50 | 0.1275 | 0.1350 | 0.0765 | 0.1035 |
| 55 | 0.1250 | 0.1325 | 0.0910 | 0.1015 |
| 60 | 0.1195 | 0.1410 | 0.0945 | 0.1070 |
| 65 | 0.1280 | 0.1425 | 0.1090 | 0.1130 |
| 70 | 0.1300 | 0.1475 | 0.1060 | 0.1230 |
| 75 | 0.1300 | 0.1490 | 0.1135 | 0.1230 |
| 80 | 0.1325 | 0.1460 | 0.1210 | 0.1140 |
| 85 | 0.1350 | 0.1375 | 0.1170 | 0.1225 |
| 90 | 0.1345 | 0.1420 | 0.1245 | 0.1215 |
| 95 | 0.1350 | 0.1520 | 0.1405 | 0.1180 |
| 100 | 0.1365 | 0.1500 | 0.1350 | 0.1235 |

## The candidates for the difference

From the preregistration's table of differences by design; none was isolated here.

- **es-at-scale adds the update into bf16 weights.** It sums the update in f32 but adds it
  to vLLM's bf16 parameters in place. At alpha / N = 1.7e-5 per unit of the z-weighted
  noise sum, a typical element's update is about 1e-4, comparable to one bf16 step at the
  weights' scale, so many updates round away or to a single step. Ours keeps an f32 master
  and applies every update exactly. This is the most direct candidate for slower learning.
- **es-at-scale restores by subtracting the noise**, which in bf16 leaves a residue after
  every member; ours writes the view back exactly.
- **es-at-scale's noise reuses one seed per member for every tensor**, so tensors of the
  same shape receive the same draw; ours draws each leaf from its own stream.
- vLLM 0.11.0 and 0.30.0 decode the start model slightly differently (start 0.044 against
  0.041); that shifts the curves at the start, not their slope.

## Also measured

- Our runs: every engine check passed (after every restore, 100, and member 0 every ten
  iterations, 10, per run), digests every five iterations. Median iteration 66 and 69 s,
  84% of it decoding, update 0.8 s. es-at-scale: median iteration 98 and 103 s.
- Response length on the evaluation prompts fell from about 390 tokens to 58-194 in every
  run (the format reward is learned first: format score 0.94-1.0 at the end).
- Step 1 on these GPUs: `../update-check/README.md`. The preflight before the runs and the
  kernel check after them found the scan deterministic on these A100s, so the old code our
  runs used was not affected.

## Incidents

- es-at-scale's log ends with seven tracebacks after training completed and the final
  weights were saved: its SIGTERM handler, inherited by its grading pool's workers, runs
  its cleanup as they exit. Exit code 0, every iteration and evaluation present.
- es-at-scale's `run.json` records its command line. The pod's absolute paths in it were
  rewritten at harvest relative to the repository and to ~ (noted in the file), as
  `countdown_ref.py` records them since b304d42: they look like a kernel id to the
  repository's account guard. Nothing else in the records was changed.
- Not harvested: es-at-scale's raw evaluation outputs (2,000 responses per evaluation,
  about 80 MB per run; `eval.jsonl.gz` holds their per-prompt rewards, answers, format
  scores and lengths, made by `countdown_gate.py --compact` on the pod) and its 1 GB final
  checkpoint.
