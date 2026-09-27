# Tulu 3.1 pilot: results (the rerun), 2026-09-27

Preregistration: `docs/end_to_end/03-tulu31-pilot.md` (with its two amendments). Four
arms, commit fc55757, three Secure Cloud H200 SXM pods in US-NC-1 ($4.59/h; A: s1e-3 then
random, B: s5e-4, C: s2e-3), 00:52 to about 03:20 UTC, about $34. Every run exited 0 with
30 iterations; the engine was checked bit for bit after every restore and on member 0
every ten iterations, 33 checks per true-reward arm and 30 for the control, all passed.
Logs: `pod-{A,B,C}-log.txt.gz`. The first, invalid run is in `pilot1-invalid/`.

## Gate G3 (`python -m es_vllm.pilot_gate`, `pilot-gate.json`)

| arm | slope of d (per iteration) | SE | slope / SE | mean d, iterations 20-29 | pass |
|---|---|---|---|---|---|
| s5e-4 | +0.0084 | 0.0049 | 1.7 | +0.125 | no |
| s1e-3 | +0.0040 | 0.0043 | 0.9 | +0.104 | no |
| s2e-3 | +0.0057 | 0.0061 | 0.9 | +0.182 | no |

`d` is an arm's held-out reward minus the random control's at the same iteration, on the
0-10 scale. **No arm passes: the preregistered outcome is "no detectable learning within
30 iterations (RL steps 1 to 120's data)".** No sigma is selected.

## What else the logs show

Held-out reward of the current weights, mean over iterations 0-9 / 10-19 / 20-29:

| arm | 0-9 | 10-19 | 20-29 |
|---|---|---|---|
| random | 5.365 | 5.443 | 5.359 |
| s5e-4 | 5.302 | 5.521 | 5.484 |
| s1e-3 | 5.396 | 5.604 | 5.464 |
| s2e-3 | 5.422 | 5.630 | 5.542 |

- Every true-reward arm ends above the control, by 1 to 2 points of accuracy on the
  late iterations, but the pilot's power was low: with the measured noise a slope had to
  exceed about 0.01 per iteration (0.3 over the run) to pass.
- The members' mean fitness falls with sigma: 5.44 (5e-4), 5.24 (1e-3), 3.90 (2e-3),
  against a center near 5.4-5.5; at 2e-3 the perturbations visibly hurt, with 1,773
  capped responses against 1,034-1,060 at the smaller sigmas. Member fitness spread
  (median sd across members) 0.17 / 0.19 / 0.28, never zero, so every update had signal.
- Center responses stayed at a median of 313-320 tokens, max 368; no collapse.
- Iteration 0 reproduced the invalid run exactly in every arm (center 5.000; member
  fitness 4.808, 5.042, 3.695), across different hosts: greedy evaluation on one GPU type
  is deterministic here.
- An iteration took about 255 s on an H200 (17 decodes of 192 prompts, 16 member writes,
  the update, the checks).
