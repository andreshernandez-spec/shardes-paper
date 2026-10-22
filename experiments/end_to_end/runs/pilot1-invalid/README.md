# Tulu 3.1 pilot, first run: invalid, kept as evidence

Four arms (`tulu-pilot-{s5e-4,s1e-3,s2e-3,random}`), commit 2eb8506, three Secure Cloud
H200 SXM pods ($4.59/h; A: s1e-3 then random, US-NC-1; B: s5e-4 and C: s2e-3, US-CO-1),
2026-09-26 22:17 to 2026-09-27 00:46 UTC, about $34. 30 iterations per arm, every run
exited 0, every recorded bit-for-bit check passed. Logs: `pod-{A,B,C}-log.txt.gz`.

**Why invalid.** The primary metric, `center_reward`, collapsed to 0.3-0.8 at 4 to 7 of
30 iterations in every arm, the random control included, with center responses of
900-1,600 tokens against ~320 normally, and recovered at the next iteration. Member
evaluations never collapsed (no member below 2 in any arm; member lengths normal). The
center runs right after `es_restore`, which follows `es_tell`'s seconds of queued JAX
work; the JAX to torch weight handoff did not wait for JAX to finish each leaf nor keep
it alive until torch had copied it, so some restores wrote wrong weights. The checks
missed it because they ran only at iteration 0. `es_vllm/race_check.py` reproduced it on
the laptop (4 of 20 restores wrong); with the fix in `worker.py` (block_until_ready per
leaf, one leaf per load, synchronize before release) it gave 0 of 40, and the driver now
checks the engine after every restore.

`gate-as-computed.json` is the preregistered gate applied to these logs as they are: no
arm passes. It is recorded, not used: the measurement it rests on was corrupted at known
iterations, and dropping those iterations after seeing them would be a rule chosen from
the data. The pilot is rerun under the same preregistration and configs.

The master weights were not affected (the update and its digests are computed in JAX),
so the fitness logs here are valid ES trajectories whose center measurements are not.
