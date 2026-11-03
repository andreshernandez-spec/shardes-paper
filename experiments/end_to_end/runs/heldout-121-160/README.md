# Held-out context for the Tulu 3.1 pilot, 2026-09-27

Exploratory, not a gate (`docs/end_to_end/05-tulu31-heldout.md`, committed before the
run). Commit dc9625a, `es_vllm/heldout.py` with `es_vllm/heldout-121-160.yaml`: greedy,
`tulu` template, cap 2,048, the run's verifiers, on the 1,920 prompts of RL steps 121 to
160 (447 MATH, 517 GSM8K, 956 IFEval), new to every model. Two Secure Cloud H200 SXM pods
at $4.59/h: released checkpoints on one (US-CO-1, 05:36 to 06:05 UTC), ES rebuilds on
the other (US-NC-1, 05:36 to 06:34, including the checks below). About $6.6 by uptime.
Logs: `pod-rl-log.txt.gz`, `pod-es-log.txt.gz`. Table: `python -m es_vllm.heldout_table`.

| model | reward | vs DPO | MATH | gsm8k | ifeval | mean len | capped | prompts |
|---|---|---|---|---|---|---|---|---|
| dpo-start | 5.427 +- 0.114 |  | 4.68 | 8.47 | 4.13 | 304 | 19 | 1920 |
| rl-step40 | 5.870 +- 0.112 | +0.443 (+2.8 SE) | 4.63 | 8.67 | 4.94 | 319 | 29 | 1920 |
| rl-step80 | 6.401 +- 0.110 | +0.974 (+6.2 SE) | 5.03 | 8.76 | 5.76 | 344 | 32 | 1920 |
| rl-step120 | 6.688 +- 0.107 | +1.260 (+8.1 SE) | 5.15 | 8.70 | 6.32 | 336 | 29 | 1920 |
| es-s5e-4 | 5.417 +- 0.114 | -0.010 (-0.1 SE) | 4.59 | 8.28 | 4.26 | 297 | 20 | 1920 |
| es-random | 5.391 +- 0.114 | -0.036 (-0.2 SE) | 4.50 | 8.28 | 4.25 | 311 | 23 | 1920 |

Reward on the 0-10 scale. "vs DPO" uses the unpaired standard error: the records keep
means, not per-prompt rewards, so it overstates the error of a difference.

- **The RL run moved detectably by step 40**, a third of the data the pilot's 30
  iterations cover, and by step 120 it is 1.26 above the start (8 SE), mostly IFEval
  (4.13 to 6.32), then MATH (+0.47), GSM8K (+0.23).
- **The s5e-4 arm's final ES weights**, matched to RL step 120 in prompts and rollouts,
  score the same as the start and the random control. The pilot's null (no arm passes
  G3) is therefore not a case of a horizon too short for anything to move: on the same
  measurement RL moved a lot within it.
- **s1e-3 and s2e-3 are not scored.** Their rebuilds never reproduced the pilot's logged
  weights (next section), and the scorer refuses any rebuild whose digest differs from the
  log's last one. Only s5e-4 and the control were rebuilt exactly.

## Rebuilds are not reproducible

`heldout.py` rebuilds an arm by replaying every `es_tell` of its fitness log from the
start weights, then requires the engine to hold the rebuilt view bit for bit and the
view's sha256 to equal the digest the pilot logged after its last update. The ES code is
the pilot's (no change to `stream.py` or `worker.py` since fc55757), with the same
packages and GPU type. Attempts (`pod-es-log.txt.gz`):

| arm | rebuilds matching the log | digests of the others |
|---|---|---|
| s5e-4 | 1 of 2 | 11839e70 |
| s1e-3 | 0 of 5 | c7106cfd, all five |
| s2e-3 | 0 of 5 | 8185e1ce four times, 01a07a45 once |
| random | 1 of 1 | |

What was ruled out, with `diag/diag_rebuild.py` (an ad-hoc check copied to the pod,
kept here with its log `diag/diag-log.txt.gz`) and `nvidia-smi`:

- The load: after every `es_init`, and after a load that waits for each tensor before
  freeing its source, the master equals the checkpoint bit for bit on all 291 leaves.
- The hardware: ECC counters zero on the pod.
- Queueing alone: replays that computed the digest every five iterations (so JAX drained
  its queue) reached all six of s5e-4's logged digests twice, and once parted from them
  between iterations 15 and 19.

So the same replay, from the same bits, can end in different weights. In the scorer,
s1e-3 ended in the same wrong weights five times out of five; under the timing of the
check below it ended in two others. The scored weights are unaffected: every mismatch
was refused, and es-s5e-4 is bit for bit the pilot's final view.

`es_vllm/replay_check.py` (commit ebcd744, `../replay-check/`) replays one arm several
times in one process and records, per iteration, the shaping weights and, per leaf, an
exact checksum of the master and the largest change the update made:

- s5e-4, four replays: all four reach the log's final digest; shaping and every leaf's
  checksum identical at every iteration.
- s1e-3, two replays: neither reaches the log's digest, nor the `c7106cfd` of the scorer's
  five attempts, nor each other's. The shaping weights are identical throughout; the
  masters already differ after the first update, in 3 of 291 leaves, and the count grows
  every iteration (3, 6, 18, 28, ...). An update touches only its own leaf, so each
  iteration adds new leaves that diverged. In the differing leaves the largest change the
  update made is identical in both replays: no blown-up values or NaNs, though wrong
  values of ordinary size are not ruled out.

**So the per-leaf update (`stream.contracted_leaf` / `updated_leaf`) is not
deterministic here** (H200, jax 0.11.2, vLLM's engine resident in the same process):
from identical inputs it intermittently gives different bits, not at all in some
processes and in several leaves per iteration in others. The cause is not found. Two
consequences stand already. The pilot's live runs had the same nondeterminism, and
whether it only moved low bits or also wrote wrong values there is not known. And a
fitness log alone does not record an ES run's weights, so the matched run cannot rely
on replay for checkpoints or resume until this is fixed.

The es-s5e-4 record says `dirty_worktree: true`: `diag_rebuild.py` sat untracked in
`experiments/end_to_end/` on the pod while it ran. The scorer does not import it.

## Later finding, 2026-10-06

The cause is found and fixed (`../update-check/README.md`): on the H200 the jitted
`lax.scan` in `stream.contracted_leaf` returned different bits from the same inputs in
about 0.4% of calls, in most processes, in JAX alone, with vLLM playing no part. Computing
the sum as one jitted step per member (154f63b) removes it and gives the scan's correct
bits; with it, every replay reached the same digest.

