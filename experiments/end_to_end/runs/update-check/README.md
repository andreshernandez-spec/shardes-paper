# The per-leaf ES update on the H200: nondeterministic scan, found and fixed. 2026-10-06

Step 1 of the plan after the held-out run (`../heldout-121-160/README.md`): rebuilds of
the Tulu pilot's arms from their fitness logs did not reproduce the logged weights, and
two replays in one process already differed after the first update.

**Finding.** On the H200, `stream.contracted_leaf` (the noise-weighted sum over members,
a jitted `lax.scan` that the backend shares with the library's `SeedRegenerated.contract`)
intermittently returned different bits from the same inputs: in about 0.4% of calls, in
most processes but not all, in JAX alone (no vLLM, torch never on the GPU). The noise draw
by itself, the update arithmetic and a plain f32 control never differed. None of three XLA
settings removed it. Computing the same sum as one jitted step per member, called from
Python, gives no differences and the scan's correct bits; that is the fix, `154f63b`.
The RTX 3080 laptop GPU and the A100 never showed it.

Every number below: `python -m es_vllm.update_check --compare runs/update-check`, from the
records in this directory (each `.json` beside its per-leaf checksums in
`.checksums.json.gz`). H200 SXM, Secure Cloud US-CO-1, driver 580.178.04 (the
driver of the H200 where the rebuilds diverged), jax 0.11.2, vLLM 0.30.0, one pod
(`d1smeoq5nw324h`, 14:57 to 16:36 UTC; a first pod never came up and was deleted after
12 minutes). Logs: `h200-uc.log`, `h200-kc-summary.log`, `h200-kc.log.gz`,
`h200-process-logs.tar.gz`.

## The update, twice from the same inputs (`es_vllm/update_check.py`)

Synthetic fitness (seeded uniform), N = 16, sigma 1e-3, alpha 5e-4. For every leaf at
every iteration the contraction is computed twice from the same master leaf and compared
element by element, then the production update is checked against the master minus the
coefficient times the first sum; an exact checksum of every leaf after every update lets
processes be compared. Two replays per process (laptop: three).

| GPU, model | code | mode | XLA flags | processes (with differences) | differing sums / updates | final digests |
|---|---|---|---|---|---|---|
| RTX 3080 Laptop, Qwen2.5-0.5B, 30 it. | 3c3ec58 | JAX alone | | 4 (0) | 0 / 0 | one |
| | 3c3ec58 | in vLLM | | 4 (0) | 0 / 0 | the same |
| H200, Tulu 3 8B, 15 it. | 9bf6b96 | in vLLM | | 3 (3) | 13 / 15 | four |
| | 9bf6b96 | JAX alone | | 6 (4) | 170 / 216 | nine |
| | 9bf6b96 | JAX alone | autotune level 0 | 2 (1) | 44 / 51 | three |
| | 9bf6b96 | JAX alone | command buffers off | 2 (1) | 19 / 21 | three |
| | 9bf6b96 | JAX alone | deterministic ops | 2 (2) | 91 / 88 | four |
| | 9bf6b96 | in vLLM | each of the three | 2 + 2 + 2 (0) | 0 / 0 | one |
| | **154f63b** | in vLLM | | 3 (0) | 0 / 0 | one, `664fa37f` |
| | **154f63b** | JAX alone | | 3 (0) | 0 / 0 | the same |
| RTX 3080 Laptop, Qwen2.5-0.5B | **154f63b** | JAX alone | | 1 (0) | 0 / 0 | `5656f88b`, as before |
| A100 SXM 80GB (T2's pods), Qwen2.5-0.5B, 30 it. | 9bf6b96 | JAX alone | | 2 (0) | 0 / 0 | `5656f88b`, the laptop's |
| | 9bf6b96 | in vLLM | | 2 (0) | 0 / 0 | the same |

`664fa37f` is the digest every clean H200 replay reached, before and after the fix: the
fix gives the same weights the scan gives when it does not misbehave.

**What a difference looks like.** In the bad JAX-alone processes, a differing sum differs
in 17,785 to 909,753 elements of its leaf (median 501,801), by up to 0.68 of the sum's
largest value (median 0.25); the update built on it differs by up to 4.5e-4 (median 1.4e-4)
against a coefficient of 3.1e-5, so by a few update steps' worth. Only layer matrices were
hit (q, k, v, o, gate, up, down), never norms or the embedding. In a bad process, one to
three leaves per iteration, about 0.4% of the sums computed (a pair of evaluations differs
when either is wrong, so pairs differ about twice as often). These are wrong values of ordinary size, not blow-ups or NaNs,
and not rounding.

**The flags.** The six clean flag processes inside vLLM came first in the order; the
round-robin that followed found autotune level 0, command buffers off and deterministic
ops each in bad processes. Whether a process is bad is not decided by these settings.

## The pieces, repeated (`es_vllm/kernel_check.py`)

One array the size of a Llama-3.1-8B MLP matrix (14336 x 4096); each piece evaluated
repeatedly and compared with its first evaluation:

| code | processes x repeats | contraction (`lax.scan`) | single draw | update arithmetic | f32 control | unrolled scan | one step per member |
|---|---|---|---|---|---|---|---|
| 775b8bf | 6 x 200 | 3 | 0 | 0 | 0 | | |
| 26c0348 | 6 x 1,000 | 24 | 0 | 0 | 0 | 0 | 0 |
| 154f63b (contraction = one step per member) | 3 x 1,000 | 0 | 0 | 0 | 0 | 0 | 0 |
| 26c0348, A100 SXM 80GB, two hosts (drivers 590, 595) | 6 x 1,000 | 0 | 0 | 0 | 0 | 0 | 0 |

Both loop-free forms equal the scan's most common value in every process (and on the
laptop). At the scan's rate, 0.4%, zero in 6,000 by chance has probability about e^-24.
The fault is in the scan as XLA runs it on this GPU (a device-side while loop whose body
draws and accumulates), not in the draw or the arithmetic.

**Why one step per member and not the unrolled scan.** Unrolled, XLA's CPU backend fuses
the steps differently and the sum differs from the library's in the last bit
(`tests/end_to_end/test_e2e_stream.py` fails, 2 of 1,024 elements). One jitted step per
member computes exactly one iteration of the library's scan per call and passes the
bit-identity tests. It costs one dispatch per member per leaf. This probe computes three
sums per leaf: a replay iteration at 8B on the H200 went from about 4.5 to 8 s, so the
single sum in training costs about 1 s more per iteration of about 255 s. At 0.5B on the
laptop the probe's 30-iteration replay went from 30 s to 227 s (dispatch-bound).

## What follows

- The Tulu pilot's live runs used the scan on H200s, so in processes that were bad, about
  0.4% of leaf updates had stretches of wrong values of update size. That degrades the
  gradient estimate a little; it is not a plausible cause of the pilot's null, but those
  runs were not exactly the algorithm they claim to be.
- With the fix, a replay of a fitness log reproduces the weights again (every fixed
  replay reached the same digest), so fitness logs can again serve as checkpoints, with
  the digest check kept.
- T2 (`docs/end_to_end/06`) ran the old code on A100s. Its preflight (this probe at 0.5B
  on each pod, before the runs; `a100-preflight-pod*.log`) and the kernel check run on the
  same GPUs after them found nothing: 6,000 scan evaluations, all equal.
- The library's `SeedRegenerated.contract` uses the same scan, so shardes itself is
  exposed on this GPU. Its published results were measured on A100 and TPU, where nothing
  like this was seen (C6d found A100 runs bitwise deterministic across processes).
- The laptop records say `dirty_worktree: true` for seven of eight 3c3ec58 processes: the
  T2 files were being written in the same worktree while they ran. None is imported by
  the probe, whose code is unchanged from 3c3ec58.
