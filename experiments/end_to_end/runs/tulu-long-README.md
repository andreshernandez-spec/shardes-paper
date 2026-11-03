# The longer Tulu 3.1 arm: results, 2026-10-07

Preregistration: `docs/end_to_end/07-tulu31-long.md` (c725831, before any run). Runs from
c725831 on Secure Cloud H200 SXM at $4.59/h: the arm (`tulu-long-s5e-4/`, US-CO-1, 10:26
to 19:12 UTC), and on a second pod (US-NC-1, 10:26 to 11:37) the RL side
(`heldout-481-520/`) and then the control (`tulu-long-random/`). About $46 by uptime.
Every run completed: the arm 120 iterations with all 132 engine checks passing (after
every restore, and member 0 every ten iterations), median 251 s per iteration; the
control 120 iterations, all 120 restore checks passing; five held-out evaluations each.

## Gate G4: pass, and the arm ends below its start

`python -m es_vllm.long_gate` (`long-gate.json`). After 120 updates the arm is above the
control by 0.260 +- 0.091 on the 1,920 held-out prompts, paired (2.9 SE): **G4 passes as
preregistered.** But the arm is also 0.250 +- 0.087 **below its own start**. The pass
comes from the control degrading faster (to 4.995), not from the model improving: ES's
updates are a random walk whose drift lowers the held-out reward, plus a signal that
offsets part of that drift. The preregistered reading of a pass ("ES learns on Tulu 3.1
within RL steps 1 to 480's data") holds only in that relative sense; the gate did not
require the arm to beat its start, and it does not.

`python -m es_vllm.long_gate --table`, held-out reward on the 0-10 scale and paired
differences with their standard errors:

| updates (RL step) | ES arm | control | RL branch | arm - start | arm - control | RL - arm |
|---|---|---|---|---|---|---|
| 0 (0) | 5.505 | 5.505 | 5.505 |  | +0.000 +- 0.000 |  |
| 30 (120) | 5.542 | 5.406 | 6.620 | +0.036 +- 0.078 | +0.135 +- 0.081 | +1.078 +- 0.097 |
| 60 (240) | 5.469 | 5.302 | 6.969 | -0.036 +- 0.080 | +0.167 +- 0.082 | +1.500 +- 0.100 |
| 90 (360) | 5.323 | 5.286 | 7.125 | -0.182 +- 0.082 | +0.036 +- 0.087 | +1.802 +- 0.100 |
| 120 (480) | 5.255 | 4.995 | 7.370 | -0.250 +- 0.087 | +0.260 +- 0.091 | +2.115 +- 0.107 |

At matched prompts and rollouts, RL's lead grows from 1.08 at step 120 to 2.12 at step
480 (20 SE).

## Also measured

- **The first 30 iterations reproduce the pilot's s5e-4 arm exactly**: all 30 fitness
  vectors and all six digests identical, on a different host with the fixed contraction.
  The pilot's best arm was unaffected by the scan fault, and the run is deterministic
  across H200 hosts.
- **The two measurements agree on every prompt**: the arm's in-run score of the start
  and `heldout.py`'s DPO start are equal on 1,920 of 1,920 prompts.
- **In-run center reward**, each iteration's own 192 prompts: the arm-minus-control slope
  over 120 iterations is +0.0020 per iteration (2.6 SE).
- **By source** after 120 updates (start: MATH 4.60, GSM8K 8.37, IFEval 4.58): arm 3.58,
  8.31, 4.66; control 3.48, 8.31, 4.18; RL step 480 5.10, 8.91, 7.87. MATH loses most to
  the drift in both ES runs; on IFEval the arm holds its level while the control falls.
- The arm's held-out responses stay at 293-331 tokens on average, with 16-29 of 1,920
  hitting the cap.

## Why: the gradient is mostly noise

`grad-check/README.md`: at this sigma and 192 prompts per member, about 12% of the
variation between the 16 members' fitness is a property of the members (ranking
reliability 0.12, 90% interval 0.01-0.47); the rest is which of the roughly 10% of
outcomes a perturbation flips. Sampled fitness at temperature 1.0 is no better per
rollout (0.11 with common random numbers, 0.04 without), mirrored pairs are no better,
and a larger sigma buys signal only where the perturbations cost 1.45 points. Each update
is therefore mostly a random direction, which is what the arm shows.
