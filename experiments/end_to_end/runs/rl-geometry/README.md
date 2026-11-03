# How far the RL run moves the weights, 2026-10-07

Part of the question Andres asked after the longer arm: why does RL get a signal on Tulu
when ES gets mostly noise? Three records:

- `geometry.json`: `es_vllm/rl_geometry.py` at 19e54d6, on a Secure Cloud H200 pod's CPU
  (US-CO-1, 20:09 UTC; the pod finished at about 20:50 and was only deleted the next
  morning, about $46 of idle time). The DPO start against the RL branches step_120,
  step_240 and step_480, leaf by leaf in f32.
- `quantization.json`: `es_vllm/rl_quantization.py` at 1633bad, on the laptop. Five
  matrices read by range request, each change measured in bf16 steps of the start weight.
- `effective-norm.json`: `es_vllm/bf16_effective_norm.py` at 1fba629, on the laptop. How
  much of a small perturbation survives the engine's bf16 rounding, on the same five.

## The RL run moves the weights very little

| | distance from the start | relative to the weights' norm (1,248) |
|---|---|---|
| RL step 120 | 0.476 | 3.8e-4 |
| RL step 240 | 0.751 | 6.0e-4 |
| RL step 480 | 1.134 | 9.1e-4 |
| ES after 30 updates (matched to step 120) | 61.4 | 4.9e-2 |
| ES after 120 updates (matched to step 480) | 122.7 | 9.8e-2 |
| one ES member's perturbation (sigma 5e-4) | 44.8 | 3.6e-2 |

ES's distances are exact: its displacement is a sum of z-weighted Gaussian vectors of
norm `alpha sqrt(T d / N)`. A single ES probe is about 90 times longer than the RL run's
whole move to step 120, and ES's own displacement over the same data is about 130 times
longer.

The RL run keeps its direction: the cosine between its moves to step 120 and step 240 is
0.51, between 240 and 480 0.52, between 120 and 480 0.32. A positive cosine needs the
same weights to move with the same sign. (The increment from 120 to 240 has cosine -0.15
with the move to 120: part of what changed is undone, as rounding at the next checkpoint
also does.) Consecutive ES updates are independent random directions, of cosine about
`sqrt(N / d)`, nil.

By module type the RL move is spread about as the parameters are: the MLP matrices carry
77% of its squared norm (70% of the parameters), embedding and output layer 5% (13%),
normalization weights nothing.

## Most of the stored difference is bf16 rounding

The checkpoints are stored in bf16. The RL run's typical change per weight, about 1e-5,
is below one bf16 step at a typical weight (about 6e-5). In the five matrices checked,
the changed weights moved by a median of exactly one step (53-66% by one step, 13-17% by
two), changes occur almost only on the smallest weights (in layer 15's q_proj at step 120,
46% of the smallest quarter changed and none of the largest), and the size of a change
follows the size of the weight (correlation 0.8). An update below half a step rounds back
to the start exactly, and only small weights, where the grid is fine, show it.

So three statistics in `geometry.json` describe the bf16 grid and the weights' magnitudes
more than the update, and are not evidence of a structured update: the share of weights
unchanged (86% at step 120), the share of the squared norm in the largest 1% of changes
(31%, against 9.3% for a Gaussian), and the lower effective rank of single matrices. The
true f32 update is not observable from the release; its parts below half a step on large
weights are missing from the stored difference. What the comparison needs holds anyway:
the released step-120 model, which scores 1 to 1.1 points above the start, is 0.476 away
from it in the bf16 weights the engine runs.
