# 05. Held-out context for the Tulu 3.1 pilot

Written 2026-09-27, before the run; exploratory, not a gate. The pilot (`03`) found no
detectable learning within 30 iterations. This measures whether the RL run itself moved
detectably on the same measurement by the same point, which says whether 30 iterations
(RL steps 1 to 120's data) was long enough to see anything.

Prompts: the 1,920 prompts of RL steps 121 to 160 (stream steps [120, 160)), new to every
model scored. Models: the DPO start; the RL run's branches `step_40`, `step_80`, `step_120`;
each pilot arm's final ES weights (after 30 iterations, matched to RL step 120 in prompts
and rollouts), rebuilt from its fitness log and checked bit for bit and against the logged
digest. Measurement: the pilot's, greedy, `tulu` template, cap 2,048, the run's verifiers,
mean reward on the 0-10 scale with its standard error and per source.

Run: `es_vllm/heldout.py` with `es_vllm/heldout-121-160.yaml`, one model per process, two
H200 SXM pods (the pilot's GPU type), released checkpoints on one and ES rebuilds on the
other. Reported whatever it shows; the next step stays Andres's decision.
