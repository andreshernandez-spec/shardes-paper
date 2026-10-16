# 01. Tulu 3.1 first: plan to the first matched ES run

Status 2026-09-26: plan only. Decided by Andres after the throughput probe: Tulu 3.1 8B
goes first, since it is affordable as designed; Olmo 3 RL-Zero IF waits for a cheaper
design (options in `00-plan.md`, "Throughput probe results"). Nothing here runs without
a go.

## The setting, as fixed so far

Start `allenai/Llama-3.1-Tulu-3-8B-DPO`; data `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`
(29,946 rows: 7,473 GSM8K, 7,500 MATH, 14,973 IF); RL reference
`allenai/Llama-3.1-Tulu-3.1-8B`, branches every 40 steps, `main` = `step_1920`, 768
rollouts per step (48 prompts x 16 samples); open-instruct `3f37c29`; `tulu` template,
cap 2,048, stop on eos. Published: GSM8K 90.0, MATH 47.8, IFEval 83.9 (DPO start 84.3,
42.0, 81.1). Probe: $122-159 for an ES run matched to `step_1920` at 192 prompts per
member, on A100 or H100.

## Work, in order

### T0. Desk (laptop, $0)

1. **Rebuild the run's data path from open-instruct at 3f37c29**, as Phase 0 did for
   Olmo 3: the legacy `grpo_vllm_thread_ray_gtrl.py` loader with `dataset_mixer_list ...
   1.0`, its transforms and length filters (prompt 2,048, total 2,048), the seed-1
   shuffle, and the per-step draw of 48 prompts. Record the training set and the
   prompt stream for steps 1 to 2440. The card's command gives `local_rollout_batch_size
   8` on 6 actor GPUs; confirm 48 prompts per step from the code, not the card.
2. **Settle what the card leaves open**, from the code: whether prompts carried a bos
   token (`--add_bos` is absent from the command), what `--non_stop_penalty` with
   `--penalty_reward_value 0.0` does to a response without eos, and how the async policy
   lag works (it only matters for describing the RL side).
3. **Vendor the verifiers at 3f37c29** (GSM8K last number, MATH flex, IF through
   `verify_ifeval_sample`; reward 10 x correct), with tests on known-good and known-bad
   responses, and pin their file hashes.
4. **Evaluation harness**, pinned: OLMES (the harness AI2 reported with) through vLLM,
   for GSM8K, MATH, IFEval as the card reports them; IFBench for unseen constraints;
   an open reward model (not a paid judge) for response quality; a retention set from
   the Tulu 3 suite. The card and the paper disagree on whether IFEval is "strict" or
   "prompt loose"; the gate below decides which one reproduces.

### T1. The ES backend on vLLM (laptop first)

The part that does not exist yet: shardes trains, vLLM generates.

1. **JAX with CUDA beside vLLM's torch in one venv.** The probe's venv has JAX for CPU
   only. The worker needs `jax[cuda13]` next to torch 2.13+cu130; check the two resolve
   together and share a GPU (memory split: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, vLLM's
   `gpu_memory_utilization`). Laptop, CUDA 13 driver. If they cannot coexist, the
   fallback is a separate JAX process handing weights over by CUDA IPC.
2. **Member weights.** The worker regenerates member `i`'s noise with shardes'
   `member_noise`, forms `view + sigma * eps_i`, and writes it into the engine's
   parameters through the `qkv_proj` / `gate_up_proj` fusion; restore recomputes the view
   from the master. Test: bit-identical weights after perturb and after restore, on
   Qwen2.5-0.5B locally (same fused layout as Llama).
3. **The update.** Single GPU first: the f32 master (32 GB at 8B) on the same card as the
   engine (16 GB) leaves ~25 GB of KV cache on an 80 GB GPU, enough at 192 prompts; an
   H200 is the roomy fallback. Multi-GPU later, with the master sharded and the
   placement paper's contraction; device-count invariance means the single-GPU run is
   the same run.
4. **Checkpoints as logs.** Store per-iteration fitness, not weights; any checkpoint is
   rebuilt from the start weights and the log. Each run hashes one checkpoint on the pod
   so the rebuild can be checked.
5. **Driver hygiene** the probe taught: shut the engine down on SIGTERM; prefix caching
   off (KV from one member's weights is wrong for the next); FlashInfer's sampler off.

### T2. Implementation check (~$15-30)

Qiu et al.'s released ES code (es-at-scale v1.0.0) against ours on Countdown, same
config, Qwen2.5-0.5B or 1.5B, both on vLLM, 2 seeds each, ~100 iterations. Gate: the
curves overlap within seed spread. Checks the ES core and the backend together against
a reference, which a published number cannot do (Qiu's numbers mostly do not reproduce).

### T3. Harness gate (~$5-10)

Our harness scores the DPO start and the released Tulu 3.1 within a stated tolerance of
the card's numbers. Then it scores the RL branches at the points ES will be compared at.

### T4. Pilot, preregistered (~$30)

Committed first: primary metrics, the matched axis, the sigma rule, what counts as a
negative. Then N=32, 192 prompts per member (four RL steps' prompts, in the run's own
order), three sigmas around 1e-3 for ~20 iterations each, beside the free random-reward
walk. Gate: one sigma climbs clearly above the walk.

### T5. The matched run (~$120-160 per seed)

To `step_1920`: 1,474,560 rollouts = 240 iterations of 32 x 192. Fitness logged every
iteration; comparison points every 5 iterations (30,720 rollouts, one RL branch every 40
steps). One seed first, three if the budget allows. ~38 H100-hours: about 5 h on an
8-GPU node, or about 1.6 days on one GPU.

### T6. Evaluation (~$30-60)

Rebuild ES checkpoints from their logs; score them and the matching RL branches on
GSM8K, MATH, IFEval, IFBench, response quality, retention and KL to the start. Random-
reward control. One GPU type and one engine setup throughout (the probe found greedy
outputs differ between A100 and H100).

**Total to a first result: about $200-300 with one seed.**

## Decisions (Andres, 2026-09-26)

1. **Greedy ES rollouts**; temperature 1.0 (the RL run's) as a later ablation.
2. **192 prompts per member**: four RL steps per ES iteration, in the run's own prompt
   order; 2.2x cheaper per rollout than 48.
3. **Keep T2**, the Countdown check against Qiu's code.
4. **One GPU for T5 is fine even if it takes days**, as long as the run is equivalent.
   It is: the same algorithm, data, prompt order and rollout budget, and shardes' update
   does not depend on the device count. What must not change is the GPU type, for the run
   and for the evaluation of its checkpoints, since greedy outputs differ between A100 and
   H100. The two cost the same per result ($127-145 A100, $122-159 H100, measured); an
   H100 takes ~1.6 days against ~3.8, halving the exposure to a community host reclaiming
   the pod, and the run resumes from its fitness log either way. Leaning H100; confirmed
   at the pilot by availability.

## Decisions (Andres, 2026-10-07)

5. **T2's high-side fail is read as "the backend works"** (`06`): our backend learns
   Countdown faster than es-at-scale under the same settings and reaches the same level.
6. **The longer arm next** (`07`): the pilot's s5e-4 arm to 120 iterations (RL step 480)
   with a control at the same sigma, scored on held-out prompts against the RL branches at
   matched points, before any run matched to `step_1920`.
