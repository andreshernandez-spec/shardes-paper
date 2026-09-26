# 00. Plan: ES fine-tuning against released RL checkpoints

Status 2026-09-26: Phase 0 and the throughput probe are done (results below). Paused
for assessment; nothing further runs without a go.

## The question

Retrain an open model with shardes' evolution strategies on exactly the data a
released RL checkpoint was trained on, starting from the same checkpoint, and compare
the two at matched rollouts: on the benchmarks the data targets, on generalization,
on response quality, on retention off-task, and on drift from the start.

The RL side is never trained here. It is always the released checkpoints, including
their intermediate ones, so the baseline is somebody else's tuned run and cannot be
called weak by construction.

## Decisions (Andres, 2026-09-25 and 26)

1. No RL training by us. "From scratch" is the ES side only.
2. Benchmarks may be scored here when no published score exists, as long as the
   benchmark is relevant to the data the RL checkpoint was trained on.
3. Held equal: the same training data and matched rollouts.
4. Engine: vLLM generates, shardes trains (below). Chosen on cost.
5. A paper of its own, independent of the placement paper and its deadline.
6. A100 or H100, decided after the throughput probe.
7. One seed per ES arm first, three if the budget allows.
8. Settings: Olmo 3 7B RL-Zero IF first, Tulu 3.1 8B second, OLMo 2 1B RLVR1 last and
   only if budget remains once real costs are known.

## What makes a setting usable

A release qualifies only if all five hold:

- (a) the RL start checkpoint is released;
- (b) the exact training data version is released;
- (c) the reward or verifier code is released at a known commit;
- (d) RL checkpoints are released with a known number of rollouts per step, ideally
  intermediate ones, so a matched-rollout point exists;
- (e) the prompt template and generation settings are known.

This turned out to be the hard part. Of the releases checked on 2026-09-25 (AI2's
Olmo 3 RL-Zero, Olmo 3 Instruct, Tulu 3 and 3.1, OLMo 2 1B and 7B; RLMT; SimpleRL-Zoo,
Oat-Zero, Open-Reasoner-Zero; ToolRL; Countdown releases), only Tulu 3.1 8B met all
five without a gap.

## Settings

### Olmo 3 7B base, RL-Zero instruction following (first)

| | |
|---|---|
| start (a) | `allenai/Olmo-3-1025-7B`; its `main` is byte-identical to `stage3-step11921`, the path the run logged (checked by LFS hashes) |
| data (b) | the run drew 13,314 of the 88,556 rows of `saurabh5/IF_multi_constraints_upto5_filtered_olmo_completions_filtered` (public, created 2025-10-01, before the run), then filtered by length. `allenai/Dolci-RL-Zero-IF-7B` (13,179 rows) came after the run. Phase 0 rebuilds the run's set and prompt stream and compares |
| verifier (c) | open-instruct `d928a7c`, `ground_truth_utils.IFEvalVerifier` and `IFEvalG/`: partial credit, 10 x fraction of constraints met |
| checkpoints (d) | `allenai/Olmo-3-7B-RL-Zero-IF` branches `step_100` to `step_1900`, `main` = step 2000; 256 rollouts per step (32 prompts x 8), so 512,000 at step 2000. `step_100` and `step_1000` hold identical weights; every branch is hashed before use |
| template (e) | `olmo_thinker` (generation prompt `<\|im_start\|>assistant\n<think>`), stop `</answer>`, cap 16,384, temperature 1.0 |
| run log | public W&B `ai2-llm/Olmo-3-7B-RL-Zero`, run `wn9zgjj3` |

The paper's text disagrees with the logged run on the template, the reward and the
advantage normalization. The logged config is what produced the released weights, so
it is what we match.

Response lengths are short: the per-step mean over steps 1-2000 has median ~99 and
mean ~340 tokens. That is what ES cost scales with.

### Tulu 3.1 8B (second)

| | |
|---|---|
| start (a) | `allenai/Llama-3.1-Tulu-3-8B-DPO` |
| data (b) | `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`, 29,946 rows; the data file has not changed since 2024-11-18 |
| verifier (c) | open-instruct `3f37c29` (the commit the model card names), GSM8K / MATH / IF verifiers, reward 10 |
| checkpoints (d) | `allenai/Llama-3.1-Tulu-3.1-8B` branches every 40 steps to `step_2440`; `main` = `step_1920`; 768 rollouts per step (48 x 16), so 1,474,560 at release |
| template (e) | `tulu`, temperature 1.0, cap 2,048 |
| published | GSM8K 90.0, MATH 47.8, IFEval 83.9 (DPO start: 84.3, 42.0, 81.1) |

RL here follows SFT and DPO, the stage production post-training runs. Responses
~310-400 tokens (read off the card's plot).

### Considered and not used

- Olmo 3 RL-Zero Math, Code, General, Mix: released runs average 6,500-12,000
  response tokens (IF: 340), so a matched ES run costs 20-35x an IF run; General needs
  a Qwen3-32B judge per rollout, Code an execution service, Mix has no public run.
- OLMo 2 7B Instruct: card and paper contradict each other on data and batch size.
- RLMT (RLHF with a public reward model): final checkpoint only; a candidate for a
  later RLHF-flavored setting.
- Countdown on Qwen2.5 (Qiu et al.): old model, covered by several ES papers, and the
  published numbers mostly fail independent reproduction. Used only as an
  implementation check against Qiu's released code.

## What is held equal

The same start checkpoint, the same prompts, the same verifier and template, and the
same number of rollouts at every compared point. Where the RL run's prompt stream can
be reconstructed (Olmo 3 IF), ES iteration `i` sees the prompts RL step `i` saw.
Released branches give RL points every 25,600 rollouts (Olmo 3 IF) and 30,720 (Tulu
3.1); ES checkpoints are saved at the same counts. GPU-hours are reported beside the
results, not equalized, since the hardware differs.

## Engine: vLLM generates, shardes trains

shardes owns the training: the f32 master and its sharding, the noise (seed contract,
i.i.d. per leaf), the exact weights each member is evaluated with (`view + sigma *
eps_i`, or rank-r factors under EGGROLL), fitness shaping, the update, checkpoints and
export. vLLM owns only "weights plus prompts to text"; the verifier turns text into
rewards.

The boundary is tested, not asserted:

- after a perturbation, the engine's weight slices equal shardes' `view + sigma * eps_i`
  bit for bit, through vLLM's fused `qkv_proj` and `gate_up_proj` layouts;
- restore recomputes the view from the master; it never subtracts noise, since a bf16
  add and subtract leaves rounding residue that accumulates over thousands of members.

Design: each vLLM worker regenerates its member's noise with shardes' own
`member_noise` (JAX on the same GPU, handed to torch by DLPack into torch-owned
buffers). The f32 master is sharded across GPUs and updated with the placement
paper's contraction; the new bf16 view is all-gathered into every engine. EGGROLL maps
to per-request rank-r LoRA through vLLM multi-LoRA.

vLLM for rollouts also matches the RL side: both released runs generated with vLLM,
so a sampler difference cannot explain a gap.

## Evaluation

1. Every number comes from an HF safetensors export loaded in a separate process.
2. The harness is proven first on published numbers: Olmo 3 7B Instruct (IFEval 85.8,
   IFBench 32.3, report Table 24) and Tulu 3.1 (GSM8K, MATH, IFEval). Then every RL
   branch and every ES checkpoint is scored with it.
3. The authors' own scorers where they exist.
4. Dimensions: in-distribution (IFEval; GSM8K, MATH, IFEval for Tulu), generalization
   (IFBench's unseen constraints), response quality on the same prompts with an open
   reward model (over-optimization), retention off-task, KL to the start (k3) and
   parameter drift.
5. Controls: random-reward ES (free for ES: with fitness independent of the outputs
   nothing needs generating), greedy and temperature 0.6 decoding for every model.
6. Provenance as in the rest of the repository: configs committed before runs and
   cited by SHA, records that stamp both repositories, writers that refuse to
   overwrite, and assertions where each number is produced that the work was done.

## Phase 0 results (2026-09-26)

Records: `experiments/end_to_end/pins/` and `experiments/end_to_end/data/olmo3_if/`.

- Every pin resolves. Olmo 3 base `main` is byte-identical to `stage3-step11921`, the
  IF run's logged start. RL-Zero-IF `main` (step 2000) is on no branch; its `step_100`
  and `step_1000` hold identical weights, so neither is used until we know which step
  they are. Tulu 3.1 `main` is byte-identical to `step_1920`. The Tulu data file is
  the 2024-11-18 upload. Both open-instruct commits exist; the IF run's logged config
  agrees with every value in `releases.py`.
- **The IF run's training data is reconstructed exactly.** open-instruct's path at
  d928a7c samples 13,314 of the 88,556 source rows with `RandomState(42)`; the
  length filter keeps 13,179. Dolci-RL-Zero-IF-7B is that filtered set in the run's
  pre-shuffle order, with every prompt identical byte for byte, so the released set
  and the run's set are the same. Our tokenizer (transformers 5.x) counts one kept
  prompt at 2,049 tokens where the run's counted at most 2,048, which is why the set
  is taken from Dolci after checking the order, not from our own filter; with one row
  fewer every later permutation would differ.
- **The per-step prompt stream is reconstructed**: `Dataset.shuffle(seed=1)` then
  open-instruct's `ShufflingIterator` (seed 1, 32 per step), recorded for steps 1 to
  2000. ES iteration `i` can see exactly the prompts RL step `i` saw.
- Prompt lengths under `olmo_thinker`: median 223 tokens, mean 285, p95 592, max 2,018
  (our tokenizer, the rows our filter kept).
- For the backend: vLLM's prefix caching must be off (or reset per member) under
  full-rank ES, since KV computed under one member's weights is wrong for the next.

## Throughput probe results (2026-09-26)

Full record: `experiments/end_to_end/probe/README.md`. Cost of an ES run matched to the
released checkpoint, measured per 32-, 48- or 192-prompt member batch, 1.2x uptime:

- **Tulu 3.1: $122-159** at 192 prompts per member, on A100 or H100 alike; $41-53 to
  step 640. The estimate above held.
- **Olmo 3 RL-Zero IF: $4,800-6,200 greedy, $1,850-2,100 at temperature 1.0.** The
  estimate above was off by 50x. Under greedy decoding the base model runs to the
  16,384-token cap on about half the prompts; sampled at the run's temperature it stops
  looping, but averages 2,873 tokens (p90 6,556), and a full-rank member waits for its
  longest sample.
- Greedy outputs differ between A100 and H100 on most prompts, so a run and its
  evaluation stay on one GPU type.
- A full weight rewrite costs 12-19 ms per member: negligible.

Options for the Olmo 3 setting, for the assessment (none decided):

1. A lower response cap (2,048 to 4,096). Bounds each member's time; the RL run's cap was
   16,384, but it also masked truncated completions, so its gradient never came from
   them. Costs fidelity to the run's generation budget.
2. More prompts per member for the same rollouts (fewer, larger iterations): on Tulu,
   4x the prompts cost 2.2x less per rollout.
3. EGGROLL mode through per-request LoRA: every member in one continuously batched
   engine, so no member waits alone for its longest sample. This is the scheduling
   advantage the plan wanted measured; it now has a measured reason, and is unmeasured
   itself.
4. Stop a response when it starts repeating, scored as a failure.
5. Swap the order: Tulu 3.1 first, since it is affordable as designed.

## Phases and gates

- **Phase 0, desk.** Pin every release by LFS hash; record the run's logged config;
  rebuild the Olmo 3 IF training set and prompt stream and compare with Dolci; pin the
  Tulu 3.1 data and code.
- **Throughput probe.** Done 2026-09-26, about $3.9 (results above). **Paused here for
  assessment.**
- Phase 1: the vLLM backend for shardes, export, harness. Gate G1: harness reproduces
  the published numbers above.
- Phase 2: implementation check against Qiu's released ES code on Countdown at 0.5B or
  1.5B, same config, both on vLLM. Gate G2: curves overlap within seed spread.
- Phase 3: Olmo 3 IF pilot, preregistered (N=32, three sigmas, 30 iterations). Gate
  G3: one sigma climbs clearly above the random-reward walk.
- Phase 4: the ES runs at matched rollouts; Phase 5: evaluation and write-up.

## Cost model (the pre-probe estimates, kept for the record; superseded above)

One Olmo 3 IF ES run matched to step 2000 (512,000 rollouts; ~250 prompt tokens each;
mean response length L unknown until the probe). Assumed throughput: A100 decode 2-3k
tok/s per GPU (KV-bound, full multi-head attention, 512 KiB per token), prefill ~9k;
H100 decode ~1.7x, prefill ~2.8x; x1.3 overhead.

| L | A100 GPU-h | A100 $ (1.39-1.59/h) | H100 GPU-h | H100 $ (2.69-3.49/h) |
|---|---|---|---|---|
| 150 | ~16 | 22-26 | ~8 | 22-29 |
| 340 | ~30 | 42-48 | ~17 | 45-58 |
| 800 | ~64 | 89-102 | ~37 | 99-128 |

Tulu 3.1 to its released step (1.47M rollouts, ~357 tokens): ~56 A100-h ($78-89) or
~31 H100-h ($84-109). Per result the two GPU types cost about the same; H100 roughly
halves wall clock. One 80 GB card is memory-starved at 7B once the f32 master (29 GB)
sits beside the engine; eight GPUs shard the master.

## Risks

- ES may not lift a 7B base model on IF within 512k rollouts; the pilot shows it first.
- Base-model greedy decoding can loop to the cap; full-rank members run one after
  another on a GPU, so one looping sequence stalls it. The probe measures how often.
- Fidelity to the logged configs, not the papers or today's scripts.
- JAX and vLLM share each GPU: memory split and process model.
- es-at-scale v1.0.0 (Qiu et al.'s PyTorch + vLLM library) has the same shape; what
  shardes adds is i.i.d. per-leaf noise, exact restore, full rank and EGGROLL behind
  one switch, and the sharded update.
