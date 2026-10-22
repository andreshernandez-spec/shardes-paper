# Goal, decisions, problems and open questions

## The goal

Retrain an open model with shardes' evolution strategies on exactly the data a released
RL checkpoint was trained on, from the same start, and compare the two at matched
rollouts: do they learn the same, and if not, why (`docs/end_to_end/00-plan.md`). The RL
side is never trained here; it is the released checkpoints, including the intermediate
ones, so the baseline is somebody else's tuned run. The setting chosen first was Tulu 3.1
8B (Llama-3.1-Tulu-3-8B-DPO, GRPO on GSM8K, MATH and IFEval-style prompts); Olmo 3 waits
for a cheaper design. The result should hold up under the repository's rules: every
number from a committed record, runs preregistered, claims no wider than the evidence.

Beyond the comparison, the medium-term aim stated on 2026-10-10 was to find where ES has
value against RL: comparable data efficiency, or failing that a structural advantage
(inference-only memory, scalar communication, any servable model, less tuning, more
stability), or else a careful negative result.

## Decisions, in order

| date | decision |
|---|---|
| 2026-09-25 / 26 | Settings and budget: Tulu 3.1 first, greedy decoding, 192 prompts per member, N = 16 to match RL's 16 samples per prompt, one H200 per run, $350. |
| 2026-09-27 | Pilot (30 iterations, three sigmas, random control) passes no gate; held-out context measured; the next step left to Andres. |
| 2026-10-06 | Step 1 (the H200 update nondeterminism) and step 2 (the Countdown check against es-at-scale) approved and run; the Countdown check fails its gate on the high side (ours learns faster). |
| 2026-10-07 | The high-side fail is read as "the backend works"; the longer 120-iteration run approved; its gate passes against the control but the model ends below its start. Andres: "if ES has barely moved, diagnosis is in order, and looking at the calculated gradients is the place to start"; then "no more levers, what we need is understanding ... even if it means ES is doomed, at least understand why". Andres asks whether ES uses the same prompts as RL, and proposes a contrastive fitness. |
| 2026-10-08 | Andres: "first analyze the two saved results", then "run the contrastive-fitness probe". After the mechanism is laid out he asks why RL learns despite the same dead prompts and whether the records are accurate; the Countdown cross-prompt probe and a literature search are run; the one-update step probe is run on his approval. |
| 2026-10-09 | Andres: the answer is in hundreds of members, low-rank, with shardes' splitting; the N-scaling probe is run and the low-rank backend built. Andres asks about wall clock, approves the accumulation test and a literature search on acceleration, and flags that the dead-prompt screen would miss prompts the model has no idea about. |
| 2026-10-10 | Andres asks for a medium-term goal, whether the gradient estimation could be broken, about variance reduction, LoRA, replicating RL and hybrids; the exact-gradient test is run on his approval. He asks whether ES's steps are too short or too long, then about AdamW and Muon, then whether the result is a property of Tulu. He asks for the write-up (`08`), then approves step 2 (best achievable efficiency, a second budget of about $100). |
| 2026-10-11 | Step 2's run fails its gate; Andres asks for the evidence explained from scratch, then for this review package. |

## Problems met, and what each cost

| problem | how it showed | cause and fix | cost |
|---|---|---|---|
| JAX-to-torch handoff race (first pilot) | the center reward collapsed to 0.3 to 0.8 at 4 to 7 of 30 iterations | async dispatch let torch reuse memory JAX was still writing; every handoff now blocks until ready and syncs; the invalid run kept as evidence | about $34, one night |
| H200 update nondeterminism | rebuilds from fitness logs did not reproduce the logged weights | a jitted `lax.scan` returned wrong bits in about 0.4% of calls on the H200 in JAX alone; the sum is now one jitted step per member | about $9 and the pilot's rebuilt arms were unscorable |
| An idle pod | a pod finished its job and billed about 10 hours before deletion | no notification path from the pod; the rule since: delete the moment a job finishes, list pods after any interruption | about $46 |
| Verifier pipe deadlock | the contrastive probe stalled at 0% GPU | 6,144 items written to the verifier before any reply was read filled the pipe; scoring now goes in chunks | about $1.50 |
| bf16 artifacts read as structure | sparsity, concentration and rank claims about the RL displacement | the stored differences are mostly one-step bf16 rounding on small weights; the claims were withdrawn and lengths are now reported as what survived rounding | a day of analysis |
| The gradient scan's step grid | every point along the exact gradient broke the model | the grid was sized for a Gaussian direction's rounding; the gradient is concentrated in few weights; a second scan at a hundredth of the lengths was added | about $3 |
| The pooled cost model | the 1,024-member run with per-type steps fell 0.32 on held-out | the random part's cost is five to six times larger on MATH than on IFEval in the one-update record, and the pooled quadratic model that set the step averaged it away | about $83, the stage-B run |
| The laptop GPU | driver and library versions out of sync after a suspend | unresolved; all GPU work moved to pods | laptop probes abandoned |
| Pod hosts that never came up | two pods in one data center showed no runtime for minutes | deleted and recreated elsewhere | about $1 |

## Open questions

- Would a step sized by the reasoning tasks' cost, far below the pooled optimum, let a
  large population accumulate IFEval gains without losing MATH, and at what compute?
  The 512-member run at a third of the step still lost MATH 0.46 in eight updates.
- Is the task-specific cost a property of the low-rank perturbation's shape or of the
  tasks? The dense members' cost by task was not measured.
- One seed per configuration everywhere; how much of the eight-update results is seed
  variance is unknown.
- RL's wall clock and pipeline overheads on the same hardware are estimated from
  per-token rates, not measured.
- The structural advantages (inference-only memory, scalar communication, quantized
  models) are untested; on this evidence they pay for a compute penalty of several times
  only where RL cannot run.
- Whether the long-horizon argument for ES (noise independent of the answer length) holds
  at lengths where policy gradients degrade was not tested.

## Budget

About $425 of $450 by pod uptime: $350 for the comparison and its diagnosis, about $94 of
a second $100 for the best-achievable stage. The records' READMEs carry the cost of each
run.
