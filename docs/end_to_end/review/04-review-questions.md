# What feedback is wanted

Answer each question in the review, with the evidence (a record, a number, a passage) the
answer rests on. The author's own doubts are listed last so the reviewer can confirm,
dismiss or add to them.

## On the chain of inference

1. The paper's claim is that the ES update is a correct gradient estimate whose useful
   component is at most `alpha` long beside a random component of `alpha sqrt(d / N)`,
   and that this, not the fitness, the step rule or the implementation, is what separates
   ES from RL on this stream. Does the evidence in `02` support each link: (a) the
   estimate is correct (E17), (b) one update gains what its random part costs (E13), (c)
   the gain is constant in N and the cost falls as 1/N (E14), (d) the signal is small
   because the stream's gains are prompt-specific (E7, E8, E10, E11)?
2. The "ceiling" argument (E17): a perfect zeroth-order estimate of RL's gradient would
   gain `alpha` x slope-along-the-gradient per unit, and that is 0.8 of what ES already
   gains. Is the comparison legitimate, given that the slope was measured at effective
   length 0.022 where curvature is already visible, and that ES's useful direction is the
   gradient of a smoothed, greedy objective rather than GRPO's?
3. The odd/even decomposition is applied at lengths (45 to 112) where the first-order
   and quadratic models are approximations. The paper reports the gain per unit falling
   from 0.0056 at length 45 to 0.0047 at 112. Is "first-order gain" the right name, and
   is the best-net formula `G^2 / (4 C)` a fair summary?
4. Cross-prompt correlation is estimated as the split-half covariance of member means
   divided by the per-prompt effect variance, with an adjustment for the share of
   uncertain prompts (E8, E11). Is the estimator sound, and is 0.004 against 0.30 a fair
   comparison across tasks with different reward scales?
5. The Countdown comparison uses the same model and the same noise directions but 200
   prompts of a different task with partial-credit rewards. Does it establish that the
   stream, not the model, sets the signal?

## On the statistics and the claims

6. Every configuration ran once. Which claims depend on a single seed in a way that a
   second seed could overturn? Where should the wording be narrowed?
7. Standard errors are over prompts, paired. Are there places where a claim treats a
   difference within two standard errors as a finding, or a null as "no effect"?
8. The RL side is the released checkpoints scored by us; the run was not reproduced, and
   its compute is estimated from per-token rates. Which comparisons does that weaken,
   and how should they be stated?
9. The held-out prompts are from the training distribution (later steps of the same
   stream). Does any claim need a benchmark outside it?
10. Are there claims in `draft.md` wider than the records: universal words, "proves",
    "cannot", "only", "no X can", where the evidence is one model and one stream?

## On what is missing

11. What control or measurement would most change the conclusion if it came out the
    other way, and what would it cost? Candidates the author sees: a second seed of the
    eight-update runs; the dense members' cost by task; a step sized by MATH's cost; a
    local RL replication for data matching and wall clock.
12. Is the dead-prompt screen (E16, E19) a fair throughput device, given that it skips
    prompts the model fails and could not learn from a rare success on them?
13. The per-type step (E18, E19) was set by a pooled cost model and over-stepped the
    reasoning tasks. Was that foreseeable from the records available before the run,
    and should the preregistration have used the per-task cost?

## On the paper

14. Title: which of the three candidates in `draft.md`'s header, or another?
15. Is the central model (Section 3, Eq. 1 and 2) stated early enough and used
    consistently? Are "member", "update", "useful part", "random part", "effective
    length" defined once and used with one name each?
16. What should be cut? What is project history rather than finding? What is a second
    question attached to the first?
17. Is the related-work placement accurate for the sources the author verified (Abdi,
    Hoy, Tulu 3, the model card) and appropriately hedged for those only reported by the
    literature agent?
18. What would a reviewer at a venue like MLSys or COLM ask first?

## The author's own doubts

- The slope along the exact gradient (7.3 +- 1.3) is one point at one length; the
  ceiling ratio 0.8 +- 0.15 inherits that.
- The projection `rho_hat` = 0.12 bounds `rho` times a cosine, and the two are not
  separated; the paper says so but leans on the bound.
- The best-net model is quadratic in the step and additive over updates; E19 shows the
  accumulated cost on MATH exceeds eight single-step costs, so the model understates
  long-run damage.
- The eight-update runs (E15, E19) are each one seed at one step size.
- The GPU-second comparison omits RL's pipeline overheads (communication, idling), which
  favour ES by an unmeasured amount, and ES's screen and LoRA factors are measured on one
  host.
- The literature placement for sources not checked at the source (GRESO, SPEC-RL, Ba et
  al., ESSA, Palleti and Singh) is kept out of the draft; if the reviewer knows them, say
  whether they bear on the claims.
