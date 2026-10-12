# 11. The published record: ES against gradient RL, failures, theory, predictors

Compiled 2026-10-12 by a literature agent for `10`; not independently verified except
where noted. How each source was read: [PDF] from the paper's text, quoted; [HTML] from
the arXiv HTML through a summarizer, numbers checked against the page but wording may be
paraphrased; [abs] abstract only. Ratios marked "mine" are the agent's arithmetic from the
stated numbers. Hoy et al.'s Section 6 and Abdi et al.'s abstract were also read directly
for `08` and the draft.

## 1. Head-to-head comparisons

**Control.**

- [Salimans et al. 2017](https://arxiv.org/abs/1703.03864) [PDF]. Atari, 51 games, 1B
  frames per game on 720 CPUs against A3C's 320M frames: ES better on 23 games, worse on
  28. Their statement: matched A3C "while using between 3x and 10x as much data", offset
  by "roughly 3x" less computation. MuJoCo against TRPO (6 seeds, 5M timesteps): Table 1
  gives the ratio of ES to TRPO timesteps to reach 100% of TRPO's progress: Swimmer 0.30,
  HalfCheetah 0.58, InvertedPendulum 0.88, InvertedDoublePendulum 1.23, Hopper 6.94,
  Walker2d 7.88. Section 3.2 argues that steps scale with "the difficulty, or intrinsic
  dimension, of the optimization problem", not d.
- [Mania, Guy, Recht 2018, ARS](https://arxiv.org/abs/1803.07055) [PDF]. Linear policies.
  Table 4 timesteps to threshold (ARS / ES / TRPO): Hopper 2.0e6 / 3.16e7 / 4.56e6;
  Walker2d 8.14e6 / 3.79e7 / 4.81e6; HalfCheetah 5.86e5 / 2.88e6 / 5.0e6; Swimmer 6.0e4
  / 1.39e6 / 4.59e6. "At least 15 times more computationally efficient than ES". Over 100
  seeds ARS succeeds about 70% of the time, 20% on Walker2d.
- [Such et al. 2017, deep GA](https://arxiv.org/abs/1712.06567) [PDF]. 13 Atari games, a
  4M-parameter CNN, population 1,000+1, sigma 0.002, 1B frames (DQN 200M). Best score: GA
  3, ES 3, DQN 3, A3C 4 games. Random search with the same frames beats DQN on 3, ES on
  3, A3C on 6.
- [Chrabaszcz et al. 2018](https://arxiv.org/abs/1802.08842) [HTML]; [Conti et al.
  2018](https://arxiv.org/abs/1712.06560) [HTML]: about 2B frames, 5,000 samples per
  generation.
- [Wong et al. 2024](https://arxiv.org/abs/2402.06912) [HTML]: linear policies, 5e5 to
  1e7 timesteps; CMA-ES ahead on Swimmer, behind PPO and SAC on Humanoid (830 against
  2,490 and 7,282) and HalfCheetah; ES "can take more time steps to converge" on complex
  tasks.
- [Martinez et al. 2026](https://arxiv.org/abs/2604.00066) [HTML], seeds not stated: PPO
  "20x faster" on HalfCheetah; a weak source.

**Language models.**

- [Qiu et al. 2025](https://arxiv.org/abs/2509.24372) [PDF]. Countdown, 7 models 0.5B to
  8B, N = 30, sigma 1e-3, alpha 5e-4, 500 iterations, 200 training prompts, greedy
  decoding. Appendix A.1: "For all the ES and RL baselines, the total number of sample
  evaluations was the same". ES / GRPO-z(30) / Dr.GRPO-v accuracy: Qwen-3B 60.5 / 32.5 /
  43.8; Qwen-7B 66.8 / 52.8 / 57.5; Llama-8B 61.2 / 51.3 / 50.2; Llama-1B 16.8 / 13.0 /
  13.9. Conciseness (2 training prompts, N = 30, 1,000 iterations): ES reward 0.889 to
  0.919 at KL 0.27 to 0.81 against GRPO 0.867 to 0.911 at KL 0.86 to 1.59. FLOPs per
  sample 2PL (ES) against 8PL (GRPO). The "ES smooths the jagged reward landscape"
  mechanism is stated as a hypothesis: "Direct evidence for this hypothesis still needs
  to be obtained."
- [Hoy, Wang, Pan 2026](https://arxiv.org/abs/2604.01499) (COLM) [HTML, v3].
  Qwen3-4B-Instruct, 200 training prompts, ES N = 30, sigma 0.0015, alpha sigma/2; GRPO
  80 steps, batch about 63, G = 8, no KL. Appendix B.8: ES at 100 iterations uses 3.72x
  GRPO's FLOPs. Generations 100 x 30 x 200 = 600,000 against 80 x 63 x 8 = 40,320, about
  15x at 100 iterations and 45x at 300 (mine). Accuracy ES100 / ES300 / GRPO: Countdown
  75.1 / 78.6 / 74.5; Math 72.4 / 74.0 / 68.8; Chemistry 68.1 / 76.5 / 74.9; BoolQ 87.2
  / 88.1 / 86.6. Update norm after four sequential tasks 173.0 against 1.84; off-task KL
  2.34x GRPO's.
- [Abdi et al. 2026](https://arxiv.org/abs/2601.20861) (ACL short) [HTML].
  Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct; ES N = 30, sigma 1e-3, alpha 5e-4; GRPO
  30 rollouts "to match ES population size", batch 200; rollouts roughly matched (the
  agent's reading). ES / GRPO: Qwen Countdown 53.0 / 56.4, GSM8K 77.4 / 80.4, MATH 59.1 /
  63.2, OlympiadBench 15.2 / 18.2; Llama Countdown 15.2 / 37.6, GSM8K 55.2 / 53.8, MATH
  32.2 / 35.6, Olympiad 5.6 / 5.9. GRPO wins 7 of 8 cells.
- [Ba et al. 2026](https://arxiv.org/abs/2608.27351) [HTML]. FLOP-matched ES N = 32
  against GRPO G = 8 (4x the generations for ES, mine). GRPO wins Pass@1 (Qwen-1.5B GSM8K
  42.9 against 41.5; DeepSeek-R1-Distill-1.5B math 52.9 against 49.9), ES wins Pass@32
  (80.9 against 79.9). ES L2 drift 40.7 to 44.1x GRPO's. Population needed falls with
  scale: N = 16 suffices at 1.5B and 3B, N = 32 at 0.5B.
- [Palleti and Singh](https://alphaxiv.org/blog/es-for-fine-tuning-llms) (blog) [HTML];
  budget matching not stated. Qwen2.5-3B-Instruct GSM8K, ES / GRPO: 10% data 89.0 / 85.5;
  40% 86.5 / 90.9; 70% 83.0 / 89.6. Countdown 10%: 36.0 / 34.1. Base models on Countdown:
  Qwen ES 15 against GRPO 58.4, Llama ES 2%: "ES collapses on both base models". N = 30
  against 8 adds 6 to 10 points on Countdown.
- [EGGROLL 2025](https://arxiv.org/abs/2511.16652) [HTML]. RWKV-7 1.5B Countdown 35%
  against GRPO 23% "given the same hardware and wall-clock time", with 1,024 x 618 against
  64 x 915 generations (about 10.8x, mine); 7B GSM8K 8,192 x 260 against 256 x 340 (about
  24x, mine). Integer-RNN pretraining matched a backprop transformer at "around 180 times
  more GPU-hours"; a MeZO-style population of 2 "significantly underperforms".
- [ESSA](https://arxiv.org/abs/2507.04453) [HTML]: about 2,000 parameters (LoRA singular
  values); no sample counts. [Hyper-ES](https://arxiv.org/abs/2608.05541) [HTML]: plain
  CMA-ES plus LoRA 52.76 against GRPO plus LoRA 56.23 against Hyper-ES 57.13 (Qwen2.5-0.5B
  arithmetic average); CMA-ES update cosine to GRPO's 8.6e-5. [ESSAM](https://arxiv.org/abs/2602.01003)
  [abs]; [QES](https://arxiv.org/abs/2602.03120) [abs]; [Cho and Han 2026](https://arxiv.org/abs/2607.19408)
  [HTML] has no RL baseline.

## 2. Reported failures

- Salimans: worse than A3C on 28 of 51 games [PDF]. Conti: ES stuck in the Humanoid
  deceptive trap in every run (median 5.3 against 37.1 for NSRA-ES); Montezuma 0.0;
  Frostbite 367 against 3,785 [HTML]. Chrabaszcz: local optima on Seaquest and Enduro
  [HTML]. ARS: Walker2d solved on 20% of 100 seeds [PDF].
- [MeZO](https://arxiv.org/abs/2305.17333) [PDF]: without a prompt, RoBERTa-large SST-2
  51.9 against 89.6, TREC 19.5 against 66.7 (Table 5); 100K MeZO steps against 1,000
  fine-tuning steps. [ZO-Bench](https://arxiv.org/abs/2402.11592) [HTML]: 20,000 ZO
  against 625 FO iterations; WinoGrande gap 5 to 6 points on OPT-13B; without prompt
  alignment ZO-SGD loses about 10 points.
- Abdi: Llama-1B Countdown 15.2 against 37.6; HellaSwag drops about 10% under ES while
  GRPO is stable; ES weight change "three orders of magnitude larger". [Schweighofer et
  al. 2026](https://arxiv.org/abs/2605.30148) [HTML]: ES N = 30 on Countdown costs GSM8K
  -9.5, HellaSwag -8.2, MMLU-Pro -6.5 (median prior-task -2.5 against GRPO +0.3); GRPO also
  forgets with ProofWriter as target. Hoy: 300 ES iterations in sequential training
  "causes pronounced degradation in earlier tasks".
- Cho and Han: N = 2 with z-score normalization "collapses toward zero accuracy in every
  size/task cell"; Qwen2.5-0.5B on GSM8K "only avoids collapse without improving".
  Palleti and Singh: collapse on base models.

## 3. Theory of when zeroth-order is slow

- Nesterov and Spokoiny (FoCM 2017): Gaussian random-direction methods need at most n
  (dimension) times more iterations than gradient methods [secondary]. MeZO [PDF]:
  without assumptions the zeroth-order step rate is n/(d+n-1) of SGD's; Assumption 1
  (local r-effective rank of the Hessian) and Theorem 1: the per-step decrease is slowed by
  a factor of order r/n; Lemma 3: iterations O((r/n + 1) x the SGD rate). Justified by
  fine-tuning subspaces of under 200 parameters ([Aghajanyan et al. 2020](https://arxiv.org/abs/2012.13255)).
  ZO-Bench: estimator variance O(d/q).
- Hyper-ES Lemma 1: Pr(|<u, e1>| >= tau) <= 2G exp(-d tau^2 / 2): G random directions
  align with a fixed descent direction only if log G is of order d.
- Hoy Section 6 and Appendix D [HTML]: linearized ES step theta_{t+1} = H theta_t + eta_t
  with eta ~ N(0, alpha^2 / N I); E|theta_ES - theta_0|^2 = O(r) + alpha^2 T (d - r) / N:
  the signal lives in the rank-r curved subspace, the diffusion in the d - r flat ones.
  Flat landscape: E|delta theta|^2 = alpha^2 d / N. Linear landscape: the on-gradient
  fraction chi = (1 + (N+1) s) / (d + (N+1) s), negligible at N = 30, d about 4e9. E
  cos(ES, GRPO) = O(sqrt(r / d)). Empirically |delta theta|^2 grows linearly in t (R^2
  0.9999), d_eff / d_total = 96.4% (Qwen) and 102.4% (Llama). Schweighofer gives the same
  E|theta_T - theta_0|^2 = alpha^2 T d / N.
- EGGROLL Theorem 1 [HTML]: with sigma_d = o(d^{-1/2}) the ES gradient converges to the
  true gradient; for sigma_d much larger than d^{-1/2}, |grad J| = Theta(sigma_d^2 d)
  diverges. [Vemula et al. 2019](https://arxiv.org/abs/1901.11503) [abs]: parameter-space
  exploration cost depends on the parameter dimension, action-space on the action
  dimension and horizon. [Zhang, Clune, Stanley 2017](https://arxiv.org/abs/1712.06564)
  [HTML]: about 3M-parameter MNIST net, 10,000 perturbations give 3.9% correlation with
  the true gradient, about 274,000 for 20%. [Lehman et al. 2018](https://arxiv.org/abs/1712.06568):
  ES optimizes the sigma-smoothed objective.
- Gradient diversity: [Yin et al. 2018](https://arxiv.org/abs/1706.05699) [PDF],
  Definition 4, Delta_S(w) = sum |grad f_i|^2 / |sum grad f_i|^2; Theorem 3: batch size up
  to delta n Delta_S + 1 keeps a linear speedup. The agent found no paper tying Delta_S
  to ES or zeroth-order efficiency. Nearest: [GRZO 2026](https://arxiv.org/abs/2606.02857)
  [abs] (one perturbation per example), and Cho and Han's correctness correlation rho.
  The agent's inference: ES's signal is the gradient of the batch-mean reward, so low
  sharedness (high Delta_S) shrinks |v| in Hoy's chi while the diffusion term is
  unchanged.

## 4. A priori predictors

- [Li et al. 2018](https://arxiv.org/abs/1804.08838) [PDF]: intrinsic dimension d_int90,
  measured with ES as the trainer: InvertedPendulum 4, Humanoid 700, Pong 6,000 (MNIST FC
  750, CIFAR-10 9,000). No ES-success correlation claimed.
- [Liang et al. 2026](https://arxiv.org/abs/2602.00170) [HTML]: ES as a landscape probe;
  the expected best-of-N improvement Delta*_N(theta, sigma) as an accessibility proxy;
  saturates by N about 30 to 40 on Qwen2.5 0.5B to 7B (GSM8K, ARC-C, WinoGrande, sigma
  3e-4); a quadratic-model noise floor (alpha / 4) d_eff(alpha) sigma^2 / N; no failure
  documented.
- Cho and Han: zero-advantage probability q about 1 / sqrt(4 pi B p0 (1 - p0) (1 - rho));
  a "zero-training probe" (200 perturbation pairs) predicted q with MAE 0.020 over 12
  configurations; availability alone did not explain collapse (normalization did).
- Hoy's d_eff from update-norm growth; [ASEBO 2019](https://arxiv.org/abs/1903.04268)
  [abs] estimates the gradient subspace by online PCA. Qiu's jagged-landscape hypothesis
  has no test.

## Summary table

| Source | Domain | Task | ES setting | RL setting | Budget ratio (ES:RL) | Outcome |
|---|---|---|---|---|---|---|
| Salimans 2017 [PDF] | Atari | 51 games | CNN, 1B frames, 720 CPUs | A3C, 320M frames | 3.1x data (stated "3x to 10x") | ES better 23, worse 28 |
| Salimans 2017 [PDF] | MuJoCo | 6 tasks | 6 seeds | TRPO, 5M steps | 0.30 to 7.88x timesteps | ES ahead on 3, 7 to 8x behind on Hopper, Walker2d |
| Mania 2018 [PDF] | MuJoCo | 4 tasks | ARS, linear | TRPO | ES:TRPO 0.3 to 7.9x | ARS beats TRPO on 3 of 4; Walker2d 20% seed success |
| Such 2017 [PDF] | Atari | 13 games | GA pop 1,000, 1B frames | DQN 200M, A3C | 5x frames vs DQN | GA beats each of ES/A3C/DQN on 6 of 13 |
| Conti 2018 [HTML] | MuJoCo | Humanoid trap | ES | none | n/a | ES stuck every run |
| Wong 2024 [HTML] | MuJoCo | Humanoid | CMA-ES, linear | PPO, SAC | <= 1e7 steps each | 830 vs 2,490 (PPO) vs 7,282 (SAC) |
| Qiu 2025 [PDF] | LLM | Countdown, 7 models | N=30, sigma 1e-3, 500 it | PPO/GRPO, equal sample evaluations | 1x | ES wins all 7, e.g. Qwen-3B 60.5 vs 43.8 |
| Hoy 2026 [HTML] | LLM | 4 tasks, Qwen3-4B | N=30, sigma 1.5e-3 | GRPO G=8, 80 steps | 3.72x FLOPs, ~15x rollouts (mine) | ES >= GRPO; update norm 94x |
| Abdi 2026 [HTML] | LLM | 4 math tasks, 1B/1.5B | N=30, sigma 1e-3 | GRPO 30 rollouts | ~1x | GRPO wins 7 of 8; Llama Countdown 15.2 vs 37.6 |
| Ba 2026 [HTML] | LLM | GSM8K, math | N=32 | GRPO G=8 | FLOP-matched (4x generations) | GRPO wins Pass@1, ES wins Pass@32 |
| Palleti and Singh [HTML] | LLM | GSM8K, Countdown, 3B | N=8/30 | GRPO | not stated | ES wins at 10% data, loses at 40 to 70%; collapses on base models |
| EGGROLL 2025 [HTML] | LLM | Countdown, RWKV-7 1.5B | pop 1,024, low-rank | GRPO 64/step | ~10.8x generations, equal wall-clock | 35% vs 23% |
| Hyper-ES [HTML] | LLM | arithmetic, 0.5B | CMA-ES on LoRA | GRPO+LoRA | not stated | 52.76 vs 56.23; cosine 8.6e-5 |
| MeZO 2023 [PDF] | LM | RoBERTa-large, OPT | n=1, 100K steps | Adam FT, 1K steps | 100x steps | within a few points with a prompt; 51.9 vs 89.6 without |
| ZO-Bench 2024 [HTML] | LM | OPT-13B and others | ZO, 20,000 it | FO, 625 it | 32x iterations | gap 1 to 6 points, widens with difficulty |

## Relation to our measurements (the agent's reading)

Our finding (ES at N = 16 ineffective at matched rollouts on the Tulu 3.1 stream, needing
roughly 30 to 100x RL's rollouts, with the random component damaging reasoning) agrees
with Abdi (GRPO wins 7 of 8 at matched rollouts), Ba (GRPO wins Pass@1 at matched FLOPs),
Hoy (ES matches GRPO only at about 15 to 45x the generations), EGGROLL (11 to 24x
generations behind a wall-clock win), MeZO and ZO-Bench (32 to 100x steps), and with
every drift measurement (Hoy's alpha^2 T (d - r) / N, Abdi, Schweighofer, Ba's 40x L2).
It disagrees with Qiu's matched-sample Countdown wins and with Cho and Han's N = 2
success on GSM8K, and both disagreements sit on single, high-sharedness tasks with
capable instruct models, which is where our own Countdown run also learns. The classic
control gap (0.3 to 8x against TRPO) is smaller than ours, consistent with Li's intrinsic
dimensions of 4 to 6,000 there. The most informative settings for a one-iteration
diagnostic: Abdi's four tasks on the same two models (ES loses mildly on Qwen, heavily on
Llama Countdown); Palleti and Singh's base-against-instruct Countdown collapse; Cho and
Han's Qwen-0.5B GSM8K no-progress case against 1.5B and 7B; Hoy's Chemistry task (slowest)
against BoolQ; and a single-task against mixed-stream contrast on one model.
