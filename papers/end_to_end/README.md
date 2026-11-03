# End to end: ES fine-tuning against released RL checkpoints

The second paper. It retrains an open model with shardes' evolution strategies on exactly
the data a released RL checkpoint was trained on, from the same start, and compares the
two at matched rollouts.

`draft.md` is the manuscript draft (markdown; its header says how its numbers are to be
regenerated): the Tulu 3.1 comparison in Sections 1 to 7, and in Section 9 the Countdown
comparisons against GRPO trained here (the S3b to S4b stages of
`docs/end_to_end/10-when-does-es-work.md`). The plan is `docs/end_to_end/00-plan.md`, the
synthesis `docs/end_to_end/08-why-rl-learns-and-es-does-not.md`, the review package
`docs/end_to_end/review/`, and the experiments and their results `experiments/end_to_end/`.
