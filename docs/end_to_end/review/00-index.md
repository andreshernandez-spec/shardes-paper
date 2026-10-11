# Review package: ES against RL at matched data on Tulu 3.1

Prepared 2026-10-11 so that a reader with no prior context, a fresh Claude instance
included, can review the work and give feedback. Everything is in the `shardes-paper`
repository, branch `end-to-end-layout` (draft PR #11), worktree
`~/private/open-source/.worktrees/shardes-paper-e2e`.

## The documents, in reading order

| file | what it is | read it for |
|---|---|---|
| `docs/end_to_end/review/00-index.md` | this file | orientation and the reviewing brief |
| `docs/end_to_end/review/01-goal-and-problems.md` | the goal, the decisions in order, the problems met and what each cost, the open questions | the context a paper would omit |
| `docs/end_to_end/review/03-metrics-explained.md` | every measurement defined from scratch, with the numbers for both datasets | before reading results |
| `docs/end_to_end/review/02-experiments-and-evidence.md` | one entry per experiment: question, design, record, code version, cost, result, what it established, caveats | the evidence itself |
| `papers/end_to_end/draft.md` | the paper draft (markdown, numbers transcribed, see its header) | the argument as a paper |
| `docs/end_to_end/08-why-rl-learns-and-es-does-not.md` | the synthesis written for the repository, section by probe | a second, longer telling of the chain |
| `docs/end_to_end/09-best-achievable.md` | the preregistration and result of the last stage | an example of how runs are committed before data |
| `docs/end_to_end/review/04-review-questions.md` | what feedback is wanted, the doubts the author already has, and the standard to apply | before writing the review |
| `docs/end_to_end/00-plan.md` to `07-tulu31-long.md` | the plan and the preregistrations of the first six weeks | only if a detail is in doubt |
| `experiments/end_to_end/runs/*/README.md` | the results READMEs, one per record, each citing its script and code version | to check any number |

## How the evidence is organized

- Every number in any document comes from a record under `experiments/end_to_end/runs/`
  and a script under `experiments/end_to_end/es_vllm/`; the README beside the record
  names the command that prints the number and the commit of the code that produced it.
- Runs were preregistered in `docs/end_to_end/0N-*.md` before they ran, with predictions
  and pass/fail rules; the result sections were added after. The probes of 2026-10-07 to
  10 carry their prediction in the script's docstring, committed before the run.
- `experiments/provenance_audit.py --legacy ~/private/open-source/shardes` checks that
  every commit a record cites is reachable; `pytest` runs the driver tests on CPU.

To print the numbers behind a README, from `experiments/end_to_end/` with the
`.venv-vllm` interpreter (`../../.venv-vllm/bin/python`), for example:

    python -m es_vllm.lowrank_check --analyze runs/lowrank-check/steps.json
    python -m es_vllm.gradient_check --analyze runs/gradient-check
    python -m es_vllm.lowrank_run --analyze runs/lowrank-run-1024
    python -m es_vllm.block_check --analyze runs/block-check

## The reviewing brief

The reviewer is asked for feedback, not for changes: do not edit files, run GPUs, push,
or open issues. Read in the order above, run any analysis command to check a number, and
write the review as a document (markdown) with:

1. the three sentences the paper rests on (question, answer, scope), as the reviewer
   understood them, so the author can see whether they came across;
2. findings ordered by severity: claims wider than their evidence, gaps in the chain of
   inference, missing controls, statistical problems, errors in numbers or arithmetic;
3. what would most strengthen the work, with an estimate of what it would cost;
4. presentation: what was hard to follow, what should be cut, what is missing;
5. an answer to each question in `04-review-questions.md`.

The standard is the one the repository's paper-writing rules set
(`.claude/skills/paper-writing/SKILL.md` at the tree root): a reader who has never seen
the repository can follow the argument and check every number, and the author can defend
every sentence. Unsupported claims and reasoning that only works with the project's
history in mind are the findings that matter most.
