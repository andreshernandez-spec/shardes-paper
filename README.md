# shardes-paper

The papers built on [shardes](https://github.com/andreshernandez-spec/shardes), every
experiment behind them, and their results:

- *Update-contraction placement in sharded evolution strategies on GPUs and TPUs*,
  in `paper/`;
- an end-to-end comparison of ES fine-tuning against released RL checkpoints, in
  `papers/end_to_end/` (in progress, no manuscript yet).

The library it measures is [shardes](https://github.com/andreshernandez-spec/shardes),
installed here as a pinned dependency. This repository was split out of that one, with
its history, at `25b6cc6`.

## Which commit produced a result

Every result record stamps the commit it was measured at. There are two kinds of record
and they resolve in different places.

- **A record with no `shardes` block, or one naming the same commit as `commit`,** was
  written while the library and the experiments were one repository. Its `commit` is a
  commit of **shardes**, not of this repository. Check that SHA out there and the driver,
  the config and the library are all in one tree, as run. That repository keeps that
  history for this reason. Almost every record here is of this kind.
- **A record whose `shardes` block names a different commit** was written after the
  split. `commit` is this repository's, `shardes.commit` is the library's, and
  `shardes.source` says how the library got onto the path.

The history here was rewritten by the split, so this repository's own SHAs are new.
Commit messages were left naming the original ones, which is what the records and the
results READMEs cite.

`python experiments/provenance_audit.py --legacy ../shardes --repo shardes=../shardes`
checks that every commit any record cites can still be fetched. It runs in CI.

## The library pin

`requirements.txt` pins shardes to one commit. A result is reproduced by checking this
repository out at the commit its record stamps and installing from that file, so the
library commit follows from this repository's commit and nobody types it by hand. The
pin is always a full commit hash, since a tag can move; a release's tag is named in a
comment beside it, and CI checks that the two agree.

To work on the library and the experiments together, install a clone of it editable
(`pip install -e ../shardes`). Records then say `"source": "checkout"` with that
checkout's commit and whether it was dirty. A campaign result should never be stamped
against a dirty or unpushed library.

## Rebuild the paper

```sh
pip install -r requirements.txt
make -C paper            # re-emits every generated table from its script, then the PDF
```

The generated tables reproduce byte for byte and the figures pixel for pixel from the
committed results. Nothing here needs an accelerator.

## Rerun an experiment

Each results directory has a README naming the driver, the config, the platform, the
commit, the cost and every incident. Configs are committed before the run and cited by
SHA. `docs/06-benchmark-runbook.md` has the mechanics for Kaggle and rented pods;
`requirements-models.txt` adds what the real-model experiments need, and
`requirements-grpo.txt` the GRPO reference arm.

## One directory per paper

Each paper gets its own directory: the manuscript in `papers/<name>/`, its experiments
and results in `experiments/<name>/`, and, where they belong to that paper alone, its
tests in `tests/<name>/` and its plans in `docs/<name>/`. Anything shared stays at the
top level: the provenance harness and audit, their tests, the benchmark runbook, the
cost model.

The first paper predates this rule. Its manuscript is `paper/` and its experiments are
the top-level `experiments/phase0` to `experiments/countdown`, and they stay there,
because its build, its CI checks and the results READMEs refer to those paths.

## Layout

```
paper/                the placement paper; generated/ is written by scripts, never by hand
papers/end_to_end/    the ES-vs-RL paper
experiments/          phase0 (alignment law), phase1, phase2 (systems), countdown (real
                      model): the placement paper. harness.py (provenance) and
                      provenance_audit.py are shared
experiments/end_to_end/  the ES-vs-RL paper's experiments and results
tests/                tests of the experiment drivers. CPU, no network, seconds
tests/end_to_end/     the ES-vs-RL paper's driver tests
docs/                 campaign plans, preregistrations, the benchmark runbook
docs/end_to_end/      the ES-vs-RL paper's plan
PLAN.md               the placement paper's research program, phases and gates
```

## Tests

```sh
pytest                                   # seconds
SHARDES_REPO=../shardes pytest           # also runs the four tests that need the library's
                                         # history or its tests/gpu, and skip without it
```
