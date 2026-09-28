# CLAUDE.md

Instructions for Claude Code in this repository. The identity, the no-upstream-push rule
and the conda environment are in the tree-level file one directory up and apply here.

## What this is

The paper, its experiments and their results. The library is
[shardes](https://github.com/andreshernandez-spec/shardes), a pinned dependency
(`requirements.txt`), in its own repository since the split at `25b6cc6`. Nothing here is
a package.

## Rules

1. **Every number in the paper or a README has a script behind it**, committed and
   re-runnable, and a record stamping the commit and the environment. A number that
   cannot be reproduced from a clean checkout gets deleted.
2. **Configs are committed before the run** and cited by SHA in the results. Never add a
   configuration mid-run.
3. **A result needs a clean, pushed library.** `harness.capture_env` records which shardes
   ran. A campaign driver refuses a dirty or unpushed one; a probe may run against one
   and its record says so.
4. **Never rebase or force-push a branch that a record cites.** Records stamp commits, and
   a rebase orphans them. It happened once (`82ab1ec`, 34 records) and went unnoticed for
   three weeks. Merge main in instead. `experiments/provenance_audit.py` is the check,
   and it runs in CI. The one exception: on 2026-09-22 both repositories were rewritten
   to remove a personal account from three files. Every hash cited in this tree was
   translated, and `experiments/provenance/` holds the old-to-new maps for anything
   written down elsewhere.
5. **Moving the pin is a commit of its own**, with the driver tests green against the new
   library commit.
6. **Generated tables are never edited by hand.** `make -C paper tables` rewrites them, and
   CI fails if that changes a byte.
7. **A count or setting a record can supply is never typed into prose.** It comes from a
   generated macro, one macro per measurement family: `experiments/protocol_constants.py`
   reads the records, writes `paper/generated/protocol.tex`, and stops if a family's
   records disagree with each other or with their config. A sentence that needs a new
   constant gets a new family or macro there, and a sentence may not merge two families.
   Added 2026-09-28, after the setup said the ranking step was timed with seven repeats:
   the in-context ranking records were, the 84-configuration benchmark used five.
   Results figures follow the same rule: the generator that computes a figure writes it
   as a macro, rounded once from the exact value (`paper_evidence.py` writes `results.tex`,
   `tb3.py` `cost-common.tex`, `tb1.py` `reference.tex`, all in `paper/generated/`). A
   sentence that states a result in words ("every", "all", "only") has that claim checked
   by the same generator, which fails the build if a record ever contradicts it. Typed by
   hand, the abstract had said 1.5x for an exact 1.449.

## Records from before the split

Almost every record here cites a commit of shardes, from when both were one tree. See the
README. Do not rewrite those stamps.

## Tests

`pytest` runs the driver tests on CPU in seconds. Four of them need a checkout of the
library with history (`SHARDES_REPO`) and skip without one.
