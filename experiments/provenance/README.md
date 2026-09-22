# provenance

Both repositories were rewritten on 2026-09-22 to remove a personal account that three
files had named since 2026-08-01 (the benchmark runbook and the E17 and E17b results
READMEs). Every commit is still there with its message and its tree, except that those
files read `[account removed]` where they named it, but a commit's hash covers its
parent, so every commit after the first affected one has a new hash: 435 of 491 in
shardes, 392 of 429 here.

The two maps give every old hash and its new one, one `old new` pair per line:

- `commit-map-shardes-2026-09-22.txt`, the library, which is what result records stamp
  as `env.commit` (and `env.shardes.commit` after the split);
- `commit-map-shardes-paper-2026-09-22.txt`, this repository.

`translate_hashes.py` is the pass that rewrote every hash cited in this tree through
them: 2,688 mentions in 2,440 files, 2,297 of them result records, plus READMEs, configs
and logs. It translates a token only if the pre-rewrite repository resolves it to a commit
the map knows, keeps abbreviations at their length, and was checked by mapping every
file back and comparing byte for byte. Its arguments are the repository to rewrite, the
two pre-rewrite repositories, and the two maps.

A hash written down anywhere else, such as a pull request or a note, is an old one; look
it up here. `provenance_audit.py` keeps checking that every commit a record cites is
reachable, which after the translation they all are.
