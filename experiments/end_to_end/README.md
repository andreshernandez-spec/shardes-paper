# experiments/end_to_end

The ES-vs-RL paper's experiments (`papers/end_to_end/`, plan in
`docs/end_to_end/00-plan.md`). Nothing from the placement paper lives here.

## Phase 0: what we compare against, pinned

| file | what it is |
|---|---|
| `releases.py` | every external artifact (models, datasets, open-instruct commits, the W&B run), pinned to full commit hashes |
| `templates.py` | the `olmo_thinker` and `tulu` chat templates, copied verbatim from open-instruct, with their sha256 |
| `provenance.py` | the `env` block every record here carries, in the shape `../provenance_audit.py` reads |
| `pin_releases.py` | checks every pin still resolves and records the weight hashes behind it, per RL branch, with duplicates grouped. Writes `pins/` |
| `olmo3_if_data.py` | rebuilds the Olmo 3 RL-Zero IF run's own training set and per-step prompt stream from open-instruct's code path and the run's logged config, and compares them with the later Dolci release. Writes `data/olmo3_if/` |

Records are written once and never overwritten. Each carries `env` (this repository's
commit and dirty flag, the library block, package versions). Hugging Face and
open-instruct hashes are stored as `revision` or `sha`, never under `commit`, which the
provenance audit would read as a citation of this repository.

Run from this directory, in the `open-source` environment, network needed, no GPU:

```sh
python pin_releases.py --out pins/releases-<date>.json
python olmo3_if_data.py --out data/olmo3_if
```

## Phase 0 results

See `docs/end_to_end/00-plan.md`, "Phase 0 results". In short: every pin resolves;
Olmo 3 RL-Zero-IF `step_100` and `step_1000` are the same weights; the IF run's
training set (13,179 rows) and its per-step prompt stream are rebuilt exactly, and
the released Dolci set is that set in the run's pre-shuffle order.

## Throughput probe

`probe/`: `probe_throughput.py` decodes one ES member batch at a time with vLLM from
each start checkpoint, with the run's own prompts, template, stop strings and cap;
`probe.yaml` is the config, run once per GPU type; `probe_worker.py` is the vLLM
worker extension that times a full in-place weight rewrite; `cost.py` turns the
records into the cost of the matched ES runs; `pod.sh` bootstraps a pod and launches
the probe detached. Its own venv: `requirements-vllm.txt`.

Tests: `pytest tests/end_to_end` from the repository root.
