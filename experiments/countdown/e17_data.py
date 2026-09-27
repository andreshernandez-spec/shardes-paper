"""Paper-eligible E17b records. Unknown or dirty provenance is excluded."""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / 'results-e17b'


def load_record(strategy, how, n, d, results=RESULTS):
    path = results / f's={strategy}__how={how}__N={n}__D={d}.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get('env', {}).get('dirty_worktree') is not False:
        return None
    return record


def cell(strategy, how, n, d, results=RESULTS):
    record = load_record(strategy, how, n, d, results)
    if record is None:
        return None
    if record.get('status') == 'oom':
        return 'oom'
    return record.get('seconds_median')
