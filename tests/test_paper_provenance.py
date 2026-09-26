"""Prevent excluded timing records from becoming plotted results or memory failures."""
import importlib.util
import json
from pathlib import Path

import pytest

COUNTDOWN = Path(__file__).resolve().parents[1] / 'experiments' / 'countdown'


def load(name):
    spec = importlib.util.spec_from_file_location(name, COUNTDOWN / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('env', [{}, {'dirty_worktree': True}, {'dirty_worktree': None}])
@pytest.mark.parametrize('result', [{'seconds_median': 0.1}, {'status': 'oom'}])
def test_unverified_records_are_neither_timings_nor_oom(tmp_path, env, result):
    data = load('e17_data')
    path = tmp_path / 's=rank1__how=A__N=32__D=8.json'
    path.write_text(json.dumps({'env': env, **result}))
    assert data.cell('rank1', 'A', 32, 8, tmp_path) is None


@pytest.mark.parametrize('result, expected', [({'seconds_median': 0.1}, 0.1),
                                             ({'status': 'oom'}, 'oom')])
def test_clean_records_retain_measured_outcome(tmp_path, result, expected):
    data = load('e17_data')
    path = tmp_path / 's=rank1__how=A__N=32__D=8.json'
    path.write_text(json.dumps({'env': {'dirty_worktree': False}, **result}))
    assert data.cell('rank1', 'A', 32, 8, tmp_path) == expected


@pytest.mark.parametrize('pair, expected', [((None, 0.1), '--'), ((0.1, None), '--'),
                                           ((None, 'oom'), '--'), ((0.1, 0.2), '100'),
                                           (('oom', 0.2), '200'), (('oom', 'oom'), 'OOM')])
def test_comparison_requires_two_eligible_records(monkeypatch, pair, expected):
    monkeypatch.syspath_prepend(str(COUNTDOWN))
    table = load('tb5_e17')
    monkeypatch.setattr(table, 'cell', lambda strategy, how, n, d: pair['AB'.index(how)])
    assert table.best('rank1', 32, 8) == expected
