"""Accounting checks for the analyses that derive paper claims from saved probes."""
import importlib.util
from pathlib import Path

import pytest

PHASE2 = Path(__file__).resolve().parents[1] / 'experiments' / 'phase2'


def load(name):
    spec = importlib.util.spec_from_file_location(name, PHASE2 / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('gather', [0.0, 0.001])
def test_reconstruction_probes_count_the_gather_once(monkeypatch, gather):
    model = load('timemodel')
    key = ('mirrored_lr1', 512, 256)
    # Measured A and B probe slopes are 10 and 7 ms. Their difference is 3 ms,
    # irrespective of how much of A's measured slope comes from its weight gather.
    record = {'contraction_seconds': 0.010, 'contraction_local_seconds': 0.002,
              'allreduce_insitu_seconds': 0.005}
    monkeypatch.setattr(model, 'load_sweep', lambda *args: {key: {'A': 0.110, 'B': 0.107}})
    monkeypatch.setattr(model, 'load_contraction', lambda *args: {key: record})
    fabric = model.Fabric({6 * 512**2 * 4: 0.005}, {256: gather}, 'test')
    monkeypatch.setattr(model.Fabric, 'from_ladder', lambda *args: fabric)
    rows, _ = model.rows('test', {'sweeps': (), 'kind': 'test', 'ladder': 'unused'})
    assert rows[0]['delta_predicted'] == pytest.approx(0.003)
    assert rows[0]['contraction_measured'] == pytest.approx(0.010 - gather)
    assert record['contraction_seconds'] == 0.010  # historical records stay intact


def test_hypothetical_link_keeps_the_original_probe_gather_correction(monkeypatch):
    model = load('timemodel')
    key = ('mirrored_lr1', 512, 256)
    record = {'contraction_seconds': 0.010, 'contraction_local_seconds': 0.002,
              'allreduce_insitu_seconds': 0.005}
    monkeypatch.setattr(model, 'load_sweep', lambda *args: {key: {'A': 0.110, 'B': 0.107}})
    monkeypatch.setattr(model, 'load_contraction', lambda *args: {key: record})
    measured = model.Fabric({6 * 512**2 * 4: 0.005}, {256: 0.001}, 'measured')
    monkeypatch.setattr(model.Fabric, 'from_ladder', lambda *args: measured)
    hypothetical = model.Fabric({6 * 512**2 * 4: 0.005}, {256: 0.003}, 'hypothetical')
    rows, _ = model.rows('test', {'sweeps': (), 'kind': 'test', 'ladder': 'unused'},
                         fabric=hypothetical)
    assert rows[0]['contraction_measured'] == pytest.approx(0.009)


def test_cost_comparison_uses_only_configurations_shared_by_both_platforms():
    table = load('tb3')
    def records(extra_n, extra_ratio):
        return {(512, 64, 'iid_gaussian', 'bfloat16'): 10.0,
                (512, 64, 'mirrored_lr1', 'bfloat16'): 2.0,
                (512, 64, 'seed_regenerated', 'bfloat16'): 15.0,
                (512, extra_n, 'iid_gaussian', 'bfloat16'): 10.0,
                (512, extra_n, 'seed_regenerated', 'bfloat16'): 15.0,
                (512, extra_n, 'mirrored_lr1', 'bfloat16'): 10.0 * extra_ratio}
    gpu_records, tpu_records = records(128, 0.001), records(256, 1000)
    for recs in (gpu_records, tpu_records):
        recs.update({(512, 512, 'iid_gaussian', 'bfloat16'): 10.0,
                     (512, 512, 'mirrored_lr1', 'bfloat16'): 5.0,
                     (512, 512, 'seed_regenerated', 'bfloat16'): 15.0})
    # The plot compares all three variants, so seed must also fit on both platforms.
    tpu_records[512, 512, 'seed_regenerated', 'bfloat16'] = None
    count, gpu, tpu = table.common_rank_cost(gpu_records, tpu_records)
    assert count == 1
    assert gpu == pytest.approx(0.2)
    assert tpu == pytest.approx(0.2)
