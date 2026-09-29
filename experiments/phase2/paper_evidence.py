#!/usr/bin/env python
"""Reproduce the worked example and supporting numerical checks in the paper.

Uses saved records and the model configuration, without an accelerator. --latex
writes the worked example and paper/generated/results.tex, the macros that carry the
results figures the prose cites; stdout contains the remaining numerical audits.

A figure the abstract and a section both state is computed once here and rounded once,
from the exact value. Typed by hand, the abstract said 1.5x where the exact lower end is
1.449: it had rounded the section's already-rounded 1.45.
"""
from __future__ import annotations

import argparse
import collections
from decimal import ROUND_CEILING, Decimal
import json
import math
import re
from pathlib import Path
import statistics
import sys

import timemodel

HERE = Path(__file__).resolve().parent
COUNTDOWN = HERE.parent / 'countdown'
sys.path.insert(0, str(HERE / 'multihost'))
sys.path.insert(0, str(HERE.parent))
from protocol_constants import rounded, spelled  # noqa: E402  (one way to write a number in prose)
import costmodel  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def sweep_records(spec, devices):
    out = {}
    for folder in spec['sweeps']:
        for path in (HERE / folder).glob('*.json'):
            r = read(path)
            c = r.get('config', {})
            if c.get('mode') == 'strong' and c.get('devices') == devices and 'seconds_median' in r:
                out[c['strategy'], c['d_model'], c['population'], c['how']] = r
    return out


def resolved(a, b):
    """The repeat envelopes do not overlap; this is not a confidence test."""
    return max(a) < min(b) or max(b) < min(a)


def block_checks():
    report, examples, precision = {}, [], []
    for platform, spec in timemodel.PLATFORMS.items():
        rows, _ = timemodel.rows(platform, spec)
        repeats = sweep_records(spec, 8)
        audit = {name: {'correct': 0, 'resolved_correct': 0, 'resolved_misses': []}
                 for name in ('measured_components', 'ideal_isolated', 'ideal_in_context')}
        n_resolved, resolution = 0, []
        for r in rows:
            key = r['strategy'], r['d_model'], r['population']
            is_resolved = resolved(*(repeats[*key, h]['seconds_all'] for h in 'AB'))
            n_resolved += is_resolved
            resolution.append([*key, is_resolved])
            c, ag = r['contraction_measured'], r['allgather_seconds']
            predictions = {'measured_components': r['delta_predicted'],
                           'ideal_isolated': c * 7 / 8 + ag - r['allreduce_seconds'],
                           'ideal_in_context': c * 7 / 8 + ag - r['allreduce_insitu']}
            for name, pred in predictions.items():
                agrees = (pred > 0) == (r['delta_measured'] > 0)
                audit[name]['correct'] += agrees
                audit[name]['resolved_correct'] += agrees and is_resolved
                if is_resolved and not agrees:
                    audit[name]['resolved_misses'].append(list(key))
        for strategy in ('seed_regenerated', 'mirrored_lr1'):
            r = next(r for r in rows if (r['strategy'], r['d_model'], r['population']) == (strategy, 2048, 256))
            examples.append((platform, r))
        low = [r for r in rows if r['strategy'] in timemodel.LOW_RANK and r['d_model'] == 2048]
        report[platform] = {'configurations': len(rows), 'resolved': n_resolved, 'audit': audit,
            'resolution': resolution,
            'drawn_full_rank_speedup': [min(r['t_A']/r['t_B'] for r in rows if r['strategy'] in ('iid_gaussian','seed_regenerated')),
                                        max(r['t_A']/r['t_B'] for r in rows if r['strategy'] in ('iid_gaussian','seed_regenerated'))],
            'large_lowrank_naive_ms': [min(r['contraction_measured']*7/8+r['allgather_seconds']-r['allreduce_seconds'] for r in low)*1e3,
                                      max(r['contraction_measured']*7/8+r['allgather_seconds']-r['allreduce_seconds'] for r in low)*1e3],
            'large_lowrank_measured_ms': [min(r['delta_measured'] for r in low)*1e3,max(r['delta_measured'] for r in low)*1e3],
            'dense_speedup': [min(r['t_A']/r['t_B'] for r in rows if r['strategy'] == 'iid_gaussian'),
                              max(r['t_A']/r['t_B'] for r in rows if r['strategy'] == 'iid_gaussian')],
            'large_lowrank_split_slower_pct': [min(r['t_B']/r['t_A']-1 for r in low)*100,
                                               max(r['t_B']/r['t_A']-1 for r in low)*100]}
        one = sweep_records(spec, 1)
        precision.append({(d,n,h): r['seconds_median']/one['iid_gaussian',d,n,h]['seconds_median']
                          for (s,d,n,h),r in one.items() if s=='mirrored_lr1' and ('iid_gaussian',d,n,h) in one})
    common = precision[0].keys() & precision[1].keys()
    report['matched_precision'] = {'count': len(common), 'configurations': sorted(common),
        'rank1_over_dense': [math.exp(statistics.mean(math.log(rs[k]) for k in common)) for rs in precision]}
    return report, examples


def span(lo, hi, places):
    return f"{rounded(lo, places)}--{rounded(hi, places)}"


def example_values(examples):
    """Macros for the paragraph that reads Table 2, from the rows the table prints."""
    ms = lambda r, k: r[k] * 1e3  # noqa: E731
    row = {('A100' if 'A100' in p else 'v5e', r['strategy']): r for p, r in examples}
    seed = [row[p, 'seed_regenerated'] for p in ('A100', 'v5e')]
    rank = {p: row[p, 'mirrored_lr1'] for p in ('A100', 'v5e')}
    # The paragraph says these in words; check them rather than trust them.
    if not all(ms(r, 'contraction_measured') < 1 for r in rank.values()):
        raise SystemExit('Section 6.1 says rank-1 reconstruction takes under 1 ms in the worked example')
    if not ms(rank['v5e'], 't_A') < ms(rank['A100'], 't_A'):
        raise SystemExit('Section 6.1 says the v5e replicated generation is the shorter one')
    saving = [ms(r, 'contraction_measured') - ms(r, 'contraction_local_measured') for r in seed]
    return {
        'ExSeedSaving': span(min(saving), max(saving), 0),
        'ExSeedComm': span(min(ms(r, 'allreduce_insitu') for r in seed),
                           max(ms(r, 'allreduce_insitu') for r in seed), 0),
        'ExRankPredGPU': rounded(-ms(rank['A100'], 'delta_predicted'), 2),
        'ExRankPredTPU': rounded(-ms(rank['v5e'], 'delta_predicted'), 2),
        'ExRankMeasGPU': rounded(-ms(rank['A100'], 'delta_measured'), 2),
        'ExRankMeasTPU': rounded(-ms(rank['v5e'], 'delta_measured'), 2),
        'ExRankGenGPU': rounded(ms(rank['A100'], 't_A'), 2),
        'ExRankGenTPU': rounded(ms(rank['v5e'], 't_A'), 2),
        'ExRankPenaltyGPU': rounded(-rank['A100']['delta_measured'] / rank['A100']['t_A'] * 100, 0),
        'ExRankPenaltyTPU': rounded(-rank['v5e']['delta_measured'] / rank['v5e']['t_A'] * 100, 0),
    }


def qwen_values(qwen):
    """Macros for Section 6.2's first paragraph, with its claims checked."""
    pairs = qwen['clean_comparisons']
    full = [p for p in pairs if 'seed' in p['variant']]
    low = [p for p in pairs if 'seed' not in p['variant']]
    def check(ok, claim):
        if not ok:
            raise SystemExit(f'Section 6.2 says {claim}')
    check(not qwen['dirty_records_by_devices'].get(8), 'every eight-device record is clean')
    check(all(p['repeat_ratio'][1] < 1 for p in full), 'splitting is faster for every full-rank comparison')
    check(max(p['ratio'] for p in full if p['D'] == 8) < min(p['ratio'] for p in full if p['D'] == 4),
          'splitting gains more at eight devices than at four')
    check(all(p['repeat_ratio'][0] > 1 for p in low), 'replication is faster in every low-rank comparison')
    closest = min(low, key=lambda p: p['ratio'])
    check((closest['variant'], closest['N']) == ('mirrored_lr16', 128),
          'the closest low-rank comparison is rank 16 at N = 128')
    # The payload paragraph: an extrapolated all-reduce time against the measured penalty.
    penalty = [p['B_minus_A_ms'] for p in low
               if p['variant'] in ('mirrored_lr1', 'mirrored_lr4') and p['D'] in (4, 8)]
    check(min(penalty) <= qwen['communication_estimate_ms'] <= max(penalty),
          'the extrapolated all-reduce time accounts for the rank-1 and rank-4 penalty')
    # The population and rank paragraph.
    at = {(p['variant'], p['N'], p['D']): p for p in low}
    one8 = {n: at['mirrored_lr1', n, 8] for n in (32, 64, 128) if ('mirrored_lr1', n, 8) in at}
    check(sorted(one8) == [32, 64, 128], 'rank 1 at eight devices covers populations 32 to 128')
    check(one8[32]['ratio'] > one8[64]['ratio'] > one8[128]['ratio'],
          "rank 1's ratio at eight devices falls as the population grows")
    check(at['mirrored_lr1', 32, 8]['ratio'] > at['mirrored_lr1', 32, 4]['ratio'],
          "rank 1's ratio at N = 32 rises from four devices to eight")
    check(at['mirrored_lr1', 32, 8]['B_minus_A_ms'] > at['mirrored_lr1', 32, 4]['B_minus_A_ms'],
          "rank 1's absolute penalty at N = 32 rises from four devices to eight")
    sixteen = [at['mirrored_lr16', n, 8]['B_minus_A_ms'] for n in (32, 64, 128)]
    check(sixteen[0] > sixteen[1] > sixteen[2], "rank 16's penalty falls as the population grows")
    check(all(at['mirrored_lr16', n, 8]['ratio'] < one8[n]['ratio'] for n in one8),
          "the predicted smaller margin for rank 16 than for rank 1 held")
    check(at['mirrored_lr4', 64, 8]['ratio'] > one8[64]['ratio'],
          "the predicted shrinking margin with rank fails at N = 64 (rank 4 above rank 1)")
    # Appendix B's record counts: every record is a timing or a memory failure.
    known = qwen['record_outcomes']['clean']
    unknown = qwen['record_outcomes']['dirty']
    check(sum(known.values()) + sum(unknown.values()) == qwen['records_total'],
          'every record is either a timing or a memory failure')
    return {
        'QwenRecords': spelled(qwen['records_total']),
        'QwenUnknownRecords': spelled(sum(unknown.values())),
        'QwenKnownRecords': spelled(sum(known.values())),
        'QwenKnownTimings': spelled(known['timings']),
        'QwenKnownOOM': spelled(known['oom']),
        'QwenRankOnePenaltyMin': rounded(min(p['B_minus_A_ms'] for p in one8.values()), 0),
        'QwenRankOnePenaltyMax': rounded(max(p['B_minus_A_ms'] for p in one8.values()), 0),
        'QwenRankOneRatioSmallN': rounded(one8[32]['ratio'], 2),
        'QwenRankOneRatioMidN': rounded(one8[64]['ratio'], 2),
        'QwenRankOneRatioLargeN': rounded(one8[128]['ratio'], 2),
        'QwenRankOneRatioFourDevices': rounded(at['mirrored_lr1', 32, 4]['ratio'], 2),
        'QwenRankSixteenPenalty': rounded(sixteen[2], 1),
        'QwenRankFourRatioMidN': rounded(at['mirrored_lr4', 64, 8]['ratio'], 2),
        'QwenParams': f"{qwen['parameters']:,}",
        'QwenUpdateGiB': rounded(qwen['update_GiB'], 2),
        'QwenCommEstimate': rounded(qwen['communication_estimate_ms'], 1),
        'QwenLowPenalty': span(min(penalty), max(penalty), 0),
        # Digits for both, since one sentence compares them.
        'QwenFullComparisons': str(len(full)),
        'QwenLowComparisons': str(len(low)),
        'QwenNearTieRatio': rounded(closest['ratio'], 3),
        'QwenNearTieBar': rounded(closest['repeat_ratio'][0], 4),
    }


def memory_values(qwen):
    """Appendix C: where Qwen runs out of memory, from the records whose code version is known."""
    def check(ok, claim):
        if not ok:
            raise SystemExit(f'Appendix C says {claim}')
    low = ('mirrored_lr1', 'mirrored_lr4', 'mirrored_lr16')
    outcomes = qwen['known_outcomes']
    eight = collections.defaultdict(dict)
    for s, n, d, h, o in outcomes:
        if d == 8:
            eight[s, h][n] = o
    populations = sorted({n for v in eight.values() for n in v})
    check(set(eight) == {(s, h) for s in low + ('mirrored_seed',) for h in 'AB'}
          and all(sorted(v) == populations for v in eight.values()),
          'every variant ran at every population on eight devices')
    fits = max(n for n in populations if all(eight[s, h][n] == 'timed' for s in low for h in 'AB'))
    oom = [n for n in populations if n > fits]
    check(len(oom) == 1 and all(eight[s, h][n] == ('timed' if n <= fits else 'oom')
                                for s in low for h in 'AB' for n in populations),
          'ranks 1, 4 and 16 run up to one population and run out of memory at the next, '
          'under both placements')
    check(all(o == 'timed' for (s, _), v in eight.items() if s == 'mirrored_seed' for o in v.values()),
          'seed, mirrored runs at every population on eight devices')
    per = [(n / d, o, s, h) for s, n, d, h, o in outcomes if s in low]
    fit_max = max(m for m, o, *_ in per if o == 'timed')
    oom_min = min(m for m, o, *_ in per if o == 'oom')
    check(fit_max < oom_min and {o for _, o, *_ in per} == {'timed', 'oom'},
          'whether a low-rank configuration fits depends only on the candidates per device')
    both = {(s, h) for s in low for h in 'AB'}
    check({(s, h) for _, o, s, h in per if o == 'oom'} == both == {(s, h) for _, o, s, h in per if o == 'timed'},
          'the per-device limit holds for all three ranks and both placements')
    memory_probe_checks()
    return {'QwenLowFitMaxN': str(fits), 'QwenLowOOMN': str(oom[0]),
            'QwenPopulationCount': spelled(len(populations)),
            'QwenFitPerDevice': spelled(int(fit_max)), 'QwenOOMPerDevice': spelled(int(oom_min))}


def memory_probe_checks():
    """Appendix C's account of e17_memory_probe.py, checked against its saved log."""
    def check(ok, claim):
        if not ok:
            raise SystemExit(f'Appendix C says {claim} (results-e17b-memory/probe.log)')
    tables, section = collections.defaultdict(list), None
    for line in (COUNTDOWN / 'results-e17b-memory/probe.log').read_text().splitlines():
        head = re.match(r'([A-D])\. ', line)
        if head:
            section = head.group(1)
        elif re.match(r'\s+\d+\s', line) and section:
            tables[section].append([float(x.rstrip('G')) for x in line.split()])
        elif section == 'C' and re.match(r'\s+\w+: [0-9.]+G', line):
            tables['C'].append(float(line.split(':')[1].strip().rstrip('G')))
    doubles = lambda col: all(1.9 <= b / a <= 2.1 for a, b in zip(col, col[1:]))  # noqa: E731
    column = lambda t, i: [row[i] for row in tables[t]]  # noqa: E731
    # A and B: columns seed, rank 1, rank 4, rank 16, one row per doubling.
    check(all(doubles(column('A', i)) for i in (2, 3, 4)),
          'low-rank temporary memory grows in proportion to the candidates per device')
    check(all(doubles(column('B', i)) for i in (2, 3, 4)),
          'low-rank temporary memory grows in proportion to the prompts')
    check(all(row[4] <= row[2] for t in 'AB' for row in tables[t]) and tables['C'][3] <= tables['C'][1],
          'temporary memory is no larger at rank 16 than at rank 1')
    seed = column('A', 1)
    check(all(b <= a for a, b in zip(seed, seed[1:])), "seed's temporary memory does not grow with the candidates")
    check(doubles(column('D', 2)), "evaluating seed's candidates together reproduces the growth")


def model_errors():
    """Appendix A.2 on Eq. 2's terms and errors, per platform, from timemodel.py's rows."""
    out = {}
    for platform, spec in timemodel.PLATFORMS.items():
        rows, _ = timemodel.rows(platform, spec)
        low = lambda r: r['strategy'] in timemodel.LOW_RANK  # noqa: E731
        split = lambda r: r['contraction_measured'] / (8 * r['contraction_local_measured'])  # noqa: E731
        err = [abs(r['delta_predicted'] - r['delta_measured']) for r in rows]
        misses = [r for r in rows if (r['delta_predicted'] > 0) != (r['delta_measured'] > 0)]
        out['GPU' if 'A100' in platform else 'TPU'] = {
            'split_full': [split(r) for r in rows if not low(r)],
            'split_low': [split(r) for r in rows if low(r)],
            'add_over_isolated': [r['allreduce_insitu'] / r['allreduce_seconds']
                                  for r in rows if low(r) and r['d_model'] == 2048],
            'median_ms': statistics.median(err) * 1e3,
            'max_ms': max(err) * 1e3,
            'configurations': len(rows),
            'correct': len(rows) - len(misses),
            'miss_gaps_ms': [abs(r['delta_measured']) * 1e3 for r in misses],
        }
    return out


def model_values(tm):
    """Appendix A.2: Eq. 2's measured terms and its errors on one host."""
    gaps = [g for p in tm.values() for g in p['miss_gaps_ms']]
    if not gaps or max(gaps) >= min(p['median_ms'] for p in tm.values()):
        raise SystemExit("Appendix A.2 presents Eq. 2's misses as near-ties: each measured difference "
                         "must be below the median error")
    if not all(max(p['split_low']) < 1 for p in tm.values()):
        raise SystemExit('Appendix A.4 says splitting takes more than 1/D of the replicated '
                         'reconstruction for low-rank noise (q > 1/D)')
    rng = lambda xs, places: f"{rounded(min(xs), places)}--{rounded(max(xs), places)}"  # noqa: E731
    return {
        **{f'TmSplitFull{k}': rng(v['split_full'], 2) for k, v in tm.items()},
        **{f'TmSplitLow{k}': rng(v['split_low'], 2) for k, v in tm.items()},
        **{f'TmAddOverIsolated{k}': rng(v['add_over_isolated'], 2) for k, v in tm.items()},
        **{f'TmMedianErr{k}': rounded(v['median_ms'], 2) for k, v in tm.items()},
        **{f'TmMaxErr{k}': rounded(v['max_ms'], 2) for k, v in tm.items()},
        **{f'TmConfigs{k}': spelled(v['configurations']) for k, v in tm.items()},
        **{f'TmCorrect{k}': spelled(v['correct']) for k, v in tm.items()},
        'TmMisses': spelled(len(gaps)),
        'TmMissGaps': rng(gaps, 2),
    }


def scaling_values():
    """Appendix D: parallel efficiency, as plot.py draws it, throughput per device relative to
    the series' smallest measured device count."""
    import plot
    def check(ok, claim):
        if not ok:
            raise SystemExit(f'Appendix D says {claim}')
    eff = {}
    for platform, spec in timemodel.PLATFORMS.items():
        k = 'GPU' if 'A100' in platform else 'TPU'
        series = collections.defaultdict(dict)
        for r in plot.load([HERE / s for s in spec['sweeps']]):
            c = r['config']
            per = c['population'] // c['devices'] if c['mode'] == 'weak' else c['population']
            series[c['mode'], c['d_model'], per, c['strategy'], c['how']][c['devices']] = \
                c['population'] / r['seconds_median']
        for key, by in series.items():
            d0 = min(by)
            eff[k, *key] = {d: (t / d) / (by[d0] / d0) for d, t in by.items()}
    weak8 = lambda k, s, h: eff[k, 'weak', 2048, 32, s, h][8]  # noqa: E731
    rank1 = {k: [weak8(k, s, h) for s in timemodel.LOW_RANK for h in 'AB'] for k in ('GPU', 'TPU')}
    repl = {(k, s): weak8(k, s, 'A') for k in ('GPU', 'TPU') for s in ('iid_gaussian', 'seed_regenerated')}
    split = [weak8(k, s, 'B') for k in ('GPU', 'TPU') for s in ('iid_gaussian', 'seed_regenerated')]
    check(max(repl.values()) < min(split), 'dense and seed keep more efficiency under splitting than under replication')
    above = [(e, key, d) for key, by in eff.items() for d, e in by.items()
             if key[0] == 'TPU' and key[1] == 'strong' and e > 1]
    check({key[2] for _, key, _ in above} == {512, 2048}, 'v5e strong-scaling efficiencies exceed 1 at both widths')
    top, key, d = max(a for a in above if a[1][2] == 512)
    check((key[4], key[3], d) == ('mirrored_lr1', 1024, 2), 'the largest at width 512 is rank 1 at N = 1024 on two devices')
    wide = max(e for e, key, _ in above if key[2] == 2048)
    return {
        **{f'ScalingRankOne{k}': span(min(v), max(v), 2) for k, v in rank1.items()},
        **{f'ScalingRepl{"Dense" if s == "iid_gaussian" else "Seed"}{k}': rounded(v, 2) for (k, s), v in repl.items()},
        'ScalingSplitFull': span(min(split), max(split), 2),
        'ScalingStrongMax': rounded(top, 2),
        'ScalingStrongMaxWide': rounded(wide, 2),
    }


def overlap_values():
    """Appendix A.2 on overlap: replication's lead if splitting overlapped perfectly."""
    import overlap_bound
    wide = [r for r in overlap_bound.rows() if r[2] == 2048]
    with_overlap = [r[8] for r in wide]
    if {r[0] for r in wide} != {'NVIDIA A100-SXM4-80GB', 'TPU v5 lite'} or min(with_overlap) <= 0:
        raise SystemExit('Appendix A.2 says replication stays faster with perfect overlap in every '
                         'low-rank configuration at width 2048, on both platforms')
    return {'OverlapLead': span(min(with_overlap), max(with_overlap), 2)}


def simpler_versions(gpu, tpu, tm):
    """Appendix A.2 on Eq. 2 against its two simpler versions, with the claims it makes in words."""
    platforms = {'GPU': gpu, 'TPU': tpu}
    for k, p in platforms.items():
        if (p['configurations'], p['audit']['measured_components']['correct']) != \
                (tm[k]['configurations'], tm[k]['correct']):
            raise SystemExit(f'the {k} audit and timemodel.py disagree on the configurations')
    strategies = {k: {row[0] for row in p['resolution']} for k, p in platforms.items()}
    unresolved = {k: sum(not row[3] for row in p['resolution']) for k, p in platforms.items()}
    if strategies['GPU'] - strategies['TPU'] != {'mirrored_seed'} or strategies['TPU'] - strategies['GPU']:
        raise SystemExit('Appendix A.2 says the mirrored full-rank variant is the only one run on one platform')
    if not unresolved['GPU'] < unresolved['TPU']:
        raise SystemExit('Appendix A.2 says fewer A100 comparisons are unresolved')
    misses = sorted(map(tuple, gpu['audit']['ideal_isolated']['resolved_misses']))
    if misses != [('lowrank_r1', 2048, 256), ('mirrored_lr1', 512, 256)]:
        raise SystemExit(f'Appendix A.2 names the resolved A100 misses of the simpler version; they are {misses}')
    for k, p in platforms.items():
        measured, in_context = p['audit']['measured_components'], p['audit']['ideal_in_context']
        if in_context['resolved_misses']:
            raise SystemExit(f'Appendix A.2 says the version with T_add has no resolved {k} misses')
        if measured['correct'] > in_context['correct']:
            raise SystemExit(f'Appendix A.2 says measuring C_B does not help on the {k}')
    naive, measured = gpu['large_lowrank_naive_ms'], gpu['large_lowrank_measured_ms']
    signed = lambda x: ('+' if x > 0 else '') + rounded(x, 2)  # noqa: E731
    return {
        **{f'ModelResolved{k}': spelled(p['resolved']) for k, p in platforms.items()},
        **{f'IdealCorrect{k}': spelled(p['audit']['ideal_isolated']['correct']) for k, p in platforms.items()},
        **{f'InContextCorrect{k}': spelled(p['audit']['ideal_in_context']['correct']) for k, p in platforms.items()},
        'IdealLowRankLoGPU': signed(naive[0]), 'IdealLowRankHiGPU': signed(naive[1]),
        'MeasuredLowRankLoGPU': signed(measured[0]), 'MeasuredLowRankHiGPU': signed(measured[1]),
    }


def update_agreement():
    """The one- against eight-device update error, from the committed C6d log (the run needed
    eight A100s; the log is the record). Section 4 and Appendix A both state it."""
    log = (COUNTDOWN / 'results/c6d-a100x8-2026-08-18/decompose-verdicts.txt').read_text()
    m = re.search(r'D=1 vs D=8: norm rel err ([0-9.]+e[-+][0-9]+)', log)
    if not m:
        raise SystemExit('the C6d log no longer states the D=1 against D=8 update error')
    value = float(m.group(1))
    exponent = math.floor(math.log10(value))
    return f'{rounded(value / 10**exponent, 1)}\\times10^{{{exponent}}}'


def results_latex(block, examples=(), qwen=None, host=None, tm=None):
    """Macros for the results figures the abstract and Section 6 state."""
    gpu, tpu = block['8x A100-SXM4-80GB (NVLink)'], block['TPU v5e-8 (ICI)']
    both = lambda key: (min(gpu[key][0], tpu[key][0]), max(gpu[key][1], tpu[key][1]))  # noqa: E731
    matched = block['matched_precision']
    values = {
        # Section 4 and Appendix A: the update computed on one and on eight A100s.
        'UpdateRelErr': update_agreement(),
        # Section 6.4: the platform difference at matched (float32, highest) precision.
        'MatchedRankOneGPU': rounded(matched['rank1_over_dense'][0], 2),
        'MatchedRankOneTPU': rounded(matched['rank1_over_dense'][1], 2),
        'MatchedPairs': spelled(matched['count']),
        # Section 6.1 and the abstract: splitting against replication at eight devices.
        'FullRankSpeedup': span(*both('drawn_full_rank_speedup'), 1),
        'DenseSpeedupGPU': span(*gpu['dense_speedup'], 1),
        'DenseSpeedupTPU': span(*tpu['dense_speedup'], 1),
        'LowRankSlowdown': span(*both('large_lowrank_split_slower_pct'), 0),
        'LowRankSlowdownGPU': span(*gpu['large_lowrank_split_slower_pct'], 0),
        'LowRankSlowdownTPU': span(*tpu['large_lowrank_split_slower_pct'], 0),
    }
    # Section 6.1 says which comparisons the repeats resolve; check the claim, count the rest.
    comparisons = [row for p in (gpu, tpu) for row in p['resolution']]
    low = lambda s: s in timemodel.LOW_RANK  # noqa: E731
    must = [row for row in comparisons if not low(row[0]) or row[1] == 2048]
    if not all(row[3] for row in must):
        raise SystemExit('Section 6.1 says every full-rank and every width-2048 low-rank '
                         f'comparison is resolved; these are not: {[r for r in must if not r[3]]}')
    narrow = [row for row in comparisons if low(row[0]) and row[1] == 512]
    # Section 6.1 on how often Eq. 2 gives the faster placement.
    audit = lambda name, key: sum(p['audit'][name][key] for p in (gpu, tpu))  # noqa: E731
    configs = sum(p['configurations'] for p in (gpu, tpu))
    resolved_n = sum(p['resolved'] for p in (gpu, tpu))
    correct = audit('measured_components', 'correct')
    if audit('measured_components', 'resolved_correct') != resolved_n:
        raise SystemExit('Section 6.1 says every miss of Eq. 2 is an unresolved configuration')
    if tpu['audit']['ideal_isolated']['resolved_misses']:
        raise SystemExit('Section 6.1 says the simpler model misses only A100 configurations')
    values['ModelConfigs'] = spelled(configs)
    values['ModelResolved'] = spelled(resolved_n)
    values['ModelCorrect'] = spelled(correct)
    values['ModelMisses'] = spelled(configs - correct)
    values['IdealMissesGPU'] = spelled(len(gpu['audit']['ideal_isolated']['resolved_misses']))
    if tm:
        values.update(model_values(tm))
        values.update(simpler_versions(gpu, tpu, tm))
        values.update(overlap_values())
        values.update(scaling_values())
    values['NarrowLowRank'] = spelled(len(narrow))
    values['NarrowLowRankUnresolved'] = spelled(sum(not row[3] for row in narrow))
    if examples:
        values.update(example_values(examples))
    if qwen:
        values.update(qwen_values(qwen))
        values.update(memory_values(qwen))
    if host:
        values['HostConfigsCap'] = spelled(host['two_host_cells']).capitalize()
        values['HostRate'] = span(*host['cross_host_rate_GiB_s'], 1)
        values['HostBreakEvenSmallN'] = rounded(host[128]['break_even_GiB_s'], 1)
        values['HostBreakEvenLargeN'] = rounded(host[256]['break_even_GiB_s'], 1)
        # Appendix A.3: the inputs of the break-even rate, and its range over the repeats.
        values['HostSavingSmallN'] = rounded(host[128]['saving_ms'], 1)
        values['HostSavingLargeN'] = rounded(host[256]['saving_ms'], 1)
        values['HostIntra'] = rounded(host[128]['intra_bandwidth_term_ms'], 2)
        values['HostBoundsSmallN'] = span(*host[128]['repeat_bounds_GiB_s'], 1)
        values['HostBoundsLargeN'] = span(*host[256]['repeat_bounds_GiB_s'], 1)
        values['HostIsolatedIntra'] = rounded(host['isolated_intra_ms'], 2)
        # "at most": an upper bound, so rounded up.
        values['HostIsolatedShift'] = str(Decimal(repr(host['isolated_shift_GiB_s'])).quantize(
            Decimal('0.1'), rounding=ROUND_CEILING))
        if host['expected_matches'] != host['expected_cells']:
            raise SystemExit('Section 6.3 says Eq. 2 gives the faster placement in every two-host configuration')
        if not all(v > 0 for v in host['expected_over_sixteen_ms']):
            raise SystemExit('Section 6.3 says Eq. 2 overestimates the penalty for splitting on sixteen devices')
        values['HostExpectedMatches'] = spelled(host['expected_matches'])
        values['HostExpectedErrorEight'] = rounded(host['expected_error_eight_ms'], 0)
        values['HostExpectedOverSixteen'] = rounded(max(host['expected_over_sixteen_ms']), 0)
        values['HostBoundaryDrop'] = span(min(host['boundary_drop_ms']), max(host['boundary_drop_ms']), 0)
        values['HostSeedSaving'] = span(min(host['seed_one_host_saving_ms']), max(host['seed_one_host_saving_ms']), 0)
    lines = ['% generated by experiments/phase2/paper_evidence.py --latex; do not edit']
    lines += [f'\\newcommand{{\\{name}}}{{{value}}}' for name, value in values.items()]
    return '\n'.join(lines) + '\n'


def example_latex(examples):
    out = [r'% generated by experiments/phase2/paper_evidence.py; do not edit',
           r'\begin{table*}[t]', r'\centering', r'\small',
           r'\caption{Worked example of Eq.~\ref{eq:crossover} at width 2048, population 256 and eight devices; '
           r'all times in milliseconds. $C_A$ and $C_B$ are the separately measured reconstruction times. '
           r'$T_{\mathrm{add}}$ is the time the all-reduce adds to the split reconstruction, measured in context; '
           r'it stands in for $T_{\mathrm{ar}}(4P)$. The expected difference is $C_A-C_B+T_{\mathrm{ag}}-T_{\mathrm{add}}$, '
           r"where $T_{\mathrm{ag}}$ is replication's second coefficient gather. Positive differences favor "
           r'splitting. $t_A$ is the replicated generation time, the denominator of the ratio in Figure~\ref{fig:f2}.}',
           r'\label{tab:worked}', r'\begin{tabular}{llrrrrrr}',r'\toprule',
           r'platform & variant & $C_A$ & $C_B$ & $T_{\mathrm{add}}$ & expected $t_A-t_B$ & measured $t_A-t_B$ & $t_A$ \\', r'\midrule']
    for platform, r in examples:
        label = 'A100' if 'A100' in platform else 'v5e'
        name = 'seed' if r['strategy']=='seed_regenerated' else 'rank 1'
        fields = [r[k]*1e3 for k in ('contraction_measured','contraction_local_measured','allreduce_insitu','delta_predicted','delta_measured','t_A')]
        out.append(f'{label} & {name} & ' + ' & '.join(rounded(v, 2) for v in fields) + r' \\')
    return '\n'.join(out+[r'\bottomrule',r'\end{tabular}',r'\end{table*}',''])


def qwen_checks():
    # eval_shape counts every leaf, including the tied embedding once, without weights.
    import jax
    import jax.numpy as jnp
    from shardes.problems import qwen2
    shape = jax.eval_shape(lambda: qwen2.init(jax.random.key(0), qwen2.Config.qwen25_05b(), dtype=jnp.bfloat16))
    p = sum(math.prod(x.shape) for x in jax.tree.leaves(shape))
    payload = 4*p
    ladder = read(HERE/'results-ladder/ladder-tpu-v5-lite-D8.json')
    t96 = ladder['allreduce'][str(96*2**20)]['step_seconds']
    records = [read(path) for path in sorted((COUNTDOWN/'results-e17b').glob('s=*.json'))]
    counts = collections.Counter((r['env']['commit'], r['env']['dirty_worktree']) for r in records)
    outcomes = collections.Counter(
        ('dirty' if r['env']['dirty_worktree'] else 'clean',
         'timings' if 'seconds_median' in r else r['status']) for r in records)
    clean = {(r['config']['strategy'],r['config']['population'],r['config']['devices'],r['config']['how']):r
             for r in records if r['env']['dirty_worktree'] is False}
    pairs = []
    for (s,n,d,h),a in clean.items():
        if h!='A' or (s,n,d,'B') not in clean:
            continue
        b=clean[s,n,d,'B']
        if 'seconds_median' not in a or 'seconds_median' not in b:
            continue
        pairs.append({'variant':s,'N':n,'D':d,'ratio':b['seconds_median']/a['seconds_median'],
                      'B_minus_A_ms':(b['seconds_median']-a['seconds_median'])*1e3,
                      'repeat_ratio':[min(b['seconds_all'])/max(a['seconds_all']),max(b['seconds_all'])/min(a['seconds_all'])]})
    return {'parameters':p,'update_GiB':payload/2**30,'communication_estimate_ms':payload/(96*2**20)*t96*1e3,
            'alpha_beta_estimate_ms':(ladder['alpha_seconds']+payload/ladder['beta_bytes_per_second'])*1e3,
            'provenance':[{'commit':sha,'dirty':dirty,'records':n} for (sha,dirty),n in sorted(counts.items())],
            'record_outcomes': {state: {outcome: outcomes[state, outcome]
                                        for outcome in ('timings', 'oom')}
                                for state in ('clean', 'dirty')},
            'clean_comparisons':pairs,
            'records_total':len(records),
            'outcomes':[[r['config'][k] for k in ('strategy','population','devices','how')]
                        + ['timed' if 'seconds_median' in r else r.get('status')] for r in records],
            'known_outcomes':[[r['config'][k] for k in ('strategy','population','devices','how')]
                              + ['timed' if 'seconds_median' in r else r.get('status')]
                              for r in records if r['env']['dirty_worktree'] is False],
            'dirty_records_by_devices':dict(collections.Counter(
                r['config']['devices'] for r in records if r['env']['dirty_worktree']))}


def host_checks():
    import tb7_e18  # multihost/, on the path above; Table 7's cells and device map
    folder=HERE/'multihost/results-e18'
    pre=read(folder/'preflight-1x8.json')
    _,beta=costmodel.ladder_alpha_beta(pre,8)
    payload=96*2**20
    intra=payload/beta  # bandwidth term only; ignore latency in this approximation
    out={}
    for n in (128,256):
        a,b=[read(folder/f'arm=seed_regenerated__how={h}__d=2048__N={n}__topo=1x8.json') for h in 'AB']
        saving=a['seconds_median']-b['seconds_median']
        bounds=[min(a['seconds_all'])-max(b['seconds_all']),max(a['seconds_all'])-min(b['seconds_all'])]
        out[n]={'saving_ms':saving*1e3,'intra_bandwidth_term_ms':intra*1e3,
                'break_even_GiB_s':payload/(saving+intra)/2**30,
                'repeat_bounds_GiB_s':sorted(payload/(v+intra)/2**30 for v in bounds)}
    # The threshold paragraph: a break-even rate only means something below the one-host rate.
    if not all(v['break_even_GiB_s'] < beta / 2**30 for v in out.values()):
        raise SystemExit('Section 6.3 gives break-even rates that must lie below the one-host all-reduce rate')
    # Appendix A.3 reads S from Table 9's 1x8 column, and compares T_intra with Table 8's time.
    one_host = {n: -v[0] * 1e3 for s, d, n, v in tb7_e18.table_rows()[2] if (s, d) == ('seed', 2048)}
    if any(abs(one_host[n] - out[n]['saving_ms']) > 1e-9 for n in (128, 256)):
        raise SystemExit("Appendix A.3 says S is Table 9's 1x8 column")
    import tb8
    ladder = next(rec for _, md, rec, _ in tb8.load() if 'A100' in md)
    isolated = ladder['allreduce'][str(payload)]['step_seconds']
    out['isolated_intra_ms'] = isolated * 1e3
    out['isolated_shift_GiB_s'] = max(abs(payload / (v['saving_ms'] / 1e3 + isolated) - payload / (v['saving_ms'] / 1e3 + intra))
                                      for v in (out[128], out[256])) / 2**30
    # Section 6.3's opening: how many configurations ran on two hosts, and at what rate.
    two_host = [c for c in tb7_e18.CELLS
                if all(tb7_e18.measured(s, d, n, topo) is not None
                       for s, d, n in [c[1:]] for topo in ('2x4', '2x8'))]
    rates = [costmodel.ladder_alpha_beta(read(folder/f'preflight-{topo}.json'), tb7_e18.DEVICES[topo])[1] / 2**30
             for topo in ('2x4', '2x8')]
    out['two_host_cells'] = len(two_host)
    # The last paragraph of Section 6.3: Eq. 2 with the rates measured on these hosts.
    _, _, rows = tb7_e18.table_rows()
    sign = lambda v: v > 0  # noqa: E731
    out['expected_matches'] = sum(sign(v[1]) == sign(v[2]) and sign(v[3]) == sign(v[4]) for *_, v in rows)
    out['expected_cells'] = len(rows)
    out['expected_error_eight_ms'] = max(abs(v[1] - v[2]) for *_, v in rows) * 1e3
    out['expected_over_sixteen_ms'] = [(v[4] - v[3]) * 1e3 for *_, v in rows]
    # The next paragraph, in t_B - t_A as the records hold it (Table 7 prints t_A - t_B).
    m = {(c[1], c[2], c[3], topo): tb7_e18.measured(c[1], c[2], c[3], topo)
         for c in two_host for topo in ('1x8', '2x4', '2x8')}
    wide = [c for c in two_host if c[2] == 2048]
    out['boundary_drop_ms'] = [(m[s, d, n, '2x4'] - m[s, d, n, '1x8']) * 1e3 for _, s, d, n in wide]
    out['seed_one_host_saving_ms'] = [-m[s, d, n, '1x8'] * 1e3 for _, s, d, n in wide if s == 'seed_regenerated']
    def claim(ok, text):
        if not ok:
            raise SystemExit(f'Section 6.3 says {text}')
    claim(all(m[s, d, n, topo] > 0 for _, s, d, n in wide if s == 'seed_regenerated' for topo in ('2x4', '2x8')),
          'replication is faster for seed at width 2048 on two hosts')
    claim(all(m[s, d, n, topo] < 0 for _, s, d, n in two_host if s == 'seed_regenerated' and d == 512
              for topo in ('1x8', '2x4', '2x8')), 'splitting stays faster for seed at width 512')
    claim(all(m[s, d, n, '1x8'] >= 0 and m[s, d, n, '2x4'] > m[s, d, n, '1x8'] and m[s, d, n, '2x8'] > m[s, d, n, '1x8']
              for _, s, d, n in two_host if s == 'mirrored_lr1' and d == 2048),
          "replication was already faster for rank 1 at width 2048, and its advantage grows on two hosts")
    for _, s, d, n in two_host:
        if s != 'mirrored_lr1' or d != 512:
            continue
        for topo in ('1x8', '2x4', '2x8'):
            a, b = [read(folder/f'arm={s}__how={h}__d={d}__N={n}__topo={topo}.json')['seconds_all']
                    for h in 'AB']
            if topo == '1x8':
                claim(not resolved(a, b), 'rank 1 at width 512 is a near-tie on one host')
            else:
                claim(max(a) < min(b), 'rank 1 at width 512 clearly favors replication on two hosts')
    out['cross_host_rate_GiB_s'] = [min(rates), max(rates)]
    return out


def training_checks():
    folder=COUNTDOWN/'results/e13-a100-2026-08-22-clean'
    out={}
    for stem in ('es-mirrored-seed','es-mirrored-lr1','es-mirrored-lr4','es-mirrored-lr16','es-lr1-frozen-embed'):
        records=[[json.loads(ln) for ln in (folder/f'{stem}-s{s}-eval.jsonl').read_text().splitlines()] for s in (0,1,2)]
        out[stem]={}
        for gen in (50,500):
            rows=[next(r for r in rs if r['generation']==gen) for rs in records]
            distance=[r['param_l2_from_init'] for r in rows]
            solve=[100*r['eval_solved'] for r in rows]
            out[stem][gen]={'distance_range':[min(distance),max(distance)],'solve_mean':statistics.mean(solve),'solve_range':[min(solve),max(solve)]}
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--latex',action='store_true');args=ap.parse_args()
    block,examples=block_checks()
    barrier=HERE/'results-barrier-tpu-v5e8'
    ranking=[read(barrier/f'N=262144__D={d}__s=centered_ranks.json')['seconds_median'] for d in (1,8)]
    report={'block':block,'qwen':qwen_checks(),'host':host_checks(),'training':training_checks(),
            'ranking_D8_minus_D1_us':(ranking[1]-ranking[0])*1e6}
    print(json.dumps(report,indent=2))
    if args.latex:
        (HERE.parent.parent/'paper/generated/worked-example.tex').write_text(example_latex(examples))
        (HERE.parent.parent/'paper/generated/results.tex').write_text(results_latex(block, examples, report['qwen'], report['host'], model_errors()))


if __name__=='__main__':
    main()
