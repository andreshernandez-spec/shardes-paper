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
import json
import math
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
    # The memory paragraph: at eight devices and N = 240, low rank runs out of memory under
    # both placements and full rank fits.
    cell = {tuple(o[:4]): o[4] for o in qwen['outcomes']}
    check(all(cell.get((s, 240, 8, h)) == 'oom' for s in ('mirrored_lr1', 'mirrored_lr4', 'mirrored_lr16')
              for h in 'AB'), 'every low-rank variant runs out of memory at eight devices and N = 240')
    check(all(cell.get(('mirrored_seed', 240, 8, h)) == 'timed' for h in 'AB'),
          'full rank fits at eight devices and N = 240')
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


def results_latex(block, examples=(), qwen=None, host=None):
    """Macros for the results figures the abstract and Section 6 state."""
    gpu, tpu = block['8x A100-SXM4-80GB (NVLink)'], block['TPU v5e-8 (ICI)']
    both = lambda key: (min(gpu[key][0], tpu[key][0]), max(gpu[key][1], tpu[key][1]))  # noqa: E731
    matched = block['matched_precision']
    values = {
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
    values['NarrowLowRank'] = spelled(len(narrow))
    values['NarrowLowRankUnresolved'] = spelled(sum(not row[3] for row in narrow))
    if examples:
        values.update(example_values(examples))
    if qwen:
        values.update(qwen_values(qwen))
    if host:
        values['HostConfigsCap'] = spelled(host['two_host_cells']).capitalize()
        values['HostRate'] = span(*host['cross_host_rate_GiB_s'], 1)
        values['HostBreakEvenSmallN'] = rounded(host[128]['break_even_GiB_s'], 1)
        values['HostBreakEvenLargeN'] = rounded(host[256]['break_even_GiB_s'], 1)
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
        (HERE.parent.parent/'paper/generated/results.tex').write_text(results_latex(block, examples, report['qwen'], report['host']))


if __name__=='__main__':
    main()
