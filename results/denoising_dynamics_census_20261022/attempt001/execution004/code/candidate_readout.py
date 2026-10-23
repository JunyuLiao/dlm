"""Reproduce the fixed-candidate analytical oracle readout; no native execution."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]

def analyze(rows):
    unique = {}
    for row in rows:
        key = tuple(row[k] for k in ('request', 'phase', 'head', 'kept_prefix_blocks', 'support_sha256'))
        if key in unique:
            for field in ('response_gain', 'output_bias_relative_max', 'reuse'):
                if unique[key][field] != row[field]:
                    raise ValueError('identical support has inconsistent diagnostic values')
        else:
            unique[key] = row
    groups = collections.defaultdict(list)
    for row in unique.values():
        groups[tuple(row[k] for k in ('request', 'phase', 'head', 'kept_prefix_blocks'))].append(row)
    readout = []
    def choice(row):
        return dict(candidate=row['candidate'], support_sha256=row['support_sha256'],
            response_gain=row['response_gain']['total'], current_error=row['output_bias_relative_max'],
            future_bias_motion=row['reuse'][0]['bias_motion_relative_max'],
            future_output_bias=row['reuse'][0]['output_bias_relative_max'])
    for key, candidates in groups.items():
        mass = next(row for row in candidates if row['candidate'] == 'mass')
        feasible = [row for row in candidates if row['response_guard_feasible']]
        response = min(feasible, key=lambda r: (r['response_gain']['total'], r['candidate'])) if feasible else mass
        error = min(feasible, key=lambda r: (r['output_bias_relative_max'], r['candidate'])) if feasible else mass
        sham = min(feasible, key=lambda r: r['support_sha256']) if feasible else mass
        result = dict(request=key[0], phase=key[1], head=key[2], K=key[3],
            split=mass['split'], dataset=mass['dataset'], candidate_count=len(candidates),
            feasible_candidates=len(feasible), fallback=not feasible,
            response=choice(response), error=choice(error), mass=choice(mass), sham=choice(sham))
        for name, baseline in [('error', error), ('mass', mass), ('sham', sham)]:
            result['response_minus_' + name + '_future_bias_motion'] = (
                response['reuse'][0]['bias_motion_relative_max'] - baseline['reuse'][0]['bias_motion_relative_max'])
        readout.append(result)
    summary = {}
    for split in ('development', 'heldout'):
        for dataset in sorted({row['dataset'] for row in readout}):
            selected = [row for row in readout if row['split'] == split and row['dataset'] == dataset]
            if not selected:
                continue
            result = dict(groups=len(selected), fallback_groups=sum(r['fallback'] for r in selected),
                same_support_as_error=sum(r['response']['support_sha256'] == r['error']['support_sha256'] for r in selected))
            for baseline in ('error', 'mass', 'sham'):
                values = [r['response_minus_' + baseline + '_future_bias_motion'] for r in selected]
                result['versus_' + baseline] = dict(lower=sum(v < -1e-12 for v in values),
                    equal=sum(abs(v) <= 1e-12 for v in values), higher=sum(v > 1e-12 for v in values),
                    mean_delta=statistics.mean(values))
            summary[split + '/' + dataset] = result
    return dict(raw_records=len(rows), unique_physical_records=len(unique), physical_groups=len(groups),
        summary=summary, groups=readout)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'operator_diagnostic002/candidate_readout_reproduced.json')
    args = parser.parse_args()
    source = ROOT/'operator_diagnostic002/records.json'
    result = analyze(json.loads(source.read_text()))
    result.update(schema='response_candidate_readout_v1',
        scope='Posthoc analytical readout of frozen candidate bank, guards and prior-state directions; no thresholds fitted',
        input_records_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        controls=dict(response='minimum total mask-induced Q/K/V response among feasible candidates',
            error='minimum current attention output error among same feasible candidates',
            mass='frozen mass candidate',
            sham='lexicographic support hash among same feasible candidates; deterministic outcome-independent sham',
            fallback='mass when no feasible candidate'), native_promotion=False,
        limitations=['Inherited V31 states, one layer and sampled rows',
            'No downstream/native continuation supervision', 'Only one heldout request per family',
            'Feasible-set and candidate bank oracle only, not global support optimum'])
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    previous = json.loads((ROOT/'operator_diagnostic002/candidate_readout.json').read_text())
    if result != previous:
        raise RuntimeError('candidate readout does not reproduce the original analytical output')
    print(json.dumps(dict(reproduced=True, summary=result['summary']), indent=2))

if __name__ == '__main__':
    main()
