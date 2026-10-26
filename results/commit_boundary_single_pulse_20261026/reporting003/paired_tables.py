"""Independent paired readout; preserves both seeds for primary timing CIs.

The seed-pair convention was declared in ANALYSIS_CONVENTIONS before C outcomes.
No model calls, selection changes or favorable retiming are performed here.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math
import numpy as np

R = Path(__file__).resolve().parents[1]
C = R / 'panel/attempt004'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    inputs = {}

    def read(p):
        inputs[str(p.relative_to(R))] = sha(p)
        return json.loads(p.read_text())

    assert read(C / 'execution_complete.json')['passed']
    assert read(C / 'fairness001/summary.json')['passed']
    protocol = read(C / 'protocol.json')
    manifest = read(C / 'manifest_identity.json')
    rows = {}
    for arm in protocol['arms']:
        assert read(C / f'runs/panel/{arm}/final_complete.json')['passed']
        for i in range(24):
            a = read(C / f'runs/panel/{arm}/audit{i:03d}.json')
            c = read(C / f'runs/panel/{arm}/clean{i:03d}.json')
            pair = read(C / f'runs/panel/{arm}/pair{i:03d}.json')
            assert pair['passed']
            assert pair['audit_sha256'] == sha(C / f'runs/panel/{arm}/audit{i:03d}.json')
            assert pair['clean_sha256'] == sha(C / f'runs/panel/{arm}/clean{i:03d}.json')
            key = (a['id'], a['seed'], arm)
            assert key not in rows and key[:2] == (manifest['identities'][i//2]['id'], [42, 43][i%2])
            rows[key] = (a, c)

    rng = np.random.default_rng(20261026)

    def estimate(v, geometric=True):
        if not v:
            return None
        v = np.asarray(v, dtype=float)
        boot = v[rng.integers(0, len(v), size=(20000, len(v)))].mean(1)
        f = (lambda x: float(np.exp(x))) if geometric else float
        return dict(point=f(v.mean()), CI95=[f(np.quantile(boot, .025)), f(np.quantile(boot, .975))], question_clusters=len(v))

    timing = []
    for task in ['aime26', 'humaneval', 'longbench_v2']:
        for stratum in ['all', 'targeted', 'coverage']:
            ids = [m['id'] for m in manifest['identities'] if m['benchmark'] == task and (stratum == 'all' or m['stratum'] == stratum)]
            for arm in protocol['arms'][3:]:
                for ref in ['corrected_dense_full', 'matched_allkept_layer29', 'current_error_single_pulse', 'support_sham_single_pulse']:
                    if arm == ref:
                        continue
                    valid, missing = [], []
                    for identity in ids:
                        checks = [all(rows[(identity, seed, ar)][1]['timing_qualified'] for ar in [arm, ref]) for seed in [42, 43]]
                        (valid if all(checks) else missing).append(identity)
                    endpoints = {}
                    for name in ['W', 'S', 'S_per_N']:
                        values = [np.mean([math.log(rows[(identity, seed, arm)][1][name] / rows[(identity, seed, ref)][1][name]) for seed in [42, 43]]) for identity in valid]
                        endpoints[name] = estimate(values)
                    timing.append(dict(task=task, stratum=stratum, method=arm, reference=ref, valid_questions=valid, excluded_questions=missing, seed_cells=2*len(valid), estimates=endpoints))

    controls = []
    for task in ['aime26', 'humaneval', 'longbench_v2']:
        for stratum in ['all', 'targeted', 'coverage']:
            ids = [m['id'] for m in manifest['identities'] if m['benchmark'] == task and (stratum == 'all' or m['stratum'] == stratum)]
            for arm, ref in [('matched_allkept_layer29', 'corrected_dense_full'), ('matched_allkept_layer29', 'dense_piecewise'), ('dense_piecewise', 'corrected_dense_full')]:
                estimates = {}
                for name in ['N', 'C', 'T', 'N_per_C', 'quality_difference', 'W', 'S', 'S_per_N']:
                    values = []
                    for identity in ids:
                        timed = name in ['W', 'S', 'S_per_N']
                        if timed and not all(rows[(identity, seed, ar)][1]['timing_qualified'] for seed in [42, 43] for ar in [arm, ref]):
                            continue
                        v = []
                        for seed in [42, 43]:
                            a = rows[(identity, seed, arm)][1 if timed else 0]
                            b = rows[(identity, seed, ref)][1 if timed else 0]
                            v.append(int(a['correct'])-int(b['correct']) if name=='quality_difference' else math.log(a[name]/b[name]))
                        values.append(np.mean(v))
                    estimates[name] = estimate(values, geometric=name!='quality_difference')
                transitions = [dict(id=identity, seed=seed, reference_correct=rows[(identity,seed,ref)][0]['correct'], method_correct=rows[(identity,seed,arm)][0]['correct']) for identity in ids for seed in [42,43]]
                controls.append(dict(task=task,stratum=stratum,method=arm,reference=ref,estimates=estimates,transitions=transitions,
                    correct_to_wrong=sum(x['reference_correct'] and not x['method_correct'] for x in transitions),wrong_to_correct=sum(not x['reference_correct'] and x['method_correct'] for x in transitions)))

    banks, local = [], []
    for i, m in enumerate(manifest['identities']):
        for seed in [42, 43]:
            a = rows[(m['id'], seed, 'matched_allkept_layer29')][0]
            methods = {arm: rows[(m['id'], seed, arm)][0] for arm in protocol['arms'][3:]}
            available = [v['runtime']['candidate_bank'] for v in methods.values() if v['runtime']['candidate_bank'] is not None]
            bank = available[0] if available else None
            assert not available or all(v == bank for v in available)
            groups = defaultdict(list)
            for arm, v in methods.items():
                support = v['runtime']['selected_support']
                if support:
                    groups[support['support_sha256']].append(arm)
                if v['runtime']['pulse_calls']:
                    assert a['pre_intervention_native_state'] == v['pre_intervention_native_state']
                    local.append(dict(id=m['id'], seed=seed, benchmark=m['benchmark'], stratum=m['stratum'], arm=arm,
                        remaining=v['remaining_calls'], A_remaining=a['remaining_calls'], remaining_change=v['remaining_calls']-a['remaining_calls'],
                        N=v['N'], A_N=a['N'], C=v['C'], A_C=a['C'], T=v['T'], A_T=a['T'],
                        correct=v['correct'], A_correct=a['correct'], output_cap=v['output_capped'], A_output_cap=a['output_capped'],
                        native_canvas_caps=sum(x['cap_reached'] for x in v['native_trace'] if not x['commit']),
                        terminal_argmax_equal=v['target_terminal']['fingerprint']['native_winner']==a['target_terminal']['fingerprint']['native_winner'],
                        emitted_target_equal=next(x['emitted_sha256'] for x in v['native_trace'] if x['commit'] and x['canvas']==v['anchor']['canvas'])==next(x['emitted_sha256'] for x in a['native_trace'] if x['commit'] and x['canvas']==a['anchor']['canvas']),
                        whole_output_equal=v['output_sha256']==a['output_sha256']))
            banks.append(dict(id=m['id'], seed=seed, benchmark=m['benchmark'], stratum=m['stratum'], anchor=a['anchor'],
                feasible_four_units=bank['feasible_common'] if bank else None, direction_pair=bank['qualified'] if bank else None,
                unit_candidates=[len(u['candidates']) for u in bank['units']] if bank else [],
                unit_feasible=[u['feasible'] for u in bank['units']] if bank else [],
                unit_matched_pairs=[len(u['matched_pairs']) for u in bank['units']] if bank else [],
                support_groups=dict(groups), distinct_exposed_supports=len(groups),
                abstentions={arm:v['runtime']['abstention'] for arm,v in methods.items()},
                directional_contrast=bank['directional_contrast'] if bank else None))

    aggregate=[]
    for task in ['aime26','humaneval','longbench_v2']:
        b=[x for x in banks if x['benchmark']==task]
        aggregate.append(dict(task=task, questions=len({x['id'] for x in b}), cells=len(b), anchor_cells=sum(x['anchor'] is not None for x in b),
            feasible_four_unit_cells=sum(x['feasible_four_units'] is True for x in b), direction_pair_cells=sum(x['direction_pair'] is True for x in b),
            four_distinct_support_cells=sum(x['distinct_exposed_supports']==4 for x in b),
            distinct_support_counts=dict(Counter(x['distinct_exposed_supports'] for x in b))))
    out=C/'paired_readout001'
    out.mkdir(exist_ok=False)
    result=dict(primary_timing=timing, control_comparisons=controls, bank_cells=banks, bank_coverage=aggregate, local_effects=local,
        protocol_sha256=sha(C/'protocol.json'), manifest_sha256=sha(C/'manifest_identity.json'),
        timing_rule='Both seeds of a question must have valid timing in BOTH compared arms. No favorable retiming.',
        caution='Descriptive small diagnostic cohort. Masks, seeds and routing units are not independent questions. CE contrasts may be unmatched. No-pulse correctness is not sparse preservation.')
    for path, data in [(out/'summary.json', result), (out/'input_hashes.json', inputs), (out/'final_complete.json', dict(passed=True, model_forwards=0))]:
        with path.open('x') as f:
            json.dump(data,f,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps(dict(bank_coverage=aggregate, exposed_cells=len(local), timing_comparisons=len(timing))))


if __name__ == '__main__':
    main()
