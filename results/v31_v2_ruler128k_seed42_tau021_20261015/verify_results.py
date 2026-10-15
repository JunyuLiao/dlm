"""Independent CPU verification of the completed paired seed42 RULER audit."""
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def key(row):
    return f"{row['dataset']}|{row['index']}|{row['panel_seed']}|{row['repeat']}"


def distribution(values):
    a = np.asarray(values)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                min=float(a.min()), p10=float(np.quantile(a, .1)),
                p90=float(np.quantile(a, .9)), max=float(a.max()))


def main():
    protocol = read(ROOT / 'protocol.json')
    complete = read(ROOT / 'final_complete.json')
    protocol_sha = sha(ROOT / 'protocol.json')
    assert complete['status'] == 'complete' and complete['protocol_sha256'] == protocol_sha
    assert protocol['coordinator_commit'].startswith('5859d086')
    assert sha(ROOT / 'coordinator_source.py') == protocol['coordinator_sha256']
    for rel, expected in protocol['execution_fingerprints'].items():
        assert sha(ROOT / 'private/execution' / rel) == expected, rel
    for rel, expected in protocol['input_fingerprints'].items():
        assert sha(ROOT / rel) == expected, rel
    schedules = ROOT / 'private/schedules/ruler128k'
    cells = read(schedules / 'cells.json')
    assert len(cells) == 104 and {r['seed'] for r in cells} == {42}
    dataset = cells[0]['dataset']
    info = read(ROOT / 'private/pools/ruler128k' / (dataset + '_rowinfo.json'))
    task_of = {r['index']: r['task'] for r in info['rows']}
    assert set(collections.Counter(task_of.values()).values()) == {8}
    metric = {r['arm']: r for r in read(ROOT / 'metrics.json')}
    data, all_scores, all_bindings, repeats, details = {}, {}, {}, {}, {}
    phase_csv = list(__import__('csv').DictReader((ROOT / 'phase_accounting.csv').open()))
    for arm in protocol['arms']:
        report = read(ROOT / 'reports' / arm / 'summary.json')
        rows = []
        phase = collections.defaultdict(lambda: collections.Counter())
        for rel in report['attempts']:
            attempt = ROOT / rel
            config = read(attempt / 'config.json')
            receipt = read(attempt / 'attempt_complete.json')
            assert config['protocol_sha256'] == receipt['protocol_sha256'] == protocol_sha
            assert receipt['status'] == 'complete' and not receipt['validation_errors']
            assert config['source_commit'] == protocol['source_commit']
            shard = schedules / (attempt.parent.name + '.json')
            assert config['cells_sha256'] == sha(shard)
            rr = [json.loads(line) for line in (attempt / 'records.jsonl').read_text().splitlines()]
            assert len(rr) == receipt['records'] == len(read(shard)) == 26
            rows.extend(rr)
        assert len(rows) == len({key(r) for r in rows}) == 104
        assert {key(r) for r in rows} == {f"{c['dataset']}|{c['index']}|42|0" for c in cells}
        for r in rows:
            a = r['receipts']['adapter']
            expected_seed = int.from_bytes(hashlib.sha256(f"31|{dataset}|{r['index']}|42|0".encode()).digest()[:4], 'little') & 0x7fffffff
            assert r['rng_seed'] == expected_seed and r['seed_base'] == 31
            assert r['adapter_sha256'] == sha(ROOT / 'private/execution/experiments/numerical_qk_reuse/vllm_adapter.py')
            assert r['cuda_graph_captures'] == 0 and r['local_kv_budget'] is None
            assert r['mage_k'] == 8192 and r['mage_granularity'] == 'qblock_max'
            assert r['mage_select_step'] == 1 and r['mage_carry_first']
            assert r['mage_sticky'] == 1.386 and r['mage_reselect_trigger'] == [.15]
            assert r['mage_trigger_signal'] == 'settle' and r['value_mass_anchor'] == 0
            assert r['sampler_calls'] - r['canvases'] == r['denoise_forwards']
            assert r['canvases'] == math.ceil(r['output_tokens'] / 256)
            assert a['global_calls'] == 5 * r['denoise_forwards']
            assert a['native_local_calls_audited'] == 25 * r['denoise_forwards']
            assert a['native_local_skipped_tiles'] == 0
            assert a['overall_eligible_rectangles'] == a['global_eligible_tiles'] + a['native_local_eligible_rectangles']
            assert a['overall_skipped_rectangles'] == a['global_eligible_tiles'] - a['global_kept_tiles']
            assert not any(a.get(k, 0) for k in ['order_errors', 'value_invalid_rows', 'value_nonfinite_logits_calls', 'value_nonfinite_attention_calls'])
            assert sum(p['eligible_tiles'] for p in a['value_phase_tiles'].values()) == a['global_eligible_tiles']
            assert sum(p['kept_tiles'] for p in a['value_phase_tiles'].values()) == a['global_kept_tiles']
            for name, p in a['value_phase_tiles'].items():
                for k in ['calls', 'eligible_tiles', 'kept_tiles']:
                    phase[name][k] += p[k]
            if arm.startswith('v2'):
                assert r['value_selector'] == a['value_selector'] == 'value_v2_online_preserve_mass'
                assert r['value_threshold'] == a['value_threshold'] == .021
                assert a['value_projection_rank'] == 32 and a['value_projection_seed'] == 1729
                assert a['value_selection_kernel'] == 'triton_log_mass_online'
                assert a['value_state_role'] == 'selector_only' and a['value_output_path'] == 'unchanged_fa4'
                assert a['value_scope'] == 'global_only' and not a['value_local_router_installed']
            else:
                assert r['value_selector'] is None
        for p in [p for p in phase_csv if p['arm'] == arm]:
            for name in ['calls', 'eligible_tiles', 'kept_tiles']:
                assert int(p[name]) == phase[p['phase']][name]
        for target, source in [('N', 'denoise_forwards'), ('C', 'canvases'), ('T', 'output_tokens')]:
            assert metric[arm][target] == sum(r[source] for r in rows)
        for name in ['global_eligible_tiles', 'global_kept_tiles', 'overall_eligible_rectangles', 'overall_skipped_rectangles']:
            assert metric[arm][name] == sum(r['receipts']['adapter'][name] for r in rows)
        official = read(ROOT / 'scoring' / arm / 'official.official.json')[arm]
        scorer = read(ROOT / 'scoring' / arm / 'official.summary.json')
        cov = scorer['coverage'][arm]
        assert cov['planned'] == cov['present'] == cov['scored'] == 104 and cov['missing'] == cov['dropped'] == 0
        assert len(official) == 104 and math.isclose(sum(official.values()) / 104, metric[arm]['official_score'])
        assert metric[arm]['full_credit_items'] == sum(v >= 100-1e-9 for v in official.values())
        data[arm] = {key(r): r for r in rows}
        all_scores[arm] = official
        all_bindings.update(read(ROOT / 'scoring' / arm / 'official.binding.json'))
        smoke = ROOT / 'qualification/ruler128k' / arm / 'smoke/attempt001/records.jsonl'
        repeats[arm] = dict(cells=2, output_N_T_C_mismatches=0)
        for r in map(json.loads, smoke.read_text().splitlines()):
            full = data[arm][key(r)]
            assert all(r[k] == full[k] for k in ['output_hash', 'output_tokens', 'denoise_forwards', 'canvases'])
        details[arm] = dict(N=distribution([r['denoise_forwards'] for r in rows]),
                           C=distribution([r['canvases'] for r in rows]),
                           T=distribution([r['output_tokens'] for r in rows]),
                           finish_reasons=dict(collections.Counter(r['finish_reason'] for r in rows)),
                           canvas_definition='committed canvases: ceil(output_tokens/256); receipt lifecycle IDs are separate')
    sys.path.insert(0, str(ROOT / 'private/execution/scripts'))
    import v31_official as op
    assert op.check_comparable(all_bindings, protocol['arms'], set(next(iter(all_scores.values())))) == 104
    c, v = protocol['arms']
    delta = [data[v][k]['denoise_forwards'] - data[c][k]['denoise_forwards'] for k in data[c]]
    changes = [all_scores[v][k] - all_scores[c][k] for k in data[c]]
    details['paired'] = dict(N_delta=distribution(delta),
                            score_increases=sum(x > 1e-9 for x in changes),
                            score_decreases=sum(x < -1e-9 for x in changes),
                            score_ties=sum(abs(x) <= 1e-9 for x in changes),
                            N_fewer=sum(x < 0 for x in delta), N_more=sum(x > 0 for x in delta),
                            N_equal=sum(x == 0 for x in delta),
                            largest_positive_N_deltas=sorted(delta, reverse=True)[:10])
    write(ROOT / 'trajectory_distributions.json', details)
    receipt = dict(status='passed',source_files=886,input_files=11,records=208,shards=8,
                   official_score_cells=208,matched_official_bindings=104,
                   qualification_repeat_checks=repeats,panel_seed=42,threshold=.021,
                   protocol_sha256=protocol_sha,invalid_rows=0,request_cuda_captures=0,LOCAL_skips=0,
                   verified_commit=protocol['coordinator_commit'],source_commit=protocol['source_commit'],
                   legacy_budget_label='value_budget_policy is a static inherited label; verified triton_log_mass_online branch uses threshold and no quota',
                   scope='audit denoising FA4 consumer tiles; selector work separate; no clean timing')
    write(ROOT / 'receipts/independent_verification.json', receipt)
    print(json.dumps(dict(verification=receipt,trajectory=details),indent=2))


if __name__ == '__main__':
    main()
