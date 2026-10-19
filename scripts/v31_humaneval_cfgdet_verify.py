"""Independent completion/input/metric checks; reads numeric and private bindings."""
import argparse
import json
import math
from pathlib import Path
import statistics

from scripts import v31_humaneval_cfgdet_study as study
from scripts import v31_official as official


def verify(root):
    protocol = study.verify(root)
    report = study.read(root/'summary.json')
    marker = study.read(root/'native_complete.json')
    assert marker['status'] == 'native_and_primary_scoring_complete'
    assert marker['protocol_sha256'] == report['protocol_sha256'] == study.sha(root/'protocol.json')
    for name, digest in marker['files_sha256'].items():
        assert study.sha(root/name) == digest
    assert study.read(root/'qualification/scorer_gate.json')['passed'] == 328
    all_rows = all_scores = 0
    bindings = {}
    for mode in study.MODES:
        pool = root/f'private/pools/{mode}'
        manifest = study.read(pool/'humaneval_generation_manifest.json')
        assert len(manifest) == 164
        assert all(m['generation_budget'] == 8192 and m['thinking'] == (mode == 'thinking_on') for m in manifest)
        for pass_name in ('audit', 'clean'):
            scores = study.read(root/f'private/scoring/{mode}/{pass_name}.json')
            for name, digest in scores['input_sha256'].items():
                assert study.sha(root/name) == digest
            paths = study.selected_paths(root, mode, pass_name)
            for label in study.LABELS:
                rows = [r for p in paths[label] for r in study.base.jsonl(p/'records.jsonl')]
                assert len(rows) == 164 and {r['index'] for r in rows} == set(range(164))
                by_index = {r['index']: r for r in rows}
                assert set(scores['scores'][label]) == {str(i) for i in range(164)}
                for p in paths[label]:
                    private = study.base.jsonl(p/f'private/run_{label}.private.jsonl')
                    for r in private:
                        _, bound, valid = official.bind(r, official.Pools(pool, ('thinking',)), False)
                        assert valid and r['id'] == manifest[r['index']]['id']
                        key = (mode, pass_name, r['index'])
                        fields = {k: r[k] for k in official.COMPARE_FIELDS}
                        if key in bindings:
                            assert fields == bindings[key]
                        else:
                            bindings[key] = fields
                for r in rows:
                    assert r['panel_seed'] == 42 and r['repeat'] == 0
                    assert r['budget'] == 8192 and r['mage_k'] in (None, protocol['prefix_budget_tokens'])
                    assert r['cuda_graph_captures'] == 0
                    assert r['canvases'] == math.ceil(r['output_tokens']/256)
                    assert r['denoise_forwards'] == r['sampler_calls']-r['canvases']
                    assert r['native_interval_trace_available'] is True
                    assert sum(r['native_interval_steps'])==r['denoise_forwards']
                    assert r['fix_51994'] and r['fa4_local_fix']
                    if label == study.V2:
                        assert r['receipts']['adapter']['humaneval_threshold'] == report['modes'][mode]['threshold']['threshold']
                    if pass_name == 'audit':
                        a = r['receipts']['adapter']
                        assert a['global_calls'] == 5*r['denoise_forwards']
                        assert a['native_local_calls_audited'] == 25*r['denoise_forwards']
                        assert a['native_local_skipped_tiles'] == 0
                        assert 0 <= a['global_kept_tiles'] <= a['global_eligible_tiles']
                        if label not in study.REFERENCES:
                            assert a['humaneval_prefix_budget_tokens'] == protocol['prefix_budget_tokens']
                            if label == study.V2:
                                assert a['humaneval_threshold'] == report['modes'][mode]['threshold']['threshold']
                            else:
                                assert a['humaneval_fixed_budget_errors'] == 0
                    else:
                        assert r['timer_policy'] == 'request_and_prefill_boundaries'
                        assert abs(r['prefill_s']+r['decode_s']-r['wall_s']) < 2e-5
                m = report['modes'][mode]['methods'][label]
                if pass_name == 'clean':
                    assert m['correct'] == sum(x['passed'] for x in scores['scores'][label].values())
                    for field, key in [('N_mean', 'denoise_forwards'), ('C_mean', 'canvases'),
                                       ('T_mean', 'output_tokens'), ('W_mean_s', 'wall_s'), ('S_mean_s', 'decode_s')]:
                        assert abs(m[field]-statistics.mean(r[key] for r in rows)) < 1e-10
                else:
                    e = sum(r['receipts']['adapter']['global_eligible_tiles'] for r in rows)
                    k = sum(r['receipts']['adapter']['global_kept_tiles'] for r in rows)
                    local = sum(r['receipts']['adapter']['native_local_eligible_rectangles'] for r in rows)
                    assert abs(m['global_sparsity']-(e-k)/e) < 1e-12
                    assert abs(m['overall_sparsity']-(e-k)/(e+local)) < 1e-12
                all_rows += len(rows)
                all_scores += len(scores['scores'][label])
    assert all_rows == all_scores == 5904
    for done in (root/'runs').rglob('attempt_complete.json'):
        leaf=done.parent
        study.verify_leaf(root,leaf)
        c=study.read(leaf/'config.json')
        cg='default' if c['family']=='dense_full_fix51994' else 'PIECEWISE'
        assert not study.policy_errors(leaf/'private/kernel_inventory.json',cg)
    assert all(x['passed'] for x in study.read(root/'receipts/independent_prefix_replication.json')['comparisons'])
    assert all(x['passed'] for x in study.read(root/'receipts/full164_audit_clean_equivalence.json')['comparisons'])
    return dict(status='passed', source_commit=protocol['source_commit'],
                protocol_sha256=study.sha(root/'protocol.json'), validated_records=all_rows,
                officially_scored_records=all_scores, settings=2, questions_per_arm=164,
                passes=['audit', 'clean'], labels=list(study.LABELS))


def rescore(root):
    study.he.sandbox_preflight()
    jobs, expected = [], []
    for mode in study.MODES:
        gold = study.read(root/f'private/pools/{mode}/gold.json')
        for pass_name in ('audit', 'clean'):
            scores = study.read(root/f'private/scoring/{mode}/{pass_name}.json')['scores']
            paths = study.selected_paths(root, mode, pass_name)
            for label in study.LABELS:
                for path in paths[label]:
                    records = study.base.jsonl(path/f'private/run_{label}.private.jsonl')
                    for r in records:
                        problem = gold[r['id']]
                        code, _ = study.extract(r['completion'], problem['entry_point'], mode == 'thinking_on')
                        jobs.append((problem, code))
                        expected.append(scores[label][str(r['index'])]['passed'])
    actual, timeouts = study.run_all(jobs, study.he)
    mismatches = sum(bool(a[0]) != b for a, b in zip(actual, expected))
    assert len(actual) == 5904 and mismatches == 0
    qualification=study.read(root/'private/scoring/qualification.json')['records']
    expected_qualification={(r['mode'],r['pass_name'],r['label'],r['seed'],r['index']):r['passed'] for r in qualification}
    jobs=[];expected=[]
    for mode in study.MODES:
        gold=study.read(root/f'private/pools/{mode}/gold.json')
        for part in ('clean','audit'):
            paths=study.selected_paths(root,mode,part,'qualification')
            for label in study.LABELS:
                for path in paths[label]:
                    for row in study.base.jsonl(path/f'private/run_{label}.private.jsonl'):
                        code,_=study.extract(row['completion'],gold[row['id']]['entry_point'],mode=='thinking_on')
                        jobs.append((gold[row['id']],code))
                        expected.append(expected_qualification[(mode,part,label,row['panel_seed'],row['index'])])
    checked,q_timeouts=study.run_all(jobs,study.he)
    q_mismatches=sum(bool(a[0])!=b for a,b in zip(checked,expected))
    assert len(checked)==432 and q_mismatches==0
    return dict(records=len(actual), mismatches=mismatches, timeouts_first_pass=timeouts,
                timeouts_after_rerun=sum(x[1] == 'timeout' for x in actual),
                qualification_records=len(checked),qualification_mismatches=q_mismatches,
                qualification_timeouts_first_pass=q_timeouts,
                scorer_sha256=study.sha(Path(study.he.__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--study', type=Path, default=study.DEFAULT)
    parser.add_argument('--rescore', action='store_true')
    a = parser.parse_args()
    result = verify(a.study.absolute())
    study.write(a.study/'independent_verification.json', result, True)
    if a.rescore:
        scored = rescore(a.study.absolute())
        study.write(a.study/'independent_rescoring.json', scored, True)
        result['independent_rescoring'] = scored
    print(json.dumps(result))
