"""Export sanitized per-item evidence and bind the completed study artifacts."""
import json
from pathlib import Path
from scripts import v31_humaneval_value_study as study


def publish(root):
    study.verify(root)
    assert study.read(root/'independent_verification.json')['status'] == 'passed'
    assert study.read(root/'independent_rescoring.json')['mismatches'] == 0
    dest = root/'analysis/attempt001/per_item_metrics.jsonl'
    exported = []
    for mode in study.MODES:
        for pass_name in ('clean', 'audit'):
            paths = study.selected_paths(root, mode, pass_name)
            scores = study.read(root/f'private/scoring/{mode}/{pass_name}.json')['scores']
            for label in study.LABELS:
                rows = [r for p in paths[label] for r in study.base.jsonl(p/'records.jsonl')]
                for r in sorted(rows, key=lambda r:r['index']):
                    a = r['receipts']['adapter'] if pass_name=='audit' else None
                    row = dict(mode=mode, pass_name=pass_name, method=label,
                        index=r['index'], seed=r['panel_seed'], rng_seed=r['rng_seed'],
                        correct=scores[label][str(r['index'])]['passed'],
                        pass_and_finished=scores[label][str(r['index'])]['pass_and_finished'],
                        finish=scores[label][str(r['index'])]['finish'],
                        N=r['denoise_forwards'], C=r['canvases'], T=r['output_tokens'],
                        output_hash=r['output_hash'],
                        S=r['decode_s'] if pass_name=='clean' else None,
                        W=r['wall_s'] if pass_name=='clean' else None,
                        global_eligible_tiles=a['global_eligible_tiles'] if pass_name=='audit' else None,
                        global_kept_tiles=a['global_kept_tiles'] if pass_name=='audit' else None,
                        local_eligible_rectangles=a['native_local_eligible_rectangles'] if pass_name=='audit' else None)
                    exported.append(row)
    assert len(exported) == 5904
    with dest.open('x') as f:
        for row in exported:
            f.write(json.dumps(row, sort_keys=True)+'\n')
    inventory = []
    for marker in sorted(root.rglob('attempt_complete.json')):
        receipt = study.verify_leaf(root, marker.parent)
        config = study.read(marker.parent/'config.json')
        assert receipt['status']=='complete' and not receipt['errors']
        inventory.append(dict(attempt=str(marker.parent.relative_to(root)),
            marker_sha256=study.sha(marker), stage=config['stage'],
            mode=config['mode'], pass_name=config['pass_name'], family=config['family'],
            records=receipt['records'], measured_records=receipt['measured_records'],
            gpu_reservation_s=receipt['gpu_reservation_s'], files_sha256=receipt['files_sha256']))
    assert len(inventory)==58
    study.write(root/'receipts/attempt_inventory.json', dict(attempts=inventory,
        completed_engine_jobs=len(inventory), failed_engine_jobs=0,
        native_saved_records=sum(r['records'] for r in inventory),
        native_measured_records=sum(r['measured_records'] for r in inventory),
        gpu_reservation_s=sum(r['gpu_reservation_s'] for r in inventory)), True)
    paths = ['protocol.json', 'spec.md', 'model_source_fingerprints.json', 'preflight_tests.xml',
        'receipts/source_fingerprints.json', 'receipts/attempt_inventory.json',
        'qualification/scorer_gate.json', 'summary.json', 'summary.md', 'final_complete.json',
        'independent_verification.json', 'independent_rescoring.json',
        'analysis/attempt001/summary.json', 'analysis/attempt001/summary.md',
        'analysis/attempt001/final_complete.json', 'analysis/attempt001/per_item_metrics.jsonl']
    paths += [str(p.relative_to(root)) for p in sorted((root/'calibration').rglob('*.json'))]
    private_scores = {str(p.relative_to(root)):study.sha(p) for p in (root/'private/scoring').glob('*/*.json')}
    receipt = dict(status='verified_complete_with_cross_pass_trajectory_limitations',
        protocol_sha256=study.sha(root/'protocol.json'), native_source_commit=study.read(root/'protocol.json')['source_commit'],
        published_per_item_records=len(exported), official_rescore_mismatches=0,
        clean_physical_work_match='unverified; audit work is a separate population',
        files_sha256={name:study.sha(root/name) for name in paths},
        private_scoring_sha256=private_scores,
        reporting_sources_sha256={name:study.sha(study.ROOT/name) for name in
            ['scripts/v31_humaneval_value_verify.py','scripts/v31_humaneval_value_report.py','scripts/v31_humaneval_value_publish.py']})
    study.write(root/'verified_complete.json', receipt, True)
    print(json.dumps(dict(status=receipt['status'], exported_records=len(exported), engine_jobs=58)))


if __name__=='__main__':
    publish(study.DEFAULT)
