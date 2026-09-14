"""Small synthetic raw-shard audit; never loads the model or CUDA."""
import json

from experiments.diffusion_gemma_value_aware_gpu import report
from experiments.diffusion_gemma_value_aware_gpu.protocol import CONFIGS, METHODS, TARGETS
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_value_aware.protocol import sha


def test_full1040_report_reproduces_and_refuses_one_missing_shard(tmp_path, monkeypatch):
    rows = []
    for benchmark, n in (('aime26', 30), ('longbench_v2', 50)):
        for i in range(n):
            rows.append(dict(id=f'{benchmark}/{i}', benchmark=benchmark,
                task='AIME26' if benchmark == 'aime26' else f'domain{i%3}',
                source_id=i, seed=42, generation_budget=2048 if benchmark=='aime26' else 128,
                prompt_hash=f'prompt/{benchmark}/{i}', expected='1' if benchmark=='aime26' else 'B',
                calibration=benchmark=='aime26' and i<6, previous_manifest_member=benchmark=='aime26' or i<30,
                truncated=False))
    conditions = {'dense':dict(config={}, target=0., thresholds={})}
    for target in TARGETS:
        for name in METHODS:
            conditions[f'{name}_s{int(target*100)}'] = dict(config=CONFIGS[name], target=target,
                thresholds={b:{k:dict(log_threshold=0.) for k in ('local','global')} for b in ('aime26','longbench_v2')})
    setup = dict(final=rows, conditions=list(conditions), revision='fixture')
    contract = dict(fingerprint='fixture')
    monkeypatch.setattr(report, 'prepare', lambda root:setup)
    monkeypatch.setattr(report, 'execution', lambda root:contract)
    # Real source files are still read/hashed and all scorer/pair/aggregation
    # logic runs; only model-dimension/fingerprint checks are outside this unit.
    def cached(adapter, root, row, stage, name, config, policy, execution):
        return json.loads(shard_path(root, stage, name, row['id']).read_text())
    monkeypatch.setattr(report, 'cached', cached)
    for label, c in conditions.items():
        path = tmp_path/'final_configs'/f'{label}.json'
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(dict(c, fingerprint='fixture', sources={})))
        for row in rows:
            skip = 0 if label == 'dense' else (5 if c['target']==.5 else 7)
            records = [dict(layer=layer, head=0, step=0, attention_type=kind, probe='execution',
                eligible=10, skipped=skip, softmax_skipped=skip, pv_omitted=skip, compensated=0,
                rows=2, mass_sum=2. if skip==0 else 1.5, denominator_mass_sum=2. if skip==0 else 1.5,
                error_sq=0. if skip==0 else .01, dense_sq=4., prefix_eligible=8, prefix_skipped=skip,
                canvas_eligible=2, canvas_skipped=0, boundary_eligible=0, boundary_skipped=0,
                rescued_rows=0, calls=1) for layer, kind in ((0,'local'),(5,'global'))]
            data = dict(id=row['id'], prompt_hash=row['prompt_hash'], seed=row['seed'], generation_budget=row['generation_budget'],
                config=c['config'], thresholds=c['thresholds'].get(row['benchmark']), records=records,
                prediction='1' if row['benchmark']=='aime26' else 'The correct answer is (B).', completion_tokens=[1,2],
                generation_metadata={}, termination_reason='eos', backend='fixture', score=1.)
            path = shard_path(tmp_path, 'dense' if label=='dense' else 'final', label, row['id'])
            path.parent.mkdir(exist_ok=True, parents=True); path.write_text(json.dumps(data))
    first = report.regenerate(tmp_path)
    assert first['complete'] and first['completed'] == 1040
    second = report.regenerate(tmp_path)
    assert first['artifacts'] == second['artifacts']
    from experiments.diffusion_gemma_value_aware_gpu.verify import verify
    proof = verify(tmp_path)
    assert proof['passed'] and not proof['inference_performed']
    assert proof['artifact_count'] == len(second['artifacts'])
    summaries = json.loads((tmp_path/'summary.json').read_text())
    assert len([r for r in summaries if r['split']=='full']) == 26
    assert all(r['count'] == 24 for r in summaries if r['split']=='noncalibration24')
    assert all(r['count'] == 50 for r in summaries if r['benchmark']=='longbench_v2' and r['split']=='full')
    missing = shard_path(tmp_path, 'final', 'value_s75', rows[-1]['id'])
    missing.unlink()
    result = report.regenerate(tmp_path)
    assert not result['complete'] and result['completed'] == 1039
    assert len(result['missing']) == 1 and not result['violations']
