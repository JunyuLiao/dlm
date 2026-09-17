"""Orchestration/report checks independent of model inference."""
import json
from pathlib import Path

import pytest

from experiments.diffusion_gemma_jl_output_aware import config, protocol, calibration


def test_historical_aime_policies_are_compatible_without_inference(tmp_path):
    setup = json.loads((protocol.ROOT/'setup.json').read_text())
    for name, cfg in config.BASELINES.items():
        if name == 'unweighted_centered': continue
        for target in config.TARGETS:
            out = calibration.import_aime_baseline(tmp_path, name, target, cfg, setup, {'fingerprint':'test'})
            assert out['imported_aime'] and not out['heldout_used']
            assert out['calibration_ids'] == [r['id'] for r in setup['calibration'] if r['benchmark']=='aime26']
            calibration.audit_policy(tmp_path, out, setup, {'fingerprint':'test'})


def test_report_complete_matrix_raw_only_and_deterministic(tmp_path, monkeypatch):
    from experiments.diffusion_gemma_jl_output_aware import report
    from experiments.diffusion_gemma_value_aware_gpu.routing import FIELDS
    from experiments.diffusion_gemma_value_aware.report_metrics import aggregate
    setup = json.loads((protocol.ROOT/'setup.json').read_text())
    monkeypatch.setattr(report, 'prepare', lambda root: setup)
    monkeypatch.setattr(report, 'execution', lambda root: {'fingerprint':'test'})
    monkeypatch.setattr(report, 'audit_policy', lambda *args: None)
    monkeypatch.setattr(report, 'supporting_audit', lambda *args: ([], [], {}))
    monkeypatch.setattr(report, 'diagnostic_summary', lambda *args: ([], {}))
    monkeypatch.setattr(report, 'seed_summary', lambda *args: ([], {}))
    monkeypatch.setattr(report, 'audit_matrices', lambda *args: None)
    # Plotting and actual NeMo scoring have separate integration tests; this
    # test covers the full37x80 matrix, raw aggregation and report reproducibility.
    monkeypatch.setattr(report, 'plots', lambda *args: [])
    raw = tmp_path/'raw.json'; raw.write_text('{}')
    monkeypatch.setattr(report, 'shard_path', lambda *args: raw)
    def output(adapter, root, row, stage, label, name, cfg, thresholds, contract):
        assert adapter is None
        records = [dict.fromkeys(FIELDS, 0) for _ in range(2)]
        for idx, rec in enumerate(records):
            rec.update(layer=idx*5, step=0, head=0, attention_type=('local','global')[idx],
                probe='execution', calls=1, eligible=10*(idx+1), skipped=0 if label=='dense' else 5*(idx+1),
                rows=2, mass_sum=2 if label=='dense' else 1.5, dense_sq=2, error_sq=0 if label=='dense' else .1)
        return dict(prediction='A', completion_tokens=[1,2], records=records,
            records_source={'path':str(raw),'sha256':protocol.sha(raw.read_bytes())})
    monkeypatch.setattr(report, 'cached', output)
    def pair(row, out, dense):
        return dict(id=row['id'],benchmark=row['benchmark'],task=row['task'],calibration=row['calibration'],
            accuracy=1.,dense_accuracy=1.,matching=2,compared=2,exact_match=True,termination_reason='stop',
            unparsed_answer=False,output_length=2,aggregates={k:aggregate([r for r in out['records']
                if k=='overall' or r['attention_type']==k]) for k in ('overall','global','local')})
    monkeypatch.setattr(report.evidence, 'pair', pair)
    monkeypatch.setattr(report, 'nemo_pairs', lambda row, outs: ([dict(condition=c,**pair(row,o,outs['dense'])) for c,o in outs.items()], []))
    (tmp_path/'final_configs').mkdir()
    for label in setup['conditions']:
        name = label.rsplit('_s',1)[0] if label!='dense' else label
        target = int(label.rsplit('_s',1)[1])/100 if label!='dense' else 0.
        cfg = {} if name=='dense' else {**config.BASELINES,**config.PROJECTED}[name]
        c = dict(fingerprint='test',name=name,target=target,config=cfg,sources={},
            thresholds={b:None if name=='dense' else {k:{'log_threshold':-2.} for k in ('local','global')} for b in ('aime26','longbench_v2')})
        (tmp_path/'final_configs'/f'{label}.json').write_text(json.dumps(c))
        for b in ('aime26','longbench_v2'):
            if name=='dense': continue
            path = tmp_path/'policies'/b/f'{label}.json'; path.parent.mkdir(exist_ok=True,parents=True)
            path.write_text(json.dumps(dict(benchmark=b,name=name,target=target,policy=c['thresholds'][b])))
    a = report.regenerate(tmp_path)
    assert a['complete'] and a['completed']==2960
    rows = json.loads((tmp_path/'summary.json').read_text())
    assert len(rows) == 37*4
    assert all(r['overall_physical_sparsity'] == (.0 if r['name']=='dense' else .5) for r in rows)
    assert a == report.regenerate(tmp_path)
    # A single absent sparse shard cannot silently be a completed experiment.
    def absent(*args, **kwargs):
        if args[2]['id']==setup['final'][0]['id'] and args[4]=='jl_gaussian_r8_s50':
            raise FileNotFoundError('synthetic missing shard')
        return output(*args, **kwargs)
    monkeypatch.setattr(report, 'cached', absent)
    b = report.regenerate(tmp_path)
    assert not b['complete'] and b['completed']==2959 and b['missing']
