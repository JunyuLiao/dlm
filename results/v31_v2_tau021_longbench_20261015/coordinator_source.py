"""Complete only frozen V2 tau=.021 LongBench; leave other campaigns paused."""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

NAME = 'v31_v2_tau021_longbench_20261015'
PARENT = 'v31_v2_local_threshold_sweep_20261014_attempt002'
ARMS = ['current_v31_control', 'dense_full_fix51994', 'dense_piecewise', 'allkept_fa4']
TAU = .021
PYTHON = '/home/exouser/dyh/v31_env_20261005/bin/python'


def read(p): return json.loads(Path(p).read_text())


def module(path):
    spec = importlib.util.spec_from_file_location('frozen_v2_campaign', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def load(study):
    return module(study / 'private/execution/scripts/v31_v2_local_threshold_sweep.py')


def prepare(study, parent):
    if (study / 'manifest.json').exists(): return
    if study.exists(): raise RuntimeError('Preserve incomplete preparation; use a new attempt directory')
    old = load(parent); protocol = read(parent / 'protocol.json')
    old.verify(parent, protocol)
    if old.runtime_identity() != protocol['runtime']: raise RuntimeError('Runtime drift: fresh matched controls required')
    marker = read(parent / 'controls_report_complete.json')
    for rel, digest in marker['files_sha256'].items():
        if old.sha(parent / rel) != digest: raise RuntimeError('Frozen control artifact drift: ' + rel)
    study.mkdir(); (study / 'private').mkdir(); (study / 'receipts').mkdir()
    for directory in ['execution', 'pools', 'schedules']:
        shutil.copytree(parent / 'private' / directory, study / 'private' / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for rel in ['protocol.json', 'model_source_fingerprints.json', 'receipts/source_fingerprints.json']:
        shutil.copyfile(parent / rel, study / rel)
    m = load(study); m.verify(study, protocol)
    reference_hashes = {}
    for arm in ARMS:
        for section in ['reports', 'scoring']:
            src = parent / section / 'longbench_v2' / arm
            dst = study / section / 'longbench_v2' / arm
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('scorer.log'))
        dst = study / 'runs/longbench_v2' / arm
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(parent / 'runs/longbench_v2' / arm, target_is_directory=True)
        for f in (parent / 'reports/longbench_v2' / arm).glob('*.json'):
            reference_hashes[str(f.relative_to(parent))] = m.sha(f)
        s = read(parent / 'reports/longbench_v2' / arm / 'summary.json')
        for rel in s['attempts']:
            for name in ['config.json', 'receipt.json', 'attempt_complete.json', 'records.jsonl']:
                reference_hashes[rel + '/' + name] = m.sha(parent / rel / name)
    shutil.copyfile(Path(__file__), study / 'coordinator_source.py')
    manifest = dict(schema='v2_tau021_longbench_resume_v1', date='2026-10-15',
        parent_campaign=parent.name, authorized_scope='503 LongBench-v2 seed-1 cells; V2 tau=0.021 only',
        source_commit=protocol['source_commit'], protocol_sha256=m.sha(study / 'protocol.json'),
        coordinator_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        coordinator_sha256=m.sha(study / 'coordinator_source.py'),
        selector='value_v2_online_preserve_mass', threshold=TAU, native_sampler_unchanged=True,
        source_files=len(protocol['execution_fingerprints']), input_files=len(protocol['input_fingerprints']),
        reference_artifact_hashes=reference_hashes, references=ARMS,
        qualification='repeat two cells per control and V2; if control trajectories differ, generate fresh matched controls',
        deferred='all other thresholds, finer sweep, AIME/MATH generation, clean timing, and separate refresh campaign',
        timing='instrumented audit only; clean timing paused')
    m.write(study / 'manifest.json', manifest, True)
    m.write(study / 'receipts/preflight.json', dict(status='passed', runtime_identical=True,
        source_files_verified=manifest['source_files'], input_files_verified=manifest['input_files'],
        parent_control_report_hashes_verified=len(marker['files_sha256']),
        prior_source_matched_qualification=read(parent / 'qualification/tests_receipt.json')), True)
    (study / 'README.md').write_text('# V2 tau=0.021 LongBench-v2 completion\n\n'
        'Only the 503-item seed-1 V2 arm is authorized. The original frozen execution and protocol are copied bytewise. '
        'Matched V31, FULL fix51994, PIECEWISE and all-kept controls are referenced from the preceding completed panel, '
        'subject to fresh control reproducibility checks. Original reports and paused campaigns remain immutable. '
        'Clean timing and all other threshold experiments stay paused.\n\n'
        'Run the frozen coordinator with `--stage run --study <this-directory>` under the pinned interpreter. '
        'Completed shards are validated and skipped; partial attempts are preserved and receive a new attempt number.\n')


def monitor(study, stop):
    while not stop.is_set():
        state = read(study / 'state.json') if (study / 'state.json').exists() else {}
        a = study / state.get('attempt', 'unused') / 'records.jsonl'
        gpu = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw',
                              '--format=csv,noheader'], capture_output=True, text=True)
        item = dict(time_unix=time.time(), state=state, gpu=gpu.stdout.strip(),
                    records=sum(1 for _ in a.open()) if a.exists() else 0,
                    candidate_completed_shards=len(list((study / 'runs/longbench_v2/v2_tau_0p021').glob('shard*/attempt*/attempt_complete.json'))))
        with (study / 'monitor.jsonl').open('a') as f: f.write(json.dumps(item)+'\n')
        print(json.dumps(dict(event='monitor', **item)), flush=True)
        stop.wait(1800)


def records(m, study, summary):
    return [r for rel in summary['attempts'] for r in m.jsonl(study / rel / 'records.jsonl')]


def qualify(m, study, protocol):
    receipt = study / 'qualification/reproducibility.json'
    if receipt.exists():
        decision = read(receipt)
    else:
        result = {}; shard = study / 'private/schedules/longbench_v2/smoke.json'
        for arm in ARMS + ['value_v2_online_preserve_mass']:
            path = m.run_shard(study, protocol, 'longbench_v2', arm, TAU if arm.startswith('value_') else None, shard, True)
            rows = m.jsonl(path / 'records.jsonl')
            if arm in ARMS:
                ref = {m.key(r):r for r in records(m, study, read(study / 'reports/longbench_v2' / arm / 'summary.json'))}
                result[arm] = {field:sum(r[field] != ref[m.key(r)][field] for r in rows)
                               for field in ['output_hash','output_tokens','denoise_forwards','canvases']}
        decision = dict(status='complete', cells_per_arm=2, control_mismatches=result,
                        reuse_compatible=not any(v for x in result.values() for v in x.values()))
        m.write(receipt, decision, True)
    if not decision['reuse_compatible']:
        # Retain old references but prevent their use in the final comparison.
        for section in ['reports','scoring','runs']:
            for arm in ARMS:
                old = study / section / 'longbench_v2' / arm
                retained = study / 'private/historical_references' / section / arm
                retained.parent.mkdir(parents=True, exist_ok=True)
                if not retained.exists(): old.rename(retained)
        for arm in ARMS: m.summarize_arm(study, 'longbench_v2', arm, None)


def publish(m, study):
    sys.path.insert(0, str(study / 'private/execution/scripts'))
    import v31_official as op
    candidate = read(study / 'reports/longbench_v2/v2_tau_0p021/summary.json')
    xs = [read(study / 'reports/longbench_v2' / arm / 'summary.json') for arm in ARMS] + [candidate]
    vrows = records(m, study, candidate)
    vscores = read(study / 'scoring/longbench_v2/v2_tau_0p021/official.official.json')['v2_tau_0p021']
    vb = read(study / 'scoring/longbench_v2/v2_tau_0p021/official.binding.json')
    comparisons = {}
    for ref in xs[:-1]:
        name = ref['arm']; cr = records(m, study, ref)
        cs = read(study / 'scoring/longbench_v2' / name / 'official.official.json')[name]
        cb = read(study / 'scoring/longbench_v2' / name / 'official.binding.json')
        op.check_comparable({**cb, **vb}, [name, 'v2_tau_0p021'], set(vscores))
        comparisons[name] = m.paired(cs, vscores, cr, vrows)
    m.write(study / 'paired_comparisons.json', comparisons, True)
    # Rank actual observed denoising canvas IDs and require a paired visible
    # canvas cohort at each prefix. No fabricated zero-valued early canvases.
    vmap = {m.key(r):(r['canvases'],m.ordinal_canvas(r['receipts']['adapter'])) for r in vrows}
    early=[]
    for ref in xs[:-1]:
        cmap={m.key(r):(r['canvases'],m.ordinal_canvas(r['receipts']['adapter'])) for r in records(m,study,ref)}
        for prefix in [1,2,4,8,12,16,20]:
            keys=[k for k in cmap if cmap[k][1] and vmap[k][1]
                  and min(cmap[k][0],vmap[k][0])>=prefix
                  and min(len(cmap[k][1]),len(vmap[k][1]))>=prefix]
            if not keys:continue
            def through(mapping,key,field):
                return sum(a[field] for i,a in mapping[key][1].items() if i<=prefix)
            ck=[through(cmap,k,'kept_tiles') for k in keys];vk=[through(vmap,k,'kept_tiles') for k in keys]
            ce=sum(through(cmap,k,'eligible_tiles') for k in keys);ve=sum(through(vmap,k,'eligible_tiles') for k in keys)
            early.append(dict(reference=ref['arm'],canvas_prefix=prefix,common_items=len(keys),
                reference_eligible_tiles=ce,candidate_eligible_tiles=ve,
                reference_kept_tiles=sum(ck),candidate_kept_tiles=sum(vk),
                reference_sparsity=1-sum(ck)/ce,candidate_sparsity=1-sum(vk)/ve,
                kept_work_ratio=sum(vk)/sum(ck),per_item_kept_delta=m.quantiles([v-c for c,v in zip(ck,vk)])))
    m.write(study / 'cumulative_canvas_work.json',early,True)
    with (study/'cumulative_canvas_work.csv').open('w',newline='') as f:
        fields=['reference','canvas_prefix','common_items','reference_eligible_tiles','candidate_eligible_tiles',
                'reference_kept_tiles','candidate_kept_tiles','reference_sparsity','candidate_sparsity','kept_work_ratio']
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for row in early:w.writerow({k:row[k] for k in fields})
    with (study / 'phase_accounting.csv').open('w', newline='') as f:
        fields = ['arm','phase','calls','eligible_tiles','kept_tiles','skipped_tiles','kept_fraction','sparsity']
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for s in xs:
            for phase,a in s['work']['phases'].items():w.writerow(dict(arm=s['arm'],phase=phase,**a))
    cost=[]
    for s in xs:
        a=s['work'];cost.append(dict(arm=s['arm'],**{k:a[k] for k in [
            'global_eligible_tiles','global_kept_tiles','global_skipped_tiles','overall_eligible_rectangles',
            'overall_skipped_rectangles','selector_discovery_qk_rectangles','selector_candidate_evaluations',
            'selector_discovery_ms','selector_selection_ms','value_selected_prefix_tiles','N','C','T',
            'audit_wall_s','audit_decode_s']},
            worker_gpu_reservation_s=sum(read(study/rel/'receipt.json')['worker_reservation_seconds'] for rel in s['attempts']),
            timing_status='instrumented audit; not clean latency'))
    m.write(study / 'cost_separation.json',cost,True)
    lines=['# V2 tau=0.021: complete LongBench-v2 comparison','',
        '503 paired seed-1 problems; unchanged native sampler and selector-only FA4 execution. Previously examined benchmark; no held-out claim.',
        '','| Method | Correct | Accuracy | N/problem | Median N | C/problem | N/C | Tokens/problem | Kept GLOBAL tiles | GLOBAL sparsity | Overall sparsity |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in xs:
        a=s['work'];lines.append(f"| {s['arm']} | {s['correct']}/503 | {s['accuracy']:.2%} | {a['mean_N']:.2f} | {a['median_N']:.1f} | {a['mean_C']:.2f} | {a['N_per_C']:.2f} | {a['mean_T']:.1f} | {a['global_kept_tiles']:,} | {a['global_sparsity']:.2%} | {a['overall_sparsity']:.2%} |")
    lines+=['','| V2 versus | Accuracy delta (pp) | Paired bootstrap 95% CI (pp) | McNemar p | N ratio | Kept GLOBAL ratio |',
            '|---|---:|---|---:|---:|---:|']
    for s in xs[:-1]:
        p=comparisons[s['arm']];low,high=p['problem_cluster_bootstrap_ci95_pp'];a=s['work'];v=candidate['work']
        lines.append(f"| {s['arm']} | {p['delta_pp']:+.3f} | [{low:+.3f}, {high:+.3f}] | {p['mcnemar_exact_p']:.4f} | {v['N']/a['N']:.5f} | {v['global_kept_tiles']/a['global_kept_tiles']:.5f} |")
    lines+=['','## Official difficulty and length strata','',
            '| Method | Easy | Hard | Short | Medium | Long | Capped | Unparsed | No final response | Input-truncated |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in xs:
        d=next(iter(s['official_scorer']['arms'][s['arm']].values()));r=d['runs']['seed1_repeat0']
        lines.append('| '+s['arm']+' | '+' | '.join(f'{r[k]:.1f}%' for k in ['Easy','Hard','Short','Medium','Long'])+' | '+' | '.join(str(d.get(k,0)) for k in ['capped','unparsed','no_final_response','truncated_items'])+' |')
    lines+=['','GLOBAL work is receipt-measured physical FA4 consumer work. Selector discovery and selection are additional costs.',
            'Equal sparsity, equal retained work, equal native trajectory and equal throughput are distinct claims.',
            'Native stopping stays enabled. Audit wall time is instrumented; clean timing and every other threshold remain paused.','']
    (study / 'summary.md').write_text('\n'.join(lines))


def run(study):
    m=load(study);protocol=read(study/'protocol.json');manifest=read(study/'manifest.json')
    if m.sha(Path(__file__))!=manifest['coordinator_sha256']:raise RuntimeError('Coordinator bytes changed')
    if (study/'final_complete.json').exists():return
    m.verify(study,protocol)
    if m.runtime_identity()!=protocol['runtime']:raise RuntimeError('Runtime identity changed')
    stop=threading.Event();t=threading.Thread(target=monitor,args=(study,stop),daemon=True);t.start()
    try:
        qualify(m,study,protocol)
        m.summarize_arm(study,'longbench_v2','value_v2_online_preserve_mass',TAU)
        publish(m,study)
        m.verify(study,protocol)
        parent=study.parent/manifest['parent_campaign']
        for rel,digest in manifest['reference_artifact_hashes'].items():
            if m.sha(parent/rel)!=digest:raise RuntimeError('Original reference artifact changed')
        m.write(study/'receipts/fairness_audit.json',dict(status='passed',source_protocol_runtime_model_seed_matched=True,
            selector='value_v2_online_preserve_mass',threshold=TAU,local_skipped_tiles=0,
            invalid_rows=0,unexpected_request_graph_captures=0,official_cells=503,
            other_campaigns_unchanged=True,other_thresholds_launched=False,clean_timing='paused'),True)
        curve=read(study/'curve.json');curve['status']='complete';m.write(study/'curve.json',curve)
        m.write(study/'state.json',dict(status='complete',updated_unix=time.time()))
        m.write(study/'final_complete.json',dict(status='complete',scope=manifest['authorized_scope'],records=503,
            source_commit=protocol['source_commit'],protocol_sha256=m.sha(study/'protocol.json'),
            other_thresholds='paused',timing='paused'),True)
        print(json.dumps(dict(event='complete',study=study.name)),flush=True)
    finally:stop.set();t.join(timeout=5)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,default=Path.cwd()/'results'/NAME)
    ap.add_argument('--stage',choices=['prepare','run'],default='run');args=ap.parse_args();study=args.study.resolve()
    if args.stage=='prepare':prepare(study,study.parent/PARENT)
    else:run(study)


if __name__=='__main__':main()
