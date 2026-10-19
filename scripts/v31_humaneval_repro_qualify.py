"""Resumable baseline qualification; full accuracy remains separately gated."""
import argparse
import json
from pathlib import Path

from scripts.v31_humaneval_repro_study import DEFAULT, ROOT, job, read, write, sha, base

FIELDS = ('output_hash','denoise_forwards','canvases','output_tokens','sampler_calls')
REQUIRED_STATE = {'canvas_sha256','argmax_sha256','step_sha256','encoder_phase_sha256',
    'confident_sha256','sc_sha256','history_sha256','history_len_sha256',
    'valid_canvas_len_sha256','cpu_rng_sha256','cuda_rng_sha256'}


def inspect_job(study,name):
    leaf = study/'diagnosis'/name
    marker = read(leaf/'attempt_complete.json')
    config = read(leaf/'config.json')
    if not marker.get('policy_checks') or not all(marker['policy_checks'].values()):
        raise RuntimeError('ineffective numerical policy')
    for rel,digest in marker['files_sha256'].items():
        if sha(leaf/rel) != digest:
            raise RuntimeError('qualification artifact drift')
    rows = base.jsonl(leaf/'records.jsonl')
    if len(rows) != 26 or config['variants'] != ['clean','observe','clean']:
        raise RuntimeError('qualification schedule mismatch')
    changes = []
    for i in range(8,26,3):
        fields = [f for f in FIELDS if any(rows[i][f] != rows[i+j][f] for j in (1,2))]
        if fields:
            changes.append(dict(index=rows[i]['index'],fields=fields))
    traces = []
    for index,i in enumerate(range(9,26,3)):
        path = leaf/'private/trajectory'/f'request{i+1:03d}.json'
        trace = read(path)
        if (trace['native_sampler_calls'] != rows[i]['sampler_calls'] or
            len(trace['calls']) != rows[i]['sampler_calls'] or trace['extra_forwards'] != 0):
            raise RuntimeError('native trajectory call-count mismatch')
        if not REQUIRED_STATE <= trace['initial'].keys() or not all(
            REQUIRED_STATE <= x['before'].keys() and
            REQUIRED_STATE | {'sampled_sha256','num_sampled_sha256','draft_sha256'} <= x['after'].keys()
            for x in trace['calls']):
            raise RuntimeError('required trajectory field absent')
        traces.append(dict(index=rows[i]['index'],path=path,digest=sha(path)))
    captures = sum(r['cuda_graph_captures'] for r in rows[8:])
    scope_ok = all(r.get('mage_k') == 256 and r.get('local_kv_budget') is None for r in rows[8:]) if config['family']=='v31' else True
    summary = dict(job=name,unique_questions=6,measured_records=18,
        bookend_changes=changes,measured_captures=captures,requested_v31_budget_scope_ok=scope_ok,
        marker_sha256=sha(leaf/'attempt_complete.json'),
        trace_sha256={str(t['index']):t['digest'] for t in traces})
    return rows,traces,summary


def compare(left,right):
    a,ta,sa = left; b,tb,sb = right
    if [(r['index'],r['rng_seed'],r['prompt_sha256']) for r in a] != [
            (r['index'],r['rng_seed'],r['prompt_sha256']) for r in b]:
        raise RuntimeError('qualified request identities or schedules differ')
    diffs = {f:sum(u[f] != v[f] for u,v in zip(a[8:],b[8:])) for f in FIELDS}
    changed = []
    for x,y in zip(ta,tb):
        if x['index'] != y['index']:
            raise RuntimeError('trajectory question mismatch')
        if x['digest'] != y['digest']:
            u,v = read(x['path']),read(y['path'])
            first = next((i for i,(p,q) in enumerate(zip(u['calls'],v['calls'])) if p != q),None)
            changed.append(dict(index=x['index'],initial_changed=u['initial'] != v['initial'],
                calls=[len(u['calls']),len(v['calls'])],first_changed_call=first,
                first_changed_fields=[side+'.'+k for side in ('before','after')
                    for k in u['calls'][first][side] if u['calls'][first][side].get(k) != v['calls'][first][side].get(k)]
                    if first is not None else []))
    return dict(left=sa['job'],right=sb['job'],differences=diffs,trajectory_changes=changed,
                passed=not any(diffs.values()) and not changed)


def qualify(study,out):
    policy = study/'private/source_addenda/attempt027/worker.py'
    observer = study/'private/source_addenda/attempt031/worker.py'
    freeze = dict(protocol_sha256=sha(study/'protocol.json'),policy_sha256=sha(policy),
        observer_sha256=sha(observer),controller_sha256=sha(ROOT/'scripts/v31_humaneval_repro_study.py'),
        source_sha256=sha(Path(__file__)),seed=42,cohort=read(study/'protocol.json')['cohort'],
        per_query_prefix_budget_tokens=256,full_accuracy_released=False)
    if not out.exists():
        out.mkdir(parents=True);write(out/'manifest.json',freeze,True)
    elif read(out/'manifest.json') != freeze:
        raise RuntimeError('new qualification attempt required after source/protocol drift')
    first = read(study/'receipts/first_cfgdet_native_repeat.json')
    if any(first['differences'].values()):
        raise RuntimeError('independent unobserved native repeat failed')
    summaries = []; comparisons = []
    # Dense-on already has an independent same-schedule unobserved clean pair.
    # Other cells obtain independent same-mode repeats in the observer schedule.
    for mode,short in [('thinking_on','on'),('thinking_off','off')]:
        for family,label in [('dense_full','dense'),('v31','v31')]:
            inspected = []
            passes = [('clean','001'),('audit','001')] if short=='on' and label=='dense' else [
                      ('clean','001'),('clean','002'),('audit','001')]
            for pass_name,rep in passes:
                name = f'{label}_{short}_cfgdet_traj_{pass_name}{rep}'
                leaf = study/'diagnosis'/name
                if not leaf.exists():
                    job(study,name,mode,family,pass_name,wrapper=observer,
                        variants=['clean','observe','clean'],overrides=dict(
                            TORCHINDUCTOR_DETERMINISTIC='1',HUMANEVAL_POLICY_REQUIRED='inductor_cfgdet',
                            HUMANEVAL_POLICY_SOURCE=str(policy),HUMANEVAL_POLICY_SOURCE_SHA256=sha(policy),
                            HUMANEVAL_KERNEL_INVENTORY=str(leaf/'private/kernel_inventory.json'),
                            HUMANEVAL_TRAJECTORY_DIR=str(leaf/'private/trajectory')))
                result = inspect_job(study,name); summary = result[2]
                summary_file = out/f'{name}.json'
                if not summary_file.exists():write(summary_file,summary,True)
                elif read(summary_file) != summary:raise RuntimeError('qualification summary drift')
                if summary['bookend_changes'] or summary['measured_captures'] or not summary['requested_v31_budget_scope_ok']:
                    raise RuntimeError('observer/native graph/budget scope qualification failed')
                summaries.append(summary); inspected.append(result)
                if len(inspected) > 1:
                    contrast = compare(inspected[0],result)
                    contrast_file = out/f"compare_{inspected[0][2]['job']}_{name}.json"
                    if not contrast_file.exists():write(contrast_file,contrast,True)
                    elif read(contrast_file) != contrast:raise RuntimeError('qualification contrast drift')
                    comparisons.append(contrast)
                    if not contrast['passed']:
                        raise RuntimeError('native full-state process/pass reproducibility failed')
    write(out/'baseline_runtime_qualified.json',dict(status='baseline_runtime_qualified',
        all_requested_methods_qualified=False,full_runs_qualified=False,manifest_sha256=sha(out/'manifest.json'),
        jobs=summaries,comparisons=comparisons,unobserved_native_repeat=first),True)
    print(json.dumps(dict(status='baseline_runtime_qualified',engines=len(summaries),full_runs_qualified=False)))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,default=DEFAULT)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();qualify(a.study,a.out)
