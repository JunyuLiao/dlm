"""Prespecified historical aligned-norm control, queued after the frozen main run.

No main worker, source, configuration or output is modified. Module patches
below adapt existing dispatch/reporting in THIS separate process only; the
actual aligned attention implementation remains the historical GPU baseline.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch

from dllm.models import create_adapter
from experiments.diffusion_gemma_value_aware.operators import Config as LegacyConfig, block_state, value_summaries, screen_risks, streaming_mask
from experiments.diffusion_gemma_value_aware.protocol import frozen_write, sha
from experiments.diffusion_gemma_value_aware_followup.evidence import check_sources, pair
from experiments.diffusion_gemma_value_aware_followup.engine import check_result
from experiments.diffusion_gemma_value_aware_followup.run import gpu_idle
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _append, _fingerprint
from . import protocol, calibration, workflow, shared_analysis, report
from .runner import cached, load_output, write_output

ROOT = protocol.ROOT/'aligned_addendum'
CONFIG = dict(method='aligned')


def identity(root=ROOT):
    setup = protocol.prepare(protocol.ROOT)
    parent = protocol.execution(protocol.ROOT)
    data = dict(parent_fingerprint=parent['fingerprint'], config=CONFIG, targets=[.5,.75],
        sample_ids=[r['id'] for r in setup['final']],
        protocol='Same frozen prompts/seeds/budgets; AIME2048, NeMo LongBench4096;128x64, prefix+canvas',
        purpose='Include the historical token-aligned value-norm control without altering the ongoing37-condition study',
        selection='Added by historical operator inventory before seeing any JL final scores, not chosen by accuracy',
        sources={**parent['sources'], str(Path(__file__)):sha(Path(__file__).read_bytes()),
            str(protocol.ROOT/'setup.json'):sha((protocol.ROOT/'setup.json').read_bytes())})
    data['fingerprint'] = _fingerprint(data)
    frozen_write(root/'execution_contract.json', data)
    return setup, data


def prerequisites():
    root = protocol.ROOT
    terminal = root/'supervisor_terminal.json'
    if not terminal.exists(): return False
    if json.loads(terminal.read_text())['exit_code'] != 0:
        raise RuntimeError('Primary worker failed; addendum remains unstarted, preserving its evidence')
    audit = json.loads((root/'audit.json').read_text())
    proof = json.loads((root/'regeneration_verification.json').read_text())
    if not audit['complete'] or not proof['passed'] or proof['audit_sha256'] != sha((root/'audit.json').read_bytes()):
        raise RuntimeError('Primary final audit/report must finish before the addendum')
    for name,digest in audit['artifacts'].items():
        if sha((root/name).read_bytes()) != digest: raise ValueError('Primary completed artifact changed')
    return True


def import_dense(root, row, stage, contract):
    source = shard_path(protocol.ROOT, stage, 'dense', row['id'])
    out = load_output(source)
    check_result(out,row,contract['parent_fingerprint'],{},None)
    path = shard_path(root,stage,'dense',row['id'])
    if path.exists(): return cached(None,root,row,stage,'dense','dense',{},None,contract)
    out = dict(out,fingerprint=contract['fingerprint'],imported_source=dict(
        path=str(source),sha256=sha(source.read_bytes()),original_fingerprint=contract['parent_fingerprint']))
    write_output(path,out)
    return out


def screen(root, sources, contract):
    """Reuse native aligned-logit risk and exactly the main calibration states."""
    index=[]
    for source in sources:
        path=Path(source['path'])
        if source['split']!='calibration' or sha(path.read_bytes())!=source['sha256']:
            raise ValueError('Invalid aligned calibration state')
        dest=root/'shared_screen'/(path.stem+'.npz'); proof=dest.with_suffix('.json')
        ident=dict(fingerprint=contract['fingerprint'],source=source)
        if proof.exists():
            item=json.loads(proof.read_text())
            if item['identity']!=ident or sha(dest.read_bytes())!=item['arrays_sha256']:
                raise ValueError('Aligned proposal cache changed')
            index.append(item);continue
        data=torch.load(path,map_location='cuda',weights_only=True)
        state=block_state(data['scores'],data['valid'],data['values'])
        meta=value_summaries(data['values'],data['kv_valid'])
        risks,_=screen_risks(state,meta,LegacyConfig(**CONFIG))
        array=risks[0,0][state['eligible'][0,0]].float().cpu().numpy()
        dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_suffix('.tmp')
        with temp.open('wb') as f: np.savez_compressed(f,aligned=array)
        temp.replace(dest)
        item=dict(identity=ident,path=str(dest),arrays_sha256=sha(dest.read_bytes()),
            benchmark=source['id'].split('/')[0],attention_type=source['attention_type'],count=len(array))
        _write(proof,item);index.append(item)
    frozen_write(root/'shared_screen_index.json',index)


def smoke(adapter, root, setup, contract):
    dest=root/'smoke.json'
    if dest.exists():
        out=json.loads(dest.read_text());check_sources(out['sources']);return out
    primary=protocol.ROOT/'validation'/contract['parent_fingerprint']
    validation=json.loads((primary/'smoke.json').read_text())
    native_folder=Path(validation['imported_validation']['path']).parent if validation.get('imported_validation') else primary
    cases=[];sources={}
    examples=[next(r for r in setup['calibration'] if r['benchmark']=='aime26'),
        max(setup['development'],key=lambda r:len(r['prompt_tokens']))]
    for original in examples:
        row=dict(original,generation_budget=16)
        p=native_folder/(sha(row['id'])+'.native.json');native=json.loads(p.read_text())
        sources[str(p)]=sha(p.read_bytes())
        for logt in (-1000.,-2.):
            policy={k:dict(log_threshold=logt) for k in ('local','global')}
            tag='unpruned' if logt==-1000. else 'pruned'
            out=cached(adapter,root,row,'validation',tag,'aligned',CONFIG,policy,contract,validate=True)
            if tag=='unpruned' and (out['completion_tokens']!=native['completion_tokens'] or any(r['skipped'] for r in out['records'])):
                raise AssertionError('Aligned unpruned/native dense parity failed')
            for key in ('native_canvas_length','thinking','sampling','denoising_configuration'):
                if out['generation_metadata'].get(key)!=native['generation_metadata'].get(key):
                    raise AssertionError('Aligned decoding settings changed')
            cases.append(dict(id=row['id'],case=tag,passed=True,checked_calls=len(out['kernel_validation'])))
            file=shard_path(root,'validation',tag,row['id'])
            sources[str(file)]=sha(file.read_bytes())
    out=dict(passed=True,fingerprint=contract['fingerprint'],sources=sources,cases=cases)
    frozen_write(dest,out);return out


def work(root=ROOT):
    if not prerequisites(): raise RuntimeError('Primary study is still running')
    setup,contract=identity(root);check_sources(contract['sources'])
    gpu_idle();torch.backends.cuda.matmul.allow_tf32=False
    adapter=create_adapter('diffusion_gemma',protocol.MODEL,device='cuda',precision='bfloat16',revision=protocol.REVISION).load()
    smoke(adapter,root,setup,contract)
    for row in setup['final']: import_dense(root,row,'dense',contract)
    states=json.loads((protocol.ROOT/'shared_state_index.json').read_text())
    frozen_write(root/'shared_state_index.json',states)
    screen(root,states,contract)
    # Reuse the existing local/global fit/search, including imported AIME policy.
    with patch.object(calibration,'BASELINES',dict(aligned=CONFIG)):
        for target in (.5,.75):
            for benchmark in ('aime26','longbench_v2'):
                try: calibration.fit_one(adapter,root,setup,contract,benchmark,'aligned',CONFIG,target)
                except Exception: workflow.failure(root,'calibration',benchmark=benchmark,target=target)
    conditions={'dense':dict(name='dense',config={},target=0.,thresholds={b:None for b in ('aime26','longbench_v2')})}
    for target in (.5,.75):
        label=f'aligned_s{int(100*target)}';thresholds={};sources=dict(contract['sources'])
        for b in ('aime26','longbench_v2'):
            path=root/'policies'/b/f'{label}.json'
            if not path.exists():continue
            p=json.loads(path.read_text());calibration.audit_policy(root,p,setup,contract)
            thresholds[b]=p['policy'];sources[str(path)]=sha(path.read_bytes())
        c=dict(name='aligned',config=CONFIG,target=target,thresholds=thresholds,sources=sources,fingerprint=contract['fingerprint'])
        frozen_write(root/'final_configs'/f'{label}.json',c);conditions[label]=c
    # Import matching dense development outputs; never run dense a second time.
    dev_ids=json.loads((protocol.ROOT/'development_policy.json').read_text())['seed_check_ids']
    for row in setup['calibration']+setup['development']:
        if row['id'] in dev_ids: import_dense(root,row,'development',contract)
    workflow.development(adapter,root,setup,contract,conditions)
    # Frozen native reference mask, on the same retained supports as this control.
    def aligned_mask(state,values,kv_valid,name,config,policy,native_state=None):
        if 'log_scale' in policy: raise ValueError('Aligned uses scalar local/global thresholds')
        return streaming_mask(native_state,value_summaries(values,kv_valid),LegacyConfig(**CONFIG,log_threshold=policy['log_threshold']))
    try:
        with patch.object(shared_analysis,'BASELINES',dict(aligned=CONFIG)),patch.object(shared_analysis,'PROJECTED',{}),patch.object(shared_analysis,'baseline_mask',aligned_mask):
            shared_analysis.analyze(root,states,contract)
    except Exception: workflow.failure(root,'shared_diagnostics')
    for label,c in conditions.items():
        if label=='dense':continue
        for row in sorted(setup['final'],key=lambda r:(workflow.tokens(r),r['id'])):
            if row['benchmark'] not in c['thresholds']:continue
            try:
                threshold=c['thresholds'][row['benchmark']]
                from .runner import import_aime
                if import_aime(root,row,label,'aligned',CONFIG,threshold,contract) is None:
                    cached(adapter,root,row,'final',label,'aligned',CONFIG,threshold,contract)
            except Exception: workflow.failure(root,'final',condition=label,id=row['id'])
    del adapter;torch.cuda.empty_cache()
    audit=regenerate(root)
    if audit['complete']:
        if regenerate(root)!=audit:raise ValueError('Addendum raw-only regeneration changed artifacts')
        consolidate(root)


def regenerate(root=ROOT):
    setup,contract=identity(root);check_sources(contract['sources'])
    rows=[];raw=[];missing=[];sources=dict(contract['sources']);policies=[];completed=0
    for example in setup['final']:
        outputs={}
        for label in ('dense','aligned_s50','aligned_s75'):
            c={} if label=='dense' else json.loads((root/'final_configs'/f'{label}.json').read_text())
            try:
                policy=None if label=='dense' else c['thresholds'][example['benchmark']]
                out=cached(None,root,example,'dense' if label=='dense' else 'final',label,'dense' if label=='dense' else 'aligned',{} if label=='dense' else CONFIG,policy,contract)
                file=shard_path(root,'dense' if label=='dense' else 'final',label,example['id'])
                sources[str(file)]=sha(file.read_bytes());sources[out['records_source']['path']]=out['records_source']['sha256']
                outputs[label]=out;completed+=1
            except Exception as error:missing.append(dict(id=example['id'],condition=label,error=str(error)))
        if 'dense' not in outputs:continue
        pairs,_=report.nemo_pairs(example,outputs) if example['benchmark']=='longbench_v2' else ([dict(condition=n,**pair(example,o,outputs['dense'])) for n,o in outputs.items()],[])
        raw.extend(dict(p,prediction=outputs[p['condition']]['prediction'],completion_tokens=outputs[p['condition']]['completion_tokens']) for p in pairs)
    groups=defaultdict(list)
    for row in raw:
        groups[row['benchmark'],'full',row['condition']].append(row)
        if row['benchmark']=='aime26':groups['aime26','calibration6' if row['calibration'] else 'noncalibration24',row['condition']].append(row)
    for (b,split,label),g in sorted(groups.items()):
        count=(30 if b=='aime26' else 50) if split=='full' else 6 if split=='calibration6' else 24
        if len(g)!=count:continue
        thresholds=None
        if label!='dense':
            path=root/'policies'/b/f'{label}.json';p=json.loads(path.read_text())
            calibration.audit_policy(root,p,setup,contract);thresholds=p['policy'];sources[str(path)]=sha(path.read_bytes())
            if split=='full':policies.append(p)
        rows.append(dict(benchmark=b,split=split,condition=label,name='dense' if label=='dense' else 'aligned',
            target=0. if label=='dense' else int(label[-2:])/100,thresholds=thresholds,
            projection_family=None,projection_rank=None,projection_seed=None,**report.summarize(g)))
    # Reuse supporting audits, but this addendum has one historical control and
    # its own four-case baseline proof instead of twelve projected candidates.
    development=json.loads((root/'development_policy.json').read_text())
    if development['failures']:missing.append(dict(stage='development',failures=development['failures']))
    diagnostics,extra=report.diagnostic_summary(root);sources.update(extra)
    expected=len(shared_analysis.selected_sources(json.loads((root/'shared_state_index.json').read_text())))*2
    actual=len(json.loads((root/'shared_diagnostics_index.json').read_text())) if (root/'shared_diagnostics_index.json').exists() else 0
    if actual!=expected:missing.append(dict(stage='shared_diagnostics',completed=actual,expected=expected))
    proof=json.loads((root/'smoke.json').read_text());check_sources(proof['sources'])
    if not proof['passed'] or len(proof['cases'])!=4:missing.append(dict(stage='smoke'))
    for file in (root/'smoke.json',root/'development_policy.json'):
        sources[str(file)]=sha(file.read_bytes())
    for name,data in (('summary',rows),('per_sample',raw),('thresholds',policies),('shared_diagnostics_summary',diagnostics)):_write(root/f'{name}.json',data)
    report.csv_write(root/'summary.csv',rows)
    audit=dict(complete=completed==240 and not missing,completed=completed,expected=240,missing=missing,sources=sources,
        inference_performed=False,primary_dense_reused=True,scope='Dense aliases plus the two aligned controls; not240 new inferences')
    audit['artifacts']={n:sha((root/n).read_bytes()) for n in ('summary.json','summary.csv','per_sample.json','thresholds.json','shared_diagnostics_summary.json')}
    _write(root/'audit.json',audit);return audit


def consolidate(root=ROOT):
    primary=protocol.ROOT;dest=primary/'consolidated';dest.mkdir(exist_ok=True)
    audits=[json.loads((p/'audit.json').read_text()) for p in (primary,root)]
    if not all(a['complete'] for a in audits):raise ValueError('Both complete audits required to consolidate')
    for folder,audit in zip((primary,root),audits):
        for name,digest in audit['artifacts'].items():
            if sha((folder/name).read_bytes())!=digest:raise ValueError('Completed consolidation input changed')
    rows=json.loads((primary/'summary.json').read_text())+[r for r in json.loads((root/'summary.json').read_text()) if r['name']!='dense']
    raw=json.loads((primary/'per_sample.json').read_text())+[r for r in json.loads((root/'per_sample.json').read_text()) if r['condition']!='dense']
    policies=json.loads((primary/'thresholds.json').read_text())+json.loads((root/'thresholds.json').read_text())
    if len(raw)!=3120 or len({(r['condition'],r['id']) for r in raw})!=3120:raise ValueError('Consolidated39x80 matrix incomplete')
    _write(dest/'summary.json',rows);report.csv_write(dest/'summary.csv',rows)
    _write(dest/'per_sample.json',raw);_write(dest/'thresholds.json',policies)
    groups=defaultdict(dict)
    for row in raw:
        groups[row['benchmark'],'full',row['condition']][row['id']]=row['accuracy']
        if row['benchmark']=='aime26':groups['aime26','calibration6' if row['calibration'] else 'noncalibration24',row['condition']][row['id']]=row['accuracy']
    compared=json.loads((primary/'comparisons.json').read_text())
    for row in rows:
        if row['name'] not in report.PROJECTED:continue
        pool=[r for r in rows if (r['benchmark'],r['split'],r['name'])==(row['benchmark'],row['split'],'aligned')]
        if not pool:continue
        ref=min(pool,key=lambda r:abs(r['overall_physical_sparsity']-row['overall_physical_sparsity']))
        a=groups[row['benchmark'],row['split'],row['condition']];b=groups[ref['benchmark'],ref['split'],ref['condition']]
        if set(a)!=set(b):raise ValueError('Aligned comparison has unpaired samples')
        gaps={k:row[f'{k}_physical_sparsity']-ref[f'{k}_physical_sparsity'] for k in ('overall','global','local')}
        compared.append(dict(benchmark=row['benchmark'],split=row['split'],candidate=row['condition'],reference=ref['condition'],
            accuracy_delta=row['accuracy']-ref['accuracy'],paired_ci95=report.paired_bootstrap_ci([a[i]-b[i] for i in sorted(a)]),
            **{f'{k}_sparsity_gap':v for k,v in gaps.items()},within3pp_overall=abs(gaps['overall'])<=.03,
            within3pp_all_types=all(abs(v)<=.03 for v in gaps.values()),interpretation='Nearest observed point; no interpolation or multiplicity correction'))
    _write(dest/'comparisons.json',compared);report.csv_write(dest/'comparisons.csv',compared)
    figures=report.plots(dest,rows)
    table_rows=[dict(benchmark=r['benchmark'],condition=r['condition'],correct=f"{r['correct']:g}/{r['count']}",
        target=r['target'],actual=r['overall_physical_sparsity'],global_sparsity=r['global_physical_sparsity'],
        local_sparsity=r['local_physical_sparsity'],mass=r['overall_mass'],agreement=r['token_agreement'],
        delta=r['delta'],paired_ci95=r['paired_ci95']) for r in rows if r['split']=='full']
    text='# Consolidated JL comparison including token-aligned value norm\n\n'
    text+='39 conditions ×80 preserved examples =3120 unique final results. No main workers, configurations or raw outputs were changed. The aligned control was prespecified from the historical operator inventory before any JL final scores were inspected. NeMo LongBench uses4096 output tokens; AIME retains2048. All other setup, exposure/calibration overlap, normalization and no-speedup-claim limitations remain in the [primary report](../report.md).\n\n'
    text+='Aligned routing uses the historical max over token-level(score +log(value norm /valid-KV RMS)), compared with the retained running score maximum, with the same strict all-query physical gate. It does not use value direction. The calibration CDF uses the primary shared calibration states and the same local/global scalar verification. Compatible AIME policies/results and all dense outputs are imported, never rerun.\n\n'
    text+=report.table(table_rows,list(table_rows[0]))+'\n\n'
    text+='Detailed raw outputs, thresholds and shared-QKV diagnostics remain in the primary bundle and `../aligned_addendum/`; the primary report contains the six JL variants, centering/cancellation controls and exploratory matched-actual comparisons. Neither improved local operator error nor nominal target matching establishes a downstream win.\n'
    text+='\n`comparisons.csv` additionally pairs every projected control with the nearest measured aligned point and reports overall/local/global budget gaps. All confidence intervals are exploratory, not multiplicity corrected.\n\n'
    for figure in figures:text+=f'![{Path(figure).stem}]({figure})\n\n'
    (dest/'report.md').write_text(text)
    sources={str(p/'audit.json'):sha((p/'audit.json').read_bytes()) for p in (primary,root)}
    _write(dest/'audit.json',dict(complete=True,completed=3120,expected=3120,sources=sources,
        artifacts={n:sha((dest/n).read_bytes()) for n in ('summary.json','summary.csv','per_sample.json','thresholds.json','comparisons.json','comparisons.csv','report.md',*figures)}))


def queue(root=ROOT):
    with (root/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not prerequisites():
            _append(root/'monitor.jsonl',dict(time=time.time(),stage='waiting_for_primary_final_audit',gpu_work_started=False))
            time.sleep(900)
        gpu_idle()
        with (root/'run.log').open('a',buffering=1) as log:
            child=subprocess.Popen([sys.executable,'-u','-m',__package__+'.aligned_addendum','work','--output',str(root)],stdout=log,stderr=subprocess.STDOUT)
            _write(root/'job.json',dict(pid=child.pid,supervisor_pid=os.getpid(),started=time.time()))
            while True:
                try:code=child.wait(timeout=900);break
                except subprocess.TimeoutExpired:
                    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
                    _append(root/'monitor.jsonl',dict(time=time.time(),stage='running',pid=child.pid,gpu=gpu.stdout.strip()))
        _write(root/'supervisor_terminal.json',dict(exit_code=code,finished=time.time()))


def launch(root=ROOT):
    identity(root)
    with (root/'supervisor.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-u','-m',__package__+'.aligned_addendum','queue','--output',str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(queued_supervisor_pid=child.pid,after='primary final audit and report',gpu_work_started=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('launch','queue','work','regenerate','consolidate'))
    parser.add_argument('--output',type=Path,default=ROOT);args=parser.parse_args()
    globals()[args.command](args.output)
