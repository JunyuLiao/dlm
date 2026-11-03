"""One frozen native DiffusionGemma current-canvas V2 benchmark attempt."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'results/optimized_refresh_benchmarks_20261022/attempt009'
EX=OLD/'private/execution'
MODEL=Path('/media/volume/new-dllm/dlm-migrated-20260921/cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b')
ARM='value_v2_current_canvas_init_gaussian32_progress_reuse'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def lines(p):return list(map(json.loads,p.read_text().splitlines()))
def pool(rs):
    receipts=[r['receipts']['adapter'] for r in rs]
    e=sum(r['global_eligible_tiles'] for r in receipts);k=sum(r['global_kept_tiles'] for r in receipts)
    return {'requests':len(rs),'eligible_tiles':e,'kept_tiles':k,'sparsity':1-k/e}


def prepare_pool(root,leaf,suite,calibration=False):
    if suite!='ruler128k':
        folder=root/'private/pools/diffusiongemma'/suite
        dataset='humaneval' if suite=='humaneval' else 'aime26'
        manifest=read(folder/f'{dataset}_generation_manifest.json')
        return folder,dataset,manifest
    name='calibration_ruler' if calibration else 'ruler128k'
    raw=read(root/'private/data/diffusiongemma'/f'{name}.json')
    dataset='ruler128k_current_tile130'
    folder=leaf/'private'/('calibration_pool' if calibration else 'pool')
    task_budget={}
    phase='calibration_ruler' if calibration else 'ruler'
    for f in (root/'private/ruler_generated/diffusiongemma'/phase).glob('*/validation.jsonl'):
        task_budget[f.parent.name]=int(lines(f)[0]['tokens_to_generate'])
    from scripts import v31_official as op
    manifest=[];rowinfo=[];gold={}
    for i,r in enumerate(raw):
        rid=f'{dataset}/{r["id"]}'
        m={'benchmark':dataset,'generation_budget':task_budget[r['task']],'id':rid,'index':i,
           'prompt':r['prompt'],'prompt_tokens':r['encoded_prompt'],'prompt_token_count':r['prompt_tokens'],
           'source_id':r['id'],'task':r['task'],'task_base':r['task_base'],'thinking':False,
           'answer_prefix':r['answer_prefix'],'prefilled':True,'target_length':131072,
           'response_prefill':op.RESPONSE_OPENER['diffusion_gemma']+r['answer_prefix']}
        manifest.append(m);gold[rid]=r['gold']
        rowinfo.append({'id':rid,'index':i,'generation_budget':m['generation_budget'],
            'prompt_sha256':op.token_ids_sha256(m['prompt_tokens']),'prompt_token_count':m['prompt_token_count'],
            'thinking':False,'task':m['task'],'task_base':m['task_base'],'prefilled':True})
    write(folder/f'{dataset}_generation_manifest.json',manifest)
    write(folder/f'{dataset}_gold.json',gold)
    write(folder/f'{dataset}_rowinfo.json',{'schema':op.ROWINFO_SCHEMA,'dataset':dataset,
        'manifest_sha256':sha(folder/f'{dataset}_generation_manifest.json'),'max_model_len':136192,'rows':rowinfo})
    return folder,dataset,manifest


def run(root,suite,attempt):
    leaf=root/'panels/diffusiongemma'/suite/f'attempt{attempt:03}'
    assert not leaf.exists();leaf.mkdir(parents=True)
    folder,dataset,manifest=prepare_pool(root,leaf,suite)
    cells=[{'dataset':dataset,'id':r['id'],'index':i,'prompt_tokens':r['prompt_tokens'],'seed':s}
           for s in ([42,43,44] if suite=='aime26' else [42]) for i,r in enumerate(manifest)]
    assert len(cells)==(90 if suite=='aime26' else 164 if suite=='humaneval' else 130)
    if suite=='ruler128k':
        cfolder,cdataset,cm=prepare_pool(root,leaf,suite,True)
        calibration=[{'dataset':cdataset,'id':r['id'],'index':i,'prompt_tokens':r['prompt_tokens'],'seed':142} for i,r in enumerate(cm)]
    else:
        cfolder=folder
        ordered=sorted(cells[:len(manifest)],key=lambda c:(len(c['prompt_tokens']),c['index']))
        calibration=[ordered[(2*i+1)*len(ordered)//12] for i in range(6)]
    settings=read(OLD/'protocol.json')['suites'].get(suite)
    if settings is None:settings={'dataset':dataset,'count':130,'seeds':[42],'mage_k':8192,'max_len':136192,'block':32,'chunk':16384,'memory':.85}
    settings=dict(settings,dataset=dataset,count=len(cells))
    source_names=['scripts/current_tile_gemma_panel.py','scripts/current_canvas_gemma_worker.py',
                  'experiments/numerical_qk_reuse/current_canvas_gemma_runtime.py','experiments/numerical_qk_reuse/v2_current_canvas_init.py']
    protocol={'model':'diffusiongemma','suite':suite,'arm':ARM,'attempt':attempt,'settings':settings,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_sha256':{n:sha(ROOT/n) for n in source_names},'frozen_execution_root_identity':sha(OLD/'protocol.json'),
        'manifest_sha256':sha(folder/f'{dataset}_generation_manifest.json'),'calibration_indices':[c['index'] for c in calibration],
        'calibration_transductive':suite!='ruler128k','calibration_count':len(calibration),'target':[.48,.52],'final_acceptance':[.4,.6],
        'scope':'GLOBAL 5/11/17/23/29 only; LOCAL/prefill/commit dense','native_lifecycle':'unchanged select step1, carry first, progress .15, sticky1.386',
        'coordinator_protocol_sha256':sha(root/'protocol.json'),'native_sampler_unchanged':True}
    write(leaf/'protocol.json',protocol)
    (leaf/'private/execution').symlink_to(EX.resolve(),target_is_directory=True)
    from scripts import v31_v2_local_threshold_sweep as inherited
    inherited.SUITES[suite]=settings
    def group(planned,pooldir,tau,dest,control=False):
        dest.mkdir(parents=True,exist_ok=False)
        schedule=dest/'private/cells.json';write(schedule,planned)
        env=inherited.env_for(leaf,suite,'value_v2_online_preserve_mass',tau)
        env.update(STUDY_ARM='allkept_fa4' if control else ARM,VALUE_AUDIT='1',VALUE_CLEAN_TIMING='0',VALUE_BOUNDARY_TIMING='0',
            TRACE='1',REPEATS='1',PYTHONHASHSEED='0',TORCHINDUCTOR_DETERMINISTIC='1',
            FROZEN_SOURCE_SHA256=sha(leaf/'protocol.json'),OPTIMIZED_REFRESH_METHOD='dense' if control else 'v2',
            OPTIMIZED_REFRESH_LIBRARY=str(OLD/'private/kernel/v2_online_sm90.so'),
            OPTIMIZED_REFRESH_OBSERVER_DIR=str(leaf/'private/cache/observer'),
            OPTIMIZED_REFRESH_BASE_WORKER=str(EX/'scripts/v31_humaneval_cfgdet_base_worker.py'),
            OPTIMIZED_KERNEL_INVENTORY=str(dest/'private/kernel_inventory.json'),
            RESIDENT_SPARSE_EXECUTION='1',RESIDENT_SPARSE_GRAPH_BUCKETS='0',RESIDENT_V31_PAGED_REFRESH='0',
            OPTIMIZED_WARM_EACH_INPUT='1',RESIDENT_ALLKEPT_EXECUTION='1',
            CURRENT_TILE_SOURCE_ROOT=str(ROOT),CURRENT_TILE_ORIGINAL_WORKER=str(EX/'scripts/optimized_refresh_worker.py'))
        env.pop('V2_PAGED_REFRESH_LIBRARY',None)
        command=[sys.executable,str(ROOT/'scripts/current_canvas_gemma_worker.py'),str(MODEL),str(pooldir),str(schedule),
                 str(dest/'records.jsonl'),str(dest/'private/generation.private.jsonl'),'allkept' if control else 'mage','PIECEWISE']
        write(dest/'config.json',{'model':'diffusiongemma','suite':suite,'arm':env['STUDY_ARM'],'threshold':tau,
            'protocol_sha256':sha(leaf/'protocol.json'),'requests':len(planned),'cells_sha256':sha(schedule),
            'manifest_sha256':sha(pooldir/f'{planned[0]["dataset"]}_generation_manifest.json')})
        start=time.time()
        with (dest/'private/worker.log').open('w') as log:
            completed=subprocess.run(command,cwd=EX,env=env,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:raise RuntimeError(f'Gemma worker {completed.returncode}: {dest}')
        rs=lines(dest/'records.jsonl');assert len(rs)==len(planned)
        for r in rs:
            a=r['receipts']['adapter'];n=r['denoise_forwards']
            assert a['native_local_skipped_tiles']==0 and a['global_calls']==5*n and a['native_local_calls_audited']==25*n
            assert not a.get('value_nonfinite_logits_calls',0) and not a.get('value_nonfinite_attention_calls',0)
            if not control:
                assert a['current_canvas_initialization'] and a['current_canvas_selector_calls']>0
                assert a['optimized_refresh_extra_model_forwards']==0
            assert r['cuda_graph_captures']==0
        write(dest/'receipt.json',{'status':'pass','requests':len(rs),'worker_seconds':time.time()-start,
                                  'records_sha256':sha(dest/'records.jsonl'),'protocol_sha256':sha(leaf/'protocol.json')})
        write(dest/'final_complete.json',{'status':'complete','requests':len(rs)})
        return rs
    # One saved all-kept control validates this native integration separately
    # from the current-selector synthetic CUDA oracle. It remains diagnostic.
    hefolder=root/'private/pools/diffusiongemma/humaneval';hm=read(hefolder/'humaneval_generation_manifest.json')
    control=[{'dataset':'humaneval','id':hm[0]['id'],'index':0,'prompt_tokens':hm[0]['prompt_tokens'],'seed':42}]
    cr=group(control,hefolder,None,leaf/'qualification/allkept',True)
    reference=[r for f in (OLD/'runs/humaneval').glob('allkept_fa4*/audit/shard*/attempt*/records.jsonl') for r in lines(f) if r['index']==0 and r['panel_seed']==42]
    assert len(reference)==1
    for key in ('output_hash','denoise_forwards','output_tokens'):assert cr[0][key]==reference[0][key],('Gemma control',key)
    write(leaf/'qualification/control_receipt.json',{'status':'pass','output_and_count_match':True})
    tau=.2 if suite=='humaneval' else .03 if suite=='aime26' else .0018
    lo,hi=1e-8,4.
    if attempt>1:
        prev=read(leaf.parent/f'attempt{attempt-1:03}/generation_complete.json')
        if prev['sparsity']>.6:hi=prev['threshold'];tau=hi/2
        else:lo=prev['threshold'];hi=max(4.,lo*16);tau=lo*2
    trials=[]
    for index in range(1,11):
        rs=group(calibration,cfolder,tau,leaf/f'calibration/candidate{index:03}')
        point={'threshold':tau,**pool(rs)};trials.append(point)
        print(json.dumps({'event':'calibration',**point}),flush=True)
        if .48<=point['sparsity']<=.52:break
        if point['sparsity']<.5:lo=tau
        else:hi=tau
        tau=math.sqrt(lo*hi)
    selected=min(trials,key=lambda x:abs(x['sparsity']-.5));tau=selected['threshold']
    write(leaf/'calibration/selected_threshold.json',{'trials':trials,**selected,'accuracy_used':False})
    result=[];start=time.time()
    for shard in range(1,5):
        rs=group([r for i,r in enumerate(cells) if i%4==shard-1],folder,tau,leaf/f'runs/shard{shard:03}/attempt001')
        result+=rs
    summary={'status':'complete','model':'diffusiongemma','suite':suite,'attempt':attempt,**pool(result),
             'threshold':tau,'worker_seconds':time.time()-start,'protocol_sha256':sha(leaf/'protocol.json')}
    summary['accepted']=.4<=summary['sparsity']<=.6
    write(leaf/'generation_complete.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('model');p.add_argument('suite');p.add_argument('--attempt',type=int,default=1)
    a=p.parse_args();run(a.root.resolve(),a.suite,a.attempt)
