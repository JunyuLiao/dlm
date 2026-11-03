"""One immutable, count-calibrated HF model/benchmark attempt."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/multi_model_value_20261030/attempt001'
ARM='value_v2_current_tile_init_gaussian32_reuse_r4'
OLD='value_v2_gaussian32_reuse_r4'
PINS={'llada22':'224f8f11173bd16e5cf10054a963c74233758cf2','trado':'8c0d14b031636ddcc708df75fa1d57aefa8b0ba4'}
INITIAL={('llada22','humaneval'):.2,('trado','humaneval'):.17782794100389226,
         ('llada22','aime26'):.012489663701836472,('trado','aime26'):.025109133251762627,
         ('llada22','ruler128k'):.00007498942093324559,('trado','ruler32k'):.0003}


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def append(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(v)+'\n');f.flush();os.fsync(f.fileno())
def pool(rs):
    e=sum(r['counts']['eligible_tiles'] for r in rs);k=sum(r['counts']['kept_tiles'] for r in rs)
    return {'requests':len(rs),'eligible_tiles':e,'kept_tiles':k,'sparsity':1-k/e}


def calibration_sample(root,model,suite,rows):
    if suite.startswith('ruler'):return read(root/'private/data'/model/'calibration_ruler.json'),142,False
    # Fixed prompt-length strata, independent of scores and generated text.
    if suite=='aime26':ordered=sorted(rows,key=lambda r:(len(r['prompt']),r['id']))
    else:ordered=sorted(rows,key=lambda r:(len(r['prompt']),r['id']))
    selected=[ordered[(2*i+1)*len(ordered)//12] for i in range(6)]
    return selected,42,True


def modules(root,model):
    snapshot=BASE/'private/execution'/PINS[model]
    package=types.ModuleType('experiments');package.__path__=[str(snapshot/'experiments')]
    sys.modules['experiments']=package
    sys.path[:0]=[str(BASE/'private/compat'/model),str(snapshot),str(snapshot/'scripts')]
    attention=importlib.import_module('experiments.numerical_qk_reuse.multi_model_attention')
    worker=importlib.import_module('multi_model_worker')
    for mod in (attention,sys.modules['experiments.numerical_qk_reuse.multi_model_runtime']):
        assert Path(mod.__file__).resolve().is_relative_to(snapshot.resolve())
    path=ROOT/'experiments/numerical_qk_reuse/v2_current_tile_init.py'
    spec=importlib.util.spec_from_file_location('experiments.numerical_qk_reuse.v2_current_tile_init',path)
    variant=importlib.util.module_from_spec(spec);spec.loader.exec_module(variant)
    return attention,worker,variant,snapshot


def run(root,model,suite,attempt):
    import torch
    leaf=root/'panels'/model/suite/f'attempt{attempt:03}'
    assert not leaf.exists();leaf.mkdir(parents=True)
    os.environ.update(HF_MODULES_CACHE=str(root/'private/hf_modules'),TRITON_CACHE_DIR=str(root/'private/triton_cache'))
    data=root/'private/data'/model/f'{suite}.json';rows=read(data)
    cells=[(r,s) for r in rows for s in r['seeds']]
    assert len(cells)==(90 if suite=='aime26' else 164 if suite=='humaneval' else 130)
    development,cal_seed,transductive=calibration_sample(root,model,suite,rows)
    cap=32768 if model=='llada22' and suite=='aime26' else 2000 if model=='trado' and suite in ('aime26','humaneval') else 1024 if suite=='humaneval' else 128
    attention,worker,variant,snapshot=modules(root,model)
    identity=read(BASE/f'{model}_model_identity.json')
    protocol={'model':model,'suite':suite,'arm':ARM,'attempt':attempt,'generation_cap':cap,
        'model_revision':identity['revision'],'manifest_sha256':sha(data),'coordinator_protocol_sha256':sha(root/'protocol.json'),
        'source_sha256':{'worker':sha(__file__),'selector':sha(ROOT/'experiments/numerical_qk_reuse/v2_current_tile_init.py'),
                         'runtime':sha(Path(sys.modules['experiments.numerical_qk_reuse.multi_model_runtime'].__file__)),
                         'attention':sha(Path(attention.__file__)),'base_worker':sha(Path(worker.__file__))},
        'calibration_ids':[r['id'] for r in development],'calibration_seed':cal_seed,'transductive_count_only':transductive,
        'target':[.48,.52],'final_acceptable':[.4,.6],'native_sampler_unchanged':True,'max_candidates':10,
        'numerical_snapshot':PINS[model],'q_tile':128,'kv_tile':64,'reuse':4,'all_denoising_layers':True}
    write(leaf/'protocol.json',protocol)
    h=worker.ModelHarness(root,model).load()
    controls=[]
    for arm in (OLD,'allkept_fa4'):
        saved={r['id']:r for p in (BASE/'runs'/model/'humaneval'/arm).glob('shard*/attempt001/records.jsonl') for r in map(json.loads,p.read_text().splitlines())}
        row=read(root/'private/data'/model/'humaneval.json')[0]
        h.runtime.arm=arm;h.runtime.threshold=.8944271909999159 if model=='llada22' else .17782794100389226
        public,_=worker.run_cell(h,row,1024 if model=='llada22' else 2000,42)
        for key in ('output_sha256','counts','output_tokens'):assert public[key]==saved[row['id']][key],('control',model,arm,key)
        controls.append({'arm':arm,'id':row['id'],'exact_output_and_counts':True})
    write(leaf/'control_receipt.json',{'status':'pass','cases':controls})
    def selector(*a,**kw):
        h.runtime.counts['current_tile_init_selection_layer_calls']+=1
        return variant.current_tile_cuda(*a,**kw)
    attention.online_cuda=selector;h.runtime.arm=OLD
    lo,hi=1e-8,4.;tau=INITIAL[model,suite]
    previous=None
    if attempt>1:
        previous=read(leaf.parent/f'attempt{attempt-1:03}'/'generation_complete.json')
        if previous['sparsity']>.6:hi=previous['threshold'];tau=hi/2
        else:lo=previous['threshold'];hi=max(4.,lo*16);tau=lo*2
    trials=[]
    for candidate in range(1,11):
        h.runtime.threshold=tau;observed=[]
        for row in development:
            item={k:row[k] for k in ('id','prompt','answer_prefix') if k in row}
            pub,_=worker.run_cell(h,item,cap,cal_seed)
            pub.update(threshold=tau,candidate=candidate)
            append(leaf/'calibration/records.jsonl',pub);observed.append(pub)
        point={'threshold':tau,**pool(observed),'candidate':candidate};trials.append(point)
        print(json.dumps({'event':'calibration',**point}),flush=True)
        if .48<=point['sparsity']<=.52:break
        if point['sparsity']<.5:lo=tau
        else:hi=tau
        below=[x for x in trials if x['sparsity']<.5];above=[x for x in trials if x['sparsity']>=.5]
        pairs=[(a,b) for a in below for b in above if a['threshold']<b['threshold']]
        if pairs:
            a,b=min(pairs,key=lambda ab:math.log(ab[1]['threshold']/ab[0]['threshold']))
            frac=min(.8,max(.2,(.5-a['sparsity'])/(b['sparsity']-a['sparsity'])))
            tau=math.exp(math.log(a['threshold'])+frac*math.log(b['threshold']/a['threshold']))
        else:tau=math.sqrt(lo*hi)
    selected=min(trials,key=lambda x:abs(x['sparsity']-.5))
    h.runtime.threshold=selected['threshold']
    write(leaf/'calibration/selected_threshold.json',{'trials':trials,**selected,'accuracy_used':False,'transductive':transductive})
    start=time.time();all_rows=[]
    for shard in range(1,5):
        dest=leaf/'runs'/f'shard{shard:03}'/'attempt001'
        write(dest/'config.json',{**protocol,'threshold':selected['threshold'],'shard':shard,'protocol_sha256':sha(leaf/'protocol.json')})
        shard_start=time.time();part=[]
        for i,(row,seed) in enumerate(cells):
            if i%4!=shard-1:continue
            pub,priv=worker.run_cell(h,row,cap,seed)
            assert pub['counts']['selection_layer_calls']==pub['counts']['current_tile_init_selection_layer_calls']>0
            pub.update(model=model,suite=suite,arm=ARM,threshold=selected['threshold'],model_revision=identity['revision'],
                       protocol_sha256=sha(leaf/'protocol.json'))
            append(dest/'records.jsonl',pub);append(dest/'private/generation.private.jsonl',dict(priv,model=model,suite=suite,arm=ARM))
            part.append(pub);all_rows.append(pub)
            print(json.dumps({'event':'cell_complete','model':model,'suite':suite,'attempt':attempt,'completed':len(all_rows),
                             'expected':len(cells),'id':row['id'],'seed':seed,'N':pub['counts']['denoising_forwards'],'T':pub['output_tokens']}),flush=True)
        write(dest/'receipt.json',{'status':'pass','records_sha256':sha(dest/'records.jsonl'),'protocol_sha256':sha(leaf/'protocol.json'),
                                  'requests':len(part),'worker_seconds':time.time()-shard_start})
        write(dest/'final_complete.json',{'status':'complete','requests':len(part)})
    result={'status':'complete','model':model,'suite':suite,'attempt':attempt,**pool(all_rows),
            'threshold':selected['threshold'],'worker_seconds':time.time()-start,'protocol_sha256':sha(leaf/'protocol.json')}
    result['accepted']=.4<=result['sparsity']<=.6
    write(leaf/'generation_complete.json',result)
    print(json.dumps({'event':'panel_complete',**result}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('model',choices=['llada22','trado']);p.add_argument('suite');p.add_argument('--attempt',type=int,default=1)
    a=p.parse_args();run(a.root.resolve(),a.model,a.suite,a.attempt)
