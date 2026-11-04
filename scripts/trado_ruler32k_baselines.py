"""Source-bound native dense/all-kept/V31/V3a completion for TraDo RULER32K."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import types

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/multi_model_value_20261030/attempt001'
PREVIOUS=ROOT/'results/current_tile_multimodel_20261103/attempt001'
PIN='8c0d14b031636ddcc708df75fa1d57aefa8b0ba4'
ARMS=('dense_native','allkept_fa4','mass_topk_reuse_r4','value_v3a_singleton_reuse_r4')


def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(map(json.loads,Path(p).read_text().splitlines()))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def append(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(v)+'\n');f.flush();os.fsync(f.fileno())


def source_files():
    src=BASE/'private/execution'/PIN
    return {'runner':Path(__file__),**{name:src/'experiments/numerical_qk_reuse'/f'{name}.py' for name in
        ('multi_model_attention','multi_model_runtime','v31_value_kernels','v31_value_selectors','v31_value_summary','v27_fa4')},
        'worker':src/'scripts/multi_model_worker.py',
        'manifest':PREVIOUS/'private/data/trado/ruler32k.json',
        'model_identity':BASE/'trado_model_identity.json',
        'reference_summary':PREVIOUS/'analysis001/summary.json',
        'reference_scores':PREVIOUS/'analysis001/scored_records.jsonl',
        'reference_completion':PREVIOUS/'final_complete.json',
        'reference_protocol':PREVIOUS/'panels/trado/ruler32k/attempt001/protocol.json',
        'reference_marker':PREVIOUS/'panels/trado/ruler32k/attempt001/generation_complete.json'}


def freeze(root):
    assert not root.exists();root.mkdir(parents=True)
    data=read(source_files()['manifest']);assert len(data)==130
    assert Counter(r['task'] for r in data)=={t:10 for t in {r['task'] for r in data}} and len({r['task'] for r in data})==13
    assert all(r['seeds']==[42] and r['prompt_tokens']+128<=32768 for r in data)
    write(root/'protocol.json',{'status':'frozen','parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'model':'Gen-Verse/TraDo-8B-Instruct','model_revision':read(BASE/'trado_model_identity.json')['revision'],
        'numerical_source_commit':PIN,'source_sha256':{k:sha(v) for k,v in source_files().items()},
        'arms':list(ARMS),'benchmark':{'suite':'ruler32k','problems':130,'tasks':13,'per_task':10,'seed':42,'generation_cap':128,
                                     'prompt_min':min(r['prompt_tokens'] for r in data),'prompt_max':max(r['prompt_tokens'] for r in data)},
        'sampler':{'block_length':4,'denoising_steps':4,'temperature':1.,'top_k':0,'top_p':1.,'confidence_threshold':.9,'native_eos':True},
        'attention':{'query_tile_rows':128,'kv_tile_tokens':64,'layers':list(range(36)),'q_heads':32,'kv_heads':8,'head_dim':128,
            'reuse_interval_calls':4,'cross_canvas_carry':False,'dense_prefill_commit':True,'projection_rank':32,'projection_seed':'1729+layer'},
        'fixed_budget':'max(round(J/2),mandatory_count), ties-to-even, mandatory current tile included; per head/Q block',
        'sparsity_target':[.48,.52],'calibration':'none: exact geometry-dependent half-budget for V31/V3a',
        'v2_policy':'no V2 generation; reuse completed current-tile V2 scores and work,130same prompts seed42',
        'dense_consumers':{'dense_native':'native SDPA','allkept_fa4':'existing FA4 full-support consumer'},
        'qualification':'CPU budget/scoring tests; independent CPU selector and FP32 masked-attention CUDA checks; exact historical allkept control',
        'monitor_seconds':900,'gpu_hour_limit':None,'timing':'instrumented accounting only',
        'scoring':'official pinned RULER fractional sample credit; per-task scores and paired problem bootstrap',
        'preserve':'all prior campaign evidence read-only; failed/relaunched workers receive new attempts'})


def modules():
    snapshot=BASE/'private/execution'/PIN
    pkg=types.ModuleType('experiments');pkg.__path__=[str(snapshot/'experiments')];sys.modules['experiments']=pkg
    sys.path[:0]=[str(BASE/'private/compat/trado'),str(snapshot),str(snapshot/'scripts')]
    attention=importlib.import_module('experiments.numerical_qk_reuse.multi_model_attention')
    worker=importlib.import_module('multi_model_worker')
    for mod in (attention,sys.modules['experiments.numerical_qk_reuse.multi_model_runtime']):
        assert Path(mod.__file__).resolve().is_relative_to(snapshot.resolve())
    return attention,worker


def qualify(attention,root):
    import torch
    from experiments.numerical_qk_reuse.v31_value_kernels import statistics_cuda
    from experiments.numerical_qk_reuse.v31_value_selectors import statistics_reference,full_support
    gen=torch.Generator(device='cuda').manual_seed(20261104)
    cases=[]
    for nk in (224,511,1024,32476):
        q=torch.randn((1,4,4,128),device='cuda',dtype=torch.bfloat16,generator=gen)
        k=torch.randn((1,1,nk,128),device='cuda',dtype=torch.bfloat16,generator=gen)
        v=torch.randn((1,1,nk,128),device='cuda',dtype=torch.bfloat16,generator=gen)
        matrix=torch.randn((1,128,32),device='cuda',generator=gen)/32**.5
        z=(v[0].float()@matrix).unsqueeze(0);nu=v[0].float().square().sum(-1).mean(-1).sqrt()
        jt=(nk+63)//64;mandatory=torch.zeros((4,jt),device='cuda',dtype=torch.bool);mandatory[:,-1]=True
        stats=statistics_cuda(q,k,z,nu,128**-.5,jt)
        scores=(q[0].cpu().float()@k[0,0].cpu().float().T)*128**-.5
        cpu=statistics_reference(scores,z[0].cpu(),nu.cpu(),jt)
        for arm in ARMS[2:]:
            if arm==ARMS[2]:
                lm=stats.log_mass;denom=torch.logsumexp(lm,1,keepdim=True)
                ranking=(lm-torch.where(torch.isfinite(denom),denom,0.)).amax(-1)
                cx=cpu.log_mass;cd=torch.logsumexp(cx,1,keepdim=True)
                cs=(cx-torch.where(torch.isfinite(cd),cd,0.)).amax(-1)
            else:
                ranking=attention.singleton_scores(stats)
                alpha,contrib,output,g=full_support(cpu)
                remaining=alpha.sum(1)[:,None]-alpha
                error=(g.sum(1)[:,None]-g).norm(dim=-1)/(remaining.clamp_min(torch.finfo(alpha.dtype).tiny)*cpu.nu[:,None].clamp_min(1e-12))
                cs=torch.where(cpu.valid,error,0.).amax(-1)
                cs.masked_fill_((cpu.valid&(remaining<=0)).any(-1),float('inf'))
            keep=attention.budget_map(ranking,mandatory,stats.valid.any(-1))
            expected=torch.zeros((4,jt),dtype=torch.bool)
            target=max(1,round(jt/2));cs[:,-1]=float('inf')
            for h in range(4):expected[h,torch.argsort(cs[h],descending=True,stable=True)[:target]]=True
            assert torch.equal(keep.cpu(),expected),(nk,arm,'CPU mask')
            output=attention.v27_fa4.sparse_lists(q,k,v,attention.v27_fa4.block_sparse_tensors(keep.reshape(1,4,1,jt)),128**-.5)
            mask=keep.repeat_interleave(64,-1)[:,:nk]
            p=((q[0].float()@k[0].float().expand(4,-1,-1).transpose(-1,-2))*128**-.5).masked_fill(~mask[:,None],-torch.inf).softmax(-1)
            exact=(p@v[0].float().expand(4,-1,-1)).transpose(0,1).unsqueeze(0)
            error=float((output.float()-exact).norm()/exact.norm());assert error<.004
            cases.append({'arm':arm,'valid_keys':nk,'available_tiles':jt,'kept_per_head':target,'mask_mismatches':0,'consumer_relative_l2':error})
    write(root/'qualification/kernel_receipt.json',{'status':'pass','cases':cases,'count':len(cases)})


def run(root):
    import torch
    protocol=read(root/'protocol.json');assert protocol['source_sha256']=={k:sha(v) for k,v in source_files().items()}
    assert not (root/'generation_complete.json').exists()
    active={'phase':'qualification','pid':os.getpid(),'completed':0};stop=threading.Event()
    def monitor():
        while True:
            gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=30)
            append(root/'private/health/gpu_health.jsonl',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                **active,'returncode':gpu.returncode,'stdout':gpu.stdout,'stderr':gpu.stderr})
            if stop.wait(900):return
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        private=root/'private';private.mkdir(exist_ok=True)
        for name in ('models','source','compat'):
            (private/name).symlink_to((BASE/'private'/name).resolve(),target_is_directory=True)
        os.environ.update(HF_MODULES_CACHE=str(private/'hf_modules'),TRITON_CACHE_DIR=str(private/'triton_cache'))
        attention,worker=modules();qualify(attention,root)
        h=worker.ModelHarness(root,'trado').load()
        # Only dense control generation; no V2 request is executed.
        row=read(PREVIOUS/'private/data/trado/humaneval.json')[0];h.runtime.arm='allkept_fa4'
        control,_=worker.run_cell(h,row,2000,42)
        saved=[r for f in (BASE/'runs/trado/humaneval/allkept_fa4').glob('shard*/attempt001/records.jsonl') for r in rows(f) if r['id']==row['id']]
        assert len(saved)==1
        for key in ('counts','output_sha256','output_tokens','prompt_tokens'):assert control[key]==saved[0][key],('control',key)
        write(root/'qualification/control_receipt.json',{'status':'pass','arm':'allkept_fa4','id':row['id'],'exact_output_and_counts':True})
        original_budget=attention.budget_map
        def audited_budget(scores,mandatory,eligible):
            keep=original_budget(scores,mandatory,eligible)
            target=torch.maximum(torch.round(eligible.sum(-1).float()*.5).long(),(mandatory&eligible).sum(-1))
            torch._assert_async((keep.sum(-1)==target).all(),'half-budget count mismatch')
            torch._assert_async((~mandatory|keep).all(),'mandatory tile missing')
            torch._assert_async((~keep|eligible).all(),'invalid tile retained')
            h.runtime.counts['audited_half_budget_units']+=keep.shape[0]
            return keep
        attention.budget_map=audited_budget
        data=read(source_files()['manifest']);total_started=time.time()
        for arm in ARMS:
            active.update(phase='benchmark',arm=arm,completed=0)
            h.runtime.arm=arm;h.runtime.threshold=.02 # ignored for fixed-budget and dense arms
            armrows=[]
            for shard in range(1,5):
                leaf=root/'runs'/arm/f'shard{shard:03}'/'attempt001';part=[r for i,r in enumerate(data) if i%4==shard-1]
                config={'model':'trado','suite':'ruler32k','arm':arm,'shard':shard,'requests':len(part),'seed':42,
                    'model_revision':protocol['model_revision'],'manifest_sha256':protocol['source_sha256']['manifest'],
                    'protocol_sha256':sha(root/'protocol.json'),'sources':protocol['source_sha256'],
                    'native_sampler_unchanged':True,'sampler':protocol['sampler'],'attention':protocol['attention'],
                    'host_gpu_uuid':subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip(),
                    'budget_policy':protocol['fixed_budget'] if arm in ARMS[2:] else 'dense full support'}
                write(leaf/'config.json',config);shard_start=time.time()
                for row in part:
                    pub,priv=worker.run_cell(h,row,128,42);counts=pub['counts']
                    if arm in ARMS[2:]:
                        assert counts['audited_half_budget_units']==32*counts['selection_layer_calls']>0
                    else:assert counts['eligible_tiles']==counts['kept_tiles'] and not counts.get('selection_layer_calls',0)
                    pub.update(model='trado',suite='ruler32k',arm=arm,manifest_sha256=config['manifest_sha256'],
                        model_revision=config['model_revision'],protocol_sha256=config['protocol_sha256'])
                    append(leaf/'records.jsonl',pub)
                    append(leaf/'private/generation.private.jsonl',dict(priv,model='trado',suite='ruler32k',arm=arm))
                    armrows.append(pub);active['completed']+=1
                    print(json.dumps({'event':'cell_complete','arm':arm,'completed':len(armrows),'expected':130,'id':row['id'],
                                      'N':counts['denoising_forwards'],'T':pub['output_tokens']}),flush=True)
                write(leaf/'receipt.json',{'status':'pass','records_sha256':sha(leaf/'records.jsonl'),
                      'protocol_sha256':sha(root/'protocol.json'),'worker_seconds':time.time()-shard_start,'requests':len(part)})
                write(leaf/'final_complete.json',{'status':'complete','requests':len(part)})
            e=sum(r['counts']['eligible_tiles'] for r in armrows);k=sum(r['counts']['kept_tiles'] for r in armrows)
            sparsity=1-k/e
            assert len(armrows)==130 and (.48<=sparsity<=.52 if arm in ARMS[2:] else sparsity==0.)
            write(root/f'{arm}_complete.json',{'status':'complete','requests':130,'sparsity':sparsity,'eligible_tiles':e,'kept_tiles':k})
            print(json.dumps({'event':'arm_complete','arm':arm,'sparsity':sparsity}),flush=True)
        write(root/'generation_complete.json',{'status':'complete','requests':520,'arms':list(ARMS),'v2_generation_requests':0,
                                             'worker_seconds':time.time()-total_started,'protocol_sha256':sha(root/'protocol.json')})
    finally:
        stop.set();thread.join(timeout=35)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run']);p.add_argument('root',type=Path);a=p.parse_args()
    (freeze if a.mode=='freeze' else run)(a.root.resolve())
