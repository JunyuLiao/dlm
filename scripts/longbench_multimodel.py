"""Immutable native LongBench-v2 campaign; one model worker per H100."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import types

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/multi_model_value_20261030/attempt001'
PINS={'llada22':'224f8f11173bd16e5cf10054a963c74233758cf2','trado':'8c0d14b031636ddcc708df75fa1d57aefa8b0ba4'}
LIMITS={'llada22':131072,'trado':32768}
GROUPS={'llada22':('short','medium'),'trado':('short',)}
FIXED=('dense_native','allkept_fa4','mass_topk_k512_reuse_r4','value_v3a_singleton_k512_reuse_r4')
CTI='value_v2_current_tile_init_gaussian32_reuse_r4'
DATA=Path('/home/exouser/.cache/huggingface/hub/datasets--THUDM--LongBench-v2/snapshots/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/data.json')
OFFICIAL=ROOT/'reference/LongBench'
OFFICIAL_PIN='2e00731f8d0bff23dc4325161044d0ed8af94c1e'


def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def append(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(v)+'\n');f.flush();os.fsync(f.fileno())
def state(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v)+'\n');temp.replace(p)
def pool(rs):
    e=sum(r['counts']['eligible_tiles'] for r in rs);k=sum(r['counts']['kept_tiles'] for r in rs)
    return {'requests':len(rs),'eligible_tiles':e,'kept_tiles':k,'sparsity':1-k/e}


def fixed_budget_map(scores,mandatory,eligible):
    """K512 tokens = eight tiles, including mandatory current support."""
    import torch
    target=torch.minimum(eligible.sum(-1),torch.maximum(torch.full_like(eligible.sum(-1),8),(mandatory&eligible).sum(-1)))
    rank=scores.masked_fill(~eligible,-math.inf).masked_fill(mandatory&eligible,math.inf)
    order=torch.argsort(rank,descending=True,stable=True)
    keep=torch.zeros_like(eligible)
    keep.scatter_(1,order,torch.arange(scores.shape[1],device=scores.device)[None]<target[:,None])
    return keep&eligible


def chat_ids(tokenizer,prompt):
    ids=tokenizer.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,tokenize=True)
    if hasattr(ids,'input_ids'):ids=ids.input_ids
    if ids and isinstance(ids[0],list):ids=ids[0]
    return list(ids)


def fit_prompt(tokenizer,prompt,limit,cap=128):
    original=chat_ids(tokenizer,prompt);budget=limit-cap
    if len(original)<=budget:return prompt,original,len(original),False
    raw=tokenizer.encode(prompt,add_special_tokens=False)
    content_budget=budget-(len(chat_ids(tokenizer,''))+8)
    for _ in range(32):
        left=content_budget//2;right=content_budget-left
        text=tokenizer.decode(raw[:left]+raw[-right:],skip_special_tokens=True)
        encoded=chat_ids(tokenizer,text)
        if len(encoded)<=budget:return text,encoded,len(original),True
        content_budget-=max(8,len(encoded)-budget+8)
    raise ValueError('middle truncation did not fit context')


def source_files():
    result={'runner':Path(__file__),'reporter':ROOT/'scripts/longbench_multimodel_report.py',
        'tests':ROOT/'tests/test_longbench_multimodel.py','spec':ROOT/'docs/MULTIMODEL_LONGBENCH_V2_20261104.md',
        'selector':ROOT/'experiments/numerical_qk_reuse/v2_current_tile_init.py','dataset':DATA,
        **{f'official_{name.replace("/","_")}':OFFICIAL/name for name in ('pred.py','result.py','prompts/0shot.txt')}}
    for model,pin in PINS.items():
        snap=BASE/'private/execution'/pin
        result[model+'_model_identity']=BASE/f'{model}_model_identity.json'
        for name in ('multi_model_attention','multi_model_runtime','v31_value_kernels','v31_value_selectors','v31_value_summary','v27_fa4'):
            result[model+'_'+name]=snap/'experiments/numerical_qk_reuse'/f'{name}.py'
        result[model+'_worker']=snap/'scripts/multi_model_worker.py'
    return result


def verify_sources(root):
    p=read(root/'protocol.json')
    assert p['source_sha256']=={k:sha(v) for k,v in source_files().items()},'source drift'
    for m in PINS:assert sha(root/'private/data'/f'{m}.json')==p['manifests'][m]['sha256']
    return p


def prepare(root):
    from transformers import AutoTokenizer
    assert not root.exists();root.mkdir(parents=True)
    assert subprocess.check_output(['git','-C',str(OFFICIAL),'rev-parse','HEAD'],text=True).strip()==OFFICIAL_PIN
    for name in ('pred.py','result.py','prompts/0shot.txt'):
        assert (OFFICIAL/name).read_bytes()==subprocess.check_output(['git','-C',str(OFFICIAL),'show',f'{OFFICIAL_PIN}:{name}'])
    private=root/'private';private.mkdir()
    for name in ('models','source','compat'):(private/name).symlink_to((BASE/'private'/name).resolve(),target_is_directory=True)
    os.environ['HF_MODULES_CACHE']=str(private/'hf_modules')
    data=read(DATA);assert len(data)==503
    assert Counter(r['length'] for r in data)=={'short':180,'medium':215,'long':108}
    template=(OFFICIAL/'prompts/0shot.txt').read_text();manifests={}
    for model in PINS:
        tokenizer=AutoTokenizer.from_pretrained(private/'models'/model,trust_remote_code=True)
        samples=[]
        for item in data:
            if item['length'] not in GROUPS[model]:continue
            prompt=template
            for key,field in [('DOC','context'),('Q','question'),('C_A','choice_A'),('C_B','choice_B'),('C_C','choice_C'),('C_D','choice_D')]:
                prompt=prompt.replace('$'+key+'$',item[field].strip())
            text,ids,original,truncated=fit_prompt(tokenizer,prompt,LIMITS[model])
            samples.append({'id':item['_id'],'prompt':text,'gold':item['answer'],'seeds':[42],
                **{k:item[k] for k in ('length','difficulty','domain','sub_domain')},'prompt_tokens':len(ids),
                'original_prompt_tokens':original,'truncated':truncated,
                'input_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
            if len(samples)%40==0:print(json.dumps({'event':'prepared','model':model,'count':len(samples)}),flush=True)
        write(private/'data'/f'{model}.json',samples)
        sanitized=[{k:v for k,v in r.items() if k not in ('prompt','gold')} for r in samples]
        write(root/'data'/f'{model}_manifest.json',sanitized)
        calibration=[]
        for group in GROUPS[model]:
            part=sorted([r for r in samples if r['length']==group],key=lambda r:(r['prompt_tokens'],r['id']))
            calibration.extend(part[(2*i+1)*len(part)//12]['id'] for i in range(6))
        manifests[model]={'sha256':sha(private/'data'/f'{model}.json'),'public_sha256':sha(root/'data'/f'{model}_manifest.json'),
            'requests':len(samples),'length_counts':dict(Counter(r['length'] for r in samples)),
            'truncated':sum(r['truncated'] for r in samples),'prompt_min':min(r['prompt_tokens'] for r in samples),
            'prompt_max':max(r['prompt_tokens'] for r in samples),'calibration_ids':calibration}
    write(root/'protocol.json',{'status':'frozen','parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_sha256':{k:sha(v) for k,v in source_files().items()},'manifests':manifests,'seed':42,'output_cap':128,
        'context_limits':LIMITS,'fixed_arms':list(FIXED),'cti_arm':CTI,'fixed_budget_tokens':512,'fixed_budget_tiles':8,
        'mandatory_in_budget':True,'numerical_snapshots':PINS,'groups':GROUPS,'all_denoising_layers':True,
        'tile_geometry':{'Q':128,'KV':64},'reuse_calls':4,'cross_canvas_carry':False,'native_samplers_unchanged':True,
        'calibration':{'count_only':True,'transductive':True,'max_candidates':8,'min_sparsity':.95,'tolerance':.03,
            'starts':{'llada22':.01,'trado':.03},'bracket':[1e-8,4.],'max_full_panel_retries_below_95':1},
        'official_revision':OFFICIAL_PIN,'truncation':'middle of user prompt, preserve chat wrapper and 128 output-token reserve',
        'monitor_seconds':900,'gpu_hour_limit':None,'timing':'instrumented accounting, no clean latency claim'})
    print(json.dumps({'status':'frozen','manifests':manifests}),flush=True)


def modules(model):
    snapshot=BASE/'private/execution'/PINS[model]
    package=types.ModuleType('experiments');package.__path__=[str(snapshot/'experiments')];sys.modules['experiments']=package
    sys.path[:0]=[str(BASE/'private/compat'/model),str(snapshot),str(snapshot/'scripts')]
    attention=importlib.import_module('experiments.numerical_qk_reuse.multi_model_attention')
    worker=importlib.import_module('multi_model_worker')
    path=ROOT/'experiments/numerical_qk_reuse/v2_current_tile_init.py'
    spec=importlib.util.spec_from_file_location('experiments.numerical_qk_reuse.v2_current_tile_init',path)
    variant=importlib.util.module_from_spec(spec);spec.loader.exec_module(variant)
    return attention,worker,variant


def qualify(attention,variant,leaf):
    import torch
    from experiments.numerical_qk_reuse.v31_value_kernels import statistics_cuda
    from experiments.numerical_qk_reuse.v31_value_selectors import statistics_reference,full_support
    torch.set_num_threads(4);gen=torch.Generator(device='cuda').manual_seed(20261104);cases=[]
    for nk in (224,511,32764,131068):
        q=torch.randn((1,4,4,128),device='cuda',dtype=torch.bfloat16,generator=gen)
        k=torch.randn((1,1,nk,128),device='cuda',dtype=torch.bfloat16,generator=gen);v=torch.randn(k.shape,device='cuda',dtype=k.dtype,generator=gen)
        matrix=torch.randn((1,128,32),device='cuda',generator=gen)/32**.5
        z=(v[0].float()@matrix).unsqueeze(0);nu=v[0].float().square().sum(-1).mean(-1).sqrt();jt=(nk+63)//64
        stats=statistics_cuda(q,k,z,nu,128**-.5,jt)
        cpu=statistics_reference((q[0].cpu().float()@k[0,0].cpu().float().T)*128**-.5,z[0].cpu(),nu.cpu(),jt)
        mandatory=torch.zeros((4,jt),device='cuda',dtype=torch.bool);mandatory[:,-1]=True
        for method in ('v31','v3a','cti'):
            if method=='v31':
                def mass(s):
                    d=torch.logsumexp(s.log_mass,1,keepdim=True)
                    return (s.log_mass-torch.where(torch.isfinite(d),d,0.)).amax(-1)
                score=mass(stats);cs=mass(cpu)
            elif method=='v3a':
                score=attention.singleton_scores(stats)
                alpha,contrib,output,g=full_support(cpu);remain=alpha.sum(1)[:,None]-alpha
                err=(g.sum(1)[:,None]-g).norm(dim=-1)/(remain.clamp_min(torch.finfo(alpha.dtype).tiny)*cpu.nu[:,None].clamp_min(1e-12))
                cs=torch.where(cpu.valid,err,0.).amax(-1);cs.masked_fill_((cpu.valid&(remain<=0)).any(-1),math.inf)
            if method=='cti':
                keep=variant.current_tile_cuda(stats,.03,True,mandatory,torch.zeros_like(mandatory),0.)[0]
                expected=variant.current_tile_reference(cpu,.03,mandatory.cpu())[0]
            else:
                keep=fixed_budget_map(score,mandatory,stats.valid.any(-1))
                cs[:,-1]=math.inf;expected=torch.zeros((4,jt),dtype=torch.bool)
                for h in range(4):expected[h,torch.argsort(cs[h],descending=True,stable=True)[:min(8,jt)]]=True
            assert torch.equal(keep.cpu(),expected),(method,nk,'CPU mask')
            output=attention.v27_fa4.sparse_lists(q,k,v,attention.v27_fa4.block_sparse_tensors(keep.reshape(1,4,1,jt)),128**-.5)
            mask=keep.repeat_interleave(64,-1)[:,:nk]
            prob=((q[0].float()@k[0].float().expand(4,-1,-1).transpose(-1,-2))*128**-.5).masked_fill(~mask[:,None],-torch.inf).softmax(-1)
            exact=(prob@v[0].float().expand(4,-1,-1)).transpose(0,1).unsqueeze(0)
            error=float((output.float()-exact).norm()/exact.norm());assert error<.004,(method,nk,error)
            cases.append({'method':method,'valid_keys':nk,'mask_mismatches':0,'consumer_relative_l2':error})
    write(leaf/'qualification/kernel_receipt.json',{'status':'pass','cases':cases})


def panel(root,model,phase,attempt):
    import torch
    protocol=verify_sources(root);leaf=root/'panels'/model/phase/f'attempt{attempt:03}'
    assert not leaf.exists();leaf.mkdir(parents=True)
    write(leaf/'protocol.json',{'model':model,'phase':phase,'attempt':attempt,'root_protocol_sha256':sha(root/'protocol.json'),
        'manifest_sha256':protocol['manifests'][model]['sha256'],'source_sha256':protocol['source_sha256'],
        'model_revision':read(BASE/f'{model}_model_identity.json')['revision'],'seed':42,'output_cap':128})
    os.environ.update(HF_MODULES_CACHE=str(root/'private/hf_modules'),TRITON_CACHE_DIR=str(root/'private/triton_cache'))
    attention,worker,variant=modules(model);qualify(attention,variant,leaf)
    h=worker.ModelHarness(root,model).load()
    previous=ROOT/'results/current_tile_multimodel_20261103/attempt001'
    row=read(previous/'private/data'/model/'humaneval.json')[0];h.runtime.arm='allkept_fa4'
    control,_=worker.run_cell(h,row,1024 if model=='llada22' else 2000,42)
    saved=[r for f in (BASE/'runs'/model/'humaneval/allkept_fa4').glob('shard*/attempt001/records.jsonl') for r in rows(f) if r['id']==row['id']]
    assert len(saved)==1
    for key in ('counts','output_sha256','output_tokens'):assert control[key]==saved[0][key],('allkept control',key)
    write(leaf/'qualification/control_receipt.json',{'status':'pass','exact_output_and_counts':True})
    def audited_budget(scores,mandatory,eligible):
        keep=fixed_budget_map(scores,mandatory,eligible)
        target=torch.minimum(eligible.sum(-1),torch.maximum(torch.full_like(eligible.sum(-1),8),(mandatory&eligible).sum(-1)))
        torch._assert_async((keep.sum(-1)==target).all()&(~mandatory|keep).all()&(~keep|eligible).all(),'K512 budget/support')
        h.runtime.counts['audited_k512_head_units']+=keep.shape[0]
        return keep
    attention.budget_map=audited_budget
    data=read(root/'private/data'/f'{model}.json');byid={r['id']:r for r in data}
    # Assert the actual model encoder consumes exactly the frozen chat request.
    for item in data:
        ids=h.encode(item['prompt'])
        assert len(ids)==item['prompt_tokens'] and hashlib.sha256(json.dumps(ids).encode()).hexdigest()==item['input_sha256']
        assert len(ids)+128<=LIMITS[model]
    selected=None
    if phase=='cti':
        fixed_candidates=sorted((root/'panels'/model/'fixed').glob('attempt*/generation_complete.json'))
        assert fixed_candidates
        fixed=read(fixed_candidates[-1]);assert fixed['status']=='complete'
        e=sum(fixed['arms'][a]['eligible_tiles'] for a in FIXED[2:]);k=sum(fixed['arms'][a]['kept_tiles'] for a in FIXED[2:]);target=1-k/e
        def online(*a,**kw):
            h.runtime.counts['current_tile_init_selection_layer_calls']+=1
            return variant.current_tile_cuda(*a,**kw)
        attention.online_cuda=online;h.runtime.arm='value_v2_gaussian32_reuse_r4'
        lo,hi=protocol['calibration']['bracket'];tau=protocol['calibration']['starts'][model]
        if attempt>1:
            old=read(leaf.parent/f'attempt{attempt-1:03}'/'generation_complete.json')
            lo=old['threshold'];tau=lo*4;hi=max(4.,tau*4)
        trials=[];development=[byid[i] for i in protocol['manifests'][model]['calibration_ids']]
        for candidate in range(1,9):
            h.runtime.threshold=tau;observed=[]
            for item in development:
                public,_=worker.run_cell(h,{k:item[k] for k in ('id','prompt')},128,42)
                public.update(threshold=tau,candidate=candidate);append(leaf/'calibration/records.jsonl',public);observed.append(public)
            point={'threshold':tau,'candidate':candidate,**pool(observed)};trials.append(point)
            print(json.dumps({'event':'calibration','model':model,'target':target,**point}),flush=True)
            if point['sparsity']>=.95 and abs(point['sparsity']-target)<=.03:break
            if point['sparsity']<target:lo=tau
            else:hi=tau
            tau=math.sqrt(lo*hi)
        qualified=[x for x in trials if x['sparsity']>=.95 and abs(x['sparsity']-target)<=.03]
        assert qualified,'no high-sparsity calibration point qualified'
        selected=min(qualified,key=lambda x:abs(x['sparsity']-target));h.runtime.threshold=selected['threshold']
        write(leaf/'calibration/selected_threshold.json',{'status':'frozen','target_sparsity':target,'trials':trials,
            **selected,'accuracy_used':False,'transductive':True,'ids':protocol['manifests'][model]['calibration_ids']})
    started=time.time();out={}
    for arm in (FIXED if phase=='fixed' else (CTI,)):
        h.runtime.arm=arm.replace('_k512','') if phase=='fixed' else 'value_v2_gaussian32_reuse_r4'
        allrows=[]
        # Complete shards from interrupted attempts may be reused only by exact source/config/hash joins.
        for shard in range(1,9):
            dest=leaf/'runs'/arm/f'shard{shard:03}'/'attempt001';part=[r for i,r in enumerate(data) if i%8==shard-1]
            write(dest/'config.json',{'model':model,'arm':arm,'shard':shard,'requests':len(part),'seed':42,'output_cap':128,
                'root_protocol_sha256':sha(root/'protocol.json'),'panel_protocol_sha256':sha(leaf/'protocol.json'),
                'manifest_sha256':protocol['manifests'][model]['sha256'],'threshold':selected['threshold'] if selected else None,
                'budget_tokens':512 if arm in FIXED[2:] else None,
                'gpu_uuid':subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()})
            t=time.time()
            for item in part:
                pub,priv=worker.run_cell(h,item,128,42);counts=pub['counts']
                assert pub['prompt_tokens']==item['prompt_tokens']
                if arm in FIXED[:2]:assert counts['eligible_tiles']==counts['kept_tiles']
                elif arm in FIXED[2:]:assert counts['audited_k512_head_units']>0
                else:assert counts['current_tile_init_selection_layer_calls']==counts['selection_layer_calls']>0
                pub.update(model=model,suite='longbench_v2',arm=arm,threshold=selected['threshold'] if selected else None,
                    panel_protocol_sha256=sha(leaf/'protocol.json'),root_protocol_sha256=sha(root/'protocol.json'),
                    **{k:item[k] for k in ('length','difficulty','domain','sub_domain','truncated','original_prompt_tokens','input_sha256')})
                append(dest/'records.jsonl',pub);append(dest/'private/generation.private.jsonl',dict(priv,model=model,arm=arm))
                allrows.append(pub)
                state(root/'private/progress.json',{'model':model,'phase':phase,'attempt':attempt,'arm':arm,'completed':len(allrows),
                    'expected':len(data),'id':item['id'],'time':time.time(),'N':counts['denoising_forwards'],'T':pub['output_tokens']})
                print(json.dumps({'event':'cell_complete','model':model,'arm':arm,'completed':len(allrows),'expected':len(data),
                    'id':item['id'],'N':counts['denoising_forwards'],'T':pub['output_tokens']}),flush=True)
            write(dest/'receipt.json',{'status':'pass','requests':len(part),'records_sha256':sha(dest/'records.jsonl'),
                'config_sha256':sha(dest/'config.json'),'worker_seconds':time.time()-t})
            write(dest/'final_complete.json',{'status':'complete','requests':len(part)})
        assert len(allrows)==len(data) and len({r['id'] for r in allrows})==len(data)
        out[arm]=pool(allrows);write(leaf/f'{arm}_complete.json',{'status':'complete',**out[arm]})
    result={'status':'complete','model':model,'phase':phase,'arms':out,'worker_seconds':time.time()-started,
        'panel_protocol_sha256':sha(leaf/'protocol.json'),'threshold':selected['threshold'] if selected else None,
        'accepted':phase=='fixed' or out[CTI]['sparsity']>=.95}
    write(leaf/'generation_complete.json',result)
    print(json.dumps({'event':'panel_complete',**result}),flush=True)


def run(root):
    verify_sources(root);stop=threading.Event();active={'phase':'starting','pid':os.getpid()}
    def health():
        while True:
            gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=30)
            append(root/'private/health/gpu_health.jsonl',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                **active,'returncode':gpu.returncode,'nvidia_smi':gpu.stdout,'stderr':gpu.stderr})
            if stop.wait(900):return
    thread=threading.Thread(target=health,daemon=True);thread.start()
    try:
        for phase in ('fixed','cti'):
            for model in PINS:
                parent=root/'panels'/model/phase
                complete=[f for f in parent.glob('attempt*/generation_complete.json') if read(f)['accepted']]
                if complete:continue
                attempt=1+max([int(p.name[-3:]) for p in parent.glob('attempt*')],default=0)
                while True:
                    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
                    assert not gpu.strip(),'GPU occupied before launch'
                    command=[sys.executable,str(Path(__file__)), 'panel',str(root),'--model',model,'--phase',phase,'--attempt',str(attempt)]
                    log=root/'private/logs'/model/phase/f'attempt{attempt:03}.log';log.parent.mkdir(parents=True,exist_ok=True)
                    with log.open('x') as handle:
                        child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH='src:.'),stdout=handle,stderr=subprocess.STDOUT)
                        active.update(model=model,phase=phase,attempt=attempt,worker_pid=child.pid)
                        state(root/'private/active_worker.json',active);code=child.wait()
                    if code:raise RuntimeError(f'worker exit {code}; see {log.relative_to(root)}')
                    done=read(parent/f'attempt{attempt:03}'/'generation_complete.json')
                    if done['accepted']:break
                    assert phase=='cti' and attempt<2,'V2 sparsity below 95% after permitted retry'
                    attempt+=1
        # Scoring has no GPU work and uses a source-bound official extraction function.
        command=[sys.executable,str(ROOT/'scripts/longbench_multimodel_report.py'),str(root)]
        with (root/'private/scoring.log').open('x') as log:subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH='src:.'),stdout=log,stderr=subprocess.STDOUT,check=True)
        assert read(root/'final_complete.json')['status']=='complete'
    finally:
        stop.set();thread.join(timeout=35)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','panel']);p.add_argument('root',type=Path)
    p.add_argument('--model',choices=list(PINS));p.add_argument('--phase',choices=['fixed','cti']);p.add_argument('--attempt',type=int,default=1)
    a=p.parse_args();root=a.root.resolve()
    if a.mode=='panel':panel(root,a.model,a.phase,a.attempt)
    else:(prepare if a.mode=='prepare' else run)(root)
