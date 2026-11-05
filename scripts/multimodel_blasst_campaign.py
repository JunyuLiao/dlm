"""Aggressive BLASST, queued strictly after completed LongBench-v2 audit."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.longbench_multimodel import read,rows,sha,write,append,state,modules,BASE,PINS
PREVIOUS=ROOT/'results/current_tile_multimodel_20261103/attempt001'
DEPENDENCY=ROOT/'results/multimodel_longbench_v2_20261104/attempt002'
ARM='blasst_aggressive_inverse_length_reuse_r4'
ORDER=[('llada22','humaneval'),('trado','humaneval'),('trado','ruler32k'),('llada22','ruler128k'),('trado','aime26'),('llada22','aime26')]


def pool(rs):
    e=sum(r['counts']['eligible_tiles'] for r in rs);k=sum(r['counts']['kept_tiles'] for r in rs)
    return {'requests':len(rs),'eligible_tiles':e,'kept_tiles':k,'sparsity':1-k/e}


def sources():
    result={'runner':Path(__file__),'reporter':ROOT/'scripts/multimodel_blasst_report.py',
        'selector':ROOT/'experiments/numerical_qk_reuse/multi_model_blasst.py','tests':ROOT/'tests/test_multi_model_blasst.py',
        'spec':ROOT/'docs/MULTIMODEL_AGGRESSIVE_BLASST_20261105.md',
        'module_loader':ROOT/'scripts/longbench_multimodel.py',
        'core_reference':ROOT/'src/dllm/attention/blasst/core.py',
        'historical_rule':ROOT/'experiments/diffusion_gemma_value_aware_gpu/kernels.py',
        'historical_inverse_length':ROOT/'experiments/diffusion_gemma_value_aware_followup/report.py'}
    for model,pin in PINS.items():
        snap=BASE/'private/execution'/pin
        for name in ('multi_model_attention','multi_model_runtime','v31_value_kernels','v31_value_selectors','v31_value_summary','v27_fa4'):
            result[model+'_'+name]=snap/'experiments/numerical_qk_reuse'/f'{name}.py'
        result[model+'_worker']=snap/'scripts/multi_model_worker.py'
        result[model+'_identity']=BASE/f'{model}_model_identity.json'
    return result


def verify(root):
    p=read(root/'protocol.json');assert p['source_sha256']=={k:sha(v) for k,v in sources().items()},'source drift'
    for key,digest in p['input_sha256'].items():
        model,suite=key.split('/');assert sha(root/'private/data'/model/f'{suite}.json')==digest
    return p


def prepare(root):
    assert not root.exists();root.mkdir(parents=True);private=root/'private';private.mkdir()
    for name in ('models','source','compat'):(private/name).symlink_to((BASE/'private'/name).resolve(),target_is_directory=True)
    (private/'data').symlink_to((PREVIOUS/'private/data').resolve(),target_is_directory=True)
    inputs={}
    for model,suite in ORDER:
        for name in (suite,'calibration_ruler') if suite.startswith('ruler') else (suite,):
            inputs[model+'/'+name]=sha(private/'data'/model/f'{name}.json')
    write(root/'protocol.json',{'status':'frozen','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_sha256':{k:sha(v) for k,v in sources().items()},'input_sha256':inputs,'order':ORDER,'arm':ARM,
        'dependency':'results/multimodel_longbench_v2_20261104/attempt002/final_complete.json',
        'native_sources':PINS,'model_revisions':{m:read(BASE/f'{m}_model_identity.json')['revision'] for m in PINS},
        'requests':768,'seeds':{'aime26':[42,43,44],'humaneval':[42],'ruler':[42]},
        'caps':{'llada22':{'aime26':32768,'humaneval':1024,'ruler128k':128},'trado':{'aime26':2000,'humaneval':2000,'ruler32k':128}},
        'policy':'lambda(L)=exp(log_kappa)/L, preceding maximum of all seen tiles; strict skip; first/current mandatory',
        'log_kappa_bracket':[-20.,30.],'initial_log_kappa':0.,'calibration_target':[.48,.52],'full_acceptable':[.4,.6],
        'max_calibration_candidates':10,'max_target_driven_full_attempts':3,'threshold_transfer':False,
        'tile_geometry':[128,64],'reuse_calls':4,'cross_canvas_carry':False,'all_denoising_layers':True,
        'dense_prefill_commit':True,'samplers_unchanged':True,'scoring':'official matched historical rules',
        'monitor_seconds':900,'gpu_hour_limit':None,'timing':'instrumented accounting only'})
    inventory={}
    for parent in (PREVIOUS,ROOT/'results/trado_ruler32k_baselines_20261104/attempt002'):
        for p in subprocess.check_output(['git','ls-files',str(parent.relative_to(ROOT))],text=True).splitlines():inventory[p]=sha(ROOT/p)
    write(root/'baseline_readonly_inventory.json',inventory)
    print(json.dumps({'status':'frozen','panels':6,'requests':768}),flush=True)


def qualify(module,attention,leaf):
    import torch
    torch.set_num_threads(4);g=torch.Generator(device='cuda').manual_seed(20261105);checks=[]
    for nk in (224,511,32764,131068):
        q=torch.randn((1,4,4,128),device='cuda',dtype=torch.bfloat16,generator=g)
        k=torch.randn((1,1,nk,128),device='cuda',dtype=torch.bfloat16,generator=g);v=torch.randn(k.shape,device='cuda',dtype=k.dtype,generator=g)
        maxima=module.block_maxima(q,k,128**-.5);jt=(nk+63)//64
        score=(q[0].cpu().float()@k[0,0].cpu().float().T)*128**-.5
        padded=torch.full((4,128,jt*64),-math.inf);padded[:,:4,:nk]=score
        expected=padded.reshape(4,128,jt,64).amax(-1).transpose(1,2)
        finite=torch.isfinite(expected);error=float((maxima.cpu()[finite]-expected[finite]).abs().max());assert error<2e-5
        mandatory=torch.zeros((4,jt),device='cuda',dtype=torch.bool);mandatory[:,-1]=True
        for log_lambda in (-math.inf,-2.,0.,math.log(2.),4.):
            keep=module.select_maxima(maxima,log_lambda,mandatory)
            ref=module.reference(expected,log_lambda,mandatory.cpu());assert torch.equal(keep.cpu(),ref),(nk,log_lambda)
            output=attention.v27_fa4.sparse_lists(q,k,v,attention.v27_fa4.block_sparse_tensors(keep.reshape(1,4,1,jt)),128**-.5)
            mask=keep.repeat_interleave(64,-1)[:,:nk]
            scores=(q[0].float()@k[0].float().expand(4,-1,-1).transpose(-1,-2))*128**-.5
            exact=(scores.masked_fill(~mask[:,None],-torch.inf).softmax(-1)@v[0].float().expand(4,-1,-1)).transpose(0,1).unsqueeze(0)
            relative=float((output.float()-exact).norm()/exact.norm());assert relative<.004
            checks.append({'valid_keys':nk,'log_lambda':log_lambda if math.isfinite(log_lambda) else 'dense_negative_infinity',
                'mask_mismatches':0,'maximum_abs_error':error,'consumer_relative_l2':relative})
    write(leaf/'qualification/kernel_receipt.json',{'status':'pass','cases':checks,'lambda_above_one_tested':True})


def panel(root,model,suite,attempt):
    import torch
    p=verify(root);assert read(DEPENDENCY/'final_complete.json')['status']=='complete'
    leaf=root/'panels'/model/suite/f'attempt{attempt:03}';assert not leaf.exists();leaf.mkdir(parents=True)
    data=read(root/'private/data'/model/f'{suite}.json');cap=p['caps'][model][suite];cells=[(r,s) for r in data for s in r['seeds']]
    if suite.startswith('ruler'):development=read(root/'private/data'/model/'calibration_ruler.json');seed=142;transductive=False
    else:
        ordered=sorted(data,key=lambda r:(len(r['prompt']),r['id']));development=[ordered[(2*i+1)*len(ordered)//12] for i in range(6)];seed=42;transductive=True
    write(leaf/'protocol.json',{'model':model,'suite':suite,'arm':ARM,'attempt':attempt,'root_protocol_sha256':sha(root/'protocol.json'),
        'manifest_sha256':p['input_sha256'][model+'/'+suite],'cap':cap,'calibration_ids':[r['id'] for r in development],
        'calibration_seed':seed,'transductive':transductive,'model_revision':p['model_revisions'][model]})
    os.environ.update(HF_MODULES_CACHE=str(root/'private/hf_modules'),TRITON_CACHE_DIR=str(root/'private/triton_cache'))
    attention,worker,_=modules(model)
    path=ROOT/'experiments/numerical_qk_reuse/multi_model_blasst.py';spec=importlib.util.spec_from_file_location('standalone_multimodel_blasst',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    qualify(module,attention,leaf);h=worker.ModelHarness(root,model).load()
    row=read(PREVIOUS/'private/data'/model/'humaneval.json')[0];h.runtime.arm='allkept_fa4'
    pub,_=worker.run_cell(h,row,1024 if model=='llada22' else 2000,42)
    saved=[r for f in (BASE/'runs'/model/'humaneval/allkept_fa4').glob('shard*/attempt001/records.jsonl') for r in rows(f) if r['id']==row['id']]
    assert len(saved)==1
    for key in ('counts','output_sha256','output_tokens'):assert pub[key]==saved[0][key],('allkept',model,key)
    write(leaf/'qualification/control_receipt.json',{'status':'pass','exact_allkept_output_and_counts':True})
    module.install(h,type(h.runtime));lo,hi=p['log_kappa_bracket'];theta=p['initial_log_kappa'];trials=[]
    if attempt>1 and (leaf.parent/f'attempt{attempt-1:03}'/'generation_complete.json').exists():
        prior=read(leaf.parent/f'attempt{attempt-1:03}'/'generation_complete.json')
        if prior['sparsity']>.6:hi=prior['log_kappa'];theta=hi-1.
        else:lo=prior['log_kappa'];theta=lo+1.
    for candidate in range(1,11):
        h.runtime.log_kappa=theta;rs=[]
        for row in development:
            sample={k:row[k] for k in ('id','prompt','answer_prefix') if k in row}
            pub,_=worker.run_cell(h,sample,cap,seed);pub.update(candidate=candidate,log_kappa=theta)
            append(leaf/'calibration/records.jsonl',pub);rs.append(pub)
        point={'candidate':candidate,'log_kappa':theta,**pool(rs)};trials.append(point)
        print(json.dumps({'event':'calibration','model':model,'suite':suite,**point}),flush=True)
        if .48<=point['sparsity']<=.52:break
        if point['sparsity']<.5:lo=theta
        else:hi=theta
        below=[x for x in trials if x['sparsity']<.5];above=[x for x in trials if x['sparsity']>=.5]
        pairs=[(a,b) for a in below for b in above if a['log_kappa']<b['log_kappa']]
        if pairs:
            a,b=min(pairs,key=lambda pair:pair[1]['log_kappa']-pair[0]['log_kappa'])
            fraction=min(.8,max(.2,(.5-a['sparsity'])/(b['sparsity']-a['sparsity'])))
            theta=a['log_kappa']+fraction*(b['log_kappa']-a['log_kappa'])
        else:theta=(lo+hi)/2
    chosen=min(trials,key=lambda x:abs(x['sparsity']-.5));h.runtime.log_kappa=chosen['log_kappa']
    write(leaf/'calibration/selected_threshold.json',{'status':'frozen','trials':trials,**chosen,'accuracy_used':False,'v2_threshold_used':False})
    started=time.time();allrows=[]
    for shard in range(1,5):
        dest=leaf/'runs'/f'shard{shard:03}'/'attempt001';part=[cell for i,cell in enumerate(cells) if i%4==shard-1]
        write(dest/'config.json',{'model':model,'suite':suite,'arm':ARM,'requests':len(part),'shard':shard,'log_kappa':chosen['log_kappa'],
            'root_protocol_sha256':sha(root/'protocol.json'),'panel_protocol_sha256':sha(leaf/'protocol.json'),
            'manifest_sha256':p['input_sha256'][model+'/'+suite],'model_revision':p['model_revisions'][model],
            'gpu_uuid':subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()})
        t=time.time()
        for row,seed in part:
            pub,priv=worker.run_cell(h,row,cap,seed);c=pub['counts'];assert c['blasst_selection_layer_calls']==c['selection_layer_calls']>0
            assert not c.get('projected_value_tokens',0)
            pub.update(model=model,suite=suite,arm=ARM,log_kappa=chosen['log_kappa'],lambda_min=h.runtime.lambda_min,
                lambda_max=h.runtime.lambda_max,panel_protocol_sha256=sha(leaf/'protocol.json'),root_protocol_sha256=sha(root/'protocol.json'))
            append(dest/'records.jsonl',pub);append(dest/'private/generation.private.jsonl',dict(priv,model=model,suite=suite,arm=ARM));allrows.append(pub)
            progress={'event':'cell_complete','model':model,'suite':suite,'attempt':attempt,'completed':len(allrows),
                'expected':len(cells),'N':c['denoising_forwards'],'T':pub['output_tokens'],'time':time.time()}
            state(root/'private/progress.json',progress);print(json.dumps(progress),flush=True)
        write(dest/'receipt.json',{'status':'pass','requests':len(part),'records_sha256':sha(dest/'records.jsonl'),
            'config_sha256':sha(dest/'config.json'),'worker_seconds':time.time()-t})
        write(dest/'final_complete.json',{'status':'complete','requests':len(part)})
    result={'status':'complete','model':model,'suite':suite,'log_kappa':chosen['log_kappa'],**pool(allrows),
        'worker_seconds':time.time()-started,'panel_protocol_sha256':sha(leaf/'protocol.json')}
    result['accepted']=.4<=result['sparsity']<=.6;write(leaf/'generation_complete.json',result)
    print(json.dumps({'event':'panel_complete',**result}),flush=True)


def run(root):
    verify(root);active={'phase':'waiting_for_longbench','pid':os.getpid()};stop=threading.Event()
    def health():
        while True:
            g=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=30)
            append(root/'private/health/gpu_health.jsonl',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                **active,'returncode':g.returncode,'nvidia_smi':g.stdout,'stderr':g.stderr})
            if stop.wait(900):return
    thread=threading.Thread(target=health,daemon=True);thread.start()
    try:
        state(root/'private/active_worker.json',active)
        while not (DEPENDENCY/'final_complete.json').exists():
            if (root/'STOP_REQUESTED.json').exists():return
            time.sleep(30)
        done=read(DEPENDENCY/'final_complete.json');assert done['status']=='complete' and done['requests']==2875
        assert done['receipt_sha256']==sha(DEPENDENCY/done['analysis']/'receipt.json')
        if not (root/'dependency_receipt.json').exists():write(root/'dependency_receipt.json',{'status':'pass',
            'completion_sha256':sha(DEPENDENCY/'final_complete.json'),'analysis_receipt_sha256':done['receipt_sha256']})
        for model,suite in ORDER:
            parent=root/'panels'/model/suite
            if any(read(f)['accepted'] for f in parent.glob('attempt*/generation_complete.json')):continue
            attempt=1+max([int(p.name[-3:]) for p in parent.glob('attempt*')],default=0)
            while True:
                assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
                log=root/'private/logs'/model/suite/f'attempt{attempt:03}.log';log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('x') as f:
                    child=subprocess.Popen([sys.executable,str(Path(__file__)),'panel',str(root),'--model',model,'--suite',suite,'--attempt',str(attempt)],
                        cwd=ROOT,env=dict(os.environ,PYTHONPATH='src:.'),stdout=f,stderr=subprocess.STDOUT)
                    active.update(phase='calibration_and_generation',model=model,suite=suite,attempt=attempt,worker_pid=child.pid)
                    state(root/'private/active_worker.json',active);code=child.wait()
                if code:raise RuntimeError(f'worker exit {code}; see {log.relative_to(root)}')
                if read(parent/f'attempt{attempt:03}'/'generation_complete.json')['accepted']:break
                assert attempt<3,'sparsity remains outside 40–60% after three attempts'
                attempt+=1
        active['phase']='scoring'
        with (root/'private/scoring.log').open('x') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/multimodel_blasst_report.py'),str(root)],cwd=ROOT,
                env=dict(os.environ,PYTHONPATH='src:.'),stdout=log,stderr=subprocess.STDOUT,check=True)
    finally:stop.set();thread.join(timeout=35)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','panel']);p.add_argument('root',type=Path)
    p.add_argument('--model',choices=list(PINS));p.add_argument('--suite');p.add_argument('--attempt',type=int,default=1)
    a=p.parse_args();root=a.root.resolve()
    if a.mode=='panel':panel(root,a.model,a.suite,a.attempt)
    else:(prepare if a.mode=='prepare' else run)(root)
