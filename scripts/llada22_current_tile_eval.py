"""Frozen current-tile-initialized V2 calibration and native HumanEval panel."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import types

import torch

ARM = "value_v2_current_tile_init_gaussian32_reuse_r4"
BASE_ARM = "value_v2_gaussian32_reuse_r4"
PIN = "224f8f11173bd16e5cf10054a963c74233758cf2"
REV = "e4f77d25ab5ca86f2223c6cdc6a72b887e20c8a5"
REPO = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        json.dump(v, f, indent=2)
        f.write("\n")


def append(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(v) + "\n")
        f.flush()
        os.fsync(f.fileno())


def frozen_sources(base):
    snap = base / "private/execution" / PIN
    return {"runner": Path(__file__),
            "variant": REPO / "experiments/numerical_qk_reuse/v2_current_tile_init.py",
            **{n: snap / "experiments/numerical_qk_reuse" / f"{n}.py" for n in
               ("multi_model_runtime", "multi_model_attention", "v31_value_kernels", "v31_value_selectors", "v27_fa4")},
            "worker": snap / "scripts/multi_model_worker.py",
            "manifest": base / "private/data/llada22/humaneval.json",
            "development": base / "private/data/llada22/calibration_code_contextmatched001.json"}


def freeze(base, out):
    assert not out.exists()
    out.mkdir(parents=True)
    sources = frozen_sources(base)
    assert len(read(sources['manifest'])) == 164 and len(read(sources['development'])) == 6
    write(out / 'protocol.json', {
        'status': 'frozen', 'arm': ARM, 'parent': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'source_sha256': {k: sha(v) for k,v in sources.items()}, 'model_revision': REV,
        'initialization': 'last mandatory KV64 tile once; then chronological tiles 0..J-2',
        'sampler': {'block_length':32, 'steps':32, 'threshold':.5, 'editing_threshold':0.,
                    'max_post_steps':16, 'temperature':0., 'eos_early_stop':True, 'cap':1024},
        'calibration': {'prompts':6, 'seed':142, 'bracket':[.01,4.], 'max_candidates':8,
                        'target':[.48,.52], 'selection':'closest pooled tile sparsity to .5; no scoring',
                        'confirmation_requests':6, 'no_post_benchmark_recalibration':True},
        'benchmark': {'problems':164, 'seed':42, 'shards':4, 'requests_per_shard':41},
        'controls':'first four canonical items each: original V2, V31, allkept FA4; exact saved hashes/counts',
        'scope': {'layers':list(range(20)), 'q_heads':16, 'q_tile':128, 'kv_tile':64,
                  'projection':32, 'projection_seed':'1729+layer', 'reuse':4, 'cross_canvas_carry':False},
        'monitor_interval_seconds':900, 'latency':'instrumented accounting only',
        'interpretation':'previously examined HumanEval follow-up; exploratory, not held-out confirmation'})


def modules(base):
    import importlib
    snap = base / 'private/execution' / PIN
    package = types.ModuleType('experiments')
    package.__path__ = [str(snap / 'experiments')]
    sys.modules['experiments'] = package
    sys.path[:0] = [str(snap), str(snap / 'scripts')]
    attention = importlib.import_module('experiments.numerical_qk_reuse.multi_model_attention')
    worker = importlib.import_module('multi_model_worker')
    for module in (attention, sys.modules['experiments.numerical_qk_reuse.multi_model_runtime']):
        assert Path(module.__file__).resolve().is_relative_to(snap.resolve())
    path = REPO / 'experiments/numerical_qk_reuse/v2_current_tile_init.py'
    spec = importlib.util.spec_from_file_location('experiments.numerical_qk_reuse.v2_current_tile_init', path)
    variant = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(variant)
    return attention, worker, variant


def qualify(variant, attention, out):
    from experiments.numerical_qk_reuse.v31_value_kernels import statistics_cuda
    from experiments.numerical_qk_reuse.v31_value_selectors import Stats
    cases = []
    gen = torch.Generator(device='cuda').manual_seed(729)
    for nk in (32, 96, 160, 224, 288, 544):
        q = torch.randn((1,4,32,128),device='cuda',dtype=torch.bfloat16,generator=gen)
        k = torch.randn((1,1,nk,128),device='cuda',dtype=torch.bfloat16,generator=gen)
        v = torch.randn((1,1,nk,128),device='cuda',dtype=torch.bfloat16,generator=gen)
        projection = torch.randn((1,128,32),device='cuda',generator=gen) / 32**.5
        z = (v[0].float() @ projection).unsqueeze(0)
        nu = v[0].float().square().sum(-1).mean(-1).sqrt()
        stats = statistics_cuda(q,k,z,nu,128**-.5,(nk+63)//64)
        cpu = Stats(*(x.cpu() for x in (stats.log_mass,stats.mean,stats.nu,stats.invalid)),
                    stats.prefix_tiles,stats.heads,stats.blocks)
        mandatory = torch.zeros((4,(nk+63)//64),device='cuda',dtype=torch.bool)
        mandatory[:,-1] = True
        for tau in (0.,.05,.2,.8,4.):
            actual = variant.current_tile_cuda(stats,tau,True,mandatory,torch.zeros_like(mandatory),0.)
            expected = variant.current_tile_reference(cpu,tau,mandatory.cpu())
            assert torch.equal(actual[0].cpu(),expected[0]), (nk,tau,'mask')
            assert torch.allclose(actual[1].cpu(),expected[1],atol=2e-5,rtol=2e-5), (nk,tau,'state')
            assert torch.allclose(actual[2].cpu(),expected[2],atol=2e-5,rtol=2e-5), (nk,tau,'mass')
            keep = actual[0]
            output = attention.v27_fa4.sparse_lists(q,k,v,attention.v27_fa4.block_sparse_tensors(keep.reshape(1,4,1,-1)),128**-.5)
            mask = keep.repeat_interleave(64,-1)[:,:nk]
            prob = ((q[0].float() @ k[0].float().expand(4,-1,-1).transpose(-1,-2))*128**-.5).masked_fill(~mask[:,None],-torch.inf).softmax(-1)
            exact = (prob @ v[0].float().expand(4,-1,-1)).transpose(0,1).unsqueeze(0)
            err = ((output.float()-exact).norm()/exact.norm()).item()
            assert err < .004, (nk,tau,'consumer',err)
            cases.append({'valid_keys':nk,'threshold':tau,'kept':int(keep.sum()),'mask_mismatches':0,'consumer_relative_l2':err})
    write(out/'qualification/kernel_receipt.json',{'status':'pass','cases':cases,'count':len(cases)})


def run(base,out):
    protocol = read(out/'protocol.json')
    assert protocol['source_sha256'] == {k:sha(v) for k,v in frozen_sources(base).items()}
    assert read(base/'llada22_model_identity.json')['revision'] == REV
    assert not (out/'generation_complete.json').exists()
    os.environ.update(HF_MODULES_CACHE=str(base/'private/hf_modules'),TRITON_CACHE_DIR=str(base/'private/triton_cache'))
    attention,worker,variant = modules(base)
    state = {'phase':'qualification','completed':0,'pid':os.getpid()}
    stop = threading.Event()
    def monitor():
        while True:
            gpu = subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=30)
            append(out/'health/gpu_health.jsonl',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                **state,'returncode':gpu.returncode,'nvidia_smi':gpu.stdout,'stderr':gpu.stderr})
            if stop.wait(900):
                return
    thread = threading.Thread(target=monitor,daemon=True)
    thread.start()
    try:
        qualify(variant,attention,out)
        h = worker.ModelHarness(base,'llada22').load()
        dataset = read(base/'private/data/llada22/humaneval.json')
        controls=[]
        state['phase']='native_control_replay'
        for arm in (BASE_ARM,'mass_topk_reuse_r4','allkept_fa4'):
            saved = {r['id']:r for p in (base/'runs/llada22/humaneval'/arm).glob('shard*/attempt001/records.jsonl')
                     for r in map(json.loads,p.read_text().splitlines())}
            h.runtime.arm,h.runtime.threshold = arm,.8944271909999159
            for row in dataset[:4]:
                pub,_ = worker.run_cell(h,row,1024,42)
                for key in ('output_sha256','output_tokens','counts','finished','prompt_tokens'):
                    assert pub[key] == saved[row['id']][key], ('control_parity',arm,row['id'],key)
                controls.append({'arm':arm,'id':row['id'],'status':'exact_hash_and_counts_match'})
        write(out/'qualification/control_receipt.json',{'status':'pass','cases':controls})
        original = attention.online_cuda
        def current_init(*args,**kwargs):
            h.runtime.counts['current_tile_init_selection_layer_calls'] += 1
            return variant.current_tile_cuda(*args,**kwargs)
        attention.online_cuda = current_init
        h.runtime.arm = BASE_ARM  # Original dispatch path; public arm is the separately named variant.
        out_private=out/'private'
        out_private.mkdir(exist_ok=True)
        (out_private/'data').symlink_to(base/'private/data',target_is_directory=True)
        state['phase']='calibration'
        development = read(frozen_sources(base)['development'])
        lo,hi=protocol['calibration']['bracket']
        trials=[]
        for index in range(1,9):
            tau=(lo*hi)**.5
            h.runtime.threshold=tau
            rows=[]
            for row in development:
                pub,priv=worker.run_cell(h,row,1024,142)
                pub.update(threshold=tau,candidate=index)
                append(out/'calibration/records.jsonl',pub)
                append(out/'calibration/private/generation.private.jsonl',dict(priv,threshold=tau,candidate=index))
                rows.append(pub)
            e=sum(r['counts']['eligible_tiles'] for r in rows); k=sum(r['counts']['kept_tiles'] for r in rows)
            trial={'candidate':index,'threshold':tau,'eligible_tiles':e,'kept_tiles':k,'sparsity':1-k/e}
            trials.append(trial)
            print(json.dumps({'event':'calibration_candidate',**trial}),flush=True)
            if .48<=trial['sparsity']<=.52:
                break
            if trial['sparsity']<.5:lo=tau
            else:hi=tau
        selected=min(trials,key=lambda x:abs(x['sparsity']-.5))
        h.runtime.threshold=selected['threshold']
        confirmation=[]
        for row in development:
            pub,priv=worker.run_cell(h,row,1024,142)
            append(out/'calibration/confirmation.jsonl',pub)
            append(out/'calibration/private/confirmation.private.jsonl',priv)
            confirmation.append(pub)
        e=sum(r['counts']['eligible_tiles'] for r in confirmation);k=sum(r['counts']['kept_tiles'] for r in confirmation)
        assert (e,k)==(selected['eligible_tiles'],selected['kept_tiles']), 'calibration repeat changed'
        write(out/'calibration/selected_threshold.json',{'trials':trials,**selected,'confirmed_sparsity':1-k/e,
              'target_met':.48<=1-k/e<=.52,'development_requests':6,'seed':142,'score_used':False})
        state['phase']='benchmark'
        start=time.time()
        for shard in range(1,5):
            cells=[r for i,r in enumerate(dataset) if i%4==shard-1]
            leaf=out/'runs/llada22/humaneval'/ARM/f'shard{shard:03}'/'attempt001'
            config={'model':'llada22','suite':'humaneval','arm':ARM,'internal_dispatch_arm':BASE_ARM,
                    'threshold':selected['threshold'],'shard':shard,'cells':41,'model_revision':REV,
                    'protocol_sha256':sha(out/'protocol.json'),'sources':protocol['source_sha256'],
                    'host_gpu_uuid':subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip(),
                    'sampler':protocol['sampler'],'scope':protocol['scope'],'kernel':'existing FA4 D128 sparse consumer; original optimized online chunk after current-tile state initialization'}
            write(leaf/'config.json',config)
            shard_start=time.time()
            for row in cells:
                pub,priv=worker.run_cell(h,row,1024,42)
                assert pub['counts']['selection_layer_calls']==pub['counts']['current_tile_init_selection_layer_calls']>0
                pub.update(model='llada22',suite='humaneval',arm=ARM,threshold=selected['threshold'],
                           protocol_sha256=config['protocol_sha256'],model_revision=REV)
                append(leaf/'records.jsonl',pub)
                append(leaf/'private/generation.private.jsonl',dict(priv,model='llada22',suite='humaneval',arm=ARM))
                state['completed']+=1
                print(json.dumps({'event':'cell_complete','id':row['id'],'completed':state['completed'],
                      'output_tokens':pub['output_tokens'],'N':pub['counts']['denoising_forwards'],'C':pub['counts']['canvases']}),flush=True)
            write(leaf/'receipt.json',{**config,'status':'pass','records_sha256':sha(leaf/'records.jsonl'),
                                     'gpu_worker_seconds':time.time()-shard_start})
            write(leaf/'final_complete.json',{'status':'complete','cells':len(cells)})
        attention.online_cuda=original
        write(out/'generation_complete.json',{'status':'complete','arm':ARM,'requests':164,'seed':42,
              'threshold':selected['threshold'],'gpu_worker_seconds':time.time()-start,
              'protocol_sha256':sha(out/'protocol.json')})
        state['phase']='generation_complete'
    finally:
        stop.set(); thread.join(timeout=35)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['freeze','run'])
    p.add_argument('base',type=Path)
    p.add_argument('out',type=Path)
    a=p.parse_args()
    (freeze if a.mode=='freeze' else run)(a.base.resolve(),a.out.resolve())


if __name__=='__main__':
    main()
