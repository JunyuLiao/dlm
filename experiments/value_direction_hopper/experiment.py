"""Frozen, resumable native-model evaluation; one cooperative GPU worker.

No dataset selection or calibration happens here. End-to-end wall timing is
separate from optional CUDA-event attention profiling. Report generation reads
completed shards only and never invokes the model.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import torch

from dllm.models import create_adapter
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _request, _set_context
from experiments.diffusion_gemma_value_aware_followup.protocol import MODEL, REVISION
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_ruler8k_jl import score
from .capture import SOURCE
from .integration import install

METHODS=('native_dense','kernel_gaussian32','kernel_blasst')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def setup(root,library,split,limit,methods,profile,torch_library=None):
    import transformers
    manifest=SOURCE/f'{split}_manifest.json'
    rows=json.loads(manifest.read_text())
    if limit:rows=rows[:limit]
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Invalid frozen manifest')
    policies={m:SOURCE/'policies/ruler4k'/f'{name}_s70.json' for m,name in
              [('kernel_gaussian32','jl_gaussian_r32'),('kernel_blasst','blasst')]}
    files=[manifest,library,*policies.values(),Path(__file__),Path(__file__).with_name('integration.py'),
           Path(__file__).with_name('cuda.py'),Path(__file__).with_name('masks.py'),Path(__file__).with_name('projection.py'),
           Path('experiments/diffusion_gemma_jl_output_aware/projections.py'),
           Path('experiments/diffusion_gemma_solattn_blasst_multibench/runner.py'),
           Path('src/dllm/models/adapters/diffusion_gemma.py')]
    # The tested binary, its source archive and build command—not the moving
    # working tree—are authoritative for kernel provenance.
    build_manifest=library.with_name(library.stem.removeprefix('value_direction_')+'.json')
    if build_manifest.exists():files.append(build_manifest)
    if torch_library is not None:files.extend([torch_library,torch_library.parent/'provenance.json',Path(__file__).parent/'csrc/torch_bridge.cpp'])
    contract=dict(schema='hopper_native_eval_v1',model=MODEL,revision=REVISION,
        source=str(SOURCE),split=split,samples=len(rows),ids=[r['id'] for r in rows],
        task_counts=dict(Counter(r['task'] for r in rows)),methods=list(methods),
        source_hashes={str(p.resolve()):sha(p) for p in files},library=str(library.resolve()),
        torch_library=None if torch_library is None else str(torch_library.resolve()),
        policies={m:json.loads(p.read_text())['policy'] for m,p in policies.items()},
        tile=[128,64],projection=dict(family='gaussian',rank=32,seed=1729,backend='fused_tf32x3_fp32'),
        target_physical_sparsity=.70,precision='tf32x3_register',schedule='tma_value_cp_blasst',
        profile=profile,torch=torch.__version__,transformers=transformers.__version__,
        cuda=torch.version.cuda,decoding='unchanged _request / native DiffusionGemma temperature schedule',
        timing='synchronized generation wall time; bindings, warmup and post-generation reduction excluded',
        warmup='One 1-token inference per method and encoded prompt length, excluded from timing; prevents shape-JIT costs contaminating final latency',
        dense_backend='original repository SDPA; D512 is not established FA3',
        mask_caveat='historical sparse local window differs from original native SDPA None-mask convention',
        exposure='Previously examined frozen samples, not fresh held-out confirmation')
    contract['fingerprint']=fingerprint(contract)
    root.mkdir(parents=True,exist_ok=True)
    path=root/'configuration.json'
    if path.exists():
        if json.loads(path.read_text())!=contract:raise ValueError('Cannot mix configurations in an existing run')
    else:
        atomic(path,contract);atomic(root/'manifest.json',rows)
    from .provenance import snapshot
    snapshot(root)
    return contract,rows


def shard_path(root,method,row):
    return root/'shards'/method/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')


def validate_shard(record,row,contract,method):
    for key,wanted in [('id',row['id']),('prompt_hash',row['prompt_hash']),('seed',row['seed']),
                       ('generation_budget',row['generation_budget']),('fingerprint',contract['fingerprint']),('method',method)]:
        if record.get(key)!=wanted:raise ValueError(f'Shard mismatch: {key}')
    if record.get('status')!='complete':raise ValueError('Incomplete shard')


def generate(adapter,row,method,contract,projections,*,warmup=False):
    if method=='native_dense':context=nullcontext((None,None))
    else:
        context=install(adapter,contract['library'],contract['policies'][method],
            mode='value' if method=='kernel_gaussian32' else 'blasst',projections=projections,
            torch_library=contract.get('torch_library'),
            collect=not warmup,profile=contract['profile'] and not warmup)
    with context as (binding,router):
        if binding is not None:_set_context(binding,row)
        torch.cuda.synchronize();start=time.perf_counter()
        generated=adapter.generate(_request(row))
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        if warmup:return None
        routing=[] if router is None else router.records()
        profile=[] if router is None else router.profile_records()
        work={} if router is None else dict(projected_tokens=router.cache.projected_tokens,
            reused_tokens=router.cache.reused_tokens,projection_madds=router.cache.projection_madds,
            peak_sketch_and_norm_bytes=router.cache.peak_storage_bytes,
            qk='computed for every traversed tile',block_softmax='all value tiles; retained BLASST tiles',
            projected_pv='rank32 on all value candidate tiles; absent for BLASST',
            full_pv='retained physical tiles only')
        record=dict(status='complete',fingerprint=contract['fingerprint'],method=method,
            id=row['id'],task=row['task'],prompt_hash=row['prompt_hash'],seed=row['seed'],
            generation_budget=row['generation_budget'],prediction=generated.text,
            completion_tokens=generated.completion_tokens,metadata=generated.metadata,
            score=score(row,generated.text),wall_seconds=seconds,profile=profile,work=work,
            effective_lambda=[] if router is None else router.lambda_records)
        if router is not None:
            record['projection_matrices']=router.cache.projections.manifest if router.mode=='value' else {}
        return record,routing


def run(root,library,*,split='final',limit=None,methods=METHODS,profile=False,torch_library=None):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        contract,rows=setup(root,library,split,limit,methods,profile,torch_library)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
        projections=Projections();done=0;errors=0;started=time.time();warmed=set()
        for index,row in enumerate(rows):
            prompt_length=len(adapter.encode_prompt(row['prompt'],extra={'thinking':False}))
            # Deterministic interleaving reduces fixed method-order clock bias.
            order=methods[index%len(methods):]+methods[:index%len(methods)]
            for method in order:
                dest=shard_path(root,method,row)
                if dest.exists():
                    validate_shard(json.loads(dest.read_text()),row,contract,method);done+=1;continue
                atomic(root/'status.json',dict(status='running',pid=os.getpid(),current_id=row['id'],method=method,
                    completed=done,expected=len(rows)*len(methods),errors=errors,started=started,updated=time.time()))
                try:
                    shape_key=(method,prompt_length)
                    if shape_key not in warmed:
                        generate(adapter,dict(row,generation_budget=1),method,contract,projections,warmup=True)
                        warmed.add(shape_key)
                    record,routing=generate(adapter,row,method,contract,projections)
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    if routing:
                        raw=dest.with_suffix('.routing.json.gz')
                        with gzip.open(raw,'wt') as f:json.dump(routing,f)
                        record['routing_path']=str(raw.resolve());record['routing_sha256']=sha(raw)
                    atomic(dest,record);done+=1
                    print(json.dumps(dict(id=row['id'],method=method,score=record['score'],seconds=record['wall_seconds'],completed=done)),flush=True)
                except Exception as exc:
                    errors+=1
                    error=dict(id=row['id'],method=method,error=repr(exc),traceback=traceback.format_exc(),time=time.time())
                    with (root/'failures.jsonl').open('a') as f:f.write(json.dumps(error)+'\n')
                    print(json.dumps(error),flush=True)
                    if 'device-side assert' in str(exc) or 'illegal memory access' in str(exc):raise
                    torch.cuda.empty_cache()
        atomic(root/'projection_matrices.json',projections.manifest)
        atomic(root/'status.json',dict(status='complete' if not errors else 'incomplete',completed=done,
            expected=len(rows)*len(methods),errors=errors,started=started,updated=time.time()))
    from .report import report
    report(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('run','report'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--library',type=Path)
    p.add_argument('--torch-library',type=Path)
    p.add_argument('--split',choices=('development','final'),default='final');p.add_argument('--limit',type=int)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS));p.add_argument('--profile',action='store_true')
    a=p.parse_args()
    if a.command=='report':
        from .report import report
        report(a.root)
    else:
        if a.library is None:p.error('--library is required for run')
        run(a.root,a.library,split=a.split,limit=a.limit,methods=a.methods,profile=a.profile,torch_library=a.torch_library)
