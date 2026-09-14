"""Resumable six-question calibration and full 30-question AIME26 evaluation."""
import argparse
import json
import math
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from dllm.models import create_adapter
from dllm.attention.blasst.core import Blasst2DConfig,Blasst2DStats,blasst_2d_attention_forward
from experiments.diffusion_gemma_aime30.protocol import MODEL,REVISION,sha,prompt,numeric_score
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense,_set_context,_request,_write,_append,_fingerprint
from experiments.diffusion_gemma_solattn_blasst_multibench.diagnostics import CheckedAttention
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_report import aggregate
from .calibration import Collector,fit

ROOT=Path('results/diffusion_gemma_aime26_blasst_aggressive')
DATA_REV='79037aebdb6580008fb960d17cb21fd3099083e3'
TARGETS=(.25,.5,.75,.9)

def prepare(root):
    from huggingface_hub import hf_hub_download
    source=Path(hf_hub_download('math-ai/aime26','aime2026.jsonl',repo_type='dataset',revision=DATA_REV))
    raw=[json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    if len(raw)!=30 or len({r['id'] for r in raw})!=30: raise ValueError('expected full AIME26')
    # Three per 15-question paper, one from each consecutive five-question band.
    ordered=sorted(raw,key=lambda r:int(r['id']))
    selected={min(ordered[i:i+5],key=lambda r:sha(f'42|{r["id"]}'))['id'] for i in range(0,30,5)}
    tok=create_adapter('diffusion_gemma',MODEL,device='cpu',precision='float32',revision=REVISION).load_tokenizer()
    rows=[]
    for r in ordered:
        p=prompt(r['problem'],[]);ids=tok.encode_prompt(p,{'thinking':False})
        rows.append(dict(id=f'aime26/{r["id"]}',source_id=r['id'],benchmark='aime26',task='AIME26',
            prompt=p,prompt_hash=sha(p),prompt_tokens=ids,expected=str(r['answer']),calibration=r['id'] in selected,
            generation_budget=2048,seed=42))
    setup=dict(dataset_revision=DATA_REV,dataset_sha256=sha(source.read_bytes()),model=MODEL,revision=REVISION,
        tile_size=[128,64],calibration_ids=sorted(selected),total=30,heldout=24,budget=2048,
        decoding='native seeded schedule 0.4–0.8, thinking=False, seed=42, canvas256, max48 denoising steps',
        threshold='previous-running-maximum; all valid queries must skip; lambda>1 enabled; prefix+canvas eligible',
        calibration='paper exponential fit separately local/global, dense-scale bisection, sparse verification/refinement (up to 8 rounds, 2pp tolerance)',
        evaluation='full30 includes6 calibration questions; heldout24 also reported; no accuracy used in threshold selection')
    root.mkdir(parents=True,exist_ok=True)
    for name,data in [('manifest.json',rows),('setup.json',setup)]:
        path=root/name
        if path.exists() and json.loads(path.read_text())!=data: raise ValueError('incompatible manifest/setup')
        _write(path,data)
    return rows

def generate(adapter,row,scales=None,collect=False):
    binding=_install_dense(adapter)
    if collect:
        stats=Collector();fn=stats
    else:
        stats=Blasst2DStats(record_layers=True,record_heads=False)
        policy={k:dict(alpha=math.exp(v),gamma=0.,target_sparsity=0.) for k,v in scales.items()} if scales else None
        binding.runtime.config=Blasst2DConfig(enable_blasst_2d=True,q_tile_size=128,kv_tile_size=64,
            allow_lambda_above_one=True,collect_blasst_stats=True,apply_blasst_mask=scales is not None,
            length_aware_policy=policy)
        binding.runtime.stats=stats;fn=blasst_2d_attention_forward
    checked=CheckedAttention(fn);binding.runtime.attention_override=checked
    try:
        _set_context(binding,row);out=adapter.generate(_request(row))
        if out.prompt_tokens!=row['prompt_tokens']: raise RuntimeError('prompt token mismatch')
        calls=stats.calls if collect else list(stats.per_step.values())
        layers=calls if collect else list(stats.per_layer.values())
        if {c['layer'] for c in layers}!=set(range(30)): raise RuntimeError('missing layer coverage')
        return dict(id=row['id'],prompt_hash=row['prompt_hash'],seed=row['seed'],generation_budget=row['generation_budget'],
            prediction=out.text,completion_tokens=out.completion_tokens,termination_reason=out.termination_reason,
            generation_metadata=out.metadata,score=numeric_score(out.text,row['expected']),calls=calls,per_layer=layers,
            log_scales=scales,finite_calls=checked.calls),stats.arrays() if collect else None
    finally: binding.close()

def cached(adapter,row,path,fingerprint,scales=None,collect=False):
    npz=path.with_suffix('.npz')
    if path.exists() and (not collect or npz.exists()):
        r=json.loads(path.read_text())
        if r['fingerprint']!=fingerprint or r['log_scales']!=scales: raise RuntimeError('incompatible cached shard')
        return r,dict(np.load(npz)) if collect else None
    if path.is_relative_to(ROOT):
        _write(ROOT/'active_generation.json',dict(id=row['id'],shard=str(path),started=time.time(),log_scales=scales))
    r,a=generate(adapter,row,scales,collect);r['fingerprint']=fingerprint
    path.parent.mkdir(parents=True,exist_ok=True)
    if a is not None: np.savez_compressed(npz,**a)
    _write(path,r);print(time.strftime('%FT%TZ',time.gmtime()),str(path),flush=True)
    return r,a

def execute(root):
    rows=prepare(root)
    files=list(Path(__file__).parent.glob('*.py'))+[Path('src/dllm/attention/blasst/core.py'),Path('src/dllm/models/adapters/diffusion_gemma.py')]
    config=dict(setup=json.loads((root/'setup.json').read_text()),manifest_sha=sha((root/'manifest.json').read_bytes()),
        code={str(p):sha(p.read_bytes()) for p in files},torch=torch.__version__,transformers=__import__('transformers').__version__)
    fp=_fingerprint(config)
    if (root/'experiment.json').exists() and json.loads((root/'experiment.json').read_text())!=config: raise RuntimeError('code/protocol changed')
    _write(root/'experiment.json',config)
    adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    _write(root/'model_runtime.json',dict(generation=adapter.model.generation_config.to_dict(),model=adapter.model.config.to_dict()))
    if not (root/'smoke.json').exists():
        r=dict(rows[0],generation_budget=32)
        binding=_install_dense(adapter)
        try: _set_context(binding,r);ref=adapter.generate(_request(r))
        finally: binding.close()
        dense,_=generate(adapter,r,collect=True)
        # lambda always >1 for this smoke; layer type and finite checks are in generate.
        sparse,_=generate(adapter,r,{'local':math.log(1e5),'global':math.log(1e5)})
        passed=ref.completion_tokens==dense['completion_tokens'] and sparse['finite_calls']>0
        _write(root/'smoke.json',dict(passed=passed,fingerprint=fp,sparse=aggregate(sparse['calls'])))
        if not passed: raise RuntimeError('smoke failed')
    elif not json.loads((root/'smoke.json').read_text()).get('passed'): raise RuntimeError('smoke failed')
    cal=[r for r in rows if r['calibration']]
    if not (root/'policy.json').exists():
        parts={k:[] for k in ('local','global')}
        for row in cal:
            _,a=cached(adapter,row,root/'dense'/f'{row["source_id"]}.json',fp,collect=True)
            for k in parts: parts[k].append(a[k])
        fits={k:fit(v) for k,v in parts.items()};_write(root/'paper_fit.json',fits)
        policy={}
        for target in TARGETS:
            scales={k:fits[k]['targets'][str(target)]['log_scale'] for k in parts}
            observations={k:[] for k in parts};rounds=[];best=None
            for iteration in range(8):
                outputs=[]
                for row in cal:
                    p=root/'verification'/f's{int(target*100)}'/f'round{iteration}'/f'{row["source_id"]}.json'
                    result,_=cached(adapter,row,p,fp,scales);outputs.append(result)
                stats=aggregate(c for r in outputs for c in r['calls'])
                measured={k:stats[k]['full_tile_sparsity'] for k in parts}
                error=max(abs(v-target) for v in measured.values())
                record=dict(log_scales=dict(scales),measured=measured,error=error,round=iteration)
                rounds.append(record)
                if best is None or error<best['error']: best=record
                if error<=.02: break
                for k in parts:
                    observations[k].append((scales[k],measured[k]))
                    below=[x for x,s in observations[k] if s<target];above=[x for x,s in observations[k] if s>=target]
                    if below and above: scales[k]=(max(below)+min(above))/2
                    else: scales[k]+=max(-8.,min(8.,(target-measured[k])*max(1.,fits[k]['gamma'])))
                    scales[k]=min(80.,max(-20.,scales[k]))
            policy[str(target)]=dict(best,rounds=rounds,within_two_points=best['error']<=.02)
        _write(root/'policy.json',dict(fingerprint=fp,targets=policy))
    policy=json.loads((root/'policy.json').read_text())
    if policy['fingerprint']!=fp: raise RuntimeError('policy provenance mismatch')
    failures=[]
    for name,target in [('dense',None)]+[(f'blasst_s{int(t*100)}',t) for t in TARGETS]:
        scales=None if target is None else policy['targets'][str(target)]['log_scales']
        for row in rows:
            _write(root/'progress.json',dict(condition=name,id=row['id'],started=time.time()))
            try:
                cached(adapter,row,root/name/f'{row["source_id"]}.json',fp,scales,collect=target is None)
            except Exception as exc:
                failure=dict(condition=name,id=row['id'],error=repr(exc),traceback=traceback.format_exc())
                failures.append(failure);_append(root/'failures.jsonl',failure)
    _write(root/'status.json',dict(finished=True,failures=failures))
    from .report import report
    report(root)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','report']);p.add_argument('--output-dir',type=Path,default=ROOT)
    args=p.parse_args()
    if args.command=='prepare': prepare(args.output_dir)
    elif args.command=='run': execute(args.output_dir)
    else:
        from .report import report
        report(args.output_dir)

if __name__=='__main__': main()
