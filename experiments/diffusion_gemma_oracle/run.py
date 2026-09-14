"""CLI: prepare, smoke, run, report. Run from repository root with PYTHONPATH=src."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import torch
from dllm.models import create_adapter
from experiments.diffusion_gemma_solattn_blasst_multibench.config import MODEL, REVISION, condition_map
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _rows, _write, _append, _install_dense, _request, _set_context, _fingerprint
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_runner import shard_path
from .routing import OracleAttention

BASE = Path('results/diffusion_gemma_solattn_blasst_multibench_controlled')
ROOT = Path('results/diffusion_gemma_oracle')


def conditions():
    configs = {'dense_observer': dict(observer=True)}
    for signal in ('qk','mass','contribution','blasst'):
        for s in ((.1,.25,.5,.75,.9) if signal != 'blasst' else (.25,.5,.75,.9)):
            configs[f'{signal}_topk_s{int(s*100)}'] = dict(signal=signal,mode='topk',amount=s)
    for signal in ('mass','contribution'):
        for p in (.9,.95,.99):
            configs[f'{signal}_topp_p{int(p*100)}'] = dict(signal=signal,mode='topp',amount=p)
    return configs


def prepare(root=ROOT):
    rows = _rows(BASE/'manifest.jsonl'); selected = []
    for task in ('qasper','hotpotqa','gov_report','trec','passage_retrieval_en'):
        selected += sorted([r for r in rows if r['benchmark']=='longbench' and r['task']==task],key=lambda r:r['id'])[:2]
    aime = sorted([r for r in rows if r['benchmark']=='aime24'],key=lambda r:r['id'])
    # Interleaved positions span the existing independently selected AIME subset.
    selected += [aime[i] for i in (0,2,4,5,7,9)]
    cache = {}
    for r in selected:
        cache[r['id']] = {}
        for name in condition_map():
            p = shard_path(BASE,name,r['id']); d = json.loads(p.read_text())
            assert all(d[k] == r[k] for k in ('prompt','seed','generation_budget'))
            cache[r['id']][name] = dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    setup = dict(schema='oracle_v1',samples=selected,cached=cache,model=MODEL,revision=REVISION,
        conditions=conditions(),tile_size=64,region='prefix_plus_canvas',
        protocol=json.loads((BASE/'decoding_protocol.json').read_text()),
        observer_steps=[0,1,4,12,24,47],observer_calls='first call per layer/step per sample',
        selection='task-local lexicographic ID; 2 each LongBench; AIME sorted positions 0,2,4,5,7,9; no score selection')
    if (root/'setup.json').exists() and json.loads((root/'setup.json').read_text()) != setup:
        raise RuntimeError('existing setup differs')
    _write(root/'setup.json',setup)
    return setup


def fingerprint(root):
    return _fingerprint(dict(setup=json.loads((root/'setup.json').read_text()),
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__),Path(__file__).with_name('routing.py'),
            Path('src/dllm/attention/blasst/core.py'),Path('src/dllm/models/adapters/diffusion_gemma.py'))}))


def generate(adapter,row,config):
    binding = _install_dense(adapter); router = OracleAttention(**config)
    binding.runtime.attention_override = router
    try:
        _set_context(binding,row); out = adapter.generate(_request(row))
    finally: binding.close()
    return out,router


def smoke(root=ROOT):
    setup = prepare(root)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable: cannot run required H100 smoke')
    adapter = create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    tests = []
    for benchmark in ('longbench','aime24'):
        row = dict(next(r for r in setup['samples'] if r['benchmark']==benchmark),generation_budget=32)
        binding = _install_dense(adapter)
        try:
            _set_context(binding,row); dense=adapter.generate(_request(row))
        finally: binding.close()
        for config in (dict(observer=True),dict(amount=0.),dict(signal='contribution',amount=.5),dict(mode='topp',amount=.95)):
            out,router = generate(adapter,row,config)
            unpruned = config.get('observer') or config.get('amount') == 0.
            assert out.completion_tokens and router.records
            if unpruned: assert out.completion_tokens == dense.completion_tokens
            assert {r['attention_type'] for r in router.records} == {'local','global'}
            assert len({r['layer'] for r in router.records}) == adapter.model.config.text_config.num_hidden_layers
            tests.append(dict(benchmark=benchmark,config=config,passed=True,calls=router.calls))
    _write(root/'smoke.json',dict(passed=True,fingerprint=fingerprint(root),tests=tests))


def run(root=ROOT,only=None):
    setup = prepare(root); fp = fingerprint(root)
    audit = json.loads((root/'smoke.json').read_text())
    assert audit['passed'] and audit['fingerprint']==fp, 'matching CUDA smoke required'
    adapter = create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    failures = []
    for name,config in conditions().items():
        if only and name not in only: continue
        for row in sorted(setup['samples'],key=lambda r:(r['generation_budget'],r['id'])):
            path = shard_path(root,name,row['id'])
            if path.exists():
                assert json.loads(path.read_text())['fingerprint']==fp
                continue
            _write(root/'progress.json',dict(condition=name,id=row['id'],started=time.time()))
            try:
                out,router = generate(adapter,row,config)
                if name=='dense_observer':
                    d=json.loads(Path(setup['cached'][row['id']]['dense']['path']).read_text())
                    assert out.completion_tokens==d['completion_tokens'], 'cached dense parity failed'
                _write(path,dict(id=row['id'],condition=name,fingerprint=fp,prediction=out.text,
                    completion_tokens=out.completion_tokens,records=router.records,rankings=router.rankings))
                print(time.strftime('%FT%TZ',time.gmtime()),name,row['id'],'completed',flush=True)
            except Exception:
                error=dict(condition=name,id=row['id'],traceback=traceback.format_exc())
                failures.append(error); _append(root/'failures.jsonl',error)
                print(error,flush=True)
            finally: torch.cuda.empty_cache()
    _write(root/'run_status.json',dict(finished=True,failures=failures))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','smoke','run','report'])
    parser.add_argument('--output',type=Path,default=ROOT);parser.add_argument('--conditions',nargs='+')
    args=parser.parse_args()
    if args.command=='run': run(args.output,args.conditions)
    elif args.command=='report':
        from .report import report
        report(args.output)
    else: globals()[args.command](args.output)


if __name__=='__main__': main()
