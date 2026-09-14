"""Resumable staged study; run from repo root with PYTHONPATH=src."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import torch
from dllm.models import create_adapter
from experiments.diffusion_gemma_solattn_blasst_multibench.config import MODEL,REVISION
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _rows,_write,_append,_install_dense,_request,_set_context,_fingerprint
from .config import ROOT,BASE,ORACLE,screening_configs
from .summaries import ProducerSummaries
from .routing import ScreeningAttention


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def heldout_selection(dev,pool,aime):
    used={r['id'] for r in dev};held=[]
    key=lambda r:hashlib.sha256(('preqk-heldout42|'+r['id']).encode()).hexdigest()
    for task in ('qasper','hotpotqa','gov_report','trec','passage_retrieval_en'):
        held+=sorted([r for r in pool if r['benchmark']=='longbench' and r['task']==task and r['id'] not in used],key=key)[:2]
    old_aime={r['source_id'] for r in pool if r['benchmark']=='aime24'}
    for row in sorted([r for r in aime if r['id'] not in old_aime],key=key)[:10]:
        number=int(row['id'].rsplit('-',1)[1])
        prompt='Solve the problem. Put the final integer answer in \\boxed{}.\n\n'+row['problem']
        held.append(dict(id='aime24/'+row['id'],source_id=row['id'],benchmark='aime24',task=f'AIME_{1+(number-1)//15}',
            prompt=prompt,prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),expected=str(row['expected_answer']),generation_budget=2048))
    held=[dict(r,paired_seeds=[314159,271828]) for r in held]
    assert len(held)==20 and not used & {r['id'] for r in held}
    assert not {r['prompt_hash'] for r in dev}&{r['prompt_hash'] for r in held}
    return held


def prepare(root=ROOT):
    old=json.loads((ORACLE/'setup.json').read_text());dev=old['samples']
    for row in dev:
        cached=old['cached'][row['id']]['dense'];path=Path(cached['path'])
        assert digest(path)==cached['sha256']
        d=json.loads(path.read_text())
        assert all(row[k]==d[k] for k in ('prompt','seed','generation_budget'))
    aime_path=Path('/home/exouser/ljy/Skills/nemo_skills/dataset/aime24/test.txt')
    held=heldout_selection(dev,_rows(BASE/'manifest.jsonl'),_rows(aime_path))
    protocol=dict(schema='preqk_v1',model=MODEL,revision=REVISION,development=dev,heldout_reserved=held,
        cached_dense={r['id']:old['cached'][r['id']]['dense'] for r in dev},decoding=old['protocol'],
        tile_size=64,region='prefix_plus_canvas',epsilon={'longbench':.03,'aime24':.05},
        uncertainty='paired prompt bootstrap; average paired seeds within prompt; preliminary preservation is not established noninferiority',
        screening=dict(steps=list(range(8)),canvas='first only',layers='all',heads='all',query_tiles='all',
            targets={'aime24':[.4,.5],'longbench':[.5,.75]},configs={k:v.to_dict() for k,v in screening_configs().items()},
            warmup='step0 masks are cold-reference diagnostics only; temporal screening uses steps1..7',
            limitation='early consecutive steps only; later-step generalization must be checked in online rollouts',
            last_mask='previous exact mass top-k mask, dense-history reference, not cheap online acquisition',
            value_cost='prefix V initialization scan once/cache; canvas norms at production; scalar norms pooled per step',
            proxy_cost='current full K summary scan explicitly paid; no QK used; candidate-specific costs measured separately before advancement'),
        gates=['retain positional categories only if better than equal-count random controls',
            'do not retain neighborhood changes with improved average but worse tails without revision',
            'reject value multiplier absent repeatable benefit; inspect norm variability and deletion risk',
            'advance at most two rankings plus plain last-mass control; compare dense-reference vs own sparse history',
            'test coarse allocation only after useful ranking; freeze before heldout; no heldout tuning',
            'profile complete online costs before claiming savings; diagnostic dense emulation saves no executed QK/PV'],
        source_hashes={'oracle_setup':digest(ORACLE/'setup.json'),'aime':digest(aime_path)})
    path=root/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise RuntimeError('immutable protocol differs')
    _write(path,protocol)
    _write(root/'manifest_audit.json',dict(development=len(dev),heldout_reserved=len(held),ids_disjoint=True,prompt_hashes_disjoint=True,
        development_tasks={task:sum(r['task']==task for r in dev) for task in sorted({r['task'] for r in dev})},
        heldout_not_evaluated=True,dense_cache_verified=True))
    return protocol


def fingerprint(root):
    names=('config.py','state.py','summaries.py','routing.py','run.py','audit.py')
    files=[Path(__file__).with_name(n) for n in names]+[Path('src/dllm/attention/blasst/core.py'),Path('src/dllm/models/adapters/diffusion_gemma.py')]
    return _fingerprint(dict(protocol=digest(root/'protocol.json'),code={str(p):digest(p) for p in files}))


def sample_key(row):return hashlib.sha256(row['id'].encode()).hexdigest()


def generate(adapter,row,trace_dir=None,steps=8):
    binding=_install_dense(adapter);producer=ProducerSummaries(adapter)
    router=ScreeningAttention(row['id'],row['benchmark'],trace_dir,producer,steps)
    binding.runtime.attention_override=router
    try:
        _set_context(binding,row);out=adapter.generate(_request(row))
    finally:producer.close();binding.close()
    return out,router,producer


def smoke(root=ROOT,adapter=None):
    setup=prepare(root)
    path=root/'smoke.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous.get('passed') and previous.get('fingerprint')==fingerprint(root):
            from .audit import validate_consecutive_coverage
            for test in previous['tests']:
                validate_consecutive_coverage(test['records'],layers=test['trace_audit']['layers'],heads=test['trace_audit']['heads'])
            return previous
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    adapter=adapter or create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    tests=[]
    for benchmark in ('longbench','aime24'):
        row=dict(next(r for r in setup['development'] if r['benchmark']==benchmark),generation_budget=32)
        binding=_install_dense(adapter)
        try:_set_context(binding,row);dense=adapter.generate(_request(row))
        finally:binding.close()
        out,router,producer=generate(adapter,row,root/'smoke_traces'/benchmark,steps=2)
        assert out.completion_tokens==dense.completion_tokens,'dense diagnostic parity failed'
        assert {r['attention_type'] for r in router.records}=={'local','global'}
        assert len({r['layer'] for r in router.records})==adapter.model.config.text_config.num_hidden_layers
        assert all(len(r['names'])==48 for r in router.records)
        from .audit import validate_consecutive_coverage
        coverage=validate_consecutive_coverage(router.records,layers=adapter.model.config.text_config.num_hidden_layers,
            heads=adapter.model.config.text_config.num_attention_heads)
        tests.append(dict(benchmark=benchmark,passed=True,attention_calls=router.calls,coverage=sorted(router.coverage),
            trace_audit=coverage,records=router.records,state_bytes=router.max_state_bytes,completion_tokens=len(out.completion_tokens)))
    result=dict(passed=True,fingerprint=fingerprint(root),tests=tests);_write(path,result);return result


def screen(root=ROOT,adapter=None):
    setup=prepare(root);fp=fingerprint(root);audit=json.loads((root/'smoke.json').read_text())
    assert audit['passed'] and audit['fingerprint']==fp,'matching CUDA smoke required'
    adapter=adapter or create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    failures=[]
    for row in sorted(setup['development'],key=lambda r:(r['generation_budget'],r['prompt_tokens'],r['id'])):
        key=sample_key(row);path=root/'screen'/'shards'/f'{key}.json'
        if path.exists():
            shard=json.loads(path.read_text());assert shard['fingerprint']==fp
            assert all(Path(r['path']).exists() for r in shard['records']);continue
        _write(root/'progress.json',dict(stage='screen',id=row['id'],started=time.time()))
        try:
            start=time.monotonic();out,router,producer=generate(adapter,row,root/'screen'/'traces'/key)
            dense=json.loads(Path(setup['cached_dense'][row['id']]['path']).read_text())
            assert out.completion_tokens==dense['completion_tokens'],'full-budget cached dense parity failed'
            from .audit import validate_consecutive_coverage
            coverage=validate_consecutive_coverage(router.records,layers=adapter.model.config.text_config.num_hidden_layers,
                heads=adapter.model.config.text_config.num_attention_heads)
            _write(path,dict(id=row['id'],benchmark=row['benchmark'],task=row['task'],fingerprint=fp,
                prompt_hash=row['prompt_hash'],seed=row['seed'],generation_budget=row['generation_budget'],
                prediction=out.text,completion_tokens=out.completion_tokens,records=router.records,
                max_state_bytes=router.max_state_bytes,attention_calls=router.calls,coverage=sorted(router.coverage),trace_audit=coverage,
                prefix_value_elements=producer.prefix_value_elements,canvas_value_elements=producer.canvas_value_elements,
                diagnostic_e2e_seconds=time.monotonic()-start,executed_qk_saved=0,executed_pv_saved=0))
            print(time.strftime('%FT%TZ',time.gmtime()),'screen completed',row['id'],flush=True)
        except Exception:
            error=dict(stage='screen',id=row['id'],traceback=traceback.format_exc());failures.append(error)
            _append(root/'failures.jsonl',error);print(error,flush=True)
        finally:torch.cuda.empty_cache()
    _write(root/'screen_status.json',dict(finished=True,failures=failures,expected=16,completed=len(list((root/'screen'/'shards').glob('*.json')))))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','smoke','screen','screen-pipeline','report'))
    p.add_argument('--output',type=Path,default=ROOT);args=p.parse_args()
    if args.command=='screen-pipeline':
        prepare(args.output)
        adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
        smoke(args.output,adapter);screen(args.output,adapter)
        from .report import report
        report(args.output)
    elif args.command=='report':
        from .report import report
        report(args.output)
    else:globals()[args.command](args.output)


if __name__=='__main__':main()
