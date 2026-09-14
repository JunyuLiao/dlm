"""Score-blind fresh sampling and a protocol frozen after disjoint development checks."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import copy,deepcopy
import ast
from functools import lru_cache
import json
from pathlib import Path
from threading import local

from experiments.diffusion_gemma_value_aware_followup.protocol import MODEL, REVISION, LB_DATA, LB_REVISION
from experiments.diffusion_gemma_value_aware.protocol import frozen_write, sha
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _fingerprint
from . import nemo

ROOT = Path('results/diffusion_gemma_longbench_v2_100_nemo')
PARENT = Path('results/diffusion_gemma_value_aware_gpu_50_75')
METHODS = ('blasst_original','blasst_aggressive','mass','risk')
TARGETS = (.5,.75)
CONFIGS = {name:dict(method='blasst') for name in METHODS[:2]}
CONFIGS.update(mass=dict(method='mass'),risk=dict(method='risk',pooling='mean'))
SEED = 20260914
INPUT_CANDIDATES = (32768,49152,65536,131072)
OUTPUT_CANDIDATES = (4096,8192,16384)


def apportion(counts, total):
    """Hamilton/largest-remainder proportional quotas, alphabetical tie break."""
    size = sum(counts.values())
    quotas = {k:(v*total)//size for k,v in counts.items()}
    order = sorted(counts,key=lambda k:(-(counts[k]*total % size),k))
    for k in order[:total-sum(quotas.values())]:
        quotas[k] += 1
    return quotas


@lru_cache
def items():
    return json.loads(LB_DATA.read_text())


def select(data, excluded):
    if len({r['_id'] for r in data}) != len(data):
        raise ValueError('Duplicate dataset source IDs')
    domains = Counter(r['domain'] for r in data)
    quota = apportion(domains,100)
    occupied = set(excluded)
    groups = dict(development=[],calibration=[],final=[])
    for domain in sorted(domains):
        # One development question per domain, plus one easy and one hard
        # calibration question. Neither set ever participates in final scoring.
        pool = sorted((r for r in data if r['domain']==domain and r['_id'] not in occupied),
                      key=lambda r:sha(f'lb100/dev/{SEED}/{r["_id"]}'))
        dev = pool[0]; groups['development'].append(dev); occupied.add(dev['_id'])
        for difficulty in ('easy','hard'):
            pool = sorted((r for r in data if r['domain']==domain and r['difficulty']==difficulty and r['_id'] not in occupied),
                          key=lambda r:sha(f'lb100/calibration/{SEED}/{r["_id"]}'))
            cal = pool[0]; groups['calibration'].append(cal); occupied.add(cal['_id'])
    subquotas = {}
    for domain in sorted(domains):
        counts = Counter(r['sub_domain'] for r in data if r['domain']==domain)
        subquotas[domain] = apportion(counts,quota[domain])
        for subdomain,n in sorted(subquotas[domain].items()):
            pool = sorted((r for r in data if r['domain']==domain and r['sub_domain']==subdomain and r['_id'] not in occupied),
                          key=lambda r:sha(f'lb100/final/{SEED}/{r["_id"]}'))
            if len(pool)<n:
                raise ValueError(f'Insufficient fresh examples: {domain}/{subdomain}')
            groups['final'].extend(pool[:n]); occupied.update(r['_id'] for r in pool[:n])
    return groups,quota,subquotas


def selection(root=ROOT):
    path = root/'selection.json'
    history = json.loads((PARENT/'setup.json').read_text())
    excluded = {r['source_id'] for split in ('final','calibration','development') for r in history[split]
                if r['benchmark']=='longbench_v2'}
    chosen,quotas,subquotas = select(items(),excluded)
    fields = ('_id','domain','sub_domain','difficulty','length')
    result = dict(seed=SEED,dataset_size=len(items()),dataset_revision=LB_REVISION,
        dataset_sha256=sha(LB_DATA.read_bytes()),parent_setup_sha256=sha((PARENT/'setup.json').read_bytes()),
        excluded_prior_ids=sorted(excluded),domain_population=dict(Counter(r['domain'] for r in items())),
        domain_quotas=quotas,subdomain_quotas=subquotas,
        groups={k:[{f:r[f] for f in fields} for r in v] for k,v in chosen.items()},
        rule='Fresh IDs only; largest-remainder domain and within-domain subtask quotas from all503; SHA256 ordering within subtask, no short-first/answer/score selection',
        input_candidates=list(INPUT_CANDIDATES),output_candidates=list(OUTPUT_CANDIDATES),
        pilot_rule='Highest memory-tested input cap; lowest output budget with zero length-limited and at least5/6 parsable dense development answers; never select by correctness')
    frozen_write(path,result)
    return result


def render(adapter,item,input_budget,output_budget,split):
    # Repository excerpts can literally document multimodal control tokens.
    # Escape only model-reserved spellings in source text, never chat framing.
    # This keeps every sampled problem without pretending image data exists.
    raw_hash=sha(nemo.prompt(item));escaped={};item=dict(item)
    reserved=tuple(getattr(adapter.tokenizer,'all_special_tokens',()))
    for field in ('context','question','choice_A','choice_B','choice_C','choice_D'):
        for token in reserved:
            if token and token.startswith('<') and token.endswith('>') and token in item[field]:
                escaped[token]=escaped.get(token,0)+item[field].count(token)
                item[field]=item[field].replace(token,'&lt;'+token[1:-1]+'&gt;')
    full = nemo.prompt(item)
    original_ids = adapter.encode_prompt(full,{'thinking':False})
    text,ids = full,original_ids
    context_ids = None
    if len(ids)>input_budget:
        # Truncate context only: preserve all question, choices and NeMo answer
        # instructions. Same tokenizer-aware head/tail policy for every method.
        context_ids = adapter.tokenizer.encode(item['context'],add_special_tokens=False)
        overhead = len(adapter.encode_prompt(nemo.prompt(dict(item,context='')),{'thinking':False}))
        keep = input_budget-overhead-128
        while keep>0:
            context = adapter.tokenizer.decode(context_ids[:keep//2]+context_ids[-(keep-keep//2):],skip_special_tokens=True)
            text = nemo.prompt(dict(item,context=context))
            ids = adapter.encode_prompt(text,{'thinking':False})
            if len(ids)<=input_budget:
                break
            keep -= 128
        else:
            raise ValueError('Question/instructions do not fit the input budget')
    return dict(id=f'longbench_v2/{item["_id"]}',source_id=item['_id'],benchmark='longbench_v2',
        task=item['domain'],sub_domain=item['sub_domain'],difficulty=item['difficulty'],length_band=item['length'],
        prompt=text,prompt_hash=sha(text),prompt_tokens=ids,original_prompt_hash=sha(full),
        original_prompt_tokens=len(original_ids),truncated=len(original_ids)>input_budget,
        expected=item['answer'],seed=42,generation_budget=output_budget,split=split,calibration=split=='calibration',
        previous_manifest_member=False,literal_special_token_escapes=escaped,raw_source_prompt_hash=raw_hash)


def rows_for(adapter,split,input_budget,output_budget,root=ROOT):
    selected = selection(root)
    byid = {r['_id']:r for r in items()}
    worker=local();nemo.initialize()
    current_code=Path(__file__).read_text();code_sha=sha(current_code)
    archived=root/'pre_thread_isolation_code/protocol.py'
    previous_sha=None
    if archived.exists():
        def render_ast(source):return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='render'))
        if render_ast(archived.read_text())!=render_ast(current_code):raise ValueError('Prior rendered rows use different semantics')
        previous_sha=sha(archived.read_bytes())
    def private_adapter():
        if not hasattr(worker,'adapter'):
            # HF mutates Rust tokenizer padding state. Each worker must own a
            # distinct processor/tokenizer; never deepcopy the model weights.
            worker.adapter=copy(adapter)
            worker.adapter.processor=deepcopy(adapter.processor)
            worker.adapter.tokenizer=worker.adapter.processor.tokenizer
        return worker.adapter
    def one(chosen):
        from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
        identity=dict(id=chosen['_id'],split=split,input_budget=input_budget,output_budget=output_budget,
            dataset=selected['dataset_sha256'],protocol=code_sha,nemo=sha(Path(nemo.__file__).read_bytes()))
        path=root/'prepared_rows'/f'{_fingerprint(identity)}.json'
        if path.exists():
            data=json.loads(path.read_text())
            if data['identity']!=identity:raise ValueError('Prepared-row cache mismatch')
            return data['row']
        if previous_sha is not None:
            old_identity=dict(identity,protocol=previous_sha)
            source=root/'prepared_rows'/f'{_fingerprint(old_identity)}.json'
            if source.exists():
                data=json.loads(source.read_text());row=data['row']
                if data['identity']!=old_identity or sha(row['prompt'])!=row['prompt_hash']:
                    raise ValueError('Prior prepared cache identity changed')
                if private_adapter().encode_prompt(row['prompt'],{'thinking':False})!=row['prompt_tokens']:
                    raise ValueError('Prior concurrently prepared tokens do not reproduce in isolated tokenizer')
                _write(path,dict(identity=identity,row=row,imported_source=dict(path=str(source),sha256=sha(source.read_bytes()),
                    renderer_ast_identical=True,exact_isolated_token_replay=True)))
                print('revalidated prepared cache',split,row['id'],flush=True)
                return row
        row=render(private_adapter(),byid[chosen['_id']],input_budget,output_budget,split)
        _write(path,dict(identity=identity,row=row))
        print('prepared',split,row['id'],len(row['prompt_tokens']),row['literal_special_token_escapes'],flush=True)
        return row
    # executor.map preserves manifest order regardless of completion order.
    with ThreadPoolExecutor(max_workers=4) as pool:return list(pool.map(one,selected['groups'][split]))


def audit_pilot_reuse(root,adapter=None):
    """Explicitly prove the pre-escape pilot prompts/output settings are unchanged."""
    from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
    pilot=json.loads((root/'pilot/decision.json').read_text())
    if not pilot['passed']:raise ValueError('Pilot did not pass')
    source_dir=Path(__file__).resolve().parent
    archived=root/'pilot/pre_escape_code'
    paths=[source_dir/n for n in ('pilot.py','protocol.py','runner.py','nemo.py')]
    from .pilot import identity
    fresh=identity(root)==pilot['fingerprint']
    old_sources={str(p):sha((p if fresh else archived/p.name).read_bytes()) for p in paths}
    for p in (PARENT/'execution_contract.json',PARENT/'final_configs/mass_s75.json'):old_sources[str(p)]=sha(p.read_bytes())
    old_fp=_fingerprint(dict(selection=selection(root),sources=old_sources,nemo_revision=nemo.REVISION))
    if old_fp!=pilot['fingerprint']:raise ValueError('Pilot source provenance mismatch')
    if not fresh and any(sha((archived/n).read_bytes())!=sha((source_dir/n).read_bytes()) for n in ('runner.py','nemo.py')):
        raise ValueError('Pilot generation or scorer changed; cannot reuse')
    for p,digest in pilot['sources'].items():
        if sha(Path(p).read_bytes())!=digest:raise ValueError('Cached pilot evidence changed')
    parent=json.loads((PARENT/'execution_contract.json').read_text())
    for p,digest in parent['sources'].items():
        if sha(Path(p).read_bytes())!=digest:raise ValueError('Validated inference code changed')
    if adapter is None:
        from dllm.models import create_adapter
        adapter=create_adapter('diffusion_gemma',MODEL,device='cpu',precision='float32',revision=REVISION).load_tokenizer()
    rows=rows_for(adapter,'development',pilot['input_budget'],pilot['output_budget'],root)
    for row in rows:
        path=root/'pilot'/f'dense_{pilot["input_budget"]}_{pilot["output_budget"]}'/(row['source_id']+'.json')
        out=json.loads(path.read_text())
        if (row['literal_special_token_escapes'] and not fresh) or any(row[k]!=out[k] for k in ('id','prompt_hash','seed','generation_budget')):
            raise ValueError('Pilot prompt/settings changed; old outputs cannot be reused')
    proof=dict(passed=True,original_pilot_fingerprint=old_fp,development_ids=[r['id'] for r in rows],
        original_inference_sources=parent['sources'],
        sources={str(p if fresh else archived/p.name):sha((p if fresh else archived/p.name).read_bytes()) for p in paths},
        new_protocol_sha256=sha(Path(__file__).read_bytes()),
        reason='All6 development prompts and seeds/budgets match cached outputs; unchanged pre-escape pilot requires zero escapes; no repeated GPU inference')
    _write(root/'pilot/reuse_audit.json',proof)
    return proof


def audit(setup):
    expected = dict(final=100,calibration=12,development=6)
    seen,hashes = set(),set()
    for split,n in expected.items():
        group = setup[split]
        if len(group)!=n:
            raise ValueError('Sample count mismatch')
        for r in group:
            if r['id'] in seen or r['prompt_hash'] in hashes or r['split']!=split:
                raise ValueError('Duplicate/overlapping split')
            if r['source_id'] in setup['selection']['excluded_prior_ids']:
                raise ValueError('Prior-experiment example reused')
            if sha(r['prompt'])!=r['prompt_hash'] or len(r['prompt_tokens'])>setup['input_budget']:
                raise ValueError('Prompt identity/budget mismatch')
            if r['generation_budget']!=setup['output_budget'] or r['expected'] not in tuple('ABCD'):
                raise ValueError('Output protocol mismatch')
            seen.add(r['id']);hashes.add(r['prompt_hash'])
    if dict(Counter(r['task'] for r in setup['final']))!=setup['selection']['domain_quotas']:
        raise ValueError('Domain quotas changed')
    for d,quotas in setup['selection']['subdomain_quotas'].items():
        if dict(Counter(r['sub_domain'] for r in setup['final'] if r['task']==d))!=quotas:
            raise ValueError('Subtask quotas changed')
    return dict(passed=True,counts=expected,expected_results=900,
        domain_counts=dict(Counter(r['task'] for r in setup['final'])),
        difficulty_counts=dict(Counter(r['difficulty'] for r in setup['final'])),
        length_counts=dict(Counter(r['length_band'] for r in setup['final'])),
        truncated=sum(r['truncated'] for r in setup['final']))


def prepare(root=ROOT):
    path = root/'setup.json'
    if path.exists():
        result=json.loads(path.read_text());audit(result);return result
    pilot=json.loads((root/'pilot'/'decision.json').read_text())
    if not pilot['passed']:
        raise ValueError('Development completion/memory gate not passed')
    from dllm.models import create_adapter
    adapter=create_adapter('diffusion_gemma',MODEL,device='cpu',precision='float32',revision=REVISION).load_tokenizer()
    reuse=audit_pilot_reuse(root,adapter)
    chosen=selection(root)
    result=dict(schema='longbench_v2_100_nemo_v1',model=MODEL,revision=REVISION,precision='bfloat16',
        input_budget=pilot['input_budget'],output_budget=pilot['output_budget'],tile_size=[128,64],regions='prefix_plus_canvas',
        selection=chosen,targets=list(TARGETS),configs=CONFIGS,
        conditions=['dense']+[f'{n}_s{int(t*100)}' for t in TARGETS for n in METHODS],
        scoring='Pinned NeMo-Skills eval_mcq defaults and eval/longbench/default prompt; native DiffusionGemma generation backend',
        decoding=dict(seed=42,thinking=False,temperature='native0.4-0.8 schedule; not greedy',canvas=256,max_denoising_steps=48),
        literal_special_token_policy='HTML-escape model-reserved <...> spellings in source text only; preserve sampled IDs and native chat framing',
        sources={**reuse['sources'],str(root/'pilot/reuse_audit.json'):sha((root/'pilot/reuse_audit.json').read_bytes()),
            **{str(p):sha(p.read_bytes()) for p in [LB_DATA,root/'selection.json',root/'pilot'/'decision.json',*nemo.sources()]}})
    for split in ('final','calibration','development'):
        result[split]=rows_for(adapter,split,result['input_budget'],result['output_budget'],root)
    preparation_sources=list((root/'prepared_rows').glob('*.json'))
    prior_code=root/'pre_thread_isolation_code/protocol.py'
    if prior_code.exists():preparation_sources.append(prior_code)
    result['sources'].update({str(p):sha(p.read_bytes()) for p in preparation_sources})
    frozen_write(path,result);frozen_write(root/'dataset_audit.json',audit(result))
    for split in ('final','calibration','development'):
        frozen_write(root/f'{split}_manifest.json',result[split])
    return result


def execution(root=ROOT,freeze=False):
    import torch,triton,transformers
    prepare(root)
    parent=json.loads((PARENT/'execution_contract.json').read_text())
    for path,digest in parent['sources'].items():
        if sha(Path(path).read_bytes())!=digest:
            raise ValueError('Validated parent GPU/attention source changed')
    own=[Path(__file__),Path(__file__).with_name('nemo.py'),Path(__file__).with_name('runner.py')]
    data=dict(schema='longbench100_execution_v1',parent_fingerprint=parent['fingerprint'],
        previous_fingerprint=parent['fingerprint'],setup_sha256=sha((root/'setup.json').read_bytes()),
        sources={**parent['sources'],**{str(p):sha(p.read_bytes()) for p in own}},
        runtime=dict(torch=torch.__version__,triton=triton.__version__,transformers=transformers.__version__),
        nemo_revision=nemo.REVISION,speedup_claims=False)
    data['fingerprint']=_fingerprint(data)
    path=root/'execution_contract.json'
    if freeze:frozen_write(path,data)
    elif path.exists() and json.loads(path.read_text())!=data:raise ValueError('Frozen execution changed')
    return data
