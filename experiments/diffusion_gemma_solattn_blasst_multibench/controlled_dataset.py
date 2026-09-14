"""Pinned shared samples and official task metrics, preserving partial credit."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import re
import subprocess
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from dllm.evaluation.ruler.official import load_scorers, verify_checkout
from experiments.diffusion_attention_threshold_modeling.math500 import build_math_verifier, symbolic_verify

RULER_ROOT = Path('reference/RULER')
LB_ROOT = Path('reference/LongBench/LongBench')
LB_REV = '5e628be450b7e67fb7ae6e201bd6d8f7056f7672'
LCB_REV = 'c52cd175916e995019dcd848d1054b419d2e70b5'
LB_TASKS = ('qasper', 'hotpotqa', 'gov_report', 'trec', 'passage_retrieval_en')

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def take(rows, count, key, seed=42):
    def identity(r): return r.get('_id', r.get('question_id', r.get('id')))
    if len(rows) < count: raise ValueError(f'{key}: insufficient samples')
    return sorted(rows, key=lambda r:digest(f'{seed}|{key}|{identity(r)}'))[:count]

def prepare(output_dir, *, model_path, seed=42):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from dllm.models import create_adapter
    if (output_dir/'manifest.jsonl').exists(): return audit_manifest(output_dir)
    adapter=create_adapter('diffusion_gemma',model_path,device='cpu',precision='float32').load_tokenizer()
    rows=[]
    source=Path('results/diffusion_gemma_solattn_vs_blasst_ruler16k/final.jsonl')
    for r in [json.loads(x) for x in source.read_text().splitlines() if x.strip()]:
        rows.append(dict(id=f"ruler16k/{r['sample_id']}",benchmark='ruler16k',task=r['task'],prompt=r['prompt'],expected=r['outputs'],generation_budget=int(r['generation_budget']),seed=int(r['seed']),source_id=r['source_id'],task_base=r['task_base']))
    aime_path=Path('/home/exouser/ljy/Skills/nemo_skills/dataset/aime24/test.txt')
    aime=[json.loads(x) for x in aime_path.read_text().splitlines() if x.strip()]
    for paper in range(2):
        for band,count in enumerate((2,2,1)):
            for r in take(aime[paper*15+band*5:paper*15+(band+1)*5],count,f'aime/{paper}/{band}',seed):
                rows.append(dict(id=f"aime24/{r['id']}",benchmark='aime24',task=f'AIME_{paper+1}',prompt='Solve the problem. Put the final integer answer in \\boxed{}.\n\n'+r['problem'],expected=str(r['expected_answer']),generation_budget=2048,seed=seed+len(rows),source_id=r['id'],difficulty_band=band))
    prompts=json.loads((LB_ROOT/'config/dataset2prompt.json').read_text())
    budgets=json.loads((LB_ROOT/'config/dataset2maxlen.json').read_text())
    archive=hf_hub_download('THUDM/LongBench','data.zip',repo_type='dataset',revision=LB_REV)
    with zipfile.ZipFile(archive) as z:
        for task in LB_TASKS:
            member=next(x for x in z.namelist() if x.endswith('/'+task+'.jsonl') or x==task+'.jsonl')
            pool=[json.loads(x) for x in z.read(member).decode().splitlines() if x.strip()]
            for r in take(pool,10,task,seed):
                prompt=prompts[task].format(**r); original_hash=digest(prompt)
                original_length=len(adapter.encode_prompt(prompt,{'thinking':False}))
                truncated=original_length>16384
                if truncated:
                    ids=adapter.tokenizer.encode(prompt,add_special_tokens=False); keep=16256
                    while True:
                        prompt=adapter.tokenizer.decode(ids[:keep//2],skip_special_tokens=True)+adapter.tokenizer.decode(ids[-keep//2:],skip_special_tokens=True)
                        if len(adapter.encode_prompt(prompt,{'thinking':False}))<=16384: break
                        keep-=128
                rows.append(dict(id=f"longbench/{task}/{r['_id']}",benchmark='longbench',task=task,prompt=prompt,expected=r['answers'],all_classes=r.get('all_classes',[]),generation_budget=budgets[task],seed=seed+len(rows),source_id=r['_id'],original_prompt_hash=original_hash,original_prompt_tokens=original_length,truncated=truncated))
    lcb=[dict(r) for r in load_dataset('livecodebench/code_generation_lite','release_v6',split='test',revision=LCB_REV)]
    for difficulty in ('easy','medium','hard'):
        for r in take([x for x in lcb if x['difficulty']==difficulty],4,'lcb/'+difficulty,seed):
            starter=r.get('starter_code') or ''
            form=('Use this starter code:\n```python\n'+starter+'\n```') if starter else 'Read inputs from stdin and write answers to stdout.'
            rows.append(dict(id=f"livecodebench_v6/{r['question_id']}",benchmark='livecodebench_v6',task=difficulty,prompt='You are an expert Python programmer.\n\n### Question:\n'+r['question_content']+'\n\n'+form+'\nEnclose the solution in one ```python``` block.',expected=None,generation_budget=2048,seed=seed+len(rows),source_id=r['question_id'],lcb={k:r[k] for k in ('question_id','contest_date','platform','difficulty','starter_code','public_test_cases','private_test_cases','metadata')}))
    for r in rows:
        r['prompt_hash']=digest(r['prompt']);r['prompt_tokens']=len(adapter.encode_prompt(r['prompt'],{'thinking':False}))
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/'manifest.jsonl').write_text(''.join(json.dumps(r,sort_keys=True,default=str)+'\n' for r in rows))
    provenance=dict(ruler=verify_checkout(RULER_ROOT),ruler_manifest_sha256=digest(source.read_text()),aime_sha256=digest(aime_path.read_text()),longbench_dataset_revision=LB_REV,lcb_dataset_revision=LCB_REV,lcb_release='release_v6 (May 2023–April 2025)',model_path=model_path,source_commits={n:subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path('reference')/n,text=True).strip() for n in ('LongBench','LiveCodeBench')})
    (output_dir/'dataset_provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
    return audit_manifest(output_dir)

def audit_manifest(output_dir):
    rows=[json.loads(x) for x in (output_dir/'manifest.jsonl').read_text().splitlines() if x.strip()]
    counts=dict(Counter(r['benchmark'] for r in rows))
    if counts!={'ruler16k':50,'longbench':50,'aime24':10,'livecodebench_v6':12} or len({r['id'] for r in rows})!=122 or len({r['prompt_hash'] for r in rows})!=122: raise ValueError(f'invalid selection: {counts}')
    if any(digest(r['prompt'])!=r['prompt_hash'] for r in rows): raise ValueError('prompt hash mismatch')
    audit=dict(counts=counts,count=len(rows),tasks=dict(Counter(r['benchmark']+'/'+r['task'] for r in rows)),unique_ids=True,unique_prompts=True,longbench_truncated=sum(r.get('truncated',False) for r in rows))
    (output_dir/'manifest_audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    return audit

@lru_cache
def lb_metrics():
    spec=importlib.util.spec_from_file_location('official_longbench_metrics',LB_ROOT/'metrics.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

@lru_cache
def ruler_scorers(): return load_scorers(RULER_ROOT)

@lru_cache
def verifier(): return build_math_verifier()

def score(row,prediction,**unused):
    if row['benchmark']=='ruler16k': return float(ruler_scorers()[row['task_base']]([prediction.strip()],[[str(v) for v in row['expected']]]))/100
    if row['benchmark']=='aime24': return float(symbolic_verify(verifier(),str(row['expected']),prediction)[0])
    if row['benchmark']=='longbench':
        fn=getattr(lb_metrics(),{'qasper':'qa_f1_score','hotpotqa':'qa_f1_score','gov_report':'rouge_score','trec':'classification_score','passage_retrieval_en':'retrieval_score'}[row['task']])
        if row['task']=='trec': prediction=prediction.lstrip('\n').split('\n')[0]
        return max(fn(prediction,a,all_classes=row['all_classes']) for a in row['expected'])
    return None

def code_from_prediction(text):
    blocks=re.findall(r'```(?:python)?\s*\n?(.*?)```',text,flags=re.S|re.I)
    return (blocks[0] if blocks else text).strip()
