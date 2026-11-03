"""Private data preparation for the nine-panel current-tile campaign."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/multi_model_value_20261030/attempt001'
DG=ROOT/'results/optimized_refresh_benchmarks_20261022/attempt009'
MODEL=Path('/media/volume/new-dllm/dlm-migrated-20260921/cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b')


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f);f.write('\n')


def prepare(root):
    from transformers import AutoTokenizer
    import yaml
    from dllm.evaluation.ruler.official import PAPER_TASKS,verify_checkout
    from scripts import v31_official as op
    ruler=ROOT/'reference/RULER';identity=verify_checkout(ruler)
    config=yaml.safe_load((ruler/'scripts/synthetic.yaml').read_text())
    for model in ('llada22','trado'):
        target=root/'private/data'/model;target.mkdir(parents=True,exist_ok=True)
        for name in ('humaneval','aime26','calibration_code_contextmatched001','calibration_math'):
            src=BASE/'private/data'/model/f'{name}.json';dst=target/src.name
            if not dst.exists():dst.symlink_to(src.resolve())
        if model=='llada22':
            for name in ('ruler128k','calibration_ruler'):
                dst=target/f'{name}.json'
                if not dst.exists():dst.symlink_to((BASE/'private/data'/model/dst.name).resolve())
    dgpool=root/'private/pools/diffusiongemma'
    for suite in ('humaneval','aime26'):
        dst=dgpool/suite;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():dst.symlink_to((DG/'private/pools'/suite).resolve(),target_is_directory=True)
    generated=[]
    for model,length in (('trado',32768),('diffusiongemma',131072)):
        modelpath=MODEL if model=='diffusiongemma' else BASE/'private/models/trado'
        tokenizer=AutoTokenizer.from_pretrained(modelpath,trust_remote_code=True)
        for phase,number,seed in (('calibration_ruler',1,9142),('ruler',10,6464)):
            name=('ruler32k' if model=='trado' else 'ruler128k') if phase=='ruler' else phase
            dest=root/'private/data'/model/f'{name}.json'
            if dest.exists():continue
            result=[]
            for ti,task in enumerate(PAPER_TASKS):
                leaf=root/'private/ruler_generated'/model/phase
                output=leaf/task/'validation.jsonl'
                if not output.exists():
                    command=[sys.executable,str(ruler/'scripts/data/prepare.py'),'--save_dir',str(leaf),
                        '--benchmark','synthetic','--task',task,'--tokenizer_path',str(modelpath),
                        '--tokenizer_type','hf','--max_seq_length',str(length-256),'--model_template_type','base',
                        '--num_samples',str(number),'--random_seed',str(seed+ti*1000),'--subset','validation']
                    leaf.mkdir(parents=True,exist_ok=True)
                    with (leaf/f'{task}.log').open('w') as log:
                        subprocess.run(command,cwd=ruler,env=dict(os.environ,PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')),stdout=log,stderr=subprocess.STDOUT,check=True)
                samples=[json.loads(x) for x in output.read_text().splitlines() if x.strip()]
                assert len(samples)==number
                for i,s in enumerate(samples):
                    prefix=s.get('answer_prefix','')
                    if model=='trado':
                        ids=tokenizer.apply_chat_template([{'role':'user','content':s['input']}],add_generation_prompt=True,tokenize=True)
                        if hasattr(ids,'input_ids'):ids=ids.input_ids
                        if ids and isinstance(ids[0],list):ids=ids[0]
                        ids=list(ids)+tokenizer(prefix,add_special_tokens=False).input_ids
                    else:
                        from dllm.models import create_adapter
                        if 'adapter' not in locals():adapter=create_adapter('diffusion_gemma',str(MODEL),device='cpu',precision='float32').load_tokenizer()
                        ids,_=op.ruler_prompt_ids(adapter,s['input'],prefix,op.RESPONSE_OPENER['diffusion_gemma'])
                    assert len(ids)+128<=length,(model,task,len(ids))
                    result.append({'id':f'{task}/{i:03}','task':task,'task_base':config[task]['task'],
                        'prompt':s['input'],'answer_prefix':prefix,'gold':s['outputs'],'prompt_tokens':len(ids),
                        'seeds':[42 if phase=='ruler' else 142],
                        **({'encoded_prompt':ids} if model=='diffusiongemma' else {})})
                print(json.dumps({'model':model,'phase':phase,'task':task,'samples':number}),flush=True)
            assert Counter(r['task'] for r in result)=={t:number for t in PAPER_TASKS}
            write(dest,result)
            generated.append({'model':model,'phase':phase,'requests':len(result),'data_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
                              'prompt_min':min(r['prompt_tokens'] for r in result),'prompt_max':max(r['prompt_tokens'] for r in result)})
    write(root/'data_preparation_complete.json',{'status':'complete','ruler':identity,'generated':generated})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    os.environ['HF_MODULES_CACHE']=str(a.root.resolve()/'private/hf_modules')
    prepare(a.root.resolve())
