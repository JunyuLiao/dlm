"""Freeze source datasets and an untouched held-out evaluation contract."""
import json
import zipfile
from collections import Counter
from pathlib import Path
from dllm.models import create_adapter
from experiments.diffusion_gemma_aime30.protocol import MODEL, REVISION, sha, numeric_score
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _rows, _write, _fingerprint
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_dataset import LB_ROOT, LB_TASKS, LB_REV, take, score as lb_score

ROOT = Path('results/diffusion_gemma_value_aware_128x64')
AIME = Path('results/diffusion_gemma_aime26_blasst_aggressive/refined')
BASE = Path('results/diffusion_gemma_solattn_blasst_multibench_controlled')
TARGETS = (.25, .5, .75, .9)


def frozen_write(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError(f'incompatible immutable protocol: {path}')
    _write(path, value)


def prepare(root=ROOT):
    if (root/'setup.json').exists():
        setup=json.loads((root/'setup.json').read_text())
        audit(setup); return setup
    aime=json.loads((AIME/'manifest.json').read_text())
    final=[dict(r, split='final') for r in aime]
    calibration=[dict(r, split='calibration') for r in aime if r['calibration']]
    lb=[r for r in _rows(BASE/'manifest.jsonl') if r['benchmark']=='longbench']
    final += [dict(r, split='final', calibration=False) for r in lb]
    tokenizer=create_adapter('diffusion_gemma',MODEL,device='cpu',precision='float32',revision=REVISION).load_tokenizer()
    archive=next(Path('/home/exouser/.cache/huggingface/hub/datasets--THUDM--LongBench/snapshots').glob(f'{LB_REV}/data.zip'))
    prompts=json.loads((LB_ROOT/'config/dataset2prompt.json').read_text())
    budgets=json.loads((LB_ROOT/'config/dataset2maxlen.json').read_text())
    development=[]
    with zipfile.ZipFile(archive) as z:
        for task in LB_TASKS:
            member=next(n for n in z.namelist() if n.endswith('/'+task+'.jsonl') or n==task+'.jsonl')
            pool=[json.loads(x) for x in z.read(member).decode().splitlines() if x.strip()]
            excluded={r['source_id'] for r in lb if r['task']==task}
            for i,r in enumerate(take([r for r in pool if r['_id'] not in excluded],3,task,42)):
                p=prompts[task].format(**r); original=sha(p)
                n=len(tokenizer.encode_prompt(p,{'thinking':False})); truncated=n>16384
                if truncated:
                    ids=tokenizer.tokenizer.encode(p,add_special_tokens=False); keep=16256
                    while True:
                        p=tokenizer.tokenizer.decode(ids[:keep//2],skip_special_tokens=True)+tokenizer.tokenizer.decode(ids[-keep//2:],skip_special_tokens=True)
                        if len(tokenizer.encode_prompt(p,{'thinking':False}))<=16384: break
                        keep-=128
                row=dict(id=f'longbench/{task}/{r["_id"]}',source_id=r['_id'],benchmark='longbench',task=task,
                    prompt=p,prompt_hash=sha(p),prompt_tokens=len(tokenizer.encode_prompt(p,{'thinking':False})),
                    expected=r['answers'],all_classes=r.get('all_classes',[]),seed=42+i,
                    generation_budget=budgets[task],calibration=i<2,split='calibration' if i<2 else 'development',
                    original_prompt_hash=original,original_prompt_tokens=n,truncated=truncated)
                (calibration if i<2 else development).append(row)
    setup=dict(schema='value_aware_128x64_v1',model=MODEL,revision=REVISION,precision='bfloat16',
        tile_size=[128,64],regions='prefix_plus_canvas',final=final,calibration=calibration,development=development,
        decoding=json.loads((BASE/'decoding_protocol.json').read_text()),
        selection='AIME26 all30; existing6 calibration IDs; LongBench exact controlled50; next2/task for calibration and next1/task for development, SHA256 seed42 ordering',
        source_hashes={str(p):sha(p.read_bytes()) for p in [AIME/'manifest.json', AIME/'policy.json', BASE/'manifest.jsonl', archive,
            LB_ROOT/'metrics.py', LB_ROOT/'config/dataset2prompt.json', LB_ROOT/'config/dataset2maxlen.json']},
        screening='same dense-state masks; steps0,4,12,24 first call each layer; all heads and both query blocks; no heldout accuracy',
        threshold_rule='new risks independently calibrated local/global scalar log thresholds; BLASST uses existing inverse valid-KV-length scale, cap1 for original only',
        stages=['correctness_and_smoke','dense_calibration_screen','pooling_selection_and_sparse_calibration',
            'development_failure_analysis','freeze_final_conditions','full30_and_longbench50','paired_seed_confirmation','report_and_audit'],
        constraints='no kernels; no speedup claims; no current dense output in online routing; compensation is not complete deletion')
    audit(setup); frozen_write(root/'setup.json',setup)
    for name in ('final','calibration','development'): frozen_write(root/f'{name}.json',setup[name])
    frozen_write(root/'dataset_audit.json',audit(setup))
    return setup


def audit(setup):
    f,c,d=[setup[k] for k in ('final','calibration','development')]
    assert Counter(r['benchmark'] for r in f)=={'aime26':30,'longbench':50}
    assert Counter(r['benchmark'] for r in c)=={'aime26':6,'longbench':10}
    assert len(d)==5 and all(r['benchmark']=='longbench' for r in d)
    assert {int(r['source_id']) for r in c if r['benchmark']=='aime26'}=={2,8,14,20,23,30}
    for rows in (f,c,d):
        assert len({r['id'] for r in rows})==len(rows)
        assert len({r['prompt_hash'] for r in rows})==len(rows)
        assert all(sha(r['prompt'])==r['prompt_hash'] for r in rows)
    heldout=[r for r in f if not r['calibration']]
    assert not {r['prompt_hash'] for r in heldout}&{r['prompt_hash'] for r in c+d}
    assert not {r['id'] for r in heldout}&{r['id'] for r in c+d}
    assert not {r['id'] for r in c}&{r['id'] for r in d}
    return dict(passed=True,final=80,calibration=16,development=5,aime26_full=30,aime26_heldout=24,
        aime26_calibration_ids=[2,8,14,20,23,30],longbench_final_tasks=dict(Counter(r['task'] for r in f if r['benchmark']=='longbench')),
        final_calibration_overlap='Only the six pre-existing AIME26 calibration problems, separately reported')


def score(row, text):
    return float(numeric_score(text,row['expected'])['correct']) if row['benchmark']=='aime26' else lb_score(row,text)


def fingerprint(root):
    # Future selection/report orchestration must not invalidate existing shared
    # dense states. Changes to the scientific operator/protocol/generation do.
    paths=[Path(__file__).with_name(n) for n in ('operators.py','routing.py','protocol.py','run.py')]
    paths += [Path('src/dllm/attention/blasst/core.py'),Path('src/dllm/models/adapters/diffusion_gemma.py'),
        Path('experiments/diffusion_gemma_oracle/routing.py'),Path('experiments/diffusion_gemma_solattn_blasst_multibench/runner.py')]
    return _fingerprint(dict(setup_sha=sha((root/'setup.json').read_bytes()),code={str(p):sha(p.read_bytes()) for p in paths}))
