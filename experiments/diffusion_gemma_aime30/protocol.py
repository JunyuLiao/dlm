"""Frozen prompts, disjoint demonstrations/calibration, final-answer scoring."""
import hashlib
import html
import json
import re
from decimal import Decimal, InvalidOperation, DecimalException
from pathlib import Path

from experiments.diffusion_gemma_solattn_blasst_multibench.config import MODEL, REVISION, TARGETS, BETAS
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _rows, _write, _fingerprint

ROOT = Path('results/diffusion_gemma_aime30_zero_vs_five_shot')
SOURCES = Path('/home/exouser/ljy/Skills/nemo_skills/dataset')
SEED = 42
BUDGET = 2048
MODES = ('zero_shot', 'five_shot')
INSTRUCTION = 'Solve the problem. Show your reasoning and put the final integer answer in \\boxed{}.'


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def conditions():
    return [dict(name='dense', method='dense', target=0., beta=None)] + [
        dict(name=f'{method}_s{int(s*100)}', method=method, target=s, beta=BETAS[s] if method=='sol' else None)
        for method in ('sol', 'blasst') for s in TARGETS]


def prompt(problem, demonstrations):
    examples=''.join(f'\n\n### Worked example {i+1}\nProblem: {d["problem"]}\nSolution: {d["reference_solution"]}\nFinal answer: \\boxed{{{d["expected_answer"]}}}' for i,d in enumerate(demonstrations))
    return INSTRUCTION + examples + '\n\n### Problem to solve\n' + problem + '\n\nSolution:'


def prepare(root=ROOT):
    from dllm.models import create_adapter
    paths={year:SOURCES/f'aime{year}'/'test.txt' for year in (24,25)}
    test, pool = _rows(paths[24]), _rows(paths[25])
    if len(test)!=30 or len(pool)!=30: raise ValueError('expected complete 30-question AIME sources')
    ordered=sorted(pool,key=lambda r:sha(f'{SEED}|aime25|{r["id"]}'))
    # Some cached AIME rows have no worked solution. A demonstration must
    # actually contain CoT and end with a numerically consistent answer.
    eligible=[r for r in ordered if r.get('reference_solution','').strip()
              and numeric_score(r['reference_solution'],str(r['expected_answer']))['correct']]
    demos=[dict(r,reference_solution=html.unescape(re.sub(r'<[^>]*>','',r['reference_solution']))) for r in eligible[:5]]
    if len(demos)!=5: raise ValueError('five valid worked demonstrations required')
    demo_ids={r['id'] for r in demos}
    calibration=[r for r in ordered if r['id'] not in demo_ids][:6]
    groups=[{sha(re.sub(r'\s+','',r['problem']).lower()) for r in rows} for rows in (test,demos,calibration)]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)): raise ValueError('test/demo/calibration problem overlap')
    tokenizer=create_adapter('diffusion_gemma',MODEL,device='cpu',precision='float32',revision=REVISION).load_tokenizer()
    manifests={}
    for split,rows in [('final',test),('calibration',calibration)]:
        manifest=[]
        for mode in MODES:
            for r in rows:
                p=prompt(r['problem'],demos if mode=='five_shot' else [])
                ids=tokenizer.encode_prompt(p,{'thinking':False})
                manifest.append(dict(id=f'{mode}/{r["id"]}',source_id=r['id'],mode=mode,split=split,
                    benchmark='aime24' if split=='final' else 'aime25_calibration',task='AIME',
                    problem_hash=sha(r['problem']),prompt=p,prompt_hash=sha(p),prompt_tokens=ids,
                    prompt_token_hash=_fingerprint(ids),token_count=len(ids),expected=str(r['expected_answer']),
                    generation_budget=BUDGET,seed=SEED))
        manifests[split]=manifest
    setup=dict(schema='aime30_v1',model=MODEL,revision=REVISION,precision='bfloat16',seed=SEED,
        sources={str(p):sha(p.read_bytes()) for p in paths.values()},demonstration_ids=[r['id'] for r in demos],
        calibration_ids=[r['id'] for r in calibration],test_ids=[r['id'] for r in test],
        final_generations=540,calibration_questions_per_mode=6,max_new_tokens=BUDGET,
        calibration_rule='per-mode per-type scalar lambda; existing candidate grid plus lambda=1; log bisection on dense physical-tile margins; verify on sparse calibration trajectories',
        decoding=dict(request_temperature=0.,semantics='native seeded schedule, not greedy',t_min=.4,t_max=.8,
            max_denoising_steps=48,confidence_threshold=.005,stability_threshold=1,entropy_bound=.1,thinking=False,canvas_length=256),
        tile_size=[64,64],regions='prefix + canvas, one combined Sol population',
        score='last boxed answer; otherwise last explicit final-answer marker; otherwise last numeric token; compare numeric value, absolute tolerance 1e-6')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'setup.json').exists() and json.loads((root/'setup.json').read_text())!=setup: raise ValueError('incompatible cached setup')
    for split,rows in manifests.items():
        path=root/f'{split}.json'
        if path.exists() and json.loads(path.read_text())!=rows: raise ValueError('incompatible cached manifest')
        _write(path,rows)
    _write(root/'demonstrations.json',demos);_write(root/'setup.json',setup)
    _write(root/'dataset_audit.json',dict(passed=True,final=60,calibration=12,demonstrations=5,
        unique_test_problems=30,disjoint_sources=True,prompt_tokens_by_mode={m:[r['token_count'] for r in manifests['final'] if r['mode']==m] for m in MODES}))
    return setup


NUMBER = re.compile(r'(?<![\w.])[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])')


def extract_number(text):
    """Gold-independent final-answer extraction; never scan for a matching gold."""
    boxes=[]
    for match in re.finditer(r'\\(?:boxed|fbox)\s*\{',text):
        start=match.end();depth=1;i=start
        while i<len(text) and depth:
            depth+=(text[i]=='{')-(text[i]=='}');i+=1
        if depth==0: boxes.append(text[start:i-1])
    source='last_box' if boxes else 'last_number'
    candidate=boxes[-1] if boxes else text
    if not boxes:
        markers=list(re.finditer(r'(?:final\s+(?:integer\s+)?answer|the\s+answer\s+is|answer\s*:)\s*(?:is\s*)?[:=]?\s*',text,re.I))
        if markers: candidate=text[markers[-1].end():].split('\n')[0];source='answer_marker'
    # Strip simple LaTeX styling; support numeric fractions, not arbitrary eval.
    candidate=re.sub(r'\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}',r'\1',candidate)
    fraction=re.fullmatch(r'\s*\\(?:d?frac)\{([-+]?\d+)\}\{([-+]?\d+)\}\s*',candidate)
    if fraction:
        try: return Decimal(fraction[1])/Decimal(fraction[2]),source
        except (DecimalException,ZeroDivisionError): return None,source
    matches=list(NUMBER.finditer(candidate))
    if not matches: return None,source
    # A boxed formula with multiple numbers is ambiguous; reject it.
    if boxes and len(matches)!=1: return None,source
    try: return Decimal(matches[-1][0].replace(',','')),source
    except InvalidOperation: return None,source


def numeric_score(text,expected):
    number,source=extract_number(text)
    return dict(correct=bool(number is not None and number.is_finite() and abs(number-Decimal(expected))<=Decimal('0.000001')),
        extracted=str(number) if number is not None else None,extraction=source)
