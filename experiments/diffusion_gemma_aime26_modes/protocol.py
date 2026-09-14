import json
from pathlib import Path
from experiments.diffusion_gemma_aime30.protocol import MODEL, REVISION, sha, prompt, numeric_score
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _fingerprint

ROOT = Path('results/diffusion_gemma_aime26_blasst_thinking_shots')
DATA_REV = '79037aebdb6580008fb960d17cb21fd3099083e3'
BUDGET = 32768
SEED = 42
TARGETS = (.25, .5, .75, .9)
MODES = tuple(f'{s}_{t}' for t in ('nothink', 'think') for s in ('zero_shot', 'five_shot'))
CONDITIONS = [('dense', None)] + [(f'blasst_s{int(t*100)}', t) for t in TARGETS]


def final_response(raw, thinking):
    # Never grade an intermediate boxed value inside an unfinished thought.
    if '<channel|>' in raw:
        raw = raw.rsplit('<channel|>', 1)[1]
    elif thinking:
        return ''
    for token in ('<turn|>', '<|endoftext|>', '<eos>'):
        raw = raw.split(token, 1)[0]
    return raw.strip()


def prepare(root=ROOT):
    from huggingface_hub import hf_hub_download
    from dllm.models import create_adapter
    source = Path(hf_hub_download('math-ai/aime26', 'aime2026.jsonl', repo_type='dataset', revision=DATA_REV, local_files_only=True))
    test = [json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    assert len(test) == 30 and len({r['id'] for r in test}) == 30
    old = Path('results/diffusion_gemma_aime30_zero_vs_five_shot')
    demos = json.loads((old/'demonstrations.json').read_text())
    cal_ids = json.loads((old/'setup.json').read_text())['calibration_ids']
    pool = [json.loads(x) for x in Path('/home/exouser/ljy/Skills/nemo_skills/dataset/aime25/test.txt').read_text().splitlines() if x.strip()]
    cal = [r for r in pool if r['id'] in cal_ids]
    canonical = lambda x: ''.join(x.split()).lower()
    groups = [{canonical(r['problem']) for r in g} for g in (test, demos, cal)]
    assert len(demos) == 5 and len(cal) == 6
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i+1:])
    tok = create_adapter('diffusion_gemma', MODEL, device='cpu', precision='float32', revision=REVISION).load_tokenizer()
    manifests = {}
    for split, records in [('final', sorted(test, key=lambda r: int(r['id']))), ('calibration', cal)]:
        rows = []
        for mode in MODES:
            thinking = mode.endswith('_think')
            for r in records:
                p = prompt(r['problem'], demos if mode.startswith('five_shot') else [])
                ids = tok.encode_prompt(p, {'thinking': thinking})
                rows.append(dict(id=f'{mode}/{r["id"]}', source_id=r['id'], mode=mode, thinking=thinking,
                    split=split, benchmark='aime26' if split=='final' else 'aime25_calibration', task='AIME',
                    prompt=p, prompt_hash=sha(p), prompt_tokens=ids, prompt_token_hash=_fingerprint(ids),
                    expected=str(r['answer'] if split=='final' else r['expected_answer']), generation_budget=BUDGET, seed=SEED))
        manifests[split] = rows
    setup = dict(schema='aime26_modes_v1', dataset_revision=DATA_REV, dataset_sha256=sha(source.read_bytes()),
        model=MODEL, revision=REVISION, precision='bfloat16', budget=BUDGET, seed=SEED, modes=MODES,
        final_generations=600, calibration_questions_per_mode=6, tile_size=[64,64], lambda_range=[1e-8,1.],
        demonstrations=demos, calibration_ids=cal_ids, sampler=dict(canvas=256, max_denoising_steps=48,
            t_max=.8, t_min=.4, confidence_threshold=.005, stability_threshold=1, entropy_bound=.1),
        protocol_status='paper-aligned, not exact reproduction',
        deviations=['Exact paper prompts, output budget and repetition count were not located.',
            'User-approved 32768-token safety cap and one seed42 generation per problem.',
            'Five fixed AIME2025 CoT demonstrations are an experimental extension.',
            'BF16 reference-mask execution, not the paper optimized FP8 speed setup.'],
        paper='https://arxiv.org/html/2608.00146v1',
        calibration='Disjoint AIME2025; per-mode local/global scalar lambda; dense physical margins and offline search, sparse verification; lambda=1 if unattainable; no test tuning')
    root.mkdir(parents=True, exist_ok=True)
    for name, value in [('setup.json', setup), ('final.json', manifests['final']), ('calibration.json', manifests['calibration']), ('demonstrations.json', demos)]:
        path = root/name
        # JSON canonicalization makes tuples/lists compare consistently.
        if path.exists() and _fingerprint(json.loads(path.read_text())) != _fingerprint(value):
            raise RuntimeError(f'Frozen protocol changed: {path}')
        _write(path, value)
    _write(root/'dataset_audit.json', dict(passed=True, test=30, final_prompts=120, calibration_prompts=24,
        demonstrations=5, disjoint_problem_text=True, token_lengths={m:[len(r['prompt_tokens']) for r in manifests['final'] if r['mode']==m] for m in MODES}))
    return manifests
