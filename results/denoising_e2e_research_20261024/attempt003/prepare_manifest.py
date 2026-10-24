"""Deterministic 10/10/5 sampling, original prompts, no outcome filtering."""
import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

REPO = Path('/home/exouser/ljy/dlm')
sys.path.insert(0, str(REPO))
from experiments.diffusion_gemma_sparse_attention_research.dynamics_screen import (
    LB_SOURCE, LB_CHECKOUT, MODEL, official_longbench_prompt)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    previous = REPO / 'results/denoising_native_causal_pulsehold_20261024/attempt002'
    prior = json.loads((previous / 'private/tokenized_manifest.json').read_text())
    protocol = json.loads((previous / 'protocol.json').read_text())
    blocked = set(protocol['blocked_ids'])
    short = [r for r in prior if r['benchmark'] != 'longbench_v2']
    assert len(short) == 20
    existing = next(r for r in prior if r['benchmark'] == 'longbench_v2')
    raw = json.loads(LB_SOURCE.read_text())
    domains = sorted({r['domain'] for r in raw if r.get('length') == 'long'})
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    check = tokenizer.apply_chat_template([dict(role='user', content=short[0]['prompt'])],
        tokenize=True, add_generation_prompt=True, enable_thinking=True)
    if hasattr(check, 'keys'):
        check = check['input_ids']
    assert list(check) == short[0]['prompt_tokens']
    long_rows = [existing]
    screening = [dict(id=existing['id'], domain=existing['domain'], status='preserved_original_eligible_item', tokens=len(existing['prompt_tokens']))]
    candidates = sorted([r for r in raw if r.get('length') == 'long' and 'longbench_v2/'+r['_id'] != existing['id']],
        key=lambda r: hashlib.sha256(f'denoising-e2e-expanded-v2|{r["_id"]}'.encode()).digest())
    for item in candidates:
        rid = 'longbench_v2/' + item['_id']
        if rid in blocked:
            screening.append(dict(id=rid, domain=item['domain'], status='original_block_preserved'))
            continue
        prompt = official_longbench_prompt(item)
        tokens = tokenizer.apply_chat_template([dict(role='user', content=prompt)],
            tokenize=True, add_generation_prompt=True, enable_thinking=True)
        if hasattr(tokens, 'keys'):
            tokens = tokens['input_ids']
        eligible = len(tokens) + 16384 <= 262144
        screening.append(dict(id=rid, domain=item['domain'], tokens=len(tokens), status='selected' if eligible else 'over_limit_excluded'))
        (root / 'private/eligibility_progress.json').write_text(json.dumps(screening))
        if not eligible:
            continue
        long_rows.append(dict(id=rid, source_id=item['_id'], benchmark='longbench_v2',
            domain=item['domain'], sub_domain=item['sub_domain'], difficulty=item['difficulty'], length_band='long',
            prompt=prompt, prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            prompt_tokens=list(tokens), expected=item['answer'], generation_budget=16384,
            seed=42, thinking=True))
        if len(long_rows)==5:
            break
    if len(long_rows)!=5:
        raise RuntimeError('Fewer than five eligible original long-band items; no replacement or truncation.')
    rows = short + sorted(long_rows, key=lambda r: r['id'])
    assert len(rows) == 25 and len({r['id'] for r in rows}) == 25
    with (root / 'private/tokenized_manifest.json').open('x') as f:
        json.dump(rows, f)
    public = dict(selection='Preserve original10 AIME/10 HumanEval and eligible original LongBench; four additional native-eligible long-band items via global SHA256 order; no per-domain quota, no outcome filtering.',
        identities=[{k: v for k, v in r.items() if k not in {'prompt', 'prompt_tokens', 'expected'}} |
                    dict(prompt_token_count=len(r['prompt_tokens']), seeds=[42, 43],
                         prior_exposure='prior dense/intervention exposure for original21; additional four historical exposure not certified absent') for r in rows],
        screening=screening, blocked_original_ids=sorted(blocked),
        sources=dict(original_private_manifest_sha256=sha(previous / 'private/tokenized_manifest.json'),
            longbench_data_sha256=sha(LB_SOURCE), official_template_sha256=sha(LB_CHECKOUT / 'prompts/0shot.txt'),
            private_manifest_sha256=sha(root / 'private/tokenized_manifest.json')))
    with (root / 'manifest_identity.json').open('x') as f:
        json.dump(public, f, sort_keys=True, indent=2)
    print(json.dumps(dict(items=len(rows), short=20, long=5,
        long_lengths=[len(r['prompt_tokens']) for r in long_rows], screened=len(screening), original_blocked=len(blocked))))


if __name__ == '__main__':
    main()
