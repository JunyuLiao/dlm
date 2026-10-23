"""Freeze production execution while preserving the original attempt identity."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
REPO=Path('/home/exouser/ljy/dlm')
sys.path.insert(0,str(REPO/'src'));sys.path.insert(0,str(REPO))
from worker import write,sha
from dllm.models import create_adapter

parent=ROOT.parent
protocol=json.loads((parent/'protocol.json').read_text())
rows=json.loads((parent/'private/manifest.json').read_text())
adapter=create_adapter('diffusion_gemma',protocol['sources']['model_path'],
    device='cuda',precision='bfloat16',revision=protocol['sources']['model_revision']).load_tokenizer()
audit=[];tokenized=[]
for row in rows:
    ids=adapter.encode_prompt(row['prompt'],{'thinking':row['thinking']})
    count=len(ids); fits=count+row['generation_budget']<=262144
    audit.append(dict(id=row['id'],benchmark=row['benchmark'],prompt_tokens=count,
        generation_budget=row['generation_budget'],position_limit=262144,eligible=fits,
        prompt_hash=row['prompt_hash'],tokenization_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        blocked_reason=None if fits else 'native_context_limit_exceeded'))
    if fits:tokenized.append(dict(row,prompt_tokens=ids))
    print(json.dumps(audit[-1]),flush=True)
write(ROOT/'context_audit.json',dict(items=audit,eligible=sum(r['eligible'] for r in audit),
    blocked=sum(not r['eligible'] for r in audit),truncation=False,replacement=False,
    policy='User explicitly requires over-limit items to remain blocked'))
write(ROOT/'private/tokenized_manifest.json',tokenized)
sources=[str(p) for p in sorted((ROOT/'code').glob('*.py'))]+[str(p) for p in sorted((ROOT/'tests').glob('*.py'))]
sources += [str(REPO/p) for p in [
    'scripts/v31_humaneval_cfgdet_base_worker.py','scripts/v31_humaneval_repro_policy_worker.py',
    'experiments/numerical_qk_reuse/v31_value_native_audit.py']]
sources += ['/home/exouser/dyh/v31_env_20261005/lib/python3.12/site-packages/vllm/model_executor/models/diffusion_gemma.py']
sources += [str(parent/'protocol.json'),str(parent/'manifest_identity.json'),str(parent/'private/manifest.json')]
new=dict(schema='denoising_dynamics_production_v1',base_head='dea3b7de2833a8131fe2500312420fe873bd3c0e',
    original_protocol_sha256=sha(parent/'protocol.json'),original_identity_sha256=sha(parent/'manifest_identity.json'),
    model_path=protocol['sources']['model_path'],model_revision=protocol['sources']['model_revision'],
    purpose='Dense native FULL diagnostic only; no sparse method or clean timing claim',
    native_sampler=protocol['native_sampler'],seed_policy='original manifest seed unchanged, 42; no per-canvas reseeding',
    arms=['dense_full_fix51994'],blocked_ids=[r['id'] for r in audit if not r['eligible']],
    eligible_ids=[r['id'] for r in audit if r['eligible']],
    private_tokenized_manifest_sha256=sha(ROOT/'private/tokenized_manifest.json'),
    smoke_item='humaneval/HumanEval/53',short_max_model_len=12288,long_max_model_len=202752,
    smoke_gate=dict(sequence=['warm','off_before','minimal_before','rich','minimal_after','off_after'],
        required=['identical output tokens hash','identical prompt tokens hash','identical sampler and model forward counts',
        'identical final CPU/CUDA RNG states','identical active per-call sampler state hashes',
        'zero observer RNG changes','zero reconstructed native stopping predicate mismatches']),
    state_reproduction_scope='Within-engine active sampler-state/output parity; full KV/layout fork restore is a separate Phase 2 gate',
    phases_2_3_gate='No intervention or selector before exact full-state fork restore and held-out diagnostic evidence',
    failed_item_policy='preserve failures and blocks; no truncation, replacement, forced continuation or altered native settings',
    resources=dict(gpu='one idle H100',workers_per_gpu=1,no_preemption=True,max_reservation_seconds=1800),
    sources={p:sha(p) for p in sources})
write(ROOT/'protocol.json',new)
write(ROOT/'source_fingerprints.json',new['sources'])
write(ROOT/'freeze_receipt.json',dict(protocol_sha256=sha(ROOT/'protocol.json'),
    manifest_identity_sha256=sha(parent/'manifest_identity.json'),original_frozen_files_unchanged=True,
    cpu_tests_passed=13,gpu_generation_launched=False,full30_complete=False))
