"""Build and freeze the v13 protocol BEFORE any new inference (CPU only).

Writes the private 30-question manifest (hash-verified offline AIME26 source, the
existing prompt template; the six already-used development prompts must reproduce
their recorded hashes) and the public frozen_protocol.json (IDs, seeds, arm
definitions + effective config hashes, cell IDs, full pre-generated schedule from
an independent RNG, metrics, ceilings, authorization). No gold in the public file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

from scripts.native_reuse_manifest import _from_dataset, sha_file
from scripts.v10_request_runs import arm_config
from scripts.v13_seed_runs import arm_config_hash, cell_id, plan_schedule

DATASET = Path('/home/exouser/.cache/huggingface/hub/datasets--math-ai--aime26/snapshots/'
               '79037aebdb6580008fb960d17cb21fd3099083e3/aime2026.jsonl')
OLD_SIX = '/home/exouser/dyh/numerical_qk_reuse_native_20260924/code_cp2/results/numerical_qk_reuse_20260924/private/smoke_manifest.json'
OLD_EXT = '/media/volume/dllm-1/dyh/numerical_qk_global_scope_20260925/private/extension_manifest.json'
MODEL = '/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b'
REVISION = 'f7f5b7f5fa82ffc52addd066915886d497f5517b'
DEV = ['aime26/2', 'aime26/8', 'aime26/14', 'aime26/20', 'aime26/23', 'aime26/30']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', type=Path, required=True, help='v12 arms.json (frozen definitions)')
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('dev6', 'all30'), required=True)
    parser.add_argument('--authorization', required=True)
    args = parser.parse_args()
    ids_all = [f'aime26/{i}' for i in range(1, 31)]
    rows = _from_dataset(DATASET, ids_all)
    old = {r['id']: r for r in json.load(open(OLD_SIX)) + json.load(open(OLD_EXT))}
    for r in rows:
        if r['id'] in old and (r['prompt_hash'] != old[r['id']]['prompt_hash'] or r['expected'] != old[r['id']]['expected']):
            raise RuntimeError(f'prompt/answer mapping changed for {r["id"]}')
    manifest = args.private / 'manifest_all30.json'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n')
    ids = ids_all if args.mode == 'all30' else DEV
    seeds = [17, 29]
    rename = {'D_native': 'D_native', 'T_G': 'T_G', 'G1_split': 'G1', 'G3_split': 'G3', 'B8_G': 'B8_G'}
    arms = {}
    for arm in json.loads(args.arms.read_text()):
        arms[rename[arm['name']]] = dict(arm, name=rename[arm['name']])
    root = Path(__file__).resolve().parents[1]
    policy = root / 'results/query_adaptive_v3/configs/frozen_policies.json'
    protocol_id = 'v13_global_multiseed_' + args.mode
    ns = SimpleNamespace(phase=protocol_id, ids=ids, manifest=manifest, policy=policy, model=Path(MODEL),
                         revision=REVISION, seeds=seeds)
    configs = {name: arm_config(ns, arm) for name, arm in arms.items()}
    arm_hashes = {name: arm_config_hash(c) for name, c in configs.items()}
    schedule_seed = 20260925
    schedule = plan_schedule(protocol_id, REVISION, arm_hashes, ids, seeds, schedule_seed)
    protocol = dict(
        schema='v13_frozen_protocol_v1', protocol_id=protocol_id, frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        authorization=args.authorization, mode=args.mode, ids=ids, development_ids=DEV,
        reserved_for_this_iteration=[i for i in ids if i not in DEV],
        exposure_note=('the 24 non-development questions were NOT used in v10-v12 but were used by older project arms '
                       '(results/aime26_prefill_paper_gate_20260916 at 1620e656: all 30 x seeds 17/29); they are reserved '
                       'for THIS method iteration, not pristine held-out data'),
        seeds=seeds, seed_semantics='generation seed -> GenerationRequest.seed -> adapter.generate seed_everything (native sampler RNG); '
                                    'projection seed, T constants and thresholds fixed at their existing values',
        seed42_status='v12 seed-42 results are a separate development observation, not pooled',
        model_path=MODEL, model_revision=REVISION, policy_file=str(policy), dataset=dict(path=str(DATASET), sha256=sha_file(DATASET),
        revision='79037aebdb6580008fb960d17cb21fd3099083e3'), manifest_sha256=sha_file(manifest),
        prompt_hashes={r['id']: r['prompt_hash'] for r in rows if r['id'] in ids},
        arms=arms, arm_hashes=arm_hashes,
        effective_arm_fields={n: {k: c.get(k) for k in ('condition', 'decision_interval', 'score_refresh_period', 'selector',
                              'selector_layers', 'consumer', 'guard_mode', 'telemetry', 'kernel_variant', 'collect', 'plugin',
                              'support_binary', 'library', 'policy')} for n, c in configs.items()},
        generation=dict(max_new_tokens=8192, thinking=True, native_adaptive=True, eos=True, temperature_sentinel=0.0, dtype='bfloat16'),
        schedule_seed=schedule_seed, schedule_rule='blocks (question, seed) shuffled by random.Random(schedule_seed); within block '
                      'all arms attempt 0 in a cyclic rotation by block position, then one warm repeat per arm in reversed order',
        warm_repeats=1, schedule=schedule,
        metrics=dict(quality='attempt-0 final-channel correctness; cap/unparsed/failure = incorrect',
                     time='accepted single warm API wall (same cell, no new compilation, tokens/calls/termination identical to attempt 0)',
                     primary='G3 vs T_G', secondary=['G3 vs D_native', 'G3 vs B8_G', 'G1 vs G3', 'G1 vs B8_G', 'G1 vs T_G'],
                     aggregation='per question: mean over seeds of log(candidate/reference); geometric mean over questions; '
                                 'also ratio of summed times; question-cluster bootstrap 10000 deterministic resamples',
                     robustness=['per seed', 'leave-one-question-out', 'development 6 vs reserved 24']),
        failure_policy='every scheduled execution runs once; failures are recorded and count as incorrect/missing; no silent retries; '
                       'a device error stops the worker and a clean worker resumes the remaining schedule',
        ceilings=dict(dev6=dict(executions=150, gpu_h=3), all30=dict(executions=660, gpu_h=12, elapsed_h=14)),
        planned_executions=len(schedule))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'frozen_protocol.json').write_text(json.dumps(protocol, indent=2, sort_keys=True) + '\n')
    print(protocol_id, len(schedule), arm_hashes)


if __name__ == '__main__':
    main()
