"""Build and freeze the v14 CVM-T pilot protocol BEFORE any new complete-answer run (CPU only).

Reuses the v13 hash-verified private AIME26 manifest and the v13 seed-safe identity /
schedule machinery (scripts/v13_seed_runs.py). Public file: IDs, seeds, arm definitions
+ effective config hashes, cell IDs, the full pre-generated schedule (independent RNG),
the conditional extension IDs (committed now, run only if both gates are encouraging),
metrics, ceilings. No prompts, answers or gold.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from scripts.native_reuse_manifest import sha_file
from scripts.v10_request_runs import arm_config
from scripts.v13_seed_runs import arm_config_hash, plan_schedule

MANIFEST = Path('/media/volume/dllm-1/dyh/numerical_qk_global_multiseed_20260925/private/manifest_all30.json')
MANIFEST_SHA = 'a253c357e1a2a2634d895e91620d7e91b1b10640755088a8d4ca7d2b0250b57c'
MODEL = '/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b'
REVISION = 'f7f5b7f5fa82ffc52addd066915886d497f5517b'
DEV = ['aime26/2', 'aime26/8', 'aime26/14', 'aime26/20', 'aime26/23', 'aime26/30']
EXTENSION = ['aime26/1', 'aime26/6', 'aime26/16', 'aime26/25']
V5 = '/media/volume/dllm-1/dyh/numerical_qk_cvm_t_20260926/build/v5_be53e4c706032aef/build_identity.json'
SUPPORT = ('/media/volume/dllm-1/dyh/numerical_qk_hopper_support_bridge_20260925/build/support_e3c283b8cbb9b79c/'
           'build_identity.json')
PLUGIN = 'experiments.numerical_qk_reuse.global_scope:install'
ARMS = [
    dict(name='D_native', condition='native_dense'),
    dict(name='T_G_original', condition='global_T', plugin=PLUGIN, collect=False, fast_t=True),
    dict(name='T_P', condition='global_TP', plugin=PLUGIN, fast_t=True, v5_build=V5),
    dict(name='B8_P', condition='global_B8P', plugin=PLUGIN, fast_t=True, v5_build=V5, support_build=SUPPORT, period=8,
         guard_mode='fused'),
    dict(name='CVM_T', condition='global_CVM', plugin=PLUGIN, fast_t=True, v5_build=V5, support_build=SUPPORT, period=8,
         guard_mode='fused'),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('dev6', 'extension4', 'smoke'), required=True)
    parser.add_argument('--authorization', required=True)
    args = parser.parse_args()
    if sha_file(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError('v13 private manifest changed')
    ids = {'dev6': DEV, 'extension4': EXTENSION, 'smoke': ['aime26/2']}[args.mode]
    seeds = [42] if args.mode == 'smoke' else [17, 29]           # smoke: a non-pilot seed; path check only, never scored
    arms = {a['name']: a for a in ARMS if args.mode != 'smoke' or a['name'] in ('T_P', 'B8_P', 'CVM_T')}
    root = Path(__file__).resolve().parents[1]
    policy = root / 'results/query_adaptive_v3/configs/frozen_policies.json'
    protocol_id = 'v14_cvm_t_' + args.mode
    ns = SimpleNamespace(phase=protocol_id, ids=ids, manifest=MANIFEST, policy=policy, model=Path(MODEL),
                         revision=REVISION, seeds=seeds)
    configs = {name: arm_config(ns, arm) for name, arm in arms.items()}
    arm_hashes = {name: arm_config_hash(c) for name, c in configs.items()}
    schedule_seed = 20260926 if args.mode == 'dev6' else 20260927
    schedule = plan_schedule(protocol_id, REVISION, arm_hashes, ids, seeds, schedule_seed,
                             warm_repeats=0 if args.mode == 'smoke' else 1)
    rows = {r['id']: r for r in json.loads(MANIFEST.read_text())}
    protocol = dict(
        schema='v14_frozen_protocol_v1', protocol_id=protocol_id, frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        authorization=args.authorization, mode=args.mode, ids=ids, development_ids=DEV,
        extension_ids=EXTENSION,
        extension_rule=('committed before any v14 answer score is read; run (80 executions) ONLY if both the direct '
                        'complete-forward gate (5.1) and the adaptive/quality pilot are encouraging; additional development '
                        'checks, not pristine validation (v13 used all 30)'),
        exposure_note='all ten IDs were examined in v13 (and six in v10-v12); development data, not a held-out test',
        seeds=seeds, seed_semantics='generation seed -> GenerationRequest.seed -> adapter.generate seed_everything (native sampler RNG)',
        model_path=MODEL, model_revision=REVISION, policy_file=str(policy), manifest=str(MANIFEST), manifest_sha256=MANIFEST_SHA,
        prompt_hashes={i: rows[i]['prompt_hash'] for i in ids},
        arms=arms, arm_hashes=arm_hashes,
        effective_arm_fields={n: {k: c.get(k) for k in ('condition', 'collect', 'fast_t', 'period', 'guard_mode', 'plugin',
                                                        'v5_binary', 'support_binary', 'library', 'policy')}
                              for n, c in configs.items()},
        method_notes=dict(
            T_G_original='v13 fresh original GLOBAL T (Junyu v4 binary) with the exact T-only controller fast path',
            T_P='fresh T every call through the private v5 binary with the mandatory set (current canvas + edge tiles)',
            B8_P='fresh v5 anchor (A8 / new canvas / new epoch) + held anchor bitmap, same mandatory set, v11 current-output consumer',
            CVM_T='B8_P + per-step add-only restoration: restore tile iff max_valid_rows(log rho_a + log s_t) >= log tau',
            fallback='all *_P arms and CVM_T use native SDPA when no complete immutable-prefix tile exists'),
        generation=dict(max_new_tokens=8192, thinking=True, native_adaptive=True, eos=True, temperature_sentinel=0.0, dtype='bfloat16'),
        schedule_seed=schedule_seed, schedule_rule='v13 plan_schedule: (question, seed) blocks shuffled by random.Random(schedule_seed); '
                      'within a block all arms attempt 0 in cyclic rotation by block position, then one warm repeat per arm reversed',
        warm_repeats=1, schedule=schedule,
        metrics=dict(quality='attempt-0 final-channel correctness (frozen v13 scorer); cap/unparsed/failure = incorrect',
                     time='accepted single warm API wall (same cell, no new compilation, tokens/calls/termination identical to attempt 0)',
                     primary='CVM_T vs B8_P (live protection attribution) and CVM_T vs T_P (vs fresh T, same mandatory set)',
                     secondary=['CVM_T vs D_native', 'CVM_T vs T_G_original', 'B8_P vs T_P', 'T_P vs T_G_original',
                                'T_G_original vs D_native'],
                     aggregation='per question mean over seeds of log(candidate/reference); geometric mean over questions; '
                                 'summed-time ratio = calls ratio x time/call ratio; question-cluster bootstrap (descriptive)',
                     parity_checks='D_native and T_G_original attempt-0 token hashes vs the v13 same (id, seed) cells'),
        failure_policy='every scheduled execution runs once; failures recorded and count as incorrect/missing; no silent retries',
        ceilings=dict(executions_total=224, gpu_h=6, elapsed_h=8, this_protocol=len(schedule)),
        planned_executions=len(schedule))
    args.out.mkdir(parents=True, exist_ok=True)
    name = {'dev6': 'frozen_protocol.json', 'extension4': 'frozen_protocol_extension.json', 'smoke': 'smoke_protocol.json'}[args.mode]
    (args.out / name).write_text(json.dumps(protocol, indent=2, sort_keys=True) + '\n')
    print(protocol_id, len(schedule), arm_hashes)


if __name__ == '__main__':
    main()
