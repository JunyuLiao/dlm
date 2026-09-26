"""Freeze the v15 scored LongBench-v2 long-context development panel BEFORE any panel output (CPU only).

Eligibility: every LongBench-v2 item (pinned revision) whose UNTRUNCATED NeMo prompt, rendered and
chat-templated exactly as the request path (thinking ON), has 10,000-20,000 tokens.
Selection (no answers/outputs/runtime/sparsity inspected): strata = (domain, length bin) with bins
[10000,15000) and [15000,20000]; inside a stratum items are ordered by sha256('v15/<seed>/<_id>');
round-robin passes over (domains alphabetically x bins in order) take the next item of each
non-empty stratum until N=12 (or the pool is exhausted -> smaller panel, disclosed).
The two v14 cost-profile items are excluded when the remaining pool still has >= N items.
Outputs: private generation manifest (NO gold), private gold file (scorer only), public
panel_selection.json and frozen_protocol.json (five v14 arms copied verbatim, v13 seed-safe schedule).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from scripts import v15_longbench_task as task
from scripts.v10_request_runs import arm_config
from scripts.v13_seed_runs import arm_config_hash, plan_schedule

MODEL = '/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b'
REVISION = 'f7f5b7f5fa82ffc52addd066915886d497f5517b'
SELECTION_SEED = 20260926
SCHEDULE_SEED = 20260929
N = 12
BINS = (('10-15K', 10000, 15000), ('15-20K', 15000, 20001))
PROFILE_IDS = {'66f39aa7821e116aacb2da76': 'v14 cost profile lb/21', '66f78ecfbb02136c067c2f12': 'v14 cost profile lb/40'}
V14_PROTOCOL = Path('results/numerical_qk_cvm_t_20260926/frozen_protocol.json')
PRIOR_RESULT_ROOTS = (Path('results'), Path('/home/exouser/ljy/dlm/results'))


def sha(x):
    return hashlib.sha256(x if isinstance(x, bytes) else str(x).encode()).hexdigest()


def bin_of(n):
    return next((name for name, lo, hi in BINS if lo <= n < hi), None)


def select(pool, n=N, seed=SELECTION_SEED):
    strata = {}
    for r in pool:
        strata.setdefault((r['domain'], r['bin']), []).append(r)
    for rows in strata.values():
        rows.sort(key=lambda r: sha(f'v15/{seed}/{r["source_id"]}'))
    order = [(d, b) for d in sorted({r['domain'] for r in pool}) for b, _, _ in BINS]
    chosen, depth = [], 0
    while len(chosen) < n and any(len(strata.get(k, [])) > depth for k in order):
        for key in order:
            if len(chosen) < n and len(strata.get(key, [])) > depth:
                chosen.append(strata[key][depth])
        depth += 1
    return chosen


def prior_exposure(source_ids):
    hits = {i: [] for i in source_ids}
    for root in PRIOR_RESULT_ROOTS:
        if not root.exists():
            continue
        out = subprocess.run(['grep', '-rlF', '--include=*.json', '--include=*.jsonl', '--include=*.csv',
                              *[x for i in source_ids for x in ('-e', i)], str(root)], capture_output=True, text=True).stdout
        for path in out.splitlines():
            text = Path(path).read_text(errors='ignore')
            for i in source_ids:
                if i in text:
                    hits[i].append(str(Path(path).parent))
    return {i: sorted(set(v)) for i, v in hits.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--authorization', required=True)
    a = p.parse_args()
    from dllm.models import create_adapter
    contract = task.contract_identity()
    data = json.loads(task.LB_DATA.read_text())
    if len({r['_id'] for r in data}) != len(data):
        raise ValueError('duplicate LongBench IDs')
    adapter = create_adapter('diffusion_gemma', MODEL, device='cpu', precision='float32', revision=REVISION).load_tokenizer()
    pool, rendered, too_long_chars, counts = [], {}, 0, dict(below=0, above=0, eligible=0)
    for index, item in enumerate(data):
        if len(item['context']) > 400_000:              # > 20 chars/token would be needed to fit 20K tokens
            too_long_chars += 1
            counts['above'] += 1
            continue
        r = task.render(adapter, item)
        n = r['prompt_tokens_n']
        if n < 10000:
            counts['below'] += 1
        elif n > 20000:
            counts['above'] += 1
        else:
            counts['eligible'] += 1
            rendered[item['_id']] = r
            pool.append(dict(source_id=item['_id'], dataset_index=index, domain=item['domain'], sub_domain=item['sub_domain'],
                             difficulty=item['difficulty'], length_band=item['length'], prompt_tokens_n=n, bin=bin_of(n)))
    remaining = [r for r in pool if r['source_id'] not in PROFILE_IDS]
    profile_excluded = len(remaining) >= N
    chosen = select(remaining if profile_excluded else pool)
    ids = [f'longbench_v2/{r["source_id"]}' for r in chosen]
    exposure = prior_exposure([r['source_id'] for r in chosen])
    by_id = {r['_id']: r for r in data}
    manifest, gold = [], {}
    for r in chosen:
        x = rendered[r['source_id']]
        manifest.append(dict(id=f'longbench_v2/{r["source_id"]}', source_id=r['source_id'], benchmark='longbench_v2',
                             prompt=x['prompt'], prompt_hash=x['prompt_hash'], prompt_tokens=x['prompt_tokens'],
                             prompt_tokens_n=x['prompt_tokens_n'], thinking=True, generation_budget=8192,
                             domain=r['domain'], sub_domain=r['sub_domain'], bin=r['bin']))
        gold[f'longbench_v2/{r["source_id"]}'] = by_id[r['source_id']]['answer']
    a.private.mkdir(parents=True, exist_ok=True)
    gen_path, gold_path = a.private / 'generation_manifest.json', a.private / 'gold_scorer_only.json'
    gen_path.write_text(json.dumps(manifest) + '\n')
    gold_path.write_text(json.dumps(gold, sort_keys=True) + '\n')
    strata_counts = {}
    for r in pool:
        strata_counts.setdefault(r['domain'], {}).setdefault(r['bin'], 0)
        strata_counts[r['domain']][r['bin']] += 1
    selection = dict(
        schema='v15_panel_selection_v1', frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        rule=__doc__.split('Outputs:')[0].strip(), selection_seed=SELECTION_SEED, contract=contract,
        dataset_items=len(data), prefiltered_over_400k_chars=too_long_chars, token_counts=counts,
        eligible_by_domain_bin=strata_counts, eligible_total=len(pool),
        cost_profile_items_excluded=profile_excluded, cost_profile_items=PROFILE_IDS,
        cost_profile_items_in_eligible_pool=[r['source_id'] for r in pool if r['source_id'] in PROFILE_IDS],
        panel_size=len(chosen), shortfall=N - len(chosen),
        panel=[dict(id=f'longbench_v2/{r["source_id"]}', source_id=r['source_id'], dataset_index=r['dataset_index'],
                    domain=r['domain'], sub_domain=r['sub_domain'], difficulty=r['difficulty'], length_band=r['length_band'],
                    prompt_tokens_n=r['prompt_tokens_n'], bin=r['bin'], prompt_sha256=rendered[r['source_id']]['prompt_hash'],
                    raw_prompt_sha256=rendered[r['source_id']]['raw_prompt_hash'],
                    literal_special_token_escapes=rendered[r['source_id']]['literal_special_token_escapes'],
                    prior_project_exposure=exposure[r['source_id']]) for r in chosen],
        exposure_note=('prior project LongBench panels (truncated inputs, thinking OFF, other methods) are listed per item; '
                       'this is development/generalization evidence, not a pristine held-out test'),
        private_generation_manifest_sha256=sha(gen_path.read_bytes()), private_gold_sha256=sha(gold_path.read_bytes()))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'panel_selection.json').write_text(json.dumps(selection, indent=2, sort_keys=True) + '\n')
    # ---------------- frozen protocol: five v14 arms copied verbatim
    v14 = json.loads(V14_PROTOCOL.read_text())
    arms = v14['arms']
    task_identity = dict(contract=contract, max_new_tokens=8192, thinking=True, model_revision=REVISION,
                         generation_manifest_sha256=selection['private_generation_manifest_sha256'])
    protocol_id = 'v15_lbv2_scored_' + sha(json.dumps(task_identity, sort_keys=True))[:12]
    root = Path(__file__).resolve().parents[1]
    policy = root / 'results/query_adaptive_v3/configs/frozen_policies.json'
    ns = SimpleNamespace(phase=protocol_id, ids=ids, manifest=gen_path, policy=policy, model=Path(MODEL), revision=REVISION,
                         seeds=[17, 29])
    configs = {name: arm_config(ns, arm) for name, arm in arms.items()}
    arm_hashes = {name: arm_config_hash(c) for name, c in configs.items()}
    schedule = plan_schedule(protocol_id, REVISION, arm_hashes, ids, [17, 29], SCHEDULE_SEED)
    protocol = dict(
        schema='v15_frozen_protocol_v1', protocol_id=protocol_id, frozen_utc=selection['frozen_utc'],
        authorization=a.authorization, mode='lbv2_dev12', ids=ids, development_ids=ids, seeds=[17, 29],
        task_identity=task_identity, model_path=MODEL, model_revision=REVISION, policy_file=str(policy),
        manifest=str(gen_path), manifest_sha256=selection['private_generation_manifest_sha256'],
        gold_file_scorer_only=str(gold_path), prompt_hashes={f'longbench_v2/{r["source_id"]}': rendered[r['source_id']]['prompt_hash'] for r in chosen},
        arms=arms, arms_copied_from=str(V14_PROTOCOL), arm_hashes=arm_hashes,
        effective_arm_fields={n: {k: c.get(k) for k in ('condition', 'collect', 'fast_t', 'period', 'guard_mode', 'plugin',
                                                        'v5_binary', 'support_binary', 'library', 'policy')} for n, c in configs.items()},
        generation=dict(max_new_tokens=8192, thinking=True, native_adaptive=True, eos=True, temperature_sentinel=0.0,
                        dtype='bfloat16', truncation=False, request_path='experiments/numerical_qk_reuse/runner.py:_one'),
        scoring=dict(version=task.SCORER_VERSION, module='scripts/v15_longbench_task.py',
                     task_score='NeMo eval_mcq (default config) on the final answer channel; a valid choice in the final channel '
                                'counts even at a length cap (task contract); no final channel -> unparsed -> incorrect',
                     strict_score='task-correct AND termination == eos', thinking_channel_mined=False),
        schedule_seed=SCHEDULE_SEED, schedule_rule='v13 plan_schedule: (question, seed) blocks shuffled by random.Random(seed); '
                    'attempt 0 of all arms in cyclic rotation by block position, then one warm repeat per arm in reversed order',
        warm_repeats=1, schedule=schedule,
        timing=dict(primary='accepted warm request wall: torch.cuda.synchronize() + perf_counter around adapter.generate '
                            '(includes real prefill/encoder, all denoising, anchors, commits, sampling, output handling; '
                            'excludes model load); every request builds its own KV cache',
                    warm_acceptance='same token hash, per-canvas calls, termination; no Triton compile and no new shared object '
                                    'loaded during the warm execution',
                    unaccepted_policy='an unaccepted warm row stays unaccepted and its pairs are reported as timing-incomplete; '
                                      'setup executions are NOT used to repeat blocks (frozen before inference)',
                    generation_latency='N/A: no qualified low-overhead initial-prefill-end boundary',
                    tbt='N/A: no true committed-output events'),
        comparisons=[['CVM_T', 'B8_P'], ['CVM_T', 'T_P'], ['CVM_T', 'D_native'], ['CVM_T', 'T_G_original'],
                     ['B8_P', 'D_native'], ['T_P', 'T_G_original']],
        ceilings=dict(executions_total=250, panel=len(schedule), setup_max=10, gpu_h=6, elapsed_h=8),
        planned_executions=len(schedule))
    (a.out / 'frozen_protocol.json').write_text(json.dumps(protocol, indent=2, sort_keys=True) + '\n')
    print(protocol_id, len(ids), len(schedule), counts, strata_counts, 'profile excluded', profile_excluded)
    for r in selection['panel']:
        print(r['id'], r['domain'], r['bin'], r['prompt_tokens_n'], len(r['prior_project_exposure']))


if __name__ == '__main__':
    main()
