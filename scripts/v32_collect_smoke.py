"""Collect the sanitized one-cell H100 smoke records of every arm of this study into the result
directory. Records carry no prompt, no gold and no generated text: only counts, hashes, receipts and
timings, which is what the study is allowed to publish.

usage: python scripts/v32_collect_smoke.py RESULTS_DIR [WORK_ROOT ...]
"""
from __future__ import annotations

import glob
import json
import os
import sys

KEYS = ('value_selector', 'value_threshold', 'value_budget_tokens', 'value_candidates',
        'value_evaluations', 'value_forced_keep', 'value_forced_skip', 'value_threshold_keeps',
        'value_blocked', 'value_approximation_calls', 'value_mu_passes', 'value_sketch_builds',
        'value_sketch_reuses', 'value_stat_bytes', 'value_passes', 'value_drop_fraction',
        'value_shortlist', 'value_exact_max', 'value_scan', 'value_last_counters',
        'value_last_objective', 'value_projection', 'value_statistics_kernel', 'value_consumer',
        'value_output_path', 'value_layer_scope', 'value_aggregation', 'value_protect_sink_tokens',
        'value_protect_recent_tokens')
COUNTERS = ('global_calls', 'prefix_copies', 'canvas_refreshes', 'invalidates', 'begins', 'observes',
            'order_errors', 'split_fa4_calls', 'split_list_builds', 'mage_selections',
            'mage_reused_calls', 'mage_reselections', 'mage_carried_calls', 'mage_warm_dense_calls',
            'mage_kept_prefix_tiles', 'mage_prefix_tiles', 'mage_sticky_units', 'trigger_checks',
            'triggers', 'trigger_step_sum', 'kept_prefix_tiles', 'sparse_prefix_tiles',
            'kept_prefix_fraction', 'sparse_only_kept_tiles', 'sparse_only_prefix_tiles',
            'sparse_kept_prefix_fraction', 'global_prefix_tiles', 'global_prefix_work_fraction',
            'global_eligible_tiles', 'global_kept_tiles', 'local_calls', 'local_observations',
            'local_eligible_tiles', 'local_kept_tiles')


def collect(root):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, 'run_*', 'public', '*.jsonl'))):
        run = os.path.basename(os.path.dirname(os.path.dirname(path)))
        for line in open(path, encoding='utf-8'):
            if not line.strip():
                continue
            r = json.loads(line)
            a = (r.get('receipts') or {}).get('adapter') or {}
            rows.append(dict(
                run=run, record_file=os.path.basename(path), dataset=r['dataset'], index=r['index'],
                panel_seed=r['panel_seed'], repeat=r['repeat'], arm=r['arm'],
                cudagraph_mode=r['cudagraph_mode'], fix_51994=r.get('fix_51994'),
                prompt_tokens=r['prompt_tokens'], budget=r['budget'],
                manifest_sha256=r['manifest_sha256'], prompt_sha256=r['prompt_sha256'],
                rng_seed=r['rng_seed'], adapter_sha256=r['adapter_sha256'],
                output_tokens=r['output_tokens'], canvases=r['canvases'], N=r['denoise_forwards'],
                engine_steps=r['engine_steps'], prefill_steps=r['prefill_steps'],
                prefill_s=r['prefill_s'], decode_s=r['decode_s'], wall_s=r['wall_s'],
                step_median_ms=r['step_median_ms'], finish_reason=r['finish_reason'],
                output_hash=r['output_hash'],
                **{k: a.get(k) for k in KEYS},
                counters={k: a.get(k) for k in COUNTERS if a.get(k) is not None}))
    return rows


def main():
    out_dir = sys.argv[1]
    roots = sys.argv[2:] or ['/home/exouser/dyh/ljy_value_20261007']
    rows = []
    for root in roots:
        rows += [r for r in collect(root) if r not in rows]
    path = os.path.join(out_dir, 'smoke_records.json')
    with open(path, 'w') as fh:
        json.dump(rows, fh, indent=1, sort_keys=True)
    print(len(rows), 'records ->', path)


if __name__ == '__main__':
    main()