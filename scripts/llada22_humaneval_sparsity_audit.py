"""CPU-only accounting audit of saved LLaDA2.2 HumanEval tile sparsity."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


ARMS = ('dense_native', 'allkept_fa4', 'mass_topk_reuse_r4',
        'value_v2_gaussian32_reuse_r4', 'value_v3a_singleton_reuse_r4')
UNITS_PER_FORWARD = 20 * 16  # 20 layers, 16 query heads, one Q128 block


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('base', type=Path)
    parser.add_argument('diagnostic', type=Path)
    parser.add_argument('published_summary', type=Path)
    parser.add_argument('out', type=Path)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    published = read(a.published_summary)
    sources, panels = [], {}
    for arm in ARMS:
        rows = []
        leaves = sorted((a.base / 'runs/llada22/humaneval' / arm).glob('shard*/attempt001'))
        assert len(leaves) == 4
        for leaf in leaves:
            raw = leaf / 'records.jsonl'
            receipt, config = read(leaf / 'receipt.json'), read(leaf / 'config.json')
            part = [json.loads(line) for line in raw.read_text().splitlines()]
            assert receipt['records_sha256'] == sha(raw)
            assert read(leaf / 'final_complete.json') == {'status': 'complete', 'cells': len(part)}
            for r in part:
                for key in ('arm', 'source_sha256', 'runtime_sha256', 'protocol_sha256',
                            'manifest_sha256', 'model_revision'):
                    assert r[key] == config[key], (arm, key)
                assert r['seed'] == 42 and r['prompt_tokens'] >= 64
                counts = r['counts']
                assert counts['eligible_tiles'] % UNITS_PER_FORWARD == 0
                assert counts['eligible_tiles'] >= counts['kept_tiles'] >= UNITS_PER_FORWARD * counts['denoising_forwards']
            rows += part
            sources.append(dict(path=str(raw), sha256=sha(raw)))
        assert len(rows) == len({r['id'] for r in rows}) == 164
        n = sum(r['counts']['denoising_forwards'] for r in rows)
        eligible = sum(r['counts']['eligible_tiles'] for r in rows)
        kept = sum(r['counts']['kept_tiles'] for r in rows)
        mandatory = UNITS_PER_FORWARD * n
        old = published['llada22/humaneval/' + arm]
        assert (n, eligible, kept) == (old['N'], old['eligible_tiles'], old['kept_tiles'])
        assert 1 - kept / eligible == old['consumer_sparsity']
        panels[arm] = dict(requests=164, forwards=n, eligible_tiles=eligible,
                          kept_tiles=kept, mandatory_tiles=mandatory,
                          optional_eligible_tiles=eligible-mandatory,
                          optional_kept_tiles=kept-mandatory,
                          pooled_tile_sparsity=1-kept/eligible,
                          optional_tile_sparsity=1-(kept-mandatory)/(eligible-mandatory),
                          mean_request_tile_sparsity=statistics.mean(1-r['counts']['kept_tiles']/r['counts']['eligible_tiles'] for r in rows),
                          mean_eligible_tiles_per_layer_head_call=eligible/mandatory,
                          mean_kept_tiles_per_layer_head_call=kept/mandatory,
                          mean_optional_kept_tiles_per_layer_head_call=(kept-mandatory)/mandatory,
                          mean_output_tokens=sum(r['output_tokens'] for r in rows)/164)

    native_path = a.diagnostic / 'attempt007/receipt.json'
    native = read(native_path)
    assert read(a.diagnostic / 'attempt007/final_complete.json')['receipt_sha256'] == sha(native_path)
    geometry = []
    for r in native['results']:
        steps = r['steps']
        assert len(steps) == r['counts']['denoising_forwards']
        # Every aligned 32-token block intersects exactly one mandatory KV64 tile.
        assert all(s['block_start'] % 32 == 0 for s in steps)
        histogram = Counter((s['block_start'] + 32 + 63)//64 for s in steps)
        eligible = UNITS_PER_FORWARD * sum(j*count for j,count in histogram.items())
        # Torch round uses ties-to-even: J=3 ->2, J=5 ->2, J=7 ->4.
        fixed_kept = UNITS_PER_FORWARD * sum(max(1,round(j/2))*count for j,count in histogram.items())
        assert eligible == r['counts']['eligible_tiles']
        if r['arm'] in (ARMS[2], ARMS[4]):
            assert fixed_kept == r['counts']['kept_tiles']
        elif r['arm'] == 'allkept_fa4':
            assert eligible == r['counts']['kept_tiles']
        valid_key_exposures = UNITS_PER_FORWARD * sum(s['block_start']+32 for s in steps)
        # Only the mandatory last tile can have fewer than 64 valid keys.
        # It is always kept, so every skipped tile has 64 valid keys.
        token_sparsity = 64*(eligible-r['counts']['kept_tiles'])/valid_key_exposures
        geometry.append(dict(id=r['id'], arm=r['arm'], histogram_J=dict(sorted(histogram.items())),
                             eligible_tiles=eligible, actual_kept_tiles=r['counts']['kept_tiles'],
                             v31_budget_on_same_geometry_kept_tiles=fixed_kept,
                             tile_sparsity=1-r['counts']['kept_tiles']/eligible,
                             valid_query_key_pair_sparsity=token_sparsity))

    metrics_path = a.diagnostic / 'attempt008/metrics.json'
    metrics = read(metrics_path)
    marker = read(a.diagnostic / 'attempt008/final_complete.json')
    assert marker['metrics_sha256'] == sha(metrics_path)
    same_states = []
    for item in sorted({r['id'] for r in metrics}):
        for method in ('actual_v2', 'v31_half', 'mass_same_count_each_head'):
            rows = [r for r in metrics if r['id'] == item and r['label'] == 'first_eos_write' and r['method'] == method]
            assert len(rows) == 20
            eligible, kept = sum(r['eligible'] for r in rows), sum(r['kept'] for r in rows)
            event = next(r for r in native['results'] if r['id'] == item and r['arm'] == ARMS[3])
            first = next(s for s in event['steps'] if s['eos_writes'])
            nk = first['block_start'] + 32
            assert eligible == UNITS_PER_FORWARD * ((nk+63)//64)
            same_states.append(dict(id=item, method=method, key_tokens=nk, J=eligible//UNITS_PER_FORWARD,
                                    mandatory_per_head=1, kept_per_head=kept/UNITS_PER_FORWARD,
                                    optional_kept_per_head=kept/UNITS_PER_FORWARD-1,
                                    tile_sparsity=1-kept/eligible,
                                    valid_query_key_pair_sparsity=64*(eligible-kept)/(UNITS_PER_FORWARD*nk),
                                    first_tile_keep_fraction=statistics.mean(r['tile_keep_fraction'][0] for r in rows),
                                    retained_mass=statistics.mean(r['retained_mass']['mean'] for r in rows),
                                    output_error_over_value_rms=statistics.mean(r['error_over_value_rms']['mean'] for r in rows)))
    sources += [dict(path=str(path),sha256=sha(path)) for path in (native_path,metrics_path,a.published_summary)]
    summary = dict(status='pass', cpu_only=True, no_new_generation=True,
                   definition='1 - sum actual consumed H x Q128 x KV64 tiles / sum eligible tiles over each arm native denoising trajectory; includes mandatory/current and selection forwards, excludes discovery, prefill and commit',
                   mandatory_definition='Only the KV64 tile intersecting aligned current 32-token canvas; first KV64 tile is selector-forced by V2 but optional under the shared mask',
                   optional_denominator_note='KV64 tile optional means outside mandatory tile; if current block occupies latter half of tile, mandatory also includes preceding 32 keys',
                   full_cohort=panels, traced_geometry=geometry, same_states=same_states,
                   J_budget=[dict(J=j,mandatory=1,V31_total=max(1,round(j/2)),V31_optional=max(1,round(j/2))-1,
                                  V2_if_no_optional_update_total=2,V2_if_no_optional_update_optional=1)
                             for j in range(2,11)],
                   sources=sources,source_sha256=sha(Path(__file__)))
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.out/'receipt.json').write_text(json.dumps(dict(status='pass',requests_checked=820,shards_checked=20,
                                    native_geometry_requests_checked=len(geometry),same_state_comparisons=len(same_states),
                                    old_aggregate_mismatches=0, summary_sha256=sha(a.out/'summary.json')),indent=2)+'\n')
    (a.out/'final_complete.json').write_text(json.dumps(dict(status='complete',receipt_sha256=sha(a.out/'receipt.json')),indent=2)+'\n')
    print(json.dumps(dict(status='pass',panels=panels,same_states=same_states),indent=2))


if __name__ == '__main__':
    main()
