"""Build measurement_contract_and_dispatch.{json,md} and replay_state_tests.json
from v9 step-replay JSONs (per-repetition evidence, not two precall digests)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, nargs='+', required=True)
    parser.add_argument('--deploy-sha', required=True)
    parser.add_argument('--pytest-summary', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    contract, state = dict(schema='v9_measurement_contract_v1', deploy_sha=args.deploy_sha, windows={}), \
        dict(schema='v9_replay_state_tests_v1', deploy_sha=args.deploy_sha,
             cpu_gpu_unit_tests=args.pytest_summary, windows={})
    lines = ['# v9 measurement contract and dispatch', '',
             f'Deployed source `{args.deploy_sha}`. Evidence: bounded dispatch spy (one untimed probe per row) '
             'and execution flags read at the step boundary inside `adapter.generate` vs in each replay.', '']
    for path in args.replay:
        report = json.loads(path.read_text())
        key = f"{report['id']}@canvas{report['canvas']}"
        captured = report['captured'][0]
        contract['windows'][key] = dict(
            source_json=str(path), source_sha256=sha(path),
            absolute_position=captured['absolute'],
            stored_prefix_layer0_local=captured['stored_prefix']['0'],
            stored_prefix_layer5_global=captured['stored_prefix']['5'],
            production_context=captured['context'],
            replay_context_mismatches={k: v['mismatches_vs_production'] for k, v in report['contexts'].items()},
            dispatch=report['dispatch'], snapshot_inventory=report['snapshot_inventory'],
            capture_trajectory=report['capture_trajectory'])
        rows = {}
        for name, row in report['rows'].items():
            reps = row['per_repetition']
            rows[name] = dict(
                median_ms=row['median_ms'], min_ms=row['min_ms'], max_ms=row['max_ms'],
                host_wall_median_ms=row['wall_median_ms'], first_observation_ms=row['first_observation_ms'],
                warmup=row['warmup'], reps=row['reps'], label=row.get('label'),
                inputs_identical_every_repetition=row['inputs_identical_every_repetition'],
                outputs_identical_every_repetition=row['outputs_identical_every_repetition'],
                per_repetition=[{k: r.get(k) for k in ('index', 'warmup', 'event_ms', 'input_digest',
                                                       'output_digest', 'telemetry_delta', 'router_step')}
                                for r in reps])
        state['windows'][key] = rows
        lines += [f'## {key} (absolute position {captured["absolute"]}, local stored prefix '
                  f'{captured["stored_prefix"]["0"]}, global {captured["stored_prefix"]["5"]})', '',
                  '| row | registry entry | native SDPA | dense_eager | selector | median ms | min | first |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        dispatch_for = lambda name: report['dispatch'].get(name) or report['dispatch'].get(name.split('.')[0]) \
            or report['dispatch'].get(name.replace('.anchor_inputs', '')) or {}
        for name, row in report['rows'].items():
            d = dispatch_for(name)
            lines.append(f"| {name} | `{d.get('registry_entry', '?').rsplit('.', 1)[-1]}` | {d.get('native_sdpa')} | "
                         f"{d.get('dense_eager')} | {d.get('numerical_selector')} | {row['median_ms']:.2f} | "
                         f"{row['min_ms']:.2f} | {row['first_observation_ms']:.2f} |")
        lines += ['', f"Context mismatches vs production: "
                  f"{ {k: v['mismatches_vs_production'] for k, v in report['contexts'].items()} }", '',
                  f"Production flags: `{json.dumps({k: captured['context'][k] for k in ('inference_mode', 'grad_enabled', 'autocast_cuda', 'attn_implementation', 'tf32_matmul', 'current_stream')})}`", '']
    lines += ['## Definitions', '',
              '- **native** = no binding at all; `ALL_ATTENTION_FUNCTIONS["sdpa"]` is '
              '`transformers.integrations.sdpa_attention.sdpa_attention_forward`; no T observer. '
              'Used for D in the request timing (runner condition `native_dense`, diagnostic off).',
              '- **dense_eager_same_mask** = `_install_dense` + `attention_override=None` -> the BLASST '
              'dispatcher -> `dense_eager_attention_forward`. This is what v8 labeled `native_dense` (183.24 ms). '
              'Kept only as a separately named reference.',
              '- **sparse_<selector>** = production `integration.install` (M1, preqk current output) plus exactly '
              'one `observe` wrapper; T bookkeeping is inside the timed step. Native pays no T work.',
              '- Mask disclosure: L/S use `legacy_junyu_mask` (query-relative local window crop); native SDPA '
              'receives no sliding-window crop beyond the cache\'s own window. L vs S is algorithm-preserving; '
              'L/S vs D additionally differs in support.',
              '- Event spans include host launch gaps; they are not GPU-active unions.']
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'measurement_contract_and_dispatch.json').write_text(json.dumps(contract, indent=2, sort_keys=True) + '\n')
    (args.out / 'measurement_contract_and_dispatch.md').write_text('\n'.join(lines) + '\n')
    (args.out / 'replay_state_tests.json').write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
