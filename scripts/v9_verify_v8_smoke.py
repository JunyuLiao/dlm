"""Compact, redacted verification of the v8 full-budget smoke vs v7 receipts.

Reads the EXISTING private attempt-0 receipts (never rewrites them) and emits
only hashes, counts, termination and frozen-scorer booleans. No prompt, raw
completion, prediction or gold leaves the private receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_hash(tokens) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v7-root', type=Path, required=True)
    parser.add_argument('--v8-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', default=['aime26/2', 'aime26/8'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from experiments.diffusion_gemma_aime26_modes.protocol import final_response, numeric_score
    from experiments.numerical_qk_reuse.runner import _receipt_path
    gold = {str(row['id']): str(row['expected']) for row in json.loads(args.manifest.read_text())}
    versions = dict(
        v7_legacy=dict(root=args.v7_root, phase='smoke', condition='M1',
                       config=args.v7_root / 'configs/smoke.M1.json'),
        v8_summary=dict(root=args.v8_root, phase='diagnostic', condition='M1',
                        config=args.v8_root / 'configs/diagnostic.M1.json'))
    configs = {}
    for name, spec in versions.items():
        config = json.loads(spec['config'].read_text())
        configs[name] = dict(config_path=str(spec['config']), config_sha256=sha_file(spec['config']),
                             fingerprint=config['fingerprint'], diagnostic=config['diagnostic'],
                             selector=config.get('selector', 'legacy_recompute (field absent: pre-v8)'),
                             selector_layers=config.get('selector_layers'),
                             output_mode=config['output_mode'], support=config['support'],
                             decision_interval=config['decision_interval'],
                             score_refresh_period=config['score_refresh_period'],
                             policy_sha256=config['policy_sha256'], manifest_sha256=config['manifest_sha256'],
                             model_revision=config['revision'],
                             source_hashes=config['source_hashes'])
    rows = []
    for id_ in args.ids:
        entry = dict(id=id_)
        for name, spec in versions.items():
            path = _receipt_path(spec['root'], spec['phase'], spec['condition'], 42, id_)
            receipt = json.loads(path.read_text())
            score = numeric_score(final_response(receipt['raw_completion'], True), gold[id_])
            entry[name] = dict(receipt_path=str(path), receipt_sha256=sha_file(path),
                               fingerprint=receipt['fingerprint'], phase=receipt['phase'],
                               prompt_token_hash=receipt['prompt_token_hash'],
                               completion_token_hash=token_hash(receipt['completion_tokens']),
                               output_tokens=receipt['output_tokens'],
                               decoder_calls=receipt['total_decoder_calls'],
                               canvases=len(receipt['per_canvas']),
                               per_canvas_calls=[c['decoder_calls'] for c in receipt['per_canvas']],
                               termination=receipt['termination_reason'],
                               frozen_scorer_correct=bool(score['correct']),
                               unparsed=score['extracted'] is None,
                               diagnostics_recorded=receipt.get('diagnostics') is not None,
                               summary_hits=(receipt.get('counters') or {}).get('summary_hits'),
                               request_wall_seconds_NOT_PRODUCTION_TIMING=receipt['request_wall_seconds'])
        a, b = entry['v7_legacy'], entry['v8_summary']
        entry['tokens_identical'] = a['completion_token_hash'] == b['completion_token_hash']
        entry['calls_identical'] = a['per_canvas_calls'] == b['per_canvas_calls']
        entry['termination_identical'] = a['termination'] == b['termination']
        rows.append(entry)
    report = dict(schema='v8_smoke_receipt_verification_v1', configs=configs, rows=rows,
                  all_tokens_identical=all(r['tokens_identical'] for r in rows),
                  all_calls_identical=all(r['calls_identical'] for r in rows),
                  notes=['v8 receipts ran with diagnostic=True (quantiles/entropy per step): their walls are '
                         'NOT production timings; v7 walls are pre-v9 and not contemporary either',
                         'score booleans are the frozen final-channel scorer on attempt 0; this 2-question '
                         'subset shows 2/2 for both; the v7 four-question 3/4 is historical, not a v8 score',
                         'real_state_qualify.json (v8) is the 512-token trajectory check, NOT this receipt',
                         'selector enabled on LOCAL layers only in production; GLOBAL summaries were only '
                         'checked in isolation'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({r['id']: [r['tokens_identical'], r['calls_identical'],
                                r['v7_legacy']['frozen_scorer_correct'], r['v8_summary']['frozen_scorer_correct']]
                      for r in rows}))


if __name__ == '__main__':
    main()
