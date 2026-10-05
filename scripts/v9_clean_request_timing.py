"""v9 clean paired natural-request timing: D / L / S in ONE process.

  D  native_dense                untouched native model (no binding, no T)
  L  M1 preqk current output     selector legacy_recompute
  S  identical M1                selector prefix_block_summary (LOCAL layers)

Every execution goes through ``runner._one`` -- the production adapter call
with its own ``seed_everything(42)``, native adaptive stopping, thinking ON,
8192 budget and EOS -- with ``diagnostic=False``. Round 0 is attempt 0
(quality, cold-inclusive: in-process Triton/cuBLAS specialization is paid by
whichever condition first meets a shape). Rounds 1..R are timing-only repeats
with the method order reversed/alternated; they never replace attempt 0.

Boundaries (declared, not subtracted):
  api_wall_s     synchronize -> adapter.generate -> synchronize (runner._one)
  outer_wall_s   whole runner._one: binding install/teardown, T controller,
                 generate, router record transfer, receipt assembly
Nothing is subtracted from either. Compilation is observed, not removed:
Triton in-memory specializations and disk-cache entries are counted around
each run (outside the timed region).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch

CONDITIONS = dict(
    D=dict(condition='native_dense', selector='legacy_recompute', plugin=None),
    L=dict(condition='M1', selector='legacy_recompute',
           plugin='experiments.numerical_qk_reuse.integration:install'),
    S=dict(condition='M1', selector='prefix_block_summary',
           plugin='experiments.numerical_qk_reuse.integration:install'))
ORDERS = (('D', 'L', 'S'), ('S', 'L', 'D'))


def token_hash(tokens) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def triton_specializations() -> dict[str, int]:
    """In-memory compiled specializations per JIT kernel we own (host-only read)."""
    import importlib
    out = {}
    for module_name in ('experiments.numerical_qk_reuse.cached_executor',):
        module = importlib.import_module(module_name)
        from triton.runtime.jit import JITFunction
        for name, value in vars(module).items():
            if not isinstance(value, JITFunction):
                continue
            # Triton 3.2: JITFunction.cache = {device: {specialization_key: kernel}}
            out[f'{module_name.rsplit(".", 1)[-1]}.{name}'] = sum(len(v) for v in value.cache.values())
    return out


def disk_cache_entries() -> int:
    root = Path(os.environ.get('TRITON_CACHE_DIR', Path.home() / '.triton/cache'))
    return len(os.listdir(root)) if root.is_dir() else 0


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def config_for(args, key):
    from experiments.numerical_qk_reuse.runner import _config
    spec = CONDITIONS[key]
    namespace = SimpleNamespace(
        condition=spec['condition'], phase='request_timing_v9', ids=args.ids, seeds=[42],
        manifest=args.manifest, policy=args.policy, policy_name='T_s50', model=args.model,
        revision=args.revision, decision_interval=1, score_refresh_period=8,
        support='legacy_junyu_mask', output_mode='historical_route_preqk_current_output',
        selector=spec['selector'], selector_layers='local', library=None, torch_library=None,
        plugin=spec['plugin'], diagnostic=False, timing_events=False, extra_source=[])
    config = _config(namespace)
    config['v9_label'] = key
    return config


def redacted(receipt: dict) -> dict:
    return dict(fingerprint=receipt['fingerprint'], id=receipt['id'], condition=receipt['condition'],
                prompt_token_hash=receipt['prompt_token_hash'],
                completion_token_hash=token_hash(receipt['completion_tokens']),
                output_tokens=receipt['output_tokens'], termination=receipt['termination_reason'],
                decoder_calls=receipt['total_decoder_calls'],
                per_canvas_calls=[c['decoder_calls'] for c in receipt['per_canvas']],
                canvases=len(receipt['per_canvas']), api_wall_s=receipt['request_wall_seconds'],
                counters=receipt['counters'], diagnostics_present=receipt['diagnostics'] is not None,
                routing_records=None if receipt['routing'] is None else len(receipt['routing']))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', default=['aime26/2', 'aime26/8'])
    parser.add_argument('--rounds', type=int, default=3, help='round 0 = attempt 0; max 3 -> 18 executions')
    parser.add_argument('--private', type=Path, required=True, help='raw attempt-0 receipts (never in Git)')
    parser.add_argument('--ledger', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 3 or len(args.ids) != 2:
        raise SystemExit('bounded to 2 IDs x 3 conditions x <=3 rounds (18 executions)')
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse.runner import GOLD_FIELDS, _atomic, _one, _rows, _selected
    rows = _selected([{k: v for k, v in row.items() if k not in GOLD_FIELDS}
                      for row in _rows(args.manifest)], args.ids, 'request_timing_v9')
    configs = {key: config_for(args, key) for key in CONDITIONS}
    for key, config in configs.items():
        _atomic(args.private / 'configs' / f'{key}.json', config)
        if config['diagnostic'] or config['timing_events']:
            raise SystemExit('diagnostic/timing events must be off')
    append(args.ledger, dict(event='start', pid=os.getpid(), when=time.time(),
                             gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                             cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                             fingerprints={k: c['fingerprint'] for k, c in configs.items()},
                             source_hashes=configs['S']['source_hashes'],
                             triton=__import__('triton').__version__,
                             disk_cache_entries=disk_cache_entries()))
    started = time.perf_counter()
    adapter = create_adapter('diffusion_gemma', configs['D']['model'], device='cuda',
                             precision='bfloat16', revision=configs['D']['revision']).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    append(args.ledger, dict(event='model_loaded', seconds=time.perf_counter() - started))
    seen: dict[tuple, dict] = {}
    for round_index in range(args.rounds):
        for id_position, row in enumerate(rows):
            order = ORDERS[(round_index + id_position) % 2]
            for position, key in enumerate(order):
                config = configs[key]
                specs_before, disk_before = triton_specializations(), disk_cache_entries()
                torch.cuda.reset_peak_memory_stats()
                outer_started = time.perf_counter()
                receipt = _one(adapter, row, 42, config)
                outer_wall = time.perf_counter() - outer_started
                specs_after, disk_after = triton_specializations(), disk_cache_entries()
                record = redacted(receipt)
                if receipt['diagnostics'] is not None:
                    raise AssertionError('diagnostics were recorded in a clean timing run')
                new_specs = {k: specs_after[k] - specs_before.get(k, 0) for k in specs_after
                             if specs_after[k] != specs_before.get(k, 0)}
                record.update(event='run', label=key, round=round_index, attempt0=round_index == 0,
                              order=list(order), position=position, outer_wall_s=outer_wall,
                              new_triton_specializations=new_specs,
                              new_triton_specialization_total=sum(new_specs.values()),
                              triton_disk_entries_added=disk_after - disk_before,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              warm_status=('cold_inclusive_attempt0' if round_index == 0 else
                                           ('warm_no_new_specializations' if not new_specs else
                                            'NOT_WARM_new_specializations')))
                first = seen.get((key, row['id']))
                if first is None:
                    seen[(key, row['id'])] = record
                    destination = args.private / 'attempt0' / key / f"{hashlib.sha256(row['id'].encode()).hexdigest()[:20]}.json"
                    _atomic(destination, receipt)
                    record['private_receipt'] = str(destination)
                else:
                    record['matches_own_attempt0'] = dict(
                        tokens=record['completion_token_hash'] == first['completion_token_hash'],
                        calls=record['per_canvas_calls'] == first['per_canvas_calls'],
                        termination=record['termination'] == first['termination'])
                append(args.ledger, record)
                print(json.dumps(dict(label=key, id=row['id'], round=round_index,
                                      api_wall_s=round(record['api_wall_s'], 3),
                                      calls=record['decoder_calls'], new_specs=record['new_triton_specialization_total'],
                                      tokens=record['completion_token_hash'][:12])), flush=True)
    append(args.ledger, dict(event='done', when=time.time(), disk_cache_entries=disk_cache_entries()))
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
