"""One-shot three-arm v20 device-timeline ON/OFF output parity qualification.

Run this file from a new wrapper deploy with cwd/PYTHONPATH pinned to the
qualified generation deploy. Existing bridge ON receipts are read-only; each
OFF request runs once with only timing_events toggled and a new fingerprint.
The public result contains hashes and equality flags, never answers or gold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

ARMS = ('D_native', 'M3_R2_A8_current_output', 'G75L30_nativeQ128')
SCHEMA = 'v20_timeline_parity_v1'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def off_config(on: dict) -> dict:
    """The observer flag alone changes; fingerprint follows the actual config."""
    from experiments.numerical_qk_reuse.runner import _fingerprint
    if on.get('timing_events') is not True or type(on.get('fingerprint')) is not str:
        raise ValueError('bridge ON config/timeline identity absent')
    base = {key: value for key, value in on.items() if key != 'fingerprint'}
    base['timing_events'] = False
    result = dict(base, fingerprint=_fingerprint(base))
    changes = {key for key in result if result[key] != on.get(key)}
    if changes != {'timing_events', 'fingerprint'} or result['fingerprint'] == on['fingerprint']:
        raise ValueError('OFF config changed a nonobserver field')
    return result


def compare(on: dict, off: dict, arm: str) -> dict:
    """Exact science parity, with the two timing fields checked separately."""
    from experiments.numerical_qk_reuse.runner import _fingerprint
    from scripts.v20_run import router_phase_evidence
    on_span, off_span = on.get('generation_gpu_timeline_seconds'), off.get('generation_gpu_timeline_seconds')
    timeline_qualified = (type(on_span) in (int, float) and math.isfinite(on_span)
                          and on_span > 0 and off_span is None)
    timeline_reasons = []
    if type(on_span) not in (int, float) or not math.isfinite(on_span) or on_span <= 0:
        timeline_reasons.append('ON_span_unavailable')
    if off_span is not None:
        timeline_reasons.append('OFF_span_present')
    fields = dict(tokens=on.get('completion_tokens') == off.get('completion_tokens'),
                  per_canvas=on.get('per_canvas') == off.get('per_canvas'),
                  termination=on.get('termination_reason') == off.get('termination_reason'),
                  router_phase=(router_phase_evidence(arm, on.get('counters')) ==
                                router_phase_evidence(arm, off.get('counters'))))
    if (on.get('id'), on.get('seed'), on.get('max_new_tokens')) != (off.get('id'), off.get('seed'), off.get('max_new_tokens')):
        raise ValueError('ON/OFF request identity drift')
    if not all(fields.values()) or router_phase_evidence(arm, off.get('counters')) is None:
        raise ValueError(f'ON/OFF output/work parity drift for {arm}')
    return dict(arm=arm, status='pass', exact=fields, id=on['id'], seed=on['seed'],
                on_fingerprint=on['fingerprint'], off_fingerprint=off['fingerprint'],
                completion_token_hash=_fingerprint(off['completion_tokens']),
                canvas_count=len(off['per_canvas']), decoder_calls=off['total_decoder_calls'],
                timeline_qualified=timeline_qualified, timeline_reasons=timeline_reasons,
                on_device_span_s=(float(on_span) if type(on_span) in (int, float)
                                  and math.isfinite(on_span) and on_span > 0 else None),
                off_device_span=None if off_span is None else 'present_unqualified',
                on_boundary='first actual encoder forward end to final generation CUDA event; '
                            'includes host gaps and later encoder/commit work',
                off_boundary='timing events disabled; no device span',
                performance_comparison='not_applicable_cold_bridge_ON_vs_separate_OFF')


def source_hook_identity(config: dict) -> dict:
    import experiments.numerical_qk_reuse.runner as runner
    path = Path(runner.__file__).resolve()
    digest = sha(path)
    if config.get('source_hashes', {}).get(str(path)) != digest:
        raise ValueError('imported actual generation/timeline source differs from frozen binding')
    hook = runner.InitialPrefillTimeline
    if hook.__module__ != runner.__name__:
        raise ValueError('timeline hook class import drift')
    return dict(imported_runner_path=str(path), imported_runner_sha256=digest,
                timeline_hook='InitialPrefillTimeline',
                boundary='first DiffusionGemmaEncoderModel forward end to final CUDA event',
                event_instrumentation='first encoder-end and one final CUDA event; no per-forward event loop')


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + '.writing')
    with temp.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def run(args) -> int:
    if args.timeout != 900:
        raise ValueError('timeline parity has a frozen 900-second per-request watchdog')
    import fcntl
    from experiments.numerical_qk_reuse.runner import _atomic, _one
    from scripts.v20_bridge import bridge_target, read_summary
    from scripts.v20_run import validate_inputs
    from dllm.models import create_adapter
    import torch
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                   text=True, timeout=15).splitlines()[0].strip()
    protocol, binding, rows, configs = validate_inputs(args.protocol, args.binding,
                                                        args.manifests_dir, args.host, uuid,
                                                        stage='initial')
    _, _, _, historical = validate_inputs(args.protocol, args.binding,
                                           args.manifests_dir, args.host, uuid,
                                           stage='historical')
    configs['ruler4k'][ARMS[-1]] = historical['ruler4k'][ARMS[-1]]
    id_, seed = bridge_target(protocol)
    bridge = read_summary(args.bridge_summary)
    if bridge.get('id') != id_ or bridge.get('seed') != seed or bridge.get('host') != args.host:
        raise ValueError('ON bridge target/host identity drift')
    on_rows = {row['arm']: row for row in bridge['rows']}
    if len(on_rows) != len(bridge['rows']) or any(on_rows.get(arm, {}).get('status') != 'ok' for arm in ARMS):
        raise ValueError('three successful immutable ON bridge arms required')
    if args.private.exists() or args.out.exists() or args.start.exists() or args.lock.exists():
        raise FileExistsError('timeline parity already started or finished; no retry')
    args.private.mkdir(parents=True, exist_ok=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.start.parent.mkdir(parents=True, exist_ok=True)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(args.start, dict(schema=SCHEMA, status='started', host=args.host,
                                     epoch=time.time(), bridge_summary_sha256=sha(args.bridge_summary)))
        report = dict(schema=SCHEMA, status='failed', host=args.host, gpu_uuid=uuid,
                      id=id_, seed=seed, arms=list(ARMS), rows=[],
                      bridge_summary_sha256=sha(args.bridge_summary),
                      protocol_sha256=sha(args.protocol), binding_sha256=sha(args.binding),
                      script_sha256=sha(Path(__file__)))
        try:
            for arm in ARMS:
                config = configs['ruler4k'][arm]
                on_raw = args.bridge_private / f'{arm}.json'
                on = json.loads(on_raw.read_text(encoding='utf-8'))
                if (on.get('fingerprint') != config['fingerprint'] or
                        on_rows[arm].get('config_fingerprint') != config['fingerprint']):
                    raise ValueError('ON raw/config fingerprint identity drift')
                source = source_hook_identity(config)
                report.setdefault('source_hook_identity', source)
                if report['source_hook_identity'] != source:
                    raise ValueError('timeline source differs across arms')
                off = off_config(config)
                report.setdefault('off_config_fingerprints', {})[arm] = off['fingerprint']
            adapter = create_adapter('diffusion_gemma', binding['host_models'][args.host],
                                     device='cuda', precision='bfloat16',
                                     revision=protocol['model_revision']).load()
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            class Timeout(Exception):
                pass
            def alarm(*_):
                raise Timeout()
            signal.signal(signal.SIGALRM, alarm)
            for arm in ARMS:
                config = configs['ruler4k'][arm]
                off = off_config(config)
                on = json.loads((args.bridge_private / f'{arm}.json').read_text(encoding='utf-8'))
                signal.alarm(args.timeout)
                try:
                    receipt = _one(adapter, rows[id_], seed, off)
                finally:
                    signal.alarm(0)
                private_path = args.private / f'{arm}.off.json'
                _atomic(private_path, receipt)
                result = compare(on, receipt, arm)
                result.update(on_receipt_sha256=sha(args.bridge_private / f'{arm}.json'),
                              off_receipt_sha256=sha(private_path),
                              off_raw_private_path=str(private_path))
                report['rows'].append(result)
            report['output_parity_all'] = True
            report['timeline_qualified_all'] = all(row['timeline_qualified'] for row in report['rows'])
            report['status'] = 'complete'
        except BaseException as exc:
            report['error_type'] = type(exc).__name__
        finally:
            report['end_epoch'] = time.time()
            atomic_json(args.out, report)
    return 0 if report['status'] == 'complete' else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'binding', 'manifests-dir', 'bridge-private', 'bridge-summary',
                 'private', 'out', 'start', 'lock'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--timeout', type=int, default=900)
    a = p.parse_args(argv)
    raise SystemExit(run(a))


if __name__ == '__main__':
    main()
