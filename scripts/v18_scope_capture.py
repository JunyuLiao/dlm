"""Capture 12 predetermined real T50 native-legal attention states, without gold.

Two complete requests (first RULER calibration row, first v15 LongBench selection
row), seed 101; decoder layers 0/5/29 and denoising iterations 1/3. The
attention wrapper leaves outputs unchanged. Missing iteration-3 states remain
missing and are reported. One padded-mask derivative of real Q/K/V is appended
for mask qualification and explicitly marked derived.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

LAYERS = (0, 5, 29)
STEPS = (1, 3)
SEED = 101


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def capture(config_path: Path, calibration_manifest: Path, calibration_id: str,
            longbench_manifest: Path, longbench_id: str, out: Path) -> dict:
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse.runner import GOLD_FIELDS, _fingerprint, _one, _rows
    from experiments.value_direction_hopper.integration import Attention
    from experiments.value_direction_hopper.masks import PackedMask

    config = json.loads(config_path.read_text())
    if config.get('condition') != 'native_legal_all_layers' or config.get('frontier_arm') != 'T50':
        raise ValueError('capture requires the frozen native-legal T50 config')
    sources = ((calibration_manifest, calibration_id, 'ruler_calibration_first'),
               (longbench_manifest, longbench_id, 'v15_selection_first'))
    rows = []
    for manifest, item_id, label in sources:
        raw = _rows(manifest, allow_task_budgets=True, allow_thinking_off=True)
        if any(any(k in r for k in GOLD_FIELDS) for r in raw):
            raise ValueError('generation manifest must be gold-free')
        row = next((r for r in raw if r['id'] == item_id), None)
        if row is None or raw[0]['id'] != item_id:
            raise ValueError(f'{label} must be the first row of its frozen manifest')
        expected_thinking = label != 'ruler_calibration_first'
        if row.get('thinking') is not expected_thinking:
            raise ValueError(f'{label} has unexpected thinking mode')
        rows.append((row, label, manifest))
    adapter = create_adapter('diffusion_gemma', config['model'], device='cuda', precision='bfloat16',
                             revision=config['revision']).load()
    original = Attention.__call__
    states, request_reports = [], []
    current_label = None
    canvas_index = -1
    seen = set()

    def tapped(self, module, q, k, v, mask, **kwargs):
        nonlocal canvas_index
        layer = int(module.layer_idx)
        native_iteration = int(module._blasst_2d_runtime.current_denoising_iteration)
        if layer == 0 and native_iteration == 0:
            canvas_index += 1
        step = native_iteration + 1  # native runtime is zero-based; State.iteration is one-based
        key = (current_label, layer, step)
        selected = canvas_index == 0 and layer in LAYERS and step in STEPS and key not in seen
        kernel_record = {}
        kernel = self.kernel

        class TransparentKernel:
            def __call__(self, *args, **kw):
                result = kernel(*args, **kw)
                kernel_mask = kw.get('mask')
                kernel_record.update(q=args[0].detach().cpu().clone(), k=args[1].detach().cpu().clone(),
                                     v=args[2].detach().cpu().clone(), z=args[3].detach().cpu().clone(),
                                     reference=args[4].detach().cpu().clone(),
                                     scale=kw.get('scale'), log_threshold=kw.get('log_threshold'),
                                     mode=kw.get('mode'), precision=kw.get('precision'), tma=kw.get('tma'),
                                     kernel_mask_kind=('packed' if isinstance(kernel_mask, PackedMask)
                                                       else 'none' if kernel_mask is None else 'tensor'),
                                     kernel_mask=None if kernel_mask is None else
                                     (kernel_mask.tensor if isinstance(kernel_mask, PackedMask) else kernel_mask).detach().cpu().clone(),
                                     kernel_mask_keys=kernel_mask.keys if isinstance(kernel_mask, PackedMask) else None,
                                     sensitivity=None if kw.get('sensitivity') is None else kw['sensitivity'].detach().cpu().clone(),
                                     skipped=result.skipped.detach().cpu().clone(),
                                     eligible=result.eligible.detach().cpu().clone(),
                                     kernel_output=result.output.detach().cpu().clone())
                return result

        if selected:
            self.kernel = TransparentKernel()
        try:
            result = original(self, module, q, k, v, mask, **kwargs)
        finally:
            if selected:
                self.kernel = kernel
        if selected:
            if not kernel_record:
                raise RuntimeError('selected state did not invoke the fresh kernel')
            seen.add(key)
            states.append(dict(q=kernel_record['q'], k=kernel_record['k'], v=kernel_record['v'],
                               mask=None if mask is None else mask.detach().cpu().clone(),
                               scaling=kwargs.get('scaling'), is_causal=kwargs.get('is_causal'),
                               sliding_window=kwargs.get('sliding_window'),
                               sensitivity=kernel_record['sensitivity'],
                               z=kernel_record['z'], reference=kernel_record['reference'],
                               kernel_scale=kernel_record['scale'], kernel_log_threshold=kernel_record['log_threshold'],
                               kernel_mode=kernel_record['mode'], kernel_precision=kernel_record['precision'],
                               kernel_tma=kernel_record['tma'],
                               kernel_mask_kind=kernel_record['kernel_mask_kind'],
                               kernel_mask=kernel_record['kernel_mask'], kernel_mask_keys=kernel_record['kernel_mask_keys'],
                               skipped=kernel_record['skipped'], eligible=kernel_record['eligible'],
                               kernel_output=kernel_record['kernel_output'],
                               layer=layer, step=step, canvas_index=canvas_index,
                               native_iteration=native_iteration,
                               layer_kind='local' if module.is_sliding else 'global',
                               source_id=f'{current_label}/layer{layer}/iteration{step}', derived_mask=False))
        return result

    Attention.__call__ = tapped
    try:
        for row, label, manifest in rows:
            current_label = label
            canvas_index = -1
            request_config = dict(config, thinking=bool(row['thinking']),
                                  manifest=str(manifest.resolve()), manifest_sha256=file_sha(manifest),
                                  ids=[row['id']], seeds=[SEED], phase='v18_scope_capture')
            request_config.pop('fingerprint', None)
            request_config['fingerprint'] = _fingerprint(request_config)
            try:
                receipt = _one(adapter, row, SEED, request_config)
                request_reports.append(dict(label=label, id=row['id'], seed=SEED, status='validated',
                                            thinking=request_config['thinking'], fingerprint=request_config['fingerprint'],
                                            termination=receipt['termination_reason'],
                                            decoder_calls=receipt['total_decoder_calls'],
                                            output_tokens=receipt['output_tokens']))
            except Exception as exc:
                request_reports.append(dict(label=label, id=row['id'], seed=SEED, status='validation_failed',
                                            thinking=request_config['thinking'], fingerprint=request_config['fingerprint'],
                                            error=f'{type(exc).__name__}: {exc}'[:500]))
                out.parent.mkdir(parents=True, exist_ok=True)
                checkpoint = out.with_suffix('.unqualified.pt')
                torch.save(states, checkpoint)
                checkpoint.with_suffix('.json').write_text(json.dumps(dict(
                    schema='v18_scope_capture_unqualified_v1', config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    state_file=str(checkpoint), state_file_sha256=file_sha(checkpoint),
                    requests=request_reports, captured_real_states=len(states),
                    valid_for_opportunity=False,
                    note='A complete generation failed post-generate validation; these states are diagnostic only.'), indent=2) + '\n')
                raise
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                checkpoint = out.with_suffix('.checkpoint.pt')
                torch.save(states, checkpoint)
                checkpoint.with_suffix('.json').write_text(json.dumps(dict(
                    schema='v18_scope_capture_checkpoint_v1', config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    state_file=str(checkpoint), state_file_sha256=file_sha(checkpoint),
                    requests=request_reports, captured_real_states=len(states),
                    valid_for_opportunity=False,
                    note='Intermediate checkpoint; only final validated capture file is qualified.'), indent=2) + '\n')
    finally:
        Attention.__call__ = original
    original_states = len(states)
    if states:
        source = next((s for s in states if s['k'].shape[-2] % 64), states[0])
        nk, nq = source['k'].shape[-2], source['q'].shape[-2]
        padded = torch.ones(1, 1, nq, nk, dtype=torch.bool)
        padded[..., -min(5, nk - 1):] = False
        states.append(dict(source, mask=padded, source_id=source['source_id'] + '/derived_padded_mask',
                           derived_mask=True))
    expected = {(label, layer, step) for _, label, _ in rows for layer in LAYERS for step in STEPS}
    missing = sorted(expected - seen)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(states, out)
    report = dict(schema='v18_scope_capture_v1', config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  state_file=str(out), state_file_sha256=file_sha(out),
                  requests=request_reports, expected_real_states=12, captured_real_states=original_states,
                  missing_real_states=[dict(source=label, layer=layer, step=step) for label, layer, step in missing],
                  derived_padded_states=len(states) - original_states,
                  note='Padded mask is a controlled derivative of captured Q/K/V; absent step-3 states are not synthesized.')
    out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--calibration-manifest', type=Path, required=True)
    p.add_argument('--calibration-id', required=True)
    p.add_argument('--longbench-manifest', type=Path, required=True)
    p.add_argument('--longbench-id', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = capture(args.config, args.calibration_manifest, args.calibration_id,
                     args.longbench_manifest, args.longbench_id, args.out)
    print(json.dumps(dict(captured_real_states=result['captured_real_states'],
                          missing_real_states=result['missing_real_states'])))


if __name__ == '__main__':
    main()
