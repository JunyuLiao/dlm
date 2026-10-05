"""B8 risk-export qualification and frozen 16-execution v15 subset planner.

The GPU check compares the private v5 kernel's export-on/off outputs and
support maps. Complete-request parity is checked from the v18 run ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def plan(v15_protocol: Path, selection: Path, manifest: Path, out: Path) -> dict:
    from scripts.v10_request_runs import arm_config
    from scripts.v13_seed_runs import arm_config_hash, plan_schedule

    parent = json.loads(v15_protocol.read_text())
    chosen = json.loads(selection.read_text())['panel'][:4]
    ids = [r['id'] for r in chosen]
    if len(ids) != 4 or ids != parent['ids'][:4]:
        raise ValueError('first four v15 selection-order IDs must match the frozen parent')
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != parent['manifest_sha256']:
        raise ValueError('v15 generation manifest hash mismatch')
    rows = {r['id']: r for r in json.loads(raw)}
    if any(i not in rows for i in ids):
        raise ValueError('v15 manifest lacks a selected ID')
    old = dict(parent['arms']['B8_P'])
    new = dict(old, name='B8_no_risk_export', plugin='experiments.value_direction_hopper.cvm:install_no_risk_export')
    arms = {'B8_P': old, 'B8_no_risk_export': new}
    protocol_id = 'v18_b8_no_risk_export_v1'
    ns = SimpleNamespace(phase=protocol_id, ids=ids, seeds=[17], manifest=manifest,
                         policy=Path(parent['policy_file']), model=Path(parent['model_path']),
                         revision=parent['model_revision'])
    configs = {name: arm_config(ns, arm) for name, arm in arms.items()}
    hashes = {name: arm_config_hash(config) for name, config in configs.items()}
    schedule = plan_schedule(protocol_id, parent['model_revision'], hashes, ids, [17], 20260926)
    assert len(schedule) == 16
    protocol = dict(schema='v18_b8_no_risk_export_v1', protocol_id=protocol_id,
                    source_v15_protocol=str(v15_protocol), source_v15_selection=str(selection),
                    ids=ids, seeds=[17], arms=arms, arm_hashes=hashes, schedule=schedule,
                    schedule_seed=20260926, planned_executions=16,
                    manifest_sha256=hashlib.sha256(raw).hexdigest(), model_path=parent['model_path'],
                    model_revision=parent['model_revision'], policy_file=parent['policy_file'],
                    comparison='same (question, seed) B8_P vs B8_no_risk_export first and warm; '
                               'require identical tokens, per-canvas calls, stop; warm requires no new compilation/load')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(protocol, indent=2, sort_keys=True) + '\n')
    return protocol


def qualify(v5_build: Path) -> None:
    import torch
    from experiments.value_direction_hopper.cvm import V5Kernel, load_v5
    from experiments.value_direction_hopper.masks import geometry

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    load_v5(v5_build)
    for prefix, threshold in ((129, -1.), (1645, -3.1366905212402343), (3000, 0.)):
        g = torch.Generator(device='cuda').manual_seed(prefix)
        nq, nk, h, hk, d = 256, prefix + 256, 16, 2, 512
        q = (torch.randn(1, h, nq, d, generator=g, device='cuda') * .3).to(torch.bfloat16).contiguous()
        k = (torch.randn(1, hk, nk, d, generator=g, device='cuda') * .3).to(torch.bfloat16).contiguous()
        v = torch.randn(1, hk, nk, d, generator=g, device='cuda').to(torch.bfloat16).contiguous()
        z = torch.randn(1, hk, nk, 32, generator=g, device='cuda').contiguous()
        ref = (torch.rand(1, hk, generator=g, device='cuda') + .5).contiguous()
        sens = (1 + 3 * torch.rand(1, nq, generator=g, device='cuda')).contiguous()
        mask, _ = geometry(1, nq, nk, device='cuda')
        outputs = []
        for export in (True, False):
            kernel = V5Kernel()
            kernel.export, kernel.protect_tile = export, prefix // 64
            result = kernel(q, k, v, z, ref, mask=mask, scale=d ** -.5, log_threshold=threshold,
                            mode='value', precision='tf32x3_register', tma=True, sensitivity=sens)
            outputs.append((result, kernel.last_export))
        torch.cuda.synchronize()
        on, off = outputs
        if on[1] is None or off[1] is not None:
            raise AssertionError('risk export state mismatch')
        for field in ('output', 'skipped', 'eligible', 'log_normalizer'):
            a, b = getattr(on[0], field), getattr(off[0], field)
            if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
                raise AssertionError(f'{field} changed with export flag at prefix={prefix}')
        print(json.dumps(dict(prefix=prefix, threshold=threshold, output_equal=True,
                              skipped_equal=True, eligible_equal=True)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    build = sub.add_parser('plan')
    for name in ('v15-protocol', 'selection', 'manifest', 'out'):
        build.add_argument('--' + name, type=Path, required=True)
    gpu = sub.add_parser('gpu')
    gpu.add_argument('--v5-build', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'plan':
        print(json.dumps(dict(ids=plan(args.v15_protocol, args.selection, args.manifest, args.out)['ids'], executions=16)))
    else:
        qualify(args.v5_build)


if __name__ == '__main__':
    main()
