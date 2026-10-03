"""Derive a named method variant from a frozen v21 effective config, inside the SAME deployment.

Only the declared method fields change: the GLOBAL threshold shift (`threshold_shift`, or none = the calibrated base
threshold), the query sensitivity (`sensitivity='cgate'`, the group member's C gate), the re-decision interval
(`decision_interval` 6 or 12), 64-row keep maps (`q_block=64`) and cross-canvas carry (`carry_canvases=K`, which
replaces `carry_first`; observe every K-th canvas). Every other field, the parent
v20 config and all source pins stay byte-identical; the fingerprint is recomputed with v21's own function and the
result must pass v21.validate_effective in this deployment. Run with cwd = the deployment, PYTHONPATH=src:.
usage: python v31_make_variant_config.py BASE_CONFIG OUT_CONFIG [--shift NAME|none] [--sensitivity cgate]
       [--interval 6|12] [--q-block 64] [--carry-canvases K]
"""
import argparse
import json
from pathlib import Path

from experiments.numerical_qk_reuse import v21


def main():
    p = argparse.ArgumentParser()
    p.add_argument('base', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--shift', default=None, help="threshold shift name, or 'none' for the base threshold")
    p.add_argument('--sensitivity', default=None, choices=('cgate',))
    p.add_argument('--interval', type=int, default=None, choices=(6, 12))
    p.add_argument('--q-block', type=int, default=None, choices=(64,))
    p.add_argument('--carry-canvases', type=int, default=None, choices=(2, 3, 4, 8))
    a = p.parse_args()
    base = json.loads(a.base.read_text())
    v21.validate_effective(base, base['condition'])
    cfg = dict(base)
    cfg.pop('fingerprint')
    if a.shift is not None:
        if a.shift == 'none':
            cfg.pop('threshold_shift', None)
        else:
            if a.shift not in v21.THRESHOLD_SHIFTS:
                raise ValueError(a.shift)
            cfg['threshold_shift'] = a.shift
    if a.sensitivity is not None:
        cfg['sensitivity'] = a.sensitivity
    if a.interval is not None:
        cfg['decision_interval'] = a.interval
    if a.q_block is not None:
        cfg['q_block'] = a.q_block
    if a.carry_canvases is not None:
        cfg.pop('carry_first', None)
        cfg['carry_canvases'] = a.carry_canvases
    changed = sorted(k for k in set(cfg) | set(base) if k != 'fingerprint' and cfg.get(k) != base.get(k))
    if not set(changed) <= {'threshold_shift', 'sensitivity', 'decision_interval', 'q_block', 'carry_canvases',
                            'carry_first'}:
        raise ValueError(f'undeclared change: {changed}')
    cfg['fingerprint'] = v21._fingerprint(cfg)
    v21.validate_effective(cfg, cfg['condition'])
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.write_text(json.dumps(cfg, indent=1, sort_keys=True))
    print(json.dumps(dict(out=a.out.name, changed=changed, threshold_shift=cfg.get('threshold_shift'),
                          sensitivity=cfg.get('sensitivity'), fingerprint=cfg['fingerprint'][:16])))


if __name__ == '__main__':
    main()
