"""Named M3+carry0 query-weight ablations on the unchanged V29 panel loop.

This is a new protocol/deployment, never a rewrite of frozen V29 runs.
Reference and variants retain FA4, DP/R6/A64, natural Q128, and torch copy/merge.
Junyu's confidence prior is not his C_gate or fresh value-aware algorithm.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
from scripts import v29_expanded_panel as base

MODES = ('temporal_T', 'unit_v30', 'confidence_v30')
SOURCES = ('scripts/v30_sensitivity_panel.py', 'scripts/v30_new_config.py',
           'experiments/numerical_qk_reuse/v30_sensitivity.py')


def check_sensitivity_receipt(receipts, n, mode):
    if mode == 'temporal_T':
        if 'v30_sensitivity' in receipts['method']:
            raise ValueError('Reference unexpectedly used a sensitivity override')
        return
    r = receipts['method'].get('v30_sensitivity', {})
    if (r.get('mode') != mode or r.get('begin') != n or r.get('observe') != n
            or r.get('accepted_mask_used') is not False or r.get('temporal_argmax_observed') is not False
            or r.get('router_and_carry_modified') is not False
            or r.get('weighting_state') != 'previous_completed_call_only'):
        raise ValueError('Sensitivity path/clock receipt mismatch')
    if not 0 < r.get('canvas_resets', 0) <= n:
        raise ValueError('Missing canvas reset coverage')
    if mode == 'unit_v30':
        if (r.get('unit_steps'), r.get('confidence_steps'), r.get('protected_first_steps')) != (n, 0, 0):
            raise ValueError('Unit path did extra weighting work')
    elif (r.get('unit_steps') != 0 or r.get('protected_first_steps') != r['canvas_resets']
          or r.get('confidence_steps', -1) + r['protected_first_steps'] != n):
        raise ValueError('Confidence path/reset coverage mismatch')


@contextmanager
def protocol(mode):
    if mode not in MODES:
        raise ValueError('Explicit sensitivity mode required')
    method = dict(base.METHOD, sensitivity=mode)
    validate = base.validate_receipt
    def checked(arm, receipts, n, required, settings, **kwargs):
        result = validate(arm, receipts, n, required, settings, **kwargs)
        if arm == 'method':
            check_sensitivity_receipt(receipts, n, mode)
        return result
    with ExitStack() as stack:
        stack.enter_context(patch.object(base, 'METHOD', method))
        stack.enter_context(patch.object(base, 'SOURCES', base.SOURCES + SOURCES))
        stack.enter_context(patch.object(base, 'validate_receipt', checked))
        yield


def build_spec(suite, mode, hosts):
    with protocol(mode):
        spec = base.build_spec(suite, 'v30_sensitivity001_'+suite+'_'+mode,
                               list(range(30001, 30009)), hosts)
    spec['v30_variant'] = dict(mode=mode, changes='query sensitivity only',
                              attribution='Junyu confidence prior; not C_gate/fresh value-aware' if mode == 'confidence_v30' else None,
                              matched_reference='temporal_T', final_claim_requires_full_panel=True)
    return spec


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sensitivity', choices=MODES, required=True)
    args, rest = p.parse_known_args(argv)
    with protocol(args.sensitivity):
        return base.main(rest)


if __name__ == '__main__':
    main()
