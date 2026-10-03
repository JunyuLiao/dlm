"""Uniform-threshold wiring for causal query-sensitivity methods.

The archived guardrail runner intentionally has separate early and late
threshold pairs.  This module provides the small state adapter needed by a
future runner that calibrates one local/global pair for every denoising call.
It keeps threshold selection independent of the call index while leaving the
existing router, kernel, and sampler unchanged.
"""
from numbers import Real

from .query_adaptive import State, observe


def uniform_policy(local, global_):
    """Return the router policy shape accepted by ``install``.

    ``local`` and ``global_`` are log thresholds.  Infinite values are useful
    for dense controls, so validation only rejects non-numeric values.
    """
    if not isinstance(local, Real) or not isinstance(global_, Real):
        raise TypeError('local and global thresholds must be real numbers')
    return {
        'local': {'log_threshold': float(local)},
        'global': {'log_threshold': float(global_)},
    }


class UniformThresholdState(State):
    """``State`` with one local/global threshold pair for all calls.

    ``State`` computes the selected causal method from completed previous
    sampler/logit outcomes. This subclass changes only the router's threshold
    selector; it does not suppress the first calls or inspect current-step
    logits before routing. The same pair is returned for call 1, call 2, and
    all later calls.
    """

    def __init__(self, *args, thresholds, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(thresholds, dict):
            raise TypeError('thresholds must be a local/global mapping')
        try:
            self.thresholds = {
                kind: dict(thresholds[kind]) for kind in ('local', 'global')
            }
            for kind in ('local', 'global'):
                self.thresholds[kind]['log_threshold'] = float(
                    self.thresholds[kind]['log_threshold'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('thresholds must contain local/global log_threshold values') from error

    def begin(self, cur_step, canvas):
        super().begin(cur_step, canvas)
        if self.router is None:
            return
        # The callback signature is part of integration.Attention.  Ignore
        # iteration deliberately: both thresholds are shared by every call.
        self.router.policy_selector = lambda _iteration, kind: self.thresholds[kind]
        self.current['threshold_phase'] = 'uniform'


__all__ = ['UniformThresholdState', 'uniform_policy', 'observe']
