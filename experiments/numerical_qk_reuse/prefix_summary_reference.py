"""CPU mathematical illustration for v8; NOT a GPU implementation/qualification.

Stores per-row (block log mass, weighted projected value) for full immutable
prefix blocks. Current sensitivity, RMS normalization, scan and support remain
live. Floating-point choices here do not claim bit parity with Triton.
Run: python prefix_summary_reference.py
"""
from dataclasses import dataclass
import unittest
import numpy as np


@dataclass
class Stats:
    log_mass: np.ndarray
    mu: np.ndarray
    active: np.ndarray


@dataclass
class PrefixCache:
    epoch: tuple
    prefix: int
    tile: int
    values: dict


def block_stats(scores, projected):
    """Row-masked normalized weighted mean; structural -inf is legal masking."""
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError('Malformed score: do not silently drop its block')
    if not np.isfinite(projected).all():
        raise ValueError('Malformed projected V')
    finite = np.isfinite(scores)
    active = finite.any(axis=1)
    maximum = scores.max(axis=1)
    safe_max = np.where(active, maximum, 0.)
    exp_scores = np.where(finite, np.exp(scores - safe_max[:, None]), 0.)
    denom = exp_scores.sum(axis=1)
    weights = exp_scores / np.maximum(denom, 1.e-30)[:, None]
    log_mass = np.where(active, maximum + np.log(np.maximum(denom, 1.e-30)), -np.inf)
    return Stats(log_mass, weights @ projected, active)


def build_prefix_cache(scores, projected, prefix, tile, epoch):
    if not 0 <= prefix <= scores.shape[1] or tile <= 0:
        raise ValueError('Invalid prefix/tile')
    values = {}
    for start in range(0, scores.shape[1], tile):
        end = min(scores.shape[1], start + tile)
        # Mixed prefix/canvas and incomplete edge tiles retain the old path.
        if start + tile <= prefix:
            values[start] = block_stats(scores[:, start:end], projected[start:end])
    return PrefixCache(epoch, prefix, tile, values)


def route(scores, projected, sensitivity, reference, log_threshold,
          *, prefix, tile=64, epoch=(0, 0, 0), cache=None):
    scores = np.asarray(scores, dtype=np.float64)
    projected = np.asarray(projected, dtype=np.float64)
    sensitivity = np.asarray(sensitivity, dtype=np.float64)
    q, k = scores.shape
    if projected.shape[0] != k or sensitivity.shape != (q,):
        raise ValueError('Shape mismatch')
    if not np.isfinite(sensitivity).all() or np.any(sensitivity <= 0):
        raise ValueError('Sensitivity must be finite positive')
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError('Reference must be finite positive')
    if np.isnan(log_threshold):
        raise ValueError('Threshold must not be NaN')
    if cache is not None and (cache.epoch != epoch or cache.prefix != prefix or cache.tile != tile):
        raise ValueError('Invalid cache identity')
    previous = np.full(q, -np.inf)
    output = np.zeros((q, projected.shape[1]))
    skipped = []
    recomputed = 0
    for start in range(0, k, tile):
        end = min(k, start + tile)
        if cache is not None and start in cache.values:
            s = cache.values[start]
        else:
            s = block_stats(scores[:, start:end], projected[start:end])
            recomputed += 1
        combined = np.logaddexp(previous, s.log_mass)
        safe = np.where(np.isfinite(combined), combined, 0.)
        alpha = np.where(s.active, np.exp(s.log_mass - safe), 0.)
        delta = alpha[:, None] * (s.mu - output)
        with np.errstate(divide='ignore'):
            risk = np.log(np.linalg.norm(delta, axis=1) / reference) + np.log(sensitivity)
        # First legal support must remain, including a newly active row.
        risk = np.where(s.active, np.where(np.isfinite(previous), risk, np.inf), -np.inf)
        eligible = bool(s.active.any())
        drop = eligible and bool(risk.max() < log_threshold)
        skipped.append(drop)
        if eligible and not drop:
            old = np.where(np.isfinite(previous), np.exp(previous - safe), 0.)
            output = old[:, None] * output + alpha[:, None] * s.mu
            previous = combined
    return np.asarray(skipped), previous, output, recomputed


class TestExactPrefixSummary(unittest.TestCase):
    def test_changing_canvas_sensitivity_and_rms(self):
        rng = np.random.default_rng(173)
        for prefix in (0, 31, 64, 73, 128):
            q, k, r = 11, prefix + 97, 8
            scores = rng.normal(size=(q, k))
            legal = rng.random((q, k)) > .15
            legal[0] = False  # row with no keys
            scores[~legal] = -np.inf
            z = rng.normal(size=(k, r))
            epoch = ('request-a', 2, 8, 'fixed-projection')
            cache = build_prefix_cache(scores, z, prefix, 64, epoch)
            for _ in range(5):
                live = z.copy()
                live[prefix:] = rng.normal(size=live[prefix:].shape)
                t = 1. + rng.random(q) * 3.
                ref = .5 + rng.random()
                threshold = float(rng.uniform(-3, 1))
                base = route(scores, live, t, ref, threshold,
                             prefix=prefix, epoch=epoch)
                opt = route(scores, live, t, ref, threshold,
                            prefix=prefix, epoch=epoch, cache=cache)
                for a, b in zip(base[:3], opt[:3]):
                    np.testing.assert_array_equal(a, b)
                self.assertEqual(base[3] - opt[3], len(cache.values))

    def test_decisions_are_not_cached(self):
        scores = np.zeros((2, 128))
        z = np.zeros((128, 1)); z[64:] = 2.
        cache = build_prefix_cache(scores, z, 128, 64, (0, 0, 0))
        low = route(scores, z, np.ones(2), 1., .01, prefix=128, cache=cache)[0]
        high = route(scores, z, np.array([1., 4.]), 1., .01, prefix=128, cache=cache)[0]
        self.assertEqual(low.tolist(), [False, True])
        self.assertEqual(high.tolist(), [False, False])
        # Same summary, changed RMS -> different decision: ref must stay live.
        rms = route(scores, z, np.ones(2), .5, .01, prefix=128, cache=cache)[0]
        self.assertEqual(rms.tolist(), [False, False])

    def test_strict_tie_and_zero_mass_rows(self):
        scores = np.full((2, 128), -np.inf)
        scores[1, 0] = scores[1, 64] = 0.  # exact representable unit-mass tie
        z = np.zeros((128, 1)); z[64:] = 2.
        cache = build_prefix_cache(scores, z, 128, 64, (0, 0, 0))
        result = route(scores, z, np.ones(2), 1., 0., prefix=128, cache=cache)
        self.assertEqual(result[0].tolist(), [False, False])
        self.assertTrue(np.isneginf(result[1][0]))
        self.assertEqual(result[2][0, 0], 0.)

    def test_epoch_and_malformed_score_guards(self):
        scores = np.zeros((2, 128)); z = np.ones((128, 2))
        cache = build_prefix_cache(scores, z, 64, 64, (1, 2, 3))
        with self.assertRaises(ValueError):
            route(scores, z, np.ones(2), 1., 0., prefix=64, cache=cache,
                  epoch=(1, 2, 4))
        scores[0, 4] = np.nan
        with self.assertRaises(ValueError):
            build_prefix_cache(scores, z, 64, 64, (1, 2, 4))

    def test_mutable_prefix_requires_invalidation(self):
        scores = np.zeros((1, 128)); z = np.zeros((128, 1)); z[64:] = 2.
        old = build_prefix_cache(scores, z, 128, 64, (0, 0, 0))
        changed = z.copy(); changed[64:] = 20.
        stale = route(scores, changed, np.ones(1), 1., 1., prefix=128, cache=old)[0]
        fresh = route(scores, changed, np.ones(1), 1., 1., prefix=128)[0]
        self.assertNotEqual(stale.tolist(), fresh.tolist())
        new = build_prefix_cache(scores, changed, 128, 64, (0, 0, 1))
        updated = route(scores, changed, np.ones(1), 1., 1., prefix=128,
                        cache=new, epoch=(0, 0, 1))[0]
        np.testing.assert_array_equal(updated, fresh)


if __name__ == '__main__':
    unittest.main(verbosity=2)
