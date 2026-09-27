"""CPU-only legal-pair and untimed counter-twin tests."""
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from experiments.numerical_qk_reuse.v20_counter import (
    CounterTwin, PreparedSupportFloor, count_bitmap, install_counter_twin,
    install_prepared_support_floor, pair_weights)


def bitmap(batch=1, heads=1, qb=2, kt=5, *, drop_tiles=(2,)):
    skipped = [[[[tile in drop_tiles for tile in range(kt)] for _ in range(qb)]
                for _ in range(heads)] for _ in range(batch)]
    eligible = [[[[True for _ in range(kt)] for _ in range(qb)]
                 for _ in range(heads)] for _ in range(batch)]
    return skipped, eligible


class PairWeightsTests(unittest.TestCase):
    def test_q128_kv64_tails_and_prefix_boundary_are_pair_weighted(self):
        weights = pair_weights(130, 260, 130)
        self.assertEqual((len(weights), len(weights[0])), (2, 5))
        self.assertEqual(weights[0][2], dict(static_prefix=256,
                                             current_canvas=7936, whole=8192))
        self.assertEqual(weights[1][2], dict(static_prefix=4,
                                             current_canvas=124, whole=128))
        self.assertEqual(sum(x['whole'] for row in weights for x in row), 130 * 260)
        self.assertEqual(sum(x['static_prefix'] for row in weights for x in row), 130 * 130)
        self.assertEqual(sum(x['current_canvas'] for row in weights for x in row), 130 * 130)
        row = count_bitmap(*bitmap(), nq=130, nk=260, prefix=130,
                           phase='D', head_dim=256)
        self.assertEqual(row['bitmap_shape'], [1, 1, 2, 5])
        self.assertIn(dict(q_tile=0, k_tile=2, static_prefix=256,
                           current_canvas=7936, whole=8192), row['partial_tile_pair_weights'])

    def test_real_anchor_charges_full_qk_but_only_kept_pv(self):
        skipped, eligible = bitmap()
        for phase in ('A', 'D', 'H'):
            row = count_bitmap(skipped, eligible, nq=130, nk=260, prefix=130,
                               phase=phase, head_dim=256)
            whole = row['by_segment']['whole']
            self.assertEqual(whole['eligible_pairs'], 130 * 260)
            self.assertEqual(whole['skipped_pv_pairs'], 130 * 64)
            self.assertEqual(row['by_segment']['static_prefix']['skipped_pv_pairs'], 130 * 2)
            self.assertEqual(row['by_segment']['current_canvas']['skipped_pv_pairs'], 130 * 62)
            self.assertEqual(whole['skipped_qk_pairs'], 0 if phase == 'A' else 130 * 64)
            self.assertEqual(whole['executed_qk_multiply_accumulates'],
                             whole['executed_qk_pairs'] * 256)
            self.assertFalse(row['projection_skipping_measured'])

    def test_control_dense_bootstrap_has_no_skips_without_extra_full_score(self):
        skipped, eligible = bitmap(drop_tiles=())
        row = count_bitmap(skipped, eligible, nq=130, nk=260, prefix=130,
                           phase='A', head_dim=256, full_current_qk=False)
        self.assertEqual(row['by_segment']['whole']['skipped_qk_pairs'], 0)
        self.assertEqual(row['by_segment']['whole']['executed_qk_pairs'], 130 * 260)

    def test_fail_closed_geometry_bitmap_phase_and_eligibility(self):
        skipped, eligible = bitmap()
        for args in (dict(nq=0, nk=260, prefix=130), dict(nq=130, nk=260, prefix=129),
                     dict(nq=130, nk=260, prefix=131),
                     dict(nq=130, nk=260, prefix=-1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                pair_weights(**args)
        with self.assertRaises(ValueError):
            count_bitmap(skipped, eligible, nq=130, nk=260, prefix=130,
                         phase='not_a_phase', head_dim=256)
        with self.assertRaises(ValueError):
            count_bitmap(skipped, eligible, nq=130, nk=260, prefix=130,
                         phase='D', head_dim=0)
        with self.assertRaises(ValueError):
            count_bitmap(skipped, [[[[True]]]], nq=130, nk=260, prefix=130,
                         phase='D', head_dim=256)
        eligible[0][0][0][0] = False
        with self.assertRaisesRegex(ValueError, 'native-legal bitmap'):
            count_bitmap(skipped, eligible, nq=130, nk=260, prefix=130,
                         phase='D', head_dim=256)


class TwinTests(unittest.TestCase):
    def test_fresh_native_bitmap_is_collected_once_then_restored(self):
        class DeviceBitmap:
            def __init__(self, values):
                self.values = values
            def detach(self):
                return self
            def cpu(self):
                return self
            def tolist(self):
                return self.values
        skipped, eligible = bitmap(heads=2, qb=1, kt=4, drop_tiles=(1,))
        router = SimpleNamespace(support_geometry='native_legal', collect=False,
                                 calls=0, pending=['historical-entry'])
        module = SimpleNamespace(layer_idx=3, is_sliding=False)
        q = SimpleNamespace(shape=(1, 2, 128, 4))
        k = v = SimpleNamespace(shape=(1, 1, 256, 4))
        def delegate(*_a, **_kw):
            self.assertTrue(router.collect)
            router.calls += 1
            router.pending.append((3, 7, 'global', 128,
                                   DeviceBitmap(skipped), DeviceBitmap(eligible)))
            return 'fresh output'
        twin = CounterTwin(delegate, router)
        self.assertEqual(twin(module, q, k, v, None, is_causal=False), 'fresh output')
        self.assertFalse(router.collect)
        self.assertEqual(router.pending, ['historical-entry'])
        self.assertEqual(twin.rows[0]['method'], 'fresh_T')
        self.assertEqual(twin.rows[0]['phase'], 'A')
        self.assertEqual((twin.rows[0]['anchor_step'], twin.rows[0]['decision_step'],
                          twin.rows[0]['numeric_age'], twin.rows[0]['decision_age']),
                         (7, 7, 0, 0))
        self.assertEqual(twin.rows[0]['by_segment']['whole']['skipped_qk_pairs'], 0)
        self.assertEqual(twin.rows[0]['by_segment']['whole']['skipped_pv_pairs'], 2 * 128 * 64)

    def test_fresh_pending_cleanup_even_when_delegate_fails(self):
        router = SimpleNamespace(support_geometry='native_legal', collect=False,
                                 calls=0, pending=[])
        def delegate(*_a, **_kw):
            router.pending.append('unfinished')
            raise RuntimeError('kernel failed')
        twin = CounterTwin(delegate, router)
        with self.assertRaisesRegex(RuntimeError, 'kernel failed'):
            twin(None, None, None, None, None, is_causal=False)
        self.assertFalse(router.collect)
        self.assertEqual(router.pending, [])

    def test_counter_retains_actual_decision_tensors_for_optional_floor(self):
        class FakeBitmap:
            def __init__(self, values):
                self.values = values
            def detach(self):
                return self
            def cpu(self):
                return self
            def tolist(self):
                return self.values
            def clone(self):
                return self
        skipped, eligible = bitmap(heads=2, qb=1, kt=4, drop_tiles=(1,))
        decision = SimpleNamespace(skipped=FakeBitmap(skipped), eligible=FakeBitmap(eligible))
        router = SimpleNamespace(support='native_mask',
                                 output_mode='historical_route_preqk_current_output',
                                 score_calls=0, decision_calls=0, held_calls=0,
                                 sources={3: (None, None, 0, 128)}, canvas=0, step=0,
                                 cache=SimpleNamespace(entries={3: SimpleNamespace(
                                     decision=decision, score_step=0, decision_step=0)}))
        def delegate(*_a, **_kw):
            router.score_calls += 1
            router.decision_calls += 1
            return 'output'
        twin = CounterTwin(delegate, router, retain_support=True,
                           qkv_digest=lambda *_: ('q', 'k', 'v'))
        module = SimpleNamespace(layer_idx=3, is_sliding=False)
        q = SimpleNamespace(shape=(1, 2, 128, 4))
        k = v = SimpleNamespace(shape=(1, 1, 256, 4))
        self.assertEqual(twin(module, q, k, v, None, is_causal=False), 'output')
        self.assertEqual(len(twin.support_calls), 1)
        self.assertIs(twin.support_calls[0]['skipped'], decision.skipped)
        self.assertEqual(twin.support_calls[0]['qkv_digest'], ('q', 'k', 'v'))

    def test_explicit_untimed_wrapper_restores_override_and_counts_a_d_h(self):
        skipped, eligible = bitmap(heads=2)
        decision = SimpleNamespace(skipped=skipped, eligible=eligible)
        entry = SimpleNamespace(decision=decision, score_step=0, decision_step=0)
        router = SimpleNamespace(support='native_mask',
                                 output_mode='historical_route_preqk_current_output', score_calls=0,
                                 decision_calls=0, held_calls=0, canvas=0, step=0,
                                 sources={2: (object(), object(), 0, 130)},
                                 cache=SimpleNamespace(entries={2: entry}))
        phases = iter(('A', 'D', 'H'))
        def delegate(*_args, **_kwargs):
            phase = next(phases, 'held')
            if phase == 'A':
                router.score_calls += 1
                router.decision_calls += 1
                entry.score_step = router.step + 1
                entry.decision_step = router.step + 1
            elif phase == 'D':
                router.decision_calls += 1
                entry.decision_step = router.step + 1
            else:
                router.held_calls += 1
            router.step += 1
            return 'unchanged output'
        runtime = SimpleNamespace(attention_override=delegate)
        binding = SimpleNamespace(runtime=runtime)
        module = SimpleNamespace(layer_idx=2, is_sliding=True)
        q = SimpleNamespace(shape=(1, 2, 130, 4))
        k = v = SimpleNamespace(shape=(1, 1, 260, 4))
        with self.assertRaisesRegex(ValueError, 'explicit untimed'):
            with install_counter_twin(binding, router):
                pass
        with install_counter_twin(binding, router, explicit_untimed=True) as twin:
            self.assertIsInstance(runtime.attention_override, CounterTwin)
            for _ in range(3):
                self.assertEqual(runtime.attention_override(module, q, k, v, None,
                                                            is_causal=False), 'unchanged output')
            result = twin.summary()
            self.assertEqual(result['calls'], 3)
            self.assertEqual([x['phase'] for x in result['rows']], ['A', 'D', 'H'])
            self.assertEqual(result['by_phase_kind']['A/local']['whole']['skipped_qk_pairs'], 0)
            self.assertEqual(result['by_phase_kind']['D/local']['whole']['skipped_qk_pairs'], 2 * 130 * 64)
            self.assertEqual(result['by_phase_kind']['H/local']['whole']['skipped_pv_pairs'], 2 * 130 * 64)
            self.assertEqual([(r['anchor_step'], r['decision_step'],
                               r['numeric_age'], r['decision_age']) for r in result['rows']],
                             [(1, 1, 0, 0), (1, 2, 1, 0), (1, 2, 2, 1)])
            self.assertFalse(result['accepted_timing'])
        self.assertIs(runtime.attention_override, delegate)

    def test_mask_and_legacy_support_refuse_unqualified_count(self):
        router = SimpleNamespace(support='legacy_junyu_mask')
        twin = CounterTwin(lambda *_a, **_kw: None, router)
        module = SimpleNamespace(layer_idx=2, is_sliding=False)
        q = SimpleNamespace(shape=(1, 1, 128, 64))
        k = v = SimpleNamespace(shape=(1, 1, 256, 64))
        with self.assertRaisesRegex(ValueError, 'mask=None'):
            twin(module, q, k, v, object())
        with self.assertRaisesRegex(ValueError, 'is_causal=False'):
            twin(module, q, k, v, None)
        with self.assertRaisesRegex(ValueError, 'native-legal'):
            twin(module, q, k, v, None, is_causal=False)

    def test_routing_only_current_output_refuses_preqk_skip_claim(self):
        router = SimpleNamespace(support='native_mask', output_mode='routing_only_current_output')
        twin = CounterTwin(lambda *_a, **_kw: None, router)
        module = SimpleNamespace(layer_idx=2, is_sliding=False)
        q = SimpleNamespace(shape=(1, 1, 128, 64))
        k = v = SimpleNamespace(shape=(1, 1, 256, 64))
        with self.assertRaisesRegex(ValueError, 'pre-QK'):
            twin(module, q, k, v, None, is_causal=False)

    def test_control_bootstrap_observation_held_distinguish_full_qk(self):
        skipped, eligible = bitmap(heads=2, qb=1, kt=4, drop_tiles=(1,))
        class DeviceBitmap:
            def __init__(self, values):
                self.values = values
            def detach(self):
                return self
            def cpu(self):
                return self
            def tolist(self):
                return self.values
        physical_skipped, physical_eligible = DeviceBitmap(skipped), DeviceBitmap(eligible)
        owner = SimpleNamespace(support='native_mask',
                                output_mode='historical_route_preqk_current_output',
                                sources={3: (object(), object(), 0, 128)}, canvas=0,
                                epoch=0, step=0)
        control = SimpleNamespace(owner=owner, bootstrap_calls=0, observation_calls=0,
                                  held_calls=0, maps={3: (object(), physical_skipped, physical_eligible)})
        phases = iter(('bootstrap', 'observe', 'held'))
        def delegate(*_args, **_kwargs):
            phase = next(phases, 'held')
            if phase == 'bootstrap':
                control.bootstrap_calls += 1
            elif phase == 'observe':
                control.observation_calls += 1
            else:
                control.held_calls += 1
            return 'output'
        probe_calls = []
        probe = SimpleNamespace(observe=lambda *_a, **kw: probe_calls.append(kw['phase']))
        twin = CounterTwin(delegate, control, operator_probe=probe)
        module = SimpleNamespace(layer_idx=3, is_sliding=False)
        q = SimpleNamespace(shape=(1, 2, 128, 4))
        k = v = SimpleNamespace(shape=(1, 1, 256, 4))
        for step in range(3):
            owner.step = step
            self.assertEqual(twin(module, q, k, v, None, is_causal=False), 'output')
        bootstrap, observe, held = twin.rows
        self.assertEqual([r['phase'] for r in twin.rows], ['A', 'A', 'H'])
        self.assertEqual(bootstrap['by_segment']['whole']['skipped_qk_pairs'], 0)
        self.assertFalse(bootstrap['full_current_qk'])
        self.assertEqual(observe['by_segment']['whole']['skipped_qk_pairs'], 0)
        self.assertTrue(observe['full_current_qk'])
        self.assertEqual(held['by_segment']['whole']['skipped_qk_pairs'], 2 * 128 * 64)
        self.assertEqual([(r['anchor_step'], r['decision_step'],
                           r['numeric_age'], r['decision_age']) for r in twin.rows],
                         [(None, None, None, None), (1, 1, 0, 0), (1, 1, 1, 1)])
        self.assertEqual(probe_calls, ['A', 'H'])
        owner.epoch = 1
        owner.step = 3
        with self.assertRaisesRegex(ValueError, 'same-canvas observed bitmap age'):
            twin(module, q, k, v, None, is_causal=False)

    def test_prepared_floor_consumes_each_frozen_map_only_at_matching_call(self):
        output = SimpleNamespace(transpose=lambda *_: SimpleNamespace(contiguous=lambda: 'same-consumer-output'))
        calls = []
        owner = SimpleNamespace(canvas=0, step=1, sources={3: (None, None, 0, 128)},
                                _mask_present=True)
        def consume(q, k, v, skipped, eligible, scale, window, causal):
            calls.append((skipped, eligible, scale, window, causal))
            return SimpleNamespace(output=output, invalid_scores=False)
        owner._consume = consume
        item = dict(layer=3, canvas=0, decoder_call=1, q_shape=(1, 2, 128, 4),
                    k_shape=(1, 1, 256, 4), prefix=128,
                    skipped='frozen-skip', eligible='frozen-eligible', qkv_digest=('q', 'k', 'v'))
        runtime = SimpleNamespace(attention_override=lambda *_a, **_kw: 'original')
        binding = SimpleNamespace(runtime=runtime)
        module = SimpleNamespace(layer_idx=3, training=False)
        q = SimpleNamespace(shape=item['q_shape'])
        k = v = SimpleNamespace(shape=item['k_shape'])
        fake_kernel = SimpleNamespace(fused_guard=lambda *_a: None)
        with patch.dict(sys.modules, {'experiments.numerical_qk_reuse.cached_executor': fake_kernel}):
            with install_prepared_support_floor(binding, owner, [item], explicit_untimed=True,
                                                qkv_digest=lambda *_: ('q', 'k', 'v')) as floor:
                self.assertIsInstance(runtime.attention_override, PreparedSupportFloor)
                self.assertEqual(floor(module, q, k, v, None, is_causal=False),
                                 ('same-consumer-output', None))
                floor.assert_complete()
                self.assertEqual(floor.qkv_matches, [True])
                self.assertEqual(calls, [('frozen-skip', 'frozen-eligible', .5, None, False)])
                self.assertFalse(owner._mask_present)
                with self.assertRaisesRegex(ValueError, 'exceeded'):
                    floor(module, q, k, v, None, is_causal=False)
                floor.reset()
                floor.qkv_digest = lambda *_: ('different',)
                self.assertEqual(floor(module, q, k, v, None, is_causal=False),
                                 ('same-consumer-output', None))
                self.assertEqual(floor.qkv_matches, [False])
                floor.reset()
                owner.step = 2
                with self.assertRaisesRegex(ValueError, 'geometry drift'):
                    floor(module, q, k, v, None, is_causal=False)
        self.assertIsNotNone(runtime.attention_override)


if __name__ == '__main__':
    unittest.main()
