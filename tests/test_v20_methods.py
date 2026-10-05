"""CPU-only v20 clock, effective-config and installer contract tests."""
import ast
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from experiments.numerical_qk_reuse.cache import Identity, ScoreCache
from experiments.numerical_qk_reuse import v20


def identity():
    return Identity(0, 0, 1, 5, 1, 16, 2, 256, 128, 512, 100, 0, 128,
                    1.0, 'bfloat16', 'cuda:0', (False, 0), 123)


def timeline(period, interval, length=16, *, force_at=()):
    cache, key, phases = ScoreCache(period, interval), identity(), []
    for step in range(length):
        plan = cache.plan(key, step, force_refresh=step in force_at)
        phase = 'A' if plan.score_refresh else 'D' if plan.decision_refresh else 'H'
        phases.append(phase)
        if plan.score_refresh:
            cache.publish_scores(key, step, object())
        if plan.decision_refresh:
            cache.publish_decision(key, step, object())
    return phases


class ClockTests(unittest.TestCase):
    def test_anchor_eight_resets_r3_next_decision_to_eleven(self):
        p = timeline(8, 3, 13)
        self.assertEqual(p, list('AHHDHHDHAHHDH'))
        self.assertEqual([i for i, x in enumerate(p) if x == 'D'], [3, 6, 11])

    def test_r1_equals_m1_and_r2_holds_alternating_decisions(self):
        self.assertEqual(timeline(8, 1, 12), list('ADDDDDDDADDD'))
        self.assertEqual(timeline(8, 2, 12), list('AHDHDHDHAHDH'))

    def test_r_at_least_a_equals_matched_anchor_held(self):
        anchor_held = timeline(8, 8)
        self.assertEqual(anchor_held, list('AHHHHHHHAHHHHHHH'))
        for interval in (9, 16, 100):
            self.assertEqual(timeline(8, interval), anchor_held)

    def test_a1_observes_real_score_every_call(self):
        self.assertEqual(timeline(1, 3, 12), ['A'] * 12)

    def test_mask_forces_anchor_and_identity_change_resets(self):
        self.assertEqual(timeline(8, 3, 7, force_at=(4,)), list('AHHD AHH'.replace(' ', '')))
        cache, key = ScoreCache(8, 3), identity()
        cache.publish_scores(key, 0, object())
        cache.publish_decision(key, 0, object())
        self.assertTrue(cache.plan(replace(key, canvas=1), 1).score_refresh)


class ConfigTests(unittest.TestCase):
    def base(self):
        return dict(diagnostic=False, policy={'local': {'log_threshold': -2.},
                                              'global': {'log_threshold': -3.}},
                    consumer='triton', thinking=True, max_new_tokens=8192,
                    m_ref=1., beta=3., gamma=.5, source_hashes={})

    def test_every_arm_scope_has_explicit_effective_contract(self):
        for scope in v20.SCOPES:
            for arm, interval in v20.ARM_INTERVALS.items():
                with self.subTest(scope=scope, arm=arm):
                    config = v20.effective_config(self.base(), arm, scope)
                    self.assertEqual(v20.validate_effective(config, config['condition']), (scope, interval))
                    self.assertEqual(config['score_refresh_period'], 8)
                    self.assertEqual(config['output_mode'], v20.PREQK_MODE)
                    self.assertEqual(config['support'], 'native_mask')
                    self.assertIs(config['fast_t'], True)
                    self.assertEqual(config['selector'], 'legacy_recompute')
                    self.assertEqual(config['plugin'], v20.PLUGIN)
                    self.assertEqual(config['max_cache_bytes'], 4 * 1024**3)
                    self.assertEqual(config['max_summary_bytes'], 1024**3)

    def test_published_global_m3_remains_r2(self):
        source = Path(v20.__file__).with_name('global_scope.py')
        tree = ast.parse(source.read_text())
        table = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign) and any(
                         isinstance(t, ast.Name) and t.id == 'DECISION_INTERVAL' for t in node.targets))
        self.assertEqual(table['global_M3'], 2)
        self.assertEqual(table['v20_global_M3_R3'], 3)

    def test_exact_prefix_summary_is_explicit_for_redecision_arms_only(self):
        for arm in ('M1_R1_A8_current_output', 'M3_R2_A8_current_output',
                    'M3_R3_A8_current_output'):
            config = v20.effective_config(dict(self.base(), selector='prefix_block_summary',
                                               selector_layers='all'), arm, v20.ALL_NATIVE_LEGAL)
            self.assertEqual(config['selector'], 'prefix_block_summary')
            self.assertEqual(v20.validate_effective(config, config['condition']),
                             (v20.ALL_NATIVE_LEGAL, v20.ARM_INTERVALS[arm]))
        with self.assertRaisesRegex(ValueError, 'anchor-held B'):
            v20.effective_config(dict(self.base(), selector='prefix_block_summary'),
                                 'B_A8_matched', v20.ALL_NATIVE_LEGAL)

    def test_fail_closed_mode_clock_fast_t_fingerprint_and_source(self):
        config = v20.effective_config(self.base(), 'M3_R3_A8_current_output', v20.ALL_NATIVE_LEGAL)
        for change in (dict(output_mode='cached_scores'), dict(decision_interval=2),
                       dict(fast_t=False), dict(support='legacy_junyu_mask'),
                       dict(score_refresh_period=4), dict(diagnostic=True),
                       dict(max_cache_bytes=0), dict(max_summary_bytes=-1)):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    v20.validate_effective({**config, **change}, config['condition'])
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            v20.validate_effective({**config, 'fingerprint': 'bad'}, config['condition'])
        changed = dict(config, source_hashes={**config['source_hashes'],
                      str(Path(v20.__file__).resolve()): '0' * 64})
        changed.pop('fingerprint')
        changed['fingerprint'] = v20._fingerprint(changed)
        with self.assertRaisesRegex(ValueError, 'source byte'):
            v20.validate_effective(changed, changed['condition'])
        with self.assertRaises(ValueError):
            v20.effective_config(dict(self.base(), diagnostic=True),
                                 'M1_R1_A8_current_output', v20.ALL_NATIVE_LEGAL)
        for key, value in (('max_cache_bytes', 0), ('max_summary_bytes', -1),
                           ('max_cache_bytes', 2.5), ('max_summary_bytes', True)):
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, 'memory caps'):
                    v20.effective_config(dict(self.base(), **{key: value}),
                                         'M1_R1_A8_current_output', v20.ALL_NATIVE_LEGAL)

    def test_plugin_dispatch_checks_bound_runtime_and_reports_effective_mode(self):
        config = v20.effective_config(self.base(), 'M3_R3_A8_current_output',
                                      v20.GLOBAL_ONLY_NATIVE_LOCAL)
        calls = []
        @contextmanager
        def fake_install(adapter, effective, condition):
            calls.append((adapter, effective['v20_arm'], condition))
            router = types.SimpleNamespace(cache=types.SimpleNamespace(score_period=8,
                                           decision_interval=3, max_bytes=v20.MAX_CACHE_BYTES),
                                           max_summary_bytes=v20.MAX_SUMMARY_BYTES,
                                           output_mode=v20.PREQK_MODE, support='native_mask',
                                           selector='legacy_recompute', selector_layers='all')
            state = types.SimpleNamespace(fast_t=True)
            yield dict(binding=object(), router=router, state=state,
                       counters=lambda: {'calls': 2, 'score_peak_transient_bytes': 300,
                                         'summary_peak_bytes': 100})
        fake_module = types.ModuleType('experiments.numerical_qk_reuse.global_scope')
        fake_module.install = fake_install
        adapter = object()
        with patch.dict(sys.modules, {'experiments.numerical_qk_reuse.global_scope': fake_module}):
            with v20.install(adapter, config, config['condition']) as runtime:
                self.assertEqual(runtime['counters']()['v20_arm'], config['v20_arm'])
                self.assertEqual(runtime['counters']()['decision_interval'], 3)
                self.assertEqual(runtime['counters']()['history_summary_transient_upper_bound_bytes'], 400)
        self.assertEqual(calls, [(adapter, config['v20_arm'], 'v20_global_M3_R3')])


if __name__ == '__main__':
    unittest.main()
