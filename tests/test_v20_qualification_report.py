"""CPU qualification gate checks using compact profiler-shaped receipts."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import v20_qualification_report as Q


class QualificationTests(unittest.TestCase):
    def fixture(self, root):
        name = 'GLOBAL_P0_M3_R2_A8_current_output'
        arm = dict(name=name, plugin='experiments.numerical_qk_reuse.v20:install',
                   condition='M3', config=dict(condition='M3', v20_arm='M3_R2_A8_current_output',
                                               v20_scope='GLOBAL_ONLY_NATIVE_LOCAL'))
        phases = [dict(attention_calls=30, score_refresh_calls=30,
                       decision_refresh_calls=30, held_decision_calls=0),
                  dict(attention_calls=30, score_refresh_calls=0,
                       decision_refresh_calls=0, held_decision_calls=30),
                  dict(attention_calls=30, score_refresh_calls=0,
                       decision_refresh_calls=30, held_decision_calls=0)]
        def sample(phase, step):
            return dict(layer=5, canvas=0, phase=phase, decoder_call=step,
                        v11_envelope_pass=True, new_max_abs_error=.01,
                        triton_max_abs_error=.02, relative_l2_error=.001,
                        actual_output_max_abs_error=.01,
                        actual_output_relative_l2_error=.001)
        sequence = dict(reached_calls=4, requested_calls=4,
                        native_bracket_drift=[dict(block=0, open_median_ms=1, close_median_ms=1.01)],
                        arms={name: dict(summary=dict(phase_deltas=phases),
                                         direct_epoch=dict(event_median_ms=5),
                                         blocks=[dict(triton_misses=0, triton_specializations_before=1,
                                                      triton_specializations_after=1,
                                                      triton_disk_entries_before=2,
                                                      triton_disk_entries_after=2,
                                                      peak_allocated_bytes=100)])},
                        diagnostic_replays={name: dict(status='qualified', operator_probe=dict(
                            status='qualified', rows=[sample('A', 0), sample('H', 1), sample('D', 2)]))})
        profile = dict(schema='v20_direct_full_forward_v1', revision='rev', arms=[arm],
                       runtime_identity=dict(hostname='mpk', gpu_uuid='GPU-1'),
                       operator_probe=True, counter_twins=True,
                       source_sha256={'/remote/v20_profile.py': 'abc'},
                       targets={'one': dict(dataset='aime26', id='private-id', canvas=0,
                                            requested_call=0, resolution=dict(missing=False),
                                            boundaries={'model_forward': {'N4': sequence},
                                                        'denoising_step': {'N4': copy.deepcopy(sequence)}}),
                                'two': dict(dataset='aime26', id='private-id', canvas=0,
                                            requested_call=2, resolution=dict(missing=False),
                                            boundaries={'model_forward': {'N4': sequence},
                                                        'denoising_step': {'N4': copy.deepcopy(sequence)}})})
        path = root / 'profile.json'
        path.write_text(json.dumps(profile), encoding='utf-8')
        return path, profile, name

    def test_qualified_and_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, name = self.fixture(Path(directory))
            report = Q.summarize([path])
            self.assertEqual(report['status'], 'qualified')
            self.assertNotIn('private-id', json.dumps(report))
            host = report['hosts'][0]
            self.assertEqual(host['source_sha256'], {'v20_profile.py': 'abc'})
            self.assertEqual(host['targets'][0]['boundaries']['model_forward']['arms'][name]['reached_phases'],
                             {'A': 1, 'D': 1, 'H': 1})
            self.assertEqual(host['targets'][0]['boundaries']['denoising_step']['status'], 'qualified')
            self.assertTrue(host['targets'][1]['shared_canvas_profile'])

    def test_missing_reached_phase_sample_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path, profile, name = self.fixture(Path(directory))
            seq = profile['targets']['one']['boundaries']['model_forward']['N4']
            seq['diagnostic_replays'][name]['operator_probe']['rows'].pop()
            path.write_text(json.dumps(profile), encoding='utf-8')
            arm = Q.summarize([path])['hosts'][0]['targets'][0]['boundaries']['model_forward']['arms'][name]
            self.assertEqual(arm['status'], 'failed')
            self.assertEqual(arm['missing_operator_samples'], [dict(layer=5, phase='D')])

    def test_missing_canvas_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            path, profile, _ = self.fixture(Path(directory))
            profile['targets']['one']['resolution']['missing'] = True
            path.write_text(json.dumps(profile), encoding='utf-8')
            self.assertEqual(Q.summarize([path])['status'], 'missing')

    def test_missing_late_call_on_shared_canvas_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            path, profile, _ = self.fixture(Path(directory))
            profile['targets']['two']['resolution']['missing'] = True
            path.write_text(json.dumps(profile), encoding='utf-8')
            report = Q.summarize([path])
            self.assertEqual(report['status'], 'missing')
            self.assertEqual(report['hosts'][0]['targets'][1]['status'], 'missing')

    def test_g75_requires_reached_a_h_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            path, profile, name = self.fixture(Path(directory))
            g75 = dict(name='G75', plugin='experiments.numerical_qk_reuse.v20_controls:install',
                       condition='v20_G75L30_nativeQ128', config=dict(v20_scope='ALL_NATIVE_LEGAL'))
            profile['arms'].append(g75)
            seq = profile['targets']['one']['boundaries']['model_forward']['N4']
            seq['arms']['G75'] = seq['arms'][name]
            seq['diagnostic_replays']['G75'] = dict(status='qualified', operator_probe=dict(status='qualified', rows=[]))
            seq['arms']['G75'] = dict(seq['arms']['G75'], summary=dict(phase_deltas=[
                dict(attention_calls=30, bootstrap_calls=0, bitmap_observation_calls=30, held_decision_calls=0),
                dict(attention_calls=30, bootstrap_calls=0, bitmap_observation_calls=0, held_decision_calls=30)]))
            path.write_text(json.dumps(profile), encoding='utf-8')
            arm = Q.summarize([path])['hosts'][0]['targets'][0]['boundaries']['model_forward']['arms']['G75']
            self.assertEqual(arm['status'], 'failed')
            self.assertEqual({x['phase'] for x in arm['missing_operator_samples']}, {'A', 'H'})

    def test_whole_step_jit_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path, profile, name = self.fixture(Path(directory))
            seq = profile['targets']['one']['boundaries']['denoising_step']['N4']
            seq['arms'][name]['blocks'][0]['triton_misses'] = 1
            path.write_text(json.dumps(profile), encoding='utf-8')
            report = Q.summarize([path])
            self.assertEqual(report['status'], 'failed')
            self.assertFalse(report['hosts'][0]['targets'][0]['boundaries']['denoising_step']['arms'][name]['no_new_jit'])

    def test_ratio_is_arm_over_native(self):
        with tempfile.TemporaryDirectory() as directory:
            path, profile, name = self.fixture(Path(directory))
            native = dict(name='native', condition='native_dense', plugin=None, config={})
            profile['arms'].append(native)
            seq = profile['targets']['one']['boundaries']['model_forward']['N4']
            seq['arms']['native'] = dict(seq['arms'][name], direct_epoch=dict(event_median_ms=10))
            path.write_text(json.dumps(profile), encoding='utf-8')
            method = Q.summarize([path])['hosts'][0]['targets'][0]['boundaries']['model_forward']['arms'][name]
            self.assertEqual(method['arm_over_native_event_ratio'], .5)


if __name__ == '__main__':
    unittest.main()
