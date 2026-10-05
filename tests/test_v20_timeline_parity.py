"""CPU-only timeline observer parity and source identity tests."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments.numerical_qk_reuse.runner import _fingerprint
from scripts import v20_timeline_parity as P


class TimelineParityTests(unittest.TestCase):
    def receipt(self, span):
        return dict(id='ruler4k/a', seed=101, max_new_tokens=128,
                    completion_tokens=[8, 9, 10], per_canvas=[dict(canvas_index=0,
                    decoder_calls=2, schedule_steps=[48, 47], native_stop_final_call=True,
                    iteration_cap_final_call=False, completion_slice_tokens=3)],
                    total_decoder_calls=2, termination_reason='eos', counters={'same': True},
                    fingerprint='fingerprint', generation_gpu_timeline_seconds=span)

    def test_off_config_changes_only_observer_flag_and_fingerprint(self):
        base = dict(condition='M3', timing_events=True, threshold=-3.,
                    thinking=False, generation_budget=128)
        on = dict(base, fingerprint=_fingerprint(base))
        off = P.off_config(on)
        self.assertIs(off['timing_events'], False)
        self.assertNotEqual(on['fingerprint'], off['fingerprint'])
        self.assertEqual({key for key in off if off[key] != on[key]},
                         {'timing_events', 'fingerprint'})

    def test_exact_output_work_and_timeline_are_distinct_qualifications(self):
        on, off = self.receipt(.25), self.receipt(None)
        off['fingerprint'] = 'OFF-fingerprint'
        with patch('scripts.v20_run.router_phase_evidence', return_value={'A': 3, 'D': 1}):
            result = P.compare(on, off, 'M3_R2_A8_current_output')
            self.assertTrue(result['timeline_qualified'])
            self.assertTrue(all(result['exact'].values()))
            self.assertEqual(result['off_device_span'], None)
            on['generation_gpu_timeline_seconds'] = None
            unavailable = P.compare(on, off, 'M3_R2_A8_current_output')
            self.assertEqual(unavailable['status'], 'pass')
            self.assertFalse(unavailable['timeline_qualified'])
            self.assertIn('ON_span_unavailable', unavailable['timeline_reasons'])
            off['completion_tokens'] = [8, 9, 11]
            with self.assertRaisesRegex(ValueError, 'output/work parity drift'):
                P.compare(on, off, 'M3_R2_A8_current_output')

    def test_source_hook_matches_imported_frozen_runner_bytes(self):
        import experiments.numerical_qk_reuse.runner as runner
        path = str(__import__('pathlib').Path(runner.__file__).resolve())
        matching = {'source_hashes': {path: P.sha(__import__('pathlib').Path(path))}}
        proof = P.source_hook_identity(matching)
        self.assertEqual(proof['imported_runner_sha256'], matching['source_hashes'][path])
        matching['source_hashes'][path] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'source differs'):
            P.source_hook_identity(matching)


if __name__ == '__main__':
    unittest.main()
