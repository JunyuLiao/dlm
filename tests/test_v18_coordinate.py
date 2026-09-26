import unittest
from unittest.mock import patch

from scripts.v18_coordinate import launch, marker_state


class CoordinatorTests(unittest.TestCase):
    def test_launch_waits_for_marker_without_relaunching(self):
        with patch('scripts.v18_coordinate.subprocess.run') as run, \
             patch('scripts.v18_coordinate.marker', side_effect=[None, None, {'start': 10}]) as marker, \
             patch('scripts.v18_coordinate.time.sleep'):
            launch('mpk', 'aime')
        self.assertEqual(run.call_count, 1)
        self.assertEqual(marker.call_count, 3)

    def test_marker_states_never_retry_failure(self):
        started = {'start': 10, 'pid': 1}
        self.assertEqual(marker_state(None, None), 'absent')
        self.assertEqual(marker_state(started, None), 'running')
        self.assertEqual(marker_state(started, {'start': 10, 'rc': 0}), 'complete')
        self.assertEqual(marker_state(started, {'start': 10, 'rc': 1}), 'failed')
        self.assertEqual(marker_state(started, {'start': 11, 'rc': 0}), 'failed')
        self.assertEqual(marker_state(None, {'start': 10, 'rc': 0}), 'failed')


if __name__ == '__main__':
    unittest.main()
