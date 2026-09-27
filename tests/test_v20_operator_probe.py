"""CPU-only scheduling and pinned numeric-envelope tests for the untimed probe."""
import unittest

from scripts.v20_operator_probe import OperatorProbe, within_v11_envelope


class OperatorProbeTests(unittest.TestCase):
    def test_only_first_phase_on_predeclared_layers_per_canvas(self):
        probe = OperatorProbe(repetitions=3)
        self.assertFalse(probe.wants(layer=1, canvas=0, phase='A'))
        self.assertFalse(probe.wants(layer=0, canvas=0, phase='other'))
        for layer in (0, 5):
            for phase in ('A', 'D', 'H'):
                self.assertTrue(probe.wants(layer=layer, canvas=0, phase=phase))
                probe.seen.add((0, layer, phase))
                self.assertFalse(probe.wants(layer=layer, canvas=0, phase=phase))
                self.assertTrue(probe.wants(layer=layer, canvas=1, phase=phase))

    def test_v11_envelope_is_exact_and_repetition_bound_is_strict(self):
        self.assertTrue(within_v11_envelope(1.5 * .002 + .001, .002, .01))
        self.assertFalse(within_v11_envelope(1.5 * .002 + .00101, .002, .01))
        self.assertFalse(within_v11_envelope(.001, .002, .01001))
        with self.assertRaisesRegex(ValueError, 'at least three'):
            OperatorProbe(repetitions=2)


if __name__ == '__main__':
    unittest.main()
