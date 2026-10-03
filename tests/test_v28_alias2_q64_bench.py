import unittest
import numpy as np
from scripts.v28_alias2_q64_bench import support


class SupportTests(unittest.TestCase):
    def test_refinement_preserves_row_needs_and_full_canvas(self):
        need = np.zeros((2, 256, 7), dtype=bool)
        need[0, 0, 1] = True
        need[0, 65, 2] = True
        for rows in (64, 128):
            kept = support(need, rows)
            expanded = kept.repeat(rows, axis=1)
            self.assertTrue(np.all(expanded[..., :7] | ~need))
            self.assertTrue(expanded[..., 7:].all())
        self.assertLess(support(need, 64)[..., :7].sum()*64,
                        support(need, 128)[..., :7].sum()*128)

    def test_invalid_rows(self):
        with self.assertRaises(ValueError):
            support(np.zeros((1, 256, 2), dtype=bool), 32)


if __name__ == '__main__':
    unittest.main()
