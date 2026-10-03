import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from scripts.v28_alias2_q64_bench import support, selected_paths


class SupportTests(unittest.TestCase):
    def test_select_by_length_bin_not_exact_length(self):
        paths = [Path(str(i)+'.npz') for i in range(6)]
        lengths = [490, 544, 560, 1000, 1100, 1500]
        needs = {p:np.zeros((1, 256, n), dtype=bool) for p,n in zip(paths,lengths)}
        with patch('pathlib.Path.glob', return_value=paths), patch('scripts.v28_regroup_screen.load_need', side_effect=needs.__getitem__):
            self.assertEqual(selected_paths('unused',2), [paths[0],paths[1],paths[3],paths[4],paths[5]])

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
