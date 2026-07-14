import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sparsed_quick_eval.py"
SPEC = importlib.util.spec_from_file_location("sparsed_quick_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SparseDQuickEvalTests(unittest.TestCase):
    def test_token_agreement_penalizes_length_difference(self):
        self.assertEqual(MODULE.token_agreement([1, 2, 3], [1, 2]), 2 / 3)
        self.assertEqual(MODULE.token_agreement([], []), 1.0)

    def test_reference_match_is_normalized_and_optional(self):
        self.assertTrue(MODULE.reference_match("The answer is  42.", ["answer is 42."]))
        self.assertIsNone(MODULE.reference_match("anything", []))

    def test_reads_jsonl_references_and_filters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(
                '{"id":"a","prompt":"first","answer":"one"}\n'
                '{"id":"b","prompt":"second","references":["two","2"]}\n',
                encoding="utf-8",
            )
            cases = MODULE.read_prompts(path, {"b"}, None)
        self.assertEqual([(case.prompt_id, case.references) for case in cases], [("b", ("two", "2"))])

    def test_summary_reports_dense_over_sparse_speedup(self):
        rows = [
            MODULE.Measurement("p", "dense", 0, 10, 4, 20.0, 200.0, 2.0, "", "[]", None),
            MODULE.Measurement("p", "sparse", 0, 10, 4, 10.0, 400.0, 1.5, "", "[]", None),
        ]
        summary = MODULE.summarize(rows)
        self.assertEqual(summary["efficiency"]["speedup_dense_over_sparse"], 2.0)
        self.assertEqual(summary["efficiency"]["peak_memory_ratio_sparse_over_dense"], 0.75)


if __name__ == "__main__":
    unittest.main()
