"""CPU-only checks for native run selection and frozen request identity."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.numerical_qk_reuse.runner import InitialPrefillTimeline, _config, _receipt_path, _rows, _selected, parse


class NativeRunnerConfigTest(unittest.TestCase):
    def test_smoke_requires_four_explicit_authorized_ids(self):
        rows = [dict(id=f"aime26/{i}", prompt=f"Question {i}") for i in range(1, 6)]
        chosen = _selected(rows, ["aime26/4", "aime26/1", "aime26/3", "aime26/2"], "smoke")
        self.assertEqual([x["id"] for x in chosen], ["aime26/4", "aime26/1", "aime26/3", "aime26/2"])
        with self.assertRaises(ValueError):
            _selected(rows, ["aime26/1"], "smoke")
        with self.assertRaises(ValueError):
            _selected(rows, ["aime26/1", "aime26/2", "aime26/3", "not-authorized"], "smoke")

    def test_manifest_rejects_changed_thinking_or_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            for field, value in (("thinking", False), ("generation_budget", 512)):
                path.write_text(json.dumps([{"id": "x", "prompt": "P", field: value}]))
                with self.assertRaises(ValueError):
                    _rows(path)

    def test_attempt_zero_path_is_stable(self):
        a = _receipt_path(Path("out"), "smoke", "M1", 42, "aime26/01")
        b = _receipt_path(Path("out"), "smoke", "M1", 42, "aime26/01")
        self.assertEqual(a, b)
        self.assertTrue(a.name.endswith(".attempt0.json"))

    def test_cli_rejects_smoke_seed_change(self):
        with self.assertRaises(SystemExit):
            parse(["--manifest", "x", "--output", "y", "--model", "z", "--revision", "r",
                   "--condition", "native_dense", "--ids", "a", "b", "c", "d", "--seeds", "43"])

    def test_native_config_freezes_policy_manifest_and_model_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            model = base / "model"
            model.mkdir()
            (model / "config.json").write_text('{"canvas_length":256}')
            manifest = base / "manifest.json"
            manifest.write_text('[{"id":"q1","prompt":"P"}]')
            policy = base / "policy.json"
            policy.write_text(json.dumps({"m_ref": 1.5, "beta": 3, "gamma": .5,
                                          "policies": {"T_s50": {"local": {"log_threshold": -1},
                                                                   "global": {"log_threshold": -3}}}}))
            args = parse(["--manifest", str(manifest), "--output", str(base / "out"),
                          "--model", str(model), "--revision", "rev", "--condition", "native_dense",
                          "--ids", "a", "b", "c", "d", "--policy", str(policy)])
            with patch("experiments.numerical_qk_reuse.runner._source_hashes", return_value={"source": "hash"}):
                config = _config(args)
            self.assertEqual(config["max_new_tokens"], 8192)
            self.assertTrue(config["thinking"])
            self.assertTrue(config["native_adaptive"])
            self.assertIn("config.json", config["model_metadata_hashes"])
            self.assertEqual(config["policy"]["local"]["log_threshold"], -1)
            self.assertFalse(config["timing_events"])

    def test_optional_first_encoder_event_records_once(self):
        class Event:
            def __init__(self, *, enable_timing):
                self.enable_timing = enable_timing
                self.recorded = False

            def record(self):
                self.recorded = True

            def elapsed_time(self, other):
                self.assert_recorded(other)
                return 123.0

            def assert_recorded(self, other):
                assert self.recorded and other.recorded

        class Handle:
            def __init__(self, owner):
                self.owner = owner

            def remove(self):
                self.owner.hook = None

        class DiffusionGemmaEncoderModel:
            def __init__(self):
                self.hook = None

            def register_forward_hook(self, hook):
                self.hook = hook
                return Handle(self)

        encoder = DiffusionGemmaEncoderModel()
        model = SimpleNamespace(modules=lambda: [encoder])
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with InitialPrefillTimeline(model, True) as timeline:
                encoder.hook()
                first = timeline.first_end
                encoder.hook()
                self.assertIs(timeline.first_end, first)
                timeline.mark_final()
                self.assertAlmostEqual(timeline.seconds(), .123)
        self.assertIsNone(encoder.hook)


if __name__ == "__main__":
    unittest.main()
