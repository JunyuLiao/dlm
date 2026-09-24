"""CPU checks for the separate native stop observer; no model or CUDA needed."""
from types import SimpleNamespace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from scripts import native_reuse_stop_diagnostic as diagnostic


class ConfigTest(unittest.TestCase):
    def test_default_scope_and_runner_config_shape(self):
        args = diagnostic.parse(["--manifest", "m", "--output", "o", "--model", "x",
                                 "--revision", "r"])
        self.assertEqual(args.id, "aime26/2")
        self.assertEqual(args.conditions, ["M1"])
        native = diagnostic._args_for(args, "native_dense")
        m3 = diagnostic._args_for(args, "M3")
        self.assertEqual(native.phase, "stop_diagnostic")
        self.assertEqual(native.ids, ["aime26/2"])
        self.assertEqual(m3.decision_interval, 2)
        self.assertEqual(m3.plugin, diagnostic.PLUGIN)
        self.assertEqual(m3.extra_source, [diagnostic.Path(diagnostic.__file__)])
        self.assertFalse(m3.diagnostic)
        self.assertFalse(m3.timing_events)

    def test_native_source_hash_gate_and_frozen_id_shape(self):
        args = diagnostic.parse(["--manifest", "m", "--output", "o", "--model", "x",
                                 "--revision", "r"])
        source = {"/env/generation_diffusion_gemma.py": "other"}
        with patch.object(diagnostic.runner, "_config", return_value={"source_hashes": source}):
            with self.assertRaises(RuntimeError):
                diagnostic._config(args, "M1")
        identity = json.loads(diagnostic.IDENTITY.read_text())
        self.assertEqual(identity["selected_ids"],
                         ["aime26/2", "aime26/8", "aime26/14", "aime26/20"])

    def test_runner_accepts_diagnostic_phase_and_source_hash_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            model = base / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            manifest = base / "manifest.json"
            manifest.write_text('[{"id":"aime26/2","prompt":"P"}]')
            policy = base / "policy.json"
            policy.write_text(json.dumps({
                "m_ref": 1.0, "beta": 1.0, "gamma": 1.0,
                "policies": {"T_s50": {"local": {}, "global": {}}}}))
            args = diagnostic.parse(["--manifest", str(manifest), "--output", str(base / "out"),
                                     "--model", str(model), "--revision", "rev",
                                     "--policy", str(policy)])
            native_path = "/env/generation_diffusion_gemma.py"
            native_sha = diagnostic.EXPECTED_NATIVE_SOURCE_SHA256
            with patch.object(diagnostic.runner, "_source_hashes",
                              return_value={native_path: native_sha}):
                config = diagnostic._config(args, "native_dense")
            self.assertEqual(config["phase"], "stop_diagnostic")
            self.assertEqual(config["source_hashes"][native_path], native_sha)
            self.assertFalse(config["diagnostic"])
            self.assertFalse(config["timing_events"])


try:
    import torch
except ImportError:
    torch = None


if torch is not None:
    class StableAndConfidentStoppingCriteria:
        """CPU copy of the inspected native predicate for observer tests."""

        def __init__(self, stability_threshold=1, confidence_threshold=.5):
            self.stability_threshold = stability_threshold
            self.confidence_threshold = confidence_threshold
            self.argmax_canvas_history = None
            self.calls = 0

        def __call__(self, argmax_canvas, logits, **kwargs):
            self.calls += 1
            if self.stability_threshold == 0:
                stable = torch.ones((logits.shape[0],), dtype=torch.bool)
            else:
                if self.argmax_canvas_history is None:
                    self.argmax_canvas_history = torch.full(
                        (self.stability_threshold, *argmax_canvas.shape), -1,
                        dtype=argmax_canvas.dtype)
                stable = (self.argmax_canvas_history == argmax_canvas[None]).all(-1).all(0)
                self.argmax_canvas_history = torch.roll(self.argmax_canvas_history, -1, 0)
                self.argmax_canvas_history[-1] = argmax_canvas
            confident = torch.distributions.Categorical(logits=logits).entropy().mean(-1) < self.confidence_threshold
            return stable & confident


    class FakeSampler:
        def __init__(self, mask):
            self.accepted_token_mask = mask
            self.accept_calls = self.renoise_calls = 0

        def accept_canvas(self, current, proposed, logits, cur_step):
            self.accept_calls += 1
            return torch.where(self.accepted_token_mask, proposed, current)

        def renoise_canvas(self, accepted_canvas, cur_step):
            self.renoise_calls += 1
            return torch.where(~self.accepted_token_mask,
                               torch.full_like(accepted_canvas, -1), accepted_canvas)


@unittest.skipIf(torch is None, "CPU PyTorch not installed")
class NativeSignalTest(unittest.TestCase):
    def test_stop_spy_matches_conjunction_and_delegates_once(self):
        native = StableAndConfidentStoppingCriteria()
        top = torch.tensor([[0, 0, 0]])
        logits = torch.tensor([[[20., 0.], [20., 0.], [20., 0.]]])
        first, second = {}, {}
        result1 = diagnostic.StopSpy(native, first, None)(top, logits)
        result2 = diagnostic.StopSpy(native, second, top)(top, logits)
        self.assertEqual(native.calls, 2)
        self.assertFalse(result1.item())
        self.assertFalse(first["stable"].item())
        self.assertTrue(first["confident"].item())
        self.assertTrue(result2.item())
        self.assertTrue(second["stable"].item())
        self.assertTrue(second["confident"].item())
        self.assertEqual(second["top1_flips_vs_previous_completed"].item(), 0)
        self.assertTrue(torch.equal(second["native_criterion_stop"],
                                    second["stable"] & second["confident"]))

    def test_sampler_acceptance_partition_and_delegation(self):
        native = FakeSampler(torch.tensor([[True, False, True]]))
        record = {}
        spy = diagnostic.SamplerSpy(native, record)
        current = torch.tensor([[1, 1, 1]])
        proposed = torch.tensor([[2, 2, 2]])
        accepted = spy.accept_canvas(current, proposed, None, 48)
        renoised = spy.renoise_canvas(accepted, 48)
        self.assertEqual(native.accept_calls, 1)
        self.assertEqual(native.renoise_calls, 1)
        self.assertEqual(record["accepted"].item(), 2)
        self.assertEqual(record["renoised"].item(), 1)
        self.assertEqual(record["accepted"].item() + record["renoised"].item(), 3)
        self.assertEqual(accepted.tolist(), [[2, 1, 2]])
        self.assertEqual(renoised.tolist(), [[2, -1, 2]])

    def test_canvas_reset_and_decoder_call_indexing(self):
        class FakeModel:
            def _denoising_step(self, **kwargs):
                current = kwargs["current_canvas"]
                proposed = kwargs["argmax_canvas"]
                logits = kwargs["logits"]
                sampler = kwargs["sampler"]
                accepted = sampler.accept_canvas(current, proposed, logits, kwargs["cur_step"])
                renoised = sampler.renoise_canvas(accepted, kwargs["cur_step"])
                stopped = kwargs["diffusion_stopping_criteria"](proposed, logits)
                return renoised, proposed, logits, stopped

        model = FakeModel()
        observer = diagnostic.NativeStopObserver()
        top = torch.tensor([[0, 0, 0]])
        logits = torch.tensor([[[20., 0.], [20., 0.], [20., 0.]]])
        current = torch.zeros_like(top)
        first_stop = StableAndConfidentStoppingCriteria()
        second_stop = StableAndConfidentStoppingCriteria()
        with observer.observe(model):
            for schedule_step, criterion in ((48, first_stop), (47, first_stop), (48, second_stop)):
                model._denoising_step(cur_step=schedule_step, current_canvas=current,
                                      argmax_canvas=top, logits=logits,
                                      sampler=FakeSampler(torch.tensor([[True, False, True]])),
                                      diffusion_stopping_criteria=criterion)
        output = SimpleNamespace(metadata={"actual_denoising_step_count": 3,
                                           "native_canvas_length": 3},
                                 completion_tokens=[10, 11, 12, 13, 14, 15],
                                 termination_reason="length")
        steps, canvases = observer.finish(output)
        self.assertEqual([(x["canvas_index"], x["decoder_call"]) for x in steps],
                         [(0, 1), (0, 2), (1, 1)])
        self.assertEqual([x["actual_decoder_calls"] for x in canvases], [2, 1])
        self.assertIsNone(steps[0]["top1_flips_vs_previous_completed"])
        self.assertEqual(steps[1]["top1_flips_vs_previous_completed"], 0)
        self.assertIsNone(steps[2]["top1_flips_vs_previous_completed"])


if __name__ == "__main__":
    unittest.main()
