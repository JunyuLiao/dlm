import argparse
from types import SimpleNamespace
import unittest

import torch

from scripts.llada_blasst_vs_sparsed import (
    answer_metrics,
    clear_sparsed_runtime_state,
    generate_memory_efficient,
    get_num_transfer_tokens,
    length_scaled_schedule_values,
    normalize_answer,
    parse_csv,
    validate_args,
)


class _ToyOfficialLLaDA(torch.nn.Module):
    """Small model with the same final-norm/output-head layout as LLaDA."""

    def __init__(self, vocab_size: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.model = torch.nn.Module()
        self.model.transformer = torch.nn.Module()
        self.model.transformer.ln_f = torch.nn.Identity()
        self.observed_logit_lengths: list[int] = []

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids, SparseD_param=None):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = positions.remainder(self.vocab_size)
        hidden = torch.nn.functional.one_hot(
            positions, num_classes=self.vocab_size
        ).float().unsqueeze(0)
        hidden = self.model.transformer.ln_f(hidden)
        self.observed_logit_lengths.append(hidden.shape[1])
        return SimpleNamespace(logits=hidden * 10.0)


class OfficialSparseDComparisonTest(unittest.TestCase):
    def test_parse_csv_requires_unique_nonempty_values(self):
        self.assertEqual(parse_csv("4k, 8k,64k"), ["4k", "8k", "64k"])
        with self.assertRaises(ValueError):
            parse_csv("")
        with self.assertRaises(ValueError):
            parse_csv("4k,4k")

    def test_answer_normalization_and_prefix_scoring(self):
        self.assertEqual(normalize_answer("\\boxed{5}."), "5")
        self.assertTrue(answer_metrics("Yes", "yes")["exact_expected_answer"])
        self.assertTrue(answer_metrics("Yes, they were.", "yes")["expected_answer_prefix"])
        self.assertFalse(answer_metrics("No", "yes")["expected_answer_prefix"])

    def test_length_scaling_preserves_boundaries_and_inverse_rule(self):
        base = length_scaled_schedule_values(4096)
        long = length_scaled_schedule_values(65536)
        self.assertEqual(base["low_noise_lambda"], 1.0)
        self.assertAlmostEqual(long["low_noise_lambda"], 1.0 / 16.0)
        self.assertAlmostEqual(
            long["mid_noise_lambda"], base["mid_noise_lambda"] / 16.0
        )
        self.assertEqual(long["high_noise_boundary"], 0.75)
        self.assertEqual(long["low_noise_boundary"], 0.25)

    def test_transfer_schedule_matches_official_uniform_allocation(self):
        mask = torch.tensor([[True, True, True, True, True], [True, True, False, False, False]])
        result = get_num_transfer_tokens(mask, 4)
        self.assertEqual(result.tolist(), [[2, 1, 1, 1], [1, 1, 0, 0]])

    def test_paper_defaults_validate(self):
        args = argparse.Namespace(
            prompts="4k,8k,16k,32k,64k",
            seq_len=128,
            steps=128,
            block_length=32,
            warmup=1,
            repeats=1,
            sparsed_skip=0.2,
            sparsed_select=0.3,
            temperature=0.0,
        )
        self.assertEqual(
            validate_args(args, ["short_context", "4k", "8k", "16k", "32k", "64k"]),
            ["4k", "8k", "16k", "32k", "64k"],
        )

    def test_generation_projects_only_suffix_and_restores_model(self):
        model = _ToyOfficialLLaDA()
        prompt = torch.tensor([[1, 1]])
        output = generate_memory_efficient(
            model,
            prompt,
            steps=4,
            gen_length=4,
            block_length=2,
            temperature=0.0,
            remasking="low_confidence",
            mask_id=7,
        )
        self.assertEqual(output.tolist(), [[1, 1, 2, 3, 4, 5]])
        self.assertEqual(model.observed_logit_lengths, [4, 4, 4, 4])
        self.assertEqual(model(prompt).logits.shape[1], 2)

    def test_sparse_runtime_state_cleanup(self):
        model = torch.nn.Sequential(torch.nn.Identity())
        model[0].fine_mask = torch.ones(1)
        model[0].block_mask = object()
        model[0].last = 3
        clear_sparsed_runtime_state(model)
        self.assertIsNone(model[0].fine_mask)
        self.assertIsNone(model[0].block_mask)
        self.assertIsNone(model[0].last)


if __name__ == "__main__":
    unittest.main()
