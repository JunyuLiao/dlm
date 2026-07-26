import math
import unittest

import torch

from spa import (
    SPAConfig,
    SPASupport,
    build_oracle_support,
    dense_attention_reference,
    exact_shared_support_attention,
)


def explicit_masked_attention(q, k, v, support, *, query_valid=None, key_valid=None, causal=False):
    batch, queries, heads, dim = q.shape
    keys = k.shape[1]
    if query_valid is None:
        query_valid = torch.ones(batch, queries, dtype=torch.bool)
    if key_valid is None:
        key_valid = torch.ones(batch, keys, dtype=torch.bool)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / math.sqrt(dim)
    groups = torch.arange(queries) // support.group_size
    allowed = support.key_mask[:, :, groups, :]
    allowed &= key_valid[:, None, None, :]
    allowed &= query_valid[:, None, :, None]
    if causal:
        allowed &= torch.arange(keys)[None, :] <= torch.arange(queries)[:, None]
    scores.masked_fill_(~allowed, -torch.inf)
    probabilities = torch.softmax(scores, -1)
    probabilities = torch.where(torch.isfinite(probabilities), probabilities, 0.0)
    return torch.einsum("bhqk,bkhd->bqhd", probabilities, v.float()).to(q.dtype)


class SharedPageAttentionReferenceTest(unittest.TestCase):
    def test_all_pages_retained_matches_dense(self):
        torch.manual_seed(1)
        q = torch.randn(2, 7, 3, 4)
        k = torch.randn(2, 9, 3, 4)
        v = torch.randn(2, 9, 3, 5)
        config = SPAConfig(group_size=4, page_size=4, support="pages", density=1.0)
        support, _ = build_oracle_support(q, k, config)
        actual = exact_shared_support_attention(q, k, v, support)
        expected = dense_attention_reference(q, k, v)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_one_page_is_renormalized_over_only_that_page(self):
        q = torch.ones(1, 4, 1, 1)
        k = torch.tensor([0.0, 1.0, 5.0, 6.0]).reshape(1, 4, 1, 1)
        v = torch.tensor([10.0, 20.0, 30.0, 40.0]).reshape(1, 4, 1, 1)
        support, _ = build_oracle_support(
            q, k, SPAConfig(group_size=4, page_size=2, density=0.5)
        )
        self.assertTrue(bool(support.key_mask[0, 0, 0, 2:].all()))
        actual = exact_shared_support_attention(q, k, v, support)
        expected_probability = torch.softmax(torch.tensor([5.0, 6.0]), 0)
        expected = (expected_probability * torch.tensor([30.0, 40.0])).sum()
        torch.testing.assert_close(actual, torch.full_like(actual, expected))

    def test_partial_final_page_and_invalid_keys(self):
        torch.manual_seed(2)
        q = torch.randn(1, 5, 1, 3)
        k = torch.randn(1, 6, 1, 3)
        v = torch.randn(1, 6, 1, 2)
        key_valid = torch.tensor([[True, True, True, True, True, False]])
        config = SPAConfig(
            group_size=4, page_size=4, density=0.5, mandatory_last=True
        )
        support, _ = build_oracle_support(q, k, config, key_valid_mask=key_valid)
        self.assertTrue(bool(support.key_mask[..., 4].all()))
        self.assertFalse(bool(support.key_mask[..., 5].any()))
        actual = exact_shared_support_attention(q, k, v, support, key_valid_mask=key_valid)
        expected = explicit_masked_attention(q, k, v, support, key_valid=key_valid)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_mixed_valid_and_invalid_query_rows(self):
        torch.manual_seed(3)
        q = torch.randn(2, 6, 2, 3)
        k = torch.randn(2, 6, 2, 3)
        v = torch.randn(2, 6, 2, 4)
        query_valid = torch.tensor(
            [[True, True, False, False, True, False], [False, True, True, True, False, True]]
        )
        support, _ = build_oracle_support(
            q, k, SPAConfig(group_size=3, page_size=2, density=2 / 3),
            query_valid_mask=query_valid,
        )
        actual = exact_shared_support_attention(
            q, k, v, support, query_valid_mask=query_valid
        )
        expected = explicit_masked_attention(
            q, k, v, support, query_valid=query_valid
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        self.assertTrue(bool((actual[~query_valid] == 0).all()))

    def test_bidirectional_attention_retains_future_key(self):
        q = torch.ones(1, 3, 1, 1)
        k = torch.tensor([0.0, 0.0, 10.0]).reshape(1, 3, 1, 1)
        v = torch.tensor([0.0, 0.0, 7.0]).reshape(1, 3, 1, 1)
        support, _ = build_oracle_support(
            q, k, SPAConfig(group_size=3, page_size=1, density=1 / 3)
        )
        output = exact_shared_support_attention(q, k, v, support, causal=False)
        self.assertGreater(float(output[0, 0, 0, 0]), 6.9)

    def test_causal_mask_is_respected(self):
        torch.manual_seed(4)
        q = torch.randn(1, 7, 2, 3)
        k = torch.randn(1, 7, 2, 3)
        v = torch.randn(1, 7, 2, 2)
        support, _ = build_oracle_support(
            q, k, SPAConfig(group_size=4, page_size=3, density=1.0), causal=True
        )
        actual = exact_shared_support_attention(q, k, v, support, causal=True)
        expected = dense_attention_reference(q, k, v, causal=True)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_heads_and_requests_select_independently(self):
        q = torch.ones(2, 2, 2, 1)
        k = torch.zeros(2, 4, 2, 1)
        v = torch.randn(2, 4, 2, 1)
        k[0, :2, 0] = 8
        k[0, 2:, 1] = 8
        k[1, 2:, 0] = 8
        k[1, :2, 1] = 8
        support, _ = build_oracle_support(
            q, k, SPAConfig(group_size=2, page_size=2, density=0.5)
        )
        chosen = support.ordinary_page_mask[..., 0, :]
        self.assertEqual(chosen[0, 0].tolist(), [True, False])
        self.assertEqual(chosen[0, 1].tolist(), [False, True])
        self.assertEqual(chosen[1, 0].tolist(), [False, True])
        self.assertEqual(chosen[1, 1].tolist(), [True, False])

    def test_explicit_dense_mask_equivalence_for_all_formats(self):
        torch.manual_seed(5)
        q = torch.randn(1, 7, 2, 4)
        k = torch.randn(1, 9, 2, 4)
        v = torch.randn(1, 9, 2, 3)
        for support_format, sinks in (("columns", 0), ("pages", 0), ("hybrid", 1)):
            with self.subTest(support=support_format):
                support, _ = build_oracle_support(
                    q, k, SPAConfig(
                        group_size=4, page_size=3, support=support_format,
                        density=1 / 3, sink_pages=sinks,
                    )
                )
                actual = exact_shared_support_attention(q, k, v, support)
                expected = explicit_masked_attention(q, k, v, support)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_coverage_selector_protects_minority_row(self):
        # Page 0 wins the mean, page 1 is essential for a single row, and page
        # 2 is the best second mean page. Coverage should choose page 1 second.
        probabilities = torch.tensor(
            [[0.30, 0.30, 0.0, 0.0, 0.20, 0.20]] * 3
            + [[0.0, 0.0, 0.49, 0.49, 0.01, 0.01]]
        ).reshape(1, 1, 4, 6)
        # Construct Q/K is unnecessary for this exact selector distinction;
        # use values whose softmax realizes the desired row distributions.
        q = torch.eye(4).reshape(1, 4, 1, 4)
        logits = probabilities[0, 0].clamp_min(1e-8).log()
        k = (logits.T * 2).reshape(1, 6, 1, 4)  # scale is 1/sqrt(4)
        mean, _ = build_oracle_support(
            q, k, SPAConfig(
                group_size=4, page_size=2, density=2 / 3, selector="oracle-mean"
            )
        )
        coverage, _ = build_oracle_support(
            q, k, SPAConfig(
                group_size=4, page_size=2, density=2 / 3,
                selector="oracle-coverage", coverage_objective="minimum",
            )
        )
        self.assertFalse(bool(mean.ordinary_page_mask[0, 0, 0, 1]))
        self.assertTrue(bool(coverage.ordinary_page_mask[0, 0, 0, 1]))

if __name__ == "__main__":
    unittest.main()
