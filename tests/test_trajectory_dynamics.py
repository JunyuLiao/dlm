import numpy as np
import pytest

from experiments.diffusion_gemma_sparse_attention_research.trajectory_dynamics import (
    attention_value_key_response,
    attention_state_directional_response,
    canvas_mass_amplification,
    correction_break_delta,
    deleted_mass_output,
    margin_safe,
    query_key_radius,
    retained_mass_lower_bound,
    stable_confidence_attribution,
    summarize_stop_bottleneck,
)


def test_stable_confident_attribution_requires_the_requested_dwell():
    argmax = np.array([[1, 2], [1, 3], [1, 3], [1, 3]])
    entropy = np.array([[0.20, 0.20], [0.10, 0.20], [0.01, 0.10], [0.01, 0.01]])
    result = stable_confidence_attribution(argmax, entropy, 0.05, dwell=2)

    # Stable transitions are calls 2 and 3 only for position 0; position 1
    # becomes stable after two consecutive equal transitions at call 3.
    assert result.first_stable_call.tolist() == [2, 3]
    assert result.first_confident_call.tolist() == [3, -1]
    assert result.terminal_stable.tolist() == [True, True]
    assert result.terminal_confident.tolist() == [True, False]
    assert result.terminal_stable.shape == (2,)


def test_covariance_response_matches_scalar_counterexample():
    keys = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    values = keys.copy()
    dense = np.full((1, 4), 0.25)
    low = np.array([[0.0, 0.5, 0.5, 0.0]])
    high = np.array([[0.5, 0.0, 0.0, 0.5]])
    dense_gain = attention_value_key_response(dense, keys, values)["jacobian"][0, 0, 0]
    low_gain = attention_value_key_response(low, keys, values)["jacobian"][0, 0, 0]
    high_gain = attention_value_key_response(high, keys, values)["jacobian"][0, 0, 0]
    assert dense_gain == pytest.approx(2.5)
    assert low_gain == pytest.approx(1.0)
    assert high_gain == pytest.approx(4.0)


def test_response_rejects_non_normalized_probability_rows():
    with pytest.raises(ValueError, match="sum to one"):
        attention_value_key_response(np.array([[0.2, 0.2]]), np.zeros((2, 1)), np.zeros((2, 1)))


def test_covariance_response_with_unequal_dimensions_matches_finite_difference():
    rng = np.random.default_rng(51994)
    query = rng.normal(size=(2, 3))
    keys = rng.normal(size=(5, 3))
    values = rng.normal(size=(5, 4))
    direction = rng.normal(size=(3,))
    scale = 0.6

    def attention(q):
        scores = scale * q @ keys.T
        scores -= scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=-1, keepdims=True)
        return p, p @ values

    p, _ = attention(query)
    result = attention_value_key_response(p, keys, values, scale=scale, directions=direction[None])
    epsilon = 1e-6
    finite_difference = (attention(query + epsilon * direction)[1] -
                         attention(query - epsilon * direction)[1]) / (2 * epsilon)
    assert result["jacobian"].shape == (2, 4, 3)
    np.testing.assert_allclose(result["directional"][:, 0], finite_difference, rtol=1e-6, atol=1e-8)


def test_deleted_mass_and_canvas_amplification_are_consistent():
    retained = np.array([[2.0, 0.0]])
    deleted = np.array([[0.0, 2.0]])
    delta = deleted_mass_output(retained, deleted, 0.25)
    np.testing.assert_allclose(delta, np.array([[0.5, -0.5]]))
    assert canvas_mass_amplification(0.25) == pytest.approx(4.0 / 3.0)


def test_full_state_response_includes_query_key_and_value_motion():
    rng = np.random.default_rng(260823)
    q, k, v = rng.normal(size=(2, 3)), rng.normal(size=(5, 3)), rng.normal(size=(5, 4))
    dq, dk, dv = rng.normal(size=q.shape), rng.normal(size=k.shape), rng.normal(size=v.shape)
    scale = 0.7

    def output(q_, k_, v_):
        scores = scale * q_ @ k_.T
        weights = np.exp(scores - scores.max(-1, keepdims=True))
        p = weights / weights.sum(-1, keepdims=True)
        return p, p @ v_

    p, _ = output(q, k, v)
    response = attention_state_directional_response(p, q, k, v, dq, dk, dv, scale=scale)
    epsilon = 1e-6
    finite = (output(q+epsilon*dq, k+epsilon*dk, v+epsilon*dv)[1] -
              output(q-epsilon*dq, k-epsilon*dk, v-epsilon*dv)[1]) / (2*epsilon)
    np.testing.assert_allclose(response['total'], finite, rtol=1e-6, atol=1e-8)
    assert not np.allclose(response['query'], finite)


def test_retained_mass_bound_matches_logit_form():
    bound = retained_mass_lower_bound(0.8, 0.1, 0.2)
    expected = 1.0 / (1.0 + np.exp(-(np.log(4.0) - 0.3)))
    assert bound == pytest.approx(expected)
    assert retained_mass_lower_bound(0.8, 0.0, 0.0) == pytest.approx(0.8)


def test_query_key_radius_and_margin_guard():
    radius = query_key_radius(np.array([[3.0, 4.0]]), 2.0, 0.5)
    assert radius.tolist() == pytest.approx([5.0])
    assert margin_safe(np.array([1.0, 1.0]), np.array([0.4, 0.5])).tolist() == [True, False]


def test_error_flow_and_stop_bottleneck_counts():
    assert correction_break_delta(0.8, 0.1, 0.02) == pytest.approx(-0.076)
    result = summarize_stop_bottleneck(
        np.array([True, False, True, False]),
        np.array([False, True, True, False]),
        np.array([False, False, False, True]),
    )
    assert result == {"stability_only": 1, "entropy_only": 1, "both": 1, "cap": 1, "none": 0}

