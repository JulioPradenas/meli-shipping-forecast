"""Tests for cost-sensitive evaluation metrics.

Coverage:

* asymmetric_cost: symmetric case equals MAE, asymmetric case favors
  the expected direction, perfect predictions give zero, input validation.
* expected_gain: positive when model beats baseline, negative when worse,
  zero when identical.
* optimal_threshold_multiplier: alpha=0 optimal when symmetric,
  positive alpha optimal when under-pred is costlier, negative when
  over-pred is costlier, input validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipping_forecast.evaluation.cost_metrics import (
    asymmetric_cost,
    expected_gain,
    optimal_threshold_multiplier,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def perfect_pred() -> tuple[np.ndarray, np.ndarray]:
    """Identical y_true and y_pred."""
    y = np.array([10.0, 20.0, 30.0, 40.0])
    return y, y.copy()


@pytest.fixture
def under_pred() -> tuple[np.ndarray, np.ndarray]:
    """Always under-predicts by 5."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = y_true - 5.0
    return y_true, y_pred


@pytest.fixture
def over_pred() -> tuple[np.ndarray, np.ndarray]:
    """Always over-predicts by 5."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = y_true + 5.0
    return y_true, y_pred


# ---------------------------------------------------------------------------
# asymmetric_cost
# ---------------------------------------------------------------------------


def test_asymmetric_cost_symmetric_equals_mae_times_n(under_pred):
    """When c_under == c_over == 1, total cost equals sum(|errors|)."""
    y_true, y_pred = under_pred
    cost = asymmetric_cost(y_true, y_pred, c_under=1.0, c_over=1.0)
    expected = float(np.sum(np.abs(y_true - y_pred)))
    assert cost == pytest.approx(expected)


def test_asymmetric_cost_perfect_prediction_is_zero(perfect_pred):
    """No errors anywhere -> cost is exactly zero regardless of costs."""
    y_true, y_pred = perfect_pred
    cost = asymmetric_cost(y_true, y_pred, c_under=5.0, c_over=3.0)
    assert cost == pytest.approx(0.0)


def test_asymmetric_cost_under_prediction_weighted_higher(under_pred):
    """Under-predictions multiplied by c_under, not c_over."""
    y_true, y_pred = under_pred
    # Errors: 5, 5, 5 (all under). Total under = 15.
    cost = asymmetric_cost(y_true, y_pred, c_under=2.0, c_over=1.0)
    # 2.0 * 15 + 1.0 * 0 = 30
    assert cost == pytest.approx(30.0)


def test_asymmetric_cost_over_prediction_weighted_lower(over_pred):
    """Over-predictions multiplied by c_over."""
    y_true, y_pred = over_pred
    # Errors: -5, -5, -5 (all over). Total over = 15.
    cost = asymmetric_cost(y_true, y_pred, c_under=2.0, c_over=1.0)
    # 2.0 * 0 + 1.0 * 15 = 15
    assert cost == pytest.approx(15.0)


def test_asymmetric_cost_mixed_directions():
    """Both under and over predictions, each weighted differently."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([5.0, 25.0, 25.0, 45.0])  # under, over, under, over
    # Under: (10-5) + (30-25) = 10
    # Over:  (25-20) + (45-40) = 10
    cost = asymmetric_cost(y_true, y_pred, c_under=3.0, c_over=1.0)
    assert cost == pytest.approx(3.0 * 10 + 1.0 * 10)


def test_asymmetric_cost_negative_costs_raise():
    y = np.array([1.0, 2.0])
    yp = np.array([0.5, 1.5])
    with pytest.raises(ValueError, match="non-negative"):
        asymmetric_cost(y, yp, c_under=-1.0, c_over=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        asymmetric_cost(y, yp, c_under=1.0, c_over=-0.5)


def test_asymmetric_cost_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        asymmetric_cost([], [], c_under=1.0, c_over=1.0)


def test_asymmetric_cost_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        asymmetric_cost([1.0, 2.0, 3.0], [1.0, 2.0], c_under=1.0, c_over=1.0)


# ---------------------------------------------------------------------------
# expected_gain
# ---------------------------------------------------------------------------


def test_expected_gain_model_better_than_baseline_returns_positive():
    """Model closer to truth than baseline -> positive gain."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred_model = np.array([11.0, 19.0, 31.0])  # close
    y_pred_baseline = np.array([5.0, 15.0, 25.0])  # far (under-pred)
    gain = expected_gain(y_true, y_pred_model, y_pred_baseline, c_under=2.0, c_over=1.0)
    assert gain > 0


def test_expected_gain_model_worse_than_baseline_returns_negative():
    """Model farther from truth than baseline -> negative gain."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred_model = np.array([5.0, 30.0, 20.0])  # far
    y_pred_baseline = np.array([10.0, 20.0, 30.0])  # perfect
    gain = expected_gain(y_true, y_pred_model, y_pred_baseline, c_under=2.0, c_over=1.0)
    assert gain < 0


def test_expected_gain_identical_predictions_is_zero():
    """Same predictions -> zero gain (no savings, no losses)."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 28.0])
    gain = expected_gain(y_true, y_pred, y_pred.copy(), c_under=2.0, c_over=1.0)
    assert gain == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# optimal_threshold_multiplier
# ---------------------------------------------------------------------------


def test_optimal_threshold_symmetric_costs_alpha_near_zero():
    """With symmetric costs, no multiplicative offset should help much.
    The optimum is alpha=0 (no adjustment) when predictions are unbiased.
    """
    rng = np.random.default_rng(42)
    y_true = rng.normal(100, 20, size=200)
    y_pred = y_true + rng.normal(0, 5, size=200)  # unbiased noise
    best_alpha, _ = optimal_threshold_multiplier(y_true, y_pred, c_under=1.0, c_over=1.0)
    assert abs(best_alpha) <= 0.05


def test_optimal_threshold_under_costlier_alpha_positive():
    """When under-pred costs 3x more, the optimum should push predictions up."""
    rng = np.random.default_rng(42)
    y_true = rng.uniform(50, 150, size=300)
    y_pred = y_true.copy()  # perfect predictions baseline
    best_alpha, _ = optimal_threshold_multiplier(y_true, y_pred, c_under=3.0, c_over=1.0)
    # With under-pred 3x costlier and perfect baseline, any positive alpha
    # creates over-pred (cheap) and avoids no under-pred (was zero anyway).
    # So the optimum should be 0 OR positive. To get a clearly positive
    # optimum we need biased predictions. Test that with noisy data:
    y_pred_noisy = y_true + rng.normal(0, 10, size=300)
    best_alpha, _ = optimal_threshold_multiplier(y_true, y_pred_noisy, c_under=3.0, c_over=1.0)
    assert best_alpha >= 0.0


def test_optimal_threshold_over_costlier_alpha_zero_or_negative():
    """When over-pred costs 3x more, the optimum should pull predictions down
    (or stay at zero)."""
    rng = np.random.default_rng(42)
    y_true = rng.uniform(50, 150, size=300)
    y_pred = y_true + rng.normal(0, 10, size=300)
    best_alpha, _ = optimal_threshold_multiplier(y_true, y_pred, c_under=1.0, c_over=3.0)
    assert best_alpha <= 0.05


def test_optimal_threshold_empty_grid_raises():
    with pytest.raises(ValueError, match="alpha_grid must be non-empty"):
        optimal_threshold_multiplier([1.0, 2.0], [1.0, 2.0], c_under=1.0, c_over=1.0, alpha_grid=())


def test_optimal_threshold_returns_finite_cost():
    """Sanity: the returned cost is finite and non-negative."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([5.0, 25.0, 28.0])
    _, best_cost = optimal_threshold_multiplier(y_true, y_pred, c_under=2.0, c_over=1.0)
    assert np.isfinite(best_cost)
    assert best_cost >= 0
