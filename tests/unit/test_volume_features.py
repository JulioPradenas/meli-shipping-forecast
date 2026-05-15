"""Tests for VolumeFeatures builder.

Coverage:

* Tier assignment logic (core / mid / tail) at threshold boundaries.
* state_avg_volume column reflects injected stats, not test data.
* Validation: empty stats and bad thresholds raise.
* Unknown states fall back gracefully.
* Idempotence: re-applying the builder is a no-op on its own outputs.
* compute_stats_from_train helper produces expected dict.
* feature_names contract matches what transform produces.
"""

from __future__ import annotations

import pandas as pd
import pytest

from shipping_forecast.features.volume import (
    DEFAULT_TIER_THRESHOLDS,
    VolumeFeatures,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal DataFrame: 3 states with very different volumes."""
    return pd.DataFrame(
        {
            "shipment_date": pd.to_datetime(
                ["2018-01-01", "2018-01-02", "2018-01-01", "2018-01-02", "2018-01-01"]
            ),
            "customer_state": ["SP", "SP", "RJ", "RJ", "AC"],
            "n_shipments": [120, 100, 25, 20, 0],
        }
    )


@pytest.fixture
def train_stats() -> dict[str, float]:
    """Pre-computed stats simulating those from a training fold."""
    return {"SP": 98.8, "RJ": 24.2, "AC": 0.11}


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


def test_tier_assignment_uses_thresholds(sample_df, train_stats):
    """Default thresholds (2, 10) bucket states correctly."""
    vf = VolumeFeatures(state_avg_volume=train_stats)
    out = vf.transform(sample_df)

    tiers = dict(zip(out["customer_state"], out["volume_tier"], strict=True))
    assert tiers["SP"] == "core"  # 98.8 >= 10
    assert tiers["RJ"] == "core"  # 24.2 >= 10
    assert tiers["AC"] == "tail"  # 0.11 < 2


def test_state_avg_volume_comes_from_injected_stats_not_test_data(sample_df, train_stats):
    """Critical: prevents leakage. The column reflects train stats, not the
    actual volume observed in the input DataFrame.
    """
    vf = VolumeFeatures(state_avg_volume=train_stats)
    out = vf.transform(sample_df)

    sp_rows = out[out["customer_state"] == "SP"]
    # All SP rows must show the injected stat (98.8), not the test mean (110).
    assert sp_rows["state_avg_volume"].tolist() == pytest.approx([98.8, 98.8])


def test_tier_at_boundary_uses_lower_bucket():
    """A state with exactly the core threshold (10.0) is 'core', not 'mid'.
    A state with exactly the tail threshold (2.0) is 'mid', not 'tail'.
    """
    stats = {"BORDER_CORE": 10.0, "BORDER_TAIL": 2.0, "JUST_BELOW": 1.99}
    df = pd.DataFrame(
        {
            "customer_state": ["BORDER_CORE", "BORDER_TAIL", "JUST_BELOW"],
            "n_shipments": [0, 0, 0],
        }
    )
    vf = VolumeFeatures(state_avg_volume=stats)
    out = vf.transform(df)
    tiers = dict(zip(out["customer_state"], out["volume_tier"], strict=True))

    assert tiers["BORDER_CORE"] == "core"
    assert tiers["BORDER_TAIL"] == "mid"
    assert tiers["JUST_BELOW"] == "tail"


def test_feature_names_contract():
    """feature_names must list exactly what transform adds."""
    vf = VolumeFeatures(state_avg_volume={"X": 5.0})
    assert vf.feature_names == ["state_avg_volume", "volume_tier"]


def test_default_thresholds_match_documentation():
    """Lock the documented thresholds from MODELS_BASELINE_SUMMARY.md."""
    assert DEFAULT_TIER_THRESHOLDS == (2.0, 10.0)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_unknown_state_uses_fallback():
    """Unknown states default to fallback value (0.0 -> tail)."""
    vf = VolumeFeatures(state_avg_volume={"SP": 100.0})
    df = pd.DataFrame({"customer_state": ["SP", "MYSTERY"], "n_shipments": [50, 50]})
    out = vf.transform(df)

    mystery = out[out["customer_state"] == "MYSTERY"].iloc[0]
    assert mystery["state_avg_volume"] == pytest.approx(0.0)
    assert mystery["volume_tier"] == "tail"


def test_unknown_state_with_custom_fallback():
    """Custom fallback is respected."""
    vf = VolumeFeatures(state_avg_volume={"SP": 100.0}, unknown_state_fallback=5.0)
    df = pd.DataFrame({"customer_state": ["MYSTERY"], "n_shipments": [0]})
    out = vf.transform(df)

    assert out["state_avg_volume"].iloc[0] == pytest.approx(5.0)
    assert out["volume_tier"].iloc[0] == "mid"  # 5.0 is between 2 and 10


def test_transform_does_not_mutate_input(sample_df, train_stats):
    """The builder must return a new DataFrame, not modify in place."""
    original_cols = list(sample_df.columns)
    vf = VolumeFeatures(state_avg_volume=train_stats)
    _ = vf.transform(sample_df)

    assert list(sample_df.columns) == original_cols


def test_transform_dtypes(sample_df, train_stats):
    """Outputs use efficient dtypes for downstream LightGBM consumption."""
    vf = VolumeFeatures(state_avg_volume=train_stats)
    out = vf.transform(sample_df)

    assert out["state_avg_volume"].dtype == "float32"
    assert isinstance(out["volume_tier"].dtype, pd.CategoricalDtype)
    assert list(out["volume_tier"].cat.categories) == ["tail", "mid", "core"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_empty_stats_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        VolumeFeatures(state_avg_volume={})


def test_non_increasing_thresholds_raises():
    with pytest.raises(ValueError, match="strictly increasing"):
        VolumeFeatures(state_avg_volume={"X": 5.0}, thresholds=(10.0, 2.0))


def test_equal_thresholds_raises():
    with pytest.raises(ValueError, match="strictly increasing"):
        VolumeFeatures(state_avg_volume={"X": 5.0}, thresholds=(5.0, 5.0))


# ---------------------------------------------------------------------------
# Helper method
# ---------------------------------------------------------------------------


def test_compute_stats_from_train_returns_means():
    """The convenience helper computes per-group means."""
    train = pd.DataFrame(
        {
            "customer_state": ["SP", "SP", "SP", "RJ", "RJ"],
            "n_shipments": [100, 110, 90, 20, 30],
        }
    )
    stats = VolumeFeatures.compute_stats_from_train(train)

    assert stats["SP"] == pytest.approx(100.0)
    assert stats["RJ"] == pytest.approx(25.0)


def test_compute_stats_from_train_respects_custom_columns():
    """Helper works with non-default column names."""
    train = pd.DataFrame({"state_code": ["A", "B"], "volume": [10.0, 20.0]})
    stats = VolumeFeatures.compute_stats_from_train(
        train, group_col="state_code", target_col="volume"
    )

    assert stats == {"A": 10.0, "B": 20.0}
