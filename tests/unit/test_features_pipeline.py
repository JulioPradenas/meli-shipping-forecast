"""Unit tests for FeaturePipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from shipping_forecast.features.base import FeatureBuilder
from shipping_forecast.features.calendar import CalendarFeatures
from shipping_forecast.features.lags import LagFeatures
from shipping_forecast.features.pipeline import FeaturePipeline

# --------------------------------------------------------------------- fixtures


@pytest.fixture
def shipments_df() -> pd.DataFrame:
    """A minimal DataFrame for testing pipeline composition."""
    return pd.DataFrame(
        {
            "customer_state": ["SP"] * 5 + ["RJ"] * 5,
            "shipment_date": pd.concat(
                [pd.Series(pd.date_range("2024-01-01", periods=5))] * 2
            ).reset_index(drop=True),
            "n_shipments": [10, 20, 30, 40, 50, 5, 15, 25, 35, 45],
        }
    )


class BrokenBuilder(FeatureBuilder):
    """A builder that lies about what it adds — for negative-path tests."""

    @property
    def feature_names(self) -> list[str]:
        return ["this_column_will_never_appear"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Note: does NOT actually add the declared column
        return df.copy()


# --------------------------------------------------------------------- tests


def test_pipeline_chains_builders(shipments_df: pd.DataFrame) -> None:
    """A pipeline must apply every builder in order and add all features."""
    pipeline = FeaturePipeline(
        [
            LagFeatures(lags=[1]),
            CalendarFeatures(features=["day_of_week", "month"]),
        ]
    )
    out = pipeline.transform(shipments_df)

    for col in ["lag_1", "day_of_week", "month"]:
        assert col in out.columns


def test_pipeline_feature_names_aggregated() -> None:
    """feature_names must concatenate the names of every builder in order."""
    pipeline = FeaturePipeline(
        [
            LagFeatures(lags=[1, 7]),
            CalendarFeatures(features=["day_of_week"]),
        ]
    )
    assert pipeline.feature_names == ["lag_1", "lag_7", "day_of_week"]


def test_pipeline_rejects_empty_builders() -> None:
    """An empty builders list must raise at construction time."""
    with pytest.raises(ValueError, match="at least one builder"):
        FeaturePipeline(builders=[])


def test_pipeline_rejects_duplicate_feature_names() -> None:
    """Two builders producing the same column must fail at construction."""
    with pytest.raises(ValueError, match="Duplicate feature"):
        FeaturePipeline(
            [
                LagFeatures(lags=[1]),
                LagFeatures(lags=[1]),  # produces lag_1 again
            ]
        )


def test_pipeline_detects_broken_builder(shipments_df: pd.DataFrame) -> None:
    """If a builder declares but doesn't add a column, transform must raise."""
    pipeline = FeaturePipeline([BrokenBuilder()])
    with pytest.raises(RuntimeError, match="missing in output"):
        pipeline.transform(shipments_df)


def test_pipeline_validation_can_be_disabled(shipments_df: pd.DataFrame) -> None:
    """With validate_outputs=False, a broken builder is silently tolerated."""
    pipeline = FeaturePipeline([BrokenBuilder()], validate_outputs=False)
    # Should not raise
    out = pipeline.transform(shipments_df)
    # And the declared column is genuinely absent
    assert "this_column_will_never_appear" not in out.columns


def test_pipeline_preserves_original_columns(shipments_df: pd.DataFrame) -> None:
    """The output must keep all the original columns of the input."""
    pipeline = FeaturePipeline(
        [
            LagFeatures(lags=[1]),
            CalendarFeatures(features=["month"]),
        ]
    )
    out = pipeline.transform(shipments_df)

    for col in shipments_df.columns:
        assert col in out.columns


def test_pipeline_input_dataframe_not_mutated(shipments_df: pd.DataFrame) -> None:
    """The pipeline must not modify the input DataFrame in place."""
    original_cols = shipments_df.columns.tolist()
    original_shape = shipments_df.shape

    pipeline = FeaturePipeline(
        [
            LagFeatures(lags=[1]),
            CalendarFeatures(features=["month"]),
        ]
    )
    pipeline.transform(shipments_df)

    assert shipments_df.columns.tolist() == original_cols
    assert shipments_df.shape == original_shape


def test_pipeline_repr_mentions_builders() -> None:
    """__repr__ should list the builder class names for easy debugging."""
    pipeline = FeaturePipeline([LagFeatures(lags=[1]), CalendarFeatures(features=["month"])])
    r = repr(pipeline)
    assert "FeaturePipeline" in r
    assert "LagFeatures" in r
    assert "CalendarFeatures" in r
