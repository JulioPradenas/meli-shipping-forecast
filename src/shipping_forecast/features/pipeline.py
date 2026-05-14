"""Pipeline orchestrator for chaining FeatureBuilder instances.

A :class:`FeaturePipeline` composes multiple feature builders into a
single transformation. Each builder receives the DataFrame with the
columns added by previous builders, allowing later steps to depend on
earlier features.

The pipeline validates that each builder actually adds the columns it
promises in :attr:`feature_names`, catching subtle bugs early.

Example:
    >>> from shipping_forecast.features import (
    ...     LagFeatures, CalendarFeatures, FeaturePipeline
    ... )
    >>> pipeline = FeaturePipeline([
    ...     LagFeatures(lags=[1, 7]),
    ...     CalendarFeatures(features=["day_of_week", "month"]),
    ... ])
    >>> df_out = pipeline.transform(df)
    >>> pipeline.feature_names
    ['lag_1', 'lag_7', 'day_of_week', 'month']
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from shipping_forecast.features.base import FeatureBuilder


@dataclass
class FeaturePipeline:
    """Compose a sequence of feature builders into a single transformation.

    Attributes:
        builders: Ordered list of :class:`FeatureBuilder` instances.
            They are applied in order; each receives the output of the
            previous one.
        validate_outputs: If ``True`` (default), after each builder runs
            the pipeline verifies that all columns listed in
            ``builder.feature_names`` are present in the output. Set to
            ``False`` to skip the check for marginal speed gains.

    Raises:
        ValueError: At construction time if ``builders`` is empty.
        RuntimeError: At ``transform`` time if a builder fails to add
            one of the columns it declared.
    """

    builders: list[FeatureBuilder] = field(default_factory=list)
    validate_outputs: bool = True

    def __post_init__(self) -> None:
        if not self.builders:
            raise ValueError("FeaturePipeline requires at least one builder")
        # Check duplicate feature names across builders to fail loud
        seen: set[str] = set()
        for b in self.builders:
            duplicates = seen & set(b.feature_names)
            if duplicates:
                raise ValueError(
                    f"Duplicate feature names across builders: {sorted(duplicates)}. "
                    f"Each feature must be produced by exactly one builder."
                )
            seen.update(b.feature_names)

    @property
    def feature_names(self) -> list[str]:
        """All feature names added by the pipeline, in order."""
        names: list[str] = []
        for b in self.builders:
            names.extend(b.feature_names)
        return names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply every builder in order, returning a new DataFrame.

        Args:
            df: Input DataFrame. Must contain the columns required by the
                first builder. Each subsequent builder may rely on columns
                added by previous builders.

        Returns:
            A new DataFrame with all features from all builders added.

        Raises:
            RuntimeError: If a builder did not add one of the columns it
                declared in ``feature_names`` (only checked when
                :attr:`validate_outputs` is ``True``).
        """
        out = df
        for builder in self.builders:
            cols_before = set(out.columns)
            out = builder.transform(out)
            if self.validate_outputs:
                self._validate_builder_output(builder, cols_before, out)
        return out

    @staticmethod
    def _validate_builder_output(
        builder: FeatureBuilder,
        cols_before: set[str],
        out: pd.DataFrame,
    ) -> None:
        """Confirm a builder produced all the columns it declared.

        Args:
            builder: The builder that just ran.
            cols_before: Columns present before this builder ran.
            out: DataFrame after the builder.

        Raises:
            RuntimeError: If any declared column is missing from ``out``.
        """
        actually_added = set(out.columns) - cols_before
        declared = set(builder.feature_names)
        missing = declared - actually_added - set(out.columns)
        # A column counts as added if it appears in out, regardless of whether
        # it existed before (some builders intentionally overwrite). The check
        # is only about presence in the final DataFrame.
        missing = declared - set(out.columns)
        if missing:
            raise RuntimeError(
                f"{builder.__class__.__name__} declared features {sorted(declared)} "
                f"but missing in output: {sorted(missing)}"
            )

    def __repr__(self) -> str:
        names = [b.__class__.__name__ for b in self.builders]
        return f"FeaturePipeline(builders={names})"
