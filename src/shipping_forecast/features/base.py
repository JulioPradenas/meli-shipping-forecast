"""Base class for feature builders.

A :class:`FeatureBuilder` is any class that takes a time-series DataFrame
and returns a new DataFrame with one or more additional columns. Builders
are designed to be:

* **Stateless**: no internal state between calls; reproducible.
* **Non-mutating**: never modify the input DataFrame in place.
* **Composable**: can be chained via :class:`FeaturePipeline`.
* **Introspectable**: expose the names of the columns they add.

Subclasses must implement :meth:`transform` and :attr:`feature_names`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class FeatureBuilder(ABC):
    """Abstract base for any feature engineering step on a time-series DataFrame.

    Each concrete builder is responsible for a single coherent family of
    features (lags, rolling stats, calendar, events, etc.). This keeps the
    code aligned with the single-responsibility principle and makes the
    pipeline composable.

    Subclasses must override :meth:`transform` (which adds columns to the
    DataFrame) and :attr:`feature_names` (which lists the names of those
    columns without executing the transformation).

    Example:
        >>> class ConstantFeature(FeatureBuilder):
        ...     def __init__(self, name: str, value: float):
        ...         self.name = name
        ...         self.value = value
        ...
        ...     @property
        ...     def feature_names(self) -> list[str]:
        ...         return [self.name]
        ...
        ...     def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...         out = df.copy()
        ...         out[self.name] = self.value
        ...         return out
    """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame with this builder's features added.

        Args:
            df: The input DataFrame. Must contain at least the columns this
                builder depends on (declared in subclass docstrings).

        Returns:
            A new DataFrame containing all original columns plus the columns
            listed in :attr:`feature_names`. The input is not modified.
        """

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """List of column names this builder adds to the DataFrame.

        Returns:
            A list of strings, one per added column. The order should match
            the order in which columns are added in :meth:`transform`.
        """
