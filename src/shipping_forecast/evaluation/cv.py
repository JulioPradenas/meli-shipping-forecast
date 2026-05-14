"""Time series cross-validation with expanding window.

Implements the validation strategy decided for this project:

* **Expanding window**: each fold trains on everything available up to
  the test start date. Reflects how a real production system would
  retrain over time.
* **4 folds**: each test period covers a distinct business context
  (Black Friday, steady state, Dia dos Namorados, final holdout).
* **No data leakage**: the test set always starts strictly after the
  train set ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class Fold:
    """A single train/test split.

    Attributes:
        fold_id: 1-indexed identifier of this fold.
        train: Training DataFrame.
        test: Test DataFrame.
        train_period: Inclusive (start, end) dates of the training data.
        test_period: Inclusive (start, end) dates of the test data.
        label: Short human description (e.g. "fold 1: covers Black Friday").
    """

    fold_id: int
    train: pd.DataFrame
    test: pd.DataFrame
    train_period: tuple[date, date]
    test_period: tuple[date, date]
    label: str

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)


# The 4 folds for the Olist dataset, defined per the project design.
DEFAULT_FOLDS: list[dict] = [
    {
        "train_end": date(2017, 10, 31),
        "test_start": date(2017, 11, 1),
        "test_end": date(2018, 1, 31),
        "label": "fold 1: covers Black Friday 2017",
    },
    {
        "train_end": date(2018, 1, 31),
        "test_start": date(2018, 2, 1),
        "test_end": date(2018, 4, 30),
        "label": "fold 2: steady state post-BF",
    },
    {
        "train_end": date(2018, 4, 30),
        "test_start": date(2018, 5, 1),
        "test_end": date(2018, 6, 30),
        "label": "fold 3: covers Dia dos Namorados",
    },
    {
        "train_end": date(2018, 6, 30),
        "test_start": date(2018, 7, 1),
        "test_end": date(2018, 8, 31),
        "label": "fold 4: final holdout",
    },
]


def time_series_split(
    df: pd.DataFrame,
    date_col: str = "shipment_date",
    train_start: date | None = None,
    folds_spec: list[dict] | None = None,
) -> list[Fold]:
    """Build expanding-window folds for time series validation.

    Args:
        df: DataFrame containing the date column and any other data.
        date_col: Name of the column containing dates.
        train_start: First date to include in training. Defaults to the
            minimum date in ``df``.
        folds_spec: List of dicts describing each fold. Each dict must have
            keys ``train_end``, ``test_start``, ``test_end``, ``label``.
            Defaults to :data:`DEFAULT_FOLDS`.

    Returns:
        List of :class:`Fold` objects, one per spec entry.

    Raises:
        ValueError: If a fold has zero training or test rows, or if the
            specified date column is not in the DataFrame.
    """
    if date_col not in df.columns:
        raise KeyError(f"date_col not in DataFrame: {date_col!r}")

    spec = folds_spec if folds_spec is not None else DEFAULT_FOLDS
    dates = pd.to_datetime(df[date_col]).dt.date

    if train_start is None:
        train_start = dates.min()

    folds: list[Fold] = []
    for i, fold_spec in enumerate(spec, start=1):
        train_end: date = fold_spec["train_end"]
        test_start: date = fold_spec["test_start"]
        test_end: date = fold_spec["test_end"]
        label: str = fold_spec["label"]

        if test_start <= train_end:
            raise ValueError(
                f"Fold {i}: test_start ({test_start}) must be strictly "
                f"after train_end ({train_end})"
            )

        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask = (dates >= test_start) & (dates <= test_end)

        train_df = df.loc[train_mask].reset_index(drop=True)
        test_df = df.loc[test_mask].reset_index(drop=True)

        if train_df.empty:
            raise ValueError(
                f"Fold {i} has empty training set "
                f"(train_start={train_start}, train_end={train_end})"
            )
        if test_df.empty:
            raise ValueError(
                f"Fold {i} has empty test set (test_start={test_start}, test_end={test_end})"
            )

        folds.append(
            Fold(
                fold_id=i,
                train=train_df,
                test=test_df,
                train_period=(train_start, train_end),
                test_period=(test_start, test_end),
                label=label,
            )
        )

    return folds
