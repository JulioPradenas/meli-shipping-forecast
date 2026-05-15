"""Volume-based features for cross-state regime differentiation.

The Phase 5 baseline analysis revealed that prediction error is dominated
by volume, not structural difficulty: Spearman(volume, WAPE) = -0.867.
Three operational clusters emerged:

* ``core`` (>10 shipments/day): SP, RJ, MG, PR, RS, SC. Reliably modelable.
* ``mid`` (2-10 shipments/day): BA, DF, GO, ES, PE, CE. Noisy but useful.
* ``tail`` (<2 shipments/day): AC, AP, RR, AL, RN, MA. Poisson noise dominates.

This builder exposes that regime structure to LightGBM via two features:

* ``state_avg_volume``: mean daily shipments of the state, from training data.
* ``volume_tier``: categorical bucket in ``{'core', 'mid', 'tail'}``.

**Crucial design choice**: the stats are *injected* at construction time
rather than computed inside :meth:`transform`. This makes the builder
stateless (consistent with the rest of the pipeline) and forces the
caller (typically ``prepare_fold_data``) to compute stats on the training
set only, preventing temporal leakage from test data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from shipping_forecast.features.base import FeatureBuilder

# Thresholds derived from Phase 5 per-state WAPE analysis (fold 4, holdout).
# See notebooks/MODELS_BASELINE_SUMMARY.md for the rationale.
DEFAULT_TIER_THRESHOLDS: tuple[float, float] = (2.0, 10.0)


@dataclass
class VolumeFeatures(FeatureBuilder):
    """Add per-state volume regime features from pre-computed statistics.

    Attributes:
        state_avg_volume: Mapping ``state -> mean daily shipments`` computed
            on the training set only. The caller is responsible for ensuring
            this dict was built without leakage from test data.
        group_col: Column identifying the group (default ``customer_state``).
        thresholds: Two cutoffs ``(tail_max, core_min)`` defining the tiers.
            Defaults to ``(2.0, 10.0)``: ``<2 = tail``, ``2-10 = mid``,
            ``>10 = core``. Tunable for sensitivity analysis.
        unknown_state_fallback: Value assigned to ``state_avg_volume`` when
            a state in the input was not present in ``state_avg_volume``.
            Defaults to 0.0 (treated as long tail). Should never trigger
            in practice if the training set covers all 27 states.

    Raises:
        ValueError: If ``state_avg_volume`` is empty or ``thresholds`` are
            not strictly increasing.

    Example:
        >>> import pandas as pd
        >>> stats = {"SP": 98.8, "RJ": 24.2, "AC": 0.11}
        >>> vf = VolumeFeatures(state_avg_volume=stats)
        >>> df = pd.DataFrame({
        ...     "customer_state": ["SP", "RJ", "AC"],
        ...     "n_shipments": [120, 25, 0],
        ... })
        >>> out = vf.transform(df)
        >>> out["volume_tier"].tolist()
        ['core', 'core', 'tail']
    """

    state_avg_volume: dict[str, float] = field(default_factory=dict)
    group_col: str = "customer_state"
    thresholds: tuple[float, float] = DEFAULT_TIER_THRESHOLDS
    unknown_state_fallback: float = 0.0

    def __post_init__(self) -> None:
        if not self.state_avg_volume:
            raise ValueError(
                "state_avg_volume cannot be empty. Compute it from the "
                "training set before constructing VolumeFeatures."
            )
        tail_max, core_min = self.thresholds
        if not (tail_max < core_min):
            raise ValueError(f"thresholds must be strictly increasing, got {self.thresholds}")

    @property
    def feature_names(self) -> list[str]:
        return ["state_avg_volume", "volume_tier"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame with volume regime features added."""
        out = df.copy()

        avg = out[self.group_col].map(self.state_avg_volume).fillna(self.unknown_state_fallback)
        out["state_avg_volume"] = avg.astype("float32")

        tail_max, core_min = self.thresholds
        tier = pd.Series("mid", index=out.index, dtype="object")
        tier[avg < tail_max] = "tail"
        tier[avg >= core_min] = "core"
        out["volume_tier"] = tier.astype(
            pd.CategoricalDtype(categories=["tail", "mid", "core"], ordered=True)
        )

        return out

    @staticmethod
    def compute_stats_from_train(
        train_df: pd.DataFrame,
        group_col: str = "customer_state",
        target_col: str = "n_shipments",
    ) -> dict[str, float]:
        """Compute ``state -> mean daily shipments`` from a training set.

        Args:
            train_df: Training DataFrame. Must contain ``group_col`` and
                ``target_col``.
            group_col: Column with the state identifier.
            target_col: Column with the daily shipment count.

        Returns:
            Dict mapping each state to its mean daily volume in the training set.

        Note:
            This is a convenience helper for the typical case. Callers can
            also pass a manually-computed dict to the constructor — e.g.
            to use median instead of mean, or to compute stats on
            operational days only.
        """
        result = train_df.groupby(group_col)[target_col].mean().to_dict()
        return {str(k): float(v) for k, v in result.items()}
