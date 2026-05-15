"""Tune LightGBMForecaster hyperparameters with Optuna.

Implements the Phase 6 design decisions:

* **Objective**: WAPE averaged over folds 1-3. Fold 4 (final holdout) is
  reserved as a true out-of-sample test set and is NOT touched during
  tuning. Only the final, frozen best-params model touches fold 4.
* **Search space**: 8 hyperparameters covering capacity, regularisation,
  and stochasticity.
* **Loss + early stopping**: regression_l1 (MAE proxy for WAPE) with
  early stopping rounds=50 on the test fold of each CV split.
* **Sampler**: TPE (Tree-structured Parzen Estimator, Optuna default).
* **Pruner**: MedianPruner — kills trials that are worse than the median
  after the first fold, ~2x speedup without losing quality.
* **Storage**: SQLite at data/optuna_studies.db so studies are
  resumable across interruptions.

Usage:

    python scripts/tune_lightgbm.py --n-trials 100 --study-name lgbm_v1

    # Resume an interrupted study
    python scripts/tune_lightgbm.py --n-trials 100 --study-name lgbm_v1

    # Quick smoke test
    python scripts/tune_lightgbm.py --n-trials 5 --study-name smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from shipping_forecast.evaluation import time_series_split, wape
from shipping_forecast.models import LightGBMForecaster

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "shipping.db"
STORAGE_PATH = PROJECT_ROOT / "data" / "optuna_studies.db"
BEST_PARAMS_PATH = PROJECT_ROOT / "data" / "processed" / "best_lgbm_params.json"

# We use folds 1, 2, 3 for tuning. Fold 4 is the HOLDOUT and is NEVER
# touched during tuning — only by the final notebook in Phase 6.5.
TUNING_FOLDS_IDX = (0, 1, 2)

# Fixed seed for reproducibility of the TPE sampler.
SEED = 42

# Early stopping patience inside each fold's fit.
EARLY_STOPPING_ROUNDS = 50

# Logging setup.
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_panel(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load the daily shipments panel from SQLite."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"Shipping DB not found at {db_path}. Run scripts/load_data.py first."
        )
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT shipment_date, customer_state, n_shipments "
            "FROM fact_daily_shipments_by_state "
            "ORDER BY customer_state, shipment_date",
            conn,
            parse_dates=["shipment_date"],
        )
    return df


# ---------------------------------------------------------------------------
# Search space (8 hyperparameters per Phase 6.3 design)
# ---------------------------------------------------------------------------


def sample_params(trial: optuna.Trial) -> dict[str, object]:
    """Sample one hyperparameter configuration."""
    return {
        # Fixed across the search (set by design, not tuned).
        "objective": "regression_l1",
        "n_estimators": 1000,  # actual stopping point set by early_stopping
        "verbose": -1,
        "random_state": SEED,
        # Tuned hyperparameters.
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 5,  # fixed; tuning bagging_freq is low-yield
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 5, 7, 10]),
    }


# ---------------------------------------------------------------------------
# Per-fold evaluation
# ---------------------------------------------------------------------------


def evaluate_fold(params: dict[str, object], train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Train LightGBMForecaster on `train`, predict `test`, return WAPE.

    Uses `test` as the early-stopping validation set inside fit(). This is
    a justified shortcut for the inner loop of Optuna: we are NOT reporting
    early-stopping iteration as a hyperparameter; the test set leakage via
    early stopping is acceptable for tuning purposes because:

    * The leakage only affects the *iteration count*, not feature signal.
    * The holdout (fold 4) is untouched, so generalisation is still measured.
    * Without early stopping, every trial wastes ~30s on full 1000 trees.
    """
    horizon = test["shipment_date"].nunique()
    model = LightGBMForecaster(params=params)
    model.fit(train, eval_set=test, early_stopping_rounds=EARLY_STOPPING_ROUNDS)

    preds = model.predict(train, horizon=horizon)
    merged = test.merge(preds, on=["shipment_date", "customer_state"], how="inner")
    return float(wape(merged["n_shipments"].to_numpy(), merged["y_pred"].to_numpy()))


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def make_objective(folds: list, fold_indices: tuple[int, ...] = TUNING_FOLDS_IDX):
    """Build the Optuna objective function bound to a list of folds.

    The objective evaluates `params` on each fold in `fold_indices` and
    returns the mean WAPE. It also reports per-fold WAPE so MedianPruner
    can cut bad trials early.
    """

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial)
        wapes: list[float] = []

        for step, idx in enumerate(fold_indices):
            fold = folds[idx]
            fold_wape = evaluate_fold(params, fold.train, fold.test)
            wapes.append(fold_wape)

            # Report partial result so MedianPruner can intervene.
            trial.report(fold_wape, step=step)
            if trial.should_prune():
                logger.info(
                    "  trial %d pruned at fold %d (wape=%.4f)", trial.number, idx + 1, fold_wape
                )
                raise optuna.TrialPruned()

        return sum(wapes) / len(wapes)

    return objective


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune LightGBMForecaster with Optuna")
    parser.add_argument("--n-trials", type=int, default=100, help="number of Optuna trials")
    parser.add_argument(
        "--study-name",
        type=str,
        default="lgbm_v1",
        help="study name (allows resuming an interrupted study by reusing the same name)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="max seconds to run; overrides --n-trials if reached",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # less noisy than INFO

    logger.info("Loading panel from %s", DB_PATH)
    df = load_panel()
    folds = time_series_split(df)
    logger.info("Generated %d folds; using folds %s for tuning", len(folds), TUNING_FOLDS_IDX)
    for idx in TUNING_FOLDS_IDX:
        logger.info("  %s", folds[idx].label)
    logger.info("Fold 4 (holdout) RESERVED — never touched during tuning.")

    STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{STORAGE_PATH}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="minimize",
        sampler=TPESampler(seed=SEED, multivariate=True),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        load_if_exists=True,
    )

    existing = len(study.trials)
    if existing > 0:
        logger.info("Resuming study '%s' with %d existing trials", args.study_name, existing)

    objective = make_objective(folds)

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=True)
    elapsed = time.time() - t0

    # Summary stats
    n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    n_failed = sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials)
    logger.info("Study finished in %.1fs", elapsed)
    logger.info("  trials complete=%d, pruned=%d, failed=%d", n_complete, n_pruned, n_failed)
    logger.info("Best trial: #%d, WAPE=%.4f", study.best_trial.number, study.best_value)
    logger.info("Best params:")
    for k, v in study.best_params.items():
        logger.info("  %s = %r", k, v)

    # Persist best params for downstream notebook in Phase 6.5.
    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "study_name": args.study_name,
        "best_wape": study.best_value,
        "best_trial_number": study.best_trial.number,
        "n_trials_complete": n_complete,
        "n_trials_pruned": n_pruned,
        "best_params": study.best_params,
        "tuning_folds": list(TUNING_FOLDS_IDX),
        "elapsed_seconds": elapsed,
    }
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Best params written to %s", BEST_PARAMS_PATH)

    return 0


if __name__ == "__main__":
    sys.exit(main())
