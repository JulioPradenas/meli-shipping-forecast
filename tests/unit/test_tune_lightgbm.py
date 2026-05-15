"""Tests for scripts/tune_lightgbm.py.

The script orchestrates an Optuna study; these tests focus on the
*structural* correctness of its building blocks rather than the
empirical quality of the resulting study (which is data-dependent and
non-deterministic across runs).

Coverage:

* Module imports cleanly (no syntax errors, no missing deps).
* sample_params returns the expected hyperparameter keys in valid ranges.
* evaluate_fold returns a finite, non-negative WAPE.
* make_objective produces a callable that returns a float.
* TUNING_FOLDS_IDX excludes fold 4 (holdout protection).
* load_panel raises a clear error when the DB is missing.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def tune_module():
    """Import scripts/tune_lightgbm.py as a module for testing."""
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts" / "tune_lightgbm.py"
    spec = importlib.util.spec_from_file_location("tune_lightgbm", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tune_lightgbm"] = module
    spec.loader.exec_module(module)
    return module


def _make_panel(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    """Small synthetic panel with weekly seasonality for fast tests."""
    rng = np.random.default_rng(seed)
    start = date(2017, 1, 1)
    states = ["SP", "RJ", "AC"]
    base = {"SP": 100.0, "RJ": 20.0, "AC": 1.0}
    rows = []
    for i in range(n_days):
        d = start + timedelta(days=i)
        wd = d.weekday()
        for st in states:
            b = base[st]
            n = 0 if wd == 6 else max(0, int(rng.normal(b * (0.08 if wd == 5 else 1.0), b * 0.2)))
            rows.append({"shipment_date": pd.Timestamp(d), "customer_state": st, "n_shipments": n})
    return pd.DataFrame(rows)


@pytest.fixture
def small_panel() -> pd.DataFrame:
    return _make_panel(n_days=200)


def test_module_imports_cleanly(tune_module):
    """The script must import without errors; ensures no dead imports."""
    assert hasattr(tune_module, "main")
    assert hasattr(tune_module, "sample_params")
    assert hasattr(tune_module, "evaluate_fold")
    assert hasattr(tune_module, "make_objective")
    assert hasattr(tune_module, "load_panel")


def test_tuning_folds_excludes_holdout(tune_module):
    """Fold 4 (index 3) MUST NOT be in the tuning folds; it's the holdout."""
    assert 3 not in tune_module.TUNING_FOLDS_IDX
    assert tune_module.TUNING_FOLDS_IDX == (0, 1, 2)


def test_seed_is_fixed(tune_module):
    """A fixed seed makes the TPE sampler reproducible."""
    assert isinstance(tune_module.SEED, int)
    assert tune_module.SEED == 42


def test_sample_params_has_expected_keys(tune_module):
    """Lock the search space: 8 tuned + 5 fixed keys."""
    import optuna

    study = optuna.create_study()
    trial = study.ask()
    params = tune_module.sample_params(trial)

    tuned = {
        "num_leaves",
        "learning_rate",
        "feature_fraction",
        "bagging_fraction",
        "min_child_samples",
        "reg_alpha",
        "reg_lambda",
        "max_depth",
    }
    fixed = {"objective", "n_estimators", "verbose", "random_state", "bagging_freq"}

    assert tuned <= set(params.keys())
    assert fixed <= set(params.keys())


def test_sample_params_values_in_valid_ranges(tune_module):
    """Sampled values respect the documented bounds."""
    import optuna

    study = optuna.create_study()
    trial = study.ask()
    p = tune_module.sample_params(trial)

    assert 15 <= p["num_leaves"] <= 127
    assert 0.01 <= p["learning_rate"] <= 0.3
    assert 0.5 <= p["feature_fraction"] <= 1.0
    assert 0.5 <= p["bagging_fraction"] <= 1.0
    assert 5 <= p["min_child_samples"] <= 100
    assert 1e-3 <= p["reg_alpha"] <= 10.0
    assert 1e-3 <= p["reg_lambda"] <= 10.0
    assert p["max_depth"] in {-1, 5, 7, 10}
    assert p["objective"] == "regression_l1"
    assert p["random_state"] == tune_module.SEED


def test_evaluate_fold_returns_finite_non_negative_wape(small_panel, tune_module):
    """The function returns a float in [0, inf)."""
    train = small_panel[small_panel["shipment_date"] < pd.Timestamp("2017-06-01")]
    test = small_panel[
        (small_panel["shipment_date"] >= pd.Timestamp("2017-06-01"))
        & (small_panel["shipment_date"] < pd.Timestamp("2017-06-15"))
    ]
    params = {
        "objective": "regression_l1",
        "n_estimators": 30,
        "verbose": -1,
        "random_state": 42,
        "num_leaves": 15,
        "learning_rate": 0.1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "max_depth": -1,
    }
    w = tune_module.evaluate_fold(params, train, test)
    assert isinstance(w, float)
    assert math.isfinite(w)
    assert w >= 0


def test_make_objective_is_callable_and_returns_float(small_panel, tune_module):
    """The objective produced by make_objective runs end-to-end on synthetic folds."""
    import optuna

    from shipping_forecast.evaluation import Fold

    cut1 = pd.Timestamp("2017-04-01")
    cut2 = pd.Timestamp("2017-05-15")
    cut3 = pd.Timestamp("2017-06-30")
    folds = [
        Fold(
            fold_id=1,
            train=small_panel[small_panel["shipment_date"] < cut1],
            test=small_panel[
                (small_panel["shipment_date"] >= cut1) & (small_panel["shipment_date"] < cut2)
            ],
            train_period=(date(2017, 1, 1), cut1.date()),
            test_period=(cut1.date(), cut2.date()),
            label="synthetic 1",
        ),
        Fold(
            fold_id=2,
            train=small_panel[small_panel["shipment_date"] < cut2],
            test=small_panel[
                (small_panel["shipment_date"] >= cut2) & (small_panel["shipment_date"] < cut3)
            ],
            train_period=(date(2017, 1, 1), cut2.date()),
            test_period=(cut2.date(), cut3.date()),
            label="synthetic 2",
        ),
    ]

    objective = tune_module.make_objective(folds, fold_indices=(0,))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1, show_progress_bar=False)

    assert study.best_value is not None
    assert math.isfinite(study.best_value)
    assert study.best_value >= 0


def test_load_panel_raises_clear_error_when_db_missing(tune_module, tmp_path):
    """If the SQLite DB doesn't exist, the error must explain how to fix it."""
    bogus = tmp_path / "nonexistent.db"
    with pytest.raises(FileNotFoundError, match=r"Run scripts/load_data\.py"):
        tune_module.load_panel(db_path=bogus)
