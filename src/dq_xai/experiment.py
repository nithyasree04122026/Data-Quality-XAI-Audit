"""Reproducible, leakage-aware benchmark experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Config:
    seed: int = 42
    missing_rate: float = 0.0
    label_flip_rate: float = 0.0
    repeats: int = 3

    def validate(self) -> None:
        if not 0 <= self.missing_rate <= 1:
            raise ValueError("missing_rate must be in [0, 1]")
        if not 0 <= self.label_flip_rate <= 1:
            raise ValueError("label_flip_rate must be in [0, 1]")
        if self.repeats < 1:
            raise ValueError("repeats must be positive")


def split_data(seed: int):
    data = load_breast_cancer()
    x = data.data.copy()
    y = data.target.copy()
    indices = np.arange(len(y))
    train_idx, held_idx = train_test_split(
        indices, test_size=0.4, random_state=seed, stratify=y
    )
    val_idx, test_idx = train_test_split(
        held_idx, test_size=0.5, random_state=seed + 1, stratify=y[held_idx]
    )
    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx]), (x[test_idx], y[test_idx])


def corrupt_training(x: np.ndarray, y: np.ndarray, config: Config):
    """Return copies; neither validation nor test arrays are passed here."""
    config.validate()
    rng = np.random.default_rng(config.seed)
    damaged_x = x.copy()
    damaged_y = y.copy()
    missing_mask = rng.random(damaged_x.shape) < config.missing_rate
    damaged_x[missing_mask] = np.nan
    n_flip = int(round(len(y) * config.label_flip_rate))
    if n_flip:
        flip_idx = rng.choice(len(y), size=n_flip, replace=False)
        damaged_y[flip_idx] = 1 - damaged_y[flip_idx]
    return damaged_x, damaged_y, int(missing_mask.sum()), n_flip


def _model(seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=seed),
    )


def _importance(model, x_val, y_val, seed: int, repeats: int):
    result = permutation_importance(
        model, x_val, y_val, scoring="roc_auc", n_repeats=repeats, random_state=seed
    )
    return result.importances_mean


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of stable ordinal ranks; NaN when rank variance is zero."""
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def run(config: Config) -> dict:
    config.validate()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_data(config.seed)
    baseline = _model(config.seed).fit(x_train, y_train)
    baseline_importance = _importance(baseline, x_val, y_val, config.seed, config.repeats)
    x_damaged, y_damaged, missing_count, flip_count = corrupt_training(
        x_train, y_train, config
    )
    model = _model(config.seed).fit(x_damaged, y_damaged)
    damaged_importance = _importance(model, x_val, y_val, config.seed, config.repeats)
    proba = model.predict_proba(x_test)[:, 1]
    return {
        **asdict(config),
        "dataset": "sklearn breast_cancer (UCI Wisconsin Diagnostic Breast Cancer)",
        "train_rows": len(y_train),
        "validation_rows": len(y_val),
        "test_rows": len(y_test),
        "missing_cells": missing_count,
        "flipped_labels": flip_count,
        "test_accuracy": float(accuracy_score(y_test, model.predict(x_test))),
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
        "validation_explanation_rank_correlation": rank_correlation(
            baseline_importance, damaged_importance
        ),
    }
