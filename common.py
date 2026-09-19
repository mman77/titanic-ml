"""Shared helpers for the Titanic pipeline (imported by 05-09).

Single source of truth for: paths, seed, feature engineering,
preprocessing, CV splitters, and baselines. This fixes the
train/serve skew and duplicated splits across scripts.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2

BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "titanic.csv"
DATA_CLEAN = BASE_DIR / "titanic_clean.csv"
MODEL_PATH = BASE_DIR / "titanic_best_model.pkl"
PARAMS_PATH = BASE_DIR / "best_params.json"
PLOTS_DIR = BASE_DIR / "plots"

# Raw columns expected from titanic_clean.csv
RAW = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]

ENGINEERED = [
    "family_size",
    "alone",
    "age_missing",
    "is_child",
    "fare_per_person",
    "age_x_pclass",
]

FEATURES = RAW + ENGINEERED

# Column groups: numeric (scaled), binary (passthrough, no scaling needed),
# categorical (one-hot; pclass is treated as categorical, not a number)
NUM = ["age", "sibsp", "parch", "fare", "family_size",
       "fare_per_person", "age_x_pclass"]
BIN = ["alone", "age_missing", "is_child"]
CAT = ["sex", "embarked", "pclass"]


def add_features(X):
    """Derive engineered features from RAW columns (no leakage: row-wise ops)."""
    X = X.copy()
    X["family_size"] = X["sibsp"] + X["parch"] + 1
    X["alone"] = (X["family_size"] == 1).astype(int)
    X["age_missing"] = X["age"].isna().astype(int)
    X["is_child"] = (X["age"] < 12).astype(int)  # NaN -> False (0), flagged by age_missing
    X["fare_per_person"] = X["fare"] / X["family_size"]
    X["age_x_pclass"] = X["age"] * X["pclass"]
    return X


def make_preprocess():
    """Fresh preprocessing objects on every call (never share fitted state)."""
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUM),
        ("bin", "passthrough", BIN),
        ("cat", cat_pipe, CAT),
    ])


def make_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def make_repeated_cv():
    return RepeatedStratifiedKFold(
        n_splits=5, n_repeats=3, random_state=RANDOM_STATE
    )


def split(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


class SexRuleClassifier(BaseEstimator, ClassifierMixin):
    """Baseline: predict survived iff passenger is female. CV-compatible."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return (np.asarray(X["sex"]) == "female").astype(int)

    def predict_proba(self, X):
        p = (np.asarray(X["sex"]) == "female").astype(float)
        return np.column_stack([1 - p, p])
