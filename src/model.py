"""Model training helpers using scikit-learn pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    threshold: float
    f1: float

    def save(self, path: str) -> None:
        joblib.dump(self, path)


def to_2d_array(x):
    """
    Helper for sklearn FunctionTransformer: converts Series/array-like to 2D numpy array.
    Must be defined at module top-level to be picklable by joblib.
    """
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def load_trained_model(path: str) -> TrainedModel:
    """Load a persisted :class:`TrainedModel` from disk."""
    return joblib.load(path)


NUMERIC_FEATURES = [
    "path_length",
    "depth",
    "num_parts",
    "num_hidden",
    "has_hidden",
    "name_length",
    "extension_length",
    "is_tmp",
    "suspicious_score",
    "size",
]

def select_path_column(df):
    # Возвращаем Series с путями (1D). TfidfVectorizer это ожидает.
    return df["path"]

def select_numeric_columns(df):
    # Возвращаем DataFrame с числовыми признаками
    return df[NUMERIC_FEATURES]


# def _build_preprocessor() -> ColumnTransformer:
#     text_pipeline = Pipeline(
#         steps=[
#             ("selector", FunctionTransformer(lambda df: df["path"], validate=False)),
#             (
#                 "tfidf",
#                 TfidfVectorizer(
#                     analyzer="char_wb",
#                     ngram_range=(3, 5),
#                     min_df=2,
#                     max_features=20000,
#                 ),
#             ),
#         ]
#     )
#     numeric_pipeline = Pipeline(
#         steps=[
#             ("selector", FunctionTransformer(lambda df: df[NUMERIC_FEATURES], validate=False)),
#             ("imputer", SimpleImputer(strategy="median")),
#             ("scaler", StandardScaler()),
#         ]
#     )
#     return ColumnTransformer(
#         transformers=[
#             ("text", text_pipeline, ["path"]),
#             ("num", numeric_pipeline, NUMERIC_FEATURES),
#         ]
#     )

def _build_preprocessor() -> ColumnTransformer:
    text_pipeline = Pipeline(
        steps=[
            ("selector", FunctionTransformer(select_path_column, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=20000,
                ),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("selector", FunctionTransformer(select_numeric_columns, validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("text", text_pipeline, ["path"]),
            ("num", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )


def _candidate_models() -> Dict[str, Pipeline]:
    preprocessor = _build_preprocessor()

    logistic = Pipeline(
        steps=[
            ("features", preprocessor),
            (
                "clf",
                LogisticRegression(max_iter=500, n_jobs=4, class_weight="balanced"),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("features", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    n_jobs=4,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    gradient_boosting = Pipeline(
        steps=[
            ("features", preprocessor),
            ("clf", GradientBoostingClassifier()),
        ]
    )

    return {
        "logistic_regression": logistic,
        "random_forest": random_forest,
        "gradient_boosting": gradient_boosting,
    }


def tune_threshold(probs: np.ndarray, y_true: pd.Series) -> Tuple[float, float]:
    """Find the probability threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    thresholds = np.append(thresholds, 1.0)  # align lengths
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1_scores))
    return thresholds[best_idx], f1_scores[best_idx]


def train_models(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Dict[str, TrainedModel]:
    """Train all candidate models and return them with tuned thresholds."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    trained: Dict[str, TrainedModel] = {}
    for name, pipeline in _candidate_models().items():
        pipeline.fit(X_train, y_train)
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_val)[:, 1]
        else:
            # fall back to decision function scaled to [0,1]
            decision = pipeline.decision_function(X_val)
            probs = (decision - decision.min()) / (decision.max() - decision.min() + 1e-8)
        threshold, f1 = tune_threshold(probs, y_val)
        trained[name] = TrainedModel(name=name, pipeline=pipeline, threshold=threshold, f1=f1)
    return trained


def select_best_model(models: Dict[str, TrainedModel]) -> TrainedModel:
    """Return the model with the highest F1 score."""
    return max(models.values(), key=lambda m: m.f1)
