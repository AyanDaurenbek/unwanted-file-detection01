"""Preprocessing steps combining feature engineering and label handling."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .data import map_binary_labels, DEFAULT_NEGATIVE, DEFAULT_POSITIVE
from .features import build_feature_frame


def prepare_training_frame(
    df: pd.DataFrame,
    positive_labels: Iterable = DEFAULT_POSITIVE,
    negative_labels: Iterable = DEFAULT_NEGATIVE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Convert a raw dataset into feature and label components."""
    if "label" not in df.columns:
        raise ValueError("Training data must include a 'label' column")
    if "path" not in df.columns:
        raise ValueError("Training data must include a 'path' column")

    # 1) Binarize labels: benign -> 0, else -> 1 (твоя правка в data.py)
    y = map_binary_labels(df["label"], positive_labels, negative_labels)

    # 2) Provide "size" in bytes for the pipeline.
    # Dataset may contain size_kb instead of size.
    if "size" in df.columns:
        sizes = df["size"]
    elif "size_kb" in df.columns:
        sizes = df["size_kb"] * 1024.0
    else:
        sizes = None

    # 3) Build feature frame from path + size
    X = build_feature_frame(df["path"], sizes)
    return X, y
