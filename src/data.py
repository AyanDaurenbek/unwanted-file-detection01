"""Data loading utilities and label handling for the unwanted file detector."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

DEFAULT_POSITIVE = {"unwanted", "malicious", "1", 1, True, "true", "yes", "y"}
DEFAULT_NEGATIVE = {"wanted", "benign", "0", 0, False, "false", "no", "n"}


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset into a DataFrame with basic validation."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(path)
    if "path" not in df.columns:
        raise ValueError("Dataset must include a 'path' column")
    return df


def map_binary_labels(
    labels: Sequence,
    positive_labels: Iterable = DEFAULT_POSITIVE,
    negative_labels: Iterable = DEFAULT_NEGATIVE,
) -> pd.Series:
    """
    Binary label mapping for unwanted file detection.

    Rule:
    - benign -> 0 (normal)
    - any other label -> 1 (unwanted)

    Parameters positive_labels and negative_labels are kept
    for backward compatibility but are not used.
    """

    def _map(value: object) -> int:
        normalized = str(value).strip().lower()
        return 0 if normalized == "benign" else 1

    return pd.Series([_map(v) for v in labels], name="label")

