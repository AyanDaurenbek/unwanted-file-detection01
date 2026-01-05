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
    """Map assorted label representations to a clean binary format of 0/1.

    Any value contained in *positive_labels* is mapped to 1, any value in
    *negative_labels* is mapped to 0. Values not found in either set raise
    a ValueError to guard against silent mislabeling.
    """

    pos_set = {str(v).lower() for v in positive_labels}
    neg_set = {str(v).lower() for v in negative_labels}

    def _map(value: object) -> int:
        normalized = str(value).lower()
        if normalized in pos_set:
            return 1
        if normalized in neg_set:
            return 0
        raise ValueError(f"Label {value!r} not recognized as positive or negative")

    return pd.Series([_map(v) for v in labels], name="label")
