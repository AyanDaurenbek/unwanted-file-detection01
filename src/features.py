"""Feature engineering for file paths and metadata."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SUSPICIOUS_TOKENS = (
    r"tmp", r"cache", r"backup", r"infected", r"virus", r"malware", r"\.exe$",
    r"\.bat$", r"\.scr$", r"\.pif$", r"\.vbs$", r"\.js$", r"\.jar$",
)


def _extract_path_parts(path_str: str) -> dict:
    path = Path(path_str)
    parts = path.parts
    name = path.name
    extension = path.suffix.lower().lstrip(".")
    depth = max(len(parts) - 1, 0)
    num_hidden = sum(1 for p in parts if p.startswith("."))
    suspicious_score = sum(bool(re.search(token, path_str, re.IGNORECASE)) for token in SUSPICIOUS_TOKENS)

    return {
        "path": path_str,
        "path_length": len(path_str),
        "depth": depth,
        "num_parts": len(parts),
        "num_hidden": num_hidden,
        "has_hidden": int(num_hidden > 0),
        "name_length": len(name),
        "extension": extension or "none",
        "extension_length": len(extension or "none"),
        "is_tmp": int("tmp" in parts or name.endswith("~")),
        "suspicious_score": suspicious_score,
    }


def build_feature_frame(paths: Iterable[str], sizes: Iterable[float] | None = None) -> pd.DataFrame:
    """Build a feature DataFrame for the provided file paths.

    Parameters
    ----------
    paths: iterable of str
        File paths to featurize.
    sizes: iterable of float, optional
        File sizes in bytes. When provided, they are included as a numeric feature.
    """

    base_rows = [_extract_path_parts(p) for p in paths]
    df = pd.DataFrame(base_rows)
    if sizes is not None:
        df["size"] = list(sizes)
    else:
        df["size"] = np.nan
    return df


def detect_potentially_dangerous(df: pd.DataFrame) -> pd.Series:
    """Heuristic flag for obviously risky files to aid scanning safeguards."""
    has_exec_ext = df["extension"].str.lower().isin(
        ["exe", "bat", "cmd", "scr", "pif", "vbs", "js", "jar", "msi", "dll"]
    )
    large_files = df["size"].fillna(0) > 100_000_000  # >100MB
    highly_suspicious = df["suspicious_score"] >= 2
    return (has_exec_ext | large_files | highly_suspicious).astype(int)
