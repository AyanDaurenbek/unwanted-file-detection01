"""Filesystem scanning utilities with safeguards."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from tqdm import tqdm

from .features import build_feature_frame, detect_potentially_dangerous
from .model import TrainedModel


SAFETY_NOTICE = (
    "Scanner never deletes files. Predictions are advisory and should be "
    "reviewed before any action is taken."
)


def iter_files(
    root: Path,
    exclude_hidden: bool = True,
    follow_symlinks: bool = False,
    max_files: Optional[int] = None,
) -> Iterable[Path]:
    """Yield file paths under *root* honoring basic safeguards."""
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if exclude_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            filenames = [f for f in filenames if not f.startswith(".")]
        for name in filenames:
            path = Path(dirpath) / name
            if not follow_symlinks and path.is_symlink():
                continue
            yield path
            count += 1
            if max_files and count >= max_files:
                return


def scan_directory(
    directory: str | Path,
    model: TrainedModel,
    threshold: Optional[float] = None,
    max_files: Optional[int] = None,
    exclude_hidden: bool = True,
) -> pd.DataFrame:
    """Scan a directory tree and produce predictions for contained files."""
    paths: List[Path] = list(
        tqdm(
            iter_files(Path(directory), exclude_hidden=exclude_hidden, max_files=max_files),
            desc="Scanning files",
        )
    )
    sizes = [p.stat().st_size for p in paths]
    features = build_feature_frame([str(p) for p in paths], sizes)
    probs = model.pipeline.predict_proba(features)[:, 1]
    use_threshold = threshold if threshold is not None else model.threshold
    preds = (probs >= use_threshold).astype(int)
    result = features.copy()
    result["probability"] = probs
    result["prediction"] = preds
    result["dangerous_flag"] = detect_potentially_dangerous(features)
    return result
