"""Reporting utilities for predictions and evaluations."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

from .evaluate import METRIC_COLUMNS


def save_predictions_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_markdown_report(
    summary: Dict[str, float],
    predictions_path: Path,
    output_path: Path,
    extra_notes: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Unwanted File Detection Report", ""]
    lines.append(f"Generated: {datetime.utcnow().isoformat()} UTC")
    lines.append("")
    if summary:
        lines.append("## Evaluation Metrics")
        for key in METRIC_COLUMNS:
            if key in summary:
                lines.append(f"- **{key.title()}**: {summary[key]:.3f}")
        lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- Predictions CSV: `{predictions_path}`")
    if extra_notes:
        lines.append("")
        lines.append("## Notes")
        lines.append(extra_notes)
    output_path.write_text("\n".join(lines))
    return output_path
