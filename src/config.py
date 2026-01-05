"""Configuration utilities for paths and defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class ProjectPaths:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    artifacts_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.artifacts_dir = self.root / "artifacts"
        self.reports_dir = self.root / "reports"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def ensure_dirs(self, extra: Iterable[Path] | None = None) -> None:
        """Ensure that common directories plus any provided extras exist."""
        for path in [self.artifacts_dir, self.reports_dir, *(extra or [])]:
            path.mkdir(parents=True, exist_ok=True)


def get_paths() -> ProjectPaths:
    """Helper to get configured project paths."""
    return ProjectPaths()
