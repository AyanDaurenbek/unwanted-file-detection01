from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_RANDOM_SEED = 42
DEFAULT_N_ESTIMATORS = 300


@dataclass(frozen=True)
class AblationConfig:
    random_seed: int = DEFAULT_RANDOM_SEED
    n_estimators: int = DEFAULT_N_ESTIMATORS
    test_size: float = 0.2
    target_column: str = "label"
    id_like_columns: Tuple[str, ...] = (
        "id", "file_id", "hash", "sha256", "md5", "path", "full_path", "filepath", "filename", "name"
    )


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def infer_target_column(df: pd.DataFrame) -> str:
    candidates = ["label", "y", "target", "class", "is_unwanted", "is_malicious", "malicious"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Не найден столбец целевой переменной. "
        "Ожидалось одно из: label, y, target, class, is_unwanted, is_malicious, malicious"
    )


def drop_leaky_columns(df: pd.DataFrame, cfg: AblationConfig) -> pd.DataFrame:
    cols = set(df.columns)
    to_drop = []
    for c in cols:
        cl = c.lower()
        if cl in cfg.id_like_columns:
            to_drop.append(c)
        if cl.endswith("_id") or cl in {"uuid", "guid"}:
            to_drop.append(c)
    if to_drop:
        return df.drop(columns=sorted(set(to_drop)), errors="ignore")
    return df


def split_xy(
    df: pd.DataFrame,
    cfg: AblationConfig,
    explicit_target: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, str]:
    target = explicit_target or cfg.target_column
    if target not in df.columns:
        target = infer_target_column(df)

    y = df[target]
    X = df.drop(columns=[target])

    # нормализация target к 0/1
    if y.dtype == object:
        y = y.astype(str).str.lower().map({"0": 0, "1": 1, "false": 0, "true": 1, "normal": 0, "unwanted": 1})
        if y.isna().any():
            raise ValueError("Целевая переменная не приведена к 0/1. Проверь значения label.")
    y = y.astype(int)

    return X, y, target


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: AblationConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=cfg.test_size, random_state=cfg.random_seed
    )
    (train_idx, test_idx) = next(splitter.split(X, y))
    return X.iloc[train_idx].copy(), X.iloc[test_idx].copy(), y.iloc[train_idx].copy(), y.iloc[test_idx].copy()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(cfg: AblationConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_seed,
        n_jobs=-1,
        class_weight="balanced",
    )


def evaluate(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    precision = float(precision_score(y_test, pred, zero_division=0))
    recall = float(recall_score(y_test, pred, zero_division=0))
    f1 = float(f1_score(y_test, pred, zero_division=0))

    # roc_auc требует вариативности классов в y_test
    roc_auc = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def default_feature_groups(columns: List[str]) -> Dict[str, List[str]]:
    """
    Автогруппировка по семантике названий колонок.
    Если у тебя в проекте свои имена колонок, это все равно отработает,
    но при необходимости можно переопределить через JSON (см. ниже).
    """
    cols = list(columns)

    structural_patterns = [
        r"size", r"bytes", r"kb", r"mb",
        r"ext", r"extension",
        r"mime", r"filetype", r"type",
        r"entropy", r"magic",
        r"has_signature", r"pe_", r"elf_", r"header", r"sections",
        r"compression", r"packed",
    ]
    temporal_patterns = [
        r"created", r"modified", r"accessed", r"ctime", r"mtime", r"atime",
        r"timestamp", r"hour", r"day", r"month", r"weekday", r"week", r"year",
        r"age", r"delta",
    ]
    behavioral_patterns = [
        r"read_", r"write_", r"open_", r"exec_", r"delete_", r"rename_",
        r"access_count", r"op_count", r"ops_", r"activity", r"events", r"hits",
        r"fail", r"error", r"attempt",
    ]
    contextual_patterns = [
        r"path_", r"dir_", r"depth", r"folder",
        r"share", r"network", r"smb", r"nas",
        r"owner", r"group", r"acl", r"perm", r"permission",
        r"dept", r"project", r"host", r"machine",
    ]

    def pick(patterns: List[str]) -> List[str]:
        rx = re.compile("|".join(f"(?:{p})" for p in patterns), flags=re.IGNORECASE)
        return [c for c in cols if rx.search(c)]

    structural = pick(structural_patterns)
    temporal = pick(temporal_patterns)
    behavioral = pick(behavioral_patterns)
    contextual = pick(contextual_patterns)

    # Остальные признаки считаем нейтральными и включаем в All / NoContext автоматически.
    return {
        "ALL": cols,
        "STRUCTURAL_ONLY": sorted(set(structural)),
        "STRUCTURAL_TEMPORAL": sorted(set(structural + temporal)),
        "STRUCTURAL_TEMPORAL_BEHAVIORAL": sorted(set(structural + temporal + behavioral)),
        "ALL_WITHOUT_CONTEXT": sorted(set([c for c in cols if c not in set(contextual)])),
        "__meta__contextual_detected": sorted(set(contextual)),
    }


def load_feature_groups_override(path: Path) -> Dict[str, List[str]]:
    """
    Формат JSON:
    {
      "STRUCTURAL_ONLY": ["col1", "col2"],
      "STRUCTURAL_TEMPORAL": [...],
      "STRUCTURAL_TEMPORAL_BEHAVIORAL": [...],
      "ALL_WITHOUT_CONTEXT": [...],
      "ALL": [...]
    }
    """
    if not path.exists():
        raise FileNotFoundError(f"Override JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Override JSON must be an object")
    return {k: list(v) for k, v in data.items()}


def run_ablation(
    df: pd.DataFrame,
    cfg: AblationConfig,
    feature_groups_override: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    df = drop_leaky_columns(df, cfg)
    X, y, target = split_xy(df, cfg, explicit_target=cfg.target_column)

    # единый split для всех запусков
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, cfg)

    groups = feature_groups_override or default_feature_groups(list(X.columns))

    # Наборы по ТЗ
    requested_sets = [
        ("Все признаки", "ALL"),
        ("Только структурные признаки", "STRUCTURAL_ONLY"),
        ("Структурные + временные признаки", "STRUCTURAL_TEMPORAL"),
        ("Структурные + временные + поведенческие признаки", "STRUCTURAL_TEMPORAL_BEHAVIORAL"),
        ("Без контекстных признаков", "ALL_WITHOUT_CONTEXT"),
    ]

    rows = []
    for human_name, key in requested_sets:
        cols = groups.get(key, [])
        cols = [c for c in cols if c in X.columns]

        if len(cols) == 0:
            raise ValueError(
                f"Для набора {human_name} не найдено ни одного признака. "
                f"Проверь имена колонок или дай override JSON."
            )

        Xtr = X_train[cols].copy()
        Xte = X_test[cols].copy()

        pre = build_preprocessor(Xtr)
        clf = build_model(cfg)
        pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

        pipe.fit(Xtr, y_train)
        m = evaluate(pipe, Xte, y_test)

        rows.append(
            {
                "feature_set": human_name,
                "n_features_raw": int(len(cols)),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
                "random_seed": cfg.random_seed,
                "n_estimators": cfg.n_estimators,
                "target": target,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """
    Запуск:
      python -m src.experiments.ablation_features --data data/dataset.csv
    Опционально:
      --target label
      --override artifacts/feature_groups_override.json
      --out artifacts/ablation_results.csv
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to input CSV dataset")
    parser.add_argument("--target", type=str, default="label", help="Target column name")
    parser.add_argument("--override", type=str, default="", help="Path to override JSON for feature groups")
    parser.add_argument("--out", type=str, default="artifacts/ablation_results.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed")
    parser.add_argument("--n_estimators", type=int, default=DEFAULT_N_ESTIMATORS, help="RandomForest n_estimators")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size for split, e.g. 0.2")
    args = parser.parse_args()

    cfg = AblationConfig(
        random_seed=args.seed,
        n_estimators=args.n_estimators,
        test_size=args.test_size,
        target_column=args.target,
    )

    df = load_dataframe(Path(args.data))

    override = None
    if args.override:
        override = load_feature_groups_override(Path(args.override))

    results = run_ablation(df, cfg, feature_groups_override=override)

    out_path = Path(args.out)
    _safe_mkdir(out_path.parent)
    results.to_csv(out_path, index=False, encoding="utf-8")

    print("Ablation done")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
