"""Command line interface for the unwanted file detector."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from . import data, evaluate, features, model, preprocess, reporter, scanner
from .config import get_paths


@click.group()
def cli() -> None:
    """Tools for training, evaluating, and applying the unwanted file detector."""


@cli.command()
@click.argument("dataset", type=click.Path(exists=True, dir_okay=False))
@click.option("--model-out", type=click.Path(), default=None, help="Where to save the best model joblib file.")
@click.option("--test-size", type=float, default=0.2, show_default=True, help="Validation split size for model selection.")
@click.option("--positive-label", multiple=True, help="Label values treated as positive (unwanted)")
@click.option("--negative-label", multiple=True, help="Label values treated as negative (wanted)")
def train(dataset: str, model_out: Optional[str], test_size: float, positive_label: tuple[str], negative_label: tuple[str]) -> None:
    """Train candidate models on DATASET and save the best one."""
    paths = get_paths()
    df = data.load_dataset(dataset)
    X, y = preprocess.prepare_training_frame(
        df,
        positive_labels=positive_label or data.DEFAULT_POSITIVE,
        negative_labels=negative_label or data.DEFAULT_NEGATIVE,
    )

    click.echo("Training models...")
    trained = model.train_models(X, y, test_size=test_size)
    best = model.select_best_model(trained)
    out_path = Path(model_out) if model_out else paths.artifacts_dir / "best_model.joblib"
    best.save(str(out_path))
    click.echo(f"Saved best model ({best.name}) with F1={best.f1:.3f} to {out_path}")

    metrics_summary = {name: best.f1 for name, best in trained.items()}
    all_metrics = {
        name: evaluate.evaluate_model(trained_model, X, y) for name, trained_model in trained.items()
    }
    summary_df = evaluate.summarize_metrics(all_metrics)
    metrics_path = paths.reports_dir / "metrics.csv"
    summary_df.to_csv(metrics_path, index=False)
    click.echo(f"Metrics saved to {metrics_path}")

    eval_plots_dir = paths.reports_dir
    evaluate.save_confusion_matrix(best, X, y, eval_plots_dir / "confusion_matrix.png")
    evaluate.save_roc_curve(best, X, y, eval_plots_dir / "roc_curve.png")
    evaluate.save_precision_recall_curve(best, X, y, eval_plots_dir / "precision_recall_curve.png")
    click.echo(f"Evaluation plots saved under {eval_plots_dir}")


@cli.command()
@click.argument("model-path", type=click.Path(exists=True, dir_okay=False))
@click.argument("dataset", type=click.Path(exists=True, dir_okay=False))
def evaluate_model(model_path: str, dataset: str) -> None:
    """Evaluate an existing MODEL_PATH on DATASET."""
    paths = get_paths()
    clf = model.load_trained_model(model_path)
    df = data.load_dataset(dataset)
    X, y = preprocess.prepare_training_frame(df)
    metrics = evaluate.evaluate_model(clf, X, y)

    click.echo("Evaluation metrics:")
    for key, value in metrics.items():
        click.echo(f"- {key}: {value:.3f}")

    evaluate.save_confusion_matrix(clf, X, y, paths.reports_dir / "confusion_matrix.png")
    evaluate.save_roc_curve(clf, X, y, paths.reports_dir / "roc_curve.png")
    evaluate.save_precision_recall_curve(clf, X, y, paths.reports_dir / "precision_recall_curve.png")

    reporter.write_markdown_report(metrics, paths.reports_dir / "predictions.csv", paths.reports_dir / "evaluation.md")


@cli.command()
@click.argument("model-path", type=click.Path(exists=True, dir_okay=False))
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--threshold", type=float, default=None, help="Override decision threshold")
@click.option("--max-files", type=int, default=None, help="Limit number of files to scan")
@click.option("--exclude-hidden/--include-hidden", default=True, show_default=True)
@click.option("--output", type=click.Path(), default=None, help="CSV path for predictions")
def scan(model_path: str, directory: str, threshold: Optional[float], max_files: Optional[int], exclude_hidden: bool, output: Optional[str]) -> None:
    """Scan DIRECTORY using MODEL_PATH and write predictions to CSV."""
    paths = get_paths()
    clf = model.load_trained_model(model_path)
    click.echo(scanner.SAFETY_NOTICE)
    predictions = scanner.scan_directory(directory, clf, threshold=threshold, max_files=max_files, exclude_hidden=exclude_hidden)
    out_path = Path(output) if output else paths.reports_dir / "scan_predictions.csv"
    reporter.save_predictions_csv(predictions, out_path)
    click.echo(f"Predictions saved to {out_path}")


@cli.command()
@click.argument("model-path", type=click.Path(exists=True, dir_okay=False))
@click.argument("dataset", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", type=click.Path(), default=None, help="Where to save predictions CSV")
def predict(model_path: str, dataset: str, output: Optional[str]) -> None:
    """Run predictions on a CSV DATASET containing file paths (and optional sizes)."""
    paths = get_paths()
    clf = model.load_trained_model(model_path)
    df = data.load_dataset(dataset)
    sizes = df["size"] if "size" in df.columns else None
    feats = features.build_feature_frame(df["path"], sizes)
    probs = clf.pipeline.predict_proba(feats)[:, 1]
    preds = (probs >= clf.threshold).astype(int)
    result = feats.copy()
    result["probability"] = probs
    result["prediction"] = preds

    out_path = Path(output) if output else paths.reports_dir / "predictions.csv"
    reporter.save_predictions_csv(result, out_path)
    reporter.write_markdown_report({}, out_path, paths.reports_dir / "prediction_report.md")
    click.echo(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    cli()
