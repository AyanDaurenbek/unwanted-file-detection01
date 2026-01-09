# Unwanted File Detector

A reference project for training and applying a machine-learning model to flag potentially unwanted files based on their paths and metadata. The CLI supports training, evaluation, scanning filesystem trees, and predicting labels for CSV inputs.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout

- `src/`: Core Python modules
  - `cli.py`: Entry point for the CLI commands
  - `config.py`: Common path helpers
  - `data.py`: Dataset loading and binary label mapping utilities
  - `preprocess.py`: Wiring for labels and feature construction
  - `features.py`: Feature engineering from file paths and optional sizes
  - `model.py`: Model pipelines, training, and threshold tuning
  - `evaluate.py`: Metrics and plot helpers
  - `scanner.py`: Filesystem walker and predictor with safeguards
  - `reporter.py`: CSV and Markdown report generation
- `artifacts/`: Saved models and intermediate artifacts
- `reports/`: Metrics, plots, and prediction reports

## Datasets

Training and prediction CSV files must contain a `path` column. For training/evaluation, include a `label` column. A `size` column (bytes) is optional but recommended. Place your files under the `data/` directory. Example names already supported in the commands below:

- `data/synthetic_unwanted_files_v3_train.csv`
- `data/synthetic_unwanted_files_v3_val.csv`
- `data/synthetic_unwanted_files_v3_test.csv`
- `data/synthetic_unwanted_files_v3.csv` (for batch predictions)

Example format:
Training and prediction CSV files must contain a `path` column. For training/evaluation, include a `label` column. A `size` column (bytes) is optional but recommended.

Example:

```csv
path,size,label
/home/user/tmp/setup.exe,1048576,unwanted
/home/user/Documents/report.pdf,20480,wanted
```

## CLI usage

Run all commands via `python -m src.cli <command>` (or `python -m cli` if the package is on `PYTHONPATH`).

### Train models

```bash
# With the provided synthetic splits
python -m src.cli train data/synthetic_unwanted_files_v3_train.csv --test-size 0.2 --model-out artifacts/unwanted_model.joblib

# Or with your own CSV
python -m src.cli train data/training.csv --test-size 0.2 --model-out artifacts/unwanted_model.joblib
```

Options:
- `--positive-label` / `--negative-label`: Override label vocabularies (can be provided multiple times)
- `--test-size`: Proportion reserved for validation during model selection
- `--model-out`: Where to save the best-performing model (default `artifacts/best_model.joblib`)

Artifacts: metrics CSV and plots are written to `reports/`.

### Evaluate an existing model

```bash
# Using the provided validation or test split
python -m src.cli evaluate-model artifacts/unwanted_model.joblib data/synthetic_unwanted_files_v3_val.csv

# Against your own holdout CSV
python -m src.cli evaluate-model artifacts/unwanted_model.joblib data/holdout.csv
```

Outputs metrics to the console and refreshes the confusion matrix, ROC curve, and precision-recall curve in `reports/`.

### Scan a directory

```bash
python -m src.cli scan artifacts/unwanted_model.joblib /home/user/downloads --max-files 1000 --output reports/scan.csv
```

Safeguards:
- Hidden files are skipped by default (`--include-hidden` to override)
- Symlinks are ignored
- A safety notice reminds you that no deletion occurs; predictions are advisory

### Predict from CSV

```bash
# Batch predictions on the synthetic prediction set
python -m src.cli predict artifacts/unwanted_model.joblib data/synthetic_unwanted_files_v3.csv --output reports/predict.csv

# Or with your own CSV
python -m src.cli predict artifacts/unwanted_model.joblib data/predict.csv --output reports/predict.csv
```

Generates probabilities and binary predictions using the model's tuned threshold.

## Reports and artifacts

- Models saved under `artifacts/`
- Metrics and plots under `reports/`
- CSV predictions and Markdown reports also stored in `reports/`

## Notes

- Thresholds are tuned to maximize F1-score on a validation split.
- Features include character n-gram TF-IDF on paths plus numeric metadata such as depth, extension length, presence of hidden components, and heuristic suspicious scores.
- The scanner never deletes files; manual review is required before taking action.
