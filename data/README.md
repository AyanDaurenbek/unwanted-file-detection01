# Data directory

Place your training, validation, test, and prediction CSV files here. The CLI examples assume files such as:

- `synthetic_unwanted_files_v3_train.csv`
- `synthetic_unwanted_files_v3_val.csv`
- `synthetic_unwanted_files_v3_test.csv`
- `synthetic_unwanted_files_v3.csv` (for bulk predictions)

Each CSV must include a `path` column; include `label` for supervised training/evaluation and optionally `size` (bytes).
