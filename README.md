# Data Quality and Explanation Stability in Healthcare ML

A small, reproducible research prototype for Nithya's PhD portfolio. It tests how **training-data missingness** and **label errors** affect classification and global explanation rankings on the public Wisconsin Diagnostic Breast Cancer benchmark bundled with scikit-learn. This is a methodological demonstration, not a clinical model or a published research result.

## Why this project

Data quality and explanation quality can fail together. A classifier may retain reasonable accuracy while its feature-ranking explanation changes under corrupted training data. This project makes both effects measurable in a controlled experiment. It connects data validation, model evaluation, and explainable AI without claiming that one dataset establishes a general law.

## What it does

The CLI loads the benchmark locally; creates stratified training, validation, and held-out test partitions; injects missing feature cells and randomly flipped labels into **training data only**; fits a median-imputation, standardization, logistic-regression pipeline; computes test accuracy and ROC-AUC; and compares validation-set permutation-importance rankings against a clean-training baseline. The clean test partition is evaluated only after fitting. Each run records its seed and corruption counts.

The explanation statistic is rank correlation between *global permutation importances* from the clean and damaged models. It measures ranking stability on this validation set; it is not a causal explanation, per-patient explanation, or guarantee of clinical reliability. Correlated features can make permutation importances unstable even without corruption.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
dq-xai --config configs/smoke.json --output results/smoke.csv
```

The smoke matrix runs one seed across clean, 20% missingness, 10% label flips, and combined corruption (four rows). For a broader **exploratory** matrix:

```bash
dq-xai --config configs/research_matrix.json --output results/research_matrix.csv
```

The larger matrix uses five seeds and a 4-by-4 corruption grid. Treat seed-level outcomes and uncertainty as essential; do not publish a single run as a robust conclusion.

## Architecture

`configs/` defines experiment matrices. `src/dq_xai/experiment.py` owns data splitting, train-only corruption, modeling, and evaluation. `src/dq_xai/cli.py` runs and exports a matrix. `tests/` checks deterministic behavior, corruption isolation, and metric bounds. Generated results stay under `results/` and are ignored by Git.

## Limitations and next research steps

- The dataset is small, public, and historical; no real patient records or clinical deployment are involved.
- Missingness is independent random cell deletion and label noise is uniform random flipping. Real quality failures can be systematic and subgroup-specific.
- A single model and global explanation method are used. Add tree models, calibration, subgroup metrics, and local explanation methods before stronger claims.
- Investigate feature correlations and explanation uncertainty, with confidence intervals across seeds.
- Define externally meaningful quality failures and validate on another licensed, de-identified dataset before drawing healthcare conclusions.

## Data and ethics

The dataset is distributed with scikit-learn and originally comes from the UCI Wisconsin Diagnostic Breast Cancer dataset. No private health data, CVs, academic records, or credentials belong in this repository. This code is for research and education only.
