# titanic-pipeline

A classical machine learning pipeline built on the Titanic survival dataset. Covers the full ML workflow — EDA, preprocessing, feature engineering, model selection, evaluation, and hyperparameter tuning. Supports binary classification, multiclass classification, and regression, for both single-file and separate train/test datasets.

## Install

```bash
pip install titanic-pipeline
```

## Usage

### Single dataset (binary classification)

```python
from titanic_pipeline import print_pipeline

print_pipeline()                    # print full pipeline
print_pipeline('preprocessing')     # print a single section
```

### Separate train / test datasets (binary classification)

```python
from titanic_pipeline import print_pipeline_separate_dataset

print_pipeline_separate_dataset()
print_pipeline_separate_dataset('tuning')
```

### Separate train / test datasets (multiclass classification)

```python
from titanic_pipeline import print_pipeline_multi_class

print_pipeline_multi_class()
print_pipeline_multi_class('tuning')
```

### Separate train / test datasets (regression)

```python
from titanic_pipeline import print_pipeline_regression

print_pipeline_regression()
print_pipeline_regression('tuning')
```

## Available Sections

Pass any of these as the `section` argument to any function:

| Key | Stage |
|---|---|
| `'imports'` | All library imports |
| `'eda'` | Data loading and exploration |
| `'preprocessing'` | Imputation, feature engineering, encoding, scaling |
| `'model_selection'` | Train and compare 5 models |
| `'evaluation'` | Metrics suited to the task type |
| `'tuning'` | GridSearchCV on all 5 models |
| `'final'` | Select best model, retrain on full data, save predictions to CSV |

## Function Comparison

| Function | Task | Models | Key Metrics |
|---|---|---|---|
| `print_pipeline` | Binary classification | LR, DT, k-NN, RF, SVM | Accuracy, F1, ROC-AUC |
| `print_pipeline_separate_dataset` | Binary classification | LR, DT, k-NN, RF, SVM | Accuracy, F1, ROC-AUC |
| `print_pipeline_multi_class` | Multiclass classification | LR, DT, k-NN, RF, SVM | Weighted F1, ROC-AUC (OvR) |
| `print_pipeline_regression` | Regression | Ridge, DT, k-NN, RF, SVR | MAE, RMSE, R² |

## Pipeline Stages

| # | Stage | What it covers |
|---|-------|----------------|
| 1 | **EDA** | Shape, dtypes, missing values, target distribution |
| 2 | **Preprocessing & Feature Engineering** | Imputation, family size, age bins, fare-per-person, encoding, scaling |
| 3 | **Model Selection & Training** | Train and compare 5 models side-by-side |
| 4 | **Evaluation** | Metrics suited to task type |
| 5 | **Hyperparameter Tuning** | GridSearchCV on all 5 models |
| Final | **Best Model & Predictions** | Retrain on full data, predict on test set, save `predictions.csv` |

## Requirements

- Python >= 3.8
- numpy, pandas, matplotlib, seaborn, scikit-learn

## License

MIT
