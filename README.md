# titanic-pipeline

A classical machine learning pipeline built on the Titanic survival dataset. Covers the full ML workflow — from raw data to evaluated models — with each stage clearly separated. Supports binary classification, multiclass classification, and regression.

## Install

```bash
pip install titanic-pipeline
```

## Usage

### Single dataset (binary classification)

Use this when you have one combined CSV (or load via seaborn):

```python
from titanic_pipeline import print_pipeline

print_pipeline()                    # print full pipeline
print_pipeline('preprocessing')     # print a single section
```

### Separate train / test datasets (binary classification)

Use this when you have separate `train.csv` and `test.csv` files (e.g. Kaggle competitions):

```python
from titanic_pipeline import print_pipeline_separate_dataset

print_pipeline_separate_dataset()                # print full pipeline
print_pipeline_separate_dataset('preprocessing') # print a single section
```

### Separate train / test datasets (multiclass classification)

Use this when your target column has 3 or more classes:

```python
from titanic_pipeline import print_pipeline_multi_class

print_pipeline_multi_class()                # print full pipeline
print_pipeline_multi_class('tuning')        # print a single section
```

### Separate train / test datasets (regression)

Use this when your target column is a continuous numeric value:

```python
from titanic_pipeline import print_pipeline_regression

print_pipeline_regression()                # print full pipeline
print_pipeline_regression('tuning')        # print a single section
```

All functions print annotated, ready-to-run Python code to stdout.

## Available Sections

Pass any of these as the `section` argument to any function:

| Key | Stage |
|---|---|
| `'imports'` | All library imports |
| `'eda'` | Data loading and exploration |
| `'preprocessing'` | Imputation, feature engineering, encoding, scaling |
| `'model_selection'` | Train and compare 5 models |
| `'evaluation'` | Metrics (varies by task — see below) |
| `'tuning'` | GridSearchCV on all 5 models |
| `'final'` | Select best model, retrain on full data, save predictions to CSV |

## Pipeline Stages

| # | Stage | What it covers |
|---|-------|----------------|
| 1 | **EDA** | Shape, dtypes, missing values, class/target balance |
| 2 | **Preprocessing & Feature Engineering** | Imputation, family size, age bins, fare-per-person, encoding, scaling |
| 3 | **Model Selection & Training** | Train and compare 5 models side-by-side |
| 4 | **Evaluation** | Metrics suited to the task type |
| 5 | **Hyperparameter Tuning** | GridSearchCV on all 5 models |
| Final | **Best Model & Predictions** | Retrain on full data, predict on test set, save `predictions.csv` |

## Function Comparison

| Function | Task | Models | Key Metrics | GridSearchCV scoring |
|---|---|---|---|---|
| `print_pipeline` | Binary classification | LR, DT, k-NN, RF, SVM | Accuracy, F1, ROC-AUC | `f1` |
| `print_pipeline_separate_dataset` | Binary classification | LR, DT, k-NN, RF, SVM | Accuracy, F1, ROC-AUC | `f1` |
| `print_pipeline_multi_class` | Multiclass classification | LR, DT, k-NN, RF, SVM | Weighted F1, ROC-AUC (OvR) | `f1_weighted` |
| `print_pipeline_regression` | Regression | Ridge, DT, k-NN, RF, SVR | MAE, RMSE, R² | `r2` |

## Key Differences by Task Type

### Binary vs Multiclass Classification

| | Binary | Multiclass |
|---|---|---|
| `predict_proba` | `[:, 1]` (one column) | all columns |
| `f1_score` / `precision` / `recall` | no `average` needed | `average='weighted'` |
| `roc_auc_score` | default | `multi_class='ovr', average='macro'` |
| Confusion matrix | `cm.ravel()` for TP/TN/FP/FN | full NxN matrix printed |
| Target encoding | raw binary column | `LabelEncoder` on target |
| GridSearchCV scoring | `'f1'` | `'f1_weighted'` |

### Classification vs Regression

| | Classification | Regression |
|---|---|---|
| Models | Classifier variants | `Ridge`, `DecisionTreeRegressor`, `KNeighborsRegressor`, `RandomForestRegressor`, `SVR` |
| Metrics | Accuracy, F1, ROC-AUC | MAE, RMSE, R² |
| CV / GridSearch scoring | `'accuracy'` / `'f1_weighted'` | `'r2'` |
| Best model selection | highest F1 | highest R² |
| `stratify` in split | yes | no |
| `predict_proba` | used | not available — removed |
| SVR tuning params | `C`, `kernel`, `gamma` | `C`, `kernel`, `epsilon` |

## Separate Dataset Notes

All three separate-dataset functions (`print_pipeline_separate_dataset`, `print_pipeline_multi_class`, `print_pipeline_regression`):
- Use PascalCase column names (`Age`, `Sex`, `Fare`, etc.) matching the real Kaggle Titanic CSV format
- Drop `PassengerId`, `Ticket`, `Cabin`, `Name` instead of seaborn-specific columns
- Fit all encoders and scalers on `train.csv` only — never on `test.csv`
- Align dummy columns between train and test after one-hot encoding
- **Retrain the best model on the full `train.csv`** before predicting on `test.csv`
- Save the original `test.csv` columns plus a `Predicted` / `PredictedValue` column to `predictions.csv`

## Requirements

- Python >= 3.8
- numpy, pandas, matplotlib, seaborn, scikit-learn

## License

MIT
