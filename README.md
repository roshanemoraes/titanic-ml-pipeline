# titanic-pipeline

A classical machine learning pipeline built on the Titanic survival dataset. Covers the full ML workflow — from raw data to evaluated models — with each stage clearly separated.

## Install

```bash
pip install titanic-pipeline
```

## Usage

### Single dataset

Use this when you have one combined CSV (or load via seaborn):

```python
from titanic_pipeline import print_pipeline

print_pipeline()                    # print full pipeline
print_pipeline('preprocessing')     # print a single section
```

### Separate train / test datasets

Use this when you have separate `train.csv` and `test.csv` files (e.g. Kaggle competitions):

```python
from titanic_pipeline import print_pipeline_separate_dataset

print_pipeline_separate_dataset()                # print full pipeline
print_pipeline_separate_dataset('preprocessing') # print a single section
```

Both functions print annotated, ready-to-run Python code to stdout.

## Available Sections

Pass any of these as the `section` argument to either function:

| Key | Stage |
|---|---|
| `'imports'` | All library imports |
| `'eda'` | Data loading and exploration |
| `'preprocessing'` | Imputation, feature engineering, encoding, scaling |
| `'model_selection'` | Train and compare 5 models |
| `'evaluation'` | Metrics — accuracy, precision, recall, F1, ROC-AUC |
| `'tuning'` | GridSearchCV on all 5 models |
| `'final'` | Select best model and save predictions to CSV |

## Pipeline Stages

| # | Stage | What it covers |
|---|-------|----------------|
| 1 | **EDA** | Shape, dtypes, missing values, class balance |
| 2 | **Preprocessing & Feature Engineering** | Imputation, family size, age bins, fare-per-person, encoding, scaling |
| 3 | **Model Training** | Logistic Regression, Decision Tree, k-NN, Random Forest, SVM |
| 4 | **Evaluation** | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, CV scores |
| 5 | **Hyperparameter Tuning** | GridSearchCV on all 5 models |
| 6 | **Final** | Best model selection, predictions saved to `predictions.csv` |

## Models Compared

- Logistic Regression
- Decision Tree
- k-Nearest Neighbors
- Random Forest
- Support Vector Machine

All models are tuned with GridSearchCV and compared side-by-side across accuracy, precision, recall, F1, and ROC-AUC.

## Separate Dataset Notes

`print_pipeline_separate_dataset()` is adapted for the real Kaggle Titanic CSV format:
- Column names are PascalCase (`Survived`, `Age`, `Sex`, `Fare`, etc.)
- Drops `PassengerId`, `Ticket`, `Cabin`, `Name` instead of seaborn-specific columns
- Fits all encoders and scalers on `train.csv` only — never on `test.csv`
- Aligns dummy columns between train and test after one-hot encoding
- Final predictions are made on the actual `test.csv` and saved to `predictions.csv`

## Requirements

- Python >= 3.8
- numpy, pandas, matplotlib, seaborn, scikit-learn

## License

MIT
