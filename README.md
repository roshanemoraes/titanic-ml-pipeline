# titanic-pipeline

A classical machine learning pipeline built on the Titanic survival dataset. Covers the full ML workflow — from raw data to evaluated models — with each stage clearly separated.

## Install

```bash
pip install titanic-pipeline
```

## Usage

```python
from titanic_pipeline import print_pipeline

print_pipeline()
```

Prints the full annotated pipeline code to stdout, broken into sections.

## Pipeline Stages

| # | Stage | What it covers |
|---|-------|---------------|
| 1 | **EDA** | Shape, dtypes, missing values, class balance, distributions |
| 2 | **Preprocessing** | Imputation, label encoding, one-hot encoding, train/test split, scaling |
| 3 | **Feature Engineering** | Family size, is-alone flag, age bins, fare-per-person |
| 4 | **Model Training** | Logistic Regression, Decision Tree, k-NN, Random Forest, SVM |
| 5 | **Evaluation** | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix |
| 6 | **Hyperparameter Tuning** | GridSearchCV on Decision Tree and k-NN |

## Models Compared

- Logistic Regression
- Decision Tree
- k-Nearest Neighbors
- Random Forest
- Support Vector Machine

All models are compared side-by-side across accuracy, precision, recall, F1, and ROC-AUC.

## Requirements

- Python >= 3.8
- numpy, pandas, matplotlib, seaborn, scikit-learn

## License

MIT
