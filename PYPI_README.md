# titanic-pipeline

A classical ML pipeline for the Titanic survival dataset.
Covers the full workflow: EDA, preprocessing, feature engineering, model selection, evaluation, and hyperparameter tuning — all accessible via a single `print_pipeline()` call.

## Install

```bash
pip install titanic-pipeline
```

## Usage

```python
from titanic_pipeline import print_pipeline

# Print the entire pipeline
print_pipeline()

# Print a specific section
print_pipeline('eda')
print_pipeline('preprocessing')
print_pipeline('feature_engineering')
print_pipeline('model_selection')
print_pipeline('evaluation')
print_pipeline('tuning')
```

## Sections

| Key | Contents |
|---|---|
| `imports` | All library imports |
| `eda` | Data loading, missing value audit, target distribution, visualizations |
| `preprocessing` | Drop redundant columns, impute missing values, encode categoricals, scale, train/test split |
| `feature_engineering` | `family_size`, `is_alone`, `age_bin`, `fare_per_person` with survival correlation plots |
| `model_selection` | Logistic Regression, Decision Tree, k-NN, Random Forest, SVM — side-by-side comparison |
| `evaluation` | Confusion matrix, classification report, ROC curves, feature importance |
| `tuning` | GridSearchCV for Decision Tree and k-NN |
