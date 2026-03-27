"""
titanic_pipeline.pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~
Contains the full Titanic ML pipeline code as named section strings,
and the print_pipeline() entrypoint.
"""

_SECTIONS = {
    "imports": {
        "title": "IMPORTS",
        "code": """\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')""",
    },

    "eda": {
        "title": "SECTION 1 -- Data Loading & Exploration (EDA)",
        "code": """\
df = sns.load_dataset('titanic')
print('Shape:', df.shape)
print(df.head())

df.info()
print(df.describe())

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'count': missing, 'pct': missing_pct})
print(missing_df[missing_df['count'] > 0])

print('Survived value counts:')
print(df['survived'].value_counts())
print('\\nClass balance (%):')
print(df['survived'].value_counts(normalize=True) * 100)

# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# df['survived'].value_counts().plot(kind='bar', ax=axes[0], color=['salmon', 'steelblue'])
# axes[0].set_title('Survival Count')
# axes[0].set_xticklabels(['Died (0)', 'Survived (1)'], rotation=0)
# df['age'].dropna().hist(bins=30, ax=axes[1], color='steelblue', edgecolor='black')
# axes[1].set_title('Age Distribution')
# df.groupby('pclass')['survived'].mean().plot(kind='bar', ax=axes[2], color='steelblue')
# axes[2].set_title('Survival Rate by Class')
# axes[2].set_xticklabels(['1st', '2nd', '3rd'], rotation=0)
# plt.tight_layout()
# plt.show()

# numeric_cols = df.select_dtypes(include=[np.number]).columns
# plt.figure(figsize=(8, 6))
# sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
# plt.title('Correlation Heatmap')
# plt.show()""",
    },

    "preprocessing": {
        "title": "SECTION 2 -- Preprocessing & Feature Engineering",
        "code": """\
data = df.copy()

# Drop redundant/duplicate columns
cols_to_drop = ['alive', 'embark_town', 'who', 'adult_male', 'alone', 'deck', 'class']
data.drop(columns=cols_to_drop, inplace=True)
print('Columns after dropping:', list(data.columns))

# Handle missing values
data['age']      = data['age'].fillna(data['age'].median())
data['embarked'] = data['embarked'].fillna(data['embarked'].mode()[0])
print('Missing values after imputation:')
print(data.isnull().sum())

# Feature engineering
data['family_size']    = data['sibsp'] + data['parch'] + 1
data['is_alone']       = (data['family_size'] == 1).astype(int)
data['age_bin']        = pd.cut(data['age'], bins=[0, 12, 60, 100],
                                labels=['child', 'adult', 'senior'])
data['fare_per_person'] = data['fare'] / data['family_size']

print('Correlation of new features with survived:')
print(data[['family_size', 'is_alone', 'fare_per_person', 'survived']].corr()['survived'])

# Encode categoricals
data['sex'] = LabelEncoder().fit_transform(data['sex'])
data = pd.get_dummies(data, columns=['embarked', 'age_bin'], drop_first=True)
print('Columns after encoding:', list(data.columns))

# Split into features and target
X_fe = data.drop('survived', axis=1)
X_fe = X_fe.fillna(X_fe.median(numeric_only=True))  # catch any NaN from get_dummies on Categorical
y_fe = data['survived']

# Train/test split (stratified to preserve class balance)
X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(
    X_fe, y_fe, test_size=0.2, random_state=42, stratify=y_fe
)
print(f'Train: {X_train_fe.shape}  |  Test: {X_test_fe.shape}')
print('Train class balance:', y_train_fe.value_counts(normalize=True).to_dict())

# Scale -- required for k-NN, LR, SVM; fit only on train
scaler_fe = StandardScaler()
X_train_fe_scaled = scaler_fe.fit_transform(X_train_fe)
X_test_fe_scaled  = scaler_fe.transform(X_test_fe)
print('Feature set:', list(X_fe.columns))""",
    },

    "model_selection": {
        "title": "SECTION 3 -- Model Selection & Training",
        "code": """\
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'k-NN (k=5)':         KNeighborsClassifier(n_neighbors=5),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':                 SVC(probability=True),
}

results = {}

for name, model in models.items():
    if name in ['Decision Tree', 'Random Forest']:
        model.fit(X_train_fe, y_train_fe)
        y_pred = model.predict(X_test_fe)
        y_prob = model.predict_proba(X_test_fe)[:, 1]
        cv_scores = cross_val_score(model, X_fe, y_fe, cv=5, scoring='accuracy')
    else:
        model.fit(X_train_fe_scaled, y_train_fe)
        y_pred = model.predict(X_test_fe_scaled)
        y_prob = model.predict_proba(X_test_fe_scaled)[:, 1]
        cv_scores = cross_val_score(model, X_train_fe_scaled, y_train_fe, cv=5, scoring='accuracy')

    results[name] = {
        'accuracy':  accuracy_score(y_test_fe, y_pred),
        'precision': precision_score(y_test_fe, y_pred),
        'recall':    recall_score(y_test_fe, y_pred),
        'f1':        f1_score(y_test_fe, y_pred),
        'roc_auc':   roc_auc_score(y_test_fe, y_prob),
        'cv_mean':   cv_scores.mean(),
        'cv_std':    cv_scores.std(),
        'y_pred':    y_pred,
        'y_prob':    y_prob,
    }

print('Training complete.')

summary = pd.DataFrame({
    name: {
        'Accuracy':  f"{r['accuracy']:.3f}",
        'Precision': f"{r['precision']:.3f}",
        'Recall':    f"{r['recall']:.3f}",
        'F1':        f"{r['f1']:.3f}",
        'ROC-AUC':   f"{r['roc_auc']:.3f}",
        'CV Mean':   f"{r['cv_mean']:.3f} +/- {r['cv_std']:.3f}",
    }
    for name, r in results.items()
}).T
print(summary)

metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
compare_df = pd.DataFrame(
    {name: {m: r[m] for m in metrics} for name, r in results.items()}
).T
# compare_df.plot(kind='bar', figsize=(12, 5), ylim=(0.5, 1.0))
# plt.title('Model Comparison')
# plt.xticks(rotation=20, ha='right')
# plt.legend(loc='lower right')
# plt.tight_layout()
# plt.show()

# dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
# dt_model.fit(X_train_fe, y_train_fe)
# plt.figure(figsize=(16, 6))
# plot_tree(dt_model, feature_names=X_train_fe.columns.tolist(),
#           class_names=['Died', 'Survived'], filled=True, rounded=True, fontsize=9)
# plt.title('Decision Tree (max_depth=3)')
# plt.show()

k_range  = range(1, 21)
k_scores = []
for k in k_range:
    knn   = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train_fe_scaled, y_train_fe, cv=5, scoring='accuracy').mean()
    k_scores.append(score)

# plt.figure(figsize=(8, 4))
# plt.plot(k_range, k_scores, marker='o')
# plt.xlabel('k (number of neighbors)')
# plt.ylabel('CV Accuracy')
# plt.title('k-NN: Accuracy vs k')
# plt.xticks(k_range)
# plt.grid(True)
# plt.show()

best_k = k_range[k_scores.index(max(k_scores))]
print(f'Best k: {best_k}  |  CV Accuracy: {max(k_scores):.3f}')""",
    },

    "evaluation": {
        "title": "SECTION 4 -- Evaluation Metrics",
        "code": """\
best_name = max(results, key=lambda k: results[k]['f1'])
best_res  = results[best_name]
print(f'Best model: {best_name}')

cm = confusion_matrix(y_test_fe, best_res['y_pred'])
# plt.figure(figsize=(5, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title(f'Confusion Matrix -- {best_name}')
# plt.show()

tn, fp, fn, tp = cm.ravel()
print(f'TN={tn}  FP={fp}  FN={fn}  TP={tp}')
print(f'Precision = TP/(TP+FP) = {tp/(tp+fp):.3f}')
print(f'Recall    = TP/(TP+FN) = {tp/(tp+fn):.3f}')
print(classification_report(y_test_fe, best_res['y_pred'], target_names=['Died', 'Survived']))

# plt.figure(figsize=(8, 6))
# for name, r in results.items():
#     fpr, tpr, _ = roc_curve(y_test_fe, r['y_prob'])
#     plt.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})")
# plt.plot([0, 1], [0, 1], 'k--', label='Random')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curves')
# plt.legend()
# plt.grid(True)
# plt.show()

# rf_model    = models['Random Forest']
# importances = pd.Series(rf_model.feature_importances_, index=X_train_fe.columns)
# importances.sort_values(ascending=True).plot(kind='barh', figsize=(8, 6))
# plt.title('Feature Importances -- Random Forest')
# plt.tight_layout()
# plt.show()""",
    },

    "tuning": {
        "title": "SECTION 5 -- Hyperparameter Tuning (GridSearchCV)",
        "code": """\
# --- Decision Tree ---
param_grid_dt = {
    'max_depth':         [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'criterion':         ['gini', 'entropy'],
}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42),
                       param_grid_dt, cv=5, scoring='f1', n_jobs=-1)
grid_dt.fit(X_train_fe, y_train_fe)
print('Best DT params:', grid_dt.best_params_)
print('Best CV F1:    ', round(grid_dt.best_score_, 3))

# --- k-NN ---
param_grid_knn = {
    'n_neighbors': list(range(1, 21)),
    'weights':     ['uniform', 'distance'],
    'metric':      ['euclidean', 'manhattan'],
}
grid_knn = GridSearchCV(KNeighborsClassifier(),
                        param_grid_knn, cv=5, scoring='f1', n_jobs=-1)
grid_knn.fit(X_train_fe_scaled, y_train_fe)
print('Best k-NN params:', grid_knn.best_params_)
print('Best CV F1:       ', round(grid_knn.best_score_, 3))

# --- Logistic Regression ---
param_grid_lr = {
    'C':       [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver':  ['liblinear'],
}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000),
                       param_grid_lr, cv=5, scoring='f1', n_jobs=-1)
grid_lr.fit(X_train_fe_scaled, y_train_fe)
print('Best LR params:', grid_lr.best_params_)
print('Best CV F1:    ', round(grid_lr.best_score_, 3))

# --- Random Forest ---
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [3, 5, 7, None],
    'max_features': ['sqrt', 'log2'],
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42),
                       param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
grid_rf.fit(X_train_fe, y_train_fe)
print('Best RF params:', grid_rf.best_params_)
print('Best CV F1:    ', round(grid_rf.best_score_, 3))

# --- SVM ---
param_grid_svm = {
    'C':      [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma':  ['scale', 'auto'],
}
grid_svm = GridSearchCV(SVC(probability=True),
                        param_grid_svm, cv=5, scoring='f1', n_jobs=-1)
grid_svm.fit(X_train_fe_scaled, y_train_fe)
print('Best SVM params:', grid_svm.best_params_)
print('Best CV F1:     ', round(grid_svm.best_score_, 3))

# --- Evaluate all tuned models ---
tuned_models = {
    'Tuned Decision Tree':       (grid_dt.best_estimator_,  X_test_fe),
    'Tuned k-NN':                (grid_knn.best_estimator_, X_test_fe_scaled),
    'Tuned Logistic Regression': (grid_lr.best_estimator_,  X_test_fe_scaled),
    'Tuned Random Forest':       (grid_rf.best_estimator_,  X_test_fe),
    'Tuned SVM':                 (grid_svm.best_estimator_, X_test_fe_scaled),
}
tuned_results = {}
for name, (model, X_test_input) in tuned_models.items():
    y_pred = model.predict(X_test_input)
    print(f'\\n{name}')
    print(f'  Accuracy : {accuracy_score(y_test_fe, y_pred):.3f}')
    print(f'  F1 Score : {f1_score(y_test_fe, y_pred):.3f}')
    print(f'  ROC-AUC  : {roc_auc_score(y_test_fe, model.predict_proba(X_test_input)[:, 1]):.3f}')
    tuned_results[name] = {'f1': f1_score(y_test_fe, y_pred), 'y_pred': y_pred}""",
    },

    "final": {
        "title": "FINAL -- Select Best Model & Save Predictions",
        "code": """\
best_tuned_name = max(tuned_results, key=lambda k: tuned_results[k]['f1'])
best_tuned_pred = tuned_results[best_tuned_name]['y_pred']

print(f'\\nSelected model: {best_tuned_name}  (F1 = {tuned_results[best_tuned_name]["f1"]:.3f})')

output = pd.DataFrame({
    'actual':    y_test_fe.values,
    'predicted': best_tuned_pred,
})
output.to_csv('predictions.csv', index=False)
print('Predictions saved to predictions.csv')""",
    },
}

_SECTION_ORDER = [
    "imports",
    "eda",
    "preprocessing",
    "model_selection",
    "evaluation",
    "tuning",
    "final",
]


def print_pipeline(section: str = None) -> None:
    """Print the full Titanic ML pipeline code, or a single named section.

    Parameters
    ----------
    section : str, optional
        One of: 'imports', 'eda', 'preprocessing',
        'model_selection', 'evaluation', 'tuning', 'final'.
        If omitted, every section is printed in order.

    Examples
    --------
    >>> from titanic_pipeline import print_pipeline
    >>> print_pipeline()                   # prints everything
    >>> print_pipeline('preprocessing')    # prints Section 2 (preprocessing + feature engineering)
    >>> print_pipeline('tuning')           # prints Section 5 (hyperparameter tuning)
    >>> print_pipeline('final')            # prints model selection & CSV export
    """
    if section is not None:
        section = section.lower().strip()
        if section not in _SECTIONS:
            valid = ", ".join(f"'{k}'" for k in _SECTION_ORDER)
            raise ValueError(
                f"Unknown section '{section}'. Valid options: {valid}"
            )
        _print_section(section)
        return

    for key in _SECTION_ORDER:
        _print_section(key)
        print()


def _print_section(key: str) -> None:
    sec = _SECTIONS[key]
    border = "#" + "=" * 72
    print(border)
    print(f" # {sec['title']}")
    print(border)
    print(sec["code"])


# =============================================================================
#  Separate dataset variant (train.csv + test.csv)
# =============================================================================

_SECTIONS_SEP = {
    "imports": _SECTIONS["imports"],

    "eda": {
        "title": "SECTION 1 -- Data Loading & Exploration (EDA)",
        "code": """\
train = pd.read_csv('datasets/titanic/train.csv')
test  = pd.read_csv('datasets/titanic/test.csv')
df = train  # EDA runs on train only

print('Train shape:', train.shape)
print('Test shape: ', test.shape)
print(df.head())

df.info()
print(df.describe())

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'count': missing, 'pct': missing_pct})
print(missing_df[missing_df['count'] > 0])

print('Survived value counts:')
print(df['Survived'].value_counts())
print('Survived balance (%):')
print(df['Survived'].value_counts(normalize=True) * 100)""",
    },

    "preprocessing": {
        "title": "SECTION 2 -- Preprocessing & Feature Engineering",
        "code": """\
def preprocess(df):
    df = df.copy()
    cols_to_drop = ['PassengerId', 'Ticket', 'Cabin', 'Name']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    df['Age']      = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['family_size']    = df['SibSp'] + df['Parch'] + 1
    df['is_alone']       = (df['family_size'] == 1).astype(int)
    df['age_bin']        = pd.cut(df['Age'], bins=[0, 12, 60, 100],
                                  labels=['child', 'adult', 'senior'])
    df['fare_per_person'] = df['Fare'] / df['family_size']
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df = pd.get_dummies(df, columns=['Embarked', 'age_bin'], drop_first=True)
    return df

train_data = preprocess(train)
test_data  = preprocess(test)

# Align columns — test may be missing some dummy columns
train_data, test_data = train_data.align(test_data, join='left', axis=1, fill_value=0)

X_fe = train_data.drop('Survived', axis=1).fillna(train_data.median(numeric_only=True))
y_fe = train_data['Survived']
print(X_fe.dtypes)
X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(X_fe, y_fe, test_size=0.2, random_state=42)

# Scale -- fit only on train
scaler_fe = StandardScaler()
X_train_fe_scaled = scaler_fe.fit_transform(X_train_fe)
X_test_fe_scaled  = scaler_fe.transform(X_test_fe)
print('Feature set:', list(X_train_fe.columns))""",
    },

    "model_selection": _SECTIONS["model_selection"],

    "evaluation": {
        "title": "SECTION 4 -- Evaluation (CV on train only — test has no labels)",
        "code": """\
# test.csv has no 'Survived' column so we cannot compute test metrics directly.
# Cross-validation on the training set is the performance estimate.

cv_results = {}
for name, model in models.items():
    if name in ['Decision Tree', 'Random Forest']:
        scores = cross_val_score(model, X_train_fe, y_train_fe, cv=5, scoring='f1')
    else:
        scores = cross_val_score(model, X_train_fe_scaled, y_train_fe, cv=5, scoring='f1')
    cv_results[name] = scores
    print(f'{name}: CV F1 = {scores.mean():.3f} +/- {scores.std():.3f}')""",
    },

    "tuning": _SECTIONS["tuning"],

    "final": {
        "title": "FINAL -- Select Best Model & Save Predictions",
        "code": """\
best_tuned_name = max(tuned_results, key=lambda k: tuned_results[k]['f1'])
best_model, _ = tuned_models[best_tuned_name]

print(f'\\nSelected model: {best_tuned_name}  (CV F1 = {tuned_results[best_tuned_name]["f1"]:.3f})')

# Retrain the best model on the FULL train.csv (not just the 80% split)
# so all available labeled data is used for the final prediction
X_full = X_fe
y_full = y_fe

if best_tuned_name in ['Tuned Decision Tree', 'Tuned Random Forest']:
    best_model.fit(X_full, y_full)
else:
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_full)
    best_model.fit(X_full_scaled, y_full)

# Prepare test.csv features using the same preprocessing
X_final = test_data.drop('Survived', axis=1, errors='ignore').fillna(X_full.median())

if best_tuned_name in ['Tuned Decision Tree', 'Tuned Random Forest']:
    final_pred = best_model.predict(X_final)
else:
    X_final_scaled = scaler_full.transform(X_final)
    final_pred = best_model.predict(X_final_scaled)

# Add predictions column to original test.csv and save
output = test.copy()
output['Predicted'] = final_pred
output.to_csv('predictions.csv', index=False)
print('Predictions saved to predictions.csv')""",
    },
}

_SECTION_ORDER_SEP = [
    "imports",
    "eda",
    "preprocessing",
    "model_selection",
    "evaluation",
    "tuning",
    "final",
]


def print_pipeline_separate_dataset(section: str = None) -> None:
    """Print the Titanic ML pipeline adapted for separate train.csv and test.csv files.

    Parameters
    ----------
    section : str, optional
        One of: 'imports', 'eda', 'preprocessing',
        'model_selection', 'evaluation', 'tuning', 'final'.
        If omitted, every section is printed in order.

    Examples
    --------
    >>> from titanic_pipeline import print_pipeline_separate_dataset
    >>> print_pipeline_separate_dataset()               # prints everything
    >>> print_pipeline_separate_dataset('preprocessing')  # prints Section 2 only
    >>> print_pipeline_separate_dataset('final')          # prints prediction export
    """
    if section is not None:
        section = section.lower().strip()
        if section not in _SECTIONS_SEP:
            valid = ", ".join(f"'{k}'" for k in _SECTION_ORDER_SEP)
            raise ValueError(
                f"Unknown section '{section}'. Valid options: {valid}"
            )
        _print_section_sep(section)
        return

    for key in _SECTION_ORDER_SEP:
        _print_section_sep(key)
        print()


def _print_section_sep(key: str) -> None:
    sec = _SECTIONS_SEP[key]
    border = "#" + "=" * 72
    print(border)
    print(f" # {sec['title']}")
    print(border)
    print(sec["code"])


# =============================================================================
#  Multiclass variant (train.csv + test.csv, 3+ target classes)
# =============================================================================

_SECTIONS_MULTI = {
    "imports": _SECTIONS["imports"],

    "eda": _SECTIONS_SEP["eda"],

    "preprocessing": {
        "title": "SECTION 2 -- Preprocessing & Feature Engineering",
        "code": """\
def preprocess(df):
    df = df.copy()
    cols_to_drop = ['PassengerId', 'Ticket', 'Cabin', 'Name']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    df['Age']      = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['family_size']    = df['SibSp'] + df['Parch'] + 1
    df['is_alone']       = (df['family_size'] == 1).astype(int)
    df['age_bin']        = pd.cut(df['Age'], bins=[0, 12, 60, 100],
                                  labels=['child', 'adult', 'senior'])
    df['fare_per_person'] = df['Fare'] / df['family_size']
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df = pd.get_dummies(df, columns=['Embarked', 'age_bin'], drop_first=True)
    return df

train_data = preprocess(train)
test_data  = preprocess(test)

# Align columns — test may be missing some dummy columns
train_data, test_data = train_data.align(test_data, join='left', axis=1, fill_value=0)

X_fe = train_data.drop('target', axis=1).fillna(train_data.median(numeric_only=True))

# Encode target if it contains string class labels
le_target = LabelEncoder()
y_fe = le_target.fit_transform(train_data['target'])
print('Classes:', list(le_target.classes_))

print(X_fe.dtypes)
X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(X_fe, y_fe, test_size=0.2, random_state=42)

# Scale -- fit only on train
scaler_fe = StandardScaler()
X_train_fe_scaled = scaler_fe.fit_transform(X_train_fe)
X_test_fe_scaled  = scaler_fe.transform(X_test_fe)
print('Feature set:', list(X_train_fe.columns))""",
    },

    "model_selection": {
        "title": "SECTION 3 -- Model Selection & Training",
        "code": """\
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'k-NN (k=5)':         KNeighborsClassifier(n_neighbors=5),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':                 SVC(probability=True),
}

results = {}

for name, model in models.items():
    if name in ['Decision Tree', 'Random Forest']:
        model.fit(X_train_fe, y_train_fe)
        y_pred = model.predict(X_test_fe)
        y_prob = model.predict_proba(X_test_fe)          # all class columns
        cv_scores = cross_val_score(model, X_fe, y_fe, cv=5, scoring='accuracy')
    else:
        model.fit(X_train_fe_scaled, y_train_fe)
        y_pred = model.predict(X_test_fe_scaled)
        y_prob = model.predict_proba(X_test_fe_scaled)   # all class columns
        cv_scores = cross_val_score(model, X_train_fe_scaled, y_train_fe, cv=5, scoring='accuracy')

    results[name] = {
        'accuracy':  accuracy_score(y_test_fe, y_pred),
        'precision': precision_score(y_test_fe, y_pred, average='weighted'),
        'recall':    recall_score(y_test_fe, y_pred, average='weighted'),
        'f1':        f1_score(y_test_fe, y_pred, average='weighted'),
        'roc_auc':   roc_auc_score(y_test_fe, y_prob, multi_class='ovr', average='macro'),
        'cv_mean':   cv_scores.mean(),
        'cv_std':    cv_scores.std(),
        'y_pred':    y_pred,
        'y_prob':    y_prob,
    }

print('Training complete.')

summary = pd.DataFrame({
    name: {
        'Accuracy':  f"{r['accuracy']:.3f}",
        'Precision': f"{r['precision']:.3f}",
        'Recall':    f"{r['recall']:.3f}",
        'F1':        f"{r['f1']:.3f}",
        'ROC-AUC':   f"{r['roc_auc']:.3f}",
        'CV Mean':   f"{r['cv_mean']:.3f} +/- {r['cv_std']:.3f}",
    }
    for name, r in results.items()
}).T
print(summary)

k_range  = range(1, 21)
k_scores = []
for k in k_range:
    knn   = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train_fe_scaled, y_train_fe, cv=5, scoring='accuracy').mean()
    k_scores.append(score)

best_k = k_range[k_scores.index(max(k_scores))]
print(f'Best k: {best_k}  |  CV Accuracy: {max(k_scores):.3f}')""",
    },

    "evaluation": {
        "title": "SECTION 4 -- Evaluation Metrics",
        "code": """\
best_name = max(results, key=lambda k: results[k]['f1'])
best_res  = results[best_name]
print(f'Best model: {best_name}')

cm = confusion_matrix(y_test_fe, best_res['y_pred'])
print('Confusion Matrix:')
print(cm)

# cm.ravel() is binary-only — for multiclass use the full matrix above
print(classification_report(y_test_fe, best_res['y_pred'],
                             target_names=list(le_target.classes_)))""",
    },

    "tuning": {
        "title": "SECTION 5 -- Hyperparameter Tuning (GridSearchCV)",
        "code": """\
# --- Decision Tree ---
param_grid_dt = {
    'max_depth':         [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'criterion':         ['gini', 'entropy'],
}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42),
                       param_grid_dt, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_dt.fit(X_train_fe, y_train_fe)
print('Best DT params:', grid_dt.best_params_)
print('Best CV F1:    ', round(grid_dt.best_score_, 3))

# --- k-NN ---
param_grid_knn = {
    'n_neighbors': list(range(1, 21)),
    'weights':     ['uniform', 'distance'],
    'metric':      ['euclidean', 'manhattan'],
}
grid_knn = GridSearchCV(KNeighborsClassifier(),
                        param_grid_knn, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_knn.fit(X_train_fe_scaled, y_train_fe)
print('Best k-NN params:', grid_knn.best_params_)
print('Best CV F1:       ', round(grid_knn.best_score_, 3))

# --- Logistic Regression ---
param_grid_lr = {
    'C':       [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver':  ['liblinear'],
}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000),
                       param_grid_lr, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_lr.fit(X_train_fe_scaled, y_train_fe)
print('Best LR params:', grid_lr.best_params_)
print('Best CV F1:    ', round(grid_lr.best_score_, 3))

# --- Random Forest ---
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [3, 5, 7, None],
    'max_features': ['sqrt', 'log2'],
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42),
                       param_grid_rf, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_rf.fit(X_train_fe, y_train_fe)
print('Best RF params:', grid_rf.best_params_)
print('Best CV F1:    ', round(grid_rf.best_score_, 3))

# --- SVM ---
param_grid_svm = {
    'C':      [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma':  ['scale', 'auto'],
}
grid_svm = GridSearchCV(SVC(probability=True),
                        param_grid_svm, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_svm.fit(X_train_fe_scaled, y_train_fe)
print('Best SVM params:', grid_svm.best_params_)
print('Best CV F1:     ', round(grid_svm.best_score_, 3))

# --- Evaluate all tuned models ---
tuned_models = {
    'Tuned Decision Tree':       (grid_dt.best_estimator_,  X_test_fe),
    'Tuned k-NN':                (grid_knn.best_estimator_, X_test_fe_scaled),
    'Tuned Logistic Regression': (grid_lr.best_estimator_,  X_test_fe_scaled),
    'Tuned Random Forest':       (grid_rf.best_estimator_,  X_test_fe),
    'Tuned SVM':                 (grid_svm.best_estimator_, X_test_fe_scaled),
}
tuned_results = {}
for name, (model, X_test_input) in tuned_models.items():
    y_pred = model.predict(X_test_input)
    y_prob = model.predict_proba(X_test_input)
    print(f'\\n{name}')
    print(f'  Accuracy : {accuracy_score(y_test_fe, y_pred):.3f}')
    print(f'  F1 Score : {f1_score(y_test_fe, y_pred, average="weighted"):.3f}')
    print(f'  ROC-AUC  : {roc_auc_score(y_test_fe, y_prob, multi_class="ovr", average="macro"):.3f}')
    tuned_results[name] = {'f1': f1_score(y_test_fe, y_pred, average='weighted'), 'y_pred': y_pred}""",
    },

    "final": {
        "title": "FINAL -- Select Best Model & Save Predictions",
        "code": """\
best_tuned_name = max(tuned_results, key=lambda k: tuned_results[k]['f1'])
best_model, _ = tuned_models[best_tuned_name]

print(f'\\nSelected model: {best_tuned_name}  (F1 = {tuned_results[best_tuned_name]["f1"]:.3f})')

# Retrain the best model on the FULL train.csv (not just the 80% split)
X_full = X_fe
y_full = y_fe

if best_tuned_name in ['Tuned Decision Tree', 'Tuned Random Forest']:
    best_model.fit(X_full, y_full)
else:
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_full)
    best_model.fit(X_full_scaled, y_full)

# Prepare test.csv features
X_final = test_data.drop('target', axis=1, errors='ignore').fillna(X_full.median())

if best_tuned_name in ['Tuned Decision Tree', 'Tuned Random Forest']:
    final_pred_encoded = best_model.predict(X_final)
else:
    X_final_scaled = scaler_full.transform(X_final)
    final_pred_encoded = best_model.predict(X_final_scaled)

# Decode numeric predictions back to original class labels
final_pred = le_target.inverse_transform(final_pred_encoded)

# Add predictions column to original test.csv and save
output = test.copy()
output['Predicted'] = final_pred
output.to_csv('predictions.csv', index=False)
print('Predictions saved to predictions.csv')""",
    },
}

_SECTION_ORDER_MULTI = [
    "imports",
    "eda",
    "preprocessing",
    "model_selection",
    "evaluation",
    "tuning",
    "final",
]


def print_pipeline_multi_class(section: str = None) -> None:
    """Print the Titanic ML pipeline adapted for multiclass classification.

    Parameters
    ----------
    section : str, optional
        One of: 'imports', 'eda', 'preprocessing',
        'model_selection', 'evaluation', 'tuning', 'final'.
        If omitted, every section is printed in order.

    Examples
    --------
    >>> from titanic_pipeline import print_pipeline_multi_class
    >>> print_pipeline_multi_class()                # prints everything
    >>> print_pipeline_multi_class('preprocessing') # prints Section 2 only
    >>> print_pipeline_multi_class('tuning')        # prints Section 5 only
    >>> print_pipeline_multi_class('final')         # prints prediction export
    """
    if section is not None:
        section = section.lower().strip()
        if section not in _SECTIONS_MULTI:
            valid = ", ".join(f"'{k}'" for k in _SECTION_ORDER_MULTI)
            raise ValueError(
                f"Unknown section '{section}'. Valid options: {valid}"
            )
        _print_section_multi(section)
        return

    for key in _SECTION_ORDER_MULTI:
        _print_section_multi(key)
        print()


def _print_section_multi(key: str) -> None:
    sec = _SECTIONS_MULTI[key]
    border = "#" + "=" * 72
    print(border)
    print(f" # {sec['title']}")
    print(border)
    print(sec["code"])
