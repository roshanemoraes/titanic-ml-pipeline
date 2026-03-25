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
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

print('All imports successful')""",
    },

    "eda": {
        "title": "SECTION 1 -- Data Loading & Exploration (EDA)",
        "code": """\
# Load Titanic dataset (built into seaborn)
df = sns.load_dataset('titanic')
print('Shape:', df.shape)
df.head()

# Data types and basic info
df.info()

# Summary statistics
df.describe()

# Check missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'count': missing, 'pct': missing_pct})
print(missing_df[missing_df['count'] > 0])

# Target distribution
print('Survived value counts:')
print(df['survived'].value_counts())
print('\\nClass balance (%):')
print(df['survived'].value_counts(normalize=True) * 100)

# Visualize distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

df['survived'].value_counts().plot(kind='bar', ax=axes[0], color=['salmon', 'steelblue'])
axes[0].set_title('Survival Count')
axes[0].set_xticklabels(['Died (0)', 'Survived (1)'], rotation=0)

df['age'].dropna().hist(bins=30, ax=axes[1], color='steelblue', edgecolor='black')
axes[1].set_title('Age Distribution')

df.groupby('pclass')['survived'].mean().plot(kind='bar', ax=axes[2], color='steelblue')
axes[2].set_title('Survival Rate by Class')
axes[2].set_xticklabels(['1st', '2nd', '3rd'], rotation=0)

plt.tight_layout()
plt.show()

# Correlation heatmap (numeric features only)
numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()""",
    },

    "preprocessing": {
        "title": "SECTION 2 -- Data Preprocessing",
        "code": """\
# Work on a copy
data = df.copy()

# Drop duplicate/redundant columns
# 'alive' = string version of 'survived', 'embark_town' = same as 'embarked'
# 'class' = string version of 'pclass' (already numeric)
# 'who', 'adult_male', 'alone' are derived -- keep raw features instead
cols_to_drop = ['alive', 'embark_town', 'who', 'adult_male', 'alone', 'deck', 'class']
data.drop(columns=cols_to_drop, inplace=True)
print('Columns after dropping:', list(data.columns))

# --- Handle Missing Values ---
# Age: fill with median (robust to outliers)
data['age'].fillna(data['age'].median(), inplace=True)

# Embarked: fill with mode (most frequent port)
data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)

print('Missing values after imputation:')
print(data.isnull().sum())

# --- Encode Categorical Variables ---
# Binary encoding: sex (male=1, female=0)
data['sex'] = LabelEncoder().fit_transform(data['sex'])

# One-hot encoding: embarked (C, Q, S)
data = pd.get_dummies(data, columns=['embarked'], drop_first=True)
print('Columns after encoding:', list(data.columns))

# --- Split into Features and Target ---
X = data.drop('survived', axis=1)
y = data['survived']

# Train / Test Split (80/20, stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'Train: {X_train.shape}  |  Test: {X_test.shape}')
print('Train class balance:', y_train.value_counts(normalize=True).to_dict())

# --- Feature Scaling ---
# Required for k-NN, Logistic Regression, SVM (distance/gradient-based)
# NOT required for tree-based models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on train
X_test_scaled  = scaler.transform(X_test)         # transform test using train stats
print('Scaling done. Train mean (should be ~0):', X_train_scaled.mean(axis=0).round(2))""",
    },

    "feature_engineering": {
        "title": "SECTION 3 -- Feature Engineering",
        "code": """\
# Re-start from clean data to add engineered features
data_fe = df.copy()
data_fe.drop(columns=['alive', 'embark_town', 'who', 'adult_male', 'alone', 'deck', 'class'], inplace=True)
data_fe['age'].fillna(data_fe['age'].median(), inplace=True)
data_fe['embarked'].fillna(data_fe['embarked'].mode()[0], inplace=True)

# --- New Feature 1: Family Size ---
data_fe['family_size'] = data_fe['sibsp'] + data_fe['parch'] + 1  # include self

# --- New Feature 2: Is Alone ---
data_fe['is_alone'] = (data_fe['family_size'] == 1).astype(int)

# --- New Feature 3: Age Bin (child / adult / senior) ---
data_fe['age_bin'] = pd.cut(
    data_fe['age'],
    bins=[0, 12, 60, 100],
    labels=['child', 'adult', 'senior']
)

# --- New Feature 4: Fare per Person ---
data_fe['fare_per_person'] = data_fe['fare'] / data_fe['family_size']

print('Correlation of new features with survived:')
print(data_fe[['family_size', 'is_alone', 'fare_per_person', 'survived']].corr()['survived'])

# Visualize impact of engineered features
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

data_fe.groupby('family_size')['survived'].mean().plot(kind='bar', ax=axes[0])
axes[0].set_title('Survival Rate by Family Size')

data_fe.groupby('age_bin')['survived'].mean().plot(kind='bar', ax=axes[1], color='darkorange')
axes[1].set_title('Survival Rate by Age Group')

data_fe.groupby('is_alone')['survived'].mean().plot(kind='bar', ax=axes[2], color='green')
axes[2].set_title('Survival Rate: Solo vs With Family')
axes[2].set_xticklabels(['With Family', 'Alone'], rotation=0)

plt.tight_layout()
plt.show()

# Encode and build final feature set WITH engineered features
data_fe['sex'] = LabelEncoder().fit_transform(data_fe['sex'])
data_fe = pd.get_dummies(data_fe, columns=['embarked', 'age_bin'], drop_first=True)

X_fe = data_fe.drop('survived', axis=1)
y_fe = data_fe['survived']

X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(
    X_fe, y_fe, test_size=0.2, random_state=42, stratify=y_fe
)

scaler_fe = StandardScaler()
X_train_fe_scaled = scaler_fe.fit_transform(X_train_fe)
X_test_fe_scaled  = scaler_fe.transform(X_test_fe)

print('Feature set with engineering:', list(X_fe.columns))""",
    },

    "model_selection": {
        "title": "SECTION 4 -- Model Selection & Training",
        "code": """\
# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'k-NN (k=5)':         KNeighborsClassifier(n_neighbors=5),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':                 SVC(probability=True),
}

results = {}

for name, model in models.items():
    # Tree models use unscaled data; others use scaled
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

# Summary table
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

# Bar chart comparison
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
compare_df = pd.DataFrame(
    {name: {m: r[m] for m in metrics} for name, r in results.items()}
).T
compare_df.plot(kind='bar', figsize=(12, 5), ylim=(0.5, 1.0))
plt.title('Model Comparison')
plt.xticks(rotation=20, ha='right')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Visualize Decision Tree structure
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train_fe, y_train_fe)

plt.figure(figsize=(16, 6))
plot_tree(
    dt_model,
    feature_names=X_train_fe.columns.tolist(),
    class_names=['Died', 'Survived'],
    filled=True, rounded=True, fontsize=9
)
plt.title('Decision Tree (max_depth=3)')
plt.show()

# k-NN: choosing the right k
k_range = range(1, 21)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train_fe_scaled, y_train_fe, cv=5, scoring='accuracy').mean()
    k_scores.append(score)

plt.figure(figsize=(8, 4))
plt.plot(k_range, k_scores, marker='o')
plt.xlabel('k (number of neighbors)')
plt.ylabel('CV Accuracy')
plt.title('k-NN: Accuracy vs k')
plt.xticks(k_range)
plt.grid(True)
plt.show()

best_k = k_range[k_scores.index(max(k_scores))]
print(f'Best k: {best_k}  |  CV Accuracy: {max(k_scores):.3f}')""",
    },

    "evaluation": {
        "title": "SECTION 5 -- Evaluation Metrics",
        "code": """\
# Pick best model by F1 score
best_name = max(results, key=lambda k: results[k]['f1'])
best_res = results[best_name]
print(f'Best model: {best_name}')

# Confusion Matrix
cm = confusion_matrix(y_test_fe, best_res['y_pred'])

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Died', 'Survived'],
    yticklabels=['Died', 'Survived']
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix -- {best_name}')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f'TN={tn}  FP={fp}  FN={fn}  TP={tp}')
print(f'Precision = TP/(TP+FP) = {tp/(tp+fp):.3f}  (of predicted survived, how many actually did)')
print(f'Recall    = TP/(TP+FN) = {tp/(tp+fn):.3f}  (of actual survivors, how many did we catch)')

# Full classification report
print(classification_report(y_test_fe, best_res['y_pred'], target_names=['Died', 'Survived']))

# ROC Curves for all models
plt.figure(figsize=(8, 6))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test_fe, r['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance (Random Forest)
rf_model = models['Random Forest']
importances = pd.Series(rf_model.feature_importances_, index=X_train_fe.columns)
importances.sort_values(ascending=True).plot(kind='barh', figsize=(8, 6))
plt.title('Feature Importances -- Random Forest')
plt.tight_layout()
plt.show()""",
    },

    "tuning": {
        "title": "SECTION 6 -- Hyperparameter Tuning (GridSearchCV)",
        "code": """\
# --- Tune Decision Tree ---
param_grid_dt = {
    'max_depth':         [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'criterion':         ['gini', 'entropy'],
}

grid_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid_dt,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_dt.fit(X_train_fe, y_train_fe)
print('Best DT params:', grid_dt.best_params_)
print('Best CV F1:    ', round(grid_dt.best_score_, 3))

# --- Tune k-NN ---
param_grid_knn = {
    'n_neighbors': list(range(1, 21)),
    'weights':     ['uniform', 'distance'],
    'metric':      ['euclidean', 'manhattan'],
}

grid_knn = GridSearchCV(
    KNeighborsClassifier(),
    param_grid_knn,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_knn.fit(X_train_fe_scaled, y_train_fe)
print('Best k-NN params:', grid_knn.best_params_)
print('Best CV F1:       ', round(grid_knn.best_score_, 3))

# --- Evaluate tuned models on test set ---
tuned_models = {
    'Tuned Decision Tree': (grid_dt.best_estimator_, X_test_fe),
    'Tuned k-NN':          (grid_knn.best_estimator_, X_test_fe_scaled),
}

for name, (model, X_test_input) in tuned_models.items():
    y_pred = model.predict(X_test_input)
    print(f'\\n{name}')
    print(f'  Accuracy : {accuracy_score(y_test_fe, y_pred):.3f}')
    print(f'  F1 Score : {f1_score(y_test_fe, y_pred):.3f}')
    print(f'  ROC-AUC  : {roc_auc_score(y_test_fe, model.predict_proba(X_test_input)[:, 1]):.3f}')""",
    },
}

_SECTION_ORDER = [
    "imports",
    "eda",
    "preprocessing",
    "feature_engineering",
    "model_selection",
    "evaluation",
    "tuning",
]


def print_pipeline(section: str = None) -> None:
    """Print the full Titanic ML pipeline code, or a single named section.

    Parameters
    ----------
    section : str, optional
        One of: 'imports', 'eda', 'preprocessing', 'feature_engineering',
        'model_selection', 'evaluation', 'tuning'.
        If omitted, every section is printed in order.

    Examples
    --------
    >>> from titanic_pipeline import print_pipeline
    >>> print_pipeline()                       # prints everything
    >>> print_pipeline('preprocessing')        # prints only Section 2
    >>> print_pipeline('feature_engineering')  # prints only Section 3
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
    border = "=" * 72
    print(border)
    print(f"  {sec['title']}")
    print(border)
    print(sec["code"])
