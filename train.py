
"""Train and evaluate placement prediction models.

Uses group-aware splitting so identical input profiles stay together
during hold-out evaluation and cross-validation.
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42

DATA_PATH = Path("data/train.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "PlacementStatus"

NUM = [
    "CGPA",
    "Internships",
    "Projects",
    "Workshops/Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "SSC_Marks",
    "HSC_Marks",
]

CAT = [
    "ExtracurricularActivities",
    "PlacementTraining",
]

FEATURES = NUM + CAT
EXPECTED_LABELS = {"Placed", "NotPlaced"}


# --------------------------------------------------
# Load and validate data
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}. "
        "Check that data/train.csv exists."
    )

df = pd.read_csv(DATA_PATH)

required_columns = FEATURES + [TARGET]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Drop rows with missing target labels.
df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype(str).str.strip()

unexpected_labels = set(df[TARGET].unique()) - EXPECTED_LABELS

if unexpected_labels:
    raise ValueError(
        f"Unexpected target labels: {unexpected_labels}"
    )

if df.empty:
    raise ValueError("No rows remain after target validation.")

# Encode target explicitly: NotPlaced = 0, Placed = 1.
X = df[FEATURES].copy()
y = df[TARGET].map({
    "NotPlaced": 0,
    "Placed": 1,
}).astype(int)

# Build group IDs from input features only.
# Identical input profiles will belong to the same group.
groups = pd.util.hash_pandas_object(
    X, index=False
).reset_index(drop=True)

# Report possible data-quality issues.
exact_duplicates = int(df.duplicated().sum())
repeated_profiles = int(X.duplicated(keep=False).sum())

profile_label_counts = (
    pd.concat([X, y.rename(TARGET)], axis=1)
    .groupby(FEATURES, dropna=False)[TARGET]
    .nunique()
)

conflicting_profiles = int(
    (profile_label_counts > 1).sum()
)

print("=" * 60)
print("DATASET VALIDATION")
print("=" * 60)
print("Dataset shape:", df.shape)
print("\nPlacement distribution:")
print(df[TARGET].value_counts())
print("\nMissing feature values:")
print(X.isna().sum())
print("\nExact duplicate rows:", exact_duplicates)
print("Rows with repeated input profiles:", repeated_profiles)
print(
    "Profiles with conflicting labels:",
    conflicting_profiles,
)

print(
    "\nNote: duplicate and conflicting profiles are retained. "
    "Identical profiles are grouped during splitting."
)


# --------------------------------------------------
# Exploratory data analysis
# --------------------------------------------------

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

sns.countplot(
    data=df,
    x=TARGET,
    ax=axes[0, 0],
)
axes[0, 0].set_title("Placement Distribution")

sns.boxplot(
    data=df,
    x=TARGET,
    y="CGPA",
    ax=axes[0, 1],
)
axes[0, 1].set_title("CGPA by Placement Status")

placement_rate = (
    pd.DataFrame({
        "Internships": df["Internships"],
        "Placed": y,
    })
    .groupby("Internships", observed=True)["Placed"]
    .mean()
)

placement_rate.plot(
    kind="bar",
    ax=axes[1, 0],
)
axes[1, 0].set_title("Placement Rate by Internships")
axes[1, 0].set_ylabel("Placement Rate")
axes[1, 0].set_ylim(0, 1)

training_rate = (
    pd.DataFrame({
        "PlacementTraining": df["PlacementTraining"],
        "Placed": y,
    })
    .groupby("PlacementTraining", observed=True)["Placed"]
    .mean()
)

training_rate.plot(
    kind="bar",
    ax=axes[1, 1],
)
axes[1, 1].set_title("Placement Rate by Placement Training")
axes[1, 1].set_ylabel("Placement Rate")
axes[1, 1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(REPORT_DIR / "eda.png", dpi=120)
plt.close()

# Correlations among numerical features.
corr_df = X[NUM].copy()
corr_df[TARGET] = y

plt.figure(figsize=(9, 7))
sns.heatmap(
    corr_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
)
plt.title("Numerical Feature Correlations")
plt.tight_layout()
plt.savefig(REPORT_DIR / "correlation.png", dpi=120)
plt.close()


# --------------------------------------------------
# Group-aware train/test split
# --------------------------------------------------

# A profile must not appear in both train and test sets.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()
y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

train_groups = groups.iloc[train_idx]
test_groups = groups.iloc[test_idx]

shared_groups = set(train_groups) & set(test_groups)

if shared_groups:
    raise RuntimeError(
        "Data leakage check failed: some profiles occur in both sets."
    )

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "The group-aware split did not place both classes in "
        "training and testing. Try another SEED."
    )

print("\n" + "=" * 60)
print("GROUP-AWARE HOLD-OUT SPLIT")
print("=" * 60)
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Training class counts:")
print(y_train.value_counts().sort_index())
print("Test class counts:")
print(y_test.value_counts().sort_index())
print("Shared input profiles:", len(shared_groups))


# --------------------------------------------------
# Preprocessing pipeline
# --------------------------------------------------

def make_pipeline(model, scale_numeric=False):
    """Create preprocessing and classification pipeline."""

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUM),
        ("categorical", categorical_pipeline, CAT),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# --------------------------------------------------
# Models
# --------------------------------------------------

models = {
    "Logistic Regression": make_pipeline(
        LogisticRegression(
            max_iter=1500,
            random_state=SEED,
        ),
        scale_numeric=True,
    ),

    "Random Forest": make_pipeline(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=SEED,
            n_jobs=-1,
        )
    ),

    "XGBoost": make_pipeline(
        XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        )
    ),
}


# --------------------------------------------------
# Group-aware cross-validation and evaluation
# --------------------------------------------------

# Each group remains entirely within one fold.
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

results = []

for name, pipeline in models.items():
    print(f"\nTraining {name}...")

    # Cross-validation is performed only on training data.
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        groups=train_groups,
        cv=cv,
        scoring="f1",
        n_jobs=1,
    )

    # Fit on all training data.
    pipeline.fit(X_train, y_train)

    # Evaluate on untouched hold-out data.
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test, probabilities
        ),
        "CV F1 mean": cv_scores.mean(),
        "CV F1 std": cv_scores.std(),
    })

results_df = pd.DataFrame(results).set_index("Model")
results_df = results_df.round(4)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string())

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv"
)

# Select using held-out F1 for consistency with the original project.
# For a final unbiased model comparison, avoid repeatedly tuning
# decisions against the same hold-out test set.
best_name = results_df["F1"].idxmax()
best_model = models[best_name]

print("\nSelected model by hold-out F1:", best_name)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1],
)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["NotPlaced", "Placed"],
    yticklabels=["NotPlaced", "Placed"],
)
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "confusion_matrix.png",
    dpi=120,
)
plt.close()


# --------------------------------------------------
# Permutation feature importance
# --------------------------------------------------

permutation = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=5,
    random_state=SEED,
    n_jobs=1,
)

importance = pd.Series(
    permutation.importances_mean,
    index=FEATURES,
).sort_values()

importance.to_csv(
    REPORT_DIR / "feature_importance.csv",
    header=["F1 importance"],
)

plt.figure(figsize=(8, 6))
importance.plot(kind="barh")
plt.title("Permutation Feature Importance — Test Set")
plt.xlabel("Mean decrease in F1 when shuffled")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "feature_importance.png",
    dpi=120,
)
plt.close()


# --------------------------------------------------
# Save model and metadata
# --------------------------------------------------

joblib.dump(
    best_model,
    MODEL_DIR / "best_model.joblib",
)

meta = {
    "best_model": best_name,
    "target_mapping": {
        "NotPlaced": 0,
        "Placed": 1,
    },
    "features": FEATURES,
    "numeric_features": NUM,
    "categorical_features": CAT,
    "positive_class": "Placed",
    "metrics": {
        key: float(value)
        for key, value in results_df.loc[best_name].items()
    },
    "data_source": str(DATA_PATH),
    "split_strategy": "GroupShuffleSplit",
    "cross_validation_strategy": "StratifiedGroupKFold",
    "duplicate_profiles_kept": True,
    "data_note": (
        "Identical input profiles are kept in the same group during "
        "hold-out splitting and cross-validation. Conflicting labels "
        "are retained. The dataset's provenance and independent "
        "real-world predictive validity have not been established."
    ),
}

with (
    MODEL_DIR / "meta.json"
).open("w", encoding="utf-8") as file:
    json.dump(meta, file, indent=2)

print("\nSaved model: models/best_model.joblib")
print("Saved metadata: models/meta.json")
print("Saved reports in:", REPORT_DIR.resolve())
