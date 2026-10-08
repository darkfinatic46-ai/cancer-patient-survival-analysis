# Cancer Patient Survival Analysis & Outcome Prediction
# Portfolio Project | Junior Data Analyst
#
# IMPORTANT:
# This project is for educational/data-analytics purposes only.
# It must not be used for clinical diagnosis or treatment decisions.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# =========================================================
# 1. LOAD DATA
# =========================================================

DATA_PATH = "india_cancer_patients_2022_2025 (2).csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

display(df.head())

# =========================================================
# 2. DATA QUALITY CHECK
# =========================================================

print("\nData types:")
display(df.dtypes)

print("\nMissing values:")
display(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nDescriptive statistics:")
display(df.describe(include="all").T)

# Remove exact duplicates
df = df.drop_duplicates().copy()

# Convert diagnosis date
if "Diagnosis_Date" in df.columns:
    df["Diagnosis_Date"] = pd.to_datetime(
        df["Diagnosis_Date"], errors="coerce"
    )

# =========================================================
# 3. DATASET OVERVIEW
# =========================================================

print("\nStatus distribution:")
display(df["Status"].value_counts())

print("\nCancer type distribution:")
display(df["Cancer_Type"].value_counts())

print("\nStage distribution:")
display(df["Stage"].value_counts())

print("\nTreatment distribution:")
display(df["Treatment_Type"].value_counts())

# =========================================================
# 4. EXPLORATORY DATA ANALYSIS
# =========================================================

# Patient status
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Status")
plt.title("Cancer Patient Status")
plt.xlabel("Patient Status")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

# Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", bins=25, kde=True)
plt.title("Age Distribution of Cancer Patients")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

# Cancer type distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y="Cancer_Type",
              order=df["Cancer_Type"].value_counts().index)
plt.title("Cancer Type Distribution")
plt.xlabel("Number of Patients")
plt.ylabel("Cancer Type")
plt.tight_layout()
plt.show()

# Stage distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Stage",
              order=df["Stage"].value_counts().index)
plt.title("Cancer Stage Distribution")
plt.xlabel("Cancer Stage")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

# Treatment distribution
plt.figure(figsize=(11, 6))
sns.countplot(data=df, y="Treatment_Type",
              order=df["Treatment_Type"].value_counts().index)
plt.title("Treatment Type Distribution")
plt.xlabel("Number of Patients")
plt.ylabel("Treatment Type")
plt.tight_layout()
plt.show()

# Gender vs Status
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Gender", hue="Status")
plt.title("Gender vs Patient Status")
plt.xlabel("Gender")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

# Cancer Type vs Status
plt.figure(figsize=(12, 6))
sns.countplot(
    data=df,
    x="Cancer_Type",
    hue="Status",
    order=df["Cancer_Type"].value_counts().index
)
plt.title("Cancer Type vs Patient Status")
plt.xlabel("Cancer Type")
plt.ylabel("Number of Patients")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Stage vs Status
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Stage", hue="Status")
plt.title("Cancer Stage vs Patient Status")
plt.xlabel("Cancer Stage")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

# Treatment vs Status
plt.figure(figsize=(12, 6))
sns.countplot(
    data=df,
    x="Treatment_Type",
    hue="Status",
    order=df["Treatment_Type"].value_counts().index
)
plt.title("Treatment Type vs Patient Status")
plt.xlabel("Treatment Type")
plt.ylabel("Number of Patients")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# =========================================================
# 5. SURVIVAL ANALYSIS
# =========================================================

survival_by_status = (
    df.groupby("Status")["Survival_Months"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

print("Survival statistics by patient status:")
display(survival_by_status)

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Status", y="Survival_Months")
plt.title("Survival Months by Patient Status")
plt.xlabel("Patient Status")
plt.ylabel("Survival Months")
plt.tight_layout()
plt.show()

# Average survival by cancer type
survival_cancer = (
    df.groupby("Cancer_Type")["Survival_Months"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
sns.barplot(x=survival_cancer.values, y=survival_cancer.index)
plt.title("Average Survival Months by Cancer Type")
plt.xlabel("Average Survival Months")
plt.ylabel("Cancer Type")
plt.tight_layout()
plt.show()

# =========================================================
# 6. OUTLIER ANALYSIS
# =========================================================

plt.figure(figsize=(10, 5))
sns.boxplot(data=df[["Age", "Survival_Months"]])
plt.title("Outlier Inspection: Age and Survival Months")
plt.tight_layout()
plt.show()

# IQR-based outlier counts
for col in ["Age", "Survival_Months"]:
    if col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"{col} outliers: {len(outliers)}")

# =========================================================
# 7. CORRELATION ANALYSIS
# =========================================================

numeric_df = df.select_dtypes(include=np.number)

if numeric_df.shape[1] >= 2:
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="Blues"
    )
    plt.title("Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

# =========================================================
# 8. GEOGRAPHICAL ANALYSIS
# =========================================================

if "State" in df.columns:
    state_counts = df["State"].value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=state_counts.values, y=state_counts.index)
    plt.title("Cancer Patients by State")
    plt.xlabel("Number of Patients")
    plt.ylabel("State")
    plt.tight_layout()
    plt.show()

# =========================================================
# 9. MACHINE LEARNING
# =========================================================
#
# Target: Status
#
# IMPORTANT:
# Survival_Months is excluded from the predictive model because it
# can be information that becomes available after/alongside the
# patient outcome and may create target leakage.
#
# The analysis section still uses Survival_Months for descriptive
# survival analysis.

TARGET = "Status"

candidate_features = [
    "Age",
    "Gender",
    "State",
    "City",
    "Cancer_Type",
    "Stage",
    "Treatment_Type",
]

features = [c for c in candidate_features if c in df.columns]

X = df[features].copy()
y = df[TARGET].copy()

print("\nML Features:")
print(features)

print("\nTarget distribution:")
display(y.value_counts())

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# =========================================================
# 10. MODEL COMPARISON
# =========================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        random_state=42
    ),
    "SVM": SVC(random_state=42),
}

results = []

for name, estimator in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results.append({
        "Model": name,
        "Accuracy": accuracy
    })

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions))

results_df = (
    pd.DataFrame(results)
    .sort_values("Accuracy", ascending=False)
)

print("\nModel comparison:")
display(results_df)

plt.figure(figsize=(9, 5))
sns.barplot(data=results_df, x="Accuracy", y="Model")
plt.title("Model Accuracy Comparison")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()

# =========================================================
# 11. RANDOM FOREST HYPERPARAMETER TUNING
# =========================================================

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(random_state=42)
        ),
    ]
)

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
}

random_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)

random_search.fit(X_train, y_train)

print("Best parameters:")
print(random_search.best_params_)

print("\nBest cross-validation accuracy:")
print(round(random_search.best_score_, 4))

best_model = random_search.best_estimator_

best_predictions = best_model.predict(X_test)

print("\nTuned Random Forest test accuracy:")
print(round(accuracy_score(y_test, best_predictions), 4))

print("\nClassification report:")
print(classification_report(y_test, best_predictions))

# Confusion matrix
cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Tuned Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# =========================================================
# 12. EXPORT RESULTS
# =========================================================

results_df.to_csv("model_comparison.csv", index=False)
survival_by_status.to_csv("survival_by_status.csv")

print("\nAnalysis complete.")
print("Saved:")
print("- model_comparison.csv")
print("- survival_by_status.csv")

# =========================================================
# 13. DISCLAIMER
# =========================================================

print(
    "\nDISCLAIMER: This project is for educational and portfolio "
    "purposes. Predictions must not be used for clinical diagnosis "
    "or treatment decisions."
)
