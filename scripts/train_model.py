"""Train infidelity prediction model using RandomForest + SHAP.

Reads from DuckDB marts_infidelity_features table, trains a RandomForestClassifier,
computes SHAP values, and saves the model + explainer.
"""
import os
import pickle
import numpy as np
import pandas as pd
import duckdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import shap

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "infidelity_predictor.duckdb")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models_ml")

FEATURE_COLUMNS = [
    "age",
    "education_years",
    "religiousness",
    "occupation",
    "honesty_humility",
    "emotionality",
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "openness",
    "years_in_relationship",
    "has_children",
    "satisfaction_rating",
    "love_rating",
    "desire_rating",
]

TARGET = "had_affair"


def load_data():
    """Load training data from DuckDB."""
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.sql("SELECT * FROM main_marts.marts_infidelity_features").fetchdf()
    con.close()
    print(f"Loaded {len(df)} rows from marts_infidelity_features")
    print(f"  Data sources: {df['data_source'].value_counts().to_dict()}")
    print(f"  Target distribution: {df[TARGET].value_counts().to_dict()}")
    return df


def prepare_features(df):
    """Prepare feature matrix and target vector."""
    # Select available features
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    print(f"  Using {len(available)} features: {available}")

    X = df[available].copy()

    # Convert boolean to int
    for col in X.columns:
        if X[col].dtype == "bool":
            X[col] = X[col].astype(float)

    # Impute missing values with median
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    y = df[TARGET].astype(int)

    # Remove rows where target is NaN
    valid_mask = y.notna()
    X_imputed = X_imputed[valid_mask]
    y = y[valid_mask]

    print(f"  Training samples: {len(y)} (positive: {y.sum()}, negative: {(~y.astype(bool)).sum()})")
    return X_imputed, y, imputer, available


def train_model(X, y):
    """Train RandomForest with cross-validation."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    print(f"\n  Cross-validation AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    f1_scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    print(f"  Cross-validation F1:  {f1_scores.mean():.3f} (+/- {f1_scores.std():.3f})")

    # Train final model on all data
    model.fit(X, y)

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n  Feature importance (top 10):")
    for feat, imp in importance.head(10).items():
        print(f"    {feat}: {imp:.4f}")

    return model


def compute_shap(model, X):
    """Compute SHAP explainer."""
    print("\n  Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    # Compute on a sample for efficiency
    sample = X.sample(min(500, len(X)), random_state=42)
    shap_values = explainer.shap_values(sample)
    print(f"  SHAP values computed for {len(sample)} samples")
    return explainer


def save_artifacts(model, explainer, imputer, feature_names):
    """Save model, explainer, and metadata."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    artifacts = {
        "model": model,
        "explainer": explainer,
        "imputer": imputer,
        "feature_names": feature_names,
    }

    output_path = os.path.join(MODEL_DIR, "model.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\n  Saved model artifacts to {output_path}")


def main():
    print("=" * 60)
    print("Training Infidelity Prediction Model")
    print("=" * 60)

    df = load_data()
    X, y, imputer, feature_names = prepare_features(df)
    model = train_model(X, y)
    explainer = compute_shap(model, X)
    save_artifacts(model, explainer, imputer, feature_names)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
