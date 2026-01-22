"""
precision_oncology_pipeline.py
Breast Cancer Precision Oncology Pipeline
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


# ---------------------------------------------------------
# 1. LOAD CSV
# ---------------------------------------------------------

def load_variant_data(path="variants_breast_cancer_raw.csv"):
    """
    Load the variant table from CSV.

    Expected columns:
        variant_id, gene, consequence, population_AF, label_actionable
    """
    df = pd.read_csv(path)
    required_cols = [
        "variant_id", "gene", "consequence",
        "population_AF", "label_actionable"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")
    return df


# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------

def engineer_features(df):
    """
    Create medically meaningful features:
      - is_truncating: 1 if frameshift/stop_gained/nonsense
      - is_rare: 1 if population_AF < 0.01
      - is_cancer_gene: 1 if BRCA1/2, TP53, PIK3CA, ERBB2
      - One-hot encoded consequence types
    """
    df = df.copy()

    # Normalise consequence text
    df["consequence"] = df["consequence"].str.lower()

    # 2.1 Truncating = high-impact variants
    truncating_terms = ["frameshift", "stop_gained", "nonsense"]
    df["is_truncating"] = df["consequence"].isin(truncating_terms).astype(int)

    # 2.2 Rarity based on population allele frequency
    df["population_AF"] = df["population_AF"].fillna(1.0)
    df["is_rare"] = (df["population_AF"] < 0.01).astype(int)

    # 2.3 Known breast cancer genes
    cancer_genes = ["BRCA1", "BRCA2", "TP53", "PIK3CA", "ERBB2"]
    df["is_cancer_gene"] = df["gene"].isin(cancer_genes).astype(int)

    # 2.4 One-hot encode consequence categories
    consq_dummies = pd.get_dummies(df["consequence"], prefix="consq")
    df = pd.concat([df, consq_dummies], axis=1)

    feature_cols = ["is_truncating", "is_rare", "is_cancer_gene"] + list(consq_dummies.columns)

    X = df[feature_cols].astype(float)
    y = df["label_actionable"].astype(int)


    return X, y, df, feature_cols


# ---------------------------------------------------------
# 3. TRAIN MODEL
# ---------------------------------------------------------

def train_model(X, y):
    """
    Train a Random Forest classifier to predict label_actionable.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print("\n=== MODEL PERFORMANCE ===")
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, (X_test, y_test, y_prob)


# ---------------------------------------------------------
# 4. FLAG CLINICALLY ACTIONABLE VARIANTS
# ---------------------------------------------------------

def flag_clinically_actionable(df_feats, X, model, threshold=0.8):
    """
    Use model + rules to flag variants as clinically_actionable_flag.

    Conditions:
      - High predicted probability (>= threshold)
      - AND rare (is_rare == 1)
      - AND in a key breast cancer gene (is_cancer_gene == 1)
    """
    df = df_feats.copy()
    df["pred_prob_actionable"] = model.predict_proba(X)[:, 1]

    df["clinically_actionable_flag"] = (
        (df["pred_prob_actionable"] >= threshold) &
        (df["is_rare"] == 1) &
        (df["is_cancer_gene"] == 1)
    )

    return df


# ---------------------------------------------------------
# 5. MAIN PIPELINE ENTRY POINT
# ---------------------------------------------------------

def main():
    print("Loading variant data...")
    df_raw = load_variant_data()

    print("Engineering features...")
    X, y, df_feats, feature_cols = engineer_features(df_raw)
    print("Features used:", feature_cols)

    print("Training model...")
    model, (X_test, y_test, y_prob) = train_model(X, y)

    print("Flagging clinically actionable variants...")
    results_df = flag_clinically_actionable(df_feats, X, model, threshold=0.8)

    out_path = "breast_cancer_actionable_variants_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

    print("\nSample of flagged variants:")
    print(results_df[results_df["clinically_actionable_flag"] == True].head())


if __name__ == "__main__":
    main()
