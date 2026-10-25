"""
train_model.py
---------------
Trains an XGBoost binary classifier to predict P(company gets acquired)
from the fundamentals + sentiment feature matrix, evaluates it with a
stratified train/test split (small-N dataset, so this is a leave-out
validation rather than a claim of production-grade generalization —
disclosed explicitly in the README), and saves:
  - models/xgb_ma_screener.json   (trained model)
  - outputs/feature_importance.csv
  - outputs/metrics.json
  - outputs/roc_curve.png
  - outputs/feature_importance.png
"""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score
import xgboost as xgb

from feature_engineering import build_feature_matrix, FEATURE_COLUMNS

RANDOM_SEED = 42


def main():
    df = build_feature_matrix()
    X = df[FEATURE_COLUMNS]
    y = df["was_acquired"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_auc": round(float(roc_auc_score(y_test, proba_test)), 3),
        "test_accuracy": round(float(accuracy_score(y_test, pred_test)), 3),
        "test_precision": round(float(precision_score(y_test, pred_test, zero_division=0)), 3),
        "test_recall": round(float(recall_score(y_test, pred_test, zero_division=0)), 3),
        "test_f1": round(float(f1_score(y_test, pred_test, zero_division=0)), 3),
    }

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importance
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv("outputs/feature_importance.csv", index=False)

    # Plots
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"XGBoost (AUC={metrics['test_auc']})", color="#2c3e50", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("M&A Target Screener - ROC Curve (test set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("outputs/roc_curve.png", dpi=150)
    plt.close()

    top_n = importance.head(10)
    plt.figure(figsize=(7, 5))
    plt.barh(top_n["feature"][::-1], top_n["importance"][::-1], color="#2980b9")
    plt.xlabel("XGBoost Feature Importance")
    plt.title("Top Drivers of Acquisition-Target Score")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()

    model.save_model("models/xgb_ma_screener.json")

    print("Metrics:", json.dumps(metrics, indent=2))
    print("\nTop 5 features by importance:")
    print(importance.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
