"""
predict_targets.py
--------------------
Loads the trained model and scores the FULL company universe (including
training rows — this is a portfolio screener demo, not a live trading
signal) to produce a ranked "acquisition likelihood" leaderboard, the
core deliverable of the tool.
"""

import pandas as pd
import xgboost as xgb

from feature_engineering import build_feature_matrix, FEATURE_COLUMNS


def main():
    df = build_feature_matrix()
    X = df[FEATURE_COLUMNS]

    model = xgb.XGBClassifier()
    model.load_model("models/xgb_ma_screener.json")

    df["acquisition_score"] = model.predict_proba(X)[:, 1]
    ranked = df[[
        "company", "sector", "acquisition_score", "was_acquired",
        "ma_rumor_mentions_6m", "avg_news_sentiment", "analyst_target_gap_pct",
        "activist_investor_present",
    ]].sort_values("acquisition_score", ascending=False)

    ranked.to_csv("outputs/target_rankings.csv", index=False)
    print(ranked.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
