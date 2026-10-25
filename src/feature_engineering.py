"""
feature_engineering.py
-----------------------
Merges fundamentals + aggregated news sentiment into the final feature
matrix used to train the XGBoost classifier. Also one-hot encodes
sector and adds a couple of engineered ratio features that are commonly
used in real M&A screening (valuation gap, leverage-adjusted cash
position, growth-vs-market re-rating potential).
"""

import pandas as pd


def build_feature_matrix() -> pd.DataFrame:
    fundamentals = pd.read_csv("data/companies_fundamentals.csv")
    sentiment = pd.read_csv("data/company_sentiment.csv")

    df = fundamentals.merge(sentiment, on="company", how="left")

    # Engineered features
    df["cash_adjusted_leverage"] = df["debt_to_equity"] - df["cash_to_market_cap"]
    df["valuation_discount_signal"] = df["analyst_target_gap_pct"] / 100.0
    df["growth_margin_gap"] = df["revenue_growth_yoy"] - df["ebitda_margin"]

    sector_dummies = pd.get_dummies(df["sector"], prefix="sector")
    df = pd.concat([df, sector_dummies], axis=1)

    return df


FEATURE_COLUMNS = [
    "revenue_growth_yoy", "ebitda_margin", "debt_to_equity", "cash_to_market_cap",
    "pe_ratio", "price_to_sales", "rule_of_40", "insider_ownership_pct",
    "institutional_ownership_pct", "short_interest_pct", "analyst_target_gap_pct",
    "activist_investor_present", "years_public", "ma_rumor_mentions_6m",
    "avg_news_sentiment", "min_news_sentiment",
    "cash_adjusted_leverage", "valuation_discount_signal", "growth_margin_gap",
]


if __name__ == "__main__":
    df = build_feature_matrix()
    df.to_csv("data/feature_matrix.csv", index=False)
    print(f"Built feature matrix: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Base features used for modeling: {len(FEATURE_COLUMNS)}")
