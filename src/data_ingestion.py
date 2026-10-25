"""
data_ingestion.py
------------------
Builds the company universe for the M&A Target Screener.

METHODOLOGY NOTE:
This project uses a curated list of REAL, named companies split into two
groups based on real, publicly documented outcomes:
  - ACQUIRED  = companies that were genuinely acquired/taken private in
                2021-2023 (Twitter, Slack, Splunk, VMware, etc.)
  - STANDALONE = comparable companies in similar sectors that remained
                independent public companies through the same window.

The financial FEATURES (growth rate, margins, leverage, ownership,
sentiment, etc.) are synthetically generated, not scraped from a live
filings feed. They are generated from realistic sector-conditioned
distributions so the dataset has learnable signal, structured exactly
the way real fundamental + alt-data features would look (this mirrors
how you'd ingest data from SEC EDGAR / Bloomberg / Refinitiv in
production). This is a deliberate, disclosed design choice for a
portfolio project — swap `generate_synthetic_fundamentals()` for a real
EDGAR/yfinance/Refinitiv puller and the rest of the pipeline (feature
engineering, FinBERT scoring, XGBoost model) is unchanged.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# Real companies that were acquired / taken private (2021-2023 wave)
ACQUIRED_COMPANIES = [
    ("Twitter", "Social Media", 2022),
    ("Activision Blizzard", "Gaming", 2023),
    ("VMware", "Enterprise Software", 2023),
    ("Slack Technologies", "Collaboration SaaS", 2021),
    ("Zendesk", "Customer Support SaaS", 2022),
    ("Citrix Systems", "Enterprise Software", 2022),
    ("Anaplan", "Planning SaaS", 2022),
    ("Coupa Software", "Procurement SaaS", 2023),
    ("Splunk", "Data/Security SaaS", 2023),
    ("Qualtrics International", "Experience Mgmt SaaS", 2023),
    ("Cerner Corporation", "Healthcare IT", 2022),
    ("Change Healthcare", "Healthcare IT", 2022),
    ("Nuance Communications", "AI/Speech Software", 2021),
    ("Zynga", "Gaming", 2022),
    ("SailPoint Technologies", "Identity Security SaaS", 2022),
    ("Datto Holding", "MSP Software", 2022),
    ("Ping Identity", "Identity Security SaaS", 2022),
    ("ForgeRock", "Identity Security SaaS", 2023),
    ("UserTesting", "UX Research SaaS", 2022),
    ("Mandiant", "Cybersecurity", 2022),
]

# Comparable companies that remained independent/public through the same window
STANDALONE_COMPANIES = [
    ("Adobe", "Enterprise Software"),
    ("Salesforce", "Collaboration SaaS"),
    ("ServiceNow", "Enterprise Software"),
    ("Workday", "Enterprise Software"),
    ("Intuit", "Fintech SaaS"),
    ("Autodesk", "Enterprise Software"),
    ("Synopsys", "Semiconductor Software"),
    ("Cadence Design Systems", "Semiconductor Software"),
    ("Fortinet", "Cybersecurity"),
    ("Palo Alto Networks", "Cybersecurity"),
    ("CrowdStrike Holdings", "Cybersecurity"),
    ("Datadog", "Data/Security SaaS"),
    ("Atlassian", "Collaboration SaaS"),
    ("HubSpot", "Marketing SaaS"),
    ("Paycom Software", "HR SaaS"),
    ("Paylocity Holding", "HR SaaS"),
    ("Veeva Systems", "Healthcare IT"),
    ("Trimble", "Enterprise Software"),
    ("Roper Technologies", "Diversified Software"),
    ("Take-Two Interactive", "Gaming"),
]

SECTOR_PROFILE = {
    # (growth_mean, margin_mean, leverage_mean) rough sector priors
    "Social Media": (0.10, 0.20, 0.30),
    "Gaming": (0.08, 0.22, 0.25),
    "Enterprise Software": (0.14, 0.28, 0.35),
    "Collaboration SaaS": (0.22, 0.10, 0.20),
    "Customer Support SaaS": (0.18, 0.08, 0.20),
    "Planning SaaS": (0.20, 0.05, 0.15),
    "Procurement SaaS": (0.19, 0.07, 0.20),
    "Data/Security SaaS": (0.25, 0.12, 0.25),
    "Experience Mgmt SaaS": (0.17, 0.06, 0.20),
    "Healthcare IT": (0.11, 0.18, 0.40),
    "AI/Speech Software": (0.13, 0.15, 0.30),
    "Identity Security SaaS": (0.21, 0.09, 0.20),
    "MSP Software": (0.16, 0.11, 0.25),
    "Cybersecurity": (0.24, 0.14, 0.20),
    "UX Research SaaS": (0.15, 0.02, 0.15),
    "Fintech SaaS": (0.15, 0.30, 0.25),
    "Semiconductor Software": (0.13, 0.32, 0.20),
    "Marketing SaaS": (0.20, 0.10, 0.20),
    "HR SaaS": (0.19, 0.15, 0.15),
    "Diversified Software": (0.10, 0.25, 0.35),
}


def _sample_features(sector: str, was_acquired: int) -> dict:
    growth_mean, margin_mean, lev_mean = SECTOR_PROFILE.get(sector, (0.15, 0.15, 0.25))

    # Acquired companies historically skew toward: decelerating growth,
    # thinner margins, higher activist/ownership churn, wider valuation gap
    # to peers, and elevated M&A-speculation media mentions in the lead-up.
    acq_growth_shift = -0.05 if was_acquired else 0.0
    acq_margin_shift = -0.04 if was_acquired else 0.0
    acq_gap_shift = 0.12 if was_acquired else 0.0
    acq_activist_boost = 0.35 if was_acquired else 0.08
    acq_rumor_boost = 2.5 if was_acquired else 1

    revenue_growth_yoy = float(np.clip(rng.normal(growth_mean + acq_growth_shift, 0.08), -0.15, 0.55))
    ebitda_margin = float(np.clip(rng.normal(margin_mean + acq_margin_shift, 0.08), -0.20, 0.55))
    debt_to_equity = float(np.clip(rng.normal(lev_mean, 0.12), 0.0, 1.3))
    cash_to_market_cap = float(np.clip(rng.normal(0.09, 0.05), 0.01, 0.35))
    pe_ratio = float(np.clip(rng.normal(28, 12), 6, 80))
    price_to_sales = float(np.clip(rng.normal(6.5, 3.5), 0.8, 22))
    rule_of_40 = revenue_growth_yoy * 100 + ebitda_margin * 100
    insider_ownership_pct = float(np.clip(rng.normal(9, 6), 0.5, 35))
    institutional_ownership_pct = float(np.clip(rng.normal(75, 10), 30, 96))
    short_interest_pct = float(np.clip(rng.normal(3.2, 2.0), 0.2, 15))
    analyst_target_gap_pct = float(np.clip(rng.normal(15 + acq_gap_shift * 100, 10), -10, 55))
    activist_investor_present = int(rng.random() < acq_activist_boost)
    years_public = float(np.clip(rng.normal(9, 5), 1, 30))
    ma_rumor_mentions_6m = int(np.clip(rng.poisson(1 + acq_rumor_boost), 0, 25))

    return dict(
        revenue_growth_yoy=round(revenue_growth_yoy, 4),
        ebitda_margin=round(ebitda_margin, 4),
        debt_to_equity=round(debt_to_equity, 3),
        cash_to_market_cap=round(cash_to_market_cap, 4),
        pe_ratio=round(pe_ratio, 2),
        price_to_sales=round(price_to_sales, 2),
        rule_of_40=round(rule_of_40, 2),
        insider_ownership_pct=round(insider_ownership_pct, 2),
        institutional_ownership_pct=round(institutional_ownership_pct, 2),
        short_interest_pct=round(short_interest_pct, 2),
        analyst_target_gap_pct=round(analyst_target_gap_pct, 2),
        activist_investor_present=activist_investor_present,
        years_public=round(years_public, 1),
        ma_rumor_mentions_6m=ma_rumor_mentions_6m,
    )


def build_dataset() -> pd.DataFrame:
    rows = []
    for name, sector, year in ACQUIRED_COMPANIES:
        feats = _sample_features(sector, was_acquired=1)
        rows.append({"company": name, "sector": sector, "was_acquired": 1, "acquired_year": year, **feats})
    for name, sector in STANDALONE_COMPANIES:
        feats = _sample_features(sector, was_acquired=0)
        rows.append({"company": name, "sector": sector, "was_acquired": 0, "acquired_year": None, **feats})

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


if __name__ == "__main__":
    df = build_dataset()
    out_path = "data/companies_fundamentals.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} companies to {out_path}")
    print(df["was_acquired"].value_counts())
