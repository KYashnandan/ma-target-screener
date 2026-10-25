"""
news_headlines.py
------------------
Generates a small sample news corpus (3-5 headlines per company) used as
input text for the FinBERT sentiment stage. In production this would be
pulled from a news API (NewsAPI, Benzinga, RavenPack) filtered to the
trailing 6 months; here it's a hand-written sample corpus covering the
kinds of headlines that typically precede/accompany M&A activity
(activist stakes, strategic review announcements, rumor coverage) versus
neutral/positive standalone-company coverage.
"""

import pandas as pd
from data_ingestion import build_dataset

MA_SIGNAL_TEMPLATES = [
    "{company} shares jump on report of takeover interest from private equity",
    "{company} confirms it is exploring strategic alternatives, including a sale",
    "Activist investor builds stake in {company}, pushes for board changes",
    "{company} reportedly in early talks with potential acquirer",
    "Analysts flag {company} as an M&A target amid sector consolidation",
]

NEUTRAL_TEMPLATES = [
    "{company} reports quarterly revenue in line with analyst expectations",
    "{company} announces new product roadmap at annual conference",
    "{company} CEO reiterates long-term standalone growth strategy",
    "{company} beats earnings estimates, raises full-year guidance",
    "{company} expands partnership network in enterprise segment",
]


def build_headlines() -> pd.DataFrame:
    df = build_dataset()
    rows = []
    for _, r in df.iterrows():
        company = r["company"]
        was_acquired = r["was_acquired"]
        templates = MA_SIGNAL_TEMPLATES if was_acquired else NEUTRAL_TEMPLATES
        # acquired companies still get 1 neutral headline mixed in (realistic noise)
        # and standalone companies occasionally get a speculative headline
        n_signal = 4 if was_acquired else 1
        chosen = templates[:n_signal]
        other = NEUTRAL_TEMPLATES if was_acquired else MA_SIGNAL_TEMPLATES
        chosen += other[:1]
        for t in chosen:
            rows.append({"company": company, "headline": t.format(company=company)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_headlines()
    df.to_csv("data/news_headlines.csv", index=False)
    print(f"Wrote {len(df)} headlines for {df['company'].nunique()} companies")
