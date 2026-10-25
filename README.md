# DealRadar: Macro-Resilient M&A Target Screener

Scores public companies on how likely they are to become acquisition targets, using SEC financials and news sentiment.

**Stack:** Python, pandas, XGBoost, FinBERT (`ProsusAI/finbert`), SEC EDGAR XBRL API, scikit-learn.

## Results (real data, 40 companies)

| Metric | Value |
|---|---|
| Repeated 5-fold CV AUC | **0.81 +/- 0.06** |
| Companies | 40 (20 acquired 2021-2023, 20 comparable independents) |
| Top features | operating margin, net margin, share of negative headlines, revenue growth, R&D intensity, average sentiment |

Run with `python src/real_pipeline.py` (needs internet + torch, e.g. Colab).

## Data

- **Labels:** real outcomes. 20 companies acquired or taken private in 2021-2023 (Twitter, Slack, Splunk, VMware, Activision Blizzard, ...) vs 20 comparable software/gaming companies that stayed public.
- **Financials:** SEC EDGAR XBRL filings for the fiscal year before the deal closed (2022 for independents): revenue growth, operating and net margin, debt-to-equity, cash-to-assets, R&D intensity, size.
- **News:** up to 25 Google News headlines per company for the same year, scored by FinBERT (positive minus negative probability). Features: average, minimum, and share of negative headlines.
- **Evaluation:** repeated stratified 5-fold CV; rankings use out-of-fold scores.

## Limitations

- Only 40 companies, so the AUC has a wide error band and is not a production claim.
- Headline volume varies by company (Google News RSS), and for some deals the snapshot year overlaps the announcement date, so a little leakage is possible.
- No price, ownership or valuation features (they need paid data).

## Next steps

Larger universe (500+), time-based validation, exact pre-announcement windows, a licensed news feed, probability calibration, sector-relative valuation features.

## Architecture (offline demo)

```
                 ┌────────────────────┐
                 │  Company Universe  │  (fundamentals: growth, margins,
                 │  data_ingestion.py │   leverage, ownership, valuation)
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │   News Headlines   │  sample corpus per company
                 │ news_headlines.py  │  (swap for NewsAPI/RavenPack feed)
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │  FinBERT Sentiment │  ProsusAI/finbert -> signed
                 │ sentiment_finbert  │  sentiment score per headline
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │ Feature Engineering│  merge + engineer ratios
                 │feature_engineering │  (valuation gap, leverage-adj cash)
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │   XGBoost Model    │  binary classifier:
                 │   train_model.py   │  P(company gets acquired)
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │  Ranked Target     │  outputs/target_rankings.csv
                 │ predict_targets.py │  + feature importance + ROC
                 └────────────────────┘
```

## Offline demo (synthetic data)

The original pipeline (`data_ingestion.py` through `predict_targets.py`, run via `run_pipeline.sh`) uses real labels but synthetic financials, template headlines and a keyword sentiment fallback. It exists to show the full pipeline offline and its AUC of 1.0 on a 10-company split is not meaningful. The real-data results above are the ones to cite.
