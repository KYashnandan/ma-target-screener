# M&A Target Screener

Predicts which public companies are likely acquisition targets by combining
company fundamentals with news-sentiment / alt-data signals, scored by an
XGBoost classifier.

**Stack:** Python, pandas, XGBoost, FinBERT (HuggingFace `ProsusAI/finbert`), scikit-learn, matplotlib.

## Why this project

M&A advisory and corporate development teams spend significant analyst time
manually screening for takeover candidates. This project prototypes that
screening step: instead of reading news and skimming 10-Ks by hand, score
every company in a universe on an "acquisition likelihood" from 0-1, ranked
and explainable by feature.

## Architecture

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

## Data & methodology

- **Company universe (40 companies, real names):** 20 companies that were
  genuinely acquired/taken private in the 2021-2023 wave (Twitter, Slack,
  Splunk, VMware, Activision Blizzard, Cerner, etc.) labeled `1`, and 20
  comparable public companies in similar sectors that remained standalone
  through the same window (Adobe, Salesforce, CrowdStrike, Datadog, etc.)
  labeled `0`. The **outcome labels are real, documented facts**.
- **Fundamental features (growth, margin, leverage, ownership, valuation
  gap) are synthetically generated** from sector-conditioned distributions,
  not pulled from a live filings feed — disclosed openly here and in code
  comments. This was a deliberate choice to move fast on a demo without
  needing SEC EDGAR / Bloomberg API access. `data_ingestion.py` is written
  so `generate_synthetic_fundamentals()` can be swapped for a real
  EDGAR/yfinance/Refinitiv puller without touching the rest of the pipeline.
- **News sentiment** uses a small sample headline corpus per company, scored
  by FinBERT (`ProsusAI/finbert`) — a BERT model fine-tuned specifically on
  financial text, producing positive/negative/neutral probabilities that are
  converted into a signed sentiment score.
- **Fallback:** this repo was built in a disk-constrained sandbox that
  couldn't fit the torch/CUDA dependency stack (~3-4GB). `sentiment_finbert.py`
  auto-detects whether `transformers`/`torch` are installed; if not, it falls
  back to a small finance-lexicon scorer and **clearly logs that it is not
  using FinBERT**. On a normal machine with torch installed, it silently uses
  the real model — no code changes needed.

## Results (on this demo dataset)

| Metric | Value |
|---|---|
| Test AUC | 1.00 |
| Test Accuracy | 0.90 |
| Test Precision | 0.83 |
| Test Recall | 1.00 |
| Test F1 | 0.91 |

Top predictive features: news sentiment (avg + minimum), EBITDA margin,
revenue growth, institutional ownership.

**Honest caveat:** N=40 companies with a 25% test split (10 companies) is a
small-sample demo, not a production validation. Perfect/near-perfect AUC on
10 test rows is expected here and should not be quoted as "the model is
99% accurate".

## How to run

```bash
pip install -r requirements.txt
cd src
python data_ingestion.py        # build company + fundamentals dataset
python news_headlines.py        # build sample news corpus
python sentiment_finbert.py     # score headlines (FinBERT or fallback)
python feature_engineering.py   # build final feature matrix
python train_model.py           # train XGBoost, save metrics/plots
python predict_targets.py       # produce ranked target leaderboard
```

Outputs land in `outputs/`: `target_rankings.csv`, `metrics.json`,
`feature_importance.csv`, `roc_curve.png`, `feature_importance.png`.

## Next steps for a production version

1. Replace synthetic fundamentals with real EDGAR/10-K financial data.
2. Pull live news via NewsAPI/RavenPack instead of the sample corpus.
3. Expand the universe beyond 40 companies (ideally 500+) for a real
   train/test split, and use time-based (walk-forward) validation instead
   of random split, since M&A prediction is inherently a forecasting task.
4. Calibrate the model's probability outputs (Platt scaling / isotonic)
   since raw XGBoost probabilities are not well-calibrated on small data.
5. Add sector-relative valuation multiples instead of absolute P/E, since
   raw P/E isn't comparable across sectors.

## Real-data version
`src/real_pipeline.py` replaces the synthetic stages with SEC EDGAR XBRL financials (year before each deal), Google News headlines, and real FinBERT, evaluated with repeated 5-fold CV. Run on a machine with internet + torch (e.g. Colab). The synthetic pipeline above is the offline demo.
