#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python src/data_ingestion.py
python src/news_headlines.py
python src/sentiment_finbert.py
python src/feature_engineering.py
python src/train_model.py
python src/predict_targets.py
echo "Done. See outputs/ for results."
