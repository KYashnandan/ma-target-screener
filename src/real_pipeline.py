"""
Real-data version. Run on your machine/Colab (needs internet + torch):
  pip install -r requirements.txt requests
  python src/real_pipeline.py
Data: SEC EDGAR XBRL (financials), Google News RSS (headlines), real FinBERT.
Snapshot year = year BEFORE deal closing (acquired) / 2022 (standalone), so
features come from before the deal, not after.
"""
import re, time, json, requests, numpy as np, pandas as pd, xgboost as xgb
import xml.etree.ElementTree as ET
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from data_ingestion import ACQUIRED_COMPANIES, STANDALONE_COMPANIES

UA = {"User-Agent": "Yash KYashnandan research yash@example.com"}  # EDGAR requires contact UA: put your real email
TAGS = {
 "rev": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
 "opinc": ["OperatingIncomeLoss"], "ni": ["NetIncomeLoss"],
 "debt": ["LongTermDebt", "LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "ConvertibleNotesPayable"], "eq": ["StockholdersEquity"],
 "cash": ["CashAndCashEquivalentsAtCarryingValue"], "assets": ["Assets"],
 "rnd": ["ResearchAndDevelopmentExpense"],
}

ENT = {}

def get(url, **kw):
    time.sleep(0.2)
    r = requests.get(url, headers=UA, timeout=30, **kw); r.raise_for_status(); return r

CIK_FIX = {"Palo Alto Networks": "0001327567", "Roper Technologies": "0000882835", "Take-Two Interactive": "0000946581", "Nuance Communications": "0001002517"}

def find_cik(name):
    if name in CIK_FIX: return CIK_FIX[name]
    q = re.sub(r"\b(Inc|Corporation|Holdings?|Holding|Technologies|International|Systems)\b", "", name).strip()
    t = get("https://www.sec.gov/cgi-bin/browse-edgar", params={"company": q, "type": "10-K", "dateb": "", "owner": "include", "count": 5, "output": "atom"}).text
    m = re.search(r"CIK=(\d{10})", t) or re.search(r"data/(\d+)/", t)
    return m.group(1).zfill(10) if m else None

def annual(facts, key):
    for tag in TAGS[key]:
        d = facts.get("us-gaap", {}).get(tag, {}).get("units", {}).get("USD")
        if d:
            rows = {}
            for x in d:
                if x.get("form") in ("10-K", "10-K/A") and "start" in x and 330 <= (pd.Timestamp(x["end"]) - pd.Timestamp(x["start"])).days <= 380:
                    rows[x["end"]] = x["val"]
                elif x.get("form") in ("10-K", "10-K/A") and "start" not in x:
                    rows[x["end"]] = x["val"]  # instant (balance sheet) items
            if rows: return pd.Series(rows).sort_index()
    return pd.Series(dtype=float)

def at(s, end_limit, back=0):
    if s.empty: return np.nan
    s = s[s.index <= end_limit]
    return s.iloc[-1 - back] if len(s) > back else np.nan

def fin_features(cik, ref):
    j = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json").json(); facts = j["facts"]; ENT[cik] = j.get("entityName")
    lim = f"{ref}-12-31"; S = {k: annual(facts, k) for k in TAGS}
    rev, rev0 = at(S["rev"], lim), at(S["rev"], lim, 1)
    return dict(
        rev_growth=rev / rev0 - 1 if rev0 else np.nan,
        op_margin=at(S["opinc"], lim) / rev if rev else np.nan,
        net_margin=at(S["ni"], lim) / rev if rev else np.nan,
        debt_to_equity=at(S["debt"], lim) / at(S["eq"], lim) if at(S["eq"], lim) else np.nan,
        cash_to_assets=at(S["cash"], lim) / at(S["assets"], lim) if at(S["assets"], lim) else np.nan,
        rd_intensity=at(S["rnd"], lim) / rev if rev else np.nan,
        log_assets=np.log(at(S["assets"], lim)) if at(S["assets"], lim) > 0 else np.nan)

def headlines(name, ref, n=25):
    q = f'"{name}" after:{ref}-01-01 before:{ref}-12-31'
    root = ET.fromstring(get("https://news.google.com/rss/search", params={"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}).content)
    return [i.findtext("title") for i in root.iter("item")][:n]

def main():
    cos = [(n, s, 1, y - 1) for n, s, y in ACQUIRED_COMPANIES] + [(n, s, 0, 2022) for n, s in STANDALONE_COMPANIES]
    from transformers import pipeline
    fb = pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True)
    rows = []
    for name, sector, y, ref in cos:
        try:
            cik = find_cik(name); f = fin_features(cik, ref)
            hl = headlines(name, ref)
            sc = [(r["score"] if r["label"] == "positive" else -r["score"] if r["label"] == "negative" else 0.0) for r in fb(hl)] if hl else []
            f.update(avg_sent=np.mean(sc) if sc else np.nan, min_sent=min(sc) if sc else np.nan,
                     neg_share=np.mean([s < -0.3 for s in sc]) if sc else np.nan, n_headlines=len(sc))
            rows.append(dict(company=name, sector=sector, was_acquired=y, ref_year=ref, cik=cik, **f)); print("ok", name, "|", ENT.get(cik), "|", len(sc), "headlines")
        except Exception as e:
            print("SKIP", name, "->", e)
    df = pd.DataFrame(rows); df.to_csv("data/real_features.csv", index=False)
    X, y = df.drop(columns=["company", "sector", "was_acquired", "ref_year", "cik"]), df["was_acquired"]
    mk = lambda: xgb.XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42)
    aucs, oof = [], np.zeros(len(df))
    for i in range(5):
        p = cross_val_predict(mk(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=i), method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p)); oof += p / 5
    print(f"Repeated 5-fold CV AUC: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}  (n={len(df)})")
    df["acq_score_oof"] = oof
    df.sort_values("acq_score_oof", ascending=False)[["company", "sector", "was_acquired", "acq_score_oof"]].to_csv("outputs/real_rankings.csv", index=False)
    m = mk().fit(X, y); print(pd.Series(m.feature_importances_, X.columns).sort_values(ascending=False).head(6))
    json.dump({"cv_auc_mean": float(np.mean(aucs)), "cv_auc_std": float(np.std(aucs)), "n": len(df)}, open("outputs/real_metrics.json", "w"))

if __name__ == "__main__":
    main()
