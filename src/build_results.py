"""Builds docs/data/results.json + models/xgb_real.json from the real run files."""
import json, numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
f = pd.read_csv("data/real_features.csv"); r = pd.read_csv("outputs/real_rankings.csv")
FIN = ["rev_growth","op_margin","net_margin","debt_to_equity","cash_to_assets","rd_intensity","log_assets"]
SENT = ["avg_sent","min_sent","neg_share","n_headlines"]; ALL = FIN + SENT; y = f.was_acquired
mk = lambda: xgb.XGBClassifier(n_estimators=100, max_depth=2, learning_rate=.08, subsample=.8, colsample_bytree=.8, random_state=42)
def cv(cols):
    a = [roc_auc_score(y, cross_val_predict(mk(), f[cols], y, cv=StratifiedKFold(5, shuffle=True, random_state=i), method="predict_proba")[:,1]) for i in range(5)]
    return {"auc": round(float(np.mean(a)),3), "std": round(float(np.std(a)),3), "per_repeat": [round(float(x),3) for x in a]}
abl = {k: cv(c) for k, c in {"Financials only": FIN, "News sentiment only": SENT, "Financials + sentiment": ALL}.items()}
m = mk().fit(f[ALL], y); m.save_model("models/xgb_real.json")
imp = sorted(zip(ALL, map(float, m.feature_importances_)), key=lambda t: -t[1])
p = (r.acq_score_oof >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(r.was_acquired, p).ravel()
pr, rc, f1, sp = precision_recall_fscore_support(r.was_acquired, p)
rows = r.merge(f[["company"] + ALL], on="company").round(4).where(lambda d: d.notna(), None).to_dict("records")
top = r.sort_values("acq_score_oof", ascending=False)
json.dump({"n": len(f), "n_acquired": int(y.sum()), "n_headlines": int(f.n_headlines.sum()), "feature_columns": ALL,
  "cv": abl["Financials + sentiment"], "ablation": abl, "importance": [{"feature": a, "value": round(b,4)} for a, b in imp],
  "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
  "per_class": {k: {"precision": round(float(pr[i]),3), "recall": round(float(rc[i]),3), "f1": round(float(f1[i]),3), "support": int(sp[i])} for i, k in enumerate(["Independent","Acquired"])},
  "top5_acquired": int(top.head(5).was_acquired.sum()), "top10_acquired": int(top.head(10).was_acquired.sum()),
  "score_curve": [round(float(x),3) for x in top.acq_score_oof], "rows": rows}, open("docs/data/results.json", "w"))
print(abl, tp, fp, fn, tn)
