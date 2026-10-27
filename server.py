"""DealRadar API + static site.  Run: uvicorn server:app --port 8000"""
import io, json
from pathlib import Path
import pandas as pd, xgboost as xgb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
ROOT = Path(__file__).parent
COLS = json.load(open(ROOT / "docs/data/results.json"))["feature_columns"]
model = xgb.XGBClassifier(); model.load_model(str(ROOT / "models/xgb_real.json"))
app = FastAPI(title="DealRadar API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
def score(df): return model.predict_proba(df[COLS].astype(float))[:, 1].round(4).tolist()
@app.get("/api/health")
def health(): return {"status": "ok", "features": COLS}
@app.post("/api/analyse")
def analyse(body: dict):
    df = pd.DataFrame([{c: body.get(c) for c in COLS}])
    return {"acquisition_score": score(df)[0], "missing": [c for c in COLS if body.get(c) in (None, "")]}
@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    df = pd.read_csv(io.BytesIO(await file.read()))
    miss = [c for c in COLS if c not in df.columns]
    if miss: raise HTTPException(400, f"CSV missing columns: {miss}")
    df["acquisition_score"] = score(df)
    name = "company" if "company" in df.columns else df.columns[0]
    return {"rows": df[[name, "acquisition_score"]].sort_values("acquisition_score", ascending=False).to_dict("records")}
app.mount("/", StaticFiles(directory=ROOT / "docs", html=True), name="site")
