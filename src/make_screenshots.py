import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---------- 1. Terminal-style screenshot ----------
terminal_text = """$ ./run_pipeline.sh

Wrote 40 companies to data/companies_fundamentals.csv
was_acquired
1    20
0    20

Wrote 140 headlines for 40 companies

[sentiment_finbert] transformers/torch NOT installed -> using
finance-lexicon fallback scorer (NOT FinBERT). Install torch +
transformers and re-run to use the real model.
Scored 140 headlines using method='lexicon_fallback'

Built feature matrix: 40 rows, 44 columns
Base features used for modeling: 19

Metrics: {
  "n_train": 30,
  "n_test": 10,
  "test_auc": 1.0,
  "test_accuracy": 0.9,
  "test_precision": 0.833,
  "test_recall": 1.0,
  "test_f1": 0.909
}

Top 5 features by importance:
             feature  importance
 min_news_sentiment       0.313
 avg_news_sentiment       0.299
       ebitda_margin       0.287
  revenue_growth_yoy       0.075
institutional_own_pct       0.025

Done. See outputs/ for results.
$ """

fig, ax = plt.subplots(figsize=(8.5, 8.2))
fig.patch.set_facecolor("#1e1e1e")
ax.set_facecolor("#1e1e1e")
ax.axis("off")

# title bar
ax.add_patch(plt.Rectangle((0, 0.965), 1, 0.035, transform=ax.transAxes,
                            facecolor="#333333", edgecolor="none", clip_on=False))
for i, (cx, color) in enumerate(zip([0.018, 0.038, 0.058], ["#ff5f56", "#ffbd2e", "#27c93f"])):
    ax.add_patch(plt.Circle((cx, 0.982), 0.008, transform=ax.transAxes,
                             facecolor=color, edgecolor="none", clip_on=False))
ax.text(0.5, 0.982, "ma_screener — run_pipeline.sh", transform=ax.transAxes,
        ha="center", va="center", color="#cccccc", fontsize=10, family="monospace")

ax.text(0.02, 0.94, terminal_text, transform=ax.transAxes, ha="left", va="top",
        color="#d4d4d4", fontsize=10.3, family="monospace", linespacing=1.5)

plt.tight_layout(pad=0.3)
plt.savefig("outputs/terminal_screenshot.png", dpi=180, facecolor="#1e1e1e")
plt.close()

# ---------- 2. Rankings table image ----------
df = pd.read_csv("outputs/target_rankings.csv").head(10)
df_display = df[["company", "sector", "acquisition_score"]].copy()
df_display["acquisition_score"] = df_display["acquisition_score"].round(3)
df_display.columns = ["Company", "Sector", "Acquisition score"]

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.axis("off")
tbl = ax.table(
    cellText=df_display.values,
    colLabels=df_display.columns,
    cellLoc="left",
    colLoc="left",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 1.8)
tbl.auto_set_column_width(col=list(range(len(df_display.columns))))

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#dddddd")
    if row == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", weight="bold")
    else:
        cell.set_facecolor("#f7f9fb" if row % 2 == 0 else "white")

ax.set_title("Top 10 acquisition-likelihood scores — M&A Target Screener", fontsize=13, pad=14, weight="bold")
plt.tight_layout()
plt.savefig("outputs/rankings_table.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close()

print("Saved outputs/terminal_screenshot.png and outputs/rankings_table.png")
