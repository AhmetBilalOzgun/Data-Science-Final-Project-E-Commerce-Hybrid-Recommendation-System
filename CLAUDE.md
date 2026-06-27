# Final Project — E-Commerce Hybrid Recommendation System

## Goal
Data Science lecture final project (due 2026-07-03).
Hybrid recommendation system on UCI Online Retail dataset (implicit-purchase recommender).
Theme: E-Commerce & Customer Behavior.

## Environment
- Python 3.14.3 venv: `/Users/ahmetbilalozgun/Documents/Projects/Dersler/veribilimi/.venv`
- Install packages: `uv pip install <pkg> --python /Users/ahmetbilalozgun/Documents/Projects/Dersler/veribilimi/.venv/bin/python3`
- Run notebook: `/Users/ahmetbilalozgun/Documents/Projects/Dersler/veribilimi/.venv/bin/jupyter lab`

## Dataset — UCI Online Retail (single CSV `data/raw/Online_Retail.csv`)
Source: https://archive.ics.uci.edu/dataset/352/online+retail  
License: Open/Public (UCI Archive)

| Column | Role |
|--------|------|
| CustomerID | CF user axis (float64 → cast to int → str) |
| StockCode | CF/CB item axis (filter: `^[0-9]{5}[A-Za-z]?$`) |
| Description | CB metadata text (modal per StockCode) |
| InvoiceDate | Temporal split axis (cutoff 2011-10-01) |
| InvoiceNo | Purchase event; prefix 'C' = cancellation (drop) |
| Quantity | Drop ≤ 0 |
| UnitPrice | Drop ≤ 0; median per StockCode → price_bucket |
| Country | EDA only |

**Post-cleaning**: ~396k rows, 4,334 customers, 3,658 products, 18,401 invoices.

## Architecture
1. **Collaborative Filtering**: scikit-surprise SVD on `customer_id × product_id` matrix  
   Value = `log1p(InvoiceNo.nunique())` scaled to [1,5] **within each temporal window** (no leakage)
2. **Content-Based**: sentence-transformers (`all-MiniLM-L6-v2`) → 384-dim embeddings → L2-normalize → FAISS IndexFlatIP  
   Feature string: `f"{product_description} {price_bucket}"`
3. **Hybrid**: `0.6 × CF_score_normalized + 0.4 × CB_cosine`; exclude already-purchased products

## Key Decisions (DO NOT CHANGE)
- No ChromaDB (57-package dep tree)
- No TF-IDF (replaced by semantic embeddings)
- Rating proxy = purchase frequency (log1p+minmax), NOT review scores
- Synthetic rating computed per temporal window separately (no future leakage)
- User-item matrix at product_id (StockCode) level, not category level
- RANDOM_STATE = 42 everywhere
- Temporal cutoff: `"2011-10-01"` (~80/20 train/test)

## Research Questions
1. Do loyal customers (≥3 invoices) show higher product diversity per invoice than occasional buyers? (Mann-Whitney U + violin, metric = diversity_rate = n_distinct_products / invoice_count)
2. How much does SVD improve over global-mean baseline? (temporal split, RMSE table, warm-user focus)
3. Are sentence-transformer embeddings semantically coherent? (FAISS top-5 prefix similarity check); does hybrid blend CF and CB signals? (Jaccard@5)

## Naming Conventions
- DataFrames: `df_raw`, `df`, `df_products`, `df_master`, `df_cb`, `df_loyalty`, `df_ui_train`, `df_ui_test`, `df_ui_all`
- Figures: `fig, ax = plt.subplots(...)` → `plt.tight_layout()` → `plt.show()`
- Random state: `RANDOM_STATE = 42` constant in setup cell
- Embedding cache: `data/cb_embeddings.npy`

## File Structure
```
finalProject/
├── CLAUDE.md           ← this file
├── README.md
├── todos.md
├── design.md
├── data/raw/           ← Online_Retail.csv (UCI)
├── data/cb_embeddings.npy  ← cached sentence-transformer embeddings
├── notebook.ipynb      ← MAIN DELIVERABLE
├── report/             ← PDF report (3-5 pages)
├── prompts.md          ← prompt log (10-15 entries)
└── requirements.txt
```

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Author a backlog-ready spec/issue → invoke /spec

## Deliverables Checklist
- [ ] notebook.ipynb (data loading, cleaning, EDA, 3 RQs, modeling, interpretation)
- [ ] report/report.pdf (3-5 pages: problem, dataset, methods, findings, limitations)
- [ ] prompts.md (10-15 AI prompts chronologically)
- [ ] README.md (team info, setup, libraries, data source + license)
- [ ] GitHub repository link submitted
