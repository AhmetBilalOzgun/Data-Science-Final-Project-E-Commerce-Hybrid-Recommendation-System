# Final Project — E-Commerce Hybrid Recommendation System

## Goal
Data Science lecture final project (due 2026-07-03).
Hybrid recommendation system on Olist Brazilian E-Commerce dataset.
Theme: E-Commerce & Customer Behavior.

## Environment
- Python 3.14.3 venv: `/Users/ahmetbilalozgun/Documents/Projects/Dersler/veribilimi/.venv`
- Install packages: `uv pip install <pkg> --python /Users/ahmetbilalozgun/Documents/Projects/Dersler/veribilimi/.venv/bin/python3`
- Run notebook: `/Users/ahmetbilalozgun/Documents/Projects/Dersler/veribilimi/.venv/bin/jupyter lab`

## Dataset — Olist Brazilian E-Commerce (8 CSVs in `data/raw/`)
| File | Key columns |
|------|-------------|
| olist_orders_dataset.csv | order_id, customer_id, order_status, order_purchase_timestamp |
| olist_order_items_dataset.csv | order_id, product_id, price, freight_value |
| olist_products_dataset.csv | product_id, product_category_name, product_description_lenght, product_weight_g |
| olist_customers_dataset.csv | customer_id, customer_state, customer_zip_code_prefix |
| olist_order_reviews_dataset.csv | order_id, review_score |
| olist_order_payments_dataset.csv | order_id, payment_value |
| olist_sellers_dataset.csv | seller_id, seller_state |
| olist_geolocation_dataset.csv | geolocation_zip_code_prefix, geolocation_lat, geolocation_lng, geolocation_state |

**Filter**: `order_status == "delivered"` only.

## Architecture
1. **Collaborative Filtering**: scikit-surprise SVD on `customer_id × product_category_name` matrix (value = mean review_score)
2. **Content-Based**: sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) → 384-dim embeddings → L2-normalize → FAISS IndexFlatIP
3. **Hybrid**: `0.6 × CF_score_normalized + 0.4 × CB_cosine`

## Key Decisions (DO NOT CHANGE)
- No ChromaDB (57-package dep tree)
- No TF-IDF (replaced by semantic embeddings)
- User-item matrix uses product_category_name (~70 categories), not product_id (too sparse)
- RANDOM_STATE = 42 everywhere

## Research Questions
1. Do multi-category buyers rate differently by Brazilian region? (Mann-Whitney U + violin plot)
2. How much does SVD improve over global-mean baseline? (5-fold CV, RMSE table)
3. How coherent are semantic recommendations vs actual co-purchases? (FAISS top-5, overlap@5)

## Naming Conventions
- DataFrames: `df_orders`, `df_items`, `df_products`, `df_customers`, `df_reviews`, `df_payments`, `df_master`
- Figures: `fig, ax = plt.subplots(...)` → `plt.tight_layout()` → `plt.show()`
- Random state: `RANDOM_STATE = 42` constant in setup cell

## File Structure
```
finalProject/
├── CLAUDE.md           ← this file
├── README.md
├── todos.md
├── design.md
├── data/raw/           ← 8 Olist CSVs
├── notebook.ipynb      ← MAIN DELIVERABLE
├── report/             ← PDF report (3-5 pages)
├── prompts.md          ← prompt log (10-15 entries)
└── requirements.txt
```

## Deliverables Checklist
- [ ] notebook.ipynb (data loading, cleaning, EDA, 3 RQs, modeling, interpretation)
- [ ] report/report.pdf (3-5 pages: problem, dataset, methods, findings, limitations)
- [ ] prompts.md (10-15 AI prompts chronologically)
- [ ] README.md (team info, setup, libraries, data source + license)
- [ ] GitHub repository link submitted
