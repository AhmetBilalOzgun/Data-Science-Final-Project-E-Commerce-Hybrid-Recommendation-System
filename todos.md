# Final Project — Task Checklist

## Phase 1 — Setup
- [ ] Download Olist dataset from Kaggle → place 8 CSVs in `data/raw/`
- [ ] Install deps: `uv pip install scikit-learn scipy matplotlib seaborn scikit-surprise faiss-cpu sentence-transformers --python /Users/ahmetbilalozgun/Documents/Projects/Dersler/veribilimi/.venv/bin/python3`
- [ ] Verify all imports work in a test cell
- [ ] Create `.gitignore` (ignore `data/raw/` if CSVs >50MB, ignore `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`)
- [ ] Create `requirements.txt`

## Phase 2 — Data Loading & Cleaning
- [ ] Load all 8 CSVs with correct dtypes and date parsing (`order_purchase_timestamp` → datetime)
- [ ] Filter to `order_status == "delivered"` only
- [ ] Handle nulls: `product_category_name` (drop rows), `product_description_lenght` (fill with 0)
- [ ] Join orders → order_items → products → customers → reviews → `df_master`
- [ ] Verify: `df_orders.shape[0] > 90_000`, no full-null columns
- [ ] Add price bucket and weight bucket columns to `df_products`

## Phase 3 — EDA (5 visualization types)
- [ ] 3.1 Line chart: monthly order volume over time
- [ ] 3.2 Horizontal bar: top-15 product categories by order count
- [ ] 3.3 Histogram + KDE: review score distribution
- [ ] 3.4 Box plot: payment_value grouped by review_score (1–5)
- [ ] 3.5 Bar/scatter: customer count by Brazilian state
- [ ] Add markdown interpretation cell after each viz

## Phase 4 — RQ1: Regional Rating Bias
- [ ] Derive `is_multi_category` flag per customer (bought from 2+ categories → True)
- [ ] Join with `customer_state` from `df_customers`
- [ ] Compute mean review score by (buyer_type × state)
- [ ] Violin/box plot: top-5 states, split by buyer type
- [ ] Mann-Whitney U test (`scipy.stats.mannwhitneyu`)
- [ ] Write 3–5 sentence answer to RQ1

## Phase 5 — RQ2: Collaborative Filtering (SVD)
- [ ] Build user-item DataFrame: `(customer_id, product_category_name, mean_review_score)`
- [ ] `surprise.Dataset.load_from_df()` with `Reader(rating_scale=(1, 5))`
- [ ] `train_test_split(test_size=0.2, random_state=42)`
- [ ] Fit `SVD(n_factors=100, n_epochs=20, random_state=42)`
- [ ] `cross_validate(SVD, data, cv=5)` → print RMSE/MAE table
- [ ] Fit `NormalPredictor` baseline, compare RMSE
- [ ] Assert SVD RMSE < NormalPredictor RMSE AND < 1.5
- [ ] Generate top-5 category recommendations for 3 sample customers
- [ ] Write RQ2 answer with RMSE improvement %

## Phase 6 — RQ3: Content-Based Filtering (sentence-transformers + FAISS)
- [ ] Build product metadata string per product_id: `category + " " + price_bucket + " " + weight_bucket`
- [ ] Load `SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")`
- [ ] Encode all product metadata → 384-dim vectors, L2-normalize
- [ ] Build `faiss.IndexFlatIP`, add normalized vectors
- [ ] Verify: self-query cosine similarity == 1.0
- [ ] Query top-5 similar products for 5 sample products
- [ ] Demo free-text semantic search (e.g. "presente leve para crianças")
- [ ] Build co-purchase ground truth from `df_items` (products in same order)
- [ ] Compute overlap@5 for top-50 products by order volume
- [ ] Display recommendation table for 3 example products
- [ ] Write RQ3 answer

## Phase 7 — Hybrid Demo
- [ ] Normalize CF predicted ratings: `(score - 1) / 4` → [0, 1]
- [ ] CB cosine score already in [0, 1]
- [ ] `hybrid_score = 0.6 * cf_normalized + 0.4 * cb_cosine`
- [ ] Demo hybrid recommendations for 3 customers
- [ ] Explain α=0.6 choice (user behavior > content metadata)

## Phase 8 — Report (PDF, 3–5 pages)
- [ ] Section 1: Introduction & motivation (0.5 pp)
- [ ] Section 2: Dataset description & cleaning decisions (1 pp)
- [ ] Section 3: Methods — CF and CB approaches (1 pp)
- [ ] Section 4: Results — answer each RQ with key figures (1.5 pp)
- [ ] Section 5: Conclusions & limitations (0.5 pp)
- [ ] Export to PDF → `report/report.pdf`

## Phase 9 — Final Polish & Submission
- [ ] `prompts.md`: log 10–15 Claude prompts used, chronological, with phase labels
- [ ] `README.md`: student info (student number, name), setup instructions, libraries, data source + license
- [ ] `requirements.txt`: freeze exact versions (`uv pip freeze`)
- [ ] Kernel > Restart & Run All — zero errors, all 5 figures render
- [ ] Push to GitHub, verify all files visible
- [ ] Submit GitHub URL by 2026-07-03, 12:30

## Submission Requirements Checklist
- [ ] notebook.ipynb: data loading/cleaning ✓ EDA 4+ viz types ✓ 3+ RQs answered ✓ ML model ✓ interpretation ✓
- [ ] report/report.pdf (3–5 pages)
- [ ] prompts.md (10–15 prompts)
- [ ] README.md
- [ ] GitHub repository
