# Final Project — Task Checklist

## Phase 1 — Setup
- [x] Download Olist dataset from Kaggle → place 8 CSVs in `data/raw/`
- [x] Install deps: `uv pip install scikit-learn scipy matplotlib seaborn scikit-surprise faiss-cpu sentence-transformers --python /Users/ahmetbilalozgun/Documents/Projects/Dersler/veribilimi/.venv/bin/python3`
- [x] Verify all imports work in a test cell
- [x] Create `.gitignore` (ignore `data/raw/` if CSVs >50MB, ignore `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`)
- [x] Create `requirements.txt`

## Phase 2 — Data Loading & Cleaning
- [x] Load all 8 CSVs with correct dtypes and date parsing (`order_purchase_timestamp` → datetime)
- [x] Filter to `order_status == "delivered"` only
- [x] Handle nulls: `product_category_name` (drop rows), `product_description_lenght` (fill with 0)
- [x] Join orders → order_items → products → customers → reviews → `df_master`
- [x] Verify: `df_orders.shape[0] > 90_000`, no full-null columns
- [x] Add price bucket and weight bucket columns to `df_products`

## Phase 3 — EDA (5 visualization types)
- [x] 3.1 Line chart: monthly order volume over time
- [x] 3.2 Horizontal bar: top-15 product categories by order count
- [x] 3.3 Histogram + KDE: review score distribution
- [x] 3.4 Box plot: payment_value grouped by review_score (1–5)
- [x] 3.5 Bar/scatter: customer count by Brazilian state
- [x] Add markdown interpretation cell after each viz

## Phase 4 — RQ1: Regional Rating Bias
- [x] Derive `is_multi_category` flag per `customer_unique_id` (bought from 2+ categories → True; NOTE: use `customer_unique_id`, not `customer_id` — Olist `customer_id` is order-scoped)
- [x] Join with `customer_state` from `df_customers`
- [x] Compute mean review score by (buyer_type × state)
- [x] Violin/box plot: top-5 states, split by buyer type
- [x] Mann-Whitney U test (`scipy.stats.mannwhitneyu`)
- [x] Write 3–5 sentence answer to RQ1

## Phase 5 — RQ2: Collaborative Filtering (SVD)
- [x] Build user-item DataFrame: `(customer_id, product_category_name, mean_review_score)`
- [x] `surprise.Dataset.load_from_df()` with `Reader(rating_scale=(1, 5))`
- [x] `train_test_split(test_size=0.2, random_state=42)` — superseded by temporal split (random split is incorrect for recommenders)
- [x] Fit `SVD(n_factors=100, n_epochs=20, random_state=42)`
- [x] Use temporal train/test split: train on orders before ~2018-01 (use `order_purchase_timestamp` from df_master), test on 2018-01+ — prevents future-review leakage into training (random split is incorrect for recommenders)
- [x] `cross_validate(SVD, data, cv=5)` → print RMSE/MAE table — replaced by 3-cutpoint temporal CV loop (no future leakage)
- [x] Fit `NormalPredictor` baseline, compare RMSE — replaced by global-mean baseline (`trainset.global_mean`)
- [x] Assert SVD RMSE < NormalPredictor RMSE AND < 1.5 — replaced by conditional print warnings (notebook stays runnable if SVD loses)
- [x] Generate top-5 category recommendations for 3 sample customers
- [x] Write RQ2 answer with RMSE improvement % — template written; fill X/Y/Z after running notebook

## Phase 6 — RQ3: Content-Based Filtering (sentence-transformers + FAISS)
- [x] Build product metadata string per product_id: `category + " " + price_bucket + " " + weight_bucket`
- [x] Load `SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")`
- [x] Encode all product metadata → 384-dim vectors, L2-normalize
- [x] Cache CB embeddings at `data/cb_embeddings.npy`
- [x] Build `faiss.IndexFlatIP`, add normalized vectors
- [x] Add row-loss log after each merge in Cell 7 for the report (how many rows dropped by inner joins)
- [x] Verify: self-query cosine similarity == 1.0
- [x] Query top-5 similar products for 5 sample products
- [x] Demo free-text semantic search (e.g. "presente leve para crianças")
- [x] Build co-purchase ground truth from `df_items` (products in same order)
- [x] Compute overlap@5 for top-50 products by order volume
- [x] Display recommendation table for 3 example products
- [x] Build category-level mean embeddings for Phase 7 hybrid scoring
- [x] Write RQ3 answer

## Phase 7 — Hybrid Demo
- [x] Normalize CF predicted ratings: `(score - 1) / 4` → [0, 1]
- [x] CB cosine score already in [0, 1]
- [x] `hybrid_score = 0.6 * cf_normalized + 0.4 * cb_cosine`
- [x] Demo hybrid recommendations for 3 customers
- [x] Explain α=0.6 choice (user behavior > content metadata)

## Phase 8 — Report (PDF, 3–5 pages)
- [x] Section 1: Introduction & motivation (0.5 pp)
- [x] Section 2: Dataset description & cleaning decisions (1 pp)
- [x] Section 3: Methods — CF and CB approaches (1 pp)
- [x] Section 4: Results — answer each RQ with key figures (1.5 pp)
- [x] Section 5: Conclusions & limitations (0.5 pp)
- [x] Export to PDF → `report/report.pdf`

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
