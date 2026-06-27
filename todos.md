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

## Phase 3 EDA — Viz 3.6 (Regional Buyer Behavior, demoted from RQ)
- [x] Derive `is_multi_category` flag per `customer_unique_id` (bought from 2+ categories → True)
- [x] Violin plot: top-5 states by order volume, split by buyer type
- [x] Brief EDA interpretation (motivates CB component for thin-history users)

## Phase 4 — RQ1: Does SVD-Based CF Outperform a Non-Personalized Baseline?
- [x] Build user-item DataFrame: `(customer_id, product_category_name, mean_review_score)`
- [x] `surprise.Dataset.load_from_df()` with `Reader(rating_scale=(1, 5))`
- [x] Temporal train/test split: train before 2018-01, test on 2018-01+
- [x] Fit `SVD(n_factors=100, n_epochs=20, random_state=42)`
- [x] 3-cutpoint temporal CV loop (no future leakage)
- [x] Global-mean baseline comparison
- [x] Generate top-5 category recommendations for 3 sample customers
- [x] Write RQ1 answer — mention 100% cold-start → motivates hybrid CB component

## Phase 5 — RQ2: Do Sentence-Transformer Embeddings Capture Meaningful Category Relationships?
- [x] Build product metadata string per product_id: `category + " " + price_bucket + " " + weight_bucket`
- [x] Load `SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")`
- [x] Encode all product metadata → 384-dim vectors, L2-normalize
- [x] Cache CB embeddings at `data/cb_embeddings.npy`
- [x] Build `faiss.IndexFlatIP`, add normalized vectors
- [x] Verify: self-query cosine similarity == 1.0
- [x] Query top-5 similar products for 5 sample products
- [x] Demo free-text semantic search (e.g. "presente leve para crianças")
- [x] Display recommendation table for 3 example products
- [x] Build category-level mean embeddings for Phase 6+7 hybrid scoring
- [x] RQ2 Part A: Nearest-neighbor category table (5 seed categories via category-level FAISS)
- [x] RQ2 Part B: PCA 2D scatter of all 73 category embeddings
- [x] Write RQ2 answer — confirm semantic clusters visible, motivates CB in hybrid

## Phase 6+7 — RQ3: Does the Hybrid Produce More Coherent Recommendations Than CF or CB Alone?
- [x] Normalize CF predicted ratings: `(score - 1) / 4` → [0, 1]
- [x] CB cosine score already in [0, 1]
- [x] `hybrid_score = 0.6 * cf_normalized + 0.4 * cb_cosine`
- [x] Phase 7 hybrid demo for 3 sample customers (CF scores + CB cosine + hybrid score table)
- [x] Side-by-side comparison: CF top-5 | CB top-5 | Hybrid top-5 for same 3 customers
- [x] Jaccard@5 computation: Jaccard(CF,CB), Jaccard(CF,Hybrid), Jaccard(CB,Hybrid)
- [x] Grouped bar chart: mean Jaccard across 3 customers
- [x] Write RQ3 answer — confirm CF and CB are complementary, hybrid blends both

## Phase 8 — Report (PDF, 3–5 pages)
- [x] Section 1: Introduction & motivation (0.5 pp)
- [x] Section 2: Dataset description & cleaning decisions (1 pp)
- [x] Section 3: Methods — CF and CB approaches (1 pp)
- [x] Section 4: Results — answer each RQ with key figures (1.5 pp)
- [x] Section 5: Conclusions & limitations (0.5 pp)
- [x] Export to PDF → `report/report.pdf`

## Phase 9 — Final Polish & Submission
- [x] `prompts.md`: log 10–15 Claude prompts used, chronological, with phase labels
- [x] `README.md`: student info (student number, name), setup instructions, libraries, data source + license
- [x] `requirements.txt`: freeze exact versions (`uv pip freeze`)
- [x] Kernel > Restart & Run All — zero errors, all 5 figures render
- [x] Push to GitHub, verify all files visible
- [ ] Submit GitHub URL by 2026-07-03, 12:30

## Phase 10 — UCI Dataset Migration

- [x] Download UCI Online Retail dataset from UCI Archive; save as `data/raw/Online_Retail.csv`
- [x] Phase 2 rewrite: load CSV/Excel; cast CustomerID: `astype(int).astype(str)`; filter non-product StockCodes `^[0-9]{5}[A-Za-z]?$`; drop cancelled/null/negative; build df_master
- [x] Phase 2: aggregate median UnitPrice per product_id → price_bucket; aggregate modal Description per product_id → product_description
- [x] Phase 3 EDA rewrite: order volume, top 20 products, country distribution, purchase frequency histogram (log scale), synthetic_rating distribution
- [x] Adapt RQ1: "Do loyal customers (≥3 invoices) show higher product diversity rate?" Mann-Whitney U + violin on diversity_rate = n_distinct_products / invoice_count
- [x] Phase 5: temporal split 2011-10-01; compute synthetic_rating within each window separately (no leakage); df_ui_train + df_ui_test from train-window freq only
- [x] Verify warm-start: confirm >50% test users seen in train — **71.8% warm** (vs 0% on Olist) ✓
- [x] Update CB: use `all-MiniLM-L6-v2` model; Description → metadata_str; remove weight_bucket; keep price_bucket (£ thresholds)
- [x] CB verification: same-description-prefix products have higher cosine than cross-prefix — **5/5 prefixes ✓**
- [x] Verify SVD lift: **+2.6% RMSE** vs global-mean baseline for warm users ✓
- [x] Update hybrid demo: pick 3 UCI customers with ≥5 invoices; exclude already-purchased products from recommendations
- [x] Update CLAUDE.md: new dataset schema, RQ1 text (diversity_rate), naming conventions, model name
- [x] Update design.md: add dataset-switch rationale; frame as implicit-purchase recommender
- [x] Run notebook kernel restart + Run All; fix any errors — **0 errors** ✓
- [x] Update report sections 2–4: dataset description, rating synthesis explanation, findings (after notebook runs)
- [x] Update README.md: new data source URL (UCI Archive), license (Open/Public), setup instructions

## Submission Requirements Checklist
- [x] notebook.ipynb: data loading/cleaning ✓ EDA 4+ viz types ✓ 3+ RQs answered ✓ ML model ✓ interpretation ✓
- [x] report/report.pdf (3–5 pages)
- [x] prompts.md (10–15 prompts)
- [x] README.md
- [x] GitHub repository
