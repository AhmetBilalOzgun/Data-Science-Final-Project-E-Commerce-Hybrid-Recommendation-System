# Design Document — E-Commerce Hybrid Recommendation System

## Architecture Overview

```
Olist Dataset (8 CSVs)
        │
        ▼
 Data Loading & Cleaning
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
Collaborative Filtering (CF)          Content-Based Filtering (CB)
customer_id × category matrix         sentence-transformer embeddings
      SVD (matrix factorization)             FAISS IndexFlatIP
      top-N category predictions             semantic similarity search
        │                                         │
        └──────────────┬──────────────────────────┘
                       ▼
              Hybrid Score
       0.6 × CF_normalized + 0.4 × CB_cosine
```

---

## Dataset

**Source**: Brazilian E-Commerce Public Dataset by Olist (Kaggle, CC BY-NC-SA 4.0)
**Tables & key columns used**:

| Table | Key columns |
|-------|-------------|
| orders | order_id, customer_id, order_status, order_purchase_timestamp |
| order_items | order_id, product_id, price, freight_value |
| products | product_id, product_category_name, product_description_lenght, product_weight_g |
| customers | customer_id, customer_state |
| order_reviews | order_id, review_score |
| order_payments | order_id, payment_value |
| sellers | seller_id, seller_state |
| geolocation | geolocation_zip_code_prefix, geolocation_lat, geolocation_lng, geolocation_state |
| product_category_name_translation | product_category_name (PT), product_category_name_english |

**Filter**: `order_status == "delivered"` — removes cancelled, unavailable, in-transit orders.

---

## User-Item Matrix (for Collaborative Filtering)

| Dimension | Choice | Reason |
|-----------|--------|--------|
| Rows | customer_id | ~100k unique customers |
| Columns | product_category_name | ~70 categories — dense enough for SVD (vs product_id which is too sparse) |
| Values | mean(review_score) per (customer, category) | Proxy for preference intensity |
| Rating scale | 1–5 | Native Olist review scale |

**Sparsity note**: Most Olist customers have exactly 1 order. CF will be sparse; cold-start customers fall back to CB component.

---

## Collaborative Filtering — SVD

**Library**: scikit-surprise
**Algorithm**: Simon Funk SVD (regularized matrix factorization)

**Hyperparameters** (academic scope — defaults sufficient):
```python
SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
```

**Evaluation**:
- 5-fold cross-validation (`cross_validate(cv=5)`) → RMSE + MAE
- Baseline: `NormalPredictor` (random from training distribution)
- SVD must beat baseline (assert SVD RMSE < NormalPredictor RMSE)

**Top-N generation**: predict score for all unrated categories for a target customer → sort descending → top-5.

---

## Content-Based Filtering — Sentence Transformers + FAISS

**Embedding model**: `paraphrase-multilingual-MiniLM-L12-v2`
- 384-dimensional vectors
- Handles Portuguese natively
- ~471MB download, cached at `~/.cache/huggingface/`

**Feature string per product** (built at preprocessing time):
```python
metadata = f"{product_category_name} {price_bucket} {weight_bucket}"
# price_bucket: budget (<50R$) | mid (50-150R$) | premium (150-500R$) | luxury (500R$+)
# weight_bucket: light (<500g) | medium (500-2000g) | heavy (2-10kg) | bulky (10kg+)
```

**Index construction**:
```python
embeddings = model.encode(metadata_list, normalize_embeddings=True)  # L2-normalized
index = faiss.IndexFlatIP(384)  # inner product on normalized = cosine similarity
index.add(embeddings)
```

**Query** (product similarity):
```python
D, I = index.search(query_embedding, k=6)  # k+1 to skip self
```

**Semantic search** (free-text):
```python
query_vec = model.encode(["presente leve para crianças"], normalize_embeddings=True)
D, I = index.search(query_vec, k=5)
```

---

## Hybrid Combination

```python
ALPHA = 0.6  # weight toward CF (user behavior over content metadata)

cf_score_normalized = (predicted_rating - 1) / 4  # map [1,5] → [0,1]
cb_score = cosine_similarity                        # already in [0,1]

hybrid_score = ALPHA * cf_score_normalized + (1 - ALPHA) * cb_score
```

α=0.6 justification: we have real purchase+review data, so user behavior signal is stronger than product metadata alone. For cold-start users (0 prior purchases), fall back to α=0 (pure CB).

---

## Research Questions

### RQ1: Regional Rating Bias by Customer Type
**Question**: Do multi-category buyers rate products differently by Brazilian region?
**Multi-category flag**: customer bought from ≥2 distinct product categories.
**Method**: Mann-Whitney U test (`scipy.stats.mannwhitneyu`) comparing review scores of multi-cat vs single-cat customers.
**Visualization**: Violin plot — top-5 states by order volume, split by buyer_type.

### RQ2: SVD vs Baseline Performance
**Question**: How much does SVD improve over global-mean baseline?
**Method**: Surprise 5-fold CV comparison (SVD vs NormalPredictor), RMSE + MAE table.
**Extension**: Segment analysis by purchase frequency (1 order, 2–3, 4+) to surface cold-start limitations.

### RQ3: Semantic Recommendation Coherence
**Question**: How coherent are sentence-transformer recommendations vs actual co-purchases?
**Metric**: overlap@5 — for top-50 products by order volume, compute:
```
overlap@5 = |FAISS_top5 ∩ co_purchased_products| / 5
```
**Bonus**: Show free-text semantic search working (demonstrates vector DB value).

---

## EDA Visualization Plan

| # | Viz Type | Variables | Library |
|---|----------|-----------|---------|
| 3.1 | Line chart | Monthly order count (time series) | matplotlib |
| 3.2 | Horizontal bar | Top-15 categories by order count | matplotlib |
| 3.3 | Histogram + KDE | review_score distribution | seaborn |
| 3.4 | Box plot | payment_value grouped by review_score | seaborn |
| 3.5 | Bar chart | Customer count by state | matplotlib |

5 distinct types → satisfies assignment's 4+ requirement.

---

## Package Dependencies

```
pandas>=2.0
numpy>=2.0
matplotlib>=3.9
seaborn>=0.13
scikit-learn>=1.9       # train_test_split, preprocessing
scipy>=1.18             # stats.mannwhitneyu
scikit-surprise>=1.1    # SVD, NormalPredictor, Dataset, Reader, cross_validate
faiss-cpu>=1.14         # IndexFlatIP
sentence-transformers>=3.0  # SentenceTransformer, encode
```

Install:
```bash
uv pip install scikit-learn scipy matplotlib seaborn scikit-surprise faiss-cpu sentence-transformers \
  --python /Users/ahmetbilalozgun/Documents/Projects/Dersler/veribilimi/.venv/bin/python3
```

---

## Code Conventions

| Convention | Rule |
|------------|------|
| Random state | `RANDOM_STATE = 42`, used in every stochastic operation |
| DataFrame names | `df_orders`, `df_items`, `df_products`, `df_customers`, `df_reviews`, `df_payments`, `df_master` |
| Figures | `fig, ax = plt.subplots(...)` → `plt.tight_layout()` → `plt.show()` |
| Notebook sections | Must match section outline in this doc |
| Language | English code + variable names; Turkish/English markdown interpretations acceptable |
| Git | Commit after each phase completes |

---

## Limitations (for report)

1. **Cold-start**: customers with 0 prior orders → CF component unusable → pure CB fallback
2. **Order sparsity**: ~90% of Olist customers have exactly 1 order → CF learns minimal personalization
3. **Category-level resolution**: CF on categories not product_ids → coarse recommendations
4. **Delivery filter**: only delivered orders → no signal from abandoned carts or returns
5. **Embedding language**: product descriptions are Portuguese; model handles it but descriptions are short (metadata string only, not full description text)
6. **No temporal dynamics**: model ignores when purchases happened (no recency weighting)
