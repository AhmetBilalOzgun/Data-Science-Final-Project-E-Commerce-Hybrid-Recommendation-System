# Design Document — E-Commerce Hybrid Recommendation System

## Dataset Switch Rationale

**Original dataset**: Olist Brazilian E-Commerce (8 CSVs)  
**Switched to**: UCI Online Retail (single CSV, 541,909 rows)

**Why the switch**: Olist had 98.62% sparsity, average 1.01 interactions per user, and 100% cold-start
test users — SVD produced only +0.2% RMSE over the global mean, indistinguishable from noise.
UCI Online Retail has avg 4.25 invoices per customer and 71.8% warm test users, enabling SVD to learn
real latent factors. Final SVD lift: **+2.6% RMSE** over global mean for warm users (vs 0% on Olist).

**Framing change**: No explicit star ratings in UCI. This is now an **implicit-purchase recommender**:
rating proxy = purchase frequency (log1p + min-max scaled to [1,5]) computed *within each temporal
window* to prevent future-leakage into training labels.

---

## Architecture Overview

```
Online_Retail.csv (UCI)
        │
        ▼
 Phase 2: Load + Clean
 ├── Drop null CustomerID (135k rows, anonymous guests)
 ├── Cast CustomerID float→str (Excel artifact)
 ├── Drop cancelled invoices (InvoiceNo starts with 'C')
 ├── Drop Quantity ≤ 0 and UnitPrice ≤ 0
 ├── Filter non-product StockCodes (regex ^[0-9]{5}[A-Za-z]?$)
 └── Aggregate: modal Description + median UnitPrice per StockCode
        │ df_master (~396k rows, 4334 users, 3658 products)
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
Collaborative Filtering (CF)          Content-Based Filtering (CB)
customer_id × product_id matrix       sentence-transformer embeddings
Temporal split: cutoff 2011-10-01     all-MiniLM-L6-v2, 384-dim
Rating: log1p(InvoiceNo.nuniq)        Feature: "{description} {price_bucket}"
        per window (no leakage)       FAISS IndexFlatIP (cosine on L2-norm)
SVD(n_factors=100, n_epochs=20)       Item-level index (one embed per StockCode)
        │                                         │
        └──────────────┬──────────────────────────┘
                       ▼
              Hybrid Score
       0.6 × CF_normalized + 0.4 × CB_cosine
       (exclude already-purchased products)
```

---

## Dataset

**Source**: UCI Online Retail Dataset (UC Irvine Machine Learning Repository, 2015)  
**URL**: https://archive.ics.uci.edu/dataset/352/online+retail  
**License**: Open/Public (UCI Machine Learning Repository)  
**File**: `data/raw/Online_Retail.csv` (45.8MB, converted from .xlsx)

| Column | Role | Notes |
|--------|------|-------|
| CustomerID | CF user axis | float64 in CSV → cast int→str |
| StockCode | CF/CB item axis | regex filter removes service codes |
| Description | CB feature text | modal per StockCode (varies across invoices) |
| InvoiceDate | Temporal split | datetime, cutoff 2011-10-01 |
| InvoiceNo | Purchase event ID | prefix 'C' = cancellation, drop |
| Quantity | Item count | drop ≤ 0 (returns/corrections) |
| UnitPrice | Item price (£) | drop ≤ 0; median per product → price_bucket |
| Country | EDA segmentation | 91% UK customers |

**Post-cleaning stats**: 396,046 rows | 4,334 customers | 3,658 products | 18,401 invoices | 37 countries

---

## User-Item Matrix (for Collaborative Filtering)

| Dimension | Choice | Reason |
|-----------|--------|--------|
| Rows | customer_id | 4,334 unique customers |
| Columns | product_id (StockCode) | 3,658 items — item-level personalization |
| Values | synthetic_rating (see below) | No explicit ratings; use purchase frequency |
| Rating scale | 1–5 | Compatible with surprise Reader(rating_scale=(1,5)) |

**Rating synthesis** (within-window, no leakage):
```python
freq = df_train.groupby(["customer_id","product_id"])["invoice_no"].nunique()
log_freq = np.log1p(freq)
synthetic_rating = 1 + 4 * (log_freq - log_freq.min()) / (log_freq.max() - log_freq.min())
synthetic_rating = synthetic_rating.clip(1, 5)
```
Applied separately to train and test windows. A rating of 2.3 means "moderate repurchase frequency
relative to the training distribution", not "2.3 stars".

**Sparsity**: train 98.5%, test 98.6% — typical for retail implicit feedback data.

---

## Collaborative Filtering — SVD

**Library**: scikit-surprise  
**Algorithm**: Simon Funk SVD (regularized matrix factorization)

**Hyperparameters**:
```python
SVD(n_factors=100, n_epochs=20, random_state=42)
```

**Evaluation**:
- Single temporal split (cutoff 2011-10-01) — train on stable season, test on holiday spike
- Warm-user RMSE: SVD 0.3776 vs global-mean baseline 0.3877 → **+2.6% lift**
- All-user RMSE: SVD 0.3517 vs baseline 0.3593 → **+2.1% lift**
- 71.8% test users seen in training (warm) — enables meaningful personalization

**Top-N generation**: predict score for all un-purchased products → sort descending → top-5.

---

## Content-Based Filtering — Sentence Transformers + FAISS

**Embedding model**: `all-MiniLM-L6-v2`  
- 384-dimensional vectors (same as multilingual predecessor)
- English-optimized (~2× faster than `paraphrase-multilingual-MiniLM-L12-v2`)
- Cached at `data/cb_embeddings.npy` after first encode

**Feature string per product**:
```python
metadata_str = f"{product_description} {price_bucket}"
# price_bucket: budget (<£2) | low (£2-5) | mid (£5-20) | premium (£20+)
```

**Index construction**:
```python
embeddings = model.encode(metadata_list, normalize_embeddings=True)  # L2-normalized
index = faiss.IndexFlatIP(384)  # inner product on normalized = cosine similarity
index.add(embeddings)
```

**Customer profile for CB scoring**:
```python
profile_emb = embeddings[purchased_product_indices].mean(axis=0)
profile_emb /= np.linalg.norm(profile_emb) + 1e-9
cb_score = candidate_embeddings @ profile_emb
```

**Semantic coherence verification**: within-prefix cosine > cross-prefix cosine for all top-5 prefixes
(SET, PINK, BLUE, RED, VINTAGE) — confirmed ✓ in notebook.

---

## Hybrid Combination

```python
ALPHA = 0.6  # weight toward CF (purchase history over content metadata)

cf_score_normalized = (predicted_rating - 1) / 4  # map [1,5] → [0,1]
cb_score = cosine_similarity                        # already in [0,1]

hybrid_score = ALPHA * cf_score_normalized + (1 - ALPHA) * max(cb_score, 0)
```

**α=0.6 justification**: purchase frequency is a direct behavioural signal (stronger than metadata);
CB provides cold-start coverage and semantic diversity. Exclusion of already-purchased products
ensures recommendations represent discovery.

---

## Research Questions

### RQ1: Do Loyal Customers Show Higher Per-Invoice Product Diversity?
**Question**: Loyal customers have more invoices — but is their *per-invoice* product variety different?  
**Metric**: `diversity_rate = n_distinct_products / invoice_count` (normalizes for exposure)  
**Method**: Mann-Whitney U test (non-parametric, no normality assumption) on diversity_rate; violin plot.  
**Segment**: Loyal (≥3 invoices, n=1,998) vs Occasional (1–2 invoices, n=2,336)  
**Result**: U = 2,045,312, p = 2.2e-12 — highly significant difference.

### RQ2: Does SVD Outperform a Non-Personalized Baseline?
**Question**: Can learned latent factors beat global-mean prediction for purchase frequency?  
**Method**: Single temporal split (2011-10-01). SVD vs global-mean baseline. Warm-user filter.  
**Result**: SVD +2.6% RMSE improvement for warm users — meaningful for implicit feedback at this sparsity.

### RQ3: CB Coherence + Hybrid Blending
**RQ3a**: Do sentence-transformer embeddings capture meaningful product relationships?  
→ Verified via within-prefix vs cross-prefix cosine similarity (5/5 prefixes ✓)  
**RQ3b**: Does hybrid blend CF and CB signals?  
→ CF vs CB Jaccard@5 = 0.00 (fully complementary); CB vs Hybrid = 0.37 (CB contributes to hybrid)

---

## EDA Visualization Plan

| # | Viz Type | Variables | Library |
|---|----------|-----------|---------|
| 3.1 | Line chart | Monthly invoice volume over time | matplotlib |
| 3.2 | Horizontal bar | Top-20 products by invoice count | matplotlib |
| 3.3 | Bar chart | Top-10 countries by unique customers | matplotlib |
| 3.4 | Histogram (log scale) | Purchase frequency per customer | matplotlib |
| 3.5 | Histogram | Synthetic rating distribution preview | matplotlib |
| 3.6 | Violin plot | Diversity rate: loyal vs occasional | seaborn |

6 distinct viz types → satisfies assignment's 4+ requirement.

---

## Package Dependencies

```
pandas>=2.0
numpy>=2.0
matplotlib>=3.9
seaborn>=0.13
scipy>=1.18             # stats.mannwhitneyu
scikit-surprise>=1.1    # SVD, Dataset, Reader, accuracy
faiss-cpu>=1.14         # IndexFlatIP
sentence-transformers>=3.0  # SentenceTransformer, encode
```

---

## Limitations (for report)

1. **Implicit feedback**: purchase frequency ≠ preference; customers may buy items they return or dislike
2. **UK concentration**: 91% UK customers → country-level analysis unreliable
3. **B2B vs B2C mix**: UCI includes wholesale buyers (very high quantities) alongside retail — different behaviour patterns not distinguished
4. **Cold-start (28%)**: test users not in training fall back to global mean for CF component
5. **No temporal dynamics**: model ignores recency — recent purchases not weighted higher
6. **Price_bucket only**: CB metadata is description + price bucket only; no image, size, category hierarchy
