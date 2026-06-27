# E-Commerce Recommendation System
### Data Science Final Project — 2025/2026 Spring Semester

## Project Team
| Student ID | Name |
|------------|------|
| 1306240133 | Ahmet Bilal Özgün |

---

## Project Summary

A hybrid product recommendation system built on the UCI Online Retail dataset (implicit-purchase recommender). Combines two approaches:

- **Collaborative Filtering**: SVD on purchase-frequency synthetic ratings (scikit-surprise), `customer_id × product_id` matrix
- **Content-Based Filtering**: Semantic product similarity via English sentence embeddings (`all-MiniLM-L6-v2`) and FAISS vector search index
- **Hybrid**: `0.4 × CF_normalized + 0.6 × CB_cosine` weighted blend; already-purchased products are excluded

### Research Questions
1. Do loyal customers (≥3 invoices) show higher product diversity per invoice than occasional buyers? (Mann-Whitney U + diversity_rate)
2. How much does SVD improve over a global-mean baseline? (temporal split, warm-user focus)
3. Are sentence-transformer embeddings semantically coherent, and does the hybrid blend CF and CB signals? (cosine similarity + Jaccard@5)

---

## Dataset

**UCI Online Retail Dataset**
- Source: [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)
- License: Open/Public (UCI Archive)
- Single CSV file, 541,909 rows, December 2010 – December 2011
- Post-cleaning: 396,046 rows | 4,334 customers | 3,658 products | 18,401 invoices

> The data file is not included in the repository due to size. See setup steps below.

---

## Setup

### 1. Clone the repository
```bash
git clone <repo-url>
cd finalProject
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` requires PyTorch (~800 MB). First install may take a while.
> The `all-MiniLM-L6-v2` model (~90 MB) is downloaded automatically on first run.

### 4. Download the data
Download `Online Retail.xlsx` from [UCI Archive](https://archive.ics.uci.edu/dataset/352/online+retail), convert to CSV, and place in `data/raw/`:

```python
import pandas as pd
df = pd.read_excel("Online Retail.xlsx")
df.to_csv("data/raw/Online_Retail.csv", index=False)
```

```
data/raw/
└── Online_Retail.csv    ← 45.8 MB, 541,909 rows
```

### 5. Run the notebook
```bash
jupyter lab notebook.ipynb
```

Use Kernel → **Restart & Run All** to execute all cells.  
On first run, `data/cb_embeddings.npy` is generated (~2 min). Subsequent runs load from cache.

---

## Libraries

For exact reproducibility, use the pinned versions in `requirements.txt`. Key packages:

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | 3.0.1 | Data loading and processing |
| numpy | 2.4.3 | Numerical computations |
| matplotlib | 3.11.0 | Visualization |
| seaborn | 0.13.2 | Statistical visualization |
| scipy | 1.18.0 | Statistical tests (Mann-Whitney U) |
| scikit-surprise | 1.1.5 | SVD collaborative filtering |
| faiss-cpu | 1.14.3 | Vector similarity index |
| sentence-transformers | 5.6.0 | Semantic embeddings (all-MiniLM-L6-v2) |
| jupyterlab | 4.5.6 | Notebook execution |
| reportlab | 5.0.0 | PDF report generation |

---

## Project Structure

```
finalProject/
├── README.md
├── CLAUDE.md                    # AI development environment context
├── todos.md                     # Task tracking list
├── design.md                    # Architecture and design decisions
├── notebook.ipynb               # Main deliverable
├── prompts.md                   # AI prompt log
├── requirements.txt             # Dependencies
├── data/
│   ├── raw/Online_Retail.csv    # UCI dataset (gitignored)
│   └── cb_embeddings.npy        # Embedding cache (generated on first run)
└── report/
    ├── report.pdf               # Short report (3–5 pages)
    └── build_report.py          # Report generation script
```

---

## Deadline

**July 3, 2026 at 12:30**
