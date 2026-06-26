"""Build the Phase 8 PDF report for the Olist recommendation project.

Run from the repository root:
    python report/build_report.py
"""

from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from scipy import stats
from surprise import Dataset, Reader, SVD, accuracy


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw"
EMBED_CACHE = ROOT / "data" / "cb_embeddings.npy"
TMP_DIR = ROOT / "tmp" / "pdfs" / "phase8"
OUT_PDF = ROOT / "report" / "report.pdf"
RANDOM_STATE = 42

os.environ.setdefault("MPLCONFIGDIR", str(TMP_DIR / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_and_clean_data() -> dict[str, pd.DataFrame]:
    df_orders_raw = pd.read_csv(
        DATA_PATH / "olist_orders_dataset.csv",
        parse_dates=["order_purchase_timestamp"],
    )
    df_items = pd.read_csv(DATA_PATH / "olist_order_items_dataset.csv")
    df_products_raw = pd.read_csv(DATA_PATH / "olist_products_dataset.csv")
    df_customers = pd.read_csv(DATA_PATH / "olist_customers_dataset.csv")
    df_reviews = pd.read_csv(
        DATA_PATH / "olist_order_reviews_dataset.csv",
        parse_dates=["review_answer_timestamp"],
    )
    df_payments = pd.read_csv(DATA_PATH / "olist_order_payments_dataset.csv")
    df_trans = pd.read_csv(DATA_PATH / "product_category_name_translation.csv")

    df_orders = df_orders_raw[df_orders_raw["order_status"] == "delivered"].reset_index(drop=True)
    df_products = df_products_raw.dropna(subset=["product_category_name"]).reset_index(drop=True)
    df_products["product_description_lenght"] = (
        df_products["product_description_lenght"].fillna(0).astype(int)
    )

    df_payments_agg = (
        df_payments.groupby("order_id", as_index=False)
        .agg(payment_value=("payment_value", "sum"), payment_installments=("payment_installments", "max"))
    )
    df_reviews_dedup = (
        df_reviews[["order_id", "review_score", "review_answer_timestamp"]]
        .sort_values("review_answer_timestamp", ascending=False, na_position="last")
        .drop_duplicates("order_id")[["order_id", "review_score"]]
    )

    df_master = (
        df_orders.merge(df_items, on="order_id", how="inner")
        .merge(df_products, on="product_id", how="inner")
        .merge(df_customers, on="customer_id", how="inner")
        .merge(df_reviews_dedup, on="order_id", how="left")
        .merge(df_payments_agg, on="order_id", how="left")
    )

    return {
        "orders_raw": df_orders_raw,
        "orders": df_orders,
        "items": df_items,
        "products_raw": df_products_raw,
        "products": df_products,
        "customers": df_customers,
        "reviews": df_reviews,
        "payments": df_payments,
        "translations": df_trans,
        "master": df_master,
    }


def label_categories(values: pd.Series, translations: pd.DataFrame) -> pd.Series:
    lookup = dict(
        zip(
            translations["product_category_name"],
            translations["product_category_name_english"],
            strict=False,
        )
    )
    return values.map(lookup).fillna(values).str.replace("_", " ").str.title()


def compute_metrics(data: dict[str, pd.DataFrame]) -> dict[str, object]:
    df_master = data["master"].copy()
    df_items = data["items"]
    df_products = data["products"].copy()
    order_view = df_master.drop_duplicates("order_id")

    metrics: dict[str, object] = {
        "orders_raw": len(data["orders_raw"]),
        "orders_delivered": len(data["orders"]),
        "products_raw": len(data["products_raw"]),
        "products_after_drop": len(df_products),
        "master_rows": len(df_master),
        "master_cols": df_master.shape[1],
    }

    review_scores = order_view["review_score"].dropna()
    metrics["review_mean"] = float(review_scores.mean())
    metrics["review_distribution"] = (
        review_scores.value_counts(normalize=True).sort_index().mul(100).round(1).to_dict()
    )

    monthly = order_view.set_index("order_purchase_timestamp").resample("ME").size()
    metrics["monthly_peak_month"] = monthly.idxmax().strftime("%Y-%m")
    metrics["monthly_peak_orders"] = int(monthly.max())

    top_states = (
        data["customers"].drop_duplicates("customer_unique_id")["customer_state"]
        .value_counts(normalize=True)
        .head(5)
        .mul(100)
        .round(1)
    )
    metrics["top_states_pct"] = top_states.to_dict()

    cat_counts = df_master.groupby("customer_unique_id")["product_category_name"].nunique().rename("n_categories")
    df_rq1 = df_master.merge(cat_counts, on="customer_unique_id").dropna(subset=["review_score"])
    df_rq1["is_multi_category"] = df_rq1["n_categories"] >= 2
    df_rq1["buyer_type"] = np.where(df_rq1["is_multi_category"], "multi-category", "single-category")
    top5_states = df_rq1["customer_state"].value_counts().head(5).index.tolist()
    rq1_means = (
        df_rq1.groupby(["customer_state", "buyer_type"])["review_score"]
        .mean()
        .unstack()
        .loc[top5_states]
    )
    metrics["rq1_top5_states"] = top5_states
    metrics["rq1_means"] = rq1_means

    cust_scores = (
        df_rq1.groupby(["customer_unique_id", "is_multi_category"])["review_score"]
        .mean()
        .reset_index()
        .dropna(subset=["review_score"])
    )
    multi_scores = cust_scores[cust_scores["is_multi_category"]]["review_score"]
    single_scores = cust_scores[~cust_scores["is_multi_category"]]["review_score"]
    stat, p_value = stats.mannwhitneyu(multi_scores, single_scores, alternative="two-sided")
    n_multi, n_single = len(multi_scores), len(single_scores)
    z_score = (stat - n_multi * n_single / 2) / np.sqrt(
        n_multi * n_single * (n_multi + n_single + 1) / 12
    )
    effect_r = abs(z_score) / np.sqrt(n_multi + n_single)
    metrics["rq1_test"] = {
        "u": float(stat),
        "p": float(p_value),
        "n_multi": n_multi,
        "n_single": n_single,
        "z": float(z_score),
        "r": float(effect_r),
        "multi_mean": float(multi_scores.mean()),
        "single_mean": float(single_scores.mean()),
    }

    df_ui = (
        df_master.dropna(subset=["review_score"])
        .groupby(["customer_id", "product_category_name"], as_index=False)["review_score"]
        .mean()
        .rename(columns={"review_score": "mean_review_score"})
    )
    n_users = df_ui["customer_id"].nunique()
    n_cats = df_ui["product_category_name"].nunique()
    metrics["rq2_matrix"] = {
        "pairs": len(df_ui),
        "users": n_users,
        "categories": n_cats,
        "sparsity": 1 - len(df_ui) / (n_users * n_cats),
        "avg_cats_per_user": len(df_ui) / n_users,
    }

    reader = Reader(rating_scale=(1, 5))
    cutoff = "2018-01-01"
    mask_train = df_master["order_purchase_timestamp"] < cutoff
    df_train = (
        df_master[mask_train].dropna(subset=["review_score"])
        .groupby(["customer_id", "product_category_name"], as_index=False)["review_score"]
        .mean()
        .rename(columns={"review_score": "mean_review_score"})
    )
    df_test = (
        df_master[~mask_train].dropna(subset=["review_score"])
        .groupby(["customer_id", "product_category_name"], as_index=False)["review_score"]
        .mean()
        .rename(columns={"review_score": "mean_review_score"})
    )
    users_train = set(df_train["customer_id"])
    users_test = set(df_test["customer_id"])
    warm_pct = len(users_train & users_test) / len(users_test) * 100

    trainset = Dataset.load_from_df(
        df_train[["customer_id", "product_category_name", "mean_review_score"]], reader
    ).build_full_trainset()
    testset = Dataset.load_from_df(
        df_test[["customer_id", "product_category_name", "mean_review_score"]], reader
    ).build_full_trainset().build_testset()
    svd = SVD(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=RANDOM_STATE,
    )
    svd.fit(trainset)
    preds = svd.test(testset)
    global_mean = trainset.global_mean
    test_list = list(testset)
    rmse_svd = accuracy.rmse(preds, verbose=False)
    rmse_base = float(np.sqrt(np.mean([(r - global_mean) ** 2 for _, _, r in test_list])))
    warm_preds = [pred for pred in preds if pred.uid in users_train]
    rmse_warm = accuracy.rmse(warm_preds, verbose=False) if warm_preds else np.nan

    metrics["rq2_primary"] = {
        "train_pairs": len(df_train),
        "test_pairs": len(df_test),
        "warm_pct": warm_pct,
        "cold_pct": 100 - warm_pct,
        "svd_rmse": float(rmse_svd),
        "base_rmse": rmse_base,
        "improvement_pct": (rmse_base - rmse_svd) / rmse_base * 100,
        "warm_rmse": float(rmse_warm) if not np.isnan(rmse_warm) else None,
        "global_mean": float(global_mean),
    }

    cv_rows = []
    for cv_cutoff in ["2017-07-01", "2017-10-01", "2018-01-01"]:
        cutoff_dt = pd.Timestamp(cv_cutoff)
        end_dt = cutoff_dt + pd.DateOffset(months=3)
        cv_train_mask = df_master["order_purchase_timestamp"] < cutoff_dt
        cv_test_mask = (
            (df_master["order_purchase_timestamp"] >= cutoff_dt)
            & (df_master["order_purchase_timestamp"] < end_dt)
        )
        cv_train = (
            df_master[cv_train_mask].dropna(subset=["review_score"])
            .groupby(["customer_id", "product_category_name"], as_index=False)["review_score"]
            .mean()
            .rename(columns={"review_score": "mean_review_score"})
        )
        cv_test = (
            df_master[cv_test_mask].dropna(subset=["review_score"])
            .groupby(["customer_id", "product_category_name"], as_index=False)["review_score"]
            .mean()
            .rename(columns={"review_score": "mean_review_score"})
        )
        cv_trainset = Dataset.load_from_df(
            cv_train[["customer_id", "product_category_name", "mean_review_score"]], reader
        ).build_full_trainset()
        cv_testset = Dataset.load_from_df(
            cv_test[["customer_id", "product_category_name", "mean_review_score"]], reader
        ).build_full_trainset().build_testset()
        cv_model = SVD(
            n_factors=100,
            n_epochs=20,
            lr_all=0.005,
            reg_all=0.02,
            random_state=RANDOM_STATE,
        )
        cv_model.fit(cv_trainset)
        cv_preds = cv_model.test(cv_testset)
        cv_global_mean = cv_trainset.global_mean
        cv_rmse_svd = accuracy.rmse(cv_preds, verbose=False)
        cv_rmse_base = float(
            np.sqrt(np.mean([(r - cv_global_mean) ** 2 for _, _, r in list(cv_testset)]))
        )
        cv_rows.append(
            {
                "cutoff": cv_cutoff,
                "window": f"{cv_cutoff[:7]} to {end_dt.strftime('%Y-%m')}",
                "train": len(cv_train),
                "test": len(cv_test),
                "svd_rmse": float(cv_rmse_svd),
                "base_rmse": cv_rmse_base,
                "improvement_pct": (cv_rmse_base - cv_rmse_svd) / cv_rmse_base * 100,
            }
        )
    metrics["rq2_cv"] = cv_rows
    metrics["rq2_cv_mean_svd"] = float(np.mean([row["svd_rmse"] for row in cv_rows]))
    metrics["rq2_cv_std_svd"] = float(np.std([row["svd_rmse"] for row in cv_rows], ddof=1))

    metrics["rq3"] = compute_rq3_metrics(data)

    return metrics


def compute_rq3_metrics(data: dict[str, pd.DataFrame]) -> dict[str, object]:
    if not EMBED_CACHE.exists():
        raise FileNotFoundError(f"Missing embedding cache: {EMBED_CACHE}")

    df_items = data["items"]
    df_products = data["products"].copy()
    df_cb = (
        df_products[["product_id", "product_category_name", "product_description_lenght", "product_weight_g"]]
        .copy()
        .merge(df_items.groupby("product_id")["price"].median().rename("price_median"), on="product_id", how="left")
    )
    df_cb["price_bucket"] = pd.cut(
        df_cb["price_median"].fillna(df_cb["price_median"].median()),
        bins=[-np.inf, 50, 150, 500, np.inf],
        labels=["budget", "mid", "premium", "luxury"],
    ).astype(str)
    df_cb["weight_bucket"] = pd.cut(
        df_cb["product_weight_g"].fillna(0),
        bins=[-np.inf, 500, 2000, 10000, np.inf],
        labels=["light", "medium", "heavy", "bulky"],
    ).astype(str)

    embeddings = np.load(EMBED_CACHE).astype(np.float32)
    if embeddings.shape[0] != len(df_cb):
        raise ValueError(
            f"Embedding cache row count mismatch: {embeddings.shape[0]} vectors for {len(df_cb)} products."
        )
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    pid_to_idx = {pid: i for i, pid in enumerate(df_cb["product_id"])}

    def recommend_similar(product_id: str, k: int = 5) -> set[str]:
        source_idx = pid_to_idx[product_id]
        _, indices = index.search(embeddings[source_idx : source_idx + 1], k + 1)
        recs: list[str] = []
        for idx in indices[0]:
            if idx == source_idx:
                continue
            recs.append(str(df_cb.iloc[idx]["product_id"]))
            if len(recs) >= k:
                break
        return set(recs)

    order_groups = df_items.groupby("order_id")["product_id"].apply(set)
    order_item_counts = df_items.groupby("order_id").size()
    copurchase_map: defaultdict[str, set[str]] = defaultdict(set)
    for products in order_groups:
        for product_id in products:
            copurchase_map[product_id] |= products - {product_id}

    top_products = [
        product_id
        for product_id in df_items["product_id"].value_counts().head(50).index
        if product_id in pid_to_idx
    ]
    overlap_scores = [
        len(recommend_similar(product_id, k=5) & copurchase_map.get(product_id, set()))
        for product_id in top_products
    ]

    return {
        "top_n": len(overlap_scores),
        "mean_overlap": float(np.mean(overlap_scores)),
        "nonempty_copurchase": sum(1 for product_id in top_products if copurchase_map.get(product_id)),
        "single_item_pct": float((order_item_counts == 1).mean() * 100),
        "single_unique_pct": float((order_groups.apply(len) == 1).mean() * 100),
    }


def make_charts(data: dict[str, pd.DataFrame], metrics: dict[str, object]) -> dict[str, Path]:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 180, "font.size": 9})

    df_master = data["master"]
    translations = data["translations"]
    order_view = df_master.drop_duplicates("order_id")
    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    monthly = order_view.set_index("order_purchase_timestamp").resample("ME").size()
    monthly.plot(ax=ax, color="#24577a", linewidth=2)
    ax.set_title("Monthly delivered order volume")
    ax.set_xlabel("")
    ax.set_ylabel("Orders")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    paths["monthly"] = TMP_DIR / "monthly_orders.png"
    fig.savefig(paths["monthly"], bbox_inches="tight")
    plt.close(fig)

    top_cats = df_master["product_category_name"].value_counts().head(10).sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    labels = label_categories(top_cats.index.to_series(), translations)
    ax.barh(labels, top_cats.values, color="#4f7f55")
    ax.set_title("Top product categories by item rows")
    ax.set_xlabel("Item rows")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    paths["categories"] = TMP_DIR / "top_categories.png"
    fig.savefig(paths["categories"], bbox_inches="tight")
    plt.close(fig)

    rq1_means = metrics["rq1_means"]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    rq1_means[["single-category", "multi-category"]].plot(kind="bar", ax=ax, color=["#587a9d", "#b05a42"])
    ax.set_title("Mean review score by state and buyer type")
    ax.set_xlabel("Customer state")
    ax.set_ylabel("Mean review score")
    ax.set_ylim(3.4, 4.35)
    ax.legend(title="", loc="lower right", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths["rq1"] = TMP_DIR / "rq1_state_buyer_type.png"
    fig.savefig(paths["rq1"], bbox_inches="tight")
    plt.close(fig)

    return paths


def stylesheet() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
            spaceAfter=16,
        ),
        "Heading": ParagraphStyle(
            "Heading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=15,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#1f3344"),
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=12.5,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "Small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            spaceAfter=4,
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#555555"),
            spaceAfter=6,
        ),
    }
    return styles


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def metric_table(rows: list[list[object]], col_widths: list[float] | None = None) -> Table:
    table = Table(rows, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f3344")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c7cdd3")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fa")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def page_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(1.7 * cm, 1.1 * cm, "E-Commerce Hybrid Recommendation System")
    canvas.drawRightString(A4[0] - 1.7 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(metrics: dict[str, object], chart_paths: dict[str, Path]) -> None:
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    styles = stylesheet()
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        rightMargin=1.65 * cm,
        leftMargin=1.65 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.65 * cm,
        title="E-Commerce Hybrid Recommendation System",
        author="Ahmet Bilal Ozgun",
    )
    story = []

    story.append(para("E-Commerce Hybrid Recommendation System", styles["Title"]))
    story.append(
        para(
            "Olist Brazilian E-Commerce Dataset | Veri Bilimi Final Project | Ahmet Bilal Ozgun",
            styles["Subtitle"],
        )
    )

    story.append(para("1. Introduction and Motivation", styles["Heading"]))
    story.append(
        para(
            "This project builds and evaluates a hybrid recommendation system for the Olist Brazilian "
            "E-Commerce dataset. The practical goal is to combine observed customer behavior with product "
            "metadata so that recommendations can reflect both preference history and semantic product "
            "similarity. The project studies three questions: whether multi-category buyers rate products "
            "differently by region, whether category-level SVD improves over a global-mean baseline, and "
            "whether semantic product similarity aligns with real co-purchase behavior.",
            styles["Body"],
        )
    )

    story.append(para("2. Dataset and Cleaning Decisions", styles["Heading"]))
    story.append(
        para(
            "The analysis uses the public Olist marketplace data from Kaggle under the CC BY-NC-SA 4.0 "
            "license. The raw data contains orders, order items, products, customers, reviews, payments, "
            "sellers, geolocation records, and category translations. The modeling dataset keeps only "
            "delivered orders because cancelled, unavailable, or in-transit orders do not represent complete "
            "purchase experiences with reliable review signals.",
            styles["Body"],
        )
    )
    cleaning_rows = [
        ["Step", "Result"],
        ["Raw orders", f"{metrics['orders_raw']:,}"],
        ["Delivered orders", f"{metrics['orders_delivered']:,}"],
        ["Products after null-category drop", f"{metrics['products_after_drop']:,}"],
        ["Final master table", f"{metrics['master_rows']:,} rows x {metrics['master_cols']} columns"],
        ["Mean review score", f"{metrics['review_mean']:.3f}"],
    ]
    story.append(metric_table(cleaning_rows, [6.8 * cm, 6.4 * cm]))
    story.append(Spacer(1, 0.12 * cm))
    story.append(
        para(
            "Rows with missing product categories were dropped because category is required by both the "
            "collaborative-filtering matrix and content metadata strings. Payment rows were aggregated to "
            "one row per order to avoid join fan-out, while duplicate review rows were resolved by keeping "
            "the latest answer timestamp. The resulting master table is item-level, so order-level summaries "
            "deduplicate by order_id.",
            styles["Body"],
        )
    )

    story.append(
        KeepTogether(
            [
                Image(str(chart_paths["monthly"]), width=16.2 * cm, height=6.25 * cm),
                para(
                    f"Figure 1. Delivered orders peak in {metrics['monthly_peak_month']} "
                    f"with {metrics['monthly_peak_orders']:,} orders.",
                    styles["Caption"],
                ),
            ]
        )
    )
    story.append(PageBreak())

    story.append(para("3. Methods", styles["Heading"]))
    story.append(
        para(
            "Collaborative filtering uses scikit-surprise SVD on a customer_id by product_category_name "
            "matrix, where each value is the mean review score for that customer-category pair. Category "
            "granularity is intentional: product_id-level modeling is too sparse for this dataset. Evaluation "
            "uses temporal splits rather than random splits, preventing future reviews from leaking into "
            "training.",
            styles["Body"],
        )
    )
    story.append(
        para(
            "Content-based filtering represents each product by a short metadata string containing category, "
            "price bucket, and weight bucket. Sentence-transformer embeddings are L2-normalized and indexed "
            "with FAISS IndexFlatIP, so inner product equals cosine similarity. The hybrid demo combines "
            "normalized SVD predictions and content similarity as hybrid_score = 0.6 * CF + 0.4 * CB, giving "
            "slightly more weight to observed behavior while retaining semantic fallback value.",
            styles["Body"],
        )
    )
    story.append(
        KeepTogether(
            [
                Image(str(chart_paths["categories"]), width=16.2 * cm, height=7.4 * cm),
                para("Figure 2. Category distribution is highly concentrated in a small number of categories.", styles["Caption"]),
            ]
        )
    )

    story.append(para("4. Results", styles["Heading"]))
    story.append(
        para(
            "RQ1 asks whether multi-category buyers rate products differently by Brazilian region. Across the "
            "top five states, multi-category buyers have lower mean review scores than single-category buyers. "
            "The Mann-Whitney U test is statistically significant, but the effect size is negligible, so the "
            "practical difference is small despite the large sample.",
            styles["Body"],
        )
    )
    rq1_test = metrics["rq1_test"]
    rq1_rows = [
        ["Metric", "Value"],
        ["U statistic", f"{rq1_test['u']:,.0f}"],
        ["p-value", f"{rq1_test['p']:.3e}"],
        ["n multi-category", f"{rq1_test['n_multi']:,}"],
        ["n single-category", f"{rq1_test['n_single']:,}"],
        ["Effect size r", f"{rq1_test['r']:.4f}"],
        ["Mean score, multi vs single", f"{rq1_test['multi_mean']:.3f} vs {rq1_test['single_mean']:.3f}"],
    ]
    story.append(metric_table(rq1_rows, [6.0 * cm, 5.2 * cm]))
    story.append(Spacer(1, 0.1 * cm))
    story.append(
        KeepTogether(
            [
                Image(str(chart_paths["rq1"]), width=16.0 * cm, height=6.8 * cm),
                para("Figure 3. Multi-category customers rate slightly lower in each of the five largest states.", styles["Caption"]),
            ]
        )
    )

    story.append(para("4. Results, Continued", styles["Heading"]))
    rq2_matrix = metrics["rq2_matrix"]
    rq2_primary = metrics["rq2_primary"]
    story.append(
        para(
            "RQ2 evaluates whether SVD improves over a global-mean baseline. The matrix contains "
            f"{rq2_matrix['pairs']:,} customer-category pairs for {rq2_matrix['users']:,} users and "
            f"{rq2_matrix['categories']} categories, with sparsity {rq2_matrix['sparsity']:.4f}. "
            "Because most Olist customers buy once, the post-2018-01 temporal test split is entirely "
            "cold-start at the customer level. SVD therefore behaves close to the global mean, producing "
            f"RMSE {rq2_primary['svd_rmse']:.4f} versus {rq2_primary['base_rmse']:.4f}, an improvement of "
            f"{rq2_primary['improvement_pct']:.2f}%.",
            styles["Body"],
        )
    )
    rq2_rows = [
        ["Split / Metric", "SVD RMSE", "Global RMSE", "Improvement"],
        [
            "Primary: train pre-2018-01, test 2018-01+",
            f"{rq2_primary['svd_rmse']:.4f}",
            f"{rq2_primary['base_rmse']:.4f}",
            f"{rq2_primary['improvement_pct']:.2f}%",
        ],
    ]
    for row in metrics["rq2_cv"]:
        rq2_rows.append(
            [
                row["window"],
                f"{row['svd_rmse']:.4f}",
                f"{row['base_rmse']:.4f}",
                f"{row['improvement_pct']:.2f}%",
            ]
        )
    story.append(metric_table(rq2_rows, [7.0 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm]))
    story.append(Spacer(1, 0.12 * cm))
    story.append(
        para(
            f"Across the three temporal validation windows, mean SVD RMSE is "
            f"{metrics['rq2_cv_mean_svd']:.4f} with standard deviation {metrics['rq2_cv_std_svd']:.4f}. "
            "The result is academically useful because it demonstrates the main limitation of this marketplace "
            "setting: sparse repeat-user history limits personalization.",
            styles["Body"],
        )
    )

    rq3 = metrics["rq3"]
    story.append(
        para(
            "RQ3 compares FAISS semantic neighbors to products that appear in the same order. The result is "
            f"mean overlap@5 = {rq3['mean_overlap']:.3f} / 5 over {rq3['top_n']} high-volume products. "
            f"This near-zero overlap is expected because {rq3['single_item_pct']:.1f}% of orders contain one "
            f"item and {rq3['single_unique_pct']:.1f}% contain one unique product after collapsing repeated "
            "quantities. In this dataset, co-purchase overlap is too sparse to be a strong content-based "
            "evaluation signal.",
            styles["Body"],
        )
    )
    rq3_rows = [
        ["RQ3 Metric", "Value"],
        ["Products evaluated", f"{rq3['top_n']}"],
        ["Mean overlap@5", f"{rq3['mean_overlap']:.3f} / 5"],
        ["Products with non-empty co-purchase set", f"{rq3['nonempty_copurchase']} / {rq3['top_n']}"],
        ["Single-item orders", f"{rq3['single_item_pct']:.1f}%"],
        ["Single-unique-product orders", f"{rq3['single_unique_pct']:.1f}%"],
    ]
    story.append(metric_table(rq3_rows, [7.4 * cm, 4.2 * cm]))

    story.append(para("5. Conclusions and Limitations", styles["Heading"]))
    story.append(
        para(
            "The hybrid approach is appropriate for the project because it combines behavioral preference "
            "learning with semantic similarity, but the data strongly constrains the measurable gains. The "
            "regional analysis finds a statistically significant but practically tiny rating difference for "
            "multi-category buyers. The SVD model barely improves over the global mean because the temporal "
            "test set is dominated by cold-start customers. The content-based system produces coherent metadata "
            "neighbors, but co-purchase validation is weak because most orders contain only one product.",
            styles["Body"],
        )
    )
    story.append(
        para(
            "Main limitations are cold-start customers, high matrix sparsity, category-level rather than "
            "product-level collaborative filtering, short product metadata strings, and no recency weighting. "
            "Future work should add richer product text, repeat-buyer segmentation, and a ranking metric based "
            "on held-out future purchases when enough user history is available.",
            styles["Body"],
        )
    )

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)


def main() -> None:
    data = load_and_clean_data()
    metrics = compute_metrics(data)
    chart_paths = make_charts(data, metrics)
    build_pdf(metrics, chart_paths)
    print(f"Wrote {OUT_PDF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
