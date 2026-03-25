import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DEFAULT = "data/processed/real_estate_nlp_base.parquet"
OUTPUT_PARQUET = "listing_clusters.parquet"
REPORT_JSON = "listing_cluster_report.json"
GENERIC_CLUSTER_TERMS = {
    "and",
    "the",
    "with",
    "of",
    "in",
    "to",
    "for",
    "this",
    "is",
    "your",
    "home",
    "room",
    "living",
    "property",
}


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_top_terms(tfidf_matrix, vectorizer: TfidfVectorizer, cluster_ids: np.ndarray, cluster_id: int, top_n: int = 8) -> list[str]:
    cluster_mask = cluster_ids == cluster_id
    if cluster_mask.sum() == 0:
        return []
    mean_scores = np.asarray(tfidf_matrix[cluster_mask].mean(axis=0)).ravel()
    top_indices = mean_scores.argsort()[::-1][:80]
    feature_names = vectorizer.get_feature_names_out()
    terms = []
    for idx in top_indices:
        term = str(feature_names[idx]).strip().lower()
        if mean_scores[idx] <= 0 or term in GENERIC_CLUSTER_TERMS:
            continue
        if len(term) <= 2:
            continue
        terms.append(term)
        if len(terms) >= top_n:
            break
    return terms


def label_cluster(top_terms: list[str], dominant_type: str) -> str:
    term_blob = " ".join(top_terms).lower()
    if any(term in term_blob for term in ["veridian", "promotions", "specifications", "discretion", "change at"]):
        return "Builder / Template Copy"
    if any(term in term_blob for term in ["construction", "custom", "plan", "build", "builder"]):
        return "New Construction / Planned Homes"
    if any(term in term_blob for term in ["acre", "lot", "land", "parcel", "mountain"]):
        return "Land / Acreage"
    if any(term in term_blob for term in ["rent", "tenant", "lease", "unit", "apartment", "income"]):
        return "Investment / Income Property"
    if any(term in term_blob for term in ["historic", "colonial", "victorian", "charming", "character"]):
        return "Historic / Character Homes"
    if any(term in term_blob for term in ["granite", "quartz", "suite", "views", "luxury", "custom cabinetry"]):
        return "Upscale Finishes / Premium Homes"
    if dominant_type in {"land", "farm"}:
        return "Land / Acreage"
    if dominant_type in {"multi_family", "apartment", "duplex_triplex"}:
        return "Multi-Unit / Income Listings"
    if dominant_type in {"condos", "townhomes", "coop", "condop"}:
        return "Attached / Urban Homes"
    return "Everyday Residential Listings"


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster listing descriptions into thematic groups.")
    parser.add_argument("--input-path", default=INPUT_DEFAULT, help="Path to the NLP base table")
    parser.add_argument("--output-dir", default="data/processed", help="Directory where outputs should be written")
    parser.add_argument("--clusters", type=int, default=10, help="Number of clusters")
    args = parser.parse_args()

    input_path = resolve_project_path(args.input_path)
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    cluster_df = df[df["is_quality_filtered_text"]].copy()
    text = cluster_df["text_clean"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=5)
    tfidf_matrix = vectorizer.fit_transform(text)
    svd = TruncatedSVD(n_components=50, random_state=42)
    embedding = svd.fit_transform(tfidf_matrix)
    kmeans = KMeans(n_clusters=args.clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(embedding)

    cluster_df["cluster_id"] = cluster_ids
    cluster_df["cluster_x"] = embedding[:, 0]
    cluster_df["cluster_y"] = embedding[:, 1]
    cluster_df["cluster_z"] = embedding[:, 2]

    summary_rows = []
    for cluster_id in sorted(cluster_df["cluster_id"].unique()):
        cluster_slice = cluster_df[cluster_df["cluster_id"] == cluster_id].copy()
        top_terms = get_top_terms(tfidf_matrix, vectorizer, cluster_ids, cluster_id)
        dominant_type = str(cluster_slice["type"].mode().iloc[0]) if cluster_slice["type"].notna().any() else "missing"
        summary_rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_label": label_cluster(top_terms, dominant_type),
                "row_count": int(len(cluster_slice)),
                "avg_price": float(cluster_slice["list_price"].mean()),
                "avg_quality": float(cluster_slice["quality_score"].mean()),
                "dominant_type": dominant_type,
                "top_terms": top_terms,
                "cluster_x": float(cluster_slice["cluster_x"].mean()),
                "cluster_y": float(cluster_slice["cluster_y"].mean()),
                "cluster_z": float(cluster_slice["cluster_z"].mean()),
            }
        )

    cluster_summary = pd.DataFrame(summary_rows).sort_values("avg_price", ascending=False)

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_df.to_parquet(output_dir / OUTPUT_PARQUET, index=False)
    with (output_dir / REPORT_JSON).open("w", encoding="utf-8") as handle:
        json.dump(cluster_summary.round(4).to_dict(orient="records"), handle, indent=2)

    print(cluster_summary.to_string(index=False))


if __name__ == "__main__":
    main()
