import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DEFAULT = "data/processed/real_estate_nlp_base.parquet"


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Find semantically similar listing descriptions using TF-IDF cosine similarity.")
    parser.add_argument("--input-path", default=INPUT_DEFAULT, help="Path to the NLP base table")
    parser.add_argument("--record-id", type=int, help="Record id to use as the search anchor")
    parser.add_argument("--query-text", help="Free-text query to search against the listing corpus")
    parser.add_argument("--top-k", type=int, default=5, help="Number of matches to return")
    args = parser.parse_args()

    if not args.record_id and not args.query_text:
        raise SystemExit("Provide either --record-id or --query-text")

    input_path = resolve_project_path(args.input_path)
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    search_df = df[df["is_quality_filtered_text"]].copy()
    corpus = search_df["text_clean"].fillna("").astype(str)
    vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=3)
    matrix = vectorizer.fit_transform(corpus)

    if args.record_id:
        anchor = search_df.loc[search_df["record_id"] == args.record_id]
        if anchor.empty:
            raise SystemExit(f"record_id {args.record_id} not found")
        query_vector = vectorizer.transform(anchor["text_clean"].astype(str))
    else:
        query_vector = vectorizer.transform([args.query_text])

    similarity = cosine_similarity(query_vector, matrix).ravel()
    top_idx = similarity.argsort()[::-1][: args.top_k]
    results = search_df.iloc[top_idx][["record_id", "state", "type", "list_price", "text"]].copy()
    results["similarity"] = similarity[top_idx]
    print(results.to_string(index=False, max_colwidth=160))


if __name__ == "__main__":
    main()
