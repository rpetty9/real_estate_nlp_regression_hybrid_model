import argparse
import json
from pathlib import Path

import pandas as pd

from nlp_feature_utils import QUALITY_LABELS, extract_amenity_flags, score_listing_quality

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = "real_estate_nlp_base.csv"
OUTPUT_PARQUET = "real_estate_nlp_base.parquet"
REPORT_JSON = "real_estate_nlp_base_report.json"


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    enriched = df.copy()

    amenity_flags = extract_amenity_flags(enriched["text_clean"])
    enriched = pd.concat([enriched, amenity_flags], axis=1)

    quality_outputs = enriched.apply(
        lambda row: score_listing_quality(row["text"], bool(row["has_duplicate_text_within_state"])),
        axis=1,
    )

    enriched["quality_score"] = [item[0] for item in quality_outputs]
    enriched["quality_label"] = [item[1] for item in quality_outputs]
    enriched["word_count"] = [item[2]["word_count"] for item in quality_outputs]
    enriched["sentence_count"] = [item[2]["sentence_count"] for item in quality_outputs]
    enriched["digit_count"] = [item[2]["digit_count"] for item in quality_outputs]
    enriched["amenity_hit_count"] = [item[2]["amenity_hit_count"] for item in quality_outputs]
    enriched["unique_word_count"] = [item[2]["unique_word_count"] for item in quality_outputs]

    enriched["is_quality_filtered_text"] = (
        ~enriched["is_blank_text"]
        & ~enriched["is_placeholder_text"]
        & (enriched["quality_label"].isin(["usable", "rich"]))
    )

    report = {
        "row_count": int(len(enriched)),
        "quality_label_counts": {
            label: int((enriched["quality_label"] == label).sum())
            for label in QUALITY_LABELS
        },
        "quality_filtered_rows": int(enriched["is_quality_filtered_text"].sum()),
        "avg_quality_score": round(float(enriched["quality_score"].mean()), 2),
        "avg_word_count": round(float(enriched["word_count"].mean()), 2),
        "top_amenities": {
            column: int(enriched[column].sum())
            for column in amenity_flags.columns
        },
        "rows_by_state": {
            key: int(value)
            for key, value in enriched["state"].value_counts().sort_index().items()
        },
    }
    return enriched, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an NLP-ready feature table from the cleaned real-estate listings.")
    parser.add_argument(
        "--input-path",
        default="data/processed/combined_real_estate_clean.parquet",
        help="Path to the cleaned combined dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where the NLP feature table should be written",
    )
    args = parser.parse_args()

    input_path = resolve_project_path(args.input_path)
    if not input_path.exists():
        raise SystemExit(f"Input dataset not found: {input_path}")

    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    enriched, report = build_feature_table(df)

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_dir / OUTPUT_CSV, index=False)
    enriched.to_parquet(output_dir / OUTPUT_PARQUET, index=False)
    with (output_dir / REPORT_JSON).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved NLP feature table to {(output_dir / OUTPUT_PARQUET).resolve()}")


if __name__ == "__main__":
    main()
