import argparse
import json
from pathlib import Path

import pandas as pd

OUTPUT_CSV = "combined_real_estate_clean.csv"
OUTPUT_PARQUET = "combined_real_estate_clean.parquet"
REPORT_JSON = "combined_real_estate_report.json"

DATASET_LABELS = {
    "wisconsin-real-estate-data-2026": "Wisconsin",
    "connecticut-real-estate-data-2026": "Connecticut",
    "real-estate-data-oregon-2026": "Oregon",
    "new-york-real-estate-data-2026": "New York",
    "new-hampshire-real-estate-data-2026": "New Hampshire",
    "indiana-real-estate-data-2026": "Indiana",
    "kentucky-real-estate-data-2026": "Kentucky",
    "colorado-real-estate-market-2026": "Colorado",
}

CORE_COLUMNS = [
    "state",
    "source_dataset",
    "source_file",
    "record_id",
    "type",
    "sub_type",
    "text",
    "text_clean",
    "list_price",
    "sqft",
    "stories",
    "beds",
    "baths",
    "baths_full",
    "baths_full_calc",
    "garage",
    "year_built",
    "text_length",
    "is_placeholder_text",
    "is_blank_text",
    "has_duplicate_text_within_state",
]

TYPE_MAP = {
    "condo": "condos",
    "condos": "condos",
    "condo_townhome_rowhome_coop": "condos",
    "coop": "coop",
    "condop": "condop",
    "land": "land",
    "farm": "farm",
    "apartment": "apartment",
    "mobile": "mobile",
    "single_family": "single_family",
    "multi_family": "multi_family",
    "multifamily": "multi_family",
    "duplex_triplex": "duplex_triplex",
    "townhome": "townhomes",
    "townhomes": "townhomes",
    "unknown": pd.NA,
}

SUB_TYPE_MAP = {
    "condo": "condo",
    "townhouse": "townhouse",
    "co_op": "co_op",
    "cond_op": "cond_op",
}

PLACEHOLDER_TEXTS = {
    "no description provided.",
    "shortsale.",
}


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def normalize_type(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().lower()
    return TYPE_MAP.get(text, text)


def normalize_sub_type(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().lower()
    return SUB_TYPE_MAP.get(text, text)


def load_frames(data_dir: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(data_dir.rglob("*.csv")):
        dataset_folder = csv_path.parent.name
        state = DATASET_LABELS.get(dataset_folder, dataset_folder)
        frame = pd.read_csv(csv_path)
        frame["state"] = state
        frame["source_dataset"] = dataset_folder
        frame["source_file"] = csv_path.name
        frames.append(frame)
    return frames


def build_raw_report(frames: list[pd.DataFrame]) -> dict[str, object]:
    raw = pd.concat(frames, ignore_index=True, sort=False)
    text = raw["text"].fillna("").astype(str).str.strip().str.lower()
    report = {
        "dataset_count": len(frames),
        "row_count_raw": int(len(raw)),
        "column_count_raw": int(len(raw.columns)),
        "exact_duplicate_rows_raw": int(raw.duplicated().sum()),
        "duplicate_text_rows_raw": int(raw["text"].fillna("").astype(str).str.strip().duplicated().sum()),
        "placeholder_text_rows_raw": int(text.isin(PLACEHOLDER_TEXTS).sum()),
        "blank_text_rows_raw": int(raw["text"].fillna("").astype(str).str.strip().eq("").sum()),
        "states": sorted(raw["state"].dropna().unique().tolist()),
    }
    return report


def clean_combined_dataset(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, object]]:
    combined = pd.concat(frames, ignore_index=True, sort=False)
    original_rows = len(combined)

    combined = combined.rename(
        columns={
            "listPrice": "list_price",
        }
    )

    combined["type"] = combined["type"].apply(normalize_type).astype("string")
    combined["sub_type"] = combined["sub_type"].apply(normalize_sub_type).astype("string")
    combined["text"] = combined["text"].apply(normalize_text).astype("string")
    combined["text_clean"] = combined["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

    for column in ["list_price", "sqft", "stories", "beds", "baths", "baths_full", "baths_full_calc", "garage", "year_built"]:
        if column in combined.columns:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")
        else:
            combined[column] = pd.NA

    combined.loc[combined["list_price"] <= 0, "list_price"] = pd.NA
    combined.loc[combined["sqft"] <= 0, "sqft"] = pd.NA
    for column in ["stories", "beds", "baths", "baths_full", "baths_full_calc", "garage"]:
        combined.loc[combined[column] < 0, column] = pd.NA
    combined.loc[(combined["year_built"] < 1700) | (combined["year_built"] > 2035), "year_built"] = pd.NA

    combined["is_placeholder_text"] = combined["text_clean"].isin(PLACEHOLDER_TEXTS)
    combined["is_blank_text"] = combined["text_clean"].eq("")
    combined["text_length"] = combined["text"].str.len().astype("Int64")
    combined["has_duplicate_text_within_state"] = combined.duplicated(subset=["state", "text_clean"], keep=False) & ~combined["is_blank_text"]

    dedupe_columns = [column for column in combined.columns if column != "record_id"]
    combined = combined.drop_duplicates(subset=dedupe_columns).reset_index(drop=True)
    combined["record_id"] = range(1, len(combined) + 1)

    for column in ["state", "source_dataset", "source_file"]:
        combined[column] = combined[column].astype("string")

    combined = combined[CORE_COLUMNS].sort_values(["state", "source_dataset", "record_id"]).reset_index(drop=True)

    clean_report = {
        "row_count_clean": int(len(combined)),
        "rows_removed_as_exact_duplicates": int(original_rows - len(combined)),
        "blank_text_rows_clean": int(combined["is_blank_text"].sum()),
        "placeholder_text_rows_clean": int(combined["is_placeholder_text"].sum()),
        "rows_with_duplicate_text_within_state": int(combined["has_duplicate_text_within_state"].sum()),
        "null_counts": {column: int(combined[column].isna().sum()) for column in combined.columns},
        "rows_by_state": {key: int(value) for key, value in combined["state"].value_counts().sort_index().items()},
    }
    return combined, clean_report


def save_outputs(cleaned: pd.DataFrame, report: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_dir / OUTPUT_CSV, index=False)
    cleaned.to_parquet(output_dir / OUTPUT_PARQUET, index=False)
    with (output_dir / REPORT_JSON).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and unify the cross-state real-estate dataset family.")
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing the downloaded raw dataset folders")
    parser.add_argument("--output-dir", default="data/processed", help="Directory where cleaned combined outputs should be written")
    args = parser.parse_args()

    frames = load_frames(Path(args.data_dir))
    if not frames:
        raise SystemExit(f"No CSV files found under {args.data_dir}")

    raw_report = build_raw_report(frames)
    cleaned, clean_report = clean_combined_dataset(frames)
    full_report = {
        "raw_summary": raw_report,
        "clean_summary": clean_report,
    }
    save_outputs(cleaned, full_report, Path(args.output_dir))

    print(json.dumps(full_report, indent=2))
    print(f"\nSaved cleaned outputs to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
