import argparse
from pathlib import Path

import pandas as pd

READABLE_EXTENSIONS = {".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl"}


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".txt":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect downloaded Kaggle real-estate datasets.")
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing dataset subfolders")
    parser.add_argument("--preview-rows", type=int, default=3, help="Rows to preview per readable table")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    dataset_files = sorted(path for path in data_dir.rglob("*") if path.is_file())
    if not dataset_files:
        raise SystemExit(f"No files found under: {data_dir}")

    print(f"Found {len(dataset_files)} file(s) in {data_dir.resolve()}\n")
    for path in dataset_files:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"- {path.relative_to(data_dir)} ({size_mb:.2f} MB)")

    print("\nTable previews\n")
    for path in dataset_files:
        if path.suffix.lower() not in READABLE_EXTENSIONS:
            continue
        print(f"=== {path.relative_to(data_dir)} ===")
        try:
            frame = load_table(path)
            print(f"shape: {frame.shape}")
            print("columns:", list(frame.columns))
            print(frame.head(args.preview_rows).to_string(index=False))
        except Exception as exc:
            print(f"Could not read file: {exc}")
        print()


if __name__ == "__main__":
    main()
