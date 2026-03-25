import argparse
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_DATASETS = [
    "kanchana1990/wisconsin-real-estate-data-2026",
    "kanchana1990/connecticut-real-estate-data-2026",
    "kanchana1990/real-estate-data-oregon-2026",
    "kanchana1990/new-york-real-estate-data-2026",
    "kanchana1990/new-hampshire-real-estate-data-2026",
    "kanchana1990/indiana-real-estate-data-2026",
    "kanchana1990/kentucky-real-estate-data-2026",
    "kanchana1990/colorado-real-estate-market-2026",
]


def parse_dataset_list() -> list[str]:
    load_dotenv()
    raw_value = os.getenv("KAGGLE_REAL_ESTATE_DATASETS", "")
    if not raw_value.strip():
        return DEFAULT_DATASETS
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def dataset_folder_name(dataset_slug: str) -> str:
    return dataset_slug.split("/", 1)[1]


def download_dataset(dataset_slug: str, base_output_dir: Path) -> None:
    output_dir = base_output_dir / dataset_folder_name(dataset_slug)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "python",
        "-m",
        "kaggle.cli",
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "-p",
        str(output_dir),
        "--unzip",
    ]

    print(f"Downloading {dataset_slug} into {output_dir.resolve()}...")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one or more Kaggle real-estate datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset slug in the form owner/dataset-slug. Repeat the flag to download multiple datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Base directory where dataset subfolders should be created",
    )
    args = parser.parse_args()

    kaggle_token = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_token.exists():
        raise SystemExit(
            "Kaggle API token not found. Place kaggle.json at "
            f"{kaggle_token} before downloading."
        )

    datasets = args.dataset if args.dataset else parse_dataset_list()
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_slug in datasets:
        download_dataset(dataset_slug, base_output_dir)

    print(f"\nDownloaded {len(datasets)} dataset(s) into {base_output_dir.resolve()}")


if __name__ == "__main__":
    main()
