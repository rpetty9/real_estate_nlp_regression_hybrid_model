import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from build_hybrid_imputation_table import (
    TARGETS,
    combine_predictions,
    extract_with_regex,
    fit_target_bundle,
    predict_text_values,
    prepare_frame,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DEFAULT = "data/processed/real_estate_nlp_base.parquet"
REPORT_JSON = "hybrid_imputer_evaluation_report.json"
REPORT_CSV = "hybrid_imputer_evaluation_metrics.csv"

DEFAULT_SAMPLE_SIZE = 1200


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def compute_tolerance_accuracy(actual: pd.Series, predicted: pd.Series, target: str) -> float:
    diff = (actual - predicted).abs()
    if target in {"beds", "garage"}:
        tolerance = 1.0
    elif target in {"baths", "stories"}:
        tolerance = 0.5
    elif target == "sqft":
        tolerance = 250.0
    elif target == "year_built":
        tolerance = 5.0
    else:
        tolerance = 1.0
    return float((diff <= tolerance).mean())


def evaluate_target(df: pd.DataFrame, target: str, include_price: bool, sample_size: int) -> dict[str, float | int | str]:
    available = df[df[target].notna()].copy()
    if available.empty:
        return {"target": target, "mode": "analysis_safe" if include_price else "price_model_safe", "rows_evaluated": 0}

    mask_count = min(sample_size, len(available))
    masked_rows = available.sample(n=mask_count, random_state=42).index

    working = df.copy()
    actual = working.loc[masked_rows, target].astype(float)
    working.loc[masked_rows, target] = np.nan

    bundle = fit_target_bundle(working, target, include_price)
    structured_model = bundle["structured_model"]
    structured_features = bundle["structured_features"]
    text_model = bundle["text_model"]
    structured_weight = float(bundle["structured_weight"])

    regex_predictions = {}
    hybrid_predictions = {}
    regex_hits = 0

    remaining = []
    for idx in masked_rows:
        text_value = str(working.at[idx, "text_clean"]) if pd.notna(working.at[idx, "text_clean"]) else ""
        regex_value, _ = extract_with_regex(text_value, target)
        if regex_value is not None:
            regex_predictions[idx] = regex_value
            hybrid_predictions[idx] = regex_value
            regex_hits += 1
        else:
            remaining.append(idx)

    if remaining:
        X_struct = prepare_frame(working.loc[remaining], structured_features)
        struct_pred = pd.Series(structured_model.predict(X_struct), index=remaining)
        text_pred = predict_text_values(text_model, working.loc[remaining], target)

        for idx in remaining:
            structured_value = float(struct_pred.loc[idx])
            text_value = float(text_pred.loc[idx])
            hybrid_value = combine_predictions(structured_value, text_value, target, structured_weight)
            hybrid_predictions[idx] = hybrid_value

    predicted = pd.Series(hybrid_predictions).sort_index()
    actual = actual.loc[predicted.index]

    mae = mean_absolute_error(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)
    tolerance_acc = compute_tolerance_accuracy(actual, predicted, target)

    return {
        "target": target,
        "mode": "analysis_safe" if include_price else "price_model_safe",
        "rows_evaluated": int(len(actual)),
        "regex_coverage": round(regex_hits / len(masked_rows), 4),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "tolerance_accuracy": round(float(tolerance_acc), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the hybrid imputer by masking known values and predicting them back.")
    parser.add_argument("--input-path", default=INPUT_DEFAULT, help="Path to the NLP base table")
    parser.add_argument("--output-dir", default="data/processed", help="Directory where evaluation outputs should be written")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Masked holdout rows per target")
    args = parser.parse_args()

    input_path = resolve_project_path(args.input_path)
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    rows = []
    for include_price in [True, False]:
        for target in TARGETS:
            rows.append(evaluate_target(df, target, include_price=include_price, sample_size=args.sample_size))

    metrics_df = pd.DataFrame(rows)
    report = {
        "sample_size_per_target": args.sample_size,
        "results": metrics_df.to_dict(orient="records"),
    }

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / REPORT_CSV, index=False)
    with (output_dir / REPORT_JSON).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
