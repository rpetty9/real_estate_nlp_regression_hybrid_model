import argparse
import json
import runpy
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
DEFAULT_INPUTS = {
    "analysis_safe": "real_estate_hybrid_imputed_analysis.parquet",
    "price_model_safe": "real_estate_hybrid_imputed_price_safe.parquet",
}
TARGETS = ["year_built", "beds", "baths", "sqft"]


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_known_value_examples(df: pd.DataFrame, target: str, include_price: bool, helpers: dict[str, object]) -> pd.DataFrame:
    required = [target, "text", "text_clean", "type", "state", "record_id"]
    if any(column not in df.columns for column in required):
        return pd.DataFrame()

    fit_structured_predictor = helpers["fit_structured_predictor"]
    fit_text_predictor = helpers["fit_text_predictor"]
    predict_text_values = helpers["predict_text_values"]
    prepare_frame = helpers["prepare_frame"]
    clamp_prediction = helpers["clamp_prediction"]
    combine_predictions = helpers["combine_predictions"]
    extract_with_regex = helpers["extract_with_regex"]

    working = df[df[target].notna() & df["text"].notna() & df["text_clean"].notna()].copy()
    if target in {"year_built", "beds", "baths", "sqft"}:
        working = working[working["type"].fillna("").astype(str).str.lower() != "land"]
    if len(working) < 100:
        return pd.DataFrame()
    if len(working) > 4000:
        working = working.sample(n=4000, random_state=42)

    validation_rows = max(100, int(len(working) * 0.2))
    validation_index = working.sample(n=validation_rows, random_state=42).index
    train_df = working.loc[working.index.difference(validation_index)].copy()
    validation_df = working.loc[validation_index].copy()

    structured_model, structured_features = fit_structured_predictor(train_df, target, include_price)
    text_model = fit_text_predictor(train_df, target)
    structured_weight = 0.5

    regex_values = []
    regex_confidences = []
    keep_rows = []
    for idx in validation_df.index:
        regex_value, regex_confidence = extract_with_regex(str(validation_df.at[idx, "text_clean"]), target)
        regex_values.append(regex_value)
        regex_confidences.append(regex_confidence)
        keep_rows.append(regex_value is None)

    eval_df = validation_df.loc[pd.Series(keep_rows, index=validation_df.index)].copy()
    if eval_df.empty:
        return pd.DataFrame()

    X_struct = prepare_frame(eval_df, structured_features)
    struct_pred = structured_model.predict(X_struct)
    text_pred = predict_text_values(text_model, eval_df, target)

    eval_df[f"{target}_structured_prediction"] = [clamp_prediction(float(value), target) for value in struct_pred]
    eval_df[f"{target}_text_prediction"] = [clamp_prediction(float(value), target) for value in text_pred]
    eval_df[f"{target}_hybrid_prediction"] = [
        combine_predictions(structured_value, text_value, target, structured_weight)
        for structured_value, text_value in zip(eval_df[f"{target}_structured_prediction"], eval_df[f"{target}_text_prediction"])
    ]
    eval_df[f"{target}_regex_candidate"] = pd.NA
    eval_df["actual_value"] = eval_df[target].astype(float)
    eval_df["prediction_error"] = (eval_df[f"{target}_hybrid_prediction"] - eval_df["actual_value"]).abs()
    eval_df["structured_error"] = (eval_df[f"{target}_structured_prediction"] - eval_df["actual_value"]).abs()
    eval_df["text_error"] = (eval_df[f"{target}_text_prediction"] - eval_df["actual_value"]).abs()
    eval_df["model_gap"] = (eval_df[f"{target}_structured_prediction"] - eval_df[f"{target}_text_prediction"]).abs()
    eval_df["structured_weight"] = structured_weight
    gap_scale = max(float(eval_df["model_gap"].median()), 1.0)
    eval_df["confidence"] = (1.0 / (1.0 + (eval_df["model_gap"] / gap_scale))).clip(lower=0.12, upper=0.90).round(3)
    eval_df["mode"] = "analysis_safe" if include_price else "price_model_safe"
    eval_df["target"] = target

    if target == "year_built":
        eval_df = eval_df[
            eval_df["actual_value"].between(1850, 2025)
            & eval_df[f"{target}_structured_prediction"].between(1850, 2025)
            & eval_df[f"{target}_text_prediction"].between(1850, 2025)
            & (eval_df["prediction_error"] <= 20)
            & (eval_df["model_gap"] <= 80)
        ]
    elif target == "beds":
        eval_df = eval_df[
            eval_df["actual_value"].between(1, 8)
            & eval_df[f"{target}_structured_prediction"].between(1, 8)
            & eval_df[f"{target}_text_prediction"].between(1, 8)
            & (eval_df["prediction_error"] <= 1)
            & (eval_df["model_gap"] <= 3)
        ]
    elif target == "baths":
        eval_df = eval_df[
            eval_df["actual_value"].between(0.5, 6)
            & eval_df[f"{target}_structured_prediction"].between(0.5, 6)
            & eval_df[f"{target}_text_prediction"].between(0.5, 6)
            & (eval_df["prediction_error"] <= 1)
            & (eval_df["model_gap"] <= 2)
        ]
    elif target == "sqft":
        eval_df = eval_df[
            eval_df["actual_value"].between(400, 8000)
            & eval_df[f"{target}_structured_prediction"].between(400, 8000)
            & eval_df[f"{target}_text_prediction"].between(400, 8000)
            & (eval_df["prediction_error"] <= 500)
            & (eval_df["model_gap"] <= 2000)
        ]

    if eval_df.empty:
        return pd.DataFrame()

    eval_df = eval_df[eval_df["text"].astype(str).str.len() >= 120]
    return eval_df.sort_values(["prediction_error", "model_gap"]).head(50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build row-level known-value evaluation artifacts for the hybrid imputer.")
    parser.add_argument("--output-dir", default=str(PROCESSED), help="Directory where outputs should be written")
    args = parser.parse_args()

    helpers = runpy.run_path(str(PROJECT_ROOT / "scripts" / "build_hybrid_imputation_table.py"))
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    summary: dict[str, dict[str, int]] = {}
    for mode, filename in DEFAULT_INPUTS.items():
        df = pd.read_parquet(PROCESSED / filename)
        include_price = mode == "analysis_safe"
        summary[mode] = {}
        for target in TARGETS:
            examples = build_known_value_examples(df, target, include_price, helpers)
            summary[mode][target] = int(len(examples))
            if not examples.empty:
                frames.append(examples)

    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output.to_parquet(output_dir / "hybrid_imputer_known_value_examples.parquet", index=False)
    output.to_csv(output_dir / "hybrid_imputer_known_value_examples.csv", index=False)
    with (output_dir / "hybrid_imputer_known_value_examples_report.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
