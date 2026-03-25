import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DEFAULT = "data/processed/real_estate_hybrid_imputed_price_safe.parquet"
REPORT_JSON = "price_model_comparison_report.json"
PREDICTIONS_CSV = "price_model_comparison_predictions.csv"


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path

NUMERIC_FEATURES = [
    "beds",
    "baths",
    "sqft",
    "garage",
    "stories",
    "year_built",
    "quality_score",
    "word_count",
    "sentence_count",
    "digit_count",
    "amenity_hit_count",
    "unique_word_count",
    "text_length",
]

CATEGORICAL_FEATURES = ["state", "type", "sub_type", "quality_label"]

BOOLEAN_FEATURES = [
    "has_fireplace",
    "has_granite",
    "has_stainless_steel",
    "has_hardwood",
    "has_quartz",
    "has_walk_in_closet",
    "has_finished_basement",
    "has_lake_access",
    "has_pool",
    "has_garage",
    "has_deck",
    "has_patio",
    "has_fenced_yard",
    "has_acreage",
    "has_new_construction",
    "has_investor_angle",
    "has_fixer_upper_angle",
    "has_hoa",
    "has_views",
    "has_updated_kitchen",
    "has_open_floor_plan",
    "has_primary_suite",
]


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in NUMERIC_FEATURES:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        prepared[column] = prepared[column].astype("object")
        prepared[column] = prepared[column].where(pd.notna(prepared[column]), np.nan)
    for column in BOOLEAN_FEATURES:
        prepared[column] = prepared[column].astype(float)
    if "text_clean" in prepared.columns:
        prepared["text_clean"] = prepared["text_clean"].fillna("").astype(str)
    return prepared


def score_predictions(name: str, pred_log, y_test) -> dict[str, float | str]:
    pred_price = np.expm1(pred_log)
    true_price = np.expm1(y_test)
    return {
        "model": name,
        "mae": round(float(mean_absolute_error(true_price, pred_price)), 2),
        "rmse": round(float(root_mean_squared_error(true_price, pred_price)), 2),
        "r2": round(float(r2_score(true_price, pred_price)), 4),
    }


def evaluate_model(name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test) -> dict[str, float | str]:
    pipeline.fit(X_train, y_train)
    pred_log = pipeline.predict(X_test)
    return score_predictions(name, pred_log, y_test)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare text-only, structured-only, and combined price prediction models.")
    parser.add_argument("--input-path", default=INPUT_DEFAULT, help="Path to the price-safe hybrid-imputed table")
    parser.add_argument("--output-dir", default="data/processed", help="Directory where outputs should be written")
    args = parser.parse_args()

    input_path = resolve_project_path(args.input_path)
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    model_df = df[df["list_price"].notna()].copy()
    y = np.log1p(model_df["list_price"])

    text_X = model_df["text_clean"].fillna("").astype(str)
    structured_X = prepare_model_frame(model_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES].copy())
    combined_X = prepare_model_frame(model_df[["text_clean"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES].copy())

    train_idx, test_idx = train_test_split(model_df.index, test_size=0.2, random_state=42)
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    text_model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=3)),
            ("model", Ridge(alpha=4.0)),
        ]
    )

    structured_model = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            CATEGORICAL_FEATURES,
                        ),
                        ("bool", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), BOOLEAN_FEATURES),
                    ]
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=120,
                    random_state=42,
                    n_jobs=1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )

    text_result = evaluate_model("text_only", text_model, text_X.loc[train_idx], text_X.loc[test_idx], y_train, y_test)
    structured_result = evaluate_model(
        "structured_only",
        structured_model,
        structured_X.loc[train_idx],
        structured_X.loc[test_idx],
        y_train,
        y_test,
    )

    blend_train_idx, blend_val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

    text_model.fit(text_X.loc[blend_train_idx], y.loc[blend_train_idx])
    structured_model.fit(structured_X.loc[blend_train_idx], y.loc[blend_train_idx])
    val_text_pred = text_model.predict(text_X.loc[blend_val_idx])
    val_struct_pred = structured_model.predict(structured_X.loc[blend_val_idx])

    best_weight = 0.5
    best_rmse = float("inf")
    for structured_weight in np.linspace(0.0, 1.0, 11):
        blended = ((1.0 - structured_weight) * val_text_pred) + (structured_weight * val_struct_pred)
        rmse = root_mean_squared_error(np.expm1(y.loc[blend_val_idx]), np.expm1(blended))
        if rmse < best_rmse:
            best_rmse = float(rmse)
            best_weight = float(structured_weight)

    text_model.fit(text_X.loc[train_idx], y_train)
    structured_model.fit(structured_X.loc[train_idx], y_train)
    text_pred_log = text_model.predict(text_X.loc[test_idx])
    structured_pred_log = structured_model.predict(structured_X.loc[test_idx])
    combined_pred_log = ((1.0 - best_weight) * text_pred_log) + (best_weight * structured_pred_log)

    results = [
        text_result,
        structured_result,
        score_predictions("text_plus_structured", combined_pred_log, y_test),
    ]

    predictions = pd.DataFrame(
        {
            "actual_price": np.expm1(y_test),
            "predicted_price": np.expm1(combined_pred_log),
            "text_only_predicted_price": np.expm1(text_pred_log),
            "structured_only_predicted_price": np.expm1(structured_pred_log),
        },
        index=test_idx,
    ).reset_index(names="row_index")

    report = {
        "row_count_used": int(len(model_df)),
        "best_structured_weight_for_blend": round(best_weight, 2),
        "results": results,
    }

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / PREDICTIONS_CSV, index=False)
    with (output_dir / REPORT_JSON).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
