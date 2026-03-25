import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DEFAULT = "data/processed/real_estate_nlp_base.parquet"
OUTPUT_CSV = "real_estate_price_modeling_table.csv"
OUTPUT_PARQUET = "real_estate_price_modeling_table.parquet"
REPORT_JSON = "real_estate_price_model_report.json"
IMPORTANCE_CSV = "price_feature_importance.csv"
MAX_IMPORTANCE_SAMPLE = 5000
MAX_IMPUTATION_TRAIN_ROWS = 40000
MAX_IMPUTE_FEATURES = 5


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path

IMPUTE_TARGETS = [
    "sqft",
    "beds",
    "baths",
    "baths_full",
    "baths_full_calc",
    "garage",
    "stories",
    "year_built",
]

NUMERIC_FEATURES = [
    "sqft",
    "beds",
    "baths",
    "baths_full",
    "baths_full_calc",
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

CATEGORICAL_FEATURES = [
    "state",
    "type",
    "sub_type",
    "quality_label",
]

BOOLEAN_FEATURES = [
    "is_placeholder_text",
    "is_blank_text",
    "has_duplicate_text_within_state",
    "is_quality_filtered_text",
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


def prepare_model_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    prepared = df[feature_columns].copy()
    for column in feature_columns:
        if column in CATEGORICAL_FEATURES:
            prepared[column] = prepared[column].astype("object")
            prepared[column] = prepared[column].where(pd.notna(prepared[column]), np.nan)
        elif column in BOOLEAN_FEATURES:
            prepared[column] = prepared[column].astype("float")
        else:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
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
    )


def build_price_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
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


def round_imputed_values(series: pd.Series, column: str) -> pd.Series:
    if column in {"beds", "garage", "year_built"}:
        return series.round().clip(lower=0)
    if column in {"sqft"}:
        return series.round().clip(lower=1)
    if column in {"stories"}:
        return series.round(1).clip(lower=0)
    if column in {"baths", "baths_full", "baths_full_calc"}:
        return series.round(1).clip(lower=0)
    return series


def evaluate_price_model(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], Pipeline]:
    model_df = df[df["list_price"].notna()].copy()
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
    X = prepare_model_frame(model_df, feature_columns)
    y = np.log1p(model_df["list_price"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline = build_price_model()
    pipeline.fit(X_train, y_train)

    pred_log = pipeline.predict(X_test)
    pred_price = np.expm1(pred_log)
    true_price = np.expm1(y_test)

    metrics = {
        "row_count_used_for_price_model": int(len(model_df)),
        "test_mae": round(float(mean_absolute_error(true_price, pred_price)), 2),
        "test_rmse": round(float(root_mean_squared_error(true_price, pred_price)), 2),
        "test_r2": round(float(r2_score(true_price, pred_price)), 4),
    }

    importance_X = X_test
    importance_y = y_test
    if len(X_test) > MAX_IMPORTANCE_SAMPLE:
        sampled_index = X_test.sample(n=MAX_IMPORTANCE_SAMPLE, random_state=42).index
        importance_X = X_test.loc[sampled_index]
        importance_y = y_test.loc[sampled_index]

    result = permutation_importance(
        pipeline,
        importance_X,
        importance_y,
        n_repeats=3,
        random_state=42,
        n_jobs=1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    return importance_df, metrics, pipeline


def impute_feature(df: pd.DataFrame, target_column: str) -> tuple[pd.Series, pd.Series, dict[str, float | int]]:
    feature_columns = (
        ["list_price"]
        + [column for column in NUMERIC_FEATURES if column != target_column]
        + CATEGORICAL_FEATURES
        + BOOLEAN_FEATURES
    )

    train_df = df[df[target_column].notna()].copy()
    if len(train_df) > MAX_IMPUTATION_TRAIN_ROWS:
        train_df = train_df.sample(n=MAX_IMPUTATION_TRAIN_ROWS, random_state=42)
    missing_mask = df[target_column].isna()
    predict_df = prepare_model_frame(df.loc[missing_mask], feature_columns)

    if predict_df.empty:
        return df[target_column], pd.Series(False, index=df.index), {"rows_imputed": 0}

    X_train = prepare_model_frame(train_df, feature_columns)
    y_train = train_df[target_column]

    numeric_features = ["list_price"] + [column for column in NUMERIC_FEATURES if column != target_column]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
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
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=80,
                    random_state=42,
                    n_jobs=1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    predictions = pd.Series(model.predict(predict_df), index=predict_df.index)
    predictions = round_imputed_values(predictions, target_column)

    output = df[target_column].copy()
    output.loc[predictions.index] = predictions

    imputed_flag = pd.Series(False, index=df.index)
    imputed_flag.loc[predictions.index] = True

    return output, imputed_flag, {"rows_imputed": int(len(predictions))}


def build_price_modeling_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    working = df.copy()

    importance_df, metrics, _ = evaluate_price_model(working)

    importance_targets = (
        importance_df[importance_df["feature"].isin(IMPUTE_TARGETS)]
        .sort_values("importance_mean", ascending=False)
        .head(MAX_IMPUTE_FEATURES)
        .reset_index(drop=True)
    )

    imputation_summary: dict[str, dict[str, float | int]] = {}
    original_missing_counts = {column: int(working[column].isna().sum()) for column in IMPUTE_TARGETS}

    for column in importance_targets["feature"].tolist():
        imputed_values, imputed_flag, summary = impute_feature(working, column)
        working[column] = imputed_values
        working[f"{column}_was_imputed"] = imputed_flag
        imputation_summary[column] = summary

    final_missing_counts = {column: int(working[column].isna().sum()) for column in IMPUTE_TARGETS}

    report = {
        "price_model_metrics": metrics,
        "important_price_features": importance_df.head(20).round(6).to_dict(orient="records"),
        "important_structured_features_for_imputation": importance_targets.round(6).to_dict(orient="records"),
        "original_missing_counts": original_missing_counts,
        "final_missing_counts": final_missing_counts,
        "imputation_summary": imputation_summary,
    }
    return working, report, importance_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a price-modeling table with feature importance analysis and targeted imputation."
    )
    parser.add_argument("--input-path", default=INPUT_DEFAULT, help="Path to the NLP base table")
    parser.add_argument("--output-dir", default="data/processed", help="Directory where outputs should be written")
    args = parser.parse_args()

    input_path = resolve_project_path(args.input_path)
    if not input_path.exists():
        raise SystemExit(f"Input dataset not found: {input_path}")

    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    modeling_df, report, importance_df = build_price_modeling_table(df)

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modeling_df.to_csv(output_dir / OUTPUT_CSV, index=False)
    modeling_df.to_parquet(output_dir / OUTPUT_PARQUET, index=False)
    importance_df.to_csv(output_dir / IMPORTANCE_CSV, index=False)
    with (output_dir / REPORT_JSON).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved price modeling outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
