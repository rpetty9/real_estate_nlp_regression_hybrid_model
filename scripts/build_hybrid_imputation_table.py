import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DEFAULT = "data/processed/real_estate_nlp_base.parquet"
ANALYSIS_OUTPUT_PARQUET = "real_estate_hybrid_imputed_analysis.parquet"
PRICE_SAFE_OUTPUT_PARQUET = "real_estate_hybrid_imputed_price_safe.parquet"
ANALYSIS_OUTPUT_CSV = "real_estate_hybrid_imputed_analysis.csv"
PRICE_SAFE_OUTPUT_CSV = "real_estate_hybrid_imputed_price_safe.csv"
REPORT_JSON = "real_estate_hybrid_imputation_report.json"

MAX_TRAIN_ROWS = 22000
MAX_CALIBRATION_ROWS = 3500
TARGETS = ["beds", "baths", "sqft", "garage", "stories", "year_built"]
TEXT_ONLY_TARGETS = {"beds", "baths", "sqft", "garage", "stories", "year_built"}

ANALYSIS_NUMERIC_FEATURES = [
    "list_price",
    "quality_score",
    "word_count",
    "sentence_count",
    "digit_count",
    "amenity_hit_count",
    "unique_word_count",
    "text_length",
    "beds",
    "baths",
    "baths_full",
    "baths_full_calc",
    "sqft",
    "garage",
    "stories",
    "year_built",
]

PRICE_SAFE_NUMERIC_FEATURES = [
    feature for feature in ANALYSIS_NUMERIC_FEATURES if feature != "list_price"
]

CATEGORICAL_FEATURES = ["state", "type", "sub_type", "quality_label"]

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

TARGET_ERROR_SCALES = {
    "beds": 1.0,
    "baths": 1.0,
    "sqft": 1200.0,
    "garage": 2.0,
    "stories": 1.0,
    "year_built": 25.0,
}

NUMBER_WORDS = {
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
}

NEGATIVE_YEAR_CONTEXT = [
    "roof",
    "hvac",
    "water heater",
    "furnace",
    "a/c",
    "ac ",
    "installed",
    "updated",
    "remodeled",
    "renovated",
    "new carpet",
    "new roof",
    "new furnace",
    "new ac",
]

POSITIVE_YEAR_CONTEXT = [
    "built",
    "constructed",
    "year built",
    "craftsman-style home",
    "new construction",
    "brand new",
    "to be built",
    "house is framed",
    "finish",
    "completion",
]

TEXT_ALPHA_BY_TARGET = {
    "beds": 2.0,
    "baths": 2.5,
    "sqft": 4.0,
    "garage": 2.0,
    "stories": 2.5,
    "year_built": 6.0,
}

EXCLUDED_TEXT_TYPES_BY_TARGET = {
    "beds": {"land", "missing"},
    "baths": {"land", "missing"},
    "sqft": {"land", "missing"},
    "garage": {"land", "missing"},
    "stories": {"land", "missing"},
    "year_built": {"land", "missing"},
}

TEXT_HINTS_BY_TARGET = {
    "beds": [
        ("primary suite", "hint_primary_suite"),
        ("guest room", "hint_guest_room"),
        ("office", "hint_office"),
        ("bonus room", "hint_bonus_room"),
        ("bunk room", "hint_bunk_room"),
    ],
    "baths": [
        ("half bath", "hint_half_bath"),
        ("powder room", "hint_powder_room"),
        ("ensuite", "hint_ensuite"),
        ("en suite", "hint_ensuite"),
        ("jack and jill", "hint_jack_jill"),
        ("full bath", "hint_full_bath"),
    ],
    "sqft": [
        ("open floor plan", "hint_open_floor_plan"),
        ("spacious", "hint_spacious"),
        ("large", "hint_large"),
        ("expansive", "hint_expansive"),
        ("cozy", "hint_cozy"),
        ("acre", "hint_acreage"),
    ],
    "garage": [
        ("attached garage", "hint_attached_garage"),
        ("detached garage", "hint_detached_garage"),
        ("workshop", "hint_workshop"),
        ("barn", "hint_barn"),
        ("carport", "hint_carport"),
    ],
    "stories": [
        ("main level", "hint_main_level"),
        ("upper level", "hint_upper_level"),
        ("walkout basement", "hint_walkout"),
        ("split level", "hint_split_level"),
        ("loft", "hint_loft"),
    ],
    "year_built": [
        ("antique", "hint_antique"),
        ("historic", "hint_historic"),
        ("colonial", "hint_colonial"),
        ("mid-century", "hint_midcentury"),
        ("mid century", "hint_midcentury"),
        ("craftsman", "hint_craftsman"),
        ("new construction", "hint_new_construction"),
        ("brand new", "hint_brand_new"),
        ("to be built", "hint_to_be_built"),
        ("fully renovated", "hint_fully_renovated"),
    ],
}

PROPERTY_FAMILY_MAP = {
    "single_family": "detached",
    "mobile": "detached",
    "farm": "detached",
    "townhomes": "attached",
    "condos": "attached",
    "coop": "attached",
    "condop": "attached",
    "multi_family": "multi_unit",
    "duplex_triplex": "multi_unit",
    "apartment": "multi_unit",
    "land": "land",
    "missing": "unknown",
}


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    prepared = df[feature_columns].copy()
    for column in feature_columns:
        if column in CATEGORICAL_FEATURES:
            prepared[column] = prepared[column].astype("object")
            prepared[column] = prepared[column].where(pd.notna(prepared[column]), np.nan)
        elif column in BOOLEAN_FEATURES:
            prepared[column] = prepared[column].astype(float)
        elif column == "text_clean":
            prepared[column] = prepared[column].fillna("").astype(str)
        else:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared


def build_structured_model(numeric_features: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
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
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=60,
                    random_state=42,
                    n_jobs=1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def build_text_model(target: str) -> Pipeline:
    alpha = TEXT_ALPHA_BY_TARGET.get(target, 3.0)
    return Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    transformer_list=[
                        (
                            "word",
                            TfidfVectorizer(
                                max_features=8000,
                                ngram_range=(1, 2),
                                min_df=3,
                                max_df=0.95,
                                strip_accents="unicode",
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=4,
                                max_features=2500,
                                strip_accents="unicode",
                            ),
                        ),
                    ]
                ),
            ),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def normalize_type(value: object) -> str:
    if pd.isna(value):
        return "missing"
    return str(value).strip().lower() or "missing"


def infer_property_family(value: object) -> str:
    return PROPERTY_FAMILY_MAP.get(normalize_type(value), "other")


def sanitize_text_for_model(text: object) -> str:
    if pd.isna(text):
        return ""
    cleaned = str(text).lower()
    cleaned = cleaned.replace("[redacted name]", " ")
    boilerplate_patterns = [
        r"all prices, promotions and specifications are subject to change[^.]*\.?",
        r"information deemed reliable but not guaranteed[^.]*\.?",
        r"listing information is deemed reliable[^.]*\.?",
        r"buyer to verify[^.]*\.?",
        r"equal housing opportunity[^.]*\.?",
        r"copyright \d{4}[^.]*\.?",
    ]
    for pattern in boilerplate_patterns:
        cleaned = re.sub(pattern, " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_text_signal_tokens(text: str, target: str) -> list[str]:
    tokens: list[str] = []
    lower = text.lower()

    for phrase, token in TEXT_HINTS_BY_TARGET.get(target, []):
        if phrase in lower:
            tokens.append(token)

    unit_match = re.search(r"\b(\d+)\s*unit\b", lower)
    if unit_match:
        unit_count = int(unit_match.group(1))
        capped = min(unit_count, 10)
        tokens.extend([f"unit_count_{capped}", "hint_multi_unit"])

    if any(word in lower for word in ["duplex", "triplex", "fourplex", "quadplex"]):
        tokens.append("hint_multi_unit_named")

    if target == "year_built":
        for year_match in re.finditer(r"\b(18\d{2}|19\d{2}|20[0-3]\d)\b", lower):
            year = int(year_match.group(1))
            decade = (year // 10) * 10
            tokens.append(f"year_mention_{decade}s")
        if "century" in lower:
            tokens.append("hint_century_language")
    elif target == "sqft":
        if re.search(r"\b\d+(\.\d+)?\s*acre", lower):
            tokens.append("hint_acre_count")
        if "open concept" in lower or "open-concept" in lower:
            tokens.append("hint_open_concept")
    elif target == "baths":
        if "washer/dryer hookup" in lower or "washer dryer hookup" in lower:
            tokens.append("hint_washer_dryer")
        if "primary bath" in lower:
            tokens.append("hint_primary_bath")

    return tokens


def compose_text_corpus(df: pd.DataFrame, target: str) -> pd.Series:
    rows = []
    for _, row in df.iterrows():
        cleaned = sanitize_text_for_model(row.get("text_clean", row.get("text", "")))
        signal_tokens = build_text_signal_tokens(cleaned, target)
        rows.append(" ".join(part for part in [cleaned, " ".join(signal_tokens)] if part).strip())
    return pd.Series(rows, index=df.index, dtype="string")


def filter_text_training_rows(df: pd.DataFrame, target: str) -> pd.DataFrame:
    working = df.copy()
    if "text_clean" in working.columns:
        working = working[working["text_clean"].fillna("").astype(str).str.len() >= 30]
    if "is_placeholder_text" in working.columns:
        working = working[~working["is_placeholder_text"].fillna(False)]
    if "is_blank_text" in working.columns:
        working = working[~working["is_blank_text"].fillna(False)]
    if "type" in working.columns:
        excluded = EXCLUDED_TEXT_TYPES_BY_TARGET.get(target, set())
        working = working[~working["type"].map(normalize_type).isin(excluded)]
    return working


def predict_text_values(model_bundle: dict[str, object], df: pd.DataFrame, target: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)

    text_input = compose_text_corpus(df, target)
    global_model = model_bundle["global_model"]
    predictions = pd.Series(global_model.predict(text_input), index=df.index, dtype=float)

    family_models = model_bundle.get("family_models", {})
    if family_models and "type" in df.columns:
        families = df["type"].map(infer_property_family)
        for family, family_model in family_models.items():
            family_index = families[families == family].index
            if len(family_index) == 0:
                continue
            family_input = text_input.loc[family_index]
            family_predictions = pd.Series(family_model.predict(family_input), index=family_index, dtype=float)
            predictions.loc[family_index] = (0.65 * family_predictions) + (0.35 * predictions.loc[family_index])

    return predictions


def get_numeric_features(include_price: bool, target: str) -> list[str]:
    numeric_features = ANALYSIS_NUMERIC_FEATURES if include_price else PRICE_SAFE_NUMERIC_FEATURES
    return [column for column in numeric_features if column != target]


def get_structured_feature_columns(include_price: bool, target: str) -> list[str]:
    return get_numeric_features(include_price, target) + CATEGORICAL_FEATURES + BOOLEAN_FEATURES


def clamp_prediction(value: float, target: str) -> float:
    if pd.isna(value):
        return np.nan
    if target == "beds":
        return float(np.clip(round(value), 0, 20))
    if target == "garage":
        return float(np.clip(round(value), 0, 10))
    if target == "sqft":
        return float(np.clip(round(value), 150, 25000))
    if target == "stories":
        return float(np.clip(round(value * 2) / 2, 0, 6))
    if target == "baths":
        return float(np.clip(round(value * 2) / 2, 0, 20))
    if target == "year_built":
        return float(np.clip(round(value), 1700, 2035))
    return float(value)


def combine_predictions(structured_value: float, text_value: float, target: str, structured_weight: float) -> float:
    return clamp_prediction((structured_weight * structured_value) + ((1.0 - structured_weight) * text_value), target)


def extract_with_regex(text: str, target: str) -> tuple[float | None, float]:
    if not text:
        return None, 0.0
    lower = text.lower()

    if target == "beds":
        patterns = [
            r"(\d+(?:\.\d+)?)\s*(?:bed|beds|bedroom|bedrooms)\b",
            r"\b(\d+(?:\.\d+)?)\s*br\b",
            r"\b(\d+(?:\.\d+)?)\s*bd\b",
            r"\b(\d+(?:\.\d+)?)\s*bdrm\b",
        ]
    elif target == "baths":
        patterns = [
            r"(\d+(?:\.\d+)?)\s*(?:bath|baths|bathroom|bathrooms)\b",
            r"\b(\d+(?:\.\d+)?)\s*ba\b",
            r"(\d+(?:\.\d+)?)\s*(?:full\s*)?bath",
        ]
    elif target == "sqft":
        patterns = [
            r"(\d{3,5}(?:,\d{3})?(?:\.\d+)?)\s*(?:sq\.?\s*ft\.?|square feet|sqft)\b",
            r"(\d{3,5}(?:,\d{3})?(?:\.\d+)?)\s*sf\b",
            r"(\d{3,5}(?:,\d{3})?(?:\.\d+)?)\s*(?:square foot|square-foot)\b",
        ]
    elif target == "garage":
        patterns = [
            r"(\d+(?:\.\d+)?)\s*-\s*car garage\b",
            r"(\d+(?:\.\d+)?)\s*car garage\b",
            r"garage for (\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*bay garage\b",
        ]
        for word, value in NUMBER_WORDS.items():
            if f"{word}-car garage" in lower or f"{word} car garage" in lower or f"{word}-bay garage" in lower:
                return value, 0.94
    elif target == "stories":
        text_map = {
            "single-story": 1.0,
            "single story": 1.0,
            "one-story": 1.0,
            "one story": 1.0,
            "single-level": 1.0,
            "two-story": 2.0,
            "two story": 2.0,
            "three-story": 3.0,
            "three story": 3.0,
            "split-level": 2.0,
        }
        for key, value in text_map.items():
            if key in lower:
                return value, 0.95
        patterns = [r"(\d+(?:\.\d+)?)\s*story\b", r"(\d+(?:\.\d+)?)\s*stories\b"]
    elif target == "year_built":
        direct_patterns = [
            r"built in (\d{4})\b",
            r"built (\d{4})\b",
            r"constructed in (\d{4})\b",
            r"year built[: ]+(\d{4})\b",
            r"(\d{4})\s+craftsman-style home",
            r"(\d{4})\s+craftsman home",
        ]
        for pattern in direct_patterns:
            match = re.search(pattern, lower)
            if match:
                return clamp_prediction(float(match.group(1)), target), 0.96

        year_candidates: list[tuple[float, float]] = []
        for match in re.finditer(r"\b(18\d{2}|19\d{2}|20[0-3]\d)\b", lower):
            year = float(match.group(1))
            start = max(0, match.start() - 45)
            end = min(len(lower), match.end() + 45)
            context = lower[start:end]
            score = 0.5
            if any(keyword in context for keyword in POSITIVE_YEAR_CONTEXT):
                score += 0.3
            if any(keyword in context for keyword in NEGATIVE_YEAR_CONTEXT):
                score -= 0.35
            if "new construction" in lower and year >= 2020:
                score += 0.15
            year_candidates.append((year, score))

        if year_candidates:
            year_candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
            best_year, best_score = year_candidates[0]
            if best_score >= 0.55:
                confidence = 0.82 if best_score < 0.75 else 0.93
                return clamp_prediction(best_year, target), confidence
        return None, 0.0
    else:
        return None, 0.0

    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            raw = match.group(1).replace(",", "")
            value = clamp_prediction(float(raw), target)
            return value, 0.95
    return None, 0.0


def fit_structured_predictor(df: pd.DataFrame, target: str, include_price: bool) -> tuple[Pipeline, list[str]]:
    feature_columns = get_structured_feature_columns(include_price, target)
    train_df = df[df[target].notna()].copy()
    if len(train_df) > MAX_TRAIN_ROWS:
        train_df = train_df.sample(n=MAX_TRAIN_ROWS, random_state=42)
    X_train = prepare_frame(train_df, feature_columns)
    y_train = train_df[target]
    model = build_structured_model(get_numeric_features(include_price, target))
    model.fit(X_train, y_train)
    return model, feature_columns


def fit_text_predictor(df: pd.DataFrame, target: str) -> Pipeline:
    train_df = df[df[target].notna()].copy()
    train_df = filter_text_training_rows(train_df, target)
    if train_df.empty:
        train_df = df[df[target].notna()].copy()
    if len(train_df) > MAX_TRAIN_ROWS:
        train_df = train_df.sample(n=MAX_TRAIN_ROWS, random_state=42)
    text_input = compose_text_corpus(train_df, target)

    global_model = build_text_model(target)
    global_model.fit(text_input, train_df[target])

    family_models: dict[str, Pipeline] = {}
    if "type" in train_df.columns:
        families = train_df["type"].map(infer_property_family)
        train_df = train_df.assign(property_family=families.values)
        for family, family_df in train_df.groupby("property_family"):
            if target not in {"baths", "sqft", "year_built"}:
                continue
            if family in {"unknown", "other", "land"} or len(family_df) < 900:
                continue
            family_input = compose_text_corpus(family_df, target)
            family_model = build_text_model(target)
            family_model.fit(family_input, family_df[target])
            family_models[family] = family_model

    return {
        "global_model": global_model,
        "family_models": family_models,
        "target": target,
    }


def fit_target_bundle(df: pd.DataFrame, target: str, include_price: bool) -> dict[str, object]:
    available = df[df[target].notna()].copy()
    if len(available) > MAX_CALIBRATION_ROWS:
        available = available.sample(n=MAX_CALIBRATION_ROWS, random_state=42)

    validation_mae = TARGET_ERROR_SCALES[target]
    best_weight = 0.5
    feature_columns = get_structured_feature_columns(include_price, target)

    if len(available) >= 200:
        validation_rows = max(100, int(len(available) * 0.2))
        validation_index = available.sample(n=validation_rows, random_state=42).index
        train_df = df.loc[df.index.difference(validation_index)]
        validation_df = df.loc[validation_index].copy()

        structured_model, _ = fit_structured_predictor(train_df, target, include_price)
        text_model = fit_text_predictor(train_df, target)

        val_actual = validation_df[target].astype(float)
        regex_predictions = {}
        remaining = []
        for idx in validation_df.index:
            regex_value, _ = extract_with_regex(str(validation_df.at[idx, "text_clean"]), target)
            if regex_value is not None:
                regex_predictions[idx] = regex_value
            else:
                remaining.append(idx)

        structured_preds = pd.Series(dtype=float)
        text_preds = pd.Series(dtype=float)
        if remaining:
            X_val_struct = prepare_frame(validation_df.loc[remaining], feature_columns)
            structured_preds = pd.Series(structured_model.predict(X_val_struct), index=remaining)
            text_preds = predict_text_values(text_model, validation_df.loc[remaining], target)

        best_mae = float("inf")
        for structured_weight in np.linspace(0.0, 1.0, 11):
            trial_predictions = {}
            for idx, value in regex_predictions.items():
                trial_predictions[idx] = value
            for idx in remaining:
                structured_value = clamp_prediction(float(structured_preds.loc[idx]), target)
                text_value = clamp_prediction(float(text_preds.loc[idx]), target)
                trial_predictions[idx] = combine_predictions(structured_value, text_value, target, structured_weight)
            pred_series = pd.Series(trial_predictions).sort_index()
            actual_series = val_actual.loc[pred_series.index]
            mae = float((actual_series - pred_series).abs().mean())
            if mae < best_mae:
                best_mae = mae
                best_weight = float(structured_weight)
        validation_mae = best_mae

    structured_model, structured_features = fit_structured_predictor(df, target, include_price)
    text_model = fit_text_predictor(df, target)
    return {
        "structured_model": structured_model,
        "structured_features": structured_features,
        "text_model": text_model,
        "structured_weight": best_weight,
        "validation_mae": validation_mae,
    }


def compute_model_confidence(target: str, structured_value: float, text_value: float, validation_mae: float) -> float:
    scale = max(TARGET_ERROR_SCALES[target], validation_mae)
    disagreement = abs(structured_value - text_value)
    confidence = 0.88 - (disagreement / (2.5 * scale)) - (validation_mae / (6 * scale))
    return round(float(max(0.12, min(0.9, confidence))), 3)


def hybrid_impute(df: pd.DataFrame, include_price: bool) -> tuple[pd.DataFrame, dict[str, object]]:
    working = df.copy()
    summary: dict[str, object] = {
        "mode": "analysis_safe" if include_price else "price_model_safe",
        "targets": {},
    }

    for target in TARGETS:
        missing_mask = working[target].isna()
        target_summary = {
            "missing_before": int(missing_mask.sum()),
            "regex_imputed": 0,
            "hybrid_model_imputed": 0,
            "missing_after": 0,
        }
        if not missing_mask.any():
            summary["targets"][target] = target_summary
            continue

        bundle = fit_target_bundle(working, target, include_price)
        structured_model = bundle["structured_model"]
        structured_features = bundle["structured_features"]
        text_model = bundle["text_model"]
        structured_weight = float(bundle["structured_weight"])
        validation_mae = float(bundle["validation_mae"])

        working[f"{target}_imputation_source"] = pd.Series(pd.NA, index=working.index, dtype="string")
        working[f"{target}_imputation_confidence"] = np.nan
        working[f"{target}_regex_candidate"] = np.nan
        working[f"{target}_structured_prediction"] = np.nan
        working[f"{target}_text_prediction"] = np.nan

        regex_indices = []
        model_indices = []

        for idx in working.index[missing_mask]:
            text_value = str(working.at[idx, "text_clean"]) if pd.notna(working.at[idx, "text_clean"]) else ""
            regex_value, regex_conf = extract_with_regex(text_value, target)
            if regex_value is not None:
                working.at[idx, target] = regex_value
                working.at[idx, f"{target}_regex_candidate"] = regex_value
                working.at[idx, f"{target}_imputation_source"] = "regex"
                working.at[idx, f"{target}_imputation_confidence"] = regex_conf
                regex_indices.append(idx)

        remaining_mask = working[target].isna()
        if remaining_mask.any():
            X_struct = prepare_frame(working.loc[remaining_mask], structured_features)
            struct_pred = pd.Series(structured_model.predict(X_struct), index=X_struct.index)
            text_pred = predict_text_values(text_model, working.loc[remaining_mask], target)

            for idx in X_struct.index:
                structured_value = clamp_prediction(float(struct_pred.loc[idx]), target)
                text_value = clamp_prediction(float(text_pred.loc[idx]), target)
                ensemble_value = combine_predictions(structured_value, text_value, target, structured_weight)
                confidence = compute_model_confidence(target, structured_value, text_value, validation_mae)

                working.at[idx, target] = ensemble_value
                working.at[idx, f"{target}_structured_prediction"] = structured_value
                working.at[idx, f"{target}_text_prediction"] = text_value
                working.at[idx, f"{target}_imputation_source"] = "hybrid_model"
                working.at[idx, f"{target}_imputation_confidence"] = confidence
                model_indices.append(idx)

        working[f"{target}_was_imputed"] = working[f"{target}_imputation_source"].notna()
        target_summary["regex_imputed"] = int(len(regex_indices))
        target_summary["hybrid_model_imputed"] = int(len(model_indices))
        target_summary["missing_after"] = int(working[target].isna().sum())
        target_summary["structured_weight"] = round(structured_weight, 2)
        target_summary["validation_mae"] = round(validation_mae, 4)
        summary["targets"][target] = target_summary

    return working, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build hybrid-imputed analysis-safe and leakage-safe tables using regex, structured models, and NLP models."
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

    analysis_df, analysis_summary = hybrid_impute(df, include_price=True)
    price_safe_df, price_safe_summary = hybrid_impute(df, include_price=False)

    report = {
        "row_count": int(len(df)),
        "analysis_safe": analysis_summary,
        "price_model_safe": price_safe_summary,
    }

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_df.to_parquet(output_dir / ANALYSIS_OUTPUT_PARQUET, index=False)
    analysis_df.to_csv(output_dir / ANALYSIS_OUTPUT_CSV, index=False)
    price_safe_df.to_parquet(output_dir / PRICE_SAFE_OUTPUT_PARQUET, index=False)
    price_safe_df.to_csv(output_dir / PRICE_SAFE_OUTPUT_CSV, index=False)
    with (output_dir / REPORT_JSON).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved hybrid imputation outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
