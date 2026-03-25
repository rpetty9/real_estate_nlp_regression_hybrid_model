import json
import re
import html
import runpy
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
TARGET_ORDER = ["beds", "baths", "garage", "stories", "sqft", "year_built"]
TARGET_LABELS = {
    "beds": "Bedrooms",
    "baths": "Bathrooms",
    "garage": "Garage",
    "stories": "Stories",
    "sqft": "Square Footage",
    "year_built": "Year Built",
}
AUDIENCE_VIEWS = {
    "ops": "Listing Ops",
    "pricing": "Buyer / Pricing",
    "copy": "Description Strategy",
}
STATE_INFO = {
    "Colorado": {"abbr": "CO", "fips": 8, "color": "#5b8e7d"},
    "Connecticut": {"abbr": "CT", "fips": 9, "color": "#8bb8c7"},
    "Indiana": {"abbr": "IN", "fips": 18, "color": "#2f5d62"},
    "Kentucky": {"abbr": "KY", "fips": 21, "color": "#7fb7be"},
    "New Hampshire": {"abbr": "NH", "fips": 33, "color": "#c7dfe6"},
    "New York": {"abbr": "NY", "fips": 36, "color": "#6ea5b3"},
    "Oregon": {"abbr": "OR", "fips": 41, "color": "#6c9fa1"},
    "Wisconsin": {"abbr": "WI", "fips": 55, "color": "#3e6f73"},
}
STATE_COORDS = {
    "Colorado": {"lat": 39.0, "lon": -105.5},
    "Connecticut": {"lat": 41.6, "lon": -72.7},
    "Indiana": {"lat": 39.9, "lon": -86.3},
    "Kentucky": {"lat": 37.8, "lon": -85.8},
    "New Hampshire": {"lat": 43.7, "lon": -71.6},
    "New York": {"lat": 42.9, "lon": -75.5},
    "Oregon": {"lat": 43.9, "lon": -120.6},
    "Wisconsin": {"lat": 44.5, "lon": -89.5},
}
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "your",
    "you",
    "are",
    "has",
    "have",
    "home",
    "homes",
    "property",
    "properties",
    "listing",
    "listings",
    "into",
    "just",
    "its",
    "it's",
    "their",
    "they",
    "them",
    "will",
    "all",
    "not",
    "but",
    "can",
    "was",
    "were",
    "our",
    "out",
    "new",
    "one",
    "two",
    "three",
    "four",
    "bedroom",
    "bedrooms",
    "bathroom",
    "bathrooms",
    "bath",
    "baths",
    "square",
    "feet",
    "foot",
}


def format_target_metric(value: float, target: str) -> str:
    if pd.isna(value):
        return "n/a"
    if target == "year_built":
        return str(int(round(float(value))))
    return f"{float(value):,.2f}"

st.set_page_config(
    page_title="Real Estate NLP Storyboard",
    page_icon="house",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        .hero {
            padding: 1.5rem 1.6rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #f8f2e9 0%, #eef6f1 48%, #f6f8fb 100%);
            border: 1px solid rgba(15, 23, 42, 0.08);
            margin-bottom: 1rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
        }
        .hero h1 {
            color: #102a43;
            margin-bottom: 0.2rem;
            font-size: 2.45rem;
        }
        .hero p {
            color: #334e68;
            margin-bottom: 0;
            font-size: 1.02rem;
        }
        .eyebrow {
            display: inline-block;
            margin-bottom: 0.6rem;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.12);
            color: #0f766e;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }
        .story-card {
            padding: 1rem 1.05rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(15, 23, 42, 0.06);
            min-height: 140px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        }
        .story-card h3 {
            margin-top: 0;
            color: #102a43;
            font-size: 1rem;
        }
        .story-card p {
            color: #486581;
            font-size: 0.95rem;
            margin-bottom: 0;
        }
        .section-note {
            padding: 0.8rem 1rem;
            border-left: 4px solid #0f766e;
            background: rgba(15, 118, 110, 0.08);
            border-radius: 8px;
            color: #134e4a;
            margin-bottom: 0.8rem;
        }
        .mini-note {
            color: #486581;
            font-size: 0.93rem;
            margin-top: -0.2rem;
            margin-bottom: 0.8rem;
        }
        .audience-banner {
            padding: 0.95rem 1rem;
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(15, 118, 110, 0.08), rgba(16, 42, 67, 0.04));
            border: 1px solid rgba(15, 23, 42, 0.06);
            margin-bottom: 0.8rem;
        }
        .audience-banner h3 {
            margin: 0 0 0.25rem 0;
            color: #102a43;
        }
        .audience-banner p {
            margin: 0;
            color: #486581;
        }
        .info-card {
            padding: 1rem 1.05rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(15, 23, 42, 0.06);
            min-height: 150px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        }
        .info-card h3 {
            margin-top: 0;
            color: #102a43;
            font-size: 1rem;
        }
        .info-card p {
            color: #486581;
            margin-bottom: 0;
        }
        .generated-copy {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            background: #fffdf8;
            border: 1px solid rgba(180, 83, 9, 0.15);
            color: #5f370e;
            line-height: 1.55;
        }
        .explain-text {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(15, 23, 42, 0.08);
            color: #243b53;
            line-height: 1.75;
            font-size: 1rem;
        }
        .cue-highlight {
            background: rgba(245, 158, 11, 0.28);
            padding: 0.08rem 0.18rem;
            border-radius: 0.28rem;
            box-shadow: inset 0 0 0 1px rgba(180, 83, 9, 0.18);
        }
        .cue-chip {
            display: inline-block;
            margin: 0.18rem 0.35rem 0.18rem 0;
            padding: 0.2rem 0.5rem;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.1);
            color: #0f766e;
            font-size: 0.86rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_reports() -> dict[str, object]:
    reports = {}
    report_files = {
        "hybrid_imputer": "hybrid_imputer_evaluation_report.json",
        "price_models": "price_model_comparison_report.json",
        "hybrid_summary": "real_estate_hybrid_imputation_report.json",
        "clusters": "listing_cluster_report.json",
    }
    for key, filename in report_files.items():
        path = PROCESSED / filename
        if path.exists():
            reports[key] = json.loads(path.read_text(encoding="utf-8"))
    return reports


@st.cache_data
def load_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    parquet_files = {
        "combined_clean": "combined_real_estate_clean.parquet",
        "analysis_imputed": "real_estate_hybrid_imputed_analysis.parquet",
        "price_safe_imputed": "real_estate_hybrid_imputed_price_safe.parquet",
        "clustered": "listing_clusters.parquet",
    }
    for key, filename in parquet_files.items():
        path = PROCESSED / filename
        if path.exists():
            tables[key] = pd.read_parquet(path)
    return tables


@st.cache_data
def load_known_value_examples_artifact() -> pd.DataFrame:
    path = PROCESSED / "hybrid_imputer_known_value_examples.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_resource
def load_hybrid_imputation_helpers() -> dict[str, object]:
    return runpy.run_path(str(PROJECT_ROOT / "scripts" / "build_hybrid_imputation_table.py"))


@st.cache_data
def build_known_value_examples(target: str, mode: str = "analysis_safe") -> pd.DataFrame:
    table_name = "real_estate_hybrid_imputed_analysis.parquet" if mode == "analysis_safe" else "real_estate_hybrid_imputed_price_safe.parquet"
    df = pd.read_parquet(PROCESSED / table_name)
    required = [target, "text", "text_clean", "type", "state", "record_id"]
    if any(column not in df.columns for column in required):
        return pd.DataFrame()

    helpers = load_hybrid_imputation_helpers()
    fit_target_bundle = helpers["fit_target_bundle"]
    predict_text_values = helpers["predict_text_values"]
    prepare_frame = helpers["prepare_frame"]
    clamp_prediction = helpers["clamp_prediction"]
    combine_predictions = helpers["combine_predictions"]
    extract_with_regex = helpers["extract_with_regex"]

    include_price = mode == "analysis_safe"
    working = df[df[target].notna() & df["text"].notna() & df["text_clean"].notna()].copy()
    if target in {"year_built", "beds", "baths", "sqft"}:
        working = working[working["type"].fillna("").astype(str).str.lower() != "land"]
    if len(working) < 100:
        return pd.DataFrame()
    if len(working) > 4000:
        working = working.sample(n=4000, random_state=42)

    bundle = fit_target_bundle(working, target, include_price)
    structured_model = bundle["structured_model"]
    structured_features = bundle["structured_features"]
    text_model = bundle["text_model"]
    structured_weight = float(bundle["structured_weight"])

    eval_df = working.copy()
    regex_values = []
    structured_values = []
    text_values = []
    final_values = []
    for idx in eval_df.index:
        regex_value, _ = extract_with_regex(str(eval_df.at[idx, "text_clean"]), target)
        regex_values.append(regex_value)

    no_regex_mask = pd.Series(regex_values, index=eval_df.index).isna()
    eval_df = eval_df.loc[no_regex_mask].copy()
    if eval_df.empty:
        return pd.DataFrame()

    X_struct = prepare_frame(eval_df, structured_features)
    struct_pred = structured_model.predict(X_struct)
    text_pred = predict_text_values(text_model, eval_df, target)
    for s_val, t_val in zip(struct_pred, text_pred):
        structured_value = clamp_prediction(float(s_val), target)
        text_value = clamp_prediction(float(t_val), target)
        final_value = combine_predictions(structured_value, text_value, target, structured_weight)
        structured_values.append(structured_value)
        text_values.append(text_value)
        final_values.append(final_value)

    eval_df[f"{target}_structured_prediction"] = structured_values
    eval_df[f"{target}_text_prediction"] = text_values
    eval_df[f"{target}_demo_prediction"] = final_values
    eval_df["actual_value"] = eval_df[target].astype(float)
    eval_df["prediction_error"] = (eval_df[f"{target}_demo_prediction"] - eval_df["actual_value"]).abs()
    eval_df["model_gap"] = (eval_df[f"{target}_structured_prediction"] - eval_df[f"{target}_text_prediction"]).abs()

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
    return eval_df.sort_values(["prediction_error", "model_gap"]).head(20)


@st.cache_data
def load_price_feature_importance() -> pd.DataFrame:
    path = PROCESSED / "price_feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])


@st.cache_resource
def fit_text_detail_model(target: str, mode: str = "analysis_safe"):
    helpers = load_hybrid_imputation_helpers()
    fit_text_predictor = helpers["fit_text_predictor"]
    table_name = "real_estate_hybrid_imputed_analysis.parquet" if mode == "analysis_safe" else "real_estate_hybrid_imputed_price_safe.parquet"
    df = pd.read_parquet(PROCESSED / table_name)
    if target not in df.columns or "text_clean" not in df.columns:
        return None
    train_df = df[df[target].notna() & df["text_clean"].notna()].copy()
    if train_df.empty:
        return None
    return fit_text_predictor(train_df, target)


def explain_text_prediction_terms(target: str, text: str, mode: str = "analysis_safe", top_n: int = 8) -> pd.DataFrame:
    if not text:
        return pd.DataFrame(columns=["term", "contribution"])
    helpers = load_hybrid_imputation_helpers()
    model_bundle = fit_text_detail_model(target, mode)
    if model_bundle is None:
        return pd.DataFrame(columns=["term", "contribution"])
    compose_text_corpus = helpers.get("compose_text_corpus")
    if isinstance(model_bundle, dict):
        global_model = model_bundle.get("global_model")
    else:
        global_model = model_bundle
    if global_model is None:
        return pd.DataFrame(columns=["term", "contribution"])
    if compose_text_corpus is not None:
        text_frame = pd.DataFrame({"text_clean": [text], "text": [text]})
        transformed_text = compose_text_corpus(text_frame, target).iloc[0]
    else:
        transformed_text = str(text)
    if "features" in global_model.named_steps:
        features_union = global_model.named_steps["features"]
        ridge = global_model.named_steps["model"]
    elif "tfidf" in global_model.named_steps:
        features_union = None
        ridge = global_model.named_steps["model"]
        vectorizer = global_model.named_steps["tfidf"]
    else:
        return pd.DataFrame(columns=["term", "contribution"])

    if features_union is None:
        transformed = vectorizer.transform([transformed_text])
        if transformed.shape[1] == 0:
            return pd.DataFrame(columns=["term", "contribution"])
        contributions = transformed.multiply(ridge.coef_).toarray().ravel()
        feature_names = vectorizer.get_feature_names_out()
        rows = [{"term": feature_names[idx], "contribution": float(contributions[idx])} for idx in transformed.nonzero()[1]]
        if not rows:
            return pd.DataFrame(columns=["term", "contribution"])
        contribution_df = pd.DataFrame(rows).sort_values("contribution", ascending=False)
        return contribution_df.head(top_n)

    transformed = features_union.transform([transformed_text])
    if transformed.shape[1] == 0:
        return pd.DataFrame(columns=["term", "contribution"])

    rows = []
    start_idx = 0
    for name, vectorizer in features_union.transformer_list:
        part = vectorizer.transform([transformed_text])
        width = part.shape[1]
        if width == 0:
            continue
        part_coef = ridge.coef_[start_idx : start_idx + width]
        part_contrib = part.multiply(part_coef).toarray().ravel()
        feature_names = vectorizer.get_feature_names_out()
        nonzero = part.nonzero()[1]
        for idx in nonzero:
            rows.append(
                {
                    "term": f"{name}: {feature_names[idx]}",
                    "contribution": float(part_contrib[idx]),
                }
            )
        start_idx += width

    if not rows:
        return pd.DataFrame(columns=["term", "contribution"])
    contribution_df = pd.DataFrame(rows).sort_values("contribution", ascending=False)
    return contribution_df.head(top_n)


def render_ranked_bar_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    color: str = "#3e6f73",
    percent_axis: bool = False,
    height: int = 320,
) -> None:
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color=color)
        .encode(
            x=alt.X(
                f"{value_col}:Q",
                title=title,
                axis=alt.Axis(format=".0%" if percent_axis else ","),
            ),
            y=alt.Y(f"{category_col}:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip(f"{category_col}:N", title=category_col.replace("_", " ").title()),
                alt.Tooltip(f"{value_col}:Q", title=title, format=".1%" if percent_axis else ",.2f"),
            ],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch")


def render_grouped_metric_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    color_col: str,
    title: str,
    color_range: list[str],
    percent_axis: bool = False,
    height: int = 330,
) -> None:
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X(f"{category_col}:N", title=None, axis=alt.Axis(labelAngle=-20)),
            y=alt.Y(
                f"{value_col}:Q",
                title=title,
                axis=alt.Axis(format=".0%" if percent_axis else ","),
            ),
            color=alt.Color(
                f"{color_col}:N",
                scale=alt.Scale(range=color_range),
                legend=alt.Legend(title=None, orient="top"),
            ),
            xOffset=f"{color_col}:N",
            tooltip=[
                alt.Tooltip(f"{category_col}:N", title=category_col.replace("_", " ").title()),
                alt.Tooltip(f"{color_col}:N", title=color_col.replace("_", " ").title()),
                alt.Tooltip(f"{value_col}:Q", title=title, format=".1%" if percent_axis else ",.2f"),
            ],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch")


@st.cache_data
def load_price_prediction_rows() -> pd.DataFrame:
    path = PROCESSED / "price_model_comparison_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def bootstrap_price_model_metrics(n_boot: int = 250, seed: int = 42) -> pd.DataFrame:
    predictions = load_price_prediction_rows()
    if predictions.empty:
        return pd.DataFrame()

    model_map = {
        "text_only": "text_only_predicted_price",
        "structured_only": "structured_only_predicted_price",
        "text_plus_structured": "predicted_price",
    }
    actual = predictions["actual_price"].to_numpy(dtype=float)
    n = len(actual)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []

    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        denom = np.sum((y_true - y_true.mean()) ** 2)
        if denom == 0:
            return float("nan")
        return 1 - np.sum((y_true - y_pred) ** 2) / denom

    for model_name, pred_col in model_map.items():
        pred = predictions[pred_col].to_numpy(dtype=float)
        mae_samples = []
        rmse_samples = []
        r2_samples = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            y_true = actual[idx]
            y_pred = pred[idx]
            err = y_true - y_pred
            mae_samples.append(np.mean(np.abs(err)))
            rmse_samples.append(np.sqrt(np.mean(err ** 2)))
            r2_samples.append(r2_score(y_true, y_pred))
        for metric_name, samples in [("mae", mae_samples), ("rmse", rmse_samples), ("r2", r2_samples)]:
            rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "mean": float(np.mean(samples)),
                    "std": float(np.std(samples, ddof=1)),
                    "ci_low": float(np.quantile(samples, 0.025)),
                    "ci_high": float(np.quantile(samples, 0.975)),
                    "n_boot": n_boot,
                }
            )
    return pd.DataFrame(rows)


def build_target_summary(impute_df: pd.DataFrame, hybrid_summary: dict[str, object], mode: str) -> pd.DataFrame:
    rows = []
    mode_summary = hybrid_summary[mode]["targets"]
    for target in TARGET_ORDER:
        result_row = impute_df[(impute_df["mode"] == mode) & (impute_df["target"] == target)].iloc[0]
        summary_row = mode_summary[target]
        rows.append(
            {
                "target": target,
                "label": TARGET_LABELS[target],
                "tolerance_accuracy": result_row["tolerance_accuracy"],
                "mae": result_row["mae"],
                "regex_coverage": result_row["regex_coverage"],
                "missing_before": summary_row["missing_before"],
                "regex_imputed": summary_row["regex_imputed"],
                "hybrid_model_imputed": summary_row["hybrid_model_imputed"],
                "structured_weight": summary_row.get("structured_weight"),
                "validation_mae": summary_row.get("validation_mae"),
            }
        )
    return pd.DataFrame(rows)


def format_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    if "tolerance_accuracy" in formatted:
        formatted["tolerance_accuracy"] = formatted["tolerance_accuracy"].map(lambda x: f"{x:.1%}")
    if "regex_coverage" in formatted:
        formatted["regex_coverage"] = formatted["regex_coverage"].map(lambda x: f"{x:.1%}")
    if "mae" in formatted:
        formatted["mae"] = formatted["mae"].map(lambda x: f"{x:,.2f}")
    if "validation_mae" in formatted:
        formatted["validation_mae"] = formatted["validation_mae"].map(
            lambda x: "" if pd.isna(x) else f"{x:,.2f}"
        )
    if "structured_weight" in formatted:
        formatted["structured_weight"] = formatted["structured_weight"].map(
            lambda x: "" if pd.isna(x) else f"{x:.2f}"
        )
    formatted = formatted.rename(
        columns={
            "label": "Target",
            "tolerance_accuracy": "Tolerance Accuracy",
            "mae": "MAE",
            "regex_coverage": "Regex Coverage",
            "missing_before": "Missing Before",
            "regex_imputed": "Recovered By Regex",
            "hybrid_model_imputed": "Recovered By Hybrid Model",
            "structured_weight": "Structured Weight",
            "validation_mae": "Validation MAE",
            "rows_evaluated": "Rows Evaluated",
            "rmse": "RMSE",
            "tolerance_error_rate": "Tolerance Error Rate",
            "rmse_to_mae_ratio": "RMSE / MAE Ratio",
        }
    )
    return formatted


def safe_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> pd.DataFrame:
    existing = [col for col in columns if col in df.columns]
    out = df[existing]
    return out.head(limit) if limit is not None else out


def show_story_cards(
    mode_summary: pd.DataFrame, price_df: pd.DataFrame, cluster_df: pd.DataFrame, audience_view: str
) -> None:
    best_target = mode_summary.sort_values("tolerance_accuracy", ascending=False).iloc[0]
    weakest_target = mode_summary.sort_values("tolerance_accuracy", ascending=True).iloc[0]
    structured_r2 = float(price_df.loc[price_df["model"] == "structured_only", "r2"].iloc[0])
    text_r2 = float(price_df.loc[price_df["model"] == "text_only", "r2"].iloc[0])
    richest_cluster = cluster_df.sort_values("avg_price", ascending=False).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="story-card">
                <h3>Imputer Strength</h3>
                <p><strong>{best_target['label']}</strong> is the strongest recovered field right now, with about <strong>{best_target['tolerance_accuracy']:.1%}</strong> tolerance accuracy in masked holdout tests.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="story-card">
                <h3>Main Caveat</h3>
                <p><strong>{weakest_target['label']}</strong> is still the weakest target at <strong>{weakest_target['tolerance_accuracy']:.1%}</strong>. The system can fill it, but it is not yet the most trustworthy field.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="story-card">
                <h3>Price Modeling</h3>
                <p>The current structured baseline reaches <strong>R² = {structured_r2:.3f}</strong>, while text-only reaches <strong>R² = {text_r2:.3f}</strong>. Text carries signal, but structure is still stronger.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        title = "Cluster Story" if audience_view != "copy" else "Language Pattern"
        st.markdown(
            f"""
            <div class="story-card">
                <h3>{title}</h3>
                <p>Cluster <strong>{int(richest_cluster['cluster_id'])}</strong> has the highest average price at about <strong>${richest_cluster['avg_price']:,.0f}</strong>, showing the text segmentation is finding meaningfully different listing groups.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def summarize_audience(audience_view: str) -> tuple[str, str]:
    if audience_view == "ops":
        return (
            "Listing Operations View",
            "This view is for teams trying to repair messy inventory, recover missing fields, and understand which imputed attributes are reliable enough to trust downstream.",
        )
    if audience_view == "pricing":
        return (
            "Buyer / Pricing View",
            "This view focuses on price signal: which attributes matter most, how well current models explain price, and how the cleaned/imputed features support pricing analysis.",
        )
    return (
        "Description Strategy View",
        "This view focuses on what high-value listings sound like, which themes show up in richer descriptions, and how the NLP layer can guide better listing copy.",
    )


@st.cache_data
def build_copywriter_insights(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df[df["is_quality_filtered_text"] & df["text_clean"].notna() & df["list_price"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=["term", "high_value_share", "rest_share", "lift"]), pd.DataFrame()

    threshold = working["list_price"].quantile(0.9)
    working["is_high_value"] = working["list_price"] >= threshold

    def count_terms(text_series: pd.Series) -> Counter:
        counter: Counter = Counter()
        for text in text_series.dropna():
            tokens = re.findall(r"\b[a-z]{4,}\b", str(text).lower())
            counter.update(token for token in tokens if token not in STOPWORDS)
        return counter

    high_counter = count_terms(working.loc[working["is_high_value"], "text_clean"])
    rest_counter = count_terms(working.loc[~working["is_high_value"], "text_clean"])
    high_total = max(sum(high_counter.values()), 1)
    rest_total = max(sum(rest_counter.values()), 1)

    rows = []
    candidate_terms = set(high_counter).union(rest_counter)
    for term in candidate_terms:
        high_count = high_counter.get(term, 0)
        if high_count < 30:
            continue
        rest_count = rest_counter.get(term, 0)
        high_share = high_count / high_total
        rest_share = rest_count / rest_total
        lift = high_share / max(rest_share, 1e-9)
        if lift >= 1.5:
            rows.append(
                {
                    "term": term,
                    "high_value_share": high_share,
                    "rest_share": rest_share,
                    "lift": lift,
                }
            )

    term_df = pd.DataFrame(rows).sort_values(["lift", "high_value_share"], ascending=[False, False]).head(15)
    example_df = working.loc[working["is_high_value"], ["record_id", "state", "type", "list_price", "text"]].head(25)
    return term_df, example_df


@st.cache_data
def build_copy_recommendation_inputs(df: pd.DataFrame) -> dict[str, object]:
    working = df[df["is_quality_filtered_text"] & df["text_clean"].notna() & df["list_price"].notna()].copy()
    if working.empty:
        return {"states": [], "types": [], "price_threshold": None}
    return {
        "states": sorted(working["state"].dropna().unique().tolist()),
        "types": sorted(working["type"].dropna().unique().tolist()),
        "price_threshold": float(working["list_price"].quantile(0.9)),
    }


def top_amenities_for_slice(df: pd.DataFrame, top_n: int = 8) -> list[str]:
    amenity_cols = [col for col in df.columns if col.startswith("has_")]
    rows = []
    for col in amenity_cols:
        share = df[col].fillna(False).mean()
        if share > 0:
            rows.append((col.replace("has_", "").replace("_", " "), share))
    rows.sort(key=lambda item: item[1], reverse=True)
    return [name for name, _share in rows[:top_n]]


def build_description_recommendation(
    df: pd.DataFrame,
    state: str | None,
    property_type: str | None,
) -> tuple[pd.DataFrame, list[str], str, pd.DataFrame]:
    working = df[df["is_quality_filtered_text"] & df["text_clean"].notna() & df["list_price"].notna()].copy()
    if state and state != "All":
        working = working[working["state"] == state]
    if property_type and property_type != "All":
        working = working[working["type"] == property_type]
    if working.empty:
        return pd.DataFrame(), [], "Not enough matching listings to generate a recommendation yet.", pd.DataFrame()

    threshold = working["list_price"].quantile(0.85)
    high_value = working[working["list_price"] >= threshold].copy()
    if high_value.empty:
        high_value = working.nlargest(min(200, len(working)), "list_price").copy()

    term_df, _ = build_copywriter_insights(working)
    top_terms = term_df["term"].head(6).tolist() if not term_df.empty else []
    top_amenities = top_amenities_for_slice(high_value, top_n=6)

    lead_bits = []
    if property_type and property_type != "All":
        lead_bits.append(property_type.replace("_", " "))
    if state and state != "All":
        lead_bits.append(f"in {state}")
    lead_phrase = " ".join(lead_bits).strip() or "home"

    recommendation = (
        f"Start with a strong opening that immediately positions the {lead_phrase}. "
        f"Then highlight the most marketable features such as {', '.join(top_amenities[:3]) if top_amenities else 'updated finishes, standout amenities, and location advantages'}. "
        f"Use concrete language that signals value and lifestyle, and weave in terms like {', '.join(top_terms[:4]) if top_terms else 'spacious, updated, private, and inviting'} when they are truthful to the listing. "
        "Close with one sentence about livability or buyer appeal rather than generic filler."
    )

    examples = high_value.loc[:, ["record_id", "state", "type", "list_price", "text"]].head(12)
    return term_df, top_amenities, recommendation, examples


def cluster_display_label(cluster_row: pd.Series | dict[str, object]) -> str:
    cluster_id = int(cluster_row["cluster_id"])
    label = cluster_row.get("cluster_label", f"Cluster {cluster_id}")
    return f"Cluster {cluster_id}: {label}"


def build_cluster_projection_chart(clustered_table: pd.DataFrame, x_col: str, y_col: str, height: int = 360) -> None:
    sample = clustered_table.dropna(subset=[x_col, y_col, "cluster_id"]).copy()
    if len(sample) > 5000:
        sample = sample.sample(n=5000, random_state=42)
    sample["Cluster"] = sample["cluster_id"].astype(int).astype(str)
    chart = (
        alt.Chart(sample)
        .mark_circle(size=58, opacity=0.55)
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_col.replace("cluster_", "").upper()),
            y=alt.Y(f"{y_col}:Q", title=y_col.replace("cluster_", "").upper()),
            color=alt.Color("Cluster:N", legend=alt.Legend(title="Cluster")),
            tooltip=[
                alt.Tooltip("cluster_id:Q", title="Cluster"),
                alt.Tooltip("state:N", title="State"),
                alt.Tooltip("type:N", title="Property type"),
                alt.Tooltip("list_price:Q", title="List price", format="$,.0f"),
                alt.Tooltip("quality_score:Q", title="Quality", format=".1f"),
            ],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def build_cluster_3d_chart(clustered_table: pd.DataFrame, cluster_df: pd.DataFrame, height: int = 520) -> None:
    sample = clustered_table.dropna(subset=["cluster_x", "cluster_y", "cluster_z", "cluster_id"]).copy()
    if len(sample) > 2500:
        sample = sample.sample(n=2500, random_state=42)
    if sample.empty:
        st.info("Not enough clustered rows were available to render the 3D cluster view.")
        return
    palette = {
        0: "#3e6f73",
        1: "#6ea5b3",
        2: "#8cb8c7",
        3: "#5b8e7d",
        4: "#6a9fa1",
        5: "#7fb7be",
        6: "#6f8f5e",
        7: "#bf9254",
        8: "#817cbf",
        9: "#5378ad",
    }
    for raw_col, scaled_col in [("cluster_x", "x_plot"), ("cluster_y", "y_plot"), ("cluster_z", "z_plot")]:
        centered = sample[raw_col] - sample[raw_col].mean()
        scale = float(sample[raw_col].std()) if float(sample[raw_col].std()) > 1e-6 else 1.0
        sample[scaled_col] = (centered / scale) * 150.0

    cluster_label_map = {}
    if not cluster_df.empty and "cluster_label" in cluster_df.columns:
        for _, row in cluster_df.iterrows():
            cluster_label_map[int(row["cluster_id"])] = str(row["cluster_label"])

    sample["cluster_color"] = sample["cluster_id"].astype(int).map(lambda x: palette.get(int(x), "#3e6f73"))
    points = []
    for _, row in sample.iterrows():
        cluster_id = int(row["cluster_id"])
        cluster_name = cluster_label_map.get(cluster_id, f"Cluster {cluster_id}")
        points.append(
            {
                "x": round(float(row["x_plot"]), 3),
                "y": round(float(row["y_plot"]), 3),
                "z": round(float(row["z_plot"]), 3),
                "color": row["cluster_color"],
                "cluster": cluster_id,
                "cluster_name": cluster_name,
                "state": str(row.get("state", "")),
                "ptype": str(row.get("type", "")),
                "price": f"${float(row.get('list_price', 0) or 0):,.0f}",
                "quality": f"{float(row.get('quality_score', 0) or 0):.1f}",
            }
        )

    html_payload = json.dumps(points)
    legend_items = []
    for cluster_id, label in sorted(cluster_label_map.items()):
        legend_items.append(
            f"<div style='display:flex; align-items:center; gap:8px; margin:4px 10px 4px 0; min-width:260px;'>"
            f"<span style='display:inline-block; width:12px; height:12px; border-radius:999px; background:{palette.get(cluster_id, '#3e6f73')};'></span>"
            f"<span style='font:13px/1.35 sans-serif; color:#4a646c;'><strong>C{cluster_id}</strong> {label}</span></div>"
        )
    legend_html = "".join(legend_items)
    html_block = f"""
    <div style="width:100%; background:#f7faf8; border:1px solid #d7e5e2; border-radius:18px; padding:12px 12px 6px 12px;">
      <div style="font:600 13px/1.4 sans-serif; color:#48636a; margin:0 0 8px 2px;">
        Drag to rotate. Scroll to zoom. Hover near a point to inspect a listing.
      </div>
      <canvas id="cluster3d" width="980" height="520" style="width:100%; height:520px; display:block; background:linear-gradient(180deg,#fbfdfc 0%,#f0f7f5 100%); border-radius:14px;"></canvas>
      <div id="cluster-tooltip" style="position:relative;"></div>
      <div style="display:flex; flex-wrap:wrap; margin-top:10px; padding:8px 4px 2px 4px; border-top:1px solid #e0ebe8;">
        {legend_html}
      </div>
    </div>
    <script>
    (function() {{
      const points = {html_payload};
      const canvas = document.getElementById("cluster3d");
      const ctx = canvas.getContext("2d");
      const tooltip = document.getElementById("cluster-tooltip");
      let width = canvas.width;
      let height = canvas.height;
      let yaw = 0.7;
      let pitch = -0.45;
      let zoom = 1.0;
      let dragging = false;
      let lastX = 0;
      let lastY = 0;

      function resize() {{
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = rect.width * dpr;
        canvas.height = 520 * dpr;
        width = rect.width;
        height = 520;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
      }}

      function rotatePoint(p) {{
        const cy = Math.cos(yaw), sy = Math.sin(yaw);
        const cx = Math.cos(pitch), sx = Math.sin(pitch);
        let x = p.x * cy - p.z * sy;
        let z = p.x * sy + p.z * cy;
        let y = p.y * cx - z * sx;
        z = p.y * sx + z * cx;
        return {{x, y, z}};
      }}

      function project(p) {{
        const scale = 2.6;
        const depth = 440;
        const perspective = depth / (depth + p.z + 260);
        return {{
          x: width / 2 + p.x * scale * perspective,
          y: height / 2 + p.y * scale * perspective,
          r: Math.max(2.4, 5.8 * perspective),
          depth: p.z,
        }};
      }}

      function drawAxes(transformPoint) {{
        const axes = [
          {{a: {{x:-180,y:0,z:0}}, b: {{x:180,y:0,z:0}}, color:"#94a9b2", label:"X: text dimension 1"}},
          {{a: {{x:0,y:-180,z:0}}, b: {{x:0,y:180,z:0}}, color:"#94a9b2", label:"Y: text dimension 2"}},
          {{a: {{x:0,y:0,z:-180}}, b: {{x:0,y:0,z:180}}, color:"#94a9b2", label:"Z: text dimension 3"}},
        ];
        ctx.font = "12px sans-serif";
        ctx.fillStyle = "#607d86";
        axes.forEach(axis => {{
          const pa = transformPoint(project(rotatePoint(axis.a)));
          const pb = transformPoint(project(rotatePoint(axis.b)));
          ctx.beginPath();
          ctx.strokeStyle = axis.color;
          ctx.lineWidth = 1.2;
          ctx.moveTo(pa.x, pa.y);
          ctx.lineTo(pb.x, pb.y);
          ctx.stroke();
          ctx.fillText(axis.label, pb.x + 6, pb.y - 4);
        }});
      }}

      function draw(mouseX=null, mouseY=null) {{
        ctx.clearRect(0, 0, width, height);
        let projected = points.map(p => {{
          const rotated = rotatePoint(p);
          const screen = project(rotated);
          return {{...p, ...screen}};
        }});

        const xs = projected.map(p => p.x);
        const ys = projected.map(p => p.y);
        const minX = Math.min(...xs), maxX = Math.max(...xs);
        const minY = Math.min(...ys), maxY = Math.max(...ys);
        const clusterWidth = maxX - minX;
        const clusterHeight = maxY - minY;
        const fitScale = Math.min((width * 0.62) / Math.max(clusterWidth, 1), (height * 0.66) / Math.max(clusterHeight, 1), 1.35);
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const visualScale = fitScale * zoom;
        const transformPoint = (p) => ({{
          x: ((p.x - centerX) * visualScale) + (width * 0.5),
          y: ((p.y - centerY) * visualScale) + (height * 0.52),
        }});
        projected = projected.map(p => {{
          const shifted = transformPoint(p);
          return {{
            ...p,
            x: shifted.x,
            y: shifted.y,
            r: Math.max(2.8, p.r * Math.min(1.3, fitScale + 0.12) * Math.sqrt(zoom)),
          }};
        }});

        projected = projected.sort((a,b) => a.depth - b.depth);

        drawAxes(transformPoint);

        const clusterGroups = new Map();
        projected.forEach(p => {{
          if (!clusterGroups.has(p.cluster)) clusterGroups.set(p.cluster, []);
          clusterGroups.get(p.cluster).push(p);
        }});

        for (const [clusterId, pts] of clusterGroups.entries()) {{
          const cx = pts.reduce((s,p) => s + p.x, 0) / pts.length;
          const cy = pts.reduce((s,p) => s + p.y, 0) / pts.length;
          const avgR = pts.reduce((s,p) => s + Math.hypot(p.x - cx, p.y - cy), 0) / pts.length;
          const haloR = Math.max(52, Math.min(165, avgR * 0.95));
          ctx.beginPath();
          ctx.fillStyle = pts[0].color + "22";
          ctx.strokeStyle = pts[0].color + "55";
          ctx.lineWidth = 1.2;
          ctx.arc(cx, cy, haloR, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = "#46616a";
          ctx.font = "600 12px sans-serif";
          const clusterName = pts[0].cluster_name.length > 18 ? pts[0].cluster_name.slice(0, 18) + "..." : pts[0].cluster_name;
          ctx.fillText(`C${{clusterId}} ${{clusterName}}`, cx + haloR + 8, cy);
        }}

        let hovered = null;
        projected.forEach(p => {{
          ctx.beginPath();
          ctx.fillStyle = p.color;
          ctx.globalAlpha = 0.82;
          ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
          ctx.fill();
          if (mouseX !== null && mouseY !== null) {{
            const dx = p.x - mouseX;
            const dy = p.y - mouseY;
            if (dx*dx + dy*dy <= Math.max(64, p.r*p.r*3)) hovered = p;
          }}
        }});
        ctx.globalAlpha = 1;

        if (hovered) {{
          tooltip.innerHTML = `<div style="margin-top:8px; padding:10px 12px; background:#15303a; color:white; border-radius:10px; font:13px/1.45 sans-serif; width:max-content; max-width:100%;">
            <strong>Cluster:</strong> C${{hovered.cluster}} ${{hovered.cluster_name}}<br>
            <strong>State:</strong> ${{hovered.state}}<br>
            <strong>Property Type:</strong> ${{hovered.ptype}}<br>
            <strong>List Price:</strong> ${{hovered.price}}<br>
            <strong>Quality:</strong> ${{hovered.quality}}
          </div>`;
        }} else {{
          tooltip.innerHTML = `<div style="margin-top:8px; color:#607d86; font:13px/1.45 sans-serif;">
            <strong>What you are looking at:</strong> each dot is one listing description projected into reduced text space. Colored halos show the rough center of each language cluster. Denser overlap means those descriptions use more similar wording.
          </div>`;
        }}
      }}

      canvas.addEventListener("mousedown", (e) => {{
        dragging = true;
        lastX = e.offsetX;
        lastY = e.offsetY;
      }});
      window.addEventListener("mouseup", () => dragging = false);
      canvas.addEventListener("mousemove", (e) => {{
        if (dragging) {{
          yaw += (e.offsetX - lastX) * 0.01;
          pitch += (e.offsetY - lastY) * 0.01;
          pitch = Math.max(-1.3, Math.min(1.3, pitch));
          lastX = e.offsetX;
          lastY = e.offsetY;
          draw(e.offsetX, e.offsetY);
        }} else {{
          draw(e.offsetX, e.offsetY);
        }}
      }});
      canvas.addEventListener("wheel", (e) => {{
        e.preventDefault();
        zoom *= e.deltaY < 0 ? 1.12 : 0.9;
        zoom = Math.max(0.65, Math.min(3.4, zoom));
        draw();
      }}, {{passive:false}});

      window.addEventListener("resize", resize);
      resize();
    }})();
    </script>
    """
    components.html(html_block, height=height + 180)


def render_cluster_analysis(cluster_df: pd.DataFrame, clustered_table: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-note">Clusters are language-based segments discovered from the listing descriptions. They are not hand-labeled categories; they are recurring patterns the NLP pipeline found in how different homes are described.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "This section explains what the clusters mean, how they can be used, and why they matter. The core idea is simple: if different language groups also separate on price, quality, and property mix, then the text is carrying real market information."
    )

    if cluster_df.empty or clustered_table.empty:
        st.info("Cluster artifacts are missing. Rebuild the cluster outputs to view this section.")
        return

    cluster_view = cluster_df.copy()
    if "cluster_label" not in cluster_view.columns:
        cluster_view["cluster_label"] = cluster_view["cluster_id"].apply(lambda x: f"Cluster {int(x)}")
    cluster_view["Cluster"] = cluster_view.apply(cluster_display_label, axis=1)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Language clusters", f"{cluster_view['cluster_id'].nunique()}")
    with m2:
        largest = cluster_view.sort_values("row_count", ascending=False).iloc[0]
        st.metric("Largest cluster", cluster_display_label(largest), f"{int(largest['row_count']):,} listings")
    with m3:
        priciest = cluster_view.sort_values("avg_price", ascending=False).iloc[0]
        st.metric("Highest-price cluster", cluster_display_label(priciest), f"${priciest['avg_price']:,.0f} avg price")

    st.subheader("How These Clusters Can Be Used")
    st.markdown(
        "They help turn the NLP work into something interpretable and useful: segment listing styles, compare premium versus lower-value language, support semantic search, and build marketing or pricing views around real text patterns rather than only structured fields."
    )

    st.subheader("Cluster Summary")
    summary_table = cluster_view.loc[:, ["Cluster", "row_count", "avg_price", "avg_quality", "dominant_type", "top_terms"]].copy()
    summary_table["row_count"] = summary_table["row_count"].map(lambda x: f"{int(x):,}")
    summary_table["avg_price"] = summary_table["avg_price"].map(lambda x: f"${x:,.0f}")
    summary_table["avg_quality"] = summary_table["avg_quality"].map(lambda x: f"{x:,.1f}")
    summary_table["top_terms"] = summary_table["top_terms"].apply(lambda x: ", ".join(x[:6]) if isinstance(x, list) else str(x))
    summary_table = summary_table.rename(
        columns={
            "row_count": "Listings",
            "avg_price": "Average Price",
            "avg_quality": "Average Quality",
            "dominant_type": "Dominant Property Type",
            "top_terms": "Top Terms",
        }
    )
    st.dataframe(summary_table, width="stretch", hide_index=True)
    st.caption(
        "Interpretation: the combination of top terms, dominant property type, and average price is what turns a cluster from a machine grouping into a human-readable listing segment."
    )

    st.subheader("2D / 3D Cluster Map")
    st.markdown(
        "The 3D chart below is rotatable and shows listings in reduced text space. Each point is one listing description, and points that sit closer together use more similar language. The axes are not raw business variables; they are the first three reduced text dimensions produced by the clustering pipeline."
    )
    st.markdown(
        """
        <div class="mini-note">
            <strong>Axis guide:</strong> X, Y, and Z are reduced NLP dimensions, not literal columns like price or square footage.
            They summarize combinations of language patterns across the descriptions. Use them to judge separation and shape, not to read a direct business meaning from the axis values themselves.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("This tab can take a minute or two to fully render the cluster visuals, especially right after the app launches. The 3D view is sampling and drawing a large number of listings in the browser.")
    build_cluster_3d_chart(clustered_table, cluster_df, height=520)
    st.caption(
        "How to use the 3D view: drag to rotate, zoom in on dense groups, and hover over points to inspect which state, property type, and price range sit inside each language cluster."
    )
    st.markdown("#### Supporting 2D Projections")
    st.markdown(
        "These flatter views make it easier to see separation without perspective effects. If a cluster stays grouped across multiple projections, that is a stronger sign that the language segment is real."
    )
    build_cluster_projection_chart(clustered_table, "cluster_x", "cluster_y", height=420)
    c1, c2 = st.columns(2)
    with c1:
        build_cluster_projection_chart(clustered_table, "cluster_x", "cluster_z", height=300)
    with c2:
        build_cluster_projection_chart(clustered_table, "cluster_y", "cluster_z", height=300)
    st.caption(
        "Interpretation: when the same colors form pockets across these projections, it suggests the text-defined segments are stable and meaningful rather than randomly mixed."
    )

    st.subheader("Cluster Detail")
    options = {cluster_display_label(row): int(row["cluster_id"]) for _, row in cluster_view.sort_values("avg_price", ascending=False).iterrows()}
    selected_label = st.selectbox("Choose a cluster to inspect", list(options.keys()))
    selected_cluster = options[selected_label]
    selected_summary = cluster_view[cluster_view["cluster_id"] == selected_cluster].iloc[0]
    selected_listings = clustered_table[clustered_table["cluster_id"] == selected_cluster].copy()

    st.markdown(
        f"""
        <div class="mini-note">
            <strong>{selected_label}</strong> is best read as a <strong>{selected_summary['cluster_label']}</strong> segment.
            It contains roughly <strong>{int(selected_summary['row_count']):,}</strong> listings, averages about <strong>${selected_summary['avg_price']:,.0f}</strong>,
            and is dominated by <strong>{selected_summary['dominant_type']}</strong> listings.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("Top terms:", ", ".join(selected_summary["top_terms"][:8]) if isinstance(selected_summary["top_terms"], list) else selected_summary["top_terms"])

    sample_examples = selected_listings.sort_values("list_price", ascending=False).loc[:, ["record_id", "state", "type", "list_price", "quality_score", "text"]].head(5).copy()
    sample_examples["list_price"] = sample_examples["list_price"].map(lambda x: f"${x:,.0f}")
    sample_examples["quality_score"] = sample_examples["quality_score"].map(lambda x: f"{x:,.1f}")
    sample_examples = sample_examples.rename(
        columns={
            "record_id": "Record ID",
            "state": "State",
            "type": "Property Type",
            "list_price": "List Price",
            "quality_score": "Quality Score",
            "text": "Example Listing Text",
        }
    )
    st.dataframe(sample_examples, width="stretch", hide_index=True)
    st.caption(
        "Interpretation: this is where the clusters become useful. If the top terms and sample listings feel consistent, the cluster can support market segmentation, search, and copy strategy."
    )


def build_nlp_influence_examples(df: pd.DataFrame, target: str) -> pd.DataFrame:
    required = [
        "record_id",
        "state",
        "type",
        "text",
        target,
        f"{target}_was_imputed",
        f"{target}_imputation_source",
        f"{target}_imputation_confidence",
        f"{target}_regex_candidate",
        f"{target}_structured_prediction",
        f"{target}_text_prediction",
    ]
    if any(col not in df.columns for col in required):
        return pd.DataFrame()

    working = df.loc[
        df[f"{target}_was_imputed"]
        & (df[f"{target}_imputation_source"] == "hybrid_model")
        & df[f"{target}_text_prediction"].notna()
        & df[f"{target}_structured_prediction"].notna()
        & df["text"].notna()
    , required].copy()
    if working.empty:
        return working

    working["prediction_gap"] = (
        working[f"{target}_text_prediction"] - working[f"{target}_structured_prediction"]
    ).abs()
    working = working[working[f"{target}_regex_candidate"].isna()]
    if working.empty:
        return working
    working = working[working[f"{target}_imputation_confidence"].fillna(0) >= 0.22]
    if working.empty:
        return working

    type_series = working["type"].fillna("").astype(str).str.lower()
    non_land_mask = ~type_series.eq("land")

    if target == "year_built":
        antique_terms = r"antique|historic|colonial|century|prewar|victorian|cape"
        working["text_signal"] = working["text"].str.contains(antique_terms, case=False, na=False)
        working = working[
            non_land_mask
            & working[target].between(1850, 2025)
            & working[f"{target}_structured_prediction"].between(1850, 2025)
            & working[f"{target}_text_prediction"].between(1850, 2025)
            & working["prediction_gap"].between(8, 45)
        ]
        working = working.sort_values(
            ["text_signal", f"{target}_imputation_confidence", "prediction_gap"],
            ascending=[False, False, False],
        )
    elif target == "beds":
        working = working[
            non_land_mask
            & working[target].between(1, 8)
            & working[f"{target}_structured_prediction"].between(1, 8)
            & working[f"{target}_text_prediction"].between(1, 8)
            & working["prediction_gap"].between(0.5, 2.5)
        ].sort_values([f"{target}_imputation_confidence", "prediction_gap"], ascending=[False, False])
    elif target == "baths":
        working = working[
            non_land_mask
            & working[target].between(0.5, 6)
            & working[f"{target}_structured_prediction"].between(0.5, 6)
            & working[f"{target}_text_prediction"].between(0.5, 6)
            & working["prediction_gap"].between(0.5, 1.5)
        ].sort_values([f"{target}_imputation_confidence", "prediction_gap"], ascending=[False, False])
    elif target == "garage":
        working = working[
            non_land_mask
            & working[target].between(0, 4)
            & working[f"{target}_structured_prediction"].between(0, 4)
            & working[f"{target}_text_prediction"].between(0, 4)
            & working["prediction_gap"].between(0.5, 2.0)
        ].sort_values([f"{target}_imputation_confidence", "prediction_gap"], ascending=[False, False])
    elif target == "stories":
        working = working[
            non_land_mask
            & working[target].between(1, 4)
            & working[f"{target}_structured_prediction"].between(1, 4)
            & working[f"{target}_text_prediction"].between(1, 4)
            & working["prediction_gap"].between(0.5, 2.0)
        ].sort_values([f"{target}_imputation_confidence", "prediction_gap"], ascending=[False, False])
    elif target == "sqft":
        working = working[
            non_land_mask
            & working[target].between(400, 8000)
            & working[f"{target}_structured_prediction"].between(400, 8000)
            & working[f"{target}_text_prediction"].between(400, 8000)
            & working["prediction_gap"].between(150, 1200)
        ].sort_values([f"{target}_imputation_confidence", "prediction_gap"], ascending=[False, False])
    else:
        working = working.sort_values([f"{target}_imputation_confidence", "prediction_gap"], ascending=[False, False])
    working = working[working["text"].astype(str).str.len() >= 80]
    return working.head(20)


def summarize_nlp_influence_example(row: pd.Series, target: str) -> str:
    target_label = TARGET_LABELS.get(target, target.replace("_", " ").title())
    structured_pred = row.get(f"{target}_structured_prediction")
    text_pred = row.get(f"{target}_text_prediction")
    final_value = row.get(target)
    confidence = row.get(f"{target}_imputation_confidence")
    return (
        f"This record was missing `{target_label}` and regex did not recover a direct value. "
        f"The structured model suggested `{structured_pred:,.2f}`, the text model suggested `{text_pred:,.2f}`, "
        f"and the final hybrid fill landed at `{final_value:,.2f}` with confidence `{confidence:.2f}`. "
        f"That is the clearest way to see the NLP layer contributing to one specific record."
    )


def get_explainer_patterns(target: str) -> list[re.Pattern[str]]:
    pattern_map = {
        "beds": [
            r"\b\d+\s*(?:bed|beds|bedroom|bedrooms)\b",
            r"\b(?:one|two|three|four|five|six|seven|eight)\s*(?:bed|bedroom)s?\b",
        ],
        "baths": [
            r"\b\d+(?:\.\d+)?\s*(?:bath|baths|bathroom|bathrooms)\b",
            r"\bhalf bath\b",
            r"\bfull bath\b",
        ],
        "sqft": [
            r"\b\d{3,5}\s*(?:sq\.?\s*ft\.?|sqft|square feet)\b",
            r"\b\d{3,5}\s*sf\b",
        ],
        "year_built": [
            r"\bbuilt in \d{4}\b",
            r"\bcirca \d{4}\b",
            r"\b(?:historic|antique|colonial|victorian|prewar|century-old|century)\b",
            r"\b\d{4}\b",
        ],
    }
    return [re.compile(pattern, flags=re.IGNORECASE) for pattern in pattern_map.get(target, [])]


def highlight_text_evidence(text: str, target: str) -> tuple[str, list[str]]:
    if not text:
        return "", []

    matches: list[tuple[int, int, str]] = []
    for pattern in get_explainer_patterns(target):
        for match in pattern.finditer(text):
            matches.append((match.start(), match.end(), match.group(0)))

    if not matches:
        return html.escape(text).replace("\n", "<br>"), []

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    merged: list[tuple[int, int, str]] = []
    clue_terms: list[str] = []
    last_end = -1
    for start, end, phrase in matches:
        if start < last_end:
            continue
        merged.append((start, end, phrase))
        clue_terms.append(phrase)
        last_end = end

    parts: list[str] = []
    cursor = 0
    for start, end, _phrase in merged:
        parts.append(html.escape(text[cursor:start]))
        parts.append(f'<span class="cue-highlight">{html.escape(text[start:end])}</span>')
        cursor = end
    parts.append(html.escape(text[cursor:]))

    return "".join(parts).replace("\n", "<br>"), list(dict.fromkeys(term.strip() for term in clue_terms if term.strip()))


def render_nlp_explainer(table: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-note">This explainer zooms into one missing record at a time and highlights clue phrases in the listing text. These highlights are explanation heuristics, not literal model attention weights.</div>',
        unsafe_allow_html=True,
    )
    explainer_targets = [
        target for target in ["year_built", "beds", "baths", "sqft"] if not build_nlp_influence_examples(table, target).empty
    ]
    if not explainer_targets:
        st.warning("No explainable NLP-assisted examples are available in the current table.")
        return

    explainer_target = st.selectbox(
        "Field to explain",
        explainer_targets,
        index=explainer_targets.index("year_built") if "year_built" in explainer_targets else 0,
        format_func=lambda x: TARGET_LABELS.get(x, x.replace("_", " ").title()),
        help="This focuses on fields where the listing text can plausibly contain direct or indirect clues about the missing value.",
        key="nlp_explainer_target",
    )

    examples = build_nlp_influence_examples(table, explainer_target)
    if examples.empty:
        st.warning("No explainable examples were available for the selected field.")
        return

    known_examples = load_known_value_examples_artifact()
    if not known_examples.empty:
        known_examples = known_examples[
            (known_examples["mode"] == "analysis_safe")
            & (known_examples["target"] == explainer_target)
        ].copy()
        known_examples = known_examples.sort_values(["prediction_error", "model_gap"], ascending=[True, False])
    if not known_examples.empty:
        st.subheader("1. Sanity-Check Example With A Known Value")
        st.caption(
            "This first example is a calibration view. The true value is known, so we can show a case where the text-supported hybrid guess lands reasonably close before moving to a genuinely missing record."
        )
        known_labels = known_examples.apply(
            lambda row: f"{row['record_id']} | {row['state']} | {str(row['type']).replace('_', ' ').title()}",
            axis=1,
        ).tolist()
        known_selected_label = st.selectbox(
            "Known-value record",
            known_labels,
            key="nlp_known_record",
        )
        known_row = known_examples.iloc[known_labels.index(known_selected_label)]
        known_highlighted_text, known_clue_terms = highlight_text_evidence(str(known_row.get("text", "")), explainer_target)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Actual known value", format_target_metric(known_row["actual_value"], explainer_target))
        with k2:
            st.metric(
                "Structured model guess",
                format_target_metric(known_row[f"{explainer_target}_structured_prediction"], explainer_target),
            )
        with k3:
            st.metric(
                "Text model guess",
                format_target_metric(known_row[f"{explainer_target}_text_prediction"], explainer_target),
            )
        with k4:
            st.metric("Hybrid guess", format_target_metric(known_row[f"{explainer_target}_hybrid_prediction"], explainer_target))

        st.caption(
            f"This known-value example helps anchor the explainer: the hybrid guess is off by **{known_row['prediction_error']:,.2f}**, so you can see what a reasonably close text-supported estimate looks like before switching to a truly missing record."
        )
        if known_clue_terms:
            known_clue_html = "".join(f'<span class="cue-chip">{html.escape(term)}</span>' for term in known_clue_terms[:10])
            st.markdown(known_clue_html, unsafe_allow_html=True)
        st.markdown(f'<div class="explain-text">{known_highlighted_text}</div>', unsafe_allow_html=True)
        known_term_contrib = explain_text_prediction_terms(explainer_target, str(known_row.get("text_clean", known_row.get("text", ""))))
        if not known_term_contrib.empty:
            st.markdown("**Top text-model terms pushing this guess**")
            term_view = known_term_contrib.copy()
            term_view["contribution"] = term_view["contribution"].map(lambda x: f"{x:+.3f}")
            term_view = term_view.rename(columns={"term": "Term", "contribution": "Contribution Toward Prediction"})
            st.dataframe(term_view, width="stretch", hide_index=True)
            st.caption(
                "These are TF-IDF term contributions from the linear text model. Positive contributions push the text-only prediction upward for the selected field."
            )
        else:
            st.caption(
                "This record does not expose especially interpretable text-term contributions, which is itself useful to know: sometimes the text-side prediction comes more from diffuse language patterns than one obvious phrase."
            )

        known_snapshot = pd.DataFrame(
            [
                {
                    "Record ID": known_row["record_id"],
                    "State": known_row["state"],
                    "Property Type": str(known_row["type"]).replace("_", " ").title(),
                    "Observed Field": TARGET_LABELS.get(explainer_target, explainer_target),
                    "Actual Value": known_row["actual_value"],
                    "Hybrid Guess": known_row[f"{explainer_target}_hybrid_prediction"],
                }
            ]
        )
        st.dataframe(known_snapshot, width="stretch", hide_index=True)

    st.subheader("2. Missing-Value Example")
    st.caption(
        "Now that the explainer has shown a known-value sanity check, this second example uses a record that was actually missing the field and needed the hybrid system to fill it."
    )
    example_labels = examples.apply(
        lambda row: f"{row['record_id']} | {row['state']} | {str(row['type']).replace('_', ' ').title()}",
        axis=1,
    ).tolist()
    selected_label = st.selectbox(
        "Missing record",
        example_labels,
        help="These records are filtered to missing-value cases where regex did not directly recover the answer and the text model contributed a meaningful prediction.",
        key="nlp_explainer_record",
    )
    row = examples.iloc[example_labels.index(selected_label)]

    highlighted_text, clue_terms = highlight_text_evidence(str(row.get("text", "")), explainer_target)

    x1, x2, x3, x4 = st.columns(4)
    with x1:
        st.metric(
            "Structured model guess",
            format_target_metric(row[f"{explainer_target}_structured_prediction"], explainer_target),
        )
    with x2:
        st.metric("Text model guess", format_target_metric(row[f"{explainer_target}_text_prediction"], explainer_target))
    with x3:
        st.metric("Final hybrid fill", format_target_metric(row[explainer_target], explainer_target))
    with x4:
        st.metric("Confidence", f"{row[f'{explainer_target}_imputation_confidence']:.2f}")

    st.caption(
        f"This record was missing **{TARGET_LABELS.get(explainer_target, explainer_target)}**. The highlighted phrases below are the kinds of textual clues that make the NLP side of the fill more understandable."
    )

    if clue_terms:
        clue_html = "".join(f'<span class="cue-chip">{html.escape(term)}</span>' for term in clue_terms[:10])
        st.markdown(clue_html, unsafe_allow_html=True)
    else:
        st.caption("No explicit clue phrase was matched by the explanation heuristics for this record.")

    st.markdown(f'<div class="explain-text">{highlighted_text}</div>', unsafe_allow_html=True)
    missing_term_contrib = explain_text_prediction_terms(explainer_target, str(row.get("text_clean", row.get("text", ""))))
    if not missing_term_contrib.empty:
        st.markdown("**Top text-model terms pushing this missing-value guess**")
        missing_term_view = missing_term_contrib.copy()
        missing_term_view["contribution"] = missing_term_view["contribution"].map(lambda x: f"{x:+.3f}")
        missing_term_view = missing_term_view.rename(columns={"term": "Term", "contribution": "Contribution Toward Prediction"})
        st.dataframe(missing_term_view, width="stretch", hide_index=True)
    else:
        st.caption(
            "No especially strong text-term contributions were available for this missing-value example."
        )

    explainer_snapshot = pd.DataFrame(
        [
            {
                "Record ID": row["record_id"],
                "State": row["state"],
                "Property Type": str(row["type"]).replace("_", " ").title(),
                "Missing Field": TARGET_LABELS.get(explainer_target, explainer_target),
                "Regex Candidate": row.get(f"{explainer_target}_regex_candidate"),
            }
        ]
    )
    st.dataframe(explainer_snapshot, width="stretch", hide_index=True)
    st.caption(
        "How to read this: the highlights are not a literal peek inside the model. They are an explanation aid showing which pieces of the description are most likely to support the text-side prediction."
    )


def render_project_info(
    analysis_df: pd.DataFrame,
    mode_summary_df: pd.DataFrame,
    price_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> None:
    st.markdown(
        '<div class="section-note">This first page is the project overview: what problem the portfolio piece tackles, what modeling ideas it demonstrates, and how the pieces fit together.</div>',
        unsafe_allow_html=True,
    )
    total_rows = len(analysis_df)
    states = analysis_df["state"].nunique()
    best_target = mode_summary_df.sort_values("tolerance_accuracy", ascending=False).iloc[0]
    best_price_model = price_df.sort_values("r2", ascending=False).iloc[0]
    richest_cluster = cluster_df.sort_values("avg_price", ascending=False).iloc[0]

    st.markdown(
        """
        <div class="hero">
            <span class="eyebrow">Portfolio Project</span>
            <h1>Real Estate Recovery And NLP Studio</h1>
            <p>This is one umbrella project with two connected modules: a hybrid missing-data recovery system for structured housing fields, and an NLP enrichment layer for listing-language analysis, clustering, and record-level support.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Combined listings",
            f"{total_rows:,}",
            help="Total number of rows in the working combined dataset. Useful for communicating project scale. It does not by itself tell you whether the rows are balanced, unique, or high quality.",
        )
    with m2:
        st.metric(
            "States covered",
            f"{states}",
            help="Number of U.S. states represented in the combined dataset. Good for showing geographic breadth, but it does not capture whether each state contributes similar volume or data quality.",
        )
    with m3:
        st.metric(
            "Best imputed field",
            best_target["label"],
            help="The target with the highest tolerance accuracy in masked holdout testing. This is a quick way to summarize the strongest imputation result, but it does not show uncertainty or how far ahead it is from the next-best field.",
        )
    with m4:
        st.metric(
            "Best pricing model R2",
            f"{best_price_model['r2']:.3f}",
            help="The highest R² among the current price models. R² is useful for summarizing explained variance, but it can hide whether errors are still large in dollar terms.",
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="info-card">
                <h3>What This Project Is</h3>
                <p>This portfolio project combines multiple state-level real-estate datasets, cleans and unifies them, then splits the work into two clear modules: structured-data recovery and NLP-based listing enrichment.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="info-card">
                <h3>Why It Matters</h3>
                <p>Real listing data is messy. Missing values, inconsistent text, and uneven schemas make downstream analysis weak. This project shows how to turn that into a usable modeling and decision-support dataset.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="info-card">
                <h3>What It Demonstrates</h3>
                <p>Data cleaning, leakage-aware imputation, masked holdout evaluation, price modeling, clustering, and NLP-based description analysis, organized as one coherent workflow rather than two unrelated experiments.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("What This Project Is Trying To Do")
    st.write(
        "The goal is to start with messy listing data, recover missing property attributes in a leakage-aware way, "
        "measure how reliable those imputations are, and then use listing text to enrich the repaired dataset with "
        "language-based insights. The pricing comparison is part of the evaluation story, but it is not the only test "
        "of whether the NLP layer was useful."
    )

    st.subheader("Project Structure")
    module_map = pd.DataFrame(
        [
            {
                "Module": "Module 1: Structured Recovery",
                "Main question": "Can I repair important missing housing attributes without creating leakage problems?",
                "What lives here": "Hybrid imputation, regex extraction, structured models, masked holdout evaluation, before-versus-after completeness gains.",
            },
            {
                "Module": "Module 2: NLP Enrichment",
                "Main question": "Once the dataset is repaired, what useful signal lives in the listing language?",
                "What lives here": "Text-based support for imputation, clustering, listing-quality analysis, amenity/copy patterns, and record-level explanation examples.",
            },
        ]
    )
    st.dataframe(module_map, width="stretch", hide_index=True)

    st.subheader("How I Attacked The Problem")
    attack_plan = pd.DataFrame(
        [
            {
                "Question I asked": "What am I actually starting with?",
                "How I answered it": "Profiled the cross-state datasets, compared schemas, and measured missingness and text quality.",
            },
            {
                "Question I asked": "Can I turn this into one usable dataset?",
                "How I answered it": "Standardized the schema, unified fields across states, and created a combined cleaned table.",
            },
            {
                "Question I asked": "How should I fill missing core attributes?",
                "How I answered it": "Built Module 1: a hybrid imputer using regex extraction, structured models, and text-based models.",
            },
            {
                "Question I asked": "Can I prove the imputer works?",
                "How I answered it": "Ran masked holdout tests so the model had to recover values we already knew.",
            },
            {
                "Question I asked": "What extra value does the text add after recovery?",
                "How I answered it": "Built Module 2: compared price models, clustered listing language, and analyzed copy patterns from higher-value homes.",
            },
        ]
    )
    st.dataframe(attack_plan, width="stretch", hide_index=True)

    st.subheader("Model Ideas Demonstrated")
    model_ideas = pd.DataFrame(
        [
            {
                "Component": "Module 1: Hybrid imputation system",
                "What it does": "Fills missing beds, baths, sqft, garage, stories, and year built using regex extraction plus structured and NLP models.",
                "Why it matters": "Shows nuanced data recovery instead of simple mean or median imputation.",
            },
            {
                "Component": "Module 1: Masked holdout evaluation",
                "What it does": "Hides known values and tests whether the imputer can recover them.",
                "Why it matters": "Makes the imputation story measurable and credible.",
            },
            {
                "Component": "Bridge evaluation: Price prediction comparison",
                "What it does": "Compares text-only, structured-only, and blended price models.",
                "Why it matters": "Shows whether listing text adds marginal pricing value beyond the repaired structured fields.",
            },
            {
                "Component": "Module 2: Listing-language clustering",
                "What it does": "Groups descriptions into themes like premium or distinctive listing styles.",
                "Why it matters": "Demonstrates unsupervised NLP on real marketplace text.",
            },
            {
                "Component": "Module 2: Amenity and copy pattern analysis",
                "What it does": "Extracts recurring signals from higher-value listings and turns them into description guidance.",
                "Why it matters": "Connects NLP output to a practical business use case.",
            },
            {
                "Component": "Module 2: Semantic-search foundation",
                "What it does": "Supports similar-listing search and future retrieval-style demos.",
                "Why it matters": "Shows the project can evolve from analysis into product-like NLP applications.",
            },
        ]
    )
    st.dataframe(model_ideas, width="stretch", hide_index=True)

    st.subheader("Current Headline Findings")
    findings = pd.DataFrame(
        [
            {"Finding": "Module 1 headline", "Current readout": f"{best_target['label']} is the strongest imputed field at {best_target['tolerance_accuracy']:.1%} tolerance accuracy"},
            {"Finding": "Bridge evaluation", "Current readout": f"{best_price_model['model']} is the strongest pricing setup with R2 of {best_price_model['r2']:.3f}"},
            {"Finding": "Module 2 headline", "Current readout": f"Cluster {int(richest_cluster['cluster_id'])} has the highest average price, showing language-based grouping is meaningful"},
        ]
    )
    st.dataframe(findings, width="stretch", hide_index=True)

    st.subheader("Suggested Reading Order")
    reading_order = pd.DataFrame(
        [
            {"Step": "1", "Where to go next": "Data Background + EDA", "Why it matters": "Shows the original mess and why simple row dropping was not a good solution."},
            {"Step": "2", "Where to go next": "Game Plan", "Why it matters": "Explains why the project was intentionally split into recovery and NLP-enrichment modules."},
            {"Step": "3", "Where to go next": "Results: Imputation", "Why it matters": "Shows Module 1, the hybrid recovery system, and the completeness gains it delivered."},
            {"Step": "4", "Where to go next": "Results: Pricing", "Why it matters": "Shows the bridge evaluation and makes clear that structured features still dominate price prediction."},
            {"Step": "5", "Where to go next": "Results: NLP", "Why it matters": "Shows Module 2, where text helps through clustering, quality analysis, and record-level support."},
            {"Step": "6", "Where to go next": "Conclusions", "Why it matters": "Wraps up what each module accomplished and where the clearest business value sits."},
        ]
    )
    st.dataframe(reading_order, width="stretch", hide_index=True)

    st.subheader("High-Level Workflow")
    workflow = pd.DataFrame(
        [
            {"Step": "1. Ingest", "What happens": "Download and combine Kaggle real-estate datasets from multiple states."},
            {"Step": "2. Clean", "What happens": "Standardize schema, normalize fields, and flag low-quality listing text."},
            {"Step": "3. Module 1: Recover", "What happens": "Recover missing fields using regex extraction plus structured and text models."},
            {"Step": "4. Evaluate", "What happens": "Hide known values and test whether the imputer can recover them accurately."},
            {"Step": "5. Bridge test", "What happens": "Compare text-only versus structured price models and surface key price drivers."},
            {"Step": "6. Module 2: Enrich", "What happens": "Cluster listing language and extract copy patterns from high-value homes."},
        ]
    )
    st.dataframe(workflow, width="stretch", hide_index=True)


def render_background_info(starting_df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-note">This page explains the source data, what the records look like, and why this was a meaningful cleanup and NLP project instead of a ready-made modeling dataset.</div>',
        unsafe_allow_html=True,
    )

    source_df = (
        starting_df.groupby(["state", "source_dataset"], dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values(["state", "row_count"], ascending=[True, False])
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Source family", "Kaggle multi-state listings")
    with m2:
        st.metric("Datasets represented", f"{source_df['source_dataset'].nunique()}")
    with m3:
        st.metric("Raw listing rows", f"{len(starting_df):,}")

    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.subheader("Where The Data Came From")
        st.write(
            "The project uses a related family of Kaggle real-estate datasets posted by the same uploader, "
            "covering multiple U.S. states. Each state arrived as its own CSV-style dataset with listing text "
            "plus structured attributes like price, beds, baths, square footage, garage, and property type."
        )
        source_display = source_df.rename(
            columns={
                "state": "State",
                "source_dataset": "Source Dataset Slug",
                "row_count": "Listing Rows",
            }
        )
        source_display["Listing Rows"] = source_display["Listing Rows"].map(lambda x: f"{int(x):,}")
        st.dataframe(source_display, width="stretch", hide_index=True)
    with c2:
        st.subheader("Starting Challenges")
        st.write(
            "The files were related, but they still needed cleanup. Key pricing fields were missing, "
            "listing text quality varied, and coverage was uneven across states, so the project had to solve "
            "a real recovery problem before any serious NLP or pricing work."
        )
        challenges = pd.DataFrame(
            [
                {"challenge": "Cross-state schema drift", "why_it_matters": "Some fields were missing or inconsistently populated across states."},
                {"challenge": "Missing property attributes", "why_it_matters": "Important pricing fields like beds, baths, sqft, garage, and year built were often blank."},
                {"challenge": "Messy listing text", "why_it_matters": "Descriptions varied widely in quality and included placeholders, boilerplate, and repeated patterns."},
                {"challenge": "Leakage risk", "why_it_matters": "If price is used carelessly during imputation, downstream price models can become misleading."},
            ]
        )
        challenge_display = challenges.rename(
            columns={
                "challenge": "Challenge Area",
                "why_it_matters": "Why It Matters For Modeling",
            }
        )
        st.dataframe(challenge_display, width="stretch", hide_index=True)

    st.subheader("Coverage By State")
    coverage_df = (
        starting_df.groupby("state", dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values("row_count", ascending=False)
    )
    coverage_df["abbr"] = coverage_df["state"].map(lambda s: STATE_INFO.get(s, {}).get("abbr", s))
    coverage_df["fips"] = coverage_df["state"].map(lambda s: STATE_INFO.get(s, {}).get("fips"))
    coverage_df["fips"] = coverage_df["fips"].map(lambda x: f"{int(x):02d}" if pd.notna(x) else None)
    coverage_df["color"] = coverage_df["state"].map(lambda s: STATE_INFO.get(s, {}).get("color", "#0f766e"))
    state_color_domain = [state for state in STATE_INFO if state in set(coverage_df["state"])]
    state_color_range = [STATE_INFO[state]["color"] for state in state_color_domain]
    c1, c2 = st.columns([0.95, 1.05])
    with c1:
        bar_chart = (
            alt.Chart(coverage_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("state:N", sort="-y", title="State", axis=alt.Axis(labelAngle=-20)),
                y=alt.Y("row_count:Q", title="Listing rows"),
                color=alt.Color(
                    "state:N",
                    scale=alt.Scale(domain=state_color_domain, range=state_color_range),
                    legend=None,
                ),
                tooltip=[alt.Tooltip("state:N", title="State"), alt.Tooltip("row_count:Q", title="Listing rows", format=",")],
            )
            .properties(height=320)
        )
        st.altair_chart(bar_chart, width="stretch")
    with c2:
        map_df = coverage_df.dropna(subset=["fips"]).copy()
        if not map_df.empty:
            states = alt.topo_feature(
                "https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json",
                "states",
            )
            base_map = (
                alt.Chart(states)
                .mark_geoshape(
                    fill="#f3f4f6",
                    stroke="#cbd5e1",
                    strokeWidth=0.9,
                )
                .project(type="albersUsa")
                .properties(height=320)
            )
            choropleth = (
                alt.Chart(states)
                .mark_geoshape(stroke="white", strokeWidth=1.25)
                .transform_lookup(
                    lookup="id",
                    from_=alt.LookupData(map_df, "fips", ["state", "abbr", "row_count", "color"]),
                )
                .transform_filter("isValid(datum.row_count)")
                .encode(
                    color=alt.Color(
                        "state:N",
                        scale=alt.Scale(domain=state_color_domain, range=state_color_range),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("state:N", title="State"),
                        alt.Tooltip("row_count:Q", title="Listing rows", format=","),
                    ],
                )
                .project(type="albersUsa")
                .properties(height=320)
            )
            labels_source = map_df.copy()
            labels_source["lat"] = labels_source["state"].map(lambda s: STATE_COORDS.get(s, {}).get("lat"))
            labels_source["lon"] = labels_source["state"].map(lambda s: STATE_COORDS.get(s, {}).get("lon"))
            labels = (
                alt.Chart(labels_source.dropna(subset=["lat", "lon"]))
                .mark_text(fontSize=11, fontWeight="bold", color="#102a43")
                .encode(
                    longitude="lon:Q",
                    latitude="lat:Q",
                    text="abbr:N",
                )
            )
            st.altair_chart(base_map + choropleth + labels, width="stretch")
        else:
            st.info("Map coordinates were not available for the current state set.")

    st.subheader("What A Record Looks Like")
    example_cols = ["state", "type", "list_price", "sqft", "beds", "baths", "text"]
    example_rows = (
        starting_df[example_cols]
        .dropna(subset=["text"])
        .sample(n=min(3, len(starting_df.dropna(subset=["text"]))), random_state=42)
        .reset_index(drop=True)
    )
    st.dataframe(example_rows, width="stretch", hide_index=True)

    st.subheader("Why This Was Messy To Start")
    st.write(
        "Even after the initial unification layer, the starting point still had the kinds of issues that make real data projects hard: "
        "missing values in core pricing fields, uneven text quality, placeholder-style descriptions, and state-to-state differences in coverage."
    )

    issue_summary = pd.DataFrame(
        [
            {"Starting issue": "Missing square footage", "Share of starting records": f"{starting_df['sqft'].isna().mean():.1%}"},
            {"Starting issue": "Missing year built", "Share of starting records": f"{starting_df['year_built'].isna().mean():.1%}"},
            {"Starting issue": "Placeholder text", "Share of starting records": f"{starting_df['is_placeholder_text'].mean():.1%}"},
            {"Starting issue": "Blank text", "Share of starting records": f"{starting_df['is_blank_text'].mean():.1%}"},
            {"Starting issue": "Duplicate text within state", "Share of starting records": f"{starting_df['has_duplicate_text_within_state'].mean():.1%}"},
        ]
    )
    st.dataframe(issue_summary, width="stretch", hide_index=True)


def render_audience_view(
    audience_view: str,
    mode: str,
    summary_df: pd.DataFrame,
    price_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
) -> None:
    title, description = summarize_audience(audience_view)
    st.markdown(
        f"""
        <div class="audience-banner">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if audience_view == "ops":
        best = summary_df.sort_values("tolerance_accuracy", ascending=False).iloc[0]
        weakest = summary_df.sort_values("tolerance_accuracy", ascending=True).iloc[0]
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Best target", best["label"], f"{best['tolerance_accuracy']:.1%} usable accuracy")
        with m2:
            st.metric("Weakest target", weakest["label"], f"{weakest['tolerance_accuracy']:.1%} usable accuracy")
        with m3:
            st.metric("Mode", "Analysis-safe" if mode == "analysis_safe" else "Price-model-safe")

        st.markdown(
            '<div class="section-note">For an operations team, the key question is which missing attributes can be filled with enough confidence to improve listing completeness without silently introducing junk.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            format_scorecard(
                summary_df[
                    [
                        "label",
                        "tolerance_accuracy",
                        "regex_coverage",
                        "missing_before",
                        "regex_imputed",
                        "hybrid_model_imputed",
                        "structured_weight",
                    ]
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        return

    if audience_view == "pricing":
        feature_view = feature_df.head(10).copy()
        if not feature_view.empty:
            feature_view["importance_mean"] = feature_view["importance_mean"].map(lambda x: f"{x:.3f}")
            feature_view["importance_std"] = feature_view["importance_std"].map(lambda x: f"{x:.3f}")
        best_model = price_df.sort_values("r2", ascending=False).iloc[0]
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Best pricing model", best_model["model"])
        with m2:
            st.metric("Best R²", f"{best_model['r2']:.3f}")
        with m3:
            st.metric("Top driver", feature_df.iloc[0]["feature"] if not feature_df.empty else "n/a")

        st.markdown(
            '<div class="section-note">For a buyer or pricing audience, the main story is which features matter most, how much price signal exists in the text, and why leakage-safe imputations matter before pricing analysis.</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([1, 1.1])
        with c1:
            st.bar_chart(price_df.set_index("model")[["r2"]])
        with c2:
            st.dataframe(feature_view, width="stretch", hide_index=True)
        return

    term_df, example_df = build_copywriter_insights(analysis_df)
    inputs = build_copy_recommendation_inputs(analysis_df)
    state_options = ["All"] + inputs["states"]
    type_options = ["All"] + inputs["types"]
    csel1, csel2 = st.columns(2)
    with csel1:
        state_choice = st.selectbox("Focus state", state_options, key="copy_state_choice")
    with csel2:
        type_choice = st.selectbox("Focus property type", type_options, key="copy_type_choice")

    rec_terms, top_amenities, recommendation, example_slice = build_description_recommendation(
        analysis_df,
        state_choice,
        type_choice,
    )
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("High-value terms surfaced", len(term_df))
    with m2:
        quality_share = analysis_df["is_quality_filtered_text"].mean()
        st.metric("Quality-filtered text share", f"{quality_share:.1%}")
    with m3:
        top_term = rec_terms.iloc[0]["term"] if not rec_terms.empty else "n/a"
        st.metric("Most lifted term", top_term)

    st.markdown(
        '<div class="section-note">This is the reverse-engineering view: instead of only modeling descriptions, it surfaces language patterns and amenities that show up more often in higher-value listings so teams can write stronger copy.</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([0.9, 1.1])
    with c1:
        if not rec_terms.empty:
            display_terms = rec_terms.copy()
            display_terms["high_value_share"] = display_terms["high_value_share"].map(lambda x: f"{x:.2%}")
            display_terms["rest_share"] = display_terms["rest_share"].map(lambda x: f"{x:.2%}")
            display_terms["lift"] = display_terms["lift"].map(lambda x: f"{x:.2f}x")
            st.dataframe(display_terms.head(10), width="stretch", hide_index=True)
        else:
            st.info("Not enough text rows were available to build copywriting term insights.")
    with c2:
        st.subheader("Recommended Description Ingredients")
        if top_amenities:
            st.write(", ".join(top_amenities))
        else:
            st.write("No clear amenity pattern was strong enough for this slice.")
        st.markdown(f'<div class="generated-copy">{recommendation}</div>', unsafe_allow_html=True)

    st.subheader("Example High-Value Listings For This Slice")
    st.dataframe(example_slice, width="stretch", hide_index=True)


inject_styles()
reports = load_reports()
tables = load_tables()
feature_importance_df = load_price_feature_importance()

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Portfolio Demo</div>
        <h1>Real Estate NLP Storyboard</h1>
        <p>
            A presentation-friendly dashboard for the cross-state real-estate project.
            It highlights what the hybrid imputer does well, where the current models still need work,
            and how the text layer contributes to pricing, clustering, and description strategy.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

required_reports = {"hybrid_imputer", "price_models", "hybrid_summary", "clusters"}
required_tables = {"combined_clean", "analysis_imputed", "price_safe_imputed", "clustered"}
if not required_reports.issubset(reports) or not required_tables.issubset(tables):
    st.error("Some processed reports or tables are missing. Rebuild the project artifacts before using the dashboard.")
    st.stop()

impute_df = pd.DataFrame(reports["hybrid_imputer"]["results"])
price_df = pd.DataFrame(reports["price_models"]["results"])
cluster_df = pd.DataFrame(reports["clusters"])
hybrid_summary = reports["hybrid_summary"]
combined_clean = tables["combined_clean"]
analysis_imputed = tables["analysis_imputed"]
price_safe_imputed = tables["price_safe_imputed"]
clustered_table = tables["clustered"]

default_mode = "analysis_safe"
default_target_focus = "beds"
default_audience_view = "ops"
mode_summary_df = build_target_summary(impute_df, hybrid_summary, default_mode)
active_table = analysis_imputed

project_tab, background_tab, eda_tab, gameplan_tab, results_tab, explainer_tab, cluster_tab, stats_tab, applications_tab, conclusions_tab, explorer_tab = st.tabs(
    ["Project Overview", "Data Background", "EDA", "Game Plan", "Results", "NLP Detail", "Cluster Analysis", "Statistical Analysis", "Applications", "Conclusions", "Appendix"]
)

with project_tab:
    render_project_info(analysis_imputed, mode_summary_df, price_df, cluster_df)

with background_tab:
    render_background_info(combined_clean)

with eda_tab:
    st.markdown(
        '<div class="section-note">The EDA phase answered two questions: how messy the unified dataset really was, and whether there was enough usable structure and text to justify a smarter recovery strategy.</div>',
        unsafe_allow_html=True,
    )
    missingness = pd.DataFrame(
        [
            {"Field": "Square footage", "Missing share": combined_clean["sqft"].isna().mean()},
            {"Field": "Year built", "Missing share": combined_clean["year_built"].isna().mean()},
            {"Field": "Beds", "Missing share": combined_clean["beds"].isna().mean()},
            {"Field": "Baths", "Missing share": combined_clean["baths"].isna().mean()},
            {"Field": "Garage", "Missing share": combined_clean["garage"].isna().mean()},
            {"Field": "Stories", "Missing share": combined_clean["stories"].isna().mean()},
        ]
    ).sort_values("Missing share", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Missingness In Core Pricing Fields")
        render_ranked_bar_chart(
            missingness,
            category_col="Field",
            value_col="Missing share",
            title="Missing share",
            color="#3e6f73",
            percent_axis=True,
        )
        st.caption(
            "Interpretation: square footage and year built are the clearest recovery targets because they are both pricing-relevant and missing often enough that dropping rows would waste a lot of data."
        )
    with c2:
        st.subheader("Text Quality Snapshot")
        text_quality = pd.DataFrame(
            [
                {"Issue": "Blank text", "Share": combined_clean["is_blank_text"].mean()},
                {"Issue": "Placeholder text", "Share": combined_clean["is_placeholder_text"].mean()},
                {"Issue": "Duplicate text within state", "Share": combined_clean["has_duplicate_text_within_state"].mean()},
            ]
        )
        render_ranked_bar_chart(
            text_quality,
            category_col="Issue",
            value_col="Share",
            title="Share",
            color="#8bb8c7",
            percent_axis=True,
            height=250,
        )
        st.caption(
            "Interpretation: text quality is imperfect, but the low blank and placeholder shares mean the text is still rich enough to justify NLP features and extraction rules."
        )

    st.subheader("State-Level Missingness Pattern")
    state_missing = (
        combined_clean.groupby("state", dropna=False)
        .agg(
            missing_sqft=("sqft", lambda s: s.isna().mean()),
            missing_year_built=("year_built", lambda s: s.isna().mean()),
            missing_beds=("beds", lambda s: s.isna().mean()),
            missing_baths=("baths", lambda s: s.isna().mean()),
            blank_text=("is_blank_text", "mean"),
            placeholder_text=("is_placeholder_text", "mean"),
        )
        .reset_index()
    )
    state_missing_view = state_missing.copy()
    state_missing_view = state_missing_view.rename(
        columns={
            "state": "State",
            "missing_sqft": "Missing Square Footage",
            "missing_year_built": "Missing Year Built",
            "missing_beds": "Missing Bedrooms",
            "missing_baths": "Missing Bathrooms",
            "blank_text": "Blank Description Text",
            "placeholder_text": "Placeholder Description Text",
        }
    )
    for col in state_missing_view.columns[1:]:
        state_missing_view[col] = state_missing_view[col].map(lambda x: f"{x:.1%}")
    st.dataframe(state_missing_view, width="stretch", hide_index=True)
    st.caption(
        "Interpretation: missingness is not uniform across states, which supports evaluating one unified pipeline carefully instead of assuming performance will look identical everywhere."
    )

    st.subheader("Text Coverage And Richness")
    text_stats = pd.DataFrame(
        [
            {"Metric": "Rows with non-blank text", "Value": f"{(1 - combined_clean['is_blank_text'].mean()):.1%}"},
            {"Metric": "Rows without placeholder text", "Value": f"{(1 - combined_clean['is_placeholder_text'].mean()):.1%}"},
            {"Metric": "Median text length", "Value": f"{combined_clean['text_length'].median():,.0f} characters"},
            {"Metric": "Rows with duplicate text within state", "Value": f"{combined_clean['has_duplicate_text_within_state'].mean():.1%}"},
        ]
    )
    text_stats = text_stats.rename(columns={"Metric": "Text Coverage Check", "Value": "Current Readout"})
    st.dataframe(text_stats, width="stretch", hide_index=True)
    st.caption(
        "Interpretation: the descriptions are noisy but broadly available, which is exactly the kind of setting where a text-assisted imputation strategy can add value."
    )

    st.subheader("Attribute Availability Before Imputation")
    attribute_ready = pd.DataFrame(
        [
            {
                "Check": "Rows with all four pricing-core fields present",
                "Share": (
                    combined_clean[["beds", "baths", "sqft", "year_built"]]
                    .notna()
                    .all(axis=1)
                    .mean()
                ),
            },
            {
                "Check": "Rows with at least three of four pricing-core fields present",
                "Share": (
                    combined_clean[["beds", "baths", "sqft", "year_built"]]
                    .notna()
                    .sum(axis=1)
                    .ge(3)
                    .mean()
                ),
            },
            {
                "Check": "Rows with non-blank text plus at least two core fields",
                "Share": (
                    ((~combined_clean["is_blank_text"]) &
                     (combined_clean[["beds", "baths", "sqft", "year_built"]].notna().sum(axis=1).ge(2)))
                    .mean()
                ),
            },
        ]
    )
    attribute_ready["Share"] = attribute_ready["Share"].map(lambda x: f"{x:.1%}")
    attribute_ready = attribute_ready.rename(columns={"Check": "Availability Check", "Share": "Share Of Rows"})
    st.dataframe(attribute_ready, width="stretch", hide_index=True)
    st.caption(
        "Interpretation: enough rows still contain partial structure plus usable text, so the pipeline has real signal to learn from even before imputation."
    )

    st.subheader("What The EDA Told Me")
    eda_findings = pd.DataFrame(
        [
            {"Finding": "There was substantial missingness in pricing-critical fields.", "Why it mattered": "That made simple row dropping too wasteful for a dataset this size."},
            {"Finding": "Descriptions were noisy, but not empty overall.", "Why it mattered": "That created room for NLP features and regex-style extraction."},
            {"Finding": "The same schema existed across states, but with uneven sparsity.", "Why it mattered": "That justified building one unified recovery pipeline instead of separate state models."},
            {"Finding": "Both structured fields and text were available on most rows.", "Why it mattered": "That made a hybrid structured-plus-text imputation strategy realistic."},
        ]
    )
    eda_findings = eda_findings.rename(columns={"Finding": "EDA Finding", "Why it mattered": "Why It Mattered"})
    st.dataframe(eda_findings, width="stretch", hide_index=True)

    st.subheader("Decision Implications For The Modeling Plan")
    decision_bridge = pd.DataFrame(
        [
            {"EDA signal": "High missingness in sqft and year_built", "Decision supported": "Imputation was worth attempting because those fields are too important to discard."},
            {"EDA signal": "Most rows still have usable text", "Decision supported": "A text-based model could contribute information when numeric fields are sparse."},
            {"EDA signal": "Some rows still retain multiple structured fields", "Decision supported": "Structured regression models had enough signal to support hybrid recovery."},
            {"EDA signal": "State-specific sparsity patterns differ", "Decision supported": "A unified but carefully evaluated pipeline was more defensible than assuming every state behaves the same."},
            {"EDA signal": "Duplicate and placeholder text exist but are limited", "Decision supported": "Text required quality controls, but not wholesale removal from the pipeline."},
        ]
    )
    decision_bridge = decision_bridge.rename(columns={"EDA signal": "EDA Signal", "Decision supported": "Decision It Supported"})
    st.dataframe(decision_bridge, width="stretch", hide_index=True)

with gameplan_tab:
    st.markdown(
        '<div class="section-note">The game plan came directly from the EDA: missing values were too important to ignore, and the dataset had both numeric structure and listing text, so a hybrid recovery approach was worth trying.</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Why A Hybrid Imputation Plan Made Sense")
    st.write(
        "Some fields could be recovered directly from the description when the listing explicitly mentioned them, while other rows needed a predictive model. Because the dataset includes both structured housing attributes and narrative text, the cleanest plan was to combine rule extraction, structured modeling, and NLP signals."
    )
    plan_table = pd.DataFrame(
        [
            {"Layer": "Rule extraction", "Role in the pipeline": "Recover values directly stated in the listing text.", "Why it was included": "Direct evidence is better than prediction when the description clearly gives the answer."},
            {"Layer": "Structured model", "Role in the pipeline": "Predict missing attributes from the numeric and categorical fields already present.", "Why it was included": "Home attributes often move together in predictable ways."},
            {"Layer": "Text model", "Role in the pipeline": "Use listing language to estimate missing values when descriptions imply size, style, or home characteristics.", "Why it was included": "The text contains real property clues even when fields are blank."},
            {"Layer": "Leakage-safe mode", "Role in the pipeline": "Build a separate version that avoids using list price when imputing predictors for downstream pricing work.", "Why it was included": "This keeps the pricing story honest."},
        ]
    )
    plan_table = plan_table.rename(
        columns={
            "Layer": "Pipeline Layer",
            "Role in the pipeline": "Role In The Pipeline",
            "Why it was included": "Why It Was Included",
        }
    )
    st.dataframe(plan_table, width="stretch", hide_index=True)
    st.subheader("How The EDA Drove The Plan")
    gameplan_links = pd.DataFrame(
        [
            {"Observation from EDA": "Square footage and year built were the most missing major fields.", "Game-plan consequence": "Prioritize them as explicit imputation targets and report their evaluation separately."},
            {"Observation from EDA": "Listing text remained widely available even when some attributes were blank.", "Game-plan consequence": "Use NLP and regex extraction alongside structured predictors rather than relying on one data source."},
            {"Observation from EDA": "Not all uses of the final dataset are the same.", "Game-plan consequence": "Maintain both analysis-safe and price-model-safe outputs to avoid leakage in pricing workflows."},
            {"Observation from EDA": "Text quality varies across listings.", "Game-plan consequence": "Keep quality flags in the pipeline and interpret weak text differently from rich text."},
        ]
    )
    gameplan_links = gameplan_links.rename(
        columns={
            "Observation from EDA": "Observation From EDA",
            "Game-plan consequence": "Game-Plan Consequence",
        }
    )
    st.dataframe(gameplan_links, width="stretch", hide_index=True)

with results_tab:
    result_mode = st.radio(
        "Evaluation mode",
        options=["analysis_safe", "price_model_safe"],
        format_func=lambda x: "Analysis-safe" if x == "analysis_safe" else "Price-model-safe",
        horizontal=True,
        help="Analysis-safe can use list price when filling other fields, which is useful for cleaning and completeness. Price-model-safe avoids using list price for those imputations so downstream pricing analysis does not leak target information.",
    )
    result_summary_df = build_target_summary(impute_df, hybrid_summary, result_mode)
    result_table = analysis_imputed if result_mode == "analysis_safe" else price_safe_imputed
    result_target = default_target_focus
    comparison_mode = "price_model_safe" if result_mode == "analysis_safe" else "analysis_safe"
    comparison_summary_df = build_target_summary(impute_df, hybrid_summary, comparison_mode)
    mode_compare_view = result_summary_df.merge(
        comparison_summary_df,
        on="target",
        suffixes=("_current", "_other"),
    )
    mode_compare_view["tolerance_delta"] = (
        mode_compare_view["tolerance_accuracy_current"] - mode_compare_view["tolerance_accuracy_other"]
    )
    mode_compare_view["mae_delta"] = mode_compare_view["mae_current"] - mode_compare_view["mae_other"]
    biggest_accuracy_shift = mode_compare_view.iloc[
        mode_compare_view["tolerance_delta"].abs().sort_values(ascending=False).index[0]
    ]
    current_mode_label = "Analysis-safe" if result_mode == "analysis_safe" else "Price-model-safe"
    comparison_mode_label = "Price-model-safe" if result_mode == "analysis_safe" else "Analysis-safe"
    avg_tolerance_current = result_summary_df["tolerance_accuracy"].mean()
    avg_tolerance_other = comparison_summary_df["tolerance_accuracy"].mean()
    st.markdown(
        '<div class="section-note">This is the payoff section: what improved, how well the hybrid imputer performed, what pricing signal remained after cleanup, and what the NLP layer found in the descriptions.</div>',
        unsafe_allow_html=True,
    )
    st.info(
        f"You are viewing **{current_mode_label}** results. The biggest visible difference versus **{comparison_mode_label}** is on **{biggest_accuracy_shift['label_current']}**, where tolerance accuracy changes by **{biggest_accuracy_shift['tolerance_delta']:+.1%}**. Most other metrics stay fairly close, so the switch is meaningful but subtle."
    )
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric(
            "Current avg tolerance accuracy",
            f"{avg_tolerance_current:.1%}",
            help="Average tolerance accuracy across the tracked imputation targets in the currently selected mode. This gives a quick sense of overall imputation reliability, but it hides target-by-target variation.",
        )
    with mc2:
        st.metric(
            f"{comparison_mode_label} avg tolerance accuracy",
            f"{avg_tolerance_other:.1%}",
            delta=f"{(avg_tolerance_current - avg_tolerance_other):+.1%}",
            help="This shows the average tolerance accuracy in the comparison mode. The delta helps you see whether switching modes materially changed overall imputation performance.",
        )
    with mc3:
        st.metric(
            "Largest mode shift",
            biggest_accuracy_shift["label_current"],
            f"{biggest_accuracy_shift['tolerance_delta']:+.1%}",
            help="This highlights the target whose tolerance accuracy changed the most when switching modes. It is the fastest way to see why the mode selector matters.",
        )
    mode_delta_display = mode_compare_view[["label_current", "tolerance_delta", "mae_delta"]].copy()
    mode_delta_display["tolerance_delta"] = mode_delta_display["tolerance_delta"].map(lambda x: f"{x:+.1%}")
    mode_delta_display["mae_delta"] = mode_delta_display["mae_delta"].map(lambda x: f"{x:+,.2f}")
    mode_delta_display = mode_delta_display.rename(
        columns={
            "label_current": "Target",
            "tolerance_delta": f"Tolerance Accuracy Change Vs {comparison_mode_label}",
            "mae_delta": f"MAE Change Vs {comparison_mode_label}",
        }
    )
    st.dataframe(mode_delta_display, width="stretch", hide_index=True)
    show_story_cards(result_summary_df, price_df, cluster_df, default_audience_view)

    st.markdown("### 1. Imputation Results")
    st.markdown(
        '<div class="mini-note">Start here when presenting. This block shows whether the hybrid recovery layer actually solved the missing-data problem and which targets it handles best.</div>',
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Best field accuracy",
            f"{result_summary_df['tolerance_accuracy'].max():.1%}",
            help="Highest tolerance accuracy among the imputed targets. Useful for a quick headline, but it can overstate overall performance if weaker targets are hidden behind one strong result.",
        )
    with m2:
        st.metric(
            "Missing values tackled",
            f"{int(result_summary_df['missing_before'].sum()):,}",
            help="Total missing target values encountered across the evaluated imputation targets. Useful for showing scale of the recovery problem, but it does not say whether every fill was equally reliable.",
        )
    with m3:
        st.metric(
            "Direct regex recoveries",
            f"{int(result_summary_df['regex_imputed'].sum()):,}",
            help="Number of missing values recovered directly from the listing text using extraction rules. This shows transparent wins from direct evidence, but it does not capture the harder cases that required modeling.",
        )
    with m4:
        st.metric(
            "Hybrid model fills",
            f"{int(result_summary_df['hybrid_model_imputed'].sum()):,}",
            help="Number of missing values filled by the predictive hybrid layer after rule extraction. This shows the scale of actual modeling work, but not all modeled fills are equally accurate across targets.",
        )

    st.subheader("Imputation Scorecard")
    st.dataframe(
        format_scorecard(
            result_summary_df[
                ["label", "tolerance_accuracy", "mae", "regex_coverage", "missing_before", "regex_imputed", "hybrid_model_imputed", "structured_weight", "validation_mae"]
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    strongest_target = result_summary_df.sort_values("tolerance_accuracy", ascending=False).iloc[0]
    weakest_target = result_summary_df.sort_values("tolerance_accuracy", ascending=True).iloc[0]
    st.caption(
        f"Readout: `{strongest_target['label']}` is the strongest target in `{result_mode}` mode, while `{weakest_target['label']}` remains the weakest. That pattern suggests the pipeline is best at discrete home attributes and less stable on harder continuous or weakly signaled fields."
    )

    st.subheader("Masked Holdout Evaluation")
    c1, c2 = st.columns([1, 1.2])
    mode_df = impute_df[impute_df["mode"] == result_mode].copy()
    mode_df["label"] = mode_df["target"].map(TARGET_LABELS)
    with c1:
        metric = st.selectbox("Imputation metric", ["tolerance_accuracy", "regex_coverage", "mae", "rmse"])
        render_ranked_bar_chart(
            mode_df[["label", metric]].rename(columns={"label": "Target"}),
            category_col="Target",
            value_col=metric,
            title=metric.replace("_", " ").title(),
            color="#5b8e7d",
            percent_axis=metric in {"tolerance_accuracy", "regex_coverage"},
        )
        st.caption(
            "How to read this: higher is better for tolerance accuracy and regex coverage, while lower is better for MAE and RMSE. This chart helps separate usable targets from riskier ones."
        )
    with c2:
        review_df = mode_df[["label", "rows_evaluated", "regex_coverage", "mae", "rmse", "tolerance_accuracy"]].copy()
        review_df["regex_coverage"] = review_df["regex_coverage"].map(lambda x: f"{x:.1%}")
        review_df["tolerance_accuracy"] = review_df["tolerance_accuracy"].map(lambda x: f"{x:.1%}")
        review_df["mae"] = review_df["mae"].map(lambda x: f"{x:,.2f}")
        review_df["rmse"] = review_df["rmse"].map(lambda x: f"{x:,.2f}")
        review_df = review_df.rename(
            columns={
                "label": "Target",
                "rows_evaluated": "Rows Evaluated",
                "regex_coverage": "Regex Coverage",
                "mae": "MAE",
                "rmse": "RMSE",
                "tolerance_accuracy": "Tolerance Accuracy",
            }
        )
        st.dataframe(review_df, width="stretch", hide_index=True)

    st.subheader("Before Vs After Missingness")
    before_after = pd.DataFrame(
        [
            {"Field": "Square footage", "Starting": combined_clean["sqft"].isna().mean(), "Current": result_table["sqft"].isna().mean()},
            {"Field": "Year built", "Starting": combined_clean["year_built"].isna().mean(), "Current": result_table["year_built"].isna().mean()},
            {"Field": "Beds", "Starting": combined_clean["beds"].isna().mean(), "Current": result_table["beds"].isna().mean()},
            {"Field": "Baths", "Starting": combined_clean["baths"].isna().mean(), "Current": result_table["baths"].isna().mean()},
        ]
    )
    before_after["Recovered"] = before_after["Starting"] - before_after["Current"]
    recovered_share = (
        before_after["Recovered"].sum() / before_after["Starting"].sum()
        if before_after["Starting"].sum() > 0
        else 0.0
    )
    fully_recovered_fields = int((before_after["Current"] == 0).sum())
    b1, b2 = st.columns([1.2, 1])
    with b1:
        recovery_chart = before_after.sort_values("Recovered", ascending=False).copy()
        recovery_chart["Recovery status"] = np.where(
            recovery_chart["Current"] == 0,
            "Fully recovered",
            "Partially recovered",
        )
        render_grouped_metric_chart(
            recovery_chart.rename(columns={"Recovered": "Missing Share Removed"}),
            category_col="Field",
            value_col="Missing Share Removed",
            color_col="Recovery status",
            title="Missing share removed",
            color_range=["#2f7d6b", "#c97a40"],
            percent_axis=True,
            height=320,
        )
    with b2:
        st.metric(
            "Recovered missing-share mass",
            f"{recovered_share:.1%}",
            help="This summarizes how much of the combined missing-share burden across the highlighted fields was removed by the recovery pipeline. It is useful as a concise completeness headline, but it does not reflect whether all recovered values were equally accurate.",
        )
        st.metric(
            "Fields fully recovered",
            f"{fully_recovered_fields}/{len(before_after)}",
            help="This shows how many of the highlighted fields were driven all the way to zero remaining missingness in the post-imputation table.",
        )
        improvement_view = before_after[["Field", "Recovered"]].copy()
        improvement_view["Recovered"] = improvement_view["Recovered"].map(lambda x: f"{x:.1%}")
        improvement_view = improvement_view.rename(columns={"Recovered": "Missing Share Removed"})
        st.dataframe(improvement_view, width="stretch", hide_index=True)
    before_after_view = before_after.copy()
    before_after_view["Starting"] = before_after_view["Starting"].map(lambda x: f"{x:.1%}")
    before_after_view["Current"] = before_after_view["Current"].map(lambda x: f"{x:.1%}")
    before_after_view["Recovered"] = before_after_view["Recovered"].map(lambda x: f"{x:.1%}")
    before_after_view = before_after_view.rename(
        columns={
            "Field": "Field",
            "Starting": "Starting Missing Share",
            "Current": "Current Missing Share",
            "Recovered": "Missing Share Removed",
        }
    )
    st.dataframe(before_after_view, width="stretch", hide_index=True)
    st.caption(
        "Interpretation: this view now emphasizes the improvement rather than the zero-height after bars. The chart shows how much missingness was removed in each business-critical field, while the summary metrics show that the highlighted fields were fully recovered in the final table."
    )

    st.markdown("---")
    st.markdown("### 2. Pricing Results")
    st.markdown(
        '<div class="mini-note">Once the missing-data problem is under control, the next question is how much pricing signal the cleaned dataset actually contains and whether text adds useful predictive power.</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Pricing Signal")
    best_r2_row = price_df.sort_values("r2", ascending=False).iloc[0]
    p1, p2, p3 = st.columns(3)
    with p1:
        st.metric(
            "Best model",
            best_r2_row["model"],
            help="The price model with the strongest current validation result. Useful for summarizing the winner, but it does not explain whether the gap is statistically meaningful or practically large.",
        )
    with p2:
        st.metric(
            "Best R²",
            f"{best_r2_row['r2']:.3f}",
            help="R² measures the share of price variation explained by the model. It is intuitive and widely used, but it can still look decent when dollar errors remain large.",
        )
    with p3:
        st.metric(
            "Blend weight",
            reports["price_models"].get("best_structured_weight_for_blend", "n/a"),
            help="The structured-model weight used in the blended pricing setup. This is useful for showing whether the final blend leaned more on structured or text signals, but it does not by itself prove the blend improved performance.",
        )
    price_view = price_df.copy()
    price_view["mae"] = price_view["mae"].map(lambda x: f"{x:,.2f}")
    price_view["rmse"] = price_view["rmse"].map(lambda x: f"{x:,.2f}")
    price_view["r2"] = price_view["r2"].map(lambda x: f"{x:.3f}")
    price_view = price_view.rename(columns={"model": "Model", "mae": "MAE", "rmse": "RMSE", "r2": "R²"})
    st.dataframe(price_view, width="stretch", hide_index=True)
    st.caption(
        "Interpretation: the current results show that text carries real pricing signal, but structured home attributes are still doing most of the predictive work in this pipeline."
    )
    st.info(
        "What this does and does not mean: in the current price-prediction setup, adding text features did not beat the structured-only baseline. That is an important result, but it does not mean the NLP layer is useless. The text model can still help with imputation, clustering, listing-quality analysis, and record-level reasoning when structured fields are missing."
    )
    if not feature_importance_df.empty:
        st.subheader("Price Driver Snapshot")
        feature_view = feature_importance_df.head(12).copy()
        feature_view = feature_view.rename(
            columns={
                "feature": "Feature",
                "importance_mean": "Average Importance",
                "importance_std": "Importance Std Dev",
            }
        )
        st.dataframe(feature_view, width="stretch", hide_index=True)
        render_ranked_bar_chart(
            feature_view.rename(columns={"Average Importance": "Importance"}),
            category_col="Feature",
            value_col="Importance",
            title="Permutation importance",
            color="#6ea5b3",
            percent_axis=False,
            height=360,
        )
        st.caption(
            "Interpretation: the importance chart shows which variables move the pricing model most when perturbed. It is useful for model explanation, but importance is not the same as causal effect."
        )

    st.markdown("---")
    st.markdown("### 3. NLP Results")
    st.markdown(
        '<div class="mini-note">This block shows what the text layer is contributing beyond imputation: recurring language patterns, cluster structure, and examples of how the model reasons about specific filled values.</div>',
        unsafe_allow_html=True,
    )
    st.subheader("NLP Insights")
    cluster_view = cluster_df.copy()
    if "avg_price" in cluster_view:
        cluster_view = cluster_view.sort_values("avg_price", ascending=False)
    if "cluster_label" in cluster_view.columns:
        cluster_view["Cluster"] = cluster_view.apply(cluster_display_label, axis=1)
    if "top_terms" in cluster_view.columns:
        cluster_view["top_terms"] = cluster_view["top_terms"].apply(lambda x: ", ".join(x[:5]) if isinstance(x, list) else str(x))
    if "avg_price" in cluster_view:
        cluster_view["avg_price"] = cluster_view["avg_price"].map(lambda x: f"${x:,.0f}")
    if "avg_quality" in cluster_view:
        cluster_view["avg_quality"] = cluster_view["avg_quality"].map(lambda x: f"{x:,.1f}")
    display_cols = [col for col in ["Cluster", "row_count", "avg_price", "avg_quality", "dominant_type", "top_terms"] if col in cluster_view.columns]
    cluster_view = cluster_view[display_cols].rename(
        columns={
            "row_count": "Listings In Cluster",
            "avg_price": "Average Price",
            "avg_quality": "Average Quality Score",
            "dominant_type": "Dominant Type",
            "top_terms": "Top Terms",
        }
    )
    st.dataframe(cluster_view, width="stretch", hide_index=True)
    st.caption(
        "Interpretation: if clusters separate on both average price and average quality score, that is evidence that the language layer is finding meaningful structure rather than random text groupings."
    )

    st.subheader("NLP-Assisted Imputation Example")
    st.caption(
        "This section answers the practical question: can we point to a real listing where the text model helped recover a missing value? The example below intentionally focuses on rows where regex did not directly recover a value, so the comparison is really between the structured prediction and the text-aware hybrid fill."
    )
    preferred_example_targets = ["year_built", "beds", "baths"]
    available_example_targets = [
        target for target in preferred_example_targets if not build_nlp_influence_examples(result_table, target).empty
    ]
    if not available_example_targets:
        st.warning("No presentation-worthy NLP-assisted examples were available in the current mode.")
        available_example_targets = ["year_built"]
    if len(available_example_targets) == 1:
        nlp_example_target = available_example_targets[0]
        st.markdown(
            f"**Example field:** {TARGET_LABELS.get(nlp_example_target, nlp_example_target.replace('_', ' ').title())}"
        )
    else:
        nlp_example_target = st.selectbox(
            "Example field",
            available_example_targets,
            index=available_example_targets.index("year_built") if "year_built" in available_example_targets else 0,
            format_func=lambda x: TARGET_LABELS.get(x, x.replace("_", " ").title()),
            help="Choose a field to inspect at the record level. This list is intentionally limited to the fields that currently have believable, presentation-worthy NLP-assisted examples.",
            key="nlp_example_target",
        )
    nlp_examples = build_nlp_influence_examples(result_table, nlp_example_target)
    if nlp_examples.empty:
        st.warning("No strong NLP-assisted examples were available for this field in the current mode.")
    else:
        example_labels = nlp_examples.apply(
            lambda row: f"{row['record_id']} | {row['state']} | {str(row['type']).replace('_', ' ').title()}",
            axis=1,
        ).tolist()
        selected_example_label = st.selectbox(
            "Example record",
            example_labels,
            help="These examples are prefiltered to rows where the hybrid model filled a missing value, the text model contributed a numeric prediction, and regex did not already extract a direct value from the listing.",
            key="nlp_example_record",
        )
        selected_example = nlp_examples.iloc[example_labels.index(selected_example_label)]
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            st.metric(
                "Structured prediction",
                f"{selected_example[f'{nlp_example_target}_structured_prediction']:,.2f}",
                help="This is the prediction coming from the structured feature model alone. It is useful because it reflects what the numeric and categorical inputs suggest even without the text.",
            )
        with e2:
            st.metric(
                "Text prediction",
                f"{selected_example[f'{nlp_example_target}_text_prediction']:,.2f}",
                help="This is the prediction coming from the NLP model using listing description text. It is useful when the written listing contains clues that are not fully represented in the structured columns.",
            )
        with e3:
            st.metric(
                "Final hybrid fill",
                f"{selected_example[nlp_example_target]:,.2f}",
                help="This is the final imputed value after blending the structured and text signals. It shows how the hybrid system translates multiple partial signals into one operational fill.",
            )
        with e4:
            st.metric(
                "Confidence",
                f"{selected_example[f'{nlp_example_target}_imputation_confidence']:.2f}",
                help="This is the model's confidence indicator for the final fill. It is useful for prioritization, but it is still not a guarantee that the imputed value is correct.",
            )
        st.markdown(summarize_nlp_influence_example(selected_example, nlp_example_target))
        with st.expander("Show listing description and record details"):
            record_snapshot = pd.DataFrame(
                [
                    {
                        "Record ID": selected_example["record_id"],
                        "State": selected_example["state"],
                        "Property Type": str(selected_example["type"]).replace("_", " ").title(),
                        "Filled Field": TARGET_LABELS.get(nlp_example_target, nlp_example_target),
                        "Final Filled Value": selected_example[nlp_example_target],
                    }
                ]
            )
            st.dataframe(record_snapshot, width="stretch", hide_index=True)
            st.markdown("**Listing description**")
            st.write(selected_example["text"])
        st.caption(
            "Interpretation: this is the clearest place to see NLP helping. Even if text does not improve the final price model much, it can still move a missing-value fill toward a more plausible value for a specific record when the listing language contains useful clues."
        )

    st.subheader("Provenance Example")
    result_target = st.selectbox(
        "Target field for provenance explorer",
        TARGET_ORDER,
        index=TARGET_ORDER.index(default_target_focus),
        format_func=lambda x: TARGET_LABELS[x],
        help="This selector only affects the provenance explorer below. Use it to inspect how the system filled one specific field, such as Bedrooms or Year Built, rather than expecting it to change the whole Results page.",
        key="results_provenance_target",
    )
    source_filter = st.selectbox("Imputation source", ["all", "regex", "hybrid_model"])
    confidence_floor = st.slider(
        "Minimum confidence",
        min_value=0.00,
        max_value=1.00,
        value=0.80,
        step=0.01,
        help="This provenance explorer is meant for presentation and inspection, so it starts in the high-confidence range by default.",
    )
    mask = result_table[f"{result_target}_was_imputed"]
    if source_filter != "all":
        mask &= result_table[f"{result_target}_imputation_source"] == source_filter
    mask &= result_table[f"{result_target}_imputation_confidence"].fillna(0) >= confidence_floor
    cols = [
        "record_id", "state", "type", result_target,
        f"{result_target}_imputation_source", f"{result_target}_imputation_confidence",
        f"{result_target}_regex_candidate", f"{result_target}_structured_prediction",
        f"{result_target}_text_prediction", "text",
    ]
    provenance_view = safe_table(result_table.loc[mask], cols, 80).rename(
        columns={
            "record_id": "Record ID",
            "state": "State",
            "type": "Property Type",
            result_target: "Final Filled Value",
            f"{result_target}_imputation_source": "Imputation Source",
            f"{result_target}_imputation_confidence": "Imputation Confidence",
            f"{result_target}_regex_candidate": "Regex Candidate",
            f"{result_target}_structured_prediction": "Structured Model Prediction",
            f"{result_target}_text_prediction": "Text Model Prediction",
            "text": "Listing Description",
        }
    )
    st.dataframe(provenance_view, width="stretch", hide_index=True)
    st.caption(
        "Interpretation: these provenance rows are useful because they make the imputation process inspectable. The upside is transparency; the downside is that a few examples never replace aggregate validation."
    )

    st.markdown("---")
    st.markdown("### 4. Bottom-Line Readout")
    bottom_line = pd.DataFrame(
        [
            {"Takeaway": "The hybrid imputer is strongest on discrete home attributes.", "What that means": "Bedrooms, bathrooms, garage, and stories are the most reliable recovery targets right now."},
            {"Takeaway": "Square footage and year built remain the hardest fields.", "What that means": "Those targets still need more cautious interpretation and likely more feature work."},
            {"Takeaway": "Structured pricing features still outperform text-only pricing.", "What that means": "Text is useful, but it is not yet the dominant pricing signal in this pipeline."},
            {"Takeaway": "The text layer still adds business value.", "What that means": "Clustering, extraction, and copy analysis make the project broader than a pure imputation exercise."},
        ]
    )
    bottom_line = bottom_line.rename(columns={"Takeaway": "Bottom-Line Takeaway", "What that means": "What It Means"})
    st.dataframe(bottom_line, width="stretch", hide_index=True)

with explainer_tab:
    render_nlp_explainer(analysis_imputed)

with cluster_tab:
    render_cluster_analysis(cluster_df, clustered_table)

with stats_tab:
    st.markdown(
        '<div class="section-note">This is the technical evaluation page. It goes beyond the headline scorecards and focuses on how the models were assessed, what each metric means, and where the strongest or weakest statistical evidence sits.</div>',
        unsafe_allow_html=True,
    )
    stats_mode = st.radio(
        "Imputation evaluation mode",
        options=["analysis_safe", "price_model_safe"],
        format_func=lambda x: "Analysis-safe" if x == "analysis_safe" else "Price-model-safe",
        horizontal=True,
    )
    stats_df = impute_df[impute_df["mode"] == stats_mode].copy()
    stats_df["label"] = stats_df["target"].map(TARGET_LABELS)
    stats_df["tolerance_error_rate"] = 1 - stats_df["tolerance_accuracy"]
    stats_df["rmse_to_mae_ratio"] = stats_df["rmse"] / stats_df["mae"].replace(0, pd.NA)
    stats_df["tolerance_se"] = (
        (stats_df["tolerance_accuracy"] * (1 - stats_df["tolerance_accuracy"]) / stats_df["rows_evaluated"]).pow(0.5)
    )
    stats_df["tolerance_ci_low"] = (stats_df["tolerance_accuracy"] - 1.96 * stats_df["tolerance_se"]).clip(lower=0)
    stats_df["tolerance_ci_high"] = (stats_df["tolerance_accuracy"] + 1.96 * stats_df["tolerance_se"]).clip(upper=1)
    stats_df["regex_se"] = (
        (stats_df["regex_coverage"] * (1 - stats_df["regex_coverage"]) / stats_df["rows_evaluated"]).pow(0.5)
    )
    stats_df["regex_ci_low"] = (stats_df["regex_coverage"] - 1.96 * stats_df["regex_se"]).clip(lower=0)
    stats_df["regex_ci_high"] = (stats_df["regex_coverage"] + 1.96 * stats_df["regex_se"]).clip(upper=1)
    price_boot = bootstrap_price_model_metrics()

    st.subheader("Evaluation Design")
    eval_design = pd.DataFrame(
        [
            {"Concept": "Masked holdout evaluation", "What it means here": "Known values were hidden, then the hybrid imputer had to recover them.", "Why it matters": "This simulates real missing-data recovery instead of grading on the training data."},
            {"Concept": "Tolerance accuracy", "What it means here": "Share of predictions that landed close enough to count as practically usable.", "Why it matters": "More informative than strict exact-match accuracy for regression-style imputation."},
            {"Concept": "MAE", "What it means here": "Average absolute prediction error.", "Why it matters": "Shows the typical miss size in natural units like bathrooms, square feet, or years."},
            {"Concept": "RMSE", "What it means here": "Root mean squared error, which penalizes large misses more heavily than MAE.", "Why it matters": "Useful for spotting whether a target has a heavy tail of bad misses."},
            {"Concept": "Regex coverage", "What it means here": "Share of holdout rows where direct text extraction supplied a candidate.", "Why it matters": "Separates easy direct recoveries from cases that required actual modeling."},
        ]
    )
    eval_design = eval_design.rename(
        columns={
            "Concept": "Metric Or Design Choice",
            "What it means here": "What It Means Here",
            "Why it matters": "Why It Matters",
        }
    )
    st.dataframe(eval_design, width="stretch", hide_index=True)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Per-Target Evaluation Table")
        stats_view = stats_df[["label", "rows_evaluated", "tolerance_accuracy", "tolerance_error_rate", "regex_coverage", "mae", "rmse", "rmse_to_mae_ratio"]].copy()
        stats_view["tolerance_accuracy"] = stats_view["tolerance_accuracy"].map(lambda x: f"{x:.1%}")
        stats_view["tolerance_error_rate"] = stats_view["tolerance_error_rate"].map(lambda x: f"{x:.1%}")
        stats_view["regex_coverage"] = stats_view["regex_coverage"].map(lambda x: f"{x:.1%}")
        stats_view["mae"] = stats_view["mae"].map(lambda x: f"{x:,.2f}")
        stats_view["rmse"] = stats_view["rmse"].map(lambda x: f"{x:,.2f}")
        stats_view["rmse_to_mae_ratio"] = stats_view["rmse_to_mae_ratio"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        stats_view = stats_view.rename(
            columns={
                "label": "Target",
                "rows_evaluated": "Rows Evaluated",
                "tolerance_accuracy": "Tolerance Accuracy",
                "tolerance_error_rate": "Tolerance Error Rate",
                "regex_coverage": "Regex Coverage",
                "mae": "MAE",
                "rmse": "RMSE",
                "rmse_to_mae_ratio": "RMSE / MAE Ratio",
            }
        )
        st.dataframe(stats_view, width="stretch", hide_index=True)
    with c2:
        st.subheader("Metric Rankings")
        ranking_metric = st.selectbox("Ranking metric", ["tolerance_accuracy", "regex_coverage", "mae", "rmse", "rmse_to_mae_ratio"])
        rank_view = stats_df[["label", ranking_metric]].sort_values(ranking_metric, ascending=ranking_metric not in {"tolerance_accuracy", "regex_coverage"})
        render_ranked_bar_chart(
            rank_view.rename(columns={"label": "Target"}),
            category_col="Target",
            value_col=ranking_metric,
            title=ranking_metric.replace("_", " ").title(),
            color="#2f5d62",
            percent_axis=ranking_metric in {"tolerance_accuracy", "regex_coverage"},
            height=280,
        )
        st.caption(
            "Interpretation: the ranking view is useful for ordering targets by one evaluation lens at a time, but it should always be read alongside the other metrics because no single metric captures the whole error profile."
        )

    st.subheader("Approximate Confidence Intervals")
    st.caption("For the imputation classification-style rates, these are normal-approximation 95% confidence intervals based on the masked holdout sample size for each target.")
    ci_view = stats_df[["label", "rows_evaluated", "tolerance_accuracy", "tolerance_ci_low", "tolerance_ci_high", "regex_coverage", "regex_ci_low", "regex_ci_high"]].copy()
    ci_view["tolerance_accuracy"] = ci_view["tolerance_accuracy"].map(lambda x: f"{x:.1%}")
    ci_view["tolerance_ci_low"] = ci_view["tolerance_ci_low"].map(lambda x: f"{x:.1%}")
    ci_view["tolerance_ci_high"] = ci_view["tolerance_ci_high"].map(lambda x: f"{x:.1%}")
    ci_view["regex_coverage"] = ci_view["regex_coverage"].map(lambda x: f"{x:.1%}")
    ci_view["regex_ci_low"] = ci_view["regex_ci_low"].map(lambda x: f"{x:.1%}")
    ci_view["regex_ci_high"] = ci_view["regex_ci_high"].map(lambda x: f"{x:.1%}")
    ci_view = ci_view.rename(
        columns={
            "label": "Target",
            "rows_evaluated": "Rows Evaluated",
            "tolerance_accuracy": "Tolerance Accuracy",
            "tolerance_ci_low": "Tolerance 95% CI Low",
            "tolerance_ci_high": "Tolerance 95% CI High",
            "regex_coverage": "Regex Coverage",
            "regex_ci_low": "Regex 95% CI Low",
            "regex_ci_high": "Regex 95% CI High",
        }
    )
    st.dataframe(ci_view, width="stretch", hide_index=True)
    st.caption(
        "Interpretation: tighter confidence intervals mean the estimated rates are more stable under the masked holdout sample size. Wider intervals mean more caution is warranted even if the point estimate looks strong."
    )

    st.subheader("Mode Comparison")
    mode_compare = impute_df.pivot(index="target", columns="mode", values=["tolerance_accuracy", "mae", "rmse"]).reset_index()
    mode_compare.columns = [
        "target",
        "analysis_tolerance_accuracy",
        "price_safe_tolerance_accuracy",
        "analysis_mae",
        "price_safe_mae",
        "analysis_rmse",
        "price_safe_rmse",
    ]
    mode_compare["label"] = mode_compare["target"].map(TARGET_LABELS)
    mode_compare["tolerance_accuracy_delta"] = mode_compare["analysis_tolerance_accuracy"] - mode_compare["price_safe_tolerance_accuracy"]
    mode_compare["mae_delta"] = mode_compare["analysis_mae"] - mode_compare["price_safe_mae"]
    mode_compare["rmse_delta"] = mode_compare["analysis_rmse"] - mode_compare["price_safe_rmse"]
    compare_view = mode_compare[["label", "tolerance_accuracy_delta", "mae_delta", "rmse_delta"]].copy()
    compare_view["tolerance_accuracy_delta"] = compare_view["tolerance_accuracy_delta"].map(lambda x: f"{x:+.1%}")
    compare_view["mae_delta"] = compare_view["mae_delta"].map(lambda x: f"{x:+,.2f}")
    compare_view["rmse_delta"] = compare_view["rmse_delta"].map(lambda x: f"{x:+,.2f}")
    compare_view = compare_view.rename(
        columns={
            "label": "Target",
            "tolerance_accuracy_delta": "Tolerance Accuracy Delta",
            "mae_delta": "MAE Delta",
            "rmse_delta": "RMSE Delta",
        }
    )
    st.dataframe(compare_view, width="stretch", hide_index=True)
    st.caption("Positive tolerance deltas mean analysis-safe performed better. Negative MAE/RMSE deltas mean analysis-safe had lower error.")
    mode_compare_chart = mode_compare.melt(
        id_vars=["label"],
        value_vars=["analysis_tolerance_accuracy", "price_safe_tolerance_accuracy"],
        var_name="mode_label",
        value_name="value",
    )
    mode_compare_chart["mode_label"] = mode_compare_chart["mode_label"].map(
        {
            "analysis_tolerance_accuracy": "Analysis-safe",
            "price_safe_tolerance_accuracy": "Price-model-safe",
        }
    )
    render_grouped_metric_chart(
        mode_compare_chart.rename(columns={"label": "Target"}),
        category_col="Target",
        value_col="value",
        color_col="mode_label",
        title="Tolerance accuracy",
        color_range=["#3e6f73", "#8bb8c7"],
        percent_axis=True,
        height=300,
    )
    st.caption(
        "Interpretation: if the gap between analysis-safe and price-model-safe is small, the leakage-safe version is retaining most of the value. Larger gaps suggest price was materially helping the recovery task."
    )

    st.subheader("Price Model Comparison In Statistical Terms")
    price_stats = price_df.copy()
    baseline_text = float(price_stats.loc[price_stats["model"] == "text_only", "r2"].iloc[0])
    baseline_struct = float(price_stats.loc[price_stats["model"] == "structured_only", "r2"].iloc[0])
    price_stats["r2_vs_text_lift"] = price_stats["r2"] - baseline_text
    price_stats["r2_vs_structured_lift"] = price_stats["r2"] - baseline_struct
    price_stats["mae"] = price_stats["mae"].map(lambda x: f"{x:,.2f}")
    price_stats["rmse"] = price_stats["rmse"].map(lambda x: f"{x:,.2f}")
    price_stats["r2"] = price_stats["r2"].map(lambda x: f"{x:.3f}")
    price_stats["r2_vs_text_lift"] = price_stats["r2_vs_text_lift"].map(lambda x: f"{x:+.3f}")
    price_stats["r2_vs_structured_lift"] = price_stats["r2_vs_structured_lift"].map(lambda x: f"{x:+.3f}")
    price_stats = price_stats.rename(
        columns={
            "model": "Model",
            "mae": "MAE",
            "rmse": "RMSE",
            "r2": "R²",
            "r2_vs_text_lift": "R² Lift Vs Text-Only",
            "r2_vs_structured_lift": "R² Lift Vs Structured-Only",
        }
    )
    st.dataframe(price_stats, width="stretch", hide_index=True)
    st.caption(
        "Interpretation: the lift columns help you see whether a model is actually adding information beyond the baselines. This is more useful than only reporting the winning R² in isolation."
    )

    if not price_boot.empty:
        st.subheader("Bootstrapped Price-Model Uncertainty")
        st.caption("These intervals come from bootstrap resampling of the held-out prediction rows, so they show how stable the price metrics look under repeated re-sampling of the evaluation set.")
        boot_view = price_boot.copy()
        for metric in ["mae", "rmse"]:
            mask_metric = boot_view["metric"] == metric
            boot_view.loc[mask_metric, ["mean", "std", "ci_low", "ci_high"]] = (
                boot_view.loc[mask_metric, ["mean", "std", "ci_low", "ci_high"]].map(lambda x: float(x))
            )
        boot_view["mean"] = boot_view.apply(lambda r: f"{r['mean']:,.2f}" if r["metric"] in {"mae", "rmse"} else f"{r['mean']:.3f}", axis=1)
        boot_view["std"] = boot_view.apply(lambda r: f"{r['std']:,.2f}" if r["metric"] in {"mae", "rmse"} else f"{r['std']:.3f}", axis=1)
        boot_view["ci_low"] = boot_view.apply(lambda r: f"{r['ci_low']:,.2f}" if r["metric"] in {"mae", "rmse"} else f"{r['ci_low']:.3f}", axis=1)
        boot_view["ci_high"] = boot_view.apply(lambda r: f"{r['ci_high']:,.2f}" if r["metric"] in {"mae", "rmse"} else f"{r['ci_high']:.3f}", axis=1)
        boot_view = boot_view.rename(
            columns={
                "model": "Model",
                "metric": "Metric",
                "mean": "Bootstrap Mean",
                "std": "Bootstrap Std",
                "ci_low": "95% CI Low",
                "ci_high": "95% CI High",
                "n_boot": "Bootstrap Samples",
            }
        )
        st.dataframe(boot_view, width="stretch", hide_index=True)
        st.caption(
            "Interpretation: the bootstrap intervals show how much the evaluation metric moves when the held-out rows are resampled. Narrower intervals suggest a more stable estimate; overlapping intervals mean claims of improvement should stay modest."
        )

    st.subheader("Interpretation Notes")
    interpretation = pd.DataFrame(
        [
            {"Pattern": "High tolerance accuracy with moderate regex coverage", "Technical read": "The hybrid model is adding value beyond direct extraction."},
            {"Pattern": "High RMSE-to-MAE ratio", "Technical read": "A target likely has a subset of large misses, even if average error looks manageable."},
            {"Pattern": "Minimal lift from text-plus-structured over structured-only", "Technical read": "The current pricing pipeline is mostly driven by structured variables, with text adding limited marginal gain."},
            {"Pattern": "Large gap between analysis-safe and price-model-safe", "Technical read": "Price may be helping imputations materially, which is useful operationally but risky for downstream pricing if leakage is not controlled."},
        ]
    )
    interpretation = interpretation.rename(columns={"Pattern": "Observed Pattern", "Technical read": "Technical Read"})
    st.dataframe(interpretation, width="stretch", hide_index=True)
    st.caption("For a future pass, the next rigor upgrade would be repeated resampling or bootstrap intervals for MAE, RMSE, and price-model R² instead of only point estimates.")

with applications_tab:
    st.markdown(
        '<div class="section-note">This section translates the technical work into practical use cases for different audiences, from listing operations to pricing teams to copy strategy.</div>',
        unsafe_allow_html=True,
    )
    audience_view = st.selectbox(
        "Choose audience",
        options=list(AUDIENCE_VIEWS.keys()),
        format_func=lambda x: AUDIENCE_VIEWS[x],
        index=list(AUDIENCE_VIEWS.keys()).index(default_audience_view),
    )
    audience_mode = st.radio(
        "Application mode",
        options=["analysis_safe", "price_model_safe"],
        format_func=lambda x: "Analysis-safe" if x == "analysis_safe" else "Price-model-safe",
        horizontal=True,
        help="Analysis-safe is better for operational completeness because price can help fill related fields. Price-model-safe is better when the output will support downstream price prediction and you want to avoid leakage.",
    )
    audience_summary_df = build_target_summary(impute_df, hybrid_summary, audience_mode)
    render_audience_view(
        audience_view,
        audience_mode,
        audience_summary_df,
        price_df,
        feature_importance_df,
        analysis_imputed,
    )

with conclusions_tab:
    st.markdown(
        '<div class="section-note">This final page summarizes what the project accomplished, what the models proved, and what should be treated as the clearest strengths versus the remaining caveats.</div>',
        unsafe_allow_html=True,
    )
    st.subheader("What The Project Accomplished")
    best_target_row = result_summary_df.sort_values("tolerance_accuracy", ascending=False).iloc[0]
    weakest_target_row = result_summary_df.sort_values("tolerance_accuracy", ascending=True).iloc[0]
    best_price_row = price_df.sort_values("r2", ascending=False).iloc[0]
    conclusion_cards = st.columns(4)
    with conclusion_cards[0]:
        st.metric(
            "Best recovered field",
            best_target_row["label"],
            f"{best_target_row['tolerance_accuracy']:.1%} tolerance accuracy",
            help="This highlights the strongest-performing imputation target in masked holdout evaluation.",
        )
    with conclusion_cards[1]:
        st.metric(
            "Hardest recovered field",
            weakest_target_row["label"],
            f"{weakest_target_row['tolerance_accuracy']:.1%} tolerance accuracy",
            help="This highlights the weakest target so the final story stays honest about what still needs work.",
        )
    with conclusion_cards[2]:
        st.metric(
            "Best pricing model",
            str(best_price_row["model"]).replace("_", " ").title(),
            f"R² = {best_price_row['r2']:.3f}",
            help="This summarizes the strongest current pricing result from the model comparison.",
        )
    with conclusion_cards[3]:
        richest_cluster = cluster_df.sort_values("avg_price", ascending=False).iloc[0]
        st.metric(
            "Highest-price cluster",
            f"Cluster {int(richest_cluster['cluster_id'])}",
            f"${richest_cluster['avg_price']:,.0f} average price",
            help="This summarizes the highest-priced language cluster discovered by the NLP clustering step.",
        )

    accomplishments = pd.DataFrame(
        [
            {
                "Model Or Layer": "Hybrid imputer",
                "What it accomplished": "Recovered missing values in pricing-critical fields instead of dropping large portions of the dataset.",
                "Strongest evidence": f"{best_target_row['label']} reached about {best_target_row['tolerance_accuracy']:.1%} tolerance accuracy in masked holdout testing.",
            },
            {
                "Model Or Layer": "Leakage-safe recovery design",
                "What it accomplished": "Separated operational data repair from downstream price-modeling use cases.",
                "Strongest evidence": "Both analysis-safe and price-model-safe views are evaluated directly rather than assuming the same pipeline fits every purpose.",
            },
            {
                "Model Or Layer": "Pricing models",
                "What it accomplished": "Showed that the cleaned dataset carries meaningful price signal and established a structured baseline.",
                "Strongest evidence": f"The best current pricing result is {str(best_price_row['model']).replace('_', ' ')} with R² = {best_price_row['r2']:.3f}.",
            },
            {
                "Model Or Layer": "NLP layer",
                "What it accomplished": "Added value through text-based recovery clues, clustering, and listing-language interpretation even when text did not beat the structured pricing baseline.",
                "Strongest evidence": "The Results tab now shows record-level NLP-assisted imputation examples and cluster separation by average price and quality score.",
            },
        ]
    )
    accomplishments = accomplishments.rename(
        columns={
            "Model Or Layer": "Model Or Layer",
            "What it accomplished": "What It Accomplished",
            "Strongest evidence": "Strongest Evidence",
        }
    )
    st.dataframe(accomplishments, width="stretch", hide_index=True)

    st.subheader("Bottom-Line Conclusions")
    conclusions = pd.DataFrame(
        [
            {
                "Conclusion": "The project turned a messy multi-state listing corpus into a modeling-ready dataset.",
                "Why it matters": "That alone is a meaningful data-engineering and data-quality achievement because the starting fields were too incomplete for naive analysis.",
            },
            {
                "Conclusion": "Hybrid imputation worked best on discrete home attributes such as bedrooms, bathrooms, garage, and stories.",
                "Why it matters": "Those are now the most defensible fields to highlight when presenting the recovery pipeline.",
            },
            {
                "Conclusion": "Square footage and year built remain the fields that need the most caution.",
                "Why it matters": "The project shows maturity by making those limits explicit instead of overselling every imputed value equally.",
            },
            {
                "Conclusion": "Text did not improve the current price model over the structured baseline, but it still added value elsewhere.",
                "Why it matters": "That is an honest and nuanced finding: NLP helped more with recovery, grouping, and language analysis than with marginal price lift in the current setup.",
            },
            {
                "Conclusion": "The overall portfolio story is stronger because it demonstrates cleaning, evaluation, leakage awareness, NLP, and business interpretation in one workflow.",
                "Why it matters": "It reads like a full end-to-end applied machine learning project rather than a single isolated model notebook.",
            },
        ]
    )
    st.dataframe(conclusions, width="stretch", hide_index=True)
    st.caption(
        "If you were presenting this live, the shortest honest summary would be: the project successfully repaired major missing-data problems, established a defensible pricing baseline, and showed that the text layer adds the most value through imputation support and listing-language structure rather than pure price lift."
    )

with explorer_tab:
    appendix_mode = st.radio(
        "Appendix mode",
        options=["analysis_safe", "price_model_safe"],
        format_func=lambda x: "Analysis-safe" if x == "analysis_safe" else "Price-model-safe",
        horizontal=True,
    )
    active_table = analysis_imputed if appendix_mode == "analysis_safe" else price_safe_imputed
    st.subheader("Record Explorer")
    st.caption("This appendix-style page is for drilling into example records after the main project story has already been told.")
    state_options = ["All"] + sorted(active_table["state"].dropna().unique().tolist())
    state = st.selectbox("State filter", state_options)
    quality_only = st.checkbox("Only quality-filtered text", value=True)
    imputed_only = st.checkbox("Only rows with any imputed target", value=False)
    limit = st.slider("Rows to show", min_value=20, max_value=250, value=60, step=10)

    subset = active_table.copy()
    if state != "All":
        subset = subset[subset["state"] == state]
    if quality_only:
        subset = subset[subset["is_quality_filtered_text"]]
    if imputed_only:
        any_imputed = pd.Series(False, index=subset.index)
        for target in TARGET_ORDER:
            any_imputed |= subset[f"{target}_was_imputed"].fillna(False)
        subset = subset[any_imputed]

    cols = [
        "record_id",
        "state",
        "type",
        "list_price",
        "beds",
        "baths",
        "garage",
        "stories",
        "sqft",
        "year_built",
        "text",
    ]
    st.dataframe(safe_table(subset, cols, limit), width="stretch", hide_index=True)

