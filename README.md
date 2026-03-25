# NLP Real Estate

Portfolio project built around one umbrella problem: turn messy multi-state real-estate listings into a usable modeling dataset, then show what the listing text adds after recovery.

## Links

- GitHub repo: https://github.com/rpetty9/real_estate_nlp_regression_hybrid_model
- Live app: add Streamlit deployment URL here after publish

The repo is best understood as two connected modules:

1. `Module 1: Hybrid missing-data recovery`
   Recover pricing-critical housing attributes with regex extraction, structured models, and text-assisted imputation.
2. `Module 2: NLP enrichment`
   Use listing language for clustering, quality scoring, copy-pattern analysis, and record-level support after the structured dataset has been repaired.

The price-model comparison sits between those two modules as a bridge evaluation. It tests whether text adds marginal lift for price prediction, but it is not the only measure of whether the NLP layer was useful.

## Portfolio roadmap

This repo is being shaped around a recovery-first workflow plus several portfolio-friendly NLP tracks:

1. `Hybrid missing-data imputation with regex + NLP + regression`
2. `Imputation evaluation with masked holdout tests`
3. `Listing quality scorer`
4. `Amenity extraction + interactive analysis`
5. `Topic modeling or clustering of listing language`
6. `Semantic search / similar listings system`
7. `Price prediction from text vs text + structured features`

Recommended build order:

1. Download and unify the multi-state datasets.
2. Run EDA to understand schema, text quality, and duplicates.
3. Build the hybrid imputation layer that combines direct text extraction, structured models, and NLP-only predictions.
4. Evaluate those imputations by hiding known values and testing whether the system can recover them.
5. Build an NLP base table with cleaned text, quality labels, and amenity flags.
6. Use that base table for clustering, search, and modeling experiments.
7. Compare text-only, structured-only, and combined price models as a bridge evaluation.
8. Add a polished notebook or app layer for portfolio presentation.

## Current dataset family

These Kaggle datasets currently appear to belong to the same `Kanchana1990` series:

- `kanchana1990/wisconsin-real-estate-data-2026`
- `kanchana1990/connecticut-real-estate-data-2026`
- `kanchana1990/real-estate-data-oregon-2026`
- `kanchana1990/new-york-real-estate-data-2026`
- `kanchana1990/new-hampshire-real-estate-data-2026`
- `kanchana1990/indiana-real-estate-data-2026`
- `kanchana1990/kentucky-real-estate-data-2026`
- `kanchana1990/colorado-real-estate-market-2026`

## Setup

1. Make sure your Kaggle token is available at `%USERPROFILE%\.kaggle\kaggle.json`.
2. From this folder, install dependencies with `python -m pip install -r requirements.txt`.
3. Optionally copy `.env.example` to `.env` and edit the dataset list.

## Typical commands

- `python scripts/download_datasets.py`
- `python scripts/download_datasets.py --dataset kanchana1990/wisconsin-real-estate-data-2026`
- `python scripts/inspect_downloads.py --data-dir data/raw`
- `python scripts/clean_unify_datasets.py`
- `python scripts/build_nlp_feature_table.py`
- `python scripts/build_price_modeling_table.py`
- `python scripts/build_hybrid_imputation_table.py`
- `python scripts/evaluate_hybrid_imputer.py`
- `python scripts/compare_price_models.py`
- `python scripts/build_listing_clusters.py`
- `python scripts/similar_listing_search.py --record-id 123`
- `streamlit run app.py`

## Notes

- Each dataset is downloaded into its own subfolder under `data/raw`.
- The downloader uses an explicit dataset catalog so we can keep a stable multi-state collection even if the Kaggle search results change.
- The cleaned combined dataset lives in `data/processed/combined_real_estate_clean.*`.
- The NLP feature base table lives in `data/processed/real_estate_nlp_base.*`.
- The price-ready table with targeted imputations lives in `data/processed/real_estate_price_modeling_table.*`.
- The hybrid-imputed analysis-safe and leakage-safe tables live in `data/processed/real_estate_hybrid_imputed_*.*`.
