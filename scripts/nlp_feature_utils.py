import re
from typing import Iterable

import pandas as pd

PLACEHOLDER_TEXTS = {
    "",
    "no description provided.",
    "shortsale.",
}

AMENITY_PATTERNS: dict[str, str] = {
    "has_fireplace": r"\bfireplace\b",
    "has_granite": r"\bgranite\b",
    "has_stainless_steel": r"\bstainless steel\b",
    "has_hardwood": r"\bhardwood\b",
    "has_quartz": r"\bquartz\b",
    "has_walk_in_closet": r"\bwalk[- ]in closet|\bwalk in closet",
    "has_finished_basement": r"\bfinished basement|\bwalk-out basement|\bwalkout basement",
    "has_lake_access": r"\blake access|\blakefront|\blake view|\bwaterfront",
    "has_pool": r"\bpool\b|\bin-ground pool|\binground pool",
    "has_garage": r"\bgarage\b",
    "has_deck": r"\bdeck\b",
    "has_patio": r"\bpatio\b",
    "has_fenced_yard": r"\bfenced yard|\bfully fenced|\bfenced in yard",
    "has_acreage": r"\bacres?\b|\bacreage\b",
    "has_new_construction": r"\bnew construction|\bbrand-new|\bbrand new|\bto be built|\bnew build",
    "has_investor_angle": r"\binvestor\b|\binvestment opportunity|\bflip\b|\bflipper\b",
    "has_fixer_upper_angle": r"\bfixer\b|\bneeds work\b|\bhandyman\b|\btlc\b|\bas is\b",
    "has_hoa": r"\bhoa\b|\bhomeowners association\b",
    "has_views": r"\bviews?\b|\bmountain view|\bsweeping view|\bscenic view",
    "has_updated_kitchen": r"\bupdated kitchen|\bremodeled kitchen|\brenovated kitchen",
    "has_open_floor_plan": r"\bopen floor plan|\bopen-concept|\bopen concept",
    "has_primary_suite": r"\bprimary suite|\bowner'?s suite|\bensuite|\ben suite",
}

QUALITY_LABELS = ["placeholder", "weak", "usable", "rich"]


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def count_pattern_hits(text: str, patterns: Iterable[str]) -> int:
    return sum(bool(re.search(pattern, text)) for pattern in patterns)


def score_listing_quality(text: str, is_duplicate_text_within_state: bool) -> tuple[int, str, dict[str, int | bool]]:
    clean_text = normalize_text(text)
    lower_text = clean_text.lower()

    word_count = len(re.findall(r"\b\w+\b", clean_text))
    sentence_count = len(re.findall(r"[.!?]+", clean_text))
    digit_count = sum(ch.isdigit() for ch in clean_text)
    amenity_hit_count = count_pattern_hits(lower_text, AMENITY_PATTERNS.values())
    unique_word_count = len(set(re.findall(r"\b[a-zA-Z]{3,}\b", lower_text)))

    is_placeholder = lower_text in PLACEHOLDER_TEXTS
    is_blank = lower_text == ""

    if is_blank or is_placeholder:
        return 0, "placeholder", {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "digit_count": digit_count,
            "amenity_hit_count": amenity_hit_count,
            "unique_word_count": unique_word_count,
            "is_placeholder": is_placeholder,
            "is_blank": is_blank,
        }

    score = 40
    score += min(word_count, 220) * 0.18
    score += min(sentence_count, 8) * 2.0
    score += min(amenity_hit_count, 8) * 3.0
    score += min(unique_word_count, 80) * 0.18
    score += min(digit_count, 20) * 0.25

    if word_count < 12:
        score -= 28
    elif word_count < 25:
        score -= 15
    elif word_count > 120:
        score += 6

    if sentence_count <= 1:
        score -= 8

    if is_duplicate_text_within_state:
        score -= 14

    score = max(0, min(100, round(score)))

    if score < 20:
        label = "weak"
    elif score < 50:
        label = "usable"
    else:
        label = "rich"

    return score, label, {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "digit_count": digit_count,
        "amenity_hit_count": amenity_hit_count,
        "unique_word_count": unique_word_count,
        "is_placeholder": is_placeholder,
        "is_blank": is_blank,
    }


def extract_amenity_flags(text_series: pd.Series) -> pd.DataFrame:
    cleaned = text_series.fillna("").astype(str).str.lower()
    features = {}
    for column_name, pattern in AMENITY_PATTERNS.items():
        features[column_name] = cleaned.str.contains(pattern, regex=True)
    return pd.DataFrame(features, index=text_series.index)
