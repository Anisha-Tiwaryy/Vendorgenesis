"""Pairwise comparison features for candidate pairs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.fuzz import partial_ratio, token_set_ratio, token_sort_ratio
from sklearn.feature_extraction.text import TfidfVectorizer

FEATURES = ["name_jw", "name_token_set", "name_token_sort", "name_partial", "core_tfidf",
            "first_token_equal", "core_len_diff", "legal_form_equal",
            "gstin_equal", "pan_equal", "phone_equal", "pincode_equal", "addr_token_set",
            "bank_equal", "pan_distance", "both_gstin_valid"]


def _tri_state(a, b) -> np.ndarray:
    """1 if both present and equal, 0 if both present and different, -1 if either missing.
    Missing is informative on its own, so it is kept as a separate value instead of imputed."""
    a, b = pd.Series(a), pd.Series(b)
    both = a.notna().to_numpy() & b.notna().to_numpy()
    eq = (a.to_numpy() == b.to_numpy())
    return np.where(both, eq.astype(int), -1)


def _raw_pan(g):
    return g[2:12] if isinstance(g, str) and len(g) == 15 else None


def _pan_distance(a, b) -> int:
    """Edit distance between the PAN parts of two GSTINs, even if one fails validation.
    1 usually means a keying error on the same vendor; 3+ between two check-digit-valid GSTINs
    means two different legal entities. -1 if either is missing."""
    pa, pb = _raw_pan(a), _raw_pan(b)
    return -1 if pa is None or pb is None else Levenshtein.distance(pa, pb)


def pair_features(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    A, B = df.iloc[i].reset_index(drop=True), df.iloc[j].reset_index(drop=True)

    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3)).fit(df["core"])
    M = tfidf.transform(df["core"])
    core_cos = np.asarray(M[i].multiply(M[j]).sum(axis=1)).ravel()

    f = pd.DataFrame({
        "name_jw": [JaroWinkler.similarity(a, b) for a, b in zip(A["norm_name"], B["norm_name"])],
        "name_token_set": [token_set_ratio(a, b) / 100 for a, b in zip(A["core"], B["core"])],
        "name_token_sort": [token_sort_ratio(a, b) / 100 for a, b in zip(A["core"], B["core"])],
        "name_partial": [partial_ratio(a, b) / 100 for a, b in zip(A["core"], B["core"])],
        "core_tfidf": core_cos,
        "first_token_equal": (A["skel1"].to_numpy() == B["skel1"].to_numpy()).astype(int),
        "core_len_diff": np.abs(A["core"].str.split().str.len() - B["core"].str.split().str.len()),
        "legal_form_equal": (A["legal_form"].to_numpy() == B["legal_form"].to_numpy()).astype(int),
        "gstin_equal": _tri_state(A["gstin_u"], B["gstin_u"]),
        "pan_equal": _tri_state(A["pan"], B["pan"]),
        "phone_equal": _tri_state(A["phone_n"], B["phone_n"]),
        "pincode_equal": _tri_state(A["pincode_f"], B["pincode_f"]),
        "addr_token_set": [token_set_ratio(a, b) / 100 for a, b in zip(A["addr_n"], B["addr_n"])],
        "bank_equal": _tri_state(A["bank_n"], B["bank_n"]),
        "pan_distance": [_pan_distance(a, b) for a, b in zip(A["gstin_u"], B["gstin_u"])],
        "both_gstin_valid": (A["gstin_valid"].to_numpy() & B["gstin_valid"].to_numpy()).astype(int),
    })
    return f[FEATURES]
