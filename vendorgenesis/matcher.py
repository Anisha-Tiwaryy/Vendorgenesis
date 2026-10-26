"""Pair classification: rule baselines vs learned matchers, evaluated without entity leakage.

Pairs are split by vendor entity: a record's entity is assigned to one fold, and a pair is used
only if both records sit in the same fold. A model is never tested on a vendor it trained on.
Every record then gets an out-of-fold match score (2-fold cross-fitting).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES


def labels(df: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    ent = df["entity_id"].to_numpy()
    return (ent[pairs["i"].to_numpy()] == ent[pairs["j"].to_numpy()]).astype(int)


def entity_fold(entity_id: str, k: int = 2, salt: str = "vg") -> int:
    return int(hashlib.md5(f"{salt}{entity_id}".encode()).hexdigest(), 16) % k


def pair_folds(df: pd.DataFrame, pairs: pd.DataFrame, k: int = 2) -> np.ndarray:
    """Fold id per pair, or -1 when its two records fall in different folds (pair unused in training)."""
    f = df["entity_id"].map(lambda e: entity_fold(e, k)).to_numpy()
    fi, fj = f[pairs["i"].to_numpy()], f[pairs["j"].to_numpy()]
    return np.where(fi == fj, fi, -1)


def make_models(seed: int = 0) -> dict:
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=seed),
    }


def rule_baselines(df: pd.DataFrame, pairs: pd.DataFrame, X: pd.DataFrame) -> dict[str, np.ndarray]:
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    return {
        "Rule: exact normalised name": (df["norm_name"].to_numpy()[i] == df["norm_name"].to_numpy()[j]).astype(int),
        "Rule: fuzzy name >= 0.90": (X["name_token_set"].to_numpy() >= 0.90).astype(int),
        "Rule: same GSTIN or fuzzy >= 0.90": ((X["gstin_equal"].to_numpy() == 1) | (X["name_token_set"].to_numpy() >= 0.90)).astype(int),
    }


def cross_fit(df: pd.DataFrame, pairs: pd.DataFrame, X: pd.DataFrame, y: np.ndarray,
              model_name: str = "Gradient Boosting", k: int = 2, seed: int = 0,
              features: list[str] | None = None) -> np.ndarray:
    """Out-of-fold match probability for every candidate pair. A cross-fold pair (its two records
    in different folds) is always a true non-match by construction; it is scored by averaging
    both fold models, and it is excluded from the reported pair metrics."""
    feats = features or FEATURES
    folds = pair_folds(df, pairs, k)
    scores = np.zeros(len(pairs))
    models = []
    for f in range(k):
        train = (folds != f) & (folds != -1)
        m = make_models(seed)[model_name].fit(X[feats][train], y[train])
        models.append(m)
        test = folds == f
        scores[test] = m.predict_proba(X[feats][test])[:, 1]
    cross = folds == -1
    if cross.any():
        scores[cross] = np.mean([m.predict_proba(X[feats][cross])[:, 1] for m in models], axis=0)
    return scores


def evaluate_pairs(df, pairs, X, y, threshold: float = 0.5, seed: int = 0,
                   features: list[str] | None = None, rules: bool = True) -> tuple[pd.DataFrame, dict]:
    """Compare rules and models on out-of-fold pairs only (cross-fold pairs excluded)."""
    folds = pair_folds(df, pairs)
    keep = folds != -1
    rows, scores = [], {}
    if rules:
        for name, pred in rule_baselines(df, pairs, X).items():
            rows.append(_metrics(name, y[keep], pred[keep], None))
    for name in make_models(seed):
        s = cross_fit(df, pairs, X, y, name, seed=seed, features=features)
        scores[name] = s
        rows.append(_metrics(name, y[keep], (s[keep] >= threshold).astype(int), s[keep]))
    return pd.DataFrame(rows), scores


def _metrics(name, y, pred, score):
    return {"method": name, "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "pr_auc": average_precision_score(y, score) if score is not None else np.nan,
            "false_merges": int(((pred == 1) & (y == 0)).sum()),
            "missed_duplicates": int(((pred == 0) & (y == 1)).sum())}
