"""End-to-end run: base -> legacy systems -> prepare -> block -> match -> cluster -> golden -> export -> impact."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .base import load_registry, synthetic_base
from .blocking import candidate_pairs, pair_completeness
from .cluster import bcubed, clusters_from_pairs, golden_records, review_simulation
from .features import pair_features
from .impact import compare
from .legacy import build_invoices, build_legacy
from .matcher import cross_fit, evaluate_pairs, labels
from .prepare import prepare, validation_summary
from .readiness import d365_export, readiness

REVIEW_LO, REVIEW_HI = 0.05, 0.98


def run(registry_csv: str | None = None, n: int = 3000, seed: int = 0, verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    log = (lambda *a: print(f"[{time.perf_counter() - t0:6.1f}s]", *a)) if verbose else (lambda *a: None)

    base = load_registry(registry_csv, n, seed) if registry_csv else synthetic_base(n, seed)
    legacy = build_legacy(base, seed)
    invoices = build_invoices(legacy, seed)
    log(f"base vendors {len(base)}, legacy records {len(legacy)}, invoices {len(invoices)}")

    df = prepare(legacy)
    pairs, block_stats = candidate_pairs(df)
    block_stats["pair_completeness"] = pair_completeness(df, pairs)
    log(f"candidate pairs {len(pairs):,} of {block_stats['all_pairs']:,}")

    X = pair_features(df, pairs)
    y = labels(df, pairs)
    pair_table, scores = evaluate_pairs(df, pairs, X, y, seed=seed)
    log("pair models evaluated")

    s = scores["Gradient Boosting"]
    review = review_simulation(y, s, REVIEW_LO, REVIEW_HI)
    truth = df["entity_id"].to_numpy()
    cluster_rows = []
    for name, accept in [("Rule: same GSTIN or fuzzy >= 0.90", ((X["gstin_equal"] == 1) | (X["name_token_set"] >= .9)).to_numpy()),
                         ("Gradient Boosting, auto only (>= 0.5)", s >= 0.5),
                         (f"Gradient Boosting + review of {REVIEW_LO}-{REVIEW_HI} band", review["accepted"])]:
        cl = clusters_from_pairs(len(df), pairs, accept)
        cluster_rows.append({"method": name, **bcubed(cl, truth)})
    final_cluster = clusters_from_pairs(len(df), pairs, review["accepted"])

    golden = golden_records(df, final_cluster)
    load, fix, xref = d365_export(golden)
    # score legacy records with what the pipeline found (not the answer key): duplicate flags and
    # bank conflicts both come from the predicted clusters
    grp = df.groupby(final_cluster)
    dup_flags = grp["legacy_id"].transform("size").to_numpy() > 1
    legacy_view = df.assign(bank_conflict=grp["bank_n"].transform(lambda b: b.dropna().nunique() > 1).to_numpy())
    before = readiness(legacy_view, duplicate_flags=dup_flags)
    after = readiness(golden, duplicate_flags=np.zeros(len(golden), dtype=bool))
    impact = compare(invoices, df, final_cluster)
    log("done")

    importances = feature_importance(df, pairs, X, y, seed)

    return {"base": base, "legacy": legacy, "invoices": invoices, "prepared": df, "pairs": pairs,
            "features": X, "y": y, "scores": s, "block_stats": block_stats, "pair_table": pair_table,
            "review": {k: v for k, v in review.items() if k != "accepted"}, "cluster_table": pd.DataFrame(cluster_rows),
            "clusters": final_cluster, "golden": golden, "load": load, "fix": fix, "xref": xref,
            "readiness_before": before, "readiness_after": after, "impact": impact,
            "validation": validation_summary(df), "importances": importances,
            "seconds": time.perf_counter() - t0, "source": "registry CSV" if registry_csv else "synthetic base"}


def feature_importance(df, pairs, X, y, seed: int = 0) -> pd.Series:
    """Permutation importance on held-out pairs (train on fold 1, measure on fold 0)."""
    from sklearn.inspection import permutation_importance
    from .features import FEATURES
    from .matcher import make_models, pair_folds
    folds = pair_folds(df, pairs)
    m = make_models(seed)["Gradient Boosting"].fit(X[folds == 1], y[folds == 1])
    test = folds == 0
    imp = permutation_importance(m, X[test], y[test], scoring="average_precision", n_repeats=5, random_state=seed)
    return pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False)
