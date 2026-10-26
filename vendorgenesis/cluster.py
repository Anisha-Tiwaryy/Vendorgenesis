"""From pair scores to vendor clusters, golden records and a human review queue."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

SOURCE_PRIORITY = {"LEGACY_ERP": 2, "TALLY": 1, "EXCEL_AP": 0}


def clusters_from_pairs(n: int, pairs: pd.DataFrame, accept: np.ndarray) -> np.ndarray:
    """Connected components of the accepted-match graph. Returns a cluster id per record."""
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(zip(pairs["i"].to_numpy()[accept], pairs["j"].to_numpy()[accept]))
    labels = np.empty(n, dtype=int)
    for cid, comp in enumerate(nx.connected_components(g)):
        labels[list(comp)] = cid
    return labels


def bcubed(pred: np.ndarray, truth: np.ndarray) -> dict:
    """B-cubed precision/recall: for each record, how pure and how complete its predicted cluster is."""
    df = pd.DataFrame({"p": pred, "t": truth})
    both = df.groupby(["p", "t"]).size()
    p_size, t_size = df.groupby("p").size(), df.groupby("t").size()
    inter = both.reindex(pd.MultiIndex.from_arrays([df["p"], df["t"]])).to_numpy()
    prec = np.mean(inter / p_size.reindex(df["p"]).to_numpy())
    rec = np.mean(inter / t_size.reindex(df["t"]).to_numpy())
    return {"bcubed_precision": prec, "bcubed_recall": rec, "bcubed_f1": 2 * prec * rec / (prec + rec),
            "predicted_vendors": int(df["p"].nunique()), "true_vendors": int(df["t"].nunique())}


def review_simulation(y: np.ndarray, scores: np.ndarray, lo: float, hi: float) -> dict:
    """Auto-accept >= hi, auto-reject < lo, send the band in between to a reviewer
    (simulated with the answer key). Returns workload and resulting pair precision/recall."""
    auto_yes, review = scores >= hi, (scores >= lo) & (scores < hi)
    accepted = auto_yes | (review & (y == 1))
    tp = int((accepted & (y == 1)).sum())
    return {"band": f"{lo:.2f}-{hi:.2f}", "pairs_to_review": int(review.sum()),
            "precision": tp / max(int(accepted.sum()), 1), "recall": tp / max(int(y.sum()), 1),
            "accepted": accepted}


def golden_records(df: pd.DataFrame, cluster: np.ndarray) -> pd.DataFrame:
    """Survivorship: per cluster, keep the record with a valid GSTIN, most complete fields and the
    most trusted source; fill its gaps field by field from the other members. Conflicting bank
    accounts are never auto-resolved: they are flagged for verification (payment-fraud risk)."""
    d = df.assign(cluster=cluster)
    fields = ["gstin_u", "phone_n", "pincode_f", "bank_n", "ifsc"]
    d["_score"] = (d["gstin_valid"].astype(int) * 10 + d[fields].notna().sum(axis=1) * 2
                   + d["source_system"].map(SOURCE_PRIORITY).fillna(0))
    out = []
    for cid, grp in d.sort_values("_score", ascending=False).groupby("cluster", sort=False):
        best = grp.iloc[0].copy()
        for f in fields:
            if pd.isna(best[f]):
                vals = grp[f].dropna()
                if f == "gstin_u":
                    vals = grp.loc[grp["gstin_valid"], f]
                if len(vals):
                    best[f] = vals.iloc[0]
        banks = grp["bank_n"].dropna().unique()
        best["bank_conflict"] = len(banks) > 1
        best["gstin_valid"] = bool(isinstance(best["gstin_u"], str) and grp.loc[grp["gstin_u"] == best["gstin_u"], "gstin_valid"].any())
        best["member_legacy_ids"] = ";".join(grp["legacy_id"])
        best["members"] = len(grp)
        best["sources"] = ";".join(sorted(grp["source_system"].unique()))
        out.append(best)
    g = pd.DataFrame(out).drop(columns=["_score"]).reset_index(drop=True)
    g["vendor_account"] = [f"V-TG-{i + 1:05d}" for i in range(len(g))]
    return g
