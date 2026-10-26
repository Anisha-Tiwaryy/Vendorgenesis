"""What duplicate vendors cost: duplicate payments that a per-vendor-record check cannot see."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

WINDOW_DAYS = 45


def invoice_core(number: str) -> str:
    """'INV/2025/0042', 'INV-2025-42', '42', '2025/0042' -> '42' (the sequence number)."""
    digits = re.findall(r"\d+", str(number))
    if not digits:
        return ""
    last = digits[-1]
    if len(digits) == 1 and len(last) >= 8 and last.startswith("20"):   # 'INV20250042': year glued on
        last = last[4:]
    return last.lstrip("0") or "0"


def flag_duplicates(inv: pd.DataFrame, vendor_key: pd.Series) -> pd.DataFrame:
    """Pairs of invoices under the same vendor key with the same amount, the same invoice
    sequence number and dates within WINDOW_DAYS. Same amount alone is not enough: recurring
    fixed-fee vendors bill the same amount every month legitimately."""
    d = inv.assign(vkey=vendor_key.to_numpy(), core=inv["invoice_no"].map(invoice_core)).reset_index(drop=True)
    flagged = []
    for _, grp in d.groupby(["vkey", "amount", "core"]):
        if len(grp) < 2:
            continue
        g = grp.sort_values("invoice_date")
        rows = g.to_dict("records")
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                if (rows[b]["invoice_date"] - rows[a]["invoice_date"]).days <= WINDOW_DAYS:
                    flagged.append((rows[a]["invoice_key"], rows[b]["invoice_key"], rows[b]["amount"]))
    return pd.DataFrame(flagged, columns=["original", "suspected_duplicate", "amount"])


def score_detection(inv: pd.DataFrame, flagged: pd.DataFrame) -> dict:
    planted = inv[inv["is_planted_dup"]]
    caught = set(flagged["suspected_duplicate"]) | set(flagged["original"])
    hit = planted["invoice_key"].isin(caught)
    true_flags = flagged["suspected_duplicate"].isin(planted["invoice_key"]) | flagged["original"].isin(planted["invoice_key"])
    return {"planted_duplicates": len(planted), "planted_value": float(planted["amount"].sum()),
            "caught": int(hit.sum()), "caught_value": float(planted.loc[hit, "amount"].sum()),
            "flags_raised": len(flagged), "false_flags": int((~true_flags).sum())}


def compare(inv: pd.DataFrame, legacy: pd.DataFrame, cluster: np.ndarray) -> pd.DataFrame:
    """Before: the check runs per legacy vendor record (what an AP system does on its own).
    After: the same check runs per deduplicated vendor cluster."""
    cluster_of = pd.Series(cluster, index=legacy["legacy_id"].to_numpy())
    before = score_detection(inv, flag_duplicates(inv, inv["vendor_legacy_id"]))
    after = score_detection(inv, flag_duplicates(inv, inv["vendor_legacy_id"].map(cluster_of)))
    return pd.DataFrame([{"check runs per": "legacy vendor record", **before},
                         {"check runs per": "deduplicated vendor (VendorGenesis)", **after}])
