"""Blocking: only compare records that share at least one cheap key, instead of all n^2 pairs."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

MAX_BLOCK = 150   # very common keys (e.g. 'sri') would explode the pair count


def block_keys(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "pincode+first_token": df["pincode_f"].fillna("") + "|" + df["skel1"],
        "first_two_tokens": df["skel2"],
        "pan": df["pan"],
        "phone": df["phone_n"],
    }


def candidate_pairs(df: pd.DataFrame, max_block: int = MAX_BLOCK) -> tuple[pd.DataFrame, dict]:
    pairs: set[tuple[int, int]] = set()
    per_key, skipped = {}, 0
    for key_name, key in block_keys(df).items():
        before = len(pairs)
        valid = key.notna() & (key.astype(str).str.strip("| ") != "")
        for _, idx in df.index[valid].to_series().groupby(key[valid]):
            ids = idx.to_numpy()
            if len(ids) < 2:
                continue
            if len(ids) > max_block:
                skipped += 1
                continue
            pairs.update(combinations(sorted(ids), 2))
        per_key[key_name] = len(pairs) - before
    out = pd.DataFrame(sorted(pairs), columns=["i", "j"])
    n = len(df)
    stats = {"records": n, "all_pairs": n * (n - 1) // 2, "candidate_pairs": len(out),
             "reduction_ratio": 1 - len(out) / max(n * (n - 1) // 2, 1),
             "new_pairs_by_key": per_key, "oversized_blocks_skipped": skipped}
    return out, stats


def pair_completeness(df: pd.DataFrame, pairs: pd.DataFrame) -> float:
    """Share of all true duplicate pairs that survive blocking (the recall ceiling for matching)."""
    sizes = df.groupby("entity_id").size()
    true_total = int((sizes * (sizes - 1) // 2).sum())
    ent = df["entity_id"].to_numpy()
    found = int((ent[pairs["i"].to_numpy()] == ent[pairs["j"].to_numpy()]).sum())
    return found / true_total if true_total else 1.0
