"""Per-record validation and normalisation, done once before blocking and matching."""
from __future__ import annotations

import pandas as pd

from .normalize import core_tokens, extract_pincode, legal_form, normalize_address, normalize_name, skeleton
from .validate import (is_valid_gstin, is_valid_ifsc, is_valid_pincode, normalize_phone, pan_from_gstin)


def prepare(legacy: pd.DataFrame, state_prefixes: tuple[str, ...] = ("50",)) -> pd.DataFrame:
    df = legacy.copy().reset_index(drop=True)
    df["norm_name"] = df["vendor_name"].map(normalize_name)
    df["core"] = df["norm_name"].map(lambda s: " ".join(core_tokens(s)))
    toks = df["core"].str.split()
    df["skel1"] = toks.map(lambda t: skeleton(t[0]) if t else "")
    df["skel2"] = toks.map(lambda t: " ".join(skeleton(x) for x in t[:2]))
    df["legal_form"] = df["norm_name"].map(legal_form)
    df["gstin_u"] = df["gstin"].map(lambda g: g.strip().upper() if isinstance(g, str) else None)
    df["gstin_valid"] = df["gstin_u"].map(is_valid_gstin)
    df["pan"] = df["gstin_u"].map(pan_from_gstin)
    df["phone_n"] = df["phone"].map(normalize_phone)
    df["pincode_f"] = [p if isinstance(p, str) and p else extract_pincode(a)
                       for p, a in zip(df["pincode"], df["address"])]
    df["pincode_valid"] = df["pincode_f"].map(lambda p: is_valid_pincode(p, state_prefixes))
    df["addr_n"] = df["address"].map(normalize_address)
    df["ifsc_valid"] = df["ifsc"].map(is_valid_ifsc)
    df["bank_n"] = df["bank_account"].map(lambda b: b.strip() if isinstance(b, str) and b.strip() else None)
    return df


def validation_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("GSTIN present", df["gstin_u"].notna().mean()),
        ("GSTIN valid (format + check digit), of present", df.loc[df["gstin_u"].notna(), "gstin_valid"].mean()),
        ("Phone usable after normalising", df["phone_n"].notna().mean()),
        ("Pincode present (field or address)", df["pincode_f"].notna().mean()),
        ("Bank account present", df["bank_n"].notna().mean()),
        ("IFSC valid", df["ifsc_valid"].mean()),
    ]
    return pd.DataFrame(rows, columns=["check", "share"])
