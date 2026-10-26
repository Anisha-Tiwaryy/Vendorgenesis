"""ERP-readiness scoring, Dynamics 365-shaped export, fix list and PII masking."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .validate import TELANGANA_STATE_CODE, is_valid_ifsc

REQUIRED = {"vendor_name": "name", "address": "address", "pincode_f": "pincode", "phone_n": "phone",
            "gstin_u": "GSTIN", "bank_n": "bank account", "ifsc": "IFSC"}


def record_issues(r: pd.Series) -> list[str]:
    issues = [f"missing {label}" for col, label in REQUIRED.items() if pd.isna(r.get(col)) or r.get(col) == ""]
    if isinstance(r.get("gstin_u"), str) and not r.get("gstin_valid"):
        issues.append("GSTIN fails format/check digit")
    if isinstance(r.get("gstin_u"), str) and r.get("gstin_valid") and isinstance(r.get("pincode_f"), str):
        if r["gstin_u"][:2] == TELANGANA_STATE_CODE and not r["pincode_f"].startswith("50"):
            issues.append("GSTIN state (36, Telangana) does not match pincode")
    if isinstance(r.get("ifsc"), str) and not is_valid_ifsc(r["ifsc"]):
        issues.append("IFSC invalid")
    if isinstance(r.get("pincode_f"), str) and not r.get("pincode_valid", True):
        issues.append("pincode invalid or outside state")
    if r.get("bank_conflict"):
        issues.append("bank account differs across duplicate records: verify before payment")
    return issues


def readiness(df: pd.DataFrame, duplicate_flags: np.ndarray | None = None) -> dict:
    """Four dimensions, each 0-1: completeness, validity, uniqueness, consistency."""
    present = df[list(REQUIRED)].notna().mean(axis=1)
    gst_ok = df["gstin_valid"].where(df["gstin_u"].notna())
    validity = pd.concat([gst_ok, df["ifsc"].map(is_valid_ifsc).where(df["ifsc"].notna()),
                          df["pincode_valid"].where(df["pincode_f"].notna())], axis=1).mean(axis=1).fillna(0)
    uniq = 1 - (duplicate_flags.astype(float) if duplicate_flags is not None else np.zeros(len(df)))
    cons = df.apply(lambda r: 0.0 if any("does not match" in x or "differs" in x for x in record_issues(r)) else 1.0, axis=1)
    dims = {"completeness": present.mean(), "validity": validity.mean(),
            "uniqueness": float(np.mean(uniq)), "consistency": cons.mean()}
    dims["overall"] = float(np.mean(list(dims.values())))
    ready = np.array([not record_issues(r) for _, r in df.iterrows()])
    # a record with known duplicates cannot be loaded as-is, so it does not count as ready
    dims["records_ready_to_load"] = float(np.mean(ready & (uniq == 1)))
    return dims


def mask_account(acct) -> str | None:
    if not isinstance(acct, str):
        return None
    return "X" * (len(acct) - 4) + acct[-4:]


def mask_phone(phone) -> str | None:
    if not isinstance(phone, str):
        return None
    return phone[:2] + "XXXXXX" + phone[-2:]


def d365_export(golden: pd.DataFrame, masked: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (load file, fix list, legacy-id cross reference).
    Column names follow the Dynamics 365 Finance 'Vendors V2' entity naming style; the file is
    shaped like that entity but has not been imported into a live Dynamics instance."""
    issues = golden.apply(record_issues, axis=1)
    ok = issues.map(len) == 0
    load = pd.DataFrame({
        "VENDORACCOUNTNUMBER": golden["vendor_account"],
        "VENDORORGANIZATIONNAME": golden["vendor_name"].str.replace(r"^m/s\.?\s*", "", case=False, regex=True)
                                   .str.replace(r"\s+", " ", regex=True).str.strip().str.title(),
        "VENDORGROUPID": "DOM-TG",
        "ADDRESSDESCRIPTION": golden["address"],
        "ADDRESSZIPCODE": golden["pincode_f"],
        "ADDRESSSTATEID": "TG",
        "ADDRESSCOUNTRYREGIONID": "IND",
        "PRIMARYCONTACTPHONE": golden["phone_n"].map(mask_phone) if masked else golden["phone_n"],
        "TAXREGISTRATIONNUMBER_GSTIN": golden["gstin_u"],
        "BANKACCOUNTNUMBER": golden["bank_n"].map(mask_account) if masked else golden["bank_n"],
        "BANKIFSC": golden["ifsc"],
    })[ok.to_numpy()]
    fix = pd.DataFrame({"vendor_account": golden["vendor_account"], "vendor_name": golden["vendor_name"],
                        "issues": issues.map("; ".join)})[~ok.to_numpy()]
    xref = golden[["vendor_account", "member_legacy_ids"]].assign(
        legacy_id=golden["member_legacy_ids"].str.split(";")).explode("legacy_id")[["legacy_id", "vendor_account"]]
    return load.reset_index(drop=True), fix.reset_index(drop=True), xref.reset_index(drop=True)
