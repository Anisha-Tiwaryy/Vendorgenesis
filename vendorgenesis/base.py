"""Clean 'true' vendor universe: either the real MCA Registrar of Companies master data
(data.gov.in) or a built-in Telangana-style generator when that file is not available.

Each true vendor gets an entity_id plus PAN, GSTIN, phone, bank account and IFSC.
Registry files carry no tax or bank details, so those are always generated (and valid).
"""
from __future__ import annotations

import re
import string
from pathlib import Path

import numpy as np
import pandas as pd

from .normalize import core_tokens, extract_pincode, normalize_name
from .validate import make_gstin

LOCALITIES = [
    ("Banjara Hills", "500034"), ("Jubilee Hills", "500033"), ("Madhapur", "500081"),
    ("Gachibowli", "500032"), ("Kukatpally", "500072"), ("Ameerpet", "500016"),
    ("Begumpet", "500016"), ("Secunderabad", "500003"), ("Uppal", "500039"),
    ("LB Nagar", "500074"), ("Kondapur", "500084"), ("Miyapur", "500049"),
    ("Abids", "500001"), ("Himayatnagar", "500029"), ("Somajiguda", "500082"),
    ("Charminar", "500002"), ("Balanagar", "500037"), ("Jeedimetla", "500055"),
    ("Patancheru", "502319"), ("Shamshabad", "501218"), ("Warangal", "506002"),
    ("Karimnagar", "505001"), ("Nizamabad", "503001"), ("Khammam", "507001"),
]
PREFIXES = ["Sri", "Sri Lakshmi", "Sri Venkateswara", "Sri Sai", "Sri Balaji", "Deccan", "Golconda",
            "Kakatiya", "Charminar", "Godavari", "Krishna", "Srinivasa", "Vijaya", "Annapurna",
            "Bhagyanagar", "Pragathi", "Navya", "Mohammed", "Nizam", "Ameer", "Sunrise", "Trident",
            "Apex", "Prime", "Global", "Sai Krishna", "Lakshmi Ganapathi", "Telangana", "Hyderabad",
            "Siddhi Vinayaka", "Mahalakshmi", "Venkata Ramana", "Raghavendra", "Shiva Shakti",
            "Al Noor", "Zamzam", "Rehmat", "Madina", "Taj", "Kohinoor", "Vasavi", "Durga", "Padmavathi",
            "Vamsi", "Harsha", "Keerthi", "Manjeera", "Musi", "Orbit", "Vertex", "Neo", "Greenfield"]
INDUSTRIES = ["Agro Industries", "Infrastructure", "Steels", "Pharmaceuticals", "Textiles", "Traders",
              "Enterprises", "Constructions", "Logistics", "Technologies", "Software Solutions",
              "Foods", "Polymers", "Electricals", "Packaging", "Chemicals", "Engineering Works",
              "Projects", "Industries", "Builders", "Exports", "Motors", "Hospitality", "Castings",
              "Granites", "Cement Products", "Paper Mills", "Plastics", "Spices", "Dairy Products"]
BANKS = ["SBIN", "HDFC", "ICIC", "UTIB", "KKBK", "CNRB", "UBIN", "BARB", "IDIB", "PUNB"]


def _pan(rng, name: str) -> str:
    first = next((c for c in name.upper() if c.isalpha()), "X")
    letters = "".join(rng.choice(list(string.ascii_uppercase), 3))
    return f"{letters}C{first}{rng.integers(1000, 10000)}{rng.choice(list(string.ascii_uppercase))}"


def _identities(df: pd.DataFrame, rng) -> pd.DataFrame:
    n = len(df)
    df = df.reset_index(drop=True)
    df["entity_id"] = [f"E{i:05d}" for i in range(n)]
    pans, seen = [], set()
    for name in df["name"]:
        p = _pan(rng, name)
        while p in seen:
            p = _pan(rng, name)
        seen.add(p)
        pans.append(p)
    df["pan"] = pans
    df["gstin"] = [make_gstin(p) for p in pans]
    df["phone"] = [str(rng.integers(6, 10)) + "".join(map(str, rng.integers(0, 10, 9))) for _ in range(n)]
    df["bank_account"] = ["".join(map(str, rng.integers(0, 10, int(rng.integers(11, 17))))) for _ in range(n)]
    df["ifsc"] = [f"{rng.choice(BANKS)}0{rng.integers(0, 1_000_000):06d}" for _ in range(n)]
    return df


def synthetic_base(n: int = 3000, seed: int = 0, sibling_share: float = 0.15) -> pd.DataFrame:
    """Plausible Telangana company universe. `sibling_share` of companies get a near-namesake
    (same prefix and locality, different line of business): the hard negatives a matcher must
    not merge."""
    rng = np.random.default_rng(seed)
    rows, names = [], set()
    while len(rows) < n:
        src_idx = -1
        if rows and rng.random() < sibling_share:
            src_idx = int(rng.integers(len(rows)))
            src = rows[src_idx]
            prefix, (loc, pin) = src["_prefix"], (src["_loc"], src["pincode"])
        else:
            prefix = rng.choice(PREFIXES)
            loc, pin = LOCALITIES[int(rng.integers(len(LOCALITIES)))]
        industry = rng.choice(INDUSTRIES)
        form = rng.choice(["Private Limited", "Limited", "LLP"], p=[.85, .1, .05])
        name = f"{prefix} {industry} {form}"
        if name in names:
            continue
        names.add(name)
        plot = f"{'Plot' if rng.random() < .5 else 'H.No'} {rng.integers(1, 400)}"
        road = f"Road No {rng.integers(1, 15)}" if rng.random() < .6 else f"{rng.choice(['Main', 'Industrial', 'Station', 'MG'])} Road"
        city = "Hyderabad" if pin.startswith("500") else loc
        rows.append({"name": name, "address": f"{plot}, {road}, {loc}, {city}, Telangana {pin}",
                     "pincode": pin, "_prefix": prefix, "_loc": loc, "_sibling_of": src_idx})
    df = pd.DataFrame(rows)
    sib = df["_sibling_of"].to_numpy()
    df = _identities(df.drop(columns=["_prefix", "_loc", "_sibling_of"]), rng)
    # group companies often share a registered office, and sometimes an accounts phone line
    for k, src in enumerate(sib):
        if src >= 0:
            if rng.random() < .5:
                df.loc[k, "address"] = df.loc[src, "address"]
            if rng.random() < .3:
                df.loc[k, "phone"] = df.loc[src, "phone"]
    df["sibling_of"] = sib
    return df


def _pick(columns, *patterns):
    for p in patterns:
        for c in columns:
            if re.search(p, c, re.I):
                return c
    return None


def load_registry(path: str | Path, n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Load a data.gov.in RoC-wise company master CSV (column names vary between releases).
    Keeps active companies with a usable address and pincode, samples n of them."""
    raw = pd.read_csv(path, dtype=str, encoding_errors="ignore", low_memory=False)
    name_c = _pick(raw.columns, r"^company_?name$", r"name")
    addr_c = _pick(raw.columns, r"registered_?office_?address", r"address")
    status_c = _pick(raw.columns, r"company_?status", r"status")
    if name_c is None or addr_c is None:
        raise ValueError(f"could not find name/address columns in {list(raw.columns)[:12]}")
    df = raw[[name_c, addr_c] + ([status_c] if status_c else [])].copy()
    df.columns = ["name", "address"] + (["status"] if status_c else [])
    if status_c:
        df = df[df["status"].str.contains("active", case=False, na=False)].drop(columns="status")
    df["pincode"] = df["address"].map(extract_pincode)
    df = df.dropna(subset=["name", "pincode"])
    df["name"] = df["name"].str.strip().str.title()
    df = df[df["name"].map(lambda s: len(core_tokens(normalize_name(s))) >= 1)]
    df = df.drop_duplicates(subset=["name"])
    rng = np.random.default_rng(seed)
    if len(df) > n:
        df = df.sample(n, random_state=seed)
    return _identities(df[["name", "address", "pincode"]], rng)
