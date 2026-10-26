"""Turn the clean vendor universe into three messy legacy systems, plus an AP invoice ledger
with planted duplicate payments. Everything planted is recorded so results can be scored."""
from __future__ import annotations

import string

import numpy as np
import pandas as pd

from .base import LOCALITIES

SOURCES = ["LEGACY_ERP", "TALLY", "EXCEL_AP"]
SHORT_FORMS = {"Private Limited": ["Pvt Ltd", "Pvt. Ltd.", "P Ltd", "(P) Ltd", "Private Ltd", "PVT LTD"],
               "Limited": ["Ltd", "Ltd.", "LTD"], "LLP": ["L.L.P.", "Llp"]}
VARIANTS = {"Sri": ["Shri", "Sree", "Shree"], "Lakshmi": ["Laxmi"], "Venkateswara": ["Venkateshwara"],
            "Mohammed": ["Mohd", "Md", "Muhammad", "Mohamed"], "Srinivasa": ["Srinivas", "Sreenivasa"],
            "Industries": ["Inds", "Ind"], "Engineering": ["Engg"], "Enterprises": ["Entp", "Ent"],
            "Technologies": ["Tech"], "Pharmaceuticals": ["Pharma"], "Infrastructure": ["Infra"],
            "Hyderabad": ["Hyd"], "Electricals": ["Elec"], "Constructions": ["Constn"]}


def _typo(rng, s: str) -> str:
    letters = [i for i, c in enumerate(s) if c.isalpha()]
    if len(letters) < 6:
        return s
    i = int(rng.choice(letters[1:-1]))
    op = rng.integers(3)
    if op == 0:                                   # drop a letter
        return s[:i] + s[i + 1:]
    if op == 1 and i + 1 < len(s):                # swap neighbours
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + rng.choice(list("aeiou")) + s[i + 1:]   # wrong vowel


def mess_name(rng, name: str, level: float) -> str:
    s = name
    for full, shorts in SHORT_FORMS.items():
        if s.endswith(full) and rng.random() < .7 * level + .2:
            s = s[: -len(full)] + rng.choice(shorts)
    for word, alts in VARIANTS.items():
        if word in s.split() and rng.random() < .5 * level:
            s = " ".join(rng.choice(alts) if t == word else t for t in s.split())
    if rng.random() < .25 * level:
        s = f"M/s {s}" if rng.random() < .5 else s.replace(" ", "  ", 1)
    if rng.random() < .35 * level:
        s = _typo(rng, s)
    if rng.random() < .3 * level:
        s = s.upper() if rng.random() < .7 else s.lower()
    tokens = s.split()
    if len(tokens) > 4 and rng.random() < .12 * level:           # a middle word gets dropped
        del tokens[int(rng.integers(1, len(tokens) - 2))]
        s = " ".join(tokens)
    return s


def mess_address(rng, addr: str, level: float) -> str:
    s = addr
    if rng.random() < .5 * level:
        s = s.replace("Road No", rng.choice(["Rd No", "Road No.", "Rd. No."])).replace(" Road", " Rd")
    if rng.random() < .3 * level:
        s = s.replace(", Telangana", "").replace("Hyderabad", rng.choice(["Hyd", "HYDERABAD"]))
    if rng.random() < .25 * level:
        s = s.rsplit(" ", 1)[0]                    # pincode dropped from the text
    if rng.random() < .2 * level:
        s = s.replace("H.No", "H No").replace("Plot", "Plot No")
    return s


def mess_phone(rng, phone: str) -> str | None:
    r = rng.random()
    if r < .15:
        return None
    fmt = rng.integers(4)
    return [phone, f"+91 {phone[:5]} {phone[5:]}", f"0{phone}", f"{phone[:5]}-{phone[5:]}"][fmt]


def mess_gstin(rng, gstin: str) -> str | None:
    r = rng.random()
    if r < .18:
        return None
    if r < .23:                                     # one-character keying error
        i = int(rng.integers(2, 14))
        return gstin[:i] + rng.choice([c for c in string.ascii_uppercase + string.digits if c != gstin[i]]) + gstin[i + 1:]
    if r < .25:
        return gstin.lower()
    return gstin


def build_legacy(base: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Each true vendor appears 1-3 times across the three systems (about 45% of vendors end
    up duplicated). The rates here are assumptions chosen for the simulation, not industry data."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, v in base.iterrows():
        copies = int(rng.choice([1, 2, 3], p=[.55, .30, .15]))
        sources = list(rng.choice(SOURCES, size=copies, replace=True))   # same-system duplicates happen too
        for k, src in enumerate(sources):
            level = 0.3 if k == 0 else 1.0           # first copy is the cleanest
            addr, pin = v["address"], v["pincode"]
            if k > 0 and rng.random() < .2:          # vendor moved, or a works address was used
                loc, pin = LOCALITIES[int(rng.integers(len(LOCALITIES)))]
                addr = f"Plot {rng.integers(1, 400)}, {rng.choice(['Industrial Area', 'Phase 2', 'IDA'])}, {loc}, Telangana {pin}"
            phone = v["phone"] if rng.random() > .08 else str(rng.integers(6, 10)) + "".join(map(str, rng.integers(0, 10, 9)))
            bank = v["bank_account"] if (k == 0 or rng.random() > .12) else "".join(map(str, rng.integers(0, 10, 12)))
            rows.append({
                "legacy_id": None, "source_system": src, "vendor_name": mess_name(rng, v["name"], level),
                "address": mess_address(rng, addr, level),
                "pincode": pin if rng.random() > .12 else None,
                "phone": mess_phone(rng, phone), "gstin": mess_gstin(rng, v["gstin"]),
                "bank_account": bank if rng.random() > .1 else None,
                "ifsc": v["ifsc"] if rng.random() > .1 else None,
                "entity_id": v["entity_id"],
            })
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    counters = {s: 0 for s in SOURCES}
    ids = []
    for s in df["source_system"]:
        counters[s] += 1
        ids.append(f"{s[:3]}-{counters[s]:05d}")
    df["legacy_id"] = ids
    return df


def _inv_variant(rng, number: str) -> str:
    digits = number.split("/")[-1]
    return rng.choice([f"INV-{number.split('/')[1]}-{int(digits)}", digits.lstrip("0"),
                       number.replace("/", ""), f"{number.split('/')[1]}/{digits}"])


def build_invoices(legacy: pd.DataFrame, seed: int = 0, dup_rate: float = 0.03,
                   recurring_share: float = 0.15) -> pd.DataFrame:
    """12 months of AP invoices. Each invoice is posted against one legacy vendor record.
    For vendors with several records, `dup_rate` of invoices are paid a second time through a
    different record of the same vendor (planted, flagged with is_planted_dup).
    `recurring_share` of vendors bill a fixed monthly amount: legitimate repeats that a naive
    'same vendor, same amount' rule would wrongly flag."""
    rng = np.random.default_rng(seed + 1)
    rows = []
    start = pd.Timestamp("2025-04-01")
    for entity, grp in legacy.groupby("entity_id"):
        recs = list(grp["legacy_id"])
        recurring = rng.random() < recurring_share
        fixed = round(float(rng.lognormal(np.log(60000), .5)), -2)
        n_inv = int(rng.poisson(12)) + 1
        for j in range(n_inv):
            date = start + pd.Timedelta(days=int(rng.integers(0, 365)))
            amount = fixed if recurring else round(float(rng.lognormal(np.log(85000), .9)), 2)
            number = f"INV/{date.year}/{int(rng.integers(1, 9999)):04d}"
            rec = recs[int(rng.integers(len(recs)))]
            rows.append({"invoice_no": number, "vendor_legacy_id": rec, "invoice_date": date,
                         "amount": amount, "entity_id": entity, "is_planted_dup": False, "dup_of": None})
            if len(recs) > 1 and rng.random() < dup_rate:
                other = rng.choice([r for r in recs if r != rec])
                rows.append({"invoice_no": _inv_variant(rng, number), "vendor_legacy_id": other,
                             "invoice_date": date + pd.Timedelta(days=int(rng.integers(3, 26))),
                             "amount": amount, "entity_id": entity, "is_planted_dup": True, "dup_of": number})
    inv = pd.DataFrame(rows).reset_index(drop=True)
    inv["invoice_key"] = [f"AP{i:06d}" for i in range(len(inv))]
    return inv
