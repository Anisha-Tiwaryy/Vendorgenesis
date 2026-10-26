"""Indian vendor master field validation: GSTIN (with check digit), PAN, pincode, phone, IFSC."""
from __future__ import annotations

import re

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_RE = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
PAN_RE = re.compile(r"^[A-Z]{3}[ABCFGHLJPT][A-Z][0-9]{4}[A-Z]$")
IFSC_RE = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")
TELANGANA_STATE_CODE = "36"


def gstin_check_digit(first14: str) -> str:
    """GSTN check digit: alternate weights 1,2 over base-36 values, sum quotient and remainder."""
    total = 0
    for i, ch in enumerate(first14):
        product = CHARS.index(ch) * (1 if i % 2 == 0 else 2)
        total += product // 36 + product % 36
    return CHARS[(36 - total % 36) % 36]


def is_valid_gstin(gstin) -> bool:
    if not isinstance(gstin, str):
        return False
    g = gstin.strip().upper()
    return bool(GSTIN_RE.match(g)) and gstin_check_digit(g[:14]) == g[14]


def pan_from_gstin(gstin) -> str | None:
    """Characters 3-12 of a GSTIN are the holder's PAN. Returned even if the check digit fails,
    because a typo in the check digit does not change who the vendor is."""
    if not isinstance(gstin, str):
        return None
    g = gstin.strip().upper()
    pan = g[2:12] if len(g) == 15 else None
    return pan if pan and PAN_RE.match(pan) else None


def is_valid_pan(pan) -> bool:
    return isinstance(pan, str) and bool(PAN_RE.match(pan.strip().upper()))


def normalize_phone(phone) -> str | None:
    """Reduce '+91 98480 12345', '098480-12345' etc. to the 10-digit mobile number."""
    if not isinstance(phone, str):
        return None
    digits = re.sub(r"\D", "", phone)
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    return digits if len(digits) == 10 and digits[0] in "6789" else None


def is_valid_pincode(pin, state_prefixes: tuple[str, ...] = ("50",)) -> bool:
    """Telangana pincodes start with 50. Pass state_prefixes=() to accept any 6-digit pincode."""
    if pin is None:
        return False
    p = str(pin).strip().split(".")[0]
    if not re.fullmatch(r"[1-9][0-9]{5}", p):
        return False
    return not state_prefixes or p.startswith(state_prefixes)


def is_valid_ifsc(ifsc) -> bool:
    return isinstance(ifsc, str) and bool(IFSC_RE.match(ifsc.strip().upper()))


def make_gstin(pan: str, state: str = TELANGANA_STATE_CODE, entity: str = "1") -> str:
    first14 = f"{state}{pan}{entity}Z"
    return first14 + gstin_check_digit(first14)
