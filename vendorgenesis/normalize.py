"""Name and address normalisation for Indian (and Arabic-origin) business names."""
from __future__ import annotations

import re
import unicodedata

# legal-form variants -> one canonical token (checked longest first)
LEGAL_FORMS = [
    (r"\bprivate\s+limited\b", "pvt ltd"), (r"\bpvt\.?\s*ltd\.?\b", "pvt ltd"),
    (r"\(\s*p\s*\)\s*ltd\.?\b", "pvt ltd"), (r"\bp\.?\s*ltd\.?\b", "pvt ltd"),
    (r"\bprivate\s+ltd\.?\b", "pvt ltd"), (r"\bpvt\.?\s+limited\b", "pvt ltd"),
    (r"\blimited\b", "ltd"), (r"\bltd\.?\b", "ltd"),
    (r"\bl\.?\s*l\.?\s*p\.?\b", "llp"),
]
LEGAL_TOKENS = {"pvt", "ltd", "llp", "private", "limited"}
NOISE_TOKENS = {"m/s", "ms", "the", "and", "&", "co", "company"}

# spelling / transliteration variants -> canonical form
TRANSLIT = {
    "mohd": "mohammed", "md": "mohammed", "muhammad": "mohammed", "mohammad": "mohammed",
    "mohamed": "mohammed", "muhammed": "mohammed",
    "shri": "sri", "sree": "sri", "shree": "sri", "sre": "sri",
    "laxmi": "lakshmi", "luxmi": "lakshmi", "lakshmy": "lakshmi",
    "venkateshwara": "venkateswara", "venkateshwar": "venkateswara", "venkatesvara": "venkateswara",
    "srinivas": "srinivasa", "sreenivasa": "srinivasa",
    "balaji": "balaji", "bhagyanagar": "bhagyanagar", "hyd": "hyderabad",
    "intl": "international", "inds": "industries", "ind": "industries", "engg": "engineering",
    "entp": "enterprises", "ent": "enterprises", "constn": "constructions", "techno": "technologies",
    "tech": "technologies", "trdg": "trading", "infra": "infrastructure", "pharma": "pharmaceuticals",
    "elec": "electricals", "agri": "agro",
}
ADDRESS_ABBR = {"rd": "road", "st": "street", "no": "number", "opp": "opposite", "nr": "near",
                "hyd": "hyderabad", "secbad": "secunderabad", "sec'bad": "secunderabad",
                "colny": "colony", "nagr": "nagar", "x": "cross", "ts": "telangana"}


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def normalize_name(name) -> str:
    """Lower-case, strip punctuation, canonicalise legal forms and transliteration variants."""
    if not isinstance(name, str):
        return ""
    s = _ascii(name).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"^\s*m\s*/\s*s\.?\s+", " ", s)          # 'M/s ' prefix (messrs) before punctuation is stripped
    for pattern, repl in LEGAL_FORMS:
        s = re.sub(pattern, f" {repl} ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    tokens = [TRANSLIT.get(t, t) for t in s.split()]
    return " ".join(tokens)


def core_tokens(norm_name: str) -> list[str]:
    """Name tokens without legal-form and noise words: what actually identifies the business."""
    return [t for t in norm_name.split() if t not in LEGAL_TOKENS and t not in NOISE_TOKENS]


def skeleton(token: str) -> str:
    """Consonant skeleton used as a cheap phonetic key: lakshmi/laxmi -> lksm."""
    t = token.lower().replace("x", "ks").replace("sh", "s").replace("ph", "f").replace("ck", "k")
    t = t.replace("w", "v").replace("z", "j").replace("q", "k")
    t = re.sub(r"[aeiouyh]", "", t)
    return re.sub(r"(.)\1+", r"\1", t)


def legal_form(norm_name: str) -> str:
    if "llp" in norm_name.split():
        return "llp"
    if "pvt" in norm_name.split():
        return "pvt ltd"
    if "ltd" in norm_name.split():
        return "ltd"
    return "other"


def normalize_address(addr) -> str:
    if not isinstance(addr, str):
        return ""
    s = re.sub(r"[^a-z0-9 ]", " ", _ascii(addr).lower())
    tokens = [ADDRESS_ABBR.get(t, t) for t in s.split()]
    return " ".join(t for t in tokens if not re.fullmatch(r"\d{6}", t))


def extract_pincode(addr) -> str | None:
    if not isinstance(addr, str):
        return None
    m = re.search(r"(?<!\d)([1-9]\d{2})\s?(\d{3})(?!\d)", addr)
    return m.group(1) + m.group(2) if m else None
