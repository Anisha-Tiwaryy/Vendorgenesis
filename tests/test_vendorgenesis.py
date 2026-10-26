import numpy as np
import pandas as pd
import pytest

from vendorgenesis.base import load_registry, synthetic_base
from vendorgenesis.blocking import candidate_pairs, pair_completeness
from vendorgenesis.cluster import bcubed, clusters_from_pairs, golden_records, review_simulation
from vendorgenesis.features import FEATURES, pair_features
from vendorgenesis.impact import compare, flag_duplicates, invoice_core
from vendorgenesis.legacy import build_invoices, build_legacy
from vendorgenesis.matcher import labels, pair_folds
from vendorgenesis.normalize import core_tokens, extract_pincode, normalize_name, skeleton
from vendorgenesis.pipeline import run
from vendorgenesis.prepare import prepare
from vendorgenesis.readiness import d365_export, mask_account, mask_phone, record_issues
from vendorgenesis.validate import (gstin_check_digit, is_valid_gstin, is_valid_pan, make_gstin,
                                    normalize_phone, pan_from_gstin)


@pytest.fixture(scope="module")
def small():
    return run(n=600, seed=1, verbose=False)


# ---------- validation ----------
def test_gstin_check_digit_on_published_example():
    # 27AAPFU0939F1ZV is the sample GSTIN used in GST portal documentation
    assert gstin_check_digit("27AAPFU0939F1Z") == "V"
    assert is_valid_gstin("27AAPFU0939F1ZV")


def test_every_single_character_typo_breaks_the_check_digit():
    g = make_gstin("ABCCD1234E")
    assert is_valid_gstin(g)
    for i in range(2, 14):
        for c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if c != g[i]:
                bad = g[:i] + c + g[i + 1:]
                assert not is_valid_gstin(bad)


def test_pan_rules():
    assert pan_from_gstin("36ABCCD1234E1Z5") == "ABCCD1234E"
    assert is_valid_pan("ABCCD1234E") and not is_valid_pan("ABCXD1234E")


@pytest.mark.parametrize("raw,expected", [("+91 98480 12345", "9848012345"), ("098480-12345", "9848012345"),
                                          ("9848012345", "9848012345"), ("12345", None), ("5848012345", None)])
def test_phone_normalisation(raw, expected):
    assert normalize_phone(raw) == expected


# ---------- normalisation ----------
@pytest.mark.parametrize("a,b", [
    ("M/s. Sri Sai Steels Pvt Ltd", "Sri Sai Steels Private Limited"),
    ("Sri Lakshmi Traders Private Limited", "SHRI LAXMI TRADERS PVT. LTD."),
    ("Mohammed Steels (P) Ltd", "Mohd Steels Pvt Ltd"),
    ("M/s Deccan Engg Works Ltd.", "Deccan Engineering Works Limited"),
])
def test_variants_normalise_to_the_same_core(a, b):
    assert core_tokens(normalize_name(a)) == core_tokens(normalize_name(b))


def test_skeleton_folds_spelling_variants():
    assert skeleton("lakshmi") == skeleton("laxmi")
    assert skeleton("venkateswara") == skeleton("venkateshwara")


def test_extract_pincode():
    assert extract_pincode("Plot 4, Road No 2, Banjara Hills, Hyderabad 500 034") == "500034"
    assert extract_pincode("no pin here 12") is None


# ---------- data generation ----------
def test_synthetic_base_identities_are_valid_and_unique():
    b = synthetic_base(400, seed=3)
    assert b["gstin"].map(is_valid_gstin).all()
    assert b["pan"].is_unique and b["name"].is_unique
    assert (b["sibling_of"] >= 0).sum() > 20          # hard negatives exist


def test_legacy_keeps_answer_key():
    b = synthetic_base(300, seed=4)
    leg = build_legacy(b, seed=4)
    assert set(leg["entity_id"]) == set(b["entity_id"])
    assert leg["legacy_id"].is_unique
    assert (leg.groupby("entity_id").size() > 1).mean() > 0.3


def test_registry_loader_handles_real_column_names(tmp_path):
    csv = tmp_path / "roc.csv"
    pd.DataFrame({
        "CORPORATE_IDENTIFICATION_NUMBER": ["U1", "U2", "U3"],
        "COMPANY_NAME": ["SRI SAI STEELS PRIVATE LIMITED", "DECCAN FOODS LIMITED", "OLD CO LIMITED"],
        "COMPANY_STATUS": ["Active", "ACTIVE", "Strike Off"],
        "REGISTERED_OFFICE_ADDRESS": ["Plot 1, Uppal, Hyderabad 500039", "Kukatpally, Hyderabad 500072", "x 500001"],
    }).to_csv(csv, index=False)
    b = load_registry(csv, n=10)
    assert len(b) == 2 and set(b["pincode"]) == {"500039", "500072"}
    assert b["gstin"].map(is_valid_gstin).all()


# ---------- blocking, features, matching ----------
def test_blocking_reduces_pairs_and_keeps_most_duplicates(small):
    bs = small["block_stats"]
    assert bs["reduction_ratio"] > 0.98 and bs["pair_completeness"] > 0.93


def test_features_are_complete_and_bounded(small):
    X = small["features"]
    assert list(X.columns) == FEATURES and not X.isna().any().any()
    for col in ["name_jw", "name_token_set", "core_tfidf", "addr_token_set"]:
        assert X[col].between(0, 1.0001).all()


def test_pair_folds_never_split_a_vendor(small):
    df, pairs = small["prepared"], small["pairs"]
    folds = pair_folds(df, pairs)
    y = labels(df, pairs)
    assert (folds[y == 1] != -1).all()                # true duplicates always share a fold


def test_model_beats_every_rule(small):
    t = small["pair_table"].set_index("method")
    # on small samples rules look fine (few name collisions); the model must still win
    assert t.loc["Gradient Boosting", "f1"] > t.iloc[:3]["f1"].max()
    assert t.loc["Gradient Boosting", "f1"] > 0.97


def test_review_band_trades_workload_for_accuracy():
    y = np.array([1, 1, 0, 0, 1, 0])
    s = np.array([.99, .6, .5, .01, .2, .97])
    auto = review_simulation(y, s, .5, .5)
    reviewed = review_simulation(y, s, .1, .98)
    assert reviewed["pairs_to_review"] == 4
    assert reviewed["precision"] == 1.0 and auto["precision"] < 1.0


# ---------- clustering and golden records ----------
def test_connected_components_and_bcubed():
    pairs = pd.DataFrame({"i": [0, 1, 3], "j": [1, 2, 4]})
    cl = clusters_from_pairs(5, pairs, np.array([True, True, False]))
    assert len(set(cl[:3])) == 1 and cl[3] != cl[4]
    perfect = bcubed(np.array([0, 0, 1]), np.array(["a", "a", "b"]))
    assert perfect["bcubed_f1"] == 1.0
    merged = bcubed(np.array([0, 0, 0]), np.array(["a", "a", "b"]))
    assert merged["bcubed_precision"] < 1.0 and merged["bcubed_recall"] == 1.0


def test_golden_record_survivorship_and_bank_conflict():
    g = make_gstin("ABCCS1234E")
    df = pd.DataFrame({
        "legacy_id": ["A", "B"], "source_system": ["EXCEL_AP", "LEGACY_ERP"],
        "vendor_name": ["Sri Steels Pvt Ltd", "SRI STEELS PRIVATE LIMITED"], "address": ["x", "y"],
        "gstin_u": [None, g], "gstin_valid": [False, True], "phone_n": ["9848012345", None],
        "pincode_f": ["500034", "500034"], "bank_n": ["111122223333", "999988887777"], "ifsc": ["SBIN0001234", None],
    })
    gold = golden_records(df, np.array([0, 0]))
    row = gold.iloc[0]
    assert row["gstin_u"] == g and row["phone_n"] == "9848012345" and row["ifsc"] == "SBIN0001234"
    assert row["bank_conflict"] and row["members"] == 2
    assert any("verify before payment" in x for x in record_issues(row))


# ---------- export and masking ----------
def test_masking():
    assert mask_account("123456789012") == "XXXXXXXX9012"
    assert mask_phone("9848012345") == "98XXXXXX45"


def test_export_is_masked_and_complete(small):
    load, fix, xref = small["load"], small["fix"], small["xref"]
    assert len(load) + len(fix) == len(small["golden"])
    assert load["BANKACCOUNTNUMBER"].str.contains("X").all()
    assert load["TAXREGISTRATIONNUMBER_GSTIN"].map(is_valid_gstin).all()
    assert set(xref["legacy_id"]) == set(small["legacy"]["legacy_id"])


# ---------- duplicate payments ----------
@pytest.mark.parametrize("num", ["INV/2025/0042", "INV-2025-42", "42", "2025/0042", "INV20250042"])
def test_invoice_core(num):
    assert invoice_core(num) == "42"


def test_recurring_fixed_fees_are_not_flagged():
    inv = pd.DataFrame({"invoice_key": ["a", "b"], "invoice_no": ["INV/2025/0001", "INV/2025/0002"],
                        "invoice_date": pd.to_datetime(["2025-05-01", "2025-06-01"]), "amount": [50000.0, 50000.0]})
    assert flag_duplicates(inv, pd.Series(["V1", "V1"])).empty


def test_dedup_exposes_cross_record_duplicates(small):
    imp = small["impact"]
    assert imp.iloc[0]["caught"] == 0
    assert imp.iloc[1]["caught"] / imp.iloc[1]["planted_duplicates"] > 0.9
    assert imp.iloc[1]["false_flags"] <= 3


def test_readiness_improves(small):
    assert small["readiness_after"]["records_ready_to_load"] > small["readiness_before"]["records_ready_to_load"] + 0.3
    assert small["readiness_after"]["uniqueness"] == 1.0


def test_cli_writes_outputs(tmp_path):
    from vendorgenesis.__main__ import main
    main(["--n", "300", "--seed", "2", "--out", str(tmp_path)])
    for f in ["d365_vendor_load.csv", "vendor_fix_list.csv", "legacy_to_new_vendor_xref.csv", "legacy_vendor_records.csv"]:
        assert (tmp_path / f).exists()
    assert "entity_id" not in pd.read_csv(tmp_path / "legacy_vendor_records.csv").columns
