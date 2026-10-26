"""Run the full benchmark and write results/RESULTS.md plus sample output files.

    python evaluate.py                                  # synthetic Telangana-style base
    python evaluate.py --registry data/raw/roc.csv      # real data.gov.in RoC company master
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vendorgenesis.cluster import review_simulation
from vendorgenesis.features import FEATURES
from vendorgenesis.matcher import evaluate_pairs, pair_folds
from vendorgenesis.pipeline import run

OUT = Path("results")


def md(df: pd.DataFrame, digits: int = 3) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.astype(object).itertuples(index=False):   # astype(object) keeps ints as ints
        out.append("| " + " | ".join(
            ("n/a" if isinstance(v, (float, np.floating)) and np.isnan(v) else
             f"{v:,.{digits}f}" if isinstance(v, (float, np.floating)) and not float(v).is_integer() or isinstance(v, float) and abs(v) < 10
             else f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)) for v in r) + " |")
    return "\n".join(out)


def rupees(x: float) -> str:
    if x == 0:
        return "Rs 0"
    return f"Rs {x / 1e7:,.2f} crore" if x >= 1e7 else f"Rs {x / 1e5:,.2f} lakh"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    r = run(args.registry, seed=0)
    df, pairs, X, y, s = r["prepared"], r["pairs"], r["features"], r["y"], r["scores"]
    keep = pair_folds(df, pairs) != -1

    # ablation: what the PAN-distance features (added after error analysis) contribute
    base_feats = [f for f in FEATURES if f not in ("pan_distance", "both_gstin_valid")]
    abl, _ = evaluate_pairs(df, pairs, X, y, features=base_feats, rules=False)
    abl["features"] = "without PAN-distance features"
    full = r["pair_table"][r["pair_table"]["method"].isin(["Logistic Regression", "Gradient Boosting"])].copy()
    full["features"] = "with PAN-distance features"
    ablation = pd.concat([abl, full])[["features", "method", "precision", "recall", "f1", "false_merges", "missed_duplicates"]]

    bands = []
    for lo, hi in [(0.5, 0.5), (0.3, 0.85), (0.1, 0.95), (0.05, 0.98), (0.02, 0.99)]:
        d = review_simulation(y[keep], s[keep], lo, hi)
        acc = d["accepted"]
        bands.append({"auto-reject below / auto-accept from": f"{lo} / {hi}", "pairs reviewed": d["pairs_to_review"],
                      "precision": d["precision"], "recall": d["recall"],
                      "false_merges": int((acc & (y[keep] == 0)).sum()), "missed_duplicates": int((~acc & (y[keep] == 1)).sum())})

    # stability across seeds (different vendors, different mess)
    seed_rows = []
    for sd in range(args.seeds):
        rs = r if sd == 0 else run(args.registry, seed=sd, verbose=False)
        gb = rs["pair_table"].set_index("method").loc["Gradient Boosting"]
        best_rule = rs["pair_table"].iloc[:3]["f1"].max()
        imp = rs["impact"].iloc[1]
        seed_rows.append({"seed": sd, "records": len(rs["prepared"]), "GB pair F1": gb["f1"], "best rule F1": best_rule,
                          "B-cubed F1": rs["cluster_table"].iloc[-1]["bcubed_f1"],
                          "dup payments caught": f"{imp['caught']}/{imp['planted_duplicates']}"})

    # scale: name collisions grow with the vendor count, rules degrade, the model holds
    scale_rows = []
    for n in [500, 1000, 2000, 3000]:
        rn = r if n == 3000 else run(args.registry, n=n, seed=0, verbose=False)
        t = rn["pair_table"].set_index("method")
        scale_rows.append({"true vendors": n, "legacy records": len(rn["prepared"]),
                           "best rule F1": t.iloc[:3]["f1"].max(), "best rule false merges": int(t.iloc[1:3]["false_merges"].min()),
                           "Gradient Boosting F1": t.loc["Gradient Boosting", "f1"],
                           "GB false merges": int(t.loc["Gradient Boosting", "false_merges"])})

    bs = r["block_stats"]
    rb, ra = r["readiness_before"], r["readiness_after"]
    readiness_tbl = pd.DataFrame([{"dimension": k, "legacy records": rb[k], "golden records": ra[k]} for k in rb])
    imp = r["impact"]
    after = imp.iloc[1]
    fix_counts = r["fix"]["issues"].str.split("; ").explode().value_counts().rename_axis("issue").reset_index(name="vendors")

    # error examples (held-out, after review)
    p = pairs.assign(y=y, s=s)[keep]
    acc = review_simulation(y[keep], s[keep], 0.05, 0.98)["accepted"]
    errs = []
    for _, row in p[acc & (p.y == 0)].head(4).iterrows():
        a, b = df.iloc[int(row.i)], df.iloc[int(row.j)]
        errs.append(f"- false merge (score {row.s:.3f}): `{a.vendor_name}` vs `{b.vendor_name}`; PAN distance "
                    f"{X.loc[row.name, 'pan_distance']}, same address: {a.addr_n == b.addr_n}")
    for _, row in p[~acc & (p.y == 1)].head(3).iterrows():
        a, b = df.iloc[int(row.i)], df.iloc[int(row.j)]
        errs.append(f"- missed duplicate (score {row.s:.3f}): `{a.vendor_name}` vs `{b.vendor_name}`; GSTINs "
                    f"{a.gstin_u} / {b.gstin_u}")

    lines = [
        "# VendorGenesis results", "",
        f"Data: **{r['source']}**. Generated by `python evaluate.py`; every number below is measured.", "",
        "## Input", "",
        f"- {len(r['base']):,} true vendors spread across 3 legacy systems as **{len(df):,} vendor records**; "
        f"{(df.groupby('entity_id').size() > 1).mean():.1%} of vendors appear more than once.",
        f"- {len(r['invoices']):,} AP invoices over 12 months, including {int(r['invoices']['is_planted_dup'].sum())} planted duplicate payments.", "",
        "Field validation on the raw records:", "", md(r["validation"]), "",
        "## Blocking", "",
        f"- Candidate pairs: **{bs['candidate_pairs']:,}** of {bs['all_pairs']:,} possible "
        f"(reduction ratio {bs['reduction_ratio']:.2%}).",
        f"- Pair completeness: **{bs['pair_completeness']:.1%}** of true duplicate pairs survive blocking "
        "(the recall ceiling for everything after).",
        f"- New pairs contributed by each key: {bs['new_pairs_by_key']}; oversized blocks skipped: {bs['oversized_blocks_skipped']}.", "",
        "## Pair matching (held-out vendors only, 2-fold split by vendor)", "", md(r["pair_table"], 4), "",
        "## Scale: why rules that work on a sample fail on the full vendor master", "",
        "More vendors means more near-namesakes (same prefix, same locality, different business). "
        "Fuzzy-name rules merge them; the model uses PAN, bank and address evidence instead.", "",
        md(pd.DataFrame(scale_rows), 4), "",
        "## Ablation: PAN-distance features", "",
        "Added after error analysis showed false merges between same-name companies with two different valid PANs, "
        "and misses where one GSTIN had a single keying error.", "", md(ablation, 4), "",
        "## Human review queue (held-out pairs)", "",
        "Pairs scoring between the two thresholds go to a reviewer (simulated with the answer key).", "", md(pd.DataFrame(bands), 4), "",
        "Remaining errors after review:", ""] + errs + ["",
        "## Vendor clusters (B-cubed)", "", md(r["cluster_table"], 3), "",
        "## ERP readiness", "", md(readiness_tbl, 3), "",
        f"- **{len(r['load']):,}** golden vendors pass every check and go into the Dynamics 365-shaped load file; "
        f"**{len(r['fix']):,}** go to the fix list.", "",
        "Fix list by issue:", "", md(fix_counts), "",
        "## Duplicate payments", "",
        "Same check (same amount, same invoice sequence number, within 45 days), run per legacy record vs per deduplicated vendor:", "",
        md(imp.assign(planted_value=imp["planted_value"].map(rupees), caught_value=imp["caught_value"].map(rupees))), "",
        f"Deduplication exposes **{after['caught']} of {after['planted_duplicates']}** planted duplicate payments "
        f"({rupees(after['caught_value'])} of {rupees(after['planted_value'])}) with {after['false_flags']} false flag(s). "
        "The duplicate-payment rate is an assumption of the simulation (3% of invoices for duplicated vendors), "
        "so read the rupee figure as the mechanism's size in this scenario, not a real-world estimate.", "",
        "## Stability across seeds", "", md(pd.DataFrame(seed_rows), 4), "",
        "## Feature importance (permutation, held-out pairs, drop in PR-AUC)", "",
        md(r["importances"].round(4).rename_axis("feature").reset_index(name="importance").head(8), 4), "",
        f"Full run time for seed 0: {r['seconds']:.0f}s.",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    r["load"].head(200).to_csv(OUT / "sample_d365_vendor_load_masked.csv", index=False)
    r["fix"].head(200).to_csv(OUT / "sample_fix_list.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
