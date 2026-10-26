"""python -m vendorgenesis [--registry roc.csv] [--n 3000] [--seed 0] [--out outputs]"""
import argparse
from pathlib import Path

from .pipeline import run


def main(argv=None):
    ap = argparse.ArgumentParser(description="Deduplicate a vendor master and produce an ERP load file.")
    ap.add_argument("--registry", default=None, help="data.gov.in RoC company master CSV (optional)")
    ap.add_argument("--n", type=int, default=3000, help="number of true vendors to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--unmasked", action="store_true", help="write bank/phone unmasked (internal use only)")
    a = ap.parse_args(argv)

    r = run(a.registry, a.n, a.seed)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    load = r["load"]
    if a.unmasked:
        from .readiness import d365_export
        load = d365_export(r["golden"], masked=False)[0]
    load.to_csv(out / "d365_vendor_load.csv", index=False)
    r["fix"].to_csv(out / "vendor_fix_list.csv", index=False)
    r["xref"].to_csv(out / "legacy_to_new_vendor_xref.csv", index=False)
    r["legacy"].drop(columns=["entity_id"]).to_csv(out / "legacy_vendor_records.csv", index=False)
    print(r["pair_table"].round(4).to_string(index=False))
    print(r["impact"].to_string(index=False))
    print(f"wrote {len(load)} load rows, {len(r['fix'])} fix rows to {out}/")


if __name__ == "__main__":
    main()
