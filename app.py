"""VendorGenesis: vendor master deduplication and ERP-readiness, as a Streamlit app."""
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from vendorgenesis.cluster import review_simulation
from vendorgenesis.matcher import pair_folds
from vendorgenesis.pipeline import run

st.set_page_config(page_title="VendorGenesis", layout="wide")
st.title("VendorGenesis")
st.caption("Vendor master deduplication and ERP-readiness before a Dynamics 365 go-live")

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Optional: data.gov.in RoC company master CSV", type="csv")
    n = st.slider("True vendors", 500, 3000, 1500, step=250)
    seed = st.number_input("Seed", 0, 99, 0)
    go = st.button("Run pipeline", type="primary")


@st.cache_data(show_spinner=False)
def cached_run(path, n, seed):
    return run(path, n, seed, verbose=False)


if go or "res" in st.session_state:
    if go:
        path = None
        if upload is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(upload.getvalue())
            path = tmp.name
        with st.spinner("Validating, blocking, matching, clustering"):
            st.session_state["res"] = cached_run(path, n, seed)
    r = st.session_state["res"]
    df, imp = r["prepared"], r["impact"]
    after = imp.iloc[1]
    gb = r["pair_table"].set_index("method").loc["Gradient Boosting"]

    c = st.columns(5)
    c[0].metric("Legacy records", f"{len(df):,}")
    c[1].metric("Golden vendors", f"{len(r['golden']):,}")
    c[2].metric("Match F1 (held out)", f"{gb['f1']:.3f}")
    c[3].metric("Ready to load", f"{r['readiness_after']['records_ready_to_load']:.0%}",
                f"from {r['readiness_before']['records_ready_to_load']:.0%}")
    c[4].metric("Duplicate payments exposed", f"{after['caught']}/{after['planted_duplicates']}",
                f"Rs {after['caught_value'] / 1e5:,.1f} lakh")

    t1, t2, t3, t4, t5 = st.tabs(["Legacy data", "Matching", "Review queue", "Golden records", "Duplicate payments"])
    with t1:
        st.dataframe(r["legacy"].drop(columns=["entity_id"]).head(300), width="stretch", hide_index=True)
        st.dataframe(r["validation"], hide_index=True)
        bs = r["block_stats"]
        st.write(f"Blocking kept **{bs['candidate_pairs']:,}** of {bs['all_pairs']:,} pairs "
                 f"({bs['reduction_ratio']:.2%} reduction) and **{bs['pair_completeness']:.1%}** of true duplicate pairs.")
    with t2:
        st.dataframe(r["pair_table"].round(4), width="stretch", hide_index=True)
        st.dataframe(r["cluster_table"].round(4), width="stretch", hide_index=True)
        if r["importances"] is not None:
            st.bar_chart(r["importances"].head(8))
    with t3:
        keep = pair_folds(df, r["pairs"]) != -1
        lo, hi = st.slider("Review band (auto-reject below, auto-accept above)", 0.0, 1.0, (0.05, 0.98), 0.01)
        d = review_simulation(r["y"][keep], r["scores"][keep], lo, hi)
        a, b, c3 = st.columns(3)
        a.metric("Pairs for a person to review", d["pairs_to_review"])
        b.metric("Precision", f"{d['precision']:.4f}")
        c3.metric("Recall", f"{d['recall']:.4f}")
        p = r["pairs"][keep].assign(score=r["scores"][keep])
        band = p[(p.score >= lo) & (p.score < hi)].sort_values("score", ascending=False).head(50)
        show = pd.DataFrame({"score": band["score"].round(3),
                             "record A": df.loc[band["i"], "vendor_name"].to_numpy(),
                             "record B": df.loc[band["j"], "vendor_name"].to_numpy(),
                             "GSTIN A": df.loc[band["i"], "gstin_u"].to_numpy(),
                             "GSTIN B": df.loc[band["j"], "gstin_u"].to_numpy()})
        st.dataframe(show, width="stretch", hide_index=True)
    with t4:
        g = r["golden"]
        pick = st.selectbox("Inspect a merged vendor", g.loc[g["members"] > 1, "vendor_account"].head(500))
        ids = g.loc[g["vendor_account"] == pick, "member_legacy_ids"].item().split(";")
        st.dataframe(df[df["legacy_id"].isin(ids)][["legacy_id", "source_system", "vendor_name", "address", "gstin",
                                                    "phone", "bank_account"]], width="stretch", hide_index=True)
        st.subheader("Dynamics 365-shaped load file (masked)")
        st.dataframe(r["load"].head(200), width="stretch", hide_index=True)
        st.download_button("Download load file", r["load"].to_csv(index=False), "d365_vendor_load.csv")
        st.subheader("Fix list")
        st.dataframe(r["fix"], width="stretch", hide_index=True)
        st.download_button("Download fix list", r["fix"].to_csv(index=False), "vendor_fix_list.csv")
        st.download_button("Download legacy-to-new cross reference", r["xref"].to_csv(index=False), "xref.csv")
    with t5:
        st.dataframe(imp, width="stretch", hide_index=True)
        st.caption("Same rule both times: same amount, same invoice sequence number, within 45 days. "
                   "The duplicate-payment rate is a simulation assumption.")
else:
    st.info("Pick a size in the sidebar and press **Run pipeline**. Upload the data.gov.in RoC company master CSV "
            "to run on real company names; otherwise a Telangana-style synthetic base is used.")
