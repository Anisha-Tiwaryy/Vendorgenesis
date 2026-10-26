# VendorGenesis

**Vendor master deduplication and ERP-readiness scoring before a Dynamics 365 go-live.**

Every ERP implementation hits the same wall during data migration: the client's vendor list lives in Tally, an Excel accounts-payable sheet and an old ERP, and the same supplier is entered five different ways. Load that as-is and the new ERP inherits duplicate vendors, split spend history, unreliable vendor scores and duplicate payments.

VendorGenesis takes the messy vendor records from several legacy systems and produces:

1. one **golden record per real vendor**, with every merge decision scored and auditable;
2. a **load file shaped like the Dynamics 365 Finance "Vendors V2" entity**, with bank and phone numbers masked;
3. a **fix list** of vendors that cannot be loaded yet, with the exact reason for each;
4. a **legacy-to-new vendor cross reference** for reconciliation after go-live;
5. a measured view of **duplicate payments** that only become visible once duplicates are merged.

## Results (3,000 vendors, 4,848 legacy records, seed 0)

Full tables in [`results/RESULTS.md`](results/RESULTS.md). Every number is produced by `python evaluate.py`.

| | Result |
|---|---|
| Blocking | 56,047 candidate pairs instead of 11.7 million (99.52% fewer), keeping 98.7% of true duplicate pairs |
| Pair matching (held-out vendors) | Gradient Boosting F1 **0.997** (9 false merges, 6 misses) vs best rule F1 0.701 (2,576 false merges) |
| Human review | Reviewing 46 borderline pairs cuts false merges from 9 to 4 and misses from 6 to 3 |
| Vendor clusters | B-cubed F1 **0.997** (3,011 vendors found for 3,000 real ones) |
| ERP readiness | Records ready to load rise from **17.9%** of legacy records to **64.4%** of golden records |
| Duplicate payments | **518 of 527** planted duplicate payments exposed (Rs 5.78 crore of Rs 5.94 crore), 1 false flag; the same check run per legacy record finds 0 |
| Stability | Gradient Boosting pair F1 0.996 to 0.997 across 3 seeds |

### Two findings worth more than the headline

**Rules that look fine on a sample collapse on the full vendor master.** Similar names multiply as the vendor count grows (Sri Lakshmi Traders, Sri Lakshmi Textiles, Sri Lakshmi Steels in the same locality). The best name rule scores F1 0.925 on 500 vendors and 0.701 on 3,000; the model stays at 0.990 to 0.997.

| Vendors | Best rule F1 | Rule false merges | Model F1 | Model false merges |
|---|---|---|---|---|
| 500 | 0.925 | 51 | 0.990 | 1 |
| 1,000 | 0.863 | 214 | 0.993 | 4 |
| 2,000 | 0.725 | 1,077 | 0.996 | 5 |
| 3,000 | 0.701 | 2,576 | 0.997 | 9 |

**Error analysis led to the most important feature.** The first model's mistakes fell into two groups: same-name companies with two different valid PANs (wrongly merged), and true duplicates where one GSTIN had a single keying error (wrongly split). A plain "PAN equal" feature cannot separate these, because a typo also changes the PAN. Adding the **edit distance between the PANs inside the GSTINs** plus a "both GSTINs pass the check digit" flag cut Gradient Boosting false merges from 16 to 9 and misses from 12 to 6, and lifted Logistic Regression from F1 0.885 to 0.960. It is now the top feature by permutation importance.

## How it works

```
legacy systems ─► validate & normalise ─► blocking ─► pair features ─► match model ─► review queue
                                                                                          │
   D365 load file ◄── readiness checks & masking ◄── golden records ◄── vendor clusters ◄─┘
                                                          │
                                       duplicate-payment check per real vendor
```

| Stage | What it does |
|---|---|
| `validate.py` | GSTIN format and **check digit** (GSTN algorithm), PAN embedded in GSTIN, phone, pincode, IFSC |
| `normalize.py` | Legal-form canonicalisation (Private Limited, Pvt. Ltd., (P) Ltd, P Ltd), M/s prefix, transliteration folding (Sri / Shri / Sree, Lakshmi / Laxmi, Mohammed / Mohd / Muhammad), abbreviations (Engg, Inds, Infra), consonant-skeleton phonetic key |
| `blocking.py` | Four keys (pincode + first token, first two tokens, PAN, phone); reports reduction ratio and pair completeness |
| `features.py` | 16 pair features: Jaro-Winkler, token-set, token-sort and partial ratios, character 3-gram TF-IDF cosine, PAN edit distance, GSTIN / phone / pincode / bank agreement, address similarity, legal form |
| `matcher.py` | Three rule baselines vs Logistic Regression and Gradient Boosting; **2-fold split by vendor** so a model is never tested on a vendor it trained on; out-of-fold scores for every pair |
| `cluster.py` | Connected components, B-cubed scoring, review-band simulation, golden-record survivorship (valid GSTIN, then completeness, then source trust; field-level gap filling) |
| `readiness.py` | Completeness, validity, uniqueness and consistency scores; Dynamics 365-shaped export; fix list; masking |
| `impact.py` | Same-amount, same-invoice-sequence, 45-day-window duplicate check, run per legacy record and per real vendor |

**Bank-account conflicts are never auto-resolved.** If two records of one vendor carry different bank accounts, the golden record goes to the fix list with "verify before payment". Silently picking one is how payment fraud gets into a new ERP.

## Data

- **Real company base (recommended):** the Registrar of Companies company master published on [data.gov.in](https://www.data.gov.in/resource/registrars-companies-roc-wise-company-master-data). Download the Telangana (RoC Hyderabad) file into `data/raw/` and pass it with `--registry`. Active companies with a pincode in the address are kept and sampled; names and addresses are real. Registry files have no tax or bank details, so a valid PAN, GSTIN (with correct check digit), phone, bank account and IFSC are generated for each company.
- **Built-in base:** without a registry file, a Telangana-style company universe is generated (Hyderabad localities and pincodes, regional business-name patterns). 15% of companies get a near-namesake in the same locality, half of them sharing the office address and some sharing a phone line, the way group companies do.
- **Legacy mess (both cases):** each real vendor appears 1 to 3 times across LEGACY_ERP, TALLY and EXCEL_AP with legal-form variants, transliteration variants, abbreviations, typos, dropped words, case changes, the M/s prefix, reformatted phones, missing or mistyped GSTINs, moved addresses and changed bank accounts. An answer key (`entity_id`) records which records are the same vendor; it is used only for scoring and is dropped from every output file.
- **Invoices:** 12 months of AP invoices. For vendors with several records, 3% of invoices are paid again through another record of the same vendor with a reformatted invoice number. 15% of vendors bill a fixed monthly amount, so "same vendor, same amount" alone is not treated as a duplicate.

## Run it

```bash
pip install -r requirements.txt

python evaluate.py                                   # full benchmark -> results/RESULTS.md
python evaluate.py --registry data/raw/roc_hyderabad.csv

python -m vendorgenesis --out outputs                # load file, fix list, cross reference
python -m vendorgenesis --registry data/raw/roc_hyderabad.csv --out outputs

streamlit run app.py                                 # interactive app with an adjustable review band
python -m pytest -q                                  # 35 tests
```

## Limitations

- **The duplicates are simulated.** Names and addresses can be real (registry mode), but the legacy copies and their errors are generated, so results show the method works on these kinds of errors, not that it will score 0.997 on any client's data. Real vendor masters will have error types not modelled here.
- **Rates are assumptions.** The 45% duplicated-vendor share and the 3% duplicate-payment rate are choices for the simulation, not industry statistics. Read the rupee figure as the size of the mechanism in this scenario.
- **The load file is shaped like, not tested against, Dynamics 365.** Column names follow the Vendors V2 entity naming style; it has not been imported into a live Dynamics instance, and real projects map vendor groups, tax groups and posting profiles per client.
- **Remaining hard cases:** two different companies whose names differ only by legal form (Pragathi Technologies Ltd vs Pragathi Technologies Private Ltd) when neither has a GSTIN; group companies sharing an office and a first name; records with heavy typos and no GSTIN on either side. These are exactly the cases the review band exists for.
- Connected components can chain a single wrong link into a larger merged cluster; a production version would add a cluster-coherence check before accepting large clusters.

See [TESTING.md](TESTING.md) for the test plan and the bugs the tests caught.
