# Testing

Run with `python -m pytest -q`: 35 tests, under 15 seconds. GitHub Actions runs them on every push (Python 3.11 and 3.12), then runs the benchmark.

## Coverage

| Area | What is tested |
|---|---|
| Validation | GSTIN check digit against the sample GSTIN used in GST portal documentation (27AAPFU0939F1ZV); **every** single-character substitution in positions 3 to 14 of a valid GSTIN fails the check digit (420 cases); PAN rules; 5 phone formats |
| Normalisation | 4 name-variant pairs reduce to the same core tokens (including M/s and transliterations); phonetic skeleton folds Lakshmi/Laxmi and Venkateswara/Venkateshwara; pincode extraction |
| Data | synthetic identities are valid and unique and include hard negatives; the legacy builder keeps the answer key; the registry loader handles real column names and drops struck-off companies |
| Blocking and features | reduction above 98% with pair completeness above 93%; all features present, finite and within range |
| Matching | true duplicate pairs never straddle the vendor split (no leakage); the model beats every rule; the review band behaves as designed on a hand-built example |
| Clustering | connected components; B-cubed on perfect and over-merged clusterings; golden-record survivorship fills gaps field by field and flags conflicting bank accounts |
| Outputs | masking; every golden vendor lands in exactly one of load file or fix list; every loaded GSTIN is valid; every legacy record appears in the cross reference; the CLI writes all four files and never leaks the answer key |
| Duplicate payments | 5 invoice-number formats reduce to the same sequence; recurring fixed-fee invoices are not flagged; the per-record check finds 0 cross-record duplicates while the per-vendor check finds over 90% |

## Bugs and problems found along the way

1. **The `M/s` prefix broke name matching.** Punctuation was stripped before noise words were removed, so "M/s Deccan Engg Works" became the tokens `m`, `s`, `deccan`... and never equalled "Deccan Engineering Works". Caught by `test_variants_normalise_to_the_same_core`; fixed by removing the prefix before punctuation stripping. The exact-name rule's recall rose from 0.467 to 0.542 as a side effect.

2. **The first synthetic data was too easy.** The first full run gave precision 1.000, with one feature (address similarity) doing almost all the work, because every vendor had a unique address that barely changed between copies. Real vendor masters are not like that, so the generator was changed: group companies now share a registered office (and sometimes a phone), and 20% of duplicate copies carry a moved or works address. Precision dropped to a believable 0.990 and the model had to learn from PAN, bank and name evidence instead.

3. **A feature that encoded a generator artifact.** An early "same source system" feature was removed before training: the first generator never put two copies of a vendor in the same system, so the model would have learned "same system means different vendor", which is false in real data. The generator now allows same-system duplicates too.

4. **Error analysis, not tuning, produced the best feature.** Remaining false merges were same-name companies with different valid PANs; remaining misses were duplicates where one GSTIN had a keying error. Adding PAN edit distance and a both-valid flag cut false merges from 16 to 9 and misses from 12 to 6 (see the ablation in `results/RESULTS.md`).

5. **A test assumption that was wrong.** `test_model_beats_every_rule` first required the model to beat the best rule by 0.2 F1. On the 600-vendor test fixture the best rule already scored 0.95, because a small sample has few near-namesakes. That led to the scale experiment in the results: the best rule falls from 0.925 at 500 vendors to 0.701 at 3,000, while the model holds.

6. **Readiness numbers were not comparable at first.** Legacy records scored a perfect 1.000 on consistency only because bank conflicts cannot be seen before duplicates are found, and uniqueness used the answer key. Both are now computed from the pipeline's own clusters for legacy and golden records alike.
