# V28 regroup-balance CPU screen (2026-10-02)

These are anonymous aggregates over 144 historical HF real-need snapshots used
by the prior `regroup_offline_1002` diagnostic. They contain prefix support only;
they are not current vLLM trajectory captures, GPU timings, or accuracy results.
No item identifiers, filenames, prompts, outputs, or need bits are published.

The implementation is `scripts/v28_regroup_screen.py`. It starts from natural
Q64 groups and exchanges rows between the two Q64 groups within each original
Q128 block. Heads and Q128 blocks never mix; every group retains 64 rows and the
order remains a bijection. Group support is the union of all member-row needs,
so no required prefix bit is lost. This bounded local search optimizes alias2
maximum/tail load rather than repeating the historical sort-only candidate.
The regroup idea is a cooperation direction attributed to Haowei; no peer code
is merged or copied.

Each kept-tile count produces alias2 CTA loads `floor(count / 2)` and
`ceil(count / 2)`, with a fixed CTA grid. The proxy per snapshot is
`max(max_CTA_tiles, total_tiles / 132)` in tile units. The aggregate proxy ratio
uses the sum of these per-snapshot proxies; aggregate p95 is nearest-rank p95
over pooled CTA loads. These proxies exclude selector/search, permutation,
list construction, memory locality, launch and merge costs. Canvas-tail work is
excluded (`tail_tiles=0`). The numbers cannot be interpreted as kernel time,
end-to-end speedup, or a bound on possible GPU speedup.

## Results

| Setting | Accepted snapshots | Raw candidate proxy / Q64 | Gated proxy / Q64 | Gated total tiles | Gated max / p95 CTA tiles |
|---|---:|---:|---:|---:|---:|
| Default: 4 swaps/head, 8 candidate signatures/group | 7 / 144 | 0.984770 | 0.997177 | 794491 | 294 / 111 |
| More search: 12 swaps/head, 16 candidate signatures/group | 19 / 144 | 0.966642 | 0.989376 | 794680 | 294 / 111 |

Both runs use the same conservative gate: at least 5% per-snapshot proxy
improvement, no maximum or p95 regression, at most 1% total-tile increase, and
at most 12.5% moved rows. Rejected candidates use exactly natural Q64. The
optimizer also enforces the movement budget separately for each head. Gate
rejection reason counts can overlap within a snapshot.

The common Q64 baseline has 794361 prefix tiles, 18432 CTAs, pooled maximum
294, and pooled p95 111. Its total exactly matches the prior diagnostic's
144-snapshot Q64 total. Each input passes packed-padding and original Q128
router-union validation. Thirteen CPU unit tests passed, including permutation,
head/block isolation, support preservation, gate acceptance/rejection, padding,
baseline costs and anonymous output checks.

The default gated proxy improves by 0.2823%; more search improves it by 1.0624%
while increasing total tiles by 0.0402%. This does not support advancing directly
to a separate request benchmark. A small held-permutation gather/consumer/scatter
ablation may be included in an otherwise justified identical-state native Q64
kernel qualification, with all costs charged and its own masked numerical
reference. A prospective runtime permutation selector remains unimplemented.
The finite search and historical snapshots do not rule out other regrouping
algorithms or amortization strategies.

Reports: [default.json](default.json), [moresearch.json](moresearch.json).
The more-search report is a separate sensitivity result; the default report was
not changed or overwritten. No GPU was used for this screen.
