# Alias2 merge and regroup writeback component001

Sourcee07aec657, three historical accepted supports, synthetic native-stride QKV,
same-host H100, alias2, frozen seed2903.32 rotated repetitions/warm8 each.
All Torch LSE and masked IEEE FP32 oracles passed. Merge is not bit-exact.

| Length bin | Natural standard ms | Natural fused ms | Held fused ms | Held fused / natural fused |
|---|---:|---:|---:|---:|
| 32768 | 0.363232 | 0.328144 | 0.330000 | 1.005656 |
| 65536 | 0.544928 | 0.504592 | 0.493776 | 0.978565 |
| 98304 | 0.445552 | 0.402592 | 0.406864 | 1.010611 |

Regroup incremental geometric ratio0.998177: approximately tied, mixed signs,
without confidence inference from three selected states.64K alone is0.978565,
but first construction and online signal generation are not in held timing.
Do not advance this costly held-search algorithm to requests on this evidence.
Offline CPU reconstruction is1.09-1.43seconds per state; it is not a viable
online implementation or evidence that a future GPU selector has that cost.

Standard merge fusion alone reduces natural component spans7.4-9.7%. This is
a separate implementation lead, not a regroup contribution. Apply it to natural,
main and eligible all-kept alike, then qualify real requests and accuracy.
First identity creation/allocation must remain inside request costs. The new
adapter opt-in has CPU tests only at this point; this component used direct calls.

Reserved51.200860GPU seconds including CPU selection/import/JIT/oracles;
offline selection36.475206seconds. No end-to-end or model-quality claim.
CUDA event spans include host dispatch; do not add timings from other hosts.
