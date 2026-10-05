# V28 Q64 paged alias2 component qualification

Source: `e7b5ff0619e1c2727fc4af737ba991f0be5404f2`; mpk H100; isolated vLLM 0.30.0,
torch 2.13.0+cu130, FA4 own-environment paged sparse fix. 2026-10-02 19:21 UTC-5.

**Scope:** six historical prefix-support snapshots with synthetic bf16 QKV,
actual native Q/K/V strides and random physical-page order. Full canvas kept.
Each Q64/Q128 result checked against its own IEEE FP32 masked reference; all finite,
maximum relative-max error <0.006 (qualification threshold 0.02).

| Prefix tokens | Snapshots | Q64/Q128 GPU time geomean |
|---|---:|---:|
| 34816 | 2 | 0.95859 |
| 71488 | 2 | 0.94214 |
| 92736 | 2 | 0.93666 |

Overall ratio **0.94575** (about 5.4% component time reduction).
30 rotated-order repetitions per state after warmup, including alias2 lookup,
page-table/used allocation, FA4 and LSE merge. Timed held-list builds remained zero.
Excluded: selector, initial keep-map/list/split construction, KV copy, request lifecycle
and all other model work. No accuracy or end-to-end inference is supported.

The dense diagnostic is fixed-length and is **not** the native varlen serving baseline.
Its original key was mislabeled; only that key is renamed in the published report,
with this correction recorded in `dense_scope`. No sparse/dense ratio is claimed.
The real strongest dense reference remains the matched native request panel.

Worker reserved GPU time: 18.6512 s, including imports and qualification. Attempt001
is retained privately as diagnostic only (30.7184 s); its original Q stride and
sampling issue were corrected before this run. Total component attempts: 49.3696 s.
