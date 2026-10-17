# Value-aware performance panel — AIME26 and RULER128K

This is a new, resumable performance panel for the non-exact value-aware
selectors. It is accuracy-first and does not reuse partial timing rows from the
LongBench campaign.

The AIME schedule is the full 30 problems × seeds 42, 43, and 44 (90 cells).
The fixed-budget reference is `MAGE_K=1728` 64-token units. The completed
control diagnostic pooled 140,100,640 kept tiles out of 260,392,800 eligible
GLOBAL tiles, or 53.8036% kept work / 46.1964% physical GLOBAL sparsity. Online
V1 and V2 thresholds must be calibrated on development trajectories and frozen
before the target run; the target arm is accepted only when its audited pooled
GLOBAL work fraction is within 0.03 of the reference. The calibration cohort is
historically examined AIME data, so any resulting accuracy comparison is
exploratory rather than a fresh confirmation.

The RULER schedule is a deterministic 104-cell subset at 128K: eight samples
from each of the 13 official tasks, generated from the pinned RULER checkout at
seed 6464. It uses the existing 8,192-token (128 KV-block) fixed budget. Online
thresholds are calibrated separately on the RULER development subset to the
measured fixed-budget GLOBAL work fraction; no threshold is copied from AIME or
from a different context length.

The arm set is:

- `dense_full_fix51994`, `dense_piecewise`, `allkept_fa4`
- `current_v31_control` (fixed-budget attention-mass qblock selector)
- `value_v1_online_discard_mass`
- `value_v2_online_preserve_mass`
- `value_v3a_singleton_delete`
- `value_v3b_approx_batch8`

The exact-greedy arm is intentionally absent. V1/V2 remain threshold-based;
V3a and approximate batch-8 are fixed-budget selectors. All arms keep the
GLOBAL-only scope, native LOCAL attention, native sampler, carry/refresh clock,
and the same FA4 consumer. No clean timing result is publishable until a
separate timing pass follows a complete audit pass.

Each arm is launched as deterministic shards. A shard is reusable only with a
matching frozen config, source hash, manifest hash, and cell hash, and only an
`attempt_complete.json` marker makes it eligible for merge. Killed workers are
left in place and receive a new attempt directory on resume.

Current status: both input pools are frozen and CPU-validated. GPU generation
is blocked in this environment because `nvidia-smi` reports no device and
PyTorch sees zero CUDA devices; `preflight.json` records this exact blocker.
No accuracy or latency number from this panel is complete until the audit
records and the official AIME/RULER scorers finish.
