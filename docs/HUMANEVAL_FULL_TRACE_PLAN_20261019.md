# Full diagnostic-cohort traces, before any accuracy release

The mixed-prefix pair completed400.400/400.395s. All per-arm native bookends
match, and all926 observed sampler calls, full logit fingerprints and residual
fingerprints agree. The earlier intermittent V31/V3a/residual differences remain
unresolved; no internal unequal value has yet been captured. Both new unobserved
prefixes match every original measured clean record, while a full-swap warmup
differs. This is bounded agreement, not a promoted correction.

A CPU comparison of the qualified baseline and shared K256 disk caches finds
3855 common files, no removed files, one changed FlashInfer log and40 new Triton
files. All common native compiler artifacts are unchanged. FlashInfer warmup
loads/saves zero tuning configurations; the actual MoE backend is Triton. Do not
spend another native pair assuming a disk-cache or FlashInfer-tuning explanation.

Freeze attempt002 of the mixed diagnostic. Use two identical clean engines with
the entire original80-request warmup/qualification sequence (all methods, six
questions, seeds42/43). Observe this whole first sequence with the already tested
full sampler/logit/cache observer. Append the original72 measured requests as
unobserved repeats, in the same order. No native sampler, selector, threshold,
budget, output limit, layer scope or graph setting changes. Total152 requests
per engine. Each measured case has a same-engine observed/unobserved contrast,
plus an independent observed-process contrast. This is a two-way observation
check; do not describe it as clean/observed/clean bookends. Warmup fingerprints
are descriptive and have no same-context unobserved first-use counterpart.

Require actual logit fingerprints on every observed call, complete input/output
bindings and zero measured graph captures. Inspect the first unequal logits or
sampler field, especially V3a question116,seed42 and V31/residual question129,
seed43. Evaluate observation agreement per case; if it fails, preserve and
explain the failure before treating a trace as native causal evidence. Integer
fingerprints locate a change; actual tensor comparison/replay remains required.

Two1000-second reservations initially, charged to existing phasek256, with no
total/phase budget expansion. Current closed phasek256 charge5851.700s before
this stage, leaving10348.300s under its16200s cap. The full matrices and K128
queue remain held. No favorable accuracy, speed or reproducibility claim follows
from failing to reproduce a rare difference. A later combined evaluation may
share identical dense configurations to reduce repeated cold starts, but such a
protocol must be designed/frozen and its fairness established separately.
