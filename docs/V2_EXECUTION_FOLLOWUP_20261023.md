# V2 execution optimization and requested follow-up, 2026-10-23

Preserve the completed optimized_refresh_results_20261023 panel and every old
partial attempt. Optimize execution without changing the Gaussian32 ordered
online selector, retained-block PV arithmetic, inherited reuse lifecycle, native
sampler, dense LOCAL layers, prefill or commits. Yuhan owns the inherited reuse
implementation; the value router and subsequent execution variants are distinct.

Explore immutable-prefix projection/packing reuse within a canvas, exact CTA
placement changes, and long-context shared-router alternatives. Compare masks
and outputs with the selected 200354cb919fb6b8 kernel. Reject numerical changes
from an output-preserving implementation claim. Preserve negative candidates.
Held steps must retain resident GPU metadata and avoid selectors, list building,
and KV copying. Compare identical Q/K/V and masks when attributing latency.

Freeze the selected combined implementation in a new attempt after CUDA parity,
changing-input graph tests, native audit/clean agreement, finite outputs, exact
receipts, zero extra forwards/LOCAL skips/timed captures/JIT events pass. Warm
compilation and graphs outside timing, separate audit from clean measurements.

Requested evaluation (the user's clarification supersedes short calibration):

- AIME26: V31 K1920, 30 questions x seeds42/43/44, thinking on.
- MATH500: V31 K384, all500 questions, seed42, thinking on.
- RULER128K: recalibrate only V2 to40-50% all-step GLOBAL physical sparsity,
  from actual eligible/kept counts; same104 sampled problems and seed42. Keep
  original thinking-off/prefilled task configuration. Report held, refresh,
  initial, carried, first-canvas and whole-request work separately. Calibration
  is computation-only and transductive; no accuracy-based threshold selection.
- LongBench-v2: all503 questions, seed42, V31 K8192 and V2 threshold0.021.
  This is explicit new authorization; old paused LongBench attempts stay intact.
  Do not run dense baseline panels for LongBench.

Preserve existing HumanEval/AIME/MATH V2 results. Changes to runtime/source are
disclosed when comparing historical references; they are not matched clean
timing controls. Score officially with paired/item-clustered uncertainty. Charge
all mask/index construction, copies, launches and synchronization. Report N,
N/C,T,S/N,S,W plus attention-phase times and physical work; no speed inference
from sparsity or fewer calls. Inspect running experiment processes at15-minute
intervals, with durable logs/state and resumable immutable attempt leaves.
