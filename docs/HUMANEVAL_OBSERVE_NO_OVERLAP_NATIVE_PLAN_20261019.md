# Qualify observation-only FA4 overlap correction on native HumanEval

Completed overlap screen001 sourcec7e7c67e runs8192 operator calls across two
independent engines (55.959/50.147s), reversing factor order. On residual input
shape K3532,Q256,H16,D512, automatic16-split observing/overlap=True differs in
1/256 and3/256 trials; single-split observing/overlap=True differs2/256 in each
engine. First automatic change is2322 BF16 elements, one head/query64 tile,
max0.0625. Q/K/V are bit-identical and auxiliary attention mass matches native
in all observing trials. All2048 observing overlap=False calls are stable and
match native mass byte-for-byte; all4096 plain calls are stable in this screen.
Retain raw first/changed tensors. This supports the observing overlap pipeline
as a reproducibility mechanism, not an exact instruction-level race diagnosis.

The named runtime correction passes intra_wg_overlap=False only to the existing
mass-observing FA4 calls. Bind wrappers after the native mask fix; plain FA4
arguments remain unchanged. Cover both direct observing_dense and the adapter's
auxiliary paged observation entry. Preserve current Q/K/V, fixed selector
budgets, layer scope, thresholds, anchors/reuse clocks and all native sampler
Python/settings/graphs. No model/sampler forward is introduced. Counters are CPU
only; actual library configuration False is already checked in the matched GPU
factorial. Per-request receipts bind policy source and observation-call count.
The same-state mass equality supports retaining selector definitions; attention
output floating-point evaluation changes and receives a new numerical identity.

Require four source-bound CPU checks: exact-once call/return preservation and
sole overlap argument change; untouched plain arguments; rejection of unknown
auxiliary/conflicting paths; deferred adapter binding, shared-dispatch receipt
capture and per-request reset. Do not import the observer before the native fix.
The corrected policy must survive the shared worker's receipt rebinding for all
control, online, full-row and refinement arms.

Then freeze two independent native152-request engines,1000s each. Preserve the
original80-request observed prefix (8 warmups,72 paired cells: six methods/control
bookends,six questions,seeds42/43); repeat all72 measured cells unobserved. Full
sampler/residual observer and pure repeats retain their previous definitions.
No layer29 forensic probe or full tensor capture is enabled. Compare original
and pure output hashes,N,C,T,emission intervals and captures within each engine;
compare full traces and pure metrics independently across engines. Require actual
numerical-policy receipts on every row and exercised observing calls. Native
same-method repeatability/observer equivalence remains a strict qualification
gate; zero differences in one pass alone is insufficient.

Any failure preserves unresolved status and holds both matrices. If qualification
passes, require corrected audit/clean, scope/budget and dense/all-kept controls
before full accuracy. The optimized native dense FULL path remains the headline
reference and does not use this observer. New campaign source/protocol identities
must disclose this correction and its measured cost. Recheck V2 first-canvas work
on the corrected path and recalibrate from work only if needed; K128 gets its own
calibration. Use official HumanEval scoring and independent rescoring. Do not
reuse earlier unqualified method rankings or infer speed from fewer calls.

Total new native ceiling2000s is charged to the existing phasek256 allocation,
inside the unchanged43200s total. Preserve all expired and failed attempts and
all older paused campaigns. Run one worker, resumable immutable attempts. No full
accuracy matrix is released automatically by this diagnostic controller.
