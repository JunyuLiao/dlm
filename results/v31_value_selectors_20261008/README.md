# Independent V31 value-selector study

Independent implementation of the requested V1/V2/V3 formulas, using the inherited V31 reuse lifecycle and unchanged FA4 consumer. The five new selectors and four matched controls passed model qualification; development calibration is next. No target accuracy or end-to-end claim is available.

The control is documented in SOURCE_NOTE.md. Runtime and intended protocol are in config.json. Online thresholds remain unfrozen until development-only calibration completes. RULER and HumanEval generation require the corresponding frozen private manifests and cell schedules. All raw generation and runtime caches stay in ignored private/cache directories below this result root. Failed attempts are retained. A study completion marker is deliberately absent.

V3b exact scores every candidate after every deletion; batch8 scores once per batch and is an approximation. Sticky retention is explicitly represented as a log-risk bonus for online arms and a log-removal penalty for deletion arms. The zero-sticky mathematical oracles implement the formulas directly.

The original development protocol is restored in
protocol_original_s1_20261008.json. calibration_reference.json contains a
superseded 15-item pilot and must not be used for threshold calibration. The
existing 32-item schedule at seeds 1 and 2 was reconstructed from supplied
cells without changing prompts and matches its historical frozen hash.
Kernel measurements after memory refinement are in qualification/, with
invalidated memory fields explicitly documented. Each model attempt freezes
source and manifest hashes. Audit requests add CUDA events, phase counts,
finite-output checks, and sampled private Q/K/V snapshots; clean requests
disable these diagnostics. Native LOCAL eligibility and the overall rectangle
denominator are geometry-derived from actual calls, not CUDA CTA counts.

The frozen execution protocol is `protocol_execution_attempt002_20261008.json`.
The first execution preparation freeze is retained and superseded; no generation
used it. Source `01f37fe3` passes63 focused H100 tests. Clean timing also disables
the inherited GPU tile counters; a bitwise paged-consumer test verifies identical
outputs. Complete target audit and clean passes each cover503 cells per arm.
Audit-only receipts record native denoising graph modes, phase GLOBAL/LOCAL
geometry and sampled private QKV. Native dense module timing inside FULL graphs
is not available from this geometry audit. Whole-model and offline masked-operator
diagnostics are reported with their respective scope and merge-rounding limits.
