# Independent V31 value-selector study

Independent implementation of the requested V1/V2/V3 formulas, using the inherited V31 reuse lifecycle and unchanged FA4 consumer. This study is in correctness and H100 qualification; no target accuracy or end-to-end claim is available.

The control is documented in SOURCE_NOTE.md. Runtime and intended protocol are in config.json. Online thresholds remain unfrozen until development-only calibration completes. RULER and HumanEval generation require the corresponding frozen private manifests and cell schedules. All raw generation and runtime caches stay in ignored private/cache directories below this result root. Failed attempts are retained. A study completion marker is deliberately absent.

V3b exact scores every candidate after every deletion; batch8 scores once per batch and is an approximation. Sticky retention is explicitly represented as a log-risk bonus for online arms and a log-removal penalty for deletion arms. The zero-sticky mathematical oracles implement the formulas directly.
