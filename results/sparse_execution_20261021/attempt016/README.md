# KV128 QK bundle, exact sequential KV64 decisions, 2026-10-21

Synthetic-only source variant. Q64 per CTA; each of two WGs computes one of
two adjacent KV64 QK products, then shared32KB raw scores support two ordered
Gaussian32 decisions. Q64 query stays resident in shared memory. K's128KB
workspace is reused as fullV64KB plus raw scores after QK completes. Full PV
only executes admitted tiles, with separate retained-only output normalization.

Initial f14b2b6c build fails output qualification on irregular tail geometry:
retained P reconstruction lacked explicit Q/K bounds. Corrected170c9695 adds
those bounds; no failed attempt overwritten. The candidate spills140-144bytes,
so it remains an experimental negative unless numerical and timing gates pass.
No native benchmark, workload calibration or execution-speed claim follows.

qualification002 with170c9695 still fails output qualification. The design is
unqualified and abandoned; neither build is timed or promoted. The initial
tail-bounds fix did not resolve the remaining output mismatch. Preserve both
failed numerical attempts instead of treating compilation as correctness.
