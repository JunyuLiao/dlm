# Optimized refresh follow-up: aime26_k256

User requests retaining achieved HumanEval/AIME results and completing old MATH
before stopping ahead of LongBench. This is a new frozen attempt. Runtime/model,
selected attention kernels and native sampler are unchanged. V31 budget is
256; scope is GLOBAL only and LOCAL/prefill stay dense.

MATH recalibration uses32 full-request physical-work strata of the examined500
cohort, lowers V2 threshold and targets actual held/whole GLOBAL sparsity40-50%.
This is transductive computation-only calibration. AIMEK256 changes only V31;
existing V2/dense/all-kept controls are explicitly referenced, never overwritten.
RULER uses the same104 sampled prompts (8 per13 tasks), seed42, original
thinking-off/prefilled prompts and budgets. Calibrate V2 to approximately matched
K8192 first-canvas physical kept work; disclose any remaining mismatch and actual
sparsity (expected above85% on sparse phases, not assumed for whole request).

Every new arm/profile needs fresh audit/clean trajectory agreement, exact receipt
and zero captures/JIT/extra forwards/LOCAL skips before full coverage. Warm every
input before clean timing. Charge all copies/index construction/launches. Score
officially; use paired problem-clustered intervals and13-task clusters for RULER.
Keep prior full panels and all failed/calibration attempts immutable and private
raw inputs/generations out of Git. Process-status checks are every15minutes.
