# Optimized native comparison, attempt007

Frozen sourcec31224ae uses the selected V2 library200354cb919fb6b8 and resident
FA4 consumer. All prior attempts remain preserved as partial.

User-fixed V2 thresholds: HumanEval0.5600716206066294 (saved K128 work-matched
calibration) and LongBench-v2 0.021. No new calibration on those workloads.
Only AIME26/MATH500 calibrate to actual held GLOBAL sparsity40-50%.
Observed work and mismatch are reported for all workloads.

Qualification reuses only explicitly referenced source/runtime/input/threshold-
identical completed pairs from attempt006; changed-threshold V2 pairs run anew.
All full benchmark generation is fresh in this attempt. AIME26 30xseeds42/43/44,
MATH500500x42, HumanEval164x42 use V31K128; LongBench503x42 uses V31K8192.

Raw inputs/generations stay private. No completion before final_complete.json.
Service dlm-optimized-refresh-20261022-a007.service; status checks every15minutes.
