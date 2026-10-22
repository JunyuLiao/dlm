# Optimized native comparison, attempt006

Frozen source f7318fa3, selected V2 library 200354cb919fb6b8 and resident FA4
consumer. V31 long-context selector storage and warm-up limits correct the
preserved attempt005 qualification OOM. Nine selector GPU tests pass.

This attempt qualifies all five arms and four workload profiles, calibrates V2
using full-request held GLOBAL physical work, runs full separate audit/clean
coverage, then officially scores. No benchmark is complete until final_complete.json.
AIME26 30 x seeds42/43/44, MATH500500 x42, HumanEval164 x42 use V31K128;
LongBench-v2 503 x42 uses V31K8192. Target calibrated held sparsity40-50%.

Raw inputs/generations remain private. Prior attempts remain partial and immutable.
Service dlm-optimized-refresh-20261022-a006.service; monitor state.json and coordinator.log
at the user-requested15-minute interval. See protocol.json for exact fingerprints.
