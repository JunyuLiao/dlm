# Completed optimized sparse-attention benchmark results

See summary.md for accuracy, paired uncertainty, clean timing and sparsity.
summary.json contains all five controls/variants; latency_and_trajectories.json
contains N,N/C,T,S/N,S,W and audit per-phase refresh/held attention timing.
physical_work.json gives GLOBAL/LOCAL/overall, first-canvas and phase tile counts;
work_mismatch.json discloses V2/V31 physical work differences.
paired_uncertainty.json includes V31, corrected FULL and all-kept references.
source_identity.json and verification.json bind all generations and evidence to
source/runtime/kernel/model/protocol identities. final_complete.json verifies
this completed report, not the old partial four-workload campaign.

HumanEval/AIME original panels are preserved. MATH V2tau0.2 rerun reaches
47.60% whole and49.36% held GLOBAL sparsity. AIMEV31K256 is additional.
RULER104 uses original balanced13-task prompts, seed42, thinking-off prefill,
K8192 and V2tau0.021. No LongBench generation after the user's stop instruction.
No dense speedup established. All raw inputs/generations remain private;
negative designs/calibration attempts remain preserved.
