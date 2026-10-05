# v15 post-panel attribution plan (fixed before any panel output; optional, budget permitting)

- **Counted data first.** `scripts/v15_phase_accounting.py` is fed by the panel receipts:
  - per-canvas calls, canvases, anchors/ordinary/fallback/planner counts, and metadata bytes;
  - common-native-workload and realized-history counterfactual estimates, priced with the v14 LongBench step profiles and labeled as estimates.
- **At most two predetermined panel states** for a separate diagnostic:
  - `longbench_v2/66f9625fbb02136c067c5456` (10–15K bin, first in panel order) and `longbench_v2/66f2aac2821e116aacb2a9de` (15–20K bin, first in panel order);
  - seed 17, canvas index 1;
  - run with `scripts/v14_adaptive_canvas.py`: identical native-captured canvas-start state, each arm to the native stop, counting executed/eligible tiles, restorations and churn.
- **These are diagnostic twins, not formal trajectories.** They start from a native-captured state, so they are labeled separate and never spliced into the formal accounting.
- **Not planned:** new gates, rank sweeps, dense recovery, forced acceptance, or A2.
