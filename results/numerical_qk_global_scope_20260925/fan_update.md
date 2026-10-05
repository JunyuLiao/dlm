# Update for Fan (v12, GLOBAL-only scope)

**Done**
- Closed the all-layer v11 result: the Hopper consumer works (bit-identical to fresh T on its own support), but the
  historical selector dominates the cost, and all-layer M1 stays ~20% more expensive per call than native.
- New controlled scope: temporal methods only on the 5 GLOBAL layers; all LOCAL layers are the original native
  attention in every arm. Dispatch was verified per step (25 native LOCAL, 5 routed GLOBAL calls).
- Arms: D_native, T_G (fresh Junyu T), G1 (M1), G3 (M3, R=2), B8_G (frozen anchor bitmap). 54 complete AIME runs
  in total (40 panel + 10 extension + 4 audit twins), with actual per-canvas denoising counts recorded.

**Result**
- In this scope the per-call cost of every method is within +2-6% of native (v11 all-layer: +20%).
- Quality is identical on the 4-question panel (3/4 each). Over 6 development questions: T_G 4/6, G3 4/6,
  D_native 3/6, G1 3/6, B8 3/6.
- M3 (G3) is the fastest arm: 0.78x native summed time, 0.76x T_G, via shorter trajectories.
- **M1's numerical history shows no increment over the simple frozen bitmap B8**: same quality, B8 is cheaper,
  and B8 skips more tiles.
- **GLOBAL attention is only ~3% of a denoising step**, so faster GLOBAL kernels cannot produce a meaningful E2E
  speedup. Any E2E benefit here comes from trajectory length (fewer denoising calls), which varies strongly per
  question and cannot yet be attributed to the method.

**Missing evidence**
- A larger frozen question set with seeds, to test whether G3's shorter trajectories are systematic.
- M1 vs B8 on quality at scale.
- Per-request trajectory analysis (why calls change).
- Other models: LLaDA2.1-mini weights are not local (download needed); I-DLM has no code here.

**Interface needed from the team**
- To make GLOBAL-only worthwhile: remove the ~6-7 ms/step fixed overhead of the T/binding path (T bookkeeping plus
  hooks), or target longer contexts where GLOBAL attention is a larger share of the step.
- Grouping (Haowei) is NOT integrated.
- Decision requested: run a frozen larger panel (for example the remaining AIME26 validation set, 2 seeds) comparing
  D_native, T_G, G3 and B8 to settle whether M3-style bitmap holding shortens trajectories systematically, before
  any further kernel work.
