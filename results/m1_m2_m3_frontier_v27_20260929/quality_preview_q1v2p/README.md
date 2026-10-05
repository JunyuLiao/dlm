# v27 quality preview (small subset before the full quality panel)

Protocol `quality_fa4pw_q1v2_preview` (`../specs/v27_quality_fa4pw_q1v2_preview.json`):
every 4th AIME26 problem (8) and every 3rd LongBench-v2 item (4), seed 101, the 10 arms of the planned
quality panel, substrate `piecewise_v2`, one H100 (dllm). Screening only. Per-cell scores (no text) are
in `cells.csv`.

**Infrastructure failure.** `aime26/17` failed in every arm with `model prefill masks are not plain causal /
sliding-causal`. The piecewise_v2 prefill-mask probe always used 1500 tokens, so every 257-1499-token prompt
failed its shape check. The failure hits all arms alike. Fixed in `v27_long.py` (the probe is capped at the
prompt length) with a CPU test. AIME below therefore has 7 scored problems.

## Paired against D_fa4_allkept (correct / scored)

| arm | AIME (7) | +/- | LB (4) | +/- | AIME runs hitting the 8192-token cap |
|---|---:|---|---:|---|---:|
| D_fa4_allkept | 4 | | 3 | | 3 |
| D_native | 4 | +1/-1 | 3 | 0/0 | 3 |
| M1_R1_A8_fa4 (Fan M1) | 4 | +1/-1 | 3 | 0/0 | 2 |
| M2c_R1_A8_fa4 (Fan M2) | 4 | +1/-1 | 1 | 0/-2 | 2 |
| M3_R3_A8_fa4 (Fan M3) | 3 | 0/-1 | 3 | 0/0 | 2 |
| B_A64_fused_rp_async_fa4 | 2 | 0/-2 | 1 | 0/-2 | 5 |
| M3_R6_A64_fused_rp_async_fa4 | 3 | 0/-1 | 3 | 0/0 | 3 |
| M3_R6_A64_fused_dp_async_fa4 | 3 | +1/-2 | 3 | 0/0 | 3 |
| M1_R1_A64_fused_dp_async_fa4 | 6 | +2/-0 | 3 | 0/0 | 1 |
| M3_R6_A64_fused_dp_async_gate8k_fa4 | 4 | 0/0 | 3 | 0/0 | 3 |

- The two exact dense paths already differ by +1/-1 on 7 AIME problems, so single swings are noise.
- **Warning: B.** B (hold the bootstrap map for the whole canvas) loses 2 on AIME and 2 on LB with no gains. Four
  discordant pairs all go against B (two-sided sign test p = 0.125). On AIME the losses are runs that stop
  terminating and hit the 8192-token cap.
- M1 with a fresh decision every step (R1 DP) is the only arm without a loss. The arms that hold a map longer (B,
  then M3 R6) lose more, which points at map staleness rather than sparsity alone. This is a hypothesis to test, not
  a finding: `v27_sparsity_fidelity.py` measures kept attention mass and output error per held call.
- The gate8k arm runs dense on AIME (prefixes rarely exceed 8K) and reproduces dense exactly.
- **Decision.** The full quality panel is not launched with this arm set. It waits for the fidelity sweep
  (`../specs/v27_dev_fidelity_v6.json`: M3 R6 vs R12, threshold shifts 0/-ln2/-2ln2, B, plain M1/M3).
