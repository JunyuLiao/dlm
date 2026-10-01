# 96K pooled panel: E6 + E6b (11 items x seeds 404-909 = 66 cells per arm)

- Items: every LongBench-v2 item of the frozen 96K pool (rendered tokens in [84000, 104000), first 24 by sha256(id),
  12 standard-pool ids excluded) whose prompt fits one H100 with dense prefill (<= 95,074 tokens): E6 ran pool
  positions 1-12's 6 fitting items, E6b positions 13-24's 5 fitting items. LongBench-v2 has 32 items in this bin,
  14 of which fit; the other 3 fitting items are outside the frozen 24-item pool.
- Same 7 arms, substrate piecewise_v5, seeds, and one host per cell in both panels. Pooling concatenates the two
  scored.csv files (distinct items) and passes all five ledgers to `scripts.v27_fa4_panel_summary`; ratios are
  paired per cell, CIs bootstrap over the 11 items.
- E6: 252/252 ok (dllm, mpk), protocol v27_lb96k_confirm_e6_8ab3aa35d1874ca1, deploy v27_e6_8f9f188.
- E6b: 210/210 ok (dllm, mpk, dlm2), protocol v27_lb96k_extend_e6b_d1b1c0c74a1dc69d, deploy v27_e6b_b805351
  (one later code commit, c36b1f933, adds the opt-in V-term options proj_rank and risk_value, which are off in all
  these arms); no timed new graphs; c0 receipts 5 bootstrap-dense calls per request.
