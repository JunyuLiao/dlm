# Prior-work overlap and the unresolved contribution

Bounded check of the three requested primary sources (27 September 2026); this is an abstract-level mechanism comparison, not a comprehensive novelty review.

| Primary source | Relevant overlap | What the present experiment must still establish |
|---|---|---|
| [MAGE, arXiv:2602.14209v2](https://arxiv.org/abs/2602.14209v2) | Exact first-step observation followed by within-block reuse of selected KV indices; block-wide support sharing can hurt sparse estimation. | Periodic M1 redecision using historical numerical scores and current projected V must beat a matched held-map control. All-MASK alignment in MAGE's model families is not established for this native DiffusionGemma run. |
| [LoSA, arXiv:2604.12056](https://arxiv.org/abs/2604.12056) | Different queries inflate the union of accessed KV pages; token stability supports temporal reuse. LoSA reuses prefix-attention results for stable tokens. | The present method always recomputes retained current QK/V output. Its history/geometry interaction must justify extra selector work without borrowing LoSA's stale-output mechanism or its reported gains. |
| [FG-Attn, arXiv:2509.16518v2](https://arxiv.org/abs/2509.16518v2) | Fine hardware-executable sparse tiles and overhead/occupancy tradeoffs are already studied in video diffusion. | Smaller tiles, old GQA grouping, or copy removal alone are not independent novelty. Native language-model adaptive quality, complete-forward cost and request benefit need direct evidence here. |

The testable hypothesis is narrower than “fine sparsity is novel”: row-sensitive historical decisions might retain useful selectivity at a physical granularity whose higher refresh cost can be amortized by M3. Required controls are native, original fresh T, matched-geometry fresh T, M1, coarse M3 at identical numerics, fine M3 and matched B. Until that comparison exists, the numerical/layout work is platform repair and the geometry screen is an opportunity measurement. No method advantage or paper claim is established yet.
