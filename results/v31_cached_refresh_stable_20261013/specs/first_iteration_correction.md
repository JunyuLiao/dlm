# First-iteration interpretation correction

The predecessor root remains frozen. Its 16-arm panel has complete execution records and official scores, but the reported receipt checks did not test the intended selector semantics deeply enough.

| Finding | Verified source behavior | Consequence |
|---|---|---|
| Dense layout | `_cached_prepare` filled a [H,QB,PT,128] allocation through a [H,QB*128,PT] view | Cached row and tile identities were scrambled. |
| Bounded CUDA no-op | Placeholder candidate writes run even when no outside candidate is available | An ostensibly fixed-K update can clear a held tile. Previous receipts did not assert actual maps at each iteration. |
| Cache identity | `_cached_refresh` overwrote `cache['z']` with retained fresh data | Outside tiles could use last-observed rather than immutable last-dense statistics. |
| Schedule | Periodic due was ORed with progress due; the dense branch could consume progress first | Frequency arms were not independent tests of the intended clocks. |
| Dense count | Receipts summed `dense_refresh_count=1` over final live states | 770 is an end-state count, not a verified request-lifetime refresh total. |
| C_gate | One sensitivity scalar was broadcast to every KV candidate in a unit | It could not distinguish candidates; this was not a valid sensitivity-ranking test. |
| IoU | Map comparison used current and selected maps | This is consecutive-map IoU, not IoU against V31. |
| Validation | Native error fields were null; replay used synthetic tensors | Neither real next-call output perturbation nor real trajectory cache errors were qualified. |
| Equivalent arms | Global top-K used the same global ranking regardless of M | M=160/256/512 were equivalent algorithm configurations. Small score differences need a repeatability floor. |

The measurements remain valid descriptions of executed code. They do not conclusively reject bounded swaps, decay, value information, or query-sensitive ranking as ideas. Corrected raw global re-ranking is the simple lead reference, not an established winner.

This study uses an aggregate trajectory gate separately on LongBench-v2 and AIME26. Some individual call-count increases do not alone reject a candidate. The 154-cell previously examined development panel can be completed without a 503-cell expansion; expansion is a later decision. Clean timing remains paused.
