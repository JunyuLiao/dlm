# Conditional qualification recovery: 2026-10-19

The first cold-cache H100 qualification engine used344.047 seconds, including
model/compile/kernel warmup. The original qualification allocation is900 seconds,
with a separate900-second recovery reserve inside the same14,400-second cap.
This addendum authorizes only completing missing qualification families if the
original coordinator terminates qualification as budget_limited_partial.
No launch is allowed while that coordinator is active, after a structural
validation failure, or after the recovery reserve is exhausted.

Preserve attempt001 and every completed/partial engine. A new attempt002 inherits
the exact native protocol, immutable execution snapshot, inputs and model pins;
its continuation receipt identifies the parent and this addendum. Completed
qualification engines are verified and referenced, not rerun. Missing families
use the recovery allocation, unchanged inputs/warmups/audit checks, and the
existing mutable compilation cache. The original scientific plan, candidate
gates, screen5400/confirmation3600/repair3600 limits, and total cap are unchanged.
No automatic borrowing, budget expansion, selector modification, or older
campaign resume. If recovery cannot qualify all families, screening does not run.

The recovery coordinator is committed before any recovery generation. It does
not edit the original source or protocol. Its own hash and this addendum's hash
are recorded in continuation.json. Later screen/confirmation/repair use the
original frozen coordinator against the completed qualification receipt.
