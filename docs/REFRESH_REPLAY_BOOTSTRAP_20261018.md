# Replay bootstrap correction

Source review before either isolated replay launched found that the standalone
entrypoint did not call the native benchmark's FA4_LOCAL_FIX bootstrap. The
GLOBAL arithmetic is not intended to change under that LOCAL-window patch, but
the loaded module/source identity must still match the captured native engine.
Preserve both original replay leaves as skipped_unstarted, with zero GPU charge.
Do not edit their frozen source or substitute their identity.

Use scripts/v31_refresh_replay_bootstrap.py in a new immutable corrected replay
subpanel. It installs the exact original benchmark patch before importing the
observer, checks that observer and native interface reference the patched SM90
module/class, and records its source hash. It then executes the byte-identical
operator replay implementation from attempt001. The private inputs and completed
capture receipt are hash-bound. Two independent replay processes each have at
most 300 seconds, using the existing 600-second replay phase and shared 2700-second
campaign cap. No recovery reserve or additional model requests are needed.
All original repeat counts, inputs, comparisons and gates remain unchanged.
