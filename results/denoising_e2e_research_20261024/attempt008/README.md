# Completion attempt 008

This attempt is a completion-only controller for the already frozen and
approved expanded end-to-end screen in attempt003.  It does not alter the
protocol, selectors, worker sources, question set, seeds, or arm set.  It
waits for the dedicated non-V31 long-panel supervisor (attempt006), then
launches the separately bound V31 long passes (attempt005), followed by the
CPU scoring and readout helpers bound by attempt007.  The previous completion
controller was interrupted by the controlling shell; this replacement records
that interruption separately and writes a failure receipt for any future
interruption.

Scope remains 25 questions x 2 seeds x 9 arms, with no campaign GPU or forward
ceiling.  The original nine over-limit LongBench identities remain blocked.
