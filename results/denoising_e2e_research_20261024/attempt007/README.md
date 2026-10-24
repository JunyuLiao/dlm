# Completion controller for the approved finite end-to-end screen

Wait for attempt006's8non-V31 long arms to finish or fail, then execute the
unchanged attempt005 V31 long audit/clean family on the idle GPU. Run CPU-only
official scoring and descriptive analysis after all9arms complete. Same approved
25questions x2seeds x9arms; no selector/input/seed/cap change, no method promotion
or adaptive outcome-based scheduling. One H100 worker; no campaign GPU/forward
ceiling; existing4-hour no-progress child watchdogs remain. A failure is preserved
and the controller exits instead of silently retrying or replacing coverage.

The controller checks the positively identified existing attempt006 supervisor
and fails if it exits before its finite family is complete. It does not modify
or signal another workload. Public status contains counts only; process logs and
private inputs/completions remain ignored. Final publication/document review is
still performed by the primary agent; no push.
