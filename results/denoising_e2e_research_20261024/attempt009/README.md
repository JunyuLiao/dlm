# Completion attempt 009

Completion-only replacement for attempt008. The host process namespace does
not expose the original supervisor PID to a detached controller, so this
version waits on the immutable per-arm completion receipts and fails only on a
bound nonzero receipt or a four-hour absence of any new completion artifact.
It does not launch or repeat any arm until all eight non-V31 long-arm markers
exist. The protocol and all method/scoring sources remain those frozen in
attempt003 and the bound attempts005--007.
