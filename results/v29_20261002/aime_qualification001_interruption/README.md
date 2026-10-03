# AIME qualification interruption

Frozen source1931: dense qualification completed normally (process exit0, one expected/completed timed request after warm-up). Complete supervisor process-reserved GPU time was239.994347 seconds; the worker's narrower self-reported span was237.866 seconds.

The private supervisor rejected a transient post-exit NVML idle state before starting the next arm. The remaining three arms did not launch. This was a wrapper idle-check failure, not a dense model/numerical failure. All original artifacts are retained; dense is reused without rerunning. No full scored qualification or formal benchmark is claimed.

The new private continuation permits at most60 seconds of no-compute-PID cleanup waiting, checking every2 seconds and requiring two consecutive single-card UUID/memory<128MiB/utilization0 observations. Any active compute PID immediately rejects; timeout preserves the attempt. Four CPU toy cases cover transient success, active PID rejection, timeout rejection and failed-worker stop. Source, binding, seed and8192 budget remain unchanged.
