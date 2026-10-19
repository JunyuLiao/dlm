# Sequential execution of the two frozen corrected HumanEval panels

The K256 panel is already running from deploy435511e4, protocol55370024. The
K128 panel is frozen from the identical executable source, protocol187719fc.
The queue attaches to the existing coordinator's PID and Linux starttime; it
does not launch a second GPU worker while that process remains alive. A recycled
PID or a zombie is not treated as the original coordinator. A missing or failed
K256 final completion marker stops the queue; it never resumes failed scientific
qualification automatically or relaxes an acceptance gate.

After qualified K256 completion the queue runs the existing independent verifier
and official rescorer to write the reference-owner certificates. It then runs
the existing K128 coordinator, including its fresh work-only calibration and all
frozen qualifications and primary passes, followed by its verifier/rescorer.
Only budget-independent reference families are linked by the source-bound study
code; selector runs remain distinct. No selector, kernel, sampler, frozen panel
protocol, source snapshot or allocation is modified by this queue.

The queue protocol binds both panel protocols, its own controller source hash,
the existing interpreter, attached PID/starttime and repository root. It writes
stage changes and progress snapshots every1800s, including actual GPU utilization
and memory plus saved request counts. It checks child termination cheaply every5s
without launching an experiment-status command. Failed children stop the queue,
preserve immutable attempts and write a failure marker. A new queue attempt may
resume completed panel/stage boundaries using existing source-bound completion
markers; completed panels must not be regenerated.

Four focused CPU checks cover exact completion/protocol bindings, changed
analysis rejection, absent-marker rejection, and recycled/zombie/absent process
handling. The queue consumes no additional model calls or GPU allowance beyond
the already frozen shared evaluation allocation. Existing per-job phase/total
budget accounting and1800s limits remain enforced by the study coordinator.
