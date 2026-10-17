# Paused by explicit user request

Pause completed2026-10-17 09:49:41 UTC (17:49:41 Hong Kong).
The queue was stopped at09:43:02 UTC with SIGSTOP on coordinator PID2023384
(process start_ticks90532357). No queued job was allowed to launch after that.
The already-running worker PID2062543 (start_ticks92129066),
`runs/audit/dense_piecewise/shard003/attempt001`, finished naturally with OS exit0.
Its records passed the original frozen validation/export rules. Its completion
receipt records the separate finalizer and an upper-bound reservation time.
The stopped coordinator was then terminated without resuming its queue.

There are17 completed full-panel engine jobs. Both passes have complete matched
coverage for64 questions across all methods; audit dense PIECEWISE additionally
has shard003 complete. All generations, frozen source, protocol and prior
partial/failed attempts are preserved. No full503 worker is running.

Evidence: `PAUSED_USER_REQUEST.json`, `pause_records/pause001/`, `state.json`.
The exact original commands and process identities are retained privately in
`private/pause_20261017.json`. Source ec87d397; protocol SHA256
30c01e85953b0d9115347ab2b5c885ae5ab5b814d84b28bb2a5744650f0fe022.

## Resume only after a new explicit user authorization

Do not invoke the frozen coordinator directly while this pause is pending.
The old frozen coordinator predates the pause guard and does not read this
marker. Use the new explicit authorization wrapper only after the user agrees:

```
/home/exouser/dyh/v31_env_20261005/bin/python \
/home/exouser/ljy/dlm/scripts/v31_value_e2e_resume.py \
--study /home/exouser/ljy/dlm/results/v31_value_e2e_longbench503_20261017/native_attempt001 \
--user-authorized-resume \
--authorization-note 'Record the new explicit user authorization here'
```

The wrapper records authorization, acquires a process lock, and executes the
original frozen command with its frozen PYTHONPATH. Completed17 jobs are hash-
verified and reused. Remaining shard003 jobs and subsequent shards continue;
no completed generation is rerun. A failed incomplete worker gets a new attempt
directory. The original pause receipt stays intact. Small investigation jobs
are separately authorized under results/v31_value_investigation_20261017/ and
cannot authorize or trigger this resume.
