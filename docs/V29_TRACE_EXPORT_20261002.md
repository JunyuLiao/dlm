# Private trace export for a new independent diagnostic

`v29_trace_profile.py` wraps the unchanged `v29_vllm_cost_profile.py`. It is a
CPU-tested, GPU-unrun preparation tool; no trace was retroactively produced for
002/003 and no GPU rerun is authorized by this document. Their real saved files
are published under the respective `raw/` directories. Old source54163 and all
its deployments/results stay unchanged.

Default trace export is off. A new committed source, private frozen diagnostic
spec and strict binding are required. The binding must pin **both** the new
trace tool and the original profiler (plus all original runner, adapter, metrics,
JIT/config/core sources). `panel.validate_binding` checks the frozen files;
original profiler/worker config, source identity, event switch and execution
validation still run before model construction. Rebind source paths only via
strict identical-byte checks; adding this wrapper does not waive old guards.
An old binding that omits the trace tool fails closed.

The new private spec adds this object; paths are placeholders, never Git data:

```json
{
  "diagnostic_only": true,
  "profile_ordinal": 1,
  "chrome_trace_export": {
    "enabled": true,
    "record_shapes": false,
    "with_stack": false,
    "own_root": "OWN_CAMPAIGN_ROOT",
    "private_directory": "NEW_PRIVATE_TRACE_DIRECTORY"
  }
}
```

The coordinator must choose the user's own dyh root and wholly new run/trace
paths. CLI directory, shape flag and ordinal must match the frozen spec exactly;
shape/stack defaults are off, stack capture is unsupported. The trace directory
must be inside the frozen own root, separate from the new run directory, with an
existing parent and no existing output/directory. Existing run/summary paths are
also rejected before GPU imports/model construction. No original run is edited.

Example for a separately frozen new run only:

```sh
python -m scripts.v29_trace_profile \
  --binding PRIVATE_NEW_BINDING --arm method --block 0 \
  --run-dir NEW_RUN --summary NEW_DIAGNOSTIC_SUMMARY \
  --query-block 128 --canvas-buffers legacy --profile-ordinal 1 --cuda-events \
  --trace-private-dir NEW_PRIVATE_TRACE_DIRECTORY
```

If shapes are explicitly frozen true, also pass `--trace-record-shapes`. Shape
recording can hold tensor references and increase overhead/memory pressure;
its performance is not a formal measurement. It captures available input shapes,
not an intentional tensor-value dump, but a Chrome trace can include private
paths, environment metadata and scalar arguments. Never publish it automatically.
It must undergo separate actual-file inspection and sanitization first. Preserve
original private traces; publish only verified sanitized copies and their public
artifact hashes. No sanitizer or automatic Git upload is implemented here.

Export happens once after the selected request's existing device-wide boundary
and profiler closure. The trace is first written to a newly allocated private
temporary file, then installed with an exclusive hard link: an existing or
late-arriving output is never overwritten. Cleanup removes only the allocated
temporary inode, not user files or earlier traces. On failure the new private
directory remains as a failed-attempt marker; relaunch uses a new directory.
No new per-step sync, graph mode, attention math, sampler or method is added.

Output: `chrome_trace.private.json` and `trace_receipt.private.json`, marked
private/diagnostic/nonformal and not automatically published. CUDA/CUPTI export,
shape memory effects and actual trace arguments remain runtime-unqualified.

CPU validation:9 new checks plus16 unchanged-profiler checks pass via
`python -m unittest tests.test_v29_trace_profile tests.test_v29_vllm_cost_profile -v`.
They cover frozen CLI/ordinal/shape agreement, dual-tool pinning, output boundaries,
early and late overwrite rejection, temporary cleanup, warm selection, one-time
post-boundary export, failed-request behavior and restoration of patched classes.
