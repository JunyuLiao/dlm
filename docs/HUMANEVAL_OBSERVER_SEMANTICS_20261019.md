# Exact observer semantics for remaining HumanEval diagnosis

The frozen trajectory_worker.py used by mixed attempts001 onward snapshots GPU
sampler tensors with indexed detach().clone() before and after each native
compiled sampler call. It also reads/clones the default CPU/CUDA generator state.
It retains these snapshots and computes their CPU SHA256 fingerprints only in
the request-finished engine-step hook. It does not transfer every sampler tensor
to CPU on each step. Do not describe its generic effect as per-step CPU copying
or establish synchronization as the cause from that description.

The separate residual_repro_worker.py digest() does detach().cpu().contiguous()
inside production residual cache/held-correction hooks when active and corrects
is true. That additional transfer applies to the direction-guarded residual
instrumentation, not every fully traced method. Neither observer adds a native
model forward or changes the Python sampler formula.

Graph-only mode in graph_fingerprint.py computes integer activation/logit
fingerprints and retains their small GPU results until request finish. It does
not keep full sampler-state snapshots or read generator state, and does not
activate residual cache tracing. Graph wrappers add actual GPU operations,
allocation/lifetime effects and host dispatch overhead; independence from those
effects must be tested with the declared pure native bookends. Their absence
cannot be assumed because the fingerprints contain only integers.

Earlier wording about avoiding the full sampler observer's per-step CPU tensor
copies was imprecise. This interpretation correction does not modify frozen
protocols, runtime source, failed attempts or qualification criteria. Matching
fully observed trajectories with differing pure repeats supports investigating
observation/context sensitivity; it does not identify an asynchronous race,
synchronization, buffer lifetime or allocator mechanism as the cause.
