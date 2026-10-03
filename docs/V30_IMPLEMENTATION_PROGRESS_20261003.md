## Peer build and bounded regroup screen — 2026-10-03 02:37 (UTC-5)

Root implemented the isolated build, ABI guard fix and GLOBAL selector wrapper.
Junyu b890ff494 CUDA source compiled successfully with own nvcc13.0.88 and
NVIDIA TensorRT-LLM headers796584295. Peer default flags all remain false;
fast-SFU/inline variants are not silently substituted. C++/CUDA source unchanged.
Only the Python debug buffer check now accepts both ABI3 and ABI4, preserving
shape/dtype/device/contiguity checks. Complete chosen header trees are privately
hashed; public receipt, source hashes and29.6KB compiler log are in
results/v30_20261003/junyu_build002. Preparation001 failed before compilation
on the archive root directory entry; it is retained, with0GPU seconds.
Compiler warnings include WGMMA serialization and register spills; they do not
establish a measured bottleneck. The built kernel has NOT run on GPU yet.

GLOBAL-only scope is implemented in peer_global_scope.py using a narrowed
adapter module selector, leaving LOCAL untagged and native call eligibility
unchanged. Four CPU tests cover exact module identities, pass-through arguments,
result identity, scope rejection and cleanup. ABI/build guards add three tests.
This is an HF peer-reproduction component, not a vLLM port or model parity proof.
Twenty-nine original peer kernel acceptance cases collected successfully with
CUDA hidden. Their new frozen spec rejects skips and changed parameter counts;
three queue-contract tests pass. GPU execution will follow the original formal
strict scoring barrier on idle dlm2; it is not armed in this source-freeze commit.
A matched ATen bridge and complete model/scale/scope qualification remain pending.

New independent regroup_key6 CPU prototype samples six fixed support bits,
stably groups inside each head/Q128, rebuilds exact unions, and accepts only
blocks with non-worsening total/max alias2 tile work. Five tests cover coverage,
bijection, ties, nonnested equal-count supports and adversarial/random supports.
Executed all144existing held-inventory snapshots:603/4608blocks accepted,
12.1179%rows moved, aggregate tile ratio0.997058, load-proxy sum ratio0.996582.
CPU construction median1.774874ms (range1.045137–3.961188), excluding loading.
These are prefix-only development proxies, NOT GPU/request gains. Improvement
is too small to justify claiming that row movement/build costs are covered.
Preserve raw anonymous per-snapshot values in results/v30_20261003/regroup_key_cpu001.
Haowei document snapshot pins6f7279c16 and is explicitly attributed.
