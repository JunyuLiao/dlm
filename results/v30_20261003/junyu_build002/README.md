# Isolated Junyu kernel CPU build

Successful compilation, no GPU execution or E2E claim. Fixed peer CUDA source
b890ff494; official NVIDIA header pin796584295; nvcc13.0.88. Peer default
fast_sfu/inline_roles/online_ratio flags all false. No CUDA source changes.
The isolated Python wrapper corrects its diagnostic ABI guard to accept3and4.

[Receipt](receipt.json),[source hashes](source_hashes.json),[full sanitized compiler log](compiler.log),
[header inventory counts](header_counts.json). Full source/header manifests and
private compiler command remain in the own build directory. The log records
WGMMA serialization/spill warnings; no measured performance attribution yet.
Preparation001 failed on an archive root-directory check before compilation;
its original evidence remains private. This attempt uses a new directory.
The29-case acceptance spec is separate; collection passed with CUDA hidden.
