# Summary

The compact Triton LOCAL consumer clears the dense FA4 kernel in this held-map
qualification on the H100. It is a kernel result only; it does not replace the
frozen AIME26 end-to-end panel.

| prefix | physical sparsity | dense FA4 (ms) | old FA4 Q128 (ms) | compact Triton (ms) | compact speedup |
|---:|---:|---:|---:|---:|---:|
| 187 | 14.29% | 0.05078 | 0.07101 | 0.03911 | 1.30x |
| 3500 | 37.50% | 0.05201 | 0.07137 | 0.04646 | 1.12x |
| 8192 | 39.47% | 0.05251 | 0.07178 | 0.04059 | 1.29x |

The compact output differs from the old FA4 sparse output by at most
`0.001953125` in absolute value for all three prefixes. The compact map uses
Q64 launch blocks formed by duplicating the selector's Q128 map, so the
selection and budget semantics are unchanged. A new AIME26 end-to-end panel
is still required to measure total attention and request time after integration.
