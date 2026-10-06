# Summary

The compact Triton LOCAL consumer clears the dense FA4 kernel in this held-map
qualification on the H100. It does not replace the frozen AIME26 end-to-end
panel.

| prefix | physical sparsity | dense FA4 (ms) | old FA4 Q128 (ms) | compact Triton (ms) | compact speedup |
|---:|---:|---:|---:|---:|---:|
| 187 | 14.29% | 0.05026 | 0.07055 | 0.03856 | 1.30x |
| 3500 | 37.50% | 0.05110 | 0.07035 | 0.04655 | 1.10x |
| 8192 | 39.47% | 0.05050 | 0.06998 | 0.04049 | 1.25x |

The compact output differs from the old FA4 sparse output by at most
`0.001953125` in absolute value for all three prefixes. The compact map uses
Q64 launch blocks formed by duplicating the selector's Q128 map, so the
selection and budget semantics are unchanged. A new AIME26 end-to-end panel
is still required to measure total attention and request time after integration.
