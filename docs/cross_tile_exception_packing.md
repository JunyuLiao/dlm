# Hopper cross-tile exception packing for BLASST

## Decision

**No-go. Stop before the Hopper WGMMA/TMA microkernel.**

The active-voter arithmetic has one apparent Phase-A pass: `tau=32`,
`PACK_M=64` reaches 79.05% weighted utilization, covers 48.08% of retained
P@V tiles, reduces counterfactual V loads by 5.25x, and adds 2.60% of baseline
QK FLOPs. Both q-head and kv-head configurations are identical because LLaDA
uses 32-head MHA rather than GQA.

That pass cannot preserve the existing BLASST approximation. When one or more
rows vote to keep a physical tile, the current kernel updates the online
softmax state and executes P@V for **all 128 rows**, including rows whose
individual vote was skip. At `tau=32`, positive voters are only 7.53% of those
rows. Packing only those voters drops the other 92.47% of contributions.

The semantics-preserving alternative is to defer all 128 rows from every
candidate tile. `PACK_M=128` then forms exactly one pack per original tile,
eliminating no V loads or P@V FLOPs. `PACK_M=64` forms two packs per original
tile and doubles V loads. It also recomputes 34.53% of baseline QK at `tau=32`,
above the 20% gate. Consequently no computation satisfying the correctness
invariant passes the preliminary gate, regardless of WGMMA/TMA efficiency.

## Existing-kernel semantics audit

For each row, the fused kernel computes:

```text
row_gap = local_max - running_max
row_vote_keep = row_gap >= log_threshold
tile_skip = max(valid row_gap) < log_threshold
```

Only `tile_skip` controls execution. If it is false, every valid row performs:

```text
next_m = max(running_m, local_max)
alpha = exp2(running_m - next_m)
P = exp2(scores - next_m)
l = l * alpha + sum(P)
O = O * alpha + P @ V
m = next_m
```

There is no per-row predicate around `P @ V` in the accepted 128-row path.
The row mask is a vote used to form the unanimous tile decision, not a mask on
individual output contributions. This is visible in
`blasst/triton_bidirectional.py` and is independently covered by
`test_retained_minority_tile_updates_negative_voters_too`.

Reconstructing a deferred contribution as
`exp(score-final_m)/final_l @ V` is valid only for rows actually deferred. It
does not make it valid to omit the negative-voter contributions that baseline
BLASST evaluates. Computing those negative-voter rows in the main kernel would
require the rejected small/irregular P@V path or the original full WGMMA, which
removes the proposed saving.

## Phase-A setup

| Item | Value |
| --- | --- |
| Model | `GSAI-ML/LLaDA-8B-Instruct` |
| Trace | Real nested LLaDA masked inputs, adjacent 90/89, 50/49, 15/14% states |
| Invocations | 192 = 32 layers x 6 states |
| Shape | sequence 4096, 32 query heads, 32 KV heads, head dim 128 |
| Original tile | 128 query rows x 64 KV rows |
| Dtype | BF16, 2 bytes |
| Total physical tiles | 12,582,912 |
| Retained tiles | 9,035,733 |
| Existing physical sparsity | 28.19% |

The target-only 90/50/15% trace was analyzed separately and reaches the same
decision. Request, head/KV-head and KV-tile boundaries are never crossed.

## Minority-tile distribution

`candidate work` is the fraction of currently retained full 128-row P@V tiles
whose positive-voter count is at most `tau`.

| tau | Candidate work | Exception rows | Mean voters/candidate tile | Extra active-only QK / baseline QK |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 7.99% | 722,005 | 1.00 | 0.045% |
| 2 | 12.96% | 1,619,819 | 1.38 | 0.101% |
| 4 | 19.57% | 3,678,319 | 2.08 | 0.228% |
| 8 | 27.77% | 8,376,320 | 3.34 | 0.520% |
| 16 | 37.35% | 18,875,272 | 5.59 | 1.172% |
| 32 | 48.08% | 41,898,967 | 9.64 | 2.601% |

Full count distributions are provided by layer, step, mask ratio, query head,
KV head, and KV tile in `active_count_marginals.csv` and
`active_count_by_layer_step_ratio.csv`.

## Packability

These are the requested counterfactual active-voter packs. Partial packs include
padded final packs.

| tau | PACK_M | Complete packs | Partial packs | Rows in complete packs | Rows in partial packs | Weighted utilization | V-load factor |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 0 | 229,906 | 0% | 100% | 4.91% | 3.14x |
| 2 | 64 | 0 | 258,963 | 0% | 100% | 9.77% | 4.52x |
| 4 | 64 | 24 | 280,682 | 0.04% | 99.96% | 20.47% | 6.30x |
| 8 | 64 | 21,967 | 298,285 | 16.78% | 83.22% | 40.87% | 7.83x |
| 16 | 64 | 156,377 | 313,636 | 53.02% | 46.98% | 62.75% | 7.18x |
| 32 | 64 | 498,855 | 329,354 | 76.20% | 23.80% | 79.05% | 5.25x |
| 8 | 128 | 83 | 299,517 | 0.13% | 99.87% | 21.84% | 8.37x |
| 16 | 128 | 30,988 | 316,136 | 21.01% | 78.99% | 42.48% | 9.72x |
| 32 | 128 | 181,962 | 331,908 | 55.59% | 44.41% | 63.70% | 8.45x |

`PACK_M=128` has zero complete packs for `tau<=4`. Utilization considering
only complete packs is 100% by definition, but those packs contain too small a
fraction of exception rows to be useful. Detailed leftover distributions are
in `leftover_histogram.csv`.

### Best active-voter configuration

For `tau=32`, `PACK_M=64`:

- 41,898,967 exception rows.
- 31,926,720 rows in full packs and 9,972,247 in partial packs.
- 11,106,409 padded row slots.
- 4,344,668 original V loads versus 828,209 packed loads.
- 71.18 GB V traffic before versus 13.57 GB after over the trace.
- 50.59 active rows per packed V load versus 9.64 per original load.
- 0.686 TFLOP recomputed QK, 2.60% of baseline QK.
- 10.73 GB indexed-Q reads over the trace.

It clears all five numerical gates in 170/192 invocations, appears in all 32
layers, and clears them in every state for 22 layers. The result is not a valid
replacement because it excludes 514,218,537 baseline non-voter row
contributions.

## Mask-ratio breakdown for tau=32, PACK_M=64

| Remaining masks | Candidate work | Utilization | Complete packs | Partial packs | V-load factor | Extra QK |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 90% | 25.52% | 79.69% | 67,392 | 44,226 | 4.40x | 2.12% |
| 89% | 25.37% | 79.77% | 67,759 | 44,010 | 4.38x | 2.13% |
| 50% | 56.10% | 81.32% | 103,418 | 58,283 | 5.20x | 3.13% |
| 49% | 56.12% | 81.38% | 103,668 | 58,177 | 5.20x | 3.14% |
| 15% | 77.41% | 75.89% | 78,588 | 62,362 | 5.98x | 2.55% |
| 14% | 76.74% | 75.83% | 78,030 | 62,296 | 5.97x | 2.54% |

## Metadata and output contention

The estimates use the proposed naturally aligned 40-byte tile record, a
32-byte expanded row descriptor, 8 bytes of count/offset storage per nonempty
key, and a 32-byte pack descriptor.

For `tau=32`, `PACK_M=64`, per invocation:

| Component | Mean |
| --- | ---: |
| Tile records | 0.86 MiB |
| Expanded row descriptors | 6.66 MiB |
| Count/offset arrays | 0.013 MiB |
| Pack descriptors | 0.132 MiB |
| Total temporary metadata | 7.67 MiB mean, 13.56 MiB maximum |

Each output row participates in 1.665 exception KV tiles on average; 36.50% of
rows receive more than one exception contribution and the observed maximum is
62. This predicts material FP32 atomic-add contention. At `tau=8`, the mean is
0.333, 7.14% of rows receive multiple contributions, and the maximum is 61.

## Semantics-preserving counterfactual

At `tau=32`, exact deferral must include 556,117,504 rows, not 41,898,967
positive voters:

| Exact path | Packs | Original/packed V loads | P@V avoided | Extra QK / baseline |
| --- | ---: | ---: | ---: | ---: |
| PACK_M=64 | 8,689,336 | 0.5x (twice as many loads) | 0 | 34.53% |
| PACK_M=128 | 4,344,668 | 1.0x | 0 | 34.53% |

This fails the 2x V-reduction gate, the P@V-saving objective, and the 20% QK
gate. Metadata expansion would also grow by 13.3x.

## Hardware phases

No packed CUDA/Triton kernel, PTX/SASS excerpt, or Nsight Compute profile is
reported. This is intentional, not missing validation: the task says to stop
before the Hopper microkernel when the preliminary method is not justified.
The only numerically passing computation violates the required output
semantics; the semantics-preserving computation fails multiple preliminary
gates. Therefore:

- WGMMA execution: not implemented or claimed.
- TMA/asynchronous K/V loading: not implemented or claimed.
- GPU exception-list pipeline: not implemented.
- Production flags/integration: not added; baseline remains unchanged.
- End-to-end speedup: not measured or claimed because there is no valid packed
  computation to benchmark.

## Reproduction

```bash
conda run -n ljy_dlm python tools/analyze_cross_tile_exception_packing.py \
  outputs/query_regrouping/row_masks_adjacent_4096.pt \
  --output-dir outputs/exception_packing/phase_a_adjacent \
  --num-kv-heads 32

conda run -n ljy_dlm python tools/plot_cross_tile_exception_packing.py \
  outputs/exception_packing/phase_a_adjacent

conda run -n ljy_dlm python -m unittest -v tests.test_exception_packing
```

The machine-readable decision and all aggregate metrics are in `summary.json`;
the plot is `phase_a_gate.png` in the same output directory.
