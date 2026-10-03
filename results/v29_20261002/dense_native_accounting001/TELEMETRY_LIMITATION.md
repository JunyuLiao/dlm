# Physical sparsity is not recorded in these diagnostic requests

The original003 method receipt explicitly marks `per_tile_statistics` as
unavailable under minimal telemetry. Its counters establish that sparse FA4,
observation and DP routing executed, but do not provide a measured per-layer,
per-step physical keep fraction for this request. Do not substitute historical
HF/panel sparsity percentages to predict this request's compute or wall saving.

Source: [original method receipt](../cost32k_events003/raw/method.records.jsonl).
The future homogeneous component protocol uses a predeclared synthetic mask and
reports its actual kept fraction; it is not a measurement of this model request.
Any added real-model density telemetry needs a separately labelled diagnostic
whose overhead remains outside formal performance conclusions.

## Observation storage is material even when QK is shared

`cached_executor.allocate_summary` stores FP32 `z` and32-dimensional `mu`,
plus int8 `active` and `bad`, for each query row and complete prefix tile.
For one batch,16 query heads,256 query rows and32768 prefix tokens in64-token
tiles, this is `16 * 256 * 512 * (4 + 32*4 + 1 + 1)` =281,018,368 bytes
=268 MiB per GLOBAL layer, or1340 MiB (1.30859375 GiB) across five layers.
That calculation excludes DP-state tensors, output/tail tensors, projections,
copies and allocator overhead. FP32 storage does not imply FP32 dot products.

The original003 method receipt reports `summary_peak_bytes=1569751040` as
its logical cached-summary peak across the request. It is not measured DRAM
traffic, allocator peak, observation overhead or evidence that every byte is
read/written each step. The method intentionally amortizes its observations.
Sharing QK eliminates a duplicate QK calculation; it does not eliminate full
attention output, projected statistics, their storage or subsequent DP work.
The separately frozen same-input diagnostic will measure these scopes directly.
