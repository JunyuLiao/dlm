# Controlled overlap factorial after a same-input FA4 output change

Completed operand replay002 (sourceb147fccd,21.870s) executes224 FA4 calls over
complete captured q129/42 inputs at native observation anchors100/154. Native
automatic16-split observing/plain paths are stable in this screen and reproduce
the captured outputs, including freed scratch zero/finite/NaN factors. Single-
split observing at residual anchor154 changes on repeat15:4677 BF16 elements,
max0.03125,relativeL2 0.000189771, while Q/K/V remain bit-identical. Other single-
split repeats and plain kernel are stable in this small screen. All observed
prefix auxiliary slots are finite. Both actual configurations report intra-
warp-group overlap=True, tile64x64. This proves an operator-level same-input
variation in one tested diagnostic path, not yet the native automatic failure.

Freeze two independent operator engines, forward/reversed factor order. Each
uses both complete captured shapes and256 repeats for each of eight factors:
num_splits0/1 x observing/plain x intra_wg_overlapTrue/False. Total8192 calls,
zero native model/sampler/generation calls. Keep every other input, scalar,
stride, pointer alignment modulo256 and native mask-fix source fixed. Record
actual tile/split/overlap, output-byte repeat/native comparisons, auxiliary-byte
repeat/native comparisons, input preservation and actual CUDA capture count.
Retain private first/changed tensors for each differing output/auxiliary and
sanitized head/query64 counts. Do not conflate physical allocation changes with
logical data changes. The write-only mass buffer alternates zero/NaN initially.

Only a repeated positive in overlap=True and a matched stable overlap=False
supports narrowing the numerical failure to this pipeline factor. It still
requires native full/light/pure and independent-process qualification before
promotion. A stable small replay, a different mean error, or switching split
count alone does not repair the native evaluation. Non-overlap is a named
numerical/runtime variant and its cost must be measured if later adopted; no
silent slower-mode claim. Preserve native selector budgets and all sampler
semantics; both full accuracy matrices remain gated.

Each engine reserves at most180s,360s total, within existing phasek256 and the
unchanged43200s total. Reuse the exact tested restoration/comparison sources and
bind the completed replay receipt/raw payloads. One GPU worker at a time, no
installed package edits, private tensors never Git. Preserve failed001 replay
launcher: no GPU worker, conservative55.141s reservation charge; corrected002
passed. Further race/internal-partial capture depends on observed results.
