# V2 equal absolute-work campaign

This campaign evaluates the unchanged `value_v2_online_preserve_mass` selector against a fresh, source-matched V31 control. It first sweeps one uniform GLOBAL threshold on the original 64-cell development panel, then audits thresholds selected by absolute physical GLOBAL FA4 kept tiles on the official 503-cell panel.

The threshold curve is in [sweep/curve.csv](sweep/curve.csv); calibration decisions are in [matched_calibration/summary.md](matched_calibration/summary.md). Official results are in [official_audit/summary.md](official_audit/summary.md), with physical work in [official_audit/work_accounting.csv](official_audit/work_accounting.csv), phase accounting in [official_audit/phase_accounting.csv](official_audit/phase_accounting.csv), and cumulative canvas work in [official_audit/cumulative_canvas_work.csv](official_audit/cumulative_canvas_work.csv).

The development-selected total-work threshold is 0.015. On the official trajectory it retains 1.0329x V31 GLOBAL tiles because V2 changes the number of native calls. The early-work threshold 0.021 matches the C=20 prefix within 1.2% on the official audit but retains 0.9221x total GLOBAL tiles. These are separate matched-work questions; neither is a clean latency result. Audit instrumentation and selector discovery work are reported separately from FA4 consumer work.
