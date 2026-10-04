"""Score v31 private AIME26 completions (primary: exact match, the official AIME metric; avg@k over seeds).

Rule: scripts/v31_official.py (planned cells, binding). Per completion: answer text = final_response(raw, row thinking)
(the text after the thinking channel; an unfinished thought gives '' and is wrong -- never grade a value inside the
thought), the project's gold-independent extraction (last \\boxed{}, else an answer marker, else the last number) and
exact match |x - answer| <= 1e-6 against the pinned math-ai/aime26 answer (`numeric_score` of
experiments/diffusion_gemma_aime30/protocol.py, compiled from that file alone, no torch).
Outputs (OUT_PREFIX):
  .official.json             {arm: {cell: bool}}  PRIMARY: exact match, no finish requirement
  .exact_and_finished.json   {arm: {cell: bool}}  SECONDARY: exact match AND a stop / eos finish
  .binding.json              per-cell run settings for the comparison tools (scripts/v31_paired_official.py aime)
  .summary.json              per arm: avg@k = mean over problems of each problem's mean over its cells (seeds x
                             repeats), accuracy per panel seed, cells, finished / capped / other, no_final_response
                             (thought never closed), unparsed (no number extracted), coverage, binding
A record from another pool / manifest / budget, a duplicate cell, a record outside the plan or (without --allow-missing)
a missing planned cell raises. No text, ids or answers written. Run with cwd / PYTHONPATH holding experiments/.
usage: python v31_score_aime.py OUT_PREFIX MANIFEST_DIR AIME2026_JSONL --cells CELLS.json [--repeats N]
           [--allow-missing] [--legacy-unbound] PRIVATE.jsonl [...]
  AIME2026_JSONL is the pinned dataset file (rows id, answer); the pool rows map to it through source_id
"""
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31_official as op  # noqa: E402

SECONDARY = 'exact_and_finished'


def score(records, answers, pools, final_response, numeric_score, legacy=False):
    official, strict, binding, diag = {}, {}, {}, {}
    for label, key, r in records:
        row, b, verified = op.bind(r, pools, legacy)
        text = final_response(r['completion'], bool(row['thinking']))
        result = numeric_score(text, answers[str(row['source_id'])])
        ok = bool(result['correct'])
        fc = op.finish_class(r['finish_reason'])
        op.put(official, label, key, ok)
        op.put(strict, label, key, ok and fc == 'finished')
        op.put(binding, label, key, b)
        d = diag.setdefault(label, collections.Counter())
        d['cells'] += 1
        d[fc] += 1
        d['no_final_response'] += text == ''
        d['unparsed'] += result['extracted'] is None
        d['verified' if verified else 'legacy_unbound'] += 1
    return official, {SECONDARY: strict}, binding, diag


def summarize(official, diag):
    out = {}
    for label, cells in official.items():
        per_item, per_seed = collections.defaultdict(list), collections.defaultdict(list)
        for k, v in cells.items():
            ds, index, seed, _ = k.split('|')
            per_item[(ds, index)].append(v)
            per_seed[seed].append(v)
        out[label] = dict(avg_at_k=100 * statistics.mean(statistics.mean(v) for v in per_item.values()),
                          k_per_problem=sorted({len(v) for v in per_item.values()}), problems=len(per_item),
                          accuracy_by_panel_seed={s: 100 * statistics.mean(v) for s, v in sorted(per_seed.items())},
                          correct=sum(cells.values()), **dict(diag[label]))
    return out


def main():
    a = op.scorer_cli(__doc__, ('out_prefix', 'manifest_dir', 'aime_jsonl'))
    data = Path(a.aime_jsonl)
    answers = {str(r['id']): str(r['answer']) for r in (json.loads(x) for x in data.read_text().splitlines() if x.strip())}
    final_response, fr_source = op.load_final_response()
    numeric_score, ns_source = op.aime_numeric_score()
    records, coverage = op.planned_records(a)
    official, secondary, binding, diag = score(records, answers, op.Pools(a.manifest_dir, ('source_id', 'thinking')),
                                               final_response, numeric_score, a.legacy_unbound)
    summary = dict(metric='exact match (avg@k over seeds)', answers_sha256=op.sha_file(data), answer_text=fr_source,
                   extraction=ns_source, coverage=coverage, arms=summarize(official, diag))
    op.write_outputs(a.out_prefix, official, secondary, summary, binding)
    for label, s in sorted(summary['arms'].items()):
        print(label, f"avg@k={s['avg_at_k']:.2f} k={s['k_per_problem']} correct={s['correct']}/{s['cells']} "
              f"capped={s.get('capped', 0)} no_final={s['no_final_response']} unparsed={s['unparsed']}")


if __name__ == '__main__':
    main()
