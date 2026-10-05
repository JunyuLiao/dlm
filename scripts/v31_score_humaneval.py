"""Score v31 private HumanEval completions (primary: pass@1, the official HumanEval metric).

Rule: scripts/v31_official.py (planned cells, binding); pieces of scripts/v27_humaneval.py. Per completion: the final
response after the thinking channel (no '<channel|>' -> nothing to run), the first fence tagged '' or 'py*' that parses
and defines the entry point (or a plain complete function), program = the official check_program with the task's FULL
prompt (prompt + code + test + check(entry_point)), executed in the unprivileged bwrap sandbox (no network, no host paths,
uid 65534; 5 s wall, 3 s CPU, 512 MB). Cells that time out in the parallel pass are re-run one at a time (one worker) and
the re-run decides; both counts are reported. The sandbox preflight must pass first (user namespaces: mpk only).
Outputs (OUT_PREFIX):
  .official.json            {arm: {cell: bool}}  PRIMARY: the tests pass; no finish requirement (the official harness
                            runs whatever was generated)
  .pass_and_finished.json   {arm: {cell: bool}}  SECONDARY: passed AND a stop / eos finish
  .binding.json             per-cell run settings for the comparison tools (scripts/v31_paired_official.py humaneval)
  .summary.json             per arm: pass@1 = mean over tasks of each task's pass fraction over its cells (the unbiased
                            pass@1 estimator with n = cells per task), pass rate per panel seed, cells, finished / capped /
                            other, extraction outcomes, timeouts (first pass / after the serial re-run), coverage, binding
A record from another pool / manifest / budget, a duplicate cell, a record outside the plan or (without --allow-missing)
a missing planned cell raises. No text, ids, code or tests are written.
usage: python v31_score_humaneval.py OUT_PREFIX MANIFEST_DIR GOLD_JSON V27_HUMANEVAL_DIR --cells CELLS.json
           [--repeats N] [--allow-missing] [--legacy-unbound] PRIVATE.jsonl [...]
  GOLD_JSON: scorer-only {"humaneval/<n>": {task_id, prompt, test, entry_point}}; V27_HUMANEVAL_DIR holds v27_humaneval.py
"""
import collections
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31_official as op  # noqa: E402

SECONDARY = 'pass_and_finished'


def run_all(jobs, he, workers=8):
    """[(passed, reason)] per job: a parallel pass, then every timed-out cell again with one worker."""
    def run(job):
        problem, code = job
        return tuple(he.run_test(he.program_for(problem, code))) if code is not None else (False, 'not_run')

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(run, jobs))
    first = [i for i, (_, reason) in enumerate(results) if reason == 'timeout']
    for i in first:
        results[i] = run(jobs[i])
    return results, len(first)


def score(records, gold, he, pools, legacy=False, workers=8):
    staged, binding, verified = [], {}, collections.defaultdict(collections.Counter)
    for label, key, r in records:
        _, b, ok = op.bind(r, pools, legacy)
        op.put(binding, label, key, b)
        verified[label][ok] += 1
        problem = gold[r['id']]
        code, reason = he.extract_code(r['completion'], problem['entry_point'])
        staged.append((label, key, r, problem, code, reason))
    results, timeouts_first = run_all([(s[3], s[4]) for s in staged], he, workers)
    official, strict, diag = {}, {}, {}
    for (label, key, r, _, _, reason), (passed, outcome) in zip(staged, results):
        fc = op.finish_class(r['finish_reason'])
        op.put(official, label, key, bool(passed))
        op.put(strict, label, key, bool(passed) and fc == 'finished')
        d = diag.setdefault(label, collections.Counter())
        d['cells'] += 1
        d[fc] += 1
        d['extraction_' + reason] += 1
        d['timeouts_after_rerun'] += outcome == 'timeout'
    for label in diag:
        diag[label]['verified'], diag[label]['legacy_unbound'] = verified[label][True], verified[label][False]
    return official, {SECONDARY: strict}, binding, diag, timeouts_first


def summarize(official, diag, id_of):
    out = {}
    for label, cells in official.items():
        per_task, per_seed = collections.defaultdict(list), collections.defaultdict(list)
        for k, v in cells.items():
            per_task[id_of[(label, k)]].append(v)
            per_seed[k.split('|')[2]].append(v)
        out[label] = dict(pass_at_1=100 * statistics.mean(statistics.mean(v) for v in per_task.values()), tasks=len(per_task),
                          samples_per_task=sorted({len(v) for v in per_task.values()}), passed=sum(cells.values()),
                          pass_rate_by_panel_seed={s: 100 * statistics.mean(v) for s, v in sorted(per_seed.items())},
                          **dict(diag[label]))
    return out


def main():
    a = op.scorer_cli(__doc__, ('out_prefix', 'manifest_dir', 'gold_json', 'v27_dir'))
    sys.path.insert(0, a.v27_dir)
    import v27_humaneval as he
    he.sandbox_preflight()
    gold = json.loads(Path(a.gold_json).read_text())
    records, coverage = op.planned_records(a)
    official, secondary, binding, diag, timeouts_first = score(records, gold, he, op.Pools(a.manifest_dir, ('thinking',)),
                                                               a.legacy_unbound)
    id_of = {(label, key): r['id'] for label, key, r in records}
    summary = dict(metric='pass@1 (official check_program, full prompt)', extractor=he.EXTRACTOR, program=he.PROGRAM,
                   sandbox=he.SANDBOX, scorer_sha256=op.sha_file(he.__file__), gold_sha256=op.sha_file(a.gold_json),
                   timeouts_first_pass=timeouts_first, coverage=coverage, arms=summarize(official, diag, id_of))
    op.write_outputs(a.out_prefix, official, secondary, summary, binding)
    print('timeouts in the parallel pass (re-run serially):', timeouts_first)
    for label, s in sorted(summary['arms'].items()):
        print(label, f"pass@1={s['pass_at_1']:.2f} passed={s['passed']}/{s['cells']} capped={s.get('capped', 0)} "
              f"parsed={s.get('extraction_parsed', 0)} timeouts_after_rerun={s['timeouts_after_rerun']}")


if __name__ == '__main__':
    main()
