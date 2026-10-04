"""Score v31 private HumanEval completions (primary: pass@1, the official HumanEval metric).

Rule: scripts/v31_official.py; pieces of scripts/v27_humaneval.py. Per completion: the final response after the thinking
channel (no '<channel|>' -> nothing to run), the first python fence that defines the entry point (or a plain complete
function), program = the official check_program with the task's FULL prompt (prompt + code + test +
check(entry_point): the prompt's docstring-only stub is replaced by the model's definition, helpers the prompt defines
stay available), executed in the unprivileged bwrap sandbox (no network, no host paths, uid 65534; 5 s wall, 3 s CPU,
512 MB). The sandbox preflight must pass first; unprivileged user namespaces are allowed on mpk only.
Outputs (OUT_PREFIX):
  .official.json            {arm: {cell: bool}}  PRIMARY: the tests pass; no finish requirement (the official harness
                            runs whatever was generated)
  .pass_and_finished.json   {arm: {cell: bool}}  SECONDARY: passed AND a stop / eos finish
  .summary.json             per arm: pass@1 = mean over tasks of each task's pass fraction over its cells (the unbiased
                            pass@1 estimator with n = cells per task), pass rate per panel seed, cells, finished / capped /
                            other, extraction outcomes (parsed / no_final_response / no_python_fence / syntax_error /
                            missing_entry_point)
A duplicate cell key raises. No text, ids, code or tests are written.
usage: python v31_score_humaneval.py OUT_PREFIX GOLD_JSON V27_HUMANEVAL_DIR PRIVATE.jsonl [...]
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


def score(records, gold, he, workers=8):
    jobs = []
    for label, key, r in records:
        problem = gold[r['id']]
        code, reason = he.extract_code(r['completion'], problem['entry_point'])
        jobs.append((label, key, r, problem, code, reason))

    def run(job):
        _, _, _, problem, code, _ = job
        return bool(he.run_test(he.program_for(problem, code))[0]) if code is not None else False

    with ThreadPoolExecutor(workers) as pool:
        passed = list(pool.map(run, jobs))
    official, strict, diag = {}, {}, {}
    for (label, key, r, _, _, reason), ok in zip(jobs, passed):
        fc = op.finish_class(r['finish_reason'])
        op.put(official, label, key, ok)
        op.put(strict, label, key, ok and fc == 'finished')
        d = diag.setdefault(label, collections.Counter())
        d['cells'] += 1
        d[fc] += 1
        d['extraction_' + reason] += 1
    return official, {SECONDARY: strict}, diag


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
    prefix, gold_path, v27_dir, files = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
    sys.path.insert(0, v27_dir)
    import v27_humaneval as he
    he.sandbox_preflight()
    gold = json.loads(gold_path.read_text())
    records = op.read_private(files)
    official, secondary, diag = score(records, gold, he)
    id_of = {(label, key): r['id'] for label, key, r in records}
    summary = dict(metric='pass@1 (official check_program, full prompt)', extractor=he.EXTRACTOR, program=he.PROGRAM,
                   sandbox=he.SANDBOX, scorer_sha256=op.sha_file(he.__file__), gold_sha256=op.sha_file(gold_path),
                   arms=summarize(official, diag, id_of))
    op.write_outputs(prefix, official, secondary, summary)
    for label, s in sorted(summary['arms'].items()):
        print(label, f"pass@1={s['pass_at_1']:.2f} passed={s['passed']}/{s['cells']} capped={s.get('capped', 0)} "
              f"parsed={s.get('extraction_parsed', 0)}")


if __name__ == '__main__':
    main()
