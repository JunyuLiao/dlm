"""Score v31 private HumanEval completions (booleans only) with the v27 HumanEval scorer's own pieces.

Per completion (scripts/v27_humaneval.py, unchanged): the final response after the thinking channel, the first python
fence (or a plain complete function) defining the entry point, the task's original imports + the completion + the
task's tests + check(entry_point), executed in its unprivileged bwrap sandbox (no network, no host paths, uid 65534;
5 s wall, 3 s CPU, 512 MB). The sandbox preflight must pass first; user namespaces are allowed on mpk only.
correct = tests passed and a stop / eos finish. Writes {arm_label: {"dataset|index|panel_seed|repeat": correct}}.
usage: python v31_score_humaneval.py OUT_JSON GOLD_JSON V27_HUMANEVAL_DIR PRIVATE.jsonl [...]
  GOLD_JSON: scorer-only {"humaneval/<n>": {task_id, prompt, test, entry_point}}; V27_HUMANEVAL_DIR holds v27_humaneval.py
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    out, gold_path, v27_dir, files = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
    sys.path.insert(0, v27_dir)
    import v27_humaneval as he
    he.sandbox_preflight()
    gold = json.loads(gold_path.read_text())
    jobs = []
    for f in files:
        label = Path(f).name.split('.private')[0].split('_', 1)[1]
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            problem = gold[r['id']]
            code, _ = he.extract_code(r['completion'], problem['entry_point'])
            key = f"{r['dataset']}|{r['index']}|{r['panel_seed']}|{r['repeat']}"
            jobs.append((label, key, problem, code, r['finish_reason'] in ('stop', 'eos')))

    def run(job):
        label, key, problem, code, finished = job
        if code is None or not finished:
            return label, key, False
        return label, key, bool(he.run_test(he.program_for(problem, code))[0])

    result = {}
    with ThreadPoolExecutor(8) as pool:
        for label, key, ok in pool.map(run, jobs):
            result.setdefault(label, {})[key] = ok
    Path(out).write_text(json.dumps(result, indent=1, sort_keys=True))
    for k, v in sorted(result.items()):
        print(k, sum(v.values()), len(v))


if __name__ == '__main__':
    main()
