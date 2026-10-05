"""Positive control of the HumanEval scorer: the official canonical solutions must pass 164 / 164.

Each canonical solution is wrapped like a chat completion (an empty thought block, then a ```python fence holding the
task's prompt + canonical_solution, i.e. a complete function), then goes through the scorer's own pieces of
scripts/v27_humaneval.py: extract_code -> program_for (the official check_program with the full prompt) -> run_test in
the bwrap sandbox. Cells that time out in the parallel pass are re-run one at a time. Prints counts and the task ids
of any failure only (no code, prompts or tests). Run on mpk (user namespaces) with CPU only.
usage: python v31_humaneval_positive_control.py HUMANEVAL_JSONL_GZ [V27_HUMANEVAL_DIR]
"""
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_SHA256 = 'b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef'   # the project's pinned file


def main():
    data = Path(sys.argv[1]).read_bytes()
    sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else str(HERE))
    sys.path.insert(0, str(HERE))
    import v27_humaneval as he
    from v31_score_humaneval import run_all
    digest = hashlib.sha256(data).hexdigest()
    tasks = [json.loads(x) for x in gzip.decompress(data).decode().splitlines() if x.strip()]
    if digest != EXPECTED_SHA256 or len(tasks) != 164:
        raise SystemExit('not the pinned 164-task HumanEval file')
    he.sandbox_preflight()
    jobs, reasons = [], collections.Counter()
    for t in tasks:
        raw = '<|channel>thought\n<channel|>```python\n' + t['prompt'] + t['canonical_solution'] + '\n```<turn|>'
        code, reason = he.extract_code(raw, t['entry_point'])
        reasons[reason] += 1
        jobs.append((dict(prompt=t['prompt'], test=t['test'], entry_point=t['entry_point']), code))
    results, timeouts_first = run_all(jobs, he)
    failed = [t['task_id'] for t, (passed, _) in zip(tasks, results) if not passed]
    print(json.dumps(dict(dataset_sha256=digest, tasks=len(tasks), extraction=dict(reasons), passed=len(tasks) - len(failed),
                          failed=failed, timeouts_first_pass=timeouts_first,
                          timeouts_after_rerun=sum(r == 'timeout' for _, r in results), extractor=he.EXTRACTOR,
                          program=he.PROGRAM, sandbox=he.SANDBOX)))
    sys.exit(0 if not failed else 1)


if __name__ == '__main__':
    main()
