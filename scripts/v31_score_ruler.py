"""Score v31 private completions of long RULER cells with the pinned official RULER metrics (booleans only).

Per completion: the final response (`final_response`, as the AIME/RULER v18 pipeline), RULER's own
`postprocess_prediction`, then the official metric of the task's base type (niah / variable_tracking /
common_words_extraction / freq_words_extraction / qa) against the gold `outputs`; correct = score 100.
Writes {arm_label: {"dataset|index|panel_seed|repeat": correct}} -- no text, ids or predictions. With RULER_RAW_OUT set,
also writes the official per-sample score (0-100, partial credit for the multi-answer tasks; RULER reports its mean)
as {arm_label: {key: score}} (a stop / eos finish is not required there, as in the official evaluation).
Run where the pinned RULER checkout and the private gold live (dllm), with PYTHONPATH holding the project `src`.
usage: python v31_score_ruler.py OUT_JSON RULER_ROOT MANIFEST_DIR GOLD_DIR PRIVATE.jsonl [...]
  MANIFEST_DIR / GOLD_DIR hold {dataset}_generation_manifest.json and {dataset}_gold.json; labels from file names
  <tag>_<label>.private.jsonl
"""
import json
import os
import sys
from pathlib import Path

from dllm.evaluation.ruler import official
from experiments.diffusion_gemma_aime26_modes.protocol import final_response


def main():
    out, ruler_root, man_dir, gold_dir, files = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5:]
    scorers = official.load_scorers(ruler_root)
    customized, _ = official.load_configuration(ruler_root)
    task_base = {name: str(cfg['task']) for name, cfg in customized.items()}
    golds, tasks = {}, {}
    result, raw = {}, {}
    for f in files:
        label = Path(f).name.split('.private')[0].split('_', 1)[1]
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            ds = r['dataset']
            if ds not in golds:
                golds[ds] = json.loads((gold_dir / f'{ds}_gold.json').read_text())
                tasks[ds] = {row['id']: row['task'] for row in
                             json.loads((man_dir / f'{ds}_generation_manifest.json').read_text())}
            pred = official.postprocess_prediction(final_response(r['completion'], False))
            base = task_base[tasks[ds][r['id']]]
            score = float(scorers[base]([pred], [golds[ds][r['id']]]))
            ok = score >= 100.0 - 1e-9 and r['finish_reason'] in ('stop', 'eos')
            key = f"{ds}|{r['index']}|{r['panel_seed']}|{r['repeat']}"
            result.setdefault(label, {})[key] = ok
            raw.setdefault(label, {})[key] = round(score, 4)
    Path(out).write_text(json.dumps(result, indent=1, sort_keys=True))
    if os.environ.get('RULER_RAW_OUT'):
        Path(os.environ['RULER_RAW_OUT']).write_text(json.dumps(raw, indent=1, sort_keys=True))
    for k, v in sorted(result.items()):
        print(k, sum(v.values()), len(v))


if __name__ == '__main__':
    main()
