"""Score v31 private AIME26 completions (booleans only).

Per completion: the final response after the thinking channel (`final_response`, the same boundary as the AIME / RULER
v18 pipeline: no grading of a value inside an unfinished thought), numeric extraction and exact match against the
pinned math-ai/aime26 answer (`numeric_score`, |x - answer| <= 1e-6); correct also requires a stop / eos finish, as in
v31_score_ruler.py. Writes {arm_label: {"dataset|index|panel_seed|repeat": correct}} -- no text, ids or answers.
Run with cwd = the deployment and PYTHONPATH=src:. (CPU only).
usage: python v31_score_aime.py OUT_JSON MANIFEST_DIR AIME2026_JSONL PRIVATE.jsonl [...]
  MANIFEST_DIR holds {dataset}_generation_manifest.json; AIME2026_JSONL is the pinned dataset file (rows id, answer);
  labels from file names <tag>_<label>.private.jsonl
"""
import json
import sys
from pathlib import Path

from experiments.diffusion_gemma_aime26_modes.protocol import final_response
from experiments.diffusion_gemma_aime30.protocol import numeric_score


def main():
    out, man_dir, data, files = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
    answers = {str(r['id']): str(r['answer']) for r in
               (json.loads(x) for x in data.read_text().splitlines() if x.strip())}
    rows, result = {}, {}
    for f in files:
        label = Path(f).name.split('.private')[0].split('_', 1)[1]
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            ds = r['dataset']
            if ds not in rows:
                rows[ds] = {x['id']: x for x in json.loads((man_dir / f'{ds}_generation_manifest.json').read_text())}
            row = rows[ds][r['id']]
            text = final_response(r['completion'], bool(row['thinking']))
            ok = numeric_score(text, answers[str(row['source_id'])])['correct'] and r['finish_reason'] in ('stop', 'eos')
            result.setdefault(label, {})[f"{ds}|{r['index']}|{r['panel_seed']}|{r['repeat']}"] = bool(ok)
    Path(out).write_text(json.dumps(result, indent=1, sort_keys=True))
    for k, v in sorted(result.items()):
        print(k, sum(v.values()), len(v))


if __name__ == '__main__':
    main()
