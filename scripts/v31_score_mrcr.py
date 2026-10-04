"""Score v31 private MRCR 2-needle completions with the official OpenAI MRCR grading (ratios + booleans).

Per completion: the final response (`final_response` of experiments/diffusion_gemma_aime26_modes/protocol.py with the
manifest row's thinking flag, OFF for every MRCR pool: the text after the last '<channel|>' if one is present, cut at
the first '<turn|>' / '<|endoftext|>' / '<eos>', stripped), then the README's `grade` (openai/mrcr @ f4c69fae): 0 unless
the response starts with random_string_to_prepend, otherwise difflib.SequenceMatcher(None, response, answer).ratio()
after removing that prefix from both. correct = ratio >= 0.99 and a stop / eos finish, as in v31_score_ruler.py.
Writes {arm_label: {"dataset|index|panel_seed|repeat": ratio}} and the same keys with the boolean -- no text, ids or
answers. CPU only, no torch: final_response is compiled from the protocol module's own source (importing that module
would pull in torch and the BLASST runner).
Run with cwd = a checkout or deployment that holds experiments/ (PYTHONPATH=.).
usage: python v31_score_mrcr.py OUT_RATIO_JSON OUT_BOOL_JSON MANIFEST_DIR GOLD_DIR PRIVATE.jsonl [...]
  MANIFEST_DIR / GOLD_DIR hold {dataset}_generation_manifest.json and {dataset}_gold.json
  ({id: {"answer", "random_string_to_prepend"}}); labels from file names <tag>_<label>.private.jsonl
"""
import ast
import importlib.util
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

THRESHOLD = 0.99


def load_final_response():
    """experiments.diffusion_gemma_aime26_modes.protocol.final_response, compiled from that file alone."""
    origin = importlib.util.find_spec('experiments.diffusion_gemma_aime26_modes.protocol').origin
    tree = ast.parse(Path(origin).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'final_response')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), origin, 'exec'), namespace)
    return namespace['final_response']


def grade(response, answer, random_string_to_prepend) -> float:
    """openai/mrcr README `grade`, verbatim apart from returning 0.0 as a float."""
    if not response.startswith(random_string_to_prepend):
        return 0.0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())


def main():
    out_ratio, out_bool, man_dir, gold_dir, files = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5:]
    final_response = load_final_response()
    rows, golds, ratios, correct = {}, {}, {}, {}
    for f in files:
        label = Path(f).name.split('.private')[0].split('_', 1)[1]
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            ds = r['dataset']
            if ds not in rows:
                rows[ds] = {x['id']: x for x in json.loads((man_dir / f'{ds}_generation_manifest.json').read_text(encoding='utf-8'))}
                golds[ds] = json.loads((gold_dir / f'{ds}_gold.json').read_text(encoding='utf-8'))
            gold = golds[ds][r['id']]
            text = final_response(r['completion'], bool(rows[ds][r['id']]['thinking']))
            ratio = grade(text, gold['answer'], gold['random_string_to_prepend'])
            key = f"{ds}|{r['index']}|{r['panel_seed']}|{r['repeat']}"
            ratios.setdefault(label, {})[key] = ratio
            correct.setdefault(label, {})[key] = ratio >= THRESHOLD and r['finish_reason'] in ('stop', 'eos')
    Path(out_ratio).write_text(json.dumps(ratios, indent=1, sort_keys=True))
    Path(out_bool).write_text(json.dumps(correct, indent=1, sort_keys=True))
    for k in sorted(ratios):
        v = ratios[k]
        print(k, sum(correct[k].values()), len(v), f'mean_ratio={sum(v.values()) / len(v):.4f}')


if __name__ == '__main__':
    main()
