"""Score v31 private GraphWalks completions with the official OpenAI GraphWalks extraction and set F1 (F1 + booleans).

Per completion: the final response after the thinking channel (`final_response` of
experiments/diffusion_gemma_aime26_modes/protocol.py with the manifest row's thinking flag, ON for every GraphWalks
pool: the text after the last '<channel|>', cut at the first end token, stripped; an unfinished thought gives ''), then
the README's `get_list` (openai/graphwalks @ be6cc6ec: the last line must contain 'Final Answer:', the greedy '[...]'
span on it is split on commas, items stripped, empty items dropped; otherwise unparsed) and its set precision / recall /
F1 against the gold answer nodes (unparsed -> 0; empty gold and empty prediction -> 1). correct = F1 == 1 and a stop /
eos finish, as in v31_score_ruler.py. Writes {arm_label: {"dataset|index|panel_seed|repeat": f1}} and the same keys
with the boolean -- no text, ids or answers. CPU only, no torch: final_response is compiled from the protocol module's
own source (importing that module would pull in torch and the BLASST runner).
Run with cwd = a checkout or deployment that holds experiments/ (PYTHONPATH=.).
usage: python v31_score_graphwalks.py OUT_F1_JSON OUT_BOOL_JSON MANIFEST_DIR GOLD_DIR PRIVATE.jsonl [...]
  MANIFEST_DIR / GOLD_DIR hold {dataset}_generation_manifest.json and {dataset}_gold.json ({id: [node, ...]});
  labels from file names <tag>_<label>.private.jsonl
"""
import ast
import importlib.util
import json
import re
import sys
from pathlib import Path


def load_final_response():
    """experiments.diffusion_gemma_aime26_modes.protocol.final_response, compiled from that file alone."""
    origin = importlib.util.find_spec('experiments.diffusion_gemma_aime26_modes.protocol').origin
    tree = ast.parse(Path(origin).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'final_response')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), origin, 'exec'), namespace)
    return namespace['final_response']


def get_list(response: str) -> tuple[list[str], bool]:
    """openai/graphwalks README extraction, verbatim (as a free function)."""
    # get the very last line of the response
    line = response.split("\n")[-1]
    # check if formatted correctly
    if "Final Answer:" not in line:
        return [], True
    list_part = re.search(r"\[.*\]", line)
    if list_part:
        result_list = list_part.group(0).strip("[]").split(",")
        # if the list was empty, then get [] not [""]
        result_list = [item.strip() for item in result_list if item.strip()]
        return result_list, False
    else:
        return [], True


def f1_score(response: str, answer_nodes) -> float:
    """openai/graphwalks README grading, verbatim (F1 returned as a float)."""
    sampled_list, failed_to_parse = get_list(response)
    sampled_set = set(sampled_list)
    truth_set = set(answer_nodes)
    n_golden = len(truth_set)
    n_sampled = len(sampled_set)
    if failed_to_parse:
        recall = 0.0
        precision = 0.0
        f1 = 0.0
    elif n_golden == n_sampled == 0:
        recall = 1.0
        precision = 1.0
        f1 = 1.0
    else:
        n_overlap = len(sampled_set & truth_set)
        recall = n_overlap / n_golden if n_golden > 0 else 0
        precision = n_overlap / n_sampled if n_sampled > 0 else 0
        f1 = 2 * (recall * precision) / (recall + precision) if recall + precision > 0 else 0
    return float(f1)


def main():
    out_f1, out_bool, man_dir, gold_dir, files = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5:]
    final_response = load_final_response()
    rows, golds, scores, correct = {}, {}, {}, {}
    for f in files:
        label = Path(f).name.split('.private')[0].split('_', 1)[1]
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            ds = r['dataset']
            if ds not in rows:
                rows[ds] = {x['id']: x for x in json.loads((man_dir / f'{ds}_generation_manifest.json').read_text(encoding='utf-8'))}
                golds[ds] = json.loads((gold_dir / f'{ds}_gold.json').read_text(encoding='utf-8'))
            text = final_response(r['completion'], bool(rows[ds][r['id']]['thinking']))
            f1 = f1_score(text, golds[ds][r['id']])
            key = f"{ds}|{r['index']}|{r['panel_seed']}|{r['repeat']}"
            scores.setdefault(label, {})[key] = f1
            correct.setdefault(label, {})[key] = f1 == 1.0 and r['finish_reason'] in ('stop', 'eos')
    Path(out_f1).write_text(json.dumps(scores, indent=1, sort_keys=True))
    Path(out_bool).write_text(json.dumps(correct, indent=1, sort_keys=True))
    for k in sorted(scores):
        v = scores[k]
        print(k, sum(correct[k].values()), len(v), f'mean_f1={sum(v.values()) / len(v):.4f}')


if __name__ == '__main__':
    main()
