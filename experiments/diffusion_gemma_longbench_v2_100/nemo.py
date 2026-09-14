"""Use the actual pinned NeMo prompt renderer and MCQ evaluator, not a reimplementation."""
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys
import tempfile

CHECKOUT = Path('reference/NeMo-Skills')
REVISION = 'bcf059af55c20a89f797724598f9908d126153e6'


@lru_cache
def initialize():
    actual = subprocess.check_output(['git','-C',str(CHECKOUT),'rev-parse','HEAD'],text=True).strip()
    if actual != REVISION:
        raise ValueError('NeMo-Skills checkout revision changed')
    for name in ('nemo_skills/prompt/config/eval/longbench/default.yaml',
                 'nemo_skills/evaluation/evaluator/mcq.py', 'nemo_skills/evaluation/math_grader.py'):
        committed = subprocess.check_output(['git','-C',str(CHECKOUT),'show',f'{REVISION}:{name}'])
        if (CHECKOUT/name).read_bytes() != committed:
            raise ValueError(f'NeMo source modified: {name}')
    sys.path.insert(0,str(CHECKOUT.resolve()))
    from nemo_skills.prompt.utils import get_prompt
    return get_prompt('eval/longbench/default')


def prompt(item):
    return initialize().build_user_message(item)


def evaluate(samples):
    initialize()
    from nemo_skills.evaluation.evaluator.mcq import eval_mcq
    # NeMo's evaluator writes into its input. Give it a disposable copy, never
    # the immutable model-generation shards or the authoritative dataset.
    with tempfile.TemporaryDirectory(prefix='longbench100_nemo_') as folder:
        path = Path(folder)/'predictions.jsonl'
        with path.open('w') as stream:
            for sample in samples:
                stream.write(json.dumps(sample,ensure_ascii=False)+'\n')
        eval_mcq(dict(input_file=str(path)))
        return [json.loads(line) for line in path.read_text().splitlines()]


def score(row, prediction):
    out, = evaluate([dict(index=row['source_id'], generation=prediction, expected_answer=row['expected'])])
    return float(out['symbolic_correct'])


def sources():
    initialize()
    # Pin all shipped Python/YAML sources used by the framework; imports can
    # traverse helper modules beyond the top-level scorer and prompt.
    return sorted(p for p in (CHECKOUT/'nemo_skills').rglob('*') if p.suffix in ('.py','.yaml'))
