"""v15 LongBench-v2 task contract: prompt construction and scoring (no model weights needed).

Prompt: the pinned NeMo-Skills `eval/longbench/default` user message (read-only checkout
at bcf059af, verified byte-for-byte against its git objects), after escaping model-reserved
special-token spellings inside the item fields exactly as the project's LongBench-v2-100
protocol (`experiments/diffusion_gemma_longbench_v2_100/protocol.py:render`). NO context
truncation. Length = tokens of the actual chat-templated request (thinking ON), i.e. the
same `adapter.encode_prompt(prompt, {'thinking': True})` the request path asserts.

Scoring (declared before inference, SCORER_VERSION):
  final text  = experiments.diffusion_gemma_aime26_modes.protocol.final_response(raw, True):
                text after the LAST '<channel|>' (the final answer channel), cut at end tokens;
                no final channel -> '' (the thinking channel is never mined)
  prediction  = NeMo `eval_mcq` with its default config on the final text
                (regex 'The final answer is (.+)$' then last \\boxed{}, then 'Answer: X')
  task_correct   = prediction == gold (a valid final-channel choice counts even at a length cap)
  strict_correct = task_correct AND termination == 'eos'
  parsed         = prediction is one of A-D (anything else extracted = malformed)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

NEMO = Path('/home/exouser/ljy/dlm/reference/NeMo-Skills')
NEMO_REVISION = 'bcf059af55c20a89f797724598f9908d126153e6'
NEMO_FILES = ('nemo_skills/prompt/config/eval/longbench/default.yaml', 'nemo_skills/evaluation/evaluator/mcq.py',
              'nemo_skills/evaluation/math_grader.py', 'nemo_skills/prompt/utils.py')
LB_REVISION = '2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9'
LB_DATA = Path('/home/exouser/.cache/huggingface/hub/datasets--THUDM--LongBench-v2/snapshots') / LB_REVISION / 'data.json'
SCORER_VERSION = 'v15-lbv2-nemo-mcq-default-on-final-channel-1'
FIELDS = ('context', 'question', 'choice_A', 'choice_B', 'choice_C', 'choice_D')


def sha(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else str(data).encode()).hexdigest()


@lru_cache
def nemo_prompt_config():
    head = subprocess.check_output(['git', '-C', str(NEMO), 'rev-parse', 'HEAD'], text=True).strip()
    if head != NEMO_REVISION:
        raise ValueError('NeMo-Skills checkout revision changed')
    for name in NEMO_FILES:
        committed = subprocess.check_output(['git', '-C', str(NEMO), 'show', f'{NEMO_REVISION}:{name}'])
        if (NEMO / name).read_bytes() != committed:
            raise ValueError(f'NeMo source modified: {name}')
    sys.dont_write_bytecode = True                 # never write caches into the read-only colleague checkout
    if str(NEMO) not in sys.path:
        sys.path.insert(0, str(NEMO))
    from nemo_skills.prompt.utils import get_prompt
    return get_prompt('eval/longbench/default')


def contract_identity() -> dict:
    nemo_prompt_config()
    from experiments.diffusion_gemma_aime26_modes import protocol as final_channel_source
    return dict(nemo_revision=NEMO_REVISION, nemo_files={n: sha((NEMO / n).read_bytes()) for n in NEMO_FILES},
                dataset_revision=LB_REVISION, dataset_sha256=sha(LB_DATA.read_bytes()), scorer_version=SCORER_VERSION,
                task_module_sha256=sha(Path(__file__).read_bytes()),
                final_channel_source_sha256=sha(Path(final_channel_source.__file__).read_bytes()))


def escape(item: dict, reserved) -> tuple[dict, dict]:
    """Project rule: escape literal model-reserved <...> special-token spellings in item text."""
    item, escaped = dict(item), {}
    for field in FIELDS:
        for token in reserved:
            if token and token.startswith('<') and token.endswith('>') and token in item[field]:
                escaped[token] = escaped.get(token, 0) + item[field].count(token)
                item[field] = item[field].replace(token, '&lt;' + token[1:-1] + '&gt;')
    return item, escaped


def render(adapter, item: dict) -> dict:
    """Untruncated NeMo prompt + exact request tokens (thinking ON). adapter needs only load_tokenizer()."""
    config = nemo_prompt_config()
    reserved = tuple(getattr(adapter.tokenizer, 'all_special_tokens', ()))
    escaped_item, escapes = escape(item, reserved)
    prompt = config.build_user_message(escaped_item)
    tokens = adapter.encode_prompt(prompt, {'thinking': True})
    return dict(prompt=prompt, prompt_hash=sha(prompt), prompt_tokens=tokens, prompt_tokens_n=len(tokens),
                literal_special_token_escapes=escapes, raw_prompt_hash=sha(config.build_user_message(item)))


def final_text(raw: str) -> str:
    from experiments.diffusion_gemma_aime26_modes.protocol import final_response
    return final_response(raw, True)


def predict(final_texts: list[str]) -> list[str | None]:
    """NeMo eval_mcq (default config) on final-channel texts; returns predicted letters (None = unparsed)."""
    nemo_prompt_config()
    from nemo_skills.evaluation.evaluator.mcq import eval_mcq
    with tempfile.TemporaryDirectory(prefix='v15_lbv2_mcq_') as folder:
        path = Path(folder) / 'predictions.jsonl'
        with path.open('w') as stream:
            for i, text in enumerate(final_texts):
                stream.write(json.dumps(dict(index=i, generation=text, expected_answer='?'), ensure_ascii=False) + '\n')
        eval_mcq(dict(input_file=str(path)))
        out = [json.loads(line) for line in path.read_text().splitlines()]
    if [o['index'] for o in out] != list(range(len(final_texts))):
        raise AssertionError('NeMo evaluator reordered or dropped samples')
    return [o['predicted_answer'] for o in out]


def score(raw_completions: list[str], golds: list[str], terminations: list[str]) -> list[dict]:
    if len(raw_completions) != len(golds) or len(raw_completions) != len(terminations):
        raise ValueError('scorer input lists must have equal length')
    texts = [final_text(r) for r in raw_completions]
    preds = predict(texts)
    if len(preds) != len(texts):
        raise ValueError('scorer prediction count differs from inputs')
    rows = []
    for text, pred, gold, term in zip(texts, preds, golds, terminations):
        if gold not in ('A', 'B', 'C', 'D'):
            raise ValueError('gold must be one of A-D')
        ok = pred == gold
        valid = pred in ('A', 'B', 'C', 'D')
        rows.append(dict(final_channel_present=bool(text), predicted=pred, parsed=valid,
                         malformed=pred is not None and not valid,
                         task_correct=ok, strict_correct=ok and term == 'eos', termination=term))
    return rows
