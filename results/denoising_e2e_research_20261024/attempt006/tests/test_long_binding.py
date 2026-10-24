import ast
import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'attempt003'


def test_selector_and_runtime_are_exact_parent_sources():
    b=json.loads((ROOT/'execution_binding.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in b['native_method_sources'].items())
    source=(ROOT/'code/long_worker.py').read_text()
    assert 'from online_runtime import Runtime' in source
    assert "'attempt003/code'" in source


def test_same_cohort_no_native_limit_or_score_changes():
    parent=json.loads((PARENT/'e2e_protocol.json').read_text());binding=json.loads((ROOT/'execution_binding.json').read_text())
    assert hashlib.sha256((PARENT/'e2e_protocol.json').read_bytes()).hexdigest()==binding['parent_protocol_sha256']
    assert len(parent['panel']['long'])==10 and len(parent['arms'])==9
    original=ast.parse((PARENT/'code/e2e_worker.py').read_text())
    modified=ast.parse((ROOT/'code/long_worker.py').read_text())
    def sample(tree):
        return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='sample')
    assert ast.dump(sample(original))==ast.dump(sample(modified))
    def calls(tree, name):
        return [ast.dump(n) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==name]
    assert calls(original,'LLM')==calls(modified,'LLM')
    assert calls(original,'SamplingParams')==calls(modified,'SamplingParams')


def test_native_request_bound_allows_full_original_cap_and_pipeline_tail():
    import math
    prompt=244678;budget=16384
    bound=math.ceil(prompt/16384)+(math.ceil(budget/256)+1)*49+2
    maximum_committed=math.ceil(prompt/16384)+math.ceil(budget/256)*49
    assert bound>maximum_committed+1
    assert bound<3300
