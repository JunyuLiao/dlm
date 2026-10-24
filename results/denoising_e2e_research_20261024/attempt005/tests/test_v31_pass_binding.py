import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'attempt003'


def test_inherited_selector_sources_match_parent_pins():
    b=json.loads((ROOT/'execution_binding.json').read_text())
    parent=json.loads((PARENT/'e2e_protocol.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h==parent['sources'][p] for p,h in b['native_method_sources'].items())


def test_native_sampler_and_model_arguments_unchanged():
    a=ast.parse((PARENT/'code/e2e_worker.py').read_text())
    b=ast.parse((ROOT/'code/v31_worker.py').read_text())
    def sample(tree):return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='sample')
    assert ast.dump(sample(a))==ast.dump(sample(b))
    def calls(tree,name):return [ast.dump(n) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==name]
    assert calls(a,'LLM')==calls(b,'LLM')
    assert calls(a,'SamplingParams')==calls(b,'SamplingParams')


def test_all_request_modes_equal_constructor_mode():
    tree=ast.parse((ROOT/'code/v31_worker.py').read_text())
    requests=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='request']
    assert len(requests)==2
    assert all(isinstance(n.args[-1],ast.Name) and n.args[-1].id=='pass_audit' for n in requests)
    constructor=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='VllmMethodAdapter')
    flags={k.arg:ast.unparse(k.value) for k in constructor.keywords if k.arg in {'profile','value_audit','value_clean_timing'}}
    assert flags==dict(profile='pass_audit',value_audit='pass_audit',value_clean_timing='not pass_audit')


def test_parent_scope_and_original_failed_leaf_are_preserved():
    binding=json.loads((PARENT/'result_directory_bindings.json').read_text())
    assert binding['failed_original_short_leaf_preserved']
    assert binding['result_paths']['short']['v31_optimized'].endswith('v31_optimized_separate_passes')
    assert (PARENT/'launches/short/v31_optimized/exit_receipt.json').exists()
    assert json.loads((PARENT/'launches/short/v31_optimized/exit_receipt.json').read_text())['returncode']==1
