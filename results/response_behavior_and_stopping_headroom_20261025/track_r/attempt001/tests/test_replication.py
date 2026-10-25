import importlib.util
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[3]


def test_selector_and_runtime_are_original_bytes():
    old=REPO/'results/denoising_e2e_research_20261024/attempt003/code'
    for name in ['selector.py','online_runtime.py']:
        assert (ROOT/'code'/name).read_bytes()==(old/name).read_bytes()


def test_text_rubric_keeps_native_stop_separate():
    spec=importlib.util.spec_from_file_location('rubric',ROOT/'code/text_metrics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    a=module.metrics('wait\nx\nx\n<channel|>answer \\boxed{17}<eos>')
    assert a['duplicate_line_fraction']>0 and a['revision_vocabulary_count']==1
    assert a['first_closed_box_in_final_character'] is not None and a['end_markers']==1
    assert 'correct' not in a and 'native_stop' not in a


def test_seed_cells_share_question_identity():
    import json
    manifest=json.loads((ROOT/'manifest_identity.json').read_text())
    rows=manifest['identities'];assert len(rows)==len({x['id'] for x in rows})==70
    assert all(x['seeds']==[42,43] for x in rows)
    assert sum(x['primary_confirmation'] for x in rows if x['benchmark']=='aime26')==20
    assert sum(x['primary_confirmation'] for x in rows if x['benchmark']=='humaneval')==30
    assert any(x==dict(id='humaneval/HumanEval/71',seed=43) for x in manifest['legacy_reproduction'])
