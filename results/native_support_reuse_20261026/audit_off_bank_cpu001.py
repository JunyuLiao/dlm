"""Same finite bank/choices with optional repeated-A audit disabled."""
from pathlib import Path
import sys,importlib.util,json,hashlib
S=Path(__file__).resolve().parent;R=S/'attempt004';sys.path.insert(0,str(R/'code'))
import op_bank
original=op_bank.build;checks=[]
def both(*args,**kw):
    calls=[]
    op=args[10]
    def counted(support):calls.append(support is None);return op(support)
    values=list(args);values[10]=counted
    a,pa,wa=original(*values,**kw);na=len(calls);aa=sum(calls);calls.clear()
    b,pb,wb=original(*values,**kw,audit=False)
    assert len(calls)==na-1 and sum(calls)==aa-1
    assert a['selections']==b['selections'] and a['screen']==b['screen']
    assert a['allkept_repeat_exact'] is True and b['allkept_repeat_exact'] is None
    import torch
    assert torch.equal(pa['dense'],pb['dense'])
    for x,y in zip(pa['candidates'],pb['candidates']):assert all(torch.equal(x[k],y[k]) for k in x)
    checks.append(dict(candidate_choices_exact=True,all_saved_operator_vectors_exact=True,one_optional_allkept_replay_removed=True,observed_operator_calls=na,off_operator_calls=len(calls)))
    return a,pa,wa
op_bank.build=both
spec=importlib.util.spec_from_file_location('qualified_cpu_case',R/'tests/test_reuse.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.test_common_bank_adaptations_share_feasibility_and_do_not_draw_rng()
out=S/'audit_off_bank_cpu001';out.mkdir(exist_ok=False)
with (out/'final_complete.json').open('x') as f:json.dump(dict(passed=True,checks=checks,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_forwards=0),f,indent=2)
print(json.dumps(checks))
