from pathlib import Path
import importlib.util,copy
import pytest

p=Path(__file__).resolve().parents[1]/'code/census.py'
s=importlib.util.spec_from_file_location('new_saved_control_census',p)
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)

def request(stories,study='R',cap=False,coincident=False):
    trace=[]
    for canvas,words in enumerate(stories):
        for i,w in enumerate(words,1):
            last=i==len(words);confidence=last and (not cap or coincident)
            row=dict(commit=False,native_step=i,stable=i>1 and words[i-2]==w,confidence=confidence,converged=last,cap_reached=i>=48,
                native_winner_sha256=w,entropy_mean=.001 if confidence else .02)
            row['entropy_slack']=row['entropy_mean']-.005
            row['native_history_ready' if study=='R' else 'history_ready']=i>=2
            if study!='R':row['canvas']=canvas
            trace.append(row)
        trace.append(dict(commit=True,native_step=0,**({'canvas':canvas+(study=='H')} if study!='R' else {})))
    row=dict(commit=False,native_step=1,stable=False,confidence=False,converged=False,cap_reached=False,native_winner_sha256='tail',entropy_mean=.02,entropy_slack=.015)
    row['native_history_ready' if study=='R' else 'history_ready']=False
    if study!='R':row['canvas']=len(stories)
    trace.append(row)
    return dict(native_trace=trace,N=sum(len(x) for x in stories)+1,C=len(stories),T=256*len(stories))

def test_native_commit_partition_retains_uncommitted_tail():
    x=request([['a','a'],['b','b','b']]);cs,tail=c.parse(x,'R')
    assert [len(z) for z in cs]==[2,3] and len(tail)==1 and sum(map(len,cs))+len(tail)==x['N']

@pytest.mark.parametrize('study',['C','H'])
def test_pre_and_post_increment_commit_labels(study):
    cs,tail=c.parse(request([['a','a'],['b','b']],study),'C' if study=='C' else 'H')
    assert len(cs)==2 and len(tail)==1

def test_wrong_h_commit_convention_rejected():
    x=request([['a','a']],'H');x['native_trace'][2]['canvas']=0
    with pytest.raises(AssertionError):c.parse(x,'H')

def test_first_l1_can_be_transient_while_l2_is_late_terminal():
    cs,_=c.parse(request([['x','x'],['a','a','b','b','b','b']]),'R')
    x=c.describe_canvas(cs[1],1)
    assert x['terminal_suffix_start']==3
    assert x['opportunities']['L1'][0]==dict(t=2,U=3,suffix='transient',entropy_slack=-.015)
    assert x['opportunities']['L2'][0]['t']==5 and x['opportunities']['L2'][0]['U']==0

def test_next_call_cannot_be_counted_as_saved():
    cs,_=c.parse(request([['x','x'],['a']*6]),'R');d=c.describe_canvas(cs[1],1)
    assert d['opportunities']['L1'][0]['U']==3
    assert d['opportunities']['L2'][0]['U']==2
    assert d['opportunities']['L1'][-1]['U']==0

def test_first_canvas_not_old_l1_l2_trigger():
    cs,_=c.parse(request([['a']*6]),'R')
    assert c.describe_canvas(cs[0],0)['opportunities']=={'L1':[],'L2':[]}

def test_missing_winners_do_not_become_settled_evidence():
    cs,_=c.parse(request([['a']*6]),'R');cs[0][0]['winner_hash']=None
    x=c.describe_canvas(cs[0],1)
    assert x['terminal_suffix_start'] is None and not x['full_winner_history']
    assert all(o['suffix']=='unknown' for o in x['opportunities']['L1'])

@pytest.mark.parametrize('coincident',[False,True])
def test_native_cap_and_adaptive_coincidence(coincident):
    cs,_=c.parse(request([['a']*48],cap=True,coincident=coincident),'R');d=c.describe_canvas(cs[0],1)
    assert d['termination']==('both' if coincident else 'cap')
    assert max(x['t'] for x in d['opportunities']['L1'])==46

def test_entropy_sign_correction_does_not_mutate_historical_record():
    r=request([['a','a']]);old=copy.deepcopy(r);cs,_=c.parse(r,'R')
    assert r==old and cs[0][0]['entropy_slack_canonical']==-.015

def test_nonsequential_steps_and_false_convergence_rejected():
    r=request([['a','a']]);r['native_trace'][1]['native_step']=3
    with pytest.raises(AssertionError):c.parse(r,'R')
    r=request([['a','a']]);r['native_trace'][0]['converged']=True
    with pytest.raises(AssertionError):c.parse(r,'R')
