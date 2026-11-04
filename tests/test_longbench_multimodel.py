import torch

from scripts.longbench_multimodel import fixed_budget_map,fit_prompt
from scripts.longbench_multimodel_report import extractor


def test_k512_is_eight_tiles_not_512_tiles():
    score=torch.arange(512,dtype=torch.float).expand(2,-1)
    eligible=torch.ones_like(score,dtype=torch.bool);mandatory=torch.zeros_like(eligible);mandatory[:,0]=True
    keep=fixed_budget_map(score,mandatory,eligible)
    assert keep.sum(-1).tolist()==[8,8]
    assert keep[:,0].all() and keep[:,-7:].all() and not keep[:,1:-7].any()


def test_budget_short_context_eligibility_and_mandatory_overflow():
    score=torch.ones((3,16));eligible=torch.ones_like(score,dtype=torch.bool);eligible[0,5:]=False
    mandatory=torch.zeros_like(eligible);mandatory[:,4]=True;mandatory[2,:10]=True
    keep=fixed_budget_map(score,mandatory,eligible)
    assert keep.sum(-1).tolist()==[5,8,10]
    assert not (keep&~eligible).any() and (keep|~mandatory).all()
    assert keep[1].tolist()==[True]*8+[False]*8


class CharacterTokenizer:
    def apply_chat_template(self,messages,**kwargs):return [1000,1001]+self.encode(messages[0]['content'])+[1002,1003]
    def encode(self,text,**kwargs):return list(map(ord,text))
    def decode(self,ids,**kwargs):return ''.join(map(chr,ids))


def test_middle_truncation_retains_ends_chat_and_output_reserve():
    tokenizer=CharacterTokenizer();prompt='HEAD'+('x'*300)+'TAIL'
    text,ids,original,truncated=fit_prompt(tokenizer,prompt,200,128)
    assert original==len(prompt)+4 and truncated
    assert text.startswith('HEAD') and text.endswith('TAIL')
    assert ids[:2]==[1000,1001] and ids[-2:]==[1002,1003] and len(ids)+128<=200


def test_prompt_at_context_boundary_is_unchanged():
    tokenizer=CharacterTokenizer();prompt='a'*68
    text,ids,original,truncated=fit_prompt(tokenizer,prompt,200,128)
    assert text==prompt and not truncated and len(ids)==original==72


def test_official_extraction_requires_requested_answer_phrase():
    extract=extractor()
    assert extract('**The correct answer is (B)**')=='B'
    assert extract('The correct answer is C')=='C'
    assert extract('B') is None
    assert extract('') is None
