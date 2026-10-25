"""Prospective finite roster from identity/metadata/exposure, no outcomes."""
from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[3];sys.path.insert(0,str(REPO))
from experiments.diffusion_gemma_sparse_attention_research.dynamics_screen import AIME_SOURCE,HUMANEVAL_SOURCE,LB_SOURCE,MODEL,official_longbench_prompt

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(x):return hashlib.sha256(('commit-boundary-panel-roster-20261026|'+x).encode()).hexdigest()
def main():
 old=REPO/'results/response_behavior_and_stopping_headroom_20261025';known={}
 for label,path in [('R',old/'track_r/attempt001/manifest_identity.json'),('H',old/'track_h/attempt001/manifest_identity.json'),('A1',REPO/'results/denoising_mechanism_stage_a_20261025/attempt002/manifest_identity.json'),('Response',REPO/'results/denoising_e2e_research_20261024/attempt003/manifest_identity.json')]:
  d=json.loads(path.read_text());ids=d.get('identities',d.get('items',[]))
  for x in ids:known.setdefault(x['id'],[]).append(label)
 blocked=json.loads((REPO/'results/denoising_e2e_research_20261024/attempt003/e2e_protocol.json').read_text())['blocked_ids']
 rows=[];screen=[]
 for b,path in [('aime26',AIME_SOURCE),('humaneval',HUMANEVAL_SOURCE)]:
  pool=[dict(r,id=b+'/'+str(r['source_id'])) for r in json.loads(path.read_text())]
  if b=='humaneval':pool=[r for r in pool if r['id'] not in known]
  chosen=sorted(pool,key=lambda r:key(r['id']))[:8];assert len(chosen)==8
  rows.extend(chosen)
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);raw=json.loads(LB_SOURCE.read_text())
 domains=['Single-Document QA','Multi-Document QA','Long-dialogue History Understanding','Long Structured Data Understanding']
 for domain in domains:
  chosen=[]
  for band in ['medium','long']:
   # Long unavailable/overlimit => prospective next eligible medium fallback.
   bands=[band]+(['medium'] if band=='long' else [])
   found=None
   for actual_band in bands:
    pool=sorted([x for x in raw if x['domain']==domain and x['length']==actual_band and 'longbench_v2/'+x['_id'] not in known and x['_id'] not in chosen],key=lambda x:key('longbench_v2/'+x['_id']))
    for x in pool:
     identity='longbench_v2/'+x['_id']
     if identity in blocked:continue
     prompt=official_longbench_prompt(x);tokens=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True,enable_thinking=True)
     if hasattr(tokens,'keys'):tokens=tokens['input_ids']
     eligible=32768<=len(tokens) and len(tokens)+16384+256<=262144
     screen.append(dict(id=identity,domain=domain,length_band=actual_band,prompt_token_count=len(tokens),eligible=eligible,requested_band=band))
     if eligible:
      found=dict(id=identity,source_id=x['_id'],benchmark='longbench_v2',prompt=prompt,prompt_tokens=list(tokens),prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),generation_budget=16384,thinking=True,expected=x['answer'],domain=domain,difficulty=x['difficulty'],length_band=actual_band,requested_band=band);break
    if found:break
   assert found,('unavailable roster stratum',domain,band)
   chosen.append(found['source_id']);rows.append(found)
 public=[]
 for r in rows:
  r['exposure']=known.get(r['id'],[]);r['roster_key']=key(r['id']);r['stratum']='profiling_roster'
  public.append({k:r.get(k) for k in ['id','source_id','benchmark','domain','difficulty','length_band','requested_band','exposure','roster_key','thinking','generation_budget','prompt_hash']}|dict(prompt_token_count=len(r['prompt_tokens']),prompt_tokenization_sha256=hashlib.sha256(json.dumps(r['prompt_tokens']).encode()).hexdigest(),seeds=[42,43],historical_other_exposure_not_certified_absent=True))
 assert len(rows)==24 and len({r['id'] for r in rows})==24
 with (ROOT/'private/roster.json').open('x') as f:json.dump(rows,f)
 with (ROOT/'roster_identity.json').open('x') as f:json.dump(dict(identities=public,screening=screen,blocked_ids=blocked,private_roster_sha256=sha(ROOT/'private/roster.json'),source_hashes={p.name:sha(p) for p in [AIME_SOURCE,HUMANEVAL_SOURCE,LB_SOURCE]},no_sparse_outcomes_used=True),f,indent=2,sort_keys=True)
 print(json.dumps(public,indent=1))
if __name__=='__main__':main()
