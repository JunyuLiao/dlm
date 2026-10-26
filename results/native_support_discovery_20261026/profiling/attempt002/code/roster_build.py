"""New finite roster; only identity/metadata/exposure, no sparse outcomes."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT.parents[1];REPO=ROOT.parents[3]
sys.path.insert(0,str(REPO))
from experiments.diffusion_gemma_sparse_attention_research.dynamics_screen import AIME_SOURCE,HUMANEVAL_SOURCE,LB_SOURCE,MODEL,official_longbench_prompt
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(identity):return hashlib.sha256(('native-support-discovery-roster-20261026-v1|'+identity).encode()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True)
def main():
    assert not (ROOT/'roster_identity.json').exists()
    known={};bindings={}
    base=REPO/'results'
    for label,path in [('R',base/'response_behavior_and_stopping_headroom_20261025/track_r/attempt001/manifest_identity.json'),('H',base/'response_behavior_and_stopping_headroom_20261025/track_h/attempt001/manifest_identity.json'),('A1',base/'denoising_mechanism_stage_a_20261025/attempt002/manifest_identity.json'),('Response',base/'denoising_e2e_research_20261024/attempt003/manifest_identity.json'),('C_profile',base/'commit_boundary_single_pulse_20261026/panel/attempt004/roster_identity.json')]:
        v=json.loads(path.read_text());bindings[str(path.relative_to(REPO))]=sha(path)
        for row in v.get('identities',v.get('items',[])):known.setdefault(row['id'],[]).append(label)
    blocked=json.loads((base/'denoising_e2e_research_20261024/attempt003/e2e_protocol.json').read_text())['blocked_ids']
    selected=[];screen=[]
    for task,path in [('aime26',AIME_SOURCE),('humaneval',HUMANEVAL_SOURCE)]:
        pool=[dict(x,id=task+'/'+str(x['source_id']),benchmark=task) for x in json.loads(path.read_text())]
        if task=='humaneval':pool=[x for x in pool if x['id'] not in known]
        pool=sorted(pool,key=lambda x:key(x['id']))
        if task=='aime26':
            chosen=pool[:12];assert len(chosen)==12
            for i,row in enumerate(chosen):selected.append(dict(row,partition='development' if i%2==0 else 'confirmation',roster_stratum='identity_coverage'))
        else:
            rich=[x for x in pool if len(x['prompt_tokens'])>=256][:6]
            deficit=6-len(rich)
            rich+= [x for x in pool if x not in rich][:deficit]
            coverage=[x for x in pool if x not in rich][:6];assert len(rich)==len(coverage)==6
            for name,group in [('prefix_rich_candidate',rich),('coverage',coverage)]:
                for i,row in enumerate(group):selected.append(dict(row,partition='development' if i%2==0 else 'confirmation',roster_stratum=name,prefix_rich_fallback=len(row['prompt_tokens'])<256 if name=='prefix_rich_candidate' else False))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    raw=json.loads(LB_SOURCE.read_text());cache={};used=set()
    domains=['Single-Document QA','Multi-Document QA','Long-dialogue History Understanding','Long Structured Data Understanding']
    slots={'development':[(0,'high'),(1,'low'),(2,'low'),(3,'high'),(0,'low'),(1,'high')], 'confirmation':[(0,'low'),(1,'high'),(2,'high'),(3,'low'),(2,'low'),(3,'high')]}
    for partition,items in slots.items():
        for domain_index,requested in items:
            domain=domains[domain_index];found=None
            for band in [requested]+(['low'] if requested=='high' else []):
                pool=sorted([x for x in raw if x['domain']==domain],key=lambda x:key('longbench_v2/'+x['_id']))
                for x in pool:
                    identity='longbench_v2/'+x['_id']
                    if identity in known or identity in blocked or identity in used:continue
                    if identity not in cache:
                        prompt=official_longbench_prompt(x);tokens=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True,enable_thinking=True)
                        if hasattr(tokens,'keys'):tokens=tokens['input_ids']
                        cache[identity]=(prompt,list(tokens))
                    prompt,tokens=cache[identity];eligible=(32768<=len(tokens)<131072 if band=='low' else 131072<=len(tokens)) and len(tokens)+16384+256<=262144
                    screen.append(dict(id=identity,partition=partition,domain=domain,requested_band=requested,actual_band=band,prompt_token_count=len(tokens),eligible=eligible))
                    if not eligible:continue
                    found=dict(id=identity,source_id=x['_id'],benchmark='longbench_v2',prompt=prompt,prompt_tokens=tokens,prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),expected=x['answer'],generation_budget=16384,thinking=True,domain=domain,length_band=x['length'],native_band=band,difficulty=x['difficulty'],requested_band=requested,band_fallback=band!=requested,partition=partition,roster_stratum='domain_context_coverage');break
                if found:break
            assert found,('unavailable slot before any new baseline',partition,domain,requested)
            used.add(found['id']);selected.append(found)
    assert len(selected)==36 and len({r['id'] for r in selected})==36
    public=[]
    for row in selected:
        row['roster_key']=key(row['id']);row['exposure']=known.get(row['id'],[])
        assert row['generation_budget']==(16384 if row['benchmark']=='longbench_v2' else 8192)
        v={k:row.get(k) for k in ['id','source_id','benchmark','domain','length_band','difficulty','requested_band','native_band','band_fallback','partition','roster_stratum','prefix_rich_fallback','roster_key','exposure','generation_budget','thinking','prompt_hash']}
        v.update(prompt_token_count=len(row['prompt_tokens']),prompt_tokenization_sha256=hashlib.sha256(json.dumps(row['prompt_tokens']).encode()).hexdigest(),seeds=[42,43],other_historical_exposure_not_certified_absent=True)
        public.append(v)
    for task in ['aime26','humaneval','longbench_v2']:
        for part in ['development','confirmation']:assert sum(x['benchmark']==task and x['partition']==part for x in public)==6
    save(ROOT/'private/roster.json',selected)
    save(ROOT/'roster_identity.json',dict(identities=public,screening=screen,blocked_ids=blocked,exposure_manifest_hashes=bindings,private_roster_sha256=sha(ROOT/'private/roster.json'),source_pool_hashes={p.name:sha(p) for p in [AIME_SOURCE,HUMANEVAL_SOURCE,LB_SOURCE]},no_sparse_outcomes_used=True))
    print(json.dumps(dict(roster=len(public),partitions={p:sum(x['partition']==p for x in public) for p in ['development','confirmation']},eligible_LB=[dict(id=x['id'],partition=x['partition'],domain=x['domain'],tokens=x['prompt_token_count'],fallback=x['band_fallback']) for x in public if x['benchmark']=='longbench_v2'])))
if __name__=='__main__':main()
