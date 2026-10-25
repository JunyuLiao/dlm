"""Pre-treatment identity selection using source metadata, never sparse outcomes."""
import hashlib
import json
from pathlib import Path
import sys

REPO=Path('/home/exouser/ljy/dlm')
sys.path.insert(0,str(REPO))
from experiments.diffusion_gemma_sparse_attention_research.dynamics_screen import AIME_SOURCE,HUMANEVAL_SOURCE,LB_SOURCE,MODEL,official_longbench_prompt


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def order(x):return hashlib.sha256(('stage-a-20261025|'+str(x)).encode()).digest()


def main():
    from transformers import AutoTokenizer
    root=Path(__file__).resolve().parents[1]
    prior=REPO/'results/denoising_e2e_research_20261024/attempt003'
    previous=json.loads((prior/'private/tokenized_manifest.json').read_text())
    oldids={r['id'] for r in previous}
    blocked=json.loads((prior/'e2e_protocol.json').read_text())['blocked_ids']
    rows=[]
    for benchmark,source,dev in [('aime26',AIME_SOURCE,['18','12']),('humaneval',HUMANEVAL_SOURCE,['HumanEval/124','HumanEval/70'])]:
        pool=json.loads(source.read_text()); byid={str(r['source_id']):r for r in pool}
        confirmation=sorted([r for r in pool if benchmark+'/'+str(r['source_id']) not in oldids and r['prompt_token_count']>=256],key=lambda x:order(x['source_id']))[:2]
        assert len(confirmation)==2
        for split,items in [('development',[byid[x] for x in dev]),('confirmation',confirmation)]:
            for item in items:
                row=dict(item);row.update(id=benchmark+'/'+str(item['source_id']),split=split,benchmark=benchmark,seeds=[42,43],cohort='mechanistic_context_stratified_not_representative',
                    exposure='Inside prior25 response screen' if benchmark+'/'+str(item['source_id']) in oldids else 'Outside prior25 response screen; historical V31/V2 benchmark exposure exists or is not excluded; method confirmation, not unseen benchmark')
                rows.append(row)
    original=next(r for r in previous if r['id']=='longbench_v2/66ec2755821e116aacb1b6f2')
    rows.append(dict(original,split='development',seeds=[42,43],cohort='targeted_slow_long_context',exposure='Prior dense and sparse screen; dense-only metadata shows confidence/cap difficulty'))
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    raw=json.loads(LB_SOURCE.read_text());screening=[]
    strata=[('development','Code Repository Understanding','short',0,32768),
            ('confirmation','Long-dialogue History Understanding','short',0,32768),
            ('confirmation','Long Structured Data Understanding','medium',32768,131072)]
    for split,domain,band,low,high in strata:
        pool=sorted([x for x in raw if x['domain']==domain and x['length']==band and 'longbench_v2/'+x['_id'] not in oldids],key=lambda x:order(x['_id']))
        found=False
        for item in pool:
            rid='longbench_v2/'+item['_id']
            if rid in blocked:continue
            prompt=official_longbench_prompt(item)
            tokens=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True,enable_thinking=True)
            if hasattr(tokens,'keys'):tokens=tokens['input_ids']
            eligible=low<=len(tokens)<high and len(tokens)+16384<=262144
            screening.append(dict(id=rid,domain=domain,band=band,tokens=len(tokens),eligible=eligible))
            if eligible:
                rows.append(dict(id=rid,source_id=item['_id'],benchmark='longbench_v2',split=split,seeds=[42,43],domain=domain,length_band=band,difficulty=item['difficulty'],
                    prompt=prompt,prompt_tokens=list(tokens),prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),expected=item['answer'],generation_budget=16384,thinking=True,
                    cohort='context_domain_stratified_mechanistic_not_representative',exposure='Outside prior25 response screen; historical benchmark exposure not certified absent; prospective dense profile, no sparse-outcome selection'))
                found=True;break
        if not found:raise RuntimeError('unavailable pre-treatment stratum '+domain)
    assert len(rows)==12 and len({r['id'] for r in rows})==12
    for row in rows:
        tokens=tok.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=True,add_generation_prompt=True,enable_thinking=row['thinking'])
        if hasattr(tokens,'keys'):tokens=tokens['input_ids']
        assert list(tokens)==row['prompt_tokens']
        assert len(tokens)+row['generation_budget']<=262144
    (root/'private/manifest.json').write_text(json.dumps(rows))
    public=[]
    for row in rows:
        public.append({k:row[k] for k in ['id','source_id','benchmark','split','seeds','cohort','exposure','generation_budget','thinking','prompt_hash']} |
                      dict(prompt_token_count=len(row['prompt_tokens']),prompt_tokenization_sha256=hashlib.sha256(json.dumps(row['prompt_tokens']).encode()).hexdigest(),
                           domain=row.get('domain'),length_band=row.get('length_band'),difficulty=row.get('difficulty')))
    result=dict(identities=public,blocked_original_ids=blocked,strata_screening=screening,
        selection='Two exposed short development questions; two >=256-token questions outside prior25 per short benchmark by SHA256. Long original slow item plus three outcome-blind domain/length strata. Dense difficulty prospectively profiled and reported, never used to replace identities.',
        seeds=[42,43],private_manifest_sha256=sha(root/'private/manifest.json'),source_hashes={'AIME_source':sha(AIME_SOURCE),'HumanEval_source':sha(HUMANEVAL_SOURCE),'LongBench_source':sha(LB_SOURCE)})
    (root/'manifest_identity.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(public,indent=2))


if __name__=='__main__':main()
