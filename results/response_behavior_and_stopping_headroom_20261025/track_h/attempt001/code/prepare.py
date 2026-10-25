"""Freeze H identities from OLD dense bottlenecks and outcome-blind raw metadata."""
import hashlib
import json
from pathlib import Path
import sys
import statistics

REPO=Path(__file__).resolve().parents[5];sys.path.insert(0,str(REPO))
from experiments.diffusion_gemma_sparse_attention_research.dynamics_screen import LB_SOURCE,MODEL,official_longbench_prompt


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1];study=root.parents[1]
    assert (study/'track_r/attempt001/readout_frozen.json').exists(),'R readout must be frozen'
    # Deliberately do not open R outcomes. Selection uses old dense traces only.
    census=json.loads((root/'dense_census001/summary.json').read_text())['rows']
    candidates={x['id'] for x in census};ranked=[]
    for identity in candidates:
        pair=[x for x in census if x['id']==identity]
        if len(pair)==2:
            difficulty=min(x['counts'].get('later_confidence_only_delay',0) for x in pair)
            ranked.append((-int(all(x['first_eligible_anchor'] is not None for x in pair)),-difficulty,hashlib.sha256(('H-dense-development-20261025|'+identity).encode()).digest(),identity))
    dev=[x[3] for x in sorted(ranked)[:2]];assert len(dev)==2,'two dense-exposed development identities unavailable'
    old=REPO/'results/denoising_e2e_research_20261024/attempt003'
    previous=json.loads((old/'private/tokenized_manifest.json').read_text());byid={x['id']:x for x in previous}
    excluded={x['id'] for x in previous}
    excluded|={x['id'] for x in json.loads((REPO/'results/denoising_mechanism_stage_a_20261025/attempt002/manifest_identity.json').read_text())['identities']}
    blocked=json.loads((old/'e2e_protocol.json').read_text())['blocked_ids']
    rows=[dict(byid[x],split='development',exposure='Original dense and Response panel exposure; selected ONLY using corrected FULL confidence-only delays') for x in dev]
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);raw=json.loads(LB_SOURCE.read_text());screen=[]
    metadata={x['_id']:x for x in raw}
    for row in rows:
        info=metadata[row['source_id']]
        row.update(domain=info['domain'],difficulty=info['difficulty'],length_band=info['length'])
    strata=[('Long-dialogue History Understanding','hard'),('Long Structured Data Understanding','easy')]
    for domain,difficulty in strata:
        pool=sorted([x for x in raw if x['domain']==domain and x['difficulty']==difficulty and x['length'] in ['medium','long'] and 'longbench_v2/'+x['_id'] not in excluded],
            key=lambda x:hashlib.sha256(('H-confirmation-20261025|'+x['_id']).encode()).digest())
        found=None
        for item in pool:
            identity='longbench_v2/'+item['_id']
            if identity in blocked:continue
            prompt=official_longbench_prompt(item)
            tokens=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True,enable_thinking=True)
            if hasattr(tokens,'keys'):tokens=tokens['input_ids']
            eligible=32768<=len(tokens) and len(tokens)+16384+256<=262144
            screen.append(dict(id=identity,domain=domain,difficulty=difficulty,length_band=item['length'],prompt_token_count=len(tokens),eligible=eligible))
            if eligible:
                found=dict(id=identity,source_id=item['_id'],benchmark='longbench_v2',prompt=prompt,prompt_tokens=list(tokens),prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
                    generation_budget=16384,thinking=True,expected=item['answer'],split='confirmation',domain=domain,difficulty=difficulty,length_band=item['length'],
                    exposure='Outside original Response/A1; other historical benchmark exposure is not certified absent')
                break
        assert found is not None,'unavailable prospective H stratum '+domain+' '+difficulty
        rows.append(found)
    public=[]
    for row in rows:
        assert len(row['prompt_tokens'])+16384<=262144 and row['generation_budget']==16384
        public.append({k:row.get(k) for k in ['id','source_id','split','benchmark','domain','difficulty','length_band','exposure','prompt_hash','generation_budget']} |
            dict(seeds=[42,43],prompt_token_count=len(row['prompt_tokens']),prompt_tokenization_sha256=hashlib.sha256(json.dumps(row['prompt_tokens']).encode()).hexdigest()))
    assert len(rows)==len({x['id'] for x in rows})==4
    (root/'private/manifest.json').write_text(json.dumps(rows))
    with (root/'manifest_identity.json').open('x') as f:json.dump(dict(identities=public,screening=screen,original_blocked_ids=blocked,
        selection='Development first prefers both-seed OLD dense anchors then ranks minimum later confidence-only delay counts; absence of old anchor is retained for prospective common profiling; confirmation domain/difficulty/hash strata outside Response/A1; no R outcomes read',
        source_sha256=sha(LB_SOURCE),private_manifest_sha256=sha(root/'private/manifest.json'),reserved_postcap_canvas_tokens=256,dense_census_sha256=sha(root/'dense_census001/summary.json')),f,indent=2)
    print(json.dumps(public))


if __name__=='__main__':main()
