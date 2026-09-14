"""Bounded development-only follow-up, with hashed completed-parent evidence."""
import argparse
import json
from pathlib import Path
from .config import ROOT,RouterConfig
from .allocation import choose_groups,layer_sensitivities
from .run import digest
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write


def build_conditions(selection):
    cases=selection['cases']
    if not 1<=len(cases)<=6:raise ValueError('bounded refinement requires 1–6 cases')
    if not selection.get('parent_evidence'):raise ValueError('completed parent evidence required')
    conditions={}
    for case in cases:
        benchmark=case['benchmark'];target=case['target_sparsity'];label=case['label']
        if benchmark not in ('aime24','longbench') or target not in (.4,.5):raise ValueError('unscreened refinement')
        options={k:case[k] for k in ('isolate','allocation_policy') if k in case}
        if len(options)>1:raise ValueError('isolate and allocation are independent tests')
        sources=case.get('sources',['sparse'])
        if not sources or set(sources)-{'sparse','dense_reference'}:raise ValueError('invalid history source')
        for source in sources:
            config=RouterConfig(sparsity=target,history_source=source,
                allocation='paired_layer_shift' if 'allocation_policy' in case else 'uniform')
            name=f'{benchmark}__{label}_s{int(target*100)}__{source}'
            if name in conditions:raise ValueError('duplicate refinement')
            conditions[name]=dict(benchmark=benchmark,predictor=label,config=config.to_dict(),**options)
    return conditions


def prepare(root,stage):
    from .online_report import report
    from .online_run import freeze
    parent=root/'online_v1';report(parent)
    audit=json.loads((parent/'audit.json').read_text())
    if not audit['complete']:raise RuntimeError('parent not complete')
    protocol=json.loads((root/'protocol.json').read_text())
    model_config=json.loads((Path(protocol['model'])/'config.json').read_text())['text_config']
    layer_types=model_config['layer_types']
    allocation=choose_groups(root/'screen_summary.csv',layer_types)
    sensitivity=layer_sensitivities(root/'screen_summary.csv','aime24')
    isolated={kind:max((i for i,t in enumerate(layer_types) if ('local' if t=='sliding_attention' else 'global')==kind),key=lambda i:sensitivity[i]) for kind in ('local','global')}
    cases=[dict(benchmark='longbench',label='last_mass_control',target_sparsity=.5),
        dict(benchmark='longbench',label='paired_layer_shift',target_sparsity=.5,allocation_policy=allocation),
        dict(benchmark='aime24',label='last_mass',target_sparsity=.4,sources=['dense_reference','sparse'])]
    cases += [dict(benchmark='aime24',label='isolated_'+kind,target_sparsity=.5,isolate=[layer,1]) for kind,layer in isolated.items()]
    selection=dict(phase='refinement',parent_stage=str(parent),
        parent_evidence={n:digest(parent/n) for n in ('audit.json','summary.csv','freeze.json')},cases=cases,
        hypothesis='AIME s40 may preserve score; one first-reuse perturbation separates accumulation from local sensitivity; LongBench coarse allocation may improve fixed-budget preservation.',
        decision_rationale='Reject beginning protection on AIME and diagonal on LongBench total-work grounds. Fix plain last-mass ranking; shift one tile between equal-sized high/low development-error groups within attention type. Uniform shadow masks compare the same sparse state. No held-out outcomes used.',
        controls='Fresh sparse LongBench uniform rollout is a regression and allocated-trajectory control; cached dense generations are not rerun. AIME single-call cases perturb only the first matching layer/step in the whole request.',
        primary_allocation_comparison='Exact total per-forward/head/query-tile retention within type; report total rollout work separately because denoising duration may change.',
        isolated_layers=isolated)
    path=stage/'selection.json'
    if path.exists() and json.loads(path.read_text())!=selection:raise RuntimeError('immutable selection changed')
    _write(path,selection);freeze(root,stage,path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--stage',type=Path,default=ROOT/'refinement_v1');a=p.parse_args();prepare(a.root,a.stage)
