"""Exact baseline cache imports and calibration-only policy transfers."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from experiments.diffusion_gemma_jl_output_aware import runner, calibration, screen, validation, shared_analysis
from experiments.diffusion_gemma_jl_output_aware.test_gate import require as require_old_tests
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_value_aware_followup.engine import check_result
from experiments.diffusion_gemma_value_aware_followup.evidence import check_sources
from experiments.diffusion_gemma_longbench_v2_100.calibration import condition as lb_condition
from .protocol import OLD, LB, AIME, PROJECTED, BASELINES, TARGETS, read, sha, frozen_write


@contextmanager
def dispatch():
    """Only process-local dispatch changes; frozen routing/kernel files untouched."""
    with patch.object(runner,'PROJECTED',PROJECTED):
        yield


def audit_policy(root, p, setup, contract):
    if not p.get('imported_policy'):
        return calibration.audit_policy(root,p,setup,contract)
    if p['fingerprint'] != contract['fingerprint'] or p['heldout_used']:
        raise ValueError('Imported threshold identity/selection mismatch')
    check_sources(p['sources'])
    src=Path(p['imported_policy']); old=read(src)
    source_root=Path(p['source_root'])
    if p['import_kind']=='jl_calibration':
        calibration.audit_policy(source_root,old,read(source_root/'setup.json'),read(source_root/'execution_contract.json'))
        if old['config']!=p['config'] or old['policy']!=p['policy']:
            raise ValueError('Imported JL policy/operator changed')
    elif p['import_kind']=='lb_baseline':
        checked=lb_condition(source_root,read(source_root/'setup.json'),read(source_root/'execution_contract.json'),p['name'],p['target'])
        if checked['config']!=p['config'] or checked['thresholds']['longbench_v2']!=p['policy']:
            raise ValueError('Imported LongBench baseline policy changed')
    else:
        raise ValueError('Unknown policy import convention')


def import_policies(root,setup,contract):
    """Never uses final score or a rank16 policy for rank32 contribution."""
    for benchmark in ('aime26','longbench_v2'):
        for target in TARGETS:
            for name,cfg in {**BASELINES,**{n:c for n,c in PROJECTED.items() if n!='contribution_gaussian_r32'}}.items():
                label=f'{name}_s{int(100*target)}'
                dest=root/'policies'/benchmark/f'{label}.json'
                if dest.exists():
                    audit_policy(root,read(dest),setup,contract); continue
                if name in BASELINES and benchmark=='aime26':
                    calibration.import_aime_baseline(root,name,target,cfg,setup,contract)
                    continue
                source_root=OLD if name in PROJECTED else LB
                src=source_root/('policies' if name in PROJECTED else 'verified_policies')/benchmark/f'{label}.json'
                old=read(src)
                p=dict(fingerprint=contract['fingerprint'],benchmark=benchmark,name=name,config=cfg,target=target,
                    policy=old['policy'],measured=old.get('measured'),heldout_used=False,
                    cap_one_unattainable=old.get('cap_one_unattainable'),
                    imported_policy=str(src),source_root=str(source_root),
                    import_kind='jl_calibration' if name in PROJECTED else 'lb_baseline',
                    calibration_ids=old.get('calibration_ids',old.get('source_ids')),
                    rule='Unchanged verified calibration-only policy; no final retuning; raw trace remains under its original execution fingerprint',
                    sources={str(p):sha(p.read_bytes()) for p in (src,source_root/'setup.json',source_root/'execution_contract.json')})
                audit_policy(root,p,setup,contract); frozen_write(dest,p)
                print('imported policy',benchmark,label,flush=True)


def import_baselines(root,setup,contract):
    """Losslessly compress cached buckets; exact prompt/settings/policy required."""
    source_setups={b:read(p/'setup.json') for b,p in (('aime26',AIME),('longbench_v2',LB))}
    sources={}; imported=[]
    for benchmark,parent in (('aime26',AIME),('longbench_v2',LB)):
        prior=read(parent/'execution_contract.json'); sources.update(prior['sources'])
        for label in ['dense']+[f'{n}_s{int(t*100)}' for t in TARGETS for n in BASELINES]:
            c=read(parent/'final_configs'/f'{label}.json'); sources.update(c['sources'])
    check_sources(sources)
    for row in setup['final']:
        parent=AIME if row['benchmark']=='aime26' else LB
        previous=next(r for r in source_setups[row['benchmark']]['final'] if r['id']==row['id'])
        if previous != row:
            raise ValueError('Cached row is not a verbatim source-manifest row')
        prior=read(parent/'execution_contract.json')
        for label in ['dense']+[f'{n}_s{int(t*100)}' for t in TARGETS for n in BASELINES]:
            name=label.rsplit('_s',1)[0] if label!='dense' else 'dense'
            cfg={} if name=='dense' else BASELINES[name]
            threshold=None if name=='dense' else read(root/'policies'/row['benchmark']/f'{label}.json')['policy']
            c=read(parent/'final_configs'/f'{label}.json')
            if c['config']!=cfg or (name!='dense' and c['thresholds'][row['benchmark']]!=threshold):
                raise ValueError('Baseline source config/policy mismatch')
            stage='dense' if name=='dense' else 'final'
            path=shard_path(root,stage,label,row['id'])
            source=shard_path(parent,stage,label,row['id'])
            if path.exists():
                out=runner.cached(None,root,row,stage,label,name,cfg,threshold,contract)
            else:
                out=runner.load_output(source)
                check_result(out,row,prior['fingerprint'],cfg,threshold)
                out.update(fingerprint=contract['fingerprint'],candidate=name,
                    imported_source=dict(path=str(source),sha256=sha(source.read_bytes()),original_fingerprint=prior['fingerprint']),
                    diagnostic_convention='Historical cached diagnostic reduction; uniform local-operator comparisons use saved shared QKV')
                check_result(out,row,contract['fingerprint'],cfg,threshold)
                runner.write_output(path,out)
            imported.append(dict(id=row['id'],condition=label,path=str(path),sha256=sha(path.read_bytes()),source=out['imported_source']))
        print('reused7 baseline outputs',row['id'],flush=True)
    if len(imported)!=560:
        raise ValueError('Incomplete matched baseline imports')
    frozen_write(root/'baseline_imports.json',dict(count=560,sources=sources,imports=imported))


def import_shared(root,setup,contract):
    require_old_tests(OLD,read(OLD/'execution_contract.json'))
    sources=read(OLD/'shared_state_index.json')
    ids={r['id'] for r in setup['calibration']}
    if len(sources)!=540 or any(s['split']!='calibration' or s['id'] not in ids for s in sources):
        raise ValueError('Invalid shared calibration-state coverage')
    check_sources({s['path']:s['sha256'] for s in sources})
    frozen_write(root/'shared_state_index.json',sources)
    frozen_write(root/'inherited_tensor_validation.json',dict(passed=True,
        source=str(OLD/'tensor_validation.json'),sha256=sha((OLD/'tensor_validation.json').read_bytes()),
        note='Unchanged custom kernels;48 CUDA/CPU tests passed,3 inapplicable guard cases skipped; includes Gaussian rank32 contribution'))
    return sources


def contribution_screen(root,sources,contract):
    with patch.object(screen,'PROJECTED',{'contribution_gaussian_r32':PROJECTED['contribution_gaussian_r32']}),patch.object(screen,'BASELINES',{}):
        return screen.shared_screen(root,sources,contract)


def smoke(adapter,root,setup,contract):
    # Two actual-model inputs; all three exact configs, including new rank32
    # contribution. Reusing kernels is not grounds to skip the dispatch check.
    with dispatch(),patch.object(validation,'PROJECTED',PROJECTED):
        return validation.smoke(adapter,root,setup,contract)


def diagnostics(root,sources,contract):
    with patch.object(shared_analysis,'PROJECTED',PROJECTED),patch.object(shared_analysis,'BASELINES',BASELINES):
        return shared_analysis.analyze(root,sources,contract)
