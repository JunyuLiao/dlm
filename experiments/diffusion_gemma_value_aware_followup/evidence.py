"""Read-only raw evidence and metrics shared by development and final reports."""
import json
import math
from pathlib import Path

from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_value_aware.report_metrics import aggregate
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_report import token_counts
from .engine import check_result
from .execution import provenance
from .protocol import score, official_extractor, sha

DECODING_FIELDS=('native_canvas_length','thinking','sampling','denoising_configuration')


def raw_source(path):
    path=Path(path);raw=path.read_bytes()
    return json.loads(raw),dict(path=str(path),sha256=sha(raw))


def read_result(root,row,stage,name,config,thresholds,execution,screen=False):
    """Never infer, import a new cache, or mutate the import index during audit."""
    path=shard_path(root,stage,name,row['id']);fp=execution['fingerprint'];entry=None
    if not path.exists():
        alias_path=root/'final_aliases.json'
        aliases=json.loads(alias_path.read_text()) if stage=='final' and alias_path.exists() else {}
        alias=aliases.get(f'{name}/{row["id"]}')
        imports_path=root/'imported_sources.json'
        imports=json.loads(imports_path.read_text()) if imports_path.exists() else {}
        entry=alias or imports.get(f'{stage}/{name}/{row["id"]}')
        if entry is None:
            raise FileNotFoundError(path)
        if alias and alias['source_stage']!='final':
            raise ValueError('final aliases require another completed final execution')
        if alias and alias['fingerprint']==execution['fingerprint']:
            pass
        elif row['benchmark']!='aime26' or entry['fingerprint']!=execution['previous_fingerprint']:
            raise ValueError('only an explicitly audited matching AIME source may be imported')
        path=Path(entry['path']);fp=entry['fingerprint']
    data,source=raw_source(path)
    if entry and source['sha256']!=entry['sha256']:
        raise ValueError('imported generation source changed')
    check_result(data,row,fp,config,thresholds,screen=screen)
    if not screen:
        for key,expected in provenance(config).items():
            if data.get(key)!=expected:
                raise ValueError('generation operator source changed')
    return data,dict(source,fingerprint=fp)


def pair(row,sparse,dense):
    """Same-state attention diagnostics; positional token comparison even after divergence."""
    for label,out in (('sparse',sparse),('dense',dense)):
        if any(out[k]!=row[k] for k in ('id','prompt_hash','seed','generation_budget')):
            raise ValueError(f'{label} pair prompt/seed/budget mismatch')
        if out.get('score') is not None and not math.isclose(out['score'],score(row,out['prediction']),rel_tol=0.,abs_tol=1e-12):
            raise ValueError(f'{label} cached benchmark score differs from official rescoring')
    if dense['config'] or dense['thresholds'] is not None:
        raise ValueError('dense pair reference has a sparse configuration')
    for key in DECODING_FIELDS:
        if sparse['generation_metadata'].get(key)!=dense['generation_metadata'].get(key):
            raise ValueError(f'unrelated decoding setting changed: {key}')
    records=[r for r in sparse['records'] if r['probe']=='execution']
    matching,compared=token_counts(dense['completion_tokens'],sparse['completion_tokens'])
    answer=official_extractor()(sparse['prediction'].strip()) if row['benchmark']=='longbench_v2' else None
    return dict(id=row['id'],benchmark=row['benchmark'],task=row['task'],
        calibration=row.get('calibration',False),seed=row['seed'],
        prompt_hash=row['prompt_hash'],generation_budget=row['generation_budget'],
        accuracy=score(row,sparse['prediction']),dense_accuracy=score(row,dense['prediction']),
        matching=matching,compared=compared,exact_match=sparse['completion_tokens']==dense['completion_tokens'],
        output_length=len(sparse['completion_tokens']),termination_reason=sparse['termination_reason'],
        parsed_answer=answer,unparsed_answer=answer is None and row['benchmark']=='longbench_v2',
        aggregates={kind:aggregate([r for r in records if kind=='overall' or r['attention_type']==kind])
            for kind in ('overall','local','global')})


def check_sources(sources):
    for path,digest in sources.items():
        if sha(Path(path).read_bytes())!=digest:
            raise ValueError(f'frozen evidence source changed: {path}')


def merge_sources(*parts):
    result={}
    for part in parts:
        for path,digest in part.items():
            if path in result and result[path]!=digest:
                raise ValueError(f'inconsistent source snapshots: {path}')
            result[path]=digest
    return result


def require_sparse_smoke(root,setup,execution,configs):
    """Prove each requested family's smoke from raw outputs, not a passed flag."""
    from .calibrate import smoke_verified
    examples=[next(r for r in setup['calibration'] if r['benchmark']=='aime26'),
        max((r for r in setup['calibration'] if r['benchmark']=='longbench_v2'),key=lambda r:len(r['prompt_tokens']))]
    summaries=[]
    for path in sorted((root/'sparse_smoke').glob('*.json')):
        data=json.loads(path.read_text())
        if data.get('identity',{}).get('fingerprint')==execution['fingerprint']:
            summaries.append((path,data))
    sources={}
    for name,config in configs.items():
        matches=[(p,d) for p,d in summaries if d['identity']['configs'].get(name)==config
            and d['identity']['provenance'].get(name)==provenance(config)
            and name in smoke_verified(d['tests'],[name],[r['id'] for r in examples])]
        if not matches:
            raise ValueError(f'no complete matching CUDA family smoke: {name}')
        path,summary=matches[-1];sources[str(path)]=sha(path.read_bytes())
        for example in examples:
            row=dict(example,generation_budget=16)
            native_path=shard_path(root,'sparse_smoke','native_dense16',row['id'])
            native,source=raw_source(native_path);sources[source['path']]=source['sha256']
            expected=dict(fingerprint=execution['fingerprint'],
                **{k:row[k] for k in ('id','prompt_hash','seed','generation_budget')})
            if native['identity']!=expected:
                raise ValueError('native smoke reference mismatch')
            for unpruned in (True,False):
                threshold=-100. if unpruned else (0. if config['method'] in ('blasst','value','aligned') else -1.)
                policy={kind:dict(log_threshold=threshold) for kind in ('local','global')}
                case=f'{name}_{"unpruned" if unpruned else "sparse"}_16'
                result,source=read_result(root,row,'sparse_smoke',case,config,policy,execution)
                sources[source['path']]=source['sha256']
                if unpruned and (result['completion_tokens']!=native['completion_tokens']
                        or any(r['pv_omitted'] for r in result['records'])):
                    raise ValueError('unpruned CUDA smoke parity/retention failed')
                for key in DECODING_FIELDS:
                    if result['generation_metadata'].get(key)!=native['generation_metadata'].get(key):
                        raise ValueError('CUDA smoke unrelated decoding change')
    return sources
