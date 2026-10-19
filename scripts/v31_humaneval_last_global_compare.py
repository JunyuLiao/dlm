"""Compare paired native layer29 captures without treating a diagnostic as qualification."""
import argparse
import hashlib
import json
from pathlib import Path

FIELDS=('output_hash','denoise_forwards','canvases','output_tokens','sampler_calls','native_interval_steps')
ORDER=('query','key','value','consumers','anchors','merge','correction','output','contiguous_cache','kept','held_lists')


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True)


def differences(a,b,path=''):
    """Preserve producer order and never silently compare unequal list coverage."""
    if type(a)!=type(b):return [dict(path=path,kind='type')]
    if isinstance(a,dict):
        result=[]
        for key in dict.fromkeys([*a,*b]):
            if key not in a or key not in b:result.append(dict(path=f'{path}/{key}',kind='missing'))
            else:result.extend(differences(a[key],b[key],f'{path}/{key}'))
        return result
    if isinstance(a,list):
        result=[]
        if len(a)!=len(b):result.append(dict(path=path,kind='coverage',left=len(a),right=len(b)))
        for i,(x,y) in enumerate(zip(a,b)):result.extend(differences(x,y,f'{path}/{i}'))
        return result
    return [] if a==b else [dict(path=path,kind='value')]


def event_contrast(left,right):
    rows=[];physical=[]
    for ordinal,(a,b) in enumerate(zip(left,right)):
        geometry={k:a.get(k)==b.get(k) for k in ('sampler_index','context','canvas','arm')}
        changed=[]
        for key in ORDER:
            aa,bb=a.get(key),b.get(key)
            if key=='consumers' and isinstance(aa,list) and isinstance(bb,list):
                fields=('page_table','paged_pages_including_padding')
                allocation=differences([{k:x.get(k) for k in fields} for x in aa],
                                       [{k:x.get(k) for k in fields} for x in bb],key)
                if allocation:physical.append(dict(ordinal=ordinal,sampler_index=a['sampler_index'],fields=allocation))
                aa=[{k:v for k,v in x.items() if k not in fields} for x in aa]
                bb=[{k:v for k,v in x.items() if k not in fields} for x in bb]
            changed.extend(differences(aa,bb,key))
        if changed or not all(geometry.values()):
            rows.append(dict(ordinal=ordinal,left_sampler_index=a['sampler_index'],right_sampler_index=b['sampler_index'],
                same_geometry=geometry,first=changed[0] if changed else None,changed_fields=changed,
                left_context=a['context'],right_context=b['context']))
            break
    return dict(left_events=len(left),right_events=len(right),equal_coverage=len(left)==len(right),
        equal=len(left)==len(right) and not rows,first_difference=rows[0] if rows else None,
        physical_translation_or_padding_changes=dict(events=len(physical),first=physical[0] if physical else None),
        scope='recorded layer29 logical used KV/Q, arguments, moments, cache, partials, merge and outputs; excludes page translation and unused padding')


def tensor_difference(a,b):
    import torch
    if a.shape!=b.shape or a.dtype!=b.dtype:return dict(equal=False,kind='geometry',left_shape=list(a.shape),right_shape=list(b.shape))
    x,y=a.contiguous().reshape(-1),b.contiguous().reshape(-1)
    byte_equal=torch.equal(x.view(torch.uint8),y.view(torch.uint8))
    if byte_equal:return dict(equal=True)
    result=dict(equal=False,kind='tensor',shape=list(a.shape),dtype=str(a.dtype),different_elements=int((x!=y).sum()),
        different_bytes=int((x.view(torch.uint8)!=y.view(torch.uint8)).sum()))
    if x.is_floating_point():
        squares=reference=0.;maximum=0.;nonfinite=0
        for lo in range(0,len(x),1048576):
            u=x[lo:lo+1048576].double();v=y[lo:lo+1048576].double();delta=u-v
            finite=torch.isfinite(delta);nonfinite+=int((~finite).sum())
            if bool(finite.any()):
                d=delta[finite];maximum=max(maximum,float(d.abs().max()));squares+=float((d*d).sum());reference+=float((u[finite]**2).sum())
        result.update(max_absolute=maximum,rmse=(squares/max(len(x)-nonfinite,1))**.5,
            relative_l2=(squares/max(reference,1e-300))**.5,nonfinite_differences=nonfinite)
    return result


def raw_difference(a,b,path=''):
    import torch
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
        result=tensor_difference(a,b)
        return [] if result['equal'] else [dict(path=path,**result)]
    if type(a)!=type(b):return [dict(path=path,kind='type',equal=False)]
    if isinstance(a,dict):
        result=[]
        for key in dict.fromkeys([*a,*b]):
            if key not in a or key not in b:result.append(dict(path=f'{path}/{key}',kind='missing',equal=False))
            # Allocation offsets/groups describe source layout, not numerical content.
            elif key not in ('alignment','storage_offset','storage_group'):
                result.extend(raw_difference(a[key],b[key],f'{path}/{key}'))
        return result
    if isinstance(a,list):
        result=[] if len(a)==len(b) else [dict(path=path,kind='coverage',equal=False)]
        for i,(x,y) in enumerate(zip(a,b)):result.extend(raw_difference(x,y,f'{path}/{i}'))
        return result
    return [] if a==b else [dict(path=path,kind='value',equal=False)]


def logical_fa4_inputs(row):
    """Keep selected-list order; exclude inactive list slots and page padding."""
    import torch
    result=dict(q=row['q']['value'],scalar_kwargs=row['scalar_kwargs'],
        tensor_kwargs={k:v['value'] for k,v in row['tensor_kwargs'].items() if k not in ('page_table','seqused_k')},
        aux_inputs=[None if a['input'] is None else a['input']['value'] for a in row['aux']])
    used=row['tensor_kwargs'].get('seqused_k')
    if row['layout']=='paged':
        page=row['k_metadata']['shape'][1]
        width=row['page_ids']['value'].numel()//row['q']['shape'][0]
        result['k']=[];result['v']=[]
        for batch in range(row['q']['shape'][0]):
            length=int(used['value'][batch]);result.setdefault('used_lengths',[]).append(length)
            for key in ('k','v'):
                pages=row[key+'_pages']['value'][batch*width:(batch+1)*width]
                result[key].append(pages.reshape(-1,*pages.shape[2:])[:length])
    else:
        result.update(k=row['k']['value'],v=row['v']['value'])
        if used is not None:result['used_lengths']=used['value']
    bst=row.get('block_sparse_tensors')
    if bst is not None:
        selected=dict(block_size=bst['block_size'])
        for kind in ('full','mask'):
            counts=bst[kind+'_block_cnt']['value'];indices=bst[kind+'_block_idx']['value']
            rows=[]
            for b,h,q in __import__('itertools').product(*(range(n) for n in counts.shape)):
                rows.append(indices[b,h,q,:int(counts[b,h,q])])
            selected[kind]=rows
        result['active_lists']=selected
    return result


def compare_raw(left,right):
    import torch
    a=torch.load(left,map_location='cpu',weights_only=True);b=torch.load(right,map_location='cpu',weights_only=True)
    # Match by native stage/call and occurrence. Do not align by output similarity.
    def keyed(rows):
        out={};counts={}
        for row in rows:
            key=(row['kind'],row.get('sampler_index'));i=counts.get(key,0);counts[key]=i+1
            out[(*key,i)]=row
        return out
    aa,bb=keyed(a),keyed(b);rows=[]
    for key in dict.fromkeys([*aa,*bb]):
        if key not in aa or key not in bb:rows.append(dict(stage=list(key),missing=True));continue
        changes=raw_difference(aa[key],bb[key]);entry=dict(stage=list(key),equal=not changes,differences=changes)
        if key[0]=='fa4':
            semantic=raw_difference(logical_fa4_inputs(aa[key]),logical_fa4_inputs(bb[key]))
            entry['logical_inputs']=dict(equal=not semantic,differences=semantic,
                scope='captured Q, logical used KV, ordered active sparse lists, named scalar and tensor arguments; excludes unused storage')
        rows.append(entry)
    return dict(left_sha256=sha(left),right_sha256=sha(right),stages=rows,
        allocation_metadata_excluded=['alignment','storage_offset','storage_group'],
        exact_operator_input_equality_not_assumed=True)


def analyze(root,destination):
    assert not destination.exists();destination.mkdir(parents=True)
    p=read(root/'protocol.json');assert p['last_global_probe']
    for name,digest in p['files_sha256'].items():assert sha(root/name)==digest
    engines={};bindings={};captured={};native={}
    for part,config in p['jobs'].items():
        leaf=root/'runs'/part/'attempt001'
        marker=next((leaf/name for name in ('attempt_complete.json','failed_attempt.json') if (leaf/name).exists()),None)
        if marker is None:engines[part]=dict(status='not_closed',analyzed=False);continue
        receipt=read(marker);bindings[str(marker.relative_to(root))]=sha(marker)
        for name,digest in receipt['files_sha256'].items():assert sha(leaf/name)==digest
        records=[json.loads(line) for line in (leaf/'records.jsonl').read_text().splitlines()] if (leaf/'records.jsonl').exists() else []
        groups=[]
        for group in config['bookend_groups']:
            rows={key:group[key] for key in ('original','observed','final')}
            entry=dict(arm=group['arm'],label=group['label'],rows=rows)
            for context,index in rows.items():
                if index<len(records):native[part,group['label'],context]=records[index]
            entry['native_contrasts']={}
            for label,a,b in (('full_vs_layer','original','observed'),('full_vs_pure','original','final'),('layer_vs_pure','observed','final')):
                i,j=rows[a],rows[b]
                entry['native_contrasts'][label]=dict(missing=True) if max(i,j)>=len(records) else dict(
                    equal=all(records[i][k]==records[j][k] for k in FIELDS),
                    differences={k:dict(left=records[i][k],right=records[j][k]) for k in FIELDS if records[i][k]!=records[j][k]})
            paths={key:leaf/f'private/last_global/request{rows[key]+1:03d}.json' for key in ('original','observed')}
            if all(x.exists() for x in paths.values()):
                traces={key:read(path) for key,path in paths.items()}
                for context,path in paths.items():captured[part,group['label'],context]=path
                entry['operator_contrast']=event_contrast(traces['original']['events'],traces['observed']['events'])
                raw={key:path.with_name(path.stem+'.private.pt') for key,path in paths.items()}
                if all(x.exists() for x in raw.values()):entry['raw_contrast']=compare_raw(raw['original'],raw['observed'])
            else:entry['operator_contrast']=dict(missing=[key for key,path in paths.items() if not path.exists()])
            groups.append(entry)
        engines[part]=dict(status=receipt['status'],records=len(records),expected=config['requests'],groups=groups)
    independent=[]
    if len(p['jobs'])==2:
        left,right=p['jobs']
        for group in p['jobs'][left]['bookend_groups']:
            for context in ('original','observed','final'):
                a,b=(left,group['label'],context),(right,group['label'],context)
                entry=dict(label=group['label'],context=context)
                if a in native and b in native:
                    entry['native']=dict(equal=all(native[a][k]==native[b][k] for k in FIELDS),
                        differences={k:dict(left=native[a][k],right=native[b][k]) for k in FIELDS if native[a][k]!=native[b][k]})
                else:entry['native']=dict(missing=True)
                if a in captured and b in captured:
                    entry['operator']=event_contrast(read(captured[a])['events'],read(captured[b])['events'])
                    raw_a=captured[a].with_name(captured[a].stem+'.private.pt')
                    raw_b=captured[b].with_name(captured[b].stem+'.private.pt')
                    if raw_a.exists() and raw_b.exists():entry['raw']=compare_raw(raw_a,raw_b)
                independent.append(entry)
    result=dict(status='diagnostic_only_unresolved',protocol_sha256=sha(root/'protocol.json'),source_commit=p['source_commit'],
        leaf_markers_sha256=bindings,engines=engines,independent_engines=independent,full_accuracy_released=False,clean_timing=False)
    write(destination/'summary.json',result)
    write(destination/'analysis_complete.json',dict(status='diagnostic_only_unresolved',files_sha256={'summary.json':sha(destination/'summary.json')}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('study',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args();result=analyze(args.study,args.destination)
    for name,engine in result['engines'].items():
        print(name,engine['status'],engine.get('records'))
        for group in engine.get('groups',[]):print(group['label'],group['native_contrasts'],group['operator_contrast'].get('first_difference'))
