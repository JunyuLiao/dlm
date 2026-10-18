"""Private native observation capture helpers and bounded same-state replay."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys


def tensor_meta(t):
    return dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                storage_offset=t.storage_offset(), pointer_mod256=t.data_ptr() % 256,
                device=str(t.device), requires_grad=t.requires_grad)


def digest(t):
    import torch
    return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def restore(values, meta, device='cuda'):
    import torch
    size = meta['storage_offset'] + 1 + sum((n-1)*s for n,s in zip(meta['shape'],meta['stride']))
    storage = torch.empty(size, dtype=values.dtype, device=device)
    t = storage.as_strided(meta['shape'], meta['stride'], meta['storage_offset'])
    t.copy_(values)
    if device != 'cpu' and t.data_ptr() % 256 != meta['pointer_mod256']:
        raise RuntimeError('cannot preserve captured pointer alignment')
    return t


def stable(value):
    if value is None or isinstance(value,(str,int,float,bool)): return value
    if isinstance(value,(tuple,list)): return [stable(v) for v in value]
    if isinstance(value,dict): return {str(k):stable(v) for k,v in value.items()}
    if hasattr(value,'__dataclass_fields__'):
        return {k:stable(getattr(value,k)) for k in value.__dataclass_fields__}
    return dict(type=type(value).__module__+'.'+type(value).__qualname__, text=str(value).split(' at 0x')[0])


class NativeProbe:
    """Capture return-frame locals without changing native arguments/returns."""
    def __init__(self): self.frames={}
    def callback(self,frame,event,arg):
        name=frame.f_code.co_name
        if event=='return' and name in ('_flash_attn_fwd','_flash_attn_fwd_combine') and 'vllm_flash_attn/cute/interface.py' in frame.f_code.co_filename:
            self.frames[name]=dict(frame.f_locals)
    def __enter__(self):
        self.previous=sys.getprofile()
        if self.previous is not None: raise RuntimeError('another Python profiler is active')
        sys.setprofile(self.callback);return self
    def __exit__(self,*args): sys.setprofile(self.previous)
    def extract(self):
        import torch
        if '_flash_attn_fwd' not in self.frames: raise RuntimeError('native forward return not observed')
        configs={}; tensors={}
        for name,loc in self.frames.items():
            names=('fwd_cfg','num_splits','tile_m','tile_n','pack_gqa','causal','softmax_scale','arch','is_split_kv','num_threads','mma_pv_is_rs','intra_wg_overlap','compile_key')
            config={k:stable(loc[k]) for k in names if k in loc}
            fn=loc.get(name) or self._function(name)
            key=loc.get('compile_key');compiled=fn.compile_cache[key] if fn is not None and key is not None else None
            config['compiled_object']=None if compiled is None else dict(type=type(compiled).__module__+'.'+type(compiled).__qualname__,process_object_id=id(compiled))
            configs[name]=config
            if name=='_flash_attn_fwd':
                for k in ('q','k','v','q_call','k_call','v_call','dynamic_causal','out','out_partial','lse_partial','lse'):
                    t=loc.get(k)
                    if isinstance(t,torch.Tensor): tensors[k]=t
                for i,t in enumerate(loc.get('aux_tensors') or []):tensors['aux'+str(i)]=t
        return configs,tensors
    @staticmethod
    def _function(name):
        from vllm.vllm_flash_attn.cute import interface
        return getattr(interface,name)


def compare(a,b,mask=None):
    import torch
    if a.shape!=b.shape:return dict(shape_mismatch=True,exact=False)
    if mask is not None:a,b=a[mask],b[mask]
    af,bf=a.float(),b.float();finite=torch.isfinite(af)&torch.isfinite(bf)
    equal=(a==b)|(torch.isnan(af)&torch.isnan(bf))
    delta=(af[finite]-bf[finite]).double()
    norm=float(bf[finite].double().square().sum().sqrt())
    return dict(exact=digest(a)==digest(b),unequal_fraction=float((~equal).float().mean()) if a.numel() else 0.,
        max_abs=float(delta.abs().max()) if delta.numel() else 0.,
        rms=float(delta.square().mean().sqrt()) if delta.numel() else 0.,
        relative_l2=float(delta.square().sum().sqrt())/max(norm,1e-30),
        nonfinite_pattern_equal=bool(torch.equal(torch.isfinite(af),torch.isfinite(bf))),elements=a.numel())


def outputs(tensors):
    import torch
    result={k:v.detach().cpu().clone() for k,v in tensors.items() if k in ('out','out_partial','lse_partial','lse','aux0')}
    # Native forward lse_partial is [splits,batch,heads,q]. Inactive splits have
    # -inf LSE and their output storage is not semantically consumed by combine.
    if 'out_partial' in result:
        mask=torch.isfinite(result['lse_partial']).transpose(-1,-2).unsqueeze(-1).expand_as(result['out_partial'])
        result['out_partial_active']=result['out_partial'][mask]
        result['partial_active_mask']=mask
        del result['out_partial']
    return result


def compare_outputs(a,b):
    return {k:compare(a[k],b[k]) for k in a.keys()&b.keys()}


def run(capture,output):
    import torch
    from experiments.numerical_qk_reuse import v31_fa4_observe as observe
    output.mkdir(parents=True,exist_ok=False)
    snapshots=sorted(capture.glob('snapshot*.pt'))
    if len(snapshots)!=6:raise RuntimeError('six exposed native snapshots required')
    reports=[]
    for path in snapshots:
        saved=torch.load(path,map_location='cpu',weights_only=False)
        q,k,v=[restore(saved['values'][key],saved['metadata'][key]) for key in ('q','k','v')]
        z=torch.empty(saved['metadata']['aux0']['shape'],device='cuda',dtype=torch.float32)
        cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
        input_hashes={key:digest(t) for key,t in zip(('q','k','v'),(q,k,v))}
        assert input_hashes=={key:saved['hashes'][key] for key in input_hashes}
        for _ in range(2):observe.observe_dense(q,k,v,saved['scale'],z)
        def one(splits):
            with NativeProbe() as probe:observe.observe_dense(q,k,v,saved['scale'],z,num_splits=splits)
            cfg,ts=probe.extract();data=outputs(ts)
            return cfg,data,{key:tensor_meta(t) for key,t in ts.items()}
        rows=[];reference=None;changed=False;ref_cfg=None
        for i in range(16):
            if i>=8:q,k,v=[restore(saved['values'][key],saved['metadata'][key]) for key in ('q','k','v')]
            cfg,data,meta=one(0)
            if reference is None:reference=data;ref_cfg=cfg
            diffs=compare_outputs(data,reference)
            captured=compare_outputs(data,saved['outputs'])
            changed|=any(not r['exact'] for r in diffs.values())
            rows.append(dict(repetition=i,allocation='fixed' if i<8 else 'fresh',config=cfg,metadata=meta,
                             vs_first=diffs,vs_capture=captured,hashes={k:digest(v) for k,v in data.items() if k!='partial_active_mask'}))
        fixed=[]
        if changed:
            for _ in range(2):observe.observe_dense(q,k,v,saved['scale'],z,num_splits=1)
            first=None
            for i in range(8):
                cfg,data,meta=one(1)
                if first is None:first=data
                fixed.append(dict(repetition=i,config=cfg,vs_first=compare_outputs(data,first),vs_auto_output=compare(data['out'],reference['out'])))
        end_hashes={key:digest(t) for key,t in zip(('q','k','v'),(q,k,v))}
        report=dict(snapshot=path.name,capture_config=saved['config'],capture_metadata=saved['metadata'],
                    input_hashes_match=input_hashes==end_hashes,
                    rng_unchanged=torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state()),
                    auto=rows,conditional_split1=fixed)
        reports.append(report)
        (output/(path.stem+'.json')).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(snapshot=path.name,auto_varies=changed,rows=len(rows),fixed=len(fixed))),flush=True)
        del saved,q,k,v,z,reference
    result=dict(status='complete_operator_diagnostic',snapshots=len(reports),reports=reports)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.capture,args.output)
