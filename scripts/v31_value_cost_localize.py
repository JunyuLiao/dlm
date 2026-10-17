"""Bounded saved-state cost localization; no native generation or promotion."""
import argparse
import ast
import inspect
import json
from pathlib import Path
import subprocess
import time
import torch
from scripts import v31_value_decision_replay as io
from scripts.v31_value_e2e_profile import measure
from experiments.numerical_qk_reuse.v31_value_fast import bounded_proposal_device, dense_output, selected_output, selected_ids
from experiments.numerical_qk_reuse.v31_value_fast_runtime import moments_full
from experiments.numerical_qk_reuse.v31_value_decision import bounded_proposal


def scoped(fn, boundaries):
    """Add profiler-only ranges around unchanged top-level statement groups."""
    fn=inspect.unwrap(fn)
    tree=ast.parse(inspect.getsource(fn));node=tree.body[0]
    groups=[];body=[];name='setup'
    for statement in node.body:
        text=ast.unparse(statement)
        new=next((label for prefix,label in sorted(boundaries,key=lambda x:-len(x[0])) if text.startswith(prefix)),None)
        if new and body:
            groups.append((name,body));body=[]
        if new:name=new
        body.append(statement)
    groups.append((name,body))
    node.body=[ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name(id='range_scope',ctx=ast.Load()),args=[ast.Constant('stage:'+n)],keywords=[]))],body=b) for n,b in groups]
    ast.fix_missing_locations(tree)
    namespace=dict(fn.__globals__,range_scope=torch.profiler.record_function)
    exec(compile(tree,'<profile-only unchanged statements>','exec'),namespace)
    return namespace[fn.__name__]


def profile(fn, output):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn();torch.cuda.synchronize()
    prof.export_chrome_trace(str(output))
    stages={x.key:dict(cpu_inclusive_ms=x.cpu_time_total/1000,device_kernel_ms=x.device_time_total/1000,count=x.count)
            for x in prof.key_averages() if x.key.startswith('stage:')}
    events=[x for x in prof.events() if x.device_type==torch.autograd.DeviceType.CUDA]
    intervals=sorted((x.time_range.start,x.time_range.end) for x in events)
    union=0.;end=-1.
    for a,b in intervals:
        union+=max(0,b-max(a,end));end=max(end,b)
    return dict(stages=stages,kernel_events=len(events),kernel_sum_ms=sum(b-a for a,b in intervals)/1000,
        kernel_union_ms=union/1000,kernel_span_ms=(max(b for a,b in intervals)-min(a for a,b in intervals))/1000 if intervals else None,
        operations={x.key:dict(count=x.count,self_cpu_ms=x.self_cpu_time_total/1000,self_device_ms=x.self_device_time_total/1000)
                    for x in prof.key_averages() if x.key in ('aten::mm','aten::bmm','aten::index','aten::index_select','aten::sort','aten::copy_','aten::empty','aten::_local_scalar_dense','aten::nonzero')})


def graph_compatible_selector():
    original=inspect.unwrap(bounded_proposal_device)
    source=inspect.getsource(original)
    old='pool[torch.argsort(score, descending=True, stable=True)[:min(pt, budget+extra)]] = True'
    new='pool.index_fill_(0, torch.argsort(score, descending=True, stable=True)[:min(pt, budget+extra)], True)'
    if source.count(old)!=1:raise RuntimeError('unexpected selector source')
    namespace=dict(original.__globals__)
    exec(compile(source.replace(old,new),'<diagnostic index_fill graph prototype>','exec'),namespace)
    return namespace[original.__name__]


@torch.inference_mode()
def run(output):
    if output.exists():raise RuntimeError('preserve attempts')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU busy')
    start=time.monotonic();output.mkdir(parents=True);(output/'private').mkdir()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    paths=[]
    for request in ('request001','request002'):
        found=False
        for group,files in sorted(io.group_paths().items()):
            if group[0]!=request or group[2]!='layer29':continue
            for path in files:
                d=torch.load(path,weights_only=True,map_location='cpu')
                if d['selection_event'] and d['next'] is not None:
                    paths.append(path);found=True;break
            if found:break
    if len(paths)!=2:raise RuntimeError('missing predeclared states')
    from safetensors import safe_open
    name='model.decoder.layers.29.self_attn.o_proj.weight';shard=io.read(io.MODEL/'model.safetensors.index.json')['weight_map'][name]
    with safe_open(str(io.MODEL/shard),framework='pt',device='cpu') as f:w=f.get_tensor(name).float().cuda()
    grams=[w[:,h*512:(h+1)*512].T@w[:,h*512:(h+1)*512] for h in range(16)]
    del w
    moment_profile=scoped(moments_full,[('scores =','qk_normalize'),('pt =','mass_reduce'),('num =','block_pv'),('tail =','tail')])
    selection_profile=scoped(bounded_proposal_device,[('ids =','candidate_construction'),('num, den =','anchor_and_dense_reductions'),('losses =','pair_objective'),('losses = torch.where','pair_validity'),('limit =','disjoint_rank'),('chosen =','combined_admission')])
    graph_selector=graph_compatible_selector()
    io.write(output/'config.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=io.sha(__file__),inputs={str(p.relative_to(io.SOURCE)):io.sha(p) for p in paths},
        tf32=False,selection='first actual observation request001/002; all heads and query tiles for graph parity',
        scope='isolated profile; graph is a fixed-shape diagnostic only, no runtime promotion'),True)
    results=[]
    for path in paths:
        d=torch.load(path,weights_only=True,map_location='cpu');prefix=torch.load(path.parent/d['prefix_file'],weights_only=True,map_location='cuda')
        k=torch.cat((prefix['k'],d['tail_k'].cuda()),2)[0].float();v=torch.cat((prefix['v'],d['tail_v'].cuda()),2)[0].float()
        q=d['q'][0,0,:128].cuda();anchor=d['next'][0,0,0,:d['prefix']//64].cuda().bool();budget=int(anchor.sum())
        m=moments_full(q,k[0],v[0],d['prefix'],d['scale']);gram=grams[0].clone()
        call=lambda:bounded_proposal_device(m,anchor,budget,gram=gram)
        row=dict(request=path.parent.name,prefix=d['prefix'],numerator_bytes=m.numerator.numel()*m.numerator.element_size())
        row['moments']=measure(lambda:moments_full(q,k[0],v[0],d['prefix'],d['scale']),7)
        row['selection']=measure(call,7)
        row['reference_selection']=measure(lambda:bounded_proposal(m,anchor,swaps=8,gram=gram),7)
        row['dense_selected_reductions']=measure(lambda:(dense_output(m),selected_output(m,selected_ids(anchor,budget))),7)
        mm=moment_profile(q,k[0],v[0],d['prefix'],d['scale'])
        if any(not torch.equal(getattr(mm,n),getattr(m,n)) for n in ('mass','numerator','tail_mass','tail_numerator','log_scale')):raise RuntimeError('profile wrapper changed moments')
        del mm
        if not torch.equal(selection_profile(m,anchor,budget,gram=gram)[0],call()[0]):raise RuntimeError('profile wrapper changed selection')
        row['moment_profile']=profile(lambda:moment_profile(q,k[0],v[0],d['prefix'],d['scale']),output/f'private/{path.parent.name}_moments.json')
        row['selection_profile']=profile(lambda:selection_profile(m,anchor,budget,gram=gram),output/f'private/{path.parent.name}_selection.json')
        io.write(output/f'{path.parent.name}_profile.json',row,True)
        # One bounded engineering prototype: capture the exact unchanged selector.
        # The moments/mask/Gram are fixed-address inputs updated between replays.
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):graph_selector(m,anchor,budget,gram=gram)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=graph_selector(m,anchor,budget,gram=gram)
        graph.replay();torch.cuda.synchronize()
        row['graph_selection']=measure(lambda:(graph.replay(),captured),20)
        row['graph_parity']=dict(units=0,mask_mismatches=0,swap_mismatches=0,pair_loss_mismatches=0,budget_errors=0)
        for h in range(16):
            for qt in range((d['n']+127)//128):
                q2=d['q'][0,h,qt*128:min((qt+1)*128,d['n'])].cuda()
                fresh=moments_full(q2,k[h//8],v[h//8],d['prefix'],d['scale'])
                for name in ('mass','numerator','tail_mass','tail_numerator','log_scale'):getattr(m,name).copy_(getattr(fresh,name))
                anchor.copy_(d['next'][0,h,qt,:d['prefix']//64]);gram.copy_(grams[h]);del fresh
                expected=call();graph.replay();torch.cuda.synchronize()
                p=row['graph_parity'];p['units']+=1
                p['mask_mismatches']+=not torch.equal(expected[0],captured[0])
                p['swap_mismatches']+=not torch.equal(expected[1]['swaps'],captured[1]['swaps'])
                p['pair_loss_mismatches']+=not torch.equal(expected[1]['pair_losses'],captured[1]['pair_losses'])
                p['budget_errors']+=int(captured[0].sum())!=budget
        io.write(output/f'{path.parent.name}.json',row,True);results.append(row)
        print(json.dumps(dict(request=row['request'],selection_ms=row['selection']['host_ms'],graph_ms=row['graph_selection']['host_ms'],parity=row['graph_parity'])),flush=True)
        del m,k,v,prefix,graph,captured,expected
        if time.monotonic()-start>600:raise RuntimeError('bounded profile reservation exceeded')
    io.write(output/'summary.json',dict(status='complete_diagnostic',results=results,gpu_reservation_s=time.monotonic()-start,
        limitations='Profiler ranges add overhead. Kernel sum excludes launch gaps. Graph timing excludes input copies and capture. Native fixed-shape graph workspace and lifecycle integration are unimplemented.'),True)
    io.write(output/'final_complete.json',dict(status='complete',files_sha256={p.name:io.sha(p) for p in output.glob('*.json')}),True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output.resolve())
