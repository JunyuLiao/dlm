"""Same-input, operator-only replay of the first differing fused reduction.

The extracted Triton source is byte-identical to the frozen generated kernel.
Only its observed R0_BLOCK choice changes. This does not run native generation.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import triton


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trace(leaf):
    marker = json.loads((leaf/'attempt_complete.json').read_text())
    rel = 'private/segment/request010.pt'
    if digest(leaf/rel) != marker['files_sha256'][rel]:
        raise RuntimeError('native snapshot identity drift')
    return torch.load(leaf/rel, map_location='cpu', weights_only=False)


def error(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise RuntimeError('incompatible replay output')
    delta = (a.double()-b.double()).abs()
    return dict(different=int((a != b).sum()), elements=a.numel(),
                max_abs=float(delta.max()), mean_abs=float(delta.mean()))


def replay(study, left_name, right_name, out):
    if out.exists():
        raise RuntimeError('new immutable operator replay required')
    out.mkdir(parents=True)
    private = out/'private'; private.mkdir()
    left, right = [study/'diagnosis'/s for s in (left_name,right_name)]
    a, b = load_trace(left), load_trace(right)
    indices = [i for i,e in enumerate(a) if 'triton_red_fused_add_mul_rms_norm_2.run(' in e['source']]
    if len(indices) != 1:
        raise RuntimeError('first differing reduction call not uniquely identified')
    index = indices[0]; entry = a[index]
    if entry['source_sha256'] != b[index]['source_sha256'] or entry['source'] != b[index]['source']:
        raise RuntimeError('native generated operator mismatch')
    wrapper = next((study/'private/cache/vllm').rglob(entry['file']))
    if digest(wrapper) != entry['source_sha256']:
        raise RuntimeError('generated native source identity drift')
    name = 'triton_red_fused_add_mul_rms_norm_2'
    assignments = [node for node in ast.walk(ast.parse(wrapper.read_text()))
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id == name for t in node.targets)]
    if len(assignments) != 1:
        raise RuntimeError('generated kernel source not unique')
    body = assignments[0].value.args[1].value
    path = private/'native_kernel.py'; path.write_text(body)
    spec = importlib.util.spec_from_file_location('humaneval_native_reduction_replay',path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = getattr(module,name)
    if list(kernel.mutated_arg_names) != ['in_out_ptr0']:
        raise RuntimeError('unexpected in-place operator signature')
    call = ast.parse(entry['source']).body[0].value
    prior = {}
    for previous in a[:index]:
        prior.update(previous['values'])
    # Generated memory planning reuses the o_proj output as the layer-output
    # in/out argument. Its pre-reduction value is saved after the first GEMM.
    prior['buf19'] = a[0]['values']['buf0']
    rows = entry['values']['buf19']['shape'][0]
    prompt = json.loads((left/'records.jsonl').read_text().splitlines()[8])['prompt_tokens']
    outputs = {}
    for block in (512,1024):
        config = triton.Config(dict(XBLOCK=1,R0_BLOCK=block),num_warps=8,num_stages=1)
        result = kernel._precompile_config(config)
        launcher = result.make_launcher()
        repetitions = []
        for repeat in range(2):
            arguments = []; tensors = {}
            for arg in call.args:
                if isinstance(arg,ast.Constant):
                    arguments.append(arg.value); continue
                key = arg.id
                if key == 's59':
                    arguments.append(rows); continue
                item = prior.get(key,entry['values'].get(key))
                if item is None:
                    raise RuntimeError(f'unsaved native operand {key}')
                source = item['value']
                value = torch.empty_strided(item['shape'],item['stride'],dtype=source.dtype,device='cuda')
                # Output-only values need no initialization; initializing them
                # to the saved value also detects a kernel that fails to write.
                value.copy_(source)
                tensors[key] = value; arguments.append(value)
            launcher(*arguments,stream=torch._C._cuda_getCurrentRawStream(0))
            torch.cuda.synchronize()
            repetitions.append({key:tensors[key].cpu() for key in ('buf19','buf17','buf21')})
        outputs[block] = repetitions
    torch.save(outputs,private/'replay_outputs.pt')
    summary = dict(kind='same_input_operator_only_counterfactual', native_forwards=0,
        generated_wrapper_sha256=digest(wrapper), kernel_text_sha256=digest(path),
        native_markers_sha256={p.name:digest(p/'attempt_complete.json') for p in (left,right)},
        prompt_tokens=prompt,padded_rows=rows,configs=[512,1024],num_warps=8,
        native_selected_configs={left.name:entry['launch_configs'],right.name:b[index]['launch_configs']},
        same_config_repeat_errors={}, between_config_errors={}, native_match_errors={})
    for block,reps in outputs.items():
        summary['same_config_repeat_errors'][block] = {
            key:error(reps[0][key][:prompt],reps[1][key][:prompt]) for key in reps[0]}
        summary['native_match_errors'][block] = {
            leaf.name:{key:error(reps[0][key][:prompt],trace[index]['values'][key]['value'][:prompt])
                       for key in reps[0]} for leaf,trace in ((left,a),(right,b))}
    summary['between_config_errors'] = {
        key:error(outputs[512][0][key][:prompt],outputs[1024][0][key][:prompt]) for key in outputs[512][0]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'operator_complete.json').write_text(json.dumps(dict(
        files_sha256={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}),indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('study',type=Path)
    p.add_argument('left');p.add_argument('right');p.add_argument('out',type=Path)
    args = p.parse_args();replay(args.study,args.left,args.right,args.out)
