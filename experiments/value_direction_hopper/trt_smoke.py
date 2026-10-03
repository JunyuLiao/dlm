"""Real serialized-engine parity test, intended for the pinned TRT-LLM image."""
import argparse
import json
from pathlib import Path

import torch
import tensorrt as trt
from .tensorrt import attention,add_layer
from .cuda import Kernel
from .validate import inputs
from .masks import pack


def run(plugin,kernel,output,precision=0,tma=False):
    if output.exists():raise FileExistsError(output)
    import tensorrt_llm
    from tensorrt_llm import Builder, net_guard
    from tensorrt_llm.functional import Tensor
    records=[];cuda=Kernel(kernel,allow_legacy=True)
    for width in (256,512):
        q,k,v,z,ref,mask=inputs(129,193,width,4,2,local=True)
        tensors=(q,k,v,z,ref,mask)
        builder=Builder();network=builder.create_network()
        with net_guard(network):
            ins=[Tensor(name=f'in{i}',dtype={torch.bfloat16:trt.bfloat16,torch.float32:trt.float32,torch.bool:trt.bool}[x.dtype],shape=tuple(x.shape)) for i,x in enumerate(tensors)]
            outs=attention(*ins,library=plugin,scale=width**-.5,log_threshold=-2.,precision=precision,tma=tma)
            for i,x in enumerate(outs):x.mark_output(f'out{i}',{0:trt.bfloat16,1:trt.bool,2:trt.bool}.get(i,trt.float32))
        config=builder.create_builder_config(precision='bfloat16')
        engine_bytes=builder.build_engine(network,config)
        if engine_bytes is None:raise RuntimeError('Engine build failed')
        runtime=trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine=runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:raise RuntimeError('Serialized plugin engine failed to reload')
        context=engine.create_execution_context()
        for i,x in enumerate(tensors):context.set_tensor_address(f'in{i}',x.data_ptr())
        result=[]
        for i in range(6):
            x=torch.empty(tuple(context.get_tensor_shape(f'out{i}')),device='cuda',dtype={0:torch.bfloat16,1:torch.bool,2:torch.bool}.get(i,torch.float32))
            result.append(x);context.set_tensor_address(f'out{i}',x.data_ptr())
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            if not context.execute_async_v3(stream.cuda_stream):raise RuntimeError('TensorRT enqueue failed')
            expected=cuda(q,k,v,z,ref,mask=cuda.pack_mask(mask) if precision==2 else mask,
                scale=width**-.5,log_threshold=-2.,trace=True,precision={0:'ieee',1:'tf32x3',2:'tf32x3_register'}[precision],tma=tma)
        torch.cuda.synchronize()
        same=[]
        for actual,field in zip(result,('output','skipped','eligible','log_normalizer','projected_state','risk')):
            reference=getattr(expected,field)
            torch.testing.assert_close(actual,reference,rtol=0,atol=0,equal_nan=True)
            same.append(field)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            if not context.execute_async_v3(stream.cuda_stream):raise RuntimeError('TensorRT graph enqueue failed')
        graph.replay();torch.cuda.synchronize()
        for actual,field in zip(result,('output','skipped','eligible','log_normalizer','projected_state','risk')):
            torch.testing.assert_close(actual,getattr(expected,field),rtol=0,atol=0,equal_nan=True)
        records.append(dict(width=width,precision=precision,tma=tma,exact_parity=same,graph_parity=True,engine_bytes=len(bytes(engine_bytes))))
    output.write_text(json.dumps(dict(tensorrt=trt.__version__,tensorrt_llm=tensorrt_llm.__version__,cases=records,passed=True),indent=2)+'\n')


def dynamic(plugin,kernel,output):
    """Exercise real dynamic profiles, workspace sizing and batch/tail shapes."""
    if output.exists():raise FileExistsError(output)
    logger=trt.Logger(trt.Logger.WARNING);cuda=Kernel(kernel);records=[]
    for width in (256,512):
        builder=trt.Builder(logger);network=builder.create_network();config=builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.BF16);profile=builder.create_optimization_profile()
        shapes=[(-1,4,-1,width),(-1,2,-1,width),(-1,2,-1,width),(-1,2,-1,32),(-1,2),(-1,1,-1,-1)]
        minimum=[(1,4,1,width),(1,2,1,width),(1,2,1,width),(1,2,1,32),(1,2),(1,1,1,1)]
        optimum=[(1,4,128,width),(1,2,1024,width),(1,2,1024,width),(1,2,1024,32),(1,2),(1,1,128,1024)]
        maximum=[(2,4,256,width),(2,2,8192,width),(2,2,8192,width),(2,2,8192,32),(2,2),(2,1,256,8192)]
        network_inputs=[]
        for i,shape in enumerate(shapes):
            network_inputs.append(network.add_input(f'in{i}',trt.bfloat16 if i<3 else trt.bool if i==5 else trt.float32,shape))
            if profile.set_shape(f'in{i}',minimum[i],optimum[i],maximum[i]) is False:raise RuntimeError(f'Invalid optimization profile for in{i}')
        config.add_optimization_profile(profile)
        layer=add_layer(network,network_inputs,plugin,scale=width**-.5,log_threshold=-2.,precision=2,tma=True)
        for i in range(6):
            tensor=layer.get_output(i);tensor.name=f'out{i}'
            tensor.dtype={0:trt.bfloat16,1:trt.bool,2:trt.bool}.get(i,trt.float32)
            network.mark_output(tensor)
        serialized=builder.build_serialized_network(network,config)
        if serialized is None:raise RuntimeError('Dynamic engine build failed')
        runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(serialized);context=engine.create_execution_context()
        for batch,nq,nk in ((1,63,65),(2,131,259)):
            tensors=list(inputs(nq,nk,width,4,2,local=True))
            if batch==2:tensors=[torch.cat((x,x),0).contiguous() for x in tensors]
            for i,x in enumerate(tensors):
                if not context.set_input_shape(f'in{i}',tuple(x.shape)):raise RuntimeError('Shape rejected')
                context.set_tensor_address(f'in{i}',x.data_ptr())
            outputs=[]
            for i in range(6):
                dtype=engine.get_tensor_dtype(f'out{i}')
                wanted={0:trt.bfloat16,1:trt.bool,2:trt.bool}.get(i,trt.float32)
                if dtype!=wanted:raise RuntimeError(f'Unexpected engine output dtype {i}: {dtype}, expected {wanted}')
                x=torch.empty(tuple(context.get_tensor_shape(f'out{i}')),device='cuda',dtype={0:torch.bfloat16,1:torch.bool,2:torch.bool}.get(i,torch.float32))
                outputs.append(x);context.set_tensor_address(f'out{i}',x.data_ptr())
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                if not context.execute_async_v3(stream.cuda_stream):raise RuntimeError('Dynamic enqueue failed')
                q,k,v,z,r,m=tensors
                expected=cuda(q,k,v,z,r,mask=cuda.pack_mask(m),scale=width**-.5,log_threshold=-2.,trace=True,precision='tf32x3_register',tma=True)
            torch.cuda.synchronize()
            for actual,field in zip(outputs,('output','skipped','eligible','log_normalizer','projected_state','risk')):
                torch.testing.assert_close(actual,getattr(expected,field),rtol=0,atol=0,equal_nan=True)
            records.append(dict(width=width,batch=batch,queries=nq,keys=nk,exact_all_outputs=True))
    output.write_text(json.dumps(dict(cases=records,passed=True,scope='TensorRT dynamic plugin profiles; static TRT-LLM functional/graph tests are separate'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plugin',type=Path,required=True);p.add_argument('--kernel',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--precision',type=int,choices=(0,1,2),default=0);p.add_argument('--tma',action='store_true')
    p.add_argument('--dynamic',action='store_true')
    a=p.parse_args()
    if a.dynamic:dynamic(a.plugin,a.kernel,a.output)
    else:run(a.plugin,a.kernel,a.output,a.precision,a.tma)
