"""TensorRT-LLM functional adapter for the value-direction FMHA V3 plugin.

The caller supplies post-normalization/RoPE native-head QKV and already refreshed
FP32 sketches/reference scale. This does not convert DiffusionGemma into an AR
TensorRT-LLM model or substitute paged-XQA semantics for its diffusion canvas.
"""
import ctypes
from pathlib import Path

_libraries = {}


def load_plugin(path):
    import tensorrt as trt
    key=str(Path(path).resolve())
    if key not in _libraries:
        lib=ctypes.CDLL(key,mode=ctypes.RTLD_GLOBAL)
        lib.init_value_direction_plugin.restype=ctypes.c_bool
        if not lib.init_value_direction_plugin():raise RuntimeError('Plugin registration failed')
        _libraries[key]=lib
    creator=trt.get_plugin_registry().get_creator('ValueDirectionHopper','2','dllm')
    if creator is None:raise RuntimeError('ValueDirectionHopper V3 creator missing')
    return creator


def add_layer(network,inputs,library,*,scale=1.,log_threshold=-float('inf'),mode=1,precision=0,overlap=True,tma=False):
    """Raw TensorRT adapter; six B,H,N,D/FP32-reference/bool-mask inputs."""
    import numpy as np
    import tensorrt as trt
    creator=load_plugin(library)
    values={name:np.array([value],dtype=dtype) for name,value,dtype in (
        ('scale',scale,np.float32),('log_threshold',log_threshold,np.float32),
        ('mode',mode,np.int32),('precision',precision,np.int32),('overlap',int(overlap),np.int32),('tma',int(tma),np.int32))}
    fields=trt.PluginFieldCollection([trt.PluginField(name,value,
        trt.PluginFieldType.FLOAT32 if value.dtype==np.float32 else trt.PluginFieldType.INT32) for name,value in values.items()])
    plugin=creator.create_plugin('value_direction_hopper',fields,trt.TensorRTPhase.BUILD)
    if plugin is None:raise ValueError('Invalid attention plugin configuration')
    layer=network.add_plugin_v3(inputs,[],plugin)
    if layer is None:raise RuntimeError('TensorRT rejected the attention plugin')
    return layer


def attention(q,k,v,z,reference,mask,*,library,**options):
    """Insert a typed layer in the active TensorRT-LLM network.

    Outputs: BF16 attention, physical skip/eligibility masks, retained LSE,
    projected retained state, and tile log-risk. No CPU/GPU synchronization.
    """
    from tensorrt_llm._common import default_trtnet
    from tensorrt_llm.functional import _create_tensor
    layer=add_layer(default_trtnet(),[x.trt_tensor for x in (q,k,v,z,reference,mask)],library,**options)
    return tuple(_create_tensor(layer.get_output(i),layer) for i in range(6))
