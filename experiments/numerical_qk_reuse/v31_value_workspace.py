"""Bounded selector scratch, independent of the model's native KV cache.

Each view is contiguous at its current shape. All operations use one CUDA
stream, and no returned keep map aliases this scratch. Only sequential
initial/refresh selections may reuse it.
"""
import math

import torch


class ValueWorkspace:
    def __init__(self, max_tokens):
        if max_tokens <= 0:
            raise ValueError('positive frozen context capacity required')
        self.max_tiles = math.ceil(max_tokens/64)
        self.storage = {}
        self.stream = None

    def take(self, name, shape, dtype, device):
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda',torch.cuda.current_device())
        stream = (str(device),torch.cuda.current_stream(device).cuda_stream)
        if self.stream is not None and stream != self.stream:
            raise RuntimeError('selector scratch cannot cross CUDA streams')
        self.stream = stream
        if shape[1] > self.max_tiles:
            raise ValueError('selection exceeds frozen workspace capacity')
        capacity = shape[0]*self.max_tiles*math.prod(shape[2:])
        key = (name,dtype,str(device),shape[0],tuple(shape[2:]))
        if key not in self.storage:
            self.storage[key] = torch.empty(capacity,dtype=dtype,device=device)
        return self.storage[key][:math.prod(shape)].view(shape)

    @property
    def allocated_bytes(self):
        return sum(x.numel()*x.element_size() for x in self.storage.values())
