"""Restore private forensic operands with captured strides and K/V aliasing.

Unused paged-cache storage is explicitly reconstructed, never represented as
captured native content. Every replay must disclose this boundary.
"""
import torch


def dtype_of(metadata):
    name=metadata['dtype'].removeprefix('torch.')
    dtype=getattr(torch,name)
    if not isinstance(dtype,torch.dtype):raise ValueError('expected a torch dtype')
    return dtype


def extent(shape,strides):
    if len(shape)!=len(strides) or any(n<=0 for n in shape) or any(s<0 for s in strides):
        raise ValueError('nonempty shapes and nonnegative captured strides required')
    return 1+sum((n-1)*s for n,s in zip(shape,strides))


def allocate_group(metadata,device,fill=0):
    """Preserve relative offsets, strides, aliases and pointer alignment modulo256."""
    if not metadata:raise ValueError('nonempty storage group required')
    dtype=dtype_of(metadata[0]);size=torch.empty((),dtype=dtype).element_size()
    if any(dtype_of(m)!=dtype for m in metadata):raise ValueError('mixed dtype storage group unsupported')
    origin=min(m.get('storage_offset',0) for m in metadata)
    offsets=[m.get('storage_offset',0)-origin for m in metadata]
    spans=[extent(m['shape'],m['stride']) for m in metadata]
    alignment=256//size
    storage=torch.empty(max(o+s for o,s in zip(offsets,spans))+alignment,device=device,dtype=dtype)
    if fill is not None:storage.fill_(fill)
    first=metadata[0].get('alignment',0)
    if (first-storage.data_ptr())%size:raise ValueError('unaligned captured pointer')
    pad=((first-storage.data_ptr())//size-offsets[0])%alignment
    views=[]
    for m,offset in zip(metadata,offsets):
        view=storage.as_strided(m['shape'],m['stride'],pad+offset)
        if view.data_ptr()%256!=m.get('alignment',0):raise ValueError('inconsistent shared-storage alignment')
        views.append(view)
    return views


def copy_logical(target,source):
    if target.shape!=source.shape or target.dtype!=source.dtype:raise ValueError('captured value geometry mismatch')
    # Native alias batches have stride0. Copy their one physical slice once.
    shape=list(target.shape);value=source
    for axis,stride in enumerate(target.stride()):
        if stride==0 and shape[axis]>1:
            first=value.narrow(axis,0,1)
            if not torch.equal(value.contiguous().view(torch.uint8),first.expand_as(value).contiguous().view(torch.uint8)):
                raise ValueError('captured stride0 slices disagree')
            value=first;shape[axis]=1
    target.as_strided(shape,target.stride(),target.storage_offset()).copy_(value.to(device=target.device))
    return target


def tensor(raw,device='cuda'):
    target=allocate_group([raw],device)[0]
    return copy_logical(target,raw['value'])


def paged(raw,device='cuda',unused_fill=0):
    if raw['layout']!='paged':raise ValueError('paged capture required')
    metadata=[raw['k_metadata'],raw['v_metadata']]
    groups={};views={}
    for key,m in zip(('k','v'),metadata):groups.setdefault(m['storage_group'],[]).append((key,m))
    for entries in groups.values():
        restored=allocate_group([m for _,m in entries],device,unused_fill)
        views.update((key,value) for (key,_),value in zip(entries,restored))
    ids=raw['page_ids']['value'].long().reshape(-1)
    if ids.numel()!=raw['k_pages']['value'].shape[0] or ids.numel()!=raw['v_pages']['value'].shape[0]:
        raise ValueError('referenced page coverage mismatch')
    first={}
    for i,page in enumerate(ids.tolist()):
        if page<0 or any(page>=m['shape'][0] for m in metadata):raise ValueError('captured physical page outside pool')
        if page in first:
            for key in ('k','v'):
                a=raw[key+'_pages']['value'][i].contiguous().view(torch.uint8)
                b=raw[key+'_pages']['value'][first[page]].contiguous().view(torch.uint8)
                if not torch.equal(a,b):raise ValueError('duplicated physical page payload differs')
        else:first[page]=i
    target_ids=torch.tensor(list(first),device=device,dtype=torch.long)
    positions=torch.tensor(list(first.values()),dtype=torch.long)
    for key in ('k','v'):
        payload=raw[key+'_pages']['value'].index_select(0,positions).to(device=device)
        views[key].index_copy_(0,target_ids,payload)
    return views['k'],views['v']
