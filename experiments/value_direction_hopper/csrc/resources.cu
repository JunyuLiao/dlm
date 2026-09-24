// SPDX-License-Identifier: Apache-2.0
// Read-only resource/occupancy queries, compiled from an archived kernel source.
// These are launch limits, NOT achieved occupancy or performance-counter values.
#ifndef VD_ARCHIVED_SOURCE
#error Define VD_ARCHIVED_SOURCE to the exact archived value_direction.cu
#endif
#include VD_ARCHIVED_SOURCE

struct ResourceInfo {
    int registers,local_bytes,static_shared_bytes,dynamic_shared_bytes;
    int threads,cluster_size,multiprocessors,sm_max_threads,active_clusters;
    int grid_clusters,blocks_per_sm_limit,binary_version,ptx_version;
};

template<class Function>
cudaError_t inspect(Function fn,int width,int cluster,int threads,ResourceInfo* out) {
    using namespace fmha::value_direction;
    int device;cudaDeviceProp prop{};cudaFuncAttributes attr{};
    auto status=cudaGetDevice(&device);if(status!=cudaSuccess)return status;
    status=cudaGetDeviceProperties(&prop,device);if(status!=cudaSuccess)return status;
    int shared=width==256?sizeof(Shared<256>):sizeof(Shared<512>);
    status=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,shared);if(status!=cudaSuccess)return status;
    status=cudaFuncGetAttributes(&attr,fn);if(status!=cudaSuccess)return status;
    cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(2*cluster,16,1);cfg.blockDim=dim3(threads);cfg.dynamicSmemBytes=shared;
    int clusters=0,blocks=0;
    status=cudaOccupancyMaxActiveClusters(&clusters,fn,&cfg);if(status!=cudaSuccess)return status;
    status=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,fn,threads,shared);if(status!=cudaSuccess)return status;
    *out={attr.numRegs,int(attr.localSizeBytes),int(attr.sharedSizeBytes),shared,threads,cluster,
          prop.multiProcessorCount,prop.maxThreadsPerMultiProcessor,clusters,32,blocks,attr.binaryVersion,attr.ptxVersion};
    return cudaSuccess;
}

extern "C" cudaError_t value_direction_resources(int width,int schedule,ResourceInfo* out) {
    using namespace fmha::value_direction;
    if(!out||(width!=256&&width!=512)||schedule<0||schedule>2)return cudaErrorInvalidValue;
    if(width==256) {
        if(schedule==2)return cudaErrorInvalidValue;
        return schedule?inspect(kernel<256,true,true,true>,256,2,256,out):inspect(kernel<256,true,true>,256,2,256,out);
    }
    if(schedule==2)return inspect(split_kernel<false>,512,4,256,out);
    return schedule?inspect(kernel<512,true,true,true>,512,2,384,out):inspect(kernel<512,true,true>,512,2,384,out);
}
