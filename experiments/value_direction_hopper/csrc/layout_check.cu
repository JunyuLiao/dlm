// SPDX-License-Identifier: Apache-2.0
// Host-only layout audit used before constructing TMA descriptors.
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cuda_bf16.h>
#include <cstdio>
int main() {
    using namespace cute;using BF=__nv_bfloat16;
    auto k=tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},Shape<_64,_512>{});
    auto v=tile_to_shape(GMMA::Layout_MN_SW128_Atom<BF>{},Shape<_256,_64>{});
    alignas(8192) BF storage[64*512];
    auto tk=make_tensor(make_smem_ptr(storage),k);
    auto tv=make_tensor(make_smem_ptr(storage),v);
    for(int row:{0,1,8,16,32,63})for(int col:{0,8,64,128,256}) {
        printf("K row=%d col=%d physical=%ld\n",row,col,&tk(row,col)-storage);
        if(col<256)printf("V key=%d channel=%d physical=%ld\n",row,col,&tv(col,row)-storage);
    }
}
