#include "test_func_lib.h"

__global__ void Kernel(hash_t func) {
    int tid = threadIdx.x;
    int out = (*func)(tid);
    printf("tid %d -> out %d\n", tid, out);
}

void Caller::Launch(hash_t h_func) {
    Kernel<<<1, 10>>>(h_func);
    __OPEN3D_CUDA_CHECK(cudaDeviceSynchronize(), __FILE__, __LINE__);
    __OPEN3D_CUDA_CHECK(cudaGetLastError(), __FILE__, __LINE__);
}
