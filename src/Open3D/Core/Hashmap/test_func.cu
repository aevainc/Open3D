#include "test_func_lib.h"

// inline void __OPEN3D_CUDA_CHECK(cudaError_t err,
//                                 const char *file,
//                                 const int line) {
//     if (err != cudaSuccess) {
//         printf("%s:%d CUDA runtime error: %s", file, line,
//                cudaGetErrorString(err));
//     }
// }

/// In library to be compiled alone
// typedef int (*hash_t)(int);

// __global__ void Kernel(hash_t func) {
//     int tid = threadIdx.x;
//     int out = (*func)(tid);
//     printf("tid %d -> out %d\n", tid, out);
// }

// class Caller {
// public:
//     void Launch(hash_t h_func) {
//         Kernel<<<1, 10>>>(h_func);
//         __OPEN3D_CUDA_CHECK(cudaDeviceSynchronize(), __FILE__, __LINE__);
//         __OPEN3D_CUDA_CHECK(cudaGetLastError(), __FILE__, __LINE__);
//     }
// };

/// In source
/// We still need to write some cuda code in main file, but the library doesn't
/// have to see the device function
__device__ int inc(int val) { return val + 1; }
/// This assignment cannot be ignored, otherwise there will be 'invalid program
/// counter' error
__device__ hash_t hash = inc;

int main() {
    hash_t h_func;
    cudaMemcpyFromSymbol(&h_func, hash, sizeof(hash_t));
    __OPEN3D_CUDA_CHECK(cudaGetLastError(), __FILE__, __LINE__);

    Caller caller;
    caller.Launch(h_func);
}