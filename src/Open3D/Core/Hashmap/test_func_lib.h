#pragma once

#include <stdio.h>

inline void __OPEN3D_CUDA_CHECK(cudaError_t err,
                                const char *file,
                                const int line) {
    if (err != cudaSuccess) {
        printf("%s:%d CUDA runtime error: %s", file, line,
               cudaGetErrorString(err));
    }
}

/// In library to be compiled alone
typedef int (*hash_t)(int);

class Caller {
public:
    void Launch(hash_t h_func);
};
