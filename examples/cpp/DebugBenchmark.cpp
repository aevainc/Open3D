// Compile with -DBUILD_CUDA_MODULE=ON
// This example gives error:
// cpp/open3d/core/CUDAState.cuh:86 CUDA runtime error: driver shutting down

#include "open3d/Open3D.h"

int main() {
    using namespace open3d;
    core::Tensor t = core::Tensor::Empty({2, 3}, core::Dtype::Float32,
                                         core::Device("CUDA:0"));
    utility::LogInfo("Done.");
    return 0;
}
