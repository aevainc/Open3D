// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "open3d/core/AdvancedIndexing.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"

// CUDA kernel launcher's goal is to separate scheduling (looping through each
// valid element) and computation (operations performed on each element).
//
// The kernel launch mechanism is inspired by PyTorch's launch Loops.cuh.
// See: https://tinyurl.com/y4lak257

static constexpr int64_t default_block_size = 128;
static constexpr int64_t default_thread_size = 4;

namespace open3d {
namespace core {
namespace kernel {

// Applies f for each element
// Works for unary / binary elementwise operations
template <int64_t block_size, int64_t thread_size, typename func_t>
__global__ void ElementWiseKernel(int64_t n, func_t f) {
    int64_t items_per_block = block_size * thread_size;
    int64_t idx = blockIdx.x * items_per_block + threadIdx.x;
#pragma unroll
    for (int64_t i = 0; i < thread_size; i++) {
        if (idx < n) {
            f(idx);
            idx += block_size;
        }
    }
}

class CUDALauncher {
public:
    template <typename func_t>
    static void LaunchUnaryEWKernel(const Indexer& indexer,
                                    func_t element_kernel) {
        OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);

        int64_t n = indexer.NumWorkloads();
        if (n == 0) {
            return;
        }
        int64_t items_per_block = default_block_size * default_thread_size;
        int64_t grid_size = (n + items_per_block - 1) / items_per_block;

        auto f = [=] OPEN3D_HOST_DEVICE(int64_t workload_idx) {
            element_kernel(indexer.GetInputPtr(0, workload_idx),
                           indexer.GetOutputPtr(workload_idx));
        };

        ElementWiseKernel<default_block_size, default_thread_size>
                <<<grid_size, default_block_size, 0>>>(n, f);
        OPEN3D_GET_LAST_CUDA_ERROR("LaunchUnaryEWKernel failed.");
    }

    template <typename func_t>
    static void LaunchBinaryEWKernel(const Indexer& indexer,
                                     func_t element_kernel) {
        OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);

        int64_t n = indexer.NumWorkloads();
        if (n == 0) {
            return;
        }
        int64_t items_per_block = default_block_size * default_thread_size;
        int64_t grid_size = (n + items_per_block - 1) / items_per_block;

        auto f = [=] OPEN3D_HOST_DEVICE(int64_t workload_idx) {
            element_kernel(indexer.GetInputPtr(0, workload_idx),
                           indexer.GetInputPtr(1, workload_idx),
                           indexer.GetOutputPtr(workload_idx));
        };

        ElementWiseKernel<default_block_size, default_thread_size>
                <<<grid_size, default_block_size, 0>>>(n, f);
        OPEN3D_GET_LAST_CUDA_ERROR("LaunchBinaryEWKernel failed.");
    }

    template <typename func_t>
    static void LaunchAdvancedIndexerKernel(const AdvancedIndexer& indexer,
                                            func_t element_kernel) {
        OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);

        int64_t n = indexer.NumWorkloads();
        if (n == 0) {
            return;
        }
        int64_t items_per_block = default_block_size * default_thread_size;
        int64_t grid_size = (n + items_per_block - 1) / items_per_block;

        auto f = [=] OPEN3D_HOST_DEVICE(int64_t workload_idx) {
            element_kernel(indexer.GetInputPtr(workload_idx),
                           indexer.GetOutputPtr(workload_idx));
        };

        ElementWiseKernel<default_block_size, default_thread_size>
                <<<grid_size, default_block_size, 0>>>(n, f);
        OPEN3D_GET_LAST_CUDA_ERROR("LaunchAdvancedIndexerKernel failed.");
    }

    template <typename func_t>
    static void LaunchImageUnaryKernel(const Indexer& indexer,
                                       func_t element_kernel) {
        OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);

        int64_t n = indexer.NumWorkloads();
        if (n == 0) {
            return;
        }
        int64_t items_per_block = default_block_size * default_thread_size;
        int64_t grid_size = (n + items_per_block - 1) / items_per_block;

        auto f = [=] OPEN3D_HOST_DEVICE(int64_t workload_idx) {
            int64_t x, y;
            indexer.GetWorkload2DIdx(workload_idx, x, y);
            element_kernel(x, y, indexer.GetInputPtr(0, workload_idx),
                           indexer.GetOutputPtr(0, workload_idx),
                           indexer.GetOutputPtr(1, workload_idx),
                           indexer.GetOutputPtr(2, workload_idx));
        };

        ElementWiseKernel<default_block_size, default_thread_size>
                <<<grid_size, default_block_size, 0>>>(n, f);
        cudaDeviceSynchronize();
        OPEN3D_GET_LAST_CUDA_ERROR("LaunchUnaryEWKernel failed.");
    }

    template <typename func_t>
    static void LaunchIntegrateKernel(const SparseIndexer& indexer,
                                      const Projector& projector,
                                      func_t element_kernel) {
        OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);

        int64_t n = indexer.NumWorkloads();
        if (n == 0) {
            return;
        }
        int64_t items_per_block = default_block_size * default_thread_size;
        int64_t grid_size = (n + items_per_block - 1) / items_per_block;

        auto f = [=] OPEN3D_HOST_DEVICE(int64_t workload_idx) {
            int64_t key_idx, value_idx;
            indexer.GetSparseWorkloadIdx(workload_idx, &key_idx, &value_idx);

            int64_t xl, yl, zl;
            indexer.GetWorkloadValue3DIdx(value_idx, &xl, &yl, &zl);

            void* key_ptr = indexer.GetWorkloadKeyPtr(key_idx);
            int64_t xg = *(static_cast<int64_t*>(key_ptr) + 0);
            int64_t yg = *(static_cast<int64_t*>(key_ptr) + 1);
            int64_t zg = *(static_cast<int64_t*>(key_ptr) + 2);

            int64_t resolution = indexer.sparse_tl_.data_desc_.GetStride(-2);
            int64_t x = (xg * resolution + xl);
            int64_t y = (yg * resolution + yl);
            int64_t z = (zg * resolution + zl);

            float xc, yc, zc;
            projector.Transform(static_cast<float>(x), static_cast<float>(y),
                                static_cast<float>(z), &xc, &yc, &zc);

            float u, v;
            projector.Project(xc, yc, zc, &u, &v);

            void* input_ptr = indexer.GetInputPtrFrom2D(0, u, v);
            void* tsdf_ptr = indexer.GetWorkloadValuePtr(0, key_idx, value_idx);
            void* weight_ptr =
                    indexer.GetWorkloadValuePtr(1, key_idx, value_idx);
            element_kernel(tsdf_ptr, weight_ptr, input_ptr, zc);
        };

        ElementWiseKernel<default_block_size, default_thread_size>
                <<<grid_size, default_block_size, 0>>>(n, f);
        cudaDeviceSynchronize();
        OPEN3D_GET_LAST_CUDA_ERROR("LaunchIntegrateKernel failed.");
    }
};
}  // namespace kernel
}  // namespace core
}  // namespace open3d
