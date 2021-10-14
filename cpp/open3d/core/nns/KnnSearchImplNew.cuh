// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "cub/cub.cuh"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/MiniVec.h"

namespace open3d {
namespace core {
namespace nns {
namespace impl {

namespace {

template <class T>
inline __device__ T Abs(T x) {
    return x > 0 ? x : -x;
}

template <class T>
inline __device__ void Swap(T *x, T *y) {
    T tmp = *x;
    *x = *y;
    *y = tmp;
}

template <int METRIC = L2, class T, int CHUNK = 64>
inline __device__ T NeighborsDist(const T *p1, const T *p2, const int ndim) {
    T dist = 0;
    T dist_vec[CHUNK];
    int cnt = 0;

    if (METRIC == Linf) {
        while (ndim - cnt > CHUNK) {
            for (int j = 0; j < CHUNK; ++j) {
                dist_vec[j] = Abs(p1[cnt + j] - p2[cnt + j]);
            }
            for (int j = 0; j < CHUNK; ++j) {
                dist = dist > dist_vec[j] ? dist : dist_vec[j];
                cnt += 1;
            }
        }
        for (int i = cnt; i < ndim; ++i) {
            T dist_tmp = Abs(p1[i] - p2[i]);
            dist = dist > dist_tmp ? dist : dist_tmp;
        }

    } else if (METRIC == L1) {
        while (ndim - cnt > CHUNK) {
            for (int j = 0; j < CHUNK; ++j) {
                dist_vec[j] = Abs(p1[cnt + j] - p2[cnt + j]);
            }
            for (int j = 0; j < CHUNK; ++j) {
                dist += dist_vec[j];
                cnt += 1;
            }
        }
        for (int i = cnt; i < ndim; ++i) {
            T dist_tmp = Abs(p1[i] - p2[i]);
            dist += dist_tmp;
        }
    } else {
        while (ndim - cnt > CHUNK) {
#pragma unroll
            for (int j = 0; j < CHUNK; ++j) {
                dist_vec[j] = p1[cnt + j] - p2[cnt + j];
            }
#pragma unroll
            for (int j = 0; j < CHUNK; ++j) {
                dist += dist_vec[j] * dist_vec[j];
                cnt += 1;
            }
        }
        for (int i = cnt; i < ndim; ++i) {
            T dist_tmp = p1[i] - p2[i];
            dist_tmp = dist_tmp * dist_tmp;
            dist += dist_tmp;
        }
    }
    return dist;
}

template <class T, class TIndex>
inline __device__ void Heapify(T *dist, TIndex *idx, int root, int k) {
    int child = root * 2 + 1;

    while (child < k) {
        if (child + 1 < k && dist[child + 1] > dist[child]) {
            child++;
        }
        if (dist[root] > dist[child]) {
            return;
        }
        Swap<T>(&dist[root], &dist[child]);
        Swap<TIndex>(&idx[root], &idx[child]);
        root = child;
        child = root * 2 + 1;
    }
}

template <class T, class TIndex>
__device__ void SortHeap(T *dist, TIndex *idx, int k) {
    int i;
    for (i = k - 1; i > 0; i--) {
        Swap<T>(&dist[0], &dist[i]);
        Swap<TIndex>(&idx[0], &idx[i]);
        Heapify<T, TIndex>(dist, idx, 0, i);
    }
}

template <class T, class TIndex, int METRIC = L2>
__global__ void KnnQueryKernel(TIndex *__restrict__ indices_ptr,
                               T *__restrict__ distances_ptr,
                               int ndim,
                               size_t num_points,
                               const T *__restrict__ points,
                               size_t num_queries,
                               const T *__restrict__ queries,
                               int knn) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) return;

    T best_dist[100];
    int best_idx[100];

    for (int i = 0; i < knn; i++) {
        best_dist[i] = 1e10;
        best_idx[i] = 0;
    }

    constexpr int chunk_size = 128;
    const T *queries_i = queries + ndim * query_idx;
    for (int i = 0; i < num_points; i++) {
        T dist = NeighborsDist<METRIC, T, chunk_size>(queries_i,
                                                      points + ndim * i, ndim);
        if (dist < best_dist[0]) {
            best_dist[0] = dist;
            best_idx[0] = i;
            Heapify(best_dist, best_idx, 0, knn);
        }
    }
    SortHeap(best_dist, best_idx, knn);
    for (int i = 0; i < knn; i++) {
        indices_ptr[i + knn * query_idx] = best_idx[i];
        distances_ptr[i + knn * query_idx] = best_dist[i];
    }
}

template <class T, class TIndex>
__global__ void HeapSortKernel(TIndex *__restrict__ indices_ptr,
                               T *__restrict__ distances_ptr,
                               T *__restrict__ in_distances_ptr,
                               size_t num_points,
                               size_t num_queries,
                               int knn) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) return;

    T best_dist[100];
    int best_idx[100];

    for (int i = 0; i < knn; i++) {
        best_dist[i] = 1e10;
        best_idx[i] = 0;
    }

    for (int i = 0; i < num_points; i++) {
        T dist = in_distances_ptr[query_idx * num_points + i];
        if (dist < best_dist[0]) {
            best_dist[0] = dist;
            best_idx[0] = i;
            Heapify(best_dist, best_idx, 0, knn);
        }
    }
    SortHeap(best_dist, best_idx, knn);
    for (int i = 0; i < knn; i++) {
        indices_ptr[i + knn * query_idx] = best_idx[i];
        distances_ptr[i + knn * query_idx] = best_dist[i];
    }
}

template <class T, class TIndex>
void HeapSort(const cudaStream_t &stream,
              TIndex *__restrict__ indices_ptr,
              T *__restrict__ distances_ptr,
              T *__restrict__ in_distances_ptr,
              size_t num_points,
              size_t num_queries,
              int knn) {
    const int BLOCKSIZE = 256;
    dim3 block(BLOCKSIZE, 1, 1);
    dim3 grid(0, 1, 1);
    grid.x = utility::DivUp(num_queries, block.x);

    if (grid.x) {
        HeapSortKernel<T, TIndex><<<grid, block, 0, stream>>>(
                indices_ptr, distances_ptr, in_distances_ptr, num_points,
                num_queries, knn);
    }
}

template <class T, class TIndex>
void KnnQuery(const cudaStream_t &stream,
              int ndim,
              TIndex *indices_ptr,
              T *distances_ptr,
              size_t num_points,
              const T *const points,
              size_t num_queries,
              const T *const queries,
              int knn) {
    // input: queries: (m, 3), points: (n, 3), idx: (m, knn)
    const int BLOCKSIZE = 256;
    dim3 block(BLOCKSIZE, 1, 1);
    dim3 grid(0, 1, 1);
    grid.x = utility::DivUp(num_queries, block.x);

    if (grid.x) {
        KnnQueryKernel<T, TIndex, L2><<<grid, block, 0, stream>>>(
                indices_ptr, distances_ptr, ndim, num_points, points,
                num_queries, queries, knn);
    }
}

}  // namespace

inline void chooseTileSize(int num_queries,
                           int num_points,
                           int dim,
                           int elementSize,
                           int &tileRows,
                           int &tileCols) {
    // The matrix multiplication should be large enough to be efficient, but if
    // it is too large, we seem to lose efficiency as opposed to
    // double-streaming. Each tile size here defines 1/2 of the memory use due
    // to double streaming. We ignore available temporary memory, as that is
    // adjusted independently by the user and can thus meet these requirements
    // (or not). For <= 4 GB GPUs, prefer 512 MB of usage. For <= 8 GB GPUs,
    // prefer 768 MB of usage. Otherwise, prefer 1 GB of usage.
    size_t free, total;
    GetCUDAMemoryInfo(free, total);

    int64_t targetUsage = 0;

    if (total <= ((size_t)4) * 1024 * 1024 * 1024) {
        targetUsage = 512 * 1024 * 1024;
    } else if (total <= ((size_t)8) * 1024 * 1024 * 1024) {
        targetUsage = 768 * 1024 * 1024;
    } else {
        targetUsage = 1024 * 1024 * 1024;
    }

    targetUsage /= 2 * elementSize;

    // 512 seems to be a batch size sweetspot for float32.
    // If we are on float16, increase to 512.
    // If the k size (vec dim) of the matrix multiplication is small (<= 32),
    // increase to 1024.
    int preferredTileRows = 512 * 4;
    if (dim <= 32) {
        preferredTileRows = 1024 * 4;
    }

    tileRows = std::min(preferredTileRows, num_queries);

    // tileCols is the remainder size
    tileCols = std::min(int(targetUsage / preferredTileRows), num_points);
}

template <class T>
void HeapSortCUDA(const cudaStream_t &stream,
                  int32_t *__restrict__ out_indices_ptr,
                  T *__restrict__ out_distances_ptr,
                  T *__restrict__ in_distances_ptr,
                  size_t num_points,
                  size_t num_queries,
                  int dim,
                  int knn) {
    HeapSort<T, int32_t>(stream, out_indices_ptr, out_distances_ptr,
                         in_distances_ptr, num_points, num_queries, knn);
}

template <class T, class OUTPUT_ALLOCATOR>
void KnnSearchCUDA(const cudaStream_t stream,
                   int ndim,
                   size_t num_points,
                   const T *const points,
                   size_t num_queries,
                   const T *const queries,
                   size_t points_row_splits_size,
                   const int64_t *const points_row_splits,
                   size_t queries_row_splits_size,
                   const int64_t *const queries_row_splits,
                   int knn,
                   OUTPUT_ALLOCATOR &output_allocator) {
    const int batch_size = points_row_splits_size - 1;

    const size_t num_indices = num_queries * knn;

    int32_t *indices_ptr;
    T *distances_ptr;

    output_allocator.AllocIndices(&indices_ptr, num_indices);
    output_allocator.AllocDistances(&distances_ptr, num_indices);

    for (int i = 0; i < batch_size; ++i) {
        const size_t num_queries_i =
                queries_row_splits[i + 1] - queries_row_splits[i];
        const size_t num_points_i =
                points_row_splits[i + 1] - points_row_splits[i];

        const T *const points_i = points + 3 * points_row_splits[i];
        const T *const queries_i = queries + 3 * queries_row_splits[i];
        int32_t *indices_ptr_i = indices_ptr + queries_row_splits[i] * knn;
        T *distances_ptr_i = distances_ptr + queries_row_splits[i] * knn;
        KnnQuery<T, int32_t>(stream, ndim, indices_ptr_i, distances_ptr_i,
                             num_points_i, points_i, num_queries_i, queries_i,
                             knn);
    }
}

}  // namespace impl
}  // namespace nns
}  // namespace core
}  // namespace open3d
