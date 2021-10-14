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

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/linalg/AddMM.h"
#include "open3d/core/linalg/Matmul.h"
#include "open3d/core/nns/KnnIndex.h"
#include "open3d/core/nns/KnnSearchImplNew.cuh"
#include "open3d/core/nns/NeighborSearchAllocator.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Timer.h"

namespace open3d {
namespace core {
namespace nns {

template <class T>
void KnnSearchCUDASingle(const cudaStream_t stream,
                         const Tensor& points,
                         const Tensor& queries,
                         int knn,
                         Tensor& neighbors_index,
                         Tensor& neighbors_distance) {
    int num_points = points.GetShape(0);
    int num_queries = queries.GetShape(0);
    int ndim = points.GetShape(1);

    knn = num_points > knn ? knn : num_points;

    Device device = points.GetDevice();
    NeighborSearchAllocator<T> output_allocator(device);

    int32_t* indices_ptr;
    T* distances_ptr;
    utility::Timer timer;

    timer.Start();
    output_allocator.AllocIndices(&indices_ptr, num_queries * knn);
    output_allocator.AllocDistances(&distances_ptr, num_queries * knn);
    Tensor neighbor_indices = output_allocator.NeighborsIndex();
    Tensor neighbor_distances = output_allocator.NeighborsDistance();
    timer.Stop();
    utility::LogInfo("{} {:.2f} ms.", "output alloc", timer.GetDuration());

    timer.Start();
    Tensor points_norm = points.Mul(points).Sum({1});
    Tensor queries_norm = queries.Mul(queries).Sum({1}, true);
    timer.Stop();
    utility::LogInfo("{} {:.2f} ms.", "norm", timer.GetDuration());

    int tileRow, tileCol;
    impl::chooseTileSize(num_queries, num_points, ndim, sizeof(T), tileRow,
                         tileCol);

    std::cout << "tileRow: " << tileRow << ", tileCol: " << tileCol
              << std::endl;
    int numCol = utility::DivUp(num_points, tileCol);
    Tensor temp_dist = Tensor::Empty({tileRow, num_points}, points.GetDtype(),
                                     points.GetDevice());
    double outer_slice = 0.0;
    double inner_slice = 0.0;
    double matmul = 0.0;
    double mul = 0.0;
    double add_point = 0.0;
    double add_query = 0.0;
    double heap = 0.0;

    for (int i = 0; i < num_queries; i += tileRow) {
        int num_queries_i = std::min(tileRow, num_queries - i);

        int32_t* indices_ptr_i = &indices_ptr[i * knn];
        T* distances_ptr_i = &distances_ptr[i * knn];

        timer.Start();
        Tensor queries_norm_i = queries_norm.Slice(0, i, i + num_queries_i);
        Tensor queries_i = queries.Slice(0, i, i + num_queries_i);
        Tensor temp_dist_row_view = temp_dist.Slice(0, 0, num_queries_i);
        Tensor neighbor_distances_i =
                neighbor_distances.View({num_queries, knn})
                        .Slice(0, i, i + num_queries_i);
        timer.Stop();
        outer_slice += timer.GetDuration();

        for (int j = 0; j < num_points; j += tileCol) {
            int num_points_j = std::min(tileCol, num_points - j);
            timer.Start();
            Tensor points_j = points.Slice(0, j, j + num_points_j);
            Tensor temp_dist_col_view =
                    temp_dist_row_view.Slice(1, j, j + num_points_j);
            Tensor points_norm_j =
                    points_norm.Slice(0, j, j + num_points_j)
                            .Expand({num_queries_i, num_points_j});
            timer.Stop();
            inner_slice += timer.GetDuration();

            timer.Start();
            temp_dist_col_view.AsRvalue() = points_norm_j;
            AddMM<T>(queries_i, points_j, temp_dist_col_view, -2.0, 1.0);
            timer.Stop();
            matmul += timer.GetDuration();

            timer.Start();
            //     temp_dist_col_view.Mul_(-2);
            timer.Stop();
            mul += timer.GetDuration();

            timer.Start();
            //     temp_dist_col_view.Add_(points_norm_j);
            timer.Stop();
            add_point += timer.GetDuration();
        }

        timer.Start();
        impl::HeapSortCUDA<T>(stream, indices_ptr_i, distances_ptr_i,
                              temp_dist_row_view.GetDataPtr<T>(), num_points,
                              num_queries_i, ndim, knn);
        timer.Stop();
        heap += timer.GetDuration();

        timer.Start();
        neighbor_distances_i.Add_(queries_norm_i);
        timer.Stop();
        add_query += timer.GetDuration();
    }
    utility::LogInfo("{} {:.2f} ms.", "outer_slice", outer_slice);
    utility::LogInfo("{} {:.2f} ms.", "inner_slice", inner_slice);
    utility::LogInfo("{} {:.2f} ms.", "matmul", matmul);
    utility::LogInfo("{} {:.2f} ms.", "mul", mul);
    utility::LogInfo("{} {:.2f} ms.", "add_point", add_point);
    utility::LogInfo("{} {:.2f} ms.", "add_query", add_query);
    utility::LogInfo("{} {:.2f} ms.", "heap_sort", heap);
    std::cout << std::endl;
    neighbors_index =
            output_allocator.NeighborsIndex().View({num_queries, knn});
    neighbors_distance =
            output_allocator.NeighborsDistance().View({num_queries, knn});
}

template <class T>
void KnnSearchCUDANew(const Tensor& points,
                      const Tensor& points_row_splits,
                      const Tensor& queries,
                      const Tensor& queries_row_splits,
                      int knn,
                      Tensor& neighbors_index,
                      Tensor& neighbors_distance) {
    const cudaStream_t stream = cuda::GetStream();

    //     Device device = points.GetDevice();
    //     NeighborSearchAllocator<T> output_allocator(device);

    //     int ndim = points.GetShape(1);
    //     int num_points = points.GetShape(0);
    //     int num_queries = queries.GetShape(0);
    //     knn = num_points > knn ? knn : num_points;

    int64_t num_batch = points_row_splits.GetShape()[0] - 1;

    for (auto i = 0; i < num_batch; ++i) {
        Tensor point_i = points.Slice(0, points_row_splits[i].Item<int64_t>(),
                                      points_row_splits[i + 1].Item<int64_t>());
        Tensor query_i =
                queries.Slice(0, queries_row_splits[i].Item<int64_t>(),
                              queries_row_splits[i + 1].Item<int64_t>());

        // Tensor norm_point_i = point_i.Mul(point_i).Sum({1});
        // Tensor norm_query_i = query_i.Mul(query_i).Sum({1});

        // Tensor point_query = query_i.Matmul(point_i.T());
        // Tensor distance = norm_query_i - 2 * point_query + norm_point_i;

        // impl::HeapSortCUDA(stream, num_points, distance.GetDataPtr<T>(),
        //                    num_queries, ndim, knn, output_allocator);
        KnnSearchCUDASingle<T>(stream, point_i, query_i, knn, neighbors_index,
                               neighbors_distance);
    }

    //     neighbors_index =
    //             output_allocator.NeighborsIndex().View({num_queries, knn});
    //     neighbors_distance =
    //             output_allocator.NeighborsDistance().View({num_queries,
    //             knn});
}

#define INSTANTIATE(T)                                                        \
    template void KnnSearchCUDANew<T>(                                        \
            const Tensor& points, const Tensor& points_row_splits,            \
            const Tensor& queries, const Tensor& queries_row_splits, int knn, \
            Tensor& neighbors_index, Tensor& neighbors_distance);

INSTANTIATE(float)
INSTANTIATE(double)
}  // namespace nns
}  // namespace core
}  // namespace open3d
