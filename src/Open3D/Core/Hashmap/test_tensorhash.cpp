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

#include "Open3D/Core/Hashmap/TensorHash.h"

using namespace open3d;

int main() {
    std::vector<Device> devices = {Device("CPU:0"), Device("CUDA:0")};

    for (auto device : devices) {
        /// Init
        Tensor init_coords(std::vector<float>({0, 0, 1, 1, 2, 2, 3, 3, 4, 4}),
                           {5, 2}, Dtype::Float32, device);
        Tensor init_values(std::vector<int64_t>({0, 1, 2, 3, 4}), {5},
                           Dtype::Int64, device);
        utility::LogInfo("Create");
        auto tensor_hash =
                std::make_shared<TensorHash>(init_coords, init_values);

        /// Query
        utility::LogInfo("Query");
        Tensor query_coords(std::vector<float>({0, 0, 3, 3, 1, 1, 4, 4, 8, 8}),
                            {5, 2}, Dtype::Float32, device);
        auto results = tensor_hash->Query(query_coords);

        // [Open3D INFO] IndexTensor [0 3 1 4 0]
        // Tensor[shape={5}, stride={1}, Int64, CUDA:0, 0x7f85dde01600]
        // [Open3D INFO] MaskTensor [1 1 1 1 0]
        // Tensor[shape={5}, stride={1}, UInt8, CUDA:0, 0x7f85dde01800]
        std::cout << "IndexTensor " << results.first.ToString() << "\n";
        std::cout << "MaskTensor " << results.second.ToString() << "\n";

        /// Assign: tensor[(0, 0)] = 2, tensor[(2, 2)] = 0
        utility::LogInfo("Assign");
        Tensor assign_coords(std::vector<float>({0, 0, 2, 2}), {2, 2},
                             Dtype::Float32, device);
        Tensor assign_values(std::vector<int64_t>({2, 0}), {2}, Dtype::Int64,
                             device);
        tensor_hash->Assign(assign_coords, assign_values);

        /// Query init after reassignment
        results = tensor_hash->Query(init_coords);
        // [Open3D INFO] IndexTensor [2 1 0 3 4]
        // Tensor[shape={5}, stride={1}, Int64, CUDA:0, 0x7f8615e01e00]
        // [Open3D INFO] MaskTensor [1 1 1 1 1]
        // Tensor[shape={5}, stride={1}, UInt8, CUDA:0, 0x7f8615e02000]
        std::cout << "IndexTensor " << results.first.ToString() << "\n";
        std::cout << "MaskTensor " << results.second.ToString() << "\n";

        /// Test Unique
        Tensor duplicate_coords(
                std::vector<float>({0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 4, 4, 3, 3}),
                {7, 2}, Dtype::Float32, device);
        results = Unique(duplicate_coords);

        // [Open3D INFO] IndexTensor [[0 0],
        //                            [1 1],
        //                            [2 2],
        //                            [3 3],
        //                            [0 0],
        //                            [4 4],
        //                            [0 0]]
        // Tensor[shape={7, 2}, stride={2, 1}, Float32, CUDA:0, 0x7ff65de03200]
        // [Open3D INFO] MaskTensor [1 1 1 1 0 1 0]
        // Tensor[shape={7}, stride={1}, UInt8, CUDA:0, 0x7ff65de03400]
        // -> [[0 0], [1 1], [2 2], [3 3], [4 4]] after advanced indexing with
        // masks
        std::cout << "IndexTensor " << results.first.ToString() << "\n";
        std::cout << "MaskTensor " << results.second.ToString() << "\n";
    }
}
