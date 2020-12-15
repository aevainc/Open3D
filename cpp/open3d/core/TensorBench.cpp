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

#include "open3d/core/TensorBench.h"

#include "open3d/core/Device.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Timer.h"

#ifdef BUILD_CUDA_MODULE
#include "open3d/core/CUDAState.cuh"
#endif

namespace open3d {
namespace core {

static std::vector<Device> PermuteDevices() {
#ifdef BUILD_CUDA_MODULE
    std::shared_ptr<core::CUDAState> cuda_state =
            core::CUDAState::GetInstance();
    if (cuda_state->GetNumDevices() >= 1) {
        return {core::Device("CPU:0"), core::Device("CUDA:0")};
    } else {
        return {core::Device("CPU:0")};
    }
#else
    return {core::Device("CPU:0")};
#endif
}

template <typename func_t>
static void RunBenchmark(func_t benchmark_func,
                         const std::string& name,
                         int repeats) {
    if (repeats <= 0) {
        utility::LogError("repeats must be > 0");
    }
    benchmark_func();  // Warm up.

    utility::Timer timer;
    timer.Start();
    for (int i = 0; i < repeats; i++) {
        benchmark_func();
    }
    timer.Stop();
    double avg_time = timer.GetDuration() / static_cast<double>(repeats);
    utility::LogInfo("Name: {}; Avg time: {:.2f}ms; Repeats: {}", name,
                     avg_time, repeats);
}

void RunTensorBench() {
    // Reduction.
    for (const auto& device : PermuteDevices()) {
        utility::LogInfo("Device: {}", device.ToString());
        SizeVector shape{2, 10000000};
        Tensor src(shape, Dtype::Int64, device);
        RunBenchmark([&]() { Tensor dst = src.Sum({1}); },
                     "reduction_" + device.ToString(), 10);
    }

    // Add.
    for (const auto& device : PermuteDevices()) {
        utility::LogInfo("Device: {}", device.ToString());
        SizeVector shape{2, 10000000};
        Tensor lhs = Tensor::Ones(shape, Dtype::Int64, device);
        Tensor rhs = Tensor::Ones(shape, Dtype::Int64, device);
        RunBenchmark([&]() { Tensor dst = lhs + rhs; },
                     "add_" + device.ToString(), 10);
    }
}

}  // namespace core
}  // namespace open3d
