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

#include <benchmark/benchmark.h>

#include "open3d/core/AdvancedIndexing.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CPULauncher.h"
#include "open3d/core/kernel/Kernel.h"

namespace open3d {
namespace core {

void SimpleOpGrained(benchmark::State& state) {
    core::Device device("CPU:0");
    core::kernel::cpu_launcher::SetSmallOpGrainSize(32767);

    SizeVector shape{32766};
    Tensor ones = Tensor::Ones(shape, core::Float64, device);
    Tensor zeros = Tensor::Zeros(shape, core::Float64, device);
    Tensor warm_up = ones + zeros;
    for (auto _ : state) {
        Tensor dst = ones + zeros;
    }
}

void SimpleOp(benchmark::State& state) {
    core::Device device("CPU:0");
    core::kernel::cpu_launcher::SetSmallOpGrainSize(1);

    SizeVector shape{32766};
    Tensor ones = Tensor::Ones(shape, core::Float64, device);
    Tensor zeros = Tensor::Zeros(shape, core::Float64, device);
    Tensor warm_up = ones + zeros;
    for (auto _ : state) {
        Tensor dst = ones + zeros;
    }
}

BENCHMARK(SimpleOpGrained)->Unit(benchmark::kMillisecond);
BENCHMARK(SimpleOp)->Unit(benchmark::kMillisecond);

}  // namespace core
}  // namespace open3d
