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

#include "Open3D/Core/MemoryManager.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "Open3D/Core/CUDAState.cuh"
#include "Open3D/Core/CUDAUtils.h"

namespace open3d {

static bool BlockComparator(const BlockPtr& a, const BlockPtr& b) {
    // Not on the same device: treat as smaller, will be filtered in lower_bound
    // operation.
    if (a->device_ != b->device_) {
        return true;
    }
    if (a->size_ != b->size_) {
        return a->size_ < b->size_;
    }
    return (size_t)a->ptr_ < (size_t)b->ptr_;
}

CUDAMemoryManager::CUDAMemoryManager() {
    small_block_pool_ = std::make_shared<BlockPool>(BlockComparator);
    large_block_pool_ = std::make_shared<BlockPool>(BlockComparator);
}

void* CUDAMemoryManager::Malloc(size_t byte_size, const Device& device) {
    CUDADeviceSwitcher switcher(device);
    void* ptr;

    if (device.GetType() == Device::DeviceType::CUDA) {
        byte_size = align_size(byte_size);
        BlockPtr query_block =
                std::make_shared<Block>(device.GetID(), byte_size);

        // Find corresponding pool
        auto pool = get_pool(byte_size);

        // Query block in the pool
        auto find_free_block = [&]() -> BlockPtr {
            auto it = pool->lower_bound(query_block);
            if (it != pool->end()) {
                BlockPtr block = *it;
                pool->erase(it);
                return block;
            }
            return nullptr;
        };

        BlockPtr found_block = find_free_block();
        if (found_block == nullptr) {
            // Allocate and insert to the allocated pool
            OPEN3D_CUDA_CHECK(cudaMalloc(static_cast<void**>(&ptr), byte_size));
            allocated_blocks_.insert(
                    {ptr,
                     std::make_shared<Block>(device.GetID(), byte_size, ptr)});
        } else {
            // Set raw ptr for return
            ptr = found_block->ptr_;

            // Adapt the found block
            BlockPtr head_block =
                    std::make_shared<Block>(device.GetID(), byte_size, ptr);
            allocated_blocks_.insert({ptr, head_block});

            // Manage the remaining block
            size_t tail_byte_size = found_block->size_ - byte_size;
            if (tail_byte_size > 0) {
                auto tail_pool = get_pool(tail_byte_size);
                BlockPtr tail_block = std::make_shared<Block>(
                        device.GetID(), tail_byte_size,
                        static_cast<char*>(ptr) + byte_size);
                tail_pool->emplace(tail_block);
            }
        }
    } else {
        utility::LogError("CUDAMemoryManager::Malloc: Unimplemented device");
    }
    return ptr;
}

void CUDAMemoryManager::Free(void* ptr, const Device& device) {
    CUDADeviceSwitcher switcher(device);
    if (device.GetType() == Device::DeviceType::CUDA) {
        if (ptr && IsCUDAPointer(ptr)) {
            auto it = allocated_blocks_.find(ptr);

            if (it == allocated_blocks_.end()) {
                // Should never reach here!
                utility::LogError(
                        "CUDAMemoryManager::Free: Memory leak! Block should "
                        "have been stored.");
            } else {
                // Release memory to the corresponding pool
                BlockPtr block = it->second;
                allocated_blocks_.erase(it);
                auto pool = get_pool(block->size_);
                pool->emplace(block);
            }
        } else {
            utility::LogError("CUDAMemoryManager::Free: Invalid pointer");
        }
    } else {
        utility::LogError("CUDAMemoryManager::Free: Unimplemented device");
    }
}

void CUDAMemoryManager::Memcpy(void* dst_ptr,
                               const Device& dst_device,
                               const void* src_ptr,
                               const Device& src_device,
                               size_t num_bytes) {
    if (dst_device.GetType() == Device::DeviceType::CUDA &&
        src_device.GetType() == Device::DeviceType::CPU) {
        CUDADeviceSwitcher switcher(dst_device);
        if (!IsCUDAPointer(dst_ptr)) {
            utility::LogError("dst_ptr is not a CUDA pointer");
        }
        OPEN3D_CUDA_CHECK(cudaMemcpy(dst_ptr, src_ptr, num_bytes,
                                     cudaMemcpyHostToDevice));
    } else if (dst_device.GetType() == Device::DeviceType::CPU &&
               src_device.GetType() == Device::DeviceType::CUDA) {
        CUDADeviceSwitcher switcher(src_device);
        if (!IsCUDAPointer(src_ptr)) {
            utility::LogError("src_ptr is not a CUDA pointer");
        }
        OPEN3D_CUDA_CHECK(cudaMemcpy(dst_ptr, src_ptr, num_bytes,
                                     cudaMemcpyDeviceToHost));
    } else if (dst_device.GetType() == Device::DeviceType::CUDA &&
               src_device.GetType() == Device::DeviceType::CUDA) {
        CUDADeviceSwitcher switcher(dst_device);
        if (!IsCUDAPointer(dst_ptr)) {
            utility::LogError("dst_ptr is not a CUDA pointer");
        }
        switcher.SwitchTo(src_device);
        if (!IsCUDAPointer(src_ptr)) {
            utility::LogError("src_ptr is not a CUDA pointer");
        }

        if (dst_device == src_device) {
            CUDADeviceSwitcher switcher(src_device);
            OPEN3D_CUDA_CHECK(cudaMemcpy(dst_ptr, src_ptr, num_bytes,
                                         cudaMemcpyDeviceToDevice));
        } else if (CUDAState::GetInstance()->IsP2PEnabled(src_device.GetID(),
                                                          dst_device.GetID())) {
            OPEN3D_CUDA_CHECK(cudaMemcpyPeer(dst_ptr, dst_device.GetID(),
                                             src_ptr, src_device.GetID(),
                                             num_bytes));
        } else {
            void* cpu_buf = MemoryManager::Malloc(num_bytes, Device("CPU:0"));
            CUDADeviceSwitcher switcher(src_device);
            OPEN3D_CUDA_CHECK(cudaMemcpy(cpu_buf, src_ptr, num_bytes,
                                         cudaMemcpyDeviceToHost));
            switcher.SwitchTo(dst_device);
            OPEN3D_CUDA_CHECK(cudaMemcpy(dst_ptr, cpu_buf, num_bytes,
                                         cudaMemcpyHostToDevice));
            MemoryManager::Free(cpu_buf, Device("CPU:0"));
        }
    } else {
        utility::LogError("Wrong cudaMemcpyKind");
    }
}

bool CUDAMemoryManager::IsCUDAPointer(const void* ptr) {
    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, ptr);
    if (attributes.devicePointer != nullptr) {
        return true;
    }
    return false;
}

}  // namespace open3d