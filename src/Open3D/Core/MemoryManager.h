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

#include <cstring>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "Open3D/Core/Device.h"

namespace open3d {

class DeviceMemoryManager;

class MemoryManager {
public:
    static void* Malloc(size_t byte_size, const Device& device);
    static void Free(void* ptr, const Device& device);
    static void Memcpy(void* dst_ptr,
                       const Device& dst_device,
                       const void* src_ptr,
                       const Device& src_device,
                       size_t num_bytes);
    /// Same as Memcpy, but with host (CPU:0) as default src_device
    static void MemcpyFromHost(void* dst_ptr,
                               const Device& dst_device,
                               const void* host_ptr,
                               size_t num_bytes);
    /// Same as Memcpy, but with host (CPU:0) as default dst_device
    static void MemcpyToHost(void* host_ptr,
                             const void* src_ptr,
                             const Device& src_device,
                             size_t num_bytes);

protected:
    static std::shared_ptr<DeviceMemoryManager> GetDeviceMemoryManager(
            const Device& device);
};

class DeviceMemoryManager {
public:
    virtual void* Malloc(size_t byte_size, const Device& device) = 0;
    virtual void Free(void* ptr, const Device& device) = 0;
    virtual void Memcpy(void* dst_ptr,
                        const Device& dst_device,
                        const void* src_ptr,
                        const Device& src_device,
                        size_t num_bytes) = 0;
};

class CPUMemoryManager : public DeviceMemoryManager {
public:
    CPUMemoryManager();
    void* Malloc(size_t byte_size, const Device& device) override;
    void Free(void* ptr, const Device& device) override;
    void Memcpy(void* dst_ptr,
                const Device& dst_device,
                const void* src_ptr,
                const Device& src_device,
                size_t num_bytes) override;
};

#ifdef BUILD_CUDA_MODULE
// Refrence
// https://github.com/pytorch/pytorch/blob/5fbec1f55df59156edff4084023086823227fbb0/c10/cuda/CUDACachingAllocator.cpp
struct Block;
typedef std::shared_ptr<Block> BlockPtr;
typedef bool (*Comparison)(const BlockPtr&, const BlockPtr&);
typedef std::set<BlockPtr, Comparison> BlockPool;

struct Block {
    int device_;   // gpu
    size_t size_;  // block size in bytes
    void* ptr_;    // memory address

    Block(int device, size_t size, void* ptr = nullptr)
        : device_(device), size_(size), ptr_(ptr) {}
};

class CUDAMemoryManager : public DeviceMemoryManager {
public:
    CUDAMemoryManager();
    void* Malloc(size_t byte_size, const Device& device) override;
    void Free(void* ptr, const Device& device) override;
    void Memcpy(void* dst_ptr,
                const Device& dst_device,
                const void* src_ptr,
                const Device& src_device,
                size_t num_bytes) override;

protected:
    bool IsCUDAPointer(const void* ptr);

    inline std::shared_ptr<BlockPool> get_pool(size_t byte_size) {
        // largest "small" allocation is 1 MiB (1024 * 1024)
        constexpr size_t kSmallSize = 1048576;
        return byte_size <= kSmallSize ? small_block_pool_ : large_block_pool_;
    }

    inline size_t align_size(size_t byte_size, size_t alignment = 4) {
        return ((byte_size + alignment - 1) / alignment) * alignment;
    }

    std::unordered_map<void*, BlockPtr> allocated_blocks_;
    std::shared_ptr<BlockPool> small_block_pool_;
    std::shared_ptr<BlockPool> large_block_pool_;
};
#endif

}  // namespace open3d
