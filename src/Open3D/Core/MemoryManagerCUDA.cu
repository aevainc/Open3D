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
        BlockPtr query_block = new Block(device.GetID(), byte_size);

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
        delete query_block;

        if (found_block == nullptr) {
            // Allocate and insert to the allocated pool
            OPEN3D_CUDA_CHECK(cudaMalloc(static_cast<void**>(&ptr), byte_size));

            BlockPtr new_block = new Block(device.GetID(), byte_size, ptr);
            new_block->in_use_ = true;
            allocated_blocks_.insert({ptr, new_block});
        } else {
            // Get raw ptr for return
            ptr = found_block->ptr_;

            // Split the found block and put the remains to cache
            size_t remain_byte_size = found_block->size_ - byte_size;
            if (remain_byte_size > 0) {
                found_block->size_ = byte_size;

                // found_block <-> remain_block <-> found_block->next_
                BlockPtr next_block = found_block->next_;
                BlockPtr remain_block =
                        new Block(device.GetID(), remain_byte_size,
                                  static_cast<char*>(ptr) + byte_size,
                                  found_block, next_block);

                found_block->next_ = remain_block;
                if (next_block) {
                    next_block->prev_ = remain_block;
                }

                // Place the remain block to pool
                get_pool(remain_byte_size)->emplace(remain_block);
                // utility::LogInfo("Splitted: {}--{} == > {}--{}, {}--{} -->{}
                // ",
                //                  fmt::ptr(found_block), found_block->size_,
                //                  fmt::ptr(found_block), byte_size,
                //                  fmt::ptr(remain_block), remain_byte_size,
                //                  fmt::ptr(remain_block->next_));
            }
            found_block->in_use_ = true;
            allocated_blocks_.insert({ptr, found_block});
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
                        "CUDAMemoryManager::Free: Memory leak! Block "
                        "should "
                        "have been stored.");
            } else {
                // Release memory and check if merge is required
                BlockPtr block = it->second;
                allocated_blocks_.erase(it);
                block->in_use_ = false;

                auto pool = get_pool(block->size_);
                pool->emplace(block);

                // Merge towards next
                // TODO: wrap it with a function
                // BlockPtr block = head_block;
                // while (block != nullptr && block->next_ != nullptr) {
                //     BlockPtr next_block = block->next_;
                //     if (next_block->prev_ != block) {
                //         // Double check; should never reach here.
                //         utility::LogError(
                //                 "CUDAMemoryManager::Free: linked list
                //                 nodes " "mismatch in next-direction
                //                 merge.");
                //     }

                //     if (next_block->in_use_) {
                //         break;
                //     }

                //     // Merge
                //     block->next_ = next_block->next_;
                //     block->size_ += next_block->size_;

                //     // Remove next_block from the pool
                //     auto next_block_pool = get_pool(next_block->size_);
                //     auto it = next_block_pool->find(next_block);
                //     if (it == next_block_pool->end()) {
                //         // Should never reach here
                //         utility::LogError(
                //                 "CUDAMemoryManager::Free: linked list
                //                 node not " "found in pool.");
                //     }
                //     next_block_pool->erase(it);
                //     delete next_block;

                //     block = block->next_;

                //     utility::LogInfo("Merging in the next-direction.");
                // }

                // Merge towards prev
                // TODO: wrap it with a function
                // block = head_block;
                // while (block != nullptr && block->prev_ != nullptr) {
                //     BlockPtr prev_block = block->prev_;
                //     if (prev_block->next_ != block) {
                //         // Double check; should never reach here.
                //         utility::LogError(
                //                 "CUDAMemoryManager::Free: linked list "
                //                 "nodes "
                //                 "mismatch in prev-direction merge: {} vs "
                //                 "{}.",
                //                 fmt::ptr(prev_block->next_),
                //                 fmt::ptr(block));
                //     }

                // if (prev_block->in_use_) {
                //     break;
                // }

                // // Merge
                // block->prev_ = prev_block->prev_;
                // block->size_ += prev_block->size_;
                // block->ptr_ = prev_block->ptr_;

                // // Remove next_block from the pool
                // auto prev_block_pool = get_pool(prev_block->size_);
                // auto it = prev_block_pool->find(prev_block);
                // if (it == prev_block_pool->end()) {
                //     // Should never reach here
                //     utility::LogError(
                //             "CUDAMemoryManager::Free: linked list
                //             node not " "found in pool.");
                // }
                // prev_block_pool->erase(it);
                // delete prev_block;

                //     block = block->prev_;
                //     utility::LogInfo("Checking in the prev-direction.");
                // }
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
