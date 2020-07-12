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

#include "open3d/core/MemoryManager.h"
#include "open3d/core/CUDAState.cuh"
#include "open3d/core/CUDAUtils.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <set>
#include <string>
#include <unordered_map>

namespace open3d {
namespace core {

// Refrence
// https://github.com/pytorch/pytorch/blob/5fbec1f55df59156edff4084023086823227fbb0/c10/cuda/CUDACachingAllocator.cpp
struct Block;

// We need raw pointers (instead of smart ptrs) for exact comparison and
// reference
typedef Block* BlockPtr;

struct Block {
    int device_;   // gpu id
    size_t size_;  // block size in bytes
    void* ptr_;    // memory address

    BlockPtr prev_;
    BlockPtr next_;

    bool in_use_;

    Block(int device,
          size_t size,
          void* ptr = nullptr,
          BlockPtr prev = nullptr,
          BlockPtr next = nullptr)
        : device_(device),
          size_(size),
          ptr_(ptr),
          prev_(prev),
          next_(next),
          in_use_(false) {}
};

struct BlockComparator {
    bool operator()(const BlockPtr& a, const BlockPtr& b) const {
        // Not on the same device: treat as smaller, will be filtered in
        // lower_bound operation.
        if (a->device_ != b->device_) {
            return true;
        }
        if (a->size_ != b->size_) {
            return a->size_ < b->size_;
        }
        return (size_t)a->ptr_ < (size_t)b->ptr_;
    }
};

/// Singleton
class CUDACacher {
public:
    static std::shared_ptr<CUDACacher> GetInstance() {
        if (instance_ == nullptr) {
            utility::LogDebug("CUDACacher Instance created.");
            instance_ = std::make_shared<CUDACacher>();
        }
        return instance_;
    }

public:
    typedef std::set<BlockPtr, BlockComparator> BlockPool;

    inline std::shared_ptr<BlockPool>& get_pool(size_t byte_size) {
        // largest "small" allocation is 1 MiB (1024 * 1024)
        constexpr size_t kSmallSize = 1048576;
        return byte_size <= kSmallSize ? small_block_pool_ : large_block_pool_;
    }

    inline size_t align_bytes(size_t byte_size, size_t alignment = 8) {
        return ((byte_size + alignment - 1) / alignment) * alignment;
    }

    CUDACacher() {
        small_block_pool_ = std::make_shared<BlockPool>();
        large_block_pool_ = std::make_shared<BlockPool>();
    }

    ~CUDACacher() {
        if (!allocated_blocks_.empty()) {
            // Should never reach here
            utility::LogError("[CUDACacher] Memory leak!");
        }
        ReleaseCache();
    }

    void* Malloc(size_t byte_size, const Device& device) {
        auto find_free_block = [&](BlockPtr query_block) -> BlockPtr {
            auto pool = get_pool(query_block->size_);
            auto it = pool->lower_bound(query_block);
            if (it != pool->end()) {
                BlockPtr block = *it;
                pool->erase(it);
                return block;
            }
            return nullptr;
        };

        void* ptr;
        size_t alloc_size = align_bytes(byte_size);
        Block query_block = Block(device.GetID(), alloc_size);
        BlockPtr found_block = find_free_block(&query_block);

        if (found_block == nullptr) {
            // Allocate a new block and insert it to the allocated pool
            OPEN3D_CUDA_CHECK(cudaMalloc(&ptr, alloc_size));
            BlockPtr new_block = new Block(device.GetID(), alloc_size, ptr);
            new_block->in_use_ = true;
            allocated_blocks_.insert({ptr, new_block});
        } else {
            utility::LogDebug("[CUDACacher] Reusing memory");
            ptr = found_block->ptr_;

            size_t remain_size = found_block->size_ - alloc_size;
            if (remain_size > 0) {
                // Split block
                // found_block <-> remain_block <-> found_block->next_
                BlockPtr next_block = found_block->next_;
                BlockPtr remain_block =
                        new Block(device.GetID(), remain_size,
                                  static_cast<char*>(ptr) + alloc_size,
                                  found_block, next_block);
                found_block->next_ = remain_block;
                if (next_block) {
                    next_block->prev_ = remain_block;
                }

                // Place the remain block to cache pool
                get_pool(remain_size)->emplace(remain_block);
                utility::LogDebug(
                        "[CUDACacher] Splitted: {}--{} == > {}--{}, {}--{} "
                        "-->{}",
                        fmt::ptr(found_block), found_block->size_,
                        fmt::ptr(found_block), alloc_size,
                        fmt::ptr(remain_block), remain_size,
                        fmt::ptr(remain_block->next_));
            }

            found_block->size_ = alloc_size;
            found_block->in_use_ = true;
            allocated_blocks_.insert({ptr, found_block});
        }

        return ptr;
    }

    void Free(void* ptr, const Device& device) {
        auto release_block = [&](BlockPtr block) {
            auto block_pool = get_pool(block->size_);
            auto it = block_pool->find(block);
            if (it == block_pool->end()) {
                // Should never reach here
                utility::LogError(
                        "[CUDACacher]: linked list node {} not found in pool.",
                        fmt::ptr(block));
            }
            block_pool->erase(it);
            delete block;
        };

        auto it = allocated_blocks_.find(ptr);

        if (it == allocated_blocks_.end()) {
            // Should never reach here!
            utility::LogError(
                    "[CUDACacher] Memory leak! Block should have been "
                    "recorded.");
        } else {
            // Release memory and check if merge is required
            BlockPtr block = it->second;
            allocated_blocks_.erase(it);

            // Merge free blocks towards 'next' direction
            BlockPtr block_it = block;
            while (block_it != nullptr && block_it->next_ != nullptr) {
                BlockPtr next_block = block_it->next_;
                if (next_block->prev_ != block_it) {
                    // Should never reach here.
                    utility::LogError(
                            "[CUDACacher]: linked list nodes mismatch in "
                            "forward merge.");
                }

                if (next_block->in_use_) {
                    break;
                }

                // Merge
                block_it->next_ = next_block->next_;
                if (block_it->next_) {
                    block_it->next_->prev_ = block_it;
                }
                block_it->size_ += next_block->size_;
                release_block(next_block);

                block_it = block_it->next_;

                utility::LogDebug(
                        "[CUDACacher] Merging in the forward direction.");
            }

            // Merge free blocks towards 'prev' direction
            block_it = block;
            while (block_it != nullptr && block_it->prev_ != nullptr) {
                BlockPtr prev_block = block_it->prev_;
                if (prev_block->next_ != block_it) {
                    // Double check; should never reach here.
                    utility::LogError(
                            "[CUDACacher]: linked list nodes mismatch in "
                            "prev-direction merge: {} vs {}.",
                            fmt::ptr(prev_block->next_), fmt::ptr(block_it));
                }

                if (prev_block->in_use_) {
                    break;
                }

                // Merge
                block_it->prev_ = prev_block->prev_;
                if (block_it->prev_) {
                    block_it->prev_->next_ = block_it;
                }
                block_it->size_ += prev_block->size_;
                block_it->ptr_ = prev_block->ptr_;
                release_block(prev_block);

                block_it = block_it->prev_;

                utility::LogDebug(
                        "[CUDACacher] Merging in the backward direction.");
            }

            block->in_use_ = false;
            get_pool(block->size_)->emplace(block);
        }
    }

    void ReleaseCache() {
        // remove_if does not work for set
        // https://stackoverflow.com/questions/24263259/c-stdseterase-with-stdremove-if
        // https://stackoverflow.com/questions/2874441/deleting-elements-from-stdset-while-iterating
        auto release_pool = [](std::set<BlockPtr, BlockComparator>& pool) {
            auto it = pool.begin();
            auto end = pool.end();
            while (it != end) {
                BlockPtr block = *it;
                if (block->prev_ == nullptr && block->next_ == nullptr) {
                    OPEN3D_CUDA_CHECK(cudaFree(block->ptr_));
                    delete block;
                    it = pool.erase(it);
                } else {
                    ++it;
                }
            }
        };

        release_pool(*small_block_pool_);
        release_pool(*large_block_pool_);
    }

private:
    std::unordered_map<void*, BlockPtr> allocated_blocks_;
    std::shared_ptr<BlockPool> small_block_pool_;
    std::shared_ptr<BlockPool> large_block_pool_;

    static std::shared_ptr<CUDACacher> instance_;
};

// Create instance on intialization to avoid 'cuda error driver shutdown'
std::shared_ptr<CUDACacher> CUDACacher::instance_ = CUDACacher::GetInstance();

void* CUDAMemoryManager::Malloc(size_t byte_size, const Device& device) {
    CUDADeviceSwitcher switcher(device);

    if (device.GetType() == Device::DeviceType::CUDA) {
        std::shared_ptr<CUDACacher> instance = CUDACacher::GetInstance();
        return instance->Malloc(byte_size, device);
    } else {
        utility::LogError("[CUDAMemoryManager] Malloc: Unimplemented device");
        return nullptr;
    }
    // Should never reach here
    return nullptr;
}

void CUDAMemoryManager::Free(void* ptr, const Device& device) {
    CUDADeviceSwitcher switcher(device);

    if (device.GetType() == Device::DeviceType::CUDA) {
        if (ptr && IsCUDAPointer(ptr)) {
            std::shared_ptr<CUDACacher> instance = CUDACacher::GetInstance();
            instance->Free(ptr, device);
        } else {
            utility::LogError("[CUDAMemoryManager] Free: Invalid pointer");
        }
    } else {
        utility::LogError("[CUDAMemoryManager] Free: Unimplemented device");
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


void CUDAMemoryManager::ReleaseCache() {
    utility::LogDebug("Releasing Cache");
    std::shared_ptr<CUDACacher> instance = CUDACacher::GetInstance();
    instance->ReleaseCache();
}

}  // namespace core
}  // namespace open3d
