#include "Image.h"
#include "Open3D/Core/CUDAUtils.h"
#include "Open3D/Open3D.h"

using namespace open3d;

int main() {
    Device device("CUDA:0");

    void* ptr = MemoryManager::Malloc(32, device);
    MemoryManager::Free(ptr, device);

    void* ptr1 = MemoryManager::Malloc(16, device);
    void* ptr2 = MemoryManager::Malloc(8, device);
    void* ptr3 = MemoryManager::Malloc(4, device);
    void* ptr4 = MemoryManager::Malloc(2, device);

    MemoryManager::Free(ptr2, device);
    void* ptr5 = MemoryManager::Malloc(4, device);

    MemoryManager::Free(ptr4, device);
    MemoryManager::Free(ptr3, device);
    MemoryManager::Free(ptr1, device);
    MemoryManager::Free(ptr5, device);
}
