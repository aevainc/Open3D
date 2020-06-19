#include "Image.h"
#include "Open3D/Open3D.h"

using namespace open3d;

int main() {
    Device device("CUDA:0");

    void* ptr = MemoryManager::Malloc(16, device);
    MemoryManager::Free(ptr, device);

    void* ptr1 = MemoryManager::Malloc(8, device);
    void* ptr2 = MemoryManager::Malloc(8, device);

    MemoryManager::Free(ptr1, device);
    MemoryManager::Free(ptr2, device);
}
