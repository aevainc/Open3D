#include <stdint.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_map>

struct HashFunc {
    HashFunc(size_t key_size) : key_size_(key_size) {}

    size_t operator()(void* key_ptr) const {
        uint64_t hash = UINT64_C(14695981039346656037);

        const int chunks = key_size_ / sizeof(int);
        int32_t* cast_key_ptr = (int32_t*)(key_ptr);
        for (size_t i = 0; i < chunks; ++i) {
            hash ^= cast_key_ptr[i];
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    }
    size_t key_size_;
};

struct KeyEqual {
    KeyEqual(size_t key_size) : key_size_(key_size) {}
    bool operator()(const void* lhs, const void* rhs) const {
        const int chunks = key_size_ / sizeof(int);
        int* lhs_key_ptr = (int*)(lhs);
        int* rhs_key_ptr = (int*)(rhs);

        bool res = true;
        for (size_t i = 0; i < chunks; ++i) {
            res = res && (lhs_key_ptr[i] == rhs_key_ptr[i]);
        }
        return res;
    }
    size_t key_size_;
};

int main() {
    const int num_keys = 4;

    /// Key of size int
    void* keys = std::malloc(num_keys * sizeof(int));
    /// Values of type float
    void* values = std::malloc(num_keys * sizeof(float));

    /// Initialize kv pairs
    std::cout << "Initialize\n";
    for (int i = 0; i < num_keys; ++i) {
        static_cast<int*>(keys)[i] = i;
        static_cast<float*>(values)[i] = i * 3.14;
    }
    auto hashmap = std::unordered_map<void*, void*, HashFunc, KeyEqual>(
            10, HashFunc(sizeof(int)), KeyEqual(sizeof(int)));

    std::cout << "Insert\n";
    for (int i = 0; i < num_keys; ++i) {
        void* src_key = (unsigned char*)keys + sizeof(int) * i;
        void* src_value = (unsigned char*)values + sizeof(float) * i;

        /// Copy before insert
        void* dst_key = std::malloc(sizeof(int));
        void* dst_value = std::malloc(sizeof(float));
        std::memcpy(dst_key, src_key, sizeof(int));
        std::memcpy(dst_value, src_value, sizeof(float));

        hashmap.insert({dst_key, dst_value});
    }

    /// View results
    std::cout << "Query\n";
    for (int i = 0; i < num_keys; ++i) {
        void* key = (unsigned char*)keys + sizeof(int) * i;
        void* gt_value = (unsigned char*)values + sizeof(float) * i;
        auto iter = hashmap.find(key);
        if (iter == hashmap.end()) {
            std::cout << "Error!\n";
        } else {
            void* key = iter->first;
            void* value = iter->second;
            std::cout << *(int*)(key) << " " << *(float*)(value) << "\n";
        }
    }
}
