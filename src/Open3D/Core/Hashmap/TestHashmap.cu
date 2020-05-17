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

#include "Open3D/Core/Hashmap/Hashmap.h"

using namespace open3d;

/// Compare all the iterators: should be identical
/// Launch after readwrite operations: Insert and Erase
template <typename Key, typename Value, typename Hash, typename Eq>
void CompareQueries(std::shared_ptr<Hashmap<Hash, Eq>> &hashmap,
                    std::unordered_map<Key, Value, Hash, Eq> &hashmap_gt,
                    std::vector<Key> &keys) {}

/// Compare all the iterators: should be identical
/// Launch after readwrite operations: Insert and Erase
template <typename Key, typename Value, typename Hash, typename Eq>
void CompareHashmaps(std::shared_ptr<Hashmap<Hash, Eq>> &hashmap,
                     std::unordered_map<Key, Value> &hashmap_gt) {
    // First get all iterators from hashmap
    // Sanity check: iterator_count == hashmap_gt.size()
    // Then unpack <key, values> and check if they all exist in hashmap_gt

    // For now: at most we have 32 x bucket_size iterators to store
    iterator_t *iterators =
            reinterpret_cast<iterator_t *>(MemoryManager::Malloc(
                    sizeof(iterator_t) * hashmap->bucket_count_ * 32,
                    hashmap->device_));

    size_t count = hashmap->GetIterators(iterators);
    assert(count == hashmap_gt.size());

    auto iterators_vec =
            thrust::device_vector<iterator_t>(iterators, iterators + count);

    for (size_t i = 0; i < count; ++i) {
        iterator_t iterator = iterators_vec[i];

        Key key = *(thrust::device_ptr<Key>(
                reinterpret_cast<Key *>(iterator.first)));
        Value val = *(thrust::device_ptr<Value>(
                reinterpret_cast<Value *>(iterator.second)));

        auto iterator_gt = hashmap_gt.find(key);
        assert(iterator_gt != hashmap_gt.end());
        assert(iterator_gt->first == key);
        assert(iterator_gt->second == val);
    }

    MemoryManager::Free(iterators, hashmap->device_);
}

template <typename Key, typename Value>
void TestInsert(const std::vector<Key> &keys,
                const std::vector<Value> &vals,
                size_t max_buckets) {
    auto hashmap = CreateHashmap<DefaultHash, DefaultKeyEq>(
            max_buckets, sizeof(Key), sizeof(Value), open3d::Device("CUDA:0"));
    auto hashmap_gt = std::unordered_map<Key, Value>();

    for (int i = 0; i < keys.size(); ++i) {
        hashmap_gt[keys[i]] = vals[i];
    }

    thrust::device_vector<Key> keys_cuda = keys;
    thrust::device_vector<Value> vals_cuda = vals;
    thrust::device_vector<iterator_t> iterators_cuda(keys.size());
    thrust::device_vector<uint8_t> masks_cuda(keys.size());

    hashmap->Insert(reinterpret_cast<void *>(
                            thrust::raw_pointer_cast(keys_cuda.data())),
                    reinterpret_cast<void *>(
                            thrust::raw_pointer_cast(vals_cuda.data())),
                    reinterpret_cast<iterator_t *>(
                            thrust::raw_pointer_cast(iterators_cuda.data())),
                    reinterpret_cast<uint8_t *>(
                            thrust::raw_pointer_cast(masks_cuda.data())),
                    keys.size());

    CompareHashmaps<Key, Value, DefaultHash, DefaultKeyEq>(hashmap, hashmap_gt);
    utility::LogInfo("Hashmaps are identical for bucket size {}", max_buckets);
}

int main() {
    for (size_t bucket_size = 1; bucket_size < 10; ++bucket_size) {
        TestInsert<int, int>(std::vector<int>({100, 100, 100, 200, 200, 300}),
                             std::vector<int>({1, 1, 1, 2, 2, 3}), bucket_size);
    }
}