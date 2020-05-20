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

#include <random>
#include "Open3D/Core/Hashmap/Hashmap.h"

using namespace open3d;

template <typename Key, typename Value, typename Hash, typename Eq>
void CompareFind(std::shared_ptr<Hashmap<Hash, Eq>> &hashmap,
                 std::unordered_map<Key, Value> &hashmap_gt,
                 const std::vector<Key> &keys) {
    // Prepare GPU memory
    thrust::device_vector<Key> keys_cuda = keys;
    thrust::device_vector<iterator_t> iterators_cuda(keys.size());
    thrust::device_vector<uint8_t> masks_cuda(keys.size());

    hashmap->Find(reinterpret_cast<void *>(
                          thrust::raw_pointer_cast(keys_cuda.data())),
                  reinterpret_cast<iterator_t *>(
                          thrust::raw_pointer_cast(iterators_cuda.data())),
                  reinterpret_cast<uint8_t *>(
                          thrust::raw_pointer_cast(masks_cuda.data())),
                  keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        auto iterator_gt = hashmap_gt.find(keys[i]);
        // Not found in gt => not found in ours
        if (iterator_gt == hashmap_gt.end()) {
            assert(masks_cuda[i] == 0);
        } else {  /// Found in gt => same key and value
            iterator_t iterator = iterators_cuda[i];
            Key key = *(thrust::device_ptr<Key>(
                    reinterpret_cast<Key *>(iterator.first)));
            Value val = *(thrust::device_ptr<Value>(
                    reinterpret_cast<Value *>(iterator.second)));
            assert(key == iterator_gt->first);
            assert(val == iterator_gt->second);
        }
    }
}

template <typename Key, typename Value, typename Hash, typename Eq>
void CompareInsert(std::shared_ptr<Hashmap<Hash, Eq>> &hashmap,
                   std::unordered_map<Key, Value> &hashmap_gt,
                   const std::vector<Key> &keys,
                   const std::vector<Value> &vals) {
    // Prepare groundtruth
    for (int i = 0; i < keys.size(); ++i) {
        hashmap_gt.insert(std::make_pair(keys[i], vals[i]));
    }

    // Prepare GPU memory
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

    iterator_t *iterators =
            reinterpret_cast<iterator_t *>(MemoryManager::Malloc(
                    sizeof(iterator_t) * hashmap->bucket_count_ * 64,
                    hashmap->device_));
    size_t count = hashmap->GetIterators(iterators);

    size_t insert_count = 0;
    for (int i = 0; i < keys.size(); ++i) {
        insert_count += masks_cuda[i];
    }
    std::cout << "insert_count = " << insert_count << "\n";

    auto bucket_count_total = 0;
    auto bucket_sizes = hashmap->BucketSizes();
    for (size_t i = 0; i < bucket_sizes.size(); ++i) {
        // if (bucket_sizes[i] >= 31) {
        //     utility::LogInfo("bucket {}: {}", i, bucket_sizes[i]);
        // }
        bucket_count_total += bucket_sizes[i];
    }
    std::cout << "bucket_count_total = " << bucket_count_total << "\n";

    // 1. Sanity check: iterator counts should be equal
    std::cout << count << " " << hashmap_gt.size() << "\n";
    assert(count == hashmap_gt.size());
    auto iterators_vec =
            thrust::device_vector<iterator_t>(iterators, iterators + count);

    // 2. Verbose check: every iterator should be observable in gt
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

template <typename Key, typename Value, typename Hash, typename Eq>
void CompareErase(std::shared_ptr<Hashmap<Hash, Eq>> &hashmap,
                  std::unordered_map<Key, Value> &hashmap_gt,
                  const std::vector<Key> &keys) {
    // Prepare groundtruth
    for (int i = 0; i < keys.size(); ++i) {
        hashmap_gt.erase(keys[i]);
    }

    // Prepare GPU memory
    thrust::device_vector<Key> keys_cuda = keys;
    thrust::device_vector<uint8_t> masks_cuda(keys.size());

    hashmap->Erase(reinterpret_cast<void *>(
                           thrust::raw_pointer_cast(keys_cuda.data())),
                   reinterpret_cast<uint8_t *>(
                           thrust::raw_pointer_cast(masks_cuda.data())),
                   keys.size());

    // size_t erase_count = 0;
    // for (int i = 0; i < keys.size(); ++i) {
    //     erase_count += masks_cuda[i];
    // }
    // std::cout << "erase_count = " << erase_count << "\n";

    iterator_t *iterators =
            reinterpret_cast<iterator_t *>(MemoryManager::Malloc(
                    sizeof(iterator_t) * hashmap->bucket_count_ * 64,
                    hashmap->device_));
    size_t count = hashmap->GetIterators(iterators);
    // std::cout << count << "\n";

    // 1. Sanity check: iterator counts should be equal
    assert(count == hashmap_gt.size());
    auto iterators_vec =
            thrust::device_vector<iterator_t>(iterators, iterators + count);

    // 2. Verbose check: every iterator should be observable in gt
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

int main() {
    // std::random_device rnd_device;
    std::mt19937 mersenne_engine{0};

    for (size_t bucket_count = 1000; bucket_count <= 1000000;
         bucket_count *= 10) {
        utility::LogInfo("Test with bucket_count = {}", bucket_count);
        using Key = int;
        using Value = int;

        // Generate data
        std::uniform_int_distribution<int> dist{-(int)bucket_count * 20,
                                                (int)bucket_count * 20};
        std::vector<int> keys(bucket_count * 32);
        std::vector<int> vals(bucket_count * 32);
        std::generate(std::begin(keys), std::end(keys),
                      [&]() { return dist(mersenne_engine); });
        std::sort(keys.begin(), keys.end());
        for (size_t i = 0; i < keys.size(); ++i) {
            // Ensure 1 on 1 mapping to remove hassles in duplicate keys
            vals[i] = keys[i] * 100;
        }

        auto hashmap = CreateHashmap<DefaultHash, DefaultKeyEq>(
                bucket_count, sizeof(Key), sizeof(Value),
                open3d::Device("CUDA:0"));
        auto hashmap_gt = std::unordered_map<Key, Value>();

        CompareInsert(hashmap, hashmap_gt, keys, vals);
        utility::LogInfo("TestInsert passed");

        CompareFind(hashmap, hashmap_gt, keys);
        utility::LogInfo("TestFind passed");

        CompareErase(hashmap, hashmap_gt, keys);
        utility::LogInfo("TestErase passed");
    }
}