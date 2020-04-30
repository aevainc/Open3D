#include <stdio.h>
#include <thrust/device_vector.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>
#include "Hashmap.h"
#include "Open3D/Utility/Timer.h"

using namespace open3d;
hash_t dummy_hash_fn;

template <typename Key, typename Value, typename HashFunc = std::hash<Key>>
bool Compare(iterator_t* ret_iterators,
             uint8_t* ret_masks,
             int num_queries,
             std::vector<Key>& query_keys,
             std::unordered_map<Key, Value, HashFunc>& hashmap_gt) {
    auto iterators = thrust::device_vector<iterator_t>(
            ret_iterators, ret_iterators + num_queries);
    auto masks =
            thrust::device_vector<uint8_t>(ret_masks, ret_masks + num_queries);

    for (int i = 0; i < num_queries; ++i) {
        auto iterator_gt = hashmap_gt.find(query_keys[i]);
        if (iterator_gt == hashmap_gt.end()) {  /// Not found
            if (masks[i] != 0) {
                utility::LogError(
                        "keys[{}] should be not found in cuda hashmap.");
            }
        } else {  /// Found
            iterator_t iterator = iterators[i];
            Key key = *(
                    thrust::device_ptr<Key>(reinterpret_cast<Key*>(iterator)));
            Value val = *(thrust::device_ptr<Value>(
                    reinterpret_cast<Value*>(iterator + sizeof(Key))));
            if (!(key == iterator_gt->first)) {
                utility::LogError("key[{}] is not equal to ground truth", i);
            }
            if (!(val == iterator_gt->second)) {
                utility::LogError("value[{}] is not equal to ground truth", i);
            }
        }
    }

    return true;
}

void TEST_SIMPLE() {
    /// C++ ground truth generation
    const int max_keys = 10;
    std::unordered_map<int, int> hashmap_gt;
    std::vector<int> insert_keys = {1, 3, 5};
    std::vector<int> insert_vals = {100, 300, 500};
    for (int i = 0; i < insert_keys.size(); ++i) {
        hashmap_gt[insert_keys[i]] = insert_vals[i];
    }
    std::vector<int> query_keys = {1, 2, 3, 4, 5};

    /// CUDA data generation: use thrust for simplicity
    thrust::device_vector<int> insert_keys_cuda = insert_keys;
    thrust::device_vector<int> insert_vals_cuda = insert_vals;
    thrust::device_vector<int> query_keys_cuda = query_keys;

    /// CUDA data ptr conversion
    uint8_t* insert_keys_ptr_cuda = reinterpret_cast<uint8_t*>(
            thrust::raw_pointer_cast(insert_keys_cuda.data()));
    uint8_t* insert_vals_ptr_cuda = reinterpret_cast<uint8_t*>(
            thrust::raw_pointer_cast(insert_vals_cuda.data()));
    uint8_t* query_keys_ptr_cuda = reinterpret_cast<uint8_t*>(
            thrust::raw_pointer_cast(query_keys_cuda.data()));

    /// Hashmap creation
    auto hashmap =
            CreateHashmap<DefaultHash>(max_keys, sizeof(int), sizeof(int),
                                       open3d::Device("CUDA:0"));

    /// Hashmap insertion
    hashmap->Insert(insert_keys_ptr_cuda, insert_vals_ptr_cuda,
                    insert_keys_cuda.size());

    /// Hashmap search
    iterator_t* ret_iterators;
    uint8_t* ret_masks;
    std::tie(ret_iterators, ret_masks) =
            hashmap->Search(query_keys_ptr_cuda, query_keys_cuda.size());

    /// Result parsing
    Compare<int, int>(ret_iterators, ret_masks, query_keys.size(), query_keys,
                      hashmap_gt);
    utility::LogInfo("TEST_SIMPLE() passed");
}

// struct Vector6i {
//     int x[6];

//     Vector6i(){};
//     Vector6i Random_(std::default_random_engine& generator,
//                      std::uniform_int_distribution<int>& dist) {
//         for (int i = 0; i < 6; ++i) {
//             x[i] = dist(generator);
//         }
//         return *this;
//     }

//     bool operator==(const Vector6i& other) const {
//         bool res = true;
//         for (int i = 0; i < 6; ++i) {
//             res = res && (other.x[i] == x[i]);
//         }
//         return res;
//     }
// };

// namespace std {
// template <>
// struct hash<Vector6i> {
//     std::size_t operator()(const Vector6i& k) const {
//         uint64_t h = UINT64_C(14695981039346656037);
//         for (size_t i = 0; i < 6; ++i) {
//             h ^= k.x[i];
//             h *= UINT64_C(1099511628211);
//         }
//         return h;
//     }
// };
// }  // namespace std

// void TEST_VECTOR6I_INT(int key_size, bool cmp = true) {
//     std::default_random_engine generator;
//     std::uniform_int_distribution<int> dist(-100, 100);

//     /// C++ ground truth generation
//     std::vector<Vector6i> insert_keys(key_size);
//     std::vector<int> insert_vals(key_size);
//     for (int i = 0; i < key_size; ++i) {
//         insert_keys[i].Random_(generator, dist);
//         insert_vals[i] = i;
//     }
//     std::unordered_map<Vector6i, int> hashmap_gt;
//     for (int i = 0; i < key_size; ++i) {
//         hashmap_gt[insert_keys[i]] = insert_vals[i];
//     }

//     std::vector<Vector6i> query_keys(insert_keys.size());
//     for (int i = 0; i < key_size; ++i) {
//         if (i % 3 == 2) {
//             query_keys[i] = insert_keys[i];
//         } else {
//             query_keys[i] = Vector6i().Random_(generator, dist);
//         }
//     }

//     /// CUDA data generation: use thrust for simplicity
//     thrust::device_vector<Vector6i> insert_keys_cuda = insert_keys;
//     thrust::device_vector<int> insert_vals_cuda = insert_vals;
//     thrust::device_vector<Vector6i> query_keys_cuda = query_keys;

//     /// CUDA data ptr conversion
//     uint8_t* insert_keys_ptr_cuda = reinterpret_cast<uint8_t*>(
//             thrust::raw_pointer_cast(insert_keys_cuda.data()));
//     uint8_t* insert_vals_ptr_cuda = reinterpret_cast<uint8_t*>(
//             thrust::raw_pointer_cast(insert_vals_cuda.data()));
//     uint8_t* query_keys_ptr_cuda = reinterpret_cast<uint8_t*>(
//             thrust::raw_pointer_cast(query_keys_cuda.data()));

//     /// Hashmap creation
//     auto hashmap =
//             open3d::CreateHashmap(key_size, sizeof(Vector6i), sizeof(int),
//                                   open3d::Device("CUDA:0"), dummy_hash_fn);

//     utility::Timer timer;

//     /// Hashmap insertion
//     timer.Start();
//     hashmap->Insert(insert_keys_ptr_cuda, insert_vals_ptr_cuda,
//                     insert_keys_cuda.size());
//     timer.Stop();
//     auto duration = timer.GetDuration();
//     utility::LogInfo("{} pairs inserted in {} ms, avg {} insertions/s",
//                      key_size, duration, 1000 * key_size / duration);

//     /// Hashmap search
//     iterator_t* ret_iterators;
//     uint8_t* ret_masks;
//     timer.Start();
//     std::tie(ret_iterators, ret_masks) =
//             hashmap->Search(query_keys_ptr_cuda, query_keys_cuda.size());
//     timer.Stop();
//     duration = timer.GetDuration();
//     utility::LogInfo("{} pairs searched in {} ms, avg {} queries/s",
//     key_size,
//                      duration, 1000 * key_size / duration);

//     /// Result parsing
//     if (cmp) {
//         Compare<Vector6i, int>(ret_iterators, ret_masks, query_keys.size(),
//                                query_keys, hashmap_gt);
//     }
//     utility::LogInfo("TEST_VECTOR6I_INT() passed");
// }

// template <typename T, size_t D>
// struct Coordinate {
// private:
//     T data_[D];

// public:
//     T& operator[](size_t i) { return data_[i]; }
//     const T& operator[](size_t i) const { return data_[i]; }

//     bool operator==(const Coordinate<T, D>& rhs) const {
//         bool equal = true;
//         for (size_t i = 0; i < D; ++i) {
//             equal = equal && (data_[i] == rhs[i]);
//         }
//         return equal;
//     }

//     static Coordinate<T, D> random(std::default_random_engine generator,
//                                    std::uniform_int_distribution<int> dist) {
//         Coordinate<T, D> res;
//         for (size_t i = 0; i < D; ++i) {
//             res.data_[i] = dist(generator);
//         }
//         return res;
//     }
// };

// template <typename T, size_t D>
// struct CoordinateHashFunc {
//     __device__ __host__ uint64_t operator()(const Coordinate<T, D>& key)
//     const {
//         uint64_t hash = UINT64_C(14695981039346656037);

//         /** We only support 4-byte and 8-byte types **/
//         using input_t = typename std::conditional<sizeof(T) ==
//         sizeof(uint32_t),
//                                                   uint32_t, uint64_t>::type;
//         for (size_t i = 0; i < D; ++i) {
//             hash ^= *((input_t*)(&key[i]));
//             hash *= UINT64_C(1099511628211);
//         }
//         return hash;
//     }
// };

// void TEST_COORD_INT(int key_size, bool cmp = true) {
//     const int D = 3;
//     std::default_random_engine generator;
//     std::uniform_int_distribution<int> dist(-1000, 1000);

//     std::vector<int> input_coords(key_size * D);
//     for (int i = 0; i < key_size * D; ++i) {
//         input_coords[i] = dist(generator);
//     }
//     std::vector<Coordinate<int, D>> insert_keys(key_size);
//     std::memcpy(insert_keys.data(), input_coords.data(),
//                 sizeof(int) * key_size * D);
//     std::vector<int> insert_vals(key_size);
//     std::iota(insert_vals.begin(), insert_vals.end(), 0);

//     std::unordered_map<Coordinate<int, D>, int, CoordinateHashFunc<int, D>>
//             hashmap_gt;
//     for (int i = 0; i < key_size; ++i) {
//         hashmap_gt[insert_keys[i]] = insert_vals[i];
//     }

//     std::vector<Coordinate<int, D>> query_keys(insert_keys.size());
//     for (int i = 0; i < key_size; ++i) {
//         if (i % 3 != 2) {  // 2/3 is valid
//             query_keys[i] = insert_keys[i];
//         } else {  // 1/3 is invalid
//             query_keys[i] = Coordinate<int, D>::random(generator, dist);
//         }
//     }

//     thrust::device_vector<Coordinate<int, D>> insert_keys_cuda = insert_keys;
//     thrust::device_vector<int> insert_vals_cuda = insert_vals;
//     thrust::device_vector<Coordinate<int, D>> query_keys_cuda = query_keys;

//     /// CUDA data ptr conversion
//     uint8_t* insert_keys_ptr_cuda = reinterpret_cast<uint8_t*>(
//             thrust::raw_pointer_cast(insert_keys_cuda.data()));
//     uint8_t* insert_vals_ptr_cuda = reinterpret_cast<uint8_t*>(
//             thrust::raw_pointer_cast(insert_vals_cuda.data()));
//     uint8_t* query_keys_ptr_cuda = reinterpret_cast<uint8_t*>(
//             thrust::raw_pointer_cast(query_keys_cuda.data()));

//     /// Hashmap creation
//     auto hashmap = open3d::CreateHashmap(key_size, sizeof(Coordinate<int,
//     D>),
//                                          sizeof(int),
//                                          open3d::Device("CUDA:0"),
//                                          dummy_hash_fn);
//     utility::Timer timer;

//     /// Hashmap insertion
//     timer.Start();
//     hashmap->Insert(insert_keys_ptr_cuda, insert_vals_ptr_cuda,
//                     insert_keys_cuda.size());
//     timer.Stop();
//     auto duration = timer.GetDuration();
//     utility::LogInfo("{} pairs inserted in {} ms, avg {} insertions/s",
//                      key_size, duration, 1000 * key_size / duration);

//     /// Hashmap search
//     iterator_t* ret_iterators;
//     uint8_t* ret_masks;
//     timer.Start();
//     std::tie(ret_iterators, ret_masks) =
//             hashmap->Search(query_keys_ptr_cuda, query_keys_cuda.size());
//     timer.Stop();
//     duration = timer.GetDuration();
//     utility::LogInfo("{} pairs searched in {} ms, avg {} queries/s",
//     key_size,
//                      duration, 1000 * key_size / duration);

//     /// Result parsing
//     if (cmp) {
//         Compare<Coordinate<int, D>, int, CoordinateHashFunc<int, D>>(
//                 ret_iterators, ret_masks, query_keys.size(), query_keys,
//                 hashmap_gt);
//     }
//     utility::LogInfo("TEST_COORD8D_INT() passed");
// }

int main() {
    TEST_SIMPLE();
    // const int max_kvpairs = 10000000;
    // for (int i = 1000; i <= max_kvpairs; i *= 10) {
    //     TEST_VECTOR6I_INT(i, false);
    // }

    // for (int i = 1000; i <= max_kvpairs; i *= 10) {
    //     TEST_COORD_INT(i, false);
    // }
}