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

namespace open3d {

typedef uint32_t ptr_t;
typedef uint8_t* iterator_t;

typedef uint64_t (*hash_t)(uint8_t*, uint32_t);

/// Internal Hashtable Node: (31 units and 1 next ptr) representation.
/// \member kv_pair_ptrs:
/// Each element is an internal ptr to a kv pair managed by the
/// InternalMemoryManager. Can be converted to a real ptr.
/// \member next_slab_ptr:
/// An internal ptr managed by InternalNodeManager.
class Slab {
public:
    ptr_t kv_pair_ptrs[31];
    ptr_t next_slab_ptr;
};

template <typename Key, typename Value>
struct Pair {
    Key first;
    Value second;
    __device__ __host__ Pair() {}
    __device__ __host__ Pair(const Key& key, const Value& value)
        : first(key), second(value) {}
};

template <typename Key, typename Value>
__device__ __host__ Pair<Key, Value> make_pair(const Key& key,
                                               const Value& value) {
    return Pair<Key, Value>(key, value);
}

}  // namespace open3d
