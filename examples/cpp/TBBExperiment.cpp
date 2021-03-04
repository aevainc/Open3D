#include <iostream>

#include "tbb/blocked_range.h"
#include "tbb/parallel_reduce.h"

using namespace tbb;

struct Sum {
    float value;
    Sum() : value(0) {}
    Sum(Sum& s, split) { value = 0; }
    void operator()(const blocked_range<float*>& r) {
        float temp = value;
        for (float* a = r.begin(); a != r.end(); ++a) {
            temp += *a;
        }
        value = temp;
    }
    void join(Sum& rhs) { value += rhs.value; }
};

float ParallelSum(float array[], size_t n) {
    Sum total;
    parallel_reduce(blocked_range<float*>(array, array + n), total);
    return total.value;
}

int main() {
    float array[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::cout << ParallelSum(array, 10) << std::endl;

    return 0;
}
