#include <iostream>

#include "tbb/blocked_range.h"
#include "tbb/parallel_reduce.h"

using namespace tbb;

class Mat {
public:
    // Data.
    double value[5];

    // Value initialising constructor.
    Mat() {
        for (int i = 0; i < 5; i++) {
            this->value[i] = 0.0;
        }
    }

    Mat(const double init) {
        for (int i = 0; i < 5; i++) {
            this->value[i] = init;
        }
    }

    // Mat (const Mat& init) {
    //     for(int i = 0; i < 5; i++) {
    //         this->value[i] = init.value[i];
    //     }
    // }

    Mat operator+(const Mat& b) const {
        Mat a(0);
        for (int i = 0; i < 5; i++) {
            a.value[i] = this->value[i] + b.value[i];
        }
        return a;
    }

    Mat operator+=(const Mat& b) {
        for (int i = 0; i < 5; i++) {
            this->value[i] += b.value[i];
        }
        return *this;
    }

    Mat operator=(const double b) {
        for (int i = 0; i < 5; i++) {
            this->value[i] = b;
        }
        return *this;
    }

    Mat operator=(const int b) {
        for (int i = 0; i < 5; i++) {
            this->value[i] = b;
        }
        return *this;
    }

    Mat operator=(const Mat& b) {
        for (int i = 0; i < 5; i++) {
            this->value[i] = b.value[i];
        }
        return *this;
    }
};

class Sum {
public:
    Mat value;
    Sum() : value(0) {}
    Sum(Sum& s, split) { value = 0.0; }
    void operator()(const blocked_range<Mat*>& r) {
        Mat temp = value;
        for (Mat* a = r.begin(); a != r.end(); ++a) {
            temp += *a;
        }
        value = temp;
    }
    void join(Sum& rhs) { value += rhs.value; }
};

Mat ParallelSum(Mat array[], size_t n) {
    Sum total;
    parallel_reduce(blocked_range<Mat*>(array, array + n), total);
    return total.value;
}

int main() {
    Mat array[10];
    for (int i = 0; i < 10; i++) {
        array[i] = i;
    }

    Mat ans = ParallelSum(array, 10);
    for (int i = 0; i < 5; i++) {
        std::cout << ans.value[i] << std::endl;
    }

    return 0;
}
