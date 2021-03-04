#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

using namespace tbb;

const int rows = 1000000;
const int cols = 10;

// Used as identity element.
std::vector<double> zeros(cols, 0.0);

int main(int argc, char **argv) {
    auto values = std::vector<std::vector<double>>(
            rows, std::vector<double>(cols, 0.0));
    tbb::parallel_for(tbb::blocked_range<int>(0, values.size()),
                      [&](tbb::blocked_range<int> r) {
                          for (int i = r.begin(); i < r.end(); ++i) {
                              for (int j = 0; j < cols; j++) {
                                  // Fill `values`.
                                  // This will be replaced by the processing
                                  // part, before reduction.
                                  values[i][j] = 1.0;
                              }
                          }
                      });

    std::vector<double> total = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, values.size()), zeros,
            [&](tbb::blocked_range<int> r, std::vector<double> running_total) {
                for (int i = r.begin(); i < r.end(); i++) {
                    for (int j = 0; j < cols; j++) {
                        running_total[j] += values[i][j];
                    }
                }
                return running_total;
            },
            [&](std::vector<double> a, std::vector<double> b) {
                std::vector<double> result(cols);
                for (int j = 0; j < cols; j++) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });

    for (double value : total) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}
