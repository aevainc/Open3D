#include <CL/sycl.hpp>
#include <array>
#include <chrono>
#include <iostream>

int main() {
    std::cout << "Hello RunSYCLDemo()" << std::endl;

    constexpr int size = 8;
    std::array<int, size> data;
    sycl::queue Q;
    sycl::buffer B{data};
    Q.submit([&](sycl::handler &h) {
        sycl::accessor A{B, h};
        h.parallel_for(size, [=](auto &idx) { A[idx] = idx; });
    });
    sycl::host_accessor A{B};
    for (int i = 0; i < size; i++) {
        std::cout << "data[" << i << "] = " << A[i] << "\n";
    }
}
