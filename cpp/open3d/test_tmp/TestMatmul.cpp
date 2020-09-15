#include "open3d/Open3D.h"
#include "open3d/core/linalg/Matmul.h"

using namespace open3d;
using namespace open3d::core;

int main() {
    std::vector<Device> devices{Device("CUDA:0"), Device("CPU:0")};

    std::vector<float> vals{0, 1, 2, 3, 4, 5};
    for (auto device : devices) {
        Tensor A(vals, {2, 3}, core::Dtype::Float32, device);
        Tensor R = Tensor(std::vector<float>({1, 0, 0, 0, -1, 1, 0, 0, -1}),
                          {3, 3}, Dtype::Float32, device);
        Tensor C;
        Matmul(A, R, C);

        std::cout << A.ToString() << "\n";
        std::cout << R.ToString() << "\n";
        std::cout << C.ToString() << "\n";
    }
}
