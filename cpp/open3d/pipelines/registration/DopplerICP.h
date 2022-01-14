// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#pragma once

#include <Eigen/Core>
#include <memory>

#include "open3d/pipelines/registration/Registration.h"
#include "open3d/pipelines/registration/RobustKernel.h"
#include "open3d/pipelines/registration/TransformationEstimation.h"

namespace open3d {

namespace geometry {
class PointCloud;
}

namespace pipelines {
namespace registration {

class RegistrationResult;

class TransformationEstimationForDopplerICP : public TransformationEstimation {
public:
    ~TransformationEstimationForDopplerICP() override{};

    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };
    explicit TransformationEstimationForDopplerICP(
            double lambda_geometric = 0.5,
            double doppler_outlier_threshold = 0.5,
            size_t geometric_robust_loss_min_iteration = 0,
            size_t doppler_robust_loss_min_iteration = 2,
            bool check_doppler_compatibility = false,
            std::shared_ptr<RobustKernel> geometric_kernel =
                    std::make_shared<L2Loss>(),
            std::shared_ptr<RobustKernel> doppler_kernel =
                    std::make_shared<L2Loss>())
        : lambda_geometric_(lambda_geometric),
          doppler_outlier_threshold_(doppler_outlier_threshold),
          geometric_robust_loss_min_iteration_(
                  geometric_robust_loss_min_iteration),
          doppler_robust_loss_min_iteration_(doppler_robust_loss_min_iteration),
          check_doppler_compatibility_(check_doppler_compatibility),
          geometric_kernel_(std::move(geometric_kernel)),
          doppler_kernel_(std::move(doppler_kernel)) {
        if (lambda_geometric_ < 0 || lambda_geometric_ > 1.0) {
            lambda_geometric_ = 0.5;
        }
    }

public:
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const CorrespondenceSet &corres) const override;
    Eigen::Matrix4d ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const override;
    Eigen::Matrix4d ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres,
            const std::vector<Eigen::Vector3d> &source_directions,
            const double period,
            const Eigen::Matrix4d &transformation,
            const Eigen::Matrix4d &T_V_to_S,
            const size_t iteration,
            std::vector<Eigen::Vector3d> &errors) const;

public:
    double lambda_geometric_{0.5};
    double doppler_outlier_threshold_{0.5};
    size_t geometric_robust_loss_min_iteration_{0};
    size_t doppler_robust_loss_min_iteration_{2};
    bool check_doppler_compatibility_{false};
    /// shared_ptr to an Abstract RobustKernel that could mutate at runtime.
    std::shared_ptr<RobustKernel> default_kernel_ = std::make_shared<L2Loss>();
    std::shared_ptr<RobustKernel> geometric_kernel_ =
            std::make_shared<L2Loss>();
    std::shared_ptr<RobustKernel> doppler_kernel_ = std::make_shared<L2Loss>();

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::DopplerICP;
};

/// \brief Function for Doppler ICP registration.
///
/// \param source The source point cloud.
/// \param target The target point cloud.
/// \param max_distance Maximum correspondence points-pair distance.
/// \param init Initial transformation estimation.
/// Default value: array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],
/// [0., 0., 0., 1.]]).
/// \param estimation TransformationEstimationForDopplerICP method. Can only
/// change the lambda_geometric value and the robust kernel used in the
/// optimization.
/// \param criteria Convergence criteria.
/// \param period Time period (in seconds) between the source and the target
/// point clouds. Default value: 0.1.
/// \param T_V_to_S The 4x4 transformation matrix to transform
/// sensor to vehicle frame.
/// Default value: array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],
/// [0., 0., 0., 1.]])
RegistrationResult RegistrationDopplerICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_distance,
        const Eigen::Matrix4d &init = Eigen::Matrix4d::Identity(),
        const TransformationEstimationForDopplerICP &estimation =
                TransformationEstimationForDopplerICP(),
        const ICPConvergenceCriteria &criteria = ICPConvergenceCriteria(),
        const double period = 0.1F,
        const Eigen::Matrix4d &T_V_to_S = Eigen::Matrix4d::Identity());

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
