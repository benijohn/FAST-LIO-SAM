#pragma once

#include "offline_global/types.hpp"

namespace fast_lio_sam::offline {

std::vector<ElevationConstraint> estimateElevationPrealignment(
    std::vector<Session>* sessions, const Config& config,
    const fs::path& run_root);

gtsam::Pose3 prealignedPose(const Session& session, std::size_t keyframe);

}  // namespace fast_lio_sam::offline
