#pragma once

#include <gtsam/geometry/Pose3.h>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace fast_lio_sam::offline {

namespace fs = std::filesystem;

struct TimedPose {
  double stamp = 0.0;
  gtsam::Pose3 pose;
};

struct CorrectionKnot {
  double stamp = 0.0;
  gtsam::Pose3 raw;
  gtsam::Pose3 target;
  gtsam::Pose3 correction;
};

struct InterpolationInfo {
  std::size_t left = 0;
  std::size_t right = 0;
  double alpha = 0.0;
  bool extrapolated = false;
};

struct TrajectoryRefinementConfig {
  std::string highrate_topic = "/OdometryHighFreq";
  double warn_odom_gap = 0.05;
  double warn_keyframe_bracket_gap = 0.05;
  double max_odom_gap = 0.25;
  double max_keyframe_bracket_gap = 0.25;
  double max_anchor_error_translation = 1e-5;
  double max_anchor_error_rotation_deg = 1e-4;
  double warn_correction_speed = 2.0;
  double warn_correction_angular_speed_deg = 20.0;
  double fail_interval_adjustment_translation = 10.0;
  double fail_interval_adjustment_rotation_deg = 45.0;
  bool smooth_translation = true;
  bool smooth_rotation = true;
};

struct RefinedTrajectory {
  std::vector<TimedPose> raw;
  std::vector<TimedPose> refined;
  std::vector<CorrectionKnot> knots;
  std::vector<InterpolationInfo> interpolation;
  double maximum_odom_gap = 0.0;
  double maximum_keyframe_bracket_gap = 0.0;
  double maximum_anchor_translation_error = 0.0;
  double maximum_anchor_rotation_error_deg = 0.0;
  double maximum_interval_adjustment_translation = 0.0;
  double maximum_interval_adjustment_rotation_deg = 0.0;
  double maximum_correction_speed = 0.0;
  double maximum_correction_angular_speed_deg = 0.0;
  std::size_t extrapolated_samples = 0;
  bool quality_passed = true;
  std::vector<std::string> warnings;
};

gtsam::Pose3 interpolatePose(const TimedPose& left, const TimedPose& right,
                             double stamp);

TrajectoryRefinementConfig loadTrajectoryRefinementConfig(const fs::path& path);

RefinedTrajectory refineTrajectory(const std::vector<TimedPose>& raw,
                                   const std::vector<TimedPose>& targets,
                                   const TrajectoryRefinementConfig& config);

int runTrajectoryRefiner(const fs::path& results_root,
                         const fs::path& target_root,
                         const fs::path& output_root,
                         const fs::path& config_path,
                         const std::vector<std::string>& session_filters);

}  // namespace fast_lio_sam::offline
