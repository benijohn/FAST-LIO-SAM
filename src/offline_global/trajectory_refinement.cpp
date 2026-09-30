#include "offline_global/trajectory_refinement.hpp"

#include "offline_global/io.hpp"

#include <Eigen/Geometry>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace fast_lio_sam::offline {
namespace {

constexpr double kRadiansToDegrees = 180.0 / M_PI;

double stampSeconds(const builtin_interfaces::msg::Time& stamp) {
  return static_cast<double>(stamp.sec) + 1e-9 * stamp.nanosec;
}

bool finitePose(const gtsam::Pose3& pose) {
  return pose.matrix().array().isFinite().all();
}

Eigen::Quaterniond normalizedQuaternion(const gtsam::Rot3& rotation) {
  Eigen::Quaterniond result = rotation.toQuaternion();
  result.normalize();
  return result;
}

Eigen::Quaterniond sameHemisphere(Eigen::Quaterniond value,
                                  const Eigen::Quaterniond& reference) {
  if (value.coeffs().dot(reference.coeffs()) < 0.0) value.coeffs() *= -1.0;
  return value;
}

std::pair<std::size_t, std::size_t> bracket(
    const std::vector<TimedPose>& poses, double stamp) {
  if (poses.empty()) throw std::runtime_error("cannot bracket an empty trajectory");
  const auto right = std::lower_bound(poses.begin(), poses.end(), stamp,
      [](const TimedPose& pose, double value) { return pose.stamp < value; });
  if (right == poses.begin()) return {0, 0};
  if (right == poses.end()) return {poses.size() - 1, poses.size() - 1};
  if (std::abs(right->stamp - stamp) < 1e-9) {
    const std::size_t index = std::distance(poses.begin(), right);
    return {index, index};
  }
  const std::size_t index = std::distance(poses.begin(), right);
  return {index - 1, index};
}

gtsam::Pose3 rawPoseAt(const std::vector<TimedPose>& raw, double stamp,
                       double* bracket_gap) {
  if (stamp < raw.front().stamp - 1e-9 || stamp > raw.back().stamp + 1e-9) {
    throw std::runtime_error("target keyframe lies outside high-rate odometry range");
  }
  const auto [left, right] = bracket(raw, stamp);
  *bracket_gap = right == left ? 0.0 : raw[right].stamp - raw[left].stamp;
  return left == right ? raw[left].pose : interpolatePose(raw[left], raw[right], stamp);
}

gtsam::Pose3 anchoredPoseAt(const gtsam::Pose3& raw_pose, double stamp,
                            const std::vector<CorrectionKnot>& knots,
                            const TrajectoryRefinementConfig& config,
                            InterpolationInfo* info) {
  if (stamp <= knots.front().stamp) {
    if (info) *info = {0, 0, 0.0, stamp < knots.front().stamp};
    return knots.front().correction.compose(raw_pose);
  }
  if (stamp >= knots.back().stamp) {
    const std::size_t last = knots.size() - 1;
    if (info) *info = {last, last, 0.0, stamp > knots.back().stamp};
    return knots.back().correction.compose(raw_pose);
  }
  const auto upper = std::upper_bound(knots.begin(), knots.end(), stamp,
      [](double value, const CorrectionKnot& knot) { return value < knot.stamp; });
  const std::size_t right = std::distance(knots.begin(), upper);
  const std::size_t left = right - 1;
  const double alpha = (stamp - knots[left].stamp) /
      (knots[right].stamp - knots[left].stamp);
  if (info) *info = {left, right, alpha, false};

  // Start with the raw relative motion placed at the optimized left anchor.
  // Then distribute only the local endpoint discrepancy across the interval.
  // This is invariant to the arbitrary global/map origin, unlike interpolating
  // the translation component of map_T_odom directly.
  const gtsam::Pose3 base = knots[left].target.compose(
      knots[left].raw.between(raw_pose));
  const gtsam::Pose3 predicted_right = knots[left].target.compose(
      knots[left].raw.between(knots[right].raw));
  const gtsam::Pose3 endpoint_residual = predicted_right.between(
      knots[right].target);
  gtsam::Vector6 tangent = gtsam::Pose3::Logmap(endpoint_residual);
  const double smooth = alpha * alpha * alpha *
      (alpha * (alpha * 6.0 - 15.0) + 10.0);  // C2 smootherstep.
  tangent.head<3>() *= config.smooth_rotation ? smooth : alpha;
  tangent.tail<3>() *= config.smooth_translation ? smooth : alpha;
  return base.compose(gtsam::Pose3::Expmap(tangent));
}

std::vector<TimedPose> loadTargetPoses(const fs::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read target keyframes: " + path.string());
  std::vector<TimedPose> result;
  double stamp, x, y, z, roll, pitch, yaw;
  while (input >> stamp >> x >> y >> z >> roll >> pitch >> yaw) {
    TimedPose value{stamp, poseFromRpy(x, y, z, roll, pitch, yaw)};
    if (!std::isfinite(stamp) || !finitePose(value.pose))
      throw std::runtime_error("non-finite target pose in " + path.string());
    if (!result.empty() && stamp <= result.back().stamp)
      throw std::runtime_error("non-monotonic target timestamps in " + path.string());
    result.push_back(value);
  }
  if (result.empty())
    throw std::runtime_error("target trajectory contains no keyframes: " + path.string());
  return result;
}

struct RawBagResult {
  std::vector<TimedPose> poses;
  std::string frame;
  std::string child_frame;
};

RawBagResult loadRawPoses(const fs::path& bag,
                          const TrajectoryRefinementConfig& config) {
  rosbag2_cpp::Reader reader;
  reader.open(bag.string());
  rosbag2_storage::StorageFilter filter;
  filter.topics = {config.highrate_topic};
  reader.set_filter(filter);
  rclcpp::Serialization<nav_msgs::msg::Odometry> serializer;
  RawBagResult result;
  while (reader.has_next()) {
    const auto bag_message = reader.read_next();
    nav_msgs::msg::Odometry message;
    rclcpp::SerializedMessage serialized(*bag_message->serialized_data);
    serializer.deserialize_message(&serialized, &message);
    if (result.frame.empty()) {
      result.frame = message.header.frame_id;
      result.child_frame = message.child_frame_id;
    } else if (result.frame != message.header.frame_id ||
               result.child_frame != message.child_frame_id) {
      throw std::runtime_error("high-rate odometry frame IDs changed within the bag");
    }
    const auto& p = message.pose.pose.position;
    const auto& q = message.pose.pose.orientation;
    TimedPose pose{stampSeconds(message.header.stamp),
        gtsam::Pose3(gtsam::Rot3::Quaternion(q.w, q.x, q.y, q.z),
                     gtsam::Point3(p.x, p.y, p.z))};
    if (!std::isfinite(pose.stamp) || !finitePose(pose.pose))
      throw std::runtime_error("non-finite high-rate odometry pose");
    result.poses.push_back(std::move(pose));
  }
  if (result.poses.size() < 2)
    throw std::runtime_error("high-rate odometry topic is empty or too short in " + bag.string());
  std::sort(result.poses.begin(), result.poses.end(),
      [](const auto& a, const auto& b) { return a.stamp < b.stamp; });
  for (std::size_t i = 1; i < result.poses.size(); ++i) {
    if (result.poses[i].stamp <= result.poses[i - 1].stamp)
      throw std::runtime_error("duplicate high-rate odometry header timestamps in " + bag.string());
  }
  return result;
}

void writePose(std::ostream& output, const gtsam::Pose3& pose) {
  const auto q = normalizedQuaternion(pose.rotation());
  output << pose.x() << ',' << pose.y() << ',' << pose.z() << ','
         << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w();
}

void writeSessionOutputs(const fs::path& output, const RefinedTrajectory& trajectory,
                         const RawBagResult& raw, const fs::path& target_path,
                         const TrajectoryRefinementConfig& config) {
  fs::create_directories(output);
  std::ofstream poses(output / "refined_highrate_pose.csv");
  poses << std::setprecision(17)
        << "timestamp,x,y,z,qx,qy,qz,qw,correction_x,correction_y,correction_z,"
           "correction_qx,correction_qy,correction_qz,correction_qw,left_keyframe,"
           "right_keyframe,alpha,extrapolated\n";
  for (std::size_t i = 0; i < trajectory.refined.size(); ++i) {
    const auto correction = trajectory.refined[i].pose.compose(
        trajectory.raw[i].pose.inverse());
    poses << trajectory.raw[i].stamp << ',';
    writePose(poses, trajectory.refined[i].pose); poses << ',';
    writePose(poses, correction);
    const auto& info = trajectory.interpolation[i];
    poses << ',' << info.left << ',' << info.right << ',' << info.alpha << ','
          << info.extrapolated << '\n';
  }

  std::ofstream knots(output / "correction_knots.csv");
  knots << std::setprecision(17)
        << "timestamp,target_x,target_y,target_z,target_qx,target_qy,target_qz,target_qw,"
           "raw_x,raw_y,raw_z,raw_qx,raw_qy,raw_qz,raw_qw,correction_x,correction_y,"
           "correction_z,correction_qx,correction_qy,correction_qz,correction_qw\n";
  for (const auto& knot : trajectory.knots) {
    knots << knot.stamp << ','; writePose(knots, knot.target); knots << ',';
    writePose(knots, knot.raw); knots << ','; writePose(knots, knot.correction); knots << '\n';
  }

  std::ofstream quality(output / "trajectory_quality.json");
  quality << std::setprecision(17)
          << "{\n  \"quality_passed\": " << (trajectory.quality_passed ? "true" : "false")
          << ",\n  \"target_path\": \"" << jsonEscape(target_path.string()) << "\""
          << ",\n  \"frame_id\": \"" << jsonEscape(raw.frame) << "\""
          << ",\n  \"child_frame_id\": \"" << jsonEscape(raw.child_frame) << "\""
          << ",\n  \"highrate_samples\": " << trajectory.raw.size()
          << ",\n  \"keyframe_anchors\": " << trajectory.knots.size()
          << ",\n  \"extrapolated_samples\": " << trajectory.extrapolated_samples
          << ",\n  \"maximum_odom_gap_seconds\": " << trajectory.maximum_odom_gap
          << ",\n  \"maximum_keyframe_bracket_gap_seconds\": "
          << trajectory.maximum_keyframe_bracket_gap
          << ",\n  \"maximum_anchor_translation_error_m\": "
          << trajectory.maximum_anchor_translation_error
          << ",\n  \"maximum_anchor_rotation_error_deg\": "
          << trajectory.maximum_anchor_rotation_error_deg
          << ",\n  \"maximum_interval_adjustment_m\": "
          << trajectory.maximum_interval_adjustment_translation
          << ",\n  \"maximum_interval_adjustment_deg\": "
          << trajectory.maximum_interval_adjustment_rotation_deg
          << ",\n  \"maximum_correction_speed_mps\": "
          << trajectory.maximum_correction_speed
          << ",\n  \"maximum_correction_angular_speed_degps\": "
          << trajectory.maximum_correction_angular_speed_deg << ",\n  \"warnings\": [";
  for (std::size_t i = 0; i < trajectory.warnings.size(); ++i) {
    if (i) quality << ',';
    quality << "\"" << jsonEscape(trajectory.warnings[i]) << "\"";
  }
  quality << "]\n}\n";
  if (trajectory.quality_passed) {
    std::ofstream complete(output / "COMPLETE");
    complete << "high-rate trajectory refinement complete\n";
  }
}

bool selectedSession(const std::string& id,
                     const std::vector<std::string>& filters) {
  if (filters.empty()) return true;
  return std::find(filters.begin(), filters.end(), id) != filters.end();
}

}  // namespace

gtsam::Pose3 interpolatePose(const TimedPose& left, const TimedPose& right,
                             double stamp) {
  if (right.stamp <= left.stamp) return left.pose;
  const double alpha = std::clamp((stamp - left.stamp) /
      (right.stamp - left.stamp), 0.0, 1.0);
  const gtsam::Point3 translation = (1.0 - alpha) * left.pose.translation() +
                                    alpha * right.pose.translation();
  Eigen::Quaterniond q0 = normalizedQuaternion(left.pose.rotation());
  Eigen::Quaterniond q1 = sameHemisphere(
      normalizedQuaternion(right.pose.rotation()), q0);
  return {gtsam::Rot3(q0.slerp(alpha, q1).normalized()), translation};
}

TrajectoryRefinementConfig loadTrajectoryRefinementConfig(const fs::path& path) {
  const YAML::Node root = YAML::LoadFile(path.string());
  TrajectoryRefinementConfig config;
#define READ(section, key, field) \
  if (root[section] && root[section][key]) config.field = root[section][key].as<decltype(config.field)>()
  READ("input", "highrate_topic", highrate_topic);
  READ("timing", "warn_odom_gap", warn_odom_gap);
  READ("timing", "warn_keyframe_bracket_gap", warn_keyframe_bracket_gap);
  READ("timing", "max_odom_gap", max_odom_gap);
  READ("timing", "max_keyframe_bracket_gap", max_keyframe_bracket_gap);
  READ("quality", "max_anchor_error_translation", max_anchor_error_translation);
  READ("quality", "max_anchor_error_rotation_deg", max_anchor_error_rotation_deg);
  READ("quality", "warn_correction_speed", warn_correction_speed);
  READ("quality", "warn_correction_angular_speed_deg", warn_correction_angular_speed_deg);
  READ("quality", "fail_interval_adjustment_translation", fail_interval_adjustment_translation);
  READ("quality", "fail_interval_adjustment_rotation_deg", fail_interval_adjustment_rotation_deg);
  READ("interpolation", "smooth_translation", smooth_translation);
  READ("interpolation", "smooth_rotation", smooth_rotation);
#undef READ
  if (config.max_odom_gap <= 0.0 || config.max_keyframe_bracket_gap <= 0.0)
    throw std::runtime_error("trajectory timing thresholds must be positive");
  return config;
}

RefinedTrajectory refineTrajectory(const std::vector<TimedPose>& raw,
                                   const std::vector<TimedPose>& targets,
                                   const TrajectoryRefinementConfig& config) {
  if (raw.size() < 2 || targets.empty())
    throw std::runtime_error("trajectory refinement needs two raw poses and one target pose");
  RefinedTrajectory result;
  result.raw = raw;
  for (std::size_t i = 1; i < raw.size(); ++i) {
    const double gap = raw[i].stamp - raw[i - 1].stamp;
    if (gap <= 0.0) throw std::runtime_error("raw pose timestamps are not strictly increasing");
    result.maximum_odom_gap = std::max(result.maximum_odom_gap, gap);
  }
  for (std::size_t i = 0; i < targets.size(); ++i) {
    if (i && targets[i].stamp <= targets[i - 1].stamp)
      throw std::runtime_error("target pose timestamps are not strictly increasing");
    double gap = 0.0;
    const gtsam::Pose3 raw_at_target = rawPoseAt(raw, targets[i].stamp, &gap);
    result.maximum_keyframe_bracket_gap =
        std::max(result.maximum_keyframe_bracket_gap, gap);
    result.knots.push_back({targets[i].stamp, raw_at_target, targets[i].pose,
                            targets[i].pose.compose(raw_at_target.inverse())});
  }
  result.refined.reserve(raw.size());
  result.interpolation.reserve(raw.size());
  for (const auto& sample : raw) {
    InterpolationInfo info;
    const auto refined = anchoredPoseAt(sample.pose, sample.stamp, result.knots,
                                       config, &info);
    result.refined.push_back({sample.stamp, refined});
    result.interpolation.push_back(info);
    if (info.extrapolated) ++result.extrapolated_samples;
  }
  for (const auto& knot : result.knots) {
    InterpolationInfo unused;
    const auto reconstructed = anchoredPoseAt(knot.raw, knot.stamp, result.knots,
                                              config, &unused);
    const auto error = knot.target.between(reconstructed);
    result.maximum_anchor_translation_error = std::max(
        result.maximum_anchor_translation_error, error.translation().norm());
    result.maximum_anchor_rotation_error_deg = std::max(
        result.maximum_anchor_rotation_error_deg,
        gtsam::Rot3::Logmap(error.rotation()).norm() * kRadiansToDegrees);
  }
  for (std::size_t i = 1; i < result.knots.size(); ++i) {
    const double duration = result.knots[i].stamp - result.knots[i - 1].stamp;
    const auto predicted = result.knots[i - 1].target.compose(
        result.knots[i - 1].raw.between(result.knots[i].raw));
    const auto residual = predicted.between(result.knots[i].target);
    const double translation = residual.translation().norm();
    const double rotation = gtsam::Rot3::Logmap(residual.rotation()).norm() * kRadiansToDegrees;
    result.maximum_interval_adjustment_translation =
        std::max(result.maximum_interval_adjustment_translation, translation);
    result.maximum_interval_adjustment_rotation_deg =
        std::max(result.maximum_interval_adjustment_rotation_deg, rotation);
    result.maximum_correction_speed =
        std::max(result.maximum_correction_speed, translation / duration);
    result.maximum_correction_angular_speed_deg =
        std::max(result.maximum_correction_angular_speed_deg, rotation / duration);
  }
  auto fail = [&](bool condition, const std::string& warning) {
    if (!condition) return;
    result.quality_passed = false;
    result.warnings.push_back(warning);
  };
  auto warn = [&](bool condition, const std::string& warning) {
    if (condition) result.warnings.push_back(warning);
  };
  fail(result.maximum_odom_gap > config.max_odom_gap + 1e-9,
       "high-rate odometry gap exceeds threshold");
  fail(result.maximum_keyframe_bracket_gap > config.max_keyframe_bracket_gap + 1e-9,
       "keyframe interpolation bracket exceeds threshold");
  fail(result.maximum_anchor_translation_error > config.max_anchor_error_translation,
       "refined trajectory does not reproduce keyframe translation anchors");
  fail(result.maximum_anchor_rotation_error_deg > config.max_anchor_error_rotation_deg,
       "refined trajectory does not reproduce keyframe rotation anchors");
  fail(result.maximum_interval_adjustment_translation >
       config.fail_interval_adjustment_translation,
       "keyframe interval translation adjustment exceeds threshold");
  fail(result.maximum_interval_adjustment_rotation_deg >
       config.fail_interval_adjustment_rotation_deg,
       "keyframe interval rotation adjustment exceeds threshold");
  warn(result.maximum_correction_speed > config.warn_correction_speed,
       "correction translation changes rapidly between keyframes");
  warn(result.maximum_correction_angular_speed_deg >
       config.warn_correction_angular_speed_deg,
       "correction rotation changes rapidly between keyframes");
  warn(result.maximum_odom_gap > config.warn_odom_gap,
       "high-rate odometry contains a timing gap worth reviewing");
  warn(result.maximum_keyframe_bracket_gap > config.warn_keyframe_bracket_gap,
       "a keyframe lies inside a high-rate odometry gap worth reviewing");
  return result;
}

int runTrajectoryRefiner(const fs::path& results_root,
                         const fs::path& target_root,
                         const fs::path& output_root,
                         const fs::path& config_path,
                         const std::vector<std::string>& session_filters) {
  const auto config = loadTrajectoryRefinementConfig(config_path);
  const fs::path run_root = makeTimestampedRunDirectory(output_root);
  fs::copy_file(config_path, run_root / "offline_trajectory.yaml",
                fs::copy_options::overwrite_existing);
  std::cout << "[trajectory] output: " << run_root << '\n';
  std::vector<fs::path> markers;
  for (const auto& item : fs::recursive_directory_iterator(results_root)) {
    if (item.is_regular_file() && item.path().filename() == "COMPLETE")
      markers.push_back(item.path());
  }
  std::sort(markers.begin(), markers.end());
  std::set<std::string> found_filters;
  std::size_t completed = 0, skipped = 0, failed = 0;
  std::ofstream summary(run_root / "summary.csv");
  summary << "session,status,reason,highrate_samples,keyframes,quality_passed\n";
  for (const auto& marker : markers) {
    const fs::path session_root = marker.parent_path();
    const fs::path relative = fs::relative(session_root, results_root);
    const std::string id = relative.generic_string();
    if (!selectedSession(id, session_filters)) continue;
    found_filters.insert(id);
    const fs::path target_path = target_root.empty()
        ? session_root / "KEY_FRAMES" / "keyframe_pose.txt"
        : target_root / "sessions" / relative / "optimized_keyframe_pose.txt";
    if (!fs::is_regular_file(target_path)) {
      const bool explicitly_requested = !session_filters.empty();
      summary << id << ',' << (explicitly_requested ? "failed" : "skipped")
              << ",target trajectory missing,0,0,false\n";
      if (explicitly_requested) ++failed; else ++skipped;
      continue;
    }
    try {
      const fs::path bag = session_root / "rosbag";
      if (!fs::is_regular_file(bag / "metadata.yaml"))
        throw std::runtime_error("recorded rosbag/metadata.yaml missing");
      const auto raw = loadRawPoses(bag, config);
      const auto targets = loadTargetPoses(target_path);
      const auto trajectory = refineTrajectory(raw.poses, targets, config);
      writeSessionOutputs(run_root / "sessions" / relative, trajectory, raw,
                          target_path, config);
      summary << id << ',' << (trajectory.quality_passed ? "complete" : "failed")
              << ',' << (trajectory.quality_passed ? "" : "quality checks failed")
              << ',' << trajectory.raw.size() << ',' << trajectory.knots.size()
              << ',' << trajectory.quality_passed << '\n';
      std::cout << "[trajectory] " << id << " samples=" << trajectory.raw.size()
                << " anchors=" << trajectory.knots.size()
                << " anchor_error=" << trajectory.maximum_anchor_translation_error
                << "m quality=" << trajectory.quality_passed << '\n';
      if (trajectory.quality_passed) ++completed; else ++failed;
    } catch (const std::exception& error) {
      summary << id << ",failed,\"" << error.what() << "\",0,0,false\n";
      std::cerr << "[trajectory] " << id << " failed: " << error.what() << '\n';
      ++failed;
    }
  }
  for (const auto& filter : session_filters) {
    if (!found_filters.count(filter)) {
      summary << filter << ",failed,session filter did not match a COMPLETE result,0,0,false\n";
      ++failed;
    }
  }
  summary.close();
  if (completed == 0 || failed > 0) {
    std::cerr << "[trajectory] completed=" << completed << " skipped=" << skipped
              << " failed=" << failed << "; COMPLETE not written\n";
    return 1;
  }
  writeCompleteMarker(run_root, "trajectory_refinement");
  std::cout << "[trajectory] complete sessions=" << completed
            << " skipped=" << skipped << '\n';
  return 0;
}

}  // namespace fast_lio_sam::offline
