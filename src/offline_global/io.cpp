#include "offline_global/io.hpp"

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace fast_lio_sam::offline {
namespace {

template <typename T>
T scalar(const YAML::Node& root, const char* section, const char* key,
         const T& fallback) {
  const auto node = root[section][key];
  return node ? node.as<T>() : fallback;
}

Eigen::Vector3d vector3(const YAML::Node& node, const std::string& label) {
  if (!node || !node.IsSequence() || node.size() != 3) {
    throw std::runtime_error(label + " must contain exactly three values");
  }
  return {node[0].as<double>(), node[1].as<double>(), node[2].as<double>()};
}

Eigen::Matrix3d matrix3(const YAML::Node& node, const std::string& label) {
  if (!node || !node.IsSequence() || node.size() != 9) {
    throw std::runtime_error(label + " must contain exactly nine values");
  }
  Eigen::Matrix3d value;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      value(row, col) = node[row * 3 + col].as<double>();
    }
  }
  return value;
}

double stampSeconds(const builtin_interfaces::msg::Time& stamp) {
  return static_cast<double>(stamp.sec) + 1e-9 * stamp.nanosec;
}

bool finitePose(const gtsam::Pose3& pose) {
  return pose.matrix().array().isFinite().all();
}

fs::path findRecordedBag(const fs::path& session_root) {
  const fs::path compact = session_root / "rosbag";
  if (fs::exists(compact / "metadata.yaml")) return compact;
  for (const auto& item : fs::recursive_directory_iterator(session_root)) {
    if (item.path().filename() == "metadata.yaml" &&
        item.path().parent_path().filename() != "KEY_FRAMES") {
      return item.path().parent_path();
    }
  }
  return {};
}

}  // namespace

gtsam::Pose3 poseFromRpy(double x, double y, double z, double roll,
                         double pitch, double yaw) {
  return {gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z)};
}

double rotationAngleDeg(const gtsam::Pose3& pose) {
  return gtsam::Rot3::Logmap(pose.rotation()).norm() * 180.0 / M_PI;
}

Eigen::Vector3d wgs84ToEnu(double latitude, double longitude, double altitude,
                           const Datum& datum) {
  constexpr double a = 6378137.0;
  constexpr double inv_f = 298.257223563;
  constexpr double f = 1.0 / inv_f;
  constexpr double e2 = f * (2.0 - f);
  auto ecef = [&](double lat_deg, double lon_deg, double h) {
    const double lat = lat_deg * M_PI / 180.0;
    const double lon = lon_deg * M_PI / 180.0;
    const double sin_lat = std::sin(lat);
    const double cos_lat = std::cos(lat);
    const double radius = a / std::sqrt(1.0 - e2 * sin_lat * sin_lat);
    return Eigen::Vector3d((radius + h) * cos_lat * std::cos(lon),
                           (radius + h) * cos_lat * std::sin(lon),
                           (radius * (1.0 - e2) + h) * sin_lat);
  };
  const Eigen::Vector3d delta =
      ecef(latitude, longitude, altitude) -
      ecef(datum.latitude, datum.longitude, datum.altitude);
  const double lat0 = datum.latitude * M_PI / 180.0;
  const double lon0 = datum.longitude * M_PI / 180.0;
  Eigen::Matrix3d rotation;
  rotation << -std::sin(lon0), std::cos(lon0), 0.0,
      -std::sin(lat0) * std::cos(lon0),
      -std::sin(lat0) * std::sin(lon0), std::cos(lat0),
      std::cos(lat0) * std::cos(lon0),
      std::cos(lat0) * std::sin(lon0), std::sin(lat0);
  return rotation * delta;
}

Config loadConfig(const fs::path& path) {
  const YAML::Node root = YAML::LoadFile(path.string());
  Config c;
  if (root["datum"]) {
    c.datum.latitude = scalar(root, "datum", "latitude", c.datum.latitude);
    c.datum.longitude = scalar(root, "datum", "longitude", c.datum.longitude);
    c.datum.altitude = scalar(root, "datum", "altitude", c.datum.altitude);
  }
  if (root["sessions"] && root["sessions"]["exclude"]) {
    const auto excluded = root["sessions"]["exclude"];
    if (!excluded.IsSequence())
      throw std::runtime_error("sessions/exclude must be a sequence");
    for (const auto& item : excluded) c.excluded_sessions.push_back(item.as<std::string>());
  }
#define LOAD(section, key, member) c.member = scalar(root, section, key, c.member)
  LOAD("submaps", "anchor_spacing", anchor_spacing);
  LOAD("submaps", "trajectory_window", submap_window);
  LOAD("submaps", "min_range", min_range);
  LOAD("submaps", "max_range", max_range);
  LOAD("submaps", "coarse_voxel", coarse_voxel);
  LOAD("submaps", "fine_voxel", fine_voxel);
  LOAD("submaps", "preview_voxel", preview_voxel);
  LOAD("registration", "candidate_xy", candidate_xy);
  LOAD("registration", "candidate_z", candidate_z);
  LOAD("registration", "constraint_spacing", constraint_spacing);
  LOAD("registration", "max_neighbors", max_neighbors);
  LOAD("registration", "use_ndt", use_ndt);
  LOAD("registration", "ndt_resolution", ndt_resolution);
  LOAD("registration", "ndt_iterations", ndt_iterations);
  LOAD("registration", "fine_iterations", fine_iterations);
  LOAD("registration", "fine_max_correspondence", fine_max_correspondence);
  LOAD("gates", "min_evaluated_points", min_evaluated_points);
  LOAD("gates", "min_within_1m", min_within_1m);
  LOAD("gates", "max_p75", max_p75);
  LOAD("gates", "max_correction_translation", max_correction_translation);
  LOAD("gates", "max_correction_rotation_deg", max_correction_rotation_deg);
  LOAD("gates", "max_reverse_translation", max_reverse_translation);
  LOAD("gates", "max_reverse_rotation_deg", max_reverse_rotation_deg);
  LOAD("optimization", "cauchy_scale", cauchy_scale);
  LOAD("optimization", "z_bias_prior_sigma", z_bias_prior_sigma);
  LOAD("optimization", "odom_rotation_sigma", odom_rotation_sigma);
  LOAD("optimization", "odom_translation_sigma", odom_translation_sigma);
  LOAD("optimization", "anchor_rotation_prior_sigma", anchor_rotation_prior_sigma);
  LOAD("optimization", "anchor_translation_prior_sigma", anchor_translation_prior_sigma);
  LOAD("optimization", "lidar_rotation_sigma_floor", lidar_rotation_sigma_floor);
  LOAD("optimization", "lidar_rotation_sigma_ceiling", lidar_rotation_sigma_ceiling);
  LOAD("optimization", "lidar_translation_sigma_floor", lidar_translation_sigma_floor);
  LOAD("optimization", "lidar_translation_sigma_ceiling", lidar_translation_sigma_ceiling);
  LOAD("optimization", "chi2_6d_99", chi2_6d_99);
  LOAD("optimization", "warn_z_bias", warn_z_bias);
  LOAD("optimization", "warn_xy_correction", warn_xy_correction);
  LOAD("elevation_alignment", "enable", elevation_prealign);
  LOAD("elevation_alignment", "xy_radius", elevation_xy_radius);
  LOAD("elevation_alignment", "parallel_angle_deg", elevation_parallel_angle_deg);
  LOAD("elevation_alignment", "crossing_angle_deg", elevation_crossing_angle_deg);
  LOAD("elevation_alignment", "min_overlap_length", elevation_min_overlap_length);
  LOAD("elevation_alignment", "min_overlap_samples", elevation_min_overlap_samples);
  LOAD("elevation_alignment", "factor_spacing", elevation_factor_spacing);
  LOAD("elevation_alignment", "overlap_sigma", elevation_overlap_sigma);
  LOAD("elevation_alignment", "crossing_sigma", elevation_crossing_sigma);
  LOAD("elevation_alignment", "crossing_cluster_radius", elevation_crossing_cluster_radius);
  LOAD("elevation_alignment", "offset_prior_sigma", elevation_offset_prior_sigma);
  LOAD("elevation_alignment", "max_initial_offset", elevation_max_initial_offset);
  LOAD("rigid_optimization", "roll_pitch_sigma_deg", rigid_roll_pitch_sigma_deg);
  LOAD("rigid_optimization", "yaw_sigma_deg", rigid_yaw_sigma_deg);
  LOAD("rigid_optimization", "xy_prior_sigma", rigid_xy_prior_sigma);
  LOAD("rigid_optimization", "z_prior_sigma", rigid_z_prior_sigma);
  LOAD("vertical_optimization", "prior_sigma", vertical_prior_sigma);
  LOAD("vertical_optimization", "first_difference_sigma", vertical_first_difference_sigma);
  LOAD("vertical_optimization", "second_difference_sigma", vertical_second_difference_sigma);
  LOAD("vertical_optimization", "max_local_correction", vertical_max_local_correction);
  LOAD("quality", "max_roll_pitch_deg", quality_max_roll_pitch_deg);
  LOAD("quality", "max_z_bias", quality_max_z_bias);
  LOAD("quality", "max_keyframe_correction", quality_max_keyframe_correction);
  LOAD("quality", "fail_on_violation", quality_fail_on_violation);
  LOAD("gnss", "spacing", gnss_spacing);
  LOAD("gnss", "time_tolerance", gnss_time_tolerance);
  LOAD("gnss", "xy_sigma_floor", gnss_xy_sigma_floor);
  LOAD("gnss", "z_sigma_floor", gnss_z_sigma_floor);
  LOAD("gnss", "sigma_ceiling", gnss_sigma_ceiling);
  LOAD("gnss", "huber_k", gnss_huber_k);
  LOAD("gnss", "topic", gnss_topic);
  LOAD("output", "diagnostic_limit_per_class", diagnostic_limit_per_class);
  LOAD("output", "highrate_topic", highrate_topic);
#undef LOAD
  return c;
}

std::vector<Session> discoverSessions(const fs::path& results_root,
                                      const Config& config,
                                      std::vector<ValidationIssue>* excluded) {
  std::vector<fs::path> markers;
  if (!fs::exists(results_root)) {
    throw std::runtime_error("results root does not exist: " + results_root.string());
  }
  for (const auto& item : fs::recursive_directory_iterator(results_root)) {
    if (item.is_regular_file() && item.path().filename() == "COMPLETE") {
      markers.push_back(item.path());
    }
  }
  std::sort(markers.begin(), markers.end());
  std::vector<Session> sessions;
  for (const auto& marker : markers) {
    const fs::path root = marker.parent_path();
    try {
      Session session;
      session.root = root;
      session.config_path = root / "ouster128_novatel.yaml";
      const fs::path pose_path = root / "KEY_FRAMES" / "keyframe_pose.txt";
      const fs::path scan_dir = root / "KEY_FRAMES" / "scans";
      session.recorded_bag = findRecordedBag(root);
      if (!fs::is_regular_file(session.config_path)) throw std::runtime_error("saved configuration missing");
      if (!fs::is_regular_file(pose_path)) throw std::runtime_error("keyframe_pose.txt missing");
      if (!fs::is_directory(scan_dir)) throw std::runtime_error("scan directory missing");
      if (session.recorded_bag.empty()) throw std::runtime_error("recorded GNSS bag missing");

      const auto relative = fs::relative(root, results_root);
      std::vector<std::string> parts;
      for (const auto& part : relative) parts.push_back(part.string());
      if (parts.size() < 3) throw std::runtime_error("expected <date>/<time>/<bag> layout");
      session.date = parts[0];
      session.time = parts[1];
      session.bag = parts[2];
      session.id = session.date + "/" + session.time + "/" + session.bag;
      if (std::find(config.excluded_sessions.begin(), config.excluded_sessions.end(),
                    session.date + "/" + session.time) != config.excluded_sessions.end() ||
          std::find(config.excluded_sessions.begin(), config.excluded_sessions.end(),
                    session.id) != config.excluded_sessions.end()) {
        throw std::runtime_error("explicitly excluded by offline configuration");
      }

      const YAML::Node saved = YAML::LoadFile(session.config_path.string());
      if (!saved["gnss"]["use_fixed_origin"] ||
          !saved["gnss"]["use_fixed_origin"].as<bool>()) {
        throw std::runtime_error("session was not generated with a fixed GNSS origin");
      }
      const double origin_lat = saved["gnss"]["origin_latitude"].as<double>();
      const double origin_lon = saved["gnss"]["origin_longitude"].as<double>();
      const double origin_alt = saved["gnss"]["origin_altitude"].as<double>();
      if (std::abs(origin_lat - config.datum.latitude) > 1e-9 ||
          std::abs(origin_lon - config.datum.longitude) > 1e-9 ||
          std::abs(origin_alt - config.datum.altitude) > 1e-4) {
        throw std::runtime_error("saved session datum differs from offline datum");
      }
      session.lidar_to_body_t = vector3(saved["mapping"]["extrinsic_T"], "mapping/extrinsic_T");
      session.lidar_to_body_R = matrix3(saved["mapping"]["extrinsic_R"], "mapping/extrinsic_R");
      const Eigen::Vector3d antenna_lidar =
          vector3(saved["gnss"]["extrinsic_T"], "gnss/extrinsic_T");
      session.antenna_in_body = session.lidar_to_body_t +
                                session.lidar_to_body_R * antenna_lidar;

      std::ifstream poses(pose_path);
      double stamp, x, y, z, roll, pitch, yaw;
      while (poses >> stamp >> x >> y >> z >> roll >> pitch >> yaw) {
        Keyframe key;
        key.stamp = stamp;
        key.original = poseFromRpy(x, y, z, roll, pitch, yaw);
        key.optimized = key.original;
        std::ostringstream filename;
        filename << std::fixed << std::setprecision(9) << stamp << ".pcd";
        key.scan_path = scan_dir / filename.str();
        if (!finitePose(key.original)) throw std::runtime_error("non-finite keyframe pose");
        if (!session.keyframes.empty()) {
          if (stamp <= session.keyframes.back().stamp) throw std::runtime_error("timestamps are not strictly increasing");
          key.travel = session.keyframes.back().travel +
              (key.original.translation() - session.keyframes.back().original.translation()).norm();
        }
        session.keyframes.push_back(std::move(key));
      }
      if (session.keyframes.empty()) throw std::runtime_error("no keyframe poses");
      std::size_t pcd_count = 0;
      for (const auto& file : fs::directory_iterator(scan_dir)) {
        if (file.is_regular_file() && file.path().extension() == ".pcd") ++pcd_count;
      }
      if (pcd_count != session.keyframes.size()) {
        throw std::runtime_error("pose/PCD count mismatch: " +
                                 std::to_string(session.keyframes.size()) + "/" +
                                 std::to_string(pcd_count));
      }
      for (const auto& key : session.keyframes) {
        if (!fs::is_regular_file(key.scan_path)) {
          throw std::runtime_error("PCD timestamp does not match pose: " + key.scan_path.filename().string());
        }
      }
      session.index = sessions.size();
      sessions.push_back(std::move(session));
    } catch (const std::exception& error) {
      excluded->push_back({root, error.what()});
    }
  }
  if (sessions.empty()) throw std::runtime_error("no valid COMPLETE sessions found");
  (void)config;
  return sessions;
}

void loadGnss(Session* session, const Config& config) {
  rosbag2_cpp::Reader reader;
  reader.open(session->recorded_bag.string());
  rosbag2_storage::StorageFilter filter;
  filter.topics = {config.gnss_topic};
  reader.set_filter(filter);
  rclcpp::Serialization<sensor_msgs::msg::NavSatFix> serializer;
  while (reader.has_next()) {
    auto bag_message = reader.read_next();
    if (bag_message->topic_name != config.gnss_topic) continue;
    sensor_msgs::msg::NavSatFix msg;
    rclcpp::SerializedMessage serialized(*bag_message->serialized_data);
    serializer.deserialize_message(&serialized, &msg);
    GnssSample sample;
    sample.stamp = stampSeconds(msg.header.stamp);
    sample.status = msg.status.status;
    sample.enu = wgs84ToEnu(msg.latitude, msg.longitude, msg.altitude, config.datum);
    for (int i = 0; i < 3; ++i) {
      const double variance = msg.position_covariance[i * 3 + i];
      sample.sigma[i] = variance >= 0.0 ? std::sqrt(variance) :
                                          std::numeric_limits<double>::quiet_NaN();
    }
    if (std::isfinite(sample.stamp) && sample.enu.array().isFinite().all() &&
        sample.sigma.array().isFinite().all() && sample.status >= 0) {
      session->gnss.push_back(sample);
    }
  }
  if (session->gnss.empty()) {
    throw std::runtime_error("no valid GNSS messages on " + config.gnss_topic);
  }
  std::sort(session->gnss.begin(), session->gnss.end(),
            [](const auto& a, const auto& b) { return a.stamp < b.stamp; });
}

std::string jsonEscape(const std::string& input) {
  std::ostringstream out;
  for (const char c : input) {
    switch (c) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << c;
    }
  }
  return out.str();
}

void writeValidationReport(const fs::path& path,
                           const std::vector<Session>& sessions,
                           const std::vector<ValidationIssue>& excluded) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path.string());
  out << "{\n  \"included\": [\n";
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    const auto& s = sessions[i];
    out << "    {\"id\": \"" << jsonEscape(s.id) << "\", \"keyframes\": "
        << s.keyframes.size() << ", \"gnss\": " << s.gnss.size() << "}"
        << (i + 1 == sessions.size() ? "\n" : ",\n");
  }
  out << "  ],\n  \"excluded\": [\n";
  for (std::size_t i = 0; i < excluded.size(); ++i) {
    out << "    {\"path\": \"" << jsonEscape(excluded[i].path.string())
        << "\", \"reason\": \"" << jsonEscape(excluded[i].reason) << "\"}"
        << (i + 1 == excluded.size() ? "\n" : ",\n");
  }
  out << "  ]\n}\n";
}

fs::path makeTimestampedRunDirectory(const fs::path& output_root) {
  const auto now = std::chrono::system_clock::now();
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm local{};
  localtime_r(&time, &local);
  std::ostringstream name;
  name << std::put_time(&local, "%Y%m%d_%H%M%S");
  fs::path output = output_root / name.str();
  fs::create_directories(output);
  return output;
}

void writeCompleteMarker(const fs::path& run_root, const std::string& stage) {
  std::ofstream out(run_root / "COMPLETE");
  out << "stage=" << stage << "\n";
}

}  // namespace fast_lio_sam::offline
