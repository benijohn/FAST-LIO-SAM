#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtsam/geometry/Pose3.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace fast_lio_sam::offline {

namespace fs = std::filesystem;
using Cloud = pcl::PointCloud<pcl::PointXYZ>;

struct Datum {
  double latitude = 34.399072206970;
  double longitude = -82.138620318436;
  double altitude = 146.572827416472;
};

struct Config {
  Datum datum;
  std::vector<std::string> excluded_sessions;
  double anchor_spacing = 10.0;
  double submap_window = 12.0;
  double min_range = 1.2;
  double max_range = 60.0;
  double coarse_voxel = 1.0;
  double fine_voxel = 0.30;
  double preview_voxel = 0.50;
  double candidate_xy = 15.0;
  double candidate_z = 10.0;
  double constraint_spacing = 15.0;
  int max_neighbors = 4;
  bool use_ndt = true;
  double ndt_resolution = 2.0;
  int ndt_iterations = 40;
  int fine_iterations = 80;
  double fine_max_correspondence = 5.0;
  int min_evaluated_points = 3000;
  double min_within_1m = 0.75;
  double max_p75 = 0.75;
  double max_correction_translation = 5.0;
  double max_correction_rotation_deg = 10.0;
  double max_reverse_translation = 0.30;
  double max_reverse_rotation_deg = 1.0;
  double cauchy_scale = 2.0;
  double gnss_spacing = 5.0;
  double gnss_time_tolerance = 0.10;
  double gnss_xy_sigma_floor = 0.25;
  double gnss_z_sigma_floor = 0.50;
  double gnss_sigma_ceiling = 10.0;
  double gnss_huber_k = 1.345;
  double z_bias_prior_sigma = 3.0;
  double odom_rotation_sigma = 0.01;
  double odom_translation_sigma = 0.03;
  double anchor_rotation_prior_sigma = 0.02;
  double anchor_translation_prior_sigma = 0.10;
  double lidar_rotation_sigma_floor = 0.03;
  double lidar_rotation_sigma_ceiling = 0.20;
  double lidar_translation_sigma_floor = 0.20;
  double lidar_translation_sigma_ceiling = 1.0;
  double chi2_6d_99 = 16.8119;
  double warn_z_bias = 5.0;
  double warn_xy_correction = 5.0;
  bool elevation_prealign = true;
  double elevation_xy_radius = 2.0;
  double elevation_parallel_angle_deg = 30.0;
  double elevation_crossing_angle_deg = 60.0;
  double elevation_min_overlap_length = 15.0;
  int elevation_min_overlap_samples = 10;
  double elevation_factor_spacing = 10.0;
  double elevation_overlap_sigma = 0.35;
  double elevation_crossing_sigma = 0.65;
  double elevation_crossing_cluster_radius = 5.0;
  double elevation_offset_prior_sigma = 3.0;
  double elevation_max_initial_offset = 5.0;
  double rigid_roll_pitch_sigma_deg = 0.5;
  double rigid_yaw_sigma_deg = 3.0;
  double rigid_xy_prior_sigma = 2.0;
  double rigid_z_prior_sigma = 1.0;
  double vertical_prior_sigma = 1.0;
  double vertical_first_difference_sigma = 0.20;
  double vertical_second_difference_sigma = 0.08;
  double vertical_max_local_correction = 3.0;
  double quality_max_roll_pitch_deg = 1.0;
  double quality_max_z_bias = 5.0;
  double quality_max_keyframe_correction = 10.0;
  bool quality_fail_on_violation = true;
  int diagnostic_limit_per_class = 12;
  std::string gnss_topic = "/novatel/oem7/fix";
  std::string highrate_topic = "/OdometryHighFreq";
};

struct Keyframe {
  double stamp = 0.0;
  gtsam::Pose3 original;
  gtsam::Pose3 optimized;
  fs::path scan_path;
  double travel = 0.0;
};

struct GnssSample {
  double stamp = 0.0;
  Eigen::Vector3d enu = Eigen::Vector3d::Zero();
  Eigen::Vector3d sigma = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  int status = -1;
};

struct Session {
  std::size_t index = 0;
  std::string id;
  std::string date;
  std::string time;
  std::string bag;
  fs::path root;
  fs::path config_path;
  fs::path recorded_bag;
  Eigen::Matrix3d lidar_to_body_R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d lidar_to_body_t = Eigen::Vector3d::Zero();
  Eigen::Vector3d antenna_in_body = Eigen::Vector3d::Zero();
  std::vector<Keyframe> keyframes;
  std::vector<GnssSample> gnss;
  std::vector<std::size_t> anchors;
  gtsam::Pose3 rigid_correction;
  gtsam::Pose3 optimized_anchor;
  Eigen::Matrix<double, 6, 6> anchor_covariance =
      Eigen::Matrix<double, 6, 6>::Constant(
          std::numeric_limits<double>::quiet_NaN());
  std::vector<double> vertical_corrections;
  std::vector<double> vertical_variances;
  double elevation_offset = 0.0;
  double elevation_offset_raw = 0.0;
  bool elevation_offset_clamped = false;
  bool elevation_supported = false;
  double z_bias = 0.0;
  bool lidar_supported = false;
  std::string cache_key;
};

struct ElevationConstraint {
  std::size_t session_a = 0;
  std::size_t keyframe_a = 0;
  std::size_t session_b = 0;
  std::size_t keyframe_b = 0;
  std::string kind;
  double xy_distance = 0.0;
  double heading_angle_deg = 0.0;
  double measured_a_minus_b = 0.0;
  double sigma = 1.0;
};

struct Anchor {
  std::size_t global_index = 0;
  std::size_t session = 0;
  std::size_t keyframe = 0;
  fs::path coarse_path;
  fs::path fine_path;
};

struct RegistrationMetrics {
  std::size_t evaluated = 0;
  double rms = std::numeric_limits<double>::infinity();
  double p50 = std::numeric_limits<double>::infinity();
  double p75 = std::numeric_limits<double>::infinity();
  double p90 = std::numeric_limits<double>::infinity();
  double within_1m = 0.0;
};

struct Constraint {
  std::size_t id = 0;
  std::size_t source_anchor = 0;
  std::size_t target_anchor = 0;
  gtsam::Pose3 source_to_target;
  gtsam::Pose3 measurement_source_between_target;
  RegistrationMetrics metrics;
  Eigen::Matrix<double, 6, 1> sigmas =
      Eigen::Matrix<double, 6, 1>::Ones();
  bool ndt_converged = false;
  bool fine_converged = false;
  bool accepted = false;
  bool pruned = false;
  double correction_translation = std::numeric_limits<double>::infinity();
  double correction_rotation_deg = std::numeric_limits<double>::infinity();
  double reverse_translation = std::numeric_limits<double>::infinity();
  double reverse_rotation_deg = std::numeric_limits<double>::infinity();
  double robust_weight = 0.0;
  double final_residual = std::numeric_limits<double>::quiet_NaN();
  std::string reason;
};

struct ValidationIssue {
  fs::path path;
  std::string reason;
};

gtsam::Pose3 poseFromRpy(double x, double y, double z, double roll,
                         double pitch, double yaw);
Eigen::Vector3d wgs84ToEnu(double latitude, double longitude, double altitude,
                           const Datum& datum);
double rotationAngleDeg(const gtsam::Pose3& pose);

}  // namespace fast_lio_sam::offline
