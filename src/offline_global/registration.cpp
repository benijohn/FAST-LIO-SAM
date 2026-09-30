#include "offline_global/registration.hpp"

#include "offline_global/elevation.hpp"
#include "offline_global/submaps.hpp"

#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/io/pcd_io.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace fast_lio_sam::offline {
namespace {

struct Candidate {
  std::size_t source;
  std::size_t target;
  double distance;
};

struct SessionEdge {
  std::size_t a;
  std::size_t b;
  double score;
  std::vector<Candidate> candidates;
};

class UnionFind {
 public:
  explicit UnionFind(std::size_t count) : parent_(count) {
    std::iota(parent_.begin(), parent_.end(), 0);
  }
  std::size_t find(std::size_t value) {
    if (parent_[value] != value) parent_[value] = find(parent_[value]);
    return parent_[value];
  }
  bool join(std::size_t a, std::size_t b) {
    a = find(a); b = find(b);
    if (a == b) return false;
    parent_[b] = a;
    return true;
  }
 private:
  std::vector<std::size_t> parent_;
};

gtsam::Pose3 poseFromMatrix(const Eigen::Matrix4f& matrix) {
  return {gtsam::Rot3(matrix.block<3, 3>(0, 0).cast<double>()),
          gtsam::Point3(matrix.block<3, 1>(0, 3).cast<double>())};
}

Eigen::Matrix4f matrixFromPose(const gtsam::Pose3& pose) {
  return pose.matrix().cast<float>();
}

RegistrationMetrics measure(const Cloud::ConstPtr& source,
                            const Cloud::ConstPtr& target,
                            const Eigen::Matrix4f& transform,
                            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>* pairs = nullptr) {
  RegistrationMetrics metrics;
  pcl::KdTreeFLANN<pcl::PointXYZ> tree;
  tree.setInputCloud(target);
  std::vector<double> distances;
  distances.reserve(source->size());
  std::vector<int> index(1);
  std::vector<float> squared(1);
  for (const auto& raw : *source) {
    Eigen::Vector4f transformed = transform * Eigen::Vector4f(raw.x, raw.y, raw.z, 1.0f);
    pcl::PointXYZ point(transformed.x(), transformed.y(), transformed.z());
    if (tree.nearestKSearch(point, 1, index, squared) != 1) continue;
    const double distance = std::sqrt(squared[0]);
    distances.push_back(distance);
    if (pairs && distance <= 2.0) {
      const auto& match = (*target)[index[0]];
      pairs->push_back({transformed.head<3>().cast<double>(),
                        Eigen::Vector3d(match.x, match.y, match.z)});
    }
  }
  metrics.evaluated = distances.size();
  if (distances.empty()) return metrics;
  std::sort(distances.begin(), distances.end());
  auto percentile = [&](double fraction) {
    const std::size_t i = std::min(distances.size() - 1,
        static_cast<std::size_t>(fraction * (distances.size() - 1)));
    return distances[i];
  };
  double sum_square = 0.0;
  std::size_t within = 0;
  for (double d : distances) {
    sum_square += d * d;
    if (d <= 1.0) ++within;
  }
  metrics.rms = std::sqrt(sum_square / distances.size());
  metrics.p50 = percentile(0.50);
  metrics.p75 = percentile(0.75);
  metrics.p90 = percentile(0.90);
  metrics.within_1m = static_cast<double>(within) / distances.size();
  return metrics;
}

Eigen::Matrix<double, 6, 1> estimateSigmas(
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& pairs,
    const RegistrationMetrics& metrics, const Config& config) {
  Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
  for (const auto& pair : pairs) {
    const Eigen::Vector3d& p = pair.first;
    Eigen::Matrix<double, 3, 6> jacobian;
    Eigen::Matrix3d skew;
    skew << 0.0, -p.z(), p.y(), p.z(), 0.0, -p.x(), -p.y(), p.x(), 0.0;
    jacobian.block<3, 3>(0, 0) = -skew;
    jacobian.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    hessian.noalias() += jacobian.transpose() * jacobian;
  }
  const double variance = std::max(0.01, metrics.p75 * metrics.p75);
  Eigen::Matrix<double, 6, 6> covariance =
      variance * (hessian + 1e-6 * Eigen::Matrix<double, 6, 6>::Identity()).inverse();
  Eigen::Matrix<double, 6, 1> sigmas;
  for (int i = 0; i < 6; ++i) {
    double sigma = std::sqrt(std::max(0.0, covariance(i, i)));
    if (i < 3) {
      sigma = std::clamp(sigma, config.lidar_rotation_sigma_floor,
                         config.lidar_rotation_sigma_ceiling);
    } else {
      sigma = std::clamp(sigma, config.lidar_translation_sigma_floor,
                         config.lidar_translation_sigma_ceiling);
    }
    sigmas[i] = sigma;
  }
  return sigmas;
}

std::vector<SessionEdge> buildEdges(const std::vector<Session>& sessions,
                                    const std::vector<Anchor>& anchors,
                                    const Config& config) {
  std::map<std::pair<std::size_t, std::size_t>, std::vector<Candidate>> grouped;
  for (std::size_t i = 0; i < anchors.size(); ++i) {
    const auto& ai = anchors[i];
    const auto pi = prealignedPose(sessions[ai.session], ai.keyframe).translation();
    for (std::size_t j = i + 1; j < anchors.size(); ++j) {
      const auto& aj = anchors[j];
      if (ai.session == aj.session) continue;
      const auto pj = prealignedPose(sessions[aj.session], aj.keyframe).translation();
      const double xy = (pi.head<2>() - pj.head<2>()).norm();
      if (xy > config.candidate_xy || std::abs(pi.z() - pj.z()) > config.candidate_z) continue;
      const auto pair = std::minmax(ai.session, aj.session);
      Candidate candidate{i, j, xy};
      if (ai.session != pair.first) std::swap(candidate.source, candidate.target);
      grouped[pair].push_back(candidate);
    }
  }
  std::vector<SessionEdge> edges;
  for (auto& [pair, candidates] : grouped) {
    std::set<std::size_t> unique_a, unique_b;
    for (const auto& c : candidates) {
      unique_a.insert(anchors[c.source].keyframe);
      unique_b.insert(anchors[c.target].keyframe);
    }
    const double overlap = config.anchor_spacing * std::min(unique_a.size(), unique_b.size());
    edges.push_back({pair.first, pair.second, overlap, std::move(candidates)});
  }
  std::sort(edges.begin(), edges.end(), [](const auto& a, const auto& b) {
    return a.score > b.score;
  });
  return edges;
}

std::vector<const SessionEdge*> retainEdges(const std::vector<SessionEdge>& edges,
                                            std::size_t session_count,
                                            int max_neighbors) {
  std::set<std::pair<std::size_t, std::size_t>> selected;
  UnionFind components(session_count);
  for (const auto& edge : edges) {
    if (components.join(edge.a, edge.b)) selected.insert({edge.a, edge.b});
  }
  std::vector<int> degree(session_count, 0);
  for (const auto& pair : selected) { ++degree[pair.first]; ++degree[pair.second]; }
  for (const auto& edge : edges) {
    if (selected.count({edge.a, edge.b})) continue;
    if (degree[edge.a] >= max_neighbors && degree[edge.b] >= max_neighbors) continue;
    selected.insert({edge.a, edge.b});
    ++degree[edge.a]; ++degree[edge.b];
  }
  std::vector<const SessionEdge*> result;
  for (const auto& edge : edges) {
    if (selected.count({edge.a, edge.b})) result.push_back(&edge);
  }
  return result;
}

std::vector<Candidate> thinCandidates(const SessionEdge& edge,
                                      const std::vector<Anchor>& anchors,
                                      const std::vector<Session>& sessions,
                                      double spacing) {
  std::map<std::size_t, Candidate> nearest;
  for (const auto& candidate : edge.candidates) {
    auto found = nearest.find(candidate.source);
    if (found == nearest.end() || candidate.distance < found->second.distance) {
      nearest[candidate.source] = candidate;
    }
  }
  std::vector<Candidate> ordered;
  for (const auto& item : nearest) ordered.push_back(item.second);
  std::sort(ordered.begin(), ordered.end(), [&](const auto& a, const auto& b) {
    return sessions[anchors[a.source].session].keyframes[anchors[a.source].keyframe].travel <
           sessions[anchors[b.source].session].keyframes[anchors[b.source].keyframe].travel;
  });
  std::vector<Candidate> result;
  double last_travel = -std::numeric_limits<double>::infinity();
  for (const auto& candidate : ordered) {
    const auto& anchor = anchors[candidate.source];
    const double travel = sessions[anchor.session].keyframes[anchor.keyframe].travel;
    if (travel - last_travel >= spacing || result.empty()) {
      result.push_back(candidate);
      last_travel = travel;
    }
  }
  if (!ordered.empty() && result.back().source != ordered.back().source) result.push_back(ordered.back());
  return result;
}

Constraint registerOne(std::size_t id, const Candidate& candidate,
                       const std::vector<Anchor>& anchors,
                       const std::vector<Session>& sessions,
                       const Config& config) {
  Constraint result;
  result.id = id;
  result.source_anchor = candidate.source;
  result.target_anchor = candidate.target;
  const auto& source_anchor = anchors[candidate.source];
  const auto& target_anchor = anchors[candidate.target];
  const auto source_pose = prealignedPose(sessions[source_anchor.session], source_anchor.keyframe);
  const auto target_pose = prealignedPose(sessions[target_anchor.session], target_anchor.keyframe);
  const gtsam::Pose3 initial_pose = target_pose.inverse().compose(source_pose);
  const Eigen::Matrix4f initial = matrixFromPose(initial_pose);
  const auto source_coarse = loadCloud(source_anchor.coarse_path);
  const auto target_coarse = loadCloud(target_anchor.coarse_path);
  Eigen::Matrix4f seed = initial;
  if (config.use_ndt) {
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    ndt.setResolution(config.ndt_resolution);
    ndt.setMaximumIterations(config.ndt_iterations);
    ndt.setTransformationEpsilon(0.01);
    ndt.setStepSize(0.2);
    ndt.setInputSource(source_coarse);
    ndt.setInputTarget(target_coarse);
    Cloud aligned;
    ndt.align(aligned, initial);
    result.ndt_converged = ndt.hasConverged();
    if (result.ndt_converged) {
      const gtsam::Pose3 ndt_correction = poseFromMatrix(ndt.getFinalTransformation())
                                             .compose(initial_pose.inverse());
      if (ndt_correction.translation().norm() <= 2.0 * config.max_correction_translation &&
          rotationAngleDeg(ndt_correction) <= 2.0 * config.max_correction_rotation_deg) {
        seed = ndt.getFinalTransformation();
      }
    }
  }
  const auto source_fine = loadCloud(source_anchor.fine_path);
  const auto target_fine = loadCloud(target_anchor.fine_path);
  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
  gicp.setMaximumIterations(config.fine_iterations);
  gicp.setMaxCorrespondenceDistance(config.fine_max_correspondence);
  gicp.setTransformationEpsilon(1e-5);
  gicp.setEuclideanFitnessEpsilon(1e-5);
  gicp.setInputSource(source_fine);
  gicp.setInputTarget(target_fine);
  Cloud aligned;
  gicp.align(aligned, seed);
  result.fine_converged = gicp.hasConverged();
  const Eigen::Matrix4f forward = gicp.getFinalTransformation();
  result.source_to_target = poseFromMatrix(forward);
  result.measurement_source_between_target = result.source_to_target.inverse();
  const gtsam::Pose3 correction = result.source_to_target.compose(initial_pose.inverse());
  result.correction_translation = correction.translation().norm();
  result.correction_rotation_deg = rotationAngleDeg(correction);
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> pairs;
  result.metrics = measure(source_fine, target_fine, forward, &pairs);
  result.sigmas = estimateSigmas(pairs, result.metrics, config);

  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> reverse;
  reverse.setMaximumIterations(config.fine_iterations);
  reverse.setMaxCorrespondenceDistance(config.fine_max_correspondence);
  reverse.setTransformationEpsilon(1e-5);
  reverse.setEuclideanFitnessEpsilon(1e-5);
  reverse.setInputSource(target_fine);
  reverse.setInputTarget(source_fine);
  Cloud reverse_aligned;
  reverse.align(reverse_aligned, forward.inverse());
  const gtsam::Pose3 consistency = result.source_to_target.compose(
      poseFromMatrix(reverse.getFinalTransformation()));
  result.reverse_translation = consistency.translation().norm();
  result.reverse_rotation_deg = rotationAngleDeg(consistency);

  if (!result.fine_converged) result.reason = "fine_not_converged";
  else if (result.metrics.evaluated < static_cast<std::size_t>(config.min_evaluated_points)) result.reason = "too_few_points";
  else if (result.metrics.within_1m < config.min_within_1m) result.reason = "low_inlier_ratio";
  else if (result.metrics.p75 > config.max_p75) result.reason = "p75_above_threshold";
  else if (result.correction_translation > config.max_correction_translation) result.reason = "translation_correction_too_large";
  else if (result.correction_rotation_deg > config.max_correction_rotation_deg) result.reason = "rotation_correction_too_large";
  else if (!reverse.hasConverged()) result.reason = "reverse_not_converged";
  else if (result.reverse_translation > config.max_reverse_translation) result.reason = "reverse_translation_inconsistent";
  else if (result.reverse_rotation_deg > config.max_reverse_rotation_deg) result.reason = "reverse_rotation_inconsistent";
  else { result.accepted = true; result.reason = "accepted"; }
  return result;
}

void saveDiagnostic(const Constraint& constraint, const std::vector<Anchor>& anchors,
                    const fs::path& directory, const std::string& label) {
  fs::create_directories(directory);
  const auto source = loadCloud(anchors[constraint.source_anchor].fine_path);
  const auto target = loadCloud(anchors[constraint.target_anchor].fine_path);
  Cloud transformed;
  pcl::transformPointCloud(*source, transformed, matrixFromPose(constraint.source_to_target));
  pcl::io::savePCDFileBinaryCompressed(
      (directory / (label + "_source_aligned.pcd")).string(), transformed);
  pcl::io::savePCDFileBinaryCompressed(
      (directory / (label + "_target.pcd")).string(), *target);
}

}  // namespace

std::vector<Constraint> registerCrossSessionSubmaps(
    std::vector<Session>* sessions, const std::vector<Anchor>& anchors,
    const Config& config, const fs::path& run_root) {
  const auto edges = buildEdges(*sessions, anchors, config);
  const auto retained = retainEdges(edges, sessions->size(), config.max_neighbors);
  std::vector<Constraint> constraints;
  const fs::path constraints_path = run_root / "constraints.csv";
  if (fs::is_regular_file(constraints_path)) constraints = readConstraintsCsv(constraints_path);
  std::set<std::pair<std::size_t, std::size_t>> completed;
  std::size_t next_id = 0;
  for (const auto& constraint : constraints) {
    completed.insert({constraint.source_anchor, constraint.target_anchor});
    next_id = std::max(next_id, constraint.id + 1);
  }
  for (const auto* edge : retained) {
    for (const auto& candidate : thinCandidates(*edge, anchors, *sessions,
                                                config.constraint_spacing)) {
      if (completed.count({candidate.source, candidate.target})) continue;
      Constraint constraint = registerOne(next_id++, candidate, anchors,
                                          *sessions, config);
      if (constraint.accepted) {
        (*sessions)[anchors[constraint.source_anchor].session].lidar_supported = true;
        (*sessions)[anchors[constraint.target_anchor].session].lidar_supported = true;
      }
      constraints.push_back(std::move(constraint));
      completed.insert({candidate.source, candidate.target});
      writeConstraintsCsv(constraints_path, constraints, anchors, *sessions);
      std::cout << "[register] " << constraints.size() << " pair=" << edge->a << "-"
                << edge->b << " accepted=" << constraints.back().accepted
                << " p75=" << constraints.back().metrics.p75
                << " within1=" << constraints.back().metrics.within_1m
                << " reason=" << constraints.back().reason << std::endl;
    }
  }
  writeConstraintsCsv(constraints_path, constraints, anchors, *sessions);

  std::vector<const Constraint*> accepted, borderline, rejected;
  for (const auto& c : constraints) {
    if (c.accepted) accepted.push_back(&c);
    else if (c.metrics.within_1m >= config.min_within_1m * 0.9 &&
             c.metrics.p75 <= config.max_p75 * 1.25) borderline.push_back(&c);
    else rejected.push_back(&c);
  }
  std::sort(rejected.begin(), rejected.end(), [](const auto* a, const auto* b) {
    return a->metrics.p75 > b->metrics.p75;
  });
  const auto save_group = [&](const auto& group, const std::string& name) {
    const std::size_t count = std::min<std::size_t>(group.size(), config.diagnostic_limit_per_class);
    for (std::size_t i = 0; i < count; ++i) {
      std::ostringstream label;
      label << std::setw(4) << std::setfill('0') << group[i]->id;
      saveDiagnostic(*group[i], anchors, run_root / "diagnostics" / name, label.str());
    }
  };
  save_group(accepted, "accepted");
  save_group(borderline, "borderline");
  save_group(rejected, "worst_rejected");
  return constraints;
}

void writeConstraintsCsv(const fs::path& path,
                         const std::vector<Constraint>& constraints,
                         const std::vector<Anchor>& anchors,
                         const std::vector<Session>& sessions) {
  std::ofstream out(path);
  out << std::setprecision(17);
  out << "id,source_session,target_session,source_anchor,target_anchor,accepted,pruned,reason,evaluated,rms,p50,p75,p90,within_1m,correction_m,correction_deg,reverse_m,reverse_deg,robust_weight,final_residual";
  for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) out << ",t" << r << c;
  for (int i = 0; i < 6; ++i) out << ",sigma" << i;
  out << "\n";
  for (const auto& item : constraints) {
    const auto& sa = anchors[item.source_anchor];
    const auto& ta = anchors[item.target_anchor];
    out << item.id << ',' << sessions[sa.session].id << ',' << sessions[ta.session].id
        << ',' << item.source_anchor << ',' << item.target_anchor << ',' << item.accepted
        << ',' << item.pruned << ',' << item.reason << ',' << item.metrics.evaluated
        << ',' << item.metrics.rms << ',' << item.metrics.p50 << ',' << item.metrics.p75
        << ',' << item.metrics.p90 << ',' << item.metrics.within_1m
        << ',' << item.correction_translation << ',' << item.correction_rotation_deg
        << ',' << item.reverse_translation << ',' << item.reverse_rotation_deg
        << ',' << item.robust_weight << ',' << item.final_residual;
    const auto matrix = item.source_to_target.matrix();
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) out << ',' << matrix(r, c);
    for (int i = 0; i < 6; ++i) out << ',' << item.sigmas[i];
    out << '\n';
  }
}

std::vector<Constraint> readConstraintsCsv(const fs::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read " + path.string());
  std::string line;
  std::getline(input, line);
  std::vector<Constraint> result;
  while (std::getline(input, line)) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    if (fields.size() < 42) throw std::runtime_error("malformed constraints.csv row");
    Constraint c;
    c.id = std::stoull(fields[0]);
    c.source_anchor = std::stoull(fields[3]);
    c.target_anchor = std::stoull(fields[4]);
    c.accepted = std::stoi(fields[5]);
    c.pruned = std::stoi(fields[6]);
    c.reason = fields[7];
    c.metrics.evaluated = std::stoull(fields[8]);
    c.metrics.rms = std::stod(fields[9]); c.metrics.p50 = std::stod(fields[10]);
    c.metrics.p75 = std::stod(fields[11]); c.metrics.p90 = std::stod(fields[12]);
    c.metrics.within_1m = std::stod(fields[13]);
    c.correction_translation = std::stod(fields[14]);
    c.correction_rotation_deg = std::stod(fields[15]);
    c.reverse_translation = std::stod(fields[16]); c.reverse_rotation_deg = std::stod(fields[17]);
    c.robust_weight = std::stod(fields[18]); c.final_residual = std::stod(fields[19]);
    Eigen::Matrix4d matrix;
    std::size_t offset = 20;
    for (int r = 0; r < 4; ++r) for (int col = 0; col < 4; ++col) matrix(r, col) = std::stod(fields[offset++]);
    c.source_to_target = gtsam::Pose3(matrix);
    c.measurement_source_between_target = c.source_to_target.inverse();
    for (int i = 0; i < 6; ++i) c.sigmas[i] = std::stod(fields[offset++]);
    result.push_back(c);
  }
  return result;
}

}  // namespace fast_lio_sam::offline
