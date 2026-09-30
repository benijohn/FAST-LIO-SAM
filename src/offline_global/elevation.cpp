#include "offline_global/elevation.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <unordered_map>

namespace fast_lio_sam::offline {
namespace {

struct Match {
  std::size_t a = 0;
  std::size_t b = 0;
  double xy = 0.0;
  double angle_deg = 0.0;
};

struct Cell {
  int x = 0;
  int y = 0;
  bool operator==(const Cell& other) const { return x == other.x && y == other.y; }
};

struct CellHash {
  std::size_t operator()(const Cell& cell) const {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(cell.x)) << 32) ^
           static_cast<std::uint32_t>(cell.y);
  }
};

Eigen::Vector2d tangent(const Session& session, std::size_t index) {
  if (session.keyframes.size() < 2) return Eigen::Vector2d::Zero();
  const std::size_t before = index == 0 ? 0 : index - 1;
  const std::size_t after = std::min(index + 1, session.keyframes.size() - 1);
  Eigen::Vector2d value =
      (session.keyframes[after].original.translation() -
       session.keyframes[before].original.translation()).head<2>();
  const double norm = value.norm();
  if (norm > 1e-6) value /= norm;
  else value.setZero();
  return value;
}

double undirectedAngleDeg(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
  if (a.norm() < 0.5 || b.norm() < 0.5) return 45.0;
  const double cosine = std::clamp(std::abs(a.dot(b)), 0.0, 1.0);
  return std::acos(cosine) * 180.0 / M_PI;
}

double median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  const std::size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  double result = values[middle];
  if (values.size() % 2 == 0) {
    const auto lower = std::max_element(values.begin(), values.begin() + middle);
    result = 0.5 * (result + *lower);
  }
  return result;
}

std::vector<Match> nearestMatches(const Session& a, const Session& b,
                                  const Config& config) {
  const double cell_size = config.elevation_xy_radius;
  std::unordered_map<Cell, std::vector<std::size_t>, CellHash> grid;
  for (std::size_t j = 0; j < b.keyframes.size(); ++j) {
    const auto p = b.keyframes[j].original.translation();
    grid[{static_cast<int>(std::floor(p.x() / cell_size)),
          static_cast<int>(std::floor(p.y() / cell_size))}].push_back(j);
  }
  std::vector<Match> matches;
  for (std::size_t i = 0; i < a.keyframes.size(); ++i) {
    const auto p = a.keyframes[i].original.translation();
    const Cell center{static_cast<int>(std::floor(p.x() / cell_size)),
                      static_cast<int>(std::floor(p.y() / cell_size))};
    double best_distance = config.elevation_xy_radius;
    std::size_t best = b.keyframes.size();
    for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) {
      const auto found = grid.find({center.x + dx, center.y + dy});
      if (found == grid.end()) continue;
      for (const std::size_t j : found->second) {
        const auto q = b.keyframes[j].original.translation();
        const double distance = (p.head<2>() - q.head<2>()).norm();
        if (distance < best_distance) { best_distance = distance; best = j; }
      }
    }
    if (best != b.keyframes.size()) {
      matches.push_back({i, best, best_distance,
                         undirectedAngleDeg(tangent(a, i), tangent(b, best))});
    }
  }
  return matches;
}

void addOverlapConstraints(const Session& a, const Session& b,
                           const std::vector<Match>& matches, const Config& config,
                           std::vector<ElevationConstraint>* constraints) {
  std::vector<Match> parallel;
  for (const auto& match : matches)
    if (match.angle_deg <= config.elevation_parallel_angle_deg) parallel.push_back(match);
  if (parallel.size() < static_cast<std::size_t>(config.elevation_min_overlap_samples)) return;
  std::sort(parallel.begin(), parallel.end(), [&](const auto& x, const auto& y) {
    return a.keyframes[x.a].travel < a.keyframes[y.a].travel;
  });
  const double length = a.keyframes[parallel.back().a].travel -
                        a.keyframes[parallel.front().a].travel;
  if (length < config.elevation_min_overlap_length) return;
  double last_travel = -std::numeric_limits<double>::infinity();
  for (const auto& match : parallel) {
    const double travel = a.keyframes[match.a].travel;
    if (travel - last_travel < config.elevation_factor_spacing) continue;
    constraints->push_back({a.index, match.a, b.index, match.b, "overlap",
        match.xy, match.angle_deg,
        a.keyframes[match.a].original.z() - b.keyframes[match.b].original.z(),
        config.elevation_overlap_sigma});
    last_travel = travel;
  }
}

void addCrossingConstraints(const Session& a, const Session& b,
                            const std::vector<Match>& matches, const Config& config,
                            std::vector<ElevationConstraint>* constraints) {
  std::vector<Match> crossings;
  for (const auto& match : matches)
    if (match.angle_deg >= config.elevation_crossing_angle_deg) crossings.push_back(match);
  std::sort(crossings.begin(), crossings.end(),
            [](const auto& x, const auto& y) { return x.xy < y.xy; });
  std::vector<Eigen::Vector2d> used;
  for (const auto& match : crossings) {
    const auto pa = a.keyframes[match.a].original.translation().head<2>();
    const auto pb = b.keyframes[match.b].original.translation().head<2>();
    const Eigen::Vector2d center = 0.5 * (pa + pb);
    bool duplicate = false;
    for (const auto& previous : used)
      if ((center - previous).norm() < config.elevation_crossing_cluster_radius) duplicate = true;
    if (duplicate) continue;
    constraints->push_back({a.index, match.a, b.index, match.b, "crossing",
        match.xy, match.angle_deg,
        a.keyframes[match.a].original.z() - b.keyframes[match.b].original.z(),
        config.elevation_crossing_sigma});
    used.push_back(center);
  }
}

}  // namespace

gtsam::Pose3 prealignedPose(const Session& session, std::size_t keyframe) {
  const auto& original = session.keyframes[keyframe].original;
  return {original.rotation(), original.translation() + gtsam::Point3(0, 0, session.elevation_offset)};
}

std::vector<ElevationConstraint> estimateElevationPrealignment(
    std::vector<Session>* sessions, const Config& config,
    const fs::path& run_root) {
  std::vector<ElevationConstraint> constraints;
  if (config.elevation_prealign) {
    for (std::size_t i = 0; i < sessions->size(); ++i) {
      for (std::size_t j = i + 1; j < sessions->size(); ++j) {
        const auto matches = nearestMatches((*sessions)[i], (*sessions)[j], config);
        addOverlapConstraints((*sessions)[i], (*sessions)[j], matches, config, &constraints);
        addCrossingConstraints((*sessions)[i], (*sessions)[j], matches, config, &constraints);
      }
    }
  }

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values initial;
  for (const auto& session : *sessions) {
    const auto key = gtsam::Symbol('e', session.index);
    initial.insert(key, 0.0);
    graph.add(gtsam::PriorFactor<double>(key, 0.0,
        gtsam::noiseModel::Isotropic::Sigma(1, config.elevation_offset_prior_sigma)));
  }
  for (const auto& constraint : constraints) {
    // Equal corrected heights imply offset_b - offset_a = original_z_a - original_z_b.
    auto base = gtsam::noiseModel::Isotropic::Sigma(1, constraint.sigma);
    auto robust = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(1.345), base);
    graph.add(gtsam::BetweenFactor<double>(gtsam::Symbol('e', constraint.session_a),
        gtsam::Symbol('e', constraint.session_b), constraint.measured_a_minus_b, robust));
  }
  if (!constraints.empty()) {
    gtsam::LevenbergMarquardtParams parameters;
    parameters.maxIterations = 100;
    const auto result = gtsam::LevenbergMarquardtOptimizer(graph, initial, parameters).optimize();
    for (auto& session : *sessions) {
      const double raw = result.at<double>(gtsam::Symbol('e', session.index));
      session.elevation_offset_raw = raw;
      session.elevation_offset = std::clamp(raw, -config.elevation_max_initial_offset,
                                            config.elevation_max_initial_offset);
      session.elevation_offset_clamped =
          std::abs(raw - session.elevation_offset) > 1e-9;
    }
  }
  for (const auto& constraint : constraints) {
    (*sessions)[constraint.session_a].elevation_supported = true;
    (*sessions)[constraint.session_b].elevation_supported = true;
  }

  std::ofstream out(run_root / "elevation_constraints.csv");
  out << "session_a,keyframe_a,session_b,keyframe_b,kind,xy_distance,heading_angle_deg,original_z_a_minus_b,sigma\n"
      << std::setprecision(17);
  for (const auto& c : constraints) {
    out << (*sessions)[c.session_a].id << ',' << c.keyframe_a << ','
        << (*sessions)[c.session_b].id << ',' << c.keyframe_b << ',' << c.kind << ','
        << c.xy_distance << ',' << c.heading_angle_deg << ','
        << c.measured_a_minus_b << ',' << c.sigma << '\n';
  }
  std::ofstream offsets(run_root / "elevation_prealignment.csv");
  offsets << "session,elevation_offset,raw_elevation_offset,clamped,supported\n"
          << std::setprecision(17);
  for (const auto& session : *sessions)
    offsets << session.id << ',' << session.elevation_offset << ','
            << session.elevation_offset_raw << ',' << session.elevation_offset_clamped << ','
            << session.elevation_supported << '\n';
  std::cout << "[elevation] constraints=" << constraints.size() << std::endl;
  return constraints;
}

}  // namespace fast_lio_sam::offline
