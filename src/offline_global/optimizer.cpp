#include "offline_global/optimizer.hpp"

#include "offline_global/io.hpp"
#include "offline_global/elevation.hpp"
#include "offline_global/factors.hpp"
#include "offline_global/registration.hpp"
#include "offline_global/submaps.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace fast_lio_sam::offline {
namespace {

using gtsam::symbol_shorthand::B;
gtsam::Key horizontalKey(std::size_t index) { return gtsam::Symbol('h', index); }
gtsam::Key verticalKey(std::size_t anchor) { return gtsam::Symbol('v', anchor); }

gtsam::SharedNoiseModel robustCauchy(const gtsam::Vector& sigmas, double scale) {
  auto diagonal = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
  auto cauchy = gtsam::noiseModel::mEstimator::Cauchy::Create(scale);
  return gtsam::noiseModel::Robust::Create(cauchy, diagonal);
}

gtsam::Values solve(const gtsam::NonlinearFactorGraph& graph,
                    const gtsam::Values& initial) {
  gtsam::LevenbergMarquardtParams params;
  params.maxIterations = 100;
  params.relativeErrorTol = 1e-6;
  params.absoluteErrorTol = 1e-6;
  params.verbosity = gtsam::NonlinearOptimizerParams::SILENT;
  return gtsam::LevenbergMarquardtOptimizer(graph, initial, params).optimize();
}

std::vector<std::pair<std::size_t, const GnssSample*>> matchGnss(
    const Session& session, const Config& config) {
  std::vector<std::pair<std::size_t, const GnssSample*>> matches;
  double last_travel = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < session.keyframes.size(); ++i) {
    const auto& key = session.keyframes[i];
    if (!matches.empty() && key.travel - last_travel < config.gnss_spacing) continue;
    const auto it = std::lower_bound(session.gnss.begin(), session.gnss.end(), key.stamp,
        [](const GnssSample& sample, double stamp) { return sample.stamp < stamp; });
    const GnssSample* best = nullptr;
    if (it != session.gnss.end()) best = &*it;
    if (it != session.gnss.begin()) {
      const auto* previous = &*std::prev(it);
      if (!best || std::abs(previous->stamp - key.stamp) < std::abs(best->stamp - key.stamp)) best = previous;
    }
    if (!best || std::abs(best->stamp - key.stamp) > config.gnss_time_tolerance) continue;
    if ((best->sigma.array() > config.gnss_sigma_ceiling).any()) continue;
    matches.push_back({i, best});
    last_travel = key.travel;
  }
  return matches;
}

gtsam::Pose2 pose2(const gtsam::Pose3& pose) {
  return {pose.x(), pose.y(), pose.rotation().rpy().z()};
}

gtsam::Pose2 relativePose2(const Session& session, std::size_t keyframe) {
  return pose2(session.keyframes.front().original).between(
      pose2(session.keyframes[keyframe].original));
}

std::vector<std::vector<std::size_t>> sessionAnchorGlobals(
    const std::vector<Session>& sessions, const std::vector<Anchor>& anchors) {
  std::vector<std::vector<std::size_t>> result(sessions.size());
  for (const auto& anchor : anchors) result[anchor.session].push_back(anchor.global_index);
  return result;
}

std::size_t nearestAnchor(const Session& session,
                          const std::vector<std::size_t>& globals,
                          const std::vector<Anchor>& anchors,
                          std::size_t keyframe) {
  if (globals.empty()) throw std::runtime_error("session has no vertical anchors");
  const double travel = session.keyframes[keyframe].travel;
  auto best = globals.front();
  double distance = std::numeric_limits<double>::infinity();
  for (const auto global : globals) {
    const double candidate = std::abs(
        session.keyframes[anchors[global].keyframe].travel - travel);
    if (candidate < distance) { distance = candidate; best = global; }
  }
  return best;
}

gtsam::Pose2 horizontalPose(const Session& session, std::size_t keyframe,
                            const gtsam::Values& values) {
  return values.at<gtsam::Pose2>(horizontalKey(session.index)).compose(
      relativePose2(session, keyframe));
}

gtsam::Rot3 yawAdjustedRotation(const Session& session, std::size_t keyframe,
                                const gtsam::Pose2& optimized_anchor) {
  const double yaw_delta = gtsam::Pose2::Logmap(
      pose2(session.keyframes.front().original).between(optimized_anchor)).z();
  return gtsam::Rot3::Rz(yaw_delta).compose(
      session.keyframes[keyframe].original.rotation());
}

gtsam::NonlinearFactorGraph makeHorizontalGraph(
    const std::vector<Session>& sessions, const std::vector<Anchor>& anchors,
    const std::vector<Constraint>& constraints,
    const Config& config, std::size_t* gnss_count, std::size_t* lidar_count) {
  gtsam::NonlinearFactorGraph graph;
  *gnss_count = 0; *lidar_count = 0;
  const double yaw_sigma = config.rigid_yaw_sigma_deg * M_PI / 180.0;
  const auto prior_sigmas = (gtsam::Vector3() << config.rigid_xy_prior_sigma,
      config.rigid_xy_prior_sigma, yaw_sigma).finished();
  for (const auto& session : sessions) {
    const gtsam::Pose2 original_anchor = pose2(session.keyframes.front().original);
    graph.add(gtsam::PriorFactor<gtsam::Pose2>(horizontalKey(session.index),
        original_anchor, gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas)));
    for (const auto& [index, sample] : matchGnss(session, config)) {
      const auto antenna = session.keyframes[index].original.transformFrom(
          gtsam::Point3(session.antenna_in_body));
      const auto relative = original_anchor.transformTo(gtsam::Point2(antenna.x(), antenna.y()));
      const auto sigmas = (gtsam::Vector2() <<
          std::max(sample->sigma.x(), config.gnss_xy_sigma_floor),
          std::max(sample->sigma.y(), config.gnss_xy_sigma_floor)).finished();
      auto base = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
      auto robust = gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Huber::Create(config.gnss_huber_k), base);
      graph.emplace_shared<HorizontalPositionFactor>(horizontalKey(session.index),
          relative, gtsam::Point2(sample->enu.x(), sample->enu.y()), robust);
      ++*gnss_count;
    }
  }
  for (const auto& c : constraints) {
    if (!c.accepted || c.pruned) continue;
    const auto& sa = anchors[c.source_anchor];
    const auto& ta = anchors[c.target_anchor];
    const auto measurement = pose2(c.measurement_source_between_target);
    const auto sigmas = (gtsam::Vector3() << c.sigmas[3], c.sigmas[4],
        c.sigmas[2]).finished();
    graph.emplace_shared<HorizontalAlignmentFactor>(horizontalKey(sa.session),
        horizontalKey(ta.session), relativePose2(sessions[sa.session], sa.keyframe),
        relativePose2(sessions[ta.session], ta.keyframe), measurement,
        robustCauchy(sigmas, config.cauchy_scale));
    ++*lidar_count;
  }
  return graph;
}

gtsam::Values makeHorizontalInitial(const std::vector<Session>& sessions) {
  gtsam::Values values;
  for (const auto& session : sessions)
    values.insert(horizontalKey(session.index), pose2(session.keyframes.front().original));
  return values;
}

gtsam::NonlinearFactorGraph makeVerticalGraph(
    const std::vector<Session>& sessions, const std::vector<Anchor>& anchors,
    const std::vector<Constraint>& constraints,
    const std::vector<ElevationConstraint>& elevation_constraints,
    const Config& config, const gtsam::Values& horizontal,
    std::size_t* gnss_count, std::size_t* lidar_count,
    std::size_t* elevation_count) {
  gtsam::NonlinearFactorGraph graph;
  *gnss_count = 0; *lidar_count = 0; *elevation_count = 0;
  const auto globals = sessionAnchorGlobals(sessions, anchors);
  for (const auto& session : sessions) {
    graph.add(gtsam::PriorFactor<double>(B(session.index), 0.0,
        gtsam::noiseModel::Isotropic::Sigma(1, config.z_bias_prior_sigma)));
    const auto& session_globals = globals[session.index];
    for (std::size_t i = 0; i < session_globals.size(); ++i) {
      const auto global = session_globals[i];
      graph.add(gtsam::PriorFactor<double>(verticalKey(global), session.elevation_offset,
          gtsam::noiseModel::Isotropic::Sigma(1, config.vertical_prior_sigma)));
      if (i > 0) graph.add(gtsam::BetweenFactor<double>(verticalKey(session_globals[i - 1]),
          verticalKey(global), 0.0, gtsam::noiseModel::Isotropic::Sigma(
              1, config.vertical_first_difference_sigma)));
      if (i > 1) graph.emplace_shared<SecondDifferenceFactor>(
          verticalKey(session_globals[i - 2]), verticalKey(session_globals[i - 1]),
          verticalKey(global), gtsam::noiseModel::Isotropic::Sigma(
              1, config.vertical_second_difference_sigma));
    }
    for (const auto& [index, sample] : matchGnss(session, config)) {
      const auto global = nearestAnchor(session, session_globals, anchors, index);
      const double antenna_z = session.keyframes[index].original.transformFrom(
          gtsam::Point3(session.antenna_in_body)).z();
      auto base = gtsam::noiseModel::Isotropic::Sigma(1,
          std::max(sample->sigma.z(), config.gnss_z_sigma_floor));
      auto robust = gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Huber::Create(config.gnss_huber_k), base);
      graph.emplace_shared<HeightBiasFactor>(verticalKey(global), B(session.index),
          antenna_z, sample->enu.z(), robust);
      ++*gnss_count;
    }
  }
  for (const auto& c : elevation_constraints) {
    const auto ga = nearestAnchor(sessions[c.session_a], globals[c.session_a],
                                  anchors, c.keyframe_a);
    const auto gb = nearestAnchor(sessions[c.session_b], globals[c.session_b],
                                  anchors, c.keyframe_b);
    auto base = gtsam::noiseModel::Isotropic::Sigma(1, c.sigma);
    auto robust = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(1.345), base);
    graph.add(gtsam::BetweenFactor<double>(verticalKey(ga), verticalKey(gb),
        c.measured_a_minus_b, robust));
    ++*elevation_count;
  }
  for (const auto& c : constraints) {
    if (!c.accepted || c.pruned) continue;
    const auto& sa = anchors[c.source_anchor];
    const auto& ta = anchors[c.target_anchor];
    const auto& source_session = sessions[sa.session];
    const auto& target_session = sessions[ta.session];
    const auto source_xy = horizontalPose(source_session, sa.keyframe, horizontal);
    const gtsam::Pose3 source_pose(
        yawAdjustedRotation(source_session, sa.keyframe,
            horizontal.at<gtsam::Pose2>(horizontalKey(sa.session))),
        gtsam::Point3(source_xy.x(), source_xy.y(),
                      source_session.keyframes[sa.keyframe].original.z()));
    const double desired_delta_z = source_pose.compose(
        c.measurement_source_between_target).z() - source_pose.z();
    const double original_delta_z =
        target_session.keyframes[ta.keyframe].original.z() -
        source_session.keyframes[sa.keyframe].original.z();
    auto robust = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Cauchy::Create(config.cauchy_scale),
        gtsam::noiseModel::Isotropic::Sigma(1, c.sigmas[5]));
    graph.add(gtsam::BetweenFactor<double>(verticalKey(c.source_anchor),
        verticalKey(c.target_anchor), desired_delta_z - original_delta_z, robust));
    ++*lidar_count;
  }
  return graph;
}

gtsam::Values makeVerticalInitial(const std::vector<Session>& sessions,
                                  const std::vector<Anchor>& anchors) {
  gtsam::Values values;
  for (const auto& session : sessions) values.insert(B(session.index), 0.0);
  for (const auto& anchor : anchors)
    values.insert(verticalKey(anchor.global_index), sessions[anchor.session].elevation_offset);
  return values;
}

gtsam::Pose3 optimizedAnchorPose(const Session& session, const Anchor& anchor,
                                 const gtsam::Values& horizontal,
                                 const gtsam::Values& vertical) {
  const auto xy = horizontalPose(session, anchor.keyframe, horizontal);
  return {yawAdjustedRotation(session, anchor.keyframe,
              horizontal.at<gtsam::Pose2>(horizontalKey(session.index))),
          gtsam::Point3(xy.x(), xy.y(), session.keyframes[anchor.keyframe].original.z() +
              vertical.at<double>(verticalKey(anchor.global_index)))};
}

double normalizedResidual(const Constraint& c, const Anchor& source,
                          const Anchor& target, const std::vector<Session>& sessions,
                          const gtsam::Values& horizontal,
                          const gtsam::Values& vertical) {
  const auto a = optimizedAnchorPose(sessions[source.session], source,
                                     horizontal, vertical);
  const auto b = optimizedAnchorPose(sessions[target.session], target,
                                     horizontal, vertical);
  const gtsam::Vector6 error = gtsam::Pose3::Logmap(
      c.measurement_source_between_target.between(a.between(b)));
  double value = 0.0;
  for (const int index : {2, 3, 4, 5})
    value += std::pow(error[index] / c.sigmas[index], 2);
  return value;
}

double interpolateAnchorValue(const Session& session, std::size_t keyframe,
                              const std::vector<double>& values) {
  if (session.anchors.empty() || values.size() != session.anchors.size())
    throw std::runtime_error("invalid vertical anchor values for " + session.id);
  if (session.anchors.size() == 1) return values.front();
  const double travel = session.keyframes[keyframe].travel;
  auto upper = std::lower_bound(session.anchors.begin(), session.anchors.end(), travel,
      [&](std::size_t anchor_keyframe, double value) {
        return session.keyframes[anchor_keyframe].travel < value;
      });
  if (upper == session.anchors.begin()) return values.front();
  if (upper == session.anchors.end()) return values.back();
  const std::size_t hi = static_cast<std::size_t>(upper - session.anchors.begin());
  const std::size_t lo = hi - 1;
  const double t0 = session.keyframes[session.anchors[lo]].travel;
  const double t1 = session.keyframes[session.anchors[hi]].travel;
  const double alpha = t1 > t0 ? (travel - t0) / (t1 - t0) : 0.0;
  return (1.0 - alpha) * values[lo] + alpha * values[hi];
}

Eigen::Matrix2d keyframeHorizontalCovariance(const Session& session,
                                             std::size_t keyframe) {
  const auto relative = relativePose2(session, keyframe);
  const double yaw = session.optimized_anchor.rotation().rpy().z();
  const double dx = relative.x(), dy = relative.y();
  Eigen::Matrix<double, 2, 3> jacobian;
  jacobian << 1.0, 0.0, -std::sin(yaw) * dx - std::cos(yaw) * dy,
              0.0, 1.0,  std::cos(yaw) * dx - std::sin(yaw) * dy;
  Eigen::Matrix3d covariance;
  covariance << session.anchor_covariance(3, 3), session.anchor_covariance(3, 4), session.anchor_covariance(3, 2),
                session.anchor_covariance(4, 3), session.anchor_covariance(4, 4), session.anchor_covariance(4, 2),
                session.anchor_covariance(2, 3), session.anchor_covariance(2, 4), session.anchor_covariance(2, 2);
  return jacobian * covariance * jacobian.transpose();
}

std::string colorFor(std::size_t index) {
  static const char* colors[] = {"#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
      "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
      "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173"};
  return colors[index % (sizeof(colors) / sizeof(colors[0]))];
}

template <typename Value>
void writeJsonArray(std::ostream& out, const std::vector<Value>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
}

void exportTrajectoryPlots(const std::vector<Session>& sessions,
                           const fs::path& run_root) {
  std::ofstream html(run_root / "trajectory_comparison.html");
  html << "<!doctype html><html><head><meta charset=\"utf-8\"><title>Global trajectory optimization</title>"
          "<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script></head>"
          "<body><div id=\"plot\" style=\"width:100vw;height:96vh\"></div><script>const d=[];\n";
  for (const auto& session : sessions) {
    std::vector<double> ox, oy, oz, nx, ny, nz, gx, gy, gz;
    for (const auto& key : session.keyframes) {
      ox.push_back(key.original.x()); oy.push_back(key.original.y()); oz.push_back(key.original.z());
      nx.push_back(key.optimized.x()); ny.push_back(key.optimized.y()); nz.push_back(key.optimized.z());
    }
    const std::size_t stride = std::max<std::size_t>(1, session.gnss.size() / 500);
    for (std::size_t i = 0; i < session.gnss.size(); i += stride) {
      gx.push_back(session.gnss[i].enu.x()); gy.push_back(session.gnss[i].enu.y());
      gz.push_back(session.gnss[i].enu.z());
    }
    const std::string dash = session.date == "20260803" ? "dash" : "solid";
    const std::string color = colorFor(session.index);
    auto trace = [&](const char* suffix, const auto& x, const auto& y, const auto& z,
                     const char* width, const std::string& line_color, const std::string& style) {
      html << "d.push({type:'scatter3d',mode:'lines',name:'" << session.id << " " << suffix
           << "',x:"; writeJsonArray(html, x); html << ",y:"; writeJsonArray(html, y);
      html << ",z:"; writeJsonArray(html, z);
      html << ",line:{color:'" << line_color << "',width:" << width << ",dash:'" << style << "'}});\n";
    };
    trace("original", ox, oy, oz, "2", "#aaaaaa", dash);
    trace("optimized", nx, ny, nz, "5", color, dash);
    html << "d.push({type:'scatter3d',mode:'markers',name:'" << session.id
         << " GNSS',x:"; writeJsonArray(html, gx); html << ",y:"; writeJsonArray(html, gy);
    html << ",z:"; writeJsonArray(html, gz);
    html << ",marker:{color:'" << color << "',size:1.5,opacity:0.30}});\n";
    html << "d.push({type:'scatter3d',mode:'markers',name:'" << session.id
         << " start/end',x:[" << nx.front() << ',' << nx.back() << "],y:["
         << ny.front() << ',' << ny.back() << "],z:[" << nz.front() << ',' << nz.back()
         << "],marker:{color:['#00ff00','#ff0000'],size:5,symbol:['circle','diamond']}});\n";
  }
  html << "d.push({type:'scatter3d',mode:'markers+text',name:'datum',x:[0],y:[0],z:[0],"
          "text:['global datum'],textposition:'top center',marker:{color:'#000000',size:8,symbol:'x'}});"
          "Plotly.newPlot('plot',d,{title:'Original, GNSS, and globally optimized trajectories',"
          "scene:{aspectmode:'data',xaxis:{title:'East (m)'},yaxis:{title:'North (m)'},zaxis:{title:'Up (m)'}},"
          "legend:{itemsizing:'constant'}},{responsive:true});</script></body></html>\n";

  double min_x = 0.0, max_x = 0.0, min_y = 0.0, max_y = 0.0;
  for (const auto& session : sessions) for (const auto& key : session.keyframes) {
    min_x = std::min({min_x, key.original.x(), key.optimized.x()});
    max_x = std::max({max_x, key.original.x(), key.optimized.x()});
    min_y = std::min({min_y, key.original.y(), key.optimized.y()});
    max_y = std::max({max_y, key.original.y(), key.optimized.y()});
  }
  constexpr double width = 1600.0, height = 1200.0, margin = 50.0;
  const double scale = std::min((width - 2 * margin) / std::max(1.0, max_x - min_x),
                                (height - 2 * margin) / std::max(1.0, max_y - min_y));
  auto sx = [&](double x) { return margin + (x - min_x) * scale; };
  auto sy = [&](double y) { return height - margin - (y - min_y) * scale; };
  std::ofstream svg(run_root / "trajectory_comparison.svg");
  svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1600\" height=\"1200\" viewBox=\"0 0 1600 1200\">"
         "<rect width=\"100%\" height=\"100%\" fill=\"white\"/><text x=\"50\" y=\"30\" font-size=\"22\">"
         "Original (gray), GNSS (dots), optimized (color); 0803 dashed, 0804 solid</text>";
  for (const auto& session : sessions) {
    auto polyline = [&](bool optimized, const std::string& color, double opacity) {
      svg << "<polyline fill=\"none\" stroke=\"" << color << "\" stroke-width=\""
          << (optimized ? 2.5 : 1.0) << "\" opacity=\"" << opacity << "\"";
      if (session.date == "20260803") svg << " stroke-dasharray=\"8,5\"";
      svg << " points=\"";
      for (const auto& key : session.keyframes) {
        const auto& pose = optimized ? key.optimized : key.original;
        svg << sx(pose.x()) << ',' << sy(pose.y()) << ' ';
      }
      svg << "\"/>";
    };
    polyline(false, "#999999", 0.45);
    polyline(true, colorFor(session.index), 0.95);
    const std::size_t stride = std::max<std::size_t>(1, session.gnss.size() / 250);
    for (std::size_t i = 0; i < session.gnss.size(); i += stride)
      svg << "<circle cx=\"" << sx(session.gnss[i].enu.x()) << "\" cy=\""
          << sy(session.gnss[i].enu.y()) << "\" r=\"1\" fill=\""
          << colorFor(session.index) << "\" opacity=\"0.25\"/>";
    const auto& start = session.keyframes.front().optimized;
    const auto& end = session.keyframes.back().optimized;
    svg << "<circle cx=\"" << sx(start.x()) << "\" cy=\"" << sy(start.y())
        << "\" r=\"4\" fill=\"#00aa00\"/><circle cx=\"" << sx(end.x())
        << "\" cy=\"" << sy(end.y()) << "\" r=\"4\" fill=\"#dd0000\"/>";
  }
  svg << "<path d=\"M " << sx(0) - 6 << ' ' << sy(0) << " h 12 M " << sx(0) << ' '
      << sy(0) - 6 << " v 12\" stroke=\"black\" stroke-width=\"3\"/>"
         "<text x=\"" << sx(0) + 8 << "\" y=\"" << sy(0) - 8
      << "\" font-size=\"14\">datum</text></svg>\n";
}

}  // namespace

OptimizationSummary optimizeGlobalGraph(
    std::vector<Session>* sessions, const std::vector<Anchor>& anchors,
    std::vector<Constraint>* constraints,
    const std::vector<ElevationConstraint>& elevation_constraints,
    const Config& config,
    const fs::path& run_root) {
  const auto anchor_globals = sessionAnchorGlobals(*sessions, anchors);
  gtsam::Values horizontal_initial = makeHorizontalInitial(*sessions);
  std::size_t horizontal_gnss = 0, horizontal_lidar = 0;
  auto horizontal_graph = makeHorizontalGraph(*sessions, anchors, *constraints,
      config, &horizontal_gnss, &horizontal_lidar);
  OptimizationSummary summary;
  gtsam::Values horizontal = solve(horizontal_graph, horizontal_initial);

  gtsam::Values vertical_initial = makeVerticalInitial(*sessions, anchors);
  std::size_t vertical_gnss = 0, vertical_lidar = 0, elevation_count = 0;
  auto vertical_graph = makeVerticalGraph(*sessions, anchors, *constraints,
      elevation_constraints, config, horizontal, &vertical_gnss,
      &vertical_lidar, &elevation_count);
  gtsam::Values vertical = solve(vertical_graph, vertical_initial);
  summary.initial_error = horizontal_graph.error(horizontal_initial) +
                          vertical_graph.error(vertical_initial);
  summary.first_error = horizontal_graph.error(horizontal) + vertical_graph.error(vertical);
  summary.gnss_factors = horizontal_gnss + vertical_gnss;
  summary.lidar_factors = horizontal_lidar;
  summary.elevation_factors = elevation_count;

  for (auto& c : *constraints) {
    if (!c.accepted || c.pruned) continue;
    const double residual = normalizedResidual(c, anchors[c.source_anchor],
                                               anchors[c.target_anchor], *sessions,
                                               horizontal, vertical);
    c.final_residual = residual;
    c.robust_weight = 1.0 / (1.0 + residual / (config.cauchy_scale * config.cauchy_scale));
    if (residual > config.chi2_6d_99) {
      c.pruned = true;
      c.reason = "pruned_chi2";
      ++summary.pruned_factors;
    }
  }
  if (summary.pruned_factors > 0) {
    horizontal_graph = makeHorizontalGraph(*sessions, anchors, *constraints,
        config, &horizontal_gnss, &horizontal_lidar);
    horizontal = solve(horizontal_graph, horizontal);
    vertical_graph = makeVerticalGraph(*sessions, anchors, *constraints,
        elevation_constraints, config, horizontal, &vertical_gnss,
        &vertical_lidar, &elevation_count);
    vertical = solve(vertical_graph, vertical);
  }
  for (auto& c : *constraints) {
    if (!c.accepted) continue;
    const double residual = normalizedResidual(c, anchors[c.source_anchor],
                                               anchors[c.target_anchor], *sessions,
                                               horizontal, vertical);
    c.final_residual = residual;
    c.robust_weight = 1.0 / (1.0 + residual / (config.cauchy_scale * config.cauchy_scale));
  }
  summary.final_error = horizontal_graph.error(horizontal) + vertical_graph.error(vertical);
  summary.gnss_factors = horizontal_gnss + vertical_gnss;
  summary.lidar_factors = horizontal_lidar;
  summary.elevation_factors = elevation_count;
  summary.converged = std::isfinite(summary.final_error) && summary.final_error <= summary.initial_error;
  summary.quality_passed = true;
  gtsam::Marginals horizontal_marginals(horizontal_graph, horizontal);
  gtsam::Marginals vertical_marginals(vertical_graph, vertical);
  for (auto& session : *sessions) {
    const auto horizontal_anchor = horizontal.at<gtsam::Pose2>(horizontalKey(session.index));
    session.z_bias = vertical.at<double>(B(session.index));
    session.vertical_corrections.clear();
    session.vertical_variances.clear();
    for (const auto global : anchor_globals[session.index]) {
      session.vertical_corrections.push_back(vertical.at<double>(verticalKey(global)));
      session.vertical_variances.push_back(
          vertical_marginals.marginalCovariance(verticalKey(global))(0, 0));
    }
    session.optimized_anchor = gtsam::Pose3(
        yawAdjustedRotation(session, 0, horizontal_anchor),
        gtsam::Point3(horizontal_anchor.x(), horizontal_anchor.y(),
            session.keyframes.front().original.z() + session.vertical_corrections.front()));
    session.rigid_correction = session.optimized_anchor.compose(
        session.keyframes.front().original.inverse());
    session.anchor_covariance.setZero();
    const auto hc = horizontal_marginals.marginalCovariance(horizontalKey(session.index));
    session.anchor_covariance(3, 3) = hc(0, 0);
    session.anchor_covariance(3, 4) = hc(0, 1);
    session.anchor_covariance(4, 3) = hc(1, 0);
    session.anchor_covariance(4, 4) = hc(1, 1);
    session.anchor_covariance(3, 2) = hc(0, 2);
    session.anchor_covariance(2, 3) = hc(2, 0);
    session.anchor_covariance(4, 2) = hc(1, 2);
    session.anchor_covariance(2, 4) = hc(2, 1);
    session.anchor_covariance(2, 2) = hc(2, 2);
    session.anchor_covariance(5, 5) = session.vertical_variances.front();
    double maximum_local_vertical = 0.0;
    for (const double correction : session.vertical_corrections)
      maximum_local_vertical = std::max(maximum_local_vertical,
          std::abs(correction - session.elevation_offset));
    if (std::abs(session.z_bias) > config.quality_max_z_bias ||
        maximum_local_vertical > config.vertical_max_local_correction ||
        session.elevation_offset_clamped) {
      summary.quality_passed = false;
    }
    double maximum_correction = 0.0;
    for (std::size_t i = 0; i < session.keyframes.size(); ++i) {
      const auto xy = horizontal_anchor.compose(relativePose2(session, i));
      session.keyframes[i].optimized = gtsam::Pose3(
          yawAdjustedRotation(session, i, horizontal_anchor),
          gtsam::Point3(xy.x(), xy.y(), session.keyframes[i].original.z() +
              interpolateAnchorValue(session, i, session.vertical_corrections)));
      if (!session.keyframes[i].optimized.matrix().array().isFinite().all()) {
        throw std::runtime_error("optimizer produced a non-finite pose for " + session.id);
      }
      maximum_correction = std::max(maximum_correction,
          (session.keyframes[i].optimized.translation() -
           session.keyframes[i].original.translation()).norm());
    }
    if (maximum_correction > config.quality_max_keyframe_correction)
      summary.quality_passed = false;
  }

  fs::create_directories(run_root / "internal");
  std::ofstream state(run_root / "internal" / "optimization_state.txt");
  state << "initial_error=" << summary.initial_error << "\nfirst_error=" << summary.first_error
        << "\nfinal_error=" << summary.final_error << "\n";
  state << "horizontal_error=" << horizontal_graph.error(horizontal)
        << "\nvertical_error=" << vertical_graph.error(vertical) << "\n";
  state << "quality_passed=" << summary.quality_passed << "\n";
  return summary;
}

void exportOptimizedResults(const std::vector<Session>& sessions,
                            const std::vector<Anchor>& anchors,
                            const std::vector<Constraint>& constraints,
                            const Config& config, const fs::path& run_root,
                            const OptimizationSummary& summary,
                            const std::vector<ValidationIssue>& excluded) {
  std::ofstream biases(run_root / "session_biases.csv");
  biases << "session,z_bias,elevation_prealignment,raw_elevation_prealignment,elevation_clamped,anchor_delta_x,anchor_delta_y,anchor_delta_z,anchor_delta_roll,anchor_delta_pitch,anchor_delta_yaw,max_local_vertical_deformation,lidar_supported,elevation_supported\n"
         << std::setprecision(17);
  for (const auto& session : sessions) {
    const fs::path output = run_root / "sessions" / session.date / session.time / session.bag;
    fs::create_directories(output);
    std::ofstream poses(output / "optimized_keyframe_pose.txt");
    poses << std::setprecision(17);
    for (const auto& key : session.keyframes) {
      const gtsam::Vector3 rpy = key.optimized.rotation().rpy();
      poses << key.stamp << ' ' << key.optimized.x() << ' ' << key.optimized.y() << ' '
            << key.optimized.z() << ' ' << rpy.x() << ' ' << rpy.y() << ' ' << rpy.z() << '\n';
    }
    const gtsam::Pose3 anchor_delta = session.keyframes.front().original.between(
        session.optimized_anchor);
    const gtsam::Vector3 rpy = anchor_delta.rotation().rpy();
    double maximum_local_vertical = 0.0;
    for (const double correction : session.vertical_corrections)
      maximum_local_vertical = std::max(maximum_local_vertical,
          std::abs(correction - session.elevation_offset));
    biases << session.id << ',' << session.z_bias << ',' << session.elevation_offset << ','
           << session.elevation_offset_raw << ',' << session.elevation_offset_clamped << ','
           << anchor_delta.x() << ',' << anchor_delta.y() << ',' << anchor_delta.z() << ','
           << rpy.x() << ',' << rpy.y() << ',' << rpy.z() << ','
           << maximum_local_vertical << ','
           << session.lidar_supported << ',' << session.elevation_supported << '\n';
  }

  for (const auto& session : sessions) {
    const fs::path output = run_root / "sessions" / session.date / session.time / session.bag;
    std::ofstream covariance(output / "keyframe_covariance.csv");
    covariance << "timestamp,var_x,var_y,var_z\n" << std::setprecision(17);
    for (std::size_t i = 0; i < session.keyframes.size(); ++i) {
      const Eigen::Matrix2d horizontal = keyframeHorizontalCovariance(session, i);
      const double vx = horizontal(0, 0), vy = horizontal(1, 1);
      const double vz = interpolateAnchorValue(session, i, session.vertical_variances);
      if (!std::isfinite(vx) || !std::isfinite(vy) || !std::isfinite(vz) ||
          vx < 0.0 || vy < 0.0 || vz < 0.0) {
        throw std::runtime_error("non-finite keyframe covariance for " + session.id);
      }
      covariance << session.keyframes[i].stamp << ',' << vx << ',' << vy << ',' << vz << '\n';
    }
  }

  auto global = std::make_shared<Cloud>();
  for (const auto& anchor : anchors) {
    const auto local = loadCloud(anchor.fine_path);
    Cloud placed;
    const auto& pose = sessions[anchor.session].keyframes[anchor.keyframe].optimized;
    pcl::transformPointCloud(*local, placed, pose.matrix().cast<float>());
    *global += placed;
  }
  pcl::VoxelGrid<pcl::PointXYZ> voxel;
  voxel.setLeafSize(config.preview_voxel, config.preview_voxel, config.preview_voxel);
  voxel.setInputCloud(global);
  Cloud preview;
  voxel.filter(preview);
  pcl::io::savePCDFileBinaryCompressed((run_root / "global_preview_0p5m.pcd").string(), preview);
  exportTrajectoryPlots(sessions, run_root);

  std::ofstream summary_file(run_root / "summary.json");
  std::map<std::string, std::size_t> rejection_reasons;
  for (const auto& constraint : constraints) {
    if (!constraint.accepted || constraint.pruned) ++rejection_reasons[constraint.reason];
  }
  std::vector<std::size_t> component(sessions.size());
  std::iota(component.begin(), component.end(), 0);
  const auto find_component = [&](std::size_t value, const auto& self) -> std::size_t {
    return component[value] == value ? value : (component[value] = self(component[value], self));
  };
  for (const auto& constraint : constraints) {
    if (!constraint.accepted || constraint.pruned) continue;
    const std::size_t a = find_component(anchors[constraint.source_anchor].session, find_component);
    const std::size_t b = find_component(anchors[constraint.target_anchor].session, find_component);
    if (a != b) component[b] = a;
  }
  std::set<std::size_t> unique_components;
  for (std::size_t i = 0; i < sessions.size(); ++i)
    unique_components.insert(find_component(i, find_component));
  summary_file << std::setprecision(17)
      << "{\n  \"converged\": " << (summary.converged ? "true" : "false")
      << ",\n  \"quality_passed\": " << (summary.quality_passed ? "true" : "false")
      << ",\n  \"sessions_included\": " << sessions.size()
      << ",\n  \"sessions_excluded\": " << excluded.size()
      << ",\n  \"initial_error\": " << summary.initial_error
      << ",\n  \"first_error\": " << summary.first_error
      << ",\n  \"final_error\": " << summary.final_error
      << ",\n  \"gnss_factors\": " << summary.gnss_factors
      << ",\n  \"elevation_factors\": " << summary.elevation_factors
      << ",\n  \"lidar_factors\": " << summary.lidar_factors
      << ",\n  \"pruned_lidar_factors\": " << summary.pruned_factors
      << ",\n  \"lidar_components\": " << unique_components.size()
      << ",\n  \"included\": [";
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    if (i) summary_file << ',';
    summary_file << "\n    \"" << jsonEscape(sessions[i].id) << "\"";
  }
  summary_file << "\n  ],\n  \"excluded\": [";
  for (std::size_t i = 0; i < excluded.size(); ++i) {
    if (i) summary_file << ',';
    summary_file << "\n    {\"path\":\"" << jsonEscape(excluded[i].path.string())
                 << "\",\"reason\":\"" << jsonEscape(excluded[i].reason) << "\"}";
  }
  summary_file << (excluded.empty() ? "" : "\n  ") << "],\n  \"rejection_reasons\": {";
  bool first_reason = true;
  for (const auto& [reason, count] : rejection_reasons) {
    summary_file << (first_reason ? "\n" : ",\n") << "    \"" << jsonEscape(reason)
                 << "\": " << count;
    first_reason = false;
  }
  summary_file << (first_reason ? "" : "\n  ") << "}"
      << ",\n  \"warnings\": [";
  bool first = true;
  for (const auto& session : sessions) {
    std::string warning;
    const auto anchor_delta = session.keyframes.front().original.between(
        session.optimized_anchor);
    double maximum_correction = 0.0;
    for (const auto& key : session.keyframes)
      maximum_correction = std::max(maximum_correction,
          (key.optimized.translation() - key.original.translation()).norm());
    if (session.elevation_offset_clamped)
      warning = session.id + ": elevation prealignment exceeded its safe limit";
    else if (std::abs(session.z_bias) > config.quality_max_z_bias)
      warning = session.id + ": z bias exceeds quality limit";
    double maximum_local_vertical = 0.0;
    for (const double correction : session.vertical_corrections)
      maximum_local_vertical = std::max(maximum_local_vertical,
          std::abs(correction - session.elevation_offset));
    if (warning.empty() && maximum_local_vertical > config.vertical_max_local_correction)
      warning = session.id + ": local vertical deformation exceeds quality limit";
    else if (warning.empty() && maximum_correction > config.quality_max_keyframe_correction)
      warning = session.id + ": keyframe correction exceeds quality limit";
    else if (warning.empty() && !session.lidar_supported)
      warning = session.id + ": no accepted cross-session LiDAR support";
    const double xy = anchor_delta.translation().head<2>().norm();
    if (warning.empty() && xy > config.warn_xy_correction) warning = session.id + ": rigid X/Y correction exceeds threshold";
    if (!warning.empty()) {
      summary_file << (first ? "\n" : ",\n") << "    \"" << jsonEscape(warning) << "\"";
      first = false;
    }
  }
  if (unique_components.size() > 1) {
    summary_file << (first ? "\n" : ",\n") << "    \"LiDAR overlap graph has "
                 << unique_components.size() << " disconnected components\"";
    first = false;
  }
  summary_file << (first ? "" : "\n  ") << "]\n}\n";
}

}  // namespace fast_lio_sam::offline
