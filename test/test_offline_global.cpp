#include "offline_global/factors.hpp"
#include "offline_global/elevation.hpp"
#include "offline_global/io.hpp"
#include "offline_global/trajectory_refinement.hpp"

#include <gtest/gtest.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

using namespace fast_lio_sam::offline;

TEST(OfflineGlobal, DatumMapsToEnuOrigin) {
  Datum datum;
  const Eigen::Vector3d enu = wgs84ToEnu(datum.latitude, datum.longitude,
                                         datum.altitude, datum);
  EXPECT_LT(enu.norm(), 1e-6);
  const Eigen::Vector3d east = wgs84ToEnu(datum.latitude,
      datum.longitude + 1e-5, datum.altitude, datum);
  EXPECT_GT(east.x(), 0.8);
  EXPECT_NEAR(east.y(), 0.0, 0.05);
}

TEST(OfflineGlobal, LidarBodyWorldConvention) {
  const Eigen::Matrix3d lidar_to_body =
      Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Vector3d lidar_to_body_t(1.0, 2.0, 3.0);
  const gtsam::Pose3 world_from_body(gtsam::Rot3::Rz(M_PI_2),
                                     gtsam::Point3(10.0, 20.0, 30.0));
  const Eigen::Vector3d lidar_point(1.0, 0.0, 0.0);
  const Eigen::Vector3d body = lidar_to_body * lidar_point + lidar_to_body_t;
  const Eigen::Vector3d world = world_from_body.transformFrom(gtsam::Point3(body));
  EXPECT_NEAR(world.x(), 7.0, 1e-9);
  EXPECT_NEAR(world.y(), 21.0, 1e-9);
  EXPECT_NEAR(world.z(), 33.0, 1e-9);
}

TEST(OfflineGlobal, AntennaFactorAndJacobian) {
  const gtsam::Pose3 pose(gtsam::Rot3::RzRyRx(0.1, -0.2, 0.3),
                          gtsam::Point3(3.0, 4.0, 5.0));
  const gtsam::Point3 lever(1.2, -0.4, 0.8);
  const double bias = 1.7;
  gtsam::Point3 measured = pose.transformFrom(lever);
  measured.z() += bias;
  AntennaPositionBiasFactor factor(0, 1, measured, lever,
      gtsam::noiseModel::Isotropic::Sigma(3, 1.0));
  gtsam::Matrix H_pose, H_bias;
  const auto error = factor.evaluateError(pose, bias, H_pose, H_bias);
  EXPECT_NEAR(error.x(), 0.0, 1e-12);
  EXPECT_NEAR(error.y(), 0.0, 1e-12);
  EXPECT_NEAR(error.z(), 0.0, 1e-12);
  const auto numeric_pose = gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose3>(
      [&](const gtsam::Pose3& p) { return factor.evaluateError(p, bias); }, pose);
  EXPECT_LT((H_pose - numeric_pose).norm(), 1e-5);
  EXPECT_NEAR(H_bias(2, 0), 1.0, 1e-12);
}

TEST(OfflineGlobal, SessionAlignmentKnownCorrection) {
  const gtsam::Pose3 base_a(gtsam::Rot3(), gtsam::Point3(0, 0, 0));
  const gtsam::Pose3 base_b(gtsam::Rot3::Rz(0.1), gtsam::Point3(10, 0, 1));
  const gtsam::Pose3 correction_a(gtsam::Rot3::Rz(0.02), gtsam::Point3(1, 2, 3));
  const gtsam::Pose3 correction_b(gtsam::Rot3::Rz(-0.01), gtsam::Point3(-1, 2, 2));
  const auto measured = correction_a.compose(base_a).between(correction_b.compose(base_b));
  SessionAlignmentFactor factor(0, 1, base_a, base_b, measured,
      gtsam::noiseModel::Isotropic::Sigma(6, 1.0));
  EXPECT_LT(factor.evaluateError(correction_a, correction_b).norm(), 1e-9);
}

TEST(OfflineGlobal, HorizontalFactorsMatchKnownAlignment) {
  const gtsam::Pose2 anchor_a(10.0, 20.0, 0.2);
  const gtsam::Pose2 anchor_b(12.0, 19.0, -0.1);
  const gtsam::Pose2 relative_a(4.0, 1.0, 0.05);
  const gtsam::Pose2 relative_b(-2.0, 3.0, -0.03);
  const auto measured = anchor_a.compose(relative_a).between(
      anchor_b.compose(relative_b));
  HorizontalAlignmentFactor alignment(0, 1, relative_a, relative_b, measured,
      gtsam::noiseModel::Isotropic::Sigma(3, 1.0));
  EXPECT_LT(alignment.evaluateError(anchor_a, anchor_b).norm(), 1e-10);

  const gtsam::Point2 antenna_relative(1.2, -0.4);
  const auto antenna_world = anchor_a.transformFrom(antenna_relative);
  HorizontalPositionFactor position(0, antenna_relative, antenna_world,
      gtsam::noiseModel::Isotropic::Sigma(2, 1.0));
  gtsam::Matrix jacobian;
  EXPECT_LT(position.evaluateError(anchor_a, jacobian).norm(), 1e-10);
  const auto numeric = gtsam::numericalDerivative11<gtsam::Vector2, gtsam::Pose2>(
      [&](const gtsam::Pose2& p) -> gtsam::Vector2 {
        return position.evaluateError(p);
      }, anchor_a);
  EXPECT_LT((jacobian - numeric).norm(), 1e-5);
}

TEST(OfflineGlobal, VerticalBiasAndSmoothnessFactors) {
  HeightBiasFactor height(0, 1, 100.0, 103.0,
      gtsam::noiseModel::Isotropic::Sigma(1, 1.0));
  EXPECT_NEAR(height.evaluateError(1.0, 2.0)[0], 0.0, 1e-12);
  gtsam::Matrix h_correction, h_bias;
  height.evaluateError(1.0, 2.0, h_correction, h_bias);
  EXPECT_NEAR(h_correction(0, 0), 1.0, 1e-12);
  EXPECT_NEAR(h_bias(0, 0), 1.0, 1e-12);

  SecondDifferenceFactor smooth(0, 1, 2,
      gtsam::noiseModel::Isotropic::Sigma(1, 1.0));
  EXPECT_NEAR(smooth.evaluateError(1.0, 1.5, 2.0)[0], 0.0, 1e-12);
  EXPECT_NEAR(smooth.evaluateError(1.0, 1.4, 2.0)[0], 0.2, 1e-12);
}

TEST(OfflineGlobal, Se3CorrectionInterpolationPreservesEndpoints) {
  const gtsam::Pose3 a(gtsam::Rot3::Rz(0.2), gtsam::Point3(1, 2, 3));
  const gtsam::Pose3 b(gtsam::Rot3::Rz(-0.3), gtsam::Point3(4, 6, 8));
  const auto interpolate = [&](double alpha) {
    return a.compose(gtsam::Pose3::Expmap(alpha * gtsam::Pose3::Logmap(a.between(b))));
  };
  EXPECT_LT(gtsam::Pose3::Logmap(a.between(interpolate(0.0))).norm(), 1e-12);
  EXPECT_LT(gtsam::Pose3::Logmap(b.between(interpolate(1.0))).norm(), 1e-12);
}

TEST(OfflineGlobal, CauchyLimitsFalseConstraintDamage) {
  const gtsam::Key x0 = gtsam::Symbol('x', 0), x1 = gtsam::Symbol('x', 1);
  gtsam::NonlinearFactorGraph graph;
  graph.add(gtsam::PriorFactor<gtsam::Pose3>(x0, gtsam::Pose3(),
      gtsam::noiseModel::Isotropic::Sigma(6, 0.01)));
  graph.add(gtsam::BetweenFactor<gtsam::Pose3>(x0, x1,
      gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(10, 0, 0)),
      gtsam::noiseModel::Isotropic::Sigma(6, 0.05)));
  const auto false_base = gtsam::noiseModel::Isotropic::Sigma(6, 0.05);
  const auto robust = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Cauchy::Create(2.0), false_base);
  graph.add(gtsam::BetweenFactor<gtsam::Pose3>(x0, x1,
      gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(100, 0, 0)), robust));
  gtsam::Values initial;
  initial.insert(x0, gtsam::Pose3());
  initial.insert(x1, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(9, 0, 0)));
  const auto result = gtsam::LevenbergMarquardtOptimizer(graph, initial).optimize();
  EXPECT_NEAR(result.at<gtsam::Pose3>(x1).x(), 10.0, 0.1);
}

TEST(OfflineGlobal, SessionZBiasAlignsRunsAndPreservesUndulation) {
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values initial;
  const auto tight = gtsam::noiseModel::Isotropic::Sigma(6, 0.02);
  const auto gps = gtsam::noiseModel::Isotropic::Sigma(3, 0.25);
  const gtsam::Point3 zero_lever(0, 0, 0);
  const gtsam::Key bias_a = gtsam::Symbol('b', 0), bias_b = gtsam::Symbol('b', 1);
  initial.insert(bias_a, 0.0); initial.insert(bias_b, 0.0);
  graph.add(gtsam::PriorFactor<double>(bias_a, 0.0,
      gtsam::noiseModel::Isotropic::Sigma(1, 3.0)));
  graph.add(gtsam::PriorFactor<double>(bias_b, 0.0,
      gtsam::noiseModel::Isotropic::Sigma(1, 3.0)));
  std::vector<double> terrain = {0.0, 0.4, -0.1};
  for (int session = 0; session < 2; ++session) {
    for (int i = 0; i < 3; ++i) {
      const gtsam::Key key = gtsam::Symbol(session == 0 ? 'x' : 'y', i);
      initial.insert(key, gtsam::Pose3(gtsam::Rot3(),
          gtsam::Point3(5.0 * i, 0.0, terrain[i] + (session ? 2.0 : 0.0))));
      graph.emplace_shared<AntennaPositionBiasFactor>(key,
          session ? bias_b : bias_a,
          gtsam::Point3(5.0 * i, 0.0, terrain[i] + (session ? 2.0 : 0.0)),
          zero_lever, gps);
      if (i > 0) {
        const gtsam::Key previous = gtsam::Symbol(session == 0 ? 'x' : 'y', i - 1);
        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(previous, key,
            gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(5.0, 0.0,
                terrain[i] - terrain[i - 1])), tight));
      }
      if (session == 1) {
        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::Symbol('x', i), key,
            gtsam::Pose3(), tight));
      }
    }
  }
  graph.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', 0), gtsam::Pose3(), tight));
  const auto result = gtsam::LevenbergMarquardtOptimizer(graph, initial).optimize();
  EXPECT_NEAR(result.at<double>(bias_b), 2.0, 0.15);
  const auto y0 = result.at<gtsam::Pose3>(gtsam::Symbol('y', 0));
  const auto y1 = result.at<gtsam::Pose3>(gtsam::Symbol('y', 1));
  EXPECT_NEAR(y0.z(), 0.0, 0.1);
  EXPECT_NEAR(y1.z() - y0.z(), 0.4, 0.02);
}

TEST(OfflineGlobal, SessionAnchorRotationDoesNotSwingFirstPoseAroundDatum) {
  const gtsam::Pose3 original_anchor(gtsam::Rot3(),
      gtsam::Point3(-600.0, -350.0, -40.0));
  const gtsam::Pose3 optimized_anchor(gtsam::Rot3::Ry(0.1),
      original_anchor.translation());
  const gtsam::Pose3 original_later(gtsam::Rot3(),
      gtsam::Point3(-500.0, -350.0, -40.0));
  const auto relative = original_anchor.between(original_later);
  const auto optimized_first = optimized_anchor.compose(gtsam::Pose3());
  const auto optimized_later = optimized_anchor.compose(relative);
  EXPECT_LT((optimized_first.translation() - original_anchor.translation()).norm(), 1e-12);
  EXPECT_NEAR(optimized_later.z() - original_later.z(), -100.0 * std::sin(0.1), 1e-9);
  EXPECT_LT(std::abs(optimized_later.z() - original_later.z()), 11.0);
}

TEST(OfflineGlobal, ElevationPrealignmentRecoversSharedTrackOffset) {
  std::vector<Session> sessions(2);
  for (int s = 0; s < 2; ++s) {
    sessions[s].index = s;
    sessions[s].id = "synthetic_" + std::to_string(s);
    for (int i = 0; i < 31; ++i) {
      Keyframe key;
      key.stamp = i;
      key.travel = i;
      const double terrain = 0.2 * std::sin(i * 0.2);
      key.original = gtsam::Pose3(gtsam::Rot3(),
          gtsam::Point3(i, 0.0, terrain + (s == 1 ? 2.0 : 0.0)));
      sessions[s].keyframes.push_back(key);
    }
  }
  Config config;
  config.elevation_xy_radius = 0.5;
  const auto output = std::filesystem::temp_directory_path() /
      "fast_lio_elevation_unit_test";
  std::filesystem::create_directories(output);
  const auto constraints = estimateElevationPrealignment(&sessions, config, output);
  ASSERT_FALSE(constraints.empty());
  EXPECT_TRUE(std::any_of(constraints.begin(), constraints.end(),
      [](const auto& c) { return c.kind == "overlap"; }));
  EXPECT_NEAR((2.0 + sessions[1].elevation_offset) - sessions[0].elevation_offset,
              0.0, 0.15);
}

TEST(OfflineGlobal, TerrainCrossingFactorAnchorsEqualHeight) {
  const gtsam::Pose3 anchor_a(gtsam::Rot3(), gtsam::Point3(0, 0, 1.0));
  const gtsam::Pose3 anchor_b(gtsam::Rot3(), gtsam::Point3(0, 0, 3.0));
  RigidTerrainHeightFactor factor(0, 1, gtsam::Pose3(), gtsam::Pose3(),
      gtsam::noiseModel::Isotropic::Sigma(1, 0.5));
  EXPECT_NEAR(factor.evaluateError(anchor_a, anchor_b)[0], -2.0, 1e-12);
  gtsam::Matrix h_a, h_b;
  factor.evaluateError(anchor_a, anchor_b, h_a, h_b);
  EXPECT_NEAR(h_a(0, 5), 1.0, 1e-6);
  EXPECT_NEAR(h_b(0, 5), -1.0, 1e-6);
}

TEST(OfflineGlobal, HighrateRefinementHitsEveryTargetAnchor) {
  std::vector<TimedPose> raw;
  for (int i = 0; i <= 20; ++i) {
    const double time = 0.1 * i;
    raw.push_back({time, gtsam::Pose3(gtsam::Rot3::Rz(0.05 * time),
                                      gtsam::Point3(time, 0.2 * time, 0.0))});
  }
  std::vector<TimedPose> target;
  for (int i = 0; i <= 2; ++i) {
    const auto& sample = raw[10 * i];
    const gtsam::Pose3 correction(gtsam::Rot3::Rz(0.02 * i),
                                  gtsam::Point3(0.2 * i, -0.1 * i, 0.4 * i));
    target.push_back({sample.stamp, correction.compose(sample.pose)});
  }
  TrajectoryRefinementConfig config;
  const auto refined = refineTrajectory(raw, target, config);
  ASSERT_EQ(refined.refined.size(), raw.size());
  EXPECT_TRUE(refined.quality_passed);
  EXPECT_LT(refined.maximum_anchor_translation_error, 1e-10);
  EXPECT_LT(refined.maximum_anchor_rotation_error_deg, 1e-8);
  for (int i = 0; i <= 2; ++i) {
    const auto error = target[i].pose.between(refined.refined[10 * i].pose);
    EXPECT_LT(gtsam::Pose3::Logmap(error).norm(), 1e-9);
  }
}

TEST(OfflineGlobal, ConstantCorrectionPreservesRawRelativeMotion) {
  std::vector<TimedPose> raw;
  for (int i = 0; i <= 10; ++i) {
    raw.push_back({static_cast<double>(i),
        gtsam::Pose3(gtsam::Rot3::Ry(0.01 * i),
                     gtsam::Point3(i, std::sin(0.2 * i), 0.1 * i))});
  }
  const gtsam::Pose3 correction(gtsam::Rot3::Rz(0.3),
                                gtsam::Point3(5.0, -2.0, 1.0));
  const std::vector<TimedPose> target = {
      {raw.front().stamp, correction.compose(raw.front().pose)},
      {raw.back().stamp, correction.compose(raw.back().pose)}};
  TrajectoryRefinementConfig config;
  config.max_odom_gap = 2.0;
  config.max_keyframe_bracket_gap = 2.0;
  const auto refined = refineTrajectory(raw, target, config);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const auto expected = correction.compose(raw[i].pose);
    EXPECT_LT(gtsam::Pose3::Logmap(expected.between(refined.refined[i].pose)).norm(),
              1e-9);
  }
  const auto raw_relative = raw[3].pose.between(raw[7].pose);
  const auto refined_relative = refined.refined[3].pose.between(refined.refined[7].pose);
  EXPECT_LT(gtsam::Pose3::Logmap(raw_relative.between(refined_relative)).norm(), 1e-9);
}

TEST(OfflineGlobal, LocalIntervalAdjustmentIsInvariantToDistantOrigin) {
  const std::vector<TimedPose> raw = {
      {0.0, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(1000.0, 500.0, 0.0))},
      {0.5, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(1000.5, 500.0, 0.0))},
      {1.0, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(1001.0, 500.0, 0.0))}};
  const std::vector<TimedPose> target = {
      {0.0, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.0, 0.0, 0.0))},
      {1.0, gtsam::Pose3(gtsam::Rot3::Rz(0.01), gtsam::Point3(1.0, 0.0, 0.0))}};
  TrajectoryRefinementConfig config;
  config.max_odom_gap = 1.0;
  config.max_keyframe_bracket_gap = 1.0;
  const auto refined = refineTrajectory(raw, target, config);
  EXPECT_TRUE(refined.quality_passed);
  EXPECT_LT(refined.maximum_interval_adjustment_translation, 1e-9);
  EXPECT_NEAR(refined.maximum_interval_adjustment_rotation_deg,
              0.01 * 180.0 / M_PI, 1e-9);
  EXPECT_LT((refined.refined[1].pose.translation() -
             gtsam::Point3(0.5, 0.0, 0.0)).norm(), 0.01);
}

TEST(OfflineGlobal, SingletonSessionUsesConstantCorrection) {
  const std::vector<TimedPose> raw = {
      {0.0, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0, 0, 0))},
      {1.0, gtsam::Pose3(gtsam::Rot3::Rz(0.1), gtsam::Point3(1, 0, 0))},
      {2.0, gtsam::Pose3(gtsam::Rot3::Rz(0.2), gtsam::Point3(2, 0, 0))}};
  const gtsam::Pose3 correction(gtsam::Rot3::Rz(-0.2),
                                gtsam::Point3(4, 5, 6));
  const std::vector<TimedPose> target = {
      {1.0, correction.compose(raw[1].pose)}};
  TrajectoryRefinementConfig config;
  config.max_odom_gap = 2.0;
  const auto refined = refineTrajectory(raw, target, config);
  ASSERT_EQ(refined.knots.size(), 1u);
  EXPECT_TRUE(refined.quality_passed);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const auto expected = correction.compose(raw[i].pose);
    EXPECT_LT(gtsam::Pose3::Logmap(expected.between(refined.refined[i].pose)).norm(),
              1e-9);
  }
}
