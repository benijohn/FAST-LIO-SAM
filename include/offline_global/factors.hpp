#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace fast_lio_sam::offline {

class HorizontalPositionFactor final
    : public gtsam::NoiseModelFactor1<gtsam::Pose2> {
 public:
  HorizontalPositionFactor(gtsam::Key key, const gtsam::Point2& relative,
      const gtsam::Point2& measured, const gtsam::SharedNoiseModel& model)
      : NoiseModelFactor1(model, key), relative_(relative), measured_(measured) {}
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<HorizontalPositionFactor>(*this);
  }
  gtsam::Vector evaluateError(const gtsam::Pose2& anchor,
      boost::optional<gtsam::Matrix&> H = boost::none) const override {
    auto function = [&](const gtsam::Pose2& value) -> gtsam::Vector2 {
      const gtsam::Point2 predicted = value.transformFrom(relative_);
      return predicted - measured_;
    };
    const gtsam::Vector2 error = function(anchor);
    if (H) {
      constexpr double epsilon = 1e-6;
      H->resize(2, 3);
      for (int i = 0; i < 3; ++i) {
        gtsam::Vector3 delta = gtsam::Vector3::Zero(); delta[i] = epsilon;
        H->col(i) = (function(anchor.retract(delta)) -
                     function(anchor.retract(-delta))) / (2.0 * epsilon);
      }
    }
    return error;
  }
 private:
  gtsam::Point2 relative_, measured_;
};

class HorizontalAlignmentFactor final
    : public gtsam::NoiseModelFactor2<gtsam::Pose2, gtsam::Pose2> {
 public:
  HorizontalAlignmentFactor(gtsam::Key a, gtsam::Key b,
      const gtsam::Pose2& relative_a, const gtsam::Pose2& relative_b,
      const gtsam::Pose2& measured, const gtsam::SharedNoiseModel& model)
      : NoiseModelFactor2(model, a, b), relative_a_(relative_a),
        relative_b_(relative_b), measured_(measured) {}
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<HorizontalAlignmentFactor>(*this);
  }
  gtsam::Vector evaluateError(const gtsam::Pose2& a, const gtsam::Pose2& b,
      boost::optional<gtsam::Matrix&> H_a = boost::none,
      boost::optional<gtsam::Matrix&> H_b = boost::none) const override {
    auto function = [&](const gtsam::Pose2& va, const gtsam::Pose2& vb) {
      const auto predicted = va.compose(relative_a_).between(vb.compose(relative_b_));
      return gtsam::Pose2::Logmap(measured_.between(predicted));
    };
    const gtsam::Vector3 error = function(a, b);
    constexpr double epsilon = 1e-6;
    if (H_a) {
      H_a->resize(3, 3);
      for (int i = 0; i < 3; ++i) {
        gtsam::Vector3 delta = gtsam::Vector3::Zero(); delta[i] = epsilon;
        H_a->col(i) = (function(a.retract(delta), b) -
                       function(a.retract(-delta), b)) / (2.0 * epsilon);
      }
    }
    if (H_b) {
      H_b->resize(3, 3);
      for (int i = 0; i < 3; ++i) {
        gtsam::Vector3 delta = gtsam::Vector3::Zero(); delta[i] = epsilon;
        H_b->col(i) = (function(a, b.retract(delta)) -
                       function(a, b.retract(-delta))) / (2.0 * epsilon);
      }
    }
    return error;
  }
 private:
  gtsam::Pose2 relative_a_, relative_b_, measured_;
};

class HeightBiasFactor final
    : public gtsam::NoiseModelFactor2<double, double> {
 public:
  HeightBiasFactor(gtsam::Key correction, gtsam::Key bias,
      double original_antenna_z, double measured_z,
      const gtsam::SharedNoiseModel& model)
      : NoiseModelFactor2(model, correction, bias),
        original_(original_antenna_z), measured_(measured_z) {}
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<HeightBiasFactor>(*this);
  }
  gtsam::Vector evaluateError(const double& correction, const double& bias,
      boost::optional<gtsam::Matrix&> H_c = boost::none,
      boost::optional<gtsam::Matrix&> H_b = boost::none) const override {
    if (H_c) *H_c = gtsam::Matrix::Ones(1, 1);
    if (H_b) *H_b = gtsam::Matrix::Ones(1, 1);
    gtsam::Vector1 error; error << original_ + correction + bias - measured_;
    return error;
  }
 private:
  double original_, measured_;
};

class SecondDifferenceFactor final
    : public gtsam::NoiseModelFactor3<double, double, double> {
 public:
  SecondDifferenceFactor(gtsam::Key previous, gtsam::Key current,
      gtsam::Key next, const gtsam::SharedNoiseModel& model)
      : NoiseModelFactor3(model, previous, current, next) {}
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<SecondDifferenceFactor>(*this);
  }
  gtsam::Vector evaluateError(const double& previous, const double& current,
      const double& next, boost::optional<gtsam::Matrix&> H_p = boost::none,
      boost::optional<gtsam::Matrix&> H_c = boost::none,
      boost::optional<gtsam::Matrix&> H_n = boost::none) const override {
    if (H_p) *H_p = gtsam::Matrix::Ones(1, 1);
    if (H_c) *H_c = -2.0 * gtsam::Matrix::Ones(1, 1);
    if (H_n) *H_n = gtsam::Matrix::Ones(1, 1);
    gtsam::Vector1 error; error << previous - 2.0 * current + next;
    return error;
  }
};

class AntennaPositionBiasFactor final
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, double> {
 public:
  AntennaPositionBiasFactor(gtsam::Key pose_key, gtsam::Key bias_key,
                            const gtsam::Point3& measured,
                            const gtsam::Point3& lever,
                            const gtsam::SharedNoiseModel& model)
      : NoiseModelFactor2(model, pose_key, bias_key), measured_(measured), lever_(lever) {}

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<AntennaPositionBiasFactor>(*this);
  }

  gtsam::Vector evaluateError(const gtsam::Pose3& pose, const double& bias,
      boost::optional<gtsam::Matrix&> H_pose = boost::none,
      boost::optional<gtsam::Matrix&> H_bias = boost::none) const override {
    gtsam::Matrix36 pose_jacobian;
    const gtsam::Point3 predicted = pose.transformFrom(lever_, pose_jacobian);
    if (H_pose) *H_pose = pose_jacobian;
    if (H_bias) {
      *H_bias = gtsam::Matrix::Zero(3, 1);
      (*H_bias)(2, 0) = 1.0;
    }
    gtsam::Vector3 error = predicted - measured_;
    error.z() += bias;
    return error;
  }

 private:
  gtsam::Point3 measured_;
  gtsam::Point3 lever_;
};

class SessionAlignmentFactor final
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
 public:
  SessionAlignmentFactor(gtsam::Key a, gtsam::Key b,
                         const gtsam::Pose3& base_a,
                         const gtsam::Pose3& base_b,
                         const gtsam::Pose3& measured,
                         const gtsam::SharedNoiseModel& model)
      : NoiseModelFactor2(model, a, b), base_a_(base_a), base_b_(base_b),
        measured_(measured) {}

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<SessionAlignmentFactor>(*this);
  }

  gtsam::Vector evaluateError(const gtsam::Pose3& a, const gtsam::Pose3& b,
      boost::optional<gtsam::Matrix&> H_a = boost::none,
      boost::optional<gtsam::Matrix&> H_b = boost::none) const override {
    auto function = [&](const gtsam::Pose3& ca, const gtsam::Pose3& cb) {
      const auto predicted = ca.compose(base_a_).between(cb.compose(base_b_));
      return gtsam::Pose3::Logmap(measured_.between(predicted));
    };
    const gtsam::Vector6 error = function(a, b);
    constexpr double epsilon = 1e-6;
    if (H_a) {
      H_a->resize(6, 6);
      for (int i = 0; i < 6; ++i) {
        gtsam::Vector6 delta = gtsam::Vector6::Zero(); delta[i] = epsilon;
        H_a->col(i) = (function(a.retract(delta), b) - function(a.retract(-delta), b)) /
                      (2.0 * epsilon);
      }
    }
    if (H_b) {
      H_b->resize(6, 6);
      for (int i = 0; i < 6; ++i) {
        gtsam::Vector6 delta = gtsam::Vector6::Zero(); delta[i] = epsilon;
        H_b->col(i) = (function(a, b.retract(delta)) - function(a, b.retract(-delta))) /
                      (2.0 * epsilon);
      }
    }
    return error;
  }

 private:
  gtsam::Pose3 base_a_, base_b_, measured_;
};

class RigidAntennaPositionBiasFactor final
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, double> {
 public:
  RigidAntennaPositionBiasFactor(gtsam::Key anchor_key, gtsam::Key bias_key,
      const gtsam::Pose3& anchor_to_keyframe, const gtsam::Point3& measured,
      const gtsam::Point3& lever, const gtsam::SharedNoiseModel& model)
      : NoiseModelFactor2(model, anchor_key, bias_key),
        relative_(anchor_to_keyframe), measured_(measured), lever_(lever) {}

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<RigidAntennaPositionBiasFactor>(*this);
  }

  gtsam::Vector evaluateError(const gtsam::Pose3& anchor, const double& bias,
      boost::optional<gtsam::Matrix&> H_anchor = boost::none,
      boost::optional<gtsam::Matrix&> H_bias = boost::none) const override {
    auto function = [&](const gtsam::Pose3& value, double z_bias) {
      gtsam::Vector3 error = value.compose(relative_).transformFrom(lever_) - measured_;
      error.z() += z_bias;
      return error;
    };
    const gtsam::Vector3 error = function(anchor, bias);
    constexpr double epsilon = 1e-6;
    if (H_anchor) {
      H_anchor->resize(3, 6);
      for (int i = 0; i < 6; ++i) {
        gtsam::Vector6 delta = gtsam::Vector6::Zero(); delta[i] = epsilon;
        H_anchor->col(i) = (function(anchor.retract(delta), bias) -
            function(anchor.retract(-delta), bias)) / (2.0 * epsilon);
      }
    }
    if (H_bias) {
      *H_bias = gtsam::Matrix::Zero(3, 1);
      (*H_bias)(2, 0) = 1.0;
    }
    return error;
  }

 private:
  gtsam::Pose3 relative_;
  gtsam::Point3 measured_;
  gtsam::Point3 lever_;
};

class RigidTerrainHeightFactor final
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
 public:
  RigidTerrainHeightFactor(gtsam::Key anchor_a, gtsam::Key anchor_b,
      const gtsam::Pose3& relative_a, const gtsam::Pose3& relative_b,
      const gtsam::SharedNoiseModel& model)
      : NoiseModelFactor2(model, anchor_a, anchor_b),
        relative_a_(relative_a), relative_b_(relative_b) {}

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<RigidTerrainHeightFactor>(*this);
  }

  gtsam::Vector evaluateError(const gtsam::Pose3& anchor_a,
      const gtsam::Pose3& anchor_b,
      boost::optional<gtsam::Matrix&> H_a = boost::none,
      boost::optional<gtsam::Matrix&> H_b = boost::none) const override {
    auto function = [&](const gtsam::Pose3& a, const gtsam::Pose3& b) {
      gtsam::Vector1 error;
      error << a.compose(relative_a_).z() - b.compose(relative_b_).z();
      return error;
    };
    const gtsam::Vector1 error = function(anchor_a, anchor_b);
    constexpr double epsilon = 1e-6;
    if (H_a) {
      H_a->resize(1, 6);
      for (int i = 0; i < 6; ++i) {
        gtsam::Vector6 delta = gtsam::Vector6::Zero(); delta[i] = epsilon;
        H_a->col(i) = (function(anchor_a.retract(delta), anchor_b) -
            function(anchor_a.retract(-delta), anchor_b)) / (2.0 * epsilon);
      }
    }
    if (H_b) {
      H_b->resize(1, 6);
      for (int i = 0; i < 6; ++i) {
        gtsam::Vector6 delta = gtsam::Vector6::Zero(); delta[i] = epsilon;
        H_b->col(i) = (function(anchor_a, anchor_b.retract(delta)) -
            function(anchor_a, anchor_b.retract(-delta))) / (2.0 * epsilon);
      }
    }
    return error;
  }

 private:
  gtsam::Pose3 relative_a_;
  gtsam::Pose3 relative_b_;
};

}  // namespace fast_lio_sam::offline
