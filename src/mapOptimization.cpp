// Modified from LIO-SAM: MapOptimization.cpp

#include "utility.h"
#include <pcl/registration/ndt.h>

#include <chrono>
#include "common_utils.h"
#include "GNSS_Processing.hpp"
#include "gnssYaw_factor.h"
#include "map_optimization.h"
#include "ros_utils.h"

#include <cmath>
#include <algorithm>
#include <fstream>
#include <limits>
#include <tuple>
#include <utility>
#include <unordered_set>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <pcl/common/transforms.h>
#include <pcl/common/point_tests.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "ikd-Tree/ikdtree_public.h"

using namespace gtsam;
using namespace std;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

extern std::string root_dir;

// gtsam
NonlinearFactorGraph gtSAMgraph;
Values initialEstimate;
ISAM2 *isam;
Values isamCurrentEstimate;
Eigen::MatrixXd poseCovariance;

Pcl2Publisher pubKeyPoses;
PathPublisher pubPath;

Pcl2Publisher pubLaserCloudGlobal;
Pcl2Publisher pubLaserCloudLocal;

Pcl2Publisher pubHistoryKeyFrames;
Pcl2Publisher pubIcpKeyFrames;
Pcl2Publisher pubLoopDebugSource;
Pcl2Publisher pubLoopDebugTarget;
Pcl2Publisher pubLoopDebugNdtAligned;
Pcl2Publisher pubLoopDebugIcpAligned;
Pcl2Publisher pubRecentKeyFrame;
Pcl2Publisher pubCloudRegisteredRaw;
MarkerArrayPublisher pubLoopConstraintEdge;
MarkerArrayPublisher pubKeyFrameYawMarkers;

TimeType timeLaserInfoStamp;

pcl::PointCloud<PointTypeIndex>::Ptr cloudKeyPoses3D; // Store keyframe poses and indexes 
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
pcl::PointCloud<PointTypePose>::Ptr cloudKeyOdomPoses6D;
pcl::PointCloud<PointTypeIndex>::Ptr copy_cloudKeyPoses3D;
pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;
vector<pcl::PointCloud<PointTypeIndex>::Ptr> copy_featCloudKeyFrames;

double timeLaserInfoCur;

float transformTobeMapped[6];
Eigen::Vector3d translationLidarToIMU;
Eigen::Matrix3d rotationLidarToIMU;

bool isDegenerate = false;

PathMsg globalPath;

std::mutex mtx;
std::mutex mtxLoopInfo;
std::mutex mtxGnssFactor;
std::mutex workerWaitMutex;
std::condition_variable workerWaitCondition;

bool aLoopIsClosed = false;
map<int, int> loopIndexContainer; // from new to old
vector<pair<int, int>> loopIndexQueue;
vector<gtsam::Pose3> loopPoseQueue;
vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
std::atomic<bool> loopFactorsPending{false};

std::deque<std::tuple<int, Eigen::Vector3d, Eigen::Matrix3d>> gnssPosFactorQueue;
std::deque<std::pair<int, double>> gnssYawFactorQueue;
double last_pos_t = -1.0;
double last_yaw_t = -1.0;
std::unordered_set<int> pos_keys;
std::unordered_set<int> yaw_keys;

void correctPoses();

bool waitForWorkerPeriod(const double frequencyHz)
{
    const double safeFrequency = std::max(frequencyHz, 1e-3);
    const auto period = std::chrono::duration<double>(1.0 / safeFrequency);
    std::unique_lock<std::mutex> lock(workerWaitMutex);
    return workerWaitCondition.wait_for(
        lock,
        period,
        []()
        {
            return flg_exit.load() || !ros_ok();
        });
}

void notifyMapOptimizationShutdown()
{
    workerWaitCondition.notify_all();
}

pcl::VoxelGrid<PointTypeIndex> downSizeFilterICP;

vector<pcl::PointCloud<PointTypeIndex>::Ptr> featCloudKeyFrames;

KD_TREE_PUBLIC<PointTypeIndex>::Ptr ikdtreeHistoryKeyPoses;

KD_TREE_PUBLIC<PointTypeIndex>::PointVector initPoses3D;

map<int, pair<pcl::PointCloud<PointTypeIndex>, pcl::PointCloud<PointTypeIndex>>> laserCloudMapContainer;

Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
{ 
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}

Eigen::Affine3f trans2Affine3f(float transformIn[])
{
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
}

gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
}

gtsam::Pose3 trans2gtsamPose(float transformIn[])
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}

float pointDistance(PointTypeIndex p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

float pointDistance(PointTypeIndex p1, PointTypeIndex p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

float rotationDistance(const gtsam::Pose3& poseFrom, const gtsam::Pose3& poseTo)
{
    const Eigen::Matrix3d deltaR = poseFrom.between(poseTo).rotation().matrix();
    double cosTheta = (deltaR.trace() - 1.0) * 0.5;
    cosTheta = std::max(-1.0, std::min(1.0, cosTheta));
    return static_cast<float>(std::acos(cosTheta));
}

struct LoopAlignmentMetrics
{
    bool valid = false;
    std::size_t evaluated = 0;
    double rms = std::numeric_limits<double>::quiet_NaN();
    double p50 = std::numeric_limits<double>::quiet_NaN();
    double p75 = std::numeric_limits<double>::quiet_NaN();
    double p90 = std::numeric_limits<double>::quiet_NaN();
    double gateInlierRatio = 0.0;
    double gateInlierMse = std::numeric_limits<double>::quiet_NaN();
    double correctionTranslation = std::numeric_limits<double>::quiet_NaN();
    double matrixTranslation = std::numeric_limits<double>::quiet_NaN();
    double correctionRotationDeg = std::numeric_limits<double>::quiet_NaN();
};

LoopAlignmentMetrics calculateLoopAlignmentMetrics(
    const char* stage,
    const pcl::PointCloud<PointTypeIndex>::Ptr& alignedSource,
    const pcl::PointCloud<PointTypeIndex>::Ptr& target,
    const Eigen::Matrix4f& correction,
    const PointTypePose& referencePose)
{
    LoopAlignmentMetrics metrics;
    if (alignedSource->empty() || target->empty())
        return metrics;

    pcl::KdTreeFLANN<PointTypeIndex> targetTree;
    targetTree.setInputCloud(target);

    std::vector<float> distances;
    distances.reserve(alignedSource->size());
    std::vector<int> nearestIndex(1);
    std::vector<float> nearestSquaredDistance(1);
    double squaredDistanceSum = 0.0;
    std::size_t withinHalfMeter = 0;
    std::size_t withinOneMeter = 0;
    std::size_t withinTwoMeters = 0;
    double withinHalfMeterSquaredDistanceSum = 0.0;
    double withinOneMeterSquaredDistanceSum = 0.0;
    double withinTwoMetersSquaredDistanceSum = 0.0;
    std::size_t gateInlierCount = 0;
    double gateInlierSquaredDistanceSum = 0.0;
    const double gateInlierDistance = std::max(
        0.0, static_cast<double>(robustIcpInlierDistance));

    for (const auto& point : alignedSource->points)
    {
        if (!pcl::isFinite(point) ||
            targetTree.nearestKSearch(
                point, 1, nearestIndex, nearestSquaredDistance) <= 0)
        {
            continue;
        }

        const float distance = std::sqrt(nearestSquaredDistance[0]);
        distances.push_back(distance);
        squaredDistanceSum += nearestSquaredDistance[0];
        if (distance <= 0.5f)
        {
            ++withinHalfMeter;
            withinHalfMeterSquaredDistanceSum += nearestSquaredDistance[0];
        }
        if (distance <= 1.0f)
        {
            ++withinOneMeter;
            withinOneMeterSquaredDistanceSum += nearestSquaredDistance[0];
        }
        if (distance <= 2.0f)
        {
            ++withinTwoMeters;
            withinTwoMetersSquaredDistanceSum += nearestSquaredDistance[0];
        }
        if (distance <= gateInlierDistance)
        {
            ++gateInlierCount;
            gateInlierSquaredDistanceSum += nearestSquaredDistance[0];
        }
    }

    if (distances.empty())
    {
        ROS_PRINT_WARN("[LOOP METRICS] stage=%s has no finite nearest-neighbor pairs", stage);
        return metrics;
    }

    std::sort(distances.begin(), distances.end());
    const auto percentile = [&distances](const double fraction)
    {
        const std::size_t index = static_cast<std::size_t>(
            std::round(fraction * static_cast<double>(distances.size() - 1)));
        return distances[index];
    };
    const double count = static_cast<double>(distances.size());
    const auto inlierRms = [](const double squaredDistanceSum, const std::size_t inlierCount)
    {
        return inlierCount > 0
            ? std::sqrt(squaredDistanceSum / static_cast<double>(inlierCount))
            : std::numeric_limits<double>::quiet_NaN();
    };

    Eigen::Affine3f correctionAffine(correction);
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(
        correctionAffine, x, y, z, roll, pitch, yaw);
    constexpr double radiansToDegrees = 180.0 / M_PI;

    metrics.valid = correction.allFinite();
    metrics.evaluated = distances.size();
    metrics.rms = std::sqrt(squaredDistanceSum / count);
    metrics.p50 = percentile(0.50);
    metrics.p75 = percentile(0.75);
    metrics.p90 = percentile(0.90);
    metrics.gateInlierRatio = static_cast<double>(gateInlierCount) / count;
    metrics.gateInlierMse = gateInlierCount > 0
        ? gateInlierSquaredDistanceSum / static_cast<double>(gateInlierCount)
        : std::numeric_limits<double>::quiet_NaN();
    metrics.matrixTranslation = std::sqrt(x * x + y * y + z * z);

    // The translation column of a rigid transform is not invariant to a
    // change in map origin.  For example, a small rotation of a scan located
    // hundreds of metres from the origin requires a large compensating
    // matrix translation even when the sensor pose moves only a metre.  Gate
    // the actual displacement of the current keyframe origin instead.
    const Eigen::Vector4d referencePosition(
        referencePose.x, referencePose.y, referencePose.z, 1.0);
    const Eigen::Vector4d correctedReferencePosition =
        correction.cast<double>() * referencePosition;
    const Eigen::Vector3d poseCorrection =
        correctedReferencePosition.head<3>() - referencePosition.head<3>();
    metrics.correctionTranslation = poseCorrection.norm();
    double correctionCosAngle =
        (correction.block<3, 3>(0, 0).cast<double>().trace() - 1.0) * 0.5;
    correctionCosAngle = std::max(-1.0, std::min(1.0, correctionCosAngle));
    metrics.correctionRotationDeg =
        std::acos(correctionCosAngle) * radiansToDegrees;

    if (loopDebugMetricsEnable)
    {
        ROS_PRINT_INFO(
            "[LOOP METRICS] stage=%s evaluated=%zu source=%zu target=%zu "
            "nn_rms=%.3fm p50=%.3fm p75=%.3fm p90=%.3fm "
            "within_0.5m=%.1f%% within_1.0m=%.1f%% within_2.0m=%.1f%% "
            "inlier_rms_0.5m=%.3fm inlier_rms_1.0m=%.3fm inlier_rms_2.0m=%.3fm "
            "gate_dist=%.2fm gate_inliers=%.1f%% gate_inlier_mse=%.3fm^2 "
            "correction_xyz=(%.3f, %.3f, %.3f)m correction_norm=%.3fm "
            "matrix_translation_xyz=(%.3f, %.3f, %.3f)m matrix_translation_norm=%.3fm "
            "correction_rpy=(%.2f, %.2f, %.2f)deg correction_angle=%.2fdeg",
            stage,
            distances.size(),
            alignedSource->size(),
            target->size(),
            metrics.rms,
            metrics.p50,
            metrics.p75,
            metrics.p90,
            100.0 * static_cast<double>(withinHalfMeter) / count,
            100.0 * static_cast<double>(withinOneMeter) / count,
            100.0 * static_cast<double>(withinTwoMeters) / count,
            inlierRms(withinHalfMeterSquaredDistanceSum, withinHalfMeter),
            inlierRms(withinOneMeterSquaredDistanceSum, withinOneMeter),
            inlierRms(withinTwoMetersSquaredDistanceSum, withinTwoMeters),
            gateInlierDistance,
            100.0 * metrics.gateInlierRatio,
            metrics.gateInlierMse,
            poseCorrection.x(), poseCorrection.y(), poseCorrection.z(),
            metrics.correctionTranslation,
            x, y, z,
            metrics.matrixTranslation,
            roll * radiansToDegrees,
            pitch * radiansToDegrees,
            yaw * radiansToDegrees,
            metrics.correctionRotationDeg);
    }

    return metrics;
}

PointTypePose trans2PointTypePose(float transformIn[])
{
    PointTypePose thisPose6D;
    thisPose6D.x = transformIn[3];
    thisPose6D.y = transformIn[4];
    thisPose6D.z = transformIn[5];
    thisPose6D.roll  = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw   = transformIn[2];
    return thisPose6D;
}

void setLaserCurTime(double lidar_end_time)
{
    timeLaserInfoCur = lidar_end_time;
}

pcl::PointCloud<PointTypeIndex>::Ptr transformPointCloud(pcl::PointCloud<PointTypeIndex>::Ptr cloudIn, PointTypePose* transformIn)
{
    pcl::PointCloud<PointTypeIndex>::Ptr cloudOut(new pcl::PointCloud<PointTypeIndex>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
    
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
}

void allocateMemory()
{
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointTypeIndex>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    cloudKeyOdomPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointTypeIndex>());
    copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

    ikdtreeHistoryKeyPoses.reset(new KD_TREE_PUBLIC<PointTypeIndex>());

    for (int i = 0; i < 6; ++i){
        transformTobeMapped[i] = 0;
    }
}

void MapOptimizationInit()
{
    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);

    init_ros_node();
    
    pubKeyPoses = create_publisher<PointCloud2Msg>("lio_sam/trajectory", 1);
    pubPath = create_publisher<PathMsg>("lio_sam/mapping/path", 1);
    pubLaserCloudGlobal = create_publisher<PointCloud2Msg>("lio_sam/mapping/cloud_global", 1);
    pubRecentKeyFrame = create_publisher<PointCloud2Msg>("lio_sam/mapping/cloud_recent_keyframe", 1);
    pubLoopConstraintEdge = create_publisher<MarkerArrayMsg>("lio_sam/loop_closure_constraints", 1);
    pubKeyFrameYawMarkers = create_publisher<MarkerArrayMsg>("lio_sam/mapping/keyframe_yaw", 1);
    pubLoopDebugSource = create_transient_local_publisher<PointCloud2Msg>("lio_sam/loop/source");
    pubLoopDebugTarget = create_transient_local_publisher<PointCloud2Msg>("lio_sam/loop/target");
    pubLoopDebugNdtAligned = create_transient_local_publisher<PointCloud2Msg>("lio_sam/loop/ndt_aligned");
    pubLoopDebugIcpAligned = create_transient_local_publisher<PointCloud2Msg>("lio_sam/loop/icp_aligned");

    downSizeFilterICP.setLeafSize(mappingICPSize, mappingICPSize, mappingICPSize);

    allocateMemory();
}

bool isKeyFrame()
{
    if (cloudKeyOdomPoses6D->points.empty())
        return true;

    Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyOdomPoses6D->back());
    Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                        transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

    if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
        abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
        abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
        sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
        return false;
    
    return true;
}  

void loopFindNearKeyframes(
    pcl::PointCloud<PointTypeIndex>::Ptr& nearKeyframes,
    const int& key,
    const int& searchNum,
    const int& temporalReferenceKey,
    const bool excludeTemporalOverlap)
{
    nearKeyframes->clear();

    const int cloudSize = static_cast<int>(std::min(
        copy_cloudKeyPoses6D->size(), copy_featCloudKeyFrames.size()));

    if (temporalReferenceKey < 0 || temporalReferenceKey >= cloudSize)
        return;

    for (int i = -searchNum; i <= searchNum; ++i)
    {
        const int keyNear = key + i;

        if (keyNear < 0 || keyNear >= cloudSize)
            continue;

        //
        // IMPORTANT:
        // When building the HISTORICAL target submap, do not allow
        // recent/current keyframes to leak into the target.
        //
        // searchNum == 0 is used for the current/source cloud, so it
        // must NOT be filtered here.
        //
        if (searchNum > 0 && excludeTemporalOverlap)
        {
            // Exclude target frames near the current loop keyframe in sensor
            // time, while retaining frames surrounding the historical target
            // key itself.  The reference must be the current loop key, not
            // the target key, otherwise every target-neighborhood frame is
            // filtered out.
            const double time_diff =
                std::abs(
                    copy_cloudKeyPoses6D->points[keyNear].time -
                    copy_cloudKeyPoses6D->points[temporalReferenceKey].time);

            if (time_diff <= historyKeyframeSearchTimeDiff)
                continue;
        }

        *nearKeyframes +=
            *transformPointCloud(
                copy_featCloudKeyFrames[keyNear],
                &copy_cloudKeyPoses6D->points[keyNear]);
    }

    if (nearKeyframes->empty())
        return;

    // downsample near keyframes
    pcl::PointCloud<PointTypeIndex>::Ptr cloud_temp(
        new pcl::PointCloud<PointTypeIndex>());

    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);

    *nearKeyframes = *cloud_temp;
}

bool detectLoopClosureDistance(int *latestID, int *closestID)
{
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
    int loopKeyPre = -1;

    // check loop constraint added before
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
        return false;

    // Candidate selection scans the immutable pose snapshot below. Waiting
    // for ten complete keyframes preserves the old tree-build startup gate
    // without reading the tree while the mapping thread may mutate it.
    if (copy_cloudKeyPoses3D->size() < 10)
        return false;

    // Find the closest history key frame in XY only.
    const auto &current_pose = copy_cloudKeyPoses3D->back();
    const double max_dist_sq =
        historyKeyframeSearchRadius * historyKeyframeSearchRadius;

    double best_dist_sq = std::numeric_limits<double>::infinity();

    int spatial_candidates = 0;
    int temporal_candidates = 0;
    int angular_candidates = 0;

    int nearest_spatial_id = -1;
    double nearest_spatial_dist_sq = std::numeric_limits<double>::infinity();

    for (int id = 0;
         id < static_cast<int>(copy_cloudKeyPoses3D->size());
         ++id)
    {
        if (id == loopKeyCur)
            continue;

        const auto &candidate_pose = copy_cloudKeyPoses3D->points[id];

        const double dx = candidate_pose.x - current_pose.x;
        const double dy = candidate_pose.y - current_pose.y;
        const double dist_sq = dx * dx + dy * dy;

        // Outside spatial search radius.
        if (dist_sq > max_dist_sq)
            continue;

        spatial_candidates++;

        if (dist_sq < nearest_spatial_dist_sq)
        {
            nearest_spatial_dist_sq = dist_sq;
            nearest_spatial_id = id;
        }

        // A loop must be separated from the current *keyframe* by the
        // configured history duration.  timeLaserInfoCur advances on every
        // incoming scan and is not a valid substitute when no keyframe is
        // added (for example, while the vehicle is stopped).
        const double time_diff =
            std::abs(copy_cloudKeyPoses6D->points[id].time -
                     copy_cloudKeyPoses6D->points[loopKeyCur].time);

        if (time_diff <= historyKeyframeSearchTimeDiff)
            continue;

        temporal_candidates++;

        const gtsam::Pose3 poseCur =
            pclPointTogtsamPose3(
                copy_cloudKeyPoses6D->points[loopKeyCur]);

        const gtsam::Pose3 posePre =
            pclPointTogtsamPose3(
                copy_cloudKeyPoses6D->points[id]);

        const double rotation_diff =
            rotationDistance(poseCur, posePre);

        if (rotation_diff > historyKeyframeSearchAngleThreshold)
            continue;

        angular_candidates++;

        // Keep the closest candidate that passed all gates.
        if (dist_sq < best_dist_sq)
        {
            loopKeyPre = id;
            best_dist_sq = dist_sq;
        }
    }

    const double nearest_spatial_dist =
        nearest_spatial_id >= 0
            ? std::sqrt(nearest_spatial_dist_sq)
            : -1.0;

    const double selected_dist =
        loopKeyPre >= 0
            ? std::sqrt(best_dist_sq)
            : -1.0;

    RCLCPP_INFO(
        rclcpp::get_logger("fastlio_mapping"),
        "[LOOP SEARCH] cur=%d xyz=(%.2f %.2f %.2f) "
        "spatial=%d temporal=%d angular=%d "
        "nearest=%d nearest_dist=%.2f "
        "selected=%d selected_dist=%.2f",
        loopKeyCur,
        current_pose.x,
        current_pose.y,
        current_pose.z,
        spatial_candidates,
        temporal_candidates,
        angular_candidates,
        nearest_spatial_id,
        nearest_spatial_dist,
        loopKeyPre,
        selected_dist);

    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
        return false;

    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    return true;
}

void performLoopClosure()
{
    size_t odomPoseCount = 0;
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (cloudKeyPoses3D->points.empty())
            return;

        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        copy_featCloudKeyFrames = featCloudKeyFrames;
        odomPoseCount = cloudKeyOdomPoses6D->size();
    }

    const size_t pose3Count = copy_cloudKeyPoses3D->size();
    const size_t pose6Count = copy_cloudKeyPoses6D->size();
    const size_t cloudCount = copy_featCloudKeyFrames.size();
    if (pose3Count != pose6Count || pose3Count != cloudCount ||
        pose3Count != odomPoseCount)
    {
        ROS_PRINT_ERROR(
            "[LOOP] Keyframe snapshot invariant failed: pose3=%zu pose6=%zu "
            "odom=%zu clouds=%zu; skipping closure safely",
            pose3Count, pose6Count, odomPoseCount, cloudCount);
        return;
    }

    // find keys
    int loopKeyCur;
    int loopKeyPre;
    if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false) return;

    // extract cloud
    pcl::PointCloud<PointTypeIndex>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointTypeIndex>());
    pcl::PointCloud<PointTypeIndex>::Ptr coarseSourceKeyframeCloud(
        new pcl::PointCloud<PointTypeIndex>());
    pcl::PointCloud<PointTypeIndex>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointTypeIndex>());
    const int sourceSearchNum = loopCoarseRegistrationEnable
        ? std::max(0, loopCoarseSourceKeyframeSearchNum)
        : 0;
    {
        // cloud near latest keyframe 
        // Always keep the original one-keyframe source for a strict fallback.
        loopFindNearKeyframes(
            cureKeyframeCloud,
            loopKeyCur,
            0,
            loopKeyCur,
            false);
        // The NDT path uses a configurable local source window.  It is only
        // promoted to the fine-ICP source after NDT passes its own gate.
        if (loopCoarseRegistrationEnable)
        {
            loopFindNearKeyframes(
                coarseSourceKeyframeCloud,
                loopKeyCur,
                sourceSearchNum,
                loopKeyCur,
                false);
        }
        // cloud near previous loop keyframe
        loopFindNearKeyframes(
            prevKeyframeCloud,
            loopKeyPre,
            historyKeyframeSearchNum,
            loopKeyCur,
            true);

        if (loopDebugCloudsEnable)
        {
            const auto &debug_source = loopCoarseRegistrationEnable
                ? coarseSourceKeyframeCloud
                : cureKeyframeCloud;
            publishCloudAlways(
                pubLoopDebugSource, debug_source, timeLaserInfoStamp, map_frame);
            publishCloudAlways(
                pubLoopDebugTarget, prevKeyframeCloud, timeLaserInfoStamp, map_frame);
        }
        if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
        {
            ROS_PRINT_WARN(
                "[LOOP] ICP skipped: current=%d previous=%d source_points=%zu "
                "target_points=%zu required_source=300 required_target=1000",
                loopKeyCur,
                loopKeyPre,
                cureKeyframeCloud->size(),
                prevKeyframeCloud->size());
            return;
        }
        // if (pubHistoryKeyFrames.getNumSubscribers() != 0)
        //     publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
    }

    const auto &current_pose = copy_cloudKeyPoses6D->points[loopKeyCur];
    const auto &previous_pose = copy_cloudKeyPoses6D->points[loopKeyPre];
    ROS_PRINT_INFO(
        "[LOOP] ICP attempt: current=%d previous=%d key_delta=%d "
        "pose_delta=(%.2f, %.2f, %.2f)m source_points=%zu target_points=%zu "
        "max_corr=%.2fm",
        loopKeyCur,
        loopKeyPre,
        loopKeyCur - loopKeyPre,
        current_pose.x - previous_pose.x,
        current_pose.y - previous_pose.y,
        current_pose.z - previous_pose.z,
        cureKeyframeCloud->size(),
        prevKeyframeCloud->size(),
        loopIcpMaxCorrespondenceDistance);

    Eigen::Matrix4f coarse_initial_guess = Eigen::Matrix4f::Identity();
    if (loopCoarseRegistrationEnable)
    {
        const auto ndt_start = std::chrono::steady_clock::now();
        pcl::VoxelGrid<PointTypeIndex> coarse_filter;
        coarse_filter.setLeafSize(
            loopCoarseNdtLeafSize,
            loopCoarseNdtLeafSize,
            loopCoarseNdtLeafSize);

        pcl::PointCloud<PointTypeIndex>::Ptr coarse_source(
            new pcl::PointCloud<PointTypeIndex>());
        pcl::PointCloud<PointTypeIndex>::Ptr coarse_target(
            new pcl::PointCloud<PointTypeIndex>());
        coarse_filter.setInputCloud(coarseSourceKeyframeCloud);
        coarse_filter.filter(*coarse_source);
        coarse_filter.setInputCloud(prevKeyframeCloud);
        coarse_filter.filter(*coarse_target);

        pcl::NormalDistributionsTransform<PointTypeIndex, PointTypeIndex> ndt;
        ndt.setResolution(loopCoarseNdtResolution);
        ndt.setStepSize(loopCoarseNdtStepSize);
        ndt.setTransformationEpsilon(loopCoarseNdtTransformationEpsilon);
        ndt.setMaximumIterations(loopCoarseNdtMaximumIterations);
        ndt.setInputSource(coarse_source);
        ndt.setInputTarget(coarse_target);
        pcl::PointCloud<PointTypeIndex> ndt_result;
        ndt.align(ndt_result);

        if (loopDebugCloudsEnable || loopDebugMetricsEnable)
        {
            pcl::PointCloud<PointTypeIndex>::Ptr ndt_result_ptr(
                new pcl::PointCloud<PointTypeIndex>(ndt_result));
            if (loopDebugCloudsEnable)
            {
                publishCloudAlways(
                    pubLoopDebugNdtAligned,
                    ndt_result_ptr,
                    timeLaserInfoStamp,
                    map_frame);
            }
            calculateLoopAlignmentMetrics(
                "NDT",
                ndt_result_ptr,
                prevKeyframeCloud,
                ndt.getFinalTransformation(),
                current_pose);
        }

        const bool ndt_converged = ndt.hasConverged();
        const double ndt_fitness = ndt.getFitnessScore();
        const bool ndt_usable = ndt_converged && std::isfinite(ndt_fitness) &&
            ndt_fitness <= loopCoarseNdtFitnessScore;
        const double ndt_duration_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - ndt_start).count();
        if (ndt_usable)
        {
            coarse_initial_guess = ndt.getFinalTransformation();
            cureKeyframeCloud = coarseSourceKeyframeCloud;
        }

        ROS_PRINT_INFO(
            "[LOOP] NDT coarse: current=%d previous=%d source_points=%zu "
            "target_points=%zu converged=%s fitness=%.6f threshold=%.6f "
            "used=%s source_window=%d duration=%.1fms",
            loopKeyCur,
            loopKeyPre,
            coarse_source->size(),
            coarse_target->size(),
            ndt_converged ? "true" : "false",
            ndt_fitness,
            loopCoarseNdtFitnessScore,
            ndt_usable ? "true" : "false",
            sourceSearchNum,
            ndt_duration_ms);
        if (!ndt_usable)
        {
            ROS_PRINT_WARN(
                "[LOOP] NDT coarse rejected; falling back to identity ICP guess");
        }
    }

    // ICP Settings. NDT may provide only an initial guess. Loop constraints
    // can pass either the legacy global fitness gate or the optional robust
    // overlap/inlier gate below.
    static pcl::IterativeClosestPoint<PointTypeIndex, PointTypeIndex> icp;
    icp.setMaxCorrespondenceDistance(loopIcpMaxCorrespondenceDistance);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align clouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(prevKeyframeCloud);
    LoopAlignmentMetrics initialMetrics;
    if (loopDebugMetricsEnable || useRobustIcpGating)
    {
        pcl::PointCloud<PointTypeIndex>::Ptr icpInitialAligned(
            new pcl::PointCloud<PointTypeIndex>());
        pcl::transformPointCloud(
            *cureKeyframeCloud,
            *icpInitialAligned,
            coarse_initial_guess);
        initialMetrics = calculateLoopAlignmentMetrics(
            "ICP_INITIAL",
            icpInitialAligned,
            prevKeyframeCloud,
            coarse_initial_guess,
            current_pose);
    }
    pcl::PointCloud<PointTypeIndex>::Ptr unused_result(new pcl::PointCloud<PointTypeIndex>());
    icp.align(*unused_result, coarse_initial_guess);

    if (loopDebugCloudsEnable)
    {
        publishCloudAlways(
            pubLoopDebugIcpAligned,
            unused_result,
            timeLaserInfoStamp,
            map_frame);
    }
    LoopAlignmentMetrics finalMetrics;
    if (loopDebugMetricsEnable || useRobustIcpGating)
    {
        finalMetrics = calculateLoopAlignmentMetrics(
            "ICP",
            unused_result,
            prevKeyframeCloud,
            icp.getFinalTransformation(),
            current_pose);
    }

    const bool icp_converged = icp.hasConverged();
    const double icp_fitness = icp.getFitnessScore();
    const bool legacyPass = icp_converged && std::isfinite(icp_fitness) &&
        icp_fitness <= historyKeyframeFitnessScore;

    static int lastRobustAcceptedKey = -1;
    const bool robustBaseValid = icp_converged && std::isfinite(icp_fitness) &&
        finalMetrics.valid;
    const bool robustPointCount = finalMetrics.evaluated >=
        static_cast<std::size_t>(std::max(0, robustIcpMinEvaluatedPoints));
    const bool robustTranslation = std::isfinite(finalMetrics.correctionTranslation) &&
        finalMetrics.correctionTranslation <= robustIcpMaxTranslation;
    const bool robustRotation = std::isfinite(finalMetrics.correctionRotationDeg) &&
        finalMetrics.correctionRotationDeg <= robustIcpMaxRotationDeg;
    const bool robustOverlap = std::isfinite(finalMetrics.gateInlierRatio) &&
        finalMetrics.gateInlierRatio >= robustIcpMinInlierRatio;
    const bool robustInlierMse = std::isfinite(finalMetrics.gateInlierMse) &&
        finalMetrics.gateInlierMse <= robustIcpMaxInlierMse;
    const bool robustP75 = std::isfinite(finalMetrics.p75) &&
        finalMetrics.p75 <= robustIcpMaxP75;

    const bool initialGood = initialMetrics.valid &&
        initialMetrics.evaluated >=
            static_cast<std::size_t>(std::max(0, robustIcpMinEvaluatedPoints)) &&
        initialMetrics.gateInlierRatio >= robustIcpMinInlierRatio &&
        std::isfinite(initialMetrics.gateInlierMse) &&
        initialMetrics.gateInlierMse <= robustIcpMaxInlierMse &&
        std::isfinite(initialMetrics.p75) &&
        initialMetrics.p75 <= robustIcpMaxP75;
    const bool robustImprovement = initialMetrics.valid &&
        finalMetrics.gateInlierRatio >=
            initialMetrics.gateInlierRatio + robustIcpMinOverlapImprovement &&
        finalMetrics.p75 < initialMetrics.p75;
    const bool robustNonDegrading = initialMetrics.valid &&
        finalMetrics.gateInlierRatio >=
            initialMetrics.gateInlierRatio - robustIcpOverlapTolerance &&
        finalMetrics.p75 <= initialMetrics.p75 + robustIcpP75Tolerance;
    const bool robustCooldown = lastRobustAcceptedKey < 0 ||
        loopKeyCur - lastRobustAcceptedKey >= robustIcpCooldownKeyframes;

    const bool robustCandidate = useRobustIcpGating && robustBaseValid &&
        robustPointCount && robustTranslation && robustRotation &&
        robustOverlap && robustInlierMse && robustP75 &&
        (initialGood || robustImprovement) && robustNonDegrading && robustCooldown;
    const bool robustApplied = robustCandidate && !robustIcpGatingDryRun;

    if (useRobustIcpGating)
    {
        ROS_PRINT_INFO(
            "[LOOP ROBUST] current=%d previous=%d candidate=%s applied=%s dry_run=%s "
            "base=%s points=%s translation=%s rotation=%s overlap=%s inlier_mse=%s "
            "p75=%s initial_good=%s improved=%s non_degrading=%s cooldown=%s "
            "evaluated=%zu inliers=%.1f%%/%.1f%% inlier_mse=%.3f/%.3fm^2 "
            "p75=%.3f/%.3fm correction=%.3f/%.3fm rotation=%.2f/%.2fdeg "
            "overlap_gain=%.1f%%/%.1f%%",
            loopKeyCur,
            loopKeyPre,
            robustCandidate ? "true" : "false",
            robustApplied ? "true" : "false",
            robustIcpGatingDryRun ? "true" : "false",
            robustBaseValid ? "pass" : "FAIL",
            robustPointCount ? "pass" : "FAIL",
            robustTranslation ? "pass" : "FAIL",
            robustRotation ? "pass" : "FAIL",
            robustOverlap ? "pass" : "FAIL",
            robustInlierMse ? "pass" : "FAIL",
            robustP75 ? "pass" : "FAIL",
            initialGood ? "pass" : "no",
            robustImprovement ? "pass" : "no",
            robustNonDegrading ? "pass" : "FAIL",
            robustCooldown ? "pass" : "FAIL",
            finalMetrics.evaluated,
            100.0 * finalMetrics.gateInlierRatio,
            100.0 * static_cast<double>(robustIcpMinInlierRatio),
            finalMetrics.gateInlierMse,
            robustIcpMaxInlierMse,
            finalMetrics.p75,
            robustIcpMaxP75,
            finalMetrics.correctionTranslation,
            robustIcpMaxTranslation,
            finalMetrics.correctionRotationDeg,
            robustIcpMaxRotationDeg,
            100.0 * (finalMetrics.gateInlierRatio - initialMetrics.gateInlierRatio),
            100.0 * static_cast<double>(robustIcpMinOverlapImprovement));
    }

    if (!legacyPass && !robustApplied)
    {
        const char* rejectionReason = !icp_converged ? "not_converged" :
            (!std::isfinite(icp_fitness) ? "non_finite_fitness" :
            (robustCandidate && robustIcpGatingDryRun ? "robust_gate_dry_run" :
            (useRobustIcpGating ? "legacy_and_robust_gates_failed" :
            "fitness_above_threshold")));
        ROS_PRINT_WARN(
            "[LOOP] ICP rejected: current=%d previous=%d converged=%s "
            "fitness=%.6f threshold=%.6f reason=%s",
            loopKeyCur,
            loopKeyPre,
            icp_converged ? "true" : "false",
            icp_fitness,
            historyKeyframeFitnessScore,
            rejectionReason);
        return;
    }
    if (robustApplied)
        lastRobustAcceptedKey = loopKeyCur;
    const char* acceptanceMode = legacyPass
        ? (robustApplied ? "legacy+robust" : "legacy")
        : "robust";
    ROS_PRINT_INFO(
        "[LOOP] ICP accepted: current=%d previous=%d fitness=%.6f threshold=%.6f mode=%s",
        loopKeyCur,
        loopKeyPre,
        icp_fitness,
        historyKeyframeFitnessScore,
        acceptanceMode
    );    

    // publish corrected cloud
    // if (pubIcpKeyFrames.getNumSubscribers() != 0)
    // {
    //     pcl::PointCloud<PointTypeIndex>::Ptr closed_cloud(new pcl::PointCloud<PointTypeIndex>());
    //     pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
    //     publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
    // }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    // transform from world origin to wrong pose
    Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    // transform from world origin to corrected pose
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
    pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
    gtsam::Vector Vector6(6);
    const double noiseScore = std::max(
        static_cast<double>(icp.getFitnessScore()) * loopWeight, 1e-4);
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

    // Add pose constraint
    mtx.lock();
    loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(constraintNoise);
    loopFactorsPending.store(true, std::memory_order_release);
    mtx.unlock();

    // add loop constriant
    loopIndexContainer[loopKeyCur] = loopKeyPre;
}

void addOdomFactor()
{
    if (cloudKeyPoses3D->points.empty())
    {
        // TODO: use IKFoM covariance as prior covariance
        noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
        const gtsam::Pose3 poseTo = [&]() {
            const gtsam::Pose3 poseOdom = trans2gtsamPose(transformTobeMapped);
            const Eigen::Matrix3d R_map_body = R_map_odom * poseOdom.rotation().matrix();
            const Eigen::Vector3d t_map_body = R_map_odom *
                Eigen::Vector3d(poseOdom.translation().x(),
                                poseOdom.translation().y(),
                                poseOdom.translation().z()) + t_map_odom;
            return gtsam::Pose3(gtsam::Rot3(R_map_body),
                                gtsam::Point3(t_map_body.x(), t_map_body.y(), t_map_body.z()));
        }();
        gtSAMgraph.add(PriorFactor<Pose3>(0, poseTo, priorNoise));
        initialEstimate.insert(0, poseTo);
    }else{
        noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        const gtsam::Pose3 poseFromFront = pclPointTogtsamPose3(cloudKeyOdomPoses6D->points.back());
        const gtsam::Pose3 poseToFront = trans2gtsamPose(transformTobeMapped);
        const gtsam::Pose3 odomDelta = poseFromFront.between(poseToFront);
        const gtsam::Pose3 poseFromOpt = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
        const gtsam::Pose3 poseToOpt = poseFromOpt.compose(odomDelta);
        gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), odomDelta, odometryNoise));
        initialEstimate.insert(cloudKeyPoses3D->size(), poseToOpt);
    }
}

int addLoopFactor()
{
    if (!loopFactorsPending.load(std::memory_order_acquire))
        return 0;

    vector<pair<int, int>> indexQueue;
    vector<gtsam::Pose3> poseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> noiseQueue;
    {
        // The loop-closure thread only writes these queues.  Move a complete
        // batch to the mapping thread before touching the GTSAM graph.
        std::lock_guard<std::mutex> lock(mtx);
        if (loopIndexQueue.empty())
        {
            loopFactorsPending.store(false, std::memory_order_release);
            return 0;
        }
        indexQueue.swap(loopIndexQueue);
        poseQueue.swap(loopPoseQueue);
        noiseQueue.swap(loopNoiseQueue);
        loopFactorsPending.store(false, std::memory_order_release);
    }

    for (int i = 0; i < (int)indexQueue.size(); ++i)
    {
        int indexFrom = indexQueue[i].first;
        int indexTo = indexQueue[i].second;
        gtsam::Pose3 poseBetween = poseQueue[i];
        gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = noiseQueue[i];

        gtSAMgraph.add(
            BetweenFactor<Pose3>(
                indexFrom,
                indexTo,
                poseBetween,
                noiseBetween
            )
        );

        ROS_PRINT_INFO(
            "[LOOP] GTSAM factor added: %d -> %d",
            indexFrom,
            indexTo
        );
    }

    aLoopIsClosed = true;
    return static_cast<int>(indexQueue.size());
}

void processPendingLoopFactors()
{
    // Keep the idle path to one atomic read.  In particular, do not run an
    // empty iSAM update for every LiDAR scan.
    if (!loopFactorsPending.load(std::memory_order_acquire))
        return;

    const auto updateStart = std::chrono::steady_clock::now();
    const int factorsAdded = addLoopFactor();
    if (factorsAdded == 0)
        return;

    // No new variables are introduced: loop factors connect keyframes that
    // already exist in iSAM.  Mirror the extra relinearization passes used by
    // saveKeyFramesAndFactor() when it consumes a loop constraint.
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    isam->update();
    isam->update();
    isam->update();
    isam->update();
    isam->update();

    gtSAMgraph.resize(0);
    initialEstimate.clear();
    isamCurrentEstimate = isam->calculateEstimate();

    const double durationMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - updateStart).count();
    ROS_PRINT_INFO(
        "[LOOP] optimized %d queued factor(s) without waiting for a new "
        "keyframe duration=%.1fms",
        factorsAdded,
        durationMs);
}

bool findNearestKeyframeByTime(const std::vector<PointTypePose> &keyposes,
                                      double stamp,
                                      int &key_out)
{
    key_out = -1;
    if (keyposes.empty())
        return false;

    const double first_time = keyposes.front().time;
    const double last_time = keyposes.back().time;
    constexpr double gnss_keyframe_tol = 0.12;

    auto it = std::lower_bound(
        keyposes.begin(),
        keyposes.end(),
        stamp,
        [](const PointTypePose &pose, double t)
        {
            return pose.time < t;
        });

    size_t best_idx = 0;
    double best_dt = std::numeric_limits<double>::infinity();

    if (it == keyposes.begin())
    {
        best_idx = 0;
        best_dt = std::abs(keyposes[0].time - stamp);
    }
    else if (it == keyposes.end())
    {
        best_idx = keyposes.size() - 1;
        best_dt = std::abs(keyposes.back().time - stamp);
    }
    else
    {
        const size_t upper_idx = static_cast<size_t>(std::distance(keyposes.begin(), it));
        const size_t lower_idx = upper_idx - 1;
        const double lower_dt = std::abs(keyposes[lower_idx].time - stamp);
        const double upper_dt = std::abs(keyposes[upper_idx].time - stamp);
        if (lower_dt <= upper_dt)
        {
            best_idx = lower_idx;
            best_dt = lower_dt;
        }
        else
        {
            best_idx = upper_idx;
            best_dt = upper_dt;
        }
    }

    key_out = static_cast<int>(best_idx);
    return best_dt <= gnss_keyframe_tol && stamp <= last_time + gnss_keyframe_tol;
}

void addGNSSFactor()
{
    decltype(gnssPosFactorQueue) posQueue;

    {
        std::lock_guard<std::mutex> lock(mtxGnssFactor);
        if (gnssPosFactorQueue.empty())
            return;
        posQueue.swap(gnssPosFactorQueue);
    }

    while (!posQueue.empty())
    {
        const auto &[key, pos, cov] = posQueue.front();

        const double gnss_x = pos.x();
        const double gnss_y = pos.y();
        const double gnss_z = pos.z();

        const double cov_x = cov(0, 0);
        const double cov_y = cov(1, 1);
        const double cov_z = cov(2, 2);

        gtsam::Vector sigma(3);
        sigma << std::sqrt(cov_x),
                 std::sqrt(cov_y),
                 std::sqrt(cov_z);

        gtSAMgraph.add(
            gtsam::GPSFactor(
                key,
                gtsam::Point3(gnss_x, gnss_y, gnss_z),
                gtsam::noiseModel::Diagonal::Sigmas(sigma)));

        posQueue.pop_front();
    }

    aLoopIsClosed = true;
}

void addGNSSYawFactor()
{
    decltype(gnssYawFactorQueue) yawQueue;

    {
        std::lock_guard<std::mutex> lock(mtxGnssFactor);
        if (gnssYawFactorQueue.empty())
            return;
        yawQueue.swap(gnssYawFactorQueue);
    }

    bool added = false;

    while (!yawQueue.empty())
    {
        const auto &[key, yaw] = yawQueue.front();

        const double yaw_sigma = std::max(gnss_yaw_factor_sigma, 1e-4);
        const auto yawNoise = gtsam::noiseModel::Isotropic::Sigma(1, yaw_sigma);

        gtSAMgraph.add(
            boost::shared_ptr<GnssYawFactor>(
                new GnssYawFactor(key, yaw, yawNoise)));

        yawQueue.pop_front();
    }

    aLoopIsClosed = true;
}

void processGnssPos(const std::vector<PointTypePose> &keyposes)
{
    if (!p_gnss)
        return;

    if (keyposes.size() < 5)
        return;

    const Eigen::Vector3d key_start(
        keyposes.front().x,
        keyposes.front().y,
        keyposes.front().z);
    const Eigen::Vector3d key_end(
        keyposes.back().x,
        keyposes.back().y,
        keyposes.back().z);

    if ((key_end - key_start).norm() < gpsFactorMinDis)
    {
        return;
    }

    static Eigen::Vector3d last_fpos = Eigen::Vector3d::Zero();
    static bool has_fpos = false;

    while (true)
    {
        PosData pos;
        if (!p_gnss->peekOldestPos(pos))
            return;

        int key = -1;
        if (!findNearestKeyframeByTime(keyposes, pos.t, key))
        {
            if (!keyposes.empty() && pos.t <= keyposes.back().time)
            {
                PosData dropped;
                p_gnss->popOldestPos(dropped);
                continue;
            }

            return;
        }
        if (key < 0)
            return;
        if (pos_keys.count(key) != 0 || pos.t <= last_pos_t)
        {
            PosData dropped;
            p_gnss->popOldestPos(dropped);
            continue;
        }

        const PointTypePose &pose = keyposes[key];
        const gtsam::Pose3 key_pose = pclPointTogtsamPose3(pose);
        const Eigen::Matrix3d R_map_imu = key_pose.rotation().matrix();
        Eigen::Vector3d gnss_pos = pos.p - R_map_imu * p_gnss->lever();
        Eigen::Matrix3d gnss_cov = pos.cov;
        if (!useGnssElevation)
        {
            gnss_pos.z() = pose.z;
            gnss_cov(2, 2) = 0.01;
        }

        const double covariance_scale = gnss_stddev_scale * gnss_stddev_scale;
        gnss_cov *= covariance_scale;
        const double min_variance = gnss_min_stddev * gnss_min_stddev;
        for (int axis = 0; axis < 3; ++axis)
        {
            gnss_cov(axis, axis) = std::max(gnss_cov(axis, axis), min_variance);
        }

        if (!gnss_pos.allFinite())
        {
            PosData dropped;
            p_gnss->popOldestPos(dropped);
            continue;
        }

        if (!gnss_cov.allFinite() ||
            gnss_cov(0, 0) <= 0.0 ||
            gnss_cov(1, 1) <= 0.0 ||
            gnss_cov(2, 2) <= 0.0)
        {
            PosData dropped;
            p_gnss->popOldestPos(dropped);
            continue;
        }

        if (has_fpos &&
            (gnss_pos - last_fpos).norm() < gpsFactorMinDis)
        {
            PosData dropped;
            p_gnss->popOldestPos(dropped);
            continue;
        }

        if (!p_gnss->popOldestPos(pos))
            return;

        pos_keys.insert(key);
        last_pos_t = pos.t;
        last_fpos = gnss_pos;
        has_fpos = true;

        {
            std::lock_guard<std::mutex> lock(mtxGnssFactor);
            gnssPosFactorQueue.emplace_back(
                key,
                gnss_pos,
                gnss_cov);
        }

        ROS_PRINT_INFO(
            "[GNSS] queued GPS factor: key=%d enu=(%.2f, %.2f, %.2f) "
            "sigma=(%.2f, %.2f, %.2f)m stddev_scale=%.2f min_stddev=%.2f",
            key,
            gnss_pos.x(), gnss_pos.y(), gnss_pos.z(),
            std::sqrt(gnss_cov(0, 0)),
            std::sqrt(gnss_cov(1, 1)),
            std::sqrt(gnss_cov(2, 2)),
            gnss_stddev_scale,
            gnss_min_stddev);
    }
}

void processGnssYaw(const std::vector<PointTypePose> &keyposes)
{
    if (!p_gnss)
        return;

    while (true)
    {
        YawData yaw;
        if (!p_gnss->peekOldestYaw(yaw))
            return;

        if (!useGnssYawFactor)
        {
            YawData dropped;
            p_gnss->popOldestYaw(dropped);
            continue;
        }

        int key = -1;
        if (!findNearestKeyframeByTime(keyposes, yaw.t, key))
        {
            if (!keyposes.empty() && yaw.t <= keyposes.back().time)
            {
                YawData dropped;
                p_gnss->popOldestYaw(dropped);
                continue;
            }

            return;
        }
        if (key < 0)
            return;
        if (yaw_keys.count(key) != 0 || yaw.t <= last_yaw_t)
        {
            YawData dropped;
            p_gnss->popOldestYaw(dropped);
            continue;
        }

        if (std::abs(normalizeYaw(yaw.yaw - keyposes[key].yaw)) > 60.0 * M_PI / 180.0)
        {
            YawData dropped;
            p_gnss->popOldestYaw(dropped);
            continue;
        }

        if (!std::isfinite(yaw.yaw))
        {
            YawData dropped;
            p_gnss->popOldestYaw(dropped);
            continue;
        }

        if (!p_gnss->popOldestYaw(yaw))
            return;

        yaw_keys.insert(key);
        last_yaw_t = yaw.t;

        {
            std::lock_guard<std::mutex> lock(mtxGnssFactor);
            gnssYawFactorQueue.emplace_back(
                key,
                yaw.yaw);
        }
    }
}

void performGnssMatching()
{
    if (!gnssEnableFlag || !gnss_aligned.load() || !p_gnss)
        return;

    std::vector<PointTypePose> keyposes;
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (cloudKeyPoses6D == nullptr || cloudKeyPoses6D->points.empty())
            return;
        keyposes.assign(cloudKeyPoses6D->points.begin(), cloudKeyPoses6D->points.end());
    }

    processGnssPos(keyposes);
    processGnssYaw(keyposes);
}

void updatePath(const PointTypePose& pose_in)
{
    PoseStampedMsg pose_stamped;
    pose_stamped.header.stamp = get_ros_time(pose_in.time);
    pose_stamped.header.frame_id = map_frame;
    pose_stamped.pose.position.x = pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z = pose_in.z;
    pose_stamped.pose.orientation = quaternion_from_rpy(pose_in.roll, pose_in.pitch, pose_in.yaw);

    globalPath.poses.push_back(pose_stamped);
}

void saveKeyFramesAndFactor(pcl::PointCloud<pcl::PointXYZINormal>::Ptr feats_undistort)
{
    PointTypePose OdomPose = trans2PointTypePose(transformTobeMapped);
    OdomPose.time = timeLaserInfoCur;

    // odom factor
    addOdomFactor();

    addGNSSFactor();

    addGNSSYawFactor();
    // loop factor
    addLoopFactor();

    // cout << "****************************************************" << endl;
    // gtSAMgraph.print("GTSAM Graph:\n");

    // update iSAM
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();

    if (aLoopIsClosed == true)
    {
        isam->update();
        isam->update();
        isam->update();
        isam->update();
        isam->update();
    }

    gtSAMgraph.resize(0);
    initialEstimate.clear();

    //save key poses
    PointTypeIndex thisPose3D;
    PointTypePose thisPose6D;
    Pose3 latestEstimate;

    isamCurrentEstimate = isam->calculateEstimate();
    latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
    // debug stuff might be able to remove later
    const auto t = latestEstimate.translation();
    const auto R = latestEstimate.rotation().matrix();

    if (!std::isfinite(t.x()) ||
        !std::isfinite(t.y()) ||
        !std::isfinite(t.z()) ||
        !R.allFinite())
    {
        ROS_PRINT_ERROR(
            "[BAD POSE] key=%zu t=(%.6f %.6f %.6f)",
            cloudKeyPoses3D->size(),
            t.x(), t.y(), t.z()
        );
        return;
    }

    if (std::abs(t.x()) > 10000.0 ||
        std::abs(t.y()) > 10000.0 ||
        std::abs(t.z()) > 10000.0)
    {
        ROS_PRINT_WARN(
            "[HUGE POSE] key=%zu t=(%.3f %.3f %.3f)",
            cloudKeyPoses3D->size(),
            t.x(), t.y(), t.z()
        );
    }
    // END DEBUG STUFF

    // cout << "****************************************************" << endl;
    // isamCurrentEstimate.print("Current estimate: ");

    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.roll  = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw   = latestEstimate.rotation().yaw();
    thisPose6D.time = timeLaserInfoCur;
    // cout << "****************************************************" << endl;
    // cout << "Pose covariance:" << endl;
    // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
    poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

    // pcl::PointCloud<PointTypeIndex>::Ptr featCloudKeyFrame(new pcl::PointCloud<PointTypeIndex>());
    // PointTypeIndex point;
    // for (const auto &pt : feats_undistort->points) {
    //     Eigen::Vector3d pointBodyLidar(pt.x, pt.y, pt.z);
    //     Eigen::Vector3d pointBodyImu(rotationLidarToIMU * pointBodyLidar + translationLidarToIMU);

    //     point.x = pointBodyImu(0);
    //     point.y = pointBodyImu(1);
    //     point.z = pointBodyImu(2);
    //     point.intensity = pt.intensity;
    //     featCloudKeyFrame->push_back(point);
    // }
    pcl::PointCloud<PointTypeIndex>::Ptr featCloudKeyFrame(
    new pcl::PointCloud<PointTypeIndex>());

    PointTypeIndex point;

    size_t invalid_points = 0;
    double max_abs_coord = 0.0;

    for (const auto &pt : feats_undistort->points)
    {
        if (!std::isfinite(pt.x) ||
            !std::isfinite(pt.y) ||
            !std::isfinite(pt.z))
        {
            ++invalid_points;
            continue;
        }

        Eigen::Vector3d pointBodyLidar(pt.x, pt.y, pt.z);
        Eigen::Vector3d pointBodyImu(
            rotationLidarToIMU * pointBodyLidar + translationLidarToIMU);

        if (!pointBodyImu.allFinite())
        {
            ++invalid_points;
            continue;
        }

        max_abs_coord = std::max(
            max_abs_coord,
            pointBodyImu.cwiseAbs().maxCoeff());

        point.x = pointBodyImu(0);
        point.y = pointBodyImu(1);
        point.z = pointBodyImu(2);
        point.intensity = pt.intensity;

        featCloudKeyFrame->push_back(point);
    }

    if (invalid_points > 0 || max_abs_coord > 10000.0)
    {
        ROS_PRINT_WARN(
            "[KEYFRAME CLOUD] key=%zu invalid=%zu/%zu max_abs=%.3f",
            cloudKeyPoses3D->size(),
            invalid_points,
            feats_undistort->size(),
            max_abs_coord
        );
    }

    // Publish a keyframe only after every part of it is complete. The loop
    // thread snapshots these four containers under the same mutex, so it
    // cannot observe a pose whose cloud is still being constructed.
    size_t keyIndex = 0;
    {
        std::lock_guard<std::mutex> lock(mtx);
        const size_t pose3Count = cloudKeyPoses3D->size();
        const size_t pose6Count = cloudKeyPoses6D->size();
        const size_t odomCount = cloudKeyOdomPoses6D->size();
        const size_t cloudCount = featCloudKeyFrames.size();
        if (pose3Count != pose6Count || pose3Count != odomCount ||
            pose3Count != cloudCount)
        {
            ROS_PRINT_ERROR(
                "[KEYFRAME] Container invariant failed before append: "
                "pose3=%zu pose6=%zu odom=%zu clouds=%zu; keyframe dropped",
                pose3Count, pose6Count, odomCount, cloudCount);
            return;
        }

        keyIndex = pose3Count;
        thisPose3D.intensity = static_cast<float>(keyIndex);
        thisPose6D.intensity = static_cast<float>(keyIndex);
        OdomPose.intensity = static_cast<float>(keyIndex);
        cloudKeyPoses3D->push_back(thisPose3D);
        cloudKeyPoses6D->push_back(thisPose6D);
        cloudKeyOdomPoses6D->push_back(OdomPose);
        featCloudKeyFrames.push_back(featCloudKeyFrame);
    }

    updatePath(thisPose6D);

    if (keyframe_export_en)
    {
        const std::string stamp_str = format_unix_time(thisPose6D.time);
        const std::string pcd_path = keyframe_frames_dir + "scans/" + stamp_str + ".pcd";
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinary(pcd_path, *feats_undistort);
    }

    {
        std::lock_guard<std::mutex> lock(mtx);
        if (ikdtreeHistoryKeyPoses->Root_Node == nullptr) {
            initPoses3D.push_back(thisPose3D);
            if (cloudKeyPoses3D->points.size() >= 10)
                ikdtreeHistoryKeyPoses->Build(initPoses3D);
        } else {
            ikdtreeHistoryKeyPoses->Add_Point(thisPose3D);
        }
    }
}

void ReconstructIkdTree()
{
    std::lock_guard<std::mutex> lock(mtx);
    if (ikdtreeHistoryKeyPoses->Root_Node == nullptr)
        return;
    if (cloudKeyPoses3D->points.empty())
        return;

    ikdtreeHistoryKeyPoses->delete_tree_nodes(&ikdtreeHistoryKeyPoses->Root_Node);

    KD_TREE_PUBLIC<PointTypeIndex>::PointVector pose_points;
    pose_points.reserve(cloudKeyPoses3D->points.size());
    for (const auto &pose : cloudKeyPoses3D->points)
    {
        pose_points.push_back(pose);
    }

    ikdtreeHistoryKeyPoses->Build(pose_points);
}

void correctPoses()
{
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (cloudKeyPoses3D->points.empty())
            return;
    }

    if (aLoopIsClosed == true)
    {
        {
            std::lock_guard<std::mutex> lock(mtx);
            globalPath.poses.clear();

            const size_t numPoses = std::min(
                static_cast<size_t>(isamCurrentEstimate.size()),
                std::min(cloudKeyPoses3D->size(), cloudKeyPoses6D->size()));
            for (size_t i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                updatePath(cloudKeyPoses6D->points[i]);
            }
        }

        ReconstructIkdTree();
        aLoopIsClosed = false;
    }

    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!cloudKeyOdomPoses6D->points.empty() &&
            cloudKeyOdomPoses6D->points.size() == cloudKeyPoses6D->points.size())
        {
            const size_t ref = cloudKeyPoses6D->points.size() - 1;
            const gtsam::Pose3 pose_map_ref = pclPointTogtsamPose3(cloudKeyPoses6D->points[ref]);
            const gtsam::Pose3 pose_odom_ref = pclPointTogtsamPose3(cloudKeyOdomPoses6D->points[ref]);
            const Eigen::Matrix3d R_map_odom =
                pose_map_ref.rotation().matrix() * pose_odom_ref.rotation().matrix().transpose();
            const Eigen::Vector3d t_map_odom =
                Eigen::Vector3d(pose_map_ref.translation().x(),
                                 pose_map_ref.translation().y(),
                                 pose_map_ref.translation().z()) -
                R_map_odom * Eigen::Vector3d(pose_odom_ref.translation().x(),
                                             pose_odom_ref.translation().y(),
                                             pose_odom_ref.translation().z());
            setMapOdom(R_map_odom, t_map_odom);
            publishMapToOdomTf(get_ros_time(timeLaserInfoCur));
        }
    }
}

void publishSamMsg()
{
    if (cloudKeyPoses3D->points.empty())
        return;
    // publish key poses
    publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, map_frame);
    if (ros_subscription_count(pubPath) != 0)
    {
        globalPath.header.stamp = timeLaserInfoStamp;
        globalPath.header.frame_id = map_frame;
        ros_publish(pubPath, globalPath);
    }

    if (ros_subscription_count(pubKeyFrameYawMarkers) != 0)
    {
        MarkerArrayMsg markerArray;

        MarkerMsg markerClear;
        markerClear.header.frame_id = map_frame;
        markerClear.header.stamp = timeLaserInfoStamp;
        markerClear.action = MarkerMsg::DELETEALL;
        markerArray.markers.push_back(markerClear);

        markerArray.markers.reserve(markerArray.markers.size() + cloudKeyPoses6D->points.size());
        for (size_t i = 0; i < cloudKeyPoses6D->points.size(); ++i)
        {
            const auto &pose = cloudKeyPoses6D->points[i];
            MarkerMsg marker;
            marker.header.frame_id = map_frame;
            marker.header.stamp = timeLaserInfoStamp;
            marker.ns = "keyframe_yaw";
            marker.id = static_cast<int>(i);
            marker.action = MarkerMsg::ADD;
            marker.type = MarkerMsg::ARROW;
            marker.pose.position.x = pose.x;
            marker.pose.position.y = pose.y;
            marker.pose.position.z = pose.z;
            marker.pose.orientation = quaternion_from_rpy(0.0, 0.0, pose.yaw);
            marker.scale.x = 0.8;
            marker.scale.y = 0.12;
            marker.scale.z = 0.12;
            marker.color.r = 1.0f;
            marker.color.g = 1.0f;
            marker.color.b = 0.0f;
            marker.color.a = 0.9f;
            markerArray.markers.push_back(marker);
        }

        ros_publish(pubKeyFrameYawMarkers, markerArray);
    }

    if (ros_subscription_count(pubRecentKeyFrame) != 0)
    {
        pcl::PointCloud<PointTypeIndex>::Ptr cloudOut(new pcl::PointCloud<PointTypeIndex>());
        PointTypePose thisPose6D = cloudKeyPoses6D->back();
        *cloudOut += *transformPointCloud(featCloudKeyFrames.back(),  &thisPose6D);
        publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, map_frame);
    }
}

void visualizeLoopClosure()
{
    if (loopIndexContainer.empty())
        return;
    
    MarkerArrayMsg markerArray;
    // loop nodes
    MarkerMsg markerNode;
    markerNode.header.frame_id = map_frame;
    markerNode.header.stamp = timeLaserInfoStamp;
    markerNode.action = MarkerMsg::ADD;
    markerNode.type = MarkerMsg::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.1; markerNode.scale.y = 0.1; markerNode.scale.z = 0.1; 
    markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
    markerNode.color.a = 1;
    // loop edges
    MarkerMsg markerEdge;
    markerEdge.header.frame_id = map_frame;
    markerEdge.header.stamp = timeLaserInfoStamp;
    markerEdge.action = MarkerMsg::ADD;
    markerEdge.type = MarkerMsg::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
    markerEdge.color.a = 1;

    for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
    {
        int key_cur = it->first;
        int key_pre = it->second;
        PointMsg p;
        p.x = copy_cloudKeyPoses6D->points[key_cur].x;
        p.y = copy_cloudKeyPoses6D->points[key_cur].y;
        p.z = copy_cloudKeyPoses6D->points[key_cur].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        p.x = copy_cloudKeyPoses6D->points[key_pre].x;
        p.y = copy_cloudKeyPoses6D->points[key_pre].y;
        p.z = copy_cloudKeyPoses6D->points[key_pre].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    ros_publish(pubLoopConstraintEdge, markerArray);
}

void loopClosureThread()
{
    if (loopClosureEnableFlag == false)
        return;

    ROS_PRINT_INFO("...... Loop Closure Thread Start......");

    while (ros_ok() && !flg_exit)
    {
        if (waitForWorkerPeriod(loopClosureFrequency))
            break;
        performLoopClosure();
        if (flg_exit || !ros_ok())
            break;
        visualizeLoopClosure();
    }
}

void gnssMatchingThread()
{
    if (!gnssEnableFlag)
        return;

    while (ros_ok() && !flg_exit)
    {
        if (waitForWorkerPeriod(50.0))
            break;
        if (!gnss_aligned.load())
            continue;
        performGnssMatching();
        if (gnssPathVis && p_gnss)
        {
            PosData pos;
            YawData yaw;
            if (p_gnss->latestPos(pos) && p_gnss->latestYaw(yaw))
            {
                TransformStampedMsg tf_msg;
                tf_msg.header.stamp = get_ros_time(std::max(pos.t, yaw.t));
                tf_msg.header.frame_id = map_frame;
                tf_msg.child_frame_id = "gnss_link";
                tf_msg.transform.translation.x = pos.p.x();
                tf_msg.transform.translation.y = pos.p.y();
                tf_msg.transform.translation.z = pos.p.z();
                tf_msg.transform.rotation = quaternion_from_rpy(0.0, 0.0, yaw.yaw);

            #ifdef USE_ROS1
                static tf::TransformBroadcaster br;
            #elif defined(USE_ROS2)
                static tf2_ros::TransformBroadcaster br(get_ros_node());
            #endif
                br.sendTransform(tf_msg);
            }
        }
    }
}

void publishGlobalMap() {
    if (ros_subscription_count(pubLaserCloudGlobal) == 0)
        return;

    pcl::PointCloud<PointTypeIndex>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointTypeIndex>());
    pcl::PointCloud<PointTypeIndex>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointTypeIndex>());
    pcl::PointCloud<PointTypeIndex>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointTypeIndex>());
    pcl::PointCloud<PointTypeIndex>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointTypeIndex>());

    // ikd-tree to find near key frames to visualize
    KD_TREE_PUBLIC<PointTypeIndex>::PointVector globalMapSearchPoses3D;
    pcl::PointCloud<PointTypeIndex>::Ptr pose3Snapshot(new pcl::PointCloud<PointTypeIndex>());
    pcl::PointCloud<PointTypePose>::Ptr pose6Snapshot(new pcl::PointCloud<PointTypePose>());
    vector<pcl::PointCloud<PointTypeIndex>::Ptr> cloudSnapshot;
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (cloudKeyPoses3D->points.empty() ||
            ikdtreeHistoryKeyPoses->Root_Node == nullptr)
            return;
        if (cloudKeyPoses3D->size() != cloudKeyPoses6D->size() ||
            cloudKeyPoses3D->size() != featCloudKeyFrames.size())
        {
            ROS_PRINT_ERROR(
                "[GLOBAL MAP] Keyframe invariant failed: pose3=%zu pose6=%zu clouds=%zu",
                cloudKeyPoses3D->size(), cloudKeyPoses6D->size(),
                featCloudKeyFrames.size());
            return;
        }

        *pose3Snapshot = *cloudKeyPoses3D;
        *pose6Snapshot = *cloudKeyPoses6D;
        cloudSnapshot = featCloudKeyFrames;
        ikdtreeHistoryKeyPoses->Radius_Search(
            pose3Snapshot->back(), globalMapVisualizationSearchRadius,
            globalMapSearchPoses3D);
    }

    for (int i = 0; i < (int)globalMapSearchPoses3D.size(); ++i)
    {
        const int key = static_cast<int>(globalMapSearchPoses3D[i].intensity);
        if (key >= 0 && key < static_cast<int>(pose3Snapshot->size()))
            globalMapKeyPoses->push_back(pose3Snapshot->points[key]);
    }
    // downsample near selected key frames
    pcl::VoxelGrid<PointTypeIndex> downSizeFilterGlobalMapKeyPoses; // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
    for(auto& pt : globalMapKeyPosesDS->points)
    {
        float nearestDistanceSq = std::numeric_limits<float>::infinity();
        int nearestKey = -1;
        for (const auto &candidate : pose3Snapshot->points)
        {
            const float dx = pt.x - candidate.x;
            const float dy = pt.y - candidate.y;
            const float dz = pt.z - candidate.z;
            const float distanceSq = dx * dx + dy * dy + dz * dz;
            if (distanceSq < nearestDistanceSq)
            {
                nearestDistanceSq = distanceSq;
                nearestKey = static_cast<int>(candidate.intensity);
            }
        }
        pt.intensity = static_cast<float>(nearestKey);
    }

    // extract visualized and downsampled key frames
    for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
        if (pointDistance(globalMapKeyPosesDS->points[i], pose3Snapshot->back()) > globalMapVisualizationSearchRadius)
            continue;
        int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
        if (thisKeyInd < 0 || thisKeyInd >= static_cast<int>(cloudSnapshot.size()))
            continue;
        *globalMapKeyFrames += *transformPointCloud(
            cloudSnapshot[thisKeyInd], &pose6Snapshot->points[thisKeyInd]);
    }
    // downsample visualized points
    pcl::VoxelGrid<PointTypeIndex> downSizeFilterGlobalMapKeyFrames; // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
    publishCloud(pubLaserCloudGlobal, globalMapKeyFramesDS, timeLaserInfoStamp, map_frame);
}

void visualizeGlobalMapThread()
{
    while (ros_ok() && !flg_exit){
        if (waitForWorkerPeriod(0.2))
            break;
        publishGlobalMap();
    }
}
