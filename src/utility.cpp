#include "utility.h"
#include "ros_utils.h"
using namespace std;

// CPU Params
int numberOfCores;

// Surrounding map
float surroundingkeyframeAddingDistThreshold; 
float surroundingkeyframeAddingAngleThreshold; 
float surroundingKeyframeDensity;
float surroundingKeyframeSearchRadius;

// Loop closure
bool  loopClosureEnableFlag;
float loopClosureFrequency;
int   surroundingKeyframeSize;
float historyKeyframeSearchRadius;
float historyKeyframeSearchTimeDiff;
float historyKeyframeSearchAngleThreshold;
int   historyKeyframeSearchNum;
float historyKeyframeFitnessScore;
float loopIcpMaxCorrespondenceDistance;
float loopWeight;
bool  useRobustIcpGating;
bool  robustIcpGatingDryRun;
float robustIcpInlierDistance;
float robustIcpMinInlierRatio;
float robustIcpMaxInlierMse;
float robustIcpMaxP75;
float robustIcpMinOverlapImprovement;
float robustIcpOverlapTolerance;
float robustIcpP75Tolerance;
float robustIcpMaxTranslation;
float robustIcpMaxRotationDeg;
int   robustIcpMinEvaluatedPoints;
int   robustIcpCooldownKeyframes;
bool  loopCoarseRegistrationEnable;
int   loopCoarseSourceKeyframeSearchNum;
float loopCoarseNdtLeafSize;
float loopCoarseNdtResolution;
float loopCoarseNdtStepSize;
float loopCoarseNdtTransformationEpsilon;
int   loopCoarseNdtMaximumIterations;
float loopCoarseNdtFitnessScore;
bool  loopDebugCloudsEnable;
bool  loopDebugMetricsEnable;

// global map visualization radius
float globalMapVisualizationSearchRadius;
float globalMapVisualizationPoseDensity;
float globalMapVisualizationLeafSize;

float mappingICPSize;

int ikdtreeSearchNeighborNum;

bool keyframe_export_en = false;
bool keyframe_global_pcd_en = false;

std::string map_frame = "map";
std::string odom_frame = "odom";
std::string base_frame = "base_link";
std::string high_freq_base_frame = "base_link_hf";
std::string gnss_topic = "handsfree/rtk/gnss";
std::string gnss_heading_topic = "handsfree/rtk/heading";
bool gnssEnableFlag = false;
bool gnssPathVis = false;
double gpsFactorMinDis = 5.0;
double gnss_stddev_scale = 1.0;
double gnss_min_stddev = 0.0;
bool gnss_use_fixed_origin = false;
double gnss_origin_latitude = 0.0;
double gnss_origin_longitude = 0.0;
double gnss_origin_altitude = 0.0;
std::vector<double> gnss_extrinsic_T_raw(3, 0.0);
std::vector<double> gnss_extrinsic_R_raw{1.0, 0.0, 0.0,
                                         0.0, 1.0, 0.0,
                                         0.0, 0.0, 1.0};
Eigen::Vector3d gnss_extrinsic_T = Eigen::Vector3d::Zero();
Eigen::Matrix3d gnss_extrinsic_R = Eigen::Matrix3d::Identity();
double heading_offset = 0.0;
bool useGnssYawFactor = false;
double gnss_yaw_factor_sigma = 0.10;
bool useGnssElevation = false;
std::atomic<bool> gnss_aligned(false);
bool flip_en = false;
const Eigen::Matrix3d IMU_FLIP_R = (Eigen::Matrix3d() <<
    1.0,  0.0,  0.0,
    0.0, -1.0,  0.0,
    0.0,  0.0, -1.0).finished();

int mapping_mode = 1;
bool use_online_map = true;
bool use_prior_map = false;

void set_mapping_mode()
{
    switch (mapping_mode)
    {
        case 1:
            use_online_map = true;
            use_prior_map = false;
            return;
        case 2:
            use_online_map = false;
            use_prior_map = true;
            return;
        case 3:
            use_online_map = true;
            use_prior_map = true;
            return;
        default:
            ROS_PRINT_WARN("unknown common/mode: %d, fallback to 1(online)", mapping_mode);
            mapping_mode = 1;
            use_online_map = true;
            use_prior_map = false;
            return;
    }
}

void read_liosam_params() {

    // CPU parameters
    rosparam_get("lio_sam/numberOfCores", numberOfCores, 2);

    // Keyframe Strategy
    rosparam_get("lio_sam/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0f);
    rosparam_get("lio_sam/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2f);
    rosparam_get("lio_sam/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0f);
    rosparam_get("lio_sam/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0f);

    // Loop closure parameters
    rosparam_get("lio_sam/loopClosureEnableFlag", loopClosureEnableFlag, false);
    rosparam_get("lio_sam/loopClosureFrequency", loopClosureFrequency, 1.0f);
    rosparam_get("lio_sam/surroundingKeyframeSize", surroundingKeyframeSize, 50);
    rosparam_get("lio_sam/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0f);
    rosparam_get("lio_sam/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0f);
    rosparam_get("lio_sam/historyKeyframeSearchAngleThreshold", historyKeyframeSearchAngleThreshold, 0.5f);
    rosparam_get("lio_sam/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
    rosparam_get("lio_sam/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3f);
    rosparam_get(
        "lio_sam/loopIcpMaxCorrespondenceDistance",
        loopIcpMaxCorrespondenceDistance,
        historyKeyframeSearchRadius * 2.0f);
    rosparam_get("lio_sam/loopWeight", loopWeight, 0.2f);
    rosparam_get("lio_sam/useRobustIcpGating", useRobustIcpGating, false);
    rosparam_get("lio_sam/robustIcpGatingDryRun", robustIcpGatingDryRun, true);
    rosparam_get("lio_sam/robustIcpInlierDistance", robustIcpInlierDistance, 1.0f);
    rosparam_get("lio_sam/robustIcpMinInlierRatio", robustIcpMinInlierRatio, 0.80f);
    rosparam_get("lio_sam/robustIcpMaxInlierMse", robustIcpMaxInlierMse, 0.30f);
    rosparam_get("lio_sam/robustIcpMaxP75", robustIcpMaxP75, 0.80f);
    rosparam_get("lio_sam/robustIcpMinOverlapImprovement", robustIcpMinOverlapImprovement, 0.03f);
    rosparam_get("lio_sam/robustIcpOverlapTolerance", robustIcpOverlapTolerance, 0.02f);
    rosparam_get("lio_sam/robustIcpP75Tolerance", robustIcpP75Tolerance, 0.05f);
    rosparam_get("lio_sam/robustIcpMaxTranslation", robustIcpMaxTranslation, 3.0f);
    rosparam_get("lio_sam/robustIcpMaxRotationDeg", robustIcpMaxRotationDeg, 5.0f);
    rosparam_get("lio_sam/robustIcpMinEvaluatedPoints", robustIcpMinEvaluatedPoints, 1000);
    rosparam_get("lio_sam/robustIcpCooldownKeyframes", robustIcpCooldownKeyframes, 20);
    rosparam_get("lio_sam/loopCoarseRegistrationEnable", loopCoarseRegistrationEnable, false);
    rosparam_get("lio_sam/loopCoarseSourceKeyframeSearchNum", loopCoarseSourceKeyframeSearchNum, 3);
    rosparam_get("lio_sam/loopCoarseNdtLeafSize", loopCoarseNdtLeafSize, 1.0f);
    rosparam_get("lio_sam/loopCoarseNdtResolution", loopCoarseNdtResolution, 2.0f);
    rosparam_get("lio_sam/loopCoarseNdtStepSize", loopCoarseNdtStepSize, 0.1f);
    rosparam_get("lio_sam/loopCoarseNdtTransformationEpsilon", loopCoarseNdtTransformationEpsilon, 0.01f);
    rosparam_get("lio_sam/loopCoarseNdtMaximumIterations", loopCoarseNdtMaximumIterations, 35);
    rosparam_get("lio_sam/loopCoarseNdtFitnessScore", loopCoarseNdtFitnessScore, 2.0f);
    rosparam_get("lio_sam/loopDebugCloudsEnable", loopDebugCloudsEnable, false);
    rosparam_get("lio_sam/loopDebugMetricsEnable", loopDebugMetricsEnable, false);

    // Global pointcloud visualization
    rosparam_get("lio_sam/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3f);
    rosparam_get("lio_sam/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0f);
    rosparam_get("lio_sam/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0f);

    rosparam_get("lio_sam/mappingICPSize", mappingICPSize, 0.2f);
    rosparam_get("lio_sam/ikdtreeSearchNeighborNum", ikdtreeSearchNeighborNum, 8);
    rosparam_get("lio_sam/keyframe_export_en", keyframe_export_en, false);
    rosparam_get("lio_sam/keyframe_global_pcd_en", keyframe_global_pcd_en, false);
}

void read_gnss_params() {
    rosparam_get("gnss/topic", gnss_topic, std::string("handsfree/rtk/gnss"));
    rosparam_get("gnss/heading_topic", gnss_heading_topic, std::string("handsfree/rtk/heading"));
    rosparam_get("gnss/gnssPathVis", gnssPathVis, false);
    rosparam_get("gnss/gnssEnableFlag", gnssEnableFlag, false);
    rosparam_get("gnss/gpsFactorMinDis", gpsFactorMinDis, 5.0);
    rosparam_get("gnss/stddev_scale", gnss_stddev_scale, 1.0);
    rosparam_get("gnss/min_stddev", gnss_min_stddev, 0.0);
    rosparam_get("gnss/use_fixed_origin", gnss_use_fixed_origin, false);
    rosparam_get("gnss/origin_latitude", gnss_origin_latitude, 0.0);
    rosparam_get("gnss/origin_longitude", gnss_origin_longitude, 0.0);
    rosparam_get("gnss/origin_altitude", gnss_origin_altitude, 0.0);
    gnss_stddev_scale = std::max(gnss_stddev_scale, 1e-6);
    gnss_min_stddev = std::max(gnss_min_stddev, 0.0);
    rosparam_get("gnss/extrinsic_T", gnss_extrinsic_T_raw,
                 std::vector<double>{0.0, 0.0, 0.0});
    rosparam_get("gnss/extrinsic_R", gnss_extrinsic_R_raw,
                 std::vector<double>{1.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0,
                                     0.0, 0.0, 1.0});
    gnss_extrinsic_T = Eigen::Vector3d(
        gnss_extrinsic_T_raw[0],
        gnss_extrinsic_T_raw[1],
        gnss_extrinsic_T_raw[2]);
    gnss_extrinsic_R <<
        gnss_extrinsic_R_raw[0], gnss_extrinsic_R_raw[1], gnss_extrinsic_R_raw[2],
        gnss_extrinsic_R_raw[3], gnss_extrinsic_R_raw[4], gnss_extrinsic_R_raw[5],
        gnss_extrinsic_R_raw[6], gnss_extrinsic_R_raw[7], gnss_extrinsic_R_raw[8];
    double heading_offset_deg = 0.0;
    rosparam_get("gnss/heading_offset_deg", heading_offset_deg, 0.0);
    heading_offset = heading_offset_deg * M_PI / 180.0;
    rosparam_get("gnss/useYawFactor", useGnssYawFactor, false);
    rosparam_get("gnss/yawFactorSigma", gnss_yaw_factor_sigma, 0.10);
    rosparam_get("gnss/useGnssElevation", useGnssElevation, false);

}
