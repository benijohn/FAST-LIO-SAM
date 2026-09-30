# Offline multi-session optimization

`offline_global_optimizer` is isolated from the online FAST-LIO-SAM node. It
reads completed batch results but never changes them. All cache and result files
are written beneath `--output-root`.

## Run the complete pipeline

```bash
source /opt/ros/jazzy/setup.bash
source ~/workspace/projects/auto-rzr/install/setup.bash

ros2 run fast_lio_sam offline_global_optimizer \
  --config ~/workspace/projects/auto-rzr/src/mapping/fast_lio_sam/config/offline_global.yaml \
  --results-root /data/gulches_data/results \
  --output-root /data/gulches_data/global_optimization \
  --stage all
```

The program prints the timestamped run directory when it starts. A `COMPLETE`
file is written only when the requested stage and all of its prerequisites
succeed.

## Resume expensive work

Use one run directory for the registration and optimization stages:

```bash
RUN=/data/gulches_data/global_optimization/20260924_180000

ros2 run fast_lio_sam offline_global_optimizer ... --run-dir "$RUN" --stage register
ros2 run fast_lio_sam offline_global_optimizer ... --run-dir "$RUN" --stage optimize
```

Submaps live in `global_optimization/cache/submaps` and are content-keyed by the
input scan metadata, poses, saved LiDAR-to-IMU extrinsic, and submap parameters.
Changing registration gates reuses them. Changing a submap input or relevant
submap parameter creates a new cache key without deleting the old cache.

Available stages are `validate`, `submaps`, `register`, `optimize`, and `all`.
`optimize` expects `constraints.csv` in the specified `--run-dir`.

## Coordinate and bias conventions

- A saved keyframe pose is `ENU_T_IMU`.
- A saved PCD contains LiDAR-frame points. The per-session saved mapping
  extrinsic is applied as `p_IMU = R * p_LIDAR + T`.
- The GNSS lever arm is reconstructed from the saved LiDAR-to-IMU and
  LiDAR-to-antenna extrinsics.
- The scalar session bias is defined by the residual
  `predicted_antenna_z + z_bias - measured_gnss_z`. A positive value therefore
  means the GNSS height is above the LiDAR-aligned trajectory height.
- High-rate output is produced afterward by `offline_trajectory_refiner`, which
  interpolates the solved keyframe corrections onto recorded high-rate odometry.
  Neither program rewrites an MCAP.

## Horizontal and vertical optimization

Horizontal alignment is intentionally rigid per session. One `Pose2` adjusts
X, Y, and yaw at the first keyframe; all other keyframes retain their original
relative X/Y shape. Original roll and pitch are copied unchanged. A long run
therefore cannot exploit a small pitch change to manufacture a large elevation
change.

Before LiDAR registration, the optimizer finds repeated trail locations from
the original X/Y trajectories. Parallel or reverse-direction overlaps create
sampled elevation-profile factors. Perpendicular intersections create one
clustered crossing-height factor with looser uncertainty. A robust scalar graph
estimates a constant Z pre-alignment for each supported session. This shift is
used by the LiDAR candidate search and registration initial transform.

The final vertical solve has one scalar correction at every submap anchor.
Repeated-XY terrain matches, accepted LiDAR registrations, and GNSS height
factors constrain these values. First- and second-difference priors make the
correction field change smoothly, while preserving the original FAST-LIO
terrain undulations between anchors. A separate constant GNSS-height bias is
estimated for every session. GNSS X/Y remains in the horizontal graph and uses
the recorded covariance.

The optimizer refuses to write `COMPLETE` when configured physical-quality
limits are exceeded, including a clamped elevation pre-alignment, excessive
local vertical deformation, excessive Z bias, or excessive keyframe movement.
Diagnostic outputs are still written for inspection.

Known unusable recordings can be listed under `sessions.exclude` as either
`DATE/TIME` or `DATE/TIME/BAG`. They remain untouched and appear in validation
output with an explicit exclusion reason.

Open `trajectory_comparison.svg` for a static X/Y overview or
`trajectory_comparison.html` for the interactive Plotly 3-D view. The HTML uses
the Plotly CDN and therefore needs network access when opened.
