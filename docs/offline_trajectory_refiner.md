# Offline high-rate trajectory refinement

`offline_trajectory_refiner` transfers finalized keyframe corrections onto the
recorded `/OdometryHighFreq` trajectory. It preserves FAST-LIO's measured
high-frequency relative motion while making the trajectory pass exactly through
the selected keyframe anchors.

For raw high-rate pose `H(t)` and target keyframe `K_i`, the program first places
the raw relative motion for interval `i` at `K_i`. It then measures the local
SE(3) residual required for that motion to arrive exactly at `K_(i+1)` and
distributes only that residual with a C2 smootherstep curve. This is invariant to
the arbitrary global origin; directly interpolating `map_T_odom` can turn a small
rotation about a distant origin into meters of apparent translation. Corrections
are held constant outside the first and last keyframes and those samples are
marked as extrapolated.

Singleton sessions are supported by applying the single anchor correction
rigidly to their high-rate odometry. They are reported with one anchor and no
within-session interval-adjustment metrics.

The tool never modifies source result folders, recorded bags, or global
optimization output.

## Test using finalized FAST-LIO keyframes

Omit `--target-root` to use each session's `KEY_FRAMES/keyframe_pose.txt`:

```bash
source /opt/ros/jazzy/setup.bash
source ~/workspace/projects/auto-rzr/install/setup.bash

ros2 run fast_lio_sam offline_trajectory_refiner \
  --config ~/workspace/projects/auto-rzr/src/mapping/fast_lio_sam/config/offline_trajectory.yaml \
  --results-root /data/gulches_data/results \
  --output-root /data/gulches_data/trajectory_refinement \
  --session 20260804/102029/1785853229_ros
```

`--session` may be repeated. With no session filters, all completed source
sessions are processed.

## Use globally optimized keyframes

After a global optimizer run has a `COMPLETE` marker:

```bash
GLOBAL_RUN=/data/gulches_data/global_optimization/<completed-run>

ros2 run fast_lio_sam offline_trajectory_refiner \
  --config ~/workspace/projects/auto-rzr/src/mapping/fast_lio_sam/config/offline_trajectory.yaml \
  --results-root /data/gulches_data/results \
  --target-root "$GLOBAL_RUN" \
  --output-root /data/gulches_data/trajectory_refinement
```

The output uses the same `date/time/bag` hierarchy and contains:

- `refined_highrate_pose.csv`: timestamped refined pose, interpolated correction,
  bracketing keyframe indices, interpolation fraction, and extrapolation flag.
- `correction_knots.csv`: target, time-interpolated raw pose, and exact correction
  at every keyframe.
- `trajectory_quality.json`: timing gaps, anchor errors, local interval
  adjustments and rates, frame IDs, warnings, and quality status.
- A per-session `COMPLETE` marker only when its quality checks pass.
- A run-level `summary.csv` and `COMPLETE` marker only when all requested
  sessions succeed.

This trajectory supplies body poses. Dense LiDAR accumulation must still query
the trajectory at each point or column timestamp and apply the calibrated
body-to-LiDAR transform; using one pose for an entire Ouster revolution will not
provide equivalent deskewing.
