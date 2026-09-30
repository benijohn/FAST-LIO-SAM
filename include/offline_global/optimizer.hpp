#pragma once

#include "offline_global/types.hpp"

namespace fast_lio_sam::offline {

struct OptimizationSummary {
  bool converged = false;
  double initial_error = 0.0;
  double first_error = 0.0;
  double final_error = 0.0;
  std::size_t gnss_factors = 0;
  std::size_t lidar_factors = 0;
  std::size_t pruned_factors = 0;
  std::size_t elevation_factors = 0;
  bool quality_passed = false;
};

OptimizationSummary optimizeGlobalGraph(
    std::vector<Session>* sessions, const std::vector<Anchor>& anchors,
    std::vector<Constraint>* constraints,
    const std::vector<ElevationConstraint>& elevation_constraints,
    const Config& config,
    const fs::path& run_root);

void exportOptimizedResults(const std::vector<Session>& sessions,
                            const std::vector<Anchor>& anchors,
                            const std::vector<Constraint>& constraints,
                            const Config& config, const fs::path& run_root,
                            const OptimizationSummary& summary,
                            const std::vector<ValidationIssue>& excluded);

}  // namespace fast_lio_sam::offline
