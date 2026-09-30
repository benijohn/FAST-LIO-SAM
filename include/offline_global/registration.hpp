#pragma once

#include "offline_global/types.hpp"

namespace fast_lio_sam::offline {

std::vector<Constraint> registerCrossSessionSubmaps(
    std::vector<Session>* sessions, const std::vector<Anchor>& anchors,
    const Config& config, const fs::path& run_root);
void writeConstraintsCsv(const fs::path& path,
                         const std::vector<Constraint>& constraints,
                         const std::vector<Anchor>& anchors,
                         const std::vector<Session>& sessions);
std::vector<Constraint> readConstraintsCsv(const fs::path& path);

}  // namespace fast_lio_sam::offline
