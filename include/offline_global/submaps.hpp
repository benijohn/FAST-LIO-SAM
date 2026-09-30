#pragma once

#include "offline_global/types.hpp"

namespace fast_lio_sam::offline {

std::vector<Anchor> buildOrLoadSubmaps(std::vector<Session>* sessions,
                                       const Config& config,
                                       const fs::path& cache_root);
Cloud::Ptr loadCloud(const fs::path& path);

}  // namespace fast_lio_sam::offline
