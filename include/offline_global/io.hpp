#pragma once

#include "offline_global/types.hpp"

#include <yaml-cpp/yaml.h>

#include <ostream>

namespace fast_lio_sam::offline {

Config loadConfig(const fs::path& path);
std::vector<Session> discoverSessions(const fs::path& results_root,
                                      const Config& config,
                                      std::vector<ValidationIssue>* excluded);
void loadGnss(Session* session, const Config& config);
void writeValidationReport(const fs::path& path,
                           const std::vector<Session>& sessions,
                           const std::vector<ValidationIssue>& excluded);
std::string jsonEscape(const std::string& input);
fs::path makeTimestampedRunDirectory(const fs::path& output_root);
void writeCompleteMarker(const fs::path& run_root, const std::string& stage);

}  // namespace fast_lio_sam::offline
