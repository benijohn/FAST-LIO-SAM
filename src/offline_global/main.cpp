#include "offline_global/io.hpp"
#include "offline_global/elevation.hpp"
#include "offline_global/optimizer.hpp"
#include "offline_global/registration.hpp"
#include "offline_global/submaps.hpp"

#include <filesystem>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using namespace fast_lio_sam::offline;

namespace {

struct Arguments {
  fs::path config;
  fs::path results_root;
  fs::path output_root;
  fs::path run_dir;
  std::string stage = "all";
};

void usage() {
  std::cout << "Usage: offline_global_optimizer --config FILE --results-root DIR "
               "--output-root DIR [--run-dir DIR] "
               "[--stage validate|submaps|register|optimize|all]\n";
}

Arguments parse(int argc, char** argv) {
  Arguments args;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    auto value = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("missing value after " + option);
      return argv[i];
    };
    if (option == "--config") args.config = value();
    else if (option == "--results-root") args.results_root = value();
    else if (option == "--output-root") args.output_root = value();
    else if (option == "--run-dir") args.run_dir = value();
    else if (option == "--stage") args.stage = value();
    else if (option == "--help" || option == "-h") { usage(); std::exit(0); }
    else throw std::runtime_error("unknown option: " + option);
  }
  if (args.config.empty() || args.results_root.empty() || args.output_root.empty()) {
    throw std::runtime_error("--config, --results-root, and --output-root are required");
  }
  const std::set<std::string> valid = {"validate", "submaps", "register", "optimize", "all"};
  if (!valid.count(args.stage)) throw std::runtime_error("invalid stage: " + args.stage);
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Arguments args = parse(argc, argv);
    const Config config = loadConfig(args.config);
    const fs::path run_root = args.run_dir.empty()
        ? makeTimestampedRunDirectory(args.output_root) : args.run_dir;
    fs::create_directories(run_root);
    const fs::path saved_config = run_root / "offline_global.yaml";
    if (!args.run_dir.empty() && fs::is_regular_file(saved_config)) {
      std::ifstream current(args.config, std::ios::binary), saved(saved_config, std::ios::binary);
      const std::string current_text((std::istreambuf_iterator<char>(current)), {});
      const std::string saved_text((std::istreambuf_iterator<char>(saved)), {});
      if (current_text != saved_text) {
        throw std::runtime_error("--run-dir configuration differs from its saved offline_global.yaml");
      }
    }
    fs::remove(run_root / "COMPLETE");
    fs::copy_file(args.config, run_root / "offline_global.yaml",
                  fs::copy_options::overwrite_existing);
    std::cout << "[offline] output: " << run_root << '\n';

    std::vector<ValidationIssue> excluded;
    auto discovered = discoverSessions(args.results_root, config, &excluded);
    std::vector<Session> sessions;
    sessions.reserve(discovered.size());
    for (auto& session : discovered) {
      try {
        loadGnss(&session, config);
        session.index = sessions.size();
        sessions.push_back(std::move(session));
      } catch (const std::exception& error) {
        excluded.push_back({session.root, error.what()});
      }
    }
    if (sessions.empty()) throw std::runtime_error("all COMPLETE sessions failed validation");
    writeValidationReport(run_root / "validation.json", sessions, excluded);
    std::cout << "[validate] included=" << sessions.size()
              << " excluded=" << excluded.size() << '\n';
    if (args.stage == "validate") {
      writeCompleteMarker(run_root, args.stage);
      return 0;
    }

    std::vector<ElevationConstraint> elevation_constraints;
    while (true) {
      elevation_constraints = estimateElevationPrealignment(&sessions, config, run_root);
      const bool rejected_clamped = std::any_of(sessions.begin(), sessions.end(),
          [](const Session& session) { return session.elevation_offset_clamped; });
      if (!rejected_clamped) break;
      std::vector<Session> retained;
      retained.reserve(sessions.size());
      for (auto& session : sessions) {
        if (session.elevation_offset_clamped) {
          excluded.push_back({session.root,
              "terrain elevation prealignment exceeded the configured safe limit"});
        } else {
          session.index = retained.size();
          retained.push_back(std::move(session));
        }
      }
      sessions = std::move(retained);
      if (sessions.empty())
        throw std::runtime_error("terrain prealignment rejected every session");
      std::cout << "[elevation] rejected clamped sessions; recomputing with "
                << sessions.size() << " sessions\n";
    }
    writeValidationReport(run_root / "validation.json", sessions, excluded);

    const fs::path cache_root = args.output_root / "cache" / "submaps";
    const auto anchors = buildOrLoadSubmaps(&sessions, config, cache_root);
    std::cout << "[submaps] anchors=" << anchors.size() << " cache=" << cache_root << '\n';
    if (args.stage == "submaps") {
      writeCompleteMarker(run_root, args.stage);
      return 0;
    }

    std::vector<Constraint> constraints;
    if (args.stage == "optimize") {
      constraints = readConstraintsCsv(run_root / "constraints.csv");
      for (const auto& constraint : constraints) {
        if (!constraint.accepted || constraint.pruned) continue;
        sessions[anchors.at(constraint.source_anchor).session].lidar_supported = true;
        sessions[anchors.at(constraint.target_anchor).session].lidar_supported = true;
      }
    } else {
      constraints = registerCrossSessionSubmaps(&sessions, anchors, config, run_root);
    }
    if (args.stage == "register") {
      writeCompleteMarker(run_root, args.stage);
      return 0;
    }

    auto summary = optimizeGlobalGraph(&sessions, anchors, &constraints,
                                       elevation_constraints, config, run_root);
    writeConstraintsCsv(run_root / "constraints.csv", constraints, anchors, sessions);
    exportOptimizedResults(sessions, anchors, constraints, config, run_root, summary, excluded);
    if (!summary.converged ||
        (config.quality_fail_on_violation && !summary.quality_passed)) {
      throw std::runtime_error(
          "optimizer convergence or physical-quality checks failed; COMPLETE not written");
    }
    writeCompleteMarker(run_root, args.stage);
    std::cout << "[complete] final_error=" << summary.final_error
              << " pruned=" << summary.pruned_factors << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "offline_global_optimizer: " << error.what() << '\n';
    return 1;
  }
}
