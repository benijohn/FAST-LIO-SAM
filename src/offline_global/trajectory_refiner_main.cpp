#include "offline_global/trajectory_refinement.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using fast_lio_sam::offline::runTrajectoryRefiner;

namespace {

struct Arguments {
  fs::path config;
  fs::path results_root;
  fs::path target_root;
  fs::path output_root;
  std::vector<std::string> sessions;
};

void usage() {
  std::cout
      << "Usage: offline_trajectory_refiner --config FILE --results-root DIR "
         "--output-root DIR [--target-root GLOBAL_RUN] "
         "[--session DATE/TIME/BAG ...]\n\n"
         "Without --target-root, finalized FAST-LIO keyframes are used as the "
         "anchors. With --target-root, optimized keyframes are read from "
         "GLOBAL_RUN/sessions/DATE/TIME/BAG/optimized_keyframe_pose.txt.\n";
}

Arguments parse(int argc, char** argv) {
  Arguments result;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    auto value = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("missing value after " + option);
      return argv[i];
    };
    if (option == "--config") result.config = value();
    else if (option == "--results-root") result.results_root = value();
    else if (option == "--target-root") result.target_root = value();
    else if (option == "--output-root") result.output_root = value();
    else if (option == "--session") result.sessions.push_back(value());
    else if (option == "--help" || option == "-h") {
      usage(); std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }
  if (result.config.empty() || result.results_root.empty() ||
      result.output_root.empty()) {
    throw std::runtime_error("--config, --results-root, and --output-root are required");
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = parse(argc, argv);
    return runTrajectoryRefiner(args.results_root, args.target_root,
                                args.output_root, args.config, args.sessions);
  } catch (const std::exception& error) {
    std::cerr << "offline_trajectory_refiner: " << error.what() << '\n';
    return 1;
  }
}
