#include "offline_global/submaps.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace fast_lio_sam::offline {
namespace {

std::uint64_t fnv1a(std::uint64_t hash, const void* bytes, std::size_t count) {
  const auto* data = static_cast<const unsigned char*>(bytes);
  for (std::size_t i = 0; i < count; ++i) {
    hash ^= data[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

template <typename T>
void hashValue(std::uint64_t* hash, const T& value) {
  *hash = fnv1a(*hash, &value, sizeof(value));
}

std::string cacheKey(const Session& session, const Config& config) {
  std::uint64_t hash = 1469598103934665603ULL;
  hashValue(&hash, config.anchor_spacing);
  hashValue(&hash, config.submap_window);
  hashValue(&hash, config.min_range);
  hashValue(&hash, config.max_range);
  hashValue(&hash, config.coarse_voxel);
  hashValue(&hash, config.fine_voxel);
  hash = fnv1a(hash, session.lidar_to_body_R.data(), sizeof(double) * 9);
  hash = fnv1a(hash, session.lidar_to_body_t.data(), sizeof(double) * 3);
  for (const auto& key : session.keyframes) {
    hashValue(&hash, key.stamp);
    const auto matrix = key.original.matrix();
    hash = fnv1a(hash, matrix.data(), sizeof(double) * 16);
    const auto size = fs::file_size(key.scan_path);
    const auto modified = fs::last_write_time(key.scan_path).time_since_epoch().count();
    hashValue(&hash, size);
    hashValue(&hash, modified);
  }
  std::ostringstream value;
  value << std::hex << std::setw(16) << std::setfill('0') << hash;
  return value.str();
}

Cloud::Ptr voxelize(const Cloud::ConstPtr& input, double leaf) {
  auto output = std::make_shared<Cloud>();
  pcl::VoxelGrid<pcl::PointXYZ> filter;
  filter.setLeafSize(leaf, leaf, leaf);
  filter.setInputCloud(input);
  filter.filter(*output);
  return output;
}

Cloud::Ptr loadSavedScan(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("failed to open " + path.string());
  std::vector<std::string> fields, types;
  std::vector<int> sizes, counts;
  std::size_t points = 0;
  std::string data_encoding;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream parser(line);
    std::string keyword;
    parser >> keyword;
    if (keyword == "FIELDS") { std::string v; while (parser >> v) fields.push_back(v); }
    else if (keyword == "SIZE") { int v; while (parser >> v) sizes.push_back(v); }
    else if (keyword == "TYPE") { std::string v; while (parser >> v) types.push_back(v); }
    else if (keyword == "COUNT") { int v; while (parser >> v) counts.push_back(v); }
    else if (keyword == "POINTS") parser >> points;
    else if (keyword == "DATA") { parser >> data_encoding; break; }
  }
  if (data_encoding != "binary" || points == 0 || fields.empty() ||
      fields.size() != sizes.size() || fields.size() != types.size()) {
    auto fallback = std::make_shared<Cloud>();
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(path.string(), *fallback) < 0)
      throw std::runtime_error("unsupported or malformed PCD " + path.string());
    return fallback;
  }
  if (counts.empty()) counts.assign(fields.size(), 1);
  if (counts.size() != fields.size()) throw std::runtime_error("malformed PCD COUNT in " + path.string());
  std::size_t stride = 0, x_offset = SIZE_MAX, y_offset = SIZE_MAX, z_offset = SIZE_MAX;
  for (std::size_t i = 0; i < fields.size(); ++i) {
    if (fields[i] == "x") x_offset = stride;
    if (fields[i] == "y") y_offset = stride;
    if (fields[i] == "z") z_offset = stride;
    if ((fields[i] == "x" || fields[i] == "y" || fields[i] == "z") &&
        (sizes[i] != 4 || types[i] != "F" || counts[i] != 1)) {
      throw std::runtime_error("saved scan XYZ fields must be float32 in " + path.string());
    }
    stride += static_cast<std::size_t>(sizes[i] * counts[i]);
  }
  if (x_offset == SIZE_MAX || y_offset == SIZE_MAX || z_offset == SIZE_MAX || stride == 0)
    throw std::runtime_error("saved scan has no XYZ fields: " + path.string());
  std::vector<char> record(stride);
  auto output = std::make_shared<Cloud>();
  output->reserve(points);
  for (std::size_t i = 0; i < points; ++i) {
    if (!input.read(record.data(), record.size()))
      throw std::runtime_error("truncated PCD payload in " + path.string());
    float x, y, z;
    std::memcpy(&x, record.data() + x_offset, sizeof(float));
    std::memcpy(&y, record.data() + y_offset, sizeof(float));
    std::memcpy(&z, record.data() + z_offset, sizeof(float));
    output->emplace_back(x, y, z);
  }
  output->width = output->size(); output->height = 1; output->is_dense = false;
  return output;
}

std::vector<std::size_t> selectAnchors(const Session& session, double spacing) {
  std::vector<std::size_t> result = {0};
  double next = spacing;
  for (std::size_t i = 1; i + 1 < session.keyframes.size(); ++i) {
    if (session.keyframes[i].travel >= next) {
      result.push_back(i);
      next = session.keyframes[i].travel + spacing;
    }
  }
  if (session.keyframes.size() > 1 && result.back() != session.keyframes.size() - 1) {
    result.push_back(session.keyframes.size() - 1);
  }
  return result;
}

Cloud::Ptr constructSubmap(const Session& session, std::size_t anchor_index,
                           const Config& config) {
  auto merged = std::make_shared<Cloud>();
  const auto& anchor = session.keyframes[anchor_index];
  const gtsam::Pose3 world_to_anchor = anchor.original.inverse();
  const double half_window = config.submap_window * 0.5;
  for (std::size_t i = 0; i < session.keyframes.size(); ++i) {
    if (std::abs(session.keyframes[i].travel - anchor.travel) > half_window) continue;
    const auto raw = loadSavedScan(session.keyframes[i].scan_path);
    const gtsam::Pose3 scan_to_anchor = world_to_anchor.compose(session.keyframes[i].original);
    const Eigen::Matrix3d rotation = scan_to_anchor.rotation().matrix();
    const Eigen::Vector3d translation = scan_to_anchor.translation();
    merged->reserve(merged->size() + raw->size());
    for (const auto& point : *raw) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
          !std::isfinite(point.z)) continue;
      const Eigen::Vector3d lidar(point.x, point.y, point.z);
      const double range = lidar.norm();
      if (range < config.min_range || range > config.max_range) continue;
      const Eigen::Vector3d body = session.lidar_to_body_R * lidar +
                                   session.lidar_to_body_t;
      const Eigen::Vector3d local = rotation * body + translation;
      merged->emplace_back(local.x(), local.y(), local.z());
    }
  }
  if (merged->empty()) throw std::runtime_error("constructed an empty submap");
  merged->width = merged->size();
  merged->height = 1;
  merged->is_dense = true;
  return merged;
}

}  // namespace

Cloud::Ptr loadCloud(const fs::path& path) {
  auto cloud = std::make_shared<Cloud>();
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(path.string(), *cloud) < 0) {
    throw std::runtime_error("failed to load cached cloud " + path.string());
  }
  return cloud;
}

std::vector<Anchor> buildOrLoadSubmaps(std::vector<Session>* sessions,
                                       const Config& config,
                                       const fs::path& cache_root) {
  std::vector<Anchor> anchors;
  for (auto& session : *sessions) {
    session.anchors = selectAnchors(session, config.anchor_spacing);
    session.cache_key = cacheKey(session, config);
    const fs::path session_cache = cache_root / session.date / session.time /
                                   session.bag / session.cache_key;
    fs::create_directories(session_cache);
    for (const std::size_t keyframe : session.anchors) {
      Anchor anchor;
      anchor.global_index = anchors.size();
      anchor.session = session.index;
      anchor.keyframe = keyframe;
      std::ostringstream stem;
      stem << "anchor_" << std::setw(6) << std::setfill('0') << keyframe;
      anchor.coarse_path = session_cache / (stem.str() + "_coarse.pcd");
      anchor.fine_path = session_cache / (stem.str() + "_fine.pcd");
      if (!fs::is_regular_file(anchor.coarse_path) ||
          !fs::is_regular_file(anchor.fine_path)) {
        const auto full = constructSubmap(session, keyframe, config);
        const auto coarse = voxelize(full, config.coarse_voxel);
        const auto fine = voxelize(full, config.fine_voxel);
        if (pcl::io::savePCDFileBinaryCompressed(anchor.coarse_path.string(), *coarse) < 0 ||
            pcl::io::savePCDFileBinaryCompressed(anchor.fine_path.string(), *fine) < 0) {
          throw std::runtime_error("failed writing submap cache " + session_cache.string());
        }
      }
      anchors.push_back(std::move(anchor));
    }
    std::ofstream manifest(session_cache / "manifest.txt");
    manifest << "session=" << session.id << "\ncache_key=" << session.cache_key
             << "\nanchors=" << session.anchors.size() << "\n";
  }
  return anchors;
}

}  // namespace fast_lio_sam::offline
