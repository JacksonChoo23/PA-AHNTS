#pragma once

#include "json_lite.h"

#include <array>
#include <cstdint>
#include <vector>

namespace pahnts {

struct Vertex {
    float px, py, pz;
    float nx, ny, nz;
    float tx, ty, tz;
    float u, v;
    uint32_t mat_id;
};

struct BayDrawBatch {
    uint32_t bay_idx = 0;
    uint32_t index_offset = 0;
    uint32_t index_count = 0;
    float min_xyz[3]{};
    float max_xyz[3]{};
};

struct CameraKeyframe {
    float t = 0.0f;
    float pos[3]{};
    float look[3]{};
};

struct CameraMatrices {
    float view_proj[16]{};
    float cam_pos[3]{};
    float planes[6][4]{}; // frustum planes (nx, ny, nz, d)
};

class SceneGeometry {
public:
    void load_camera_path(const std::string& json_path);
    void build_gallery();

    CameraMatrices evaluate_camera(int frame_idx, float aspect_ratio) const;
    bool is_aabb_visible(const float min_xyz[3], const float max_xyz[3], const CameraMatrices& cam) const;

    const std::vector<Vertex>& vertices() const { return vertices_; }
    const std::vector<uint32_t>& indices() const { return indices_; }
    const std::vector<BayDrawBatch>& bays() const { return bays_; }
    const std::vector<std::pair<std::string, int>>& quality_checkpoints() const { return checkpoints_; }

    int warmup_frames() const { return warmup_frames_; }
    int measured_frames() const { return measured_frames_; }

private:
    std::vector<Vertex> vertices_;
    std::vector<uint32_t> indices_;
    std::vector<BayDrawBatch> bays_;

    int num_bays_ = 16;
    float bay_length_m_ = 10.0f;
    float width_m_ = 8.0f;
    float height_m_ = 6.0f;
    int num_materials_ = 32;

    int warmup_frames_ = 120;
    int measured_frames_ = 1000;
    float fov_y_deg_ = 65.0f;
    float near_z_ = 0.1f;
    float far_z_ = 220.0f;

    float warmup_center_[3] = {0.0f, 2.2f, 5.0f};
    float warmup_rx_ = 1.2f;
    float warmup_rz_ = 2.0f;
    float warmup_look_z_ = 18.0f;
    float warmup_yaw_deg_ = 25.0f;

    std::vector<CameraKeyframe> keyframes_;
    std::vector<std::pair<std::string, int>> checkpoints_;
};

} // namespace pahnts
