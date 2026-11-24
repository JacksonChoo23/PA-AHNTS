#include "scene_geometry.h"

#include <algorithm>
#include <cmath>

namespace pahnts {

namespace {

constexpr float kPi = 3.14159265358979323846f;

struct Vec3 {
    float x, y, z;
};

Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

Vec3 cross(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

Vec3 normalize(Vec3 v) {
    float len = std::sqrt(dot(v, v));
    if (len < 1e-6f) return {0.0f, 1.0f, 0.0f};
    return {v.x / len, v.y / len, v.z / len};
}

void build_look_at_lh(Vec3 eye, Vec3 at, Vec3 up, float out[16]) {
    Vec3 zaxis = normalize(at - eye);
    Vec3 xaxis = normalize(cross(up, zaxis));
    Vec3 yaxis = cross(zaxis, xaxis);

    // Row-major DX matrix
    out[0] = xaxis.x; out[1] = yaxis.x; out[2] = zaxis.x; out[3] = 0.0f;
    out[4] = xaxis.y; out[5] = yaxis.y; out[6] = zaxis.y; out[7] = 0.0f;
    out[8] = xaxis.z; out[9] = yaxis.z; out[10] = zaxis.z; out[11] = 0.0f;
    out[12] = -dot(xaxis, eye);
    out[13] = -dot(yaxis, eye);
    out[14] = -dot(zaxis, eye);
    out[15] = 1.0f;
}

void build_perspective_lh(float fov_y_rad, float aspect, float zn, float zf, float out[16]) {
    float y_scale = 1.0f / std::tan(fov_y_rad * 0.5f);
    float x_scale = y_scale / aspect;
    std::fill(out, out + 16, 0.0f);
    out[0] = x_scale;
    out[5] = y_scale;
    out[10] = zf / (zf - zn);
    out[11] = 1.0f;
    out[14] = (-zn * zf) / (zf - zn);
}

void mul_mat4(const float a[16], const float b[16], float out[16]) {
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += a[r * 4 + k] * b[k * 4 + c];
            out[r * 4 + c] = s;
        }
    }
}

void append_quad(
    std::vector<Vertex>& verts,
    std::vector<uint32_t>& inds,
    Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3,
    Vec3 n, Vec3 t,
    float u0, float v0, float u1, float v1,
    uint32_t mat_id
) {
    uint32_t base = static_cast<uint32_t>(verts.size());
    verts.push_back({p0.x, p0.y, p0.z, n.x, n.y, n.z, t.x, t.y, t.z, u0, v0, mat_id});
    verts.push_back({p1.x, p1.y, p1.z, n.x, n.y, n.z, t.x, t.y, t.z, u1, v0, mat_id});
    verts.push_back({p2.x, p2.y, p2.z, n.x, n.y, n.z, t.x, t.y, t.z, u1, v1, mat_id});
    verts.push_back({p3.x, p3.y, p3.z, n.x, n.y, n.z, t.x, t.y, t.z, u0, v1, mat_id});
    inds.push_back(base + 0); inds.push_back(base + 1); inds.push_back(base + 2);
    inds.push_back(base + 0); inds.push_back(base + 2); inds.push_back(base + 3);
}

void append_pillar(
    std::vector<Vertex>& verts,
    std::vector<uint32_t>& inds,
    float cx, float cz, float radius, float height, uint32_t mat_id
) {
    const int sides = 8;
    for (int i = 0; i < sides; ++i) {
        float a0 = static_cast<float>(i) * (2.0f * kPi / sides);
        float a1 = static_cast<float>(i + 1) * (2.0f * kPi / sides);
        float c0 = std::cos(a0), s0 = std::sin(a0);
        float c1 = std::cos(a1), s1 = std::sin(a1);
        float am = 0.5f * (a0 + a1);
        Vec3 n{std::cos(am), 0.0f, std::sin(am)};
        Vec3 t{-std::sin(am), 0.0f, std::cos(am)};
        Vec3 b0{cx + radius * c0, 0.0f, cz + radius * s0};
        Vec3 b1{cx + radius * c1, 0.0f, cz + radius * s1};
        Vec3 t1{cx + radius * c1, height, cz + radius * s1};
        Vec3 t0{cx + radius * c0, height, cz + radius * s0};
        float u0 = static_cast<float>(i) / sides;
        float u1 = static_cast<float>(i + 1) / sides;
        append_quad(verts, inds, b0, b1, t1, t0, n, t, u0, 0.0f, u1, 2.0f, mat_id);
    }
}

void append_uv_sphere(
    std::vector<Vertex>& verts,
    std::vector<uint32_t>& inds,
    Vec3 center, float radius, uint32_t mat_id
) {
    const int rings = 16;
    const int sectors = 24;
    uint32_t base = static_cast<uint32_t>(verts.size());
    for (int r = 0; r <= rings; ++r) {
        float v = static_cast<float>(r) / rings;
        float phi = v * kPi;
        float sp = std::sin(phi), cp = std::cos(phi);
        for (int s = 0; s <= sectors; ++s) {
            float u = static_cast<float>(s) / sectors;
            float theta = u * 2.0f * kPi;
            float st = std::sin(theta), ct = std::cos(theta);
            Vec3 n{sp * ct, cp, sp * st};
            Vec3 t{-st, 0.0f, ct};
            verts.push_back({
                center.x + radius * n.x, center.y + radius * n.y, center.z + radius * n.z,
                n.x, n.y, n.z, t.x, t.y, t.z, u * 2.0f, v * 2.0f, mat_id
            });
        }
    }
    for (int r = 0; r < rings; ++r) {
        for (int s = 0; s < sectors; ++s) {
            uint32_t i0 = base + static_cast<uint32_t>(r * (sectors + 1) + s);
            uint32_t i1 = i0 + 1;
            uint32_t i2 = i0 + static_cast<uint32_t>(sectors + 1);
            uint32_t i3 = i2 + 1;
            inds.push_back(i0); inds.push_back(i2); inds.push_back(i1);
            inds.push_back(i1); inds.push_back(i2); inds.push_back(i3);
        }
    }
}

} // namespace

void SceneGeometry::load_camera_path(const std::string& json_path) {
    JsonValue root = JsonParser::parse_file(json_path);
    const auto& gal = root["gallery"];
    num_bays_ = gal["num_bays"].as_int(16);
    bay_length_m_ = gal["bay_length_m"].as_float(10.0f);
    width_m_ = gal["width_m"].as_float(8.0f);
    height_m_ = gal["height_m"].as_float(6.0f);
    num_materials_ = gal["num_materials"].as_int(32);

    warmup_frames_ = root["warmup_frames"].as_int(120);
    measured_frames_ = root["measured_frames"].as_int(1000);
    fov_y_deg_ = root["fov_y_deg"].as_float(65.0f);
    near_z_ = root["near_z"].as_float(0.1f);
    far_z_ = root["far_z"].as_float(220.0f);

    const auto& wo = root["warmup_orbit"];
    for (int i = 0; i < 3; ++i) warmup_center_[i] = wo["center"][i].as_float();
    warmup_rx_ = wo["radius_x"].as_float(1.2f);
    warmup_rz_ = wo["radius_z"].as_float(2.0f);
    warmup_look_z_ = wo["look_ahead_z"].as_float(18.0f);
    warmup_yaw_deg_ = wo["yaw_osc_deg"].as_float(25.0f);

    keyframes_.clear();
    const auto& kfs = root["keyframes"];
    for (size_t i = 0; i < kfs.size(); ++i) {
        CameraKeyframe kf;
        kf.t = kfs[i]["t"].as_float();
        for (int c = 0; c < 3; ++c) {
            kf.pos[c] = kfs[i]["pos"][c].as_float();
            kf.look[c] = kfs[i]["look"][c].as_float();
        }
        keyframes_.push_back(kf);
    }

    checkpoints_.clear();
    const auto& cks = root["quality_checkpoints"];
    for (size_t i = 0; i < cks.size(); ++i) {
        checkpoints_.emplace_back(cks[i]["id"].as_string(), cks[i]["frame_idx"].as_int());
    }
}

void SceneGeometry::build_gallery() {
    vertices_.clear();
    indices_.clear();
    bays_.clear();

    float hw = width_m_ * 0.5f;
    float mid_y = height_m_ * 0.5f;

    for (int b = 0; b < num_bays_; ++b) {
        BayDrawBatch batch;
        batch.bay_idx = static_cast<uint32_t>(b);
        batch.index_offset = static_cast<uint32_t>(indices_.size());

        float z0 = static_cast<float>(b) * bay_length_m_;
        float z1 = z0 + bay_length_m_;
        batch.min_xyz[0] = -hw - 0.1f; batch.min_xyz[1] = -0.1f; batch.min_xyz[2] = z0 - 0.1f;
        batch.max_xyz[0] =  hw + 0.1f; batch.max_xyz[1] = height_m_ + 0.1f; batch.max_xyz[2] = z1 + 0.1f;

        uint32_t m_floor  = static_cast<uint32_t>((b * 2 + 0) % num_materials_);
        uint32_t m_ceil   = static_cast<uint32_t>((b * 2 + 7) % num_materials_);
        uint32_t m_l_low  = static_cast<uint32_t>((b * 2 + 1) % num_materials_);
        uint32_t m_l_up   = static_cast<uint32_t>((b * 2 + 11) % num_materials_);
        uint32_t m_r_low  = static_cast<uint32_t>((b * 2 + 2) % num_materials_);
        uint32_t m_r_up   = static_cast<uint32_t>((b * 2 + 15) % num_materials_);
        uint32_t m_pillar = static_cast<uint32_t>((b * 2 + 19) % num_materials_);
        uint32_t m_sphere = static_cast<uint32_t>((b * 2 + 25) % num_materials_);

        // Floor
        append_quad(vertices_, indices_,
                    {-hw, 0.0f, z0}, {hw, 0.0f, z0}, {hw, 0.0f, z1}, {-hw, 0.0f, z1},
                    {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 0.0f, 0.0f, 1.0f, 1.0f, m_floor);
        // Ceiling
        append_quad(vertices_, indices_,
                    {-hw, height_m_, z1}, {hw, height_m_, z1}, {hw, height_m_, z0}, {-hw, height_m_, z0},
                    {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 0.0f, 0.0f, 1.0f, 1.0f, m_ceil);
        // Left lower & upper walls
        append_quad(vertices_, indices_,
                    {-hw, 0.0f, z1}, {-hw, 0.0f, z0}, {-hw, mid_y, z0}, {-hw, mid_y, z1},
                    {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, 0.0f, 0.0f, 1.0f, 1.0f, m_l_low);
        append_quad(vertices_, indices_,
                    {-hw, mid_y, z1}, {-hw, mid_y, z0}, {-hw, height_m_, z0}, {-hw, height_m_, z1},
                    {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, 0.0f, 0.0f, 1.0f, 1.0f, m_l_up);
        // Right lower & upper walls
        append_quad(vertices_, indices_,
                    {hw, 0.0f, z0}, {hw, 0.0f, z1}, {hw, mid_y, z1}, {hw, mid_y, z0},
                    {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.0f, 0.0f, 1.0f, 1.0f, m_r_low);
        append_quad(vertices_, indices_,
                    {hw, mid_y, z0}, {hw, mid_y, z1}, {hw, height_m_, z1}, {hw, height_m_, z0},
                    {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.0f, 0.0f, 1.0f, 1.0f, m_r_up);

        // Pillars and alternating hero spheres
        float z_mid = z0 + bay_length_m_ * 0.5f;
        append_pillar(vertices_, indices_, -2.85f, z_mid, 0.45f, height_m_, m_pillar);
        append_pillar(vertices_, indices_,  2.85f, z_mid, 0.45f, height_m_, m_pillar);
        float sx = (b % 2 == 0) ? -1.55f : 1.55f;
        append_uv_sphere(vertices_, indices_, {sx, 1.65f, z_mid}, 0.85f, m_sphere);

        batch.index_count = static_cast<uint32_t>(indices_.size()) - batch.index_offset;
        bays_.push_back(batch);
    }
}

CameraMatrices SceneGeometry::evaluate_camera(int frame_idx, float aspect_ratio) const {
    Vec3 eye{0.0f, 2.0f, 2.0f};
    Vec3 look{0.0f, 2.2f, 16.0f};

    if (frame_idx < 0) {
        int wf = std::max(1, warmup_frames_);
        float u = static_cast<float>(frame_idx + wf) / static_cast<float>(wf);
        float ang = u * 2.0f * kPi;
        eye = {
            warmup_center_[0] + warmup_rx_ * std::sin(ang),
            warmup_center_[1] + 0.15f * std::cos(ang * 2.0f),
            warmup_center_[2] + warmup_rz_ * std::cos(ang)
        };
        float yaw = (warmup_yaw_deg_ * kPi / 180.0f) * std::sin(ang);
        float dz = warmup_look_z_ - eye.z;
        look = {eye.x + dz * std::sin(yaw), 2.1f, eye.z + dz * std::cos(yaw)};
    } else if (!keyframes_.empty()) {
        float t = static_cast<float>(frame_idx) / static_cast<float>(std::max(1, measured_frames_ - 1));
        t = std::clamp(t, 0.0f, 1.0f);
        size_t seg = 0;
        while (seg + 1 < keyframes_.size() && t > keyframes_[seg + 1].t) {
            ++seg;
        }
        if (seg + 1 >= keyframes_.size()) {
            const auto& k = keyframes_.back();
            eye = {k.pos[0], k.pos[1], k.pos[2]};
            look = {k.look[0], k.look[1], k.look[2]};
        } else {
            const auto& k0 = keyframes_[seg];
            const auto& k1 = keyframes_[seg + 1];
            float dt = std::max(1e-5f, k1.t - k0.t);
            float s = std::clamp((t - k0.t) / dt, 0.0f, 1.0f);
            // Smoothstep for C1 velocity continuity
            float ss = s * s * (3.0f - 2.0f * s);
            eye = {
                k0.pos[0] + (k1.pos[0] - k0.pos[0]) * ss,
                k0.pos[1] + (k1.pos[1] - k0.pos[1]) * ss,
                k0.pos[2] + (k1.pos[2] - k0.pos[2]) * ss,
            };
            look = {
                k0.look[0] + (k1.look[0] - k0.look[0]) * ss,
                k0.look[1] + (k1.look[1] - k0.look[1]) * ss,
                k0.look[2] + (k1.look[2] - k0.look[2]) * ss,
            };
        }
    }

    float view[16], proj[16];
    build_look_at_lh(eye, look, {0.0f, 1.0f, 0.0f}, view);
    build_perspective_lh(fov_y_deg_ * kPi / 180.0f, aspect_ratio, near_z_, far_z_, proj);

    CameraMatrices out;
    mul_mat4(view, proj, out.view_proj);
    out.cam_pos[0] = eye.x;
    out.cam_pos[1] = eye.y;
    out.cam_pos[2] = eye.z;

    // Extract frustum planes from row-major ViewProj (Gribb/Hartmann for D3D [0,1] Z)
    const float* m = out.view_proj;
    // Left: col3 + col0
    out.planes[0][0] = m[3] + m[0]; out.planes[0][1] = m[7] + m[4]; out.planes[0][2] = m[11] + m[8];  out.planes[0][3] = m[15] + m[12];
    // Right: col3 - col0
    out.planes[1][0] = m[3] - m[0]; out.planes[1][1] = m[7] - m[4]; out.planes[1][2] = m[11] - m[8];  out.planes[1][3] = m[15] - m[12];
    // Bottom: col3 + col1
    out.planes[2][0] = m[3] + m[1]; out.planes[2][1] = m[7] + m[5]; out.planes[2][2] = m[11] + m[9];  out.planes[2][3] = m[15] + m[13];
    // Top: col3 - col1
    out.planes[3][0] = m[3] - m[1]; out.planes[3][1] = m[7] - m[5]; out.planes[3][2] = m[11] - m[9];  out.planes[3][3] = m[15] - m[13];
    // Near: col2
    out.planes[4][0] = m[2];        out.planes[4][1] = m[6];        out.planes[4][2] = m[10];         out.planes[4][3] = m[14];
    // Far: col3 - col2
    out.planes[5][0] = m[3] - m[2]; out.planes[5][1] = m[7] - m[6]; out.planes[5][2] = m[11] - m[10]; out.planes[5][3] = m[15] - m[14];

    for (int p = 0; p < 6; ++p) {
        float l = std::sqrt(out.planes[p][0] * out.planes[p][0] +
                            out.planes[p][1] * out.planes[p][1] +
                            out.planes[p][2] * out.planes[p][2]);
        if (l > 1e-6f) {
            for (int c = 0; c < 4; ++c) out.planes[p][c] /= l;
        }
    }
    return out;
}

bool SceneGeometry::is_aabb_visible(const float min_xyz[3], const float max_xyz[3], const CameraMatrices& cam) const {
    for (int p = 0; p < 6; ++p) {
        const float* pl = cam.planes[p];
        float px = pl[0] >= 0.0f ? max_xyz[0] : min_xyz[0];
        float py = pl[1] >= 0.0f ? max_xyz[1] : min_xyz[1];
        float pz = pl[2] >= 0.0f ? max_xyz[2] : min_xyz[2];
        if (pl[0] * px + pl[1] * py + pl[2] * pz + pl[3] < 0.0f) {
            return false;
        }
    }
    return true;
}

} // namespace pahnts
