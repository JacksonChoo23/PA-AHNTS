#pragma once

#include "../baselines/material_pack.h"
#include "../core/dx12_context.h"
#include "../core/scene_geometry.h"
#include "../telemetry/telemetry_logger.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace pahnts {

struct RamTierConfig {
    std::string config_name;
    int run_id = 0;
    uint32_t width = 1920;
    uint32_t height = 1080;
    int warmup_frames = 120;
    int measured_frames = 1000;
    uint32_t bcn_tile_cache_mib = 1536;
    uint32_t max_tile_triplets_per_frame = 48;
    uint32_t l1_latent_pool_mib = 512;
    uint32_t ram_l2_pool_mib = 2560;
    bool capture_checkpoints = false;
    std::string manifest_path = "results/assets_cache/manifest.json";
    std::string shader_dir = "build/shaders";
    std::string capture_dir = "results/phase2/captures";
    std::string frame_csv_path = "results/phase2/frame_metrics.csv";
    std::string mem_power_csv_path = "results/phase2/memory_power.csv";
    std::string pcie_trace_csv_path = "results/phase2/pcie_transfer_trace.csv";
};

struct PcieTransferRow {
    int run_id = 0;
    std::string config_name;
    uint32_t batch_idx = 0;
    int frame_idx = 0;
    uint32_t latent_tiles = 0;
    double upload_kb = 0.0;
    double copy_gpu_latency_us = 0.0;
    double cpu_submit_us = 0.0;
    double bandwidth_gbps = 0.0;
};

class RamTierRunner {
public:
    void run(
        DX12Context& ctx,
        const SceneGeometry& scene,
        TelemetryLogger& logger,
        const RamTierConfig& cfg
    );
};

} // namespace pahnts
