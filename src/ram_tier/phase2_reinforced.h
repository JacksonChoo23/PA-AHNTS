#pragma once

#include "../baselines/material_pack.h"
#include "../core/dx12_context.h"
#include "../core/scene_geometry.h"
#include "../telemetry/telemetry_logger.h"

#include <cstdint>
#include <string>

namespace pahnts {

enum class P2RMode {
    C0_VRAM_Sync,
    C1_RAM_Sync,
    C2_VRAM_Async,
    C3_RAM_Async,
    B1_SFS_BCn_Async_RamCache,
};

struct Phase2ReinforcedConfig {
    P2RMode mode = P2RMode::C3_RAM_Async;
    std::string config_name;
    int run_id = 0;
    uint32_t width = 1920;
    uint32_t height = 1080;
    int warmup_frames = 120;
    int measured_frames = 1000;
    uint32_t bcn_tile_cache_mib = 1536;
    uint32_t max_tile_triplets_per_frame = 48;
    uint32_t l1_latent_pool_mib = 512;
    uint32_t ram_l2_pool_mib = 512;
    bool capture_checkpoints = false;
    bool streamed_checkpoints = true; // Capture live streamed frames without artificial stall waits
    std::string manifest_path = "results/assets_cache/manifest.json";
    std::string shader_dir = "build/shaders";
    std::string capture_dir = "results/phase2_reinforced/captures";
    std::string out_dir = "results/phase2_reinforced";
};

class Phase2ReinforcedRunner {
public:
    void run(
        DX12Context& ctx,
        const SceneGeometry& scene,
        TelemetryLogger& logger,
        const Phase2ReinforcedConfig& cfg
    );
};

} // namespace pahnts
