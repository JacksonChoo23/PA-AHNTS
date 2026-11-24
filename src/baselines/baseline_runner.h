#pragma once

#include "../core/dx12_context.h"
#include "../core/scene_geometry.h"
#include "../telemetry/telemetry_logger.h"
#include "material_pack.h"

#include <string>
#include <vector>

namespace pahnts {

enum class BaselineMode {
    B0_BCn_Full,
    B1_SFS_BCn,
    B2_NTC_OnSample,
    B3_NTC_OnFeedback
};

struct BenchmarkConfig {
    BaselineMode mode = BaselineMode::B0_BCn_Full;
    std::string config_name;
    int run_id = 0;
    uint32_t width = 1920;
    uint32_t height = 1080;
    int warmup_frames = 120;
    int measured_frames = 1000;
    uint32_t bcn_tile_cache_mib = 1536;
    uint32_t max_tile_triplets_per_frame = 48;
    bool capture_checkpoints = false;
    std::string shader_dir = "build/shaders";
    std::string capture_dir = "results/phase1/captures";
    std::string frame_csv_path = "results/phase1/frame_metrics.csv";
    std::string mem_power_csv_path = "results/phase1/memory_power.csv";
};

class BaselineRunner {
public:
    void run(
        DX12Context& ctx,
        const SceneGeometry& scene,
        const MaterialPack& pack,
        TelemetryLogger& logger,
        const BenchmarkConfig& cfg
    );
};

} // namespace pahnts
