#include "../core/dx12_context.h"
#include "../core/json_lite.h"
#include "../core/scene_geometry.h"
#include "../ram_tier/phase2_reinforced.h"
#include "../ram_tier/ram_tiered_pool.h"
#include "../telemetry/telemetry_logger.h"
#include "baseline_runner.h"
#include "material_pack.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

pahnts::BaselineMode parse_mode(const std::string& name) {
    if (name == "B0_BCn_Full") return pahnts::BaselineMode::B0_BCn_Full;
    if (name == "B1_SFS_BCn") return pahnts::BaselineMode::B1_SFS_BCn;
    if (name == "B2_NTC_OnSample") return pahnts::BaselineMode::B2_NTC_OnSample;
    if (name == "B3_NTC_OnFeedback") return pahnts::BaselineMode::B3_NTC_OnFeedback;
    throw std::runtime_error("Unknown baseline mode: " + name);
}

} // namespace

int main(int argc, char** argv) {
    try {
        int phase = 1;
        std::string exp_cfg_path = "configs/experiments/phase1_baselines.json";
        std::string config_name = "B0_BCn_Full";
        int run_id = 0;
        int warmup_override = -1;
        int measured_override = -1;
        int l1_mib_override = -1;
        int ram_l2_mib_override = -1;
        bool capture_only = false;
        bool debug_layer = false;
        std::string out_dir = "results/phase1";
        std::string shader_dir = "build/shaders";

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--phase" && i + 1 < argc) phase = std::stoi(argv[++i]);
            else if (a == "--config" && i + 1 < argc) exp_cfg_path = argv[++i];
            else if ((a == "--baseline" || a == "--config-name") && i + 1 < argc) config_name = argv[++i];
            else if (a == "--run-id" && i + 1 < argc) run_id = std::stoi(argv[++i]);
            else if (a == "--warmup" && i + 1 < argc) warmup_override = std::stoi(argv[++i]);
            else if (a == "--frames" && i + 1 < argc) measured_override = std::stoi(argv[++i]);
            else if (a == "--l1-mib" && i + 1 < argc) l1_mib_override = std::stoi(argv[++i]);
            else if (a == "--ram-l2-mib" && i + 1 < argc) ram_l2_mib_override = std::stoi(argv[++i]);
            else if (a == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
            else if (a == "--shader-dir" && i + 1 < argc) shader_dir = argv[++i];
            else if (a == "--capture-checkpoints") capture_only = true;
            else if (a == "--debug-layer") debug_layer = true;
        }

        auto exp_cfg = pahnts::JsonParser::parse_file(exp_cfg_path);
        uint32_t width = static_cast<uint32_t>(exp_cfg["resolution"][0].as_int(1920));
        uint32_t height = static_cast<uint32_t>(exp_cfg["resolution"][1].as_int(1080));

        pahnts::SceneGeometry scene;
        scene.load_camera_path(exp_cfg["camera_path"].as_string());
        scene.build_gallery();

        pahnts::DX12Context ctx;
        ctx.init(width, height, "0x10DE", debug_layer);

        pahnts::TelemetryLogger logger;
        logger.init(&ctx);

        if (phase == 21) {
            pahnts::Phase2ReinforcedConfig rcfg;
            if (config_name.find("C0_") == 0) rcfg.mode = pahnts::P2RMode::C0_VRAM_Sync;
            else if (config_name.find("C1_") == 0) rcfg.mode = pahnts::P2RMode::C1_RAM_Sync;
            else if (config_name.find("C2_") == 0) rcfg.mode = pahnts::P2RMode::C2_VRAM_Async;
            else if (config_name.find("B1_") == 0) rcfg.mode = pahnts::P2RMode::B1_SFS_BCn_Async_RamCache;
            else rcfg.mode = pahnts::P2RMode::C3_RAM_Async;

            rcfg.config_name = config_name;
            rcfg.run_id = run_id;
            rcfg.width = width;
            rcfg.height = height;
            rcfg.warmup_frames = warmup_override >= 0 ? warmup_override : exp_cfg["warmup_frames"].as_int(120);
            rcfg.measured_frames = measured_override >= 0 ? measured_override : exp_cfg["measured_frames"].as_int(1000);
            rcfg.bcn_tile_cache_mib = static_cast<uint32_t>(exp_cfg["bcn_tile_cache_mib"].as_int(1536));
            rcfg.max_tile_triplets_per_frame = static_cast<uint32_t>(exp_cfg["max_tile_triplets_per_frame"].as_int(48));
            rcfg.l1_latent_pool_mib = l1_mib_override >= 0 ? static_cast<uint32_t>(l1_mib_override) : 512u;
            rcfg.ram_l2_pool_mib = ram_l2_mib_override >= 0 ? static_cast<uint32_t>(ram_l2_mib_override) : 512u;
            rcfg.capture_checkpoints = capture_only;
            rcfg.streamed_checkpoints = true;
            rcfg.manifest_path = exp_cfg["asset_manifest"].as_string();
            rcfg.shader_dir = shader_dir;
            rcfg.capture_dir = out_dir + "/captures";
            rcfg.out_dir = out_dir;

            std::cout << "[pa_ahnts_bench] Phase 2 Reinforced | " << rcfg.config_name
                      << " | run_id=" << rcfg.run_id
                      << " | L1_MiB=" << rcfg.l1_latent_pool_mib
                      << " | RAM_L2_MiB=" << rcfg.ram_l2_pool_mib
                      << (rcfg.capture_checkpoints ? " | STREAMED_CHECKPOINTS" : "")
                      << std::endl;
            try {
                pahnts::Phase2ReinforcedRunner runner;
                runner.run(ctx, scene, logger, rcfg);
            } catch (...) {
                ctx.dump_debug_messages();
                throw;
            }
            std::cout << "[pa_ahnts_bench] Completed " << rcfg.config_name << " (run " << rcfg.run_id << ")" << std::endl;
            return 0;
        }

        if (phase == 2) {
            pahnts::RamTierConfig rcfg;
            rcfg.config_name = config_name;
            rcfg.run_id = run_id;
            rcfg.width = width;
            rcfg.height = height;
            rcfg.warmup_frames = warmup_override >= 0 ? warmup_override : exp_cfg["warmup_frames"].as_int(120);
            rcfg.measured_frames = measured_override >= 0 ? measured_override : exp_cfg["measured_frames"].as_int(1000);
            rcfg.bcn_tile_cache_mib = static_cast<uint32_t>(exp_cfg["bcn_tile_cache_mib"].as_int(1536));
            rcfg.max_tile_triplets_per_frame = static_cast<uint32_t>(exp_cfg["max_tile_triplets_per_frame"].as_int(48));
            rcfg.l1_latent_pool_mib = l1_mib_override >= 0 ? static_cast<uint32_t>(l1_mib_override) : 512u;
            rcfg.ram_l2_pool_mib = ram_l2_mib_override >= 0 ? static_cast<uint32_t>(ram_l2_mib_override) : 2560u;
            rcfg.capture_checkpoints = capture_only;
            rcfg.manifest_path = exp_cfg["asset_manifest"].as_string();
            rcfg.shader_dir = shader_dir;
            rcfg.capture_dir = out_dir + "/captures";
            rcfg.frame_csv_path = out_dir + "/frame_metrics.csv";
            rcfg.mem_power_csv_path = out_dir + "/memory_power.csv";
            rcfg.pcie_trace_csv_path = out_dir + "/pcie_transfer_trace.csv";

            std::cout << "[pa_ahnts_bench] Phase 2 | " << rcfg.config_name
                      << " | run_id=" << rcfg.run_id
                      << " | L1_MiB=" << rcfg.l1_latent_pool_mib
                      << " | RAM_L2_MiB=" << rcfg.ram_l2_pool_mib
                      << (rcfg.capture_checkpoints ? " | CAPTURE_CHECKPOINTS" : "")
                      << std::endl;
            try {
                pahnts::RamTierRunner runner;
                runner.run(ctx, scene, logger, rcfg);
            } catch (...) {
                ctx.dump_debug_messages();
                throw;
            }
            std::cout << "[pa_ahnts_bench] Completed " << rcfg.config_name << " (run " << rcfg.run_id << ")" << std::endl;
            return 0;
        }

        pahnts::BenchmarkConfig cfg;
        cfg.mode = parse_mode(config_name);
        cfg.config_name = config_name;
        cfg.run_id = run_id;
        cfg.width = width;
        cfg.height = height;
        cfg.warmup_frames = warmup_override >= 0 ? warmup_override : exp_cfg["warmup_frames"].as_int(120);
        cfg.measured_frames = measured_override >= 0 ? measured_override : exp_cfg["measured_frames"].as_int(1000);
        cfg.bcn_tile_cache_mib = static_cast<uint32_t>(exp_cfg["bcn_tile_cache_mib"].as_int(1536));
        cfg.max_tile_triplets_per_frame = static_cast<uint32_t>(exp_cfg["max_tile_triplets_per_frame"].as_int(48));
        cfg.capture_checkpoints = capture_only;
        cfg.shader_dir = shader_dir;
        cfg.capture_dir = out_dir + "/captures";
        cfg.frame_csv_path = out_dir + "/frame_metrics.csv";
        cfg.mem_power_csv_path = out_dir + "/memory_power.csv";

        pahnts::MaterialPack pack;
        pack.load(exp_cfg["asset_manifest"].as_string());

        std::cout << "[pa_ahnts_bench] Running " << cfg.config_name
                  << " | run_id=" << cfg.run_id
                  << " | warmup=" << cfg.warmup_frames
                  << " | measured=" << cfg.measured_frames
                  << " | feedback_tier=" << ctx.sampler_feedback_tier()
                  << (cfg.capture_checkpoints ? " | CAPTURE_CHECKPOINTS" : "")
                  << std::endl;

        try {
            pahnts::BaselineRunner runner;
            runner.run(ctx, scene, pack, logger, cfg);
        } catch (...) {
            ctx.dump_debug_messages();
            throw;
        }

        std::cout << "[pa_ahnts_bench] Completed " << cfg.config_name << " (run " << cfg.run_id << ")" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[pa_ahnts_bench] FATAL: " << ex.what() << std::endl;
        return 1;
    }
}
