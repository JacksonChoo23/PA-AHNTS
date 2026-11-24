#pragma once

#include "../core/dx12_context.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace pahnts {

struct FrameMetricRow {
    int run_id = 0;
    std::string config_name;
    int frame_idx = 0;
    double timestamp_ms = 0.0;
    double cpu_frame_ms = 0.0;
    double gpu_frame_ms = 0.0;
    double present_ms = 0.0;
    double pcie_upload_mb = 0.0;
    uint32_t tiles_decoded_dgpu = 0;
    uint32_t tiles_decoded_npu = 0;
    uint32_t tiles_decoded_igpu = 0;
    uint32_t tiles_decoded_cpu = 0;
};

struct MemoryPowerRow {
    int run_id = 0;
    std::string config_name;
    int sample_idx = 0;
    double timestamp_ms = 0.0;
    double local_vram_mb = 0.0;
    double nonlocal_vram_mb = 0.0;
    double sys_ram_latent_mb = 0.0;
    double dgpu_power_w = 0.0;
    double cpu_pkg_power_w = 0.0;
    double npu_power_w = 0.0;
    double total_est_power_w = 0.0;
    uint32_t dgpu_clock_mhz = 0;
    uint32_t dgpu_temp_c = 0;
};

class TelemetryLogger {
public:
    TelemetryLogger() = default;
    ~TelemetryLogger();

    void init(const DX12Context* ctx);
    void set_sys_ram_latent_ptr(const std::atomic<double>* ptr) { sys_ram_latent_ptr_ = ptr; }
    void start_run(int run_id, const std::string& config_name, size_t expected_frames);
    void record_frame(const FrameMetricRow& row);
    void set_frame_gpu_ms(int frame_idx, double gpu_ms);
    void stop_run();

    void write_or_append_csv(const std::string& frame_csv_path, const std::string& mem_power_csv_path) const;
    double elapsed_ms() const;

private:
    void sampler_loop();

    const DX12Context* ctx_ = nullptr;
    int run_id_ = 0;
    std::string config_name_;
    uint64_t qpc_freq_ = 1;
    uint64_t run_start_qpc_ = 0;

    std::vector<FrameMetricRow> frame_rows_;
    std::vector<MemoryPowerRow> mem_rows_;
    std::atomic<bool> sampling_{false};
    std::thread sampler_thread_;

    // Dynamically loaded NVML & PDH state
    HMODULE nvml_mod_ = nullptr;
    void* nvml_dev_ = nullptr;
    bool nvml_ok_ = false;
    const std::atomic<double>* sys_ram_latent_ptr_ = nullptr;
    void* pdh_query_ = nullptr;
    void* rapl_pkg_counter_ = nullptr;
    bool pdh_ok_ = false;
};

} // namespace pahnts
