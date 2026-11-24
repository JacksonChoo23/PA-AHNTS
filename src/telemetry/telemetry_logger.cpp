#include "telemetry_logger.h"

#include <pdh.h>
#include <pdhmsg.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>

namespace pahnts {

namespace {

using PFN_nvmlInit_v2 = int (*)();
using PFN_nvmlShutdown = int (*)();
using PFN_nvmlDeviceGetHandleByIndex_v2 = int (*)(unsigned int, void**);
using PFN_nvmlDeviceGetPowerUsage = int (*)(void*, unsigned int*);
using PFN_nvmlDeviceGetTemperature = int (*)(void*, int, unsigned int*);
using PFN_nvmlDeviceGetClockInfo = int (*)(void*, int, unsigned int*);

PFN_nvmlInit_v2 g_nvmlInit = nullptr;
PFN_nvmlShutdown g_nvmlShutdown = nullptr;
PFN_nvmlDeviceGetHandleByIndex_v2 g_nvmlGetHandle = nullptr;
PFN_nvmlDeviceGetPowerUsage g_nvmlGetPower = nullptr;
PFN_nvmlDeviceGetTemperature g_nvmlGetTemp = nullptr;
PFN_nvmlDeviceGetClockInfo g_nvmlGetClock = nullptr;

uint64_t query_qpc() {
    LARGE_INTEGER li{};
    QueryPerformanceCounter(&li);
    return static_cast<uint64_t>(li.QuadPart);
}

} // namespace

TelemetryLogger::~TelemetryLogger() {
    stop_run();
    if (pdh_query_) {
        PdhCloseQuery(reinterpret_cast<PDH_HQUERY>(pdh_query_));
        pdh_query_ = nullptr;
        rapl_pkg_counter_ = nullptr;
        pdh_ok_ = false;
    }
    if (nvml_ok_ && g_nvmlShutdown) {
        g_nvmlShutdown();
        nvml_ok_ = false;
    }
    if (nvml_mod_) {
        FreeLibrary(nvml_mod_);
        nvml_mod_ = nullptr;
    }
}

void TelemetryLogger::init(const DX12Context* ctx) {
    ctx_ = ctx;
    LARGE_INTEGER freq{};
    QueryPerformanceFrequency(&freq);
    qpc_freq_ = static_cast<uint64_t>(freq.QuadPart);

    PDH_HQUERY q = nullptr;
    PDH_HCOUNTER c = nullptr;
    if (PdhOpenQueryW(nullptr, 0, &q) == ERROR_SUCCESS) {
        if (PdhAddEnglishCounterW(q, L"\\Energy Meter(RAPL_Package0_PKG)\\Power", 0, &c) == ERROR_SUCCESS) {
            PdhCollectQueryData(q);
            pdh_query_ = q;
            rapl_pkg_counter_ = c;
            pdh_ok_ = true;
        } else {
            PdhCloseQuery(q);
        }
    }

    nvml_mod_ = LoadLibraryW(L"nvml.dll");
    if (nvml_mod_) {
        g_nvmlInit = reinterpret_cast<PFN_nvmlInit_v2>(GetProcAddress(nvml_mod_, "nvmlInit_v2"));
        g_nvmlShutdown = reinterpret_cast<PFN_nvmlShutdown>(GetProcAddress(nvml_mod_, "nvmlShutdown"));
        g_nvmlGetHandle = reinterpret_cast<PFN_nvmlDeviceGetHandleByIndex_v2>(
            GetProcAddress(nvml_mod_, "nvmlDeviceGetHandleByIndex_v2"));
        g_nvmlGetPower = reinterpret_cast<PFN_nvmlDeviceGetPowerUsage>(
            GetProcAddress(nvml_mod_, "nvmlDeviceGetPowerUsage"));
        g_nvmlGetTemp = reinterpret_cast<PFN_nvmlDeviceGetTemperature>(
            GetProcAddress(nvml_mod_, "nvmlDeviceGetTemperature"));
        g_nvmlGetClock = reinterpret_cast<PFN_nvmlDeviceGetClockInfo>(
            GetProcAddress(nvml_mod_, "nvmlDeviceGetClockInfo"));

        if (g_nvmlInit && g_nvmlGetHandle && g_nvmlInit() == 0) {
            if (g_nvmlGetHandle(0, &nvml_dev_) == 0 && nvml_dev_) {
                nvml_ok_ = true;
            }
        }
    }
}

void TelemetryLogger::start_run(int run_id, const std::string& config_name, size_t expected_frames) {
    stop_run();
    run_id_ = run_id;
    config_name_ = config_name;
    frame_rows_.clear();
    frame_rows_.reserve(expected_frames);
    mem_rows_.clear();
    mem_rows_.reserve(256);

    run_start_qpc_ = query_qpc();
    sampling_.store(true, std::memory_order_release);
    sampler_thread_ = std::thread(&TelemetryLogger::sampler_loop, this);
}

void TelemetryLogger::record_frame(const FrameMetricRow& row) {
    frame_rows_.push_back(row);
}

void TelemetryLogger::set_frame_gpu_ms(int frame_idx, double gpu_ms) {
    if (frame_idx >= 0 && static_cast<size_t>(frame_idx) < frame_rows_.size() &&
        frame_rows_[static_cast<size_t>(frame_idx)].frame_idx == frame_idx) {
        frame_rows_[static_cast<size_t>(frame_idx)].gpu_frame_ms = gpu_ms;
    }
}

void TelemetryLogger::stop_run() {
    if (sampling_.exchange(false, std::memory_order_acq_rel)) {
        if (sampler_thread_.joinable()) {
            sampler_thread_.join();
        }
    }
}

double TelemetryLogger::elapsed_ms() const {
    uint64_t now = query_qpc();
    return static_cast<double>(now - run_start_qpc_) * 1000.0 / static_cast<double>(qpc_freq_);
}

void TelemetryLogger::sampler_loop() {
    int sample_idx = 0;
    while (true) {
        MemoryPowerRow row;
        row.run_id = run_id_;
        row.config_name = config_name_;
        row.sample_idx = sample_idx++;
        row.timestamp_ms = elapsed_ms();

        if (ctx_) {
            ctx_->query_vram_usage_mb(row.local_vram_mb, row.nonlocal_vram_mb);
        }

        if (nvml_ok_ && nvml_dev_) {
            unsigned int mw = 0, mhz = 0, temp = 0;
            if (g_nvmlGetPower && g_nvmlGetPower(nvml_dev_, &mw) == 0) {
                row.dgpu_power_w = static_cast<double>(mw) / 1000.0;
            }
            if (g_nvmlGetClock && g_nvmlGetClock(nvml_dev_, 0, &mhz) == 0) {
                row.dgpu_clock_mhz = mhz;
            }
            if (g_nvmlGetTemp && g_nvmlGetTemp(nvml_dev_, 0, &temp) == 0) {
                row.dgpu_temp_c = temp;
            }
        }

        if (pdh_ok_ && PdhCollectQueryData(reinterpret_cast<PDH_HQUERY>(pdh_query_)) == ERROR_SUCCESS) {
            PDH_FMT_COUNTERVALUE val{};
            if (PdhGetFormattedCounterValue(reinterpret_cast<PDH_HCOUNTER>(rapl_pkg_counter_),
                                            PDH_FMT_DOUBLE, nullptr, &val) == ERROR_SUCCESS &&
                val.CStatus == 0) {
                row.cpu_pkg_power_w = val.doubleValue / 1000.0;
            }
        }

        row.npu_power_w = 0.0;
        row.sys_ram_latent_mb = sys_ram_latent_ptr_
            ? sys_ram_latent_ptr_->load(std::memory_order_relaxed)
            : 0.0;
        row.total_est_power_w = row.dgpu_power_w + row.cpu_pkg_power_w + row.npu_power_w;
        mem_rows_.push_back(row);

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (!sampling_.load(std::memory_order_acquire)) break;
    }
}

void TelemetryLogger::write_or_append_csv(
    const std::string& frame_csv_path,
    const std::string& mem_power_csv_path
) const {
    namespace fs = std::filesystem;

    auto open_csv = [](const std::string& path, const char* header) -> std::ofstream {
        fs::path p(path);
        if (p.has_parent_path()) fs::create_directories(p.parent_path());
        bool need_header = !fs::exists(p) || fs::file_size(p) == 0;
        std::ofstream ofs(path, std::ios::app);
        if (need_header) ofs << header << "\n";
        return ofs;
    };

    {
        std::ofstream f = open_csv(
            frame_csv_path,
            "run_id,config_name,frame_idx,timestamp_ms,cpu_frame_ms,gpu_frame_ms,present_ms,"
            "pcie_upload_mb,tiles_decoded_dgpu,tiles_decoded_npu,tiles_decoded_igpu,tiles_decoded_cpu"
        );
        f << std::fixed << std::setprecision(4);
        for (const auto& r : frame_rows_) {
            f << r.run_id << "," << r.config_name << "," << r.frame_idx << ","
              << r.timestamp_ms << "," << r.cpu_frame_ms << "," << r.gpu_frame_ms << ","
              << r.present_ms << "," << r.pcie_upload_mb << ","
              << r.tiles_decoded_dgpu << "," << r.tiles_decoded_npu << ","
              << r.tiles_decoded_igpu << "," << r.tiles_decoded_cpu << "\n";
        }
    }

    {
        std::ofstream m = open_csv(
            mem_power_csv_path,
            "run_id,config_name,sample_idx,timestamp_ms,local_vram_mb,nonlocal_vram_mb,"
            "sys_ram_latent_mb,dgpu_power_w,cpu_pkg_power_w,npu_power_w,total_est_power_w,"
            "dgpu_clock_mhz,dgpu_temp_c"
        );
        m << std::fixed << std::setprecision(3);
        for (const auto& r : mem_rows_) {
            m << r.run_id << "," << r.config_name << "," << r.sample_idx << ","
              << r.timestamp_ms << "," << r.local_vram_mb << "," << r.nonlocal_vram_mb << ","
              << r.sys_ram_latent_mb << "," << r.dgpu_power_w << "," << r.cpu_pkg_power_w << ","
              << r.npu_power_w << "," << r.total_est_power_w << ","
              << r.dgpu_clock_mhz << "," << r.dgpu_temp_c << "\n";
        }
    }
}

} // namespace pahnts
