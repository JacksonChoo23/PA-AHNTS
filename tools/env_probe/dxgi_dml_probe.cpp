#include <initguid.h>
#include <windows.h>
#include <dxgi1_6.h>
#include <d3d12.h>

#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// {6dbd6437-96fd-423f-a98c-ae5e7c2a573f}
static const GUID kIID_IDMLDevice = {
    0x6dbd6437, 0x96fd, 0x423f, {0xa9, 0x8c, 0xae, 0x5e, 0x7c, 0x2a, 0x57, 0x3f}
};

using PFN_DMLCreateDevice = HRESULT(WINAPI*)(
    ID3D12Device* d3d12Device,
    UINT flags,
    REFIID riid,
    void** ppv
);

template <typename T>
struct ComPtr {
    T* ptr = nullptr;
    ComPtr() = default;
    ~ComPtr() { reset(); }
    ComPtr(const ComPtr&) = delete;
    ComPtr& operator=(const ComPtr&) = delete;
    ComPtr(ComPtr&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    ComPtr& operator=(ComPtr&& other) noexcept {
        if (this != &other) {
            reset();
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
    void reset() {
        if (ptr) {
            ptr->Release();
            ptr = nullptr;
        }
    }
    T** put() {
        reset();
        return &ptr;
    }
    void** put_void() {
        reset();
        return reinterpret_cast<void**>(&ptr);
    }
    T* get() const { return ptr; }
    T* operator->() const { return ptr; }
    explicit operator bool() const { return ptr != nullptr; }
};

static std::string wide_to_utf8(const wchar_t* wstr) {
    if (!wstr || !wstr[0]) return "";
    int needed = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, nullptr, 0, nullptr, nullptr);
    if (needed <= 1) return "";
    std::string out(static_cast<size_t>(needed - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, &out[0], needed, nullptr, nullptr);
    return out;
}

static std::string json_escape(const std::string& s) {
    std::ostringstream oss;
    for (unsigned char c : s) {
        switch (c) {
            case '\"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (c < 0x20) {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned int>(c) << std::dec;
                } else {
                    oss << static_cast<char>(c);
                }
        }
    }
    return oss.str();
}

static std::string hex_u32(uint32_t v) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::uppercase << v;
    return oss.str();
}

static std::string hex_hr(HRESULT hr) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::uppercase << std::setw(8) << std::setfill('0')
        << static_cast<uint32_t>(hr);
    return oss.str();
}

static bool check_dll(const wchar_t* name) {
    HMODULE mod = LoadLibraryW(name);
    if (!mod) return false;
    FreeLibrary(mod);
    return true;
}

static const char* feature_level_str(D3D_FEATURE_LEVEL fl) {
    switch (static_cast<uint32_t>(fl)) {
        case 0xc200: return "12_2";
        case 0xc100: return "12_1";
        case 0xc000: return "12_0";
        case 0xb100: return "11_1";
        case 0xb000: return "11_0";
        default: return "unknown";
    }
}

static std::string shader_model_str(uint32_t sm) {
    uint32_t major = (sm >> 4) & 0xF;
    uint32_t minor = sm & 0xF;
    return std::to_string(major) + "." + std::to_string(minor);
}

static void print_mem_info(std::ostream& os, const DXGI_QUERY_VIDEO_MEMORY_INFO& info, bool ok) {
    os << "{";
    os << "\"queried\": " << (ok ? "true" : "false") << ", ";
    os << "\"budget_bytes\": " << info.Budget << ", ";
    os << "\"current_usage_bytes\": " << info.CurrentUsage << ", ";
    os << "\"available_for_reservation_bytes\": " << info.AvailableForReservation << ", ";
    os << "\"current_reservation_bytes\": " << info.CurrentReservation << ", ";
    os << std::fixed << std::setprecision(2);
    os << "\"budget_mib\": " << (static_cast<double>(info.Budget) / (1024.0 * 1024.0));
    os << "}";
}

int main() {
    HRESULT co_hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);

    bool has_d3d12 = check_dll(L"d3d12.dll");
    bool has_dxgi = check_dll(L"dxgi.dll");
    bool has_dml = check_dll(L"DirectML.dll");
    bool has_dxcore = check_dll(L"dxcore.dll");
    bool has_dxcompiler = check_dll(L"dxcompiler.dll");
    bool has_d3dcompiler = check_dll(L"d3dcompiler_47.dll");
    bool has_nvml = check_dll(L"nvml.dll");

    HMODULE dml_module = LoadLibraryW(L"DirectML.dll");
    PFN_DMLCreateDevice pfn_dml_create = nullptr;
    if (dml_module) {
        pfn_dml_create = reinterpret_cast<PFN_DMLCreateDevice>(
            GetProcAddress(dml_module, "DMLCreateDevice"));
    }

    ComPtr<IDXGIFactory4> factory4;
    HRESULT factory_hr = CreateDXGIFactory2(0, IID_IDXGIFactory4, factory4.put_void());
    if (FAILED(factory_hr)) {
        std::cout << "{\n"
                  << "  \"error\": \"CreateDXGIFactory2 failed\",\n"
                  << "  \"hresult\": \"" << hex_hr(factory_hr) << "\"\n"
                  << "}\n";
        if (dml_module) FreeLibrary(dml_module);
        if (SUCCEEDED(co_hr)) CoUninitialize();
        return 1;
    }

    std::cout << "{\n";
    std::cout << "  \"dll_availability\": {\n";
    std::cout << "    \"d3d12\": " << (has_d3d12 ? "true" : "false") << ",\n";
    std::cout << "    \"dxgi\": " << (has_dxgi ? "true" : "false") << ",\n";
    std::cout << "    \"directml\": " << (has_dml ? "true" : "false") << ",\n";
    std::cout << "    \"dml_create_device_symbol\": " << (pfn_dml_create ? "true" : "false") << ",\n";
    std::cout << "    \"dxcore\": " << (has_dxcore ? "true" : "false") << ",\n";
    std::cout << "    \"dxcompiler\": " << (has_dxcompiler ? "true" : "false") << ",\n";
    std::cout << "    \"d3dcompiler_47\": " << (has_d3dcompiler ? "true" : "false") << ",\n";
    std::cout << "    \"nvml\": " << (has_nvml ? "true" : "false") << "\n";
    std::cout << "  },\n";
    std::cout << "  \"adapters\": [\n";

    std::vector<ComPtr<IDXGIAdapter1>> adapters;
    for (UINT idx = 0;; ++idx) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory4->EnumAdapters1(idx, adapter.put()) == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        adapters.push_back(std::move(adapter));
    }

    for (size_t i = 0; i < adapters.size(); ++i) {
        IDXGIAdapter1* adapter = adapters[i].get();
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);

        std::string name = wide_to_utf8(desc.Description);
        bool is_sw = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;

        DXGI_QUERY_VIDEO_MEMORY_INFO local_info{};
        DXGI_QUERY_VIDEO_MEMORY_INFO nonlocal_info{};
        bool local_ok = false;
        bool nonlocal_ok = false;

        ComPtr<IDXGIAdapter3> adapter3;
        if (SUCCEEDED(adapter->QueryInterface(IID_IDXGIAdapter3, adapter3.put_void()))) {
            local_ok = SUCCEEDED(adapter3->QueryVideoMemoryInfo(
                0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local_info));
            nonlocal_ok = SUCCEEDED(adapter3->QueryVideoMemoryInfo(
                0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &nonlocal_info));
        }

        const D3D_FEATURE_LEVEL candidate_levels[] = {
            static_cast<D3D_FEATURE_LEVEL>(0xc200), // 12_2
            D3D_FEATURE_LEVEL_12_1,
            D3D_FEATURE_LEVEL_12_0,
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0
        };

        ComPtr<ID3D12Device> device;
        D3D_FEATURE_LEVEL max_fl = static_cast<D3D_FEATURE_LEVEL>(0);
        HRESULT d3d_hr = E_FAIL;
        for (D3D_FEATURE_LEVEL fl : candidate_levels) {
            d3d_hr = D3D12CreateDevice(adapter, fl, IID_ID3D12Device, device.put_void());
            if (SUCCEEDED(d3d_hr)) {
                max_fl = fl;
                break;
            }
        }

        bool d3d_ok = static_cast<bool>(device);
        D3D12_FEATURE_DATA_D3D12_OPTIONS opts0{};
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 opts1{};
        D3D12_FEATURE_DATA_D3D12_OPTIONS3 opts3{};
        D3D12_FEATURE_DATA_D3D12_OPTIONS4 opts4{};
        D3D12_FEATURE_DATA_D3D12_OPTIONS7 opts7{};
        D3D12_FEATURE_DATA_ARCHITECTURE1 arch1{};
        uint32_t highest_sm = 0;

        uint64_t direct_ts_freq = 0;
        uint64_t copy_ts_freq = 0;
        bool direct_queue_ok = false;
        bool copy_queue_ok = false;

        bool dml_device_ok = false;
        HRESULT dml_hr = E_NOTIMPL;

        if (d3d_ok) {
            device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &opts0, sizeof(opts0));
            device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &opts1, sizeof(opts1));
            device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &opts3, sizeof(opts3));
            device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &opts4, sizeof(opts4));
            device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &opts7, sizeof(opts7));
            arch1.NodeIndex = 0;
            device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &arch1, sizeof(arch1));

            const uint32_t sm_candidates[] = {0x68, 0x67, 0x66, 0x65, 0x64, 0x63, 0x62, 0x61, 0x60};
            for (uint32_t sm : sm_candidates) {
                D3D12_FEATURE_DATA_SHADER_MODEL sm_data{};
                sm_data.HighestShaderModel = static_cast<D3D_SHADER_MODEL>(sm);
                if (SUCCEEDED(device->CheckFeatureSupport(
                        D3D12_FEATURE_SHADER_MODEL, &sm_data, sizeof(sm_data)))) {
                    highest_sm = static_cast<uint32_t>(sm_data.HighestShaderModel);
                    break;
                }
            }

            D3D12_COMMAND_QUEUE_DESC qd{};
            qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
            ComPtr<ID3D12CommandQueue> dq;
            if (SUCCEEDED(device->CreateCommandQueue(&qd, IID_ID3D12CommandQueue, dq.put_void()))) {
                direct_queue_ok = SUCCEEDED(dq->GetTimestampFrequency(&direct_ts_freq));
            }

            D3D12_COMMAND_QUEUE_DESC qc{};
            qc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
            ComPtr<ID3D12CommandQueue> cq;
            if (SUCCEEDED(device->CreateCommandQueue(&qc, IID_ID3D12CommandQueue, cq.put_void()))) {
                copy_queue_ok = SUCCEEDED(cq->GetTimestampFrequency(&copy_ts_freq));
            }

            if (pfn_dml_create) {
                ComPtr<IUnknown> dml_dev;
                dml_hr = pfn_dml_create(device.get(), 0, kIID_IDMLDevice, dml_dev.put_void());
                dml_device_ok = SUCCEEDED(dml_hr) && static_cast<bool>(dml_dev);
            }
        }

        std::cout << "    {\n";
        std::cout << "      \"adapter_index\": " << i << ",\n";
        std::cout << "      \"description\": \"" << json_escape(name) << "\",\n";
        std::cout << "      \"vendor_id\": \"" << hex_u32(desc.VendorId) << "\",\n";
        std::cout << "      \"device_id\": \"" << hex_u32(desc.DeviceId) << "\",\n";
        std::cout << "      \"subsys_id\": \"" << hex_u32(desc.SubSysId) << "\",\n";
        std::cout << "      \"revision\": " << desc.Revision << ",\n";
        std::cout << "      \"is_software\": " << (is_sw ? "true" : "false") << ",\n";
        std::cout << "      \"dedicated_video_memory_bytes\": " << desc.DedicatedVideoMemory << ",\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "      \"dedicated_video_memory_mib\": "
                  << (static_cast<double>(desc.DedicatedVideoMemory) / (1024.0 * 1024.0)) << ",\n";
        std::cout << "      \"dedicated_system_memory_bytes\": " << desc.DedicatedSystemMemory << ",\n";
        std::cout << "      \"shared_system_memory_bytes\": " << desc.SharedSystemMemory << ",\n";
        std::cout << "      \"shared_system_memory_mib\": "
                  << (static_cast<double>(desc.SharedSystemMemory) / (1024.0 * 1024.0)) << ",\n";
        std::cout << "      \"luid\": \"" << hex_u32(static_cast<uint32_t>(desc.AdapterLuid.HighPart))
                  << ":" << hex_u32(desc.AdapterLuid.LowPart) << "\",\n";
        std::cout << "      \"dxgi_local_memory\": ";
        print_mem_info(std::cout, local_info, local_ok);
        std::cout << ",\n";
        std::cout << "      \"dxgi_nonlocal_memory\": ";
        print_mem_info(std::cout, nonlocal_info, nonlocal_ok);
        std::cout << ",\n";

        std::cout << "      \"d3d12\": {\n";
        std::cout << "        \"supported\": " << (d3d_ok ? "true" : "false") << ",\n";
        std::cout << "        \"hresult\": \"" << hex_hr(d3d_hr) << "\",\n";
        std::cout << "        \"max_feature_level\": \"" << (d3d_ok ? feature_level_str(max_fl) : "none") << "\",\n";
        std::cout << "        \"highest_shader_model\": \"" << (highest_sm ? shader_model_str(highest_sm) : "unknown") << "\",\n";
        std::cout << "        \"tiled_resources_tier\": " << static_cast<int>(opts0.TiledResourcesTier) << ",\n";
        std::cout << "        \"resource_binding_tier\": " << static_cast<int>(opts0.ResourceBindingTier) << ",\n";
        std::cout << "        \"resource_heap_tier\": " << static_cast<int>(opts0.ResourceHeapTier) << ",\n";
        std::cout << "        \"wave_ops\": " << (opts1.WaveOps ? "true" : "false") << ",\n";
        std::cout << "        \"wave_lane_count_min\": " << opts1.WaveLaneCountMin << ",\n";
        std::cout << "        \"wave_lane_count_max\": " << opts1.WaveLaneCountMax << ",\n";
        std::cout << "        \"total_lane_count\": " << opts1.TotalLaneCount << ",\n";
        std::cout << "        \"copy_queue_timestamp_supported\": " << (opts3.CopyQueueTimestampQueriesSupported ? "true" : "false") << ",\n";
        std::cout << "        \"native_16bit_shader_ops\": " << (opts4.Native16BitShaderOpsSupported ? "true" : "false") << ",\n";
        std::cout << "        \"mesh_shader_tier\": " << static_cast<int>(opts7.MeshShaderTier) << ",\n";
        std::cout << "        \"sampler_feedback_tier\": " << static_cast<int>(opts7.SamplerFeedbackTier) << ",\n";
        std::cout << "        \"uma\": " << (arch1.UMA ? "true" : "false") << ",\n";
        std::cout << "        \"cache_coherent_uma\": " << (arch1.CacheCoherentUMA ? "true" : "false") << ",\n";
        std::cout << "        \"direct_timestamp_ok\": " << (direct_queue_ok ? "true" : "false") << ",\n";
        std::cout << "        \"direct_timestamp_freq_hz\": " << direct_ts_freq << ",\n";
        std::cout << "        \"copy_timestamp_ok\": " << (copy_queue_ok ? "true" : "false") << ",\n";
        std::cout << "        \"copy_timestamp_freq_hz\": " << copy_ts_freq << "\n";
        std::cout << "      },\n";

        std::cout << "      \"directml\": {\n";
        std::cout << "        \"device_created\": " << (dml_device_ok ? "true" : "false") << ",\n";
        std::cout << "        \"hresult\": \"" << hex_hr(dml_hr) << "\"\n";
        std::cout << "      }\n";
        std::cout << "    }" << (i + 1 < adapters.size() ? "," : "") << "\n";
    }

    std::cout << "  ]\n";
    std::cout << "}\n";

    if (dml_module) FreeLibrary(dml_module);
    if (SUCCEEDED(co_hr)) CoUninitialize();
    return 0;
}
