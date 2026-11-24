#pragma once

#include <windows.h>
#include <dxgi1_6.h>
#include <d3d12.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace pahnts {

constexpr uint32_t kFrameInFlightCount = 3;
constexpr DXGI_FORMAT kFormatSamplerFeedbackMinMipOpaque = static_cast<DXGI_FORMAT>(189);

template <typename T>
class ComPtr {
public:
    ComPtr() = default;
    ~ComPtr() { reset(); }
    ComPtr(const ComPtr&) = delete;
    ComPtr& operator=(const ComPtr&) = delete;
    ComPtr(ComPtr&& o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }
    ComPtr& operator=(ComPtr&& o) noexcept {
        if (this != &o) {
            reset();
            ptr_ = o.ptr_;
            o.ptr_ = nullptr;
        }
        return *this;
    }
    void reset() {
        if (ptr_) {
            ptr_->Release();
            ptr_ = nullptr;
        }
    }
    T* get() const { return ptr_; }
    T** put() {
        reset();
        return &ptr_;
    }
    void** put_void() {
        reset();
        return reinterpret_cast<void**>(&ptr_);
    }
    T* operator->() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }

private:
    T* ptr_ = nullptr;
};

inline void throw_if_failed(HRESULT hr, const char* msg) {
    if (FAILED(hr)) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s (HRESULT=0x%08X)", msg, static_cast<unsigned int>(hr));
        throw std::runtime_error(buf);
    }
}

struct QueueSync {
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    HANDLE event_handle = nullptr;
    uint64_t next_fence_value = 1;
    uint64_t timestamp_freq_hz = 1000000000ULL;

    uint64_t signal();
    void wait(uint64_t value);
    void flush();
};

class DescriptorHeapAllocator {
public:
    void init(ID3D12Device* dev, D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t capacity, bool shader_visible);
    uint32_t alloc(uint32_t count = 1);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu(uint32_t idx) const;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu(uint32_t idx) const;
    ID3D12DescriptorHeap* heap() const { return heap_.get(); }

private:
    ComPtr<ID3D12DescriptorHeap> heap_;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu_start_{};
    D3D12_GPU_DESCRIPTOR_HANDLE gpu_start_{};
    uint32_t inc_ = 0;
    uint32_t cap_ = 0;
    uint32_t used_ = 0;
    bool shader_visible_ = false;
};

class DX12Context {
public:
    DX12Context() = default;
    ~DX12Context();

    void init(uint32_t width, uint32_t height, const std::string& preferred_vendor_hex, bool enable_debug = false);
    void dump_debug_messages() const;
    void pump_messages();
    void wait_idle();

    uint32_t begin_frame();
    void end_frame(uint32_t frame_slot, bool allow_tearing = true);
    bool try_read_gpu_frame_ms(uint32_t frame_slot, double& out_ms);
    void capture_backbuffer_png(uint32_t backbuffer_idx, const std::string& png_path);

    ID3D12Device8* device() const { return device_.get(); }
    IDXGIAdapter3* adapter3() const { return adapter3_.get(); }
    ID3D12CommandQueue* direct_queue() const { return direct_.queue.get(); }
    ID3D12CommandQueue* compute_queue() const { return compute_.queue.get(); }
    ID3D12CommandQueue* copy_queue() const { return copy_.queue.get(); }
    QueueSync& direct_sync() { return direct_; }
    QueueSync& compute_sync() { return compute_; }
    QueueSync& copy_sync() { return copy_; }

    ID3D12GraphicsCommandList7* cmd_list() const { return cmd_list_.get(); }
    ID3D12Resource* current_rtv_resource(uint32_t bb_idx) const { return backbuffers_[bb_idx].get(); }
    D3D12_CPU_DESCRIPTOR_HANDLE rtv_cpu(uint32_t bb_idx) const { return rtv_heap_.cpu(bb_idx); }
    D3D12_CPU_DESCRIPTOR_HANDLE dsv_cpu() const { return dsv_heap_.cpu(0); }

    DescriptorHeapAllocator& cbv_srv_uav_heap() { return srv_heap_; }
    DescriptorHeapAllocator& cpu_uav_heap() { return cpu_uav_heap_; }

    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    double last_present_ms() const { return last_present_ms_; }
    bool sampler_feedback_supported() const { return sampler_feedback_tier_ > 0; }
    int sampler_feedback_tier() const { return sampler_feedback_tier_; }

    void query_vram_usage_mb(double& local_mb, double& nonlocal_mb) const;

private:
    HWND hwnd_ = nullptr;
    uint32_t width_ = 1920;
    uint32_t height_ = 1080;
    bool tearing_supported_ = false;
    int sampler_feedback_tier_ = 0;

    ComPtr<IDXGIFactory4> factory_;
    ComPtr<IDXGIAdapter3> adapter3_;
    ComPtr<ID3D12Device8> device_;

    QueueSync direct_;
    QueueSync compute_;
    QueueSync copy_;

    ComPtr<IDXGISwapChain3> swapchain_;
    ComPtr<ID3D12Resource> backbuffers_[kFrameInFlightCount];
    ComPtr<ID3D12Resource> depth_buffer_;

    DescriptorHeapAllocator rtv_heap_;
    DescriptorHeapAllocator dsv_heap_;
    DescriptorHeapAllocator srv_heap_;
    DescriptorHeapAllocator cpu_uav_heap_;

    ComPtr<ID3D12CommandAllocator> frame_alloc_[kFrameInFlightCount];
    uint64_t frame_fence_vals_[kFrameInFlightCount] = {0, 0, 0};
    ComPtr<ID3D12GraphicsCommandList7> cmd_list_;

    ComPtr<ID3D12QueryHeap> ts_heap_;
    ComPtr<ID3D12Resource> ts_readback_;
    bool ts_valid_[kFrameInFlightCount] = {false, false, false};

    double last_present_ms_ = 0.0;
};

} // namespace pahnts
