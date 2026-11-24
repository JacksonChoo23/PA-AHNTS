#include <initguid.h>
#include "dx12_context.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
__declspec(dllexport) extern const UINT D3D12SDKVersion = 620;
__declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\";
}

namespace pahnts {

namespace {

LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t len) {
    crc = ~crc;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int b = 0; b < 8; ++b) {
            crc = (crc >> 1) ^ (0xEDB88320U & static_cast<uint32_t>(-static_cast<int32_t>(crc & 1U)));
        }
    }
    return ~crc;
}

void write_be32(std::vector<uint8_t>& dst, uint32_t v) {
    dst.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
    dst.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    dst.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    dst.push_back(static_cast<uint8_t>(v & 0xFF));
}

void append_png_chunk(std::vector<uint8_t>& out, const char tag[4], const uint8_t* data, size_t len) {
    write_be32(out, static_cast<uint32_t>(len));
    size_t tag_pos = out.size();
    out.insert(out.end(), tag, tag + 4);
    if (len > 0) out.insert(out.end(), data, data + len);
    uint32_t crc = crc32_update(0, out.data() + tag_pos, 4 + len);
    write_be32(out, crc);
}

void write_uncompressed_png(const std::string& path, uint32_t w, uint32_t h, const uint8_t* rgba, size_t row_pitch) {
    std::vector<uint8_t> raw;
    raw.reserve(static_cast<size_t>(h) * (1 + static_cast<size_t>(w) * 4));
    for (uint32_t y = 0; y < h; ++y) {
        raw.push_back(0); // filter type 0
        const uint8_t* row = rgba + static_cast<size_t>(y) * row_pitch;
        raw.insert(raw.end(), row, row + static_cast<size_t>(w) * 4);
    }

    // Wrap raw scanlines in zlib stored (uncompressed) deflate blocks
    std::vector<uint8_t> zlib;
    zlib.reserve(raw.size() + (raw.size() / 65535 + 2) * 5 + 6);
    zlib.push_back(0x78);
    zlib.push_back(0x01);

    uint32_t s1 = 1, s2 = 0;
    for (uint8_t b : raw) {
        s1 = (s1 + b) % 65521U;
        s2 = (s2 + s1) % 65521U;
    }

    size_t offset = 0;
    while (offset < raw.size()) {
        size_t chunk = std::min<size_t>(65535, raw.size() - offset);
        uint8_t final_flag = (offset + chunk == raw.size()) ? 1 : 0;
        zlib.push_back(final_flag);
        uint16_t len16 = static_cast<uint16_t>(chunk);
        uint16_t nlen16 = static_cast<uint16_t>(~len16);
        zlib.push_back(static_cast<uint8_t>(len16 & 0xFF));
        zlib.push_back(static_cast<uint8_t>((len16 >> 8) & 0xFF));
        zlib.push_back(static_cast<uint8_t>(nlen16 & 0xFF));
        zlib.push_back(static_cast<uint8_t>((nlen16 >> 8) & 0xFF));
        zlib.insert(zlib.end(), raw.data() + offset, raw.data() + offset + chunk);
        offset += chunk;
    }
    write_be32(zlib, (s2 << 16) | s1);

    std::vector<uint8_t> png = {137, 80, 78, 71, 13, 10, 26, 10};
    std::vector<uint8_t> ihdr;
    write_be32(ihdr, w);
    write_be32(ihdr, h);
    ihdr.push_back(8); // bit depth
    ihdr.push_back(6); // RGBA
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    append_png_chunk(png, "IHDR", ihdr.data(), ihdr.size());
    append_png_chunk(png, "IDAT", zlib.data(), zlib.size());
    append_png_chunk(png, "IEND", nullptr, 0);

    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) throw std::runtime_error("Failed to open PNG output: " + path);
    std::fwrite(png.data(), 1, png.size(), fp);
    std::fclose(fp);
}

} // namespace

uint64_t QueueSync::signal() {
    uint64_t val = next_fence_value++;
    throw_if_failed(queue->Signal(fence.get(), val), "Queue::Signal failed");
    return val;
}

void QueueSync::wait(uint64_t value) {
    if (fence->GetCompletedValue() < value) {
        throw_if_failed(fence->SetEventOnCompletion(value, event_handle), "SetEventOnCompletion failed");
        WaitForSingleObject(event_handle, INFINITE);
    }
}

void QueueSync::flush() {
    uint64_t val = signal();
    wait(val);
}

void DescriptorHeapAllocator::init(
    ID3D12Device* dev, D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t capacity, bool shader_visible
) {
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = type;
    desc.NumDescriptors = capacity;
    desc.Flags = shader_visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    throw_if_failed(dev->CreateDescriptorHeap(&desc, IID_ID3D12DescriptorHeap, heap_.put_void()),
                    "CreateDescriptorHeap failed");
    cpu_start_ = heap_->GetCPUDescriptorHandleForHeapStart();
    if (shader_visible) {
        gpu_start_ = heap_->GetGPUDescriptorHandleForHeapStart();
    }
    inc_ = dev->GetDescriptorHandleIncrementSize(type);
    cap_ = capacity;
    used_ = 0;
    shader_visible_ = shader_visible;
}

uint32_t DescriptorHeapAllocator::alloc(uint32_t count) {
    if (used_ + count > cap_) {
        throw std::runtime_error("DescriptorHeapAllocator capacity exceeded");
    }
    uint32_t idx = used_;
    used_ += count;
    return idx;
}

D3D12_CPU_DESCRIPTOR_HANDLE DescriptorHeapAllocator::cpu(uint32_t idx) const {
    D3D12_CPU_DESCRIPTOR_HANDLE h = cpu_start_;
    h.ptr += static_cast<SIZE_T>(idx) * inc_;
    return h;
}

D3D12_GPU_DESCRIPTOR_HANDLE DescriptorHeapAllocator::gpu(uint32_t idx) const {
    D3D12_GPU_DESCRIPTOR_HANDLE h = gpu_start_;
    h.ptr += static_cast<UINT64>(idx) * inc_;
    return h;
}

DX12Context::~DX12Context() {
    if (device_) {
        wait_idle();
    }
    if (direct_.event_handle) CloseHandle(direct_.event_handle);
    if (compute_.event_handle) CloseHandle(compute_.event_handle);
    if (copy_.event_handle) CloseHandle(copy_.event_handle);
    if (hwnd_) DestroyWindow(hwnd_);
}

void DX12Context::init(uint32_t width, uint32_t height, const std::string& preferred_vendor_hex, bool enable_debug) {
    width_ = width;
    height_ = height;

    if (enable_debug) {
        ComPtr<ID3D12Debug> dbg;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_ID3D12Debug, dbg.put_void()))) {
            dbg->EnableDebugLayer();
        }
    }

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"PA_AHNTS_DX12_CLASS";
    RegisterClassExW(&wc);

    RECT rc{0, 0, static_cast<LONG>(width_), static_cast<LONG>(height_)};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    hwnd_ = CreateWindowExW(
        0, wc.lpszClassName, L"PA-AHNTS DX12 Streaming Testbed",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, wc.hInstance, nullptr);
    if (!hwnd_) throw std::runtime_error("CreateWindowExW failed");
    ShowWindow(hwnd_, SW_SHOWNOACTIVATE);

    throw_if_failed(CreateDXGIFactory2(0, IID_IDXGIFactory4, factory_.put_void()), "CreateDXGIFactory2 failed");

    ComPtr<IDXGIFactory5> factory5;
    if (SUCCEEDED(factory_->QueryInterface(IID_IDXGIFactory5, factory5.put_void()))) {
        BOOL allow = FALSE;
        if (SUCCEEDED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allow, sizeof(allow)))) {
            tearing_supported_ = (allow == TRUE);
        }
    }

    uint32_t preferred_vendor = 0x10DE;
    if (!preferred_vendor_hex.empty()) {
        preferred_vendor = static_cast<uint32_t>(std::stoul(preferred_vendor_hex, nullptr, 16));
    }

    ComPtr<IDXGIAdapter1> best_adapter;
    SIZE_T best_vram = 0;
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> cand;
        if (factory_->EnumAdapters1(i, cand.put()) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 desc{};
        cand->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (desc.VendorId == preferred_vendor) {
            best_adapter = std::move(cand);
            break;
        }
        if (desc.DedicatedVideoMemory > best_vram) {
            best_vram = desc.DedicatedVideoMemory;
            best_adapter = std::move(cand);
        }
    }
    if (!best_adapter) throw std::runtime_error("No hardware DXGI adapter found");
    throw_if_failed(best_adapter->QueryInterface(IID_IDXGIAdapter3, adapter3_.put_void()), "QI IDXGIAdapter3 failed");

    throw_if_failed(
        D3D12CreateDevice(best_adapter.get(), D3D_FEATURE_LEVEL_12_1, IID_ID3D12Device8, device_.put_void()),
        "D3D12CreateDevice (ID3D12Device8) failed"
    );

    D3D12_FEATURE_DATA_D3D12_OPTIONS7 opts7{};
    if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &opts7, sizeof(opts7)))) {
        sampler_feedback_tier_ = static_cast<int>(opts7.SamplerFeedbackTier);
    }

    auto make_queue = [&](D3D12_COMMAND_LIST_TYPE type, QueueSync& qs) {
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = type;
        throw_if_failed(device_->CreateCommandQueue(&qd, IID_ID3D12CommandQueue, qs.queue.put_void()),
                        "CreateCommandQueue failed");
        throw_if_failed(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_ID3D12Fence, qs.fence.put_void()),
                        "CreateFence failed");
        qs.event_handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        qs.next_fence_value = 1;
        qs.queue->GetTimestampFrequency(&qs.timestamp_freq_hz);
    };
    make_queue(D3D12_COMMAND_LIST_TYPE_DIRECT, direct_);
    make_queue(D3D12_COMMAND_LIST_TYPE_COMPUTE, compute_);
    make_queue(D3D12_COMMAND_LIST_TYPE_COPY, copy_);

    DXGI_SWAP_CHAIN_DESC1 sc_desc{};
    sc_desc.Width = width_;
    sc_desc.Height = height_;
    sc_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sc_desc.SampleDesc.Count = 1;
    sc_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sc_desc.BufferCount = kFrameInFlightCount;
    sc_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sc_desc.Flags = tearing_supported_ ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

    ComPtr<IDXGISwapChain1> sc1;
    throw_if_failed(
        factory_->CreateSwapChainForHwnd(direct_.queue.get(), hwnd_, &sc_desc, nullptr, nullptr, sc1.put()),
        "CreateSwapChainForHwnd failed"
    );
    factory_->MakeWindowAssociation(hwnd_, DXGI_MWA_NO_ALT_ENTER);
    throw_if_failed(sc1->QueryInterface(IID_IDXGISwapChain3, swapchain_.put_void()), "QI IDXGISwapChain3 failed");

    rtv_heap_.init(device_.get(), D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 8, false);
    dsv_heap_.init(device_.get(), D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 4, false);
    srv_heap_.init(device_.get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4096, true);
    cpu_uav_heap_.init(device_.get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 256, false);

    for (uint32_t i = 0; i < kFrameInFlightCount; ++i) {
        throw_if_failed(swapchain_->GetBuffer(i, IID_ID3D12Resource, backbuffers_[i].put_void()),
                        "SwapChain GetBuffer failed");
        uint32_t slot = rtv_heap_.alloc(1);
        device_->CreateRenderTargetView(backbuffers_[i].get(), nullptr, rtv_heap_.cpu(slot));

        throw_if_failed(
            device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                            IID_ID3D12CommandAllocator, frame_alloc_[i].put_void()),
            "CreateCommandAllocator failed"
        );
    }

    D3D12_HEAP_PROPERTIES def_heap{};
    def_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC depth_desc{};
    depth_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    depth_desc.Width = width_;
    depth_desc.Height = height_;
    depth_desc.DepthOrArraySize = 1;
    depth_desc.MipLevels = 1;
    depth_desc.Format = DXGI_FORMAT_D32_FLOAT;
    depth_desc.SampleDesc.Count = 1;
    depth_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    D3D12_CLEAR_VALUE dsv_clear{};
    dsv_clear.Format = DXGI_FORMAT_D32_FLOAT;
    dsv_clear.DepthStencil.Depth = 1.0f;
    throw_if_failed(
        device_->CreateCommittedResource(
            &def_heap, D3D12_HEAP_FLAG_NONE, &depth_desc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE, &dsv_clear,
            IID_ID3D12Resource, depth_buffer_.put_void()),
        "Create depth buffer failed"
    );
    uint32_t dsv_slot = dsv_heap_.alloc(1);
    device_->CreateDepthStencilView(depth_buffer_.get(), nullptr, dsv_heap_.cpu(dsv_slot));

    throw_if_failed(
        device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, frame_alloc_[0].get(), nullptr,
                                   IID_ID3D12GraphicsCommandList7, cmd_list_.put_void()),
        "CreateCommandList failed"
    );
    cmd_list_->Close();

    D3D12_QUERY_HEAP_DESC qhd{};
    qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qhd.Count = kFrameInFlightCount * 2;
    throw_if_failed(device_->CreateQueryHeap(&qhd, IID_ID3D12QueryHeap, ts_heap_.put_void()),
                    "CreateQueryHeap failed");

    D3D12_HEAP_PROPERTIES rb_heap{};
    rb_heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rb_desc{};
    rb_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rb_desc.Width = sizeof(uint64_t) * kFrameInFlightCount * 2;
    rb_desc.Height = 1;
    rb_desc.DepthOrArraySize = 1;
    rb_desc.MipLevels = 1;
    rb_desc.SampleDesc.Count = 1;
    rb_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    throw_if_failed(
        device_->CreateCommittedResource(
            &rb_heap, D3D12_HEAP_FLAG_NONE, &rb_desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_ID3D12Resource, ts_readback_.put_void()),
        "Create timestamp readback buffer failed"
    );
}

void DX12Context::dump_debug_messages() const {
    if (!device_) return;
    HRESULT reason = device_->GetDeviceRemovedReason();
    std::fprintf(stderr, "[DX12] GetDeviceRemovedReason = 0x%08X\n", static_cast<unsigned int>(reason));
    ComPtr<ID3D12InfoQueue> iq;
    if (SUCCEEDED(device_->QueryInterface(IID_ID3D12InfoQueue, iq.put_void()))) {
        UINT64 count = iq->GetNumStoredMessages();
        for (UINT64 i = 0; i < count; ++i) {
            SIZE_T len = 0;
            iq->GetMessage(i, nullptr, &len);
            if (len == 0) continue;
            std::vector<uint8_t> buf(len);
            auto* msg = reinterpret_cast<D3D12_MESSAGE*>(buf.data());
            if (SUCCEEDED(iq->GetMessage(i, msg, &len))) {
                std::fprintf(stderr, "[D3D12_DEBUG][%d] %s\n", static_cast<int>(msg->Severity), msg->pDescription);
            }
        }
        iq->ClearStoredMessages();
    }
}

void DX12Context::pump_messages() {
    MSG msg{};
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

void DX12Context::wait_idle() {
    direct_.flush();
    compute_.flush();
    copy_.flush();
}

uint32_t DX12Context::begin_frame() {
    uint32_t bb = swapchain_->GetCurrentBackBufferIndex();
    direct_.wait(frame_fence_vals_[bb]);
    throw_if_failed(frame_alloc_[bb]->Reset(), "Reset frame allocator failed");
    throw_if_failed(cmd_list_->Reset(frame_alloc_[bb].get(), nullptr), "Reset cmd_list failed");
    cmd_list_->EndQuery(ts_heap_.get(), D3D12_QUERY_TYPE_TIMESTAMP, bb * 2);
    return bb;
}

void DX12Context::end_frame(uint32_t frame_slot, bool allow_tearing) {
    cmd_list_->EndQuery(ts_heap_.get(), D3D12_QUERY_TYPE_TIMESTAMP, frame_slot * 2 + 1);
    cmd_list_->ResolveQueryData(
        ts_heap_.get(), D3D12_QUERY_TYPE_TIMESTAMP,
        frame_slot * 2, 2,
        ts_readback_.get(), frame_slot * 2 * sizeof(uint64_t)
    );
    throw_if_failed(cmd_list_->Close(), "Close cmd_list failed");

    ID3D12CommandList* lists[] = {cmd_list_.get()};
    direct_.queue->ExecuteCommandLists(1, lists);

    auto t0 = std::chrono::steady_clock::now();
    UINT flags = (allow_tearing && tearing_supported_) ? DXGI_PRESENT_ALLOW_TEARING : 0;
    throw_if_failed(swapchain_->Present(0, flags), "SwapChain::Present failed");
    auto t1 = std::chrono::steady_clock::now();
    last_present_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();

    frame_fence_vals_[frame_slot] = direct_.signal();
    ts_valid_[frame_slot] = true;
}

bool DX12Context::try_read_gpu_frame_ms(uint32_t frame_slot, double& out_ms) {
    if (!ts_valid_[frame_slot]) return false;
    direct_.wait(frame_fence_vals_[frame_slot]);
    uint64_t* ptr = nullptr;
    D3D12_RANGE range{frame_slot * 2 * sizeof(uint64_t), (frame_slot * 2 + 2) * sizeof(uint64_t)};
    if (FAILED(ts_readback_->Map(0, &range, reinterpret_cast<void**>(&ptr)))) return false;
    uint64_t t0 = ptr[frame_slot * 2];
    uint64_t t1 = ptr[frame_slot * 2 + 1];
    D3D12_RANGE empty{0, 0};
    ts_readback_->Unmap(0, &empty);
    if (t1 < t0 || direct_.timestamp_freq_hz == 0) return false;
    out_ms = static_cast<double>(t1 - t0) * 1000.0 / static_cast<double>(direct_.timestamp_freq_hz);
    return true;
}

void DX12Context::capture_backbuffer_png(uint32_t backbuffer_idx, const std::string& png_path) {
    wait_idle();

    uint64_t row_pitch = ((static_cast<uint64_t>(width_) * 4ULL) + 255ULL) & ~255ULL;
    uint64_t total_bytes = row_pitch * height_;

    D3D12_HEAP_PROPERTIES rb_heap{};
    rb_heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rb_desc{};
    rb_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rb_desc.Width = total_bytes;
    rb_desc.Height = 1;
    rb_desc.DepthOrArraySize = 1;
    rb_desc.MipLevels = 1;
    rb_desc.SampleDesc.Count = 1;
    rb_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ComPtr<ID3D12Resource> rb;
    throw_if_failed(
        device_->CreateCommittedResource(
            &rb_heap, D3D12_HEAP_FLAG_NONE, &rb_desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_ID3D12Resource, rb.put_void()),
        "Create PNG readback buffer failed"
    );

    throw_if_failed(frame_alloc_[backbuffer_idx]->Reset(), "Reset alloc for capture failed");
    throw_if_failed(cmd_list_->Reset(frame_alloc_[backbuffer_idx].get(), nullptr), "Reset cmd for capture failed");

    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = backbuffers_[backbuffer_idx].get();
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd_list_->ResourceBarrier(1, &b);

    D3D12_TEXTURE_COPY_LOCATION src_loc{};
    src_loc.pResource = backbuffers_[backbuffer_idx].get();
    src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src_loc.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION dst_loc{};
    dst_loc.pResource = rb.get();
    dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst_loc.PlacedFootprint.Offset = 0;
    dst_loc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    dst_loc.PlacedFootprint.Footprint.Width = width_;
    dst_loc.PlacedFootprint.Footprint.Height = height_;
    dst_loc.PlacedFootprint.Footprint.Depth = 1;
    dst_loc.PlacedFootprint.Footprint.RowPitch = static_cast<UINT>(row_pitch);

    cmd_list_->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);

    std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
    cmd_list_->ResourceBarrier(1, &b);
    throw_if_failed(cmd_list_->Close(), "Close capture cmd_list failed");

    ID3D12CommandList* lists[] = {cmd_list_.get()};
    direct_.queue->ExecuteCommandLists(1, lists);
    direct_.flush();

    uint8_t* mapped = nullptr;
    D3D12_RANGE range{0, static_cast<SIZE_T>(total_bytes)};
    throw_if_failed(rb->Map(0, &range, reinterpret_cast<void**>(&mapped)), "Map capture readback failed");
    write_uncompressed_png(png_path, width_, height_, mapped, static_cast<size_t>(row_pitch));
    D3D12_RANGE empty{0, 0};
    rb->Unmap(0, &empty);
}

void DX12Context::query_vram_usage_mb(double& local_mb, double& nonlocal_mb) const {
    DXGI_QUERY_VIDEO_MEMORY_INFO local_info{};
    DXGI_QUERY_VIDEO_MEMORY_INFO nonlocal_info{};
    if (adapter3_) {
        adapter3_->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local_info);
        adapter3_->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &nonlocal_info);
    }
    local_mb = static_cast<double>(local_info.CurrentUsage) / (1024.0 * 1024.0);
    nonlocal_mb = static_cast<double>(nonlocal_info.CurrentUsage) / (1024.0 * 1024.0);
}

} // namespace pahnts
