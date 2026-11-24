#include "baseline_runner.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <list>
#include <unordered_map>
#include <vector>

namespace pahnts {

namespace {

constexpr uint32_t kNumMaterials = 32;
constexpr uint32_t kVirtualTexSize = 8192;
constexpr uint32_t kBcnMipLevels = 11;       // 8192 .. 8
constexpr uint32_t kStreamedMipCount = 6;    // Mips 0..5 (8192..256)
constexpr uint32_t kTileBytes = 65536;       // 64 KB

struct FrameConstants {
    float view_proj[16];
    float cam_pos[3];
    float time_sec;
    uint32_t enable_feedback;
    uint32_t use_tile_residency;
    uint32_t pad0;
    uint32_t pad1;
};

struct TileDecodeJob {
    uint32_t mat_id;
    uint32_t mip_level;
    uint32_t tile_x;
    uint32_t tile_y;
    uint32_t dst_albedo_byte_offset;
    uint32_t dst_normal_byte_offset;
    uint32_t dst_orm_byte_offset;
    uint32_t pad;
};

std::vector<uint8_t> read_binary_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("Failed to open shader CSO: " + path);
    std::streamsize sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> out(static_cast<size_t>(sz));
    if (!ifs.read(reinterpret_cast<char*>(out.data()), sz)) {
        throw std::runtime_error("Failed to read shader CSO: " + path);
    }
    return out;
}

ComPtr<ID3D12Resource> create_default_buffer(
    ID3D12Device* dev, uint64_t bytes, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
    D3D12_RESOURCE_STATES init_state = D3D12_RESOURCE_STATE_COMMON
) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = flags;

    ComPtr<ID3D12Resource> res;
    throw_if_failed(
        dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, init_state, nullptr,
                                     IID_ID3D12Resource, res.put_void()),
        "Create default buffer failed"
    );
    return res;
}

ComPtr<ID3D12Resource> create_upload_buffer(ID3D12Device* dev, uint64_t bytes) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ComPtr<ID3D12Resource> res;
    throw_if_failed(
        dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                     IID_ID3D12Resource, res.put_void()),
        "Create upload buffer failed"
    );
    return res;
}

ComPtr<ID3D12Resource> create_readback_buffer(ID3D12Device* dev, uint64_t bytes) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ComPtr<ID3D12Resource> res;
    throw_if_failed(
        dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                     IID_ID3D12Resource, res.put_void()),
        "Create readback buffer failed"
    );
    return res;
}

ComPtr<ID3D12Resource> create_texture2d(
    ID3D12Device* dev, uint32_t w, uint32_t h, uint16_t mips, DXGI_FORMAT fmt,
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
    D3D12_RESOURCE_STATES init_state = D3D12_RESOURCE_STATE_COPY_DEST
) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = w;
    rd.Height = h;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = mips;
    rd.Format = fmt;
    rd.SampleDesc.Count = 1;
    rd.Flags = flags;

    ComPtr<ID3D12Resource> res;
    throw_if_failed(
        dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, init_state, nullptr,
                                     IID_ID3D12Resource, res.put_void()),
        "Create committed texture2D failed"
    );
    return res;
}

void upload_texture_mips_with_staging(
    DX12Context& ctx,
    ID3D12CommandAllocator* alloc,
    ID3D12Resource* dst_tex,
    ID3D12Resource* upload_staging,
    uint8_t* staging_ptr,
    DXGI_FORMAT fmt,
    uint32_t base_w,
    uint32_t base_h,
    uint16_t start_mip,
    uint16_t end_mip,
    const std::function<void(uint16_t mip, std::vector<uint8_t>& data)>& fill_fn,
    D3D12_RESOURCE_STATES final_state
) {
    ID3D12Device8* dev = ctx.device();
    ID3D12GraphicsCommandList7* cmd = ctx.cmd_list();
    D3D12_RESOURCE_DESC desc = dst_tex->GetDesc();

    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts(desc.MipLevels);
    std::vector<UINT> num_rows(desc.MipLevels);
    std::vector<UINT64> row_sizes(desc.MipLevels);
    UINT64 total_bytes = 0;
    dev->GetCopyableFootprints(&desc, 0, desc.MipLevels, 0, layouts.data(), num_rows.data(), row_sizes.data(), &total_bytes);

    throw_if_failed(alloc->Reset(), "upload alloc reset failed");
    throw_if_failed(cmd->Reset(alloc, nullptr), "upload cmd reset failed");

    uint64_t staging_cursor = 0;
    std::vector<uint8_t> mip_bytes;
    bool is_bc = (fmt == DXGI_FORMAT_BC7_UNORM || fmt == DXGI_FORMAT_BC5_UNORM);

    for (uint16_t m = start_mip; m <= end_mip; ++m) {
        fill_fn(m, mip_bytes);
        uint32_t mw = std::max(1u, base_w >> m);
        uint32_t mh = std::max(1u, base_h >> m);
        uint32_t rows = is_bc ? std::max(1u, mh / 4u) : mh;
        uint32_t src_row_bytes = static_cast<uint32_t>(mip_bytes.size() / rows);

        uint32_t dst_pitch = (src_row_bytes + 255u) & ~255u;
        staging_cursor = (staging_cursor + 511ULL) & ~511ULL;

        for (uint32_t r = 0; r < rows; ++r) {
            std::memcpy(staging_ptr + staging_cursor + static_cast<size_t>(r) * dst_pitch,
                        mip_bytes.data() + static_cast<size_t>(r) * src_row_bytes,
                        src_row_bytes);
        }

        D3D12_TEXTURE_COPY_LOCATION src_loc{};
        src_loc.pResource = upload_staging;
        src_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src_loc.PlacedFootprint.Offset = staging_cursor;
        src_loc.PlacedFootprint.Footprint.Format = fmt;
        src_loc.PlacedFootprint.Footprint.Width = mw;
        src_loc.PlacedFootprint.Footprint.Height = mh;
        src_loc.PlacedFootprint.Footprint.Depth = 1;
        src_loc.PlacedFootprint.Footprint.RowPitch = dst_pitch;

        D3D12_TEXTURE_COPY_LOCATION dst_loc{};
        dst_loc.pResource = dst_tex;
        dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst_loc.SubresourceIndex = m;

        cmd->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);
        staging_cursor += static_cast<uint64_t>(rows) * dst_pitch;
    }

    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = dst_tex;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    b.Transition.StateAfter = final_state;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd->ResourceBarrier(1, &b);

    throw_if_failed(cmd->Close(), "upload cmd close failed");
    ID3D12CommandList* lists[] = {cmd};
    ctx.direct_queue()->ExecuteCommandLists(1, lists);
    ctx.direct_sync().flush();
}

// LRU Tile Cache for 8,192 tile triplets (each triplet = 3x 64KB tiles)
struct TileKey {
    uint32_t mat_id;
    uint32_t mip;
    uint32_t tx;
    uint32_t ty;

    uint32_t pack() const {
        return ((mat_id & 0xFFu) << 16) | ((mip & 0xFu) << 12) | ((ty & 0x3Fu) << 6) | (tx & 0x3Fu);
    }
};

class TileLruCache {
public:
    void init(uint32_t max_triplets) {
        cap_ = max_triplets;
        lru_.clear();
        map_.clear();
        free_slots_.clear();
        free_slots_.reserve(max_triplets);
        for (int i = static_cast<int>(max_triplets) - 1; i >= 0; --i) {
            free_slots_.push_back(static_cast<uint32_t>(i));
        }
        std::memset(resident_mask_, 0, sizeof(resident_mask_));
        std::memset(residency_map_, 6, sizeof(residency_map_)); // fallback to Mip 6
    }

    bool contains(uint32_t packed_key) const {
        return map_.find(packed_key) != map_.end();
    }

    void touch(uint32_t packed_key) {
        auto it = map_.find(packed_key);
        if (it != map_.end()) {
            lru_.splice(lru_.begin(), lru_, it->second);
        }
    }

    struct AllocResult {
        uint32_t slot = 0;
        bool evicted = false;
        uint32_t evicted_key = 0;
    };

    AllocResult acquire_slot(uint32_t new_key) {
        AllocResult r;
        if (!free_slots_.empty()) {
            r.slot = free_slots_.back();
            free_slots_.pop_back();
        } else {
            auto tail = lru_.back();
            r.evicted = true;
            r.evicted_key = tail.key;
            r.slot = tail.slot;
            map_.erase(tail.key);
            lru_.pop_back();
            set_resident_bit(tail.key, false);
        }
        lru_.push_front({new_key, r.slot});
        map_[new_key] = lru_.begin();
        return r;
    }

    void mark_resident(uint32_t packed_key, bool on) {
        set_resident_bit(packed_key, on);
    }

    const uint32_t* residency_map_data() const { return residency_map_; }

private:
    struct Node {
        uint32_t key;
        uint32_t slot;
    };

    void set_resident_bit(uint32_t packed_key, bool on) {
        uint32_t m = (packed_key >> 16) & 0x1Fu;
        uint32_t mip = (packed_key >> 12) & 0xFu;
        uint32_t ty = (packed_key >> 6) & 0x3Fu;
        uint32_t tx = packed_key & 0x3Fu;
        if (mip >= kStreamedMipCount) return;

        // Mark bit in resident_mask_[m][mip][ty*32 + tx]
        uint32_t cell_span = 1u << mip;
        uint32_t x0 = tx * cell_span;
        uint32_t y0 = ty * cell_span;
        uint32_t bit = 1u << mip;
        for (uint32_t y = y0; y < std::min(32u, y0 + cell_span); ++y) {
            for (uint32_t x = x0; x < std::min(32u, x0 + cell_span); ++x) {
                uint32_t idx = m * 1024u + y * 32u + x;
                if (on) resident_mask_[idx] |= bit;
                else    resident_mask_[idx] &= ~bit;

                // Finest contiguous resident mip starting from 6 downward or finest resident bit
                uint32_t mask = resident_mask_[idx];
                uint32_t finest = 6u;
                for (int k = 5; k >= 0; --k) {
                    if (mask & (1u << k)) finest = static_cast<uint32_t>(k);
                    else break;
                }
                residency_map_[idx] = finest;
            }
        }
    }

    uint32_t cap_ = 0;
    std::list<Node> lru_;
    std::unordered_map<uint32_t, std::list<Node>::iterator> map_;
    std::vector<uint32_t> free_slots_;
    uint8_t resident_mask_[kNumMaterials * 1024]{};
    uint32_t residency_map_[kNumMaterials * 1024]{};
};

} // namespace

void BaselineRunner::run(
    DX12Context& ctx,
    const SceneGeometry& scene,
    const MaterialPack& pack,
    TelemetryLogger& logger,
    const BenchmarkConfig& cfg
) {
    ID3D12Device8* dev = ctx.device();
    auto& srv_heap = ctx.cbv_srv_uav_heap();

    // 1. Upload Vertex & Index buffers
    const auto& verts = scene.vertices();
    const auto& inds = scene.indices();
    uint64_t vb_bytes = verts.size() * sizeof(Vertex);
    uint64_t ib_bytes = inds.size() * sizeof(uint32_t);

    auto vb = create_default_buffer(dev, vb_bytes);
    auto ib = create_default_buffer(dev, ib_bytes);
    auto geom_up = create_upload_buffer(dev, vb_bytes + ib_bytes);
    {
        uint8_t* p = nullptr;
        throw_if_failed(geom_up->Map(0, nullptr, reinterpret_cast<void**>(&p)), "Map geom_up failed");
        std::memcpy(p, verts.data(), vb_bytes);
        std::memcpy(p + vb_bytes, inds.data(), ib_bytes);
        geom_up->Unmap(0, nullptr);
    }

    ComPtr<ID3D12CommandAllocator> init_alloc;
    throw_if_failed(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                IID_ID3D12CommandAllocator, init_alloc.put_void()),
                    "Create init_alloc failed");
    {
        auto* cmd = ctx.cmd_list();
        throw_if_failed(init_alloc->Reset(), "init_alloc reset");
        throw_if_failed(cmd->Reset(init_alloc.get(), nullptr), "init cmd reset");
        cmd->CopyBufferRegion(vb.get(), 0, geom_up.get(), 0, vb_bytes);
        cmd->CopyBufferRegion(ib.get(), 0, geom_up.get(), vb_bytes, ib_bytes);
        D3D12_RESOURCE_BARRIER bs[2]{};
        bs[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bs[0].Transition.pResource = vb.get();
        bs[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        bs[0].Transition.StateAfter = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        bs[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        bs[1] = bs[0];
        bs[1].Transition.pResource = ib.get();
        bs[1].Transition.StateAfter = D3D12_RESOURCE_STATE_INDEX_BUFFER;
        cmd->ResourceBarrier(2, bs);
        throw_if_failed(cmd->Close(), "init cmd close");
        ID3D12CommandList* lists[] = {cmd};
        ctx.direct_queue()->ExecuteCommandLists(1, lists);
        ctx.direct_sync().flush();
    }

    D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), static_cast<UINT>(vb_bytes), sizeof(Vertex)};
    D3D12_INDEX_BUFFER_VIEW ibv{ib->GetGPUVirtualAddress(), static_cast<UINT>(ib_bytes), DXGI_FORMAT_R32_UINT};

    // Frame CBs (3 in flight, 256B aligned)
    ComPtr<ID3D12Resource> frame_cbs[kFrameInFlightCount];
    uint8_t* frame_cb_ptrs[kFrameInFlightCount]{};
    for (uint32_t i = 0; i < kFrameInFlightCount; ++i) {
        frame_cbs[i] = create_upload_buffer(dev, 256);
        throw_if_failed(frame_cbs[i]->Map(0, nullptr, reinterpret_cast<void**>(&frame_cb_ptrs[i])), "Map CB failed");
    }

    // 2. Build Root Signatures & PSOs
    bool use_ntc_sample = (cfg.mode == BaselineMode::B2_NTC_OnSample);
    bool use_bcn_stream = (cfg.mode == BaselineMode::B1_SFS_BCn || cfg.mode == BaselineMode::B3_NTC_OnFeedback);
    bool use_ntc_assets = (cfg.mode == BaselineMode::B2_NTC_OnSample || cfg.mode == BaselineMode::B3_NTC_OnFeedback);

    D3D12_STATIC_SAMPLER_DESC linear_samp{};
    linear_samp.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    linear_samp.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    linear_samp.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    linear_samp.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    linear_samp.MaxLOD = D3D12_FLOAT32_MAX;
    linear_samp.ShaderRegister = 0;
    linear_samp.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    ComPtr<ID3D12RootSignature> gfx_rs;
    ComPtr<ID3D12PipelineState> gfx_pso;
    {
        std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
        std::vector<D3D12_ROOT_PARAMETER> params;

        D3D12_ROOT_PARAMETER p0{};
        p0.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        p0.Descriptor.ShaderRegister = 0;
        p0.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p0);

        if (use_ntc_sample) {
            ranges.resize(2);
            ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kNumMaterials, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kNumMaterials, 0, 1, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};

            for (int r = 0; r < 2; ++r) {
                D3D12_ROOT_PARAMETER rp{};
                rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                rp.DescriptorTable = {1, &ranges[r]};
                rp.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
                params.push_back(rp);
            }
            D3D12_ROOT_PARAMETER mlp_p{};
            mlp_p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            mlp_p.Descriptor.ShaderRegister = 0;
            mlp_p.Descriptor.RegisterSpace = 2;
            mlp_p.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
            params.push_back(mlp_p);
        } else {
            ranges.resize(4);
            ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kNumMaterials, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kNumMaterials, 0, 1, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            ranges[2] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kNumMaterials, 0, 2, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            ranges[3] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, kNumMaterials, 0, 3, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};

            for (int r = 0; r < 4; ++r) {
                D3D12_ROOT_PARAMETER rp{};
                rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                rp.DescriptorTable = {1, &ranges[r]};
                rp.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
                params.push_back(rp);
            }
            D3D12_ROOT_PARAMETER res_p{};
            res_p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            res_p.Descriptor.ShaderRegister = 0;
            res_p.Descriptor.RegisterSpace = 4;
            res_p.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
            params.push_back(res_p);
        }

        D3D12_ROOT_SIGNATURE_DESC rsd{};
        rsd.NumParameters = static_cast<UINT>(params.size());
        rsd.pParameters = params.data();
        rsd.NumStaticSamplers = 1;
        rsd.pStaticSamplers = &linear_samp;
        rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        ComPtr<ID3DBlob> sig, err;
        HRESULT hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, sig.put(), err.put());
        if (FAILED(hr)) {
            std::string msg = err ? std::string(static_cast<const char*>(err->GetBufferPointer()), err->GetBufferSize()) : "";
            throw std::runtime_error("Serialize Gfx RS failed: " + msg);
        }
        throw_if_failed(dev->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                                 IID_ID3D12RootSignature, gfx_rs.put_void()),
                        "Create Gfx RS failed");

        auto vs_bin = read_binary_file(cfg.shader_dir + "/pbr_vs.cso");
        auto ps_bin = read_binary_file(cfg.shader_dir + (use_ntc_sample ? "/pbr_ntc_sample.ps.cso" : "/pbr_bcn.ps.cso"));

        D3D12_INPUT_ELEMENT_DESC il[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TANGENT",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 36, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"MATID",    0, DXGI_FORMAT_R32_UINT,        0, 44, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        };

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = gfx_rs.get();
        pso.VS = {vs_bin.data(), vs_bin.size()};
        pso.PS = {ps_bin.data(), ps_bin.size()};
        pso.InputLayout = {il, 5};
        pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso.RasterizerState.DepthClipEnable = TRUE;
        pso.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso.DepthStencilState.DepthEnable = TRUE;
        pso.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        pso.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        pso.SampleMask = UINT_MAX;
        pso.NumRenderTargets = 1;
        pso.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        pso.SampleDesc.Count = 1;
        throw_if_failed(dev->CreateGraphicsPipelineState(&pso, IID_ID3D12PipelineState, gfx_pso.put_void()),
                        "Create Gfx PSO failed");
    }

    // 3. NTC Assets (G0, G1, MLP) for B2 and B3
    std::vector<ComPtr<ID3D12Resource>> g0_tex(kNumMaterials), g1_tex(kNumMaterials);
    ComPtr<ID3D12Resource> mlp_weights_buf;
    uint32_t g0_srv_base = 0, g1_srv_base = 0;

    if (use_ntc_assets) {
        mlp_weights_buf = create_default_buffer(dev, pack.mlp_table_bytes());
        auto mlp_up = create_upload_buffer(dev, pack.mlp_table_bytes());
        {
            uint8_t* p = nullptr;
            throw_if_failed(mlp_up->Map(0, nullptr, reinterpret_cast<void**>(&p)), "Map mlp_up failed");
            std::memcpy(p, pack.mlp_table_ptr(), pack.mlp_table_bytes());
            mlp_up->Unmap(0, nullptr);
            auto* cmd = ctx.cmd_list();
            throw_if_failed(init_alloc->Reset(), "mlp alloc reset");
            throw_if_failed(cmd->Reset(init_alloc.get(), nullptr), "mlp cmd reset");
            cmd->CopyBufferRegion(mlp_weights_buf.get(), 0, mlp_up.get(), 0, pack.mlp_table_bytes());
            D3D12_RESOURCE_BARRIER b{};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource = mlp_weights_buf.get();
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            b.Transition.StateAfter = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
            b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            cmd->ResourceBarrier(1, &b);
            throw_if_failed(cmd->Close(), "mlp cmd close");
            ID3D12CommandList* lists[] = {cmd};
            ctx.direct_queue()->ExecuteCommandLists(1, lists);
            ctx.direct_sync().flush();
        }

        // Reusable 128 MiB upload buffer for G0 (4096x4096, 13 mips) & G1 (1024x1024, 11 mips)
        auto ntc_staging = create_upload_buffer(dev, 128ULL * 1024ULL * 1024ULL);
        uint8_t* ntc_ptr = nullptr;
        throw_if_failed(ntc_staging->Map(0, nullptr, reinterpret_cast<void**>(&ntc_ptr)), "Map ntc_staging");

        g0_srv_base = srv_heap.alloc(kNumMaterials);
        g1_srv_base = srv_heap.alloc(kNumMaterials);

        D3D12_RESOURCE_STATES ntc_final_state = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
        for (uint32_t m = 0; m < kNumMaterials; ++m) {
            g0_tex[m] = create_texture2d(dev, 4096, 4096, 13, DXGI_FORMAT_R8G8B8A8_UNORM);
            upload_texture_mips_with_staging(
                ctx, init_alloc.get(), g0_tex[m].get(), ntc_staging.get(), ntc_ptr,
                DXGI_FORMAT_R8G8B8A8_UNORM, 4096, 4096, 0, 12,
                [&](uint16_t mip, std::vector<uint8_t>& out) { pack.build_g0_mip_tiled(m, mip, out); },
                ntc_final_state
            );

            g1_tex[m] = create_texture2d(dev, 1024, 1024, 11, DXGI_FORMAT_R16G16B16A16_FLOAT);
            upload_texture_mips_with_staging(
                ctx, init_alloc.get(), g1_tex[m].get(), ntc_staging.get(), ntc_ptr,
                DXGI_FORMAT_R16G16B16A16_FLOAT, 1024, 1024, 0, 10,
                [&](uint16_t mip, std::vector<uint8_t>& out) { pack.build_g1_mip_tiled(m, mip, out); },
                ntc_final_state
            );

            D3D12_SHADER_RESOURCE_VIEW_DESC sd0{};
            sd0.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            sd0.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            sd0.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            sd0.Texture2D.MipLevels = 13;
            dev->CreateShaderResourceView(g0_tex[m].get(), &sd0, srv_heap.cpu(g0_srv_base + m));

            D3D12_SHADER_RESOURCE_VIEW_DESC sd1{};
            sd1.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            sd1.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
            sd1.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            sd1.Texture2D.MipLevels = 11;
            dev->CreateShaderResourceView(g1_tex[m].get(), &sd1, srv_heap.cpu(g1_srv_base + m));
        }
        ntc_staging->Unmap(0, nullptr);
    }

    // 4. BCn Textures (B0 committed full vs B1/B3 reserved + tile heap + Sampler Feedback)
    std::vector<ComPtr<ID3D12Resource>> alb_tex(kNumMaterials), nrm_tex(kNumMaterials), orm_tex(kNumMaterials);
    uint32_t alb_srv_base = 0, nrm_srv_base = 0, orm_srv_base = 0, fb_uav_base = 0, fb_cpu_uav_base = 0;

    ComPtr<ID3D12Resource> dummy_residency_buf;
    ComPtr<ID3D12Heap> mip_tail_heap;
    ComPtr<ID3D12Heap> tile_cache_heap;
    std::vector<ComPtr<ID3D12Resource>> fb_tex(kNumMaterials), fb_resolved(kNumMaterials);
    ComPtr<ID3D12Resource> fb_readback[kFrameInFlightCount];
    uint32_t max_tile_triplets = (cfg.bcn_tile_cache_mib * 1024u) / (3u * 64u); // 8,192 for 1536 MiB
    TileLruCache lru_cache;
    ComPtr<ID3D12Resource> residency_buf, residency_upload[kFrameInFlightCount];
    uint8_t* residency_up_ptr[kFrameInFlightCount]{};

    // Streaming staging buffers (B1 PCIe upload OR B3 async compute decoded output)
    ComPtr<ID3D12Resource> pcie_upload_ring[kFrameInFlightCount];
    uint8_t* pcie_up_ptr[kFrameInFlightCount]{};
    ComPtr<ID3D12Resource> vram_tile_staging;
    ComPtr<ID3D12CommandAllocator> copy_alloc[kFrameInFlightCount];
    ComPtr<ID3D12GraphicsCommandList> copy_cmd;

    // B3 Compute decode resources
    ComPtr<ID3D12RootSignature> cs_rs;
    ComPtr<ID3D12PipelineState> cs_pso;
    ComPtr<ID3D12Resource> jobs_default, jobs_upload[kFrameInFlightCount];
    uint8_t* jobs_up_ptr[kFrameInFlightCount]{};
    ComPtr<ID3D12CommandAllocator> cs_alloc[kFrameInFlightCount];
    ComPtr<ID3D12GraphicsCommandList> cs_cmd;
    uint32_t decode_uav_slot = 0;

    if (!use_ntc_sample) {
        alb_srv_base = srv_heap.alloc(kNumMaterials);
        nrm_srv_base = srv_heap.alloc(kNumMaterials);
        orm_srv_base = srv_heap.alloc(kNumMaterials);
        fb_uav_base = srv_heap.alloc(kNumMaterials);
        fb_cpu_uav_base = ctx.cpu_uav_heap().alloc(kNumMaterials);

        if (cfg.mode == BaselineMode::B0_BCn_Full) {
            dummy_residency_buf = create_default_buffer(dev, 4096);
            auto b0_staging = create_upload_buffer(dev, 96ULL * 1024ULL * 1024ULL);
            uint8_t* b0_ptr = nullptr;
            throw_if_failed(b0_staging->Map(0, nullptr, reinterpret_cast<void**>(&b0_ptr)), "Map b0_staging");

            for (uint32_t m = 0; m < kNumMaterials; ++m) {
                alb_tex[m] = create_texture2d(dev, kVirtualTexSize, kVirtualTexSize, kBcnMipLevels, DXGI_FORMAT_BC7_UNORM);
                upload_texture_mips_with_staging(
                    ctx, init_alloc.get(), alb_tex[m].get(), b0_staging.get(), b0_ptr,
                    DXGI_FORMAT_BC7_UNORM, kVirtualTexSize, kVirtualTexSize, 0, kBcnMipLevels - 1,
                    [&](uint16_t mip, std::vector<uint8_t>& out) { pack.build_bcn_mip_tiled(m, mip, 0, out); },
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                );

                nrm_tex[m] = create_texture2d(dev, kVirtualTexSize, kVirtualTexSize, kBcnMipLevels, DXGI_FORMAT_BC5_UNORM);
                upload_texture_mips_with_staging(
                    ctx, init_alloc.get(), nrm_tex[m].get(), b0_staging.get(), b0_ptr,
                    DXGI_FORMAT_BC5_UNORM, kVirtualTexSize, kVirtualTexSize, 0, kBcnMipLevels - 1,
                    [&](uint16_t mip, std::vector<uint8_t>& out) { pack.build_bcn_mip_tiled(m, mip, 1, out); },
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                );

                orm_tex[m] = create_texture2d(dev, kVirtualTexSize, kVirtualTexSize, kBcnMipLevels, DXGI_FORMAT_BC7_UNORM);
                upload_texture_mips_with_staging(
                    ctx, init_alloc.get(), orm_tex[m].get(), b0_staging.get(), b0_ptr,
                    DXGI_FORMAT_BC7_UNORM, kVirtualTexSize, kVirtualTexSize, 0, kBcnMipLevels - 1,
                    [&](uint16_t mip, std::vector<uint8_t>& out) { pack.build_bcn_mip_tiled(m, mip, 2, out); },
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                );
            }
            b0_staging->Unmap(0, nullptr);
        } else {
            // B1 & B3 Reserved Resources + Tile Pool + Sampler Feedback
            D3D12_HEAP_DESC tail_hd{};
            tail_hd.SizeInBytes = static_cast<UINT64>(kNumMaterials) * 3ULL * kTileBytes; // 96 tiles = 6 MiB
            tail_hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
            tail_hd.Flags = D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES;
            throw_if_failed(dev->CreateHeap(&tail_hd, IID_ID3D12Heap, mip_tail_heap.put_void()), "Create mip_tail_heap");

            D3D12_HEAP_DESC pool_hd{};
            pool_hd.SizeInBytes = static_cast<UINT64>(max_tile_triplets) * 3ULL * kTileBytes;
            pool_hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
            pool_hd.Flags = D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES;
            throw_if_failed(dev->CreateHeap(&pool_hd, IID_ID3D12Heap, tile_cache_heap.put_void()), "Create tile_cache_heap");

            auto make_reserved = [&](DXGI_FORMAT fmt) {
                D3D12_RESOURCE_DESC rd{};
                rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                rd.Width = kVirtualTexSize;
                rd.Height = kVirtualTexSize;
                rd.DepthOrArraySize = 1;
                rd.MipLevels = kBcnMipLevels;
                rd.Format = fmt;
                rd.SampleDesc.Count = 1;
                rd.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
                ComPtr<ID3D12Resource> r;
                throw_if_failed(dev->CreateReservedResource(&rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                            IID_ID3D12Resource, r.put_void()),
                                "CreateReservedResource failed");
                return r;
            };

            auto tail_staging = create_upload_buffer(dev, 4ULL * 1024ULL * 1024ULL);
            uint8_t* tail_ptr = nullptr;
            throw_if_failed(tail_staging->Map(0, nullptr, reinterpret_cast<void**>(&tail_ptr)), "Map tail_staging");

            for (uint32_t m = 0; m < kNumMaterials; ++m) {
                alb_tex[m] = make_reserved(DXGI_FORMAT_BC7_UNORM);
                nrm_tex[m] = make_reserved(DXGI_FORMAT_BC5_UNORM);
                orm_tex[m] = make_reserved(DXGI_FORMAT_BC7_UNORM);

                // Query tiling on first texture (Mip 6..10 is packed tile 1365)
                UINT total_tiles = 0;
                D3D12_PACKED_MIP_INFO packed_info{};
                D3D12_TILE_SHAPE tile_shape{};
                UINT sub_count = kBcnMipLevels;
                std::vector<D3D12_SUBRESOURCE_TILING> sub_tilings(sub_count);
                dev->GetResourceTiling(alb_tex[m].get(), &total_tiles, &packed_info, &tile_shape,
                                       &sub_count, 0, sub_tilings.data());

                uint32_t tail_start_tile = packed_info.StartTileIndexInOverallResource;
                D3D12_TILED_RESOURCE_COORDINATE coord{0, 0, 0, packed_info.NumStandardMips};
                D3D12_TILE_REGION_SIZE region{packed_info.NumTilesForPackedMips, FALSE, 0, 0, 0};
                D3D12_TILE_RANGE_FLAGS rflag = D3D12_TILE_RANGE_FLAG_NONE;
                UINT rcount = packed_info.NumTilesForPackedMips;

                ID3D12Resource* pair3[3] = {alb_tex[m].get(), nrm_tex[m].get(), orm_tex[m].get()};
                DXGI_FORMAT fmts3[3] = {DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_BC5_UNORM, DXGI_FORMAT_BC7_UNORM};
                for (int k = 0; k < 3; ++k) {
                    UINT heap_off = (m * 3u + static_cast<uint32_t>(k)) * packed_info.NumTilesForPackedMips;
                    ctx.direct_queue()->UpdateTileMappings(
                        pair3[k], 1, &coord, &region, mip_tail_heap.get(), 1, &rflag, &heap_off, &rcount,
                        D3D12_TILE_MAPPING_FLAG_NONE
                    );
                    upload_texture_mips_with_staging(
                        ctx, init_alloc.get(), pair3[k], tail_staging.get(), tail_ptr,
                        fmts3[k], kVirtualTexSize, kVirtualTexSize,
                        static_cast<uint16_t>(packed_info.NumStandardMips), kBcnMipLevels - 1,
                        [&](uint16_t mip, std::vector<uint8_t>& out) { pack.build_bcn_mip_tiled(m, mip, k, out); },
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                    );
                }
                (void)tail_start_tile;
            }
            tail_staging->Unmap(0, nullptr);

            // Create 32 Sampler Feedback resources + 32x32 R8_UINT resolve targets
            D3D12_HEAP_PROPERTIES def_hp{};
            def_hp.Type = D3D12_HEAP_TYPE_DEFAULT;
            for (uint32_t m = 0; m < kNumMaterials; ++m) {
                D3D12_RESOURCE_DESC1 fbd{};
                fbd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                fbd.Width = kVirtualTexSize;
                fbd.Height = kVirtualTexSize;
                fbd.DepthOrArraySize = 1;
                fbd.MipLevels = kBcnMipLevels;
                fbd.Format = kFormatSamplerFeedbackMinMipOpaque;
                fbd.SampleDesc.Count = 1;
                fbd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                fbd.SamplerFeedbackMipRegion = {256, 256, 1};
                throw_if_failed(
                    dev->CreateCommittedResource2(
                        &def_hp, D3D12_HEAP_FLAG_NONE, &fbd,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, nullptr,
                        IID_ID3D12Resource, fb_tex[m].put_void()),
                    "CreateCommittedResource2 feedback failed"
                );
                dev->CreateSamplerFeedbackUnorderedAccessView(
                    alb_tex[m].get(), fb_tex[m].get(), srv_heap.cpu(fb_uav_base + m));
                dev->CreateSamplerFeedbackUnorderedAccessView(
                    alb_tex[m].get(), fb_tex[m].get(), ctx.cpu_uav_heap().cpu(fb_cpu_uav_base + m));

                fb_resolved[m] = create_texture2d(dev, 32, 32, 1, DXGI_FORMAT_R8_UINT);
            }

            // Readback: 32 materials * (32 rows * 256B pitch = 8192 bytes) = 262,144 bytes
            for (uint32_t i = 0; i < kFrameInFlightCount; ++i) {
                fb_readback[i] = create_readback_buffer(dev, kNumMaterials * 32ULL * 256ULL);
            }

            lru_cache.init(max_tile_triplets);
            uint64_t res_bytes = kNumMaterials * 1024ULL * sizeof(uint32_t); // 128 KB
            residency_buf = create_default_buffer(dev, res_bytes, D3D12_RESOURCE_FLAG_NONE,
                                                  D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            for (uint32_t i = 0; i < kFrameInFlightCount; ++i) {
                residency_upload[i] = create_upload_buffer(dev, res_bytes);
                throw_if_failed(residency_upload[i]->Map(0, nullptr, reinterpret_cast<void**>(&residency_up_ptr[i])),
                                "Map residency_upload");
                std::memcpy(residency_up_ptr[i], lru_cache.residency_map_data(), res_bytes);
            }

            uint64_t max_stage_bytes = static_cast<uint64_t>(cfg.max_tile_triplets_per_frame) * 3ULL * kTileBytes;
            vram_tile_staging = create_default_buffer(
                dev, max_stage_bytes,
                use_ntc_assets ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COMMON
            );

            if (cfg.mode == BaselineMode::B1_SFS_BCn) {
                for (uint32_t i = 0; i < kFrameInFlightCount; ++i) {
                    pcie_upload_ring[i] = create_upload_buffer(dev, max_stage_bytes);
                    throw_if_failed(pcie_upload_ring[i]->Map(0, nullptr, reinterpret_cast<void**>(&pcie_up_ptr[i])),
                                    "Map pcie_upload_ring");
                    throw_if_failed(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY,
                                                                IID_ID3D12CommandAllocator, copy_alloc[i].put_void()),
                                    "Create copy_alloc");
                }
                throw_if_failed(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, copy_alloc[0].get(), nullptr,
                                                       IID_ID3D12GraphicsCommandList, copy_cmd.put_void()),
                                "Create copy_cmd");
                copy_cmd->Close();
            } else if (cfg.mode == BaselineMode::B3_NTC_OnFeedback) {
                // Compute decode RS & PSO
                D3D12_DESCRIPTOR_RANGE cs_ranges[3]{};
                cs_ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kNumMaterials, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
                cs_ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kNumMaterials, 0, 1, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
                cs_ranges[2] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};

                D3D12_ROOT_PARAMETER cs_params[5]{};
                cs_params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                cs_params[0].DescriptorTable = {1, &cs_ranges[0]};
                cs_params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                cs_params[1].DescriptorTable = {1, &cs_ranges[1]};
                cs_params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
                cs_params[2].Descriptor.ShaderRegister = 0;
                cs_params[2].Descriptor.RegisterSpace = 2;
                cs_params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
                cs_params[3].Descriptor.ShaderRegister = 0;
                cs_params[3].Descriptor.RegisterSpace = 3;
                cs_params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                cs_params[4].DescriptorTable = {1, &cs_ranges[2]};

                D3D12_ROOT_SIGNATURE_DESC cs_rsd{5, cs_params, 1, &linear_samp, D3D12_ROOT_SIGNATURE_FLAG_NONE};
                ComPtr<ID3DBlob> cs_sig, cs_err;
                throw_if_failed(D3D12SerializeRootSignature(&cs_rsd, D3D_ROOT_SIGNATURE_VERSION_1, cs_sig.put(), cs_err.put()),
                                "Serialize CS RS failed");
                throw_if_failed(dev->CreateRootSignature(0, cs_sig->GetBufferPointer(), cs_sig->GetBufferSize(),
                                                         IID_ID3D12RootSignature, cs_rs.put_void()),
                                "Create CS RS failed");

                auto cs_bin = read_binary_file(cfg.shader_dir + "/ntc_decode_bcn.cs.cso");
                D3D12_COMPUTE_PIPELINE_STATE_DESC cpso{};
                cpso.pRootSignature = cs_rs.get();
                cpso.CS = {cs_bin.data(), cs_bin.size()};
                throw_if_failed(dev->CreateComputePipelineState(&cpso, IID_ID3D12PipelineState, cs_pso.put_void()),
                                "Create CS PSO failed");

                uint64_t jobs_bytes = static_cast<uint64_t>(cfg.max_tile_triplets_per_frame) * sizeof(TileDecodeJob);
                jobs_default = create_default_buffer(dev, jobs_bytes);
                for (uint32_t i = 0; i < kFrameInFlightCount; ++i) {
                    jobs_upload[i] = create_upload_buffer(dev, jobs_bytes);
                    throw_if_failed(jobs_upload[i]->Map(0, nullptr, reinterpret_cast<void**>(&jobs_up_ptr[i])),
                                    "Map jobs_upload");
                    throw_if_failed(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                                IID_ID3D12CommandAllocator, cs_alloc[i].put_void()),
                                    "Create cs_alloc");
                }
                throw_if_failed(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, cs_alloc[0].get(), nullptr,
                                                       IID_ID3D12GraphicsCommandList, cs_cmd.put_void()),
                                "Create cs_cmd");
                cs_cmd->Close();

                decode_uav_slot = srv_heap.alloc(1);
                D3D12_UNORDERED_ACCESS_VIEW_DESC uavd{};
                uavd.Format = DXGI_FORMAT_R32_TYPELESS;
                uavd.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                uavd.Buffer.NumElements = static_cast<UINT>(max_stage_bytes / 4ULL);
                uavd.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
                dev->CreateUnorderedAccessView(vram_tile_staging.get(), nullptr, &uavd, srv_heap.cpu(decode_uav_slot));
            }
        }

        // Create SRVs for Albedo, Normal, ORM
        for (uint32_t m = 0; m < kNumMaterials; ++m) {
            D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
            sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            sd.Texture2D.MipLevels = kBcnMipLevels;
            sd.Format = DXGI_FORMAT_BC7_UNORM;
            dev->CreateShaderResourceView(alb_tex[m].get(), &sd, srv_heap.cpu(alb_srv_base + m));
            sd.Format = DXGI_FORMAT_BC5_UNORM;
            dev->CreateShaderResourceView(nrm_tex[m].get(), &sd, srv_heap.cpu(nrm_srv_base + m));
            sd.Format = DXGI_FORMAT_BC7_UNORM;
            dev->CreateShaderResourceView(orm_tex[m].get(), &sd, srv_heap.cpu(orm_srv_base + m));
        }
    }

    // 5. Render Loop: warmup_frames + measured_frames
    int total_frames = cfg.warmup_frames + cfg.measured_frames;
    int slot_last_frame_idx[kFrameInFlightCount] = {-1, -1, -1};
    bool fb_slot_valid[kFrameInFlightCount] = {false, false, false};
    uint64_t fb_slot_fence[kFrameInFlightCount] = {0, 0, 0};

    struct InflightStreamBatch {
        bool active = false;
        uint64_t fence_val = 0;
        std::vector<uint32_t> keys;
        std::vector<TileLruCache::AllocResult> allocs;
    } inflight;

    LARGE_INTEGER qpc_freq_li{};
    QueryPerformanceFrequency(&qpc_freq_li);
    double qpc_freq = static_cast<double>(qpc_freq_li.QuadPart);

    const float aspect = static_cast<float>(cfg.width) / static_cast<float>(cfg.height);
    const auto& bays = scene.bays();
    const auto& checkpoints = scene.quality_checkpoints();

    if (!cfg.capture_checkpoints) {
        logger.start_run(cfg.run_id, cfg.config_name, static_cast<size_t>(cfg.measured_frames));
    }

    for (int f_iter = 0; f_iter < total_frames; ++f_iter) {
        int frame_idx = f_iter - cfg.warmup_frames;

        ctx.pump_messages();
        LARGE_INTEGER t_cpu_0{};
        QueryPerformanceCounter(&t_cpu_0);

        uint32_t slot = ctx.begin_frame();
        ID3D12GraphicsCommandList7* cmd = ctx.cmd_list();

        // Attribute exact GPU duration of the frame that previously used `slot`
        double prev_gpu_ms = 0.0;
        if (ctx.try_read_gpu_frame_ms(slot, prev_gpu_ms) && slot_last_frame_idx[slot] >= 0 && !cfg.capture_checkpoints) {
            logger.set_frame_gpu_ms(slot_last_frame_idx[slot], prev_gpu_ms);
        }

        double frame_pcie_mb = 0.0;
        uint32_t frame_tiles_decoded = 0;
        bool committed_this_frame = false;

        // 5A-1. Commit previously dispatched async Copy/Compute batch onto DirectQueue
        if (use_bcn_stream && inflight.active) {
            committed_this_frame = true;
            ID3D12Fence* async_fence = (cfg.mode == BaselineMode::B1_SFS_BCn)
                ? ctx.copy_sync().fence.get()
                : ctx.compute_sync().fence.get();
            ctx.direct_queue()->Wait(async_fence, inflight.fence_val);

            uint32_t n_jobs = static_cast<uint32_t>(inflight.keys.size());
            std::vector<D3D12_RESOURCE_BARRIER> copy_barriers;
            copy_barriers.reserve(n_jobs * 3 + 1);
            bool tex_transitioned[kNumMaterials]{};

            for (uint32_t j = 0; j < n_jobs; ++j) {
                const auto& al = inflight.allocs[j];
                if (al.evicted) {
                    uint32_t em = (al.evicted_key >> 16) & 0x1Fu;
                    uint32_t emip = (al.evicted_key >> 12) & 0xFu;
                    uint32_t ety = (al.evicted_key >> 6) & 0x3Fu;
                    uint32_t etx = al.evicted_key & 0x3Fu;
                    D3D12_TILED_RESOURCE_COORDINATE ec{etx, ety, 0, emip};
                    D3D12_TILE_REGION_SIZE er{1, FALSE, 0, 0, 0};
                    D3D12_TILE_RANGE_FLAGS ef = D3D12_TILE_RANGE_FLAG_NULL;
                    UINT ecnt = 1;
                    ctx.direct_queue()->UpdateTileMappings(alb_tex[em].get(), 1, &ec, &er, nullptr, 1, &ef, nullptr, &ecnt, D3D12_TILE_MAPPING_FLAG_NONE);
                    ctx.direct_queue()->UpdateTileMappings(nrm_tex[em].get(), 1, &ec, &er, nullptr, 1, &ef, nullptr, &ecnt, D3D12_TILE_MAPPING_FLAG_NONE);
                    ctx.direct_queue()->UpdateTileMappings(orm_tex[em].get(), 1, &ec, &er, nullptr, 1, &ef, nullptr, &ecnt, D3D12_TILE_MAPPING_FLAG_NONE);
                }

                uint32_t key = inflight.keys[j];
                uint32_t m = (key >> 16) & 0x1Fu;
                uint32_t mip = (key >> 12) & 0xFu;
                uint32_t ty = (key >> 6) & 0x3Fu;
                uint32_t tx = key & 0x3Fu;

                D3D12_TILED_RESOURCE_COORDINATE nc{tx, ty, 0, mip};
                D3D12_TILE_REGION_SIZE nr{1, FALSE, 0, 0, 0};
                D3D12_TILE_RANGE_FLAGS nf = D3D12_TILE_RANGE_FLAG_NONE;
                UINT ncnt = 1;
                UINT h0 = al.slot * 3u + 0u;
                UINT h1 = al.slot * 3u + 1u;
                UINT h2 = al.slot * 3u + 2u;

                ctx.direct_queue()->UpdateTileMappings(alb_tex[m].get(), 1, &nc, &nr, tile_cache_heap.get(), 1, &nf, &h0, &ncnt, D3D12_TILE_MAPPING_FLAG_NONE);
                ctx.direct_queue()->UpdateTileMappings(nrm_tex[m].get(), 1, &nc, &nr, tile_cache_heap.get(), 1, &nf, &h1, &ncnt, D3D12_TILE_MAPPING_FLAG_NONE);
                ctx.direct_queue()->UpdateTileMappings(orm_tex[m].get(), 1, &nc, &nr, tile_cache_heap.get(), 1, &nf, &h2, &ncnt, D3D12_TILE_MAPPING_FLAG_NONE);

                if (!tex_transitioned[m]) {
                    tex_transitioned[m] = true;
                    ID3D12Resource* r3[3] = {alb_tex[m].get(), nrm_tex[m].get(), orm_tex[m].get()};
                    for (int k = 0; k < 3; ++k) {
                        D3D12_RESOURCE_BARRIER b{};
                        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                        b.Transition.pResource = r3[k];
                        b.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
                        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                        copy_barriers.push_back(b);
                    }
                }
            }
            if (!copy_barriers.empty()) {
                cmd->ResourceBarrier(static_cast<UINT>(copy_barriers.size()), copy_barriers.data());
            }

            for (uint32_t j = 0; j < n_jobs; ++j) {
                uint32_t key = inflight.keys[j];
                uint32_t m = (key >> 16) & 0x1Fu;
                uint32_t mip = (key >> 12) & 0xFu;
                uint32_t ty = (key >> 6) & 0x3Fu;
                uint32_t tx = key & 0x3Fu;
                D3D12_TILED_RESOURCE_COORDINATE nc{tx, ty, 0, mip};
                D3D12_TILE_REGION_SIZE nr{1, FALSE, 0, 0, 0};
                UINT64 base_off = static_cast<UINT64>(j) * 3ULL * kTileBytes;
                cmd->CopyTiles(alb_tex[m].get(), &nc, &nr, vram_tile_staging.get(), base_off, D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE);
                cmd->CopyTiles(nrm_tex[m].get(), &nc, &nr, vram_tile_staging.get(), base_off + kTileBytes, D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE);
                cmd->CopyTiles(orm_tex[m].get(), &nc, &nr, vram_tile_staging.get(), base_off + 2ULL * kTileBytes, D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE);
                lru_cache.mark_resident(key, true);
            }

            for (auto& b : copy_barriers) {
                std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
            }
            if (!copy_barriers.empty()) {
                cmd->ResourceBarrier(static_cast<UINT>(copy_barriers.size()), copy_barriers.data());
            }

            // Upload updated 128 KB TileResidencyMap
            uint64_t res_bytes = kNumMaterials * 1024ULL * sizeof(uint32_t);
            std::memcpy(residency_up_ptr[slot], lru_cache.residency_map_data(), res_bytes);
            D3D12_RESOURCE_BARRIER rb{};
            rb.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            rb.Transition.pResource = residency_buf.get();
            rb.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            rb.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
            rb.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            cmd->ResourceBarrier(1, &rb);
            cmd->CopyBufferRegion(residency_buf.get(), 0, residency_upload[slot].get(), 0, res_bytes);
            std::swap(rb.Transition.StateBefore, rb.Transition.StateAfter);
            cmd->ResourceBarrier(1, &rb);

            inflight.active = false;
            inflight.keys.clear();
            inflight.allocs.clear();
        }

        // 5A-2. Read Sampler Feedback from slot's prior frame and queue async Copy/Compute (overlaps with this frame's draw)
        if (use_bcn_stream && fb_slot_valid[slot] && !inflight.active && !committed_this_frame) {
            ctx.direct_sync().wait(fb_slot_fence[slot]);

            uint8_t* fb_ptr = nullptr;
            D3D12_RANGE rb_rng{0, kNumMaterials * 32ULL * 256ULL};
            throw_if_failed(fb_readback[slot]->Map(0, &rb_rng, reinterpret_cast<void**>(&fb_ptr)),
                            "Map fb_readback failed");

            std::vector<std::vector<uint32_t>> req_by_mip(kStreamedMipCount);
            for (uint32_t m = 0; m < kNumMaterials; ++m) {
                const uint8_t* mat_fb = fb_ptr + static_cast<size_t>(m) * 32ULL * 256ULL;
                for (uint32_t ty = 0; ty < 32; ++ty) {
                    const uint8_t* row = mat_fb + static_cast<size_t>(ty) * 256ULL;
                    for (uint32_t tx = 0; tx < 32; ++tx) {
                        uint8_t min_mip = row[tx];
                        if (min_mip >= kStreamedMipCount) continue;
                        for (int k = static_cast<int>(kStreamedMipCount) - 1; k >= static_cast<int>(min_mip); --k) {
                            uint32_t uk = static_cast<uint32_t>(k);
                            uint32_t key = TileKey{m, uk, tx >> uk, ty >> uk}.pack();
                            if (lru_cache.contains(key)) {
                                lru_cache.touch(key);
                            } else {
                                req_by_mip[uk].push_back(key);
                            }
                        }
                    }
                }
            }
            D3D12_RANGE empty_rng{0, 0};
            fb_readback[slot]->Unmap(0, &empty_rng);

            for (int k = static_cast<int>(kStreamedMipCount) - 1; k >= 0; --k) {
                auto& vec = req_by_mip[static_cast<size_t>(k)];
                std::sort(vec.begin(), vec.end());
                vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
                for (uint32_t key : vec) {
                    if (inflight.keys.size() >= cfg.max_tile_triplets_per_frame) break;
                    if (!lru_cache.contains(key)) {
                        inflight.keys.push_back(key);
                        inflight.allocs.push_back(lru_cache.acquire_slot(key));
                    }
                }
                if (inflight.keys.size() >= cfg.max_tile_triplets_per_frame) break;
            }

            if (!inflight.keys.empty()) {
                uint32_t n_jobs = static_cast<uint32_t>(inflight.keys.size());
                uint64_t total_copy_bytes = static_cast<uint64_t>(n_jobs) * 3ULL * kTileBytes;

                // Ensure prior CopyTiles on DirectQueue has finished reading vram_tile_staging before overwriting it
                uint64_t prev_direct_val = ctx.direct_sync().next_fence_value - 1;

                if (cfg.mode == BaselineMode::B1_SFS_BCn) {
                    uint8_t* dst_base = pcie_up_ptr[slot];
                    for (uint32_t j = 0; j < n_jobs; ++j) {
                        uint32_t key = inflight.keys[j];
                        uint32_t m = (key >> 16) & 0x1Fu;
                        uint32_t mip = (key >> 12) & 0xFu;
                        uint32_t ty = (key >> 6) & 0x3Fu;
                        uint32_t tx = key & 0x3Fu;
                        for (int k = 0; k < 3; ++k) {
                            pack.build_bcn_tile_64kb(
                                m, mip, tx, ty, k,
                                dst_base + (static_cast<size_t>(j) * 3ULL + static_cast<size_t>(k)) * kTileBytes
                            );
                        }
                    }

                    ctx.copy_queue()->Wait(ctx.direct_sync().fence.get(), prev_direct_val);
                    throw_if_failed(copy_alloc[slot]->Reset(), "copy_alloc reset");
                    throw_if_failed(copy_cmd->Reset(copy_alloc[slot].get(), nullptr), "copy_cmd reset");
                    copy_cmd->CopyBufferRegion(vram_tile_staging.get(), 0, pcie_upload_ring[slot].get(), 0, total_copy_bytes);
                    throw_if_failed(copy_cmd->Close(), "copy_cmd close");
                    ID3D12CommandList* c_lists[] = {copy_cmd.get()};
                    ctx.copy_queue()->ExecuteCommandLists(1, c_lists);
                    inflight.fence_val = ctx.copy_sync().signal();
                    inflight.active = true;

                    frame_pcie_mb = static_cast<double>(total_copy_bytes) / (1024.0 * 1024.0);
                } else {
                    auto* jobs_ptr = reinterpret_cast<TileDecodeJob*>(jobs_up_ptr[slot]);
                    for (uint32_t j = 0; j < n_jobs; ++j) {
                        uint32_t key = inflight.keys[j];
                        uint32_t base_off = j * 3u * kTileBytes;
                        jobs_ptr[j] = {
                            (key >> 16) & 0x1Fu,
                            (key >> 12) & 0xFu,
                            key & 0x3Fu,
                            (key >> 6) & 0x3Fu,
                            base_off,
                            base_off + kTileBytes,
                            base_off + 2u * kTileBytes,
                            0u
                        };
                    }

                    ctx.compute_queue()->Wait(ctx.direct_sync().fence.get(), prev_direct_val);
                    throw_if_failed(cs_alloc[slot]->Reset(), "cs_alloc reset");
                    throw_if_failed(cs_cmd->Reset(cs_alloc[slot].get(), cs_pso.get()), "cs_cmd reset");
                    cs_cmd->CopyBufferRegion(jobs_default.get(), 0, jobs_upload[slot].get(), 0,
                                             static_cast<UINT64>(n_jobs) * sizeof(TileDecodeJob));

                    D3D12_RESOURCE_BARRIER jb{};
                    jb.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    jb.Transition.pResource = jobs_default.get();
                    jb.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                    jb.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                    jb.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                    cs_cmd->ResourceBarrier(1, &jb);

                    ID3D12DescriptorHeap* heaps[] = {srv_heap.heap()};
                    cs_cmd->SetDescriptorHeaps(1, heaps);
                    cs_cmd->SetComputeRootSignature(cs_rs.get());
                    cs_cmd->SetComputeRootDescriptorTable(0, srv_heap.gpu(g0_srv_base));
                    cs_cmd->SetComputeRootDescriptorTable(1, srv_heap.gpu(g1_srv_base));
                    cs_cmd->SetComputeRootShaderResourceView(2, mlp_weights_buf->GetGPUVirtualAddress());
                    cs_cmd->SetComputeRootShaderResourceView(3, jobs_default->GetGPUVirtualAddress());
                    cs_cmd->SetComputeRootDescriptorTable(4, srv_heap.gpu(decode_uav_slot));
                    cs_cmd->Dispatch(8, 8, n_jobs);

                    D3D12_RESOURCE_BARRIER end_bs[2]{};
                    end_bs[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    end_bs[0].Transition.pResource = vram_tile_staging.get();
                    end_bs[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                    end_bs[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
                    end_bs[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                    end_bs[1] = jb;
                    end_bs[1].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                    end_bs[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
                    cs_cmd->ResourceBarrier(2, end_bs);

                    throw_if_failed(cs_cmd->Close(), "cs_cmd close");
                    ID3D12CommandList* cs_lists[] = {cs_cmd.get()};
                    ctx.compute_queue()->ExecuteCommandLists(1, cs_lists);
                    inflight.fence_val = ctx.compute_sync().signal();
                    inflight.active = true;

                    frame_tiles_decoded = n_jobs;
                }
            }
        }

        // 5B. Clear Feedback UAVs
        ID3D12DescriptorHeap* heaps[] = {srv_heap.heap()};
        cmd->SetDescriptorHeaps(1, heaps);

        if (use_bcn_stream) {
            const UINT clr[4] = {0xFFu, 0xFFu, 0xFFu, 0xFFu};
            for (uint32_t m = 0; m < kNumMaterials; ++m) {
                cmd->ClearUnorderedAccessViewUint(
                    srv_heap.gpu(fb_uav_base + m),
                    ctx.cpu_uav_heap().cpu(fb_cpu_uav_base + m),
                    fb_tex[m].get(), clr, 0, nullptr
                );
            }
        }

        // 5C. Main PBR Raster Pass
        CameraMatrices cam = scene.evaluate_camera(frame_idx, aspect);
        {
            FrameConstants fc{};
            std::memcpy(fc.view_proj, cam.view_proj, sizeof(fc.view_proj));
            std::memcpy(fc.cam_pos, cam.cam_pos, sizeof(fc.cam_pos));
            fc.time_sec = static_cast<float>(f_iter) / 60.0f;
            fc.enable_feedback = use_bcn_stream ? 1u : 0u;
            fc.use_tile_residency = use_bcn_stream ? 1u : 0u;
            std::memcpy(frame_cb_ptrs[slot], &fc, sizeof(fc));
        }

        D3D12_RESOURCE_BARRIER rt_b{};
        rt_b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        rt_b.Transition.pResource = ctx.current_rtv_resource(slot);
        rt_b.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        rt_b.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
        rt_b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmd->ResourceBarrier(1, &rt_b);

        D3D12_CPU_DESCRIPTOR_HANDLE rtv = ctx.rtv_cpu(slot);
        D3D12_CPU_DESCRIPTOR_HANDLE dsv = ctx.dsv_cpu();
        const float clear_col[4] = {0.06f, 0.07f, 0.09f, 1.0f};
        cmd->ClearRenderTargetView(rtv, clear_col, 0, nullptr);
        cmd->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        cmd->OMSetRenderTargets(1, &rtv, FALSE, &dsv);

        D3D12_VIEWPORT vp{0.0f, 0.0f, static_cast<float>(cfg.width), static_cast<float>(cfg.height), 0.0f, 1.0f};
        D3D12_RECT sc{0, 0, static_cast<LONG>(cfg.width), static_cast<LONG>(cfg.height)};
        cmd->RSSetViewports(1, &vp);
        cmd->RSSetScissorRects(1, &sc);

        cmd->SetGraphicsRootSignature(gfx_rs.get());
        cmd->SetPipelineState(gfx_pso.get());
        cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmd->IASetVertexBuffers(0, 1, &vbv);
        cmd->IASetIndexBuffer(&ibv);

        cmd->SetGraphicsRootConstantBufferView(0, frame_cbs[slot]->GetGPUVirtualAddress());
        if (use_ntc_sample) {
            cmd->SetGraphicsRootDescriptorTable(1, srv_heap.gpu(g0_srv_base));
            cmd->SetGraphicsRootDescriptorTable(2, srv_heap.gpu(g1_srv_base));
            cmd->SetGraphicsRootShaderResourceView(3, mlp_weights_buf->GetGPUVirtualAddress());
        } else {
            cmd->SetGraphicsRootDescriptorTable(1, srv_heap.gpu(alb_srv_base));
            cmd->SetGraphicsRootDescriptorTable(2, srv_heap.gpu(nrm_srv_base));
            cmd->SetGraphicsRootDescriptorTable(3, srv_heap.gpu(orm_srv_base));
            cmd->SetGraphicsRootDescriptorTable(4, srv_heap.gpu(fb_uav_base));
            cmd->SetGraphicsRootShaderResourceView(
                5, use_bcn_stream ? residency_buf->GetGPUVirtualAddress() : dummy_residency_buf->GetGPUVirtualAddress()
            );
        }

        for (const auto& bay : bays) {
            if (scene.is_aabb_visible(bay.min_xyz, bay.max_xyz, cam)) {
                cmd->DrawIndexedInstanced(bay.index_count, 1, bay.index_offset, 0, 0);
            }
        }

        // 5D. Resolve Sampler Feedback -> Readback
        if (use_bcn_stream) {
            std::vector<D3D12_RESOURCE_BARRIER> res_b(kNumMaterials * 2);
            for (uint32_t m = 0; m < kNumMaterials; ++m) {
                res_b[m * 2 + 0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                res_b[m * 2 + 0].Transition.pResource = fb_tex[m].get();
                res_b[m * 2 + 0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                res_b[m * 2 + 0].Transition.StateAfter = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
                res_b[m * 2 + 0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

                res_b[m * 2 + 1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                res_b[m * 2 + 1].Transition.pResource = fb_resolved[m].get();
                res_b[m * 2 + 1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                res_b[m * 2 + 1].Transition.StateAfter = D3D12_RESOURCE_STATE_RESOLVE_DEST;
                res_b[m * 2 + 1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            }
            cmd->ResourceBarrier(static_cast<UINT>(res_b.size()), res_b.data());

            for (uint32_t m = 0; m < kNumMaterials; ++m) {
                cmd->ResolveSubresourceRegion(
                    fb_resolved[m].get(), 0, 0, 0, fb_tex[m].get(), 0, nullptr,
                    DXGI_FORMAT_R8_UINT, D3D12_RESOLVE_MODE_DECODE_SAMPLER_FEEDBACK
                );
            }

            for (uint32_t m = 0; m < kNumMaterials; ++m) {
                res_b[m * 2 + 0].Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
                res_b[m * 2 + 0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                res_b[m * 2 + 1].Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_DEST;
                res_b[m * 2 + 1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            }
            cmd->ResourceBarrier(static_cast<UINT>(res_b.size()), res_b.data());

            for (uint32_t m = 0; m < kNumMaterials; ++m) {
                D3D12_TEXTURE_COPY_LOCATION src_loc{};
                src_loc.pResource = fb_resolved[m].get();
                src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                src_loc.SubresourceIndex = 0;

                D3D12_TEXTURE_COPY_LOCATION dst_loc{};
                dst_loc.pResource = fb_readback[slot].get();
                dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                dst_loc.PlacedFootprint.Offset = static_cast<UINT64>(m) * 32ULL * 256ULL;
                dst_loc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8_UINT;
                dst_loc.PlacedFootprint.Footprint.Width = 32;
                dst_loc.PlacedFootprint.Footprint.Height = 32;
                dst_loc.PlacedFootprint.Footprint.Depth = 1;
                dst_loc.PlacedFootprint.Footprint.RowPitch = 256;

                cmd->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);

                res_b[m].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                res_b[m].Transition.pResource = fb_resolved[m].get();
                res_b[m].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
                res_b[m].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
                res_b[m].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            }
            cmd->ResourceBarrier(kNumMaterials, res_b.data());
        }

        std::swap(rt_b.Transition.StateBefore, rt_b.Transition.StateAfter);
        cmd->ResourceBarrier(1, &rt_b);

        ctx.end_frame(slot, true);

        if (use_bcn_stream) {
            fb_slot_valid[slot] = true;
            fb_slot_fence[slot] = ctx.direct_sync().next_fence_value - 1;
        }

        LARGE_INTEGER t_cpu_1{};
        QueryPerformanceCounter(&t_cpu_1);
        double cpu_ms = static_cast<double>(t_cpu_1.QuadPart - t_cpu_0.QuadPart) * 1000.0 / qpc_freq;

        // Checkpoint capture mode
        if (cfg.capture_checkpoints && frame_idx >= 0) {
            for (const auto& ck : checkpoints) {
                if (ck.second == frame_idx) {
                    std::filesystem::create_directories(cfg.capture_dir);
                    std::string out_png = cfg.capture_dir + "/" + cfg.config_name + "_" + ck.first + ".png";
                    ctx.capture_backbuffer_png(slot, out_png);
                }
            }
        }

        // Record measured frames (gpu_ms will be filled deterministically once `slot` resolves)
        if (!cfg.capture_checkpoints && frame_idx >= 0) {
            FrameMetricRow row;
            row.run_id = cfg.run_id;
            row.config_name = cfg.config_name;
            row.frame_idx = frame_idx;
            row.timestamp_ms = logger.elapsed_ms();
            row.cpu_frame_ms = cpu_ms;
            row.gpu_frame_ms = 0.0;
            row.present_ms = ctx.last_present_ms();
            row.pcie_upload_mb = frame_pcie_mb;
            row.tiles_decoded_dgpu = frame_tiles_decoded;
            logger.record_frame(row);
            slot_last_frame_idx[slot] = frame_idx;
        } else {
            slot_last_frame_idx[slot] = -1;
        }
    }

    ctx.wait_idle();
    if (!cfg.capture_checkpoints) {
        // Drain final 3 in-flight timestamp slots
        for (uint32_t s = 0; s < kFrameInFlightCount; ++s) {
            double tail_gpu_ms = 0.0;
            if (slot_last_frame_idx[s] >= 0 && ctx.try_read_gpu_frame_ms(s, tail_gpu_ms)) {
                logger.set_frame_gpu_ms(slot_last_frame_idx[s], tail_gpu_ms);
            }
        }
        logger.stop_run();
        logger.write_or_append_csv(cfg.frame_csv_path, cfg.mem_power_csv_path);
    }
}

} // namespace pahnts
