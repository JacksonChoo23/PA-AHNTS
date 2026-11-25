#include "../baselines/material_pack.h"
#include "../core/dx12_context.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <pdh.h>
#include <pdhmsg.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace pahnts;

constexpr uint32_t kNumMaterials = 32;
constexpr uint32_t kTileBytes = 65536;
constexpr uint32_t kLatentTileBytes = 73728;  // 64KB G0 + 8KB G1
constexpr uint32_t kBcnTileBytes = 196608;    // 3 * 64KB
constexpr uint32_t kRawTileBytes = 524288;    // 256 * 256 * 8B

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

uint64_t qpc_now() {
    LARGE_INTEGER li{};
    QueryPerformanceCounter(&li);
    return static_cast<uint64_t>(li.QuadPart);
}

uint64_t qpc_freq() {
    LARGE_INTEGER li{};
    QueryPerformanceFrequency(&li);
    return static_cast<uint64_t>(li.QuadPart);
}

std::vector<uint8_t> read_bin(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("Cannot open binary file: " + path);
    auto sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    if (!ifs.read(reinterpret_cast<char*>(buf.data()), sz)) {
        throw std::runtime_error("Failed reading: " + path);
    }
    return buf;
}

ComPtr<ID3D12Resource> make_buf(
    ID3D12Device* dev,
    uint64_t bytes,
    D3D12_HEAP_TYPE heap_type,
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
    D3D12_RESOURCE_STATES init_state = D3D12_RESOURCE_STATE_COMMON
) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = heap_type;
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
        "make_buf failed"
    );
    return res;
}

ComPtr<ID3D12Resource> make_tex2d(ID3D12Device* dev, uint32_t w, uint32_t h, DXGI_FORMAT fmt) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = w;
    rd.Height = h;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = fmt;
    rd.SampleDesc.Count = 1;
    ComPtr<ID3D12Resource> res;
    throw_if_failed(
        dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                     IID_ID3D12Resource, res.put_void()),
        "make_tex2d failed"
    );
    return res;
}

uint16_t f32_to_f16(float x) {
    uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(float));
    uint32_t sign = (u >> 16) & 0x8000u;
    int exp = static_cast<int>((u >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = u & 0x7FFFFFu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

std::vector<uint8_t> build_synthetic_mlp_weights(uint32_t hidden_dim) {
    uint32_t w1_off = 0;
    uint32_t b1_off = w1_off + 16u * hidden_dim * 2u;
    uint32_t w2_off = b1_off + hidden_dim * 2u;
    uint32_t b2_off = w2_off + hidden_dim * hidden_dim * 2u;
    uint32_t w3_off = b2_off + hidden_dim * 2u;
    uint32_t b3_off = w3_off + hidden_dim * 8u * 2u;
    uint32_t raw_bytes = b3_off + 8u * 2u;
    uint32_t stride = ((raw_bytes + 255u) / 256u) * 256u;

    std::vector<uint8_t> blob(static_cast<size_t>(kNumMaterials) * stride, 0);
    std::mt19937 rng(42u + hidden_dim);
    std::normal_distribution<float> norm(0.0f, 1.0f / std::sqrt(static_cast<float>(hidden_dim)));
    for (uint32_t m = 0; m < kNumMaterials; ++m) {
        auto write_f16 = [&](uint32_t off, uint32_t cnt, float scale) {
            uint16_t* dst = reinterpret_cast<uint16_t*>(blob.data() + static_cast<size_t>(m) * stride + off);
            for (uint32_t i = 0; i < cnt; ++i) {
                dst[i] = f32_to_f16(norm(rng) * scale);
            }
        };
        write_f16(w1_off, 16u * hidden_dim, 1.0f);
        write_f16(w2_off, hidden_dim * hidden_dim, 1.0f);
        write_f16(w3_off, hidden_dim * 8u, 1.0f);
    }
    return blob;
}

struct NtcBenchGpuEnv {
    ComPtr<ID3D12RootSignature> rs_ntc;
    ComPtr<ID3D12RootSignature> rs_pack;
    ComPtr<ID3D12PipelineState> pso_h16_bcn;
    ComPtr<ID3D12PipelineState> pso_h32_bcn;
    ComPtr<ID3D12PipelineState> pso_h64_bcn;
    ComPtr<ID3D12PipelineState> pso_h32_raw;
    ComPtr<ID3D12PipelineState> pso_pack_only;

    std::vector<ComPtr<ID3D12Resource>> g0_tex;
    std::vector<ComPtr<ID3D12Resource>> g1_tex;
    ComPtr<ID3D12Resource> mlp_h16, mlp_h32, mlp_h64;
    ComPtr<ID3D12Resource> jobs_def, jobs_up;
    ComPtr<ID3D12Resource> out_def;
    ComPtr<ID3D12Resource> raw_in_def;

    uint32_t g0_srv_base = 0;
    uint32_t g1_srv_base = 0;
    uint32_t out_uav_slot = 0;
    uint32_t raw_in_srv_slot = 0;
    uint32_t out_pack_uav_slot = 0;

    static constexpr uint32_t kMaxBatch = 256;

    void init(DX12Context& ctx, const MaterialPack& pack, const std::string& shader_dir) {
        ID3D12Device8* dev = ctx.device();
        auto& srv = ctx.cbv_srv_uav_heap();

        // Root signature for ntc_bench_decode
        D3D12_STATIC_SAMPLER_DESC samp{};
        samp.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        samp.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        samp.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        samp.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        samp.MaxLOD = D3D12_FLOAT32_MAX;
        samp.ShaderRegister = 0;
        samp.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        {
            D3D12_DESCRIPTOR_RANGE r[3]{};
            r[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kNumMaterials, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            r[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kNumMaterials, 0, 1, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            r[2] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};

            D3D12_ROOT_PARAMETER p[5]{};
            p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            p[0].DescriptorTable = {1, &r[0]};
            p[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            p[1].DescriptorTable = {1, &r[1]};
            p[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            p[2].Descriptor.ShaderRegister = 0;
            p[2].Descriptor.RegisterSpace = 2;
            p[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            p[3].Descriptor.ShaderRegister = 0;
            p[3].Descriptor.RegisterSpace = 3;
            p[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            p[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            p[4].DescriptorTable = {1, &r[2]};
            p[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            D3D12_ROOT_SIGNATURE_DESC rsd{5, p, 1, &samp, D3D12_ROOT_SIGNATURE_FLAG_NONE};
            ComPtr<ID3DBlob> sig, err;
            throw_if_failed(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, sig.put(), err.put()), "rs_ntc");
            throw_if_failed(dev->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                                     IID_ID3D12RootSignature, rs_ntc.put_void()), "create rs_ntc");
        }

        // Root signature for bcn_pack_only
        {
            D3D12_DESCRIPTOR_RANGE r[1]{};
            r[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            D3D12_ROOT_PARAMETER p[2]{};
            p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            p[0].Descriptor.ShaderRegister = 0;
            p[0].Descriptor.RegisterSpace = 0;
            p[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            p[1].DescriptorTable = {1, &r[0]};
            p[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            D3D12_ROOT_SIGNATURE_DESC rsd{2, p, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
            ComPtr<ID3DBlob> sig, err;
            throw_if_failed(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, sig.put(), err.put()), "rs_pack");
            throw_if_failed(dev->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                                     IID_ID3D12RootSignature, rs_pack.put_void()), "create rs_pack");
        }

        auto make_cs_pso = [&](ID3D12RootSignature* rs, const std::string& cso, ComPtr<ID3D12PipelineState>& out_pso) {
            auto bin = read_bin(shader_dir + "/" + cso);
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc{rs, {bin.data(), bin.size()}, 0, {}, D3D12_PIPELINE_STATE_FLAG_NONE};
            throw_if_failed(dev->CreateComputePipelineState(&desc, IID_ID3D12PipelineState, out_pso.put_void()),
                            ("Create PSO failed: " + cso).c_str());
        };
        make_cs_pso(rs_ntc.get(), "ntc_h16_bcn.cs.cso", pso_h16_bcn);
        make_cs_pso(rs_ntc.get(), "ntc_h32_bcn.cs.cso", pso_h32_bcn);
        make_cs_pso(rs_ntc.get(), "ntc_h64_bcn.cs.cso", pso_h64_bcn);
        make_cs_pso(rs_ntc.get(), "ntc_h32_raw.cs.cso", pso_h32_raw);
        make_cs_pso(rs_pack.get(), "bcn_pack_only.cs.cso", pso_pack_only);

        // Upload G0 (256x256 RGBA8) and G1 (64x64 RGBA16F) base textures + weights + jobs
        g0_tex.resize(kNumMaterials);
        g1_tex.resize(kNumMaterials);
        g0_srv_base = srv.alloc(kNumMaterials);
        g1_srv_base = srv.alloc(kNumMaterials);

        auto w16 = build_synthetic_mlp_weights(16);
        auto w64 = build_synthetic_mlp_weights(64);

        mlp_h16 = make_buf(dev, w16.size(), D3D12_HEAP_TYPE_DEFAULT);
        mlp_h32 = make_buf(dev, pack.mlp_table_bytes(), D3D12_HEAP_TYPE_DEFAULT);
        mlp_h64 = make_buf(dev, w64.size(), D3D12_HEAP_TYPE_DEFAULT);

        auto up_w16 = make_buf(dev, w16.size(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto up_w32 = make_buf(dev, pack.mlp_table_bytes(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto up_w64 = make_buf(dev, w64.size(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);

        auto fill_up = [](ID3D12Resource* r, const void* src, size_t n) {
            uint8_t* p = nullptr;
            throw_if_failed(r->Map(0, nullptr, reinterpret_cast<void**>(&p)), "Map weight up");
            std::memcpy(p, src, n);
            r->Unmap(0, nullptr);
        };
        fill_up(up_w16.get(), w16.data(), w16.size());
        fill_up(up_w32.get(), pack.mlp_table_ptr(), pack.mlp_table_bytes());
        fill_up(up_w64.get(), w64.data(), w64.size());

        // Single staging buffer for 32 x (G0 256KB + G1 32KB) ~ 12 MB
        auto tex_up = make_buf(dev, 32ULL * 1024ULL * 1024ULL, D3D12_HEAP_TYPE_UPLOAD,
                               D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        uint8_t* tup = nullptr;
        throw_if_failed(tex_up->Map(0, nullptr, reinterpret_cast<void**>(&tup)), "Map tex_up");

        ComPtr<ID3D12CommandAllocator> init_alloc;
        throw_if_failed(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                    IID_ID3D12CommandAllocator, init_alloc.put_void()), "init_alloc");
        auto* cmd = ctx.cmd_list();
        throw_if_failed(init_alloc->Reset(), "init_alloc reset");
        throw_if_failed(cmd->Reset(init_alloc.get(), nullptr), "init cmd reset");

        cmd->CopyBufferRegion(mlp_h16.get(), 0, up_w16.get(), 0, w16.size());
        cmd->CopyBufferRegion(mlp_h32.get(), 0, up_w32.get(), 0, pack.mlp_table_bytes());
        cmd->CopyBufferRegion(mlp_h64.get(), 0, up_w64.get(), 0, w64.size());

        std::vector<D3D12_RESOURCE_BARRIER> bars;
        auto add_bar = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
            D3D12_RESOURCE_BARRIER b{};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource = r;
            b.Transition.StateBefore = before;
            b.Transition.StateAfter = after;
            b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            bars.push_back(b);
        };
        add_bar(mlp_h16.get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        add_bar(mlp_h32.get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        add_bar(mlp_h64.get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);

        uint64_t cursor = 0;
        const auto& mats = pack.materials();
        for (uint32_t m = 0; m < kNumMaterials; ++m) {
            g0_tex[m] = make_tex2d(dev, 256, 256, DXGI_FORMAT_R8G8B8A8_UNORM);
            g1_tex[m] = make_tex2d(dev, 64, 64, DXGI_FORMAT_R16G16B16A16_FLOAT);

            // Upload G0 (256x256 RGBA8, rowPitch=1024)
            cursor = (cursor + 511ULL) & ~511ULL;
            const auto& g0_s = mats[m].g0_mips[0];
            std::memcpy(tup + cursor, pack.data_ptr(g0_s), g0_s.bytes);
            D3D12_TEXTURE_COPY_LOCATION s0{tex_up.get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            s0.PlacedFootprint.Offset = cursor;
            s0.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, 256, 256, 1, 256u * 4u};
            D3D12_TEXTURE_COPY_LOCATION d0{g0_tex[m].get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            d0.SubresourceIndex = 0;
            cmd->CopyTextureRegion(&d0, 0, 0, 0, &s0, nullptr);
            cursor += 256ULL * 256ULL * 4ULL;

            // Upload G1 (64x64 RGBA16F, rowPitch=512 -> aligned 256B)
            cursor = (cursor + 511ULL) & ~511ULL;
            const auto& g1_s = mats[m].g1_mips[0];
            std::memcpy(tup + cursor, pack.data_ptr(g1_s), g1_s.bytes);
            D3D12_TEXTURE_COPY_LOCATION s1{tex_up.get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            s1.PlacedFootprint.Offset = cursor;
            s1.PlacedFootprint.Footprint = {DXGI_FORMAT_R16G16B16A16_FLOAT, 64, 64, 1, 64u * 8u};
            D3D12_TEXTURE_COPY_LOCATION d1{g1_tex[m].get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            d1.SubresourceIndex = 0;
            cmd->CopyTextureRegion(&d1, 0, 0, 0, &s1, nullptr);
            cursor += 64ULL * 64ULL * 8ULL;

            add_bar(g0_tex[m].get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
            add_bar(g1_tex[m].get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);

            D3D12_SHADER_RESOURCE_VIEW_DESC sd0{};
            sd0.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            sd0.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            sd0.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            sd0.Texture2D.MipLevels = 1;
            dev->CreateShaderResourceView(g0_tex[m].get(), &sd0, srv.cpu(g0_srv_base + m));

            D3D12_SHADER_RESOURCE_VIEW_DESC sd1{};
            sd1.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            sd1.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
            sd1.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            sd1.Texture2D.MipLevels = 1;
            dev->CreateShaderResourceView(g1_tex[m].get(), &sd1, srv.cpu(g1_srv_base + m));
        }

        // Prepare kMaxBatch TileDecodeJob entries
        uint64_t jobs_bytes = static_cast<uint64_t>(kMaxBatch) * sizeof(TileDecodeJob);
        jobs_def = make_buf(dev, jobs_bytes, D3D12_HEAP_TYPE_DEFAULT);
        jobs_up = make_buf(dev, jobs_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        {
            TileDecodeJob* jp = nullptr;
            throw_if_failed(jobs_up->Map(0, nullptr, reinterpret_cast<void**>(&jp)), "Map jobs_up");
            for (uint32_t j = 0; j < kMaxBatch; ++j) {
                jp[j].mat_id = j % kNumMaterials;
                jp[j].mip_level = 0;
                jp[j].tile_x = (j / 32u) % 32u;
                jp[j].tile_y = j % 32u;
                jp[j].dst_albedo_byte_offset = j * kBcnTileBytes;
                jp[j].dst_normal_byte_offset = j * kBcnTileBytes + kTileBytes;
                jp[j].dst_orm_byte_offset = j * kBcnTileBytes + 2u * kTileBytes;
                jp[j].pad = 0;
            }
            jobs_up->Unmap(0, nullptr);
        }
        cmd->CopyBufferRegion(jobs_def.get(), 0, jobs_up.get(), 0, jobs_bytes);
        add_bar(jobs_def.get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);

        cmd->ResourceBarrier(static_cast<UINT>(bars.size()), bars.data());
        throw_if_failed(cmd->Close(), "init close");
        ID3D12CommandList* lists[] = {cmd};
        ctx.direct_queue()->ExecuteCommandLists(1, lists);
        ctx.direct_sync().flush();
        tex_up->Unmap(0, nullptr);

        // Output & Raw input buffers (max 256 * 512KB = 128 MiB)
        uint64_t max_out_bytes = static_cast<uint64_t>(kMaxBatch) * kRawTileBytes;
        out_def = make_buf(dev, max_out_bytes, D3D12_HEAP_TYPE_DEFAULT,
                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        raw_in_def = make_buf(dev, max_out_bytes, D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);

        out_uav_slot = srv.alloc(1);
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavd{};
        uavd.Format = DXGI_FORMAT_R32_TYPELESS;
        uavd.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavd.Buffer.NumElements = static_cast<UINT>(max_out_bytes / 4ULL);
        uavd.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        dev->CreateUnorderedAccessView(out_def.get(), nullptr, &uavd, srv.cpu(out_uav_slot));
    }
};

void run_compute_sweep(
    DX12Context& ctx,
    NtcBenchGpuEnv& env,
    const std::string& out_csv,
    int runs,
    int warmup_iters,
    int measure_iters,
    const std::vector<uint32_t>& custom_batches = {}
) {
    ID3D12Device8* dev = ctx.device();
    auto& srv = ctx.cbv_srv_uav_heap();

    ComPtr<ID3D12CommandAllocator> cs_alloc;
    ComPtr<ID3D12GraphicsCommandList> cs_cmd;
    throw_if_failed(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                IID_ID3D12CommandAllocator, cs_alloc.put_void()), "cs_alloc");
    throw_if_failed(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, cs_alloc.get(), nullptr,
                                           IID_ID3D12GraphicsCommandList, cs_cmd.put_void()), "cs_cmd");
    cs_cmd->Close();

    ComPtr<ID3D12QueryHeap> ts_heap;
    ComPtr<ID3D12Resource> ts_rb = make_buf(dev, sizeof(uint64_t) * 2, D3D12_HEAP_TYPE_READBACK,
                                            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_QUERY_HEAP_DESC qhd{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2, 0};
    throw_if_failed(dev->CreateQueryHeap(&qhd, IID_ID3D12QueryHeap, ts_heap.put_void()), "cs ts_heap");

    struct Variant {
        std::string backend_name;
        uint32_t hidden_dim;
        bool output_bcn;
        std::vector<uint32_t> batches;
        ID3D12PipelineState* pso;
        ID3D12Resource* mlp_buf;
    };

    std::vector<Variant> variants;
    if (!custom_batches.empty()) {
        variants = {
            {"dGPU_DX12_Compute_FP16_BCn", 32, true, custom_batches, env.pso_h32_bcn.get(), env.mlp_h32.get()},
        };
    } else {
        variants = {
            {"dGPU_DX12_Compute_FP16_BCn", 32, true, {1, 4, 16, 64, 256}, env.pso_h32_bcn.get(), env.mlp_h32.get()},
            {"dGPU_DX12_Compute_FP16_Raw", 32, false, {1, 4, 16, 64, 256}, env.pso_h32_raw.get(), env.mlp_h32.get()},
            {"dGPU_DX12_Compute_FP16_BCn", 16, true, {16}, env.pso_h16_bcn.get(), env.mlp_h16.get()},
            {"dGPU_DX12_Compute_FP16_BCn", 64, true, {16}, env.pso_h64_bcn.get(), env.mlp_h64.get()},
        };
    }

    namespace fs = std::filesystem;
    fs::path p(out_csv);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::ofstream ofs(out_csv);
    ofs << "run_id,backend,precision,hidden_dim,batch_tiles,output_format,"
           "gpu_kernel_ms,e2e_dispatch_ms,tiles_per_ms,mtexels_per_sec,effective_gflops\n";
    ofs << std::fixed << std::setprecision(5);

    double ts_freq_ms = static_cast<double>(ctx.compute_sync().timestamp_freq_hz) / 1000.0;
    double qf = static_cast<double>(qpc_freq());

    for (const auto& v : variants) {
        for (uint32_t b : v.batches) {
            // MACs per texel = 2 * (16*H + H*H + H*8) ops (approx FLOPs incl. leakyrelu/sigmoid ~ +2.5*H+24)
            double ops_per_texel = 2.0 * (16.0 * v.hidden_dim + static_cast<double>(v.hidden_dim) * v.hidden_dim + v.hidden_dim * 8.0);
            double total_texels = static_cast<double>(b) * 65536.0;

            auto exec_once = [&](double& gpu_ms, double& cpu_ms) {
                uint64_t t0 = qpc_now();
                throw_if_failed(cs_alloc->Reset(), "cs_alloc reset");
                throw_if_failed(cs_cmd->Reset(cs_alloc.get(), v.pso), "cs_cmd reset");
                cs_cmd->SetComputeRootSignature(env.rs_ntc.get());
                ID3D12DescriptorHeap* heaps[] = {srv.heap()};
                cs_cmd->SetDescriptorHeaps(1, heaps);
                cs_cmd->SetComputeRootDescriptorTable(0, srv.gpu(env.g0_srv_base));
                cs_cmd->SetComputeRootDescriptorTable(1, srv.gpu(env.g1_srv_base));
                cs_cmd->SetComputeRootShaderResourceView(2, v.mlp_buf->GetGPUVirtualAddress());
                cs_cmd->SetComputeRootShaderResourceView(3, env.jobs_def->GetGPUVirtualAddress());
                cs_cmd->SetComputeRootDescriptorTable(4, srv.gpu(env.out_uav_slot));

                cs_cmd->EndQuery(ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
                cs_cmd->Dispatch(8, 8, b);
                cs_cmd->EndQuery(ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
                cs_cmd->ResolveQueryData(ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, ts_rb.get(), 0);
                throw_if_failed(cs_cmd->Close(), "cs_cmd close");

                ID3D12CommandList* lists[] = {cs_cmd.get()};
                ctx.compute_queue()->ExecuteCommandLists(1, lists);
                ctx.compute_sync().flush();
                uint64_t t1 = qpc_now();
                cpu_ms = static_cast<double>(t1 - t0) * 1000.0 / qf;

                uint64_t* rp = nullptr;
                D3D12_RANGE rr{0, sizeof(uint64_t) * 2};
                throw_if_failed(ts_rb->Map(0, &rr, reinterpret_cast<void**>(&rp)), "Map ts_rb");
                gpu_ms = (rp[1] > rp[0]) ? static_cast<double>(rp[1] - rp[0]) / ts_freq_ms : cpu_ms;
                D3D12_RANGE empty{0, 0};
                ts_rb->Unmap(0, &empty);
            };

            for (int w = 0; w < warmup_iters; ++w) {
                double gm = 0, cm = 0;
                exec_once(gm, cm);
            }

            for (int r = 0; r < runs; ++r) {
                double sum_g = 0.0, sum_c = 0.0;
                for (int it = 0; it < measure_iters; ++it) {
                    double gm = 0, cm = 0;
                    exec_once(gm, cm);
                    sum_g += gm;
                    sum_c += cm;
                }
                double mean_g = sum_g / measure_iters;
                double mean_c = sum_c / measure_iters;
                double tiles_per_ms = static_cast<double>(b) / std::max(mean_c, 1e-6);
                double mtex_s = (total_texels / 1e6) / std::max(mean_c / 1000.0, 1e-9);
                double gflops = (total_texels * ops_per_texel / 1e9) / std::max(mean_g / 1000.0, 1e-9);
                ofs << r << "," << v.backend_name << ",FP16," << v.hidden_dim << "," << b << ","
                    << (v.output_bcn ? "BCn_192KB" : "Raw_512KB") << ","
                    << mean_g << "," << mean_c << "," << tiles_per_ms << "," << mtex_s << "," << gflops << "\n";
            }
        }
    }
    std::cout << "[dx12_decode_bench] Wrote compute sweep CSV: " << out_csv << std::endl;
}

void run_transfer_sweep(
    DX12Context& ctx,
    NtcBenchGpuEnv& env,
    const std::string& out_csv,
    int runs,
    int warmup_iters,
    int measure_iters
) {
    ID3D12Device8* dev = ctx.device();
    auto& srv = ctx.cbv_srv_uav_heap();

    uint64_t max_bytes = static_cast<uint64_t>(NtcBenchGpuEnv::kMaxBatch) * kRawTileBytes; // 128 MiB
    auto up_buf = make_buf(dev, max_bytes, D3D12_HEAP_TYPE_UPLOAD,
                           D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto dst_def = make_buf(dev, max_bytes, D3D12_HEAP_TYPE_DEFAULT,
                            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);
    {
        uint8_t* p = nullptr;
        throw_if_failed(up_buf->Map(0, nullptr, reinterpret_cast<void**>(&p)), "Map transfer up_buf");
        for (size_t i = 0; i < max_bytes; i += 64) {
            p[i] = static_cast<uint8_t>(i & 0xFFu);
        }
        up_buf->Unmap(0, nullptr);
    }

    ComPtr<ID3D12CommandAllocator> cp_alloc, cs_alloc;
    ComPtr<ID3D12GraphicsCommandList> cp_cmd, cs_cmd;
    throw_if_failed(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY,
                                                IID_ID3D12CommandAllocator, cp_alloc.put_void()), "cp_alloc");
    throw_if_failed(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, cp_alloc.get(), nullptr,
                                           IID_ID3D12GraphicsCommandList, cp_cmd.put_void()), "cp_cmd");
    cp_cmd->Close();

    throw_if_failed(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                IID_ID3D12CommandAllocator, cs_alloc.put_void()), "cs_alloc");
    throw_if_failed(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, cs_alloc.get(), nullptr,
                                           IID_ID3D12GraphicsCommandList, cs_cmd.put_void()), "cs_cmd");
    cs_cmd->Close();

    ComPtr<ID3D12QueryHeap> cp_ts_heap, cs_ts_heap;
    ComPtr<ID3D12Resource> cp_ts_rb = make_buf(dev, sizeof(uint64_t) * 2, D3D12_HEAP_TYPE_READBACK,
                                               D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    ComPtr<ID3D12Resource> cs_ts_rb = make_buf(dev, sizeof(uint64_t) * 2, D3D12_HEAP_TYPE_READBACK,
                                               D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    {
        D3D12_QUERY_HEAP_DESC qcp{D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP, 2, 0};
        throw_if_failed(dev->CreateQueryHeap(&qcp, IID_ID3D12QueryHeap, cp_ts_heap.put_void()), "cp_ts_heap");
        D3D12_QUERY_HEAP_DESC qcs{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2, 0};
        throw_if_failed(dev->CreateQueryHeap(&qcs, IID_ID3D12QueryHeap, cs_ts_heap.put_void()), "cs_ts_heap");
    }

    namespace fs = std::filesystem;
    fs::path p(out_csv);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::ofstream ofs(out_csv);
    ofs << "run_id,path_name,batch_tiles,upload_kb_per_tile,total_upload_mb,"
           "copy_gpu_ms,compute_gpu_ms,serial_e2e_ms,bandwidth_gbps\n";
    ofs << std::fixed << std::setprecision(5);

    double cp_freq_ms = static_cast<double>(ctx.copy_sync().timestamp_freq_hz) / 1000.0;
    double cs_freq_ms = static_cast<double>(ctx.compute_sync().timestamp_freq_hz) / 1000.0;
    double qf = static_cast<double>(qpc_freq());

    std::vector<uint32_t> batches = {1, 4, 16, 64, 256};
    struct TransferPath {
        std::string name;
        uint32_t bytes_per_tile;
        int compute_mode; // 0=none, 1=mlp_bcn, 2=pack_only
    };
    std::vector<TransferPath> paths = {
        {"Latent_72KB_Plus_dGPU_MLP_BCn", kLatentTileBytes, 1},
        {"HostPredecoded_BCn_192KB_Upload", kBcnTileBytes, 0},
        {"HostPredecoded_Raw_512KB_Plus_GPU_BCnPack", kRawTileBytes, 2},
    };

    for (const auto& tp : paths) {
        for (uint32_t b : batches) {
            uint64_t upload_bytes = static_cast<uint64_t>(b) * tp.bytes_per_tile;
            double upload_mb = static_cast<double>(upload_bytes) / (1024.0 * 1024.0);

            auto exec_once = [&](double& copy_ms, double& comp_ms, double& e2e_ms) {
                uint64_t t0 = qpc_now();
                throw_if_failed(cp_alloc->Reset(), "cp_alloc reset");
                throw_if_failed(cp_cmd->Reset(cp_alloc.get(), nullptr), "cp_cmd reset");
                cp_cmd->EndQuery(cp_ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
                ID3D12Resource* copy_dst = (tp.compute_mode == 2) ? env.raw_in_def.get() : dst_def.get();
                cp_cmd->CopyBufferRegion(copy_dst, 0, up_buf.get(), 0, upload_bytes);
                cp_cmd->EndQuery(cp_ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
                cp_cmd->ResolveQueryData(cp_ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, cp_ts_rb.get(), 0);
                throw_if_failed(cp_cmd->Close(), "cp_cmd close");

                ID3D12CommandList* cpl[] = {cp_cmd.get()};
                ctx.copy_queue()->ExecuteCommandLists(1, cpl);
                ctx.copy_sync().flush();

                comp_ms = 0.0;
                if (tp.compute_mode != 0) {
                    throw_if_failed(cs_alloc->Reset(), "cs_alloc reset");
                    if (tp.compute_mode == 1) {
                        throw_if_failed(cs_cmd->Reset(cs_alloc.get(), env.pso_h32_bcn.get()), "cs reset mlp");
                        cs_cmd->SetComputeRootSignature(env.rs_ntc.get());
                        ID3D12DescriptorHeap* heaps[] = {srv.heap()};
                        cs_cmd->SetDescriptorHeaps(1, heaps);
                        cs_cmd->SetComputeRootDescriptorTable(0, srv.gpu(env.g0_srv_base));
                        cs_cmd->SetComputeRootDescriptorTable(1, srv.gpu(env.g1_srv_base));
                        cs_cmd->SetComputeRootShaderResourceView(2, env.mlp_h32->GetGPUVirtualAddress());
                        cs_cmd->SetComputeRootShaderResourceView(3, env.jobs_def->GetGPUVirtualAddress());
                        cs_cmd->SetComputeRootDescriptorTable(4, srv.gpu(env.out_uav_slot));
                    } else {
                        throw_if_failed(cs_cmd->Reset(cs_alloc.get(), env.pso_pack_only.get()), "cs reset pack");
                        cs_cmd->SetComputeRootSignature(env.rs_pack.get());
                        ID3D12DescriptorHeap* heaps[] = {srv.heap()};
                        cs_cmd->SetDescriptorHeaps(1, heaps);
                        cs_cmd->SetComputeRootShaderResourceView(0, env.raw_in_def->GetGPUVirtualAddress());
                        cs_cmd->SetComputeRootDescriptorTable(1, srv.gpu(env.out_uav_slot));
                    }
                    cs_cmd->EndQuery(cs_ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
                    cs_cmd->Dispatch(8, 8, b);
                    cs_cmd->EndQuery(cs_ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
                    cs_cmd->ResolveQueryData(cs_ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, cs_ts_rb.get(), 0);
                    throw_if_failed(cs_cmd->Close(), "cs close");
                    ID3D12CommandList* csl[] = {cs_cmd.get()};
                    ctx.compute_queue()->ExecuteCommandLists(1, csl);
                    ctx.compute_sync().flush();

                    uint64_t* crp = nullptr;
                    D3D12_RANGE rr{0, sizeof(uint64_t) * 2};
                    throw_if_failed(cs_ts_rb->Map(0, &rr, reinterpret_cast<void**>(&crp)), "Map cs_ts_rb");
                    comp_ms = (crp[1] > crp[0]) ? static_cast<double>(crp[1] - crp[0]) / cs_freq_ms : 0.0;
                    D3D12_RANGE empty{0, 0};
                    cs_ts_rb->Unmap(0, &empty);
                }

                uint64_t t1 = qpc_now();
                e2e_ms = static_cast<double>(t1 - t0) * 1000.0 / qf;

                uint64_t* prp = nullptr;
                D3D12_RANGE rr{0, sizeof(uint64_t) * 2};
                throw_if_failed(cp_ts_rb->Map(0, &rr, reinterpret_cast<void**>(&prp)), "Map cp_ts_rb");
                copy_ms = (prp[1] > prp[0]) ? static_cast<double>(prp[1] - prp[0]) / cp_freq_ms : 0.0;
                D3D12_RANGE empty{0, 0};
                cp_ts_rb->Unmap(0, &empty);
            };

            for (int w = 0; w < warmup_iters; ++w) {
                double a = 0, c = 0, e = 0;
                exec_once(a, c, e);
            }
            for (int r = 0; r < runs; ++r) {
                double sc = 0, sk = 0, se = 0;
                for (int it = 0; it < measure_iters; ++it) {
                    double a = 0, c = 0, e = 0;
                    exec_once(a, c, e);
                    sc += a;
                    sk += c;
                    se += e;
                }
                double mc = sc / measure_iters;
                double mk = sk / measure_iters;
                double me = se / measure_iters;
                double bw_gbps = (mc > 1e-6) ? (static_cast<double>(upload_bytes) / 1e9) / (mc / 1000.0) : 0.0;
                ofs << r << "," << tp.name << "," << b << "," << (tp.bytes_per_tile / 1024.0) << ","
                    << upload_mb << "," << mc << "," << mk << "," << me << "," << bw_gbps << "\n";
            }
        }
    }
    std::cout << "[dx12_decode_bench] Wrote transfer sweep CSV: " << out_csv << std::endl;
}

void run_gpu_congestion_bench(
    DX12Context& ctx,
    NtcBenchGpuEnv& env,
    const std::string& out_csv,
    int runs,
    int warmup_iters,
    int measure_iters,
    bool extra_loads
) {
    ID3D12Device8* dev = ctx.device();
    auto& srv = ctx.cbv_srv_uav_heap();

    ComPtr<ID3D12CommandAllocator> bg_alloc, pr_alloc;
    ComPtr<ID3D12GraphicsCommandList> bg_cmd, pr_cmd;
    throw_if_failed(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_ID3D12CommandAllocator, bg_alloc.put_void()), "cong bg_alloc");
    throw_if_failed(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, bg_alloc.get(), nullptr, IID_ID3D12GraphicsCommandList, bg_cmd.put_void()), "cong bg_cmd");
    bg_cmd->Close();
    throw_if_failed(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_ID3D12CommandAllocator, pr_alloc.put_void()), "cong pr_alloc");
    throw_if_failed(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, pr_alloc.get(), nullptr, IID_ID3D12GraphicsCommandList, pr_cmd.put_void()), "cong pr_cmd");
    pr_cmd->Close();

    ComPtr<ID3D12QueryHeap> ts_heap;
    ComPtr<ID3D12Resource> ts_rb = make_buf(dev, sizeof(uint64_t) * 4, D3D12_HEAP_TYPE_READBACK,
                                            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_QUERY_HEAP_DESC qhd{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 4, 0};
    throw_if_failed(dev->CreateQueryHeap(&qhd, IID_ID3D12QueryHeap, ts_heap.put_void()), "cong ts_heap");

    double ts_freq_ms = static_cast<double>(ctx.compute_sync().timestamp_freq_hz) / 1000.0;
    double qf = static_cast<double>(qpc_freq());

    // Calibrate unit background dispatch duration: pso_h32_bcn Dispatch(8, 8, 2) (~0.4-0.6 ms)
    auto bind_ntc = [&](ID3D12GraphicsCommandList* cmd, ID3D12PipelineState* pso, ID3D12Resource* mlp) {
        cmd->SetComputeRootSignature(env.rs_ntc.get());
        ID3D12DescriptorHeap* heaps[] = {srv.heap()};
        cmd->SetDescriptorHeaps(1, heaps);
        cmd->SetComputeRootDescriptorTable(0, srv.gpu(env.g0_srv_base));
        cmd->SetComputeRootDescriptorTable(1, srv.gpu(env.g1_srv_base));
        cmd->SetComputeRootShaderResourceView(2, mlp->GetGPUVirtualAddress());
        cmd->SetComputeRootShaderResourceView(3, env.jobs_def->GetGPUVirtualAddress());
        cmd->SetComputeRootDescriptorTable(4, srv.gpu(env.out_uav_slot));
        cmd->SetPipelineState(pso);
    };

    double unit_ms = 0.5;
    {
        double sum_u = 0.0;
        for (int it = 0; it < 12; ++it) {
            throw_if_failed(bg_alloc->Reset(), "cal alloc");
            throw_if_failed(bg_cmd->Reset(bg_alloc.get(), env.pso_h32_bcn.get()), "cal cmd");
            bind_ntc(bg_cmd.get(), env.pso_h32_bcn.get(), env.mlp_h32.get());
            bg_cmd->EndQuery(ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
            bg_cmd->Dispatch(8, 8, 2);
            bg_cmd->EndQuery(ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
            bg_cmd->ResolveQueryData(ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, ts_rb.get(), 0);
            throw_if_failed(bg_cmd->Close(), "cal close");
            ID3D12CommandList* l[] = {bg_cmd.get()};
            ctx.compute_queue()->ExecuteCommandLists(1, l);
            ctx.compute_sync().flush();
            if (it >= 2) {
                uint64_t* rp = nullptr;
                D3D12_RANGE rr{0, sizeof(uint64_t) * 2};
                throw_if_failed(ts_rb->Map(0, &rr, reinterpret_cast<void**>(&rp)), "Map cal");
                sum_u += static_cast<double>(rp[1] - rp[0]) / ts_freq_ms;
                D3D12_RANGE emp{0, 0};
                ts_rb->Unmap(0, &emp);
            }
        }
        unit_ms = std::max(0.1, sum_u / 10.0);
        std::cout << "[dx12_decode_bench] gpu_congestion_bench unit_ms (h32 b=2) = " << unit_ms << " ms\n";
    }

    std::vector<int> target_loads = extra_loads
        ? std::vector<int>{0, 2, 5, 10, 15, 20, 30, 40, 60}
        : std::vector<int>{0, 2, 5, 10, 20, 40, 60};
    std::vector<uint32_t> probe_batches = {1, 4, 16, 64};

    namespace fs = std::filesystem;
    fs::path p(out_csv);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::ofstream ofs(out_csv);
    ofs << "run_id,iter_idx,target_load_ms,bg_dispatches,batch_tiles,"
           "bg_gpu_ms,probe_kernel_ms,measured_queue_delay_ms,total_complete_ms\n";
    ofs << std::fixed << std::setprecision(5);

    for (int load_ms : target_loads) {
        int k_bg = (load_ms <= 0) ? 0 : std::max(1, static_cast<int>(std::lround(static_cast<double>(load_ms) / unit_ms)));
        for (uint32_t b : probe_batches) {
            auto run_one = [&](double& out_bg_ms, double& out_probe_ms, double& out_q_ms, double& out_tot_ms) {
                if (k_bg > 0) {
                    throw_if_failed(bg_alloc->Reset(), "bg_alloc");
                    throw_if_failed(bg_cmd->Reset(bg_alloc.get(), env.pso_h32_bcn.get()), "bg_cmd");
                    bind_ntc(bg_cmd.get(), env.pso_h32_bcn.get(), env.mlp_h32.get());
                    bg_cmd->EndQuery(ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
                    for (int k = 0; k < k_bg; ++k) {
                        bg_cmd->Dispatch(8, 8, 2);
                    }
                    bg_cmd->EndQuery(ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
                    bg_cmd->ResolveQueryData(ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, ts_rb.get(), 0);
                    throw_if_failed(bg_cmd->Close(), "bg_cmd close");
                }

                throw_if_failed(pr_alloc->Reset(), "pr_alloc");
                throw_if_failed(pr_cmd->Reset(pr_alloc.get(), env.pso_h32_bcn.get()), "pr_cmd");
                bind_ntc(pr_cmd.get(), env.pso_h32_bcn.get(), env.mlp_h32.get());
                pr_cmd->EndQuery(ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
                pr_cmd->Dispatch(8, 8, b);
                pr_cmd->EndQuery(ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 3);
                pr_cmd->ResolveQueryData(ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 2, 2, ts_rb.get(), sizeof(uint64_t) * 2);
                throw_if_failed(pr_cmd->Close(), "pr_cmd close");

                if (k_bg > 0) {
                    ID3D12CommandList* bgl[] = {bg_cmd.get()};
                    ctx.compute_queue()->ExecuteCommandLists(1, bgl);
                }
                uint64_t t0 = qpc_now();
                ID3D12CommandList* prl[] = {pr_cmd.get()};
                ctx.compute_queue()->ExecuteCommandLists(1, prl);
                ctx.compute_sync().flush();
                uint64_t t1 = qpc_now();
                out_tot_ms = static_cast<double>(t1 - t0) * 1000.0 / qf;

                uint64_t* rp = nullptr;
                D3D12_RANGE rr{0, sizeof(uint64_t) * 4};
                throw_if_failed(ts_rb->Map(0, &rr, reinterpret_cast<void**>(&rp)), "Map ts_rb");
                out_bg_ms = (k_bg > 0 && rp[1] > rp[0]) ? static_cast<double>(rp[1] - rp[0]) / ts_freq_ms : 0.0;
                out_probe_ms = (rp[3] > rp[2]) ? static_cast<double>(rp[3] - rp[2]) / ts_freq_ms : out_tot_ms;
                D3D12_RANGE emp{0, 0};
                ts_rb->Unmap(0, &emp);
                out_q_ms = std::max(0.0, out_tot_ms - out_probe_ms);
            };

            for (int w = 0; w < warmup_iters; ++w) {
                double a, c, d, e;
                run_one(a, c, d, e);
            }
            for (int r = 0; r < runs; ++r) {
                for (int it = 0; it < measure_iters; ++it) {
                    double bg_ms = 0, pr_ms = 0, q_ms = 0, tot_ms = 0;
                    run_one(bg_ms, pr_ms, q_ms, tot_ms);
                    ofs << r << "," << it << "," << load_ms << "," << k_bg << "," << b << ","
                        << bg_ms << "," << pr_ms << "," << q_ms << "," << tot_ms << "\n";
                }
            }
        }
    }
    std::cout << "[dx12_decode_bench] Wrote GPU congestion CSV: " << out_csv << std::endl;
}

void run_sustained_dgpu_load(
    DX12Context& ctx,
    NtcBenchGpuEnv& env,
    const std::string& out_csv,
    const std::string& sync_file,
    int duration_ms,
    bool pin_dgpu_thread = false,
    int prewarm_temp_c = 0,
    bool standalone_power_only = false,
    int load_hidden = 64
) {
    ID3D12Device8* dev = ctx.device();
    auto& srv = ctx.cbv_srv_uav_heap();

    // Load NVML + PDH RAPL (PKG, PP0, PP1)
    HMODULE nvml_mod = LoadLibraryW(L"nvml.dll");
    using PFN_Init = int (*)();
    using PFN_Shutdown = int (*)();
    using PFN_Handle = int (*)(unsigned int, void**);
    using PFN_Power = int (*)(void*, unsigned int*);
    using PFN_Clock = int (*)(void*, int, unsigned int*);
    using PFN_Temp = int (*)(void*, int, unsigned int*);

    PFN_Shutdown nv_shut = nullptr;
    PFN_Power nv_pwr = nullptr;
    PFN_Clock nv_clk = nullptr;
    PFN_Temp nv_tmp = nullptr;
    void* nv_dev = nullptr;
    bool nv_ok = false;

    if (nvml_mod) {
        auto nv_init = reinterpret_cast<PFN_Init>(GetProcAddress(nvml_mod, "nvmlInit_v2"));
        nv_shut = reinterpret_cast<PFN_Shutdown>(GetProcAddress(nvml_mod, "nvmlShutdown"));
        auto nv_hnd = reinterpret_cast<PFN_Handle>(GetProcAddress(nvml_mod, "nvmlDeviceGetHandleByIndex_v2"));
        nv_pwr = reinterpret_cast<PFN_Power>(GetProcAddress(nvml_mod, "nvmlDeviceGetPowerUsage"));
        nv_clk = reinterpret_cast<PFN_Clock>(GetProcAddress(nvml_mod, "nvmlDeviceGetClockInfo"));
        nv_tmp = reinterpret_cast<PFN_Temp>(GetProcAddress(nvml_mod, "nvmlDeviceGetTemperature"));
        if (nv_init && nv_hnd && nv_init() == 0 && nv_hnd(0, &nv_dev) == 0 && nv_dev) {
            nv_ok = true;
        }
    }

    PDH_HQUERY pdh_q = nullptr;
    PDH_HCOUNTER c_pkg = nullptr, c_pp0 = nullptr, c_pp1 = nullptr;
    bool pdh_ok = false;
    if (PdhOpenQueryW(nullptr, 0, &pdh_q) == ERROR_SUCCESS) {
        bool ok_pkg = PdhAddEnglishCounterW(pdh_q, L"\\Energy Meter(RAPL_Package0_PKG)\\Power", 0, &c_pkg) == ERROR_SUCCESS;
        PdhAddEnglishCounterW(pdh_q, L"\\Energy Meter(RAPL_Package0_PP0)\\Power", 0, &c_pp0);
        PdhAddEnglishCounterW(pdh_q, L"\\Energy Meter(RAPL_Package0_PP1)\\Power", 0, &c_pp1);
        if (ok_pkg) {
            PdhCollectQueryData(pdh_q);
            pdh_ok = true;
        }
    }

    ComPtr<ID3D12CommandAllocator> cs_alloc[2];
    ComPtr<ID3D12GraphicsCommandList> cs_cmd[2];
    for (int i = 0; i < 2; ++i) {
        throw_if_failed(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                    IID_ID3D12CommandAllocator, cs_alloc[i].put_void()), "load alloc");
        throw_if_failed(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, cs_alloc[i].get(), nullptr,
                                               IID_ID3D12GraphicsCommandList, cs_cmd[i].put_void()), "load cmd");
        cs_cmd[i]->Close();
    }

    std::atomic<uint64_t> tiles_completed{0};
    std::atomic<bool> stop_flag{false};
    std::mutex stage_mtx;
    std::string cur_stage = "Idle";

    auto read_stage = [&]() -> std::string {
        if (sync_file.empty()) return "dGPU_Solo";
        std::ifstream ifs(sync_file);
        std::string line;
        if (ifs && std::getline(ifs, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) {
                line.pop_back();
            }
            if (!line.empty()) return line;
        }
        return "Idle";
    };

    struct ContentionRow {
        int sample_idx;
        double timestamp_ms;
        std::string stage;
        double dgpu_power_w;
        double cpu_pkg_power_w;
        double cpu_pp0_power_w;
        double igpu_pp1_power_w;
        double total_est_power_w;
        uint32_t dgpu_gfx_clock_mhz;
        uint32_t dgpu_sm_clock_mhz;
        uint32_t dgpu_temp_c;
        double dgpu_tiles_per_sec;
    };
    std::vector<ContentionRow> rows;
    rows.reserve(1024);

    uint64_t t_start = qpc_now();
    double qf = static_cast<double>(qpc_freq());

    std::thread gpu_worker([&]() {
        if (pin_dgpu_thread) {
            SetThreadAffinityMask(GetCurrentThread(), 1ULL);
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
        }
        // Double-buffered compute loop pushing continuous heavy workload
        uint64_t inflight_fv[2] = {0, 0};
        int slot = 0;
        while (!stop_flag.load(std::memory_order_acquire)) {
            if (standalone_power_only) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            std::string st;
            {
                std::lock_guard<std::mutex> lk(stage_mtx);
                st = cur_stage;
            }
            if (st == "Idle" || st == "STOP" || st == "dGPU_Cooldown") {
                if (inflight_fv[0] != 0) { ctx.compute_sync().wait(inflight_fv[0]); inflight_fv[0] = 0; }
                if (inflight_fv[1] != 0) { ctx.compute_sync().wait(inflight_fv[1]); inflight_fv[1] = 0; }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            if (inflight_fv[slot] != 0) {
                ctx.compute_sync().wait(inflight_fv[slot]);
                tiles_completed.fetch_add(32ULL, std::memory_order_relaxed);
                inflight_fv[slot] = 0;
            }

            ID3D12PipelineState* load_pso = (load_hidden == 32) ? env.pso_h32_bcn.get() : env.pso_h64_bcn.get();
            ID3D12Resource* load_mlp = (load_hidden == 32) ? env.mlp_h32.get() : env.mlp_h64.get();
            throw_if_failed(cs_alloc[slot]->Reset(), "load alloc reset");
            throw_if_failed(cs_cmd[slot]->Reset(cs_alloc[slot].get(), load_pso), "load cmd reset");
            cs_cmd[slot]->SetComputeRootSignature(env.rs_ntc.get());
            ID3D12DescriptorHeap* heaps[] = {srv.heap()};
            cs_cmd[slot]->SetDescriptorHeaps(1, heaps);
            cs_cmd[slot]->SetComputeRootDescriptorTable(0, srv.gpu(env.g0_srv_base));
            cs_cmd[slot]->SetComputeRootDescriptorTable(1, srv.gpu(env.g1_srv_base));
            cs_cmd[slot]->SetComputeRootShaderResourceView(2, load_mlp->GetGPUVirtualAddress());
            cs_cmd[slot]->SetComputeRootShaderResourceView(3, env.jobs_def->GetGPUVirtualAddress());
            cs_cmd[slot]->SetComputeRootDescriptorTable(4, srv.gpu(env.out_uav_slot));
            cs_cmd[slot]->Dispatch(8, 8, 32);
            throw_if_failed(cs_cmd[slot]->Close(), "load cmd close");

            ID3D12CommandList* lists[] = {cs_cmd[slot].get()};
            ctx.compute_queue()->ExecuteCommandLists(1, lists);
            inflight_fv[slot] = ctx.compute_sync().signal();
            slot ^= 1;
        }
        ctx.compute_sync().flush();
    });

    if (prewarm_temp_c > 0 && !standalone_power_only && nv_ok && nv_dev && nv_tmp) {
        {
            std::lock_guard<std::mutex> lk(stage_mtx);
            cur_stage = "dGPU_Prewarm";
        }
        uint64_t pw_start = qpc_now();
        while (true) {
            unsigned int tmp = 0;
            if (nv_tmp(nv_dev, 0, &tmp) == 0 && static_cast<int>(tmp) >= prewarm_temp_c) {
                std::cout << "[dx12_decode_bench] Prewarm target " << prewarm_temp_c << " C reached (" << tmp << " C)\n";
                break;
            }
            if ((static_cast<double>(qpc_now() - pw_start) * 1000.0 / qf) >= 15000.0) {
                std::cout << "[dx12_decode_bench] Prewarm timeout 15s (temp=" << tmp << " C)\n";
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        tiles_completed.store(0, std::memory_order_relaxed);
    }

    t_start = qpc_now();
    std::cout << "[dx12_decode_bench] READY_FOR_STAGES" << std::endl;
    int sample_idx = 0;
    uint64_t prev_tiles = 0;
    uint64_t prev_qpc = t_start;
    while (true) {
        std::string st = read_stage();
        {
            std::lock_guard<std::mutex> lk(stage_mtx);
            cur_stage = st;
        }
        if (st == "STOP") break;

        uint64_t now_qpc = qpc_now();
        double elapsed_ms = static_cast<double>(now_qpc - t_start) * 1000.0 / qf;
        if (duration_ms > 0 && elapsed_ms >= static_cast<double>(duration_ms)) break;

        double dt_s = std::max(1e-3, static_cast<double>(now_qpc - prev_qpc) / qf);
        uint64_t cur_tiles = tiles_completed.load(std::memory_order_relaxed);
        double tps = (sample_idx > 0) ? static_cast<double>(cur_tiles - prev_tiles) / dt_s : 0.0;
        prev_tiles = cur_tiles;
        prev_qpc = now_qpc;

        ContentionRow r{};
        r.sample_idx = sample_idx++;
        r.timestamp_ms = elapsed_ms;
        r.stage = st;
        r.dgpu_tiles_per_sec = tps;

        if (nv_ok && nv_dev) {
            unsigned int mw = 0, gfx = 0, sm = 0, tmp = 0;
            if (nv_pwr && nv_pwr(nv_dev, &mw) == 0) r.dgpu_power_w = mw / 1000.0;
            if (nv_clk && nv_clk(nv_dev, 0, &gfx) == 0) r.dgpu_gfx_clock_mhz = gfx;
            if (nv_clk && nv_clk(nv_dev, 1, &sm) == 0) r.dgpu_sm_clock_mhz = sm;
            if (nv_tmp && nv_tmp(nv_dev, 0, &tmp) == 0) r.dgpu_temp_c = tmp;
        }

        if (pdh_ok && PdhCollectQueryData(pdh_q) == ERROR_SUCCESS) {
            auto get_w = [&](PDH_HCOUNTER ch) -> double {
                if (!ch) return 0.0;
                PDH_FMT_COUNTERVALUE v{};
                if (PdhGetFormattedCounterValue(ch, PDH_FMT_DOUBLE, nullptr, &v) == ERROR_SUCCESS && v.CStatus == 0) {
                    return v.doubleValue / 1000.0;
                }
                return 0.0;
            };
            r.cpu_pkg_power_w = get_w(c_pkg);
            r.cpu_pp0_power_w = get_w(c_pp0);
            r.igpu_pp1_power_w = get_w(c_pp1);
        }
        r.total_est_power_w = r.dgpu_power_w + r.cpu_pkg_power_w;
        rows.push_back(r);

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    stop_flag.store(true, std::memory_order_release);
    gpu_worker.join();

    if (pdh_q) PdhCloseQuery(pdh_q);
    if (nv_ok && nv_shut) nv_shut();
    if (nvml_mod) FreeLibrary(nvml_mod);

    namespace fs = std::filesystem;
    fs::path p(out_csv);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::ofstream ofs(out_csv);
    ofs << "sample_idx,timestamp_ms,stage,dgpu_power_w,cpu_pkg_power_w,cpu_pp0_power_w,"
           "igpu_pp1_power_w,total_est_power_w,dgpu_gfx_clock_mhz,dgpu_sm_clock_mhz,dgpu_temp_c,dgpu_tiles_per_sec\n";
    ofs << std::fixed << std::setprecision(3);
    for (const auto& r : rows) {
        ofs << r.sample_idx << "," << r.timestamp_ms << "," << r.stage << ","
            << r.dgpu_power_w << "," << r.cpu_pkg_power_w << "," << r.cpu_pp0_power_w << ","
            << r.igpu_pp1_power_w << "," << r.total_est_power_w << ","
            << r.dgpu_gfx_clock_mhz << "," << r.dgpu_sm_clock_mhz << "," << r.dgpu_temp_c << ","
            << r.dgpu_tiles_per_sec << "\n";
    }
    std::cout << "[dx12_decode_bench] Wrote power contention CSV (" << rows.size() << " rows): " << out_csv << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string mode = "compute_sweep";
        std::string manifest_path = "results/assets_cache/manifest.json";
        std::string shader_dir = "build/shaders";
        std::string out_csv = "results/phase3/dx12_compute_sweep.csv";
        std::string sync_file = "";
        std::string vendor_hex = "0x10DE";
        int runs = 5;
        int warmup = 5;
        int iters = 20;
        int duration_ms = 35000;
        bool pin_dgpu_thread = false;
        int prewarm_temp_c = 0;
        bool standalone_power_only = false;
        bool extra_loads = false;
        int load_hidden = 64;
        std::vector<uint32_t> custom_batches;

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--mode" && i + 1 < argc) mode = argv[++i];
            else if (a == "--manifest" && i + 1 < argc) manifest_path = argv[++i];
            else if (a == "--shader-dir" && i + 1 < argc) shader_dir = argv[++i];
            else if (a == "--out-csv" && i + 1 < argc) out_csv = argv[++i];
            else if (a == "--sync-file" && i + 1 < argc) sync_file = argv[++i];
            else if (a == "--vendor" && i + 1 < argc) vendor_hex = argv[++i];
            else if (a == "--runs" && i + 1 < argc) runs = std::stoi(argv[++i]);
            else if (a == "--warmup" && i + 1 < argc) warmup = std::stoi(argv[++i]);
            else if (a == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
            else if (a == "--duration-ms" && i + 1 < argc) duration_ms = std::stoi(argv[++i]);
            else if (a == "--pin-dgpu-thread") pin_dgpu_thread = true;
            else if (a == "--prewarm-temp-c" && i + 1 < argc) prewarm_temp_c = std::stoi(argv[++i]);
            else if (a == "--standalone-power-only") standalone_power_only = true;
            else if (a == "--extra-loads") extra_loads = true;
            else if (a == "--load-hidden" && i + 1 < argc) load_hidden = std::stoi(argv[++i]);
            else if (a == "--batches" && i + 1 < argc) {
                std::string s = argv[++i];
                size_t pos = 0;
                while (pos < s.size()) {
                    size_t comma = s.find(',', pos);
                    std::string tok = (comma == std::string::npos) ? s.substr(pos) : s.substr(pos, comma - pos);
                    if (!tok.empty()) custom_batches.push_back(static_cast<uint32_t>(std::stoul(tok)));
                    if (comma == std::string::npos) break;
                    pos = comma + 1;
                }
            }
        }

        DX12Context ctx;
        ctx.init(1280, 720, vendor_hex, false);

        MaterialPack pack;
        pack.load(manifest_path);

        NtcBenchGpuEnv env;
        env.init(ctx, pack, shader_dir);

        if (mode == "compute_sweep") {
            run_compute_sweep(ctx, env, out_csv, runs, warmup, iters, custom_batches);
        } else if (mode == "transfer_sweep") {
            run_transfer_sweep(ctx, env, out_csv, runs, warmup, iters);
        } else if (mode == "gpu_congestion_bench") {
            run_gpu_congestion_bench(ctx, env, out_csv, runs, warmup, iters, extra_loads);
        } else if (mode == "sustained_dgpu_load") {
            run_sustained_dgpu_load(ctx, env, out_csv, sync_file, duration_ms,
                                    pin_dgpu_thread, prewarm_temp_c, standalone_power_only, load_hidden);
        } else {
            throw std::runtime_error("Unknown mode: " + mode);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[dx12_decode_bench] FATAL: " << ex.what() << std::endl;
        return 1;
    }
}
