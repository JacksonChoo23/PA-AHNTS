#include "phase2_reinforced.h"
#include "../core/json_lite.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <unordered_map>
#include <vector>

namespace pahnts {

namespace {

constexpr uint32_t kNumMaterials = 32;
constexpr uint32_t kVirtualTexSize = 8192;
constexpr uint32_t kBcnMipLevels = 11;
constexpr uint32_t kStreamedMipCount = 6;
constexpr uint32_t kTileBytes = 65536;
constexpr uint32_t kMaxL1BatchTiles = 512; // 32 MiB staging (zero-truncation invariant per batch)

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

struct LatentTileKey {
    uint32_t is_g1 = 0;
    uint32_t mat_id = 0;
    uint32_t mip = 0;
    uint32_t tx = 0;
    uint32_t ty = 0;

    uint32_t pack() const {
        return ((is_g1 & 1u) << 21) | ((mat_id & 0x1Fu) << 16) |
               ((mip & 0xFu) << 12) | ((ty & 0x3Fu) << 6) | (tx & 0x3Fu);
    }

    static LatentTileKey unpack(uint32_t k) {
        LatentTileKey out;
        out.is_g1 = (k >> 21) & 1u;
        out.mat_id = (k >> 16) & 0x1Fu;
        out.mip = (k >> 12) & 0xFu;
        out.ty = (k >> 6) & 0x3Fu;
        out.tx = k & 0x3Fu;
        return out;
    }
};

struct BcnTileKey {
    uint32_t mat_id;
    uint32_t mip;
    uint32_t tx;
    uint32_t ty;
    uint32_t pack() const {
        return ((mat_id & 0xFFu) << 16) | ((mip & 0xFu) << 12) | ((ty & 0x3Fu) << 6) | (tx & 0x3Fu);
    }
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
    D3D12_RESOURCE_STATES init_state = D3D12_RESOURCE_STATE_COPY_DEST,
    uint64_t* out_alloc_bytes = nullptr
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
    if (out_alloc_bytes) {
        auto info = dev->GetResourceAllocationInfo(0, 1, &rd);
        *out_alloc_bytes += info.SizeInBytes;
    }
    ComPtr<ID3D12Resource> res;
    throw_if_failed(
        dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, init_state, nullptr,
                                     IID_ID3D12Resource, res.put_void()),
        "Create texture2D failed"
    );
    return res;
}

ComPtr<ID3D12Resource> create_reserved_tex2d(ID3D12Device* dev, uint32_t w, uint32_t h, uint16_t mips, DXGI_FORMAT fmt) {
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = w;
    rd.Height = h;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = mips;
    rd.Format = fmt;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    ComPtr<ID3D12Resource> res;
    throw_if_failed(
        dev->CreateReservedResource(&rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                    IID_ID3D12Resource, res.put_void()),
        "CreateReservedResource failed"
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
    ID3D12GraphicsCommandList7* cmd = ctx.cmd_list();
    throw_if_failed(alloc->Reset(), "upload alloc reset");
    throw_if_failed(cmd->Reset(alloc, nullptr), "upload cmd reset");

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
        src_loc.PlacedFootprint.Footprint = {fmt, mw, mh, 1, dst_pitch};

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

    throw_if_failed(cmd->Close(), "upload cmd close");
    ID3D12CommandList* lists[] = {cmd};
    ctx.direct_queue()->ExecuteCommandLists(1, lists);
    ctx.direct_sync().flush();
}

// Supports both batched per-resource UpdateTileMappings (Async) and unbatched per-tile calls (Sync ablation)
class TileMapperHelper {
public:
    void add_unmap(ID3D12Resource* res, uint32_t mip, uint32_t tx, uint32_t ty) {
        ops_.push_back({false, res, mip, tx, ty, 0});
    }

    void add_map(ID3D12Resource* res, uint32_t mip, uint32_t tx, uint32_t ty, uint32_t heap_tile_offset) {
        ops_.push_back({true, res, mip, tx, ty, heap_tile_offset});
    }

    void flush(ID3D12CommandQueue* queue, ID3D12Heap* heap, bool batched) {
        if (!batched) {
            for (const auto& op : ops_) {
                D3D12_TILED_RESOURCE_COORDINATE c{op.tx, op.ty, 0, op.mip};
                D3D12_TILE_REGION_SIZE r{1, FALSE, 0, 0, 0};
                D3D12_TILE_RANGE_FLAGS rf = op.is_map ? D3D12_TILE_RANGE_FLAG_NONE : D3D12_TILE_RANGE_FLAG_NULL;
                UINT cnt = 1;
                queue->UpdateTileMappings(
                    op.res, 1, &c, &r, op.is_map ? heap : nullptr, 1, &rf,
                    op.is_map ? &op.heap_off : nullptr, &cnt, D3D12_TILE_MAPPING_FLAG_NONE
                );
            }
            ops_.clear();
            return;
        }

        struct ResourceBatch {
            std::vector<D3D12_TILED_RESOURCE_COORDINATE> unmap_c;
            std::vector<D3D12_TILE_REGION_SIZE> unmap_r;
            std::vector<D3D12_TILED_RESOURCE_COORDINATE> map_c;
            std::vector<D3D12_TILE_REGION_SIZE> map_r;
            std::vector<UINT> map_h;
        };
        std::unordered_map<ID3D12Resource*, ResourceBatch> grouped;
        for (const auto& op : ops_) {
            auto& b = grouped[op.res];
            if (op.is_map) {
                b.map_c.push_back({op.tx, op.ty, 0, op.mip});
                b.map_r.push_back({1, FALSE, 0, 0, 0});
                b.map_h.push_back(op.heap_off);
            } else {
                b.unmap_c.push_back({op.tx, op.ty, 0, op.mip});
                b.unmap_r.push_back({1, FALSE, 0, 0, 0});
            }
        }
        for (auto& kv : grouped) {
            auto& b = kv.second;
            if (!b.unmap_c.empty()) {
                UINT n = static_cast<UINT>(b.unmap_c.size());
                std::vector<D3D12_TILE_RANGE_FLAGS> f(n, D3D12_TILE_RANGE_FLAG_NULL);
                std::vector<UINT> cnt(n, 1);
                queue->UpdateTileMappings(kv.first, n, b.unmap_c.data(), b.unmap_r.data(),
                                          nullptr, n, f.data(), nullptr, cnt.data(), D3D12_TILE_MAPPING_FLAG_NONE);
            }
            if (!b.map_c.empty()) {
                UINT n = static_cast<UINT>(b.map_c.size());
                std::vector<D3D12_TILE_RANGE_FLAGS> f(n, D3D12_TILE_RANGE_FLAG_NONE);
                std::vector<UINT> cnt(n, 1);
                queue->UpdateTileMappings(kv.first, n, b.map_c.data(), b.map_r.data(),
                                          heap, n, f.data(), b.map_h.data(), cnt.data(), D3D12_TILE_MAPPING_FLAG_NONE);
            }
        }
        ops_.clear();
    }

private:
    struct Op {
        bool is_map;
        ID3D12Resource* res;
        uint32_t mip, tx, ty, heap_off;
    };
    std::vector<Op> ops_;
};

// SSD L3 + Pinned System RAM L2 Latent Pool (strictly sized, tracks hits/misses)
class LatentRamL2Pool {
public:
    ~LatentRamL2Pool() {
        if (file_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
        }
        if (arena_) {
            VirtualFree(arena_, 0, MEM_RELEASE);
            arena_ = nullptr;
        }
    }

    void init(const std::string& manifest_path, uint32_t ram_l2_pool_mib,
              const D3D12_TILE_SHAPE& g0_shape, const D3D12_TILE_SHAPE& g1_shape) {
        g0_shape_ = g0_shape;
        g1_shape_ = g1_shape;

        JsonValue root = JsonParser::parse_file(manifest_path);
        std::string pack_file = root["pack_file"].as_string();
        std::wstring wpack(pack_file.begin(), pack_file.end());
        file_handle_ = CreateFileW(wpack.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                   OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (file_handle_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("SSD L3 CreateFileW failed: " + pack_file);
        }

        uint64_t mlp_off = root["mlp_table"]["offset"].as_u64();
        uint64_t mlp_bytes = root["mlp_table"]["bytes"].as_u64();
        mlp_blob_.resize(static_cast<size_t>(mlp_bytes));
        read_l3_bytes(mlp_off, mlp_bytes, mlp_blob_.data());

        const auto& mats = root["materials"];
        g0_slices_.resize(kNumMaterials);
        g1_slices_.resize(kNumMaterials);
        for (size_t i = 0; i < kNumMaterials && i < mats.size(); ++i) {
            for (size_t k = 0; k < mats[i]["g0_mips"].size(); ++k) {
                g0_slices_[i].push_back({mats[i]["g0_mips"][k]["offset"].as_u64(), mats[i]["g0_mips"][k]["bytes"].as_u64()});
            }
            for (size_t k = 0; k < mats[i]["g1_mips"].size(); ++k) {
                g1_slices_[i].push_back({mats[i]["g1_mips"][k]["offset"].as_u64(), mats[i]["g1_mips"][k]["bytes"].as_u64()});
            }
        }

        cap_slots_ = (ram_l2_pool_mib * 1024u) / 64u;
        arena_bytes_ = static_cast<uint64_t>(cap_slots_) * kTileBytes;
        if (arena_bytes_ > 0) {
            arena_ = static_cast<uint8_t*>(
                VirtualAlloc(nullptr, static_cast<SIZE_T>(arena_bytes_), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
            );
            if (!arena_) throw std::runtime_error("VirtualAlloc failed for RAM L2 latent pool");
            constexpr uint64_t kPinChunk = 64ULL * 1024ULL * 1024ULL;
            for (uint64_t off = 0; off < arena_bytes_; off += kPinChunk) {
                SIZE_T sz = static_cast<SIZE_T>(std::min<uint64_t>(kPinChunk, arena_bytes_ - off));
                if (!VirtualLock(arena_ + off, sz)) break;
            }
        }

        free_slots_.reserve(cap_slots_);
        for (int i = static_cast<int>(cap_slots_) - 1; i >= 0; --i) {
            free_slots_.push_back(static_cast<uint32_t>(i));
        }
    }

    const std::vector<uint8_t>& mlp_blob() const { return mlp_blob_; }
    const std::atomic<double>* sys_ram_latent_mb_ptr() const { return &sys_ram_latent_mb_; }
    double committed_mib() const { return static_cast<double>(arena_bytes_) / (1024.0 * 1024.0); }
    double peak_populated_mib() const { return peak_populated_mib_; }

    void build_g0_mip_from_l3(uint32_t mat_id, uint32_t mip_level, std::vector<uint8_t>& out) {
        uint32_t world_dim = 4096u >> mip_level;
        uint32_t g0_idx = std::min<uint32_t>(mip_level, 8u);
        const BlobSlice& sl = g0_slices_[mat_id][g0_idx];
        std::vector<uint8_t> base(static_cast<size_t>(sl.bytes));
        read_l3_bytes(sl.offset, sl.bytes, base.data());

        uint32_t base_dim = 256u >> g0_idx;
        out.resize(static_cast<size_t>(world_dim) * world_dim * 4ULL);
        if (mip_level >= 9) {
            for (size_t i = 0; i < static_cast<size_t>(world_dim) * world_dim; ++i) {
                std::memcpy(out.data() + i * 4ULL, base.data(), 4);
            }
            return;
        }
        uint32_t base_row_bytes = base_dim * 4u;
        for (uint32_t r = 0; r < world_dim; ++r) {
            const uint8_t* srow = base.data() + (r % base_dim) * base_row_bytes;
            uint8_t* drow = out.data() + static_cast<size_t>(r) * world_dim * 4ULL;
            for (uint32_t rep = 0; rep < 16u; ++rep) {
                std::memcpy(drow + rep * base_row_bytes, srow, base_row_bytes);
            }
        }
    }

    void build_g1_mip_from_l3(uint32_t mat_id, uint32_t mip_level, std::vector<uint8_t>& out) {
        uint32_t world_dim = 1024u >> mip_level;
        uint32_t g1_idx = std::min<uint32_t>(mip_level, 6u);
        const BlobSlice& sl = g1_slices_[mat_id][g1_idx];
        std::vector<uint8_t> base(static_cast<size_t>(sl.bytes));
        read_l3_bytes(sl.offset, sl.bytes, base.data());

        uint32_t base_dim = 64u >> g1_idx;
        out.resize(static_cast<size_t>(world_dim) * world_dim * 8ULL);
        if (mip_level >= 7) {
            for (size_t i = 0; i < static_cast<size_t>(world_dim) * world_dim; ++i) {
                std::memcpy(out.data() + i * 8ULL, base.data(), 8);
            }
            return;
        }
        uint32_t base_row_bytes = base_dim * 8u;
        for (uint32_t r = 0; r < world_dim; ++r) {
            const uint8_t* srow = base.data() + (r % base_dim) * base_row_bytes;
            uint8_t* drow = out.data() + static_cast<size_t>(r) * world_dim * 8ULL;
            for (uint32_t rep = 0; rep < 16u; ++rep) {
                std::memcpy(drow + rep * base_row_bytes, srow, base_row_bytes);
            }
        }
    }

    const uint8_t* get_or_fetch_tile_64kb(uint32_t packed_key, bool& out_l2_hit) {
        auto it = l2_map_.find(packed_key);
        if (it != l2_map_.end()) {
            out_l2_hit = true;
            lru_.splice(lru_.begin(), lru_, it->second);
            return arena_ + static_cast<size_t>(it->second->slot) * kTileBytes;
        }
        out_l2_hit = false;

        uint32_t slot = 0;
        if (!free_slots_.empty()) {
            slot = free_slots_.back();
            free_slots_.pop_back();
        } else {
            auto tail = lru_.back();
            slot = tail.slot;
            l2_map_.erase(tail.key);
            lru_.pop_back();
        }
        lru_.push_front({packed_key, slot});
        l2_map_[packed_key] = lru_.begin();

        uint8_t* dst = arena_ + static_cast<size_t>(slot) * kTileBytes;
        LatentTileKey k = LatentTileKey::unpack(packed_key);
        if (k.is_g1 == 0) {
            const BlobSlice& sl = g0_slices_[k.mat_id][std::min<uint32_t>(k.mip, 8u)];
            std::vector<uint8_t> base(static_cast<size_t>(sl.bytes));
            read_l3_bytes(sl.offset, sl.bytes, base.data());
            uint32_t base_dim = 256u >> std::min<uint32_t>(k.mip, 8u);
            uint32_t tw = g0_shape_.WidthInTexels;
            uint32_t th = g0_shape_.HeightInTexels;
            uint32_t x0 = k.tx * tw;
            uint32_t y0 = k.ty * th;
            for (uint32_t r = 0; r < th; ++r) {
                uint32_t sr = (y0 + r) % base_dim;
                const uint8_t* srow = base.data() + static_cast<size_t>(sr) * base_dim * 4ULL;
                uint8_t* drow = dst + static_cast<size_t>(r) * tw * 4ULL;
                for (uint32_t c = 0; c < tw; ++c) {
                    uint32_t sc = (x0 + c) % base_dim;
                    std::memcpy(drow + c * 4ULL, srow + sc * 4ULL, 4);
                }
            }
        } else {
            const BlobSlice& sl = g1_slices_[k.mat_id][std::min<uint32_t>(k.mip, 6u)];
            std::vector<uint8_t> base(static_cast<size_t>(sl.bytes));
            read_l3_bytes(sl.offset, sl.bytes, base.data());
            uint32_t base_dim = 64u >> std::min<uint32_t>(k.mip, 6u);
            uint32_t tw = g1_shape_.WidthInTexels;
            uint32_t th = g1_shape_.HeightInTexels;
            uint32_t x0 = k.tx * tw;
            uint32_t y0 = k.ty * th;
            for (uint32_t r = 0; r < th; ++r) {
                uint32_t sr = (y0 + r) % base_dim;
                const uint8_t* srow = base.data() + static_cast<size_t>(sr) * base_dim * 8ULL;
                uint8_t* drow = dst + static_cast<size_t>(r) * tw * 8ULL;
                for (uint32_t c = 0; c < tw; ++c) {
                    uint32_t sc = (x0 + c) % base_dim;
                    std::memcpy(drow + c * 8ULL, srow + sc * 8ULL, 8);
                }
            }
        }

        double mb = static_cast<double>(l2_map_.size()) * static_cast<double>(kTileBytes) / (1024.0 * 1024.0);
        peak_populated_mib_ = std::max(peak_populated_mib_, mb);
        sys_ram_latent_mb_.store(mb, std::memory_order_relaxed);
        return dst;
    }

private:
    struct L2Node { uint32_t key; uint32_t slot; };

    void read_l3_bytes(uint64_t offset, uint64_t bytes, void* dst) {
        LARGE_INTEGER li{};
        li.QuadPart = static_cast<LONGLONG>(offset);
        if (!SetFilePointerEx(file_handle_, li, nullptr, FILE_BEGIN)) {
            throw std::runtime_error("SSD L3 SetFilePointerEx failed");
        }
        uint8_t* p = static_cast<uint8_t*>(dst);
        uint64_t rem = bytes;
        while (rem > 0) {
            DWORD chunk = static_cast<DWORD>(std::min<uint64_t>(rem, 64ULL * 1024ULL * 1024ULL));
            DWORD got = 0;
            if (!ReadFile(file_handle_, p, chunk, &got, nullptr) || got == 0) {
                throw std::runtime_error("SSD L3 ReadFile failed");
            }
            p += got;
            rem -= got;
        }
    }

    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    uint8_t* arena_ = nullptr;
    uint64_t arena_bytes_ = 0;
    uint32_t cap_slots_ = 0;
    D3D12_TILE_SHAPE g0_shape_{128, 128, 1};
    D3D12_TILE_SHAPE g1_shape_{128, 64, 1};
    std::vector<uint8_t> mlp_blob_;
    std::vector<std::vector<BlobSlice>> g0_slices_;
    std::vector<std::vector<BlobSlice>> g1_slices_;
    std::list<L2Node> lru_;
    std::unordered_map<uint32_t, std::list<L2Node>::iterator> l2_map_;
    std::vector<uint32_t> free_slots_;
    std::atomic<double> sys_ram_latent_mb_{0.0};
    double peak_populated_mib_ = 0.0;
};

// Pinned RAM L2 BCn Triplet Cache for optimized B1_SFS_BCn_Async_RamCache
class BcnRamL2Pool {
public:
    ~BcnRamL2Pool() {
        if (arena_) {
            VirtualFree(arena_, 0, MEM_RELEASE);
            arena_ = nullptr;
        }
    }

    void init(const std::string& manifest_path, uint32_t ram_mib) {
        pack_.load(manifest_path);
        cap_triplets_ = (ram_mib * 1024u) / (3u * 64u);
        arena_bytes_ = static_cast<uint64_t>(cap_triplets_) * 3ULL * kTileBytes;
        if (arena_bytes_ > 0) {
            arena_ = static_cast<uint8_t*>(
                VirtualAlloc(nullptr, static_cast<SIZE_T>(arena_bytes_), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
            );
            if (!arena_) throw std::runtime_error("VirtualAlloc failed for BcnRamL2Pool");
            constexpr uint64_t kPinChunk = 64ULL * 1024ULL * 1024ULL;
            for (uint64_t off = 0; off < arena_bytes_; off += kPinChunk) {
                SIZE_T sz = static_cast<SIZE_T>(std::min<uint64_t>(kPinChunk, arena_bytes_ - off));
                if (!VirtualLock(arena_ + off, sz)) break;
            }
        }
        free_slots_.reserve(cap_triplets_);
        for (int i = static_cast<int>(cap_triplets_) - 1; i >= 0; --i) {
            free_slots_.push_back(static_cast<uint32_t>(i));
        }
    }

    const MaterialPack& pack() const { return pack_; }
    const std::atomic<double>* sys_ram_mb_ptr() const { return &sys_ram_mb_; }
    double committed_mib() const { return static_cast<double>(arena_bytes_) / (1024.0 * 1024.0); }
    double peak_populated_mib() const { return peak_populated_mib_; }

    const uint8_t* get_or_build_triplet_192kb(uint32_t packed_key, bool& out_l2_hit) {
        auto it = map_.find(packed_key);
        if (it != map_.end()) {
            out_l2_hit = true;
            lru_.splice(lru_.begin(), lru_, it->second);
            return arena_ + static_cast<size_t>(it->second->slot) * 3ULL * kTileBytes;
        }
        out_l2_hit = false;

        uint32_t slot = 0;
        if (!free_slots_.empty()) {
            slot = free_slots_.back();
            free_slots_.pop_back();
        } else {
            auto tail = lru_.back();
            slot = tail.slot;
            map_.erase(tail.key);
            lru_.pop_back();
        }
        lru_.push_front({packed_key, slot});
        map_[packed_key] = lru_.begin();

        uint8_t* dst = arena_ + static_cast<size_t>(slot) * 3ULL * kTileBytes;
        uint32_t m = (packed_key >> 16) & 0x1Fu;
        uint32_t mip = (packed_key >> 12) & 0xFu;
        uint32_t ty = (packed_key >> 6) & 0x3Fu;
        uint32_t tx = packed_key & 0x3Fu;
        for (int k = 0; k < 3; ++k) {
            pack_.build_bcn_tile_64kb(m, mip, tx, ty, k, dst + static_cast<size_t>(k) * kTileBytes);
        }

        double mb = static_cast<double>(map_.size()) * (3.0 * 64.0 / 1024.0);
        peak_populated_mib_ = std::max(peak_populated_mib_, mb);
        sys_ram_mb_.store(mb, std::memory_order_relaxed);
        return dst;
    }

private:
    struct Node { uint32_t key; uint32_t slot; };
    MaterialPack pack_;
    uint8_t* arena_ = nullptr;
    uint64_t arena_bytes_ = 0;
    uint32_t cap_triplets_ = 0;
    std::list<Node> lru_;
    std::unordered_map<uint32_t, std::list<Node>::iterator> map_;
    std::vector<uint32_t> free_slots_;
    std::atomic<double> sys_ram_mb_{0.0};
    double peak_populated_mib_ = 0.0;
};

class VramL1Lru {
public:
    void init(uint32_t cap_tiles) {
        cap_ = cap_tiles;
        list_.clear();
        map_.clear();
        free_slots_.clear();
        free_slots_.reserve(cap_tiles);
        for (int i = static_cast<int>(cap_tiles) - 1; i >= 0; --i) {
            free_slots_.push_back(static_cast<uint32_t>(i));
        }
    }

    bool contains(uint32_t key) const { return map_.find(key) != map_.end(); }
    uint32_t used_slots() const { return static_cast<uint32_t>(map_.size()); }
    uint32_t cap_slots() const { return cap_; }

    void touch(uint32_t key, uint64_t frame_id) {
        auto it = map_.find(key);
        if (it != map_.end()) {
            it->second->last_frame = frame_id;
            list_.splice(list_.begin(), list_, it->second);
        }
    }

    struct AllocResult { uint32_t slot = 0; bool evicted = false; uint32_t evicted_key = 0; };

    AllocResult acquire(uint32_t key, uint64_t frame_id) {
        AllocResult r;
        if (!free_slots_.empty()) {
            r.slot = free_slots_.back();
            free_slots_.pop_back();
        } else {
            auto victim = std::prev(list_.end());
            uint64_t best_score = UINT64_MAX;
            auto cur = victim;
            for (int step = 0; step < 16 && !list_.empty(); ++step) {
                uint64_t score = cur->last_frame + cur->weight;
                if (score < best_score) {
                    best_score = score;
                    victim = cur;
                }
                if (cur == list_.begin()) break;
                --cur;
            }
            r.evicted = true;
            r.evicted_key = victim->key;
            r.slot = victim->slot;
            map_.erase(victim->key);
            list_.erase(victim);
        }
        LatentTileKey lk = LatentTileKey::unpack(key);
        uint32_t w = lk.is_g1 ? 24u : (lk.mip >= 2u ? 16u : (lk.mip == 1u ? 8u : 0u));
        list_.push_front({key, r.slot, frame_id, w});
        map_[key] = list_.begin();
        return r;
    }

private:
    struct Node { uint32_t key; uint32_t slot; uint64_t last_frame; uint32_t weight; };
    uint32_t cap_ = 0;
    std::list<Node> list_;
    std::unordered_map<uint32_t, std::list<Node>::iterator> map_;
    std::vector<uint32_t> free_slots_;
};

class BcnTileLru {
public:
    void init(uint32_t max_triplets) {
        lru_.clear();
        map_.clear();
        free_slots_.clear();
        free_slots_.reserve(max_triplets);
        for (int i = static_cast<int>(max_triplets) - 1; i >= 0; --i) {
            free_slots_.push_back(static_cast<uint32_t>(i));
        }
        std::memset(resident_mask_, 0, sizeof(resident_mask_));
        std::memset(residency_map_, 6, sizeof(residency_map_));
    }

    bool contains(uint32_t key) const { return map_.find(key) != map_.end(); }
    uint32_t used_slots() const { return static_cast<uint32_t>(map_.size()); }

    void touch(uint32_t key) {
        auto it = map_.find(key);
        if (it != map_.end()) lru_.splice(lru_.begin(), lru_, it->second);
    }

    struct AllocResult { uint32_t slot = 0; bool evicted = false; uint32_t evicted_key = 0; };

    AllocResult acquire(uint32_t new_key) {
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
            set_bit(tail.key, false);
        }
        lru_.push_front({new_key, r.slot});
        map_[new_key] = lru_.begin();
        return r;
    }

    void mark_resident(uint32_t key, bool on) { set_bit(key, on); }
    const uint32_t* residency_map_data() const { return residency_map_; }

    uint32_t finest_resident_mip(uint32_t m, uint32_t ty, uint32_t tx) const {
        return residency_map_[m * 1024u + ty * 32u + tx];
    }

private:
    struct Node { uint32_t key; uint32_t slot; };
    void set_bit(uint32_t packed_key, bool on) {
        uint32_t m = (packed_key >> 16) & 0x1Fu;
        uint32_t mip = (packed_key >> 12) & 0xFu;
        uint32_t ty = (packed_key >> 6) & 0x3Fu;
        uint32_t tx = packed_key & 0x3Fu;
        if (mip >= kStreamedMipCount) return;
        uint32_t span = 1u << mip;
        uint32_t x0 = tx * span, y0 = ty * span;
        uint8_t bit = static_cast<uint8_t>(1u << mip);
        for (uint32_t y = y0; y < std::min(32u, y0 + span); ++y) {
            for (uint32_t x = x0; x < std::min(32u, x0 + span); ++x) {
                uint32_t idx = m * 1024u + y * 32u + x;
                if (on) resident_mask_[idx] |= bit;
                else    resident_mask_[idx] &= ~bit;
                uint8_t mask = resident_mask_[idx];
                uint32_t finest = 6u;
                for (int k = 5; k >= 0; --k) {
                    if (mask & (1u << k)) finest = static_cast<uint32_t>(k);
                    else break;
                }
                residency_map_[idx] = finest;
            }
        }
    }

    std::list<Node> lru_;
    std::unordered_map<uint32_t, std::list<Node>::iterator> map_;
    std::vector<uint32_t> free_slots_;
    uint8_t resident_mask_[kNumMaterials * 1024]{};
    uint32_t residency_map_[kNumMaterials * 1024]{};
};

static inline int floor_div_i(int a, int b) {
    int q = a / b;
    int r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) --q;
    return q;
}

// Enumerate all unique G0 & G1 standard tiles intersecting BCn tile (mip, tx, ty) + 1-texel wrap halo
void gather_required_latent_tiles_with_halo(
    uint32_t m, uint32_t mip, uint32_t tx, uint32_t ty,
    uint32_t g0_std_mips, uint32_t g1_std_mips,
    const D3D12_TILE_SHAPE& g0_shape, const D3D12_TILE_SHAPE& g1_shape,
    const std::vector<D3D12_SUBRESOURCE_TILING>& g0_tilings,
    const std::vector<D3D12_SUBRESOURCE_TILING>& g1_tilings,
    std::vector<uint32_t>& out_keys
) {
    uint32_t bcn_dim = 32u >> mip;

    // G0 at same mip (4096 >> mip)
    if (mip < g0_std_mips) {
        int g0_w = static_cast<int>(4096u >> mip);
        int g0_h = g0_w;
        int tw = static_cast<int>(g0_shape.WidthInTexels);
        int th = static_cast<int>(g0_shape.HeightInTexels);
        int wx = static_cast<int>(g0_tilings[mip].WidthInTiles);
        int wy = static_cast<int>(g0_tilings[mip].HeightInTiles);

        int x0 = static_cast<int>((tx * static_cast<uint32_t>(g0_w)) / bcn_dim) - 1;
        int x1 = static_cast<int>(((tx + 1u) * static_cast<uint32_t>(g0_w)) / bcn_dim); // inclusive of +1 border texel
        int y0 = static_cast<int>((ty * static_cast<uint32_t>(g0_h)) / bcn_dim) - 1;
        int y1 = static_cast<int>(((ty + 1u) * static_cast<uint32_t>(g0_h)) / bcn_dim);

        int tx_s = floor_div_i(x0, tw);
        int tx_e = floor_div_i(x1, tw);
        int ty_s = floor_div_i(y0, th);
        int ty_e = floor_div_i(y1, th);

        int n_tx = std::min(tx_e - tx_s + 1, wx);
        int n_ty = std::min(ty_e - ty_s + 1, wy);
        for (int dy = 0; dy < n_ty; ++dy) {
            uint32_t uy = static_cast<uint32_t>(((ty_s + dy) % wy + wy) % wy);
            for (int dx = 0; dx < n_tx; ++dx) {
                uint32_t ux = static_cast<uint32_t>(((tx_s + dx) % wx + wx) % wx);
                out_keys.push_back(LatentTileKey{0u, m, mip, ux, uy}.pack());
            }
        }
    }

    // G1 at g1_mip = max(0, mip - 2) with rectangular 128x64 tile shape
    uint32_t g1_mip = (mip >= 2u) ? (mip - 2u) : 0u;
    if (g1_mip < g1_std_mips) {
        int g1_w = static_cast<int>(1024u >> g1_mip);
        int g1_h = g1_w;
        int tw = static_cast<int>(g1_shape.WidthInTexels);   // 128
        int th = static_cast<int>(g1_shape.HeightInTexels);  // 64
        int wx = static_cast<int>(g1_tilings[g1_mip].WidthInTiles);
        int wy = static_cast<int>(g1_tilings[g1_mip].HeightInTiles);

        int x0 = static_cast<int>((tx * static_cast<uint32_t>(g1_w)) / bcn_dim) - 1;
        int x1 = static_cast<int>(((tx + 1u) * static_cast<uint32_t>(g1_w)) / bcn_dim);
        int y0 = static_cast<int>((ty * static_cast<uint32_t>(g1_h)) / bcn_dim) - 1;
        int y1 = static_cast<int>(((ty + 1u) * static_cast<uint32_t>(g1_h)) / bcn_dim);

        int tx_s = floor_div_i(x0, tw);
        int tx_e = floor_div_i(x1, tw);
        int ty_s = floor_div_i(y0, th);
        int ty_e = floor_div_i(y1, th);

        int n_tx = std::min(tx_e - tx_s + 1, wx);
        int n_ty = std::min(ty_e - ty_s + 1, wy);
        for (int dy = 0; dy < n_ty; ++dy) {
            uint32_t uy = static_cast<uint32_t>(((ty_s + dy) % wy + wy) % wy);
            for (int dx = 0; dx < n_tx; ++dx) {
                uint32_t ux = static_cast<uint32_t>(((tx_s + dx) % wx + wx) % wx);
                out_keys.push_back(LatentTileKey{1u, m, g1_mip, ux, uy}.pack());
            }
        }
    }
}

struct P2RPcieRow {
    int run_id = 0;
    std::string config_name;
    uint32_t batch_idx = 0;
    int frame_idx = 0;
    uint32_t dma_tiles_64kb = 0;
    double upload_kb = 0.0;
    double copy_gpu_latency_us = 0.0;
    double cpu_submit_us = 0.0;
    double bandwidth_gbps = 0.0;
};

struct P2RFrameExtRow {
    int run_id = 0;
    std::string config_name;
    int frame_idx = 0;
    double display_frame_ms = 0.0;
    double cpu_work_ms = 0.0;
    double gpu_frame_ms = 0.0;
    double present_call_ms = 0.0;
    uint32_t visible_cells = 0;
    double bcn_hit_rate = 1.0;
    double l1_hit_rate = 1.0;
    double l2_hit_rate = 1.0;
    double deadline_miss_rate = 0.0;
    double mean_mip_deficit = 0.0;
    uint32_t popin_transitions = 0;
    uint32_t l1_evictions = 0;
    uint32_t bcn_evictions = 0;
};

struct P2RMemoryBreakdownRow {
    int run_id = 0;
    std::string config_name;
    std::string storage_mode;
    std::string scheduling_mode;
    uint64_t copy_queue_freq_hz = 0;
    double bcn_cache_heap_mib = 0.0;
    double bcn_tail_heap_mib = 0.0;
    double l1_latent_heap_mib = 0.0;
    double latent_tail_heap_mib = 0.0;
    double committed_latent_textures_mib = 0.0;
    double vram_staging_buffers_mib = 0.0;
    double render_targets_feedback_mib = 0.0;
    double explicit_local_vram_sum_mib = 0.0;
    double dxgi_peak_local_vram_mib = 0.0;
    double driver_overhead_mib = 0.0;
    double dxgi_peak_nonlocal_mib = 0.0;
    double upload_readback_buffers_mib = 0.0;
    double ram_arena_committed_mib = 0.0;
    double ram_working_set_populated_mib = 0.0;
    double vram_l1_populated_peak_mib = 0.0;
    double bcn_cache_populated_peak_mib = 0.0;
};

void append_csv_pcie(const std::string& path, const std::vector<P2RPcieRow>& rows) {
    namespace fs = std::filesystem;
    fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    bool hdr = !fs::exists(p) || fs::file_size(p) == 0;
    std::ofstream ofs(path, std::ios::app);
    if (hdr) ofs << "run_id,config_name,batch_idx,frame_idx,dma_tiles_64kb,upload_kb,copy_gpu_latency_us,cpu_submit_us,bandwidth_gbps\n";
    ofs << std::fixed << std::setprecision(3);
    for (const auto& r : rows) {
        ofs << r.run_id << "," << r.config_name << "," << r.batch_idx << "," << r.frame_idx << ","
            << r.dma_tiles_64kb << "," << r.upload_kb << "," << r.copy_gpu_latency_us << ","
            << r.cpu_submit_us << "," << r.bandwidth_gbps << "\n";
    }
}

void append_csv_frame_ext(const std::string& path, const std::vector<P2RFrameExtRow>& rows) {
    namespace fs = std::filesystem;
    fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    bool hdr = !fs::exists(p) || fs::file_size(p) == 0;
    std::ofstream ofs(path, std::ios::app);
    if (hdr) {
        ofs << "run_id,config_name,frame_idx,display_frame_ms,cpu_work_ms,gpu_frame_ms,present_call_ms,"
               "visible_cells,bcn_hit_rate,l1_hit_rate,l2_hit_rate,deadline_miss_rate,mean_mip_deficit,"
               "popin_transitions,l1_evictions,bcn_evictions\n";
    }
    ofs << std::fixed << std::setprecision(5);
    for (const auto& r : rows) {
        ofs << r.run_id << "," << r.config_name << "," << r.frame_idx << ","
            << r.display_frame_ms << "," << r.cpu_work_ms << "," << r.gpu_frame_ms << ","
            << r.present_call_ms << "," << r.visible_cells << "," << r.bcn_hit_rate << ","
            << r.l1_hit_rate << "," << r.l2_hit_rate << "," << r.deadline_miss_rate << ","
            << r.mean_mip_deficit << "," << r.popin_transitions << "," << r.l1_evictions << ","
            << r.bcn_evictions << "\n";
    }
}

void append_csv_mem_breakdown(const std::string& path, const P2RMemoryBreakdownRow& r) {
    namespace fs = std::filesystem;
    fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    bool hdr = !fs::exists(p) || fs::file_size(p) == 0;
    std::ofstream ofs(path, std::ios::app);
    if (hdr) {
        ofs << "run_id,config_name,storage_mode,scheduling_mode,copy_queue_freq_hz,"
               "bcn_cache_heap_mib,bcn_tail_heap_mib,l1_latent_heap_mib,latent_tail_heap_mib,"
               "committed_latent_textures_mib,vram_staging_buffers_mib,render_targets_feedback_mib,"
               "explicit_local_vram_sum_mib,dxgi_peak_local_vram_mib,driver_overhead_mib,"
               "dxgi_peak_nonlocal_mib,upload_readback_buffers_mib,ram_arena_committed_mib,"
               "ram_working_set_populated_mib,vram_l1_populated_peak_mib,bcn_cache_populated_peak_mib\n";
    }
    ofs << std::fixed << std::setprecision(3);
    ofs << r.run_id << "," << r.config_name << "," << r.storage_mode << "," << r.scheduling_mode << ","
        << r.copy_queue_freq_hz << "," << r.bcn_cache_heap_mib << "," << r.bcn_tail_heap_mib << ","
        << r.l1_latent_heap_mib << "," << r.latent_tail_heap_mib << "," << r.committed_latent_textures_mib << ","
        << r.vram_staging_buffers_mib << "," << r.render_targets_feedback_mib << "," << r.explicit_local_vram_sum_mib << ","
        << r.dxgi_peak_local_vram_mib << "," << r.driver_overhead_mib << "," << r.dxgi_peak_nonlocal_mib << ","
        << r.upload_readback_buffers_mib << "," << r.ram_arena_committed_mib << "," << r.ram_working_set_populated_mib << ","
        << r.vram_l1_populated_peak_mib << "," << r.bcn_cache_populated_peak_mib << "\n";
}

} // namespace

void Phase2ReinforcedRunner::run(
    DX12Context& ctx,
    const SceneGeometry& scene,
    TelemetryLogger& logger,
    const Phase2ReinforcedConfig& cfg
) {
    ID3D12Device8* dev = ctx.device();
    auto& srv_heap = ctx.cbv_srv_uav_heap();

    const bool is_b1_sfs = (cfg.mode == P2RMode::B1_SFS_BCn_Async_RamCache);
    const bool use_ram_tier = (cfg.mode == P2RMode::C1_RAM_Sync || cfg.mode == P2RMode::C3_RAM_Async);
    const bool use_async_batch = (cfg.mode == P2RMode::C2_VRAM_Async || cfg.mode == P2RMode::C3_RAM_Async || is_b1_sfs);

    uint64_t vram_staging_bytes = 0;
    uint64_t rt_fb_alloc_bytes = 0;
    uint64_t committed_latent_bytes = 0;
    uint64_t upload_rb_bytes = 0;

    // Track backbuffers & depth buffer via GetResourceAllocationInfo (without touching DX12Context internals)
    {
        D3D12_RESOURCE_DESC rtd{};
        rtd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rtd.Width = cfg.width;
        rtd.Height = cfg.height;
        rtd.DepthOrArraySize = 1;
        rtd.MipLevels = 1;
        rtd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rtd.SampleDesc.Count = 1;
        rtd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        rt_fb_alloc_bytes += dev->GetResourceAllocationInfo(0, 1, &rtd).SizeInBytes * kFrameInFlightCount;

        rtd.Format = DXGI_FORMAT_D32_FLOAT;
        rtd.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        rt_fb_alloc_bytes += dev->GetResourceAllocationInfo(0, 1, &rtd).SizeInBytes;
    }

    // 1. Geometry VB/IB
    const auto& verts = scene.vertices();
    const auto& inds = scene.indices();
    uint64_t vb_bytes = verts.size() * sizeof(Vertex);
    uint64_t ib_bytes = inds.size() * sizeof(uint32_t);
    vram_staging_bytes += vb_bytes + ib_bytes;
    auto vb = create_default_buffer(dev, vb_bytes);
    auto ib = create_default_buffer(dev, ib_bytes);
    {
        auto geom_up = create_upload_buffer(dev, vb_bytes + ib_bytes);
        uint8_t* p = nullptr;
        throw_if_failed(geom_up->Map(0, nullptr, reinterpret_cast<void**>(&p)), "Map geom_up");
        std::memcpy(p, verts.data(), vb_bytes);
        std::memcpy(p + vb_bytes, inds.data(), ib_bytes);
        geom_up->Unmap(0, nullptr);

        ComPtr<ID3D12CommandAllocator> init_alloc;
        throw_if_failed(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                    IID_ID3D12CommandAllocator, init_alloc.put_void()), "init_alloc");
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

    ComPtr<ID3D12CommandAllocator> init_alloc;
    throw_if_failed(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                IID_ID3D12CommandAllocator, init_alloc.put_void()), "init_alloc");

    ComPtr<ID3D12Resource> frame_cbs[kFrameInFlightCount];
    uint8_t* frame_cb_ptrs[kFrameInFlightCount]{};
    for (uint32_t i = 0; i < kFrameInFlightCount; ++i) {
        frame_cbs[i] = create_upload_buffer(dev, 256);
        upload_rb_bytes += 256;
        throw_if_failed(frame_cbs[i]->Map(0, nullptr, reinterpret_cast<void**>(&frame_cb_ptrs[i])), "Map CB");
    }

    // 2. Graphics & Compute Root Signatures + PSOs
    D3D12_STATIC_SAMPLER_DESC linear_samp{};
    linear_samp.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    linear_samp.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    linear_samp.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    linear_samp.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    linear_samp.MaxLOD = D3D12_FLOAT32_MAX;
    linear_samp.ShaderRegister = 0;
    linear_samp.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    ComPtr<ID3D12RootSignature> gfx_rs, cs_rs;
    ComPtr<ID3D12PipelineState> gfx_pso, cs_pso;
    {
        D3D12_DESCRIPTOR_RANGE ranges[4]{};
        ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kNumMaterials, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
        ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kNumMaterials, 0, 1, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
        ranges[2] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kNumMaterials, 0, 2, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
        ranges[3] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, kNumMaterials, 0, 3, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};

        D3D12_ROOT_PARAMETER params[6]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[0].Descriptor.ShaderRegister = 0;
        params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        for (int r = 0; r < 4; ++r) {
            params[1 + r].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            params[1 + r].DescriptorTable = {1, &ranges[r]};
            params[1 + r].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        }
        params[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[5].Descriptor.ShaderRegister = 0;
        params[5].Descriptor.RegisterSpace = 4;
        params[5].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

        D3D12_ROOT_SIGNATURE_DESC rsd{6, params, 1, &linear_samp, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
        ComPtr<ID3DBlob> sig, err;
        throw_if_failed(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, sig.put(), err.put()), "Serialize Gfx RS");
        throw_if_failed(dev->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                                 IID_ID3D12RootSignature, gfx_rs.put_void()), "Create Gfx RS");

        auto vs_bin = read_binary_file(cfg.shader_dir + "/pbr_vs.cso");
        auto ps_bin = read_binary_file(cfg.shader_dir + "/pbr_bcn.ps.cso");
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
        throw_if_failed(dev->CreateGraphicsPipelineState(&pso, IID_ID3D12PipelineState, gfx_pso.put_void()), "Create Gfx PSO");

        if (!is_b1_sfs) {
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
            throw_if_failed(D3D12SerializeRootSignature(&cs_rsd, D3D_ROOT_SIGNATURE_VERSION_1, cs_sig.put(), cs_err.put()), "Serialize CS RS");
            throw_if_failed(dev->CreateRootSignature(0, cs_sig->GetBufferPointer(), cs_sig->GetBufferSize(),
                                                     IID_ID3D12RootSignature, cs_rs.put_void()), "Create CS RS");
            auto cs_bin = read_binary_file(cfg.shader_dir + "/ntc_decode_bcn.cs.cso");
            D3D12_COMPUTE_PIPELINE_STATE_DESC cpso{cs_rs.get(), {cs_bin.data(), cs_bin.size()}, 0, {}, D3D12_PIPELINE_STATE_FLAG_NONE};
            throw_if_failed(dev->CreateComputePipelineState(&cpso, IID_ID3D12PipelineState, cs_pso.put_void()), "Create CS PSO");
        }
    }

    // 3. Latent Subsystem (Committed VRAM for C0/C2 vs Reserved L1 + RAM L2 for C1/C3 vs B1_SFS_RamCache)
    std::vector<ComPtr<ID3D12Resource>> g0_tex(kNumMaterials), g1_tex(kNumMaterials);
    ComPtr<ID3D12Resource> mlp_weights_buf;
    ComPtr<ID3D12Heap> l1_latent_heap, latent_tail_heap;
    uint32_t g0_srv_base = 0, g1_srv_base = 0;
    uint64_t l1_heap_bytes = 0, latent_tail_bytes = 0;

    D3D12_PACKED_MIP_INFO g0_packed{}, g1_packed{};
    D3D12_TILE_SHAPE g0_shape{128, 128, 1}, g1_shape{128, 64, 1};
    std::vector<D3D12_SUBRESOURCE_TILING> g0_tilings(13), g1_tilings(11);

    LatentRamL2Pool ram_l2;
    BcnRamL2Pool bcn_ram_l2;
    VramL1Lru l1_lru;

    if (is_b1_sfs) {
        bcn_ram_l2.init(cfg.manifest_path, cfg.ram_l2_pool_mib);
        if (!cfg.capture_checkpoints) logger.set_sys_ram_latent_ptr(bcn_ram_l2.sys_ram_mb_ptr());
    } else if (use_ram_tier) {
        for (uint32_t m = 0; m < kNumMaterials; ++m) {
            g0_tex[m] = create_reserved_tex2d(dev, 4096, 4096, 13, DXGI_FORMAT_R8G8B8A8_UNORM);
            g1_tex[m] = create_reserved_tex2d(dev, 1024, 1024, 11, DXGI_FORMAT_R16G16B16A16_FLOAT);
        }
        UINT g0_tt = 0, g1_tt = 0, g0_sc = 13, g1_sc = 11;
        dev->GetResourceTiling(g0_tex[0].get(), &g0_tt, &g0_packed, &g0_shape, &g0_sc, 0, g0_tilings.data());
        dev->GetResourceTiling(g1_tex[0].get(), &g1_tt, &g1_packed, &g1_shape, &g1_sc, 0, g1_tilings.data());

        ram_l2.init(cfg.manifest_path, cfg.ram_l2_pool_mib, g0_shape, g1_shape);
        if (!cfg.capture_checkpoints) logger.set_sys_ram_latent_ptr(ram_l2.sys_ram_latent_mb_ptr());

        vram_staging_bytes += ram_l2.mlp_blob().size();
        mlp_weights_buf = create_default_buffer(dev, ram_l2.mlp_blob().size());
        {
            auto mlp_up = create_upload_buffer(dev, ram_l2.mlp_blob().size());
            uint8_t* p = nullptr;
            throw_if_failed(mlp_up->Map(0, nullptr, reinterpret_cast<void**>(&p)), "Map mlp_up");
            std::memcpy(p, ram_l2.mlp_blob().data(), ram_l2.mlp_blob().size());
            mlp_up->Unmap(0, nullptr);
            auto* cmd = ctx.cmd_list();
            throw_if_failed(init_alloc->Reset(), "mlp alloc");
            throw_if_failed(cmd->Reset(init_alloc.get(), nullptr), "mlp cmd");
            cmd->CopyBufferRegion(mlp_weights_buf.get(), 0, mlp_up.get(), 0, ram_l2.mlp_blob().size());
            D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_FLAG_NONE};
            b.Transition = {mlp_weights_buf.get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE};
            cmd->ResourceBarrier(1, &b);
            throw_if_failed(cmd->Close(), "mlp close");
            ID3D12CommandList* l[] = {cmd};
            ctx.direct_queue()->ExecuteCommandLists(1, l);
            ctx.direct_sync().flush();
        }

        uint32_t l1_cap_tiles = (cfg.l1_latent_pool_mib * 1024u) / 64u;
        l1_lru.init(l1_cap_tiles);

        D3D12_HEAP_DESC hd{};
        l1_heap_bytes = static_cast<UINT64>(l1_cap_tiles) * kTileBytes;
        hd.SizeInBytes = l1_heap_bytes;
        hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        hd.Flags = D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES;
        throw_if_failed(dev->CreateHeap(&hd, IID_ID3D12Heap, l1_latent_heap.put_void()), "Create l1_latent_heap");

        uint32_t tail_tiles = kNumMaterials * (g0_packed.NumTilesForPackedMips + g1_packed.NumTilesForPackedMips);
        latent_tail_bytes = static_cast<UINT64>(std::max(1u, tail_tiles)) * kTileBytes;
        hd.SizeInBytes = latent_tail_bytes;
        throw_if_failed(dev->CreateHeap(&hd, IID_ID3D12Heap, latent_tail_heap.put_void()), "Create latent_tail_heap");

        auto ltail_up = create_upload_buffer(dev, 4ULL * 1024ULL * 1024ULL);
        uint8_t* ltail_ptr = nullptr;
        throw_if_failed(ltail_up->Map(0, nullptr, reinterpret_cast<void**>(&ltail_ptr)), "Map ltail_up");
        uint32_t heap_cursor = 0;
        for (uint32_t m = 0; m < kNumMaterials; ++m) {
            if (g0_packed.NumTilesForPackedMips > 0) {
                D3D12_TILED_RESOURCE_COORDINATE c{0, 0, 0, g0_packed.NumStandardMips};
                D3D12_TILE_REGION_SIZE r{g0_packed.NumTilesForPackedMips, FALSE, 0, 0, 0};
                D3D12_TILE_RANGE_FLAGS rf = D3D12_TILE_RANGE_FLAG_NONE;
                UINT rc = g0_packed.NumTilesForPackedMips;
                ctx.direct_queue()->UpdateTileMappings(g0_tex[m].get(), 1, &c, &r, latent_tail_heap.get(),
                                                       1, &rf, &heap_cursor, &rc, D3D12_TILE_MAPPING_FLAG_NONE);
                heap_cursor += rc;
                upload_texture_mips_with_staging(
                    ctx, init_alloc.get(), g0_tex[m].get(), ltail_up.get(), ltail_ptr,
                    DXGI_FORMAT_R8G8B8A8_UNORM, 4096, 4096, static_cast<uint16_t>(g0_packed.NumStandardMips), 12,
                    [&](uint16_t mip, std::vector<uint8_t>& out) { ram_l2.build_g0_mip_from_l3(m, mip, out); },
                    D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE
                );
            }
            if (g1_packed.NumTilesForPackedMips > 0) {
                D3D12_TILED_RESOURCE_COORDINATE c{0, 0, 0, g1_packed.NumStandardMips};
                D3D12_TILE_REGION_SIZE r{g1_packed.NumTilesForPackedMips, FALSE, 0, 0, 0};
                D3D12_TILE_RANGE_FLAGS rf = D3D12_TILE_RANGE_FLAG_NONE;
                UINT rc = g1_packed.NumTilesForPackedMips;
                ctx.direct_queue()->UpdateTileMappings(g1_tex[m].get(), 1, &c, &r, latent_tail_heap.get(),
                                                       1, &rf, &heap_cursor, &rc, D3D12_TILE_MAPPING_FLAG_NONE);
                heap_cursor += rc;
                upload_texture_mips_with_staging(
                    ctx, init_alloc.get(), g1_tex[m].get(), ltail_up.get(), ltail_ptr,
                    DXGI_FORMAT_R16G16B16A16_FLOAT, 1024, 1024, static_cast<uint16_t>(g1_packed.NumStandardMips), 10,
                    [&](uint16_t mip, std::vector<uint8_t>& out) { ram_l2.build_g1_mip_from_l3(m, mip, out); },
                    D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE
                );
            }
        }
        ltail_up->Unmap(0, nullptr);
    } else {
        // C0_VRAM_Sync / C2_VRAM_Async: Full G0 + G1 resident as Committed Textures in Local VRAM (~3072 MiB)
        MaterialPack pack;
        pack.load(cfg.manifest_path);
        vram_staging_bytes += pack.mlp_table_bytes();
        mlp_weights_buf = create_default_buffer(dev, pack.mlp_table_bytes());
        {
            auto mlp_up = create_upload_buffer(dev, pack.mlp_table_bytes());
            uint8_t* p = nullptr;
            throw_if_failed(mlp_up->Map(0, nullptr, reinterpret_cast<void**>(&p)), "Map mlp_up");
            std::memcpy(p, pack.mlp_table_ptr(), pack.mlp_table_bytes());
            mlp_up->Unmap(0, nullptr);
            auto* cmd = ctx.cmd_list();
            throw_if_failed(init_alloc->Reset(), "mlp alloc");
            throw_if_failed(cmd->Reset(init_alloc.get(), nullptr), "mlp cmd");
            cmd->CopyBufferRegion(mlp_weights_buf.get(), 0, mlp_up.get(), 0, pack.mlp_table_bytes());
            D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_FLAG_NONE};
            b.Transition = {mlp_weights_buf.get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE};
            cmd->ResourceBarrier(1, &b);
            throw_if_failed(cmd->Close(), "mlp close");
            ID3D12CommandList* l[] = {cmd};
            ctx.direct_queue()->ExecuteCommandLists(1, l);
            ctx.direct_sync().flush();
        }
        auto ntc_staging = create_upload_buffer(dev, 128ULL * 1024ULL * 1024ULL);
        uint8_t* ntc_ptr = nullptr;
        throw_if_failed(ntc_staging->Map(0, nullptr, reinterpret_cast<void**>(&ntc_ptr)), "Map ntc_staging");
        for (uint32_t m = 0; m < kNumMaterials; ++m) {
            g0_tex[m] = create_texture2d(dev, 4096, 4096, 13, DXGI_FORMAT_R8G8B8A8_UNORM,
                                         D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &committed_latent_bytes);
            upload_texture_mips_with_staging(
                ctx, init_alloc.get(), g0_tex[m].get(), ntc_staging.get(), ntc_ptr,
                DXGI_FORMAT_R8G8B8A8_UNORM, 4096, 4096, 0, 12,
                [&](uint16_t mip, std::vector<uint8_t>& out) { pack.build_g0_mip_tiled(m, mip, out); },
                D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE
            );
            g1_tex[m] = create_texture2d(dev, 1024, 1024, 11, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                         D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &committed_latent_bytes);
            upload_texture_mips_with_staging(
                ctx, init_alloc.get(), g1_tex[m].get(), ntc_staging.get(), ntc_ptr,
                DXGI_FORMAT_R16G16B16A16_FLOAT, 1024, 1024, 0, 10,
                [&](uint16_t mip, std::vector<uint8_t>& out) { pack.build_g1_mip_tiled(m, mip, out); },
                D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE
            );
        }
        ntc_staging->Unmap(0, nullptr);
    }

    if (!is_b1_sfs) {
        g0_srv_base = srv_heap.alloc(kNumMaterials);
        g1_srv_base = srv_heap.alloc(kNumMaterials);
        for (uint32_t m = 0; m < kNumMaterials; ++m) {
            D3D12_SHADER_RESOURCE_VIEW_DESC sd0{DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
            sd0.Texture2D.MipLevels = 13;
            dev->CreateShaderResourceView(g0_tex[m].get(), &sd0, srv_heap.cpu(g0_srv_base + m));
            D3D12_SHADER_RESOURCE_VIEW_DESC sd1{DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
            sd1.Texture2D.MipLevels = 11;
            dev->CreateShaderResourceView(g1_tex[m].get(), &sd1, srv_heap.cpu(g1_srv_base + m));
        }
    }

    // 4. BCn Cache Heap (1536 MiB) + Tail Heap + Sampler Feedback
    uint32_t max_bcn_triplets = (cfg.bcn_tile_cache_mib * 1024u) / (3u * 64u);
    uint64_t bcn_cache_heap_bytes = static_cast<UINT64>(max_bcn_triplets) * 3ULL * kTileBytes;
    uint64_t bcn_tail_heap_bytes = static_cast<UINT64>(kNumMaterials) * 3ULL * kTileBytes;
    ComPtr<ID3D12Heap> bcn_tail_heap, bcn_cache_heap;
    {
        D3D12_HEAP_DESC hd{};
        hd.SizeInBytes = bcn_tail_heap_bytes;
        hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        hd.Flags = D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES;
        throw_if_failed(dev->CreateHeap(&hd, IID_ID3D12Heap, bcn_tail_heap.put_void()), "Create bcn_tail_heap");

        hd.SizeInBytes = bcn_cache_heap_bytes;
        throw_if_failed(dev->CreateHeap(&hd, IID_ID3D12Heap, bcn_cache_heap.put_void()), "Create bcn_cache_heap");
    }

    std::vector<ComPtr<ID3D12Resource>> alb_tex(kNumMaterials), nrm_tex(kNumMaterials), orm_tex(kNumMaterials);
    std::vector<ComPtr<ID3D12Resource>> fb_tex(kNumMaterials), fb_resolved(kNumMaterials);
    uint32_t alb_srv_base = srv_heap.alloc(kNumMaterials);
    uint32_t nrm_srv_base = srv_heap.alloc(kNumMaterials);
    uint32_t orm_srv_base = srv_heap.alloc(kNumMaterials);
    uint32_t fb_uav_base = srv_heap.alloc(kNumMaterials);
    uint32_t fb_cpu_uav_base = ctx.cpu_uav_heap().alloc(kNumMaterials);

    {
        MaterialPack tail_pack;
        tail_pack.load(cfg.manifest_path);
        auto tail_up = create_upload_buffer(dev, 4ULL * 1024ULL * 1024ULL);
        uint8_t* tail_ptr = nullptr;
        throw_if_failed(tail_up->Map(0, nullptr, reinterpret_cast<void**>(&tail_ptr)), "Map tail_up");

        for (uint32_t m = 0; m < kNumMaterials; ++m) {
            alb_tex[m] = create_reserved_tex2d(dev, kVirtualTexSize, kVirtualTexSize, kBcnMipLevels, DXGI_FORMAT_BC7_UNORM);
            nrm_tex[m] = create_reserved_tex2d(dev, kVirtualTexSize, kVirtualTexSize, kBcnMipLevels, DXGI_FORMAT_BC5_UNORM);
            orm_tex[m] = create_reserved_tex2d(dev, kVirtualTexSize, kVirtualTexSize, kBcnMipLevels, DXGI_FORMAT_BC7_UNORM);

            UINT tt = 0, sc = kBcnMipLevels;
            D3D12_PACKED_MIP_INFO pi{};
            D3D12_TILE_SHAPE ts{};
            std::vector<D3D12_SUBRESOURCE_TILING> st(sc);
            dev->GetResourceTiling(alb_tex[m].get(), &tt, &pi, &ts, &sc, 0, st.data());
            D3D12_TILED_RESOURCE_COORDINATE coord{0, 0, 0, pi.NumStandardMips};
            D3D12_TILE_REGION_SIZE region{pi.NumTilesForPackedMips, FALSE, 0, 0, 0};
            D3D12_TILE_RANGE_FLAGS rf = D3D12_TILE_RANGE_FLAG_NONE;
            UINT rc = pi.NumTilesForPackedMips;

            ID3D12Resource* pair3[3] = {alb_tex[m].get(), nrm_tex[m].get(), orm_tex[m].get()};
            DXGI_FORMAT fmts3[3] = {DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_BC5_UNORM, DXGI_FORMAT_BC7_UNORM};
            for (int k = 0; k < 3; ++k) {
                UINT hoff = (m * 3u + static_cast<uint32_t>(k)) * pi.NumTilesForPackedMips;
                ctx.direct_queue()->UpdateTileMappings(pair3[k], 1, &coord, &region, bcn_tail_heap.get(),
                                                       1, &rf, &hoff, &rc, D3D12_TILE_MAPPING_FLAG_NONE);
                upload_texture_mips_with_staging(
                    ctx, init_alloc.get(), pair3[k], tail_up.get(), tail_ptr,
                    fmts3[k], kVirtualTexSize, kVirtualTexSize,
                    static_cast<uint16_t>(pi.NumStandardMips), kBcnMipLevels - 1,
                    [&](uint16_t mip, std::vector<uint8_t>& out) { tail_pack.build_bcn_mip_tiled(m, mip, k, out); },
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                );
            }

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
        tail_up->Unmap(0, nullptr);
    }

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
        rt_fb_alloc_bytes += dev->GetResourceAllocationInfo2(0, 1, &fbd, nullptr).SizeInBytes;
        throw_if_failed(
            dev->CreateCommittedResource2(&def_hp, D3D12_HEAP_FLAG_NONE, &fbd,
                                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, nullptr,
                                          IID_ID3D12Resource, fb_tex[m].put_void()),
            "CreateCommittedResource2 feedback failed"
        );
        dev->CreateSamplerFeedbackUnorderedAccessView(alb_tex[m].get(), fb_tex[m].get(), srv_heap.cpu(fb_uav_base + m));
        dev->CreateSamplerFeedbackUnorderedAccessView(alb_tex[m].get(), fb_tex[m].get(), ctx.cpu_uav_heap().cpu(fb_cpu_uav_base + m));
        fb_resolved[m] = create_texture2d(dev, 32, 32, 1, DXGI_FORMAT_R8_UINT,
                                          D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &rt_fb_alloc_bytes);
    }

    ComPtr<ID3D12Resource> fb_readback[kFrameInFlightCount];
    ComPtr<ID3D12Resource> residency_upload[kFrameInFlightCount];
    uint8_t* residency_up_ptr[kFrameInFlightCount]{};
    uint64_t res_bytes = kNumMaterials * 1024ULL * sizeof(uint32_t);
    vram_staging_bytes += res_bytes;
    ComPtr<ID3D12Resource> residency_buf = create_default_buffer(dev, res_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    BcnTileLru bcn_lru;
    bcn_lru.init(max_bcn_triplets);
    for (uint32_t i = 0; i < kFrameInFlightCount; ++i) {
        fb_readback[i] = create_readback_buffer(dev, kNumMaterials * 32ULL * 256ULL);
        upload_rb_bytes += kNumMaterials * 32ULL * 256ULL;
        residency_upload[i] = create_upload_buffer(dev, res_bytes);
        upload_rb_bytes += res_bytes;
        throw_if_failed(residency_upload[i]->Map(0, nullptr, reinterpret_cast<void**>(&residency_up_ptr[i])), "Map residency_upload");
        std::memcpy(residency_up_ptr[i], bcn_lru.residency_map_data(), res_bytes);
    }

    // 5. Async Copy & Compute Staging Resources (allocated ONLY for the active mode)
    uint64_t bcn_stage_bytes = static_cast<uint64_t>(cfg.max_tile_triplets_per_frame) * 3ULL * kTileBytes;
    vram_staging_bytes += bcn_stage_bytes;
    ComPtr<ID3D12Resource> vram_bcn_staging = create_default_buffer(
        dev, bcn_stage_bytes,
        is_b1_sfs ? D3D12_RESOURCE_FLAG_NONE : D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    );

    uint64_t copy_stage_bytes = is_b1_sfs
        ? bcn_stage_bytes
        : (use_ram_tier ? static_cast<uint64_t>(kMaxL1BatchTiles) * kTileBytes : 0ULL);
    ComPtr<ID3D12Resource> vram_l1_staging;
    ComPtr<ID3D12Resource> copy_upload_ring[kFrameInFlightCount];
    uint8_t* copy_up_ptr[kFrameInFlightCount]{};
    ComPtr<ID3D12CommandAllocator> copy_alloc[kFrameInFlightCount];
    ComPtr<ID3D12GraphicsCommandList> copy_cmd;
    ComPtr<ID3D12QueryHeap> copy_ts_heap;
    ComPtr<ID3D12Resource> copy_ts_readback;

    if (copy_stage_bytes > 0) {
        if (!is_b1_sfs) {
            vram_staging_bytes += copy_stage_bytes;
            vram_l1_staging = create_default_buffer(dev, copy_stage_bytes);
        }
        D3D12_QUERY_HEAP_DESC qhd{D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP, kFrameInFlightCount * 2, 0};
        throw_if_failed(dev->CreateQueryHeap(&qhd, IID_ID3D12QueryHeap, copy_ts_heap.put_void()), "Create copy_ts_heap");
        copy_ts_readback = create_readback_buffer(dev, sizeof(uint64_t) * kFrameInFlightCount * 2);
        upload_rb_bytes += sizeof(uint64_t) * kFrameInFlightCount * 2;
        for (uint32_t i = 0; i < kFrameInFlightCount; ++i) {
            copy_upload_ring[i] = create_upload_buffer(dev, copy_stage_bytes);
            upload_rb_bytes += copy_stage_bytes;
            throw_if_failed(copy_upload_ring[i]->Map(0, nullptr, reinterpret_cast<void**>(&copy_up_ptr[i])), "Map copy_upload_ring");
            throw_if_failed(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_ID3D12CommandAllocator, copy_alloc[i].put_void()), "copy_alloc");
        }
        throw_if_failed(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, copy_alloc[0].get(), nullptr,
                                               IID_ID3D12GraphicsCommandList, copy_cmd.put_void()), "copy_cmd");
        copy_cmd->Close();
    }

    ComPtr<ID3D12Resource> jobs_default, jobs_upload[kFrameInFlightCount];
    uint8_t* jobs_up_ptr[kFrameInFlightCount]{};
    ComPtr<ID3D12CommandAllocator> cs_alloc[kFrameInFlightCount];
    ComPtr<ID3D12GraphicsCommandList> cs_cmd;
    uint32_t decode_uav_slot = 0;

    if (!is_b1_sfs) {
        uint64_t jobs_bytes = static_cast<uint64_t>(cfg.max_tile_triplets_per_frame) * sizeof(TileDecodeJob);
        vram_staging_bytes += jobs_bytes;
        jobs_default = create_default_buffer(dev, jobs_bytes);
        for (uint32_t i = 0; i < kFrameInFlightCount; ++i) {
            jobs_upload[i] = create_upload_buffer(dev, jobs_bytes);
            upload_rb_bytes += jobs_bytes;
            throw_if_failed(jobs_upload[i]->Map(0, nullptr, reinterpret_cast<void**>(&jobs_up_ptr[i])), "Map jobs_upload");
            throw_if_failed(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_ID3D12CommandAllocator, cs_alloc[i].put_void()), "cs_alloc");
        }
        throw_if_failed(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, cs_alloc[0].get(), nullptr,
                                               IID_ID3D12GraphicsCommandList, cs_cmd.put_void()), "cs_cmd");
        cs_cmd->Close();

        decode_uav_slot = srv_heap.alloc(1);
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavd{};
        uavd.Format = DXGI_FORMAT_R32_TYPELESS;
        uavd.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavd.Buffer.NumElements = static_cast<UINT>(bcn_stage_bytes / 4ULL);
        uavd.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        dev->CreateUnorderedAccessView(vram_bcn_staging.get(), nullptr, &uavd, srv_heap.cpu(decode_uav_slot));
    }

    uint64_t verified_copy_freq_hz = 0;
    ctx.copy_queue()->GetTimestampFrequency(&verified_copy_freq_hz);

    // 6. Pipeline State & Render Loop
    struct InflightCopy {
        bool active = false;
        bool has_dma = false;
        uint64_t fence_val = 0;
        uint32_t copy_slot = 0;
        int submit_frame_idx = 0;
        double cpu_submit_us = 0.0;
        std::vector<uint32_t> l1_keys;
        std::vector<VramL1Lru::AllocResult> l1_allocs;
        std::vector<uint32_t> bcn_keys;
        std::vector<BcnTileLru::AllocResult> bcn_allocs;
    } inflight_copy;

    struct ReadyDecode {
        bool pending = false;
        std::vector<uint32_t> bcn_keys;
        std::vector<BcnTileLru::AllocResult> bcn_allocs;
    } ready_for_decode;

    struct InflightDecode {
        bool active = false;
        uint64_t fence_val = 0;
        std::vector<uint32_t> bcn_keys;
        std::vector<BcnTileLru::AllocResult> bcn_allocs;
    } inflight_decode;

    uint64_t last_l1_commit_direct_fence = 0;

    std::vector<P2RPcieRow> pcie_rows;
    std::vector<P2RFrameExtRow> ext_rows;
    ext_rows.reserve(static_cast<size_t>(cfg.measured_frames));
    uint32_t pcie_batch_counter = 0;
    TileMapperHelper tile_mapper;

    int total_frames = cfg.warmup_frames + cfg.measured_frames;
    int slot_last_frame_idx[kFrameInFlightCount] = {-1, -1, -1};
    bool fb_slot_valid[kFrameInFlightCount] = {false, false, false};
    uint64_t fb_slot_fence[kFrameInFlightCount] = {0, 0, 0};

    // Track per-cell previous resident mip for visible pop-in transition counting
    std::vector<uint8_t> prev_visible_res_mip(kNumMaterials * 1024u, 6u);
    std::vector<uint8_t> last_fb_snapshot(kNumMaterials * 1024u, 0xFFu);

    LARGE_INTEGER qpc_freq_li{};
    QueryPerformanceFrequency(&qpc_freq_li);
    double qpc_freq = static_cast<double>(qpc_freq_li.QuadPart);
    LARGE_INTEGER prev_present_qpc{};
    QueryPerformanceCounter(&prev_present_qpc);

    const float aspect = static_cast<float>(cfg.width) / static_cast<float>(cfg.height);
    const auto& bays = scene.bays();
    const auto& checkpoints = scene.quality_checkpoints();

    double peak_local_vram_mb = 0.0, peak_nonlocal_vram_mb = 0.0;
    uint32_t peak_l1_used_slots = 0, peak_bcn_used_slots = 0;

    if (!cfg.capture_checkpoints) {
        logger.start_run(cfg.run_id, cfg.config_name, static_cast<size_t>(cfg.measured_frames));
    }

    auto record_copy_ts = [&](uint32_t copy_slot, int sub_fidx, uint32_t nt, double sub_us) {
        if (cfg.capture_checkpoints || sub_fidx < 0 || nt == 0) return;
        uint64_t* ts_ptr = nullptr;
        D3D12_RANGE rng{copy_slot * 2 * sizeof(uint64_t), (copy_slot * 2 + 2) * sizeof(uint64_t)};
        if (SUCCEEDED(copy_ts_readback->Map(0, &rng, reinterpret_cast<void**>(&ts_ptr)))) {
            uint64_t t0 = ts_ptr[copy_slot * 2];
            uint64_t t1 = ts_ptr[copy_slot * 2 + 1];
            D3D12_RANGE empty{0, 0};
            copy_ts_readback->Unmap(0, &empty);
            double freq = static_cast<double>(verified_copy_freq_hz);
            double lat_us = (t1 > t0 && freq > 0.0) ? (static_cast<double>(t1 - t0) * 1e6 / freq) : 0.0;
            double kb = static_cast<double>(nt) * 64.0;
            double gbps = (lat_us > 0.01) ? ((kb * 1024.0 / 1e9) / (lat_us / 1e6)) : 0.0;
            pcie_rows.push_back({cfg.run_id, cfg.config_name, pcie_batch_counter++, sub_fidx, nt, kb, lat_us, sub_us, gbps});
        }
    };

    auto commit_bcn_tiles = [&](ID3D12GraphicsCommandList7* cmd, uint32_t slot,
                                const std::vector<uint32_t>& bkeys,
                                const std::vector<BcnTileLru::AllocResult>& ballocs,
                                uint32_t& out_evictions) {
        uint32_t n_jobs = static_cast<uint32_t>(bkeys.size());
        std::vector<D3D12_RESOURCE_BARRIER> copy_barriers;
        copy_barriers.reserve(n_jobs * 3);
        bool tex_trans[kNumMaterials]{};
        for (uint32_t j = 0; j < n_jobs; ++j) {
            const auto& al = ballocs[j];
            if (al.evicted) {
                ++out_evictions;
                uint32_t em = (al.evicted_key >> 16) & 0x1Fu;
                uint32_t emip = (al.evicted_key >> 12) & 0xFu;
                uint32_t ety = (al.evicted_key >> 6) & 0x3Fu;
                uint32_t etx = al.evicted_key & 0x3Fu;
                tile_mapper.add_unmap(alb_tex[em].get(), emip, etx, ety);
                tile_mapper.add_unmap(nrm_tex[em].get(), emip, etx, ety);
                tile_mapper.add_unmap(orm_tex[em].get(), emip, etx, ety);
            }
            uint32_t key = bkeys[j];
            uint32_t m = (key >> 16) & 0x1Fu;
            uint32_t mip = (key >> 12) & 0xFu;
            uint32_t ty = (key >> 6) & 0x3Fu;
            uint32_t tx = key & 0x3Fu;
            tile_mapper.add_map(alb_tex[m].get(), mip, tx, ty, al.slot * 3u + 0u);
            tile_mapper.add_map(nrm_tex[m].get(), mip, tx, ty, al.slot * 3u + 1u);
            tile_mapper.add_map(orm_tex[m].get(), mip, tx, ty, al.slot * 3u + 2u);
            if (!tex_trans[m]) {
                tex_trans[m] = true;
                ID3D12Resource* r3[3] = {alb_tex[m].get(), nrm_tex[m].get(), orm_tex[m].get()};
                for (int k = 0; k < 3; ++k) {
                    D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_FLAG_NONE};
                    b.Transition = {r3[k], D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST};
                    copy_barriers.push_back(b);
                }
            }
        }
        tile_mapper.flush(ctx.direct_queue(), bcn_cache_heap.get(), use_async_batch);
        if (!copy_barriers.empty()) cmd->ResourceBarrier(static_cast<UINT>(copy_barriers.size()), copy_barriers.data());
        for (uint32_t j = 0; j < n_jobs; ++j) {
            uint32_t key = bkeys[j];
            uint32_t m = (key >> 16) & 0x1Fu;
            uint32_t mip = (key >> 12) & 0xFu;
            uint32_t ty = (key >> 6) & 0x3Fu;
            uint32_t tx = key & 0x3Fu;
            D3D12_TILED_RESOURCE_COORDINATE nc{tx, ty, 0, mip};
            D3D12_TILE_REGION_SIZE nr{1, FALSE, 0, 0, 0};
            UINT64 base_off = static_cast<UINT64>(j) * 3ULL * kTileBytes;
            cmd->CopyTiles(alb_tex[m].get(), &nc, &nr, vram_bcn_staging.get(), base_off, D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE);
            cmd->CopyTiles(nrm_tex[m].get(), &nc, &nr, vram_bcn_staging.get(), base_off + kTileBytes, D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE);
            cmd->CopyTiles(orm_tex[m].get(), &nc, &nr, vram_bcn_staging.get(), base_off + 2ULL * kTileBytes, D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE);
            bcn_lru.mark_resident(key, true);
        }
        for (auto& b : copy_barriers) std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
        if (!copy_barriers.empty()) cmd->ResourceBarrier(static_cast<UINT>(copy_barriers.size()), copy_barriers.data());

        std::memcpy(residency_up_ptr[slot], bcn_lru.residency_map_data(), res_bytes);
        D3D12_RESOURCE_BARRIER rb{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_FLAG_NONE};
        rb.Transition = {residency_buf.get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                         D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST};
        cmd->ResourceBarrier(1, &rb);
        cmd->CopyBufferRegion(residency_buf.get(), 0, residency_upload[slot].get(), 0, res_bytes);
        std::swap(rb.Transition.StateBefore, rb.Transition.StateAfter);
        cmd->ResourceBarrier(1, &rb);
        peak_bcn_used_slots = std::max(peak_bcn_used_slots, bcn_lru.used_slots());
    };

    auto commit_l1_tiles = [&](ID3D12GraphicsCommandList7* cmd,
                               const std::vector<uint32_t>& lkeys,
                               const std::vector<VramL1Lru::AllocResult>& lallocs,
                               uint32_t& out_evictions) {
        uint32_t nt = static_cast<uint32_t>(lkeys.size());
        std::vector<D3D12_RESOURCE_BARRIER> l1_barriers;
        bool g0_trans[kNumMaterials]{}, g1_trans[kNumMaterials]{};
        for (uint32_t j = 0; j < nt; ++j) {
            const auto& al = lallocs[j];
            if (al.evicted) {
                ++out_evictions;
                LatentTileKey ek = LatentTileKey::unpack(al.evicted_key);
                ID3D12Resource* er = ek.is_g1 ? g1_tex[ek.mat_id].get() : g0_tex[ek.mat_id].get();
                tile_mapper.add_unmap(er, ek.mip, ek.tx, ek.ty);
            }
            LatentTileKey lk = LatentTileKey::unpack(lkeys[j]);
            ID3D12Resource* nr = lk.is_g1 ? g1_tex[lk.mat_id].get() : g0_tex[lk.mat_id].get();
            tile_mapper.add_map(nr, lk.mip, lk.tx, lk.ty, al.slot);
            if (!lk.is_g1 && !g0_trans[lk.mat_id]) {
                g0_trans[lk.mat_id] = true;
                D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_FLAG_NONE};
                b.Transition = {g0_tex[lk.mat_id].get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST};
                l1_barriers.push_back(b);
            } else if (lk.is_g1 && !g1_trans[lk.mat_id]) {
                g1_trans[lk.mat_id] = true;
                D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_FLAG_NONE};
                b.Transition = {g1_tex[lk.mat_id].get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST};
                l1_barriers.push_back(b);
            }
        }
        tile_mapper.flush(ctx.direct_queue(), l1_latent_heap.get(), use_async_batch);
        if (!l1_barriers.empty()) cmd->ResourceBarrier(static_cast<UINT>(l1_barriers.size()), l1_barriers.data());
        for (uint32_t j = 0; j < nt; ++j) {
            LatentTileKey lk = LatentTileKey::unpack(lkeys[j]);
            ID3D12Resource* dst_r = lk.is_g1 ? g1_tex[lk.mat_id].get() : g0_tex[lk.mat_id].get();
            D3D12_TILED_RESOURCE_COORDINATE nc{lk.tx, lk.ty, 0, lk.mip};
            D3D12_TILE_REGION_SIZE nr{1, FALSE, 0, 0, 0};
            cmd->CopyTiles(dst_r, &nc, &nr, vram_l1_staging.get(),
                           static_cast<UINT64>(j) * kTileBytes,
                           D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE);
        }
        for (auto& b : l1_barriers) std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
        if (!l1_barriers.empty()) cmd->ResourceBarrier(static_cast<UINT>(l1_barriers.size()), l1_barriers.data());
        peak_l1_used_slots = std::max(peak_l1_used_slots, l1_lru.used_slots());
    };

    ComPtr<ID3D12CommandAllocator> sync_alloc;
    ComPtr<ID3D12GraphicsCommandList7> sync_cmd;
    throw_if_failed(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_ID3D12CommandAllocator, sync_alloc.put_void()), "sync_alloc");
    throw_if_failed(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, sync_alloc.get(), nullptr,
                                           IID_ID3D12GraphicsCommandList7, sync_cmd.put_void()), "sync_cmd");
    sync_cmd->Close();

    int prev_sync_slot = -1;

    auto process_feedback_map = [&](uint32_t fb_slot, std::vector<uint32_t>& chosen_bcn, double& out_bcn_hit) {
        uint8_t* fb_ptr = nullptr;
        D3D12_RANGE rb_rng{0, kNumMaterials * 32ULL * 256ULL};
        throw_if_failed(fb_readback[fb_slot]->Map(0, &rb_rng, reinterpret_cast<void**>(&fb_ptr)), "Map fb_readback");

        std::vector<std::vector<uint32_t>> req_by_mip(kStreamedMipCount);
        uint64_t bcn_hits = 0, bcn_total = 0;
        for (uint32_t m = 0; m < kNumMaterials; ++m) {
            const uint8_t* mat_fb = fb_ptr + static_cast<size_t>(m) * 32ULL * 256ULL;
            for (uint32_t ty = 0; ty < 32; ++ty) {
                const uint8_t* row = mat_fb + static_cast<size_t>(ty) * 256ULL;
                for (uint32_t tx = 0; tx < 32; ++tx) {
                    uint8_t min_mip = row[tx];
                    last_fb_snapshot[m * 1024u + ty * 32u + tx] = min_mip;
                    if (min_mip >= kStreamedMipCount) continue;
                    for (int k = static_cast<int>(kStreamedMipCount) - 1; k >= static_cast<int>(min_mip); --k) {
                        uint32_t uk = static_cast<uint32_t>(k);
                        uint32_t key = BcnTileKey{m, uk, tx >> uk, ty >> uk}.pack();
                        ++bcn_total;
                        if (bcn_lru.contains(key)) {
                            bcn_lru.touch(key);
                            ++bcn_hits;
                        } else {
                            req_by_mip[uk].push_back(key);
                        }
                    }
                }
            }
        }
        D3D12_RANGE empty_rng{0, 0};
        fb_readback[fb_slot]->Unmap(0, &empty_rng);
        out_bcn_hit = (bcn_total > 0) ? (static_cast<double>(bcn_hits) / static_cast<double>(bcn_total)) : 1.0;

        for (int k = static_cast<int>(kStreamedMipCount) - 1; k >= 0; --k) {
            auto& vec = req_by_mip[static_cast<size_t>(k)];
            std::sort(vec.begin(), vec.end());
            vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
            for (uint32_t key : vec) {
                if (chosen_bcn.size() >= cfg.max_tile_triplets_per_frame) break;
                if (!bcn_lru.contains(key)) chosen_bcn.push_back(key);
            }
            if (chosen_bcn.size() >= cfg.max_tile_triplets_per_frame) break;
        }
    };

    auto launch_compute_decode = [&](uint32_t slot, const std::vector<uint32_t>& bkeys, uint64_t wait_direct_val, bool sync_wait) -> uint64_t {
        uint32_t n_jobs = static_cast<uint32_t>(bkeys.size());
        auto* jobs_ptr = reinterpret_cast<TileDecodeJob*>(jobs_up_ptr[slot]);
        for (uint32_t j = 0; j < n_jobs; ++j) {
            uint32_t key = bkeys[j];
            uint32_t base_off = j * 3u * kTileBytes;
            jobs_ptr[j] = {
                (key >> 16) & 0x1Fu, (key >> 12) & 0xFu, key & 0x3Fu, (key >> 6) & 0x3Fu,
                base_off, base_off + kTileBytes, base_off + 2u * kTileBytes, 0u
            };
        }
        if (wait_direct_val > 0) ctx.compute_queue()->Wait(ctx.direct_sync().fence.get(), wait_direct_val);
        throw_if_failed(cs_alloc[slot]->Reset(), "cs_alloc reset");
        throw_if_failed(cs_cmd->Reset(cs_alloc[slot].get(), cs_pso.get()), "cs_cmd reset");
        cs_cmd->CopyBufferRegion(jobs_default.get(), 0, jobs_upload[slot].get(), 0, static_cast<UINT64>(n_jobs) * sizeof(TileDecodeJob));
        D3D12_RESOURCE_BARRIER jb{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_FLAG_NONE};
        jb.Transition = {jobs_default.get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                         D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
        cs_cmd->ResourceBarrier(1, &jb);

        ID3D12DescriptorHeap* cs_heaps[] = {srv_heap.heap()};
        cs_cmd->SetDescriptorHeaps(1, cs_heaps);
        cs_cmd->SetComputeRootSignature(cs_rs.get());
        cs_cmd->SetComputeRootDescriptorTable(0, srv_heap.gpu(g0_srv_base));
        cs_cmd->SetComputeRootDescriptorTable(1, srv_heap.gpu(g1_srv_base));
        cs_cmd->SetComputeRootShaderResourceView(2, mlp_weights_buf->GetGPUVirtualAddress());
        cs_cmd->SetComputeRootShaderResourceView(3, jobs_default->GetGPUVirtualAddress());
        cs_cmd->SetComputeRootDescriptorTable(4, srv_heap.gpu(decode_uav_slot));
        cs_cmd->Dispatch(8, 8, n_jobs);

        D3D12_RESOURCE_BARRIER end_bs[2]{};
        end_bs[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        end_bs[0].Transition = {vram_bcn_staging.get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON};
        end_bs[1] = jb;
        end_bs[1].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        end_bs[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        cs_cmd->ResourceBarrier(2, end_bs);
        throw_if_failed(cs_cmd->Close(), "cs_cmd close");
        ID3D12CommandList* cs_lists[] = {cs_cmd.get()};
        ctx.compute_queue()->ExecuteCommandLists(1, cs_lists);
        uint64_t fv = ctx.compute_sync().signal();
        if (sync_wait) ctx.compute_sync().wait(fv);
        return fv;
    };

    for (int f_iter = 0; f_iter < total_frames; ++f_iter) {
        int frame_idx = f_iter - cfg.warmup_frames;
        ctx.pump_messages();

        LARGE_INTEGER t_cpu_0{};
        QueryPerformanceCounter(&t_cpu_0);

        double frame_pcie_mb = 0.0;
        uint32_t frame_tiles_decoded = 0;
        uint32_t frame_l1_evictions = 0;
        uint32_t frame_bcn_evictions = 0;
        double frame_bcn_hit_rate = 1.0;
        double frame_l1_hit_rate = 1.0;
        double frame_l2_hit_rate = 1.0;

        // Pending synchronous BCn commit to inject right after begin_frame()
        std::vector<uint32_t> sync_commit_bkeys;
        std::vector<BcnTileLru::AllocResult> sync_commit_ballocs;

        if (!use_async_batch && prev_sync_slot >= 0 && fb_slot_valid[prev_sync_slot]) {
            uint32_t fbs = static_cast<uint32_t>(prev_sync_slot);
            ctx.direct_sync().wait(fb_slot_fence[fbs]);
            std::vector<uint32_t> chosen_bcn;
            process_feedback_map(fbs, chosen_bcn, frame_bcn_hit_rate);

            if (!chosen_bcn.empty()) {
                if (!use_ram_tier) {
                    for (uint32_t bkey : chosen_bcn) {
                        sync_commit_bkeys.push_back(bkey);
                        sync_commit_ballocs.push_back(bcn_lru.acquire(bkey));
                    }
                } else {
                    uint64_t f_id = static_cast<uint64_t>(f_iter + 1);
                    std::vector<uint32_t> needed_l1;
                    uint64_t l1_req_tot = 0, l1_hits = 0;
                    for (uint32_t bkey : chosen_bcn) {
                        uint32_t m = (bkey >> 16) & 0x1Fu;
                        uint32_t mip = (bkey >> 12) & 0xFu;
                        uint32_t ty = (bkey >> 6) & 0x3Fu;
                        uint32_t tx = bkey & 0x3Fu;
                        std::vector<uint32_t> lkeys;
                        gather_required_latent_tiles_with_halo(
                            m, mip, tx, ty, g0_packed.NumStandardMips, g1_packed.NumStandardMips,
                            g0_shape, g1_shape, g0_tilings, g1_tilings, lkeys
                        );
                        std::vector<uint32_t> new_l1;
                        for (uint32_t lk : lkeys) {
                            if (!l1_lru.contains(lk) &&
                                std::find(needed_l1.begin(), needed_l1.end(), lk) == needed_l1.end() &&
                                std::find(new_l1.begin(), new_l1.end(), lk) == new_l1.end()) {
                                new_l1.push_back(lk);
                            }
                        }
                        if (!sync_commit_bkeys.empty() && needed_l1.size() + new_l1.size() > kMaxL1BatchTiles) {
                            break; // Defer remaining BCn tiles to next frame rather than decoding with unmapped L1
                        }
                        for (uint32_t lk : lkeys) {
                            ++l1_req_tot;
                            if (l1_lru.contains(lk)) {
                                l1_lru.touch(lk, f_id);
                                ++l1_hits;
                            }
                        }
                        needed_l1.insert(needed_l1.end(), new_l1.begin(), new_l1.end());
                        sync_commit_bkeys.push_back(bkey);
                        sync_commit_ballocs.push_back(bcn_lru.acquire(bkey));
                    }
                    if (l1_req_tot > 0) frame_l1_hit_rate = static_cast<double>(l1_hits) / static_cast<double>(l1_req_tot);

                    if (!needed_l1.empty()) {
                        std::sort(needed_l1.begin(), needed_l1.end());
                        needed_l1.erase(std::unique(needed_l1.begin(), needed_l1.end()), needed_l1.end());
                        if (needed_l1.size() > kMaxL1BatchTiles) needed_l1.resize(kMaxL1BatchTiles);

                        LARGE_INTEGER tsub0{}, tsub1{};
                        QueryPerformanceCounter(&tsub0);
                        uint32_t nt = static_cast<uint32_t>(needed_l1.size());
                        std::vector<VramL1Lru::AllocResult> lallocs;
                        lallocs.reserve(nt);
                        uint64_t l2_hits = 0;
                        uint8_t* dst_ring = copy_up_ptr[fbs];
                        for (uint32_t j = 0; j < nt; ++j) {
                            uint32_t lk = needed_l1[j];
                            lallocs.push_back(l1_lru.acquire(lk, f_id));
                            bool l2h = false;
                            const uint8_t* src_tile = ram_l2.get_or_fetch_tile_64kb(lk, l2h);
                            if (l2h) ++l2_hits;
                            std::memcpy(dst_ring + static_cast<size_t>(j) * kTileBytes, src_tile, kTileBytes);
                        }
                        frame_l2_hit_rate = static_cast<double>(l2_hits) / static_cast<double>(nt);

                        uint64_t up_bytes = static_cast<uint64_t>(nt) * kTileBytes;
                        throw_if_failed(copy_alloc[fbs]->Reset(), "copy_alloc");
                        throw_if_failed(copy_cmd->Reset(copy_alloc[fbs].get(), nullptr), "copy_cmd");
                        copy_cmd->EndQuery(copy_ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, fbs * 2);
                        copy_cmd->CopyBufferRegion(vram_l1_staging.get(), 0, copy_upload_ring[fbs].get(), 0, up_bytes);
                        copy_cmd->EndQuery(copy_ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, fbs * 2 + 1);
                        copy_cmd->ResolveQueryData(copy_ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, fbs * 2, 2,
                                                   copy_ts_readback.get(), fbs * 2 * sizeof(uint64_t));
                        throw_if_failed(copy_cmd->Close(), "copy_cmd close");
                        ID3D12CommandList* cpl[] = {copy_cmd.get()};
                        ctx.copy_queue()->ExecuteCommandLists(1, cpl);
                        ctx.copy_sync().flush();
                        QueryPerformanceCounter(&tsub1);
                        record_copy_ts(fbs, frame_idx, nt, static_cast<double>(tsub1.QuadPart - tsub0.QuadPart) * 1e6 / qpc_freq);
                        frame_pcie_mb = static_cast<double>(up_bytes) / (1024.0 * 1024.0);

                        throw_if_failed(sync_alloc->Reset(), "sync_alloc");
                        throw_if_failed(sync_cmd->Reset(sync_alloc.get(), nullptr), "sync_cmd");
                        commit_l1_tiles(sync_cmd.get(), needed_l1, lallocs, frame_l1_evictions);
                        throw_if_failed(sync_cmd->Close(), "sync_cmd close");
                        ID3D12CommandList* dl[] = {sync_cmd.get()};
                        ctx.direct_queue()->ExecuteCommandLists(1, dl);
                        ctx.direct_sync().flush();
                    }
                }
                if (!sync_commit_bkeys.empty()) {
                    launch_compute_decode(fbs, sync_commit_bkeys, 0, true);
                    frame_tiles_decoded = static_cast<uint32_t>(sync_commit_bkeys.size());
                }
            }
        }

        uint32_t slot = ctx.begin_frame();
        ID3D12GraphicsCommandList7* cmd = ctx.cmd_list();

        double prev_gpu_ms = 0.0;
        if (ctx.try_read_gpu_frame_ms(slot, prev_gpu_ms) && slot_last_frame_idx[slot] >= 0 && !cfg.capture_checkpoints) {
            logger.set_frame_gpu_ms(slot_last_frame_idx[slot], prev_gpu_ms);
            for (auto& er : ext_rows) {
                if (er.frame_idx == slot_last_frame_idx[slot]) {
                    er.gpu_frame_ms = prev_gpu_ms;
                    break;
                }
            }
        }

        bool bcn_staging_busy_this_cmd = false;
        bool l1_staging_busy_this_cmd = false;

        if (!use_async_batch) {
            if (!sync_commit_bkeys.empty()) {
                commit_bcn_tiles(cmd, slot, sync_commit_bkeys, sync_commit_ballocs, frame_bcn_evictions);
                bcn_staging_busy_this_cmd = true;
            }
        } else {
            // === ASYNC NON-BLOCKING PIPELINE (C2_VRAM_Async, C3_RAM_Async, B1_SFS_BCn_Async_RamCache) ===
            // Stage A: Poll completed Compute (or CopyQueue for B1_SFS) -> Commit BCn tiles
            if (is_b1_sfs) {
                if (inflight_copy.active) {
                    bool done = (ctx.copy_sync().fence->GetCompletedValue() >= inflight_copy.fence_val);
                    if (!done && cfg.capture_checkpoints && !cfg.streamed_checkpoints) {
                        ctx.copy_sync().wait(inflight_copy.fence_val);
                        done = true;
                    }
                    if (done) {
                        record_copy_ts(inflight_copy.copy_slot, inflight_copy.submit_frame_idx,
                                       static_cast<uint32_t>(inflight_copy.bcn_keys.size()) * 3u, inflight_copy.cpu_submit_us);
                        commit_bcn_tiles(cmd, slot, inflight_copy.bcn_keys, inflight_copy.bcn_allocs, frame_bcn_evictions);
                        inflight_copy.active = false;
                        inflight_copy.bcn_keys.clear();
                        inflight_copy.bcn_allocs.clear();
                        bcn_staging_busy_this_cmd = true;
                    }
                }
            } else {
                if (inflight_decode.active) {
                    bool done = (ctx.compute_sync().fence->GetCompletedValue() >= inflight_decode.fence_val);
                    if (!done && cfg.capture_checkpoints && !cfg.streamed_checkpoints) {
                        ctx.compute_sync().wait(inflight_decode.fence_val);
                        done = true;
                    }
                    if (done) {
                        commit_bcn_tiles(cmd, slot, inflight_decode.bcn_keys, inflight_decode.bcn_allocs, frame_bcn_evictions);
                        inflight_decode.active = false;
                        inflight_decode.bcn_keys.clear();
                        inflight_decode.bcn_allocs.clear();
                        bcn_staging_busy_this_cmd = true;
                    }
                }

                // Stage B: Poll completed L2->L1 CopyQueue DMA -> Commit L1 tiles -> promote to ready_for_decode
                if (inflight_copy.active && !ready_for_decode.pending && !inflight_decode.active) {
                    bool done = !inflight_copy.has_dma || (ctx.copy_sync().fence->GetCompletedValue() >= inflight_copy.fence_val);
                    if (!done && cfg.capture_checkpoints && !cfg.streamed_checkpoints) {
                        ctx.copy_sync().wait(inflight_copy.fence_val);
                        done = true;
                    }
                    if (done) {
                        if (inflight_copy.has_dma && !inflight_copy.l1_keys.empty()) {
                            record_copy_ts(inflight_copy.copy_slot, inflight_copy.submit_frame_idx,
                                           static_cast<uint32_t>(inflight_copy.l1_keys.size()), inflight_copy.cpu_submit_us);
                            commit_l1_tiles(cmd, inflight_copy.l1_keys, inflight_copy.l1_allocs, frame_l1_evictions);
                            l1_staging_busy_this_cmd = true;
                        }
                        ready_for_decode.pending = true;
                        ready_for_decode.bcn_keys = std::move(inflight_copy.bcn_keys);
                        ready_for_decode.bcn_allocs = std::move(inflight_copy.bcn_allocs);
                        inflight_copy.active = false;
                        inflight_copy.has_dma = false;
                        inflight_copy.l1_keys.clear();
                        inflight_copy.l1_allocs.clear();
                    }
                }
            }

            // Stage C: Poll Sampler Feedback & launch L2->L1 (or RAM BCn->VRAM for B1, or directly ReadyDecode for C2)
            bool stage_c_free = is_b1_sfs
                ? (!inflight_copy.active && !bcn_staging_busy_this_cmd)
                : (!inflight_copy.active && !l1_staging_busy_this_cmd && !ready_for_decode.pending);
            if (fb_slot_valid[slot] && stage_c_free) {
                if (ctx.direct_sync().fence->GetCompletedValue() >= fb_slot_fence[slot]) {
                    std::vector<uint32_t> chosen_bcn;
                    process_feedback_map(slot, chosen_bcn, frame_bcn_hit_rate);
                    if (!chosen_bcn.empty()) {
                        if (is_b1_sfs) {
                            LARGE_INTEGER tsub0{}, tsub1{};
                            QueryPerformanceCounter(&tsub0);
                            uint32_t n_jobs = static_cast<uint32_t>(chosen_bcn.size());
                            uint8_t* dst = copy_up_ptr[slot];
                            uint64_t l2_hits = 0;
                            for (uint32_t j = 0; j < n_jobs; ++j) {
                                uint32_t bkey = chosen_bcn[j];
                                inflight_copy.bcn_keys.push_back(bkey);
                                inflight_copy.bcn_allocs.push_back(bcn_lru.acquire(bkey));
                                bool l2h = false;
                                const uint8_t* trip = bcn_ram_l2.get_or_build_triplet_192kb(bkey, l2h);
                                if (l2h) ++l2_hits;
                                std::memcpy(dst + static_cast<size_t>(j) * 3ULL * kTileBytes, trip, 3ULL * kTileBytes);
                            }
                            frame_l2_hit_rate = static_cast<double>(l2_hits) / static_cast<double>(n_jobs);

                            uint64_t up_bytes = static_cast<uint64_t>(n_jobs) * 3ULL * kTileBytes;
                            uint64_t prev_dv = ctx.direct_sync().next_fence_value - 1;
                            ctx.copy_queue()->Wait(ctx.direct_sync().fence.get(), prev_dv);
                            throw_if_failed(copy_alloc[slot]->Reset(), "copy_alloc");
                            throw_if_failed(copy_cmd->Reset(copy_alloc[slot].get(), nullptr), "copy_cmd");
                            copy_cmd->EndQuery(copy_ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, slot * 2);
                            copy_cmd->CopyBufferRegion(vram_bcn_staging.get(), 0, copy_upload_ring[slot].get(), 0, up_bytes);
                            copy_cmd->EndQuery(copy_ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, slot * 2 + 1);
                            copy_cmd->ResolveQueryData(copy_ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, slot * 2, 2,
                                                       copy_ts_readback.get(), slot * 2 * sizeof(uint64_t));
                            throw_if_failed(copy_cmd->Close(), "copy_cmd close");
                            ID3D12CommandList* cpl[] = {copy_cmd.get()};
                            ctx.copy_queue()->ExecuteCommandLists(1, cpl);
                            inflight_copy.fence_val = ctx.copy_sync().signal();
                            inflight_copy.has_dma = true;
                            inflight_copy.copy_slot = slot;
                            inflight_copy.submit_frame_idx = frame_idx;
                            inflight_copy.active = true;
                            QueryPerformanceCounter(&tsub1);
                            inflight_copy.cpu_submit_us = static_cast<double>(tsub1.QuadPart - tsub0.QuadPart) * 1e6 / qpc_freq;
                            frame_pcie_mb = static_cast<double>(up_bytes) / (1024.0 * 1024.0);
                        } else if (!use_ram_tier) {
                            // C2_VRAM_Async: Latents already resident in VRAM -> push straight to ready_for_decode
                            for (uint32_t bkey : chosen_bcn) {
                                ready_for_decode.bcn_keys.push_back(bkey);
                                ready_for_decode.bcn_allocs.push_back(bcn_lru.acquire(bkey));
                            }
                            ready_for_decode.pending = true;
                        } else {
                            // C3_RAM_Async: Check L1 LRU atomically per triplet (zero truncation), fetch L2->L1 async on CopyQueue
                            uint64_t f_id = static_cast<uint64_t>(f_iter + 1);
                            std::vector<uint32_t> needed_l1;
                            uint64_t l1_req_tot = 0, l1_hits = 0;
                            for (uint32_t bkey : chosen_bcn) {
                                uint32_t m = (bkey >> 16) & 0x1Fu;
                                uint32_t mip = (bkey >> 12) & 0xFu;
                                uint32_t ty = (bkey >> 6) & 0x3Fu;
                                uint32_t tx = bkey & 0x3Fu;
                                std::vector<uint32_t> lkeys;
                                gather_required_latent_tiles_with_halo(
                                    m, mip, tx, ty, g0_packed.NumStandardMips, g1_packed.NumStandardMips,
                                    g0_shape, g1_shape, g0_tilings, g1_tilings, lkeys
                                );
                                std::vector<uint32_t> new_l1;
                                for (uint32_t lk : lkeys) {
                                    if (!l1_lru.contains(lk) &&
                                        std::find(needed_l1.begin(), needed_l1.end(), lk) == needed_l1.end() &&
                                        std::find(new_l1.begin(), new_l1.end(), lk) == new_l1.end()) {
                                        new_l1.push_back(lk);
                                    }
                                }
                                if (!inflight_copy.bcn_keys.empty() && needed_l1.size() + new_l1.size() > kMaxL1BatchTiles) {
                                    break; // Never decode a BCn tile unless ALL its L1 latents are mapped
                                }
                                for (uint32_t lk : lkeys) {
                                    ++l1_req_tot;
                                    if (l1_lru.contains(lk)) {
                                        l1_lru.touch(lk, f_id);
                                        ++l1_hits;
                                    }
                                }
                                needed_l1.insert(needed_l1.end(), new_l1.begin(), new_l1.end());
                                inflight_copy.bcn_keys.push_back(bkey);
                                inflight_copy.bcn_allocs.push_back(bcn_lru.acquire(bkey));
                            }
                            if (l1_req_tot > 0) frame_l1_hit_rate = static_cast<double>(l1_hits) / static_cast<double>(l1_req_tot);

                            std::sort(needed_l1.begin(), needed_l1.end());
                            needed_l1.erase(std::unique(needed_l1.begin(), needed_l1.end()), needed_l1.end());
                            if (needed_l1.size() > kMaxL1BatchTiles) needed_l1.resize(kMaxL1BatchTiles);

                            if (!needed_l1.empty()) {
                                LARGE_INTEGER tsub0{}, tsub1{};
                                QueryPerformanceCounter(&tsub0);
                                uint32_t nt = static_cast<uint32_t>(needed_l1.size());
                                uint8_t* dst_ring = copy_up_ptr[slot];
                                uint64_t l2_hits = 0;
                                for (uint32_t j = 0; j < nt; ++j) {
                                    uint32_t lk = needed_l1[j];
                                    inflight_copy.l1_keys.push_back(lk);
                                    inflight_copy.l1_allocs.push_back(l1_lru.acquire(lk, f_id));
                                    bool l2h = false;
                                    const uint8_t* src_tile = ram_l2.get_or_fetch_tile_64kb(lk, l2h);
                                    if (l2h) ++l2_hits;
                                    std::memcpy(dst_ring + static_cast<size_t>(j) * kTileBytes, src_tile, kTileBytes);
                                }
                                frame_l2_hit_rate = static_cast<double>(l2_hits) / static_cast<double>(nt);

                                uint64_t up_bytes = static_cast<uint64_t>(nt) * kTileBytes;
                                if (last_l1_commit_direct_fence > 0) {
                                    ctx.copy_queue()->Wait(ctx.direct_sync().fence.get(), last_l1_commit_direct_fence);
                                }
                                throw_if_failed(copy_alloc[slot]->Reset(), "copy_alloc");
                                throw_if_failed(copy_cmd->Reset(copy_alloc[slot].get(), nullptr), "copy_cmd");
                                copy_cmd->EndQuery(copy_ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, slot * 2);
                                copy_cmd->CopyBufferRegion(vram_l1_staging.get(), 0, copy_upload_ring[slot].get(), 0, up_bytes);
                                copy_cmd->EndQuery(copy_ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, slot * 2 + 1);
                                copy_cmd->ResolveQueryData(copy_ts_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, slot * 2, 2,
                                                           copy_ts_readback.get(), slot * 2 * sizeof(uint64_t));
                                throw_if_failed(copy_cmd->Close(), "copy_cmd close");
                                ID3D12CommandList* cpl[] = {copy_cmd.get()};
                                ctx.copy_queue()->ExecuteCommandLists(1, cpl);
                                inflight_copy.fence_val = ctx.copy_sync().signal();
                                inflight_copy.has_dma = true;
                                inflight_copy.copy_slot = slot;
                                inflight_copy.submit_frame_idx = frame_idx;
                                QueryPerformanceCounter(&tsub1);
                                inflight_copy.cpu_submit_us = static_cast<double>(tsub1.QuadPart - tsub0.QuadPart) * 1e6 / qpc_freq;
                                frame_pcie_mb = static_cast<double>(up_bytes) / (1024.0 * 1024.0);
                            } else {
                                inflight_copy.has_dma = false;
                                inflight_copy.fence_val = 0;
                            }
                            inflight_copy.active = true;
                        }
                    }
                }
            }
        }

        // Main Raster Pass
        ID3D12DescriptorHeap* heaps[] = {srv_heap.heap()};
        cmd->SetDescriptorHeaps(1, heaps);
        {
            const UINT clr[4] = {0xFFu, 0xFFu, 0xFFu, 0xFFu};
            for (uint32_t m = 0; m < kNumMaterials; ++m) {
                cmd->ClearUnorderedAccessViewUint(
                    srv_heap.gpu(fb_uav_base + m), ctx.cpu_uav_heap().cpu(fb_cpu_uav_base + m),
                    fb_tex[m].get(), clr, 0, nullptr
                );
            }
        }

        CameraMatrices cam = scene.evaluate_camera(frame_idx, aspect);
        {
            FrameConstants fc{};
            std::memcpy(fc.view_proj, cam.view_proj, sizeof(fc.view_proj));
            std::memcpy(fc.cam_pos, cam.cam_pos, sizeof(fc.cam_pos));
            fc.time_sec = static_cast<float>(f_iter) / 60.0f;
            fc.enable_feedback = 1u;
            fc.use_tile_residency = 1u;
            std::memcpy(frame_cb_ptrs[slot], &fc, sizeof(fc));
        }

        D3D12_RESOURCE_BARRIER rt_b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_FLAG_NONE};
        rt_b.Transition = {ctx.current_rtv_resource(slot), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                           D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET};
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
        cmd->SetGraphicsRootDescriptorTable(1, srv_heap.gpu(alb_srv_base));
        cmd->SetGraphicsRootDescriptorTable(2, srv_heap.gpu(nrm_srv_base));
        cmd->SetGraphicsRootDescriptorTable(3, srv_heap.gpu(orm_srv_base));
        cmd->SetGraphicsRootDescriptorTable(4, srv_heap.gpu(fb_uav_base));
        cmd->SetGraphicsRootShaderResourceView(5, residency_buf->GetGPUVirtualAddress());

        for (const auto& bay : bays) {
            if (scene.is_aabb_visible(bay.min_xyz, bay.max_xyz, cam)) {
                cmd->DrawIndexedInstanced(bay.index_count, 1, bay.index_offset, 0, 0);
            }
        }

        {
            std::vector<D3D12_RESOURCE_BARRIER> res_b(kNumMaterials * 2);
            for (uint32_t m = 0; m < kNumMaterials; ++m) {
                res_b[m * 2 + 0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                res_b[m * 2 + 0].Transition = {fb_tex[m].get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RESOLVE_SOURCE};
                res_b[m * 2 + 1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                res_b[m * 2 + 1].Transition = {fb_resolved[m].get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                               D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RESOLVE_DEST};
            }
            cmd->ResourceBarrier(static_cast<UINT>(res_b.size()), res_b.data());
            for (uint32_t m = 0; m < kNumMaterials; ++m) {
                cmd->ResolveSubresourceRegion(fb_resolved[m].get(), 0, 0, 0, fb_tex[m].get(), 0, nullptr,
                                              DXGI_FORMAT_R8_UINT, D3D12_RESOLVE_MODE_DECODE_SAMPLER_FEEDBACK);
            }
            for (uint32_t m = 0; m < kNumMaterials; ++m) {
                res_b[m * 2 + 0].Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
                res_b[m * 2 + 0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                res_b[m * 2 + 1].Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_DEST;
                res_b[m * 2 + 1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            }
            cmd->ResourceBarrier(static_cast<UINT>(res_b.size()), res_b.data());
            for (uint32_t m = 0; m < kNumMaterials; ++m) {
                D3D12_TEXTURE_COPY_LOCATION src_loc{fb_resolved[m].get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
                D3D12_TEXTURE_COPY_LOCATION dst_loc{fb_readback[slot].get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
                dst_loc.PlacedFootprint.Offset = static_cast<UINT64>(m) * 32ULL * 256ULL;
                dst_loc.PlacedFootprint.Footprint = {DXGI_FORMAT_R8_UINT, 32, 32, 1, 256};
                cmd->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);
                res_b[m].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                res_b[m].Transition = {fb_resolved[m].get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                       D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST};
            }
            cmd->ResourceBarrier(kNumMaterials, res_b.data());
        }

        std::swap(rt_b.Transition.StateBefore, rt_b.Transition.StateAfter);
        cmd->ResourceBarrier(1, &rt_b);

        ctx.end_frame(slot, true);
        LARGE_INTEGER cur_present_qpc{};
        QueryPerformanceCounter(&cur_present_qpc);
        double display_ms = static_cast<double>(cur_present_qpc.QuadPart - prev_present_qpc.QuadPart) * 1000.0 / qpc_freq;
        prev_present_qpc = cur_present_qpc;

        uint64_t this_direct_val = ctx.direct_sync().next_fence_value - 1;
        if (l1_staging_busy_this_cmd) {
            last_l1_commit_direct_fence = this_direct_val;
        }
        fb_slot_valid[slot] = true;
        fb_slot_fence[slot] = this_direct_val;
        prev_sync_slot = static_cast<int>(slot);

        // Async Stage: Launch Compute decode in parallel with next frame after Direct CopyTiles finishes
        if (use_async_batch && !is_b1_sfs && ready_for_decode.pending && !inflight_decode.active && !bcn_staging_busy_this_cmd) {
            inflight_decode.fence_val = launch_compute_decode(slot, ready_for_decode.bcn_keys, this_direct_val, false);
            inflight_decode.active = true;
            inflight_decode.bcn_keys = std::move(ready_for_decode.bcn_keys);
            inflight_decode.bcn_allocs = std::move(ready_for_decode.bcn_allocs);
            ready_for_decode.pending = false;
            frame_tiles_decoded = static_cast<uint32_t>(inflight_decode.bcn_keys.size());
        }

        LARGE_INTEGER t_cpu_1{};
        QueryPerformanceCounter(&t_cpu_1);
        double cpu_work_ms = static_cast<double>(t_cpu_1.QuadPart - t_cpu_0.QuadPart) * 1000.0 / qpc_freq;

        // Per-frame DXGI VRAM peak tracking + residency deadline miss & pop-in metrics
        double cur_loc_mb = 0.0, cur_nloc_mb = 0.0;
        ctx.query_vram_usage_mb(cur_loc_mb, cur_nloc_mb);
        peak_local_vram_mb = std::max(peak_local_vram_mb, cur_loc_mb);
        peak_nonlocal_vram_mb = std::max(peak_nonlocal_vram_mb, cur_nloc_mb);

        if (cfg.capture_checkpoints && frame_idx >= 0) {
            for (const auto& ck : checkpoints) {
                if (ck.second == frame_idx) {
                    std::filesystem::create_directories(cfg.capture_dir);
                    std::string out_png = cfg.capture_dir + "/" + cfg.config_name + "_" + ck.first + ".png";
                    ctx.capture_backbuffer_png(slot, out_png);
                }
            }
        }

        if (!cfg.capture_checkpoints && frame_idx >= 0) {
            uint32_t vis_cells = 0, miss_cells = 0, popin_cnt = 0;
            uint64_t deficit_sum = 0;
            for (uint32_t c = 0; c < kNumMaterials * 1024u; ++c) {
                uint8_t req = last_fb_snapshot[c];
                if (req < kStreamedMipCount) {
                    ++vis_cells;
                    uint32_t m = c >> 10;
                    uint32_t rem = c & 1023u;
                    uint32_t cur_res = bcn_lru.finest_resident_mip(m, rem >> 5, rem & 31u);
                    if (cur_res > req) {
                        ++miss_cells;
                        deficit_sum += (cur_res - req);
                    }
                    if (prev_visible_res_mip[c] <= 6 && cur_res < prev_visible_res_mip[c]) {
                        ++popin_cnt;
                    }
                    prev_visible_res_mip[c] = static_cast<uint8_t>(cur_res);
                } else {
                    prev_visible_res_mip[c] = 0xFFu;
                }
            }

            FrameMetricRow frow;
            frow.run_id = cfg.run_id;
            frow.config_name = cfg.config_name;
            frow.frame_idx = frame_idx;
            frow.timestamp_ms = logger.elapsed_ms();
            frow.cpu_frame_ms = display_ms;
            frow.gpu_frame_ms = 0.0;
            frow.present_ms = ctx.last_present_ms();
            frow.pcie_upload_mb = frame_pcie_mb;
            frow.tiles_decoded_dgpu = frame_tiles_decoded;
            logger.record_frame(frow);

            P2RFrameExtRow erow;
            erow.run_id = cfg.run_id;
            erow.config_name = cfg.config_name;
            erow.frame_idx = frame_idx;
            erow.display_frame_ms = display_ms;
            erow.cpu_work_ms = cpu_work_ms;
            erow.gpu_frame_ms = 0.0;
            erow.present_call_ms = ctx.last_present_ms();
            erow.visible_cells = vis_cells;
            erow.bcn_hit_rate = frame_bcn_hit_rate;
            erow.l1_hit_rate = frame_l1_hit_rate;
            erow.l2_hit_rate = frame_l2_hit_rate;
            erow.deadline_miss_rate = (vis_cells > 0) ? (static_cast<double>(miss_cells) / static_cast<double>(vis_cells)) : 0.0;
            erow.mean_mip_deficit = (vis_cells > 0) ? (static_cast<double>(deficit_sum) / static_cast<double>(vis_cells)) : 0.0;
            erow.popin_transitions = popin_cnt;
            erow.l1_evictions = frame_l1_evictions;
            erow.bcn_evictions = frame_bcn_evictions;
            ext_rows.push_back(erow);

            slot_last_frame_idx[slot] = frame_idx;
        } else {
            slot_last_frame_idx[slot] = -1;
        }
    }

    ctx.wait_idle();
    if (!cfg.capture_checkpoints) {
        for (uint32_t s = 0; s < kFrameInFlightCount; ++s) {
            double tail_gpu_ms = 0.0;
            if (slot_last_frame_idx[s] >= 0 && ctx.try_read_gpu_frame_ms(s, tail_gpu_ms)) {
                logger.set_frame_gpu_ms(slot_last_frame_idx[s], tail_gpu_ms);
                for (auto& er : ext_rows) {
                    if (er.frame_idx == slot_last_frame_idx[s]) {
                        er.gpu_frame_ms = tail_gpu_ms;
                        break;
                    }
                }
            }
        }
        logger.stop_run();
        logger.write_or_append_csv(cfg.out_dir + "/frame_metrics.csv", cfg.out_dir + "/memory_power.csv");
        append_csv_pcie(cfg.out_dir + "/pcie_transfer_trace.csv", pcie_rows);
        append_csv_frame_ext(cfg.out_dir + "/frame_extended_metrics.csv", ext_rows);

        constexpr double kMiB = 1024.0 * 1024.0;
        P2RMemoryBreakdownRow mb{};
        mb.run_id = cfg.run_id;
        mb.config_name = cfg.config_name;
        mb.storage_mode = is_b1_sfs ? "SFS_BCn_RAMCache" : (use_ram_tier ? "RAM_Tiered" : "VRAM_Resident");
        mb.scheduling_mode = use_async_batch ? "Async_Batched" : "Sync_Unbatched";
        mb.copy_queue_freq_hz = verified_copy_freq_hz;
        mb.bcn_cache_heap_mib = static_cast<double>(bcn_cache_heap_bytes) / kMiB;
        mb.bcn_tail_heap_mib = static_cast<double>(bcn_tail_heap_bytes) / kMiB;
        mb.l1_latent_heap_mib = static_cast<double>(l1_heap_bytes) / kMiB;
        mb.latent_tail_heap_mib = static_cast<double>(latent_tail_bytes) / kMiB;
        mb.committed_latent_textures_mib = static_cast<double>(committed_latent_bytes) / kMiB;
        mb.vram_staging_buffers_mib = static_cast<double>(vram_staging_bytes) / kMiB;
        mb.render_targets_feedback_mib = static_cast<double>(rt_fb_alloc_bytes) / kMiB;
        mb.explicit_local_vram_sum_mib =
            mb.bcn_cache_heap_mib + mb.bcn_tail_heap_mib + mb.l1_latent_heap_mib +
            mb.latent_tail_heap_mib + mb.committed_latent_textures_mib +
            mb.vram_staging_buffers_mib + mb.render_targets_feedback_mib;
        mb.dxgi_peak_local_vram_mib = peak_local_vram_mb;
        mb.driver_overhead_mib = peak_local_vram_mb - mb.explicit_local_vram_sum_mib;
        mb.dxgi_peak_nonlocal_mib = peak_nonlocal_vram_mb;
        mb.upload_readback_buffers_mib = static_cast<double>(upload_rb_bytes) / kMiB;
        mb.ram_arena_committed_mib = is_b1_sfs ? bcn_ram_l2.committed_mib() : (use_ram_tier ? ram_l2.committed_mib() : 0.0);
        mb.ram_working_set_populated_mib = is_b1_sfs ? bcn_ram_l2.peak_populated_mib() : (use_ram_tier ? ram_l2.peak_populated_mib() : 0.0);
        mb.vram_l1_populated_peak_mib = static_cast<double>(peak_l1_used_slots) * 64.0 / 1024.0;
        mb.bcn_cache_populated_peak_mib = static_cast<double>(peak_bcn_used_slots) * 192.0 / 1024.0;
        append_csv_mem_breakdown(cfg.out_dir + "/resource_memory_breakdown.csv", mb);
    }
}

} // namespace pahnts
