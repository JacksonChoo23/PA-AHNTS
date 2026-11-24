#include "material_pack.h"
#include "../core/json_lite.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace pahnts {

void MaterialPack::load(const std::string& manifest_path) {
    JsonValue root = JsonParser::parse_file(manifest_path);
    std::string pack_file = root["pack_file"].as_string();

    std::ifstream ifs(pack_file, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("Failed to open asset pack: " + pack_file);
    std::streamsize sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    blob_.resize(static_cast<size_t>(sz));
    if (!ifs.read(reinterpret_cast<char*>(blob_.data()), sz)) {
        throw std::runtime_error("Failed to read asset pack: " + pack_file);
    }

    mlp_offset_ = root["mlp_table"]["offset"].as_u64();
    mlp_bytes_ = root["mlp_table"]["bytes"].as_u64();
    mlp_stride_ = static_cast<uint32_t>(root["mlp_table"]["stride_bytes"].as_u64(3840));

    materials_.clear();
    const auto& mats = root["materials"];
    materials_.reserve(mats.size());
    for (size_t i = 0; i < mats.size(); ++i) {
        const auto& m = mats[i];
        MaterialAssetMeta meta;
        meta.material_id = static_cast<uint32_t>(m["material_id"].as_int());
        meta.ntc_psnr_db = m["ntc_psnr_db"].as_double();

        const auto& g0 = m["g0_mips"];
        for (size_t k = 0; k < g0.size(); ++k) {
            meta.g0_mips.push_back({g0[k]["offset"].as_u64(), g0[k]["bytes"].as_u64()});
        }
        const auto& g1 = m["g1_mips"];
        for (size_t k = 0; k < g1.size(); ++k) {
            meta.g1_mips.push_back({g1[k]["offset"].as_u64(), g1[k]["bytes"].as_u64()});
        }
        const auto& bm = m["bcn_mips"];
        for (size_t k = 0; k < bm.size(); ++k) {
            MaterialBcnMip entry;
            entry.size = static_cast<uint32_t>(bm[k]["size"].as_int());
            entry.albedo_bc7 = {bm[k]["albedo_bc7"]["offset"].as_u64(), bm[k]["albedo_bc7"]["bytes"].as_u64()};
            entry.normal_bc5 = {bm[k]["normal_bc5"]["offset"].as_u64(), bm[k]["normal_bc5"]["bytes"].as_u64()};
            entry.orm_bc7    = {bm[k]["orm_bc7"]["offset"].as_u64(),    bm[k]["orm_bc7"]["bytes"].as_u64()};
            meta.bcn_mips.push_back(entry);
        }
        materials_.push_back(std::move(meta));
    }
}

void MaterialPack::build_bcn_tile_64kb(
    uint32_t mat_id, uint32_t mip_level, uint32_t tile_x, uint32_t tile_y,
    int map_kind, uint8_t* dst_64kb
) const {
    const auto& bm = materials_[mat_id].bcn_mips[mip_level];
    const BlobSlice& sl = (map_kind == 0) ? bm.albedo_bc7 : ((map_kind == 1) ? bm.normal_bc5 : bm.orm_bc7);
    const uint8_t* src = data_ptr(sl);

    uint32_t base_blocks = bm.size / 4; // 128 at mip 0, 64 at mip 1, 32 at mip 2, ...
    constexpr uint32_t kTileBlocks = 64;
    constexpr uint32_t kBlockBytes = 16;

    if (mip_level == 0) {
        // Base is 512x512 (128x128 blocks), 1 tile = 256x256 (64x64 blocks)
        uint32_t bx0 = (tile_x & 1u) * 64u;
        uint32_t by0 = (tile_y & 1u) * 64u;
        for (uint32_t r = 0; r < kTileBlocks; ++r) {
            const uint8_t* srow = src + ((by0 + r) * base_blocks + bx0) * kBlockBytes;
            std::memcpy(dst_64kb + r * kTileBlocks * kBlockBytes, srow, kTileBlocks * kBlockBytes);
        }
    } else {
        // mip_level >= 1: base_blocks <= 64, tile repeats base_blocks seamlessly to 64x64 blocks
        for (uint32_t r = 0; r < kTileBlocks; ++r) {
            uint32_t sr = r % base_blocks;
            const uint8_t* srow = src + sr * base_blocks * kBlockBytes;
            uint8_t* drow = dst_64kb + r * kTileBlocks * kBlockBytes;
            for (uint32_t cx = 0; cx < kTileBlocks; cx += base_blocks) {
                std::memcpy(drow + cx * kBlockBytes, srow, base_blocks * kBlockBytes);
            }
        }
    }
}

void MaterialPack::build_bcn_mip_tiled(
    uint32_t mat_id, uint32_t mip_level, int map_kind, std::vector<uint8_t>& out_buf
) const {
    uint32_t world_dim = 8192u >> mip_level;
    uint32_t world_blocks = std::max(1u, world_dim / 4u);
    constexpr uint32_t kBlockBytes = 16;

    uint32_t bcn_idx = std::min<uint32_t>(mip_level, 6u);
    const auto& bm = materials_[mat_id].bcn_mips[bcn_idx];
    const BlobSlice& sl = (map_kind == 0) ? bm.albedo_bc7 : ((map_kind == 1) ? bm.normal_bc5 : bm.orm_bc7);
    const uint8_t* src = data_ptr(sl);

    if (mip_level >= 7) {
        // Mips 7..10 (64..8): world_blocks <= 16; crop/tile from 8x8 base (2x2 blocks)
        uint32_t base_blocks = bm.size / 4; // 2
        out_buf.resize(static_cast<size_t>(world_blocks) * world_blocks * kBlockBytes);
        for (uint32_t r = 0; r < world_blocks; ++r) {
            uint32_t sr = r % base_blocks;
            for (uint32_t c = 0; c < world_blocks; ++c) {
                uint32_t sc = c % base_blocks;
                std::memcpy(
                    out_buf.data() + (static_cast<size_t>(r) * world_blocks + c) * kBlockBytes,
                    src + (sr * base_blocks + sc) * kBlockBytes,
                    kBlockBytes
                );
            }
        }
        return;
    }

    // Mips 0..6 (8192..128) are 16x16 repeats of bcn_mips[mip_level] (512..8)
    uint32_t base_blocks = bm.size / 4; // world_blocks / 16
    uint32_t base_row_bytes = base_blocks * kBlockBytes;

    out_buf.resize(static_cast<size_t>(world_blocks) * world_blocks * kBlockBytes);
    for (uint32_t r = 0; r < world_blocks; ++r) {
        uint32_t sr = r % base_blocks;
        const uint8_t* srow = src + sr * base_row_bytes;
        uint8_t* drow = out_buf.data() + static_cast<size_t>(r) * world_blocks * kBlockBytes;
        for (uint32_t rep = 0; rep < 16u; ++rep) {
            std::memcpy(drow + rep * base_row_bytes, srow, base_row_bytes);
        }
    }
}

void MaterialPack::build_g0_mip_tiled(uint32_t mat_id, uint32_t mip_level, std::vector<uint8_t>& out_buf) const {
    uint32_t world_dim = 4096u >> mip_level;
    uint32_t g0_idx = std::min<uint32_t>(mip_level, 8u);
    const BlobSlice& sl = materials_[mat_id].g0_mips[g0_idx];
    const uint8_t* src = data_ptr(sl);
    uint32_t base_dim = 256u >> g0_idx;
    constexpr uint32_t kPixBytes = 4;

    out_buf.resize(static_cast<size_t>(world_dim) * world_dim * kPixBytes);
    if (mip_level >= 9) {
        // Mip 9..12 (8..1): replicate 1x1 pixel
        for (size_t i = 0; i < static_cast<size_t>(world_dim) * world_dim; ++i) {
            std::memcpy(out_buf.data() + i * kPixBytes, src, kPixBytes);
        }
        return;
    }

    uint32_t base_row_bytes = base_dim * kPixBytes;
    for (uint32_t r = 0; r < world_dim; ++r) {
        uint32_t sr = r % base_dim;
        const uint8_t* srow = src + sr * base_row_bytes;
        uint8_t* drow = out_buf.data() + static_cast<size_t>(r) * world_dim * kPixBytes;
        for (uint32_t rep = 0; rep < 16u; ++rep) {
            std::memcpy(drow + rep * base_row_bytes, srow, base_row_bytes);
        }
    }
}

void MaterialPack::build_g1_mip_tiled(uint32_t mat_id, uint32_t mip_level, std::vector<uint8_t>& out_buf) const {
    uint32_t world_dim = 1024u >> mip_level;
    uint32_t g1_idx = std::min<uint32_t>(mip_level, 6u);
    const BlobSlice& sl = materials_[mat_id].g1_mips[g1_idx];
    const uint8_t* src = data_ptr(sl);
    uint32_t base_dim = 64u >> g1_idx;
    constexpr uint32_t kPixBytes = 8; // RGBA16F

    out_buf.resize(static_cast<size_t>(world_dim) * world_dim * kPixBytes);
    if (mip_level >= 7) {
        for (size_t i = 0; i < static_cast<size_t>(world_dim) * world_dim; ++i) {
            std::memcpy(out_buf.data() + i * kPixBytes, src, kPixBytes);
        }
        return;
    }

    uint32_t base_row_bytes = base_dim * kPixBytes;
    for (uint32_t r = 0; r < world_dim; ++r) {
        uint32_t sr = r % base_dim;
        const uint8_t* srow = src + sr * base_row_bytes;
        uint8_t* drow = out_buf.data() + static_cast<size_t>(r) * world_dim * kPixBytes;
        for (uint32_t rep = 0; rep < 16u; ++rep) {
            std::memcpy(drow + rep * base_row_bytes, srow, base_row_bytes);
        }
    }
}

} // namespace pahnts
