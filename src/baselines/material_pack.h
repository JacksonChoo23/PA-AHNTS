#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pahnts {

struct BlobSlice {
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

struct MaterialBcnMip {
    uint32_t size = 0;
    BlobSlice albedo_bc7;
    BlobSlice normal_bc5;
    BlobSlice orm_bc7;
};

struct MaterialAssetMeta {
    uint32_t material_id = 0;
    double ntc_psnr_db = 0.0;
    std::vector<BlobSlice> g0_mips; // 256 .. 1 (9 mips)
    std::vector<BlobSlice> g1_mips; // 64 .. 1 (7 mips)
    std::vector<MaterialBcnMip> bcn_mips; // 512 .. 8 (7 mips)
};

class MaterialPack {
public:
    void load(const std::string& manifest_path);

    const uint8_t* data_ptr(const BlobSlice& s) const { return blob_.data() + s.offset; }
    const uint8_t* mlp_table_ptr() const { return blob_.data() + mlp_offset_; }
    uint64_t mlp_table_bytes() const { return mlp_bytes_; }
    uint32_t mlp_stride_bytes() const { return mlp_stride_; }

    const std::vector<MaterialAssetMeta>& materials() const { return materials_; }
    uint32_t material_count() const { return static_cast<uint32_t>(materials_.size()); }

    // Build a 64KB (64x64 4x4-blocks = 256x256 texels) BC7 or BC5 tile for mip in [0..5]
    // map_kind: 0=Albedo BC7, 1=Normal BC5, 2=ORM BC7
    void build_bcn_tile_64kb(
        uint32_t mat_id, uint32_t mip_level, uint32_t tile_x, uint32_t tile_y,
        int map_kind, uint8_t* dst_64kb
    ) const;

    // Build tiled 2D buffer for one mip of the 8192x8192 BCn texture (for B0_BCn_Full init)
    void build_bcn_mip_tiled(
        uint32_t mat_id, uint32_t mip_level, int map_kind, std::vector<uint8_t>& out_buf
    ) const;

    // Build tiled G0 (4096 base -> mip 0..12) or G1 (1024 base -> mip 0..10)
    void build_g0_mip_tiled(uint32_t mat_id, uint32_t mip_level, std::vector<uint8_t>& out_buf) const;
    void build_g1_mip_tiled(uint32_t mat_id, uint32_t mip_level, std::vector<uint8_t>& out_buf) const;

private:
    std::vector<uint8_t> blob_;
    uint64_t mlp_offset_ = 0;
    uint64_t mlp_bytes_ = 0;
    uint32_t mlp_stride_ = 3840;
    std::vector<MaterialAssetMeta> materials_;
};

} // namespace pahnts
