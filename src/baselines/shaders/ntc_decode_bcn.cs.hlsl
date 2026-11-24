#include "ntc_mlp.hlsli"

struct TileDecodeJob {
    uint matId;
    uint mipLevel;
    uint tileX;
    uint tileY;
    uint dstAlbedoByteOffset;
    uint dstNormalByteOffset;
    uint dstOrmByteOffset;
    uint pad;
};

Texture2D g_LatentG0[32] : register(t0, space0);
Texture2D g_LatentG1[32] : register(t0, space1);
ByteAddressBuffer g_MlpWeights : register(t0, space2);
StructuredBuffer<TileDecodeJob> g_Jobs : register(t0, space3);
RWByteAddressBuffer g_DecodedTilesOut : register(u0, space0);
SamplerState g_LinearSampler : register(s0);

uint4 pack_bc7_mode6(uint4 rgba[16]) {
    // Select endpoint pair from highest-variance channel
    uint4 bmin = rgba[0];
    uint4 bmax = rgba[0];
    [unroll]
    for (uint i = 1u; i < 16u; ++i) {
        bmin = min(bmin, rgba[i]);
        bmax = max(bmax, rgba[i]);
    }
    uint4 span = bmax - bmin;
    uint dom = 0u;
    uint maxSpan = span.x;
    if (span.y > maxSpan) { maxSpan = span.y; dom = 1u; }
    if (span.z > maxSpan) { maxSpan = span.z; dom = 2u; }
    if (span.w > maxSpan) { maxSpan = span.w; dom = 3u; }

    uint minVal = 256u;
    uint maxVal = 0u;
    uint4 e0 = rgba[0];
    uint4 e1 = rgba[0];
    [unroll]
    for (uint i = 0u; i < 16u; ++i) {
        uint v = rgba[i][dom];
        if (v < minVal) { minVal = v; e0 = rgba[i]; }
        if (v > maxVal) { maxVal = v; e1 = rgba[i]; }
    }

    uint p0 = (((e0.x & 1u) + (e0.y & 1u) + (e0.z & 1u)) >= 2u) ? 1u : 0u;
    uint p1 = (((e1.x & 1u) + (e1.y & 1u) + (e1.z & 1u)) >= 2u) ? 1u : 0u;

    float4 rec0 = float4(((e0 >> 1u) << 1u) | p0);
    float4 rec1 = float4(((e1 >> 1u) << 1u) | p1);
    float4 diff = rec1 - rec0;
    float denom = dot(diff, diff);

    uint q[16];
    [unroll]
    for (uint i = 0u; i < 16u; ++i) {
        float proj = (denom < 1e-4f) ? 0.0f : dot(float4(rgba[i]) - rec0, diff) / denom;
        q[i] = min(15u, uint(round(saturate(proj) * 15.0f)));
    }

    if (q[0] >= 8u) {
        uint4 tmpE = e0; e0 = e1; e1 = tmpE;
        uint tmpP = p0; p0 = p1; p1 = tmpP;
        [unroll]
        for (uint i = 0u; i < 16u; ++i) {
            q[i] = 15u - q[i];
        }
    }

    uint64_t lo64 =
        uint64_t(0x40u)
        | (uint64_t(e0.x >> 1u) << 7)
        | (uint64_t(e1.x >> 1u) << 14)
        | (uint64_t(e0.y >> 1u) << 21)
        | (uint64_t(e1.y >> 1u) << 28)
        | (uint64_t(e0.z >> 1u) << 35)
        | (uint64_t(e1.z >> 1u) << 42)
        | (uint64_t(e0.w >> 1u) << 49)
        | (uint64_t(e1.w >> 1u) << 56)
        | (uint64_t(p0) << 63);

    uint64_t hi64 = uint64_t(p1) | (uint64_t(q[0] & 0x7u) << 1);
    [unroll]
    for (uint i = 1u; i < 16u; ++i) {
        hi64 |= (uint64_t(q[i] & 0xFu) << (4u * i));
    }

    return uint4(
        uint(lo64 & 0xFFFFFFFFu),
        uint(lo64 >> 32),
        uint(hi64 & 0xFFFFFFFFu),
        uint(hi64 >> 32)
    );
}

uint64_t pack_bc4_channel(uint vals[16]) {
    uint e0 = vals[0];
    uint e1 = vals[0];
    [unroll]
    for (uint i = 1u; i < 16u; ++i) {
        e0 = max(e0, vals[i]);
        e1 = min(e1, vals[i]);
    }
    uint64_t w = uint64_t(e0) | (uint64_t(e1) << 8);
    if (e0 > e1) {
        float invSpan = 7.0f / float(e0 - e1);
        [unroll]
        for (uint i = 0u; i < 16u; ++i) {
            uint s = min(7u, uint(round(float(e0 - vals[i]) * invSpan)));
            uint idx = (s == 0u) ? 0u : ((s == 7u) ? 1u : (s + 1u));
            w |= (uint64_t(idx) << (16u + 3u * i));
        }
    }
    return w;
}

uint4 pack_bc5(uint nx[16], uint ny[16]) {
    uint64_t c0 = pack_bc4_channel(nx);
    uint64_t c1 = pack_bc4_channel(ny);
    return uint4(
        uint(c0 & 0xFFFFFFFFu),
        uint(c0 >> 32),
        uint(c1 & 0xFFFFFFFFu),
        uint(c1 >> 32)
    );
}

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    // Dispatch(8, 8, numJobs) -> 64x64 blocks = 256x256 tile
    uint bx = dtid.x;
    uint by = dtid.y;
    uint jobIdx = dtid.z;
    if (bx >= 64u || by >= 64u) return;

    TileDecodeJob job = g_Jobs[jobIdx];
    uint m = job.matId & 31u;
    float mipDim = float(8192u >> job.mipLevel);
    float lod = float(job.mipLevel);
    float lodAtten = saturate(1.0f - lod * 0.25f);

    uint4 albBlock[16];
    uint4 ormBlock[16];
    uint nxArr[16];
    uint nyArr[16];

    [unroll]
    for (uint iy = 0u; iy < 4u; ++iy) {
        [unroll]
        for (uint ix = 0u; ix < 4u; ++ix) {
            uint px = job.tileX * 256u + bx * 4u + ix;
            uint py = job.tileY * 256u + by * 4u + iy;
            float2 uv = (float2(px, py) + 0.5f) / mipDim;

            float4 g0 = g_LatentG0[m].SampleLevel(g_LinearSampler, uv, lod);
            float4 g1 = g_LatentG1[m].SampleLevel(g_LinearSampler, uv, max(0.0f, lod - 2.0f));

            float pbr[8];
            eval_ntc_mlp(g_MlpWeights, m, g0, g1, uv, lodAtten, pbr);

            uint idx = iy * 4u + ix;
            uint r = min(255u, uint(round(pbr[0] * 255.0f)));
            uint g = min(255u, uint(round(pbr[1] * 255.0f)));
            uint b = min(255u, uint(round(pbr[2] * 255.0f)));
            uint ao = min(255u, uint(round(pbr[3] * 255.0f)));
            uint nx = min(255u, uint(round(pbr[4] * 255.0f)));
            uint ny = min(255u, uint(round(pbr[5] * 255.0f)));
            uint rough = min(255u, uint(round(pbr[6] * 255.0f)));
            uint metal = min(255u, uint(round(pbr[7] * 255.0f)));

            albBlock[idx] = uint4(r, g, b, ao);
            ormBlock[idx] = uint4(rough, metal, ao, 255u);
            nxArr[idx] = nx;
            nyArr[idx] = ny;
        }
    }

    uint blockOff = (by * 64u + bx) * 16u;
    g_DecodedTilesOut.Store4(job.dstAlbedoByteOffset + blockOff, pack_bc7_mode6(albBlock));
    g_DecodedTilesOut.Store4(job.dstNormalByteOffset + blockOff, pack_bc5(nxArr, nyArr));
    g_DecodedTilesOut.Store4(job.dstOrmByteOffset + blockOff, pack_bc7_mode6(ormBlock));
}
