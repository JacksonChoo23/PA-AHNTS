#ifndef HIDDEN_DIM
#define HIDDEN_DIM 32u
#endif

#ifndef OUTPUT_BCN
#define OUTPUT_BCN 1u
#endif

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

void unpack_8_fp16(uint4 p, out float16_t a[8]) {
    a[0] = float16_t(f16tof32(p.x & 0xFFFFu));
    a[1] = float16_t(f16tof32(p.x >> 16u));
    a[2] = float16_t(f16tof32(p.y & 0xFFFFu));
    a[3] = float16_t(f16tof32(p.y >> 16u));
    a[4] = float16_t(f16tof32(p.z & 0xFFFFu));
    a[5] = float16_t(f16tof32(p.z >> 16u));
    a[6] = float16_t(f16tof32(p.w & 0xFFFFu));
    a[7] = float16_t(f16tof32(p.w >> 16u));
}

void eval_param_mlp(uint matId, float4 g0Val, float4 g1Val, float2 worldUV, float lodAtten, out float outPbr[8]) {
    static const uint H = HIDDEN_DIM;
    static const uint G = H / 8u;
    static const uint W1_OFF = 0u;
    static const uint B1_OFF = W1_OFF + 16u * H * 2u;
    static const uint W2_OFF = B1_OFF + H * 2u;
    static const uint B2_OFF = W2_OFF + H * H * 2u;
    static const uint W3_OFF = B2_OFF + H * 2u;
    static const uint B3_OFF = W3_OFF + H * 8u * 2u;
    static const uint RAW_BYTES = B3_OFF + 8u * 2u;
    static const uint STRIDE_BYTES = ((RAW_BYTES + 255u) / 256u) * 256u;

    uint base = matId * STRIDE_BYTES;
    float2 tileUV = frac(worldUV * 16.0f);
    float twoPi = 6.2831853f;

    float16_t x[16];
    x[0]  = float16_t(g0Val.x * 2.0f - 1.0f);
    x[1]  = float16_t(g0Val.y * 2.0f - 1.0f);
    x[2]  = float16_t(g0Val.z * 2.0f - 1.0f);
    x[3]  = float16_t(g0Val.w * 2.0f - 1.0f);
    x[4]  = float16_t(g1Val.x);
    x[5]  = float16_t(g1Val.y);
    x[6]  = float16_t(g1Val.z);
    x[7]  = float16_t(g1Val.w);
    x[8]  = float16_t(sin(twoPi * 8.0f * tileUV.x) * lodAtten);
    x[9]  = float16_t(cos(twoPi * 8.0f * tileUV.x) * lodAtten);
    x[10] = float16_t(sin(twoPi * 8.0f * tileUV.y) * lodAtten);
    x[11] = float16_t(cos(twoPi * 8.0f * tileUV.y) * lodAtten);
    x[12] = float16_t(sin(twoPi * 16.0f * tileUV.x) * lodAtten);
    x[13] = float16_t(cos(twoPi * 16.0f * tileUV.x) * lodAtten);
    x[14] = float16_t(sin(twoPi * 16.0f * tileUV.y) * lodAtten);
    x[15] = float16_t(cos(twoPi * 16.0f * tileUV.y) * lodAtten);

    float16_t h1[HIDDEN_DIM];
    for (uint g = 0u; g < G; ++g) {
        float16_t v[8];
        unpack_8_fp16(g_MlpWeights.Load4(base + B1_OFF + g * 16u), v);
        [unroll] for (uint k = 0u; k < 8u; ++k) h1[g * 8u + k] = v[k];
    }
    for (uint i = 0u; i < 16u; ++i) {
        float16_t xi = x[i];
        for (uint g = 0u; g < G; ++g) {
            float16_t w[8];
            unpack_8_fp16(g_MlpWeights.Load4(base + W1_OFF + (i * G + g) * 16u), w);
            [unroll] for (uint k = 0u; k < 8u; ++k) h1[g * 8u + k] += xi * w[k];
        }
    }
    for (uint j = 0u; j < HIDDEN_DIM; ++j) {
        h1[j] = max(h1[j], h1[j] * float16_t(0.125f));
    }

    float16_t h2[HIDDEN_DIM];
    for (uint g = 0u; g < G; ++g) {
        float16_t v[8];
        unpack_8_fp16(g_MlpWeights.Load4(base + B2_OFF + g * 16u), v);
        [unroll] for (uint k = 0u; k < 8u; ++k) h2[g * 8u + k] = v[k];
    }
    for (uint i = 0u; i < HIDDEN_DIM; ++i) {
        float16_t hi = h1[i];
        for (uint g = 0u; g < G; ++g) {
            float16_t w[8];
            unpack_8_fp16(g_MlpWeights.Load4(base + W2_OFF + (i * G + g) * 16u), w);
            [unroll] for (uint k = 0u; k < 8u; ++k) h2[g * 8u + k] += hi * w[k];
        }
    }
    for (uint j = 0u; j < HIDDEN_DIM; ++j) {
        h2[j] = max(h2[j], h2[j] * float16_t(0.125f));
    }

    float16_t y[8];
    unpack_8_fp16(g_MlpWeights.Load4(base + B3_OFF), y);
    for (uint i = 0u; i < HIDDEN_DIM; ++i) {
        float16_t hi = h2[i];
        float16_t w[8];
        unpack_8_fp16(g_MlpWeights.Load4(base + W3_OFF + i * 16u), w);
        [unroll] for (uint k = 0u; k < 8u; ++k) y[k] += hi * w[k];
    }

    [unroll]
    for (uint k = 0u; k < 8u; ++k) {
        outPbr[k] = 1.0f / (1.0f + exp(-clamp(float(y[k]), -12.0f, 12.0f)));
    }
}

uint4 pack_bc7_mode6(uint4 rgba[16]) {
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

    [loop]
    for (uint iy = 0u; iy < 4u; ++iy) {
        [loop]
        for (uint ix = 0u; ix < 4u; ++ix) {
            uint px = job.tileX * 256u + bx * 4u + ix;
            uint py = job.tileY * 256u + by * 4u + iy;
            float2 uv = (float2(px, py) + 0.5f) / mipDim;

            float4 g0 = g_LatentG0[m].SampleLevel(g_LinearSampler, uv, lod);
            float4 g1 = g_LatentG1[m].SampleLevel(g_LinearSampler, uv, max(0.0f, lod - 2.0f));

            float pbr[8];
            eval_param_mlp(m, g0, g1, uv, lodAtten, pbr);

            uint idx = iy * 4u + ix;
            uint r = min(255u, uint(round(pbr[0] * 255.0f)));
            uint g = min(255u, uint(round(pbr[1] * 255.0f)));
            uint b = min(255u, uint(round(pbr[2] * 255.0f)));
            uint ao = min(255u, uint(round(pbr[3] * 255.0f)));
            uint nx = min(255u, uint(round(pbr[4] * 255.0f)));
            uint ny = min(255u, uint(round(pbr[5] * 255.0f)));
            uint rough = min(255u, uint(round(pbr[6] * 255.0f)));
            uint metal = min(255u, uint(round(pbr[7] * 255.0f)));

#if OUTPUT_BCN
            albBlock[idx] = uint4(r, g, b, ao);
            ormBlock[idx] = uint4(rough, metal, ao, 255u);
            nxArr[idx] = nx;
            nyArr[idx] = ny;
#else
            uint w0 = r | (g << 8u) | (b << 16u) | (ao << 24u);
            uint w1 = nx | (ny << 8u) | (rough << 16u) | (metal << 24u);
            uint tileOff = jobIdx * (256u * 256u * 8u);
            uint lx = bx * 4u + ix;
            uint ly = by * 4u + iy;
            uint pixOff = (ly * 256u + lx) * 8u;
            g_DecodedTilesOut.Store2(tileOff + pixOff, uint2(w0, w1));
#endif
        }
    }

#if OUTPUT_BCN
    uint blockOff = (by * 64u + bx) * 16u;
    g_DecodedTilesOut.Store4(job.dstAlbedoByteOffset + blockOff, pack_bc7_mode6(albBlock));
    g_DecodedTilesOut.Store4(job.dstNormalByteOffset + blockOff, pack_bc5(nxArr, nyArr));
    g_DecodedTilesOut.Store4(job.dstOrmByteOffset + blockOff, pack_bc7_mode6(ormBlock));
#endif
}
