#ifndef NTC_MLP_HLSLI
#define NTC_MLP_HLSLI

static const uint NTC_MLP_STRIDE_BYTES = 3840u;

void unpack_8_fp16(uint4 p, out float16_t a0, out float16_t a1, out float16_t a2, out float16_t a3,
                   out float16_t a4, out float16_t a5, out float16_t a6, out float16_t a7) {
    a0 = float16_t(f16tof32(p.x & 0xFFFFu));
    a1 = float16_t(f16tof32(p.x >> 16u));
    a2 = float16_t(f16tof32(p.y & 0xFFFFu));
    a3 = float16_t(f16tof32(p.y >> 16u));
    a4 = float16_t(f16tof32(p.z & 0xFFFFu));
    a5 = float16_t(f16tof32(p.z >> 16u));
    a6 = float16_t(f16tof32(p.w & 0xFFFFu));
    a7 = float16_t(f16tof32(p.w >> 16u));
}

void eval_ntc_mlp(
    ByteAddressBuffer mlpWeights,
    uint matId,
    float4 g0Val,
    float4 g1Val,
    float2 worldUV,
    float lodAtten,
    out float outPbr[8]
) {
    uint base = matId * NTC_MLP_STRIDE_BYTES;

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

    // Layer 1: b1 at offset 1024 (64 bytes = 4 uint4s)
    float16_t h1[32];
    [unroll]
    for (uint g = 0u; g < 4u; ++g) {
        uint4 pb = mlpWeights.Load4(base + 1024u + g * 16u);
        unpack_8_fp16(pb, h1[g*8+0], h1[g*8+1], h1[g*8+2], h1[g*8+3],
                          h1[g*8+4], h1[g*8+5], h1[g*8+6], h1[g*8+7]);
    }
    [unroll]
    for (uint i = 0u; i < 16u; ++i) {
        float16_t xi = x[i];
        [unroll]
        for (uint g = 0u; g < 4u; ++g) {
            uint4 pw = mlpWeights.Load4(base + (i * 4u + g) * 16u);
            float16_t w0,w1,w2,w3,w4,w5,w6,w7;
            unpack_8_fp16(pw, w0,w1,w2,w3,w4,w5,w6,w7);
            h1[g*8+0] += xi * w0; h1[g*8+1] += xi * w1;
            h1[g*8+2] += xi * w2; h1[g*8+3] += xi * w3;
            h1[g*8+4] += xi * w4; h1[g*8+5] += xi * w5;
            h1[g*8+6] += xi * w6; h1[g*8+7] += xi * w7;
        }
    }
    [unroll]
    for (uint j = 0u; j < 32u; ++j) {
        h1[j] = max(h1[j], h1[j] * float16_t(0.125f));
    }

    // Layer 2: w2 at offset 1088, b2 at offset 3136
    float16_t h2[32];
    [unroll]
    for (uint g = 0u; g < 4u; ++g) {
        uint4 pb = mlpWeights.Load4(base + 3136u + g * 16u);
        unpack_8_fp16(pb, h2[g*8+0], h2[g*8+1], h2[g*8+2], h2[g*8+3],
                          h2[g*8+4], h2[g*8+5], h2[g*8+6], h2[g*8+7]);
    }
    [unroll]
    for (uint i = 0u; i < 32u; ++i) {
        float16_t hi = h1[i];
        [unroll]
        for (uint g = 0u; g < 4u; ++g) {
            uint4 pw = mlpWeights.Load4(base + 1088u + (i * 4u + g) * 16u);
            float16_t w0,w1,w2,w3,w4,w5,w6,w7;
            unpack_8_fp16(pw, w0,w1,w2,w3,w4,w5,w6,w7);
            h2[g*8+0] += hi * w0; h2[g*8+1] += hi * w1;
            h2[g*8+2] += hi * w2; h2[g*8+3] += hi * w3;
            h2[g*8+4] += hi * w4; h2[g*8+5] += hi * w5;
            h2[g*8+6] += hi * w6; h2[g*8+7] += hi * w7;
        }
    }
    [unroll]
    for (uint j = 0u; j < 32u; ++j) {
        h2[j] = max(h2[j], h2[j] * float16_t(0.125f));
    }

    // Layer 3: w3 at offset 3200, b3 at offset 3712
    float16_t y[8];
    {
        uint4 pb = mlpWeights.Load4(base + 3712u);
        unpack_8_fp16(pb, y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]);
    }
    [unroll]
    for (uint i = 0u; i < 32u; ++i) {
        float16_t hi = h2[i];
        uint4 pw = mlpWeights.Load4(base + 3200u + i * 16u);
        float16_t w0,w1,w2,w3,w4,w5,w6,w7;
        unpack_8_fp16(pw, w0,w1,w2,w3,w4,w5,w6,w7);
        y[0] += hi * w0; y[1] += hi * w1; y[2] += hi * w2; y[3] += hi * w3;
        y[4] += hi * w4; y[5] += hi * w5; y[6] += hi * w6; y[7] += hi * w7;
    }

    [unroll]
    for (uint k = 0u; k < 8u; ++k) {
        outPbr[k] = 1.0f / (1.0f + exp(-clamp(float(y[k]), -12.0f, 12.0f)));
    }
}

#endif
