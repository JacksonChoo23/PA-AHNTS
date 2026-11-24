#include "pbr_common.hlsli"

Texture2D g_Albedo[32] : register(t0, space0);
Texture2D g_Normal[32] : register(t0, space1);
Texture2D g_ORM[32]    : register(t0, space2);
FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> g_Feedback[32] : register(u0, space3);
StructuredBuffer<uint> g_TileResidencyMap : register(t0, space4);
SamplerState g_LinearSampler : register(s0);

float4 main(VSOut input) : SV_TARGET {
    uint m = input.matId & 31u;

    if (g_EnableFeedback != 0u) {
        g_Feedback[m].WriteSamplerFeedback(g_Albedo[m], g_LinearSampler, input.uv);
    }

    float minLod = 0.0f;
    if (g_UseTileResidency != 0u) {
        float2 wuv = frac(input.uv);
        uint tx = min(31u, uint(wuv.x * 32.0f));
        uint ty = min(31u, uint(wuv.y * 32.0f));
        minLod = float(g_TileResidencyMap[m * 1024u + ty * 32u + tx]);
    }

    float4 albAo = g_Albedo[m].Sample(g_LinearSampler, input.uv, int2(0, 0), minLod);
    float2 nxy   = g_Normal[m].Sample(g_LinearSampler, input.uv, int2(0, 0), minLod).rg;
    float3 orm   = g_ORM[m].Sample(g_LinearSampler, input.uv, int2(0, 0), minLod).rgb;

    return compute_pbr_lighting(
        input.worldPos,
        input.normal,
        input.tangent,
        albAo.rgb,
        albAo.a,
        nxy,
        orm.r,
        orm.g
    );
}
