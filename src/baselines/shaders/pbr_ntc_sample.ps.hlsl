#include "pbr_common.hlsli"
#include "ntc_mlp.hlsli"

Texture2D g_LatentG0[32] : register(t0, space0);
Texture2D g_LatentG1[32] : register(t0, space1);
ByteAddressBuffer g_MlpWeights : register(t0, space2);
SamplerState g_LinearSampler : register(s0);

float4 main(VSOut input) : SV_TARGET {
    uint m = input.matId & 31u;

    float2 dx = ddx(input.uv * 8192.0f);
    float2 dy = ddy(input.uv * 8192.0f);
    float rho = max(dot(dx, dx), dot(dy, dy));
    float worldLod = max(0.0f, 0.5f * log2(max(rho, 1.0f)));
    float lodAtten = saturate(1.0f - worldLod * 0.25f);

    float4 g0 = g_LatentG0[m].SampleLevel(g_LinearSampler, input.uv, worldLod);
    float4 g1 = g_LatentG1[m].SampleLevel(g_LinearSampler, input.uv, max(0.0f, worldLod - 2.0f));

    float pbr[8];
    eval_ntc_mlp(g_MlpWeights, m, g0, g1, input.uv, lodAtten, pbr);

    return compute_pbr_lighting(
        input.worldPos,
        input.normal,
        input.tangent,
        float3(pbr[0], pbr[1], pbr[2]),
        pbr[3],
        float2(pbr[4], pbr[5]),
        pbr[6],
        pbr[7]
    );
}
