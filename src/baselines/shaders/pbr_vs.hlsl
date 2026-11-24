#include "pbr_common.hlsli"

VSOut main(VSIn input) {
    VSOut o;
    o.pos = mul(float4(input.pos, 1.0f), g_ViewProj);
    o.worldPos = input.pos;
    o.normal = input.normal;
    o.tangent = input.tangent;
    o.uv = input.uv;
    o.matId = input.matId;
    return o;
}
