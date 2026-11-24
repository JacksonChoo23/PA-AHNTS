#ifndef PBR_COMMON_HLSLI
#define PBR_COMMON_HLSLI

cbuffer FrameCB : register(b0) {
    row_major float4x4 g_ViewProj;
    float3   g_CamPos;
    float    g_TimeSec;
    uint     g_EnableFeedback;
    uint     g_UseTileResidency;
    uint     g_Pad0;
    uint     g_Pad1;
};

struct VSIn {
    float3 pos     : POSITION;
    float3 normal  : NORMAL;
    float3 tangent : TANGENT;
    float2 uv      : TEXCOORD0;
    uint   matId   : MATID;
};

struct VSOut {
    float4 pos      : SV_POSITION;
    float3 worldPos : TEXCOORD0;
    float3 normal   : TEXCOORD1;
    float3 tangent  : TEXCOORD2;
    float2 uv       : TEXCOORD3;
    nointerpolation uint matId : TEXCOORD4;
};

static const float PI = 3.14159265f;

float3 fresnel_schlick(float cosTheta, float3 F0) {
    return F0 + (1.0f - F0) * pow(saturate(1.0f - cosTheta), 5.0f);
}

float distribution_ggx(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / max(PI * denom * denom, 1e-5f);
}

float geometry_smith(float NdotV, float NdotL, float roughness) {
    float r = roughness + 1.0f;
    float k = (r * r) * 0.125f;
    float gv = NdotV / (NdotV * (1.0f - k) + k);
    float gl = NdotL / (NdotL * (1.0f - k) + k);
    return gv * gl;
}

float3 eval_dir_light(
    float3 N, float3 V, float3 L, float3 radiance,
    float3 albedo, float roughness, float metallic
) {
    float3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0f);
    float NdotV = max(dot(N, V), 1e-4f);
    float NdotH = max(dot(N, H), 0.0f);

    float3 F0 = lerp(float3(0.04f, 0.04f, 0.04f), albedo, metallic);
    float3 F = fresnel_schlick(max(dot(H, V), 0.0f), F0);
    float D = distribution_ggx(NdotH, roughness);
    float G = geometry_smith(NdotV, NdotL, roughness);

    float3 spec = (D * G * F) / max(4.0f * NdotV * NdotL, 1e-4f);
    float3 kD = (1.0f - F) * (1.0f - metallic);
    float3 diff = kD * albedo / PI;
    return (diff + spec) * radiance * NdotL;
}

float4 compute_pbr_lighting(
    float3 worldPos,
    float3 geoNormal,
    float3 geoTangent,
    float3 albedoSrgb,
    float  ao,
    float2 normalXYEnc,
    float  roughness,
    float  metallic
) {
    float3 albedo = pow(saturate(albedoSrgb), 2.2f);
    roughness = clamp(roughness, 0.06f, 0.98f);
    metallic = saturate(metallic);

    float nx = normalXYEnc.x * 2.0f - 1.0f;
    float ny = normalXYEnc.y * 2.0f - 1.0f;
    float nz = sqrt(saturate(1.0f - nx * nx - ny * ny));

    float3 N0 = normalize(geoNormal);
    float3 T0 = normalize(geoTangent - N0 * dot(N0, geoTangent));
    float3 B0 = cross(N0, T0);
    float3 N = normalize(T0 * nx + B0 * ny + N0 * nz);
    float3 V = normalize(g_CamPos - worldPos);

    // Three deterministic gallery lights (overhead warm key, cool fill, warm rim)
    float3 Lo = 0.0f;
    Lo += eval_dir_light(N, V, normalize(float3(0.35f, 0.88f, -0.32f)), float3(2.8f, 2.65f, 2.45f), albedo, roughness, metallic);
    Lo += eval_dir_light(N, V, normalize(float3(-0.55f, 0.45f, 0.70f)), float3(0.95f, 1.10f, 1.35f), albedo, roughness, metallic);

    // Local overhead corridor point light every 20m along Z
    float bayLightZ = floor(worldPos.z / 20.0f) * 20.0f + 10.0f;
    float3 lightPos = float3(0.0f, 5.2f, bayLightZ);
    float3 toL = lightPos - worldPos;
    float dist2 = max(dot(toL, toL), 1.0f);
    float3 Lp = toL * rsqrt(dist2);
    float atten = 18.0f / dist2;
    Lo += eval_dir_light(N, V, Lp, float3(1.4f, 1.25f, 1.05f) * atten, albedo, roughness, metallic);

    float3 ambient = float3(0.11f, 0.12f, 0.14f) * albedo * ao;
    float3 hdr = ambient + Lo;

    // ACES Filmic tone map + sRGB encode
    float3 a = hdr * (hdr + 0.0245786f) - 0.000090537f;
    float3 b = hdr * (0.983729f * hdr + 0.4329510f) + 0.238081f;
    float3 mapped = saturate(a / b);
    return float4(pow(mapped, 1.0f / 2.2f), 1.0f);
}

#endif
