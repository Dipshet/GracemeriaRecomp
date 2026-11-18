// Ambient occlusion (AC6_AO): the passes of the renderer's own path, XeGTAO. Included by
// internal_shaders.hlsl for its AO_* variants.
//
// XeGTAO is Intel's implementation of ground truth ambient occlusion ("Practical Real-Time Strategies for Accurate
// Indirect Occlusion", Jimenez et al. 2016): thirdparty/XeGTAO (MIT, Copyright (C) 2016-2021 Intel Corporation, Filip
// Strugar, Steve Mccalla). Its passes are compute shaders writing storage images. The renderer's passes are pixel
// shaders over bindless textures, so the functions below are XeGTAO.hlsli's (its depth filter, its main pass, its
// denoiser, their helpers) with these changes:
//  - each writes its result as the pixel shader's output, and reads texels by position (Load) where the original
//    gathers or samples with a point sampler: the same texels,
//  - the view-space depth's mip levels are five images (the renderer's images have one level), chosen per sample,
//  - 32-bit depths throughout (view distances reach past a 16-bit float's 65,504) and normals made in the main pass
//    (the original's XE_GTAO_FP32_DEPTHS and XE_GTAO_GENERATE_NORMALS_INPLACE paths), no bent normals,
//  - the radius grows with the view distance (radius = max(the base, (distance - a near distance) x a slope), up to a
//    cap): a radius of a metre or so is under a pixel from a thousand metres up, where a city's blocks are the detail.
//    The base serves the cockpit and the hangar.
//
//  AO_DEPTH      the scene's depth -> view-space depth, level 0
//  AO_DEPTH_MIP  level n -> level n + 1 (XeGTAO's weighted depth filter)
//  AO_MAIN       -> the visibility (/ 1.5) and the packed edges
//  AO_DENOISE    XeGTAO's edge-aware blur, one pass (the last one scales the visibility back)
//  AO_APPLY      the scene's colour taken towards the game's shadow blend by the occlusion (XeGTAO's), into the scene
//  AO_SUBJECT    the hangar: the aircraft's own depth where it is the surface seen, nothing elsewhere
#define XE_GTAO_USE_HALF_FLOAT_PRECISION 0
#define XE_GTAO_FP32_DEPTHS 1
#define XE_GTAO_USE_DEFAULT_CONSTANTS 0
#include "../../../thirdparty/XeGTAO/XeGTAO.h"

Texture2D<float4> g_Textures[] : register(t0, space0);
[[vk::binding(0, 0)]] Texture2DMS<float4> g_TexturesMS[];   // (the same textures: the terrain's depth where it is multisampled)
SamplerState g_Samplers[] : register(s0, space3);
struct PushConstants { uint textureIndex; uint samplerIndex; int offsetX; int offsetY; uint grade0; uint grade1; uint stripes; };
[[vk::push_constant]] ConstantBuffer<PushConstants> g_Push;
struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

// The parameters (renderer.cpp RunAmbientOcclusion), at the device address in offsetX / offsetY.
struct AoParams
{
    GTAOConstants consts;
    float radiusSlope;       // the radius per unit of view distance
    float slices, steps;
    float clipZ;             // depth x view distance (the game's reversed depth with an infinite far plane)
    float farthest;          // the view distance the sky is given
    uint depthLevels[5];     // the view-space depth's images
    float strength;
    float fadeFrom, fadeTo;  // view distances between which the occlusion fades out
    float colourCap;         // the 8-bit scene's values above this are its emitter codes: left alone
    float debug;
    float last;              // the denoiser's last pass
    uint terrainDepth;       // the terrain's own depth: a pixel of the scene with that depth is terrain
    float terrainMode;       // 0: no occlusion on the terrain, 1: what objects cast on it only, 2: all (no terrain depth)
    float terrainMask;       // the apply pass leaves the terrain's pixels alone
    float terrainSamples;    // the terrain's depth is a multisampled image (read by its first sample)
    float radiusMax;         // the radius's cap in world units (0: none)
    float4 shadow;           // full occlusion leaves colour x (1 - a) + rgb: the game's own shadow blend
    float radiusNear;        // the view distance the radius starts growing from
};

AoParams LoadParams()
{
    const uint64_t a = (uint64_t(asuint(g_Push.offsetY)) << 32) | asuint(g_Push.offsetX);
    float4 r[12];
    [unroll] for (uint k = 0; k < 12; k++)
        r[k] = vk::RawBufferLoad<float4>(a + k * 16);
    AoParams p;
    p.consts.ViewportSize = int2(r[0].xy);
    p.consts.ViewportPixelSize = r[0].zw;
    p.consts.DepthUnpackConsts = 0.0;
    p.consts.CameraTanHalfFOV = float2(-r[1].z, r[1].w);
    p.consts.NDCToViewMul = r[1].xy;
    p.consts.NDCToViewAdd = r[1].zw;
    p.consts.NDCToViewMul_x_PixelSize = r[2].xy;
    p.consts.EffectRadius = r[2].z;
    p.radiusSlope = r[2].w;
    p.consts.EffectFalloffRange = r[3].x;
    p.consts.RadiusMultiplier = r[3].y;
    p.consts.Padding0 = 0.0;
    p.consts.FinalValuePower = r[3].z;
    p.consts.DenoiseBlurBeta = r[3].w;
    p.consts.SampleDistributionPower = r[4].x;
    p.consts.ThinOccluderCompensation = r[4].y;
    p.consts.DepthMIPSamplingOffset = r[4].z;
    p.consts.NoiseIndex = int(r[4].w);
    p.slices = r[5].x;
    p.steps = r[5].y;
    p.clipZ = r[5].z;
    p.farthest = r[5].w;
    p.depthLevels[0] = asuint(r[6].x);
    p.depthLevels[1] = asuint(r[6].y);
    p.depthLevels[2] = asuint(r[6].z);
    p.depthLevels[3] = asuint(r[6].w);
    p.depthLevels[4] = asuint(r[7].x);
    p.strength = r[7].y;
    p.fadeFrom = r[7].z;
    p.fadeTo = r[7].w;
    p.colourCap = r[8].x;
    p.debug = r[8].y;
    p.last = r[8].z;
    p.terrainDepth = asuint(r[8].w);
    p.terrainMode = r[9].x;
    p.terrainMask = r[9].y;
    p.terrainSamples = r[9].z;
    p.radiusMax = r[9].w;
    p.shadow = r[10];
    p.radiusNear = r[11].x;
    return p;
}

// Whether the scene's depth d, read at the texel `at`, is the terrain's: the terrain's draws are issued twice,
// with the same vertex shader and vertices, so the depths are equal but for what another pipeline may do to the
// arithmetic (a relative difference of a millionth, a few steps of a 32-bit float, is allowed). The same texel in
// both: the terrain's image can be the larger one (a single-sampled copy kept from a larger scene), its picture in the
// top left as the scene's depth is.
bool AoIsTerrain(const AoParams p, int2 at, float d)
{
    if (p.terrainMode >= 2.0 || d <= 0.0)
        return false;
    float t;
    if (p.terrainSamples != 0.0)
    {
        uint tw, th, samples;
        g_TexturesMS[p.terrainDepth].GetDimensions(tw, th, samples);
        t = g_TexturesMS[p.terrainDepth].Load(min(at, int2(tw, th) - 1), 0).x;
    }
    else
    {
        uint tw, th;
        g_Textures[p.terrainDepth].GetDimensions(tw, th);
        t = g_Textures[p.terrainDepth].Load(int3(min(at, int2(tw, th) - 1), 0)).x;
    }
    return abs(t - d) <= d * 1.0e-6;
}

// The radius at a view distance (the change described above).
float AoRadius(const AoParams p, float viewZ)
{
    const float radius = max(p.consts.EffectRadius, (viewZ - p.radiusNear) * p.radiusSlope);
    return p.radiusMax > 0.0 ? min(radius, max(p.radiusMax, p.consts.EffectRadius)) : radius;
}

// --- XeGTAO.hlsli's helpers, as they are ---------------------------------------------------------------------------
float3 XeGTAO_ComputeViewspacePosition(const float2 screenPos, const float viewspaceDepth, const GTAOConstants consts)
{
    float3 ret;
    ret.xy = (consts.NDCToViewMul * screenPos.xy + consts.NDCToViewAdd) * viewspaceDepth;
    ret.z = viewspaceDepth;
    return ret;
}

float4 XeGTAO_CalculateEdges(const float centerZ, const float leftZ, const float rightZ, const float topZ, const float bottomZ)
{
    float4 edgesLRTB = float4(leftZ, rightZ, topZ, bottomZ) - centerZ;
    float slopeLR = (edgesLRTB.y - edgesLRTB.x) * 0.5;
    float slopeTB = (edgesLRTB.w - edgesLRTB.z) * 0.5;
    float4 edgesLRTBSlopeAdjusted = edgesLRTB + float4(slopeLR, -slopeLR, slopeTB, -slopeTB);
    edgesLRTB = min(abs(edgesLRTB), abs(edgesLRTBSlopeAdjusted));
    return float4(saturate((1.25 - edgesLRTB / (centerZ * 0.011))));
}

// packing/unpacking for edges: 2 bits per edge mean 4 gradient values (0, 0.33, 0.66, 1) for smoother transitions!
float XeGTAO_PackEdges(float4 edgesLRTB)
{
    edgesLRTB = round(saturate(edgesLRTB) * 2.9);
    return dot(edgesLRTB, float4(64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0));
}

float4 XeGTAO_UnpackEdges(float _packedVal)
{
    uint packedVal = (uint)(_packedVal * 255.5);
    float4 edgesLRTB;
    edgesLRTB.x = float((packedVal >> 6) & 0x03) / 3.0;
    edgesLRTB.y = float((packedVal >> 4) & 0x03) / 3.0;
    edgesLRTB.z = float((packedVal >> 2) & 0x03) / 3.0;
    edgesLRTB.w = float((packedVal >> 0) & 0x03) / 3.0;
    return saturate(edgesLRTB);
}

float3 XeGTAO_CalculateNormal(const float4 edgesLRTB, float3 pixCenterPos, float3 pixLPos, float3 pixRPos, float3 pixTPos, float3 pixBPos)
{
    // Get this pixel's viewspace normal
    float4 acceptedNormals = saturate(float4(edgesLRTB.x * edgesLRTB.z, edgesLRTB.z * edgesLRTB.y, edgesLRTB.y * edgesLRTB.w, edgesLRTB.w * edgesLRTB.x) + 0.01);
    pixLPos = normalize(pixLPos - pixCenterPos);
    pixRPos = normalize(pixRPos - pixCenterPos);
    pixTPos = normalize(pixTPos - pixCenterPos);
    pixBPos = normalize(pixBPos - pixCenterPos);
    float3 pixelNormal = acceptedNormals.x * cross(pixLPos, pixTPos) + acceptedNormals.y * cross(pixTPos, pixRPos) +
        acceptedNormals.z * cross(pixRPos, pixBPos) + acceptedNormals.w * cross(pixBPos, pixLPos);
    return normalize(pixelNormal);
}

// http://h14s.p5r.org/2012/09/0x5f3759df.html, [Drobot2014a] Low Level Optimizations for High-Level Optimizations
float XeGTAO_FastSqrt(float x)
{
    return (float)(asfloat(0x1fbd1df5 + (asint(x) >> 1)));
}

// input [-1, 1] and output [0, PI], from https://seblagarde.wordpress.com/2014/12/01/inverse-trigonometric-functions-gpu-optimization-for-amd-gcn-architecture/
float XeGTAO_FastACos(float inX)
{
    const float PI = 3.141593;
    const float HALF_PI = 1.570796;
    float x = abs(inX);
    float res = -0.156583 * x + HALF_PI;
    res *= XeGTAO_FastSqrt(1.0 - x);
    return (inX >= 0) ? res : PI - res;
}

#define XE_GTAO_PI      (3.1415926535897932384626433832795)
#define XE_GTAO_PI_HALF (1.5707963267948966192313216916398)

// (The original's sample code: the Hilbert curve driving the R2 sequence.)
float2 SpatioTemporalNoise(uint2 pixCoord, uint temporalIndex)   // without TAA, temporalIndex is always 0
{
    uint index = HilbertIndex(pixCoord.x % 64, pixCoord.y % 64);
    index += 288 * (temporalIndex % 64);
    return float2(frac(0.5 + index * float2(0.75487766624669276005, 0.5698402909980532659114)));
}

#if defined(AO_SUBJECT)
// The hangar's aircraft alone: the depth the occlusion is made from and applied by. textureIndex =
// the scene's depth. The aircraft's own depth (its draws again, into a depth image of their own) is the parameters'
// terrainDepth. Where both are the same the aircraft is the surface seen: its depth, elsewhere 0 (as the sky: nothing
// occludes it, it occludes nothing, the apply pass leaves it alone).
float4 main(VsOut i) : SV_Target0
{
    const AoParams p = LoadParams();
    Texture2D<float4> sceneImage = g_Textures[g_Push.textureIndex];
    uint w, h;
    sceneImage.GetDimensions(w, h);
    const int2 at = min(int2(i.pos.xy), int2(w, h) - 1);
    const float d = sceneImage.Load(int3(at, 0)).x;
    AoParams q = p;
    q.terrainMode = 0.0;
    return float4(AoIsTerrain(q, at, d) ? d : 0.0, 0.0, 0.0, 1.0);
}
#elif defined(AO_DEPTH)
float4 main(VsOut i) : SV_Target0
{
    const AoParams p = LoadParams();
    Texture2D<float4> depthImage = g_Textures[g_Push.textureIndex];
    float2 size;
    depthImage.GetDimensions(size.x, size.y);
    const int2 at = min(int2(i.uv * size), int2(size) - 1);
    const float d = depthImage.Load(int3(at, 0)).x;
    // (XeGTAO_ClampDepth: the sky, depth 0, at the farthest.) The terrain's pixels: the distance negated.
    const float viewZ = d > 0.0 ? min(p.clipZ / d, p.farthest) : p.farthest;
    return float4(AoIsTerrain(p, at, d) ? -viewZ : viewZ, 0.0, 0.0, 1.0);
}
#elif defined(AO_DEPTH_MIP)
// weighted average depth filter (XeGTAO_DepthMIPFilter, the radius by the distance)
float4 main(VsOut i) : SV_Target0
{
    const AoParams p = LoadParams();
    Texture2D<float4> source = g_Textures[g_Push.textureIndex];
    uint w, h;
    source.GetDimensions(w, h);
    const int2 top = int2(w, h) - 1;
    const int2 at = int2(i.pos.xy) * 2;
    const float4 raw = float4(source.Load(int3(min(at, top), 0)).x, source.Load(int3(min(at + int2(1, 0), top), 0)).x,
        source.Load(int3(min(at + int2(0, 1), top), 0)).x, source.Load(int3(min(at + int2(1, 1), top), 0)).x);
    // The terrain's texels are negative. With the terrain taking only what objects cast (terrainMode 1) a level's
    // texel is terrain where all four are, and the filter of the others alone where any isn't: objects stay in the
    // levels for the terrain's pixels to find. Otherwise the levels above 0 hold distances only.
    float4 use = 1.0;
    float side = 1.0;
    if (p.terrainMode == 1.0)
    {
        if (any(raw > 0.0))
            use = step(0.0, raw);
        else
            side = -1.0;
    }
    const float depth0 = abs(raw.x), depth1 = abs(raw.y), depth2 = abs(raw.z), depth3 = abs(raw.w);
    float maxDepth = max(max(depth0 * use.x, depth1 * use.y), max(depth2 * use.z, depth3 * use.w));
    const float depthRangeScaleFactor = 0.75; // found empirically :)
    const float effectRadius = depthRangeScaleFactor * AoRadius(p, maxDepth) * p.consts.RadiusMultiplier;
    const float falloffRange = p.consts.EffectFalloffRange * effectRadius;
    const float falloffFrom = effectRadius * (1.0 - p.consts.EffectFalloffRange);
    // fadeout precompute optimisation
    const float falloffMul = -1.0 / falloffRange;
    const float falloffAdd = falloffFrom / falloffRange + 1.0;
    float weight0 = use.x * saturate((maxDepth - depth0) * falloffMul + falloffAdd);
    float weight1 = use.y * saturate((maxDepth - depth1) * falloffMul + falloffAdd);
    float weight2 = use.z * saturate((maxDepth - depth2) * falloffMul + falloffAdd);
    float weight3 = use.w * saturate((maxDepth - depth3) * falloffMul + falloffAdd);
    float weightSum = weight0 + weight1 + weight2 + weight3;
    return float4(side * (weight0 * depth0 + weight1 * depth1 + weight2 * depth2 + weight3 * depth3) / weightSum, 0.0, 0.0, 1.0);
}
#elif defined(AO_MAIN)
// The view-space depth at a screen position and mip level (the original: SampleLevel with a point sampler).
float DepthAt(const AoParams p, float2 uv, float mipLevel)
{
    const uint level = min(uint(mipLevel + 0.5), 4u);
    const int2 size = max(p.consts.ViewportSize >> level, int2(1, 1));
    return g_Textures[p.depthLevels[level]].Load(int3(clamp(int2(floor(uv * float2(size))), int2(0, 0), size - 1), 0)).x;
}

// XeGTAO_MainPass: the visibility (/ XE_GTAO_OCCLUSION_TERM_SCALE) and the packed edges.
float4 main(VsOut i) : SV_Target0
{
    const AoParams p = LoadParams();
    const GTAOConstants consts = p.consts;
    const uint2 pixCoord = uint2(i.pos.xy);
    const float sliceCount = p.slices, stepsPerSlice = p.steps;
    const float2 localNoise = SpatioTemporalNoise(pixCoord, uint(consts.NoiseIndex));
    float2 normalizedScreenPos = (pixCoord + 0.5.xx) * consts.ViewportPixelSize;

    Texture2D<float4> depth0 = g_Textures[p.depthLevels[0]];
    const int2 top = consts.ViewportSize - 1;
    const int2 c = int2(pixCoord);
    // viewspace Z at the center (negative: the terrain's)
    const float centreRaw = depth0.Load(int3(c, 0)).x;
    float viewspaceZ = abs(centreRaw);
    // viewspace Zs left top right bottom
    const float pixLZ = abs(depth0.Load(int3(max(c - int2(1, 0), int2(0, 0)), 0)).x);
    const float pixTZ = abs(depth0.Load(int3(max(c - int2(0, 1), int2(0, 0)), 0)).x);
    const float pixRZ = abs(depth0.Load(int3(min(c + int2(1, 0), top), 0)).x);
    const float pixBZ = abs(depth0.Load(int3(min(c + int2(0, 1), top), 0)).x);

    float4 edgesLRTB = XeGTAO_CalculateEdges(viewspaceZ, pixLZ, pixRZ, pixTZ, pixBZ);
    const float packedEdges = XeGTAO_PackEdges(edgesLRTB);
    // (The sky: nothing occludes it and it occludes nothing.)
    if (viewspaceZ >= p.farthest)
        return float4(1.0 / XE_GTAO_OCCLUSION_TERM_SCALE, packedEdges, 0.0, 1.0);
    // (The terrain: no occlusion on it, or only what isn't terrain occludes it.)
    if (centreRaw < 0.0 && p.terrainMode == 0.0)
        return float4(1.0 / XE_GTAO_OCCLUSION_TERM_SCALE, packedEdges, 0.0, 1.0);
    const bool objectsOnly = centreRaw < 0.0 && p.terrainMode == 1.0;

    float3 CENTER = XeGTAO_ComputeViewspacePosition(normalizedScreenPos, viewspaceZ, consts);
    float3 LEFT = XeGTAO_ComputeViewspacePosition(normalizedScreenPos + float2(-1, 0) * consts.ViewportPixelSize, pixLZ, consts);
    float3 RIGHT = XeGTAO_ComputeViewspacePosition(normalizedScreenPos + float2(1, 0) * consts.ViewportPixelSize, pixRZ, consts);
    float3 TOP = XeGTAO_ComputeViewspacePosition(normalizedScreenPos + float2(0, -1) * consts.ViewportPixelSize, pixTZ, consts);
    float3 BOTTOM = XeGTAO_ComputeViewspacePosition(normalizedScreenPos + float2(0, 1) * consts.ViewportPixelSize, pixBZ, consts);
    float3 viewspaceNormal = XeGTAO_CalculateNormal(edgesLRTB, CENTER, LEFT, RIGHT, TOP, BOTTOM);

    // Move center pixel slightly towards camera to avoid imprecision artifacts due to depth buffer imprecision. Offset depends on depth texture format used
    viewspaceZ *= 0.99999;     // this is good for FP32 depth buffer

    const float3 pixCenterPos = XeGTAO_ComputeViewspacePosition(normalizedScreenPos, viewspaceZ, consts);
    const float3 viewVec = normalize(-pixCenterPos);

    const float effectRadius = AoRadius(p, viewspaceZ) * consts.RadiusMultiplier;
    const float sampleDistributionPower = consts.SampleDistributionPower;
    const float thinOccluderCompensation = consts.ThinOccluderCompensation;
    const float falloffRange = consts.EffectFalloffRange * effectRadius;
    const float falloffFrom = effectRadius * (1.0 - consts.EffectFalloffRange);
    // fadeout precompute optimisation
    const float falloffMul = -1.0 / falloffRange;
    const float falloffAdd = falloffFrom / falloffRange + 1.0;

    float visibility = 0;
    // see "Algorithm 1" in https://www.activision.com/cdn/research/Practical_Real_Time_Strategies_for_Accurate_Indirect_Occlusion_NEW%20VERSION_COLOR.pdf
    {
        const float noiseSlice = localNoise.x;
        const float noiseSample = localNoise.y;

        // quality settings / tweaks / hacks
        const float pixelTooCloseThreshold = 1.3;      // if the offset is under approx pixel size (pixelTooCloseThreshold), push it out to the minimum distance

        // approx viewspace pixel size at pixCoord: approximation of NDCToViewspace( normalizedScreenPos.xy + consts.ViewportPixelSize.xy, pixCenterPos.z ).xy - pixCenterPos.xy;
        const float2 pixelDirRBViewspaceSizeAtCenterZ = viewspaceZ.xx * consts.NDCToViewMul_x_PixelSize;

        float screenspaceRadius = effectRadius / pixelDirRBViewspaceSizeAtCenterZ.x;

        // fade out for small screen radii
        visibility += saturate((10 - screenspaceRadius) / 100) * 0.5;

        const float minS = pixelTooCloseThreshold / screenspaceRadius;

        [loop] for (float slice = 0; slice < sliceCount; slice++)
        {
            float sliceK = (slice + noiseSlice) / sliceCount;
            // lines 5, 6 from the paper
            float phi = sliceK * XE_GTAO_PI;
            float cosPhi = cos(phi);
            float sinPhi = sin(phi);
            float2 omega = float2(cosPhi, -sinPhi);

            // convert to screen units (pixels) for later use
            omega *= screenspaceRadius;

            // line 8 from the paper
            const float3 directionVec = float3(cosPhi, sinPhi, 0);
            // line 9 from the paper
            const float3 orthoDirectionVec = directionVec - (dot(directionVec, viewVec) * viewVec);
            // line 10 from the paper
            // axisVec is orthogonal to directionVec and viewVec, used to define projectedNormal
            const float3 axisVec = normalize(cross(orthoDirectionVec, viewVec));
            // line 11 from the paper
            float3 projectedNormalVec = viewspaceNormal - axisVec * dot(viewspaceNormal, axisVec);
            // line 13 from the paper
            float signNorm = sign(dot(orthoDirectionVec, projectedNormalVec));
            // line 14 from the paper
            float projectedNormalVecLength = length(projectedNormalVec);
            float cosNorm = saturate(dot(projectedNormalVec, viewVec) / projectedNormalVecLength);
            // line 15 from the paper
            float n = signNorm * XeGTAO_FastACos(cosNorm);

            // this is a lower weight target. Not using -1 as in the original paper because it is under horizon, so a 'weight' has different meaning based on the normal
            const float lowHorizonCos0 = cos(n + XE_GTAO_PI_HALF);
            const float lowHorizonCos1 = cos(n - XE_GTAO_PI_HALF);

            // lines 17, 18 from the paper, manually unrolled the 'side' loop
            float horizonCos0 = lowHorizonCos0; //-1;
            float horizonCos1 = lowHorizonCos1; //-1;

            [loop] for (float step = 0; step < stepsPerSlice; step++)
            {
                // R1 sequence (http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/)
                const float stepBaseNoise = float(slice + step * stepsPerSlice) * 0.6180339887498948482;
                float stepNoise = frac(noiseSample + stepBaseNoise);

                // approx line 20 from the paper, with added noise
                float s = (step + stepNoise) / (stepsPerSlice);

                // additional distribution modifier
                s = pow(s, sampleDistributionPower);

                // avoid sampling center pixel
                s += minS;

                // approx lines 21-22 from the paper, unrolled
                float2 sampleOffset = s * omega;

                float sampleOffsetLength = length(sampleOffset);

                // note: when sampling, using point_point_point or point_point_linear sampler works, but linear_linear_linear will cause unwanted interpolation between neighbouring depth values on the same MIP level!
                const float mipLevel = clamp(log2(sampleOffsetLength) - consts.DepthMIPSamplingOffset, 0, XE_GTAO_DEPTH_MIP_LEVELS);

                // Snap to pixel center (more correct direction math, avoids artifacts due to sampling pos not matching depth texel center - messes up slope - but adds other
                // artifacts due to them being pushed off the slice). Also use full precision for high res cases.
                sampleOffset = round(sampleOffset) * consts.ViewportPixelSize;

                float2 sampleScreenPos0 = normalizedScreenPos + sampleOffset;
                const float raw0 = DepthAt(p, sampleScreenPos0, mipLevel);
                float SZ0 = abs(raw0);
                float3 samplePos0 = XeGTAO_ComputeViewspacePosition(sampleScreenPos0, SZ0, consts);

                float2 sampleScreenPos1 = normalizedScreenPos - sampleOffset;
                const float raw1 = DepthAt(p, sampleScreenPos1, mipLevel);
                float SZ1 = abs(raw1);
                float3 samplePos1 = XeGTAO_ComputeViewspacePosition(sampleScreenPos1, SZ1, consts);

                float3 sampleDelta0 = (samplePos0 - float3(pixCenterPos));
                float3 sampleDelta1 = (samplePos1 - float3(pixCenterPos));
                float sampleDist0 = length(sampleDelta0);
                float sampleDist1 = length(sampleDelta1);

                // approx lines 23, 24 from the paper, unrolled
                float3 sampleHorizonVec0 = sampleDelta0 / sampleDist0;
                float3 sampleHorizonVec1 = sampleDelta1 / sampleDist1;

                // any sample out of radius should be discarded - also use fallof range for smooth transitions. This is a modified idea from "4.3 Implementation details, Bounding the sampling area"
                // this is our own thickness heuristic that relies on sooner discarding samples behind the center
                float falloffBase0 = length(float3(sampleDelta0.x, sampleDelta0.y, sampleDelta0.z * (1 + thinOccluderCompensation)));
                float falloffBase1 = length(float3(sampleDelta1.x, sampleDelta1.y, sampleDelta1.z * (1 + thinOccluderCompensation)));
                float weight0 = saturate(falloffBase0 * falloffMul + falloffAdd);
                float weight1 = saturate(falloffBase1 * falloffMul + falloffAdd);
                // (The terrain doesn't occlude the terrain.)
                if (objectsOnly)
                {
                    weight0 = raw0 < 0.0 ? 0.0 : weight0;
                    weight1 = raw1 < 0.0 ? 0.0 : weight1;
                }

                // sample horizon cos
                float shc0 = dot(sampleHorizonVec0, viewVec);
                float shc1 = dot(sampleHorizonVec1, viewVec);

                // discard unwanted samples
                shc0 = lerp(lowHorizonCos0, shc0, weight0);
                shc1 = lerp(lowHorizonCos1, shc1, weight1);

                // this is a version where thicknessHeuristic is completely disabled
                horizonCos0 = max(horizonCos0, shc0);
                horizonCos1 = max(horizonCos1, shc1);
            }

            // I can't figure out the slight overdarkening on high slopes, so I'm adding this fudge - in the training set, 0.05 is close (PSNR 21.34) to disabled (PSNR 21.45)
            projectedNormalVecLength = lerp(projectedNormalVecLength, 1, 0.05);

            // line ~27, unrolled
            float h0 = -XeGTAO_FastACos(horizonCos1);
            float h1 = XeGTAO_FastACos(horizonCos0);
            // (XeGTAO skips clamping h0 and h1 to n -+ PI / 2 "for a tiny little bit more performance".)
            float iarc0 = (cosNorm + 2.0 * h0 * sin(n) - cos(2.0 * h0 - n)) / 4.0;
            float iarc1 = (cosNorm + 2.0 * h1 * sin(n) - cos(2.0 * h1 - n)) / 4.0;
            float localVisibility = projectedNormalVecLength * (iarc0 + iarc1);
            visibility += localVisibility;
        }
        visibility /= sliceCount;
        visibility = pow(visibility, consts.FinalValuePower);
        visibility = max(0.03, visibility); // disallow total occlusion (which wouldn't make any sense anyhow since pixel is visible but also helps with packing bent normals)
    }
    // (XeGTAO_OutputWorkingTerm)
    return float4(saturate(visibility / XE_GTAO_OCCLUSION_TERM_SCALE), packedEdges, 0.0, 1.0);
}
#elif defined(AO_DENOISE)
// XeGTAO_Denoise for one pixel: textureIndex = the visibility and the packed edges.
float4 main(VsOut i) : SV_Target0
{
    const AoParams p = LoadParams();
    Texture2D<float4> source = g_Textures[g_Push.textureIndex];
    const bool finalApply = p.last != 0.0;
    const float blurAmount = finalApply ? p.consts.DenoiseBlurBeta : p.consts.DenoiseBlurBeta / 5.0;
    const float diagWeight = 0.85 * 0.5;
    const int2 c = int2(i.pos.xy), top = p.consts.ViewportSize - 1;
#define AO_TEXEL(dx, dy) source.Load(int3(clamp(c + int2(dx, dy), int2(0, 0), top), 0)).xy
    const float2 centre = AO_TEXEL(0, 0), left = AO_TEXEL(-1, 0), right = AO_TEXEL(1, 0), above = AO_TEXEL(0, -1), below = AO_TEXEL(0, 1);
    float4 edgesL_LRTB = XeGTAO_UnpackEdges(left.y);
    float4 edgesT_LRTB = XeGTAO_UnpackEdges(above.y);
    float4 edgesR_LRTB = XeGTAO_UnpackEdges(right.y);
    float4 edgesB_LRTB = XeGTAO_UnpackEdges(below.y);
    float4 edgesC_LRTB = XeGTAO_UnpackEdges(centre.y);

    // Edges aren't perfectly symmetrical: edge detection algorithm does not guarantee that a left edge on the right pixel will match the right edge on the left pixel (although
    // they will match in majority of cases). This line further enforces the symmetricity, creating a slightly sharper blur. Works real nice with TAA.
    edgesC_LRTB *= float4(edgesL_LRTB.y, edgesR_LRTB.x, edgesT_LRTB.w, edgesB_LRTB.z);

    // this allows some small amount of AO leaking from neighbours if there are 3 or 4 edges. This reduces both spatial and temporal aliasing
    const float leak_threshold = 2.5; const float leak_strength = 0.5;
    float edginess = (saturate(4.0 - leak_threshold - dot(edgesC_LRTB, 1.xxxx)) / (4 - leak_threshold)) * leak_strength;
    edgesC_LRTB = saturate(edgesC_LRTB + edginess);

    // for diagonals, used by first and second pass
    const float weightTL = diagWeight * (edgesC_LRTB.x * edgesL_LRTB.z + edgesC_LRTB.z * edgesT_LRTB.x);
    const float weightTR = diagWeight * (edgesC_LRTB.z * edgesT_LRTB.y + edgesC_LRTB.y * edgesR_LRTB.z);
    const float weightBL = diagWeight * (edgesC_LRTB.w * edgesB_LRTB.x + edgesC_LRTB.x * edgesL_LRTB.w);
    const float weightBR = diagWeight * (edgesC_LRTB.y * edgesR_LRTB.w + edgesC_LRTB.w * edgesB_LRTB.y);

    float sumWeight = blurAmount;
    float sum = centre.x * sumWeight;
    sum += left.x * edgesC_LRTB.x;              sumWeight += edgesC_LRTB.x;
    sum += right.x * edgesC_LRTB.y;             sumWeight += edgesC_LRTB.y;
    sum += above.x * edgesC_LRTB.z;             sumWeight += edgesC_LRTB.z;
    sum += below.x * edgesC_LRTB.w;             sumWeight += edgesC_LRTB.w;
    sum += AO_TEXEL(-1, -1).x * weightTL;       sumWeight += weightTL;
    sum += AO_TEXEL(1, -1).x * weightTR;        sumWeight += weightTR;
    sum += AO_TEXEL(-1, 1).x * weightBL;        sumWeight += weightBL;
    sum += AO_TEXEL(1, 1).x * weightBR;         sumWeight += weightBR;
    float visibility = sum / sumWeight;
    // (XeGTAO_Output: the last pass scales the visibility back.)
    if (finalApply)
        visibility *= XE_GTAO_OCCLUSION_TERM_SCALE;
    return float4(saturate(visibility), centre.y, 0.0, 1.0);
}
#elif defined(AO_APPLY)
// The scene's colour taken towards the game's shadow blend as the visibility falls (black with AC6_AO_SHADOW=0).
// textureIndex = a copy of the scene's colour, samplerIndex = the visibility (XeGTAO's, at any size) | the scene's
// depth << 16. AC6_AO_DEBUG=1: the visibility alone.
float4 main(VsOut i) : SV_Target0
{
    const AoParams p = LoadParams();
    Texture2D<float4> scene = g_Textures[g_Push.textureIndex];
    Texture2D<float4> visibilityImage = g_Textures[g_Push.samplerIndex & 0xFFFF];
    Texture2D<float4> depthImage = g_Textures[g_Push.samplerIndex >> 16];
    const float4 c = scene.Load(int3(int2(i.pos.xy), 0));
    float2 depthSize;
    depthImage.GetDimensions(depthSize.x, depthSize.y);
    const int2 depthAt = min(int2(i.uv * depthSize), int2(depthSize) - 1);
    const float d = depthImage.Load(int3(depthAt, 0)).x;
    // AC6_AO_DEBUG=2: the view distance (white near, darker by the octave, red: no depth).
    if (p.debug == 2.0)
        return d > 0.0 ? float4((1.0 - saturate(log2(max(p.clipZ / d, 1.0)) / 16.0)).xxx * min(p.colourCap, 1.0), c.a) : float4(min(p.colourCap, 1.0), 0.0, 0.0, c.a);
    if (d <= 0.0)
        return p.debug == 1.0 ? float4(min(p.colourCap, 1.0).xxx, c.a) : c;   // the sky
    const bool terrain = AoIsTerrain(p, depthAt, d);
    // AC6_AO_DEBUG=3: the terrain's pixels in green over the scene.
    if (p.debug == 3.0)
        return terrain ? float4(c.r * 0.5, min(c.g * 0.5 + 0.4, min(p.colourCap, 1.0)), c.b * 0.5, c.a) : c;
    const float viewZ = p.clipZ / d;
    // AC6_AO_DEBUG=5: what isn't terrain, coloured by its view distance in sevenths of the distance the occlusion ends at
    // (AC6_AO_FADE_TO: red, yellow, green, cyan, blue, magenta, then white beyond it, the first seventh as it is): where the
    // game's objects begin.
    if (p.debug == 5.0)
    {
        const float3 bands[7] = { float3(1, 0, 0), float3(1, 1, 0), float3(0, 1, 0), float3(0, 1, 1), float3(0, 0, 1), float3(1, 0, 1), float3(1, 1, 1) };
        const int km = int(viewZ * 7.0 / p.fadeTo);
        return terrain || km < 1 ? c : float4(bands[min(km - 1, 6)] * min(p.colourCap, 1.0), c.a);
    }
    float visibility = visibilityImage.SampleLevel(g_Samplers[0], i.uv, 0).x;
    // Fading out with the distance from the eye (not the depth: the map's buildings are drawn from a distance, and
    // the game's fog is in the colour already: far away the occlusion would darken fog).
    const float distance = viewZ * length(float3(p.consts.NDCToViewMul * i.uv + p.consts.NDCToViewAdd, 1.0));
    const float fade = 1.0 - saturate((distance - p.fadeFrom) / max(p.fadeTo - p.fadeFrom, 1.0));
    visibility = 1.0 - saturate((1.0 - visibility) * p.strength) * fade;
    if (terrain && p.terrainMask != 0.0)
        visibility = 1.0;
    // Full occlusion is the game's own shadow: colour x (1 - a) + rgb, as its shadow pass blends.
    if (p.debug == 1.0)
    {
        const float white = min(p.colourCap, 1.0);
        return float4(lerp(white * (1.0 - p.shadow.a) + p.shadow.rgb, white.xxx, visibility), c.a);
    }
    // The 8-bit scene's emitter codes (values above 240 / 255 stand for brightness above white): not scaled.
    if (max(c.r, max(c.g, c.b)) > p.colourCap)
        return c;
    return float4(lerp(c.rgb * (1.0 - p.shadow.a) + p.shadow.rgb, c.rgb, visibility), c.a);
}
#endif
