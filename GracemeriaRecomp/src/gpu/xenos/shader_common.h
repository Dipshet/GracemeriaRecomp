#ifndef SHADER_COMMON_H_INCLUDED
#define SHADER_COMMON_H_INCLUDED

#define SPEC_CONSTANT_R11G11B10_NORMAL  (1 << 0)
#define SPEC_CONSTANT_ALPHA_TEST        (1 << 1)
// AC6: point lists drawn as two triangles per point.
#define SPEC_CONSTANT_POINT_LIST        (1 << 5)
// AC6: the draw writes its motion to colour output 1 (TAA's motion vectors, SPIR-V backend only).
#define SPEC_CONSTANT_MOTION            (1 << 6)
// AC6: the game's lighting in float. The draw writes the game's 8-bit code (from a shader without its code scale: the
// sun, light sprites) into a float scene surface, so its colour is written decoded, as the game's decoders read it
// (codes 241-255 as 1.1-100), in code units. Read from the shared draw flags only (no pipeline variant).
#define SPEC_CONSTANT_EMITTER_DECODE    (1 << 7)
// AC6: FSR's reactive mask: a translucent draw writes its coverage into colour output 1's alpha: its alpha, or with
// REACTIVE_LUMA (an additive draw) its brightest channel, x 0.9. Read from the shared draw flags only.
#define SPEC_CONSTANT_REACTIVE          (1 << 8)
#define SPEC_CONSTANT_REACTIVE_LUMA     (1 << 9)
// AC6: the planar reflection's mirrored draws: the vertex shader clips at the sea's plane, the clip position (as the
// game's shader wrote it) against the plane in vertex constant 255, which the renderer writes into the draw's own copy
// of the constants. Read from the shared draw flags only.
#define SPEC_CONSTANT_PLANAR_CLIP       (1 << 10)

#ifdef UNLEASHED_RECOMP
    #define SPEC_CONSTANT_BICUBIC_GI_FILTER (1 << 2)
    #define SPEC_CONSTANT_ALPHA_TO_COVERAGE (1 << 3)
    #define SPEC_CONSTANT_REVERSE_Z         (1 << 4)
#endif

#if !defined(__cplusplus) || defined(__INTELLISENSE__)

#define FLT_MIN asfloat(0xff7fffff)
#define FLT_MAX asfloat(0x7f7fffff)

// AC6: the console's reciprocal is never above the true value (it truncates), and game shaders depend on
// it: the terrain picks its fan ring slot as floor(40 * frac((v + 1) * rcp(40))), which on the console is
// v mod 40 for every vertex, the phase its ring table is authored for. An exact or rounded-up reciprocal
// lands on (v + 1) mod 40 or between the two (cracks and steep "rifts" in the terrain).
// Two ulps below the host's rcp keeps that for vertex indices well past 200000 (float32
// simulation). Powers of two, which have exact reciprocals, stay exact.
float xeRcp(float x)
{
    float r = rcp(x);
    if ((asuint(x) & 0x007fffffu) == 0 || !(abs(r) > 0.0) || !(abs(r) < asfloat(0x7f800000)))
        return r;
    return asfloat(asuint(r) - 2u);
}

#ifdef __spirv__

struct PushConstants
{
    uint64_t VertexShaderConstants;
    uint64_t PixelShaderConstants;
    uint64_t SharedConstants;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> g_PushConstants;

// AC6: 32 texture/sampler slots per kind (pixel 0-15, vertex 16-31) before these, see renderer.cpp SharedConstants.
#define g_Booleans                 vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 512)
#define g_SwappedTexcoords         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 516)
#define g_HalfPixelOffset          vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 520)
#define g_AlphaThreshold           vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + 528)
// AC6: the guest vertex index of SV_VertexID 0 (r0.x starts as the vertex index), and per stream the
// device address and size in dwords of the whole guest buffer, for fetches at a computed index.
#define g_VertexIndexBase          vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 532)
#define g_StreamAddress(STREAM)    vk::RawBufferLoad<uint64_t>(g_PushConstants.SharedConstants + 544 + (STREAM) * 8, 8)
#define g_StreamDwords(STREAM)     vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 672 + (STREAM) * 4)
// AC6: guest pixels per host pixel of the render target (0.5 on 4x MSAA surfaces drawn at twice the size).
#define g_PixelPosScale            vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + 736)
// AC6: samplers for texture fetches whose instruction overrides the fetch constant's filters (pixel
// shader entries 0-7, vertex shader 8-15), see ShaderRecompiler::samplerOverrides.
#define g_OverrideSampler(INDEX)   vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 740 + (INDEX) * 4)
// AC6: point sprites: x/y = the per-vertex diameter's clamp range in pixels, z/w = NDC
// units per pixel of the viewport. The constant diameter for shaders that don't export one.
#define g_PointParams              vk::RawBufferLoad<float4>(g_PushConstants.SharedConstants + 804)
#define g_PointConstantDiameter    vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 820)
// AC6: pixel shader inputs interpolated at the centroid, bit = input register.
#define g_InterpolatorCentroid     vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 828)
// AC6: the draw's SPEC_CONSTANT_* bits, which the SPIR-V backend reads here instead.
#define g_DrawFlags                vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 832)

// AC6: vertex shaders fetch textures too (Vulkan allows implicit-LOD sampling only in pixel shaders),
// so samples go through XE_SAMPLE, which uses LOD 0 outside the pixel stage.
#if defined(__SHADER_TARGET_STAGE) && __SHADER_TARGET_STAGE == __SHADER_STAGE_PIXEL
#define XE_SAMPLE(texture, samplerState, coord) (texture).Sample(samplerState, coord)
#else
#define XE_SAMPLE(texture, samplerState, coord) (texture).SampleLevel(samplerState, coord, 0)
#endif

[[vk::constant_id(0)]] const uint g_SpecConstants = 0;

#define g_SpecConstants() g_SpecConstants

#else

#define DEFINE_SHARED_CONSTANTS() \
    uint g_Booleans : packoffset(c16.x); \
    uint g_SwappedTexcoords : packoffset(c16.y); \
    float2 g_HalfPixelOffset : packoffset(c16.z); \
    float g_AlphaThreshold : packoffset(c17.x);

uint g_SpecConstants();

#endif

Texture2D<float4> g_Texture2DDescriptorHeap[] : register(t0, space0);
Texture3D<float4> g_Texture3DDescriptorHeap[] : register(t0, space1);
TextureCube<float4> g_TextureCubeDescriptorHeap[] : register(t0, space2);
SamplerState g_SamplerDescriptorHeap[] : register(s0, space3);

uint2 getTexture2DDimensions(Texture2D<float4> texture)
{
    uint2 dimensions;
    texture.GetDimensions(dimensions.x, dimensions.y);
    return dimensions;
}

float4 tfetch2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return XE_SAMPLE(texture, g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord + offset / getTexture2DDimensions(texture));
}

float2 getWeights2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return select(isnan(texCoord), 0.0, frac(texCoord * getTexture2DDimensions(texture) + offset - 0.5));
}

// AC6: getCompTexLOD: the LOD the hardware would pick (x). Outside the pixel stage there are no
// derivatives, so 0.
float4 getComputedLod2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
#if defined(__SHADER_TARGET_STAGE) && __SHADER_TARGET_STAGE == __SHADER_STAGE_PIXEL
    return float4(g_Texture2DDescriptorHeap[resourceDescriptorIndex].CalculateLevelOfDetailUnclamped(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord), 0.0, 0.0, 0.0);
#else
    return 0.0;
#endif
}

float w0(float a)
{
    return (1.0f / 6.0f) * (a * (a * (-a + 3.0f) - 3.0f) + 1.0f);
}

float w1(float a)
{
    return (1.0f / 6.0f) * (a * a * (3.0f * a - 6.0f) + 4.0f);
}

float w2(float a)
{
    return (1.0f / 6.0f) * (a * (a * (-3.0f * a + 3.0f) + 3.0f) + 1.0f);
}

float w3(float a)
{
    return (1.0f / 6.0f) * (a * a * a);
}

float g0(float a)
{
    return w0(a) + w1(a);
}

float g1(float a)
{
    return w2(a) + w3(a);
}

float h0(float a)
{
    return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}

float h1(float a)
{
    return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}

float4 tfetch2DBicubic(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    SamplerState samplerState = g_SamplerDescriptorHeap[samplerDescriptorIndex];
    uint2 dimensions = getTexture2DDimensions(texture);
    
    float x = texCoord.x * dimensions.x + offset.x;
    float y = texCoord.y * dimensions.y + offset.y;

    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    float g0x = g0(fx);
    float g1x = g1(fx);
    float h0x = h0(fx);
    float h1x = h1(fx);
    float h0y = h0(fy);
    float h1y = h1(fy);

    float4 r =
        g0(fy) * (g0x * XE_SAMPLE(texture, samplerState, float2(px + h0x, py + h0y) / float2(dimensions)) +
            g1x * XE_SAMPLE(texture, samplerState, float2(px + h1x, py + h0y) / float2(dimensions))) +
        g1(fy) * (g0x * XE_SAMPLE(texture, samplerState, float2(px + h0x, py + h1y) / float2(dimensions)) +
            g1x * XE_SAMPLE(texture, samplerState, float2(px + h1x, py + h1y) / float2(dimensions)));

    return r;
}

float4 tfetch3D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord)
{
    return XE_SAMPLE(g_Texture3DDescriptorHeap[resourceDescriptorIndex], g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord);
}

struct CubeMapData
{
    float3 cubeMapDirections[2];
    uint cubeMapIndex;
};

float4 tfetchCube(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, inout CubeMapData cubeMapData)
{
    return XE_SAMPLE(g_TextureCubeDescriptorHeap[resourceDescriptorIndex], g_SamplerDescriptorHeap[samplerDescriptorIndex], cubeMapData.cubeMapDirections[texCoord.z]);
}

float4 tfetchR11G11B10(uint4 value)
{
    if (g_SpecConstants() & SPEC_CONSTANT_R11G11B10_NORMAL)
    {
        return float4(
            (value.x & 0x00000400 ? -1.0 : 0.0) + ((value.x & 0x3FF) / 1024.0),
            (value.x & 0x00200000 ? -1.0 : 0.0) + (((value.x >> 11) & 0x3FF) / 1024.0),
            (value.x & 0x80000000 ? -1.0 : 0.0) + (((value.x >> 22) & 0x1FF) / 512.0),
            0.0);
    }
    else
    {
        return asfloat(value);
    }
}

// AC6: packed vertex formats, unpacked from the raw 32-bit word (component X in the low bits).
float xeFetchComponent(uint value, uint shift, uint bits, bool isSigned, bool isInteger)
{
    uint raw = (value >> shift) & ((1u << bits) - 1u);
    if (isSigned)
    {
        int v = int(raw << (32u - bits)) >> (32u - bits);
        return isInteger ? float(v) : max(float(v) / float((1u << (bits - 1u)) - 1u), -1.0);
    }
    return isInteger ? float(raw) : float(raw) / float((1u << bits) - 1u);
}

float4 xeFetch2_10_10_10(uint value, bool isSigned, bool isInteger)
{
    return float4(xeFetchComponent(value, 0, 10, isSigned, isInteger), xeFetchComponent(value, 10, 10, isSigned, isInteger),
        xeFetchComponent(value, 20, 10, isSigned, isInteger), xeFetchComponent(value, 30, 2, isSigned, isInteger));
}

float4 xeFetch10_11_11(uint value, bool isSigned, bool isInteger)
{
    return float4(xeFetchComponent(value, 0, 11, isSigned, isInteger), xeFetchComponent(value, 11, 11, isSigned, isInteger),
        xeFetchComponent(value, 22, 10, isSigned, isInteger), 1.0);
}

float4 xeFetch11_11_10(uint value, bool isSigned, bool isInteger)
{
    return float4(xeFetchComponent(value, 0, 10, isSigned, isInteger), xeFetchComponent(value, 10, 11, isSigned, isInteger),
        xeFetchComponent(value, 21, 11, isSigned, isInteger), 1.0);
}

#ifdef __spirv__
// AC6: vertex fetches at an index the shader computed (particles: corner and particle index from the
// vertex index) read the stream themselves. The renderer copies the whole buffer with each 32-bit
// word swapped, as for the input assembler, so these decode like the host vertex formats there.
// Reads outside the buffer return 0.
uint xeFetchWord(uint stream, uint index, int offset, uint stride, uint word)
{
    uint dword = index * stride + uint(offset) + word;
    if (dword >= g_StreamDwords(stream))
        return 0;
    return vk::RawBufferLoad<uint>(g_StreamAddress(stream) + uint64_t(dword) * 4, 4);
}

float4 xeFetchIndexed(uint stream, uint index, int offset, uint stride, uint format, bool isSigned, bool isInteger)
{
    uint w0 = xeFetchWord(stream, index, offset, stride, 0);
    switch (format)
    {
    case 6:    // 8_8_8_8
        return float4(xeFetchComponent(w0, 0, 8, isSigned, isInteger), xeFetchComponent(w0, 8, 8, isSigned, isInteger),
            xeFetchComponent(w0, 16, 8, isSigned, isInteger), xeFetchComponent(w0, 24, 8, isSigned, isInteger));
    case 25:   // 16_16
        return float4(xeFetchComponent(w0, 0, 16, isSigned, isInteger), xeFetchComponent(w0, 16, 16, isSigned, isInteger), 0.0, 1.0);
    case 26:   // 16_16_16_16
    {
        uint w1 = xeFetchWord(stream, index, offset, stride, 1);
        return float4(xeFetchComponent(w0, 0, 16, isSigned, isInteger), xeFetchComponent(w0, 16, 16, isSigned, isInteger),
            xeFetchComponent(w1, 0, 16, isSigned, isInteger), xeFetchComponent(w1, 16, 16, isSigned, isInteger));
    }
    case 31:   // 16_16_FLOAT
        return float4(f16tof32(w0), f16tof32(w0 >> 16), 0.0, 1.0);
    case 32:   // 16_16_16_16_FLOAT
    {
        uint w1 = xeFetchWord(stream, index, offset, stride, 1);
        return float4(f16tof32(w0), f16tof32(w0 >> 16), f16tof32(w1), f16tof32(w1 >> 16));
    }
    case 33: case 34: case 35:   // 32, 32_32, 32_32_32_32 (integer)
    {
        uint4 w = uint4(w0, format >= 34 ? xeFetchWord(stream, index, offset, stride, 1) : 0,
            format == 35 ? xeFetchWord(stream, index, offset, stride, 2) : 0, format == 35 ? xeFetchWord(stream, index, offset, stride, 3) : 1);
        return isSigned ? float4(int4(w)) : float4(w);
    }
    case 36:   // 32_FLOAT
        return float4(asfloat(w0), 0.0, 0.0, 1.0);
    case 37:   // 32_32_FLOAT
        return float4(asfloat(w0), asfloat(xeFetchWord(stream, index, offset, stride, 1)), 0.0, 1.0);
    case 57:   // 32_32_32_FLOAT
        return float4(asfloat(w0), asfloat(xeFetchWord(stream, index, offset, stride, 1)), asfloat(xeFetchWord(stream, index, offset, stride, 2)), 1.0);
    default:   // 38: 32_32_32_32_FLOAT
        return float4(asfloat(w0), asfloat(xeFetchWord(stream, index, offset, stride, 1)), asfloat(xeFetchWord(stream, index, offset, stride, 2)),
            asfloat(xeFetchWord(stream, index, offset, stride, 3)));
    }
}
#endif

float4 tfetchTexcoord(uint swappedTexcoords, float4 value, uint semanticIndex)
{
    return (swappedTexcoords & (1ull << semanticIndex)) != 0 ? value.yxwz : value;
}

float4 cube(float4 value, inout CubeMapData cubeMapData)
{
    uint index = cubeMapData.cubeMapIndex;
    cubeMapData.cubeMapDirections[index] = value.xyz;
    ++cubeMapData.cubeMapIndex;
    
    return float4(0.0, 0.0, 0.0, index);
}

float4 dst(float4 src0, float4 src1)
{
    float4 dest;
    dest.x = 1.0;
    dest.y = src0.y * src1.y;
    dest.z = src0.z;
    dest.w = src1.w;
    return dest;
}

float4 max4(float4 src0)
{
    return max(max(src0.x, src0.y), max(src0.z, src0.w));
}

float2 getPixelCoord(uint resourceDescriptorIndex, float2 texCoord)
{
    return getTexture2DDimensions(g_Texture2DDescriptorHeap[resourceDescriptorIndex]) * texCoord;
}

float computeMipLevel(float2 pixelCoord)
{
    float2 dx = ddx(pixelCoord);
    float2 dy = ddy(pixelCoord);
    float deltaMaxSqr = max(dot(dx, dx), dot(dy, dy));
    return max(0.0, 0.5 * log2(deltaMaxSqr));
}

#endif

#endif
