// The renderer's own shaders (copies, depth conversions, gamma ramp, the empty pixel shader), compiled to
// SPIR-V at build time by the Vulkan SDK's dxc.exe (tools/cmake/compile_internal_shaders.cmake), one variant per
// define: the executable needs no shader compiler.
#ifdef NULL_PIXEL
// Depth-only draws bind no pixel shader. They get this one, with colour writes off.
void main() {}
#elif defined(VOL_HARVEST) || defined(VOL_VS) || defined(VOL_PS) || defined(VOL_COMPOSE) || defined(VOL_CLOUDSCAPE)
#include "volumetrics.hlsl"
#elif defined(AO_DEPTH) || defined(AO_DEPTH_MIP) || defined(AO_MAIN) || defined(AO_DENOISE) || defined(AO_APPLY) || defined(AO_SUBJECT)
#include "ao.hlsl"
#elif defined(LENS_FX)
#include "lens_fx.hlsl"
#elif defined(MOTION_BLUR)
#include "motion_blur.hlsl"
#elif defined(BLOOM_BRIGHT) || defined(BLOOM_DOWN) || defined(BLOOM_UP) || defined(BLOOM_APPLY)
#include "bloom.hlsl"
#elif defined(LIGHT_DOWN0) || defined(LIGHT_DOWN0B) || defined(LIGHT_DOWN) || defined(LIGHT_APPLY)
#include "emitter_light.hlsl"
#elif defined(IMGUI_VS) || defined(IMGUI_PS)
// The settings overlay (src/ui/overlay.cpp): Dear ImGui's triangles in window pixels, textured and alpha blended into
// the swap chain image. scale / translate take pixels to clip space, paper: 0 for SDR, else scRGB (linear, 1.0 = 80 nits)
// at that paper-white scale, as the output pass draws the game's UI.
Texture2D<float4> g_Textures[] : register(t0, space0);
SamplerState g_Samplers[] : register(s0, space3);
struct UiPush { float2 scale; float2 translate; uint textureIndex; float paper; };
[[vk::push_constant]] ConstantBuffer<UiPush> g_Ui;
struct UiOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; float4 color : COLOR0; };
#ifdef IMGUI_VS
UiOut main([[vk::location(0)]] float2 pos : TEXCOORD0, [[vk::location(1)]] float2 uv : TEXCOORD1, [[vk::location(2)]] float4 color : TEXCOORD2)
{
    UiOut o;
    o.pos = float4(pos * g_Ui.scale + g_Ui.translate, 0.0, 1.0);
    o.uv = uv;
    o.color = color;
    return o;
}
#else
float4 main(UiOut i) : SV_Target0
{
    float4 c = i.color * g_Textures[g_Ui.textureIndex].SampleLevel(g_Samplers[0], i.uv, 0);
    if (g_Ui.paper > 0.0)
        c.rgb = pow(saturate(c.rgb), 2.2) * g_Ui.paper;
    return c;
}
#endif
#elif defined(LATE_BOUNDS)
// The late pass's meshes (the canopy glass), once each: a geometry shader after the game's vertex
// shader passes the triangles through and stores their vertices' guest clip positions (the motion vectors' input,
// location 17) at the slots the renderer reserved, for the renderer to rebuild the mesh and bound it on screen.
struct GsPush { uint addressLo; uint addressHi; uint base; uint end; };
[[vk::push_constant]] ConstantBuffer<GsPush> g_Gs;
struct GsVert
{
    float4 pos : SV_Position;
    [[vk::location(0)]] float4 t0 : TEXCOORD0;
    [[vk::location(1)]] float4 t1 : TEXCOORD1;
    [[vk::location(2)]] float4 t2 : TEXCOORD2;
    [[vk::location(3)]] float4 t3 : TEXCOORD3;
    [[vk::location(4)]] float4 t4 : TEXCOORD4;
    [[vk::location(5)]] float4 t5 : TEXCOORD5;
    [[vk::location(6)]] float4 t6 : TEXCOORD6;
    [[vk::location(7)]] float4 t7 : TEXCOORD7;
    [[vk::location(8)]] float4 t8 : TEXCOORD8;
    [[vk::location(9)]] float4 t9 : TEXCOORD9;
    [[vk::location(10)]] float4 t10 : TEXCOORD10;
    [[vk::location(11)]] float4 t11 : TEXCOORD11;
    [[vk::location(12)]] float4 t12 : TEXCOORD12;
    [[vk::location(13)]] float4 t13 : TEXCOORD13;
    [[vk::location(14)]] float4 t14 : TEXCOORD14;
    [[vk::location(15)]] float4 t15 : TEXCOORD15;
    [[vk::location(16)]] float4 t16 : TEXCOORD16;
    [[vk::location(17)]] float4 t17 : TEXCOORD17;
};

[maxvertexcount(3)]
void main(triangle GsVert v[3], uint prim : SV_PrimitiveID, inout TriangleStream<GsVert> stream)
{
    stream.Append(v[0]);
    stream.Append(v[1]);
    stream.Append(v[2]);
    stream.RestartStrip();
    const uint slot = g_Gs.base + prim * 3;
    if (slot + 3 > g_Gs.end)
        return;
    const uint64_t address = (uint64_t(g_Gs.addressHi) << 32) | g_Gs.addressLo;
    [unroll] for (uint k = 0; k < 3; k++)
        vk::RawBufferStore<float4>(address + uint64_t(slot + k) * 16, v[k].t17);
}
#else
Texture2D<float4> g_Textures[] : register(t0, space0);
SamplerState g_Samplers[] : register(s0, space3);
// grade0 / grade1: the output pass's HDR grading, bytes: exposure ((x - 0.5) x 100), highlights,
// shadows, contrast. grade1: saturation, gamut compression (%), PsychoV-30's compression (x 100), then bit 24 HDR from
// the scene and bits 25-31 the hue correction (%). stripes: the output pass's brightness stripes (AC6_HDR_STRIPES):
// bits 0-7: 0 none, 1 the display's brightness, 2 the game's values, bit 8: the stray emitter cap off. (The
// layout's 28 bytes.)
struct PushConstants { uint textureIndex; uint samplerIndex; int offsetX; int offsetY; uint grade0; uint grade1; uint stripes; };
[[vk::push_constant]] ConstantBuffer<PushConstants> g_Push;
struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
#ifdef VERTEX
VsOut main(uint id : SV_VertexID)
{
    VsOut o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.uv = uv;
    // D3D-style clip space: the shared compile path adds -fvk-invert-y for vertex shaders.
    o.pos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    return o;
}
#elif defined(DEPTH_RESAMPLE)
// EDRAM alias: the depth of another surface at the same EDRAM address, resampled to this one's size (the
// smallest depth of the covered texels, as the game's own depth downsample takes). offsetX/Y = source size,
// samplerIndex = this surface's width | height << 16.
float main(VsOut i) : SV_Depth
{
    uint dw, dh;
    g_Textures[g_Push.textureIndex].GetDimensions(dw, dh);
    const int2 src = int2(g_Push.offsetX, g_Push.offsetY);
    const int2 dst = int2(i.pos.xy);
    float2 scale = float2(src) / float2(g_Push.samplerIndex & 0xFFFF, g_Push.samplerIndex >> 16);
    int2 first = int2(float2(dst) * scale), count = max(int2(1, 1), int2(scale));
    float depth = 1.0;
    for (int y = 0; y < count.y; y++)
        for (int x = 0; x < count.x; x++)
            depth = min(depth, g_Textures[g_Push.textureIndex].Load(int3(min(first + int2(x, y), src - 1), 0)).x);
    return depth;
}
#elif defined(DEPTH_SWIZZLE) || defined(COLOR_SWIZZLE)
// EDRAM alias between a 1x surface and a 4x MSAA one drawn at twice the size: in EDRAM,
// each run of 4 pixels of the 1x view holds two MSAA pixels' samples interleaved (pixel 0 sample 0, pixel 1
// sample 0, pixel 0 sample 1, pixel 1 sample 1), on both axes, while the host MSAA image keeps samples in
// place. Either way the texel at p comes from the other image's p with the middle two of each 4 swapped.
int2 EdramMsaaSwap(int2 p)
{
    return (p & ~3) | ((p & 1) << 1) | ((p >> 1) & 1);
}
// At the internal resolution the swap moves whole guest samples: the host pixel's guest
// sample, swapped, moved by as many host pixels. offsetX/Y = the scales' float bits (0: scale 1, the plain swap).
int2 ScaledSwap(int2 p)
{
    if (g_Push.offsetX == 0)
        return EdramMsaaSwap(p);
    const float2 r = float2(asfloat(g_Push.offsetX), asfloat(g_Push.offsetY));   // per axis
    const int2 q = int2(floor((float2(p) + 0.5) / r));
    return p + int2(round(float2(EdramMsaaSwap(q) - q) * r));
}
#ifdef DEPTH_SWIZZLE
float main(VsOut i) : SV_Depth
{
    return g_Textures[g_Push.textureIndex].Load(int3(ScaledSwap(int2(i.pos.xy)), 0)).x;
}
#else
float4 main(VsOut i) : SV_Target0
{
    return g_Textures[g_Push.textureIndex].Load(int3(ScaledSwap(int2(i.pos.xy)), 0));
}
#endif
#elif defined(DEPTH_LOAD)
// A single-sampled depth image into every sample of a multisampled one.
float main(VsOut i) : SV_Depth
{
    return g_Textures[g_Push.textureIndex].Load(int3(int2(i.pos.xy), 0)).x;
}
#elif defined(COLOR_LOAD)
float4 main(VsOut i) : SV_Target0
{
    return g_Textures[g_Push.textureIndex].Load(int3(int2(i.pos.xy), 0));
}
#elif defined(TEXEL_COPY)
// Colour resolve as a draw: the source texel at (destination pixel + offset), 1:1.
float4 main(VsOut i) : SV_Target0
{
    return g_Textures[g_Push.textureIndex].Load(int3(int2(i.pos.xy) + int2(g_Push.offsetX, g_Push.offsetY), 0));
}
#elif defined(STENCIL_BIT)
// One bit plane of a stencil copy (R8): the pass writes that bit (its pipeline's write mask) where it is set.
void main(VsOut i)
{
    const uint stencil = uint(g_Textures[g_Push.textureIndex].Load(int3(int2(i.pos.xy), 0)).x * 255.0 + 0.5);
    if ((stencil & g_Push.samplerIndex) == 0)
        discard;
}
#elif defined(DEPTH_TO_COLOR)
// A depth/stencil surface's EDRAM seen through an 8_8_8_8 colour surface at the same base:
// textureIndex = the depth, samplerIndex = its stencil (R8), offsetX = 1 for 20e4 float depth, offsetY = the
// depth surface's width.
uint Float20e4(float f)
{
    f = clamp(f, 0.0, 1.99999988);
    if (f < 6.103515625e-05)
        return uint(f * 17179869184.0 + 0.5);   // denormal: m * 2^-34
    const uint u = asuint(f);
    uint v = (((u >> 23) - 112) << 20) | ((u & 0x7FFFFF) >> 3);   // exponent bias 127 -> 15, 23 -> 20 mantissa bits
    const uint dropped = u & 7;
    if (dropped > 4 || (dropped == 4 && (v & 1)))
        v++;
    return min(v, 0xFFFFFFu);
}
float4 main(VsOut i) : SV_Target0
{
    const int2 p = int2(i.pos.xy);
    float depth = 1.0;
    uint stencil = 0;
    if (p.x < g_Push.offsetY)
    {
        depth = g_Textures[g_Push.textureIndex].Load(int3(p, 0)).x;
        stencil = uint(g_Textures[g_Push.samplerIndex].Load(int3(p, 0)).x * 255.0 + 0.5);
    }
    // (Rounded, not +0.5 truncated: 16777215.5 is 2^24 in float.)
    const uint d24 = (g_Push.offsetX & 1) ? Float20e4(depth) : min(uint(round(saturate(depth) * 16777215.0)), 0xFFFFFFu);
    // The EDRAM word holds the depth in bits 0-23 and the stencil above. An 8_8_8_8 colour word holds blue,
    // green, red, alpha from bit 0. So red is the depth's high byte, which is what the game reads.
    return float4((d24 >> 16) & 255, (d24 >> 8) & 255, d24 & 255, stencil) / 255.0;
}
#elif defined(DEPTH_COPY)
// Depth resolve: the source depth texel at (destination pixel + offset) into a float colour target.
float4 main(VsOut i) : SV_Target0
{
    // samplerIndex = source host pixels per destination pixel (promoted 4x MSAA sources).
    return g_Textures[g_Push.textureIndex].Load(int3(int2(i.pos.xy) * int(max(g_Push.samplerIndex, 1u)) + int2(g_Push.offsetX, g_Push.offsetY), 0)).xxxx;
}
#elif defined(DEPTH_COPY_MS)
// Depth resolve straight from a multisampled depth surface: sample 0's depth into the float target
// and its stencil into the R8 one beside it, as a single-sampled copy resolved from sample 0 would give them.
// textureIndex = the surface's depth view, samplerIndex = its stencil view, offsetX/Y as DEPTH_COPY's.
[[vk::binding(0, 0)]] Texture2DMS<float4> g_DepthMS[];
[[vk::binding(0, 0)]] Texture2DMS<uint4> g_StencilMS[];
struct DepthStencilOut { float4 depth : SV_Target0; float4 stencil : SV_Target1; };
DepthStencilOut main(VsOut i)
{
    const int2 at = int2(i.pos.xy) + int2(g_Push.offsetX, g_Push.offsetY);
    DepthStencilOut o;
    o.depth = g_DepthMS[g_Push.textureIndex].Load(at, 0).xxxx;
    o.stencil = (float(g_StencilMS[g_Push.samplerIndex].Load(at, 0).x) / 255.0).xxxx;
    return o;
}
#elif defined(DEPTH_COPY_SS)
// DEPTH_COPY_MS's for a single-sampled depth image: its depth and stencil (an R8_UINT view) in one
// pass.
[[vk::binding(0, 0)]] Texture2D<float4> g_DepthSS[];
[[vk::binding(0, 0)]] Texture2D<uint4> g_StencilSS[];
struct DepthStencilOut { float4 depth : SV_Target0; float4 stencil : SV_Target1; };
DepthStencilOut main(VsOut i)
{
    const int3 at = int3(int2(i.pos.xy) + int2(g_Push.offsetX, g_Push.offsetY), 0);
    DepthStencilOut o;
    o.depth = g_DepthSS[g_Push.textureIndex].Load(at).xxxx;
    o.stencil = (float(g_StencilSS[g_Push.samplerIndex].Load(at).x) / 255.0).xxxx;
    return o;
}
#elif defined(SMAA_EDGE) || defined(SMAA_WEIGHT) || defined(SMAA_BLEND)
// SMAA 1x (thirdparty/smaa, Jimenez et al., MIT) on the scene image after the game resolves it: the
// unmodified SMAA.hlsl through its custom-language hooks, high preset. offsetX/Y = the image's size. The vertex stage's
// offsets are computed here (they are linear in the texture coordinate). Sampler 0 = linear clamp, 1023 = point clamp.
static float4 g_SmaaMetrics;
#define SMAA_RT_METRICS g_SmaaMetrics
#define SMAA_CUSTOM_SL 1
#define SMAA_PRESET_HIGH 1
#define SMAATexture2D(tex) Texture2D<float4> tex
#define SMAATexturePass2D(tex) tex
#define SMAASampleLevelZero(tex, coord) tex.SampleLevel(g_Samplers[0], coord, 0)
#define SMAASampleLevelZeroPoint(tex, coord) tex.SampleLevel(g_Samplers[1023], coord, 0)
#define SMAASampleLevelZeroOffset(tex, coord, offset) tex.SampleLevel(g_Samplers[0], coord, 0, offset)
#define SMAASample(tex, coord) tex.Sample(g_Samplers[0], coord)
#define SMAASamplePoint(tex, coord) tex.Sample(g_Samplers[1023], coord)
#define SMAASampleOffset(tex, coord, offset) tex.Sample(g_Samplers[0], coord, offset)
#define SMAA_FLATTEN [flatten]
#define SMAA_BRANCH [branch]
#define SMAATexture2DMS2(tex) Texture2DMS<float4, 2> tex
#define SMAALoad(tex, pos, sample) tex.Load(pos, sample)
#define SMAAGather(tex, coord) tex.Gather(g_Samplers[0], coord, 0)
#include "../../../thirdparty/smaa/SMAA.hlsl"
void SetSmaaMetrics()
{
    const float2 size = float2(g_Push.offsetX, g_Push.offsetY);
    g_SmaaMetrics = float4(1.0 / size, size);
}
#if defined(SMAA_EDGE)
// textureIndex = the colour image. Luma edges (gamma space), into an RG8 target cleared to 0 (no edge: discarded).
float4 main(VsOut i) : SV_Target0
{
    SetSmaaMetrics();
    float4 offset[3];
    SMAAEdgeDetectionVS(i.uv, offset);
    return float4(SMAALumaEdgeDetectionPS(i.uv, offset, g_Textures[g_Push.textureIndex]), 0.0, 0.0);
}
#elif defined(SMAA_WEIGHT)
// textureIndex = the edges, samplerIndex = the area texture | the search texture << 16.
float4 main(VsOut i) : SV_Target0
{
    SetSmaaMetrics();
    float2 pixcoord;
    float4 offset[3];
    SMAABlendingWeightCalculationVS(i.uv, pixcoord, offset);
    return SMAABlendingWeightCalculationPS(i.uv, pixcoord, offset, g_Textures[g_Push.textureIndex],
        g_Textures[g_Push.samplerIndex & 0xFFFF], g_Textures[g_Push.samplerIndex >> 16], float4(0.0, 0.0, 0.0, 0.0));
}
#else
// textureIndex = the colour image, samplerIndex = the blending weights.
float4 main(VsOut i) : SV_Target0
{
    SetSmaaMetrics();
    float4 offset;
    SMAANeighborhoodBlendingVS(i.uv, offset);
    return SMAANeighborhoodBlendingPS(i.uv, offset, g_Textures[g_Push.textureIndex], g_Textures[g_Push.samplerIndex]);
}
#endif
#elif defined(TAA)
// Temporal AA on the scene image: textureIndex = this frame's colour (rendered with a sub-pixel jitter),
// samplerIndex = the scene depth | the history << 16, offsetX / offsetY = the device address of the parameters: the rows of
// R = inverse(this frame's view-projection) x last frame's (row vectors, as the game's shaders use them), this frame's
// jitter in NDC, the blend factor, a mode (0 normal, 1 no usable history, 2-5 and 7 debug views), the depth nearer than
// which a pixel is taken as moving with the camera (the player's own aircraft, the cockpit), the clip z. Then the blend
// at fast motion, the motion (host pixels a frame) that reaches it, the blend for the near pixels and the variance
// clip's width. Then the motion image's descriptor (each pixel's movement as its draw wrote it, 0 none).
// Each pixel is reprojected by its draw's own motion where the draw wrote one, else by the camera. The history is read
// with a Catmull-Rom filter and clipped towards the 3x3 neighbourhood's mean (variance clipping in YCoCg, within its
// range), so moving objects don't smear or ghost. The new frame weighs more the faster the pixel moves (a reprojection
// that is off by a pixel or two in a fast roll, and the resampling, would otherwise blur detail over the frames the
// history spans).
float3 ToYCoCg(float3 c)
{
    return float3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}
float3 FromYCoCg(float3 c)
{
    return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}
// The history pulled towards the box's centre until it lies inside (clipping, not clamping per channel: the colour keeps
// its direction).
float3 ClipToBox(float3 h, float3 lo, float3 hi)
{
    const float3 centre = 0.5 * (hi + lo), extent = 0.5 * (hi - lo) + 1e-4;
    const float3 v = h - centre;
    const float3 units = abs(v / extent);
    const float m = max(units.x, max(units.y, units.z));
    return m > 1.0 ? centre + v / m : h;
}
// Catmull-Rom from five bilinear samples (the corner taps' weights dropped).
float3 SampleCatmullRom(Texture2D<float4> t, float2 uv, float2 size)
{
    const float2 pos = uv * size - 0.5;
    const float2 p1 = floor(pos) + 0.5;
    const float2 f = pos - floor(pos);
    const float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    const float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    const float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    const float2 w3 = f * f * (-0.5 + 0.5 * f);
    const float2 w12 = w1 + w2;
    const float2 uv0 = (p1 - 1.0) / size, uv3 = (p1 + 2.0) / size, uv12 = (p1 + w2 / w12) / size;
    float3 r = t.SampleLevel(g_Samplers[0], float2(uv12.x, uv0.y), 0).rgb * (w12.x * w0.y);
    r += t.SampleLevel(g_Samplers[0], float2(uv0.x, uv12.y), 0).rgb * (w0.x * w12.y);
    r += t.SampleLevel(g_Samplers[0], uv12, 0).rgb * (w12.x * w12.y);
    r += t.SampleLevel(g_Samplers[0], float2(uv3.x, uv12.y), 0).rgb * (w3.x * w12.y);
    r += t.SampleLevel(g_Samplers[0], float2(uv12.x, uv3.y), 0).rgb * (w12.x * w3.y);
    const float total = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return max(r / total, 0.0);
}
struct TaaOut { float4 color : SV_Target0; float4 history : SV_Target1; };
TaaOut main(VsOut i)
{
    Texture2D<float4> current = g_Textures[g_Push.textureIndex];
    Texture2D<float4> depthImage = g_Textures[g_Push.samplerIndex & 0xFFFF];
    Texture2D<float4> history = g_Textures[g_Push.samplerIndex >> 16];
    const uint64_t params = (uint64_t(asuint(g_Push.offsetY)) << 32) | asuint(g_Push.offsetX);
    const float4 r0 = vk::RawBufferLoad<float4>(params), r1 = vk::RawBufferLoad<float4>(params + 16);
    const float4 r2 = vk::RawBufferLoad<float4>(params + 32), r3 = vk::RawBufferLoad<float4>(params + 48);
    const float4 extra = vk::RawBufferLoad<float4>(params + 64);   // jitter x, y (NDC), blend, mode
    const float4 extra2 = vk::RawBufferLoad<float4>(params + 80);  // near depth, clip z
    const float4 extra3 = vk::RawBufferLoad<float4>(params + 96);  // fast-motion blend, its motion, near blend, clip width
    const float4 extra4 = vk::RawBufferLoad<float4>(params + 112); // the motion image's descriptor (0: none)
    const int2 p = int2(i.pos.xy);
    float2 size;
    current.GetDimensions(size.x, size.y);
    // A NaN or an infinity in the frame (0 x inf in a game shader, an overflowing float surface) is black here: kept,
    // the history would spread it into its neighbours every frame and the blot would last until a camera cut.
    float3 c = current.Load(int3(p, 0)).rgb;
    if (any(isnan(c)) || any(isinf(c)))
        c = 0.0;
    // The neighbourhood's range and its mean and spread (YCoCg), for the clip.
    const float3 cy = ToYCoCg(c);
    float3 lo = cy, hi = cy, m1 = 0.0, m2 = 0.0;
    [unroll] for (int y = -1; y <= 1; y++)
        [unroll] for (int x = -1; x <= 1; x++)
        {
            const float3 n = ToYCoCg(current.Load(int3(clamp(p + int2(x, y), int2(0, 0), int2(size) - 1), 0)).rgb);
            lo = min(lo, n);
            hi = max(hi, n);
            m1 += n;
            m2 += n * n;
        }
    m1 /= 9.0;
    const float3 sigma = sqrt(max(m2 / 9.0 - m1 * m1, 0.0));
    const float3 boxLo = max(lo, m1 - extra3.w * sigma), boxHi = min(hi, m1 + extra3.w * sigma);
    TaaOut o;
    o.color = float4(c, 1.0);
    o.history = float4(c, 1.0);
    if (extra.w == 1.0)
        return o;
    // The nearest depth around (reversed depth: the largest), so edges move with the object in front.
    float2 depthSize;
    depthImage.GetDimensions(depthSize.x, depthSize.y);
    const int2 dp = min(int2(i.uv * depthSize), int2(depthSize) - 1);
    float depth = 0.0;
    int2 nearest = dp;
    [unroll] for (int y = -1; y <= 1; y++)
        [unroll] for (int x = -1; x <= 1; x++)
        {
            const int2 q = clamp(dp + int2(x, y), int2(0, 0), int2(depthSize) - 1);
            const float d = depthImage.Load(int3(q, 0)).x;
            if (d > depth)
            {
                depth = d;
                nearest = q;
            }
        }
    if (extra.w == 4.0 || (extra.w == 5.0 && (p.x & 1)))
    {
        // The depth the reprojection uses, as view distance: 1 / depth on a log scale (near white, far dark). Mode 5:
        // odd columns only, the frame in the even ones (the depth and the picture side by side).
        o.color = float4((1.0 - saturate(log2(max(extra2.y / max(depth, 1e-9), 1.0)) / 14.0)).xxx, 1.0);
        return o;
    }
    if (extra.w == 5.0)
        return o;
    if (extra.w == 7.0)
    {
        // The motion image as the draws wrote it (x 40), magenta where unknown, grey when there is none.
        o.color = float4(0.25, 0.25, 0.25, 1.0);
        if (asuint(extra4.x) != 0)
        {
            Texture2D<float4> motionImage = g_Textures[asuint(extra4.x)];
            const float2 m = motionImage.Load(int3(p, 0)).xy;
            o.color = all(abs(m) < 4.0) ? float4(saturate(abs(m) * 40.0), 0.0, 1.0) : float4(1.0, 0.0, 1.0, 1.0);
        }
        return o;
    }
    float2 prevUv = i.uv;
    const bool near = depth >= extra2.x;
    // The motion the draws wrote, at that nearest pixel: the object's own (aircraft, missiles, the
    // own aircraft as it rolls), from where the unjittered point was last frame. None there (sky-less gaps, translucent
    // things over nothing, objects new this frame): the camera's reprojection below, or none for the near pixels.
    bool moved = false;
    if (asuint(extra4.x) != 0)
    {
        Texture2D<float4> motionImage = g_Textures[asuint(extra4.x)];
        const int2 mp = int2((float2(nearest) + 0.5) / depthSize * size);
        const float2 m = motionImage.Load(int3(mp, 0)).xy;
        if (all(abs(m) < 4.0))
        {
            prevUv = float2(i.uv.x - extra.x * 0.5, i.uv.y + extra.y * 0.5) - m;
            moved = true;
        }
    }
    if (!near && !moved)
    {
        const float2 ndc = float2(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0) - extra.xy;
        float4 prev = ndc.x * r0 + ndc.y * r1 + depth * r2 + r3;
        if (prev.w <= 0.0)
            return o;
        prev.xy /= prev.w;
        prevUv = float2(prev.x * 0.5 + 0.5, 0.5 - prev.y * 0.5);
    }
    if (extra.w == 2.0)
    {
        o.color = float4(saturate(abs(prevUv - i.uv) * 40.0), moved ? 0.5 : near ? 1.0 : 0.0, 1.0);
        return o;
    }
    if (any(prevUv < 0.0) || any(prevUv > 1.0))
        return o;
    float3 raw = SampleCatmullRom(history, prevUv, size);
    if (any(isnan(raw)) || any(isinf(raw)))
        raw = c;
    if (extra.w == 3.0)
    {
        // Reprojection check: the history where this pixel was, against the frame (x 4).
        o.color = float4(saturate(abs(raw - c) * 4.0), 1.0);
        return o;
    }
    const float3 h = FromYCoCg(ClipToBox(ToYCoCg(raw), boxLo, boxHi));
    // The new frame's weight: the base, more the faster the pixel moves, the near pixels without motion their own.
    const float motion = length((prevUv - i.uv) * size);
    const float blend = near && !moved ? max(extra.z, extra3.z) : lerp(extra.z, max(extra.z, extra3.x), saturate(motion / extra3.y));
    const float3 blended = lerp(h, c, blend);
    o.color = float4(blended, 1.0);
    o.history = float4(blended, 1.0);
    return o;
}
#elif defined(WATER_SSR)
// The sea's reflection of the scene (AC6_SSR=1): made from the scene as the effects composite leaves it
// (the world with the clouds, smoke and explosions over it, in the scene's own values) and the scene's depth, read
// by the sea's own shader in the next frame in place of its sky cube where a reflection was found (the map's pixel
// shaders, xenos/spirv_recompiler.cpp SeaReflection). textureIndex = the scene's colour, samplerIndex = the scene's
// depth, offsetX / offsetY = the device address of the parameters: the view-projection's rows (row vectors, as the
// game's shaders use them) and its inverse's, then the camera's position and the sea's height. The jitter (NDC), the clip
// z (depth x view distance) and a spare. A spare, the march's stride (pixels), the debug view and the waves' slope. The
// largest value a reflected colour may take (the 8-bit scene's codes above 240 / 255 are its emitters'), a time for the
// waves, the thickness (a share of the ray's distance) and a spare.
// Each pixel stands for the point of the sea's plane its view ray meets (where the scene shows that plane or something
// behind it: the sea's shader reads only where it draws). That point's mirror ray is
// followed across the screen in growing steps, with the depth compared in view distance: a surface the ray passes
// behind is a hit when it is near the ray (the thickness), and hides what is behind it when it is much nearer (the
// player's own aircraft). A ray that reaches the sky ends there. rgb: the scene's colour where the ray ended, a: how far
// to trust it (0: no reflection found, the sea keeps its own), fading out towards the picture's edges, so a reflection
// leaves the screen gradually.
float4 main(VsOut i) : SV_Target0
{
    Texture2D<float4> scene = g_Textures[g_Push.textureIndex];
    Texture2D<float4> depthImage = g_Textures[g_Push.samplerIndex];
    const uint64_t params = (uint64_t(asuint(g_Push.offsetY)) << 32) | asuint(g_Push.offsetX);
    float4 vp[4], inv[4];
    [unroll] for (uint r = 0; r < 4; r++)
    {
        vp[r] = vk::RawBufferLoad<float4>(params + r * 16);
        inv[r] = vk::RawBufferLoad<float4>(params + 64 + r * 16);
    }
    const float4 camera = vk::RawBufferLoad<float4>(params + 128);   // xyz, the sea's height
    const float4 view = vk::RawBufferLoad<float4>(params + 144);     // jitter x, y (NDC), clip z
    const float4 tune = vk::RawBufferLoad<float4>(params + 160);     // -, stride, debug, wave slope
    const float4 tune2 = vk::RawBufferLoad<float4>(params + 176);    // colour cap, time, thickness
    float2 size, depthSize;
    scene.GetDimensions(size.x, size.y);
    depthImage.GetDimensions(depthSize.x, depthSize.y);
    // (The picture is the steady one, after the anti-aliasing, the depth is as the scene was drawn, with the frame's
    // jitter: read where the picture's point lies in it.)
    const float2 depthShift = float2(view.x, -view.y) * 0.5;
#define SSR_DEPTH(uv) depthImage.Load(int3(clamp(int2(((uv) + depthShift) * depthSize), int2(0, 0), int2(depthSize) - 1), 0)).x
    const int debug = int(tune.z);
    // The view ray (the far point of the pixel unprojected) and its point on the sea's plane.
    const float2 ndc = float2(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0);
    const float4 h = ndc.x * inv[0] + ndc.y * inv[1] + 1e-4 * inv[2] + inv[3];
    const float3 v = normalize(h.xyz / h.w - camera.xyz);
    if (v.y >= -1e-4 || camera.y <= camera.w)
        return 0.0;
    const float dist = (camera.w - camera.y) / v.y;
    const float3 world = camera.xyz + v * dist;
    // Where the scene holds something in front of the sea's plane (land, an aircraft) no sea is seen: nothing to make
    // (most of a frame, what an aircraft hid shows for one frame with the sea's own reflection).
    const float4 clip0 = world.x * vp[0] + world.y * vp[1] + world.z * vp[2] + vp[3];
    const float w0 = clip0.w;
    if (w0 <= 0.0)
        return 0.0;
    {
        const float d = SSR_DEPTH(i.uv);
        if (d > 0.0 && view.z / d < w0 * 0.99 - 5.0)
            return 0.0;
    }
    // The water's normal: up, tilted by waves fixed in the world (sums of sines, each faded out where a pixel no longer
    // resolves it, so the far sea doesn't shimmer).
    const float footprint = dist * (2.0 / size.y) / max(-v.y, 0.03);
    float2 slope = 0.0;
    {
        const float2 dirs[4] = { float2(0.958, 0.287), float2(-0.371, 0.928), float2(0.707, -0.707), float2(-0.976, -0.217) };
        const float lengths[4] = { 160.0, 73.0, 31.0, 13.0 };
        [unroll] for (uint k = 0; k < 4; k++)
        {
            const float w = saturate(lengths[k] / (4.0 * footprint) - 0.5);
            slope += dirs[k] * cos(dot(dirs[k], world.xz) * (6.2831853 / lengths[k]) + tune2.y * (0.6 + 0.35 * k) + 1.7 * k) * w;
        }
    }
    const float3 n = normalize(float3(-slope.x * tune.w, 1.0, -slope.y * tune.w));
    float3 dir = reflect(v, n);
    dir.y = max(dir.y, 0.002);
    dir = normalize(dir);
    // The ray in clip space: clip(t) = clip0 + t x clipDir. Its far end on the screen is where it goes at infinity, or,
    // heading back past the camera, where it nears the camera's plane.
    const float4 clipDir = dir.x * vp[0] + dir.y * vp[1] + dir.z * vp[2];
    const float wd = clipDir.w;
    const bool endless = wd > w0 * 1e-7;
    const float tEnd = endless ? 0.0 : 0.95 * w0 / max(-wd, w0 * 1e-9);
    const float4 clipEnd = endless ? clipDir : clip0 + tEnd * clipDir;
    const float2 uv0 = float2(clip0.x / w0 * 0.5 + 0.5, 0.5 - clip0.y / w0 * 0.5);
    const float2 uvEnd = float2(clipEnd.x / clipEnd.w * 0.5 + 0.5, 0.5 - clipEnd.y / clipEnd.w * 0.5);
    const float2 span = uvEnd - uv0;
    // Where the screen's line leaves the picture (the share of the span).
    float sMax = 1.0;
    if (abs(span.x) > 1e-7)
        sMax = min(sMax, ((span.x > 0.0 ? 1.0 : 0.0) - uv0.x) / span.x);
    if (abs(span.y) > 1e-7)
        sMax = min(sMax, ((span.y > 0.0 ? 1.0 : 0.0) - uv0.y) / span.y);
    sMax = max(sMax, 0.0);
    const bool leaves = sMax < 1.0;
    // The steps grow by a tenth each (the first as long as the stride): fine by the sea's point, where what stands in
    // the water is reflected, coarse far along the ray, 40 at most (even steps would take up to 96).
    const float pixels = length(span * size) * sMax;
    const float growth = 1.1;
    const float count = clamp(ceil(log(1.0 + pixels * (growth - 1.0) / max(tune.y, 1.0)) / log(growth)), 4.0, 40.0);
    const int steps = int(count);
    const float whole = pow(growth, count) - 1.0;
    // (Interleaved gradient noise: the steps' banding as grain, which the anti-aliasing of the scene takes.)
    const float2 p = i.pos.xy;
    const float grain = frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
    const float wEnd = endless ? 0.0 : clipEnd.w;
    float sPrev = 0.0, sHit = -1.0;
    bool sky = false;
    [loop] for (int k = 0; k < steps; k++)
    {
        const float s = min(sMax * (pow(growth, float(k) + grain) - 1.0) / whole, 0.99995);
        // The ray's view distance at that share of the screen's line (perspective: equal steps on screen are not
        // equal steps along the ray).
        const float wq = endless ? w0 / (1.0 - s) : w0 * wEnd / (wEnd * (1.0 - s) + w0 * s);
        const float d = SSR_DEPTH(uv0 + span * s);
        const float ws = d > 0.0 ? view.z / d : 1e30;
        if (ws < wq && wq - ws < tune2.z * wq + 2.0)
        {
            sHit = s;
            break;
        }
        sPrev = s;
    }
    float2 hitUv = 0.0;
    float confidence = 0.0;
    if (sHit >= 0.0)
    {
        // The crossing, by halving.
        float lo = sPrev, hi = sHit;
        [unroll] for (int b = 0; b < 5; b++)
        {
            const float s = 0.5 * (lo + hi);
            const float wq = endless ? w0 / (1.0 - s) : w0 * wEnd / (wEnd * (1.0 - s) + w0 * s);
            const float d = SSR_DEPTH(uv0 + span * s);
            const float ws = d > 0.0 ? view.z / d : 1e30;
            if (ws < wq)
                hi = s;
            else
                lo = s;
        }
        hitUv = uv0 + span * hi;
        confidence = 1.0;
    }
    else if (endless && !leaves)
    {
        // The ray's end at infinity is on the screen: the sky there (the sky's dome has a depth of its own), or far
        // scenery the ray ends behind: anything well beyond the sea's point itself. Nearer than that it is something in
        // front of the camera (the player's aircraft), hiding where the ray went.
        hitUv = uvEnd;
        const float d = SSR_DEPTH(uvEnd);
        sky = d <= 0.0;
        confidence = sky || view.z / d > 1.5 * dist ? 1.0 : 0.0;
    }
    const float2 edge = min(hitUv, 1.0 - hitUv);
    confidence *= smoothstep(0.0, 0.08, min(edge.x, edge.y));
    if (debug == 1)
    {
        // What each ray found, in the sea's own shading: green: a surface, white: the sky, blue: scenery at the ray's far
        // end, red: nothing (the sea keeps its own reflection there), darker towards the picture's edges.
        const float3 kind = sHit >= 0.0 ? float3(0.0, 1.0, 0.0) : sky ? float3(1.0, 1.0, 1.0) : float3(0.0, 0.0, 1.0);
        return float4(lerp(float3(1.0, 0.0, 0.0), kind, confidence) * min(tune2.x, 1.0), 1.0);
    }
    if (confidence <= 0.0)
        return 0.0;
    return float4(min(scene.SampleLevel(g_Samplers[0], hitUv, 0).rgb, tune2.x), confidence);
}
#elif defined(PLANAR_BLUR)
// The sea's planar reflection (AC6_SSR=planar, RunPlanarReflection), blurred in two passes. textureIndex = the image
// read, samplerIndex = the mirrored scene's depth (the first pass), offsetX / offsetY = the device address of the
// parameters: the view-projection's rows and its inverse's, the camera and the sea's height, then the pass (0 across,
// 1 down, 2 and 3 the glow across and down), the blur's sigma in pixels, the largest value a colour may take and the
// waves' slope, then a time, the debug view, whether the sky's layers count and the image of them drawn over white,
// then the downward stretch, the bright texels' weight, whether the scene is float, the glow's strength, and its sigma.
// In the down pass samplerIndex is the glow's image (0: none). Then the blur by height's parameters (the depth,
// roughness, the largest sigma, and the way back to the world) and the lights' (strength, streak length, their image).
// Pass 4: the lights, the texels much brighter than those around them, which pass 3 stretches into streaks.
// Across: from the mirrored scene, its coverage from its depth (0: nothing drawn there). Down: into the image the sea
// reads, each pixel moved by the waves on the sea's point it stands for. Both weigh a colour less the brighter it is
// (bright specks, such as a glint in a low-resolution picture, don't spread into a blob), and keep colour and coverage
// apart: rgb is the covered pixels' colour, a how much of the pixel is covered.
// The scene's values as light: the 8-bit scene's encoding decoded (below 241 / 255 x 255 / 240, else per channel codes
// 241-245 are 1.1-1.5 and 246-255 10-100: the game's emitters), the float scene's x 255 / 240.
float3 PlanarDecode(float3 c, bool floatScene)
{
    if (floatScene || max(c.r, max(c.g, c.b)) < 241.0 / 255.0)
        return max(c, 0.0) * (255.0 / 240.0);
    const float3 q = round(saturate(c) * 255.0);
    const float3 n = max(q - 240.0, 0.0);
    return min(q, 240.0) / 240.0 + min(n, 5.0) * 0.1 + max(n - 5.0, 0.0) * 10.0 - 1.5 * step(5.5, n);
}

// What the game's bright pass takes of a value (as the enhanced bloom's BLOOM_BRIGHT): x the exposure (from the adapted
// luminance and the band, tune4.yz, the luminance image tune5.y, none: 1: PlanarExposure, once a pixel),
// x (1 - threshold / luminance) (tune4.w), then x / (x + offset) (tune5.x), and back in the scene's values (/ the exposure).
float PlanarExposure(float4 tune4, float4 tune5)
{
    float exposure = 1.0;
    if (asuint(tune5.y) != 0)
    {
        const float L = max(g_Textures[asuint(tune5.y)].SampleLevel(g_Samplers[1023], float2(0.5, 0.5), 0).x, 1.0e-6);
        exposure = L < tune4.y ? tune4.y / L : L > tune4.z ? tune4.z / L : 1.0;
    }
    return exposure;
}

float3 PlanarBright(float3 x, float exposure, float4 tune4, float4 tune5)
{
    x *= exposure;
    const float luminance = max(dot(x, float3(0.2125, 0.7154, 0.0721)), 1.0e-6);
    x *= max(1.0 - tune4.w / luminance, 0.0);
    return x / (x + max(tune5.x, 1.0e-3)) / exposure;
}

// The way reflections stretch at this pixel: the world's vertical on screen at the sea's point the pixel stands for
// (xy, a unit vector in pixels: with the camera rolled it isn't the screen's), and the stretch there (z): rippled water
// stretches what it mirrors along that line, the most at a grazing view and not at all looking straight down
// (1 + (stretch - 1) x cos^2 of the view's angle to the water).
float3 PlanarAxis(float2 uv, uint64_t params, float2 size, float stretch)
{
    float4 vp[4], inv[4];
    [unroll] for (uint r = 0; r < 4; r++)
    {
        vp[r] = vk::RawBufferLoad<float4>(params + r * 16);
        inv[r] = vk::RawBufferLoad<float4>(params + 64 + r * 16);
    }
    const float4 camera = vk::RawBufferLoad<float4>(params + 128);
    const float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    const float4 h = ndc.x * inv[0] + ndc.y * inv[1] + 1e-4 * inv[2] + inv[3];
    const float3 v = normalize(h.xyz / h.w - camera.xyz);
    if (v.y >= -1e-4 || camera.y <= camera.w)
        return float3(0.0, 1.0, stretch);
    const float dist = (camera.w - camera.y) / v.y;
    const float3 world = camera.xyz + v * dist;
    const float4 c0 = world.x * vp[0] + world.y * vp[1] + world.z * vp[2] + vp[3];
    const float4 c1 = c0 + vp[1] * (dist * 0.01);
    const float stretchHere = 1.0 + (stretch - 1.0) * (1.0 - v.y * v.y);
    if (c0.w <= 0.0 || c1.w <= 0.0)
        return float3(0.0, 1.0, stretchHere);
    const float2 d = (float2(c1.x / c1.w - c0.x / c0.w, c0.y / c0.w - c1.y / c1.w) * 0.5) * size;
    const float length2 = dot(d, d);
    return length2 > 1e-12 ? float3(d * rsqrt(length2), stretchHere) : float3(0.0, 1.0, stretchHere);
}

float4 PlanarTap(float4 c, float g, inout float3 colour, inout float colourWeight, inout float cover, inout float total, float cap,
    float specks)
{
    const float3 rgb = min(c.rgb, cap);
    const float weight = g * c.a / (1.0 + specks * max(rgb.r, max(rgb.g, rgb.b)));
    colour += rgb * weight;
    colourWeight += weight;
    cover += g * c.a;
    total += g;
    return c;
}

float4 main(VsOut i) : SV_Target0
{
    Texture2D<float4> source = g_Textures[g_Push.textureIndex];
    const uint64_t params = (uint64_t(asuint(g_Push.offsetY)) << 32) | asuint(g_Push.offsetX);
    const float4 tune = vk::RawBufferLoad<float4>(params + 144);    // pass, sigma, colour cap, wave slope
    const float4 tune2 = vk::RawBufferLoad<float4>(params + 160);   // time
    float2 size;
    source.GetDimensions(size.x, size.y);
    const float4 tune3 = vk::RawBufferLoad<float4>(params + 176);   // vertical stretch, specks' weight, float scene, glow
    const float4 tune4 = vk::RawBufferLoad<float4>(params + 192);   // the glow's sigma, the exposure band, the threshold
    const float4 tune5 = vk::RawBufferLoad<float4>(params + 208);   // the bright pass's offset, the adapted luminance
    const bool floatScene = tune3.z > 0.5;
    const float4 tune7 = vk::RawBufferLoad<float4>(params + 304);   // the lights' strength, streak length, their image
    if (tune.x > 3.5)
    {
        // The lights (4): points much brighter than the texels on both sides of them, across and down (a lamp, a
        // glint: a texel or two at this size), and bright themselves. A line (a lit edge, a cable) is as bright as the
        // texels along it, so it isn't one. From the mirrored scene, decoded.
        const int2 p = int2(i.pos.xy);
        const int2 top = int2(size) - 1;
        const float3 c = PlanarDecode(source.Load(int3(p, 0)).rgb, floatScene);
        // (The pairs across, down and along both diagonals: a line in any of those directions isn't a point.)
        const int2 pairs[4] = { int2(1, 0), int2(0, 1), int2(1, 1), int2(1, -1) };
        float3 sides = 0.0;
        [unroll] for (uint n = 0; n < 4; n++)
            sides = max(sides, 0.5 * (PlanarDecode(source.Load(int3(clamp(p + pairs[n], int2(0, 0), top), 0)).rgb, floatScene) +
                PlanarDecode(source.Load(int3(clamp(p - pairs[n], int2(0, 0), top), 0)).rgb, floatScene)));
        const float luminance = dot(c, float3(0.2125, 0.7154, 0.0721));
        return float4(max(c - sides * 1.25, 0.0) * smoothstep(0.4, 0.8, luminance), 1.0);
    }
    if (tune.x > 1.5)
    {
        // The glow: what is brighter than white, blurred across (2, from the mirrored scene) or down (3, longer).
        // tune5.w: the mirrored scene's pixels to one of the glow's (2: the glow at half its size, each tap across the
        // mean of the 2 x 2 texels it covers; down reads the half-size image at its own size). The sigma is in the
        // mirrored scene's pixels.
        const bool across = tune.x < 2.5;
        const float step = max(tune5.w, 1.0);
        const float3 axis = PlanarAxis(i.uv, params, size, max(tune3.x, 1.0));
        const float2 dir = across ? float2(-axis.y, axis.x) : axis.xy;
        const float glowSigma = max(tune4.x, 0.05) * (across ? 1.0 : axis.z) / step;
        const int glowRadius = min(int(ceil(glowSigma * 2.5)), 40);
        const int2 p = int2(i.pos.xy);
        const int2 top = int2(size) - 1;
        const float exposure = across ? PlanarExposure(tune4, tune5) : 1.0;
        const float tapStep = across ? step : 1.0;
        float3 sum = 0.0;
        float total = 0.0;
        for (int k = -glowRadius; k <= glowRadius; k++)
        {
            const float2 at = i.pos.xy * tapStep + dir * (float(k) * tapStep);
            const float g = exp(-0.5 * float(k * k) / (glowSigma * glowSigma));
            if (across && step > 1.5)
            {
                const int2 q = int2(floor(at - 0.5));
                float3 c4 = 0.0;
                [unroll] for (uint n = 0; n < 4; n++)
                {
                    const int2 t = clamp(q + int2(n & 1, n >> 1), int2(0, 0), top);
                    c4 += PlanarBright(PlanarDecode(source.Load(int3(t, 0)).rgb, floatScene), exposure, tune4, tune5);
                }
                sum += c4 * (0.25 * g);
            }
            else
            {
                const int2 q = clamp(int2(floor(at)), int2(0, 0), top);
                const float3 c = source.Load(int3(q, 0)).rgb;
                sum += (across ? PlanarBright(PlanarDecode(c, floatScene), exposure, tune4, tune5) : c) * g;
            }
            total += g;
        }
        if (across)
            return float4(sum / total, 1.0);
        // Down, with the lights' streaks: each light keeps its brightness along the streak, fading with the distance
        // (the largest of the lights above and below, not their average: on rippled water every wave facet that
        // catches a lamp shows it at full brightness).
        float3 streak = 0.0;
        if (asuint(tune7.z) != 0 && tune7.x > 0.0)
        {
            const float streakLength = max(tune7.y * axis.z / max(tune3.x, 1.0), 1.0);
            const int reach = min(int(ceil(streakLength * 3.0)), 48);
            for (int k = -reach; k <= reach; k++)
            {
                const int2 q = clamp(int2(floor(i.pos.xy + axis.xy * float(k))), int2(0, 0), top);
                const float3 l = g_Textures[asuint(tune7.z)].Load(int3(q, 0)).rgb;
                streak = max(streak, l * exp(-abs(float(k)) / streakLength));
            }
        }
        return float4(sum / total * tune3.w + streak * tune7.x, 1.0);
    }
    // The blur grows with how far above the water the reflected surface stands, against the water's distance (tune6.y
    // the water's roughness, 0: the blur's sigma everywhere): rippled water mirrors what touches it sharply and what is
    // high above it blurred, a light's streak sharpest at its base. The pixel's surface: its depth (the mirrored
    // scene's, tune6.x) back through the mirrored camera (the rows after tune6) to the world, its height above the sea.
    // Where nothing wrote depth (the sky and its layers) it is the most.
    const float4 tune6 = vk::RawBufferLoad<float4>(params + 224);   // the depth, roughness, the largest sigma
    float sigma = max(tune.y, 0.05);
    if (tune6.y > 0.0)
    {
        const int2 p = int2(i.pos.xy);
        const float depth = g_Textures[asuint(tune6.x)].Load(int3(p, 0)).x;
        float4 vp[4], inv[4], back[4];
        [unroll] for (uint r = 0; r < 4; r++)
        {
            vp[r] = vk::RawBufferLoad<float4>(params + r * 16);
            inv[r] = vk::RawBufferLoad<float4>(params + 64 + r * 16);
            back[r] = vk::RawBufferLoad<float4>(params + 240 + r * 16);
        }
        const float4 camera = vk::RawBufferLoad<float4>(params + 128);
        const float2 ndc = float2(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0);
        const float4 h = ndc.x * inv[0] + ndc.y * inv[1] + 1e-4 * inv[2] + inv[3];
        const float3 v = normalize(h.xyz / h.w - camera.xyz);
        float most = tune6.z;
        if (depth > 0.0 && v.y < -1e-4)
        {
            const float dist = (camera.w - camera.y) / v.y;
            const float4 w = ndc.x * back[0] + ndc.y * back[1] + depth * back[2] + back[3];
            const float height = max(w.y / w.w - camera.w, 0.0);
            // (Pixels per radian: the image's half height x the projection's vertical scale.)
            const float focal = size.y * 0.5 * length(float3(vp[0].y, vp[1].y, vp[2].y));
            most = min(sigma + tune6.y * height / max(dist, 1.0) * focal, tune6.z);
        }
        sigma = most;
    }
    // (Down, along the water's vertical, the blur is longer: rippled water stretches reflections, the lights' most of
    // all, into streaks.)
    const float3 axis = PlanarAxis(i.uv, params, size, max(tune3.x, 1.0));
    sigma *= tune.x < 0.5 ? 1.0 : axis.z;
    const int radius = min(int(ceil(sigma * 2.5)), 32);
    float3 colour = 0.0;
    float colourWeight = 0.0, cover = 0.0, total = 0.0;
    if (tune.x < 0.5)
    {
        Texture2D<float4> depthImage = g_Textures[g_Push.samplerIndex];
        const float2 across = float2(-axis.y, axis.x);
        const int2 top = int2(size) - 1;
        for (int k = -radius; k <= radius; k++)
        {
            const int2 q = clamp(int2(floor(i.pos.xy + across * float(k))), int2(0, 0), top);
            // Covered where something wrote depth. Elsewhere (with the sky's layers, tune2.z) the layers' coverage:
            // 1 - what they let through (the layers drawn again over white, tune2.w: C + T against C over black),
            // and their own colour C / coverage.
            float3 rgb = source.Load(int3(q, 0)).rgb;
            float covered = depthImage.Load(int3(q, 0)).x > 0.0 ? 1.0 : 0.0;
            if (covered == 0.0 && tune2.z > 0.5)
            {
                const float3 through = saturate(g_Textures[asuint(tune2.w)].Load(int3(q, 0)).rgb - rgb);
                covered = saturate(1.0 - (through.r + through.g + through.b) / 3.0);
                rgb = covered > 0.02 ? rgb / covered : 0.0;
            }
            rgb = PlanarDecode(rgb, floatScene);
            PlanarTap(float4(rgb, covered), exp(-0.5 * float(k * k) / (sigma * sigma)), colour, colourWeight,
                cover, total, tune.z, tune3.y);
        }
        return float4(colourWeight > 0.0 ? colour / colourWeight : 0.0, cover / total);
    }
    float2 uv = i.uv;
    if (tune.w > 0.0)
    {
        // The sea's point this pixel stands for (the view ray meeting the plane), and the waves there: sums of sines
        // fixed in the world, each faded out where a pixel no longer resolves it. The picture is read moved by the
        // waves' slope, a share of the distance.
        float4 vp[4], inv[4];
        [unroll] for (uint r = 0; r < 4; r++)
        {
            vp[r] = vk::RawBufferLoad<float4>(params + r * 16);
            inv[r] = vk::RawBufferLoad<float4>(params + 64 + r * 16);
        }
        const float4 camera = vk::RawBufferLoad<float4>(params + 128);
        const float2 ndc = float2(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0);
        const float4 h = ndc.x * inv[0] + ndc.y * inv[1] + 1e-4 * inv[2] + inv[3];
        const float3 v = normalize(h.xyz / h.w - camera.xyz);
        if (v.y < -1e-4 && camera.y > camera.w)
        {
            const float dist = (camera.w - camera.y) / v.y;
            const float3 world = camera.xyz + v * dist;
            const float footprint = dist * (2.0 / size.y) / max(-v.y, 0.03);
            float2 slope = 0.0;
            const float2 dirs[4] = { float2(0.958, 0.287), float2(-0.371, 0.928), float2(0.707, -0.707), float2(-0.976, -0.217) };
            const float lengths[4] = { 160.0, 73.0, 31.0, 13.0 };
            [unroll] for (uint k = 0; k < 4; k++)
            {
                const float w = saturate(lengths[k] / (4.0 * footprint) - 0.5);
                slope += dirs[k] * cos(dot(dirs[k], world.xz) * (6.2831853 / lengths[k]) + tune2.x * (0.6 + 0.35 * k) + 1.7 * k) * w;
            }
            const float3 moved = world + float3(slope.x, 0.0, slope.y) * (tune.w * dist);
            const float4 clip = moved.x * vp[0] + moved.y * vp[1] + moved.z * vp[2] + vp[3];
            if (clip.w > 0.0)
                uv = clamp(float2(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5), 0.0, 1.0);
        }
    }
    for (int k = -radius; k <= radius; k++)
    {
        const float4 c = source.SampleLevel(g_Samplers[0], uv + axis.xy * float(k) / size, 0);
        PlanarTap(c, exp(-0.5 * float(k * k) / (sigma * sigma)), colour, colourWeight, cover, total, tune.z, tune3.y);
    }
    // The lights' glow (samplerIndex: its image, 0 without), read where the waves moved this pixel.
    const float3 glow = g_Push.samplerIndex != 0 ? g_Textures[g_Push.samplerIndex].SampleLevel(g_Samplers[0], uv, 0).rgb : 0.0;
    const float3 reflected = (colourWeight > 0.0 ? colour / colourWeight : 0.0) + glow;
    // AC6_PLANAR_DEBUG=1: the reflection alone on the sea, magenta where nothing was drawn.
    if (tune2.y > 0.5)
        return float4(lerp(float3(1.0, 0.0, 1.0), reflected, cover / total), 1.0);
    return float4(reflected, cover / total);
}
#elif defined(MOTION_COMPLETE)
// FSR 3.1's motion vectors: every pixel's, from this frame's unjittered position to last frame's, in
// texture coordinates (FSR scales them by the render size). textureIndex = the draws' motion image (0: none),
// samplerIndex = the scene depth, offsetX / offsetY = the device address of the parameters: the rows of R (as TAA's: this
// frame's clip position to last frame's, for the world's view-projection), then the jitter in NDC. Where the draws wrote
// their motion it is that, turned around. Elsewhere (the sky, things new this frame) the camera's reprojection of the depth.
// Target 1: FSR's reactive mask, the draws' motion image's alpha (the translucent draws' coverage: smoke, fire, trails,
// glows, which FSR then takes more of from this frame than from its history).
struct MotionOut
{
    float4 motion : SV_Target0;
    float4 reactive : SV_Target1;
};

float2 CompleteMotion(VsOut i, float4 drawn)
{
    const uint64_t params = (uint64_t(asuint(g_Push.offsetY)) << 32) | asuint(g_Push.offsetX);
    const float4 r0 = vk::RawBufferLoad<float4>(params), r1 = vk::RawBufferLoad<float4>(params + 16);
    const float4 r2 = vk::RawBufferLoad<float4>(params + 32), r3 = vk::RawBufferLoad<float4>(params + 48);
    const float2 jitter = vk::RawBufferLoad<float4>(params + 64).xy;
    const float2 current = float2(i.uv.x - jitter.x * 0.5, i.uv.y + jitter.y * 0.5);   // the unjittered position
    if (g_Push.textureIndex != 0 && all(abs(drawn.xy) < 4.0))
        return -drawn.xy;
    Texture2D<float4> depthImage = g_Textures[g_Push.samplerIndex];
    float2 depthSize;
    depthImage.GetDimensions(depthSize.x, depthSize.y);
    const float depth = depthImage.Load(int3(min(int2(i.uv * depthSize), int2(depthSize) - 1), 0)).x;
    const float2 ndc = float2(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0) - jitter;
    const float4 prev = ndc.x * r0 + ndc.y * r1 + depth * r2 + r3;
    if (prev.w <= 0.0)
        return float2(0.0, 0.0);
    const float2 prevUv = float2(prev.x / prev.w * 0.5 + 0.5, 0.5 - prev.y / prev.w * 0.5);
    return prevUv - current;
}

MotionOut main(VsOut i)
{
    const float4 drawn = g_Push.textureIndex != 0 ? g_Textures[g_Push.textureIndex].Load(int3(int2(i.pos.xy), 0)) : float4(0.0, 0.0, 0.0, 0.0);
    MotionOut o;
    o.motion = float4(CompleteMotion(i, drawn), 0.0, 0.0);
    o.reactive = float4(saturate(drawn.w), 0.0, 0.0, 0.0);
    return o;
}
#elif defined(SHARPEN)
// Sharpening after temporal AA, on what is shown only (the history stays unsharpened): an unsharp
// mask of the four neighbours, the result kept within their range (no halos). offsetX = the strength's float bits.
float4 main(VsOut i) : SV_Target0
{
    Texture2D<float4> image = g_Textures[g_Push.textureIndex];
    int2 size;
    image.GetDimensions(size.x, size.y);
    const int2 p = int2(i.pos.xy);
    const float3 c = image.Load(int3(p, 0)).rgb;
    const float3 n = image.Load(int3(clamp(p + int2(0, -1), 0, size - 1), 0)).rgb;
    const float3 s = image.Load(int3(clamp(p + int2(0, 1), 0, size - 1), 0)).rgb;
    const float3 w = image.Load(int3(clamp(p + int2(-1, 0), 0, size - 1), 0)).rgb;
    const float3 e = image.Load(int3(clamp(p + int2(1, 0), 0, size - 1), 0)).rgb;
    const float3 lo = min(c, min(min(n, s), min(w, e))), hi = max(c, max(max(n, s), max(w, e)));
    const float strength = asfloat(g_Push.offsetX);
    const float3 sharpened = c + strength * (4.0 * c - n - s - w - e) * 0.5;
    return float4(clamp(sharpened, lo, hi), 1.0);
}
#elif defined(GLOW_MERGE)
// A multisampled layer's picture: its samples' mean (the scene's lights added into the effects
// pass's layer before its first cloud, drawn with an additive blend).
[[vk::binding(0, 0)]] Texture2DMS<float4> g_MergeSamples[];
float4 main(VsOut i) : SV_Target0
{
    Texture2DMS<float4> layer = g_MergeSamples[g_Push.textureIndex];
    uint w, h, n;
    layer.GetDimensions(w, h, n);
    float4 sum = 0.0;
    [loop] for (uint k = 0; k < n; k++)
        sum += layer.Load(int2(i.pos.xy), int(k));
    return sum / float(n);
}
#elif defined(HDR_COMPOSE)
// A layer at the canvas's pixel, or sampled where it is another size (with FSR upscaling the world's layers are
// at the render size).
float3 ComposeLayer(uint index, float2 uv, int3 p)
{
    Texture2D<float4> layer = g_Textures[index];
    uint lw, lh, cw, ch;
    layer.GetDimensions(lw, lh);
    g_Textures[g_Push.textureIndex].GetDimensions(cw, ch);
    if (lw == cw && lh == ch)
        return layer.Load(p).rgb;
    return layer.SampleLevel(g_Samplers[0], uv, 0.0).rgb;
}
// A multisampled layer as drawn (without its resolve): its samples' mean at the canvas's pixel.
[[vk::binding(0, 0)]] Texture2DMS<float4> g_LayerSamples[];
float3 ComposeLayerSamples(uint index, int2 p)
{
    Texture2DMS<float4> layer = g_LayerSamples[index];
    uint w, h, n;
    layer.GetDimensions(w, h, n);
    float3 sum = 0.0;
    [loop] for (uint k = 0; k < n; k++)
        sum += layer.Load(p, int(k)).rgb;
    return sum / float(n);
}
// HDR from the scene, after the scene composite: the float canvas (textureIndex: as it was before the
// pass, grade1's two low bits: the multisampled glow layer (1) and the clouds' layer (2) are read by their samples) with its lit
// highlights above white compressed (offsetY, as float bits: how much of the range above 1 is kept, 0-1, by the brightest
// channel, the colour kept), plus the effects' float glow (samplerIndex: the single-sampled layer in bits 0-15, the resolved
// multisampled one in 16-31, 0 = none) as values (x 255/240), rolled off towards a ceiling (offsetX, float bits: dense
// overlaps of additive sprites sum without the 8-bit clip, a fireball's to 80x white: L (1 - exp(-g / L)) by the
// brightest channel, slope 1 for faint glows).
float4 main(VsOut i) : SV_Target0
{
    const int3 p = int3(int2(i.pos.xy), 0);
    float4 base = g_Textures[g_Push.textureIndex].Load(p);
    const float keep = asfloat(g_Push.offsetY);
    const float m = max(base.r, max(base.g, base.b));
    if (m > 1.0)
        base.rgb *= (1.0 + (m - 1.0) * keep) / m;
    // The scene's ceiling (stripes, as float bits, 0 = none): the sun's disc decodes at 100x white beside its glare's
    // 1-2, and would show as a hard-edged ball at the display's peak (SDR's clip merges them). After the game's bloom,
    // what lies above a knee (0.6 x the ceiling) rolls off towards the ceiling by the brightest channel, the colour
    // kept. A second run of the pass over the sun's box only (bit 31) also spreads that light: each sample is rolled
    // off first towards the sun's own ceiling (the disc's 100x would saturate every sample touching it), then summed
    // over a small Gaussian (sigma 8 pixels at 1440 lines: the light's own bloom), which leaves the disc a soft core.
    // Over the whole frame the spread would be costly.
    // (The sun's pass: bit 31 set, bits 0-15 the frame's ceiling x 256 (its knee), 16-30 the sun's ceiling x 128.)
    const bool spread = (g_Push.stripes & 0x80000000u) != 0;
    const float ceilingWord = spread ? 0.0 : asfloat(g_Push.stripes);
    const float top = ceilingWord;
    if (ceilingWord > 0.0)
    {
        const float knee = 0.6 * top, span = top - knee, mb = max(base.r, max(base.g, base.b));
        if (mb > knee)
            base.rgb *= (knee + span * (1.0 - exp(-(mb - knee) / span))) / mb;
    }
    else if (spread)
    {
        const float knee = 0.6 * float(g_Push.stripes & 0xFFFFu) / 256.0;
        const float span = max(float((g_Push.stripes >> 16) & 0x7FFFu) / 128.0 - knee, 1e-3);
        uint w, h;
        g_Textures[g_Push.textureIndex].GetDimensions(w, h);
        const float sigma = max(1.0, float(h) / 1440.0 * 8.0);
        const int2 last = int2(int(w) - 1, int(h) - 1);
        const float mb = max(base.r, max(base.g, base.b));
        // (A dense grid, affordable as the pass covers the sun's box only: 3 rings of 12 samples would draw rings round
        // the core when the sun's ceiling is above the frame's.)
        const int stepPx = max(1, int(sigma * 0.5));
        const int reach = int(sigma * 3.0) / stepPx;
        float3 excess = 0.0;
        float total = 0.0;
        [loop] for (int dy = -reach; dy <= reach; dy++)
        {
            [loop] for (int dx = -reach; dx <= reach; dx++)
            {
                const float2 d = float2(dx, dy) * float(stepPx);
                const float weight = exp(-0.5 * dot(d, d) / (sigma * sigma));
                const int2 q = clamp(int2(i.pos.xy) + int2(d), int2(0, 0), last);
                float3 t = g_Textures[g_Push.textureIndex].Load(int3(q, 0)).rgb;
                const float mt = max(t.r, max(t.g, t.b));
                if (mt > 1.0)
                    t *= (1.0 + (mt - 1.0) * keep) / mt;
                const float m2 = max(t.r, max(t.g, t.b));
                excess += t / max(m2, 1e-6) * (span * (1.0 - exp(-max(m2 - knee, 0.0) / span))) * weight;
                total += weight;
            }
        }
        excess /= total;
        if (mb > knee || max(excess.r, max(excess.g, excess.b)) > 0.0)
            base.rgb = base.rgb * (min(mb, knee) / max(mb, 1e-6)) + excess;
    }
    // The clouds (grade0: the cloud particles' layer, their contribution over nothing, 0 = none, grade1:
    // the cloud lift as float bits): that contribution's light added again x (lift - 1), weighted by how bright it is (from
    // 0.15 of white up to full at 0.5: sunlit cloud, not its shade), in light (the canvas is 2.2-encoded).
    if (g_Push.grade0 != 0)
    {
        const float3 cloud = max((g_Push.grade1 & 2u) != 0 ? ComposeLayerSamples(g_Push.grade0, p.xy) : ComposeLayer(g_Push.grade0, i.uv, p), 0.0) *
            (255.0 / 240.0);
        const float3 cloudLin = pow(cloud, 2.2);
        const float y = dot(cloudLin, float3(0.2126, 0.7152, 0.0722));
        const float weight = smoothstep(0.15, 0.5, y);
        if (weight > 0.0)
        {
            const float3 baseLin = pow(max(base.rgb, 0.0), 2.2);
            base.rgb = pow(baseLin + cloudLin * (asfloat(g_Push.grade1 & ~3u) - 1.0) * weight, 1.0 / 2.2);
        }
    }
    float3 glow = 0.0;
    const uint a = g_Push.samplerIndex & 0xFFFF, b = g_Push.samplerIndex >> 16;
    if (a != 0)
        glow += ComposeLayer(a, i.uv, p);
    if (b != 0)
        glow += (g_Push.grade1 & 1u) != 0 ? ComposeLayerSamples(b, p.xy) : ComposeLayer(b, i.uv, p);
    glow *= 255.0 / 240.0;
    const float ceiling = asfloat(g_Push.offsetX), g = max(glow.r, max(glow.g, glow.b));
    if (g > 0.0)
        glow *= ceiling * (1.0 - exp(-g / ceiling)) / g;
    return float4(base.rgb + glow, base.a);
}
#elif defined(OUTPUT)
// The frame to the window: debanding, dithering and the HDR encode.
// samplerIndex: bit 0 deband, bit 1 dither, bit 2 scRGB (linear, x offsetX's paper-white scale), bit 3 dither to 10 bits
// (deep output) instead of 8, bits 8-15 the deband radius in source texels, bits 16-23 its threshold in 1/16 of an 8-bit
// step, bits 24-31 the HDR peak in 50-nit units (0: none), bits 4-5 the HDR tone mapping: 2 RenoDRT, any other value
// PsychoV-30 (the renderer sends 2 or 3), bit 6 the UI mask in red (AC6_HDR_UI_DEBUG), bit 7 the frame all UI.
// offsetY = the frame number (the dither's noise moves every frame). grade0 / grade1: the grading (RenoDRT, PsychoV-30).
#include "../../../thirdparty/renodx/src/shaders/tonemap/reno_drt.hlsl"
#include "../../../thirdparty/renodx/src/shaders/tonemap/psychov/test30.hlsl"
#include "../../../thirdparty/renodx/src/shaders/inverse_tonemap.hlsl"

// The grading, as RenoDX's mods offer it (their sliders' 0-100 with 50 = 1.0, x 0.02).
struct HdrGrade
{
    float exposure, highlights, shadows, contrast, saturation;
    float gamutCompression;   // PsychoV-30: 0-1
    float compression;        // PsychoV-30's finite-response power h: 0 = automatic, else 0-2
    float hueCorrection;      // towards the SDR picture's hue (the game's clip at 1): 0-1
};
HdrGrade Grade()
{
    HdrGrade g;
    g.exposure = 0.5 + float(g_Push.grade0 & 255) * 0.01;
    g.highlights = float((g_Push.grade0 >> 8) & 255) * 0.02;
    g.shadows = float((g_Push.grade0 >> 16) & 255) * 0.02;
    g.contrast = float(g_Push.grade0 >> 24) * 0.02;
    g.saturation = float(g_Push.grade1 & 255) * 0.02;
    g.gamutCompression = float((g_Push.grade1 >> 8) & 255) * 0.01;
    g.compression = float((g_Push.grade1 >> 16) & 255) * 0.01;
    g.hueCorrection = float((g_Push.grade1 >> 25) & 127) * 0.01;
    return g;
}

// Scene light from the SDR picture (RenoDRT's and PsychoV-30's): a mid-grey-preserving inverse Reinhard of luminance
// (0.18 stays 0.18), the colour kept. Its forward curve's log-log slope at mid grey is 1 - 0.18 = 0.82.
static const float kInverseSlopeAtMidGray = 0.82;
float3 SceneFromSdr(float3 lin)
{
    const float y = renodx::color::y::from::BT709(lin);
    const float scene = renodx::tonemap::inverse::ReinhardScalable(min(y, 0.999), 1.0, 0.0, 0.18, 0.18);
    return lin * (y > 0.0 ? scene / y : 0.0);
}

// The HDR colour towards the SDR picture's, for HDR from the scene: `hdr` the tone mapped colour
// (paper white units), `lin` the frame's value (the SDR picture is min(lin, 1)). RenoDX's hue correction takes the SDR
// colour's hue and keeps the HDR colour's chroma. Where the game's clip makes the SDR colour white (every channel above
// 1: the sun's glare) its hue is rounding noise (OKLab's white is not exactly 0), and the HDR colour would get that hue
// at full chroma: a flat beige star with an edge where the last channel crosses 1. Here the
// hue pull fades out as the SDR colour's chroma does, and where the picture clips (by max(lin) - 1, full at 1.25) the
// chroma is capped at the SDR colour's chromaticity at the HDR colour's lightness: what SDR shows white goes white.
float3 SdrColour(float3 hdr, float3 lin, float strength)
{
    if (strength <= 0.0)
        return hdr;
    float3 h = renodx::color::oklab::from::BT709(hdr);
    const float3 s = renodx::color::oklab::from::BT709(min(lin, 1.0));
    const float ch = length(h.yz), cs = length(s.yz);
    float2 ab = lerp(h.yz, s.yz, strength * saturate(cs * 50.0));
    const float clipped = saturate((max(lin.r, max(lin.g, lin.b)) - 1.0) * 4.0);
    const float target = lerp(ch, min(ch, cs * h.x / max(s.x, 1e-4)), clipped * strength);
    const float len = length(ab);
    h.yz = len > 1e-7 ? ab * (target / len) : float2(0.0, 0.0);
    return renodx::color::bt709::clamp::AP1(renodx::color::bt709::from::OkLab(h));
}

float3 HdrRenoDrt(float3 lin, float paperNits, float peakNits, HdrGrade g, bool scene)
{
    // RenoDRT (RenoDX): scene light (SceneFromSdr, with HDR from the scene, the frame's own values), then RenoDRT's
    // Daniele tone scale to the peak (relative to paper white = 100), hue corrected towards the SDR colour in OKLab (the
    // game's clip at 1 for the scene's values), highlights dechromatised. Mid grey (0.18) stays at 0.18 x paper white.
    renodx::tonemap::renodrt::Config config = renodx::tonemap::renodrt::config::Create();
    config.nits_peak = peakNits / paperNits * 100.0;
    config.mid_gray_value = 0.18;
    config.mid_gray_nits = 18.0;
    config.exposure = g.exposure;
    config.highlights = g.highlights;
    config.shadows = g.shadows;
    config.contrast = g.contrast;
    config.saturation = g.saturation;
    config.hue_correction_type = renodx::tonemap::renodrt::config::hue_correction_type::CUSTOM;
    config.hue_correction_source = min(lin, 1.0);
    // From the scene: SdrColour after the tone scale instead (RenoDRT's own would take the noise hue of clipped whites).
    config.hue_correction_strength = scene ? 0.0 : g.hueCorrection;
    float3 hdr = max(0.0, renodx::tonemap::renodrt::BT709(scene ? lin : SceneFromSdr(lin), config));
    if (scene)
        hdr = max(0.0, SdrColour(hdr, lin, g.hueCorrection));
    return hdr * paperNits;   // 1.0 = paper white
}

float3 HdrPsychoV30(float3 lin, float paperNits, float peakNits, HdrGrade g, bool scene)
{
    // PsychoV-30 (RenoDX, Carlos Lopez: tonemap/psychov/test30.hlsl): a cone-response (LMS) tone scale with its anchor at
    // mid grey (0.18 in, 0.18 out) and its top at the peak (in paper whites), then the colour projected into BT.2020 at
    // the peak. Called as RenoDX's mods call it (orderofthesinkingstar/tonemap.hlsli): exposure, highlights, shadows,
    // contrast, purity (saturation), the cone response, gamut compression, compression, the placeholders at theirs. Its
    // "slope reference" (the original curve's contrast kept): the cone response is the SDR curve's slope at mid grey,
    // the inverse's (0.82), so the midtones' contrast stays the SDR picture's at contrast 1. With HDR from the scene
    // the frame's own values, and the game's curve is a clip (slope 1): 1.
    const float3 hdr = renodx::tonemap::psychov::psychotm_test30(scene ? lin : SceneFromSdr(lin), peakNits / paperNits,
        g.exposure, g.highlights, g.shadows, g.contrast, g.saturation, 0.0, 100.0, 1.0, 1.0, 1,
        scene ? 1.0 : kInverseSlopeAtMidGray, 0.18, 0.18, g.gamutCompression, 1, 1.0, g.compression);
    // Hue correction (RenoDX's wiki "Hue Correction"): the hue pulled towards the SDR picture's, the game's clip at 1
    // (the grade's per-channel gamma above 1 would turn the sun orange where the SDR picture has it white). Below 1 none.
    // From the scene, SdrColour (clipped whites stay white), else RenoDX's (lin is the picture, nothing clipped).
    if (scene)
        return max(0.0, SdrColour(hdr, lin, g.hueCorrection)) * paperNits;
    return renodx::color::correct::Hue(hdr, min(lin, 1.0), g.hueCorrection) * paperNits;   // linear BT.709 (scRGB), 1.0 = paper white
}

uint PcgHash(uint v)
{
    const uint state = v * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float Rand(uint2 p, uint salt)
{
    return float(PcgHash(p.x ^ PcgHash(p.y ^ PcgHash(salt)))) * (1.0 / 4294967296.0);
}
// Brightness stripes (AC6_HDR_STRIPES): diagonal stripes, every other one a band's colour at paper white,
// over the picture where it is brighter than paper white. Mode 1, the display's brightness (the output's brightest
// channel, scRGB: 1 = 80 nits): blue, green and yellow the thirds from paper white to 90% of the peak, orange to 98%, red
// at the peak (the tone mapping's top: flat there). Mode 2, the game's own values (the frame before tone mapping, 1 = SDR white,
// its brightest channel): blue 1-2x, green 2-4x, yellow 4-16x, orange 16-256x, red above.
float3 Stripes(float3 o, float3 lin, uint2 pixel, float paper, float peak, uint mode, float height)
{
    const float period = max(8.0, round(height / 60.0));
    if (frac(float(pixel.x + pixel.y) / period) < 0.5)
        return o;
    const float3 blue = float3(0.1, 0.35, 1.0), green = float3(0.1, 1.0, 0.2), yellow = float3(1.0, 0.95, 0.1),
                 orange = float3(1.0, 0.45, 0.0), red = float3(1.0, 0.0, 0.0);
    float3 band;
    if (mode == 1)
    {
        const float y = max(o.r, max(o.g, o.b));
        if (y <= paper * 1.02)
            return o;
        // Thirds of the range from paper white to 90% of the peak (fixed multiples would leave bands empty on a low
        // peak: at 203 / 460 nits no green), then the top.
        const float top = max(peak * 0.9, paper * 1.05), third = (top - paper) / 3.0;
        band = y >= peak * 0.98 ? red : y >= top ? orange : y >= paper + 2.0 * third ? yellow : y >= paper + third ? green : blue;
    }
    else
    {
        const float y = max(lin.r, max(lin.g, lin.b));
        if (y <= 1.0)
            return o;
        band = y > 256.0 ? red : y > 16.0 ? orange : y > 4.0 ? yellow : y > 2.0 ? green : blue;
    }
    return band * paper;
}

// A float frame through the display's gamma ramp (GAMMA_RAMP_HDR's arithmetic, from the ramp's 1026 x 1 image):
// the level's entry, and above 1 the ramp continued at its slope over the last 32 levels (the image's last two texels).
float3 RampFrame(float3 v, uint rampIndex)
{
    Texture2D<float4> ramp = g_Textures[rampIndex];
    const uint3 level = uint3(saturate(v) * 1023.0 + 0.5);
    const float3 ramped = float3(ramp.Load(int3(level.r, 0, 0)).r, ramp.Load(int3(level.g, 0, 0)).g, ramp.Load(int3(level.b, 0, 0)).b);
    // (The entries are half floats, as in GAMMA_RAMP_HDR's float copy of the frame: the same output to the bit.)
    const float3 slope = ramp.Load(int3(1024, 0, 0)).rgb + ramp.Load(int3(1025, 0, 0)).rgb;
    // (Unlike GAMMA_RAMP_HDR's float copy, the result is not rounded to a half float before the tone mapping:
    // highlights differ from that copy's by under 0.2%, in up to 3% of a frame's pixels.)
    return ramped + max(v - 1.0, 0.0) * slope;
}

float4 main(VsOut i) : SV_Target0
{
    // textureIndex: the frame, and in bits 16-31 the frame before its 2D layers (0: none).
    Texture2D<float4> source = g_Textures[g_Push.textureIndex & 0xFFFF];
    const uint preUiIndex = g_Push.textureIndex >> 16;
    SamplerState linearClamp = g_Samplers[0];
    const uint flags = g_Push.samplerIndex;
    float2 size;
    source.GetDimensions(size.x, size.y);
    float3 c = source.SampleLevel(linearClamp, i.uv, 0).rgb;
    // HDR (bit 2) from the frames as the game drew them, before the gamma ramp (offsetY bit 31: offsetY holds no frame
    // number then, as HDR has no dither): the gamma ramp here, from its image (offsetY's low 16 bits, 0: no ramp set).
    const uint rampIndex = (flags & 4) != 0 && (asuint(g_Push.offsetY) & 0x80000000u) != 0 ? asuint(g_Push.offsetY) & 0xFFFFu : 0u;
    if (rampIndex != 0)
        c = RampFrame(c, rampIndex);
    // How much of this pixel the UI drew: where the frame differs from the frame before the 2D layers.
    float ui = (flags & 128) ? 1.0 : 0.0;   // a frame all UI (menus, the briefing)
    if (preUiIndex != 0)
    {
        Texture2D<float4> preUi = g_Textures[preUiIndex];
        float3 before = preUi.SampleLevel(linearClamp, i.uv, 0).rgb;
        if (rampIndex != 0)
            before = RampFrame(before, rampIndex);
        const float3 d = abs(c - before);
        ui = saturate((max(d.r, max(d.g, d.b)) - 1.5 / 255.0) * 24.0);
    }
    const uint2 pixel = uint2(i.pos.xy);
    // Debanding (after libplacebo's): at two growing radii, the average of four samples around a random direction replaces
    // the pixel where it stays within the threshold of it (a gradient's steps), not across edges or detail.
    if (flags & 1)
    {
        const float radius = float((flags >> 8) & 255);
        const float threshold = float((flags >> 16) & 255) / (16.0 * 255.0);
        [unroll] for (uint step = 1; step <= 2; step++)
        {
            const float r = radius * float(step) * 0.5 * Rand(pixel, step * 2);
            const float a = 6.28318531 * Rand(pixel, step * 2 + 1);
            const float2 o = float2(cos(a), sin(a)) * r / size, p = float2(-o.y, o.x);
            const float3 average = 0.25 * (source.SampleLevel(linearClamp, i.uv + o, 0).rgb + source.SampleLevel(linearClamp, i.uv - o, 0).rgb +
                source.SampleLevel(linearClamp, i.uv + p, 0).rgb + source.SampleLevel(linearClamp, i.uv - p, 0).rgb);
            if (all(abs(average - c) < threshold / float(step)))
                c = average;
        }
    }
    if (flags & 4)
    {
        // scRGB: linear BT.709, 1.0 = 80 nits, the picture at paper white. The SDR frame's highlights (the top of its range:
        // the sun, glints, lit cloud tops) rise towards the display's peak through the tone mapping below (RenoDRT or
        // PsychoV-30).
        const float paper = asfloat(g_Push.offsetX);
        const float peak = float(flags >> 24) * 50.0 / 80.0;
        const uint tonemap = (flags >> 4) & 3;
        // HDR from the scene (grade1 bit 24): the frame keeps the game's values above 1 (its 2.2
        // continued). RenoDRT and PsychoV-30 take them as they are.
        const bool scene = (g_Push.grade1 & (1u << 24)) != 0;
        // Stray emitter codes: the game decodes any channel written at codes 246-255 as 10-100x white
        // (the sun's, the lights'). A texel near full red (the afterburner's hot ring), filtered or blended into those
        // codes, decodes as red 100+ beside green and blue below 1: single pixels at the peak, which SDR's clip hides.
        // A channel is kept to max(1.5, 2x the pixel's middle channel) (1.5: the lit surfaces' top). Bright whites (the
        // sun: 200 / 107 / 69) are untouched. Stripes word bit 8: off (AC6_HDR_EMITTER_CAP=0).
        if (scene && (g_Push.stripes & 256u) == 0)
        {
            const float mid = max(min(c.r, c.g), min(max(c.r, c.g), c.b));
            c = min(c, max(1.5, 2.0 * mid));
        }
        const float3 lin = scene ? pow(max(c, 0.0), 2.2) : pow(saturate(c), 2.2);
        const float3 linUi = min(lin, 1.0);
        // The UI at its own white (the stripes word's bits 16-31, in nits, 0 = paper white), mixed by how
        // much of the pixel it drew.
        const float uiWhite = (g_Push.stripes >> 16) != 0 ? float(g_Push.stripes >> 16) / 80.0 : paper;
        if (peak > paper)
        {
            float3 hdr;
            if (tonemap == 2)
                hdr = HdrRenoDrt(lin, paper * 80.0, peak * 80.0, Grade(), scene);
            else
                hdr = HdrPsychoV30(lin, paper * 80.0, peak * 80.0, Grade(), scene);
            float3 o = hdr / 80.0;
            if ((g_Push.stripes & 255) != 0 && ui < 0.5)
                o = Stripes(o, lin, pixel, paper, peak, g_Push.stripes & 255, size.y);
            return float4(lerp(o, linUi * uiWhite, ui), 1.0);
        }
        // No room above paper white (a peak at or under it): the SDR picture at paper white.
        float3 o = linUi * paper;
        if ((g_Push.stripes & 255) != 0 && ui < 0.5)
            o = Stripes(o, lin, pixel, paper, peak, g_Push.stripes & 255, size.y);
        return float4(lerp(o, linUi * uiWhite, ui), 1.0);
    }
    if (flags & 64)
        c = lerp(c, float3(1.0, 0.0, 0.0), ui * 0.75);   // AC6_HDR_UI_DEBUG: the UI in red
    // Dithering: triangular noise of one output step, so the quantization to the swap chain's bits leaves no steps.
    if (flags & 2)
    {
        const uint frame = uint(g_Push.offsetY);
        const float noise = Rand(pixel, frame * 2 + 7) - Rand(pixel, frame * 2 + 8);
        c += noise * ((flags & 8) ? 1.0 / 1023.0 : 1.0 / 255.0);
    }
    return float4(c, 1.0);
}
#elif defined(PILLAR_FROST)
// Frosted glass at the sides of a 16:9 picture on a wide screen: the picture continued past its
// box mirrored, blurred until nothing of it can be made out, and shaded. samplerIndex: the pass, offsetX / offsetY: the
// box's left and right edges (fractions of the width, float bits), stripes: the blur's sigma in the small image's
// texels (float bits).
//  0: the frame (textureIndex) reduced into the small image (16 taps over each texel's footprint), mirrored at the
//     box's edges,
//  1: the small image blurred across,
//  2: that blurred down, into a copy of the frame: outside the box the glass (x grade1's float, its shade, with
//     grade0 bit 16, an 8-bit image, dithered: a smooth dark gradient bands), inside it the frame itself (grade0's
//     low 16 bits: its texture).
float4 main(VsOut i) : SV_Target0
{
    Texture2D<float4> image = g_Textures[g_Push.textureIndex];
    const float left = asfloat(g_Push.offsetX), right = asfloat(g_Push.offsetY);
    float2 size;
    image.GetDimensions(size.x, size.y);
    if (g_Push.samplerIndex == 0)
    {
        const float2 footprint = float2(ddx(i.uv.x), ddy(i.uv.y));
        float3 sum = 0.0;
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
            {
                float2 uv = i.uv + ((float2(x, y) + 0.5) * 0.25 - 0.5) * footprint;
                uv.x = uv.x < left ? 2.0 * left - uv.x : uv.x > right ? 2.0 * right - uv.x : uv.x;
                uv.x = clamp(uv.x, left + 0.5 / size.x, right - 0.5 / size.x);
                sum += image.SampleLevel(g_Samplers[0], uv, 0.0).rgb;
            }
        return float4(sum / 16.0, 1.0);
    }
    const bool across = g_Push.samplerIndex == 1;
    if (!across && i.uv.x >= left && i.uv.x < right)
        return g_Textures[g_Push.grade0 & 0xFFFF].Load(int3(int2(i.pos.xy), 0));
    const float sigma = max(asfloat(g_Push.stripes), 0.25);
    const int radius = min(int(ceil(3.0 * sigma)), 40);
    const float2 step = across ? float2(1.0 / size.x, 0.0) : float2(0.0, 1.0 / size.y);
    float3 sum = 0.0;
    float weight = 0.0;
    for (int k = -radius; k <= radius; k++)
    {
        const float w = exp(-0.5 * float(k * k) / (sigma * sigma));
        sum += w * image.SampleLevel(g_Samplers[0], i.uv + float(k) * step, 0.0).rgb;
        weight += w;
    }
    float3 c = sum / weight;
    if (across)
        return float4(c, 1.0);
    c *= asfloat(g_Push.grade1);
    if (g_Push.grade0 & 0x10000)
    {
        // (Interleaved gradient noise, half a level either way.)
        const float n = frac(52.9829189 * frac(dot(i.pos.xy, float2(0.06711056, 0.00583715))));
        c += (n - 0.5) / 255.0;
    }
    return float4(c, 1.0);
}
#elif defined(GAMMA_RAMP)
// Scan-out through the display controller's gamma ramp: offsetX/Y = the device address
// of 1024 entries, one per 10-bit input level, each red << 20 | green << 10 | blue (10-bit outputs).
float3 RampLevel(uint64_t lut, uint3 level)
{
    const uint r = vk::RawBufferLoad<uint>(lut + level.r * 4) >> 20;
    const uint g = vk::RawBufferLoad<uint>(lut + level.g * 4) >> 10;
    const uint b = vk::RawBufferLoad<uint>(lut + level.b * 4);
    return float3(r & 1023, g & 1023, b & 1023) / 1023.0;
}
#if defined(GAMMA_RAMP_LUT)
// The ramp as a 1026 x 1 image, a level a texel and in the last two its slope past the top: the
// output pass reads it for the HDR frames it takes as the game drew them (RampFrame there is GAMMA_RAMP_HDR's
// arithmetic). The slope is worked out here from the 10-bit entries (from the image's half floats it would be 3% out)
// and stored as a half float plus the remainder in the next texel (a single half float puts highlights 0.1% out).
float4 main(VsOut i) : SV_Target0
{
    const uint64_t lut = (uint64_t(asuint(g_Push.offsetY)) << 32) | asuint(g_Push.offsetX);
    const uint level = uint(i.pos.x);
    if (level > 1023u)
    {
        const float3 slope = (RampLevel(lut, uint3(1023, 1023, 1023)) - RampLevel(lut, uint3(991, 991, 991))) * (1023.0 / 32.0);
        const float3 coarse = f16tof32(f32tof16(slope));
        return float4(level == 1024u ? coarse : slope - coarse, 1.0);
    }
    return float4(RampLevel(lut, uint3(level, level, level)), 1.0);
}
#else
float4 main(VsOut i) : SV_Target0
{
    const float3 source = g_Textures[g_Push.textureIndex].Sample(g_Samplers[g_Push.samplerIndex], i.uv).rgb;
    const float3 c = saturate(source);
    const uint64_t lut = (uint64_t(asuint(g_Push.offsetY)) << 32) | asuint(g_Push.offsetX);
    float3 ramped = RampLevel(lut, uint3(c * 1023.0 + 0.5));
#if defined(GAMMA_RAMP_HDR)
    // HDR from the scene: a float frame's values above 1 continue the ramp past its top, at its slope
    // over the last 32 levels.
    const float3 slope = (RampLevel(lut, uint3(1023, 1023, 1023)) - RampLevel(lut, uint3(991, 991, 991))) * (1023.0 / 32.0);
    ramped += max(source - 1.0, 0.0) * slope;
#endif
    return float4(ramped, 1.0);
}
#endif
#else
float4 main(VsOut i) : SV_Target0
{
    return g_Textures[g_Push.textureIndex].Sample(g_Samplers[g_Push.samplerIndex], i.uv);
}
#endif
#endif
