// The enhanced bloom (settings graphics.bloom = "enhanced", AC6_BLOOM, "console" by default).
// Included by internal_shaders.hlsl for its BLOOM_* variants.
//
// The game's bloom (the DirectX SDK's HDRLighting chain): the scene decoded x the exposure, 4 x 4
// downsampled, a bright pass, blurred and halved to a 160 x 90 image (guest pixels), a Gaussian of 3 of its texels
// across, a star from it. The composite adds bloom x c100.z and star x c100.w to the scene, each image stretched 8
// times bilinearly: soft, with the corners of its 160 x 90 grid round small bright things at high resolutions.
// Enhanced: the bloom image made again from the scene at half the frame's size, with the game's own bright pass
// (its threshold and offset, the same exposure), and spread by a pyramid of seven levels whose widest are the game's
// width: the same light, a sharper core near small bright things, a smooth falloff. The star image read with a cubic
// B-spline. The composite draws with c100.zw at 0 (renderer.cpp RunDraw) and this adds both after it, which on the
// 8-bit canvas clips exactly as the composite's own sum would.
//  BLOOM_BRIGHT  the scene (encoded 8-bit) -> half size: four texels decoded, averaged, x the exposure, the bright pass
//  BLOOM_DOWN    a level -> half its size (a 13-tap filter: Jimenez, "Next generation post processing in Call of
//                Duty: Advanced Warfare", 2014)
//  BLOOM_UP      a level x its weight + the next coarser result, read with a 3 x 3 tent
//  BLOOM_APPLY   into the canvas (added by the blend): the result x the bloom's scale + the star x the star's scale
Texture2D<float4> g_Textures[] : register(t0, space0);
SamplerState g_Samplers[] : register(s0, space3);
struct PushConstants { uint textureIndex; uint samplerIndex; int offsetX; int offsetY; uint grade0; uint grade1; uint stripes; };
[[vk::push_constant]] ConstantBuffer<PushConstants> g_Push;
struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

#if defined(BLOOM_BRIGHT)
// The game's decode of its 8-bit scene: below 241 / 255 in every channel x 255 / 240, else per
// channel n = round(255 x) - 240: min(round(255 x), 240) / 240 + min(n, 5) x 0.1, + (n - 5) x 10 - 1.5 above 5.
float3 Decode(float3 c)
{
    if (max(c.r, max(c.g, c.b)) < 241.0 / 255.0)
        return c * (255.0 / 240.0);
    const float3 code = round(c * 255.0), n = code - 240.0;
    return min(code, 240.0) / 240.0 + min(max(n, 0.0), 5.0) * 0.1 + max(n - 5.0, 0.0) * 10.0 - select(n > 5.0, 1.5, 0.0);
}

// textureIndex = the scene, samplerIndex = the adapted luminance (1 x 1), offsetX / offsetY = the exposure band's low
// and high (the composite's vertex constant c106.xy), as floats, grade0 / grade1 = the bright pass's threshold and
// offset (its pixel constant c104.yz), as floats.
float4 main(VsOut i) : SV_Target0
{
    Texture2D<float4> scene = g_Textures[g_Push.textureIndex];
    uint w, h;
    scene.GetDimensions(w, h);
    const int2 top = int2(w, h) - 1, at = int2(i.uv * float2(w, h) - 0.5);
    float3 sum = 0.0;
    [unroll] for (int k = 0; k < 4; k++)
        sum += Decode(scene.Load(int3(clamp(at + int2(k & 1, k >> 1), int2(0, 0), top), 0)).rgb);
    float3 x = sum * 0.25;
    // The exposure as the composite's vertex shader makes it: L under the band's low: low / L, over its high: high / L.
    const float L = max(g_Textures[g_Push.samplerIndex].SampleLevel(g_Samplers[1023], float2(0.5, 0.5), 0).x, 1.0e-6);
    const float lo = asfloat(g_Push.offsetX), hi = asfloat(g_Push.offsetY);
    x *= L < lo ? lo / L : L > hi ? hi / L : 1.0;
    // The bright pass (psHDRBrightPassFilter): x (1 - threshold / luminance), then x / (x + offset).
    const float threshold = asfloat(g_Push.grade0), offset = asfloat(g_Push.grade1);
    const float luminance = max(dot(x, float3(0.2125, 0.7154, 0.0721)), 1.0e-6);
    x *= max(1.0 - threshold / luminance, 0.0);
    return float4(x / (x + offset), 1.0);
}
#elif defined(BLOOM_DOWN)
float4 main(VsOut i) : SV_Target0
{
    Texture2D<float4> s = g_Textures[g_Push.textureIndex];
    float2 size;
    s.GetDimensions(size.x, size.y);
    const float2 t = 1.0 / size, uv = i.uv;
    #define TAP(dx, dy) s.SampleLevel(g_Samplers[0], uv + float2(dx, dy) * t, 0).rgb
    const float3 a = TAP(-2, -2), b = TAP(0, -2), c = TAP(2, -2);
    const float3 d = TAP(-2, 0), e = TAP(0, 0), f = TAP(2, 0);
    const float3 g = TAP(-2, 2), h = TAP(0, 2), k = TAP(2, 2);
    const float3 j = TAP(-1, -1), l = TAP(1, -1), m = TAP(-1, 1), n = TAP(1, 1);
    #undef TAP
    const float3 r = e * 0.125 + (a + c + g + k) * 0.03125 + (b + d + f + h) * 0.0625 + (j + l + m + n) * 0.125;
    return float4(r, 1.0);
}
#elif defined(BLOOM_UP)
// textureIndex = this level, samplerIndex = the coarser result (0: none), offsetX = this level's weight, as a float.
float4 main(VsOut i) : SV_Target0
{
    float3 r = g_Textures[g_Push.textureIndex].SampleLevel(g_Samplers[0], i.uv, 0).rgb * asfloat(g_Push.offsetX);
    if (g_Push.samplerIndex != 0)
    {
        Texture2D<float4> s = g_Textures[g_Push.samplerIndex];
        float2 size;
        s.GetDimensions(size.x, size.y);
        const float2 t = 1.0 / size;
        #define TAP(dx, dy) s.SampleLevel(g_Samplers[0], i.uv + float2(dx, dy) * t, 0).rgb
        r += (TAP(-1, -1) + TAP(1, -1) + TAP(-1, 1) + TAP(1, 1)) * (1.0 / 16.0) +
             (TAP(0, -1) + TAP(-1, 0) + TAP(1, 0) + TAP(0, 1)) * (2.0 / 16.0) + TAP(0, 0) * (4.0 / 16.0);
        #undef TAP
    }
    return float4(r, 1.0);
}
#elif defined(BLOOM_APPLY)
// A cubic B-spline read (GPU Gems 2, chapter 20): four bilinear reads.
float3 SampleSmooth(Texture2D<float4> t, float2 uv)
{
    float2 size;
    t.GetDimensions(size.x, size.y);
    const float2 p = uv * size - 0.5, f = frac(p), base = floor(p);
    const float2 f2 = f * f, f3 = f2 * f;
    const float2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0, w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    const float2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0, w3 = f3 / 6.0;
    const float2 g0 = w0 + w1, g1 = w2 + w3;
    const float2 h0 = (w1 / g0) - 1.0 + base, h1 = (w3 / g1) + 1.0 + base;   // texels base-1 / base and base+1 / base+2
    const float2 inv = 1.0 / size;
    return (g0.y * (g0.x * t.SampleLevel(g_Samplers[0], (float2(h0.x, h0.y) + 0.5) * inv, 0) +
                    g1.x * t.SampleLevel(g_Samplers[0], (float2(h1.x, h0.y) + 0.5) * inv, 0)) +
            g1.y * (g0.x * t.SampleLevel(g_Samplers[0], (float2(h0.x, h1.y) + 0.5) * inv, 0) +
                    g1.x * t.SampleLevel(g_Samplers[0], (float2(h1.x, h1.y) + 0.5) * inv, 0))).rgb;
}

// textureIndex = the bloom (the pyramid's result), samplerIndex = the game's star image, offsetX / offsetY = the
// bloom's and the star's scales, as floats, grade0 = the debug view (1: the game's own bloom alone, 2: this bloom
// alone, each x its scale, 3: the star alone) as a float, grade1 = the game's bloom image.
float4 main(VsOut i) : SV_Target0
{
    const float bloomScale = asfloat(g_Push.offsetX), starScale = asfloat(g_Push.offsetY), debug = asfloat(g_Push.grade0);
    const float3 bloom = g_Textures[g_Push.textureIndex].SampleLevel(g_Samplers[0], i.uv, 0).rgb * bloomScale;
    const float3 star = SampleSmooth(g_Textures[g_Push.samplerIndex], i.uv) * starScale;
    if (debug == 1.0)
        return float4(g_Textures[g_Push.grade1].SampleLevel(g_Samplers[0], i.uv, 0).rgb * bloomScale, 1.0);
    if (debug == 2.0)
        return float4(bloom, 1.0);
    if (debug == 3.0)
        return float4(star, 1.0);
    return float4(max(bloom + star, 0.0), 0.0);
}
#endif
