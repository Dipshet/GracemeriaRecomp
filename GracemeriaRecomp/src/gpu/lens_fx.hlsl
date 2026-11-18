// Optional lens effects (settings graphics.film_grain, vignette, chromatic_aberration: all off by
// default): one pass over the finished 3D picture, after the game's own post effects and before its HUD and menus are
// drawn. Included by internal_shaders.hlsl for its LENS_FX variant.
//
// textureIndex = a copy of the picture, samplerIndex = the frame's number (the grain moves each frame), offsetX,
// offsetY, grade0 = the three strengths as floats (grain, vignette, chromatic aberration, 0 to 1), stripes = the
// picture's width over its height as a float.
//
// All three are sized by the picture, not by its pixels: the same look at any resolution and aspect ratio.
//  - Chromatic aberration: red sampled a little further from the centre and blue a little nearer, growing with the
//    square of the distance from the centre: nothing in the middle, 0.8% of the half diagonal at the corners at 1.
//  - Vignette: darker toward the corners, round on any aspect (the distance is measured against the half diagonal):
//    nothing inside 45% of it, a quarter of the light gone in the corners at 1.
//  - Film grain, perceptual: a step in the picture's lightness, the same size across the tones as the eye takes them
//    (the picture's values are display-encoded: equal steps look about equal), strongest in the midtones, 40% of that
//    toward white, fading out through the shadows (from 30% lightness down to nothing at black, so a night sky doesn't
//    take a full step as a quarter of its own value), in lightness only, the colour's hue and saturation kept (the
//    three channels scaled alike). A grain cell is 1 / 720 of the picture's height (2 pixels at 1440p), new each frame.
Texture2D<float4> g_Textures[] : register(t0, space0);
SamplerState g_Samplers[] : register(s0, space3);
struct PushConstants { uint textureIndex; uint samplerIndex; int offsetX; int offsetY; uint grade0; uint grade1; uint stripes; };
[[vk::push_constant]] ConstantBuffer<PushConstants> g_Push;
struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

// (A hash of three integers to 0..1: Dave Hoskins' hash13 form.)
float Hash(float3 p)
{
    p = frac(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return frac((p.x + p.y) * p.z);
}

float4 main(VsOut i) : SV_Target0
{
    Texture2D<float4> picture = g_Textures[g_Push.textureIndex];
    const float grain = asfloat(g_Push.offsetX), vignette = asfloat(g_Push.offsetY), aberration = asfloat(g_Push.grade0);
    const float aspect = asfloat(g_Push.stripes);
    // From the centre, in units of the half diagonal (the corners at 1), the picture's shape taken out.
    const float2 fromCentre = (i.uv - 0.5) * float2(aspect, 1.0) / (0.5 * sqrt(aspect * aspect + 1.0));
    const float r2 = dot(fromCentre, fromCentre);
    float4 c = picture.SampleLevel(g_Samplers[1023], i.uv, 0);
    if (aberration > 0.0)
    {
        const float2 shift = (i.uv - 0.5) * (r2 * 0.008 * aberration);
        c.r = picture.SampleLevel(g_Samplers[0], i.uv + shift, 0).r;
        c.b = picture.SampleLevel(g_Samplers[0], i.uv - shift, 0).b;
    }
    if (vignette > 0.0)
        c.rgb *= 1.0 - 0.25 * vignette * smoothstep(0.45, 1.0, sqrt(r2));
    if (grain > 0.0)
    {
        const float2 cell = floor(i.uv * float2(aspect, 1.0) * 720.0);
        const float noise = Hash(float3(cell, float(g_Push.samplerIndex % 4096u))) + Hash(float3(cell.yx + 17.0, float(g_Push.samplerIndex % 4093u))) - 1.0;
        const float luma = dot(c.rgb, float3(0.2126, 0.7152, 0.0722));
        const float tone = saturate(luma);
        const float response = smoothstep(0.0, 0.3, tone) * lerp(0.4, 1.0, 4.0 * tone * (1.0 - tone));
        // (The step, up to 3.5% of the range at 1, as a factor on the colour: lightness + step over lightness.)
        c.rgb *= max(1.0 + noise * 0.035 * grain * response / max(luma, 0.03), 0.0);
    }
    return c;
}
