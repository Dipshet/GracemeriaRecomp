// Object motion blur (settings graphics.motion_blur, 0 to 1, off by default, only with temporal AA
// or FSR, whose motion vectors it reads). Included by internal_shaders.hlsl for its MOTION_BLUR variant.
//
// One pass on the scene after the temporal anti-aliasing: each pixel is averaged with the pixels along its own motion
// over the exposure. textureIndex = a copy of the scene, samplerIndex = every pixel's motion | the scene's depth << 16.
// The motion image comes from the motion completion pass: this frame's position to last frame's, in texture
// coordinates, each object's own where its draw wrote it and the camera's elsewhere. offsetX = the share of that
// frame-to-frame motion the exposure spans, as a float (the setting x the frame rate / 120: at 1 an exposure of
// 1 / 120 s whatever the frame rate, a half-open shutter at 60 fps), offsetY = the longest blur as a share of the
// picture's height, as a float, grade0 = the picture's width over its height, as a float, grade1 = the frame's number
// (the taps' offset changes each frame).
//
// What moves with the camera stays sharp (the player's aircraft in the chase view, the cockpit): its motion is next to
// none. A pixel takes nothing from a surface in front of it that does not move itself as far as the distance between
// them (the aircraft's outline is not smeared into the ground rushing past behind it). A fast object does not
// spread over a still background beyond its own outline (that would need the largest motion of each neighbourhood).
Texture2D<float4> g_Textures[] : register(t0, space0);
SamplerState g_Samplers[] : register(s0, space3);
struct PushConstants { uint textureIndex; uint samplerIndex; int offsetX; int offsetY; uint grade0; uint grade1; uint stripes; };
[[vk::push_constant]] ConstantBuffer<PushConstants> g_Push;
struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

float4 main(VsOut i) : SV_Target0
{
    Texture2D<float4> scene = g_Textures[g_Push.textureIndex];
    Texture2D<float4> motionImage = g_Textures[g_Push.samplerIndex & 0xFFFF];
    Texture2D<float4> depthImage = g_Textures[g_Push.samplerIndex >> 16];
    const float exposure = asfloat(g_Push.offsetX), longest = asfloat(g_Push.offsetY), aspect = asfloat(g_Push.grade0);
    float2 size;
    scene.GetDimensions(size.x, size.y);
    const float4 c = scene.SampleLevel(g_Samplers[1023], i.uv, 0);
    // (Lengths in shares of the picture's height: the same blur on any aspect.)
    const float2 shape = float2(aspect, 1.0);
    float2 v = motionImage.SampleLevel(g_Samplers[1023], i.uv, 0).xy * exposure;
    float length0 = length(v * shape);
    if (length0 * size.y < 1.0)
        return c;   // under a pixel: as it is
    if (length0 > longest)
    {
        v *= longest / length0;
        length0 = longest;
    }
    const float depth0 = depthImage.SampleLevel(g_Samplers[1023], i.uv, 0).x;
    // (Interleaved gradient noise, Jimenez 2014: the taps' places differ from pixel to pixel and frame to frame.)
    const float2 at = i.pos.xy + 5.588238 * float(g_Push.grade1 % 64u);
    const float dither = frac(52.9829189 * frac(dot(at, float2(0.06711056, 0.00583715))));
    const int taps = 10;
    float3 sum = c.rgb;
    float weight = 1.0;
    [loop] for (int k = 0; k < taps; k++)
    {
        const float t = (float(k) + dither) / float(taps) - 0.5;   // along the exposure, the pixel in the middle
        const float2 uv = i.uv + v * t;
        if (any(uv < 0.0) || any(uv > 1.0))
            continue;
        const float distance = abs(t) * length0;
        const float depth = depthImage.SampleLevel(g_Samplers[1023], uv, 0).x;
        float w = 1.0;
        // (The game's depth: larger is nearer, 0 the sky.) A surface in front: only as far as it moves itself.
        if (depth > depth0 * 1.03 + 1.0e-7)
        {
            const float own = length(motionImage.SampleLevel(g_Samplers[1023], uv, 0).xy * exposure * shape);
            w = saturate((own - distance) * size.y + 1.0);
        }
        sum += scene.SampleLevel(g_Samplers[0], uv, 0).rgb * w;
        weight += w;
    }
    return float4(sum / weight, c.a);
}
