// Light from the game's emitters on what is near them (AC6_EMITTER_LIGHT=<strength>,
// an experiment, off by default): explosions, fires, flares and afterburners are drawn by the game as glowing sprites
// that light nothing. Included by internal_shaders.hlsl for its LIGHT_* variants.
//
// The emitters' draws (additive, plain colour: the same ones the HDR glow takes) are issued again by the renderer
// into a light image with a depth of their own (renderer.cpp RunEmitterLight). From it:
//  LIGHT_DOWN0   the light image and its depth at half size: the mean colour, and in alpha the mean of brightness x view
//                distance (so that brightness-weighted distance = alpha / brightness survives the averaging)
//  LIGHT_DOWN0B  the same blocks' brightness-weighted screen position: (brightness x u, brightness x v, brightness)
//  LIGHT_DOWN    a level at half the size of the one before (the same means, for either kind)
//  LIGHT_APPLY   into the scene: each level's light, as much of it as the pixel is near that light in depth and faces it
//
// The sum over the levels stands for the light of every emitter pixel falling off with the square of the distance: a
// level's texel is the mean over a block 2^(n+1) pixels wide, which at the light's view distance z is r = 2^(n+1) x the
// size of a pixel there. A surface at the same distance takes the level's light whole, one further off in depth by d
// takes 1 / (1 + (d / r)^2) of it. Ten levels: the coarsest's texel is 1024 pixels, so a light reaches across the picture.
//
// Two refinements:
//  - Bell-shaped: the levels are read with a cubic B-spline (four bilinear reads each), not bilinearly: a bilinear read of
//    a coarse level is a pyramid with corners, and the sum of ten of them a cone with a flat top where the ceiling
//    clips it. The ceiling is on the light's brightness alone, the colour kept (a ceiling per channel turns the light
//    white).
//  - Direction: each level also keeps where its light is (the brightness-weighted screen position, with the distance
//    above). The surface's normal is made from the scene's depth, and a surface takes the light as it faces it (wrapped:
//    a quarter at grazing, none facing away): a nacelle's side facing the camera with the flame behind it does not take
//    the flame's full light as if the flame were in front of it.
// Only light that is on screen: an emitter outside the picture lights nothing.
Texture2D<float4> g_Textures[] : register(t0, space0);
SamplerState g_Samplers[] : register(s0, space3);
struct PushConstants { uint textureIndex; uint samplerIndex; int offsetX; int offsetY; uint grade0; uint grade1; uint stripes; };
[[vk::push_constant]] ConstantBuffer<PushConstants> g_Push;
struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

float Luma(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

#if defined(LIGHT_DOWN0) || defined(LIGHT_DOWN0B)
// textureIndex = the light image, samplerIndex = its depth, offsetX = clip z (depth x view distance), as a float.
float4 main(VsOut i) : SV_Target0
{
    Texture2D<float4> light = g_Textures[g_Push.textureIndex];
    Texture2D<float4> depthImage = g_Textures[g_Push.samplerIndex];
    uint w, h;
    light.GetDimensions(w, h);
    const int2 top = int2(w, h) - 1, at = int2(i.pos.xy) * 2;
    const float clipZ = asfloat(g_Push.offsetX);
    float4 sum = 0.0;
    [unroll] for (int k = 0; k < 4; k++)
    {
        const int2 p = min(at + int2(k & 1, k >> 1), top);
        const float3 c = max(light.Load(int3(p, 0)).rgb, 0.0);
#if defined(LIGHT_DOWN0)
        const float d = depthImage.Load(int3(p, 0)).x;
        sum += float4(c, d > 0.0 ? Luma(c) * clipZ / d : 0.0);
#else
        const float b = Luma(c);
        sum += float4((float2(p) + 0.5) / float2(w, h) * b, b, 0.0);
#endif
    }
    return sum * 0.25;
}
#elif defined(LIGHT_DOWN)
float4 main(VsOut i) : SV_Target0
{
    Texture2D<float4> source = g_Textures[g_Push.textureIndex];
    uint w, h;
    source.GetDimensions(w, h);
    const int2 top = int2(w, h) - 1, at = int2(i.pos.xy) * 2;
    return (source.Load(int3(min(at, top), 0)) + source.Load(int3(min(at + int2(1, 0), top), 0)) +
        source.Load(int3(min(at + int2(0, 1), top), 0)) + source.Load(int3(min(at + int2(1, 1), top), 0))) * 0.25;
}
#elif defined(LIGHT_APPLY)
// textureIndex = a copy of the scene's colour, samplerIndex = the scene's depth, offsetX / offsetY = the device address
// of the parameters (28 floats): the ten levels' colour textures, the ten levels' position textures. Then clip z, the
// size of a light-image pixel at view distance 1, the emitters' intensity and the largest scene value that is a colour
// (the 8-bit scene's codes above 240 / 255 are its emitters'). Then the debug view (1: the light alone), tan of the half
// field of view across and up, and a spare.
static const uint kLevels = 10;

// A cubic B-spline read of a texture, from four bilinear reads (GPU Gems 2, chapter 20): smooth, no corners.
float4 SampleSmooth(Texture2D<float4> t, float2 uv)
{
    float2 size;
    t.GetDimensions(size.x, size.y);
    const float2 p = uv * size - 0.5, f = frac(p), base = floor(p);
    const float2 f2 = f * f, f3 = f2 * f;
    const float2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0, w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    const float2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0, w3 = f3 / 6.0;
    const float2 g0 = w0 + w1, g1 = w2 + w3;
    const float2 h0 = (w1 / g0) - 1.0 + base, h1 = (w3 / g1) + 1.0 + base;   // texels base-1 / base and base+1 / base+2 (centres at +0.5)
    const float2 inv = 1.0 / size;
    return g0.y * (g0.x * t.SampleLevel(g_Samplers[0], (float2(h0.x, h0.y) + 0.5) * inv, 0) +
                   g1.x * t.SampleLevel(g_Samplers[0], (float2(h1.x, h0.y) + 0.5) * inv, 0)) +
           g1.y * (g0.x * t.SampleLevel(g_Samplers[0], (float2(h0.x, h1.y) + 0.5) * inv, 0) +
                   g1.x * t.SampleLevel(g_Samplers[0], (float2(h1.x, h1.y) + 0.5) * inv, 0));
}

// A screen position and view distance as a view-space point (x right, y up, z forward).
float3 ViewPoint(float2 uv, float z, float2 tanHalf)
{
    return float3((uv.x * 2.0 - 1.0) * tanHalf.x, (1.0 - uv.y * 2.0) * tanHalf.y, 1.0) * z;
}

float4 main(VsOut i) : SV_Target0
{
    const uint64_t a = (uint64_t(asuint(g_Push.offsetY)) << 32) | asuint(g_Push.offsetX);
    float4 r[7];
    [unroll] for (uint k = 0; k < 7; k++)
        r[k] = vk::RawBufferLoad<float4>(a + k * 16);
    const uint colours[10] = { asuint(r[0].x), asuint(r[0].y), asuint(r[0].z), asuint(r[0].w), asuint(r[1].x), asuint(r[1].y),
        asuint(r[1].z), asuint(r[1].w), asuint(r[2].x), asuint(r[2].y) };
    const uint places[10] = { asuint(r[2].z), asuint(r[2].w), asuint(r[3].x), asuint(r[3].y), asuint(r[3].z), asuint(r[3].w),
        asuint(r[4].x), asuint(r[4].y), asuint(r[4].z), asuint(r[4].w) };
    const float clipZ = r[5].x, pixelSize = r[5].y, strength = r[5].z, colourCap = r[5].w;
    const float debug = r[6].x;
    const float2 tanHalf = r[6].yz;
    Texture2D<float4> scene = g_Textures[g_Push.textureIndex];
    Texture2D<float4> depthImage = g_Textures[g_Push.samplerIndex];
    const float4 c = scene.Load(int3(int2(i.pos.xy), 0));
    int2 depthSize;
    depthImage.GetDimensions(depthSize.x, depthSize.y);
    const int2 dp = min(int2(i.uv * float2(depthSize)), depthSize - 1);
    const float d = depthImage.Load(int3(dp, 0)).x;
    if (d <= 0.0)
        return debug != 0.0 ? float4(0.0, 0.0, 0.0, c.a) : c;   // the sky
    const float z = clipZ / d;
    const float3 P = ViewPoint(i.uv, z, tanHalf);
    // The surface's normal from the depth: of the two neighbours on each axis, the nearer in depth (no normal across an
    // object's edge), toward the camera.
    const float2 texel = 1.0 / float2(depthSize);
    float3 sideX, sideY;
    {
        const float dl = depthImage.Load(int3(max(dp - int2(1, 0), 0), 0)).x, dr = depthImage.Load(int3(min(dp + int2(1, 0), depthSize - 1), 0)).x;
        const float du = depthImage.Load(int3(max(dp - int2(0, 1), 0), 0)).x, dd = depthImage.Load(int3(min(dp + int2(0, 1), depthSize - 1), 0)).x;
        const float3 L = ViewPoint(i.uv - float2(texel.x, 0.0), clipZ / max(dl, 1.0e-9), tanHalf);
        const float3 R = ViewPoint(i.uv + float2(texel.x, 0.0), clipZ / max(dr, 1.0e-9), tanHalf);
        const float3 U = ViewPoint(i.uv - float2(0.0, texel.y), clipZ / max(du, 1.0e-9), tanHalf);
        const float3 D = ViewPoint(i.uv + float2(0.0, texel.y), clipZ / max(dd, 1.0e-9), tanHalf);
        sideX = abs(R.z - P.z) < abs(P.z - L.z) ? R - P : P - L;
        sideY = abs(D.z - P.z) < abs(P.z - U.z) ? D - P : P - U;
    }
    float3 N = normalize(cross(sideX, sideY));
    if (dot(N, -P) < 0.0)
        N = -N;
    float3 light = 0.0;
    [unroll] for (uint n = 0; n < kLevels; n++)
    {
        // (A bilinear read first: a level whose light here is under a step of the 8-bit picture is passed over, and the
        // cubic read, four times the cost, is only made where it adds something. The two finest levels stay bilinear:
        // their texels are 2 and 4 pixels, too small for corners to show.)
        float4 l = g_Textures[colours[n]].SampleLevel(g_Samplers[0], i.uv, 0);
        if (Luma(l.rgb) * strength < 0.5 / 255.0)
            continue;
        if (n >= 2)
            l = SampleSmooth(g_Textures[colours[n]], i.uv);
        const float brightness = Luma(l.rgb);
        if (brightness <= 1.0e-5)
            continue;
        const float lightZ = l.a / brightness;
        const float reach = float(2u << n) * pixelSize * lightZ;
        const float gap = (z - lightZ) / max(reach, 1.0e-3);
        // Where the level's light is, and how the surface faces it (wrapped: a quarter at grazing, none facing away). The
        // place read bilinearly (a direction: no corners to see, and a cubic read costs four bilinear ones).
        const float3 place = g_Textures[places[n]].SampleLevel(g_Samplers[0], i.uv, 0).xyz;
        const float2 lightUv = place.z > 1.0e-6 ? place.xy / place.z : i.uv;
        const float3 toLight = ViewPoint(lightUv, lightZ, tanHalf) - P;
        const float span = length(toLight);
        const float facing = span > reach * 0.05 ? saturate(0.25 + 0.75 * dot(N, toLight / span)) : 1.0;
        light += l.rgb * facing / (1.0 + gap * gap);
    }
    // Eased toward a ceiling on its brightness alone (next to a flame every level holds it), its colour kept.
    const float ceiling = 0.8;
    const float raw = Luma(light) * strength;
    if (raw > 1.0e-6)
        light *= ceiling * (1.0 - exp(-raw / ceiling)) / max(Luma(light), 1.0e-6);
    if (debug != 0.0)
        return float4(light * min(colourCap, 1.0), c.a);
    if (max(c.r, max(c.g, c.b)) > colourCap)
        return c;
    // What the surface gives back of it: its own colour's hue, at a brightness that does not go to nothing where the
    // scene is dark (a night scene's surfaces are dark by their light, not by what they are).
    const float own = max(Luma(c.rgb), 1.0e-4);
    const float3 surface = c.rgb / own * lerp(0.5, max(own, 0.5), saturate(own * 2.0));
    return float4(min(c.rgb + light * surface, colourCap), c.a);
}
#endif
