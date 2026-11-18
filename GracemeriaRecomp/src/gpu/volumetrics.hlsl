// Volumetric effects (AC6_VOLUMETRICS): the effects buffer's billboards and ribbons become
// lit, self-shadowed or glowing volumes. Included by internal_shaders.hlsl for its VOL_* variants.
//  - VOL_HARVEST, a geometry shader after the game's own vertex shader on the effect draws (identified by their pixel
//    shaders: renderer.cpp VolumetricLayout): per billboard (two triangles) one record of the particle's centre and a
//    corner (NDC), the game's colour (the sprite's mean through the draw's own colour terms), opacity and fog, at the
//    slot the renderer reserved for the draw (its order kept). The billboard itself is dropped (flags bit 0 keeps it:
//    AC6_VOLUMETRICS=debug). Trail ribbons give a segment (the two cross-sections' middles) and a half width instead.
//  - VOL_VS / VOL_PS, after the effects composite: one camera-facing quad per record around the particle's sphere (or
//    the segment's), and a ray march through it (3D noise eroding a soft ball or tube, extinction scaled so the mean
//    opacity is the billboard's coverage, or for the additive kinds emission scaled to the billboard's mean glow, a few
//    steps towards the sun for self-shadowing), clipped by the scene's depth, blended premultiplied.
//
// A record, 80 bytes: +0 the centre (a trail: one end) in NDC and the opacity at the centre, +16 a corner (a trail: a
// corner at that end) and the fog factor, +32 the colour (the additive kinds: the emission, premultiplied) and the mean
// coverage, +48 the frame's tag, +52 the soft-particle fade's range and threshold, +60 the kind (bits 0-3) and the fog
// colour (RGB8 in bits 8-31), +64 a trail's other end (NDC), or for a cloud puff: its mean alpha, view depth and
// distance fade.

// The kinds (renderer.cpp VolumetricLayout).
#define VOL_SMOKE 0          // TranslucentEdge: TankSmoke, SimpleExplosionSmoke, ImpactSand
#define VOL_SMOKE_DETAIL 1   // TranslucentEdgeDetail: RisingSmokeNeo, ImpactSmoke
#define VOL_FLAME 2          // SimpleExplosionFlame (additive)
#define VOL_ADD_DETAIL 3     // AddDetail: RisingSmokeNeo's fire (additive)
#define VOL_CLOUD 4          // Cloud_p: the cloud puffs
#define VOL_TRAIL 5          // MissileSmokeBelt: the missile trails (ribbons)
#define VOL_TRAIL_ADD 6      // the trails' additive variant
#define VOL_RECORD 80

bool VolEmissive(uint kind) { return kind == VOL_FLAME || kind == VOL_ADD_DETAIL || kind == VOL_TRAIL_ADD; }
bool VolTube(uint kind) { return kind == VOL_TRAIL || kind == VOL_TRAIL_ADD; }

#ifdef VOL_HARVEST
Texture2D<float4> g_Textures[] : register(t0, space0);
SamplerState g_Samplers[] : register(s0, space3);
// The draw's shared constants (the game shaders' uniform block 2 of set 4): fetch slots' textures (+0, one uint each)
// and samplers (+384).
cbuffer SharedBlock : register(b2, space4) { uint4 g_Shared[32]; };
// The draw's pixel shader constants (c50-c52 the flames' colours, c80 / c85 the clouds' fog colour and altitude band,
// c93 the trails' fog colour, c96 the detail ratio, c97 the smoke's fog colour, c103 DepthDelta_None).
cbuffer PixelBlock : register(b1, space4) { float4 g_Ps[224]; };
struct GsPush { uint addressLo; uint addressHi; uint base; uint flags; uint end; };   // end: past the draw's slots
// flags: bit 0 keep, 1-4 the kind, 8-31 frame tag
[[vk::push_constant]] ConstantBuffer<GsPush> g_Gs;

struct GsVert
{
    float4 pos : SV_Position;
    [[vk::location(0)]] float4 t0 : TEXCOORD0;
    [[vk::location(1)]] float4 t1 : TEXCOORD1;
    [[vk::location(2)]] float4 t2 : TEXCOORD2;
    [[vk::location(3)]] float4 t3 : TEXCOORD3;
    [[vk::location(4)]] float4 t4 : TEXCOORD4;
    [[vk::location(16)]] float4 t16 : TEXCOORD16;
};

float3 Ndc(float4 p)
{
    return float3(p.x, -p.y, p.z) / p.w;   // the vertex shaders' position has y negated (-fvk-invert-y)
}

// The sprite's mean over a cell (five taps towards the corners: centre uv, half diagonals da, dc): premultiplied
// colour and coverage, `centre` the tap at the cell's centre.
float4 SpriteMean(uint texture, uint sampler, float2 uv, float2 da, float2 dc, out float4 centre)
{
    centre = g_Textures[texture].SampleLevel(g_Samplers[sampler], uv, 1.0);
    float4 sum = float4(centre.rgb * centre.a, centre.a);
    const float2 offsets[4] = { da, -da, dc, -dc };
    [unroll] for (uint k = 0; k < 4; k++)
    {
        const float4 s = g_Textures[texture].SampleLevel(g_Samplers[sampler], uv + offsets[k], 1.0);
        sum += float4(s.rgb * s.a, s.a);
    }
    return sum * 0.2;
}

uint PackColour(float3 c)
{
    const uint3 b = uint3(saturate(c) * 255.0 + 0.5);
    return (b.r << 8) | (b.g << 16) | (b.b << 24);
}

[maxvertexcount(3)]
void main(triangle GsVert v[3], uint prim : SV_PrimitiveID, inout TriangleStream<GsVert> stream)
{
    if (g_Gs.flags & 1)
    {
        stream.Append(v[0]);
        stream.Append(v[1]);
        stream.Append(v[2]);
        stream.RestartStrip();
    }
    // Only into the draw's own slots: writes past them would land in other draws' (GPU order: a different winner each
    // frame) or past the buffer.
    if ((prim & 1) || g_Gs.base + prim / 2 >= g_Gs.end)
        return;
    const uint kind = (g_Gs.flags >> 1) & 15;
    const bool tube = VolTube(kind);
    // The sprite's texture coordinates: TEXCOORD1.xy, the trails' TEXCOORD0.xy.
    float2 uvs[3] = { v[0].t1.xy, v[1].t1.xy, v[2].t1.xy };
    if (tube)
    {
        uvs[0] = v[0].t0.xy;
        uvs[1] = v[1].t0.xy;
        uvs[2] = v[2].t0.xy;
    }
    // The diagonal: the edge whose ends differ in both texture coordinates. The third corner is a right angle.
    uint a = 0, b = 1, c = 2;
    float best = -1.0;
    [unroll] for (uint e = 0; e < 3; e++)
    {
        const uint i = e, j = (e + 1) % 3;
        const float2 d = abs(uvs[i] - uvs[j]);
        if (d.x * d.y > best)
        {
            best = d.x * d.y;
            a = i;
            b = j;
            c = (e + 2) % 3;
        }
    }
    float3 centre = Ndc((v[a].pos + v[b].pos) * 0.5);
    float3 corner = Ndc(v[c].pos);
    float3 other = 0.0;
    if (tube)
    {
        // A trail's quad spans two cross-sections (the template's v runs along the trail: MissileSmokeBelt's vertex
        // shader): the edge from the right angle whose ends share v is one. The quad's fourth corner closes the other.
        const uint across = abs(uvs[c].y - uvs[a].y) <= abs(uvs[c].y - uvs[b].y) ? a : b;
        const uint along = across == a ? b : a;
        const float4 fourth = v[a].pos + v[b].pos - v[c].pos;   // linear in clip space
        centre = Ndc((v[c].pos + v[across].pos) * 0.5);
        other = Ndc((v[along].pos + fourth) * 0.5);
        corner = Ndc(v[c].pos);
    }
    const float2 uv = (uvs[a] + uvs[b]) * 0.5;
    const float2 da = (uvs[a] - uv) * 0.5, dc = (uvs[c] - uv) * 0.5;
    // Slot 0's texture and sampler, the clouds' slot 6.
    const uint texture = kind == VOL_CLOUD ? g_Shared[1].z : g_Shared[0].x;
    const uint sampler = kind == VOL_CLOUD ? g_Shared[25].z : g_Shared[24].x;
    float4 s0;
    const float4 mean = SpriteMean(texture, sampler, uv, da, dc, s0);
    float3 colour = 0.0;          // straight (alpha kinds) or the emission (additive kinds, premultiplied)
    float opacity = 0.0, coverage = 0.0, fog = 1.0, softRange = 0.0;
    float3 fogColour = g_Ps[97].rgb;
    const float3 texColour = mean.a > 1e-4 ? mean.rgb / mean.a : s0.rgb;
    switch (kind)
    {
    case VOL_SMOKE:
    case VOL_SMOKE_DETAIL:
    {
        // Colour = sprite x TEXCOORD2.rgb + add (TEXCOORD3 / 4), opacity sprite.a x TEXCOORD2.w, fog TEXCOORD1.z / 3.x.
        const bool detail = kind == VOL_SMOKE_DETAIL;
        colour = texColour * v[a].t2.rgb + (detail ? v[a].t4.rgb : v[a].t3.rgb);
        opacity = saturate(max(s0.a, mean.a) * v[a].t2.w);
        coverage = mean.a * v[a].t2.w;
        fog = detail ? v[a].t3.x : v[a].t1.z;
        softRange = detail ? v[a].t4.w : v[a].t3.w;
        break;
    }
    case VOL_FLAME:
    {
        // Three taps of the sprite (TEXCOORD1.xy, 1.zw, 2.xy) through ApplyColor_Alpha0 (c50), Alpha1 (c52.x),
        // ApplyColor2 (c51), added: rgb x a summed, coverage their mean.
        float4 s1, s2;
        const float2 uv1 = (v[a].t1.zw + v[b].t1.zw) * 0.5, uv2 = (v[a].t2.xy + v[b].t2.xy) * 0.5;
        const float4 m1 = SpriteMean(texture, sampler, uv1, (v[a].t1.zw - uv1) * 0.5, (v[c].t1.zw - uv1) * 0.5, s1);
        const float4 m2 = SpriteMean(texture, sampler, uv2, (v[a].t2.xy - uv2) * 0.5, (v[c].t2.xy - uv2) * 0.5, s2);
        const float4 c50 = g_Ps[50], c51 = g_Ps[51];
        const float a1 = g_Ps[52].x;
        colour = mean.rgb * c50.rgb * c50.a + m1.rgb * c50.rgb * a1 + m2.rgb * c51.rgb * c51.a;
        coverage = saturate((mean.a * c50.a + m1.a * a1 + m2.a * c51.a) / 3.0);
        opacity = saturate((s0.a * c50.a + s1.a * a1 + s2.a * c51.a) / 3.0);
        softRange = v[a].t2.z;
        break;
    }
    case VOL_ADD_DETAIL:
    {
        // Sprite x TEXCOORD2 (rgb, a), x AddDouble (c50.x), x the detail texture's mix (TEXCOORD1.zw, c96.w).
        float4 sd;
        const float2 uvd = (v[a].t1.zw + v[b].t1.zw) * 0.5;
        SpriteMean(texture, sampler, uvd, (v[a].t1.zw - uvd) * 0.5, (v[c].t1.zw - uvd) * 0.5, sd);
        const float3 detail = lerp(sd.rgb, sd.aaa, g_Ps[96].w);
        const float gain = v[a].t2.w * g_Ps[50].x;
        colour = mean.rgb * v[a].t2.rgb * gain * detail;
        coverage = saturate(mean.a * gain);
        opacity = saturate(s0.a * gain);
        softRange = v[a].t4.w;
        break;
    }
    case VOL_CLOUD:
    {
        // Slot 6's sprite x its detail tap's colour (TEXCOORD1.zw, mixed by c96.w) x TEXCOORD3 (rgb, a), faded by the
        // altitude band: TEXCOORD2.x through CloudAlphaAltitudeUpperLower (c85) as the game computes it, max(lower) then
        // min(upper), less lower, over upper - lower (an inverted band gives 1). Fog TEXCOORD2.y towards CloudFogColor (c80).
        float4 sd;
        const float2 uvd = (v[a].t1.zw + v[b].t1.zw) * 0.5;
        SpriteMean(texture, sampler, uvd, (v[a].t1.zw - uvd) * 0.5, (v[c].t1.zw - uvd) * 0.5, sd);
        const float3 detail = lerp(sd.rgb, sd.aaa, g_Ps[96].w);
        const float4 band = g_Ps[85];
        const float range = band.x - band.y;
        const float altitude = abs(range) > 1e-6 ? saturate((min(max(v[a].t2.x, band.y), band.x) - band.y) / range) : 1.0;
        colour = texColour * detail * v[a].t3.rgb;
        coverage = mean.a * v[a].t3.w * altitude;
        opacity = saturate(max(s0.a, mean.a) * v[a].t3.w * altitude);
        fog = v[a].t2.y;
        fogColour = g_Ps[80].rgb;
        softRange = v[a].t2.z;
        // For the cloudscape: the sprite's mean alpha, the view depth, the game's distance fade (TEXCOORD3.w)
        // without its near fade (within DebugZ_FadeDistance.y, 100 m, of the camera: a cloud flown into stays).
        other = float3(mean.a, v[a].pos.w, v[a].pos.w < 150.0 ? 1.0 : saturate(v[a].t3.w));
        break;
    }
    case VOL_TRAIL:
    {
        // Sprite x TEXCOORD1 (rgb, a) + TEXCOORD2.rgb, fog TEXCOORD0.z towards CustomFogColor (c93), no soft fade.
        colour = texColour * v[a].t1.rgb + v[a].t2.rgb;
        coverage = mean.a * v[a].t1.w;
        opacity = saturate(max(s0.a, mean.a) * v[a].t1.w);
        fog = v[a].t0.z;
        fogColour = g_Ps[93].rgb;
        break;
    }
    default:   // VOL_TRAIL_ADD
    {
        // Sprite x TEXCOORD1 (rgb, a) x AddDouble (c50.x).
        const float gain = v[a].t1.w * g_Ps[50].x;
        colour = mean.rgb * v[a].t1.rgb * gain;
        coverage = saturate(mean.a * gain);
        opacity = saturate(s0.a * gain);
        break;
    }
    }
    // The game's additive shaders clamp each pixel's glow to 1: the mean of a clamped glow is 1 at most.
    if (VolEmissive(kind))
        colour = saturate(colour);
    const uint64_t address = ((uint64_t(g_Gs.addressHi) << 32) | g_Gs.addressLo) + uint64_t(g_Gs.base + prim / 2) * VOL_RECORD;
    vk::RawBufferStore<float4>(address, float4(centre, opacity));
    vk::RawBufferStore<float4>(address + 16, float4(corner, fog));
    vk::RawBufferStore<float4>(address + 32, float4(colour, coverage));
    vk::RawBufferStore<uint>(address + 48, g_Gs.flags >> 8);   // the frame's tag: a slot whose draw didn't run is skipped
    // The soft-particle fade's depth range and threshold, as the game's pixel shaders apply them (none for the trails).
    vk::RawBufferStore<float2>(address + 52, float2(softRange, g_Ps[103].x));
    vk::RawBufferStore<uint>(address + 60, kind | PackColour(fogColour));
    vk::RawBufferStore<float4>(address + 64, float4(other, 0.0));
}

#elif defined(VOL_COMPOSE)
// The half-resolution volumes onto the frame (premultiplied, bilinear).
Texture2D<float4> g_Textures[] : register(t0, space0);
SamplerState g_Samplers[] : register(s0, space3);
// offsetX: 1 for an 8-bit scene.
struct PushConstants { uint textureIndex; uint samplerIndex; int offsetX; int offsetY; };
[[vk::push_constant]] ConstantBuffer<PushConstants> g_Push;
struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
float4 main(VsOut i) : SV_Target0
{
    float4 c = g_Textures[g_Push.textureIndex].SampleLevel(g_Samplers[0], i.uv, 0.0);
    if (g_Push.offsetX == 1)
    {
        // In the game's 8-bit scene nothing of ours may pass 240 / 255: the codes above are its lit and emitter codes,
        // which its bloom and exposure take as 1.1 to 100 x white (a lit missile trail seen along its length would
        // reach 1.25 x white and flare like a sun). As the additive kinds' glow nears that limit it replaces the scene
        // behind it, so the sum stays under the limit too.
        const float limit = 240.0 / 255.0;
        c.a = max(c.a, saturate(max(max(c.r, c.g), c.b) / limit));
        c.rgb = min(c.rgb, c.a * limit);
    }
    return c;
}

#elif defined(VOL_CLOUDSCAPE)
// The cloudscape (AC6_VOL_CLOUDSCAPE): the cloud puffs' records as one cloud field, marched once a
// pixel, in place of a volume per billboard. As Ace Combat 7's clouds are built (a placement map
// over the layer, a height profile, noise), with the placement taken from the game's own puffs each frame:
//  - VOL_CLOUD_VS + VOL_CLOUD_MAP: each puff as a disc into a top-down map around the camera (four levels, each wider
//    and coarser, blended with MAX): the coverage, the top's and the bottom's altitude (the puff's sphere),
//  - VOL_CLOUD_VS + VOL_CLOUD_BOUNDS: each puff's sphere on screen (MAX): where a pixel's ray first enters one and last
//    leaves one, so the march only walks where clouds can be,
//  - VOL_CLOUD_MARCH: a full-screen march through the map (the puffs' balls, eroded by world-anchored noise, a few
//    steps towards the sun, the game's cloud colours, altitude band and fog), premultiplied, and the view depth the
//    cloud is seen at. The scene is not in it: the picture depends on the camera and the clouds alone,
//  - VOL_CLOUD_TEMPORAL: that picture blended into the last frames' (reprojected by the cloud's depth, clamped to what
//    is around now): the march's noise averaged out,
//  - VOL_CLOUD_OVER: the result over the scene, where the scene is behind the cloud's depth (softly), at the scene's
//    own resolution: an aircraft in front of a cloud keeps its outline.
Texture2D<float4> g_Textures[] : register(t0, space0);
Texture3D<float4> g_Textures3D[] : register(t0, space1);
SamplerState g_Samplers[] : register(s0, space3);
#if !defined(VOL_CLOUD_OVER) && !defined(VOL_CLOUD_SCENE) && !defined(VOL_CLOUD_DEPTH) && !defined(VOL_CLOUD_GATE)
// mode: the map's level (0-3), 4 for the bounds, or 5 for the clearance map.
struct CloudPush { uint paramsLo; uint paramsHi; uint particlesLo; uint particlesHi; uint depthIndex; uint noiseIndex; uint mode; };
[[vk::push_constant]] ConstantBuffer<CloudPush> g_Vol;

uint64_t Params() { return (uint64_t(g_Vol.paramsHi) << 32) | g_Vol.paramsLo; }
float4 P(uint i) { return vk::RawBufferLoad<float4>(Params() + i * 16); }
uint4 PU(uint i) { return vk::RawBufferLoad<uint4>(Params() + i * 16); }
// Rows 0-17 as the volumes' (below). The cloudscape's:
// 18 the clouds' fog colour and rate (CloudFogColor_Param)
// 19 CloudColorAmbient, 20 CloudColorDiffuse, 21 CloudColorUpperAmbient, 22 CloudColorLowerAmbient (the relighting
//    applied)
// 23 the alpha band's upper and lower altitude, the ambient band's upper and lower
// 24 the map levels' half extents
// 25, 26 their centres (x, z each)
// 27 a level's texels, the atlas's reciprocal size (x, y), the altitude reference
// 28 (uint) the map's, the bounds' and the clearance map's textures
// 29 the puffs' size scale, coverage gain, top and bottom scales
// 30 the extinction per metre, the base noise's and the detail noise's tile (metres), the detail's weight
// 31 the most steps, a fine step's least length and its growth with distance, the coarse stride in fine steps
// 32 the sun's gain, the ambient's gain, the forward lobe's g, the extinction per metre towards the sun
// 33 a debug view, the output gain, the phase's cap, the envelope's gain
// 34 towards the sun (world), 1 for a float scene (colours past 1 kept)
// 35 the finest noise's tile (metres) and weight, the base noise's weight, the edges' sharpness
// 36 the ambient at a cloud's base, the temporal blend (the new frame's share), 1 with a usable last frame, 1 when the
//    cloudscape ran this frame
// 37 (uint) the march's picture, its depth, the last frame's result
// 38-41 the last frame's view-projection rows
// 42 the thickness a puff's coverage gives, a pixel's angle x the noise's texels (its mip level), the warp's tile and
//    reach (metres)
// 43 the field's threshold, how much cloud above takes of the ambient, the march's step over the target's pixels (1,
//    or 2: a quarter of them a frame), the frame's phase

float3 Unproject(float3 ndc)
{
    const float4 h = ndc.x * P(0) + ndc.y * P(1) + ndc.z * P(2) + P(3);
    return h.xyz / h.w;
}

// A point from its place on screen and its view depth. The cloud puffs' records are placed so: their projection's depth
// scale is not the camera's the other records are unprojected with (ProjectionMatrix c40 against ViewProjectionMatrix
// c116), so their NDC depth read with it puts them at about 2.3 times their distance.
// (With the camera the records were harvested with, rows 46-51: last frame's when the clouds are drawn into the scene
// before this frame's harvest.)
float3 AtDepth(float2 ndc, float depth)
{
    const float3 eye = P(50).xyz;
    const float4 h = ndc.x * P(46) + ndc.y * P(47) + 0.5 * P(48) + P(49);
    const float3 dir = normalize(h.xyz / h.w - eye);
    return eye + dir * (depth / max(dot(dir, P(51).xyz), 1e-4));
}

// Which pixel of each 2 x 2 the march takes this frame (P(43): z the step, 1 or 2, w the frame's phase, 0-3).
int2 PhaseOffset()
{
    if (P(43).z < 2.0)
        return int2(0, 0);
    const uint phase = uint(P(43).w) & 3;
    return int2(phase == 1 || phase == 2 ? 1 : 0, phase == 1 || phase == 3 ? 1 : 0);
}

float2 LevelCentre(uint level)
{
    const float4 c = level < 2 ? P(25) : P(26);
    return (level & 1) ? c.zw : c.xy;
}

#define CLOUD_CEILING 8192.0   // the bottom is stored below this, above the reference
#endif

#if defined(VOL_CLOUD_VS) || defined(VOL_CLOUD_MAP) || defined(VOL_CLOUD_BOUNDS)
struct CloudOut
{
    float4 pos : SV_Position;
    nointerpolation float4 sphere : TEXCOORD0;   // centre, radius (the bounds: the cylinder's axis x, bottom, axis z, radius)
    nointerpolation float4 data : TEXCOORD1;     // coverage, the cylinder's top, -, the coverage x the distance fade
    float2 offset : TEXCOORD2;                   // the map: from the centre, in radii
};
#endif

#ifdef VOL_CLOUD_VS
CloudOut main(uint vertex : SV_VertexID, uint instance : SV_InstanceID)
{
    CloudOut o = (CloudOut)0;
    const uint64_t record = ((uint64_t(g_Vol.particlesHi) << 32) | g_Vol.particlesLo) + uint64_t(instance) * VOL_RECORD;
    const uint packed = vk::RawBufferLoad<uint>(record + 60);
    if ((packed & 15) != VOL_CLOUD || vk::RawBufferLoad<uint>(record + 48) != asuint(P(15).x))
        return o;
    const float4 r0 = vk::RawBufferLoad<float4>(record), r1 = vk::RawBufferLoad<float4>(record + 16);
    if (any(!isfinite(r0.xyz)) || any(!isfinite(r1.xyz)))
        return o;
    // The sprite's mean alpha and the game's distance fade (the harvest: without the near fade and the altitude band).
    const float4 raw = vk::RawBufferLoad<float4>(record + 64);
    const float3 centre = AtDepth(r0.xy, raw.y);
    const float3 corner = AtDepth(r1.xy, raw.y);
    const float radius = length(corner - centre) * 0.70710678 * P(29).x;
    // The sprite's alpha is the cloud's shape, the game's distance fade (from 20 km of view depth, which changes as the
    // camera turns) how thick it is: a far cloud thins out, it doesn't shrink (as coverage, the fade would change the
    // far clouds' shapes with the camera's angle).
    // And by its distance, not its view depth: the game fades and then culls a puff past 20 to 24 km of view depth, so
    // a puff 30 km away is there at the edge of a wide view (19 km deep) and gone when the camera turns to it. Faded over
    // the same 20 to 24 km of distance (never less than its depth), the clouds are the same whichever way the camera looks.
    const float coverage = saturate(raw.x * P(29).y);
    const float amount = coverage * min(saturate(raw.z), saturate(1.0 - (length(centre - P(8).xyz) - 20000.0) / 4000.0));
    if (!(radius > 1.0) || !(amount > 0.002))
        return o;
    const float2 uv = float2(vertex == 1 || vertex == 4 || vertex == 5 ? 1.0 : 0.0, vertex == 2 || vertex == 3 || vertex == 5 ? 1.0 : 0.0);
    const float2 q = uv * 2.0 - 1.0;
    o.sphere = float4(centre, radius);
    o.data = float4(coverage, 0.0, 0.0, amount);
    o.offset = q;
    if (g_Vol.mode < 4)
    {
        const float2 n = (centre.xz + q * radius - LevelCentre(g_Vol.mode)) / P(24)[g_Vol.mode];
        o.pos = float4(n.x, n.y, 0.5, 1.0);
        return o;
    }
    // The bounds: all the puff's cloud can be, as the map and the march make it (VOL_CLOUD_MAP: a mound from the clouds'
    // base to the ball's top, nothing under the middle of the altitude band, Warped: read up to the warp's reach away):
    // an upright cylinder. (Bounded by the puff's ball alone, the part of a tall mound under its ball would be
    // marched only where another puff's ball lies along the ray: clouds would change shape with the view.)
    const float4 band = P(23);
    const bool banded = band.x > band.y + 1.0;
    const float base = banded ? min(centre.y, band.x) : centre.y;
    const float reach = P(42).w;
    float lo = base - radius * P(29).w;
    if (banded)
        lo = max(lo, (band.x + band.y) * 0.5);
    lo -= reach * 0.6;
    const float hi = centre.y + radius * P(29).z + reach * 0.6;
    const float around = radius + reach * 0.5;
    o.sphere = float4(centre.x, lo, centre.z, around);
    o.data.y = hi;
    if (g_Vol.mode == 5)
    {
        // The clearance map: the cylinder's disc wider by the clearance and three quarters of a texel (a point in
        // an empty texel is then at least the clearance from every puff's cylinder), over the widest level's square.
        const float wide = around + P(45).x + 1.5 * P(24).w / P(45).y;
        const float2 n = (centre.xz + q * wide - LevelCentre(3)) / P(24).w;
        o.pos = float4(n.x, n.y, 0.5, 1.0);
        return o;
    }
    // Its quad: about the cylinder's bounding ball, facing the eye (not the screen: seen off the view's axis a ball's
    // outline is wider than a screen-facing quad its size, by 1 / cos of the angle: 1.7 times at a 21:9 view's corners,
    // and such a quad would cut the clouds there into upright strips that change as the camera turns).
    const float3 middle = float3(centre.x, (lo + hi) * 0.5, centre.z);
    const float ball = sqrt(around * around + (hi - lo) * (hi - lo) * 0.25);
    const float3 camera = P(8).xyz;
    const float away = length(middle - camera);
    if (away < ball * 1.6)
        o.pos = float4(q, 0.5, 1.0);   // the camera in or near it: the whole screen
    else
    {
        const float3 towards = (middle - camera) / away;
        const float3 side = normalize(cross(P(12).xyz, towards));
        const float3 upward = cross(towards, side);
        const float size = ball * 1.03 / sqrt(1.0 - (ball * ball) / (away * away));
        const float3 w = middle + (q.x * side + q.y * upward) * size;
        o.pos = w.x * P(4) + w.y * P(5) + w.z * P(6) + P(7);
    }
    return o;
}

#elif defined(VOL_CLOUD_MAP)
float4 main(CloudOut i) : SV_Target0
{
    const float u2 = dot(i.offset, i.offset);
    if (u2 >= 1.0)
        discard;
    const float dome = sqrt(1.0 - u2);
    const float reference = P(27).w;
    if (g_Vol.mode == 5)   // the clearance map: cloud near here, the highest top and the lowest bottom of it (MAX)
        return float4(1.0, max(i.data.y - reference, 0.0), max(reference + CLOUD_CEILING - i.sphere.y, 0.0), 1.0);
    // A puff above the clouds' base (the alpha band's upper edge, where most sit) is a mound rising from the base, not a
    // ball in the air: the map has one top and one bottom a column, and a ball's rim at its own altitude, over a lower
    // puff, would fill the air between them (walls, columns).
    const float4 band = P(23);
    const float base = band.x > band.y + 1.0 ? min(i.sphere.y, band.x) : i.sphere.y;
    const float top = base + (i.sphere.y - base) * (1.0 - u2) + i.sphere.w * dome * P(29).z - reference;
    const float bottom = reference + CLOUD_CEILING - (base - i.sphere.w * dome * P(29).w);
    // (The fourth: the puff's own coverage x the game's distance fade, unshaped: how thick a cloud it is.)
    return float4(i.data.x * (1.0 - u2), max(top, 0.0), max(bottom, 0.0), i.data.w);
}

#elif defined(VOL_CLOUD_BOUNDS)
float4 main(CloudOut i) : SV_Target0
{
    const float2 screen = i.pos.xy * P(13).zw;
    const float2 ndc = float2(screen.x * 2.0 - 1.0, 1.0 - screen.y * 2.0);
    const float3 camera = P(8).xyz;
    const float3 dir = normalize(Unproject(float3(ndc, 0.5)) - camera);
    // The ray's span in the puff's upright cylinder (the vertex shader: sphere = its axis x, its bottom, its axis z, its
    // radius, data.y its top).
    const float2 oc = camera.xz - i.sphere.xz;
    const float a = dot(dir.xz, dir.xz);
    float t0 = -1e30, t1 = 1e30;
    if (a > 1e-8)
    {
        const float b = dot(oc, dir.xz);
        const float h = b * b - a * (dot(oc, oc) - i.sphere.w * i.sphere.w);
        if (h <= 0.0)
            discard;
        const float root = sqrt(h);
        t0 = (-b - root) / a;
        t1 = (-b + root) / a;
    }
    else if (dot(oc, oc) > i.sphere.w * i.sphere.w)
        discard;
    if (abs(dir.y) > 1e-6)
    {
        const float s0 = (i.sphere.y - camera.y) / dir.y, s1 = (i.data.y - camera.y) / dir.y;
        t0 = max(t0, min(s0, s1));
        t1 = min(t1, max(s0, s1));
    }
    else if (camera.y < i.sphere.y || camera.y > i.data.y)
        discard;
    t0 = max(t0, 0.0);
    if (t1 <= t0)
        discard;
    // The nearest entry (as 1 / (1 + km): MAX keeps the nearest) and the farthest exit.
    return float4(1.0 / (1.0 + t0 * 0.001), min(t1, 64000.0), 0.0, 1.0);
}

#elif defined(VOL_CLOUD_TEMPORAL)
struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
// The march's picture is of one pixel in each 2 x 2 here a frame (PhaseOffset: which), or of all of them (P(43).z = 1).
// A pixel marched this frame is blended into what the last frames hold there. The others keep the last frames' alone,
// reprojected. Without a usable last frame, the march's picture stretched.
float4 main(VsOut i) : SV_Target0
{
    const int2 pixel = int2(i.pos.xy);
    const int scale = int(P(43).z);
    const int2 offset = PhaseOffset();
    Texture2D<float4> current = g_Textures[PU(37).x];
    float2 size;
    current.GetDimensions(size.x, size.y);
    const int2 last = int2(size) - 1;
    const int2 cell = min(pixel / scale, last);
    const bool fresh = scale == 1 || all((pixel & 1) == offset);
    // The march's picture at this pixel, between its samples (their centres are at pixel offset + 0.5 of each cell).
    const float2 at = ((i.pos.xy - float2(offset) - 0.5) / float(scale) + 0.5) / size;
    const float4 c = fresh ? current.Load(int3(cell, 0)) : current.SampleLevel(g_Samplers[0], at, 0.0);
    const float4 temporal = P(36);
    if (temporal.z == 0.0)
        return c;
    // What the march has around now bounds what the last frames may add.
    float4 lo = c, hi = c;
    [unroll] for (int y = -1; y <= 1; y++)
        [unroll] for (int x = -1; x <= 1; x++)
        {
            const float4 n = current.Load(int3(clamp(cell + int2(x, y), int2(0, 0), last), 0));
            lo = min(lo, n);
            hi = max(hi, n);
        }
    if (hi.a <= 0.0)
        return 0.0;
    // Where this pixel's cloud was on screen a frame ago: by its depth (the nearest of the puffs' bounds where the
    // march found none here, the clouds don't move, the camera does).
    const float2 screen = i.pos.xy * P(13).zw;
    const float2 ndc = float2(screen.x * 2.0 - 1.0, 1.0 - screen.y * 2.0);
    const float3 camera = P(8).xyz;
    const float3 dir = normalize(Unproject(float3(ndc, 0.5)) - camera);
    const float forward = max(dot(dir, P(16).xyz), 1e-4);
    float depth = g_Textures[PU(37).y].Load(int3(cell, 0)).x;
    if (depth <= 0.0)
    {
        const float4 bounds = g_Textures[PU(28).y].Load(int3(pixel, 0));
        depth = bounds.y > 0.0 ? (1.0 / max(bounds.x, 1e-5) - 1.0) * 1000.0 * forward : 20000.0;
    }
    const float3 p = camera + dir * (depth / forward);
    const float4 clip = p.x * P(38) + p.y * P(39) + p.z * P(40) + P(41);
    if (clip.w <= 0.0)
        return c;
    const float2 was = float2(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
    if (any(was < 0.0) || any(was > 1.0))
        return c;
    const float4 history = clamp(g_Textures[PU(37).z].SampleLevel(g_Samplers[0], was, 0.0), lo, hi);
    return fresh ? lerp(history, c, temporal.y) : history;
}

#elif defined(VOL_CLOUD_OVER)
struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
// alpha, beta: the scene's depth as a view depth, beta / (z - alpha), soft: the metres over which a cloud covers what
// enters it, jitter: the scene's sub-pixel jitter this frame (TAA, FSR), x and y + 1 in 1 / 32767.
struct OverPush { uint cloudIndex; uint frontIndex; uint depthIndex; float alpha; float beta; float soft; uint jitter; };
[[vk::push_constant]] ConstantBuffer<OverPush> g_Over;
float4 main(VsOut i) : SV_Target0
{
    const float4 c = g_Textures[g_Over.cloudIndex].SampleLevel(g_Samplers[0], i.uv, 0.0);
    if (c.a <= 0.002)
        discard;
    // The nearest cloud of the texels around (4 x 4: a texel without cloud has no depth, and the picture, blended over
    // frames and stretched, reaches a little past the texels that have).
    Texture2D<float4> front = g_Textures[g_Over.frontIndex];
    float2 size;
    front.GetDimensions(size.x, size.y);
    const int2 base = int2(floor(i.uv * size - 0.5)) - 1;
    float nearest = 1e30;
    [unroll] for (uint k = 0; k < 16; k++)
    {
        const float f = front.Load(int3(clamp(base + int2(k & 3, k >> 2), int2(0, 0), int2(size) - 1), 0)).x;
        if (f > 0.0)
            nearest = min(nearest, f);
    }
    // The scene's depth where this pixel is in the scene's image, which is drawn moved by the frame's jitter (and, with
    // FSR's upscaling, smaller) while this target is the resolved, steady picture: the four texels about that place, by
    // their shares. (Read at the pixel itself, the cloud's edge against an aircraft would stand up to a texel and
    // a half off the aircraft's, on another side each frame: a white outline over the aircraft where a cloud is behind
    // it.)
    Texture2D<float4> depthImage = g_Textures[g_Over.depthIndex];
    float2 depthSize;
    depthImage.GetDimensions(depthSize.x, depthSize.y);
    const float2 jitter = float2(g_Over.jitter & 0xFFFFu, g_Over.jitter >> 16) / 32767.0 - 1.0;
    const float2 at = i.uv * depthSize + jitter - 0.5;
    const int2 corner = int2(floor(at));
    const float2 share = at - floor(at);
    float cover = 0.0;
    [unroll] for (uint t = 0; t < 4; t++)
    {
        const int2 o = int2(t & 1, t >> 1);
        const float z = depthImage.Load(int3(clamp(corner + o, int2(0, 0), int2(depthSize) - 1), 0)).x;
        const float weight = (o.x ? share.x : 1.0 - share.x) * (o.y ? share.y : 1.0 - share.y);
        // Over the sky always, over the scene where it is behind the cloud (softly), and not at all where the cloud's
        // depth isn't known (a faint edge: drawn there, it would cover whatever is there, the aircraft in front too).
        const float covered = z <= 0.0 ? 1.0 : nearest > 1e29 ? 0.0
            : saturate((g_Over.beta / max(z - g_Over.alpha, 1e-12) - nearest) / g_Over.soft + 0.5);
        cover += weight * covered;
    }
    return c * cover;
}

#elif defined(VOL_CLOUD_SCENE)
// The clouds into the scene's own surface, before the scene is resolved: each of a pixel's samples is tested
// against the scene's depth there (the depth this writes out: the cloud's, 100 m nearer), so an aircraft's edge in front
// of a cloud is aircraft and cloud by its samples, and the anti-aliasing after (the resolve, TAA, FSR) blends those.
// (Drawn over the resolved picture instead, the edge's pixels would already be aircraft blended with the sky: a row of
// sky along the aircraft wherever the cloud behind is darker or lighter than the sky, a white outline.)
struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
struct OverPush { uint cloudIndex; uint frontIndex; uint depthIndex; float alpha; float beta; float soft; uint jitter; };
[[vk::push_constant]] ConstantBuffer<OverPush> g_Over;
struct SceneOut { float4 colour : SV_Target0; float depth : SV_Depth; };
SceneOut main(VsOut i)
{
    // The scene is drawn moved by the frame's jitter (TAA, FSR): the clouds' picture, which is not, is read where this
    // pixel is without it.
    const float2 jitter = float2(g_Over.jitter & 0xFFFFu, g_Over.jitter >> 16) / 32767.0 - 1.0;
    const float2 uv = i.uv - jitter * (i.uv / i.pos.xy);
    const float4 c = g_Textures[g_Over.cloudIndex].SampleLevel(g_Samplers[0], uv, 0.0);
    if (c.a <= 0.002)
        discard;
    Texture2D<float4> front = g_Textures[g_Over.frontIndex];
    float2 size;
    front.GetDimensions(size.x, size.y);
    const int2 base = int2(floor(uv * size - 0.5)) - 1;
    float nearest = 1e30;
    [unroll] for (uint k = 0; k < 16; k++)
    {
        const float f = front.Load(int3(clamp(base + int2(k & 3, k >> 2), int2(0, 0), int2(size) - 1), 0)).x;
        if (f > 0.0)
            nearest = min(nearest, f);
    }
    SceneOut o;
    // Where the cloud's depth isn't known (a faint edge): over the sky alone (its depth is 0, the farthest).
    if (nearest > 1e29)
    {
        o.colour = c;
        o.depth = 1e-12;
        return o;
    }
    // Softly over what is near the cloud's depth: by the scene's resolved depth, the farthest of the 3 x 3 about the
    // pixel (at an edge the samples behind count: the samples in front fail the depth test).
    float cover = 1.0;
    if (g_Over.depthIndex != 0)
    {
        Texture2D<float4> depthImage = g_Textures[g_Over.depthIndex];
        float2 depthSize;
        depthImage.GetDimensions(depthSize.x, depthSize.y);
        const int2 at = int2(i.uv * depthSize);
        float farthest = 1.0;
        [unroll] for (int t = 0; t < 9; t++)
            farthest = min(farthest, depthImage.Load(int3(clamp(at + int2(t % 3 - 1, t / 3 - 1), int2(0, 0), int2(depthSize) - 1), 0)).x);
        if (farthest > 0.0)
            cover = saturate((g_Over.beta / max(farthest - g_Over.alpha, 1e-12) - nearest) / g_Over.soft + 0.5);
    }
    o.colour = c * cover;
    o.depth = saturate(g_Over.alpha + g_Over.beta / max(nearest - g_Over.soft * 0.5, 1.0));
    return o;
}

#elif defined(VOL_CLOUD_GATE)
// The game's background cloud sprites behind the cloudscape. The game's effects pass draws over a copy of the
// scene (ReductionBufferHalfRes: the scene's colour, alpha 0), the sprites first. That copy as it was before them
// (frontIndex: kept by the renderer) is blended back by the clouds' alpha (colour: src alpha, 1 - src alpha. Alpha: the
// pass's alpha, its coverage, x 1 - src alpha): behind a cloud the sprites fade as the cloud thickens. (Cut by the
// clouds' depth instead, the band of sprites would end on a hard line along each cloud.)
struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
struct OverPush { uint cloudIndex; uint frontIndex; uint depthIndex; float alpha; float beta; float soft; uint jitter; };
[[vk::push_constant]] ConstantBuffer<OverPush> g_Over;
float4 main(VsOut i) : SV_Target0
{
    const float cloud = saturate(g_Textures[g_Over.cloudIndex].SampleLevel(g_Samplers[0], i.uv, 0.0).a * g_Over.soft);
    if (cloud <= 0.002)
        discard;
    return float4(g_Textures[g_Over.frontIndex].Load(int3(int2(i.pos.xy), 0)).rgb, cloud);
}

#elif defined(VOL_CLOUD_DEPTH)
// The clouds' depth, where a cloud covers more than a share of the pixel (soft: the threshold): as a colour into the
// scene's resolved depth (blended with MAX: the nearer, the depth being reversed), or, VOL_CLOUD_DEPTH_OUT, as
// the depth of a depth-only draw into the game's effects pass. What the game draws there after (tracers,
// particles) is then behind or in front of the clouds as it is of the scene.
struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
struct OverPush { uint cloudIndex; uint frontIndex; uint depthIndex; float alpha; float beta; float soft; uint jitter; };
[[vk::push_constant]] ConstantBuffer<OverPush> g_Over;
#ifdef VOL_CLOUD_DEPTH_OUT
float main(VsOut i) : SV_Depth
#else
float4 main(VsOut i) : SV_Target0
#endif
{
    const float2 jitter = float2(g_Over.jitter & 0xFFFFu, g_Over.jitter >> 16) / 32767.0 - 1.0;
    const float2 uv = i.uv - jitter * (i.uv / i.pos.xy);
    if (g_Textures[g_Over.cloudIndex].SampleLevel(g_Samplers[0], uv, 0.0).a < g_Over.soft)
        discard;
    Texture2D<float4> front = g_Textures[g_Over.frontIndex];
    float2 size;
    front.GetDimensions(size.x, size.y);
    const int2 base = int2(floor(uv * size - 0.5)) - 1;
    float nearest = 1e30;
    [unroll] for (uint k = 0; k < 16; k++)
    {
        const float f = front.Load(int3(clamp(base + int2(k & 3, k >> 2), int2(0, 0), int2(size) - 1), 0)).x;
        if (f > 0.0)
            nearest = min(nearest, f);
    }
    if (nearest > 1e29)
        discard;
#ifdef VOL_CLOUD_DEPTH_OUT
    return saturate(g_Over.alpha + g_Over.beta / max(nearest, 1.0));
#else
    return float4(saturate(g_Over.alpha + g_Over.beta / max(nearest, 1.0)), 0.0, 0.0, 0.0);
#endif
}

#else   // VOL_CLOUD_MARCH
struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
struct MarchOut { float4 colour : SV_Target0; float4 front : SV_Target1; };

float Remap(float v, float lo, float hi)
{
    return saturate((v - lo) / max(hi - lo, 1e-4));
}

struct MapSample { float cov; float top; float bottom; float amount; };

// The map at a world position: the finest level holding it.
MapSample Map(float2 xz)
{
    MapSample m;
    m.cov = 0.0;
    m.top = 0.0;
    m.bottom = 1.0;
    m.amount = 0.0;
    const float4 extents = P(24);
    uint level = 4;
    float2 n = 0.0;
    [unroll] for (uint l = 0; l < 4; l++)
    {
        const float2 nl = (xz - LevelCentre(l)) / extents[l];
        if (level == 4 && max(abs(nl.x), abs(nl.y)) < 0.98)
        {
            level = l;
            n = nl;
        }
    }
    if (level == 4)
        return m;
    const float4 atlas = P(27);
    float2 uv = float2(n.x, -n.y) * 0.5 + 0.5;
    uv = clamp(uv, 0.5 / atlas.x, 1.0 - 0.5 / atlas.x);
    const float4 s = g_Textures[PU(28).x].SampleLevel(g_Samplers[0], float2((uv.x + float(level)) * atlas.x * atlas.y, uv.y * atlas.x * atlas.z), 0.0);
    m.cov = s.x;
    m.top = atlas.w + s.y;
    m.bottom = atlas.w + CLOUD_CEILING - s.z;
    m.amount = s.w;
    return m;
}

// The clouds' field at a point: positive inside a cloud, growing inwards, hn: the height within the cloud, amount: the
// puffs' own coverage there. For one puff the map holds a (1 - u^2) (u: the distance from its axis in radii) and its
// top and bottom, so the coverage x (1 - v^2) (v: the height from the middle over half the thickness) is
// a (1 - |x - centre|^2 / r^2): the puff's ball, greatest at its centre. For several, the largest of them about. Times a
// gain and the game's altitude band (its sprites fade out below it: the flat bases, the puffs' centres sit on it), less a
// threshold: the surface is where the noise below brings the sum to 0. -10 outside the map's columns.
float Field(float3 p, out float hn, out float amount)
{
    hn = 0.0;
    amount = 0.0;
    const MapSample m = Map(p.xz);
    const float thickness = m.top - m.bottom;
    if (m.cov <= 0.003 || thickness <= 1.0)
        return -10.0;
    hn = (p.y - m.bottom) / thickness;
    if (hn <= 0.0 || hn >= 1.0)
        return -10.0;
    amount = m.amount;
    const float v = hn * 2.0 - 1.0;
    // (Over the band's upper half: a base a hundred metres deep is haze the march wades through.)
    const float4 band = P(23);
    const float fade = band.x > band.y + 1.0 ? saturate((p.y - (band.x + band.y) * 0.5) / ((band.x - band.y) * 0.5)) : 1.0;
    return m.cov * (1.0 - v * v) * P(33).w * fade - P(43).x;
}

// How far a ray can go from a point before it can meet cloud, 0 if cloud may be near. The clearance map (the
// vertex shader's mode 5) holds, a kilometre texel, whether any puff's cylinder is within the clearance of the texel, and
// the highest top and the lowest bottom of those: outside that the ray is clear for the clearance sideways, or until it
// comes down to the tops (up to the bottoms).
float Clearance(float3 p, float3 dir)
{
    const float4 c = P(45);
    const float sideways = c.x / max(length(dir.xz), 1e-3);
    const float2 n = (p.xz - LevelCentre(3)) / P(24).w;
    if (max(abs(n.x), abs(n.y)) >= 1.0)
        return sideways;
    const float2 uv = float2(n.x, -n.y) * 0.5 + 0.5;
    const float4 s = g_Textures[PU(28).z].Load(int3(min(int2(uv * c.y), int(c.y) - 1), 0));
    if (s.x <= 0.0)
        return sideways;
    const float hi = P(27).w + s.y, lo = P(27).w + CLOUD_CEILING - s.z;
    if (p.y > hi)
        return min(sideways, dir.y < 0.0 ? (p.y - hi) / -dir.y : 1e30);
    if (p.y < lo)
        return min(sideways, dir.y > 0.0 ? (lo - p.y) / dir.y : 1e30);
    return 0.0;
}

// How much cloud a puff's coverage makes: thin sprites, thin clouds.
float Thickness(float amount)
{
    return saturate(amount * P(42).x);
}

// The noise's mip level for a tile that size (metres) at that distance: a texel no smaller than a pixel.
float NoiseLevel(float tile, float t)
{
    return max(log2(max(t * P(42).y / tile, 1e-6)), 0.0);
}

// Where a point reads the clouds' shapes: pushed about by slow noise (as Ace Combat 7's clouds are read), less at the
// base than at the top: the puffs' balls and mounds become heaps with shoulders and overhangs, their outlines not
// circles.
float3 Warped(float3 p, float t)
{
    const float4 w = P(42);
    if (w.w <= 0.0)
        return p;
    Texture3D<float4> noise = g_Textures3D[g_Vol.noiseIndex];
    const float level = NoiseLevel(w.z, t);
    // (One sample: its two channels sideways, their difference upwards.)
    const float2 a = noise.SampleLevel(g_Samplers[1], p / w.z + float3(0.13, 0.57, 0.29), level).xy;
    const float4 band = P(23);
    const float rise = band.x > band.y + 1.0 ? saturate((p.y - band.y) / 600.0) : 1.0;
    return p + float3(a.x - 0.5, a.x - a.y, a.y - 0.5) * (w.w * lerp(0.35, 1.0, rise));
}

// How much of the noise a point takes: all of it well inside the clouds' shapes, none where there is no shape at all
// (At full weight everywhere the noise alone would make wisps where the shape is nothing, under the flat bases, and
// the march would stop there at every stride to look for them: it would run out of steps along the bases, and the
// clouds behind would be missing).
float NoiseWeight(float field)
{
    return saturate((field + P(43).x) * 3.0);
}

// The density, 0-1. The field plus noise about 0 moves the surface in and out (by more on a larger puff: its field
// rises more slowly): the base noise the lumps, then the cellular noise, inverted (what is between its cells goes, the
// cells stay as billows), at two sizes, the finer near the camera only. Sharpened: a cloud's edge is a surface, not a haze.
// The base noise alone for the shadow rays and the strides (margin: what the finer noise could still add).
float DensityCoarse(float3 p, float time, float t, float margin)
{
    float hn, amount;
    const float field = Field(p, hn, amount);
    if (field <= -10.0)
        return 0.0;
    const float weight = NoiseWeight(field);
    if (weight <= 0.0)
        return 0.0;
    const float base = g_Textures3D[g_Vol.noiseIndex].SampleLevel(g_Samplers[1], p / P(30).y + float3(time * 0.0007, 0.0, time * 0.0004),
        NoiseLevel(P(30).y, t)).x;
    const float thickness = Thickness(amount);
    return saturate((field + (P(35).z * (base - 0.5) + margin) * weight) * P(35).w * lerp(0.25, 1.0, thickness)) * thickness;
}

// p: the point (the noise is read there), field: the clouds' field where the point reads the shapes (Warped).
// near: 1 close to the camera, 0 far.
float Density(float3 p, float time, float field, float near, float t, float thickness)
{
    Texture3D<float4> noise = g_Textures3D[g_Vol.noiseIndex];
    const float4 n = P(30), fine = P(35);
    const float weight = NoiseWeight(field);
    if (weight <= 0.0)
        return 0.0;
    const float reach = 0.5 * (n.w + fine.y * near) * weight;
    float f = field + weight * fine.z *
        (noise.SampleLevel(g_Samplers[1], p / n.y + float3(time * 0.0007, 0.0, time * 0.0004), NoiseLevel(n.y, t)).x - 0.5);
    if (f + reach <= 0.0)
        return 0.0;
    f += weight * n.w * (0.5 - noise.SampleLevel(g_Samplers[1], p / n.z + float3(0.37, time * 0.004, 0.11), NoiseLevel(n.z, t)).y);
    if (near > 0.0)
        f += weight * fine.y * near *
            (0.5 - noise.SampleLevel(g_Samplers[1], p / fine.x + float3(0.71, time * 0.01, 0.53), NoiseLevel(fine.x, t)).y);
    // (A thin puff's cloud is a haze: its edge soft, or the noise would speckle it.)
    return saturate(f * fine.w * lerp(0.25, 1.0, thickness)) * thickness;
}

float Phase(float c, float g)
{
    return (1.0 - g * g) / pow(max(1.0 + g * g - 2.0 * g * c, 1e-4), 1.5);
}

MarchOut Debug(float4 colour)
{
    MarchOut o;
    o.colour = colour;
    o.front = 1.0;   // a metre away: over everything
    return o;
}

MarchOut main(VsOut i)
{
    // The target's pixel (P(13): its size): this cell's, of the 2 x 2 it stands for this frame.
    const float4 size = P(13);
    const int2 pixel = min(int2(i.pos.xy) * int(P(43).z) + PhaseOffset(), int2(size.xy) - 1);
    const float4 bounds = g_Textures[PU(28).y].Load(int3(pixel, 0));
    const float view = P(33).x;
    if (view == 4.0)   // the map: its four levels across the screen
    {
        const float4 m = g_Textures[PU(28).x].SampleLevel(g_Samplers[0], i.uv, 0.0);
        return Debug(float4(m.x, m.y / 4000.0, (CLOUD_CEILING - m.z) / 4000.0, 1.0) * 0.9);
    }
    if (view == 5.0)
        return Debug(float4(bounds.x, bounds.y / 30000.0, 0.0, 1.0) * 0.9);
    // (View 6: the bounds ignored, every ray marched from the camera to 60 km: what the bounds must not change. For
    // checks, with AC6_CLOUD_STEPS=512.)
    if (bounds.y <= 0.0 && view != 6.0)
        discard;
    float tNear = max((1.0 / max(bounds.x, 1e-5) - 1.0) * 1000.0 - 4.0, 0.0);
    float tFar = bounds.y + 4.0;
    if (view == 6.0)
    {
        tNear = 0.0;
        tFar = 60000.0;
    }
    const float2 screen = (float2(pixel) + 0.5) * size.zw;
    const float2 ndc = float2(screen.x * 2.0 - 1.0, 1.0 - screen.y * 2.0);
    const float3 camera = P(8).xyz;
    const float3 dir = normalize(Unproject(float3(ndc, 0.5)) - camera);
    if (view == 1.0)   // the bounds: red the entry, green the span (10 km each)
        return Debug(float4(saturate(tNear / 10000.0), saturate((tFar - tNear) / 10000.0), 0.0, 1.0) * 0.7);
    const float time = P(8).w;
    uint seed = uint(pixel.x) * 1973u + uint(pixel.y) * 9277u + uint(time * 60.0) * 26699u;
    seed = (seed ^ 61u) ^ (seed >> 16);
    seed *= 9u;
    seed ^= seed >> 4;
    seed *= 0x27D4EB2Du;
    seed ^= seed >> 15;
    const float jitter = float(seed & 0xFFFFu) / 65536.0;
    const float4 stepping = P(31);
    const uint maxSteps = uint(stepping.x);
    const float sigma = P(30).x;
    const float3 sun = P(34).xyz;
    const float4 lighting = P(32);
    const float cosine = dot(dir, sun);
    // Two lobes (forwards and a little backwards), 1 at a right angle to the sun, capped (the thin edges towards the sun).
    const float g = lighting.z;
    const float phase = min((Phase(cosine, g) + Phase(cosine, -0.25)) / (Phase(0.0, g) + Phase(0.0, -0.25)), P(33).z);
    const float forward = dot(dir, P(16).xyz);
    const float3 fogColour = P(18).rgb;
    const float4 band = P(23);
    float3 light = 0.0;
    float transmittance = 1.0, depthSum = 0.0, weightSum = 0.0;
    bool fine = false;
    uint empty = 0, used = 0;
    float resume = 0.0, thin = 1.0;
    float t = tNear + jitter * max(stepping.y, tNear * stepping.z);
    for (uint s = 0; s < maxSteps && t < tFar; s++)
    {
        used++;
        // (thin: a step up to three times as long while the last sample was thin haze or nothing: it integrates as well.)
        const float dt = max(stepping.y, t * stepping.z) * thin;
        const float3 p = camera + dir * t;
        if (!fine)
        {
            // Across the air that is known to be empty in one step (striding through it, a ray that grazes a near
            // cloud would have no steps left for the clouds behind: they would be missing about the near cloud's
            // outline).
            const float clear = Clearance(p, dir);
            if (clear > 0.0)
            {
                t += max(clear, dt * stepping.w);
                continue;
            }
            // Strides through the rest of the empty air, back one stride, in fine steps, where there is cloud.
            if (DensityCoarse(Warped(p, t), time, t, 0.5 * (P(30).w + P(35).y)) > 0.0)
            {
                fine = true;
                empty = 0;
                thin = 1.0;
                resume = t;
                t = max(t - dt * stepping.w + dt * jitter, tNear);
            }
            else
                t += dt * stepping.w;
            continue;
        }
        float hn, amount;
        const float3 shaped = Warped(p, t);
        const float field = Field(shaped, hn, amount);
        const float near = saturate(1.0 - t / 15000.0);
        const float d = field > -10.0 ? Density(p, time, field, near, t, Thickness(amount)) : 0.0;
        thin = lerp(3.0, 1.0, saturate(d * 6.0));
        if (d > 0.002)
        {
            empty = 0;
            // Towards the sun: the optical depth over four spans, each nearly three times the last (to about 1 km).
            // (The first span with the detail: the billows shade each other.)
            float tau = 0.0, start = 0.0, end = 45.0;
            {
                const float3 q = p + sun * 20.0;
                float hq, aq;
                const float fq = Field(shaped + sun * 20.0, hq, aq);
                tau += (fq > -10.0 ? Density(q, time, fq, near, t, Thickness(aq)) : 0.0) * end;
                start = end;
                end *= 2.2;
            }
            [unroll] for (uint k = 1; k < 4; k++)
            {
                tau += DensityCoarse(shaped + sun * (start + end) * 0.5, time, t, 0.0) * (end - start);
                start = end;
                end *= 2.8;
            }
            tau *= lighting.w;
            // The light that still arrives after scattering on the way (three octaves, each dimmer and less absorbed).
            const float arrived = (exp(-tau) + 0.5 * exp(-tau * 0.4) + 0.25 * exp(-tau * 0.15)) / 1.75;
            // The sky's light: less at a cloud's base and under more cloud (one sample above: the shaded sides keep
            // their relief).
            const float above = DensityCoarse(shaped + float3(0.0, 90.0, 0.0), time, t, 0.0);
            const float3 ambient = (P(19).rgb + lerp(P(22).rgb, P(21).rgb, saturate((p.y - band.w) / max(band.z - band.w, 1.0)))) *
                (lerp(P(36).x, 1.05, hn) * (1.0 - P(43).y * above));
            // As the game's pixel shaders write a colour: clamped to 1 unless the scene is float, x 240 / 255 (the
            // codes above 240 are its lit and emitter codes).
            float3 c = P(20).rgb * (arrived * phase * lighting.x) + ambient * lighting.y;
            if (P(34).w == 0.0)
                c = saturate(c);
            c *= P(33).y;
            c = lerp(fogColour, c, saturate(exp(P(18).w * t * forward)));
            const float a = 1.0 - exp(-sigma * d * dt);
            light += transmittance * a * c;
            depthSum += transmittance * a * t;
            weightSum += transmittance * a;
            transmittance *= 1.0 - a;
            if (transmittance < 0.02)
                break;
        }
        else if (t > resume && ++empty >= 4)
            fine = false;
        t += dt;
    }
    if (view == 2.0)   // the steps taken
        return Debug(float4(float(used) / float(maxSteps), float(used) / float(maxSteps) > 0.98 ? 1.0 : 0.0, 0.0, 1.0) * 0.8);
    const float alpha = 1.0 - transmittance;
    if (view == 3.0)
        return Debug(float4(alpha, alpha, alpha, 1.0));
    if (alpha < 0.002)
        discard;
    MarchOut o;
    // What was left when the march stopped is taken as covered (the last 2%).
    o.colour = transmittance < 0.02 ? float4(light / alpha, 1.0) : float4(light, alpha);
    // The view depth the cloud is seen at: its samples' mean, by what each added.
    o.front = depthSum / max(weightSum, 1e-6) * forward;
    return o;
}
#endif

#else   // VOL_VS / VOL_PS
Texture2D<float4> g_Textures[] : register(t0, space0);
Texture3D<float4> g_Textures3D[] : register(t0, space1);
SamplerState g_Samplers[] : register(s0, space3);
// params: the frame's camera and lighting (renderer.cpp RunVolumetrics), particles: the harvested records.
struct VolPush { uint paramsLo; uint paramsHi; uint particlesLo; uint particlesHi; uint depthIndex; uint noiseIndex; };
[[vk::push_constant]] ConstantBuffer<VolPush> g_Vol;

uint64_t Params() { return (uint64_t(g_Vol.paramsHi) << 32) | g_Vol.paramsLo; }
float4 P(uint i) { return vk::RawBufferLoad<float4>(Params() + i * 16); }
uint4 PU(uint i) { return vk::RawBufferLoad<uint4>(Params() + i * 16); }
// 0-3 inverse view-projection rows, 4-7 view-projection rows, 8 camera + time, 9 towards the sun + size scale,
// 10 fog colour + density gain, 11 camera right + steps, 12 camera up + light gain, 13 target size + reciprocals,
// 14 noise scale, detail erosion, ball falloff, shadow floor, 15 the frame's tag (x) and a debug view (y), 16 forward,
// 17 the trails' size scale, the additive kinds' glow gain, the trails' noise size (world units), the kinds drawn (mask).
// (18-43: the cloudscape's, above.) 44 the steps a pixel of radius on screen, the radius from which the light is marched,
// the depth buffer's a and b (view depth = b / (z - a)), 45 the clearance map's clearance (metres) and texels,
// the scene's jitter this frame (its pixels), 46-49 the records' camera's inverse view-projection rows, 50 its eye,
// 51 its forward.

float3 Unproject(float3 ndc)
{
    const float4 h = ndc.x * P(0) + ndc.y * P(1) + ndc.z * P(2) + P(3);
    return h.xyz / h.w;
}

struct VolOut
{
    float4 pos : SV_Position;
    nointerpolation float4 sphere : TEXCOORD0;   // centre, radius (a trail: its segment's bounding sphere)
    nointerpolation float4 colour : TEXCOORD1;   // the game's colour (additive kinds: the emission), opacity through the centre
    nointerpolation float4 extra : TEXCOORD2;    // fog, extinction, noise offset
    nointerpolation float4 soft : TEXCOORD3;     // the centre's NDC depth, the soft fade's range and threshold, emission per unit
    nointerpolation float4 tubeA : TEXCOORD4;    // a trail's ends (world) and radius, w 0 for the others
    nointerpolation float4 tubeB : TEXCOORD5;
    nointerpolation float4 fog : TEXCOORD6;      // the fog colour, the kind
    float3 at : TEXCOORD7;                       // a world point on the pixel's ray (the quad's own)
};

#ifdef VOL_VS
VolOut main(uint vertex : SV_VertexID, uint instance : SV_InstanceID)
{
    VolOut o = (VolOut)0;
    const uint64_t record = ((uint64_t(g_Vol.particlesHi) << 32) | g_Vol.particlesLo) + uint64_t(instance) * VOL_RECORD;
    const float4 r0 = vk::RawBufferLoad<float4>(record), r1 = vk::RawBufferLoad<float4>(record + 16);
    const float4 r2 = vk::RawBufferLoad<float4>(record + 32);
    const uint packed = vk::RawBufferLoad<uint>(record + 60);
    const uint kind = packed & 15;
    const bool emissive = VolEmissive(kind), tube = VolTube(kind);
    const bool drawn = (asuint(P(17).w) >> kind) & 1;
    // Faint additive records still glow: their coverage is no measure of what they add.
    const bool faint = emissive ? !(max(max(r2.r, r2.g), r2.b) > 0.002) : !(r0.w > 0.003);
    if ((faint || !drawn || vk::RawBufferLoad<uint>(record + 48) != asuint(P(15).x)) && (P(15).y == 0.0 || P(15).y >= 5.0))
        return o;   // degenerate: nothing drawn
    // Particles the game hides with a zero w (dead trail segments, culled puffs) come out infinite.
    if (any(!isfinite(r0.xyz)) || any(!isfinite(r1.xyz)))
        return o;
    float3 centre = Unproject(r0.xyz);
    float3 corner = Unproject(r1.xyz);
    if (kind == VOL_CLOUD)
    {
        // By their view depth: the clouds' projection has another depth scale than the camera here.
        const float depth = vk::RawBufferLoad<float4>(record + 64).y;
        const float3 eye = P(8).xyz;
        const float3 toCentre = normalize(centre - eye), toCorner = normalize(corner - eye);
        centre = eye + toCentre * (depth / max(dot(toCentre, P(16).xyz), 1e-4));
        corner = eye + toCorner * (depth / max(dot(toCorner, P(16).xyz), 1e-4));
    }
    float radius = length(corner - centre) * 0.70710678 * P(9).w;
    float chord = 1.3333 * radius;   // the mean chord through the volume, across its projected area
    if (tube)
    {
        // The segment from one cross-section's middle to the other's, as thick as the ribbon is wide.
        const float3 bNdc = vk::RawBufferLoad<float4>(record + 64).xyz;
        if (any(!isfinite(bNdc)))
            return o;
        const float3 b = Unproject(bNdc);
        const float r = length(corner - centre) * P(17).x;
        o.tubeA = float4(centre, r);
        o.tubeB = float4(b, 0.0);
        centre = (centre + b) * 0.5;
        radius = length(b - centre) + r;
        chord = 1.5708 * r;
    }
    // The noise's offset follows the particle (continuously: the noise tiles), so a rising puff churns. A trail's noise
    // is the world's (ShapeAt).
    const float2 offset = tube ? 0.0 : float2(centre.x * 0.0137 + centre.z * 0.0071, centre.y * 0.0113 + centre.z * 0.0059);
    const float3 camera = P(8).xyz;
    const float2 uv = float2(vertex == 1 || vertex == 4 || vertex == 5 ? 1.0 : 0.0, vertex == 2 || vertex == 3 || vertex == 5 ? 1.0 : 0.0);
    const float away = length(centre - camera);
    if (away < radius * 1.6)
    {
        o.pos = float4(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, 0.5, 1.0);   // the camera in or near it: the whole screen
        o.at = Unproject(float3(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, 0.5));
    }
    else
    {
        // As wide as the ball's outline from there (wider than the ball, by more the nearer), and 3% over, facing the
        // eye, not the screen (off the view's axis the outline is wider than a screen-facing quad).
        const float reach = radius * 1.03 / sqrt(1.0 - (radius * radius) / (away * away));
        const float3 towards = (centre - camera) / away;
        const float3 side = normalize(cross(P(12).xyz, towards));
        const float3 upward = cross(towards, side);
        const float3 q = centre + (uv.x * 2.0 - 1.0) * reach * side + (uv.y * 2.0 - 1.0) * reach * upward;
        o.pos = q.x * P(4) + q.y * P(5) + q.z * P(6) + P(7);
        o.at = q;
    }
    // The billboard's mean coverage (its sprite's mean alpha over the quad, about the disc's area) as the volume's mean
    // opacity: the mean chord, the density's mean and patchiness in the factor (AC6_VOL_DENSITY). Faint large
    // puffs stay faint. The centre's opacity isn't used (scaled up, it would make old expanded puffs a dark mass).
    // Thin puffs are linear (the factor is the density's mean alone), dense ones also fill the noise's gaps.
    // The additive kinds absorb nothing. Their emission is scaled the same way, so a ray through the mean chord adds
    // about the billboard's mean glow (AC6_VOL_GLOW).
    const float opacity = min(r2.w * 1.05, 0.95);
    const float extinction = emissive ? 0.0 : -log(1.0 - opacity) * P(10).w * lerp(0.4, 1.25, opacity) / chord;
    o.soft = float4(r0.z, vk::RawBufferLoad<float2>(record + 52), emissive ? P(17).y / chord : 0.0);
    o.sphere = float4(centre, radius);
    // Its radius on screen, in the target's pixels: how finely it is marched.
    o.tubeB.w = radius / max(length(centre - camera) * P(42).y / 64.0, 1e-6);
    o.colour = float4(r2.rgb, opacity);
    o.extra = float4(r1.w, extinction, offset);
    // The smoke's fog colour is the frame's (FogColor_Param), the others' their own.
    const float3 fogColour = kind <= VOL_SMOKE_DETAIL ? P(10).rgb : float3((packed >> 8) & 255, (packed >> 16) & 255, packed >> 24) / 255.0;
    o.fog = float4(fogColour, float(kind));
    return o;
}
#else
// As Guerrilla Games' Nubis clouds are made: the base noise remapped by a soft ball (cauliflower edges), then eroded
// by finer noise at the edges.
float Remap(float v, float lo, float hi)
{
    return saturate((v - lo) / max(hi - lo, 1e-4));
}

// Where x is in the volume's own frame: r2, the squared distance against its shape (a ball: from its centre over its
// radius, a tube: from its axis over its radius, inside below 1), and n, where the noise is read.
struct Shape { float r2; float3 n; };

Shape ShapeAt(float3 x, float3 centre, float radius, float4 tubeA, float4 tubeB, float2 offset)
{
    Shape s;
    if (tubeA.w > 0.0)
    {
        const float3 ba = tubeB.xyz - tubeA.xyz;
        const float h = saturate(dot(x - tubeA.xyz, ba) / max(dot(ba, ba), 1e-6));
        const float3 q = (x - tubeA.xyz - ba * h) / tubeA.w;
        s.r2 = dot(q, q);
        // World-anchored noise (a trail's segments join without seams), P(17).z world units a cell.
        s.n = x / P(17).z;
        return s;
    }
    const float3 q = (x - centre) / radius;
    s.r2 = dot(q, q);
    s.n = q * P(14).x + offset.xyx;
    return s;
}

float Density(float3 x, float3 centre, float radius, float4 tubeA, float4 tubeB, float2 offset, float time)
{
    const Shape s = ShapeAt(x, centre, radius, tubeA, tubeB, offset);
    if (s.r2 >= 1.0)
        return 0.0;
    const float4 n = P(14);   // noise scale, detail erosion, ball falloff, shadow floor
    Texture3D<float4> noise = g_Textures3D[g_Vol.noiseIndex];
    const float3 drift = float3(0.0, time * 0.03, 0.0);
    const float shape = saturate(1.0 - s.r2 * n.z);
    const float base = noise.SampleLevel(g_Samplers[1], s.n + drift, 0.0).x;
    float d = Remap(base, 1.0 - shape, 1.0);
    if (d <= 0.0)
        return 0.0;
    const float detail = noise.SampleLevel(g_Samplers[1], s.n * 3.1 + offset.yxy - offset.xyx * 3.1 - drift * 1.7, 0.0).x;
    d = Remap(d, detail * n.y * (1.0 - d), 1.0);
    return d;
}

// The base noise alone, for the shadow rays.
float DensityCoarse(float3 x, float3 centre, float radius, float4 tubeA, float4 tubeB, float2 offset, float time)
{
    const Shape s = ShapeAt(x, centre, radius, tubeA, tubeB, offset);
    if (s.r2 >= 1.0)
        return 0.0;
    const float3 drift = float3(0.0, time * 0.03, 0.0);
    const float base = g_Textures3D[g_Vol.noiseIndex].SampleLevel(g_Samplers[1], s.n + drift, 0.0).x;
    return Remap(base, 1.0 - saturate(1.0 - s.r2 * P(14).z), 1.0);
}

// The ray's span inside a tube (a finite cylinder: from a to b, radius r), false if it misses.
bool TubeSpan(float3 origin, float3 dir, float3 a, float3 b, float r, out float t0, out float t1)
{
    const float3 ba = b - a, oc = origin - a;
    const float baba = dot(ba, ba), bard = dot(ba, dir), baoc = dot(ba, oc);
    t0 = -1e30;
    t1 = 1e30;
    // The infinite cylinder.
    const float k2 = baba - bard * bard;
    const float k1 = baba * dot(oc, dir) - baoc * bard;
    const float k0 = baba * dot(oc, oc) - baoc * baoc - r * r * baba;
    if (abs(k2) > 1e-6 * baba)
    {
        const float h = k1 * k1 - k2 * k0;
        if (h <= 0.0)
            return false;
        const float root = sqrt(h);
        t0 = (-k1 - root) / k2;
        t1 = (-k1 + root) / k2;
    }
    else if (k0 > 0.0)
        return false;   // along the axis, outside
    // Between the end planes.
    if (abs(bard) > 1e-6)
    {
        const float s0 = -baoc / bard, s1 = (baba - baoc) / bard;
        t0 = max(t0, min(s0, s1));
        t1 = min(t1, max(s0, s1));
    }
    else if (baoc < 0.0 || baoc > baba)
        return false;
    return t1 > t0;
}

float4 main(VolOut i) : SV_Target0
{
    const float4 size = P(13);
    const float2 screen = i.pos.xy * size.zw;
    const float3 camera = P(8).xyz;
    // (The ray through the quad's own point: no unprojection a pixel.)
    const float3 dir = normalize(i.at - camera);
    const float3 centre = i.sphere.xyz;
    const float radius = i.sphere.w;
    const uint kind = uint(i.fog.w);
    const bool emissive = VolEmissive(kind);
    const float view = P(15).y;
    float t0, t1;
    if (i.tubeA.w > 0.0)
    {
        if (!TubeSpan(camera, dir, i.tubeA.xyz, i.tubeB.xyz, i.tubeA.w, t0, t1))
            discard;
        t0 = max(t0, 0.0);
    }
    else
    {
        const float3 oc = camera - centre;
        const float b = dot(oc, dir);
        const float h = b * b - (dot(oc, oc) - radius * radius);
        if (h <= 0.0)
            discard;
        const float root = sqrt(h);
        t0 = max(-b - root, 0.0);
        t1 = -b + root;
    }
    if (view == 1.0)
        return float4(0.5, 0.0, 0.0, 0.5);
    if (view == 2.0)
        return float4(i.colour.rgb * i.colour.a, i.colour.a);
    if (view == 5.0)   // valid volumes: red = opacity, green = radius / 100, blue = the additive kinds
        return float4(i.colour.a, saturate(radius / 100.0), emissive ? 1.0 : 0.0, 1.0) * 0.4;
    if (view == 6.0)   // the kinds: smoke grey, flames orange, clouds white, trails blue
    {
        const float3 colours[7] = { float3(0.5, 0.5, 0.5), float3(0.4, 0.4, 0.4), float3(1.0, 0.5, 0.0), float3(1.0, 0.2, 0.0),
            float3(1.0, 1.0, 1.0), float3(0.2, 0.4, 1.0), float3(0.0, 1.0, 1.0) };
        return float4(colours[min(kind, 6u)], 1.0) * 0.4;
    }
    // The scene in front: its depth (reversed, infinite far plane: 0 is the sky).
    Texture2D<float4> depthImage = g_Textures[g_Vol.depthIndex];
    float2 depthSize;
    depthImage.GetDimensions(depthSize.x, depthSize.y);
    // (Where this pixel is in the scene's jittered image, P(45).zw.)
    const float depth = depthImage.Load(int3(clamp(int2(screen * depthSize + P(45).zw), int2(0, 0), int2(depthSize) - 1), 0)).x;
    // (Its view depth is b / (z - a), P(44).zw, along the ray, that over the ray's share of the view's axis.)
    const float3 forward = P(16).xyz;
    const float along = max(dot(dir, forward), 1e-4);
    const float4 lod = P(44);
    const float scene = depth > 0.0 ? lod.w / max(depth - lod.z, 1e-12) / along : 1e30;
    t1 = min(t1, scene);
    if (t1 <= t0)
        discard;
    // The billboard's own visibility there (its camera-facing plane through the centre against the scene, softly, as
    // the game's soft particles fade): a puff whose sprite the ground or a building hid stays hidden.
    const float plane = dot(centre - camera, forward) / along;
    // and the game's soft-particle fade, in its reversed depth units: (the centre's depth - the scene's) / range when the
    // difference is past the threshold and within the range (distant puffs against the sky fade out: the game's look).
    const float difference = i.soft.x - depth;
    const float soft = difference >= i.soft.z && difference < i.soft.y ? saturate(difference / i.soft.y) : 1.0;
    const float thickness = i.tubeA.w > 0.0 ? i.tubeA.w : radius;
    const float visible = saturate((scene - plane) / (thickness * 0.6) + 0.5) * soft;
    if (visible <= 0.0)
        discard;
    if (view == 3.0)
    {
        const float3 x = camera + dir * (t0 + t1) * 0.5;
        const Shape s = ShapeAt(x, centre, radius, i.tubeA, i.tubeB, i.extra.zw);
        const float n = g_Textures3D[g_Vol.noiseIndex].SampleLevel(g_Samplers[1], s.n, 0.0).x;
        return float4(n, Density(x, centre, radius, i.tubeA, i.tubeB, i.extra.zw, P(8).w), (t1 - t0) / (2.0 * thickness), 1.0) * 0.8;
    }
    // As many steps as its size on screen shows (a step each few pixels of radius, 2 at least): most of a smoke column's
    // particles are a few pixels across, and the full count for each would make the pass far costlier.
    const uint steps = uint(clamp(ceil(i.tubeB.w * lod.x), 2.0, P(11).w));
    const float dt = (t1 - t0) / float(steps);
    // The samples' start moves per pixel and frame (a hash: interleaved gradient noise draws streaks with few steps).
    const float time = P(8).w;
    uint seed = uint(i.pos.x) * 1973u + uint(i.pos.y) * 9277u + uint(time * 60.0) * 26699u;
    seed = (seed ^ 61u) ^ (seed >> 16);
    seed *= 9u;
    seed ^= seed >> 4;
    seed *= 0x27D4EB2Du;
    seed ^= seed >> 15;
    const float jitter = float(seed & 0xFFFFu) / 65536.0;
    const float3 sun = P(9).xyz;
    const float lightStep = thickness * 0.4;
    const float4 n = P(14);
    const float3 fogColour = i.fog.rgb;
    // A trail seen along its length: the game's ribbon turns about its axis to face the camera, so it shows its width
    // once however long the ray runs inside the tube. A tube adds up its whole length there (a missile leaving the
    // player's aircraft: a white haze about the camera, its glow a fan of fire). As much as the ray crosses the
    // tube, no more: the extinction and the glow by the sine of the angle between them.
    float across = 1.0;
    if (i.tubeA.w > 0.0)
    {
        const float3 axis = i.tubeB.xyz - i.tubeA.xyz;
        across = max(length(cross(dir, axis)) / max(length(axis), 1e-4), 0.03);
    }
    const float extinction = i.extra.y * across;
    float3 light = 0.0;
    float transmittance = 1.0;
    if (emissive)
    {
        // Glow: the emission where the density is, nothing absorbed (the game adds these).
        float sum = 0.0;
        for (uint s = 0; s < steps; s++)
            sum += Density(camera + dir * (t0 + (float(s) + jitter) * dt), centre, radius, i.tubeA, i.tubeB, i.extra.zw, time);
        light = i.colour.rgb * (sum * dt * i.soft.w * visible * across);
        if (max(max(light.r, light.g), light.b) < 0.001)
            discard;
        return float4(light, 0.0);
    }
    for (uint s = 0; s < steps; s++)
    {
        const float3 x = camera + dir * (t0 + (float(s) + jitter) * dt);
        const float d = Density(x, centre, radius, i.tubeA, i.tubeB, i.extra.zw, time);
        if (d <= 0.001)
            continue;
        // Towards the sun: two coarse samples for a volume large on screen, for a small one its place in the ball
        // (lit on the sun's side, shadowed across).
        float shadow = 0.0;
        if (i.tubeB.w >= lod.y)
        {
            [unroll] for (uint k = 1; k <= 2; k++)
                shadow += DensityCoarse(x + sun * lightStep * float(k), centre, radius, i.tubeA, i.tubeB, i.extra.zw, time);
        }
        else
            shadow = 2.0 * d * saturate(0.5 - 0.5 * dot(x - centre, sun) / thickness);
        const float lit = exp(-extinction * lightStep * shadow * P(12).w);
        const float3 c = lerp(fogColour, i.colour.rgb * lerp(n.w, 1.25, lit), i.extra.x);
        const float a = 1.0 - exp(-extinction * d * dt);
        light += transmittance * a * c;
        transmittance *= 1.0 - a;
        if (transmittance < 0.01)
            break;
    }
    // Behind the cloudscape's cloud here: covered by it, softly (the volumes are composited after the clouds).
    float under = 1.0;
    if (P(36).w != 0.0)
    {
        const uint3 cloud = PU(37).xyz;
        float2 frontSize;
        g_Textures[cloud.y].GetDimensions(frontSize.x, frontSize.y);
        const float front = g_Textures[cloud.y].Load(int3(min(int2(screen * frontSize), int2(frontSize) - 1), 0)).x;
        if (front > 0.0)
        {
            const float depthHere = dot(centre - camera, forward);
            under = 1.0 - g_Textures[cloud.z].SampleLevel(g_Samplers[0], screen, 0.0).a * saturate((depthHere - front) / 300.0 + 0.5);
        }
    }
    light *= visible * under;
    const float alpha = (1.0 - transmittance) * visible * under;
    if (view == 4.0)
        return float4(alpha, i.extra.y * radius, i.colour.a, 1.0);
    if (alpha < 0.002)
        discard;
    return float4(light, alpha);
}
#endif
#endif
