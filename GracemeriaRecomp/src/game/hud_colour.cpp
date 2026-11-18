// The flight HUD's colours. Everything the HUD draws goes through its renderer's three lists, each given a colour
// (ARGB): line strips (0x820925D0, r6), polygons (0x82092850, r7) and sprites (0x82092CF0, r7). The game's HUD is
// green: 00FF20, a dimmer 00C820 and the panels' dark 2E523E, each at many alphas. Friends are blue (0080FF: radar,
// labels, the kill list), enemies red (FF1E00, FF2800: radar, target boxes and labels). AC6_HUD_COLOUR,
// AC6_HUD_FRIENDLY and AC6_HUD_ENEMY (RRGGBB) swap each family, kept as bright as it was. Greys and white stay, and all
// of it only with AC6_HUD_CUSTOM = custom. Protanopia, deuteranopia and tritanopia bring their own three colours.
// Subtitles are drawn elsewhere and keep their colours. With nothing to change the hooks change nothing.
#include "game/hud_colour.h"

#include "endian.h"
#include "memory.h"
#include "ppc.h"
#include "settings.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <string>

namespace ac6::game {

bool HudColoursClose(uint32_t a, uint32_t b)
{
    // The "redmean" approximation of how far apart two colours look.
    const int ar = (a >> 16) & 255, ag = (a >> 8) & 255, ab = a & 255, br = (b >> 16) & 255, bg = (b >> 8) & 255, bb = b & 255;
    const int mean = (ar + br) / 2, dr = ar - br, dg = ag - bg, db = ab - bb;
    const double d = std::sqrt(double((512 + mean) * dr * dr / 256 + 4 * dg * dg + (767 - mean) * db * db / 256));
    return d < 230.0;
}

HudPalette ResolveHudPalette(uint32_t hud, uint32_t friendly, uint32_t enemy)
{
    HudPalette p{ hud == kHudAuto ? kGameHud : hud, friendly, enemy };
    if (p.friendly == kHudAuto)
    {
        p.friendly = kGameFriendly;
        for (uint32_t c : { kGameFriendly, 0x00E0C0u, 0xF2F2F2u, 0xE040FFu })
            if (!HudColoursClose(c, p.hud))
            {
                p.friendly = c;
                break;
            }
    }
    if (p.enemy == kHudAuto)
    {
        p.enemy = kGameEnemy;
        for (uint32_t c : { kGameEnemy, 0xFF00C0u, 0xFFD000u, 0xF2F2F2u })
            if (!HudColoursClose(c, p.hud) && !HudColoursClose(c, p.friendly))
            {
                p.enemy = c;
                break;
            }
    }
    return p;
}

// Each preset changes only the colours that collide for its kind of colour blindness. They were chosen by simulating
// dichromacy (Machado 2009) so that the three colours stay apart and each stays clear of the sky, cloud and city.
// White separates best but is lost against cloud.
bool HudPreset(std::string_view mode, HudPalette& palette)
{
    if (mode == "protanopia")
        palette = { 0xFFFF40, kGameFriendly, kGameEnemy };
    else if (mode == "deuteranopia")
        palette = { 0x00E5FF, kGameFriendly, 0xFFD000 };
    else if (mode == "tritanopia")
        palette = { kGameHud, 0xFFD000, kGameEnemy };
    else
        return false;
    return true;
}

} // namespace ac6::game

namespace {

using namespace ac6::game;

thread_local bool t_listSprite = false;   // in the HUD's sprite list, whose sprites have their colour already

uint32_t Setting(const char* name)
{
    const std::string v = ac6::settings::Get(name);
    char* end = nullptr;
    const unsigned long c = std::strtoul(v.c_str(), &end, 16);
    return v.size() == 6 && end && !*end ? uint32_t(c) : kHudAuto;
}

struct Active
{
    bool hud = false;   // a HUD colour other than the game's green is in force
    HudPalette palette{ kGameHud, kGameFriendly, kGameEnemy };
};

// The colours in force, read again after a live change (the overlay's Apply).
Active Current()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::mutex mutex;
    static Active active;
    if (const uint64_t now = ac6::settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        std::lock_guard lock(mutex);
        const std::string mode = ac6::settings::Get("AC6_HUD_CUSTOM");
        HudPalette preset;
        if (HudPreset(mode, preset))
            active = { preset.hud != kGameHud, preset };
        else if (mode == "custom")
        {
            const uint32_t hud = Setting("AC6_HUD_COLOUR");
            active = { hud != kHudAuto, ResolveHudPalette(hud, Setting("AC6_HUD_FRIENDLY"), Setting("AC6_HUD_ENEMY")) };
        }
        else
            active = Active{};
        generation.store(now, std::memory_order_release);
        return active;
    }
    std::lock_guard lock(mutex);
    return active;
}

uint32_t Scaled(uint32_t argb, uint32_t colour, uint32_t level)
{
    auto scaled = [level](uint32_t c) { return std::min(c * level / 255, 255u); };
    return (argb & 0xFF000000u) | (scaled((colour >> 16) & 255) << 16) | (scaled((colour >> 8) & 255) << 8) | scaled(colour & 255);
}

uint32_t Recolour(uint32_t argb)
{
    const Active now = Current();
    const uint32_t r = (argb >> 16) & 255, g = (argb >> 8) & 255, b = argb & 255;
    if (g >= 40 && g * 5 >= r * 6 && g * 5 >= b * 6)
        return now.hud ? Scaled(argb, now.palette.hud, g) : argb;
    if (b >= 40 && b * 5 >= g * 7 && b * 5 >= r * 7)
        return now.palette.friendly != kGameFriendly ? Scaled(argb, now.palette.friendly, b) : argb;
    if (r >= 40 && r >= g * 2 && r >= b * 2)
        return now.palette.enemy != kGameEnemy ? Scaled(argb, now.palette.enemy, r) : argb;
    return argb;
}

} // namespace

// The picture-in-picture windows are outside the HUD: their tint is an untextured 2D sprite. Every 2D sprite is built by
// CAce62DSprite's constructor (0x82213BF0) from a description at r5, its colour floats r, g, b, a at +48: that catches it.
extern "C" PPC_FUNC(__imp__sub_82213BF0);
PPC_FUNC(sub_82213BF0)
{
    if (const Active now = Current(); now.hud && ctx.r5.u32 && !t_listSprite)
    {
        ac6::be<float>* c = ac6::g_memory.As<ac6::be<float>>(ctx.r5.u32 + 48);
        const float r = c[0], g = c[1], b = c[2];
        if (g >= 40.0f / 255.0f && g * 5.0f >= r * 6.0f && g * 5.0f >= b * 6.0f)
        {
            const uint32_t chosen = now.palette.hud;
            c[0] = float((chosen >> 16) & 255) / 255.0f * g;
            c[1] = float((chosen >> 8) & 255) / 255.0f * g;
            c[2] = float(chosen & 255) / 255.0f * g;
        }
    }
    __imp__sub_82213BF0(ctx, base);
}

// Their outlines are outside the lists too: each render view adds one as a type 6 primitive (r4), its colour (r6) from a
// virtual call. The lists' own primitives reach this function as type 4, already recoloured.
extern "C" PPC_FUNC(__imp__sub_82214088);
PPC_FUNC(sub_82214088)
{
    if (ctx.r4.u32 == 6)
        ctx.r6.u32 = Recolour(ctx.r6.u32);
    __imp__sub_82214088(ctx, base);
}

extern "C" PPC_FUNC(__imp__sub_820925D0);
PPC_FUNC(sub_820925D0)
{
    ctx.r6.u32 = Recolour(ctx.r6.u32);
    __imp__sub_820925D0(ctx, base);
}

extern "C" PPC_FUNC(__imp__sub_82092850);
PPC_FUNC(sub_82092850)
{
    ctx.r7.u32 = Recolour(ctx.r7.u32);
    __imp__sub_82092850(ctx, base);
}

extern "C" PPC_FUNC(__imp__sub_82092CF0);
PPC_FUNC(sub_82092CF0)
{
    ctx.r7.u32 = Recolour(ctx.r7.u32);
    t_listSprite = true;   // a chosen green would be scaled a second time by the constructor's hook
    __imp__sub_82092CF0(ctx, base);
    t_listSprite = false;
}
