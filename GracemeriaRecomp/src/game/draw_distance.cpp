// Level of detail. The models' LOD selection (0x82225080, from the per-model update 0x822254E8) projects the model's
// bounds with the view-projection and measures them on a 1280 x 720 screen: two constants that only it reads
// (0x823F6E4C, 0x823F6E50). AC6_LOD_SCALE=<k> (graphics.lod_scale, 1 to 8, default 1) measures them on a screen k times
// that size, so each model keeps its finer levels to k times the distance (and with them the parts shown only on near
// levels).
//
// Hooked at the per-model update, the selection's only caller: the two constants are written there when the setting
// changes, before the update runs. The update calls the selection when the model is visible, the global switch at
// 0x823F6E38 is on (it is: data, never written), the model isn't held ([+279]), is in state 1 ([+248])
// and its argument f1 is at least 0x82069B74's (1 degree in radians), else it clears the model's level flags.
#include "ppc.h"
#include "endian.h"
#include "log.h"
#include "memory.h"
#include "settings.h"
#include "game/draw_distance.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string>

namespace {

constexpr uint32_t kLodScreenWidth = 0x823F6E4C, kLodScreenHeight = 0x823F6E50;

// Live: read again when a setting changes.
double LodScale()
{
    const std::string v = ac6::settings::Get("AC6_LOD_SCALE");
    return !v.empty() ? std::clamp(std::atof(v.c_str()), 1.0, 8.0) : 1.0;
}

// Whether the update will call the selection (its tests, in its order).
bool SelectsLevel(const PPCContext& ctx, const uint8_t* base)
{
    const uint32_t m = ctx.r3.u32;
    auto u8 = [&](uint32_t a) { return uint32_t(base[a]); };
    auto u32 = [&](uint32_t a) { return __builtin_bswap32(*reinterpret_cast<const uint32_t*>(base + a)); };
    if (u8(m + 258) == 0 || u8(0x823F6E38) == 0 || u8(m + 279) != 0 || u32(m + 248) != 1)
        return false;
    const uint32_t bits = u32(0x82069B74);
    float threshold;
    std::memcpy(&threshold, &bits, sizeof(threshold));
    return !(ctx.f1.f64 < threshold);
}

} // namespace

// The per-model update: the screen the selection measures on, set again when a setting changes (the game's own
// 1280 x 720 at 1), before it runs.
extern "C" PPC_FUNC(__imp__sub_822254E8);
PPC_FUNC(sub_822254E8)
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    if (const uint64_t now = ac6::settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        generation.store(now, std::memory_order_release);
        static double applied = 1.0;
        if (const double s = LodScale(); s != applied)
        {
            applied = s;
            ac6::log::Info("LOD: model levels chosen on a {}x{} screen (AC6_LOD_SCALE={})", 1280.0 * s, 720.0 * s, s);
            *reinterpret_cast<ac6::be<float>*>(base + kLodScreenWidth) = float(1280.0 * s);
            *reinterpret_cast<ac6::be<float>*>(base + kLodScreenHeight) = float(720.0 * s);
        }
    }
    // AC6_LOD_TRACE=1 (diagnostics): the levels chosen ([model+208]), as a histogram every 20000 selections.
    static const bool trace = [] { const char* v = std::getenv("AC6_LOD_TRACE"); return v && *v == '1'; }();
    const uint32_t model = ctx.r3.u32;
    const bool selects = trace && SelectsLevel(ctx, base);
    __imp__sub_822254E8(ctx, base);
    if (selects)
    {
        static std::atomic<uint32_t> levels[8], calls{ 0 };
        const uint32_t level = __builtin_bswap32(*reinterpret_cast<uint32_t*>(base + model + 208));
        levels[std::min(level, 7u)].fetch_add(1, std::memory_order_relaxed);
        if (calls.fetch_add(1) % 20000 == 19999)
        {
            std::string text;
            for (auto& l : levels)
                text += std::format(" {}", l.exchange(0));
            ac6::log::Info("LOD levels chosen (0 = finest):{}", text);
        }
    }
}

// ---- Where the map's objects begin -------------------------------------------------------------
// A city's buildings are the map's parts: instances the map manager (CMapManager, vtable 0x8205C8F4) collects per
// visible cell (sub_820FFAD0, from the map's draw sub_820FDCD0, slot 61) where their squared distance to the camera is
// under their class's range: three ranges per map, small / medium / large at [map+29084] / [+29088] / [+29092]
// (".mapparts.distanceS / M / L": defaults 3000 / 6000 / 12000, loaded by sub_82103428 or sub_821048D0), each capped at
// 28000 (0x82069E1C, squared). Mission units of model type 5 are hidden by the same three (sub_8209FA50, through the
// map manager's slots 26 to 28). None of it is the models' level of detail above, so the LOD selection alone would
// leave every building appearing where the console shows it. With AC6_LOD_SCALE=k the three ranges are k times the
// map's own, before each draw of the map: the loaders write the fields, so a value other than the one written here is
// the map's own again.
namespace {

constexpr uint32_t kMapRangeSmall = 29084;   // then medium, large

struct MapRanges
{
    uint32_t map = 0;
    float own[3] = {};
    float written[3] = {};
    float logged[3] = {};
};

std::atomic<float> s_nearestRange{ 0.0f };

} // namespace

float ac6::game::MapPartsNearestRange()
{
    return s_nearestRange.load(std::memory_order_relaxed);
}

// AC6_MAP_RANGE_TRACE=1 (diagnostics): the map's own ranges, as they are first seen.
extern "C" PPC_FUNC(__imp__sub_820FDCD0);
PPC_FUNC(sub_820FDCD0)
{
    static MapRanges ranges;
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static double scale = 1.0;
    if (const uint64_t now = ac6::settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        generation.store(now, std::memory_order_release);
        scale = LodScale();
    }
    const uint32_t map = ctx.r3.u32;
    auto* fields = reinterpret_cast<ac6::be<float>*>(base + map + kMapRangeSmall);
    float now[3] = { fields[0], fields[1], fields[2] };
    if (map != ranges.map || std::memcmp(now, ranges.written, sizeof(now)) != 0)
    {
        ranges.map = map;
        std::memcpy(ranges.own, now, sizeof(now));
        std::memcpy(ranges.written, now, sizeof(now));
        // (Mission 01 sets them again every frame: 10000 / 12000 / 16000.)
        static const bool trace = [] { const char* v = std::getenv("AC6_MAP_RANGE_TRACE"); return v && *v == '1'; }();
        if (trace && std::memcmp(now, ranges.logged, sizeof(now)) != 0)
        {
            std::memcpy(ranges.logged, now, sizeof(now));
            ac6::log::Info("Map {:08X}: its parts begin at {} / {} / {} (small / medium / large), x {}", map, now[0], now[1], now[2], scale);
        }
    }
    float nearest = 0.0f;
    for (uint32_t i = 0; i < 3; i++)
    {
        // (A range of 0 or a nonsense one is left alone.)
        const float want = ranges.own[i] > 0.0f && ranges.own[i] < 1.0e6f ? float(ranges.own[i] * scale) : ranges.own[i];
        if (want != ranges.written[i])
        {
            fields[i] = want;
            ranges.written[i] = want;
        }
        if (want > 0.0f && (nearest == 0.0f || want < nearest))
            nearest = want;
    }
    s_nearestRange.store(nearest, std::memory_order_relaxed);
    __imp__sub_820FDCD0(ctx, base);
}
