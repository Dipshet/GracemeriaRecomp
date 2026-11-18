// The exterior body behind the cockpit view. The aircraft model is ten part groups (the group's own model at
// +144 + 4i, its flag at +259 + i): the camera mode's setter (0x82226CF0 -> 0x822239A8) shows the exterior body
// (group 0) in chase view, the cockpit's three groups (1-3: the interior, the cockpit view's own wings and stores, ...)
// in cockpit view, where the model is drawn camera-relative with its own projection: the player's object job (around
// 0x8209E8B0) submits groups 1, 2 and 3 one by one through 0x822247F0 (model, transform, group), with their render
// states. The cockpit model only covers what the game's 16:9 view at its FOV can see: wider (AC6_WIDESCREEN,
// AC6_FOV) the sky shows through past the seat.
//
// With a wider view than the game's (a positive AC6_FOV_OFFSET, or the auto aspect ratio on an output wider than 16:9,
// AC6_COCKPIT_BODY=0 / 1 forces it) the exterior body is submitted there too, with group 3's transform and states.
// Drawn whole it would cover the cockpit (its own simplified cockpit, canopy and coaming, the wings drawn twice), so
// its draws are told apart on the render thread and filtered (gpu/renderer.cpp's CockpitDraw): the draw requests made
// while a group is submitted in the cockpit view (RequestNud20DrawRigid, constructor 0x82363A30) are tagged with it,
// and while one executes (0x82364038, which issues the draw) its group is known. The cockpit groups' draws record their
// model-space boxes. The body's draws keep only the triangles not wholly inside one of them: the spine, the rear
// fuselage, the fin and the tailplanes.
#include "game/cockpit.h"
#include "game/widescreen.h"
#include "gpu/video.h"

#include "endian.h"
#include "log.h"
#include "memory.h"
#include "ppc.h"

#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace ac6::game {

namespace {

bool CockpitBody()
{
    if (const char* v = std::getenv("AC6_COCKPIT_BODY"); v && *v)
        return *v == '1';
    // Wider than the game's view: a positive field of view offset (live), or the auto aspect ratio on a wider output
    // than 16:9.
    return FovDegrees() > 46.0 || gpu::g_video.WideFactor() > 1.0f;
}

thread_local int t_submitGroup = -1;    // the group being submitted in the cockpit view (the game's job threads)
thread_local int t_executeGroup = -1;   // the group of the request executing (the render thread)
std::mutex s_mutex;
std::unordered_map<uint32_t, std::pair<int, uint32_t>> s_requests;   // request -> group, submission generation
uint32_t s_generation = 0;               // one per cockpit submission of group 3 (about a frame)

} // namespace

int CockpitDrawGroup()
{
    return t_executeGroup;
}

} // namespace ac6::game

using namespace ac6;

extern "C" PPC_FUNC(__imp__sub_822247F0);
PPC_FUNC(sub_822247F0)
{
    const uint32_t model = ctx.r3.u32, transform = ctx.r4.u32, group = ctx.r5.u32;
    // A group of an aircraft model (+184 = 1) in the cockpit display mode (+188 = 2).
    const bool cockpit = game::CockpitBody() && group <= 9 && g_memory.As<be<uint32_t>>(model + 188)->get() == 2 &&
        g_memory.As<be<uint32_t>>(model + 184)->get() == 1;
    if (!cockpit)
    {
        __imp__sub_822247F0(ctx, base);
        return;
    }
    // The exterior body first, before group 3.
    if (group == 3 && g_memory.As<be<uint32_t>>(model + 144)->get())
    {
        {
            std::lock_guard lock(game::s_mutex);
            game::s_generation++;
        }
        uint8_t& flag = *g_memory.As<uint8_t>(model + 259);
        const uint8_t shown = flag;
        flag = 1;
        ctx.r5.u64 = 0;
        game::t_submitGroup = 0;
        __imp__sub_822247F0(ctx, base);
        flag = shown;
        ctx.r3.u64 = model;
        ctx.r4.u64 = transform;
        ctx.r5.u64 = group;
        static bool logged = false;
        if (!logged)
            log::Info("Cockpit view: the exterior body behind the cockpit drawn too (AC6_COCKPIT_BODY)");
        logged = true;
    }
    game::t_submitGroup = int(group);
    __imp__sub_822247F0(ctx, base);
    game::t_submitGroup = -1;
}

namespace {

// Requests made while a cockpit group is submitted get its tag.
void TagRequest(uint32_t request)
{
    if (game::t_submitGroup < 0 || !request)
        return;
    std::lock_guard lock(game::s_mutex);
    game::s_requests[request] = { game::t_submitGroup, game::s_generation };
    if (game::s_requests.size() > 4096)
        std::erase_if(game::s_requests, [](const auto& e) { return game::s_generation - e.second.second > 4; });
}

// While a request executes (issuing its draw), its group.
int RequestGroup(uint32_t request)
{
    if (!game::CockpitBody())
        return -1;
    std::lock_guard lock(game::s_mutex);
    if (auto it = game::s_requests.find(request); it != game::s_requests.end() && game::s_generation - it->second.second <= 2)
        return it->second.first;
    return -1;
}

} // namespace

// The model draw requests' constructors and Execute (vtable slot 5): Nud10 / Nud20, rigid / skinned (the aircraft's body
// is skinned: its control surfaces move).
#define AC6_REQUEST_HOOKS(CTOR, EXECUTE)                      extern "C" PPC_FUNC(__imp__sub_##CTOR);                   PPC_FUNC(sub_##CTOR)                                      {                                                             __imp__sub_##CTOR(ctx, base);                             TagRequest(ctx.r3.u32);                               }                                                         extern "C" PPC_FUNC(__imp__sub_##EXECUTE);                PPC_FUNC(sub_##EXECUTE)                                   {                                                             game::t_executeGroup = RequestGroup(ctx.r3.u32);          __imp__sub_##EXECUTE(ctx, base);                          game::t_executeGroup = -1;                            }

AC6_REQUEST_HOOKS(82362E78, 82362E38)   // RequestNud10DrawRigid
AC6_REQUEST_HOOKS(823639E8, 823639B8)   // RequestNud10DrawSkin
AC6_REQUEST_HOOKS(82363A30, 82364038)   // RequestNud20DrawRigid
AC6_REQUEST_HOOKS(82363A78, 823644A0)   // RequestNud20DrawSkin
