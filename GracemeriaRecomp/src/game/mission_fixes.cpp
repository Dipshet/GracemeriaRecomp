// Fixes to the game's missions, each with its own switch.
//
// Mission 6's objective check (title update 1's, carried on the base image). The title update adds a step to the
// mission manager's per-update work (sub_8226C068): in mission 6 of the campaign or Free Mission (the number as the
// game-setup object gives it, sub_820943B0), once eight of the mission's objective counters have reached their targets,
// it sets mission variable 10 itself, once per mission. The base game sets it only along some orders of completing the
// objectives, and a mission script waiting on it then waits forever (a soft-lock). The check here is the
// title update's own condition, applied through the game's mission_var_apply (sub_82267258) as the title update does.
// Its once-per-mission flag (a new mission-manager field in the title update) is kept on the host, cleared by the
// manager's reset at the start of a mission (sub_8226B2F8). AC6_TU_MISSION6=0: off.
//
// Mission 11's F-22A squad in objective D, which the gun can barely hit in the original game. Hits are found in two steps
// (0x8229A760 both times). Each tick a gun round's path is tested against a sphere round every unit, of the unit's hit
// radius ([unit+360]); only a unit whose sphere it crosses goes to the round's hit handler, which tests the round
// against the aircraft's model itself (the model's own radius, then a raycast of its mesh) before any damage. The hit
// radius comes from the unit's 112-byte mission record when the record's byte +81 is 1: its float +0, copied when the
// unit is set up (0x82299430). Enemy fighters' records carry 24 (the F-22A's type, 408, has 24 in missions 13, 14 and
// 15). The six F-22As of mission 11 carry 1.0, the only such value among the game's aircraft, so a round reached the
// model test only when it passed within a metre of an F-22A's centre. Here such a unit gets the 24 the type has
// everywhere else; the model test still decides a hit. (A missile's blast is tested against the same sphere, its blast
// radius added: missiles hit those six a little closer in than other fighters.) AC6_F22_HIT_RADIUS=0: as the game
// does it.
#include "heap.h"
#include "kernel/function.h"
#include "log.h"
#include "endian.h"
#include "memory.h"
#include "ppc.h"

#include <cstdlib>
#include <cstring>

namespace {

bool On(const char* name)
{
    const char* v = std::getenv(name);
    return !(v && *v == '0');
}

uint32_t Load32(uint32_t address) { return ac6::g_memory.As<ac6::be<uint32_t>>(address)->get(); }

// --- Mission 6's objective check ---

constexpr uint32_t kSystemSingletonPtr = 0x826E4E54;   // the system singleton
constexpr uint32_t kSingletonSetup = 112;              // the game-setup object (inside the singleton)
constexpr uint32_t kSingletonMode = 120;               // 1 campaign, 2 Free Mission, 4 online
constexpr uint32_t kSetupValueA = 0x820943B0;          // the mission number the check compares with 6
constexpr uint32_t kSetupValueB = 0x820F60C0;          // a second setup value, which must not be 6
constexpr uint32_t kMissionVarApply = 0x82267258;      // (mgr, index, op*): applies an op to a mission variable
constexpr uint32_t kMgrVars = 92;                      // the mission variables: 20-byte records, value at +0

int32_t Var(uint32_t vars, uint32_t index) { return int32_t(Load32(vars + index * 20)); }

uint32_t s_doneFor = 0;   // the mission manager the check has fired for since its last reset (0: none)
bool s_logged = false;

void Check(uint32_t mgr)
{
    static const bool enabled = On("AC6_TU_MISSION6");
    if (!enabled || s_doneFor == mgr)
        return;
    const uint32_t singleton = Load32(kSystemSingletonPtr);
    if (!singleton)
        return;
    const uint32_t mode = Load32(singleton + kSingletonMode);
    if (mode != 1 && mode != 2)
        return;
    const uint32_t a = ac6::GuestCall<uint32_t>(kSetupValueA, singleton + kSingletonSetup);
    const uint32_t b = ac6::GuestCall<uint32_t>(kSetupValueB, singleton + kSingletonSetup);
    if (!s_logged)
    {
        s_logged = true;
        ac6::log::Info("Mission check (title update): mission number {} / {} (mode {}); the check is for 6", a, b, mode);
    }
    if (a != 6 || b == 6)
        return;
    const uint32_t vars = Load32(mgr + kMgrVars);
    if (!vars)
        return;
    if (Var(vars, 14) == 1 && Var(vars, 34) == 1 && Var(vars, 44) == 1 && Var(vars, 183) >= 3 && Var(vars, 73) >= 2 &&
        Var(vars, 99) >= 1 && Var(vars, 130) >= 4 && Var(vars, 121) >= 1)
    {
        // The op record: value (int16) at +0, the two operand indices at +4 / +6 (none), the op at +8 (0: set).
        static const uint32_t op = ac6::g_memory.MapVirtual(ac6::g_heap.Alloc(16));
        auto* p = static_cast<uint8_t*>(ac6::g_memory.Translate(op));
        std::memset(p, 0, 16);
        p[1] = 1;                 // value 1 (big-endian int16)
        p[4] = p[5] = 0xFF;       // no operand
        p[6] = p[7] = 0xFF;
        p[8] = 0;                 // set
        const int32_t before = Var(vars, 10);
        ac6::GuestCall(kMissionVarApply, mgr, 10u, op);
        s_doneFor = mgr;
        ac6::log::Info("Mission check (title update): objectives complete, mission variable 10 set ({} -> {})", before,
            Var(vars, 10));
    }
}

// --- Mission 11's F-22A hit radius ---

constexpr uint16_t kF22Type = 408;
constexpr float kFighterRadius = 24.0f;

void FixHitRadius(uint32_t unit)
{
    static const bool enabled = On("AC6_F22_HIT_RADIUS");
    if (!enabled || !unit)
        return;
    const uint32_t params = Load32(unit + 384);
    const uint32_t record = params ? Load32(params) : 0;
    if (!record || *ac6::g_memory.As<uint8_t>(record + 81) != 1 ||
        ac6::g_memory.As<ac6::be<uint16_t>>(record + 72)->get() != kF22Type)
        return;
    ac6::be<float>& radius = *ac6::g_memory.As<ac6::be<float>>(unit + 360);
    if (ac6::g_memory.As<ac6::be<float>>(record)->get() == 1.0f && radius.get() == 1.0f)
        radius = kFighterRadius;
}

} // namespace

extern "C" PPC_FUNC(__imp__sub_8226C068);   // the mission manager's per-update step (r3 = the manager)
PPC_FUNC(sub_8226C068)
{
    const uint32_t mgr = ctx.r3.u32;
    __imp__sub_8226C068(ctx, base);
    Check(mgr);
}

extern "C" PPC_FUNC(__imp__sub_8226B2F8);   // the mission manager's reset (vtable slot 13), at the start of a mission
PPC_FUNC(sub_8226B2F8)
{
    if (s_doneFor == ctx.r3.u32)
        s_doneFor = 0;
    s_logged = false;
    __imp__sub_8226B2F8(ctx, base);
}

extern "C" PPC_FUNC(__imp__sub_82299430);   // a unit's setup from its mission record (r3 = the unit)
PPC_FUNC(sub_82299430)
{
    const uint32_t unit = ctx.r3.u32;
    __imp__sub_82299430(ctx, base);
    FixHitRadius(unit);
}
