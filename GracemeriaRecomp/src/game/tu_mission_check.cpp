// Title update 1's mission-6 objective check, carried on the base image. The title update adds a step to the mission
// manager's per-update work (sub_8226C068): in mission 6 of the campaign or Free Mission (the number as the game-setup
// object gives it, sub_820943B0), once eight of the mission's objective counters have reached their targets, it sets
// mission variable 10 itself, once per mission. The base game sets it only along some orders of completing the
// objectives, and a mission script waiting on it then waits forever (a soft-lock). The check here is the
// title update's own condition, applied through the game's mission_var_apply (sub_82267258) as the title update does.
// Its once-per-mission flag (a new mission-manager field in the title update) is kept on the host, cleared by the
// manager's reset at the start of a mission (sub_8226B2F8). AC6_TU_MISSION6=0: off.
#include "heap.h"
#include "kernel/function.h"
#include "log.h"
#include "endian.h"
#include "memory.h"
#include "ppc.h"

#include <cstdlib>
#include <cstring>

namespace {

constexpr uint32_t kSystemSingletonPtr = 0x826E4E54;   // the system singleton
constexpr uint32_t kSingletonSetup = 112;              // the game-setup object (inside the singleton)
constexpr uint32_t kSingletonMode = 120;               // 1 campaign, 2 Free Mission, 4 online
constexpr uint32_t kSetupValueA = 0x820943B0;          // the mission number the check compares with 6
constexpr uint32_t kSetupValueB = 0x820F60C0;          // a second setup value, which must not be 6
constexpr uint32_t kMissionVarApply = 0x82267258;      // (mgr, index, op*): applies an op to a mission variable
constexpr uint32_t kMgrVars = 92;                      // the mission variables: 20-byte records, value at +0

bool Enabled()
{
    static const bool on = [] { const char* v = std::getenv("AC6_TU_MISSION6"); return !(v && *v == '0'); }();
    return on;
}

uint32_t Load32(uint32_t address) { return ac6::g_memory.As<ac6::be<uint32_t>>(address)->get(); }
int32_t Var(uint32_t vars, uint32_t index) { return int32_t(Load32(vars + index * 20)); }

uint32_t s_doneFor = 0;   // the mission manager the check has fired for since its last reset (0: none)
bool s_logged = false;

void Check(uint32_t mgr)
{
    if (!Enabled() || s_doneFor == mgr)
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
