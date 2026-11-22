// The keyboard and mouse's flight heartbeat (input/kbm.cpp): the player's flight model ticking means flight (the
// flight key set, mouse steering). When it stops (pause, menus, cutscenes, the front end) the menu set takes over.
// Every state has to beat, or the keys go dead somewhere:
//  - 0x823046A0: the airborne force step (the 40-slot flight-model classes are the player's, enemies step elsewhere).
//  - 0x82305278: the ground force step ([fm+332] & 0x40: landed, taxiing, the take-off roll).
//  - 0x82306620 / 0x82329488: the flight model's master update (base / derived class, vtable slot +60), which also runs
//    while parked ([fm+332] & 0x80 steps neither force step): only for the player's flight model.
//  - 0x82233208: the player actor's ground-state maintainer (its only caller is the player driver 0x82233FB0).
//  - 0x82191AE8: a timed gauge on the player's input block, not flight control: a secondary signal.
// And the hangar's display (below), where the pad moves the shown aircraft as in flight.
#include "endian.h"
#include "input/kbm.h"
#include "log.h"
#include "memory.h"
#include "ppc.h"

namespace {

uint32_t U32(uint32_t a)
{
    return a ? ac6::g_memory.As<ac6::be<uint32_t>>(a)->get() : 0;
}

// [[[ [0x826E4E54] + 0x29FC4 ] + 0x1008 ] + 4912]: the unit manager, its player actor, the actor's flight model.
uint32_t PlayerFlightModel()
{
    const uint32_t root = U32(0x826E4E54);
    const uint32_t manager = root ? U32(root + 0x29FC4) : 0;
    const uint32_t actor = manager ? U32(manager + 0x1008) : 0;
    return actor ? U32(actor + 4912) : 0;
}

} // namespace

#define AC6_FLIGHT_BEAT(addr)                                                                                          \
    extern "C" PPC_FUNC(__imp__sub_##addr);                                                                             \
    PPC_FUNC(sub_##addr)                                                                                                \
    {                                                                                                                   \
        ac6::input::kbm::NotifyFlightStep();                                                                            \
        __imp__sub_##addr(ctx, base);                                                                                   \
    }

#define AC6_FLIGHT_BEAT_PLAYER(addr)                                                                                   \
    extern "C" PPC_FUNC(__imp__sub_##addr);                                                                             \
    PPC_FUNC(sub_##addr)                                                                                                \
    {                                                                                                                   \
        if (ctx.r3.u32 && ctx.r3.u32 == PlayerFlightModel())                                                           \
            ac6::input::kbm::NotifyFlightStep();                                                                        \
        __imp__sub_##addr(ctx, base);                                                                                   \
    }

AC6_FLIGHT_BEAT(823046A0)
AC6_FLIGHT_BEAT(82305278)
AC6_FLIGHT_BEAT_PLAYER(82306620)
AC6_FLIGHT_BEAT_PLAYER(82329488)
AC6_FLIGHT_BEAT(82233208)
AC6_FLIGHT_BEAT(82191AE8)

// The hangar's display (the aircraft select's X): CSelectAircraftManager state 2 ([mgr+0x8C90], its slot 24). Slot 38
// (0x82148BC8, r3 = the manager) runs every scene update and moves the control surfaces from the pad in that state only.
extern "C" PPC_FUNC(__imp__sub_82148BC8);
PPC_FUNC(sub_82148BC8)
{
    const uint32_t state = U32(ctx.r3.u32 ? ctx.r3.u32 + 0x8C90 : 0);
    static uint32_t logged = UINT32_MAX;
    if (state != logged)
    {
        logged = state;
        ac6::log::Info("Hangar: state {}", state);
    }
    if (state == 2)
        ac6::input::kbm::NotifyHangarDisplay();
    __imp__sub_82148BC8(ctx, base);
}
