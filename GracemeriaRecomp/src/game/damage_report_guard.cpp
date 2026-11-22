// The mission manager's damage report (sub_82269248, reached from unit_apply_damage sub_8229BC98 through
// sub_82258068) looks up the two units the damage record names: sub_8224C250 turns a unit reference (group, slot) into
// the unit's object, or 0 when that slot is empty, the unit removed. Online, the report is skipped when either unit
// is gone. Offline the game skips that check and reads [unit+292] at once: a crash reading guest address 0x124 when
// the damage names a unit that was removed meanwhile. Here offline takes the online path's skip too, to the report's
// exit (0x822695DC).
#include "log.h"
#include "ppc.h"

#include <atomic>

bool DamageReportUnitGone(PPCRegister& r26, PPCRegister& r27)
{
    if (r26.u32 != 0 && r27.u32 != 0)
        return false;
    static std::atomic<bool> logged{ false };
    if (!logged.exchange(true))
        ac6::log::Info("Damage report skipped: the damage names a unit that is gone (the game would have crashed here)");
    return true;
}
