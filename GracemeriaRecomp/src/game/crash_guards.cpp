// Crash guards: places where the game's own code would crash, each skipping or cleaning up what it can't handle. Each
// guard logs once when it acts.
#include "endian.h"
#include "log.h"
#include "memory.h"
#include "ppc.h"

#include <atomic>
#include <cstdint>

// ---- The damage report and a unit that is gone ------------------------------------------------------------------

// The mission manager's damage report (sub_82269248, reached from unit_apply_damage sub_8229BC98 through
// sub_82258068) looks up the two units the damage record names: sub_8224C250 turns a unit reference (group, slot) into
// the unit's object, or 0 when that slot is empty, the unit removed. Online, the report is skipped when either unit
// is gone. Offline the game skips that check and reads [unit+292] at once: a crash reading guest address 0x124 when
// the damage names a unit that was removed meanwhile. Here offline takes the online path's skip too, to the report's
// exit (0x822695DC).
bool DamageReportUnitGone(PPCRegister& r26, PPCRegister& r27)
{
    if (r26.u32 != 0 && r27.u32 != 0)
        return false;
    static std::atomic<bool> logged{ false };
    if (!logged.exchange(true))
        ac6::log::Info("Damage report skipped: the damage names a unit that is gone (the game would have crashed here)");
    return true;
}

// ---- The HUD's text in a European language ---------------------------------------------------------------------

namespace {

// The game's language index (root + 0x24474, root = [0x826E4E54]): 0 Japanese, 1 English, 2 to 6 the European languages
// (game/language.cpp sets them).
bool EuropeanLanguage()
{
    const uint32_t root = ac6::g_memory.As<ac6::be<uint32_t>>(0x826E4E54)->get();
    if (!root)
        return false;
    const uint32_t index = ac6::g_memory.As<ac6::be<uint32_t>>(root + 0x24474)->get();
    return index >= 2 && index <= 6;
}

} // namespace

// Mid-asm hooks where the HUD's text measurer (0x8224E1F0, at 0x8224E1F4) and drawer (0x8224E878, at 0x8224E88C) start
// reading their UTF-16 string (r4). They look each character up in the font's glyph table, one entry per code below 512,
// and read past it for a higher code. On this disc a European language's HUD can build such a string 
// (Italian, on the autopilot in mission 02: a code 0x4342 before "EMPTY"), which crashed the game.
// With a European language, characters past the table are taken out of the string first, as the HUD
// skips a letter its font lacks.
void HudTextDrawable(PPCRegister& r4)
{
    if (!r4.u32 || !EuropeanLanguage())
        return;
    uint32_t to = r4.u32;
    bool changed = false;
    for (uint32_t from = r4.u32;; from += 2)
    {
        const uint16_t c = *ac6::g_memory.As<ac6::be<uint16_t>>(from);
        if (c >= 512)
        {
            changed = true;
            continue;
        }
        if (changed)
            *ac6::g_memory.As<ac6::be<uint16_t>>(to) = c;
        to += 2;
        if (!c)
            break;
    }
    static std::atomic<bool> logged{ false };
    if (changed && !logged.exchange(true))
        ac6::log::Info("HUD text: characters its fonts don't have taken out (a European language's HUD message)");
}
