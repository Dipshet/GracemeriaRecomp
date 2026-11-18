// The frame's 2D layers: 0x821D0D78 draws them every frame after the 3D scene and its post-processing (the HUD, the
// menus and the game's Flash (swg) screens, the picture-in-picture windows, the fades). Draws issued inside it are the
// UI's: the HDR output keeps them at paper white. The draws are marked where the layers' recorded commands run (below).
#include "game/ui_layers.h"

#include "endian.h"
#include "log.h"
#include "memory.h"
#include "ppc.h"

#include <atomic>
#include <cstdlib>

namespace ac6::game {

namespace {
thread_local int t_layers = 0;   // set by a layer's first command, cleared by its last
}

bool InUiLayers()
{
    return t_layers > 0;
}

void EnterUiLayers()
{
    t_layers = 1;
}

void LeaveUiLayers()
{
    t_layers = 0;
}

} // namespace ac6::game

// Draw2DLayers (0x821383D0) only records each layer's draw as commands, which run later on the render thread: the
// layer's viewport (0x82138758, the first) to its render-state restore (0x82138260, the last). The draws in between
// are the UI's.
extern "C" PPC_FUNC(__imp__sub_82138758);
PPC_FUNC(sub_82138758)
{
    ac6::game::EnterUiLayers();
    __imp__sub_82138758(ctx, base);
}

extern "C" PPC_FUNC(__imp__sub_82138260);
PPC_FUNC(sub_82138260)
{
    __imp__sub_82138260(ctx, base);
    ac6::game::LeaveUiLayers();
}

// Bink FMV (the intro, cutscene movies, the main menu's background): CNuMoviePlayer queues its frame's draw
// (0x821D94A8 -> 0x821D9140, one quad over the viewport), run later on the render thread. Not a 2D layer, but drawn as
// the UI is: at the frame's size when the 3D isn't (AC6_RENDER_SCALE).
extern "C" PPC_FUNC(__imp__sub_821D94A8);
PPC_FUNC(sub_821D94A8)
{
    ac6::game::EnterUiLayers();
    __imp__sub_821D94A8(ctx, base);
    ac6::game::LeaveUiLayers();
}

// The 2D vertices' frame slot. The 2D layers' vertices are written on the main thread into one of three buffers,
// picked by the frame slot [0x826E4DD0] (0x82138D58, 0x82138F00, 0x82139368). The layers' first command (0x82138048),
// run later on the render thread, binds the slot before the current one, (slot + 2) mod 3: it assumes the
// main loop has advanced the slot by then (0x821D7ECC, right after the frame's hand-off 0x82331E78). When the render
// thread gets there first, it binds the previous frame's buffer and draws this frame's draws, at their offsets, from
// last frame's vertices: whenever the menu's layout moves (a text growing by a glyph in the swg screens' scramble),
// every later quad gets another one's texture (the Garuda emblem flashing as a white box at 240 FPS).
// So Draw2DLayers passes the slot it wrote to as that command's argument (the game's is 0, unused) and the command
// binds it.
// AC6_UI_SLOT_FIX=0 keeps the game's guess.
namespace {
constexpr uint32_t kFrameSlot = 0x826E4DD0;
constexpr uint32_t kSlotTag = 0x2D5A0000;   // the argument: tag | the slot written
std::atomic<uint32_t> s_slotMisses{ 0 };
}

// Draw2DLayers (0x82138448: li r4,0), queueing the layers' first command.
void UiLayersSlotArg(PPCRegister& r4)
{
    r4.u64 = kSlotTag | (ac6::g_memory.As<ac6::be<uint32_t>>(kFrameSlot)->get() & 3);
}

// The layers' first command (0x82138078: r11 = the slot now, r4 = its argument): the slot it binds is r11 + 2 mod 3.
void UiLayersSlotUse(PPCRegister& r4, PPCRegister& r11)
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_UI_SLOT_FIX"); return !v || *v != '0'; }();
    if ((r4.u32 & 0xFFFF0000u) != kSlotTag)
        return;
    const uint32_t want = ((r4.u32 & 3) + 1) % 3;
    if (r11.u32 == want)
        return;
    const uint32_t n = ++s_slotMisses;
    if (n <= 8 || n % 1000 == 0)
        ac6::log::Info("2D layers: the render thread ran ahead of the frame slot ({} -> {}; {} time(s)){}", r11.u32, want, n,
            enabled ? "" : ", left as the game has it");
    if (enabled)
        r11.u64 = want;
}
