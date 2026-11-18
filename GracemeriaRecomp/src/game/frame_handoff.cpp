// The frame hand-off's order. At the end of each main-loop frame, the frame end (0x82331E78 -> 0x8233B5A0) waits for
// the render worker to finish the last lists, hands it this frame's, waits for the vblank and swaps, then releases the
// worker (0x8233A610: its state 1). The main loop advances the frame slot [0x826E4DD0] (the
// index of the game's three-frame rings) a few instructions after that (0x821D7ECC). The render callbacks read the slot
// assuming it has moved on: the 2D layers bind the vertex buffer of slot - 1 (0x82138048), the slot their vertices were
// written to. On the console the main thread always wins those few instructions. On a PC the woken worker sometimes
// runs first (measured: several times a second in the menus at high frame rates) and reads the slot before it moves,
// the 2D layers then drawing this frame's draws from last frame's vertices (quads with each other's textures).
// So the main loop's release waits until the slot has moved: it is held at the frame end and done at the main loop's
// next call (0x82331DE8, right after the advance). AC6_FRAME_HANDOFF_ORDER=0 keeps the game's order.
// AC6_HANDOFF_STALL=<ms> (testing only): the main thread sleeps after the frame end, before the advance, as a host
// preemption there would: in the game's order the worker then always runs first.
#include "ppc.h"

#include <chrono>
#include <cstdlib>
#include <thread>

namespace {
bool Enabled()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_FRAME_HANDOFF_ORDER"); return !v || *v != '0'; }();
    return enabled;
}

thread_local bool t_loopFrameEnd = false;   // the main loop's frame end is running (not the boot's)
thread_local uint32_t t_heldRelease = 0;    // the request manager whose worker release is held, or 0
}

// The main loop, before its frame end (0x821D7EB4: bl 0x82331E78).
void FrameHandoffLoopEnd()
{
    t_loopFrameEnd = true;
}

extern "C" PPC_FUNC(__imp__sub_8233A610);
PPC_FUNC(sub_8233A610)
{
    if (t_loopFrameEnd && !t_heldRelease && Enabled())
    {
        t_heldRelease = ctx.r3.u32;
        return;
    }
    __imp__sub_8233A610(ctx, base);
}

extern "C" PPC_FUNC(__imp__sub_82331E78);
PPC_FUNC(sub_82331E78)
{
    __imp__sub_82331E78(ctx, base);
    static const int stallMs = [] { const char* v = std::getenv("AC6_HANDOFF_STALL"); return v ? std::atoi(v) : 0; }();
    if (stallMs > 0 && t_loopFrameEnd)
        std::this_thread::sleep_for(std::chrono::milliseconds(stallMs));
    t_loopFrameEnd = false;
}

// The main loop's next call after the slot's advance (also the frame's first anywhere else: a release still held is
// done before anything else runs).
extern "C" PPC_FUNC(__imp__sub_82331DE8);
PPC_FUNC(sub_82331DE8)
{
    if (const uint32_t manager = t_heldRelease)
    {
        t_heldRelease = 0;
        ctx.r3.u64 = manager;
        __imp__sub_8233A610(ctx, base);
    }
    __imp__sub_82331DE8(ctx, base);
}
