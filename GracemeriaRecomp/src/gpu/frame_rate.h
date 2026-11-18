#pragma once
// The unlocked frame rate. The game asks for 2 vblanks a frame (30 FPS) in missions and in-engine cutscenes and 1
// (60 FPS) in menus, the briefing and the hangar. Mission frames run at AC6_FRAME_RATE instead: the display's refresh
// rate by default, a cap of at least 60, or 0 for uncapped. The game's frame loop then skips its vblank waits and the
// present paces the frame (the cap's pace, or vsync/VRR when uncapped). Cutscenes (AC6_CUTSCENE_UNLOCK) and the 60 FPS
// screens (AC6_MENU_UNLOCK) unlock on their own switches, both on by default (off: the console's timing).
// AC6_FRAME_RATE=console keeps the console's timing everywhere.

#include <cstdint>

namespace ac6::gpu {

// The missions' frame rate: the display's refresh rate unless AC6_FRAME_RATE says otherwise (a cap of at least 60, held
// to the refresh rate with vsync on), 30 with "console", 0 = uncapped.
uint32_t FrameRateTarget();

// AC6_CUTSCENE_UNLOCK (on by default, 0 off): in-engine cutscenes run unlocked too (game/cutscene_unlock.cpp).
// Otherwise at 30.
bool CutsceneUnlock();

// AC6_MENU_UNLOCK (on by default, 0 off): the 60 FPS screens (menus, briefing, hangar) run unlocked too unless
// AC6_FRAME_RATE is "console" (at 60 their pace stays the console's).
bool MenuUnlock();

// At each swap, with the game's present interval in vblanks: decides whether the frame runs unlocked and
// returns the vblanks to pace it with (0 = unpaced).
uint32_t NoteSwap(uint32_t vblanks);

// The frame being built runs unlocked.
bool Unlocked();

// A frame was handed to the display (AC6_FRAME_TIMES).
void NotePresent();

// A cutscene manager ran this frame (apu/cutscene_sync.cpp).
void NoteCutsceneFrame();
// Whether an in-engine cutscene is playing (it noted a frame within the last few).
bool InCutscene();

} // namespace ac6::gpu
