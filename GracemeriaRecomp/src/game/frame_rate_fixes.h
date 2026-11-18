#pragma once
// The frame-rate fixes of the game's frame-stepped code (game/frame_rate_fixes.cpp, game/cutscene_unlock.cpp).

namespace ac6::game {

// R = the game's measured frame time over the console's frame (x 30 in missions, x 60 on the 60 FPS screens), clamped
// to 0.01 to 1, on unlocked frames (gpu/frame_rate.h), exactly 1 otherwise.
double FrameStepRatio();

// The cutscene timeline's step for the current cutscene manager Exec, in timeline frames (apu/cutscene_sync.cpp sets
// it around each call, 1 outside them). Never above 1: a larger step would skip a frame's record block.
void SetCutsceneStep(double step);

// The timeline frames the sequencer advanced since the last call (the step hook's sum).
double TakeCutsceneFramesAdvanced();

} // namespace ac6::game
