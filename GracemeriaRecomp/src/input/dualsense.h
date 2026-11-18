#pragma once
// DualSense adaptive triggers (input/dualsense.cpp).
struct SDL_Gamepad;

namespace ac6::input::dualsense {

// Main thread, once per input update: player 1's pad (null: none). Effects only on a DualSense, in flight, with the
// setting on and the overlay closed, cleared otherwise.
void Update(SDL_Gamepad* pad, bool overlayOpen);
// The triggers back to normal (before the game closes).
void Clear();

} // namespace ac6::input::dualsense
