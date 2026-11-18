#pragma once
// The frame's 2D layers (game/ui_layers.cpp).

namespace ac6::game {

// True while the game draws its 2D layers (the UI) on this thread.
bool InUiLayers();
// Around a 2D layer's recorded commands (0x82138758 to 0x82138260).
void EnterUiLayers();
void LeaveUiLayers();

} // namespace ac6::game
