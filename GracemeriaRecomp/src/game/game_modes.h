#pragma once
// The game's mode tasks (game/game_modes.cpp).
#include <cstdint>
#include <string_view>

namespace ac6::game {

// Wraps the mode factories' function-table entries (after the table is filled).
void InstallModeHooks();
// The mode task the game is in ("CModeTaskGame", ..., "" before the first), and the frame it was entered.
const char* CurrentMode();
uint64_t ModeEnteredFrame();
// AC6_FAST_BOOT=1 (dev): the boot logos, before the first mode, run as fast as the machine can (vblank, no vsync).
bool FastBoot();
// Whether the game is in that mode (its class name, with or without "CModeTask").
bool ModeIs(std::string_view name);

} // namespace ac6::game
