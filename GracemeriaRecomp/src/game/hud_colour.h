#pragma once
// The HUD's colours (game/hud_colour.cpp).

#include <cstdint>
#include <string_view>

namespace ac6::game {

constexpr uint32_t kHudAuto = 0xFFFFFFFFu;   // not chosen
constexpr uint32_t kGameHud = 0x00FF20, kGameFriendly = 0x0080FF, kGameEnemy = 0xFF2000;

struct HudPalette
{
    uint32_t hud, friendly, enemy;   // 00RRGGBB
};

// Whether two colours would be hard to tell apart on the HUD.
bool HudColoursClose(uint32_t a, uint32_t b);
// What's drawn for these choices (each RRGGBB or kHudAuto): friendly and enemy on Auto keep the game's colours, unless
// that's too close to the HUD's (or, for the enemy, the friendly colour): then the first of a short list that isn't.
HudPalette ResolveHudPalette(uint32_t hud, uint32_t friendly, uint32_t enemy);

// The colourblind options of hud_custom (protanopia, deuteranopia, tritanopia): their colours, false for off and custom.
bool HudPreset(std::string_view mode, HudPalette& palette);

} // namespace ac6::game
