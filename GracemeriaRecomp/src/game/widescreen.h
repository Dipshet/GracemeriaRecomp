#pragma once
// The view options of game/widescreen.cpp read elsewhere.

namespace ac6::game {

// The gameplay camera's vertical field of view in first person, in degrees: the game's 46 plus AC6_FOV_OFFSET
// (0 to 40), or AC6_FOV's absolute degrees (the legacy variable, 10 to 150, which wins when set). The other gameplay
// cameras follow in proportion (tan of the half angle scaled alike).
double FovDegrees();

} // namespace ac6::game
