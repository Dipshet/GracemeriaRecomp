#pragma once
// The cockpit view's back (game/cockpit.cpp).

namespace ac6::game {

// The part group of the aircraft model whose draw request this thread is executing in the cockpit view while the
// exterior body is drawn there (AC6_COCKPIT_BODY, or a view wider than the game's): 0 the exterior body, 1-3 the
// cockpit's, or -1.
int CockpitDrawGroup();

} // namespace ac6::game
