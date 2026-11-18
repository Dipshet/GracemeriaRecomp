// Where the map's objects begin (draw_distance.cpp).
#pragma once

namespace ac6::game {

// The nearest of the map's three part ranges as the last draw of the map had them (world units, the level of detail
// scale applied): the distance within which its smallest buildings are drawn. 0 before a map has drawn (menus,
// frame capture replays).
float MapPartsNearestRange();

}
