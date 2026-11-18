#pragma once
// D3D call census and probe (gpu/d3d_census.cpp). Functions the renderer hooks itself report here.

#include "ppc.h"

#include <string_view>

namespace ac6::gpu {

int CensusIndex(std::string_view name);              // -1 if unknown
void CensusNote(int index, const PPCContext& ctx);   // count (and probe) one call

} // namespace ac6::gpu
