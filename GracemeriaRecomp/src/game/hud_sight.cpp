// The gun sight's range arc, centred in its ring. The sight is a ring sprite (shape 88) and an arc inside it that shows
// the target's range (0x8224E758: up to 128 points of radius 20, (sin, -cos) x radius around an origin), both placed at
// the sight's position by CAce6HudContainer::Draw. The HUD's line strip list (0x820925D0) moves every line vertex a
// pixel up and left (point + origin - 1.0): the HUD's frames are drawn for that, a pixel inside their fills' edges. The
// arc gets it too, so with the console's half-pixel offset for lines its centre sits half a pixel up and left of the
// ring's, on the console as well. Its origin is moved half a pixel right and down here, onto the ring's centre.
// AC6_GUN_SIGHT_CENTRE=0: as the game draws it.
#include "memory.h"
#include "endian.h"
#include "ppc.h"

#include <cstdlib>

namespace {

bool Enabled()
{
    static const bool on = [] { const char* v = std::getenv("AC6_GUN_SIGHT_CENTRE"); return !(v && *v == '0'); }();
    return on;
}

} // namespace

// DrawArc(r3 = HUD renderer, r4 = &origin {x, y}, f1 = radius, f2 = the arc's share, r6 = colour, r7 = layer). Its one
// caller is the gun sight.
extern "C" PPC_FUNC(__imp__sub_8224E758);
PPC_FUNC(sub_8224E758)
{
    if (!Enabled() || !ctx.r4.u32)
    {
        __imp__sub_8224E758(ctx, base);
        return;
    }
    ac6::be<float>* origin = ac6::g_memory.As<ac6::be<float>>(ctx.r4.u32);
    const float x = origin[0], y = origin[1];
    origin[0] = x + 0.5f;
    origin[1] = y + 0.5f;
    __imp__sub_8224E758(ctx, base);
    origin[0] = x;
    origin[1] = y;
}
