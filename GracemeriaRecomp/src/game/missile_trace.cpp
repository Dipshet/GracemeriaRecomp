// Diagnostic (AC6_MISSILE_TRACE=<calls>): CAce6Missile::steer (0x822D1718, r3 = guidance component, r4 = &target
// point, r5 = &target velocity, f1 = dt): the lead it adds is r5's vector x the intercept time. Logs every call's
// distance to the target point (MSLD) and every 8th call's velocity, speed, lead and gimbal (MSLTRACE).
#include "ppc.h"
#include "log.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {

int Limit()
{
    static const int limit = [] { const char* v = std::getenv("AC6_MISSILE_TRACE"); return v ? std::atoi(v) : 0; }();
    return limit;
}

std::atomic<int> s_calls{ 0 };

float F(uint8_t* base, uint32_t a)
{
    uint32_t v;
    std::memcpy(&v, base + a, 4);
    v = __builtin_bswap32(v);
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}

} // namespace

extern "C" PPC_FUNC(__imp__sub_822D1718);
PPC_FUNC(sub_822D1718)
{
    if (Limit())
    {
        const int n = s_calls.fetch_add(1);
        // Every call: the distance to the target point and the heading's cosine to it (closest approach per missile).
        if (n < Limit() * 16)
        {
            const uint32_t m = ctx.r3.u32, t = ctx.r4.u32, arms = __builtin_bswap32(*reinterpret_cast<uint32_t*>(base + m + 80));
            const float dx = F(base, t) - F(base, arms + 80), dy = F(base, t + 4) - F(base, arms + 84), dz = F(base, t + 8) - F(base, arms + 88);
            ac6::log::Info("MSLD {:08X} {:.5f} {:.2f} {:.4f} {}", m, ctx.f1.f64, std::sqrt(dx * dx + dy * dy + dz * dz), F(base, m + 132),
                int(base[m + 159] & 1));
        }
        if (n < Limit() * 8 && n % 8 == 0)
        {
            const uint32_t m = ctx.r3.u32, t = ctx.r4.u32, v = ctx.r5.u32;
            const float vx = F(base, v), vy = F(base, v + 4), vz = F(base, v + 8);
            ac6::log::Info("MSLTRACE dt {:.5f} m {:08X} vel {:.3f} {:.3f} {:.3f} |{:.3f}| speed {:.2f} lead {} {} rate {:.4f} gimbal {:.4f} "
                           "target {:.1f} {:.1f} {:.1f}",
                ctx.f1.f64, m, vx, vy, vz, std::sqrt(vx * vx + vy * vy + vz * vz), F(base, m + 92), int(base[m + 160]),
                int(base[m + 161]), F(base, m + 104), F(base, m + 124), F(base, t), F(base, t + 4), F(base, t + 8));
        }
    }
    __imp__sub_822D1718(ctx, base);
}
