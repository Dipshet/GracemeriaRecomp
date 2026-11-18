#include "gpu/vblank.h"
#include "gpu/renderer.h"

#include "cpu/guest_thread.h"
#include "game/game_modes.h"
#include "kernel/dispatcher.h"
#include "kernel/function.h"
#include "log.h"
#include "memory.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <chrono>

namespace ac6::gpu {

namespace {

// D1MODE VBlank status register: the D3D interrupt handler only treats source 0 as a VBlank if
// bit 0 is set here.
constexpr uint32_t kVBlankStatusRegister = 0x7FC86544;
// AC6_VBLANK_HZ=<rate>[@<frame>]: a faster interrupt lets the game's frame loop run past 60 FPS, for
// measuring how fast a frame can be (the game's timing then runs fast too), optionally only from a frame
// on: the menus' timing (and the frame-based test macros) need the normal 60 Hz.
struct FastVBlank { std::chrono::nanoseconds period{ 1'000'000'000 / 60 }; uint64_t fromFrame = 0; };
// Read on first use: the settings file (settings.cpp) sets the variables at the start of main.
const FastVBlank& FastVBlankSetting()
{
    static const FastVBlank value = [] {
        FastVBlank f;
        if (const char* v = std::getenv("AC6_VBLANK_HZ"))
        {
            f.period = std::chrono::nanoseconds(1'000'000'000 / std::max(30, std::atoi(v)));
            if (const char* at = std::strchr(v, '@'))
                f.fromFrame = std::strtoull(at + 1, nullptr, 10);
        }
        return f;
    }();
    return value;
}
constexpr auto kPeriod60 = std::chrono::nanoseconds(1'000'000'000 / 60);

std::atomic<uint32_t> s_callback{ 0 };
std::atomic<uint32_t> s_data{ 0 };
std::atomic<uint64_t> s_count{ 0 };
std::atomic<bool> s_started{ false };

void VBlankLoop(void*)
{
    using Clock = std::chrono::steady_clock;
    auto next = Clock::now();
    bool fast = false, boot = false;
    for (;;)
    {
        const bool nowFast = FastVBlankSetting().period != kPeriod60 && RendererFrame() >= FastVBlankSetting().fromFrame;
        if (nowFast && !fast)
            log::Info("VBlank: {:.0f} Hz from frame {} (AC6_VBLANK_HZ)", 1e9 / double(FastVBlankSetting().period.count()), RendererFrame());   // scripts parse this line
        fast = nowFast;
        // AC6_FAST_BOOT=1: 1 kHz through the boot logos (game/game_modes.cpp).
        const bool nowBoot = game::FastBoot();
        if (nowBoot != boot)
            log::Info("VBlank: {} (AC6_FAST_BOOT, frame {})", nowBoot ? "1 kHz through the boot logos" : "boot logos done, back to normal", RendererFrame());
        boot = nowBoot;
        next += boot ? std::chrono::nanoseconds(1'000'000) : nowFast ? FastVBlankSetting().period : kPeriod60;
        auto remaining = std::chrono::duration_cast<std::chrono::nanoseconds>(next - Clock::now()).count();
        if (remaining > 0)
            kernel::DelayExecution({ false, -(remaining / 100) });
        else
            next = Clock::now();   // fell behind (debugger, load spike): don't burst

        uint32_t callback = s_callback.load(std::memory_order_acquire);
        if (!callback)
            continue;
        auto* status = g_memory.As<be<uint32_t>>(kVBlankStatusRegister);
        *status = *status | 1u;
        s_count.fetch_add(1, std::memory_order_relaxed);
        GuestCall(callback, 0u /* source: VBlank */, s_data.load(std::memory_order_relaxed));
    }
}

} // namespace

void SetInterruptCallback(uint32_t callback, uint32_t data)
{
    s_data = data;
    s_callback.store(callback, std::memory_order_release);
    log::Info("Graphics interrupt callback {:08X}({:08X})", callback, data);

    if (callback && !s_started.exchange(true))
    {
        GuestThreadParams params;
        params.hostEntry = VBlankLoop;
        params.flags = 1u << 24;   // hardware thread 0
        if (GuestThread* thread = GuestThread::Start(params))
        {
            thread->SetPriority(15);
            thread->Resume();
        }
    }
}

uint64_t VBlankCount() { return s_count.load(std::memory_order_relaxed); }

} // namespace ac6::gpu
