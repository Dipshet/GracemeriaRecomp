#include "gpu/frame_rate.h"
#include "gpu/renderer.h"
#include "gpu/video.h"

#include "endian.h"
#include "kernel/function.h"
#include "log.h"
#include "memory.h"
#include "settings.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <format>
#include <string>
#include <thread>

namespace ac6::gpu {

namespace {

constexpr uint32_t kConsoleRate = 30;
// Frames after a cutscene manager's last call that still count as the cutscene (its Exec runs every frame
// while it exists).
constexpr uint64_t kCutsceneGapFrames = 2;

// Read on first use (the settings file sets the variables at the start of main), and again after a live change (the
// settings overlay: settings::Generation).
template<typename T, typename Read> T Live(std::atomic<uint64_t>& generation, std::atomic<T>& value, Read read)
{
    const uint64_t now = settings::Generation();
    if (generation.load(std::memory_order_acquire) != now)
    {
        value.store(read(), std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    return value.load(std::memory_order_relaxed);
}

constexpr uint32_t kMatchDisplay = UINT32_MAX;

uint32_t TargetRate()
{
    // AC6_FRAME_RATE: a cap of at least 60 (1 to 59 count as 60), 0 for uncapped, "console" for the console's rates.
    // Unset or "display": the display's refresh rate, read at each call so it follows a display mode or refresh rate
    // changed while running.
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<uint32_t> value{ kMatchDisplay };
    const uint32_t rate = Live(generation, value, [] {
        const std::string v = settings::Get("AC6_FRAME_RATE");
        if (v.empty() || v == "display")
            return kMatchDisplay;
        if (v == "console")
            return kConsoleRate;
        const uint32_t r = uint32_t(std::strtoul(v.c_str(), nullptr, 10));
        if (r != 0 && r < 60)
        {
            log::Info("Frame rate: AC6_FRAME_RATE={} is below the minimum: 60", v);
            return 60u;
        }
        return r;
    });
    const uint32_t display = std::max<uint32_t>(60, uint32_t(std::lround(g_video.refreshHz)));
    if (rate == kMatchDisplay)
        return display;
    // With VSync on, a cap above the display's refresh rate is the refresh rate: otherwise the pace and the
    // VSync fight, the pace asking for a frame every 1/cap and the swap chain taking one a refresh, so the frames run
    // 26 ms / 7 ms alternately at a 120 cap on a 60 Hz display, and the chase camera, which
    // follows by an averaged frame time, swings against the world every frame. Uncapped (0) is left to the VSync alone.
    static std::atomic<uint64_t> vsyncGeneration{ UINT64_MAX };
    static std::atomic<bool> vsync{ true };
    const bool vsyncOn = Live(vsyncGeneration, vsync, [] { return settings::Get("AC6_VSYNC") != "0"; });
    if (vsyncOn && rate != kConsoleRate && rate > display)
    {
        static std::atomic<uint64_t> noted{ 0 };
        if (const uint64_t key = (uint64_t(rate) << 32) | display; noted.exchange(key, std::memory_order_relaxed) != key)
            log::Info("Frame rate: the cap of {} FPS is above the display's {} Hz with VSync on: paced at {} FPS", rate,
                display, display);
        return display;
    }
    return rate;
}

std::atomic<bool> s_unlocked{ false };
std::atomic<uint64_t> s_cutsceneFrame{ UINT64_MAX };

// AC6_FRAME_TIMES=1: every 2 s, the spread of the presents' spacing and of the game's measured frame time (CNuTimer's
// tick delta [0x826EDDA0] times seconds per tick [0x826EDDB0]).
// Read on first use: the settings file (settings.cpp) sets the variables at the start of main.
const bool& FrameTimesOn()
{
    static const bool value = [] { const char* v = std::getenv("AC6_FRAME_TIMES"); return v && *v == '1'; }();
    return value;
}
struct Spread
{
    double sum = 0, sum2 = 0, lo = 1e9, hi = 0;
    uint32_t n = 0;
    void Add(double ms) { sum += ms; sum2 += ms * ms; lo = std::min(lo, ms); hi = std::max(hi, ms); n++; }
    std::string Text() const
    {
        const double mean = sum / n;
        return std::format("avg {:.2f} min {:.2f} max {:.2f} sd {:.3f} ms", mean, lo, hi, std::sqrt(std::max(0.0, sum2 / n - mean * mean)));
    }
};
Spread s_present, s_gameDt;
std::chrono::steady_clock::time_point s_lastPresent{}, s_lastLog{};

} // namespace

uint32_t FrameRateTarget() { return TargetRate(); }

// On unless set to 0 / false.
static bool OnByDefault(const char* env)
{
    const std::string v = settings::Get(env);
    return v != "0" && v != "false";
}

bool MenuUnlock()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<bool> value{ true };
    return Live(generation, value, [] { return OnByDefault("AC6_MENU_UNLOCK"); });
}

bool CutsceneUnlock()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<bool> value{ true };
    return Live(generation, value, [] { return OnByDefault("AC6_CUTSCENE_UNLOCK"); });
}

uint32_t NoteSwap(uint32_t vblanks)
{
    if (FrameTimesOn())
    {
        const double ticks = double(g_memory.As<be<uint64_t>>(0x826EDDA0)->get());
        s_gameDt.Add(ticks * g_memory.As<be<float>>(0x826EDDB0)->get() * 1000.0);
    }
    const uint64_t frame = RendererFrame();
    const uint64_t cutscene = s_cutsceneFrame.load(std::memory_order_relaxed);
    const bool inCutscene = cutscene != UINT64_MAX && frame <= cutscene + kCutsceneGapFrames;
    // The 60 FPS screens unlock at any rate set (60 or more): at 60 their pace stays the console's, and what they ran
    // slower (the hangar's 3D view, every second frame on the console) runs at the rate.
    const bool menus = vblanks == 1 && MenuUnlock();
    const bool unlocked = TargetRate() != kConsoleRate && ((vblanks == 2 && (!inCutscene || CutsceneUnlock())) || menus);
    if (unlocked != s_unlocked.exchange(unlocked, std::memory_order_relaxed))
    {
        if (unlocked)
            log::Info("Frame rate: unlocked ({}) from frame {} (AC6_FRAME_RATE)", TargetRate() ? std::to_string(TargetRate()) + " FPS" : std::string("uncapped"), frame);
        else
            log::Info("Frame rate: the game's from frame {} ({})", frame, inCutscene ? "cutscene" : "present interval");
    }
    if (!unlocked)
        return vblanks;
    // AC6_FRAME_STALL=<ms> (tests): one unlocked frame in six held up to that long at its swap, for
    // uneven frame times on demand (a VRR display's or a busy machine's).
    static const double stall = [] { const char* v = std::getenv("AC6_FRAME_STALL"); return v ? std::atof(v) : 0.0; }();
    if (stall > 0.0)
    {
        static uint32_t seed = 12345;
        seed = seed * 1664525u + 1013904223u;
        if ((seed >> 16) % 6 == 0)
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(stall * double((seed >> 8) & 0xFF) / 255.0));
    }
    return TargetRate() ? vblanks : 0;
}

bool Unlocked() { return s_unlocked.load(std::memory_order_relaxed); }

void NotePresent()
{
    if (!FrameTimesOn())
        return;
    using Clock = std::chrono::steady_clock;
    const auto now = Clock::now();
    if (s_lastPresent.time_since_epoch().count())
        s_present.Add(std::chrono::duration<double, std::milli>(now - s_lastPresent).count());
    s_lastPresent = now;
    if (s_lastLog.time_since_epoch().count() == 0)
        s_lastLog = now;
    if (now - s_lastLog >= std::chrono::seconds(2) && s_present.n && s_gameDt.n)
    {
        log::Info("Frame times ({}): presents {}; game dt {}", Unlocked() ? "unlocked" : "the game's", s_present.Text(), s_gameDt.Text());
        s_present = {};
        s_gameDt = {};
        s_lastLog = now;
    }
}

void NoteCutsceneFrame() { s_cutsceneFrame.store(RendererFrame(), std::memory_order_relaxed); }

bool InCutscene()
{
    const uint64_t cutscene = s_cutsceneFrame.load(std::memory_order_relaxed);
    return cutscene != UINT64_MAX && RendererFrame() <= cutscene + kCutsceneGapFrames;
}

} // namespace ac6::gpu

// The game's frame end (sub_8233B5A0) waits on the VSync thread's vblank count before
// and after its present: >= [vsync+16] (sub_8233B4D0), then > it (sub_8233B538), 2 vblanks a mission frame. Unlocked
// frames skip both, as a PC game's loop would: the present (AC6_FRAME_RATE's pace, vsync or VRR) is the only pace.
// The wait functions have no other callers, and the D3D flip ring the vblanks also serve is never filled here (no
// command-processor interrupts).
extern "C" PPC_FUNC(__imp__sub_8233B4D0);
extern "C" PPC_FUNC(__imp__sub_8233B538);
PPC_FUNC(sub_8233B4D0)
{
    if (!ac6::gpu::Unlocked())
        __imp__sub_8233B4D0(ctx, base);
}
PPC_FUNC(sub_8233B538)
{
    if (!ac6::gpu::Unlocked())
        __imp__sub_8233B538(ctx, base);
}
