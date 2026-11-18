// The game's frame-stepped code made frame-rate independent for the unlocked frame rate (gpu/frame_rate.h).
//
// Mission code was tuned for 30 calls a second (the 60 FPS screens for 60): some steps carry the reference timestep
// 1/30 where the frame time belongs, some add a fixed amount, decay or blend once per call. Each fix scales its step by
// the frame step ratio R = frame time / the console's frame (the game's own measured frame time, CNuTimer's tick delta,
// x 30 in missions, x 60 on the 60 FPS screens): an amount per call becomes amount x R, a decay k per call k^R, a blend
// toward t with weight a per call a^R. While the frame rate is the console's (not unlocked) R is exactly 1 and every hook
// leaves its value untouched. A frame longer than the console's keeps R = 1 too (the console slows down there rather
// than taking bigger steps).
//
// Most hooks are mid-asm hooks (GracemeriaRecompLib/config/AC6.toml, "frame rate fixes"), the others wrap whole
// functions. AC6_FPS_FIXES=0 turns them off, AC6_FPS_FIX_OFF=<name>,<name> single ones (names below). The frame clock
// (AC6_FRAME_SMOOTHING) and the missile fuse (AC6_MISSILE_FUSE_30HZ) have switches of their own.
#include "endian.h"
#include "game/frame_rate_fixes.h"
#include "gpu/frame_rate.h"
#include "gpu/renderer.h"
#include "log.h"
#include "memory.h"
#include "ppc.h"
#include "settings.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace {

enum Fix : uint32_t
{
    kSpeed = 1u << 0,          // aircraft speed: thrust, drag, gravity, turn bleed (flight force step)
    kSink = 1u << 1,           // aircraft sink when not level (flight-path drop after the path blend)
    kGroundPath = 1u << 2,     // runway roll: travel direction following the nose (ground step's path blend)
    kImpactKnock = 1u << 3,    // collision knock rotation
    kRudder = 1u << 4,         // rudder deflection ramp (visual)
    kNpcRateDecay = 1u << 5,   // enemy aircraft turn rates decaying without a command
    kNpcSmoothing = 1u << 6,   // enemy aircraft input smoothing
    kGroundHeading = 1u << 7,  // ground units turning to a waypoint behind them
    kSea = 1u << 8,            // the sea's wave scroll
    kMapScroll = 1u << 9,      // the map's ground texture scroll
    kGun = 1u << 11,           // the player's gun fire rate
    kNpcWeapons = 1u << 12,    // enemy aircraft weapon cooldowns
    kBursts = 1u << 13,        // enemy flak / AA bursts
    kAiCadence = 1u << 14,     // the units' every-8-frames target search
    kSwg = 1u << 15,           // the Flash UI's timeline on mission frames (pause menu, overlays)
    kMissileReload = 1u << 16, // the player's missile reload while firing continuously
    kReplayRecord = 1u << 17,  // the replay recorder's sampling cadence
    kSubCamera = 1u << 18,     // the picture-in-picture camera's follow
    kReplayCamera = 1u << 19,  // the replay viewer's stick camera
    kEffects = 1u << 20,       // effect cadences: enemy vapour emission, shoulder vapour flicker, smoke warm-up
    kSubmunition = 1u << 21,   // a sub-munition spawner's interval
    kMenus = 1u << 22,         // the 60 FPS screens' cadences: the hangar's 30 Hz scene, frame-counted UI waits
    kShake = 1u << 23,         // the camera / HUD shake shown as the console's 30 Hz samples, interpolated
    kMotion = 1u << 24,        // model motions stepped a fixed amount per update (the hangar door, unit models)
    kAttitudeDt = 1u << 25,    // the player's attitude step on uneven frame times (the aircraft / camera jitter)
};

struct FixName { const char* name; uint32_t bit; };
constexpr FixName kFixNames[] = {
    { "speed", kSpeed }, { "sink", kSink }, { "groundpath", kGroundPath }, { "knock", kImpactKnock },
    { "rudder", kRudder }, { "npcdecay", kNpcRateDecay }, { "npcsmooth", kNpcSmoothing },
    { "groundheading", kGroundHeading }, { "sea", kSea }, { "mapscroll", kMapScroll },
    { "gun", kGun }, { "npcweapons", kNpcWeapons }, { "bursts", kBursts }, { "aicadence", kAiCadence },
    { "ui", kSwg }, { "missilereload", kMissileReload }, { "replayrecord", kReplayRecord },
    { "subcamera", kSubCamera }, { "replaycamera", kReplayCamera }, { "effects", kEffects },
    { "submunition", kSubmunition }, { "menus", kMenus }, { "shake", kShake }, { "motion", kMotion }, { "attitudedt", kAttitudeDt },
};

// Read on first use: the settings file (settings.cpp) sets the variables at the start of main.
const uint32_t& FixesEnabled()
{
    static const uint32_t value = [] {
        const char* all = std::getenv("AC6_FPS_FIXES");
        if (all && *all == '0')
            return 0u;
        uint32_t mask = ~0u;
        if (const char* off = std::getenv("AC6_FPS_FIX_OFF"))
        {
            for (std::string_view list(off); !list.empty();)
            {
                const size_t comma = list.find(',');
                const std::string_view name = list.substr(0, comma);
                for (const FixName& f : kFixNames)
                    if (name == f.name)
                        mask &= ~f.bit;
                list = comma == std::string_view::npos ? std::string_view() : list.substr(comma + 1);
            }
        }
        return mask;
    }();
    return value;
}

// The game's measured frame time (CNuTimer's tick delta x seconds per tick).
double FrameSeconds()
{
    const double ticks = double(ac6::g_memory.As<ac6::be<uint64_t>>(0x826EDDA0)->get());
    return ticks * ac6::g_memory.As<ac6::be<float>>(0x826EDDB0)->get();
}

// The frame step ratio, once per frame (all hooks run on the game's threads within the frame's update).
double Ratio()
{
    static thread_local uint64_t frame = UINT64_MAX;
    static thread_local double ratio = 1.0;
    const uint64_t now = ac6::gpu::RendererFrame();
    if (now != frame)
    {
        frame = now;
        ratio = 1.0;
        if (ac6::gpu::Unlocked())
        {
            // The console's frame: the game's vblanks per present ([0x827BE1F0+12], 2 in missions, 1 on the 60 FPS
            // screens) at 60 Hz.
            const uint32_t vblanks = std::clamp<uint32_t>(ac6::g_memory.As<ac6::be<uint32_t>>(0x827BE1FC)->get(), 1, 2);
            ratio = std::clamp(FrameSeconds() * 60.0 / vblanks, 0.01, 1.0);
        }
    }
    return ratio;
}

// The ratio when the fix is on and the frame is unlocked, 1 otherwise.
double Ratio(uint32_t fix) { return (FixesEnabled() & fix) ? Ratio() : 1.0; }

// Whether this frame completes a tick of `hz` (the frame time x hz accumulated per frame): always on locked frames.
// Shared by the cadence gates (they run on the game's and the job threads). The frame's first caller decides.
struct Ticker
{
    double hz;
    std::mutex mutex;
    uint64_t frame = UINT64_MAX;
    double ticks = 0.0;
    bool tick = true;

    bool Tick()
    {
        const uint64_t now = ac6::gpu::RendererFrame();
        std::lock_guard lock(mutex);
        if (now != frame)
        {
            frame = now;
            if (!ac6::gpu::Unlocked())
            {
                ticks = 0.0;
                tick = true;
            }
            else
            {
                ticks += std::min(1.0, FrameSeconds() * hz);
                tick = ticks >= 1.0 - 1e-6;
                if (tick)
                    ticks = std::max(0.0, ticks - 1.0);
            }
        }
        return tick;
    }
};
Ticker s_tick30{ 30.0 }, s_tick60{ 60.0 };
bool Tick30() { return s_tick30.Tick(); }
// How far this frame is from its last 30 Hz tick toward the next, in ticks (0 on a tick frame at the console's rate).
double Phase30()
{
    s_tick30.Tick();
    std::lock_guard lock(s_tick30.mutex);
    return s_tick30.ticks;
}
bool Tick60() { return s_tick60.Tick(); }

// A cadence gate: skip this frame's per-call step (the fix is on, the frame unlocked and no 30 Hz tick completed).
bool SkipStep(uint32_t fix) { return (FixesEnabled() & fix) && ac6::gpu::Unlocked() && !Tick30(); }
// The same for a 60 Hz cadence (the 60 FPS screens).
bool SkipStep60(uint32_t fix) { return (FixesEnabled() & fix) && ac6::gpu::Unlocked() && !Tick60(); }

float LoadFloat(uint32_t address) { return ac6::g_memory.As<ac6::be<float>>(address)->get(); }
void StoreFloat(uint32_t address, float value) { *ac6::g_memory.As<ac6::be<float>>(address) = value; }

// The flight force step's state between its hooks (one aircraft at a time on the thread that runs it).
thread_local float t_speedBefore = 0.0f;
thread_local double t_pathWeight = -1.0;

} // namespace

namespace ac6::game {
double FrameStepRatio() { return Ratio(); }
}

// ---- The player's flight force step, sub_823046A0 ----------------------------------
// [fm+1320] is the aircraft's forward speed (km/h, body -Z). Every term that changes it runs once per call: thrust and
// gravity along the path carry the reference timestep 1/30 (0x82069DBC) where the frame time belongs, overspeed drag,
// the nose-attitude term, stall drag and the turn bleed (the High-G turn's deceleration) are flat amounts per call.
// They are all linear and nothing reads the speed between them, so scaling the call's net change by R is scaling
// each term by R. Top speed (their balance) is unchanged, and acceleration and braking don't grow with the frame rate.

// 0x82304714 (after `lfs f8,1320(r31)`): the speed before this call's terms.
void FpsFlightSpeedBegin(PPCRegister& f8)
{
    t_speedBefore = float(f8.f64);
    t_pathWeight = -1.0;
}

// 0x82304AA4 (before the first read of the final speed, `lwz r10,1312(r31)`). r30 = the aircraft's transform.
void FpsFlightSpeedEnd(PPCRegister& r30, PPCRegister& r31)
{
    const uint32_t speed = r31.u32 + 1320;
    const double r = Ratio(kSpeed);
    if (r != 1.0)
        StoreFloat(speed, float(t_speedBefore + (LoadFloat(speed) - t_speedBefore) * r));
    // AC6_FPS_TRACE=1: the speed and height every 100 ms (comparing frame rates).
    static const bool trace = [] { const char* v = std::getenv("AC6_FPS_TRACE"); return v && *v == '1'; }();
    if (trace)
    {
        using Clock = std::chrono::steady_clock;
        static thread_local Clock::time_point last{};
        const auto now = Clock::now();
        if (now - last >= std::chrono::milliseconds(100))
        {
            last = now;
            ac6::log::Info("FPS trace: fm {:08X} speed {:.2f} height {:.2f} R {:.3f}", r31.u32, -LoadFloat(speed), LoadFloat(r30.u32 + 68), Ratio());
        }
    }
}

// The flight path (the direction the aircraft travels, [fm+128..136]) follows the nose through
// dir = normalize(unit(V) + w * path_old) with w = c / (30 dt) (0x82304E00: rate-correct as shipped), and the sink
// when not level, d, is subtracted once per call after that blend (0x82304F40). Repeated every call, blend and sink
// settle the path d * (1 + w) below the nose: w doubles at 60 fps and so does the sink. At the console's rate w would
// be w * R. Scaling d by (1 + w R) / (1 + w) settles the path where the console's does.

// 0x82304E00 (after `fmuls f12,f12,f29`): w.
void FpsFlightPathWeight(PPCRegister& f12) { t_pathWeight = f12.f64; }

// 0x82304F40 (before `fsubs f11,f11,f9`): d.
void FpsFlightPathDrop(PPCRegister& f9)
{
    const double r = Ratio(kSink);
    if (r == 1.0 || t_pathWeight < 0.0)
        return;
    f9.f64 = float(f9.f64 * (1.0 + t_pathWeight * r) / (1.0 + t_pathWeight));
}

// ---- The ground step, sub_82305278 -----------------------------------------------------------------------------------
// 0x8230590C (after `lfs f0,324(r31)`): on the runway the path blends toward the nose with w = [fm+324], no 1/dt as the
// air step has (w = c / (30 dt)): the travel direction catches up with the nose R times faster per second. w / R is the
// air step's form (identical at 30 fps).
void FpsGroundPathWeight(PPCRegister& f0)
{
    const double r = Ratio(kGroundPath);
    if (r != 1.0)
        f0.f64 = float(f0.f64 / r);
}

// ---- The attitude step, sub_82305C20 ----------------------------------------------------------------------------------
// 0x823063C0 (after `lfs f1,328(r31)`): a collision knock (slot +92) rotates the aircraft by [fm+328] each call. The
// angle itself decays with dt. Per call x R.
void FpsImpactKnock(PPCRegister& f1)
{
    const double r = Ratio(kImpactKnock);
    if (r != 1.0)
        f1.f64 = float(f1.f64 * r);
}

// The aircraft / camera jitter: the aircraft is drawn at its pose from before the frame's steps, while the camera
// (0x822638B0 -> 0x8225E368) follows the attitude the attitude step has just produced, through the aircraft tick's copy
// of the transform (0x82233FB0). Their difference is one step's rotation, so with uneven frame times (VRR, a busy
// machine, the bug report frame captures) it changes from frame to frame: the aircraft jumps sideways for a frame and
// back (chase view), the view against the cockpit (first person): 1.8 px rms at 1280x720, 11 of 120 frames over 2 px,
// against 0.13 px for the attitude step alone at a constant dt. So, unlocked, the player's attitude step steps by a
// smoothed time: a running average of the frame times plus a tenth of the time it owes the real clock each frame, so
// its rotation stays in step with real time (a long frame is caught up over the next ones) while its steps are even:
// 0.17 px, none over 1.
// The force step (position) keeps the frame's own time. Frames over 0.25 s pass through.
namespace {
uint32_t PlayerFlightModel()
{
    const uint32_t root = ac6::g_memory.As<ac6::be<uint32_t>>(0x826E4E54)->get();
    const uint32_t units = root ? ac6::g_memory.As<ac6::be<uint32_t>>(root + 0x29FC4)->get() : 0;
    const uint32_t actor = units ? ac6::g_memory.As<ac6::be<uint32_t>>(units + 0x1008)->get() : 0;
    return actor ? ac6::g_memory.As<ac6::be<uint32_t>>(actor + 4912)->get() : 0;
}

struct AttitudeClock { double average = 0.0, owed = 0.0; };
AttitudeClock s_attitudeClock;
}

// The switches for the camera jitter tests, live (the overlay's Debug tab):
// AC6_ATTITUDE_SMOOTHING (default on, AC6_FPS_FIX_OFF=attitudedt turns it off too), AC6_ATT_TRACE, and
// AC6_FRAME_SMOOTHING (default on) with AC6_VSYNC for the frame clock below.
struct AttitudeSwitches { bool smoothing = true, trace = false, clock = true, vsync = true; };
AttitudeSwitches Switches()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<bool> smoothing{ true }, trace{ false }, clock{ true }, vsync{ true };
    if (const uint64_t now = ac6::settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        const std::string s = ac6::settings::Get("AC6_ATTITUDE_SMOOTHING"), t = ac6::settings::Get("AC6_ATT_TRACE"),
            c = ac6::settings::Get("AC6_FRAME_SMOOTHING");
        smoothing.store(s != "0" && s != "false", std::memory_order_relaxed);
        trace.store(t == "1" || t == "true", std::memory_order_relaxed);
        clock.store(c != "0" && c != "false", std::memory_order_relaxed);
        vsync.store(ac6::settings::Get("AC6_VSYNC") != "0", std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    return { smoothing.load(std::memory_order_relaxed), trace.load(std::memory_order_relaxed),
        clock.load(std::memory_order_relaxed), vsync.load(std::memory_order_relaxed) };
}

// ---- The frame clock: CNuTimer::advance, sub_82211F60 --------------------------------
// With VSync on, the frames reach the display evenly, a refresh apart, but the time the game measures between its frames
// can alternate. When the port's pacing is slightly faster than the display's real refresh (60.000 Hz against a
// "60 Hz" monitor), the swap chain's queue fills after about 26 s, and from then on the game thread blocks at different
// points in alternate frames: 33 / 10 / 40 / 10 / 10 ms, averaging the refresh. Everything then steps unevenly while
// the frames show evenly, and the chase camera, which follows by an averaged frame time, swings against the world every
// frame.
// AC6_PACE_TEST_HZ a little above the display's rate reproduces it: the presents' spacing then has a standard deviation
// of 0.2-0.8 ms, the game's frame times 7-8 ms. So, unlocked and paced with VSync on, the frame's delta ([0x826EDDA0],
// the one place every consumer reads it from) is replaced by a smoothed one, the attitude clock for the whole game: a
// running average plus a tenth of the time owed to the real clock each frame, so the game's time stays the real time
// while its steps are even. The timer's own accumulated time ([this+8]) is moved by the same amount. Deltas over 0.1 s
// (loading, a hitch) pass through and start it again. AC6_FRAME_SMOOTHING=0 (the Debug tab) turns it off.
namespace {
struct FrameClock { double average = 0.0, owed = 0.0; };
FrameClock s_frameClock;
}
extern "C" PPC_FUNC(__imp__sub_82211F60);
PPC_FUNC(sub_82211F60)
{
    const uint32_t timer = ctx.r3.u32;
    const bool sample = (ctx.r4.u32 & 0xFF) != 0;
    const uint64_t totalBefore = timer ? ac6::g_memory.As<ac6::be<uint64_t>>(timer + 8)->get() : 0;
    __imp__sub_82211F60(ctx, base);
    FrameClock& c = s_frameClock;
    const AttitudeSwitches sw = Switches();
    if (!sample || !sw.clock || !sw.vsync || !ac6::gpu::Unlocked() || !ac6::gpu::FrameRateTarget())
    {
        c = {};
        return;
    }
    auto* deltaTicks = ac6::g_memory.As<ac6::be<uint64_t>>(0x826EDDA0);
    const uint64_t ticks = deltaTicks->get();
    const double perTick = LoadFloat(0x826EDDB0);
    const double dt = double(ticks) * perTick;
    if (!(perTick > 0.0) || !(dt > 0.0) || dt > 0.1 || c.average <= 0.0)
    {
        c = { dt > 0.0 && dt <= 0.1 ? dt : 0.0, 0.0 };
        return;
    }
    c.average += (dt - c.average) * 0.05;
    c.owed += dt;
    const double step = std::clamp(c.average + (c.owed - c.average) * 0.1, c.average * 0.5, c.average * 2.0);
    c.owed -= step;
    const uint64_t stepTicks = uint64_t(std::llround(step / perTick));
    deltaTicks->set(stepTicks);
    // The timer's accumulated time took the measured delta, unless it is paused (then none): moved to the step.
    auto* total = ac6::g_memory.As<ac6::be<uint64_t>>(timer + 8);
    if (timer && total->get() == totalBefore + ticks)
        total->set(totalBefore + stepTicks);
}

// AC6_ATT_TRACE=1 (diagnostics): per player attitude step the frame time, the step, and the main camera's view matrix
// (rows 0 and 1, the translation, [[0x826E4E54]+0x2F9A0]+256: the previous frame's camera here).
void AttitudeTrace(const PPCContext& ctx, double dt)
{
    if (!Switches().trace || !ctx.r3.u32 || ctx.r3.u32 != PlayerFlightModel())
        return;
    const uint32_t root = ac6::g_memory.As<ac6::be<uint32_t>>(0x826E4E54)->get();
    const uint32_t m = root ? ac6::g_memory.As<ac6::be<uint32_t>>(root + 0x2F9A0)->get() : 0;
    auto v = [&](uint32_t i) { return m ? LoadFloat(m + 256 + 4 * i) : 0.0f; };
    ac6::log::Info("ATT {} dt {:.6f} step {:.6f} view {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.3f} {:.3f} {:.3f}",
        ac6::gpu::RendererFrame(), dt, ctx.f1.f64, v(0), v(1), v(2), v(4), v(5), v(6), v(12), v(13), v(14));
}

extern "C" PPC_FUNC(__imp__sub_82305C20);
PPC_FUNC(sub_82305C20)
{
    AttitudeClock& c = s_attitudeClock;
    const double dt = ctx.f1.f64;
    if (!(FixesEnabled() & kAttitudeDt) || !Switches().smoothing || !ac6::gpu::Unlocked() || !ctx.r3.u32 ||
        ctx.r3.u32 != PlayerFlightModel())
    {
        if (ctx.r3.u32 && ctx.r3.u32 == PlayerFlightModel())
            c = {};   // locked frames, or the smoothing off: start again when on
        AttitudeTrace(ctx, dt);
        __imp__sub_82305C20(ctx, base);
        return;
    }
    if (!(dt > 0.0) || dt > 0.25 || c.average <= 0.0)
        c = { dt > 0.0 && dt <= 0.25 ? dt : 0.0, 0.0 };
    else
    {
        c.average += (dt - c.average) * 0.05;
        c.owed += dt;
        const double step = std::clamp(c.average + (c.owed - c.average) * 0.1, c.average * 0.5, c.average * 2.0);
        c.owed -= step;
        ctx.f1.f64 = step;
    }
    AttitudeTrace(ctx, dt);
    __imp__sub_82305C20(ctx, base);
}

// ---- Control conditioning, sub_82329B40 (the derived player class) ---------------------------------------------------
// [fm+1456], the rudder deflection, ramps +-0.125 per call toward the input and back to 0 (only the state report reads
// it: visual). 0x82329E28 (before `fadds f0,f0,f13`, f0 = the step) and 0x82329E7C (before `fadds f0,f13,f0`, f13).
void FpsRudderRamp(PPCRegister& f0)
{
    const double r = Ratio(kRudder);
    if (r != 1.0)
        f0.f64 = float(f0.f64 * r);
}
void FpsRudderReturn(PPCRegister& f13)
{
    const double r = Ratio(kRudder);
    if (r != 1.0)
        f13.f64 = float(f13.f64 * r);
}

// ---- Enemy aircraft: the NPC flight engine step sub_82328A70, the unit update sub_82293128 --------------------------
// Without a command the roll, pitch and yaw rates ([140], [132], [136]) decay x f30 (0.8) per call: 0x82328B6C,
// 0x82328C50, 0x82328D14 (after `fmuls f0,f0,f30`): x 0.8^(R-1) more makes it 0.8^R.
void FpsNpcRateDecay(PPCRegister& f0, PPCRegister& f30)
{
    const double r = Ratio(kNpcRateDecay);
    if (r != 1.0 && f30.f64 > 0.0)
        f0.f64 = float(f0.f64 * std::pow(f30.f64, r - 1.0));
}

// [unit+792] and [unit+796] (the latter smooths the engine's accel command [592]) blend x = 0.75 x + 0.25 t per call:
// 0x822936B8 (before `stfs f12,792(r31)`, target f10) and 0x822936C0 (before `stfs f0,796(r31)`, target f9).
// x = t + (x - t) 0.75^R.
namespace {
float NpcBlend(uint32_t unit, uint32_t offset, double target, double blended)
{
    const double r = Ratio(kNpcSmoothing);
    if (r == 1.0)
        return float(blended);
    const double keep = LoadFloat(0x82069E48);   // 0.75, the constant the code multiplies by
    return float(target + (LoadFloat(unit + offset) - target) * std::pow(keep, r));
}
}
void FpsNpcSmoothing792(PPCRegister& r31, PPCRegister& f10, PPCRegister& f12) { f12.f64 = NpcBlend(r31.u32, 792, f10.f64, f12.f64); }
void FpsNpcSmoothing796(PPCRegister& r31, PPCRegister& f9, PPCRegister& f0) { f0.f64 = NpcBlend(r31.u32, 796, f9.f64, f0.f64); }

// ---- Ground units: steer to waypoint, sub_822A6F10 -----------------------------------------------------------------
// 0x822A70F0 (after `lfs f0,-25204(r11)`, 10 degrees): the heading target [unit+524] steps 10 degrees per call while
// the waypoint is directly behind. Per call x R.
void FpsGroundHeadingStep(PPCRegister& f0)
{
    const double r = Ratio(kGroundHeading);
    if (r != 1.0)
        f0.f64 = float(f0.f64 * r);
}

// ---- The sea: CMapManager slot 65, sub_82100A08 -------------------------------------------------------------------
// The wave scroll ACE_vOceanScroll / ocean_vParam2 (CMapManager +27984..+27996, bound by slot 61) advances by the sky
// data's .ocean.speScroll steps ([+24688], [+24692], x 1.2 = oceanNormal2MoveScale [+27968] for zw) once per call, each
// wrapped by fmod 1.875. The step register f13 before 0x82101300, 0x82101314, 0x82101338 and 0x8210135C: per call x R.
void FpsSeaStep(PPCRegister& f13)
{
    const double r = Ratio(kSea);
    if (r != 1.0)
        f13.f64 = float(f13.f64 * r);
}

// ---- The map's UV scroll: CMapManager slot 61, sub_820FDCD0 ---------------------------------------------------------
// ACE_vTextureUVOffset's V, the global 0x82A218A0, decreases 0.001 per call.
// The step f13 before 0x820FE718 (`fsubs f0,f0,f13`): x R.
void FpsMapScroll(PPCRegister& f13)
{
    const double r = Ratio(kMapScroll);
    if (r != 1.0)
        f13.f64 = float(f13.f64 * r);
}

// ---- The player's gun: sub_8222C0F0 ---------------------------------------------------------------------------------
// The cooldown [player+5020] counts down by dt while above 0 and a round fires when it is <= 0, at most one per update.
// The shot resets it with `=` to the interval I = reloadRateBullet x the gun's parameter. The time below 0 is lost, so
// the real interval is I rounded up to whole frames: ceil(30 I) / 30 on the console, finer (shorter) at higher frame
// rates. Unlocked, the reset (before 0x8222C3F8 `stfs f0,5020(r31)`, f0 = I) takes the console's interval and keeps
// the time already past 0: the console's fire rate at any frame rate.
void FpsGunCooldown(PPCRegister& r31, PPCRegister& f0)
{
    static const bool trace = [] { const char* v = std::getenv("AC6_FPS_TRACE"); return v && *v == '1'; }();
    if (trace)
        ac6::log::Info("FPS trace: gun round, interval {:.4f}", f0.f64);
    if (!(FixesEnabled() & kGun) || !ac6::gpu::Unlocked())
        return;
    const double frames = std::max(1.0, std::ceil(f0.f64 * 30.0 - 1e-3));
    const double past = std::min(0.0, double(LoadFloat(r31.u32 + 5020)));
    f0.f64 = float(frames / 30.0 + past);
}

// ---- Enemy aircraft weapons: sub_822A5438 (NPC aircraft vtable 0x820097B0 slot 63) -----------------------------------
// Per weapon w the cooldown [npc+416+4w] counts down by dt on the updates it is above 0 (those skip firing). One at or
// below 0 fires and is reset with `=` to the weapon's interval P (param +28). The firing update doesn't count down, so
// the console fires every ceil(30 P) + 1 frames. Unlocked, the reset (before 0x822A5B28 `stfsx f0,r24,r31`, f0 = P) takes
// that interval, less the firing update's own frame, and keeps the time already past 0.
void FpsNpcWeaponCooldown(PPCRegister& r24, PPCRegister& r31, PPCRegister& f0)
{
    if (!(FixesEnabled() & kNpcWeapons) || !ac6::gpu::Unlocked())
        return;
    const double frames = std::ceil(f0.f64 * 30.0 - 1e-3) + 1.0;
    const double past = std::min(0.0, double(LoadFloat(r31.u32 + r24.u32)));
    f0.f64 = float(frames / 30.0 - Ratio() / 30.0 + past);
}

// ---- Enemy flak / AA bursts: sub_822F6568 (vtable 0x82006A90), sub_822F7200 (0x82006BA8) ----------------------------
// A burst of [a+336] shots: the timer ([a+1476], [a+876]) counts down by dt and a shot fires when it reaches 0, then it
// is reset with `=` to [a+340]: ceil(30 x [a+340]) frames on the console. Before 0x822F66FC / 0x822F7474 (`stfs f0`):
// the console's interval plus the time already past 0.
namespace {
void BurstReset(uint32_t timer, PPCRegister& f0)
{
    if (!(FixesEnabled() & kBursts) || !ac6::gpu::Unlocked())
        return;
    const double frames = std::max(1.0, std::ceil(f0.f64 * 30.0 - 1e-3));
    f0.f64 = float(frames / 30.0 + std::min(0.0, double(LoadFloat(timer))));
}
}
void FpsBurstResetA(PPCRegister& r31, PPCRegister& f0) { BurstReset(r31.u32 + 1476, f0); }
void FpsBurstResetB(PPCRegister& r31, PPCRegister& f0) { BurstReset(r31.u32 + 876, f0); }

// ---- The units' target search cadence ------------------------------------------------------------------------------
// CAce6MissionManager::Update counts mission frames in [root+32] (0x8226D390, mission state 2). Units search for a
// target (vt +264, sub_8229AF60) when ([root+32] ^ [unit+176]) & 7 == 0: every 8 frames, staggered by unit, 8 frames
// are 267 ms on the console and shorter unlocked. Unlocked, the counter advances once per 30 Hz tick of the frame
// time (after 0x8226D394 `addi r10,r10,1`) and the searches run only on the frames that complete a tick (after the
// `clrlwi r11,r11,29` at 0x822A3130, 0x822A7248 and 0x82292AE8: r11 = 1 skips).
void FpsAiCadenceTick(PPCRegister& r10)
{
    if (SkipStep(kAiCadence))
        r10.u64 = r10.u32 - 1;   // no tick completed: the counter stays
}
void FpsAiCadenceGate(PPCRegister& r11)
{
    if (r11.u32 == 0 && SkipStep(kAiCadence))
        r11.u64 = 1;
}

// ---- The Flash UI (swg): CSwgManager slot 8, sub_820D8ED0 --------------------------------------------------------------
// Every swg timeline (menus, the intro, in-mission screens) advances through sub_8237AFF0(player, delta), a delta-time
// player with a 60 Hz tick, but its one caller passes a literal 1.0 per call (0x820D9068 `lfs f1,16908(r11)`): one
// tick per frame, 60 a second in the menus and 30 during missions. After that load: x the frame step ratio, which is
// relative to the console's frame of the screen (1/30 s in missions, 1/60 s on the 60 FPS screens).
void FpsSwgAdvance(PPCRegister& f1)
{
    const double r = Ratio(kSwg);
    if (r != 1.0)
        f1.f64 = float(f1.f64 * r);
}

// ---- The player's missile reload: sub_8222C658 ----------------------------------------------------------------------
// Firing sets the rail's timer [p+5024+4r] = T ([p+6064]) (0x8222CABC) and its rate k (0x8222CAD0). Every update the
// timer decays by k dt and is clamped at 0 (0x8222CC18), and a rail fires when it is <= 0, one launch per update: the
// console's interval is ceil(30 T / k) frames. Unlocked: the reset takes that interval and the clamp keeps the time
// past 0 on the update that crossed it (the next decay clamps an idle rail as the game does).
namespace { thread_local double t_reloadCarry = 0.0; }
void FpsMissileReloadSave(PPCRegister& r11, PPCRegister& r31)
{
    t_reloadCarry = std::min(0.0, double(LoadFloat(r31.u32 + r11.u32)));
}
void FpsMissileReloadReset(PPCRegister& r29, PPCRegister& r31, PPCRegister& f1)
{
    if (!(FixesEnabled() & kMissileReload) || !ac6::gpu::Unlocked() || f1.f64 <= 0.0)
        return;
    const double time = LoadFloat(r31.u32 + 6064);
    const double frames = std::max(1.0, std::ceil(30.0 * time / f1.f64 - 1e-4));
    StoreFloat(r31.u32 + 5024 + 4 * r29.u32, float(frames * f1.f64 / 30.0 + t_reloadCarry));
}
bool FpsMissileReloadKeepCarry(PPCRegister& r9, PPCRegister& f0, PPCRegister& f29)
{
    return (FixesEnabled() & kMissileReload) && ac6::gpu::Unlocked() && f0.f64 > -double(LoadFloat(r9.u32 + 8)) * f29.f64;
}

// ---- The replay recorder: sub_82254CF0 ------------------------------------------------------------------------------
// It samples on its call counter [ctx+60] (the player every 2nd call, others every 4th / Nth) into a ring that drops
// its oldest records when full: unlocked it would record 1/R times as often and keep a shorter stretch of the mission
// (playback is keyed to the mission time). Before 0x82254D08: frames that complete no 30 Hz tick skip the whole
// recorder, its counter included, to the epilogue 0x82255034.
bool FpsReplayRecordCadence() { return SkipStep(kReplayRecord); }

// ---- The picture-in-picture camera: sub_822F1090 (container at root+0x31090) ---------------------------------------
// Its follow gain [A+10180] ramps toward 24 by c = 1/90 twice per call, k2 = 24 - (24 - k0)(1 - c)^2 (after 0x822F1234:
// f0 = k2, f13 = 24 - k1, f12 = c), R calls' worth is 24 - f13 (1 - c)^(2R - 1). The follow moves by
// (target - pos) k dt, an Euler step with k dt = 0.8 on the console: before 0x822F125C (f0 = k, f31 = dt) the factor
// becomes the console's per-frame approach a = k dt / R spread over R frames, (1 - (1 - a)^R) / dt.
void FpsSubCameraGain(PPCRegister& f0, PPCRegister& f12, PPCRegister& f13)
{
    const double r = Ratio(kSubCamera);
    if (r != 1.0)
        f0.f64 = float(LoadFloat(0x82069C94) - f13.f64 * std::pow(1.0 - f12.f64, 2.0 * r - 1.0));
}
namespace {
thread_local bool t_subCameraExact = false;   // set while sub_822F1090's hook solves the follow: the guest's step zeroed
}
void FpsSubCameraFollow(PPCRegister& f0, PPCRegister& f31)
{
    if (t_subCameraExact)
    {
        f0.f64 = 0.0;
        return;
    }
    const double r = Ratio(kSubCamera);
    if (r == 1.0 || f31.f64 <= 0.0)
        return;
    const double a = f0.f64 * f31.f64 / r;
    f0.f64 = float((a >= 1.0 ? 1.0 : 1.0 - std::pow(1.0 - a, r)) / f31.f64);
}

// ---- The replay viewer's stick camera: sub_82276890 (mission state 3) ----------------------------------------------
// Three blends toward the stick's targets per call: zoom (before 0x822769B0, weight f13) and two rotations (before
// 0x82276AA0, weights f31 and f28): w = 1 - (1 - w)^R for weights in (0, 1).
namespace {
double Blend(double w, double r) { return w > 0.0 && w < 1.0 ? 1.0 - std::pow(1.0 - w, r) : w; }
}
void FpsReplayCameraZoom(PPCRegister& f13)
{
    const double r = Ratio(kReplayCamera);
    if (r != 1.0)
        f13.f64 = float(Blend(f13.f64, r));
}
void FpsReplayCameraRotate(PPCRegister& f28, PPCRegister& f31)
{
    const double r = Ratio(kReplayCamera);
    if (r == 1.0)
        return;
    f31.f64 = float(Blend(f31.f64, r));
    f28.f64 = float(Blend(f28.f64, r));
}

// ---- Effect cadences ------------------------------------------------------------------------------------------------
// NPCLineVapor (sub_822BE3B8) emits a trail point every N+1 calls and blanks 10 calls after a jump, ShoulderVapor
// (sub_822BF758) re-rolls its random value every 4 calls. Frames that complete no 30 Hz tick skip that cadence logic:
// before 0x822BE3D8 (to the epilogue 0x822BE44C) and 0x822BF8D4 (to 0x822BF92C). The player's wingtip vapour (one point
// per call) is not gated: its newest point would trail the wingtip by up to 1/30 s.
bool FpsNpcVaporCadence() { return SkipStep(kEffects); }
bool FpsShoulderVaporRoll() { return SkipStep(kEffects); }
// MissileSmoke's 2-call warm-up counter [g+244] (after 0x822BDD74 `addi r11,r11,-1`) counts ticks.
void FpsMissileSmokeWarmup(PPCRegister& r11)
{
    if (SkipStep(kEffects))
        r11.u64 = r11.u32 + 1;
}

// ---- A sub-munition spawner: sub_822DE218 (arms 1014, vtable 0x82007758) --------------------------------------------
// It spawns one item each time the dt timer [a+412] goes below 0 and re-arms it with `=` I = [a+328] / 12 (0x822DE2E0):
// floor(30 I) + 1 frames on the console. Unlocked: that interval plus the time already past 0.
void FpsSubmunitionInterval(PPCRegister& r31, PPCRegister& f0)
{
    if (!(FixesEnabled() & kSubmunition) || !ac6::gpu::Unlocked())
        return;
    const double frames = std::floor(30.0 * f0.f64 + 1e-4) + 1.0;
    f0.f64 = float(frames / 30.0 + std::min(0.0, double(LoadFloat(r31.u32 + 412))));
}

// ---- The 60 FPS screens (AC6_MENU_UNLOCK) ------------------------------------------
// CSelectAircraftManager slot 35 (sub_82146D58, aircraft select and hangar) toggles [mgr+0x20000-10500] every call
// (0x82146D7C) and updates the 3D scene (models, camera orbit: slots 37-39) only on the calls it is 0 (tested at
// 0x82146EC8 and 0x82146F48), with the frame times accumulated since: 30 Hz on the console's 60 FPS screen. The hangar's
// draw (sub_821498B8) alternates two passes on the same toggle, and its shadow needs them strictly alternating (held at
// 1 for several frames, the shadow turns blocky and flickers), so the toggle keeps flipping every frame. Unlocked,
// only the update's two tests (after their `lwz r11`) are steered: the scene updates on every frame with that frame's
// time (smooth, everything in the update integrates the time it is given: slots 37-39, the model motion
// sub_822A0DE0, the camera orbit slots 19 / 3), or with AC6_HANGAR_30HZ=1 on the frames that complete a 30 Hz tick, as
// on the console.
void FpsHangarUpdate(PPCRegister& r11)
{
    static const bool console30 = [] { const char* v = std::getenv("AC6_HANGAR_30HZ"); return v && *v == '1'; }();
    if ((FixesEnabled() & kMenus) && ac6::gpu::Unlocked())
        r11.u64 = !console30 || Tick30() ? 0 : 1;
}

// Frame-counted UI on those screens, counted in 60 Hz ticks: the aircraft slots' 20-frame wait before their resources
// swap (after 0x82147900 `addi r10,r10,1`, slot 36 sub_82147010) and the stat chart's morph (after 0x82151030, the
// `-1` of [80], CSelectAircraftSpider slot 3 sub_82151018): a frame without a tick undoes the step.
void FpsAircraftSlotWait(PPCRegister& r10)
{
    if (SkipStep60(kMenus))
        r10.u64 = r10.u32 - 1;
}
void FpsSpiderMorph(PPCRegister& r11)
{
    if (SkipStep60(kMenus))
        r11.u64 = r11.u32 + 1;
}

// The hangar's draw (sub_821498B8) builds one 3D image over two calls on the toggle: 0 renders the shadow maps and
// the environment, saves the scene's samples and shows the image the last 1 finished, 1 reloads the samples, draws
// the aircraft and its floor shadows and finishes the image. So the 3D view
// refreshes every second frame (30 Hz on the console's 60 FPS screen). Unlocked, every frame runs both: the draw is
// called with the toggle at 0, minus its tail (it would show the old image, FpsHangarDrawTail jumps from 0x82149E60
// to the epilogue 0x82149F58 after the 0x8000 target has ended and its buckets are closed), then at 1.
// AC6_HANGAR_FULLRATE=0 keeps the console's alternation.
namespace {
bool s_hangarSkipTail = false;
bool HangarFullRate()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_HANGAR_FULLRATE"); return !v || *v != '0'; }();
    return enabled && (FixesEnabled() & kMenus) && ac6::gpu::Unlocked();
}
}

extern "C" PPC_FUNC(__imp__sub_821498B8);
PPC_FUNC(sub_821498B8)
{
    if (!HangarFullRate())
    {
        __imp__sub_821498B8(ctx, base);
        return;
    }
    auto* toggle = ac6::g_memory.As<ac6::be<uint32_t>>(ctx.r3.u32 + 0x1D6FC);   // [mgr+0x20000-10500]
    const uint32_t saved = toggle->get();
    const PPCContext entry = ctx;
    *toggle = 0u;
    s_hangarSkipTail = true;
    __imp__sub_821498B8(ctx, base);
    s_hangarSkipTail = false;
    ctx = entry;
    *toggle = 1u;
    __imp__sub_821498B8(ctx, base);
    *toggle = saved;
}

bool FpsHangarDrawTail() { return s_hangarSkipTail; }

// ---- Camera shake: sub_8225D850 (camera modes 1-3: third person, cockpit, HUD) ------------------------------------
// The buffet / overspeed shake (sub_8225CF08) retargets every 0.1 s of dt and swings toward each target and back with
// dt-integrated motion, so its size and rate don't depend on the frame rate. But the console shows it at 30 Hz, about
// three samples of each 0.1 s swing. Unlocked, every frame shows the whole swing, which reads as a fast vibration
// (seen as HUD flicker). Unlocked, the shake added to the eye offset (r4: [out] += [cam+480..488]) is replaced by
// the console's 30 Hz samples of it, interpolated: the sample of the last 30 Hz tick and the one before, blended by the
// frame's progress toward the next tick (one tick of latency, smooth motion). The shake's own state still advances
// every frame.
namespace {
struct ShakeSamples { uint64_t frame = UINT64_MAX; float previous[3] = {}, current[3] = {}; };
}
extern "C" PPC_FUNC(__imp__sub_8225D850);
PPC_FUNC(sub_8225D850)
{
    if (!(FixesEnabled() & kShake) || !ac6::gpu::Unlocked())
    {
        static const bool trace = [] { const char* v = std::getenv("AC6_FPS_TRACE"); return v && *v == '1'; }();
        const uint32_t out = ctx.r4.u32;
        const float x = trace ? LoadFloat(out) : 0.0f, y = trace ? LoadFloat(out + 4) : 0.0f;
        __imp__sub_8225D850(ctx, base);
        if (trace && (LoadFloat(out) != x || LoadFloat(out + 4) != y))
            ac6::log::Info("FPS trace: shake (game's) frame {} raw {:.4f} {:.4f}", ac6::gpu::RendererFrame(), LoadFloat(out) - x, LoadFloat(out + 4) - y);
        return;
    }
    const uint32_t camera = ctx.r3.u32, out = ctx.r4.u32;
    float before[3];
    for (int i = 0; i < 3; i++)
        before[i] = LoadFloat(out + 4 * i);
    __imp__sub_8225D850(ctx, base);
    static std::mutex mutex;
    static std::unordered_map<uint32_t, ShakeSamples> samples;
    std::lock_guard lock(mutex);
    ShakeSamples& s = samples[camera];
    const uint64_t frame = ac6::gpu::RendererFrame();
    float shake[3];
    for (int i = 0; i < 3; i++)
        shake[i] = LoadFloat(out + 4 * i) - before[i];
    if (frame != s.frame)
    {
        if (frame != s.frame + 1)   // not shaken last frame: start from this sample
            for (int i = 0; i < 3; i++)
                s.previous[i] = s.current[i] = shake[i];
        s.frame = frame;
        if (Tick30())
            for (int i = 0; i < 3; i++)
            {
                s.previous[i] = s.current[i];
                s.current[i] = shake[i];
            }
    }
    const double t = std::clamp(Phase30(), 0.0, 1.0);
    for (int i = 0; i < 3; i++)
        StoreFloat(out + 4 * i, float(before[i] + s.previous[i] + (s.current[i] - s.previous[i]) * t));
    static const bool trace = [] { const char* v = std::getenv("AC6_FPS_TRACE"); return v && *v == '1'; }();
    if (trace && (shake[0] != 0.0f || shake[1] != 0.0f))
        ac6::log::Info("FPS trace: shake frame {} cam {:08X} t {:.3f} raw {:.4f} {:.4f} shown {:.4f} {:.4f}", frame, camera, t, shake[0], shake[1],
            s.previous[0] + (s.current[0] - s.previous[0]) * t, s.previous[1] + (s.current[1] - s.previous[1]) * t);
}

// ---- Missile fuses: sub_822D0BD8 (contact capsule, then proximity) --------------------------------------------------
// Sampled at the missile's current position on every update. Measured in mission 01: guidance is rate-correct (the
// lead uses the target's velocity in m/s, the turn limit rate x dt) and as many missiles reach their targets at 60 FPS
// as at 30, but the fuse, sampled twice as often, fires as the missile crosses its radius: the last distances 34-40 m
// at 60 FPS against 29-35 m at 30. AC6_MISSILE_FUSE_30HZ=1 (opt-in: whether the damage depends on that distance is not
// known): unlocked, the fuse is only sampled on the console's 30 Hz ticks, so missiles detonate as deep inside the
// radius as on the console.
extern "C" PPC_FUNC(__imp__sub_822D0BD8);
PPC_FUNC(sub_822D0BD8)
{
    static const bool fuse30 = [] { const char* v = std::getenv("AC6_MISSILE_FUSE_30HZ"); return v && *v == '1'; }();
    if (fuse30 && ac6::gpu::Unlocked() && !Tick30())
    {
        ctx.r3.u64 = 0;
        return;
    }
    __imp__sub_822D0BD8(ctx, base);
}

// ---- The hangar camera's move to a new aircraft: CSelectAircraftCamera slot 19, sub_82142CE0 -------------------------
// State 2 (requested by slot 36 on an aircraft change) advances its progress p by dt / T but blends the lookat by 2p
// (0x82142E30) and the eye by p (0x82142E94) once per update: tuned for the console's 30 Hz scene updates, too fast once
// the scene updates every frame. After each weight's computation (f0), with the update's time f30:
// w = 1 - (1 - w)^(30 f30).
namespace {
double HangarBlend(double w, double dt)
{
    if (!(FixesEnabled() & kMenus) || !ac6::gpu::Unlocked() || !(w > 0.0 && w < 1.0))
        return w;
    return 1.0 - std::pow(1.0 - w, std::clamp(dt * 30.0, 0.01, 1.0));
}
}
void FpsHangarCamLookat(PPCRegister& f0, PPCRegister& f30) { f0.f64 = float(HangarBlend(f0.f64, f30.f64)); }
void FpsHangarCamEye(PPCRegister& f0, PPCRegister& f30) { f0.f64 = float(HangarBlend(f0.f64, f30.f64)); }

// ---- Model motions stepped per update: sub_822EEC88 (a motion channel's transform update) -----------------------------
// A channel samples its animation at [ch+36] and advances by its step [ch+56] (the load at 0x822EED14, the argument of
// CX360Motion vt+48) once per update. dt-driven channels ([ch+66] bit 0) get [ch+56] = dt x 30 each update (rate-correct).
// The others keep a fixed step set by the game, tuned for its update rate: e.g. the hangar door at "Deploy" (2.0 on the
// two door models' channels, sub_8214B590), advanced by CSelectAircraftManager slot 39 once per 30 Hz scene update on
// the console, and mission unit models given +-2.5 by their state handlers (0x822EC350 / 0x822EC8D4 / 0x822ECE04),
// advanced once per mission frame. Unlocked, a fixed step is scaled to the update's share of the console's: inside
// slot 39 (sub_821491F8, f1 = the scene time since its last update) by that time x 30, elsewhere by the frame step
// ratio.
namespace { thread_local double t_motionFactor = -1.0; }
extern "C" PPC_FUNC(__imp__sub_821491F8);
PPC_FUNC(sub_821491F8)
{
    const double saved = t_motionFactor;
    t_motionFactor = (FixesEnabled() & kMenus) && ac6::gpu::Unlocked() ? std::clamp(ctx.f1.f64 * 30.0, 0.01, 1.0) : 1.0;
    __imp__sub_821491F8(ctx, base);
    t_motionFactor = saved;
}
void FpsModelChannelStep(PPCRegister& r31, PPCRegister& f1)
{
    if (!ac6::gpu::Unlocked() || (*ac6::g_memory.As<uint8_t>(r31.u32 + 66) & 1))
        return;
    const double factor = t_motionFactor >= 0.0 ? t_motionFactor : Ratio(kMotion);
    static const bool trace = [] { const char* v = std::getenv("AC6_FPS_TRACE"); return v && *v == '1'; }();
    if (trace && f1.f64 != 0.0)
    {
        static std::mutex mutex;
        static std::unordered_map<uint64_t, uint32_t> seen;
        std::lock_guard lock(mutex);
        const uint64_t key = (uint64_t(r31.u32) << 32) | uint32_t(int32_t(f1.f64 * 1000.0));
        if (seen.size() < 4000 && seen[key]++ == 0)
            ac6::log::Info("FPS trace: motion channel {:08X} step {:.3f} factor {:.3f} ({})", r31.u32, f1.f64, factor,
                t_motionFactor >= 0.0 ? "hangar" : "frame");
    }
    if (factor != 1.0)
        f1.f64 = float(f1.f64 * factor);
}

// ---- Air / rain / snow streaks: CEffectGeneratorAir slot 37 sub_822BA3F0, Rain / Snow sub_822BFAE0 --------------------
// The streak field around the camera drifts by [g+176] -= 10 x [g+144] per call, [g+144] being the camera's displacement
// per 1/30 s: 10x the flight speed on the console, fps / 30 times that unlocked. After the 10.0 load (0x822BA724,
// 0x822BFE8C): x R.
void FpsAirDrift(PPCRegister& f0)
{
    const double r = Ratio(kEffects);
    if (r != 1.0)
        f0.f64 = float(f0.f64 * r);
}

// ---- The afterburner flame's flicker: AfterBurner slot 37 sub_822B9FD0, BurnerLid sub_822BAA48 -----------------------
// An accumulator gains dt while below 0.03. At or above it, 0.03 is taken off (without that frame's dt) and the flame's
// random parameters are re-rolled: every second frame on the console (15 Hz), about 29 Hz at 144 fps. After the 0.03
// load (0x822BA2A4, dt f28, 0x822BAAA8, dt f31): the threshold becomes 1/15 - dt, firing on the frame that completes the
// console's period and keeping the time past it.
void FpsBurnerFlicker(PPCRegister& f13, PPCRegister& dt)
{
    if ((FixesEnabled() & kEffects) && ac6::gpu::Unlocked())
        f13.f64 = float(std::max(0.0, 1.0 / 15.0 - dt.f64));
}

// Diagnostics (AC6_FPS_TRACE): NU::Draw's object pass callback sub_82333218 (request { id, arg }): the first 40
// requests with an id of 0x0781xxxx (the hangar shadow map's two passes are 0x0781101C / 0x0781101D), with their
// arg and frame.
extern "C" PPC_FUNC(__imp__sub_82333218);
PPC_FUNC(sub_82333218)
{
    static const bool trace = [] { const char* v = std::getenv("AC6_FPS_TRACE"); return v && *v == '1'; }();
    if (trace)
    {
        const uint32_t id = ac6::g_memory.As<ac6::be<uint32_t>>(ctx.r4.u32)->get();
        static std::atomic<uint32_t> logged{ 0 };
        if ((id & 0xFFFF0000u) == 0x07810000u && logged.fetch_add(1) < 40)
            ac6::log::Info("FPS trace: object pass id {:08X} arg {:08X} frame {}", id,
                ac6::g_memory.As<ac6::be<uint32_t>>(ctx.r4.u32 + 4)->get(), ac6::gpu::RendererFrame());
    }
    __imp__sub_82333218(ctx, base);
}


// The picture-in-picture camera's aim: the point it looks at, [A+10160..68], chases the target's position
// T ([player+576] while [player+640] is set, else the target's [+80]) by x += (T - x) k dt once a frame. The target has moved
// v dt by then, so the point trails it by v dt (1 - a) / a (a = k dt): constant at the console's even 30 FPS (v / 120 at
// k = 24), but with uneven frame times it changes every frame and the target slides back and forth in the picture (measured
// at 2560x1440: 45-51 px rms with one frame in six held 25 ms, 7.6 px at 60 FPS, 5.2 px locked at 30). So the follow
// is solved exactly for a target moving steadily over the time it moved, dx/dt = l (T(t) - x): x' = T' - v / l +
// (x - T + v / l) e^-l s, v = (T' - T) / s, and the point trails by v / l whatever the frame times, l = 30 a / (1 - a) for the
// console's a = k / 30 keeps the console's trailing distance (l = 120 / s at k = 24, the target sat at 170 px of the
// picture's 400 against 168 locked at 30). T is the display pose, saved before the frame's steps, so it moved over the
// previous frame's time: s is that (this frame's time would leave 25-28 px on the held frames). That gives 0.55 px rms
// with the held frames, 0.54 at 60 FPS. The guest's own step is then none (FpsSubCameraFollow). Its gain ramp, activation and
// snaps are the game's. A new target starts from rest.
namespace {
struct SubCameraTrack { uint32_t target = 0; double last[3] = {}; double lastDt = 0.0; bool valid = false; };
SubCameraTrack s_subCameraTrack;
}

extern "C" PPC_FUNC(__imp__sub_822F1090);
PPC_FUNC(sub_822F1090)
{
    const uint32_t a = ctx.r3.u32;
    const double dt = ctx.f1.f64;
    SubCameraTrack& track = s_subCameraTrack;
    const uint32_t root = ac6::g_memory.As<ac6::be<uint32_t>>(0x826E4E54)->get();
    const uint32_t units = root ? ac6::g_memory.As<ac6::be<uint32_t>>(root + 0x29FC4)->get() : 0;
    const uint32_t player = units ? ac6::g_memory.As<ac6::be<uint32_t>>(units + 0x1008)->get() : 0;
    const uint32_t target = player ? ac6::g_memory.As<ac6::be<uint32_t>>(player + 692)->get() : 0;
    const bool exact = (FixesEnabled() & kSubCamera) && a && target && dt > 0.0 && dt <= 0.25 &&
        ac6::g_memory.As<ac6::be<uint32_t>>(a + 64)->get() != 0;   // active (an inactive one is activated inside)
    if (!exact)
    {
        track.valid = false;
        __imp__sub_822F1090(ctx, base);
        return;
    }
    // Where the guest reads T (0x822F1188..0x822F11A8).
    const uint32_t point = ac6::g_memory.As<ac6::be<uint32_t>>(player + 640)->get() ? player + 576 : target + 80;
    double t[3];
    for (int i = 0; i < 3; i++)
        t[i] = LoadFloat(point + i * 4);
    if (!track.valid || track.target != target)
    {
        track.target = target;
        std::memcpy(track.last, t, sizeof(t));
        track.lastDt = dt;
        track.valid = true;
    }
    // T is the pose saved before the frame's steps (the display pose): it moved over the previous frame's time, so the
    // follow is solved over that (measured: ~14 m a frame whatever this frame's dt, 25 m in the frame after a 30 ms one).
    const double span = track.lastDt;
    track.lastDt = dt;
    // The console's per-frame approach at the current gain, as a rate.
    const double k = LoadFloat(a + 10180);
    const double approach = std::clamp(k / 30.0, 0.0, 0.999);
    if (approach > 0.0)
    {
        const double rate = 30.0 * approach / (1.0 - approach);
        const double e = std::exp(-rate * span);
        for (int i = 0; i < 3; i++)
        {
            const double lag = (t[i] - track.last[i]) / span / rate;   // v / l
            const double x = LoadFloat(a + 10160 + i * 4);
            StoreFloat(a + 10160 + i * 4, float(t[i] - lag + (x - track.last[i] + lag) * e));
        }
    }
    std::memcpy(track.last, t, sizeof(t));
    t_subCameraExact = true;
    __imp__sub_822F1090(ctx, base);
    t_subCameraExact = false;
}

