// In-engine cutscenes follow their soundtrack.
//
// The cutscene managers' per-frame Exec (sub_82184460 and sub_821856F8) run a setup state, then a playback
// state (the state word at +0x10000+6900 or +0x10000+14084 holds 1) in which every call advances the cutscene's
// sequence by one 30 FPS frame (sub_823659F0 steps it at its set rate, the frame time passed in f1 does
// not move it). The soundtrack, a multichannel voice, is cued by the sequence and then plays on in real
// time, so a render stall (shader compiles) after the cue leaves the picture behind for good, while a
// stall before it does no harm.
//
// The soundtrack is the master from the moment a multichannel voice starts during playback (the anchor,
// a later one re-anchors): the frames due from the samples rendered since then are compared with the
// playback calls made since then. From two frames behind, extra calls run in the same frame (at most 3, so a
// long stall catches up over a few frames instead of one jump). From two frames ahead (an unpaced ac6.exe runs
// at 60 FPS), a frame's call is held back. Without an anchor nothing is corrected. AC6_CUTSCENE_RESYNC=0
// disables it. With AC6_CUTSCENE_UNLOCK each call steps R timeline frames, R being the frame time x 30, at most
// 1 (game::FrameStepRatio, game/cutscene_unlock.cpp), and the frames are counted as the sequencer advanced
// them. Catching up still runs whole frames (a larger step would skip a frame's records).
//
// The CX360DemoManagerEM manager's setup (sub_821857B0) requests the cutscene's soundtrack (CNuSound vtable +404, track =
// a table entry by cutscene index) and then, in state 10, waits for the stream to report ready (+408)
// with no timeout: a request that is lost or a stream that never starts leaves the cutscene frozen. As
// the title update does, the request is repeated every 3 s of waiting.
// AC6_DEMO_BGM_RETRY=0 disables it. AC6_DEMO_BGM_DROP=1 ignores the first soundtrack
// request of each cutscene setup, as a lost request would (reproduces the hang on demand, testing only).
#include "apu/audio_driver.h"
#include "apu/xma.h"
#include "endian.h"
#include "game/frame_rate_fixes.h"
#include "gpu/frame_rate.h"
#include "gpu/renderer.h"
#include "kernel/function.h"
#include "log.h"
#include "memory.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>

namespace {

constexpr int64_t kSamplesPerFrame = ac6::apu::kAudioSampleRate / 30;
constexpr int64_t kTolerance = 2;   // frames
constexpr int kMaxExtraFrames = 3;
// More than this many frames presented without a call mean the cutscene ended. A stall inside one presents
// no frames however long it lasts, so this is measured in frames, not time.
constexpr uint64_t kSessionGapFrames = 2;
constexpr uint32_t kPlaying = 1;

struct Manager
{
    uint32_t function;
    uint32_t stateOffset;   // the state word Exec switches on (copied from here into the current one)
};
constexpr Manager kDemoManager{ 0x82184460, 0x10000 + 6900 };     // CAce6DemoManager
constexpr Manager kDemoManagerEM{ 0x821856F8, 0x10000 + 14084 };  // CX360DemoManagerEM

struct Session
{
    uint32_t manager = 0;
    uint64_t lastCallFrame = 0;
    uint32_t anchorStarts = 0;     // LastMultichannelStart().count at the anchor (0: none yet)
    uint32_t sessionStarts = 0;    // ... when the session began
    uint64_t anchorSamples = 0;
    double framesSinceAnchor = 0;  // timeline frames played since the anchor (one per call unless unlocked)
};

Session s_session;   // only the game's main thread runs the managers

bool Enabled()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_CUTSCENE_RESYNC"); return !v || *v != '0'; }();
    return enabled;
}

uint32_t State(const PPCContext& ctx, const Manager& m)
{
    return ac6::g_memory.As<ac6::be<uint32_t>>(ctx.r3.u32 + m.stateOffset)->get();
}

// One Exec call advancing the timeline by `step` frames (game/cutscene_unlock.cpp). Returns the frames advanced.
double Run(PPCContext& ctx, uint8_t* base, void (*original)(PPCContext&, uint8_t*), double step)
{
    ac6::game::SetCutsceneStep(step);
    ac6::game::TakeCutsceneFramesAdvanced();
    original(ctx, base);
    ac6::game::SetCutsceneStep(1.0);
    return ac6::game::TakeCutsceneFramesAdvanced();
}

void Exec(PPCContext& ctx, uint8_t* base, void (*original)(PPCContext&, uint8_t*), const Manager& m)
{
    ac6::gpu::NoteCutsceneFrame();   // cutscenes run at 30 FPS when AC6_CUTSCENE_UNLOCK=0 (gpu/frame_rate.h)
    // Unlocked frames step the timeline by FrameStepRatio() frames a call (game/cutscene_unlock.cpp), 1 otherwise.
    const bool unlocked = ac6::gpu::CutsceneUnlock() && ac6::gpu::Unlocked();
    const double step = unlocked ? ac6::game::FrameStepRatio() : 1.0;
    if (!Enabled())
    {
        Run(ctx, base, original, step);
        return;
    }
    const uint64_t frame = ac6::gpu::RendererFrame();
    const ac6::apu::MultichannelStart voice = ac6::apu::LastMultichannelStart();
    if (s_session.manager != m.function || frame > s_session.lastCallFrame + kSessionGapFrames)
    {
        s_session = Session{ m.function, frame, 0, voice.count };
        ac6::log::Info("Cutscene resync: {:08X} started at frame {}", m.function, frame);
    }
    s_session.lastCallFrame = frame;

    const bool playing = State(ctx, m) == kPlaying;
    if (playing && voice.count != s_session.sessionStarts && voice.count != s_session.anchorStarts)
    {
        s_session.anchorStarts = voice.count;
        s_session.anchorSamples = voice.samples;
        s_session.framesSinceAnchor = 0;
        ac6::log::Info("Cutscene resync: soundtrack started at frame {}", frame);
    }
    if (!playing || !s_session.anchorStarts)
    {
        Run(ctx, base, original, step);
        return;
    }

    // Timeline frames due from the samples played since the anchor: whole frames as on the console's cadence, or
    // fractional ones when each call steps a fraction.
    const double samples = double(ac6::apu::RenderedSamples() - s_session.anchorSamples);
    const double due = unlocked ? samples / double(kSamplesPerFrame) : std::floor(samples / double(kSamplesPerFrame));
    const double behind = due - s_session.framesSinceAnchor;
    if (behind <= -double(kTolerance))
    {
        // Ahead of the sound: this frame's step waits.
        static uint32_t logged = 0;
        if (logged++ < 16)
            ac6::log::Info("Cutscene resync: {:.1f} frame(s) ahead of the soundtrack, holding", -behind);
        return;
    }
    const PPCContext entry = ctx;
    const double advanced = Run(ctx, base, original, step);
    s_session.framesSinceAnchor += unlocked ? advanced : 1.0;
    if (behind < double(kTolerance))
        return;
    // Behind: extra calls of one whole frame each (a larger step would skip a frame's records).
    int extra = 0;
    while (extra < kMaxExtraFrames && s_session.framesSinceAnchor < due - (unlocked ? step : 0.0) && State(entry, m) == kPlaying)
    {
        ctx = entry;
        const double more = Run(ctx, base, original, 1.0);
        s_session.framesSinceAnchor += unlocked ? more : 1.0;
        extra++;
    }
    static uint32_t logged = 0;
    if (logged++ < 64)
        ac6::log::Info("Cutscene resync: {:.1f} frame(s) behind the soundtrack, ran {} extra", behind, extra);
}

// --- The soundtrack wait of the CX360DemoManagerEM manager's setup -----------------------------------

constexpr uint32_t kSetupStateOffset = 0x10000 + 14088;   // sub_821857B0's state word
constexpr uint32_t kWaitingForSoundtrack = 10;
constexpr uint32_t kCutsceneIndexOffset = 0x10000 + 14064;
constexpr uint32_t kSoundPointer = 0x823F9B28;            // CAce6Sound* (a CNuSound)
constexpr uint32_t kTrackTable = 0x82691FD8;              // 18 track ids by cutscene index
constexpr uint32_t kRequestTrackSlot = 404, kStreamReadySlot = 408;
constexpr double kRetrySeconds = 3.0;

struct SoundtrackWait
{
    uint32_t manager = 0;     // 0: not waiting
    double waited = 0;        // since the last request
    double total = 0;
    uint32_t retries = 0;
};
SoundtrackWait s_wait;
bool s_dropNext = false;   // AC6_DEMO_BGM_DROP: the next request is ignored

bool RetryEnabled()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_DEMO_BGM_RETRY"); return !v || *v != '0'; }();
    return enabled;
}

uint32_t Load32(uint32_t address) { return ac6::g_memory.As<ac6::be<uint32_t>>(address)->get(); }

void SetupStep(PPCContext& ctx, uint8_t* base, void (*original)(PPCContext&, uint8_t*))
{
    const uint32_t manager = ctx.r3.u32;
    const uint32_t state = Load32(manager + kSetupStateOffset);
    const bool waiting = state == kWaitingForSoundtrack;
    if (!waiting)
    {
        if (state == 0)
            s_dropNext = true;   // a new setup: its first request is the one AC6_DEMO_BGM_DROP drops
        s_wait = SoundtrackWait{};
        original(ctx, base);
        return;
    }
    if (s_wait.manager != manager)
        s_wait = SoundtrackWait{ manager };
    const double dt = ctx.f1.f64;
    s_wait.waited += dt;
    s_wait.total += dt;
    if (RetryEnabled() && s_wait.waited >= kRetrySeconds)
    {
        s_wait.waited = 0;
        const uint32_t sound = Load32(kSoundPointer);
        const uint32_t vtable = sound ? Load32(sound) : 0;
        if (vtable >= 0x82000000u && vtable < 0x83000000u &&
            !(ac6::GuestCall<uint32_t>(Load32(vtable + kStreamReadySlot), sound) & 0xFF))
        {
            const int32_t index = int32_t(Load32(manager + kCutsceneIndexOffset));
            const uint32_t track = Load32(kTrackTable + 4 * uint32_t(index >= 0 && index <= 17 ? index : 0));
            ac6::GuestCall(Load32(vtable + kRequestTrackSlot), sound, track);
            s_wait.retries++;
            ac6::log::Warn("Cutscene soundtrack (track {}) not ready after {:.1f} s: requested again", track, s_wait.total);
        }
    }
    original(ctx, base);
    // The manager stops calling the setup once it leaves the wait.
    if (Load32(manager + kSetupStateOffset) != kWaitingForSoundtrack)
    {
        ac6::log::Info("Cutscene soundtrack ready after {:.2f} s{}", s_wait.total,
            s_wait.retries ? std::format(" ({} repeated request(s))", s_wait.retries) : std::string());
        s_wait = SoundtrackWait{};
    }
}

} // namespace

extern "C" PPC_FUNC(__imp__sub_821857B0);   // CX360DemoManagerEM setup step
PPC_FUNC(sub_821857B0) { SetupStep(ctx, base, __imp__sub_821857B0); }

// CNuSound::request_stream_track (vtable +404), for AC6_DEMO_BGM_DROP.
extern "C" PPC_FUNC(__imp__sub_82124DD0);
PPC_FUNC(sub_82124DD0)
{
    static const bool drop = [] { const char* v = std::getenv("AC6_DEMO_BGM_DROP"); return v && *v == '1'; }();
    if (drop && s_dropNext)
    {
        s_dropNext = false;
        ac6::log::Warn("AC6_DEMO_BGM_DROP: soundtrack request (track {}) ignored", ctx.r4.u32);
        return;
    }
    __imp__sub_82124DD0(ctx, base);
}

extern "C" PPC_FUNC(__imp__sub_82184460);   // CAce6DemoManager::Exec
extern "C" PPC_FUNC(__imp__sub_821856F8);   // CX360DemoManagerEM::Exec

PPC_FUNC(sub_82184460) { Exec(ctx, base, __imp__sub_82184460, kDemoManager); }
PPC_FUNC(sub_821856F8) { Exec(ctx, base, __imp__sub_821856F8, kDemoManagerEM); }
