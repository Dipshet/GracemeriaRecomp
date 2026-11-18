// In-engine cutscenes at the unlocked frame rate (AC6_CUTSCENE_UNLOCK, on by default).
//
// The cutscene sequencer (pump sub_823659F0 from the managers' Exec, apu/cutscene_sync.cpp) evaluates the timeline at
// its time [inner+36], a float frame, then advances it by its step [inner+44] (1.0 in play, 0 while held) once per
// call (sub_82367588, carrying the fraction across segments). Evaluation runs the record block of frame trunc(time):
// every record rebuilds per-frame state (the render manager clears it before each pump), so running a block on
// several calls in a row is what the design expects. Unlocked, each call advances S timeline frames (S = R, the frame
// step ratio of game/frame_rate_fixes.h, at most 1: a larger step would skip a block, and with it a cut's reset
// record), and catching up with the soundtrack is still done by extra calls with S = 1.
//
// A fractional step alone keeps the motion at 30 Hz: actors, the camera and the post-effects are sampled at each
// record's integer frame (payload frame - 1), not at the timeline time. So inside evaluation the frame's fraction is
// added where that integer becomes the sampling time (the motion samplers interpolate between keys). That assumes a
// record's frame goes up by one per block (inferred from the code, not confirmed).
//
// Everything here is the identity unless the cutscene frame is unlocked: S = R = 1 and the fraction is 0.
#include "game/frame_rate_fixes.h"
#include "gpu/frame_rate.h"
#include "gpu/renderer.h"
#include "log.h"
#include "endian.h"
#include "memory.h"
#include "ppc.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace {

// The cutscene managers, their sequencer and its hooks all run on the game's main thread.
double s_step = 1.0;           // this Exec call's timeline step
double s_advanced = 0.0;       // timeline frames advanced since TakeCutsceneFramesAdvanced
bool s_inEval = false;         // inside the timeline's evaluation
double s_subframe = 0.0;       // the evaluated time's fraction
double s_evalTime = 0.0;
uint32_t s_applied = 0, s_skipped = 0;   // AC6_CUTSCENE_TRACE counts

bool Active() { return ac6::gpu::CutsceneUnlock() && ac6::gpu::Unlocked(); }

// The per-presented-frame ratio, for cutscene things that run once per frame rather than once per Exec.
double FrameRatio() { return ac6::gpu::CutsceneUnlock() ? ac6::game::FrameStepRatio() : 1.0; }

float LoadFloat(uint32_t address) { return ac6::g_memory.As<ac6::be<float>>(address)->get(); }

// A sampling time that is a whole record frame gets the evaluated time's fraction, clamped to the animation's last
// key ([anim+40] = its length in frames) where the sampler has the animation at hand.
void AddSubframe(PPCRegister& f1, uint32_t anim)
{
    if (!s_inEval)
        return;
    if (s_subframe == 0.0 || f1.f64 != std::floor(f1.f64))
    {
        s_skipped++;
        return;
    }
    double t = f1.f64 + s_subframe;
    if (anim)
    {
        // An integer frame count (the camera record compares it with `cmplw`, 0x82125B24).
        const double length = double(ac6::g_memory.As<ac6::be<uint32_t>>(anim + 40)->get());
        if (!(length > 1.0))
        {
            s_skipped++;
            return;
        }
        t = std::min(t, length - 1.0);
    }
    f1.f64 = float(t);
    s_applied++;
}

} // namespace

namespace ac6::game {

void SetCutsceneStep(double step) { s_step = std::clamp(step, 0.0, 1.0); }

double TakeCutsceneFramesAdvanced()
{
    const double advanced = s_advanced;
    s_advanced = 0.0;
    return advanced;
}

} // namespace ac6::game

// 0x823675E8 (before `fadds f31,f0,f13`, sub_82367588): the advance, time + step. Its only caller is the cutscene
// sequencer's inner (vt +68).
void FpsCutsceneStep(PPCRegister& f13)
{
    if (s_step != 1.0)
        f13.f64 = float(f13.f64 * s_step);
    s_advanced += f13.f64;
}

// 0x823665B4 (after `lfs f1,36(r31)`, the time handed to the evaluation): the evaluation's scope and fraction.
// 0x823665CC (the first instruction after its call): the end of the scope.
void FpsCutsceneEvalBegin(PPCRegister& f1)
{
    // AC6_CUTSCENE_SUBFRAME=0: whole record frames only (the motion then steps at 30 Hz, for comparisons).
    static const bool subframes = [] { const char* v = std::getenv("AC6_CUTSCENE_SUBFRAME"); return !v || *v != '0'; }();
    s_inEval = true;
    s_subframe = subframes && Active() ? f1.f64 - std::floor(f1.f64) : 0.0;
    s_evalTime = f1.f64;
    s_applied = s_skipped = 0;
}
void FpsCutsceneEvalEnd()
{
    s_inEval = false;
    // AC6_CUTSCENE_TRACE=1: each evaluation's time, step and how many sampling times took the fraction.
    static const bool trace = [] { const char* v = std::getenv("AC6_CUTSCENE_TRACE"); return v && *v == '1'; }();
    if (trace)
        ac6::log::Info("Cutscene eval: frame {} time {:.3f} step {:.3f} fraction added {} skipped {}", ac6::gpu::RendererFrame(),
            s_evalTime, s_step, s_applied, s_skipped);
}

// 0x82118BAC (before `fmr f31,f1`, sub_82118B98, the motion sampler, its 8 callers are cutscene samplers): f1 is the
// sampling time, r4 the animation.
void FpsCutsceneSampleTime(PPCRegister& f1, PPCRegister& r4) { AddSubframe(f1, r4.u32); }

// Before the 9 cutscene calls of sub_82119B20 (0x82126150 ... 0x821274B8, not at its entry, also a general
// motion API): f1 the sampling time, r3 the animation.
void FpsCutsceneSampleCall(PPCRegister& f1, PPCRegister& r3) { AddSubframe(f1, r3.u32); }

// 0x8212AAE0, 0x8212AC30, 0x82136904 (skin poses, types 3 / 4 / 1) and 0x82135BA4 (record 0x2005 mode 4, the unit
// model's animation time), each after `frsp f1,f0`: no animation at hand to clamp to.
void FpsCutscenePoseTime(PPCRegister& f1) { AddSubframe(f1, 0); }

// The 1/29.97 s clock (0x8205D764, one NTSC frame) per Exec call: x S. 0x82184C58 (CX360DemoManagerDD, in-mission
// cutscenes) and 0x8218616C (CX360DemoManagerEM, standalone ones), the accompaniment clock's increment (before, f1).
// 0x821348B8 / 0x82134940 (sub_821347F0's fade timer, after, f0). 0x82134A0C (its motion, after, f1). 0x82134A3C (its
// 0.02 phase step, after, f13). 0x821353B8 (a propeller's spin, after, f1).
void FpsCutsceneClock(PPCRegister& f)
{
    if (s_step != 1.0)
        f.f64 = float(f.f64 * s_step);
}

// Once per presented frame: 0x8219566C (the demo task's accompaniment clock, after, f1), 0x82131CC0 and 0x82131D14
// (the cutscene sea's y and w steps, before, f13): x R.
void FpsCutsceneFrameStep(PPCRegister& f)
{
    const double r = FrameRatio();
    if (r != 1.0)
        f.f64 = float(f.f64 * r);
}

// AC6_CUTSCENE_TRACE=1: the camera record's sample (sub_82125A80, after the sampler call at 0x82125B58 returns at
// 0x82125B5C): position r1+96..104, rotation r1+108..120, the record frame (r28) and the evaluated fraction.
void FpsCutsceneCameraTrace(PPCRegister& r1, PPCRegister& r28)
{
    static const bool trace = [] { const char* v = std::getenv("AC6_CUTSCENE_TRACE"); return v && *v == '1'; }();
    if (!trace)
        return;
    const uint32_t s = r1.u32;
    ac6::log::Info("Cutscene camera: frame {} record {} fraction {:.3f} pos {:.3f} {:.3f} {:.3f} rot {:.4f} {:.4f} {:.4f} {:.4f}",
        ac6::gpu::RendererFrame(), r28.u32, s_subframe, LoadFloat(s + 96), LoadFloat(s + 100), LoadFloat(s + 104),
        LoadFloat(s + 108), LoadFloat(s + 112), LoadFloat(s + 116), LoadFloat(s + 120));
}
