// DualSense adaptive triggers, modelled on Ace Combat 8's own effects. In flight:
//  - the throttle (RT): no effect (the trigger's own spring).
//  - the brake (LT): none below 300 km/h, from 300 km/h resistance over the last 30% of the pull only, strength 1 of 8,
//    2 from 1420 km/h, 3 from 2000 km/h.
//  - a High-G turn: both triggers rattle, amplitude 2 of 8 at 32 Hz, all the pull.
//  - the buffet before a stall: the brake alone rattles over its last 30%, amplitude 2 at 8 Hz, a stick shaker, not in
//    the full stall.
// With the game's THROTTLE/YAW on REVERSE the triggers are the rudder pedals (the throttle and the brake are on the
// bumpers), so both feel the same: a rudder load the same over all the pull, the brake's strength at that speed (none
// below 300 km/h, then 1, 2 from 1420 km/h, 3 from 2000), the High-G rattle as above, the buffet's shaker on both, over
// all the pull. (A load growing with the pull would need its force sent again as the trigger moves, which feels like a
// knock on the trigger.) The same with Advanced controls on or off (which removes the game's dead zone from the
// triggers' yaw: game/flight_input.cpp).
// Sent as the controller's own effect packet through SDL (SDL_SendGamepadEffect, its PS5 HIDAPI driver: USB or Bluetooth,
// not through Steam Input, which sits in between). Only when the effect changes. Cleared in menus, pauses, cutscenes,
// while the settings overlay is open, and with AC6_ADAPTIVE_TRIGGERS=0 (settings controller.adaptive_triggers, default
// on, live). The strengths can be tuned in ac6_settings.toml's [env] (Tuning below).
//
// The game's state, read each frame: the player actor [[0x826E4E54] + 0x29FC4] + 0x1008,
// its flight model [actor+4912]. The High-G turn: the flight model's mode word [fm+332] bit 0x10 (sub_82282470 mirrors the
// drift sub-object's active flag into it). The buffet: sub_82281750's stall factor [fm+344], 0 at the buffet speed
// [fm+940] (346 km/h in mission 01's plane) rising as ((buffet - speed) / (buffet - stall))^2 to 1 at the stall speed
// [fm+944] (150), held to 0-1, with the mode word's bit 0x4 set while it is above 0 (bits 0x4 and 0x8 both come on at
// the buffet speed: neither is the full stall). The full stall: the factor at 1 (once stalled the game raises both speeds
// by 1.2, so it holds to about 180 km/h). The speed: [actor+2192], km/h (sub_8224D6E0 copies actor vt+124 there every
// frame, what the overspeed shake reads). In flight: the flight model's heartbeat (game/kbm_heartbeat.cpp,
// input/kbm.cpp InFlight).
#include "input/advanced.h"
#include "input/dualsense.h"
#include "input/kbm.h"
#include "endian.h"
#include "log.h"
#include "memory.h"
#include "settings.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string>
#include <utility>

namespace ac6::input::dualsense {

namespace {

// SDL's DualSense effect packet (SDL_hidapi_ps5.c DS5EffectsState_t, as its testcontroller sends it): the trigger effects
// at bytes 10 (right) and 21 (left), 11 each, applied when ucEnableBits1 has 0x04 (right) / 0x08 (left).
struct Effects
{
    uint8_t enableBits1, enableBits2;
    uint8_t rumbleRight, rumbleLeft;
    uint8_t headphoneVolume, speakerVolume, microphoneVolume, audioEnableBits, micLightMode, audioMuteBits;
    uint8_t rightTrigger[11];
    uint8_t leftTrigger[11];
    uint8_t unknown1[6];
    uint8_t ledFlags;
    uint8_t unknown2[2];
    uint8_t ledAnim, ledBrightness, padLights, ledRed, ledGreen, ledBlue;
};
static_assert(sizeof(Effects) == 47);

// A trigger's effect, in the controller's own modes (their encoding as the DualSense's documented trigger effect
// generators write them):
//  - off: 0x05.
//  - resistance (0x01, the older mode, finer than the zones' 8 steps): from `start` (0-255 of the pull) with `force`
//    0-255: bytes 1 and 2.
//  - zoned resistance ("feedback", 0x21): the pull in 10 zones, each a strength 1-8 (0 = none): bytes 1-2 the zones
//    that resist, 3-6 three bits each, the strength less one.
//  - vibration (0x26): from zone `start`, amplitude 1-8 at `frequency` Hz: bytes 1-2 the zones, 3-6 three bits of
//    amplitude each, byte 9 the frequency.
struct Trigger
{
    enum Kind : uint8_t { Off, Resistance, Zones, Vibration } kind = Off;
    std::array<uint8_t, 10> strength{};   // Zones: per tenth of the pull, 0-8
    uint8_t start = 0, amplitude = 0, frequency = 0;   // Vibration (start: Resistance too, 0-255)
    uint8_t force = 0;   // Resistance
    bool operator==(const Trigger&) const = default;
};

void Encode(const Trigger& t, uint8_t out[11])
{
    std::memset(out, 0, 11);
    uint32_t active = 0, values = 0;
    switch (t.kind)
    {
    case Trigger::Off:
        out[0] = 0x05;
        return;
    case Trigger::Resistance:
        if (t.force == 0)
            out[0] = 0x05;
        else
            out[0] = 0x01, out[1] = t.start, out[2] = t.force;
        return;
    case Trigger::Zones:
        for (uint32_t i = 0; i < 10; i++)
            if (t.strength[i] > 0)
            {
                active |= 1u << i;
                values |= uint32_t((std::min<uint8_t>(t.strength[i], 8) - 1) & 7) << (3 * i);
            }
        out[0] = 0x21;
        break;
    case Trigger::Vibration:
        for (uint32_t i = t.start; i < 10; i++)
        {
            active |= 1u << i;
            values |= uint32_t((std::clamp<uint8_t>(t.amplitude, 1, 8) - 1) & 7) << (3 * i);
        }
        out[0] = 0x26;
        out[9] = t.frequency;
        break;
    }
    out[1] = uint8_t(active), out[2] = uint8_t(active >> 8);
    out[3] = uint8_t(values), out[4] = uint8_t(values >> 8), out[5] = uint8_t(values >> 16), out[6] = uint8_t(values >> 24);
}

std::string Describe(const Trigger& t)
{
    if (t.kind == Trigger::Off)
        return "off";
    if (t.kind == Trigger::Vibration)
        return std::format("rattle {}x{} Hz from zone {}", t.amplitude, t.frequency, t.start);
    if (t.kind == Trigger::Resistance)
        return std::format("resistance {}", t.force);
    std::string out = "zones ";
    for (uint8_t s : t.strength)
        out += char('0' + s);
    return out;
}

uint32_t U32(uint32_t a)
{
    return a ? g_memory.As<be<uint32_t>>(a)->get() : 0;
}

float F32(uint32_t a)
{
    const uint32_t bits = U32(a);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

uint32_t PlayerActor()
{
    const uint32_t root = U32(0x826E4E54);
    const uint32_t manager = root ? U32(root + 0x29FC4) : 0;
    return manager ? U32(manager + 0x1008) : 0;
}

// The flight model's mode word [fm+332] (0 without one): bit 0x10 the High-G turn, 0x4 and 0x8 the stall factor above 0
// (both from the buffet speed, cleared on the ground, 0x40).
uint32_t Mode(uint32_t actor)
{
    const uint32_t fm = actor ? U32(actor + 4912) : 0;
    return fm ? U32(fm + 332) : 0;
}

bool HighG(uint32_t actor)
{
    return (Mode(actor) & 0x10) != 0;
}

// The stall factor [fm+344]: 0 above the buffet speed (and the altitude ceiling's), 1 in the full stall.
float StallFactor(uint32_t actor)
{
    const uint32_t fm = actor ? U32(actor + 4912) : 0;
    return fm && (U32(fm + 332) & 0x4) ? F32(fm + 344) : 0.0f;
}

// The buffet: the stall factor between 0 and 1. The full stall from 1 until it falls below 0.9 (no shaker flicker while
// the speed wavers at the stall's edge).
bool Buffet(uint32_t actor)
{
    static bool fullStall = false;
    const float f = StallFactor(actor);
    fullStall = f >= 0.999f || (fullStall && f >= 0.9f);
    return f > 0.0f && !fullStall;
}

// The game's THROTTLE/YAW is REVERSE: [profile + 19588] = 1, the active profile being root + 112 + index * 42976 with
// the index at [root + 112 + 0x243F8], as the player's input sample 0x822291C0 reads it.
bool Reverse()
{
    const uint32_t root = U32(0x826E4E54);
    if (!root)
        return false;
    const uint32_t index = U32(root + 112 + 0x243F8);
    return index < 4 && U32(root + 112 + index * 42976 + 19588) == 1;
}

bool Enabled()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<bool> value{ true };
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        const std::string v = settings::Get("AC6_ADAPTIVE_TRIGGERS");
        value.store(v != "0" && v != "false", std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    return value.load(std::memory_order_relaxed);
}

// The feel (AC8's effects), and its tuning by variables, in ac6_settings.toml's [env]
// (read when the settings change, a bad value keeps the default):
//  - AC6_TRIGGER_LIGHT = 0: a resistance, 0-255 (0 = none, as AC8), of the throttle and of the brake below the first
//    brake speed (the older resistance mode 0x01, finer than the zones).
//  - AC6_TRIGGER_BRAKE = "0000000111": the brake from the first brake speed, a strength 0-8 per tenth of the pull. Each
//    further brake speed adds 1 to its zones (AC8: 1, 2, 3 over the last three tenths).
//  - AC6_TRIGGER_BRAKE_SPEEDS = "300,1420,2000": the brake speeds, km/h, 1-3 of them rising: AC8's (its steps come at
//    1436 / 2001 km/h accelerating, 1412 / 1994 braking, none below 299-300). AC8's absolute speeds are kept: AC8's
//    F-22, the plane they come from, performs like AC6's, and the slower starter F-16C reaches step 2 only near its top
//    speed of about 1600 km/h.
//  - AC6_TRIGGER_HIGHG = "2,32": the High-G turn's rattle on both, amplitude 1-8, frequency in Hz (from zone 0).
//  - AC6_TRIGGER_SHAKER = "2,8,7": the buffet's rattle on the brake alone, amplitude, frequency, the first zone 0-9.
//  - AC6_TRIGGER_RUDDER = "1111111111": with THROTTLE/YAW on REVERSE, both triggers (the rudder pedals) from the first
//    brake speed, a strength 0-8 per tenth of the pull. Each further brake speed adds 1 (as the brake's).
struct Tuning
{
    uint8_t light = 0;
    std::array<uint8_t, 10> brake = { 0, 0, 0, 0, 0, 0, 0, 1, 1, 1 };
    std::array<uint8_t, 10> rudder = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    std::array<float, 3> brakeSpeeds = { 300.0f, 1420.0f, 2000.0f };
    size_t brakeSteps = 3;
    uint8_t highGAmplitude = 2, highGFrequency = 32;
    uint8_t shakerAmplitude = 2, shakerFrequency = 8, shakerStart = 7;
};

// "a,f[,z]": amplitude 1-8, frequency 1-255 Hz, (with `start`) the first zone 0-9.
void ParseRattle(const char* name, const std::string& v, uint8_t& amplitude, uint8_t& frequency, uint8_t* start = nullptr)
{
    if (v.empty())
        return;
    unsigned a = 0, f = 0, z = start ? *start : 0;
    const int n = std::sscanf(v.c_str(), "%u , %u , %u", &a, &f, &z);
    if ((n == 2 || (n == 3 && start)) && a >= 1 && a <= 8 && f >= 1 && f <= 255 && z <= 9)
    {
        amplitude = uint8_t(a), frequency = uint8_t(f);
        if (start)
            *start = uint8_t(z);
    }
    else
        log::Warn("DualSense adaptive triggers: {} \"{}\" is not amplitude,frequency{} (1-8,1-255{}): the default kept", name,
            v, start ? "[,first zone]" : "", start ? ",0-9" : "");
}

const Tuning& Tuned()
{
    static Tuning t;
    static uint64_t generation = UINT64_MAX;
    if (const uint64_t now = settings::Generation(); generation != now)
    {
        generation = now;
        t = {};
        if (const std::string v = settings::Get("AC6_TRIGGER_LIGHT"); !v.empty())
        {
            char* end = nullptr;
            const long n = std::strtol(v.c_str(), &end, 10);
            if (end && *end == 0 && n >= 0 && n <= 255)
                t.light = uint8_t(n);
            else
                log::Warn("DualSense adaptive triggers: AC6_TRIGGER_LIGHT \"{}\" is not 0-255: the default kept", v);
        }
        for (auto [name, zones] : { std::pair{ "AC6_TRIGGER_BRAKE", &t.brake }, std::pair{ "AC6_TRIGGER_RUDDER", &t.rudder } })
            if (const std::string v = settings::Get(name); !v.empty())
            {
                if (v.size() == 10 && std::all_of(v.begin(), v.end(), [](char c) { return c >= '0' && c <= '8'; }))
                    for (size_t i = 0; i < 10; i++)
                        (*zones)[i] = uint8_t(v[i] - '0');
                else
                    log::Warn("DualSense adaptive triggers: {} \"{}\" is not ten digits 0-8: the default kept", name, v);
            }
        if (const std::string v = settings::Get("AC6_TRIGGER_BRAKE_SPEEDS"); !v.empty())
        {
            float f[3] = {};
            const int n = std::sscanf(v.c_str(), "%f , %f , %f", &f[0], &f[1], &f[2]);
            bool ok = n >= 1 && f[0] >= 0.0f;
            for (int i = 1; i < n; i++)
                ok = ok && f[i] > f[i - 1];
            if (ok)
            {
                t.brakeSteps = size_t(n);
                for (int i = 0; i < n; i++)
                    t.brakeSpeeds[size_t(i)] = f[i];
            }
            else
                log::Warn("DualSense adaptive triggers: AC6_TRIGGER_BRAKE_SPEEDS \"{}\" is not 1-3 rising speeds: the default kept", v);
        }
        ParseRattle("AC6_TRIGGER_HIGHG", settings::Get("AC6_TRIGGER_HIGHG"), t.highGAmplitude, t.highGFrequency);
        ParseRattle("AC6_TRIGGER_SHAKER", settings::Get("AC6_TRIGGER_SHAKER"), t.shakerAmplitude, t.shakerFrequency,
            &t.shakerStart);
    }
    return t;
}

SDL_Gamepad* s_pad = nullptr;   // the pad the last effects went to
Trigger s_left, s_right;
bool s_sent = false;
bool s_buffetAfterHighG = false;   // a buffet entered in a High-G turn: no shaker (Update)

} // namespace

void Update(SDL_Gamepad* pad, bool overlayOpen)
{
    if (pad && SDL_GetGamepadType(pad) != SDL_GAMEPAD_TYPE_PS5)
        pad = nullptr;
    if (pad != s_pad)
    {
        s_pad = pad;
        s_sent = false;   // a new pad: its triggers are as it came
    }
    // AC6_TRIGGER_TRACE=1 (diagnostics): the state and the effects every 0.25 s, with or without a DualSense.
    static const bool trace = [] { const char* v = std::getenv("AC6_TRIGGER_TRACE"); return v && *v == '1'; }();
    if (!pad && !trace)
        return;
    Trigger left, right;
    const uint32_t actor = Enabled() && !overlayOpen && kbm::InFlight() ? PlayerActor() : 0;
    if (actor)
    {
        // The High-G turn's rattle first, then the buffet's shaker, else the throttle's and the brake's resistance. A
        // buffet already under way when a High-G turn ends gets no shaker until the stall factor is back at 0, so the
        // rattle ends without a shaker taking over. A buffet that starts after the turn has ended does shake: in AC6
        // the turn usually ends at about 410 km/h, above the 346 km/h buffet speed, so even a buffet that follows at
        // once starts in normal flight.
        // With REVERSE (the triggers the rudder pedals) both get the same: the rudder load in place of the throttle's and
        // the brake's, and the shaker on both over all the pull.
        // Each trigger's part: the game's layout (NORMAL: LT the brake, RT the throttle, REVERSE: both the
        // rudder), or, with Advanced controls on, what the trigger is bound to there (unbound: the game's layout).
        const Tuning& t = Tuned();
        const bool reverse = Reverse();
        enum class Part { Throttle, Brake, Rudder, None };
        auto partOf = [&](bool isRight) {
            switch (input::advanced::RoleOfTrigger(isRight))
            {
            case input::advanced::TriggerRole::Thrust: return Part::Throttle;
            case input::advanced::TriggerRole::Brake: return Part::Brake;
            case input::advanced::TriggerRole::Yaw: return Part::Rudder;
            case input::advanced::TriggerRole::Other: return Part::None;
            default: return reverse ? Part::Rudder : isRight ? Part::Throttle : Part::Brake;
            }
        };
        const Part leftPart = partOf(false), rightPart = partOf(true);
        const bool highG = HighG(actor);
        const bool buffet = Buffet(actor);   // (every frame: it keeps the full stall's state)
        s_buffetAfterHighG = highG || (s_buffetAfterHighG && StallFactor(actor) > 0.0f);
        const float speed = F32(actor + 2192);
        size_t step = 0;   // the brake speeds reached
        while (speed < 5000.0f && step < t.brakeSteps && speed >= t.brakeSpeeds[step])
            step++;
        // One trigger by its part: the High-G rattle on every part but Part::None. The buffet's shaker on the brake
        // (from its start zone) and the rudder (over all the pull). Else the throttle's light resistance, the brake's
        // zones (from the first brake speed) or the rudder's (none below it), nothing for Part::None.
        auto feel = [&](Part part) {
            Trigger g{};
            if (part == Part::None)
                return g;
            if (highG)
            {
                g.kind = Trigger::Vibration;
                g.amplitude = t.highGAmplitude;
                g.frequency = t.highGFrequency;
                return g;
            }
            if (buffet && !s_buffetAfterHighG && (part == Part::Brake || part == Part::Rudder))
            {
                g.kind = Trigger::Vibration;
                g.amplitude = t.shakerAmplitude, g.frequency = t.shakerFrequency;
                g.start = part == Part::Brake ? t.shakerStart : 0;
                return g;
            }
            g.kind = Trigger::Resistance;
            g.force = part == Part::Rudder ? 0 : t.light;
            if (step > 0 && part != Part::Throttle)
            {
                const auto& zones = part == Part::Brake ? t.brake : t.rudder;
                g = {};
                g.kind = Trigger::Zones;
                for (size_t i = 0; i < 10; i++)
                    g.strength[i] = zones[i] ? uint8_t(std::min<size_t>(zones[i] + step - 1, 8)) : 0;
            }
            return g;
        };
        left = feel(leftPart);
        right = feel(rightPart);
    }
    else
        s_buffetAfterHighG = false;
    if (trace)
    {
        static uint64_t next = 0;
        if (const uint64_t now = SDL_GetTicks(); now >= next)
        {
            next = now + 250;
            const uint32_t player = PlayerActor();
            log::Info("Triggers: in flight {}, mode {:02X} (High-G 10, stall factor 4|8), speed {:.0f} km/h, stall factor {:.3f}, "
                "THROTTLE/YAW {} -> LT {}, RT {}{}", kbm::InFlight(), Mode(player) & 0x1C, player ? F32(player + 2192) : 0.0f,
                StallFactor(player), Reverse() ? "REVERSE" : "NORMAL", Describe(left), Describe(right),
                pad ? "" : " (no DualSense)");
        }
    }
    if (!pad)
        return;
    if (s_sent && left == s_left && right == s_right)
        return;
    Effects e{};
    e.enableBits1 = 0x04 | 0x08;
    Encode(right, e.rightTrigger);
    Encode(left, e.leftTrigger);
    if (!SDL_SendGamepadEffect(pad, &e, sizeof(e)))
    {
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            log::Warn("DualSense adaptive triggers: SDL_SendGamepadEffect failed: {}", SDL_GetError());
        }
    }
    s_left = left, s_right = right, s_sent = true;   // (a failure isn't retried every frame)
}

void Clear()
{
    if (!s_pad || !s_sent || (s_left == Trigger{} && s_right == Trigger{}))
        return;
    Effects e{};
    e.enableBits1 = 0x04 | 0x08;
    Encode({}, e.rightTrigger);
    Encode({}, e.leftTrigger);
    SDL_SendGamepadEffect(s_pad, &e, sizeof(e));
    s_left = s_right = {};
}

} // namespace ac6::input::dualsense
