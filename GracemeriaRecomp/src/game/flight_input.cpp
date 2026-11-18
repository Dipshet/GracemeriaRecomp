// Custom flight controls (AC6_CUSTOM_FLIGHT, Advanced controls in the overlay).
// With it and Analog yaw off, these three hooks only call the game's functions: the game's controls and keyboard and
// mouse flight (input/kbm.cpp) are unchanged. On:
//  - 0x822291C0, the player's input sample: its output is where Advanced controls' bindings (input/advanced.h) and a
//    HOTAS set the flight inputs, past the stick's deadzone. AC6_FLIGHT_TEST_YAW, which tests use, sets the yaw there.
//    With THROTTLE/YAW on REVERSE the triggers' yaw is set from the pad's own pull, without the game's dead zone.
//  - 0x82303A50 and 0x82329B40, the control steps of CONTROL TYPE's NORMAL and NOVICE: the game keeps only the sign of
//    the yaw input. Here a partial input gets a partial yaw. A full one (a bumper, a trigger pulled fully) goes on as
//    the game's.
// Analog yaw (AC6_ANALOG_YAW, the Controls tab's row): the triggers' rudder under REVERSE without the game's dead zone
// and with the control steps' partial yaw. With Advanced controls on it is so anyway (the row greyed out). Off and
// Advanced controls off: the game's. On NORMAL the rudder is on the bumpers, all or nothing: nothing changes there.
#include "endian.h"
#include "input/advanced.h"
#include "input/input.h"
#include "log.h"
#include "memory.h"
#include "ppc.h"
#include "settings.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <string>

namespace {

// The aircraft's sampled inputs (the flight model's fm+44 is yawA - yawB), and the flight model's fields used here.
constexpr uint32_t kAccelerate = 2096, kBrake = 2100, kPitch = 2104, kRoll = 2108, kYawA = 2112, kYawB = 2116, kFlightModel = 4912;
// The input block's (the sample's source): the held bits, the stick, the trigger axes as THROTTLE/YAW lays them out.
constexpr uint32_t kBlockBits = 3652, kBlockPitch = 3672, kBlockRoll = 3676, kBlockAxis0 = 3680, kBlockAxis1 = 3684,
    kBlockAxis2 = 3688, kBlockAxis3 = 3692;
// The sample's pitch and roll against Advanced controls' (nose up +, roll right +). Measured: the stick down (nose up) gives
// the block's pitch +, the stick right its roll - (AC6_TEST_JOYSTICK).
constexpr float kPitchSign = 1.0f, kRollSign = -1.0f;
constexpr uint32_t kRollInput = 40, kYawInput = 44, kRate = 148, kYawCommand = 308, kRollCommand = 312, kRudder = 1456;
constexpr uint32_t kYawRise = 580, kYawFall = 596;   // inputRotAdd.Y, inputRotDec.Y (1/s)
constexpr float kFloor = 1.0f / 256;                 // less than a trigger's smallest step is no input
constexpr float kNoviceRoll = 0.125f;                // the roll NOVICE gives a full yaw input, and its rudder's step at 60 Hz

// Analog yaw: live.
bool AnalogYaw()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<bool> value{ false };
    if (const uint64_t now = ac6::settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        const std::string v = ac6::settings::Get("AC6_ANALOG_YAW");
        value.store(v == "1" || v == "true", std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    return value.load(std::memory_order_relaxed);
}

uint32_t Word(uint32_t address) { return ac6::g_memory.As<ac6::be<uint32_t>>(address)->get(); }
float Load(uint32_t address) { return ac6::g_memory.As<ac6::be<float>>(address)->get(); }
void Store(uint32_t address, float value) { *ac6::g_memory.As<ac6::be<float>>(address) = value; }

bool Custom()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<bool> value{ false };
    if (const uint64_t now = ac6::settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        const std::string v = ac6::settings::Get("AC6_CUSTOM_FLIGHT");
        value.store(v == "1" || v == "true", std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    return value.load(std::memory_order_relaxed);
}

float MoveToward(float from, float to, float step)
{
    return from < to ? std::min(from + step, to) : std::max(from - step, to);
}

// The game's THROTTLE/YAW is REVERSE: [profile + 19588] = 1 (input/dualsense.cpp reads it the same way).
bool Reverse()
{
    const uint32_t root = Word(0x826E4E54);
    if (!root)
        return false;
    const uint32_t index = Word(root + 112 + 0x243F8);
    return index < 4 && Word(root + 112 + index * 42976 + 19588) == 1;
}

// Advanced controls' value into a sampled field, unless player 1's other input there is the larger (the game's reading of
// it stays) or the game took an input and gave nothing (its own reasons: the field stays 0).
void Override(uint32_t field, float value, bool passLarger, bool gameHadInput)
{
    if (passLarger || (Load(field) == 0.0f && gameHadInput))
        return;
    Store(field, value);
}

// A trigger's yaw without the game's dead zone: the game's is (pull - 74) / 181, held to 0-1 (pulls 40, 128, 192, 255 of 255
// read 0, 0.299, 0.652, 1.000). This is pull / 255. Where the game gave 0 to a pull well past its dead zone, it meant no yaw
// (the input not taken), and that stays.
void WithoutDeadZone(uint32_t field, uint8_t pull)
{
    constexpr uint8_t kGameDeadZone = 80;   // 74 measured, with a margin for rounding
    if (Load(field) > 0.0f || pull <= kGameDeadZone)
        Store(field, pull / 255.0f);
}

bool Partial(float input)
{
    return input != 0.0f && std::fabs(input) < 1.0f;
}

} // namespace

extern "C" PPC_FUNC(__imp__sub_822291C0);
PPC_FUNC(sub_822291C0)
{
    const uint32_t aircraft = ctx.r3.u32, block = ctx.r4.u32;
    __imp__sub_822291C0(ctx, base);
    // (With Advanced controls on the triggers' rudder is analog whatever the Analog yaw row says: the row is greyed out
    // then.)
    const bool custom = Custom(), analogYaw = custom || AnalogYaw();
    if (!block || !analogYaw)
        return;

    // Advanced controls (input/advanced.h): the bindings' inputs as they are, past the game's dead zones
    // and curves.
    const bool reverse = Reverse();
    const ac6::input::advanced::Flight f = custom ? ac6::input::advanced::Current() : ac6::input::advanced::Flight{};
    const uint32_t bits = Word(block + kBlockBits);
    if (f.active)
    {
        using namespace ac6::input;
        Override(aircraft + kPitch, f.pitch * kPitchSign, std::fabs(f.passPitch) > std::fabs(f.pitch), Load(block + kBlockPitch) != 0.0f);
        Override(aircraft + kRoll, f.roll * kRollSign, std::fabs(f.passRoll) > std::fabs(f.roll), Load(block + kBlockRoll) != 0.0f);
        if (!reverse)
        {
            Override(aircraft + kAccelerate, f.thrust, f.passRightTrigger / 255.0f > f.thrust, Load(block + kBlockAxis2) != 0.0f);
            Override(aircraft + kBrake, f.brake, f.passLeftTrigger / 255.0f > f.brake, Load(block + kBlockAxis3) != 0.0f);
            Override(aircraft + kYawA, f.yawLeft, f.passButtons & kLeftShoulder, bits & 0x4);
            Override(aircraft + kYawB, f.yawRight, f.passButtons & kRightShoulder, bits & 0x8);
        }
        else
        {
            Override(aircraft + kAccelerate, f.thrust, f.passButtons & kRightShoulder, bits & 0x10);
            Override(aircraft + kBrake, f.brake, f.passButtons & kLeftShoulder, bits & 0x20);
            // (An unbound trigger: the game's yaw, or without its dead zone with Analog yaw on.)
            if (f.passLeftTrigger / 255.0f > f.yawLeft)
            {
                if (analogYaw)
                    WithoutDeadZone(aircraft + kYawA, f.passLeftTrigger);
            }
            else
                Override(aircraft + kYawA, f.yawLeft, false, Load(block + kBlockAxis0) != 0.0f);
            if (f.passRightTrigger / 255.0f > f.yawRight)
            {
                if (analogYaw)
                    WithoutDeadZone(aircraft + kYawB, f.passRightTrigger);
            }
            else
                Override(aircraft + kYawB, f.yawRight, false, Load(block + kBlockAxis1) != 0.0f);
        }
    }
    else if (reverse && analogYaw)
    {
        const ac6::input::PadState pad = ac6::input::GetState(0);   // player 1, as the game was given it
        WithoutDeadZone(aircraft + kYawA, pad.leftTrigger);
        WithoutDeadZone(aircraft + kYawB, pad.rightTrigger);
    }

    static const char* const test = std::getenv("AC6_FLIGHT_TEST_YAW");
    if (test)
    {
        const float yaw = float(std::atof(test));
        Store(aircraft + kYawA, std::max(yaw, 0.0f));
        Store(aircraft + kYawB, std::max(-yaw, 0.0f));
    }
    if (std::fabs(Load(aircraft + kYawA) - Load(aircraft + kYawB)) < kFloor)
    {
        Store(aircraft + kYawA, 0.0f);
        Store(aircraft + kYawB, 0.0f);
    }

    static const bool trace = std::getenv("AC6_FLIGHT_TRACE") != nullptr;
    static uint32_t count = 0;
    if (const uint32_t fm = Word(aircraft + kFlightModel); trace && fm && count++ % 15 == 0)
    {
        // The profile's six options, by [root + 112 + 0x243F8].
        const uint32_t root = Word(0x826E4E54), profile = root + 112 + Word(root + 112 + 0x243F8) * 42976 + 19568;
        ac6::log::Info("Flight trace: yaw {:.3f} {:.3f} command {:.3f} rate {:.3f} roll {:.3f} rudder {:.3f} | in pitch {:.3f} roll {:.3f} "
            "accel {:.3f} brake {:.3f} block pitch {:.3f} roll {:.3f} axes {:.3f} {:.3f} {:.3f} {:.3f} bits {:X} | adv {} {:.3f} {:.3f} | "
            "class {:08X} options {} {} {} {} {} {}",
            Load(aircraft + kYawA), Load(aircraft + kYawB), Load(fm + kYawCommand), Load(fm + kRate), Load(fm + kRollCommand), Load(fm + kRudder),
            Load(aircraft + kPitch), Load(aircraft + kRoll), Load(aircraft + kAccelerate), Load(aircraft + kBrake), Load(block + kBlockPitch),
            Load(block + kBlockRoll), Load(block + kBlockAxis0), Load(block + kBlockAxis1), Load(block + kBlockAxis2), Load(block + kBlockAxis3),
            bits, f.active, f.pitch, f.roll,
            Word(fm), Word(profile), Word(profile + 4), Word(profile + 8), Word(profile + 12), Word(profile + 16), Word(profile + 20));
    }
}

// NORMAL: the game moves the yaw command toward +-1 by the input's sign. A partial input moves it toward the input instead:
// at the game's rise rate going out or across zero, at its fall rate coming back.
extern "C" PPC_FUNC(__imp__sub_82303A50);
PPC_FUNC(sub_82303A50)
{
    if (!Custom() && !AnalogYaw())
    {
        __imp__sub_82303A50(ctx, base);
        return;
    }
    const uint32_t fm = ctx.r3.u32;
    const float dt = float(ctx.f1.f64), input = Load(fm + kYawInput), command = Load(fm + kYawCommand);
    __imp__sub_82303A50(ctx, base);
    if (!Partial(input))
        return;
    const bool easing = command * input > 0.0f && std::fabs(command) > std::fabs(input);
    Store(fm + kYawCommand, MoveToward(command, input, Load(fm + (easing ? kYawFall : kYawRise)) * dt));
}

// NOVICE has no yaw command: with the stick's roll at rest, a yaw input becomes a small roll of its sign (the rudder moving
// with it), and the yaw comes from the roll. Given a partial roll here, the game takes it as the stick's and centres the
// rudder, which is then set as the game would, to the input's size.
extern "C" PPC_FUNC(__imp__sub_82329B40);
PPC_FUNC(sub_82329B40)
{
    if (!Custom() && !AnalogYaw())
    {
        __imp__sub_82329B40(ctx, base);
        return;
    }
    const uint32_t fm = ctx.r3.u32;
    const float dt = float(ctx.f1.f64), input = Load(fm + kYawInput), rudder = Load(fm + kRudder);
    const bool partial = Partial(input) && std::fabs(Load(fm + kRollInput)) < 1.0f / 65536;
    if (partial)
        Store(fm + kRollInput, input * kNoviceRoll);
    __imp__sub_82329B40(ctx, base);
    if (partial)
        Store(fm + kRudder, MoveToward(rudder, input, kNoviceRoll * dt * 60.0f));
}
