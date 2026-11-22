#pragma once
// Advanced controls: player 1's flight inputs bound to any controller's or flight stick's (HOTAS)
// axes, buttons and hats, each axis tuned (dead zone, full deflection, curve, invert). The settings overlay's Advanced
// controls tab edits the bindings ([advanced_bindings] in ac6_settings.toml) and the tuning ([advanced_controls]).
//
// On with the tab's first row (AC6_CUSTOM_FLIGHT, which also gives partial yaw: game/flight_input.cpp). The flight
// actions are for flight and the hangar's display (the aircraft select's X, where the pad moves the shown aircraft as
// in flight). Outside them (menus, pauses, cutscenes) the pad is as the game has it, and the MENUS actions and the flight
// button actions bound on a joystick press their pad buttons (so a flight stick alone gets through the menus). The
// MENUS actions work in the hangar's display too. An action without bindings stays the game's. A pad input that is bound is the bindings' only in flight (the
// game doesn't see it there). The keyboard and mouse are unchanged.
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

struct SDL_Gamepad;
union SDL_Event;

namespace ac6::input {
struct PadState;
}

namespace ac6::input::advanced {

// An action's kind. Axis: -1 to 1, bound to an axis only. Half: 0 to 1, bound to half an axis, a lever, a trigger or a
// button (1 when pressed). Button: on or off, bound to any input (an axis counts past half way).
enum class Kind : uint8_t { Axis, Half, Button };
struct Action
{
    const char* name;    // the file's key
    const char* label;
    const char* group;   // the tab's heading
    Kind kind;
    const char* help;
};
const std::vector<Action>& Actions();

bool Enabled();                          // the tab's first row: AC6_CUSTOM_FLIGHT
void Init();                             // main thread: the joysticks connected at the start
void HandleEvent(const SDL_Event& e);    // main thread: joysticks added and removed

// Main thread, each frame, player 1's pad state: Mask right after the controller is read (the bound pad inputs taken
// out, in flight), Add after the keyboard and the macro (the bindings' inputs put in, and published for Current).
void Mask(SDL_Gamepad* pad, PadState& state, bool flying, bool held);
// What player 1's trigger (LT, or RT with right) is bound to, for the DualSense's trigger feel, on the
// main thread (as Mask): Unbound with Advanced controls off or nothing bound to it (the game's layout decides).
enum class TriggerRole { Unbound, Thrust, Brake, Yaw, Other };
TriggerRole RoleOfTrigger(bool right);
void Add(SDL_Gamepad* pad, PadState& state, bool flying, bool held);

// This frame's flight inputs, for the player's input sample (game/flight_input.cpp), any thread.
struct Flight
{
    bool active = false;                      // on, in flight, with bindings
    float pitch = 0.0f, roll = 0.0f;          // -1 to 1: nose up +, roll right +
    float yawLeft = 0.0f, yawRight = 0.0f;    // 0 to 1
    float thrust = 0.0f, brake = 0.0f;        // 0 to 1
    // Player 1's inputs without the bindings' (the pad's unbound ones, the keyboard's and mouse's): where they are the
    // larger, the game's own reading of them stays.
    uint16_t passButtons = 0;
    uint8_t passLeftTrigger = 0, passRightTrigger = 0;
    float passPitch = 0.0f, passRoll = 0.0f;   // the left stick, as pitch (nose up +) and roll
};
Flight Current();

// The settings overlay. The bindings: action -> its inputs, as the file has them (none for an action it leaves out).
std::map<std::string, std::vector<std::string>> ReadBindings();
bool SaveBindings(const std::vector<std::pair<std::string, std::vector<std::string>>>& changes, std::string& error);
std::string InputLabel(const std::string& input);   // "PAD A", "T.16000M AXIS 2 +", "(NOT CONNECTED) BUTTON 4"
// Waiting for an input: the first moved or pressed after BeginCapture, as an action of that kind takes it ("" until then).
void BeginCapture(Kind kind);
std::string PollCapture();
void EndCapture();
// The action's value now from those inputs, tuned (-1 to 1 or 0 to 1), whether in flight or not (the overlay's preview).
float Value(const std::string& action, const std::vector<std::string>& inputs);
// The same before the tuning: the input as read (-1 to 1 for an axis action, 0 to 1 for a half), the larger of its inputs.
float RawValue(const std::string& action, const std::vector<std::string>& inputs);
std::vector<std::string> DeviceNames();   // the flight sticks and other joysticks connected (not the gamepads)

} // namespace ac6::input::advanced
