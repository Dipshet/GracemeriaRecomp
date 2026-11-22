// Advanced controls (input/advanced.h).
//
// An input is written as the file has it: "pad|<name>" for player 1's controller by SDL's names (a, b, leftshoulder,
// dpup, leftx, lefttrigger, ...), "joy:<device>|<name>" for a joystick that isn't a gamepad (a flight stick, a throttle,
// pedals), the device its GUID (with "#2" for a second of the same) or its name, and the name axis<n>, button<n> or
// hat<n><up|down|left|right>. An axis takes a suffix for how it is read: none: the axis (-1 to 1), "!": reversed, "+" / "-":
// that half (0 to 1 from the centre), "~" / "~!": all its travel as 0 to 1 (a throttle lever, pedals resting at one end),
// from its low or its high end. A trigger reads 0 to 1 as it is.
//
// In flight the actions' values go into player 1's pad as the game reads it (its buttons, sticks, triggers and bumpers as
// THROTTLE/YAW lays them out), and as they are, without the game's dead zones, into the player's input sample
// (game/flight_input.cpp, through Current).
#include "input/advanced.h"
#include "platform/platform.h"
#include "input/input.h"
#include "input/kbm.h"
#include "endian.h"
#include "log.h"
#include "memory.h"
#include "settings.h"

#include <SDL3/SDL.h>

#define TOML_EXCEPTIONS 0
#include <toml++/toml.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace ac6::input::advanced {

namespace {

constexpr float kFloor = 1.0f / 256;   // less is no input

// The actions, in the tab's order. Positive: nose up, roll right, yaw right, look right / up.
const std::vector<Action> kActions = {
    { "pitch", "Pitch", "PITCH", Kind::Axis, "An axis for pitch" },
    { "pitch_up", "Pitch up", "PITCH", Kind::Half, "A button, hat, half of an axis or trigger" },
    { "pitch_down", "Pitch down", "PITCH", Kind::Half, "A button, hat, half of an axis or trigger" },
    { "roll", "Roll", "ROLL", Kind::Axis, "An axis for roll" },
    { "roll_left", "Roll left", "ROLL", Kind::Half, "A button, hat, half of an axis or trigger" },
    { "roll_right", "Roll right", "ROLL", Kind::Half, "A button, hat, half of an axis or trigger" },
    { "yaw", "Yaw", "YAW", Kind::Axis, "An axis for the rudder, such as pedals or a stick's twist" },
    { "yaw_left", "Yaw left", "YAW", Kind::Half, "A button, hat, half of an axis, trigger or toe brake" },
    { "yaw_right", "Yaw right", "YAW", Kind::Half, "A button, hat, half of an axis, trigger or toe brake" },
    { "thrust", "Thrust", "THROTTLE", Kind::Half, "A throttle lever or trigger, or a button for full thrust" },
    { "brake", "Brake", "THROTTLE", Kind::Half, "A lever, trigger or toe brake, or a button for full brake" },
    { "throttle_lever", "Thrust/brake lever", "THROTTLE", Kind::Half,
      "One lever for both: brake below the detent, thrust above it\nThe lever detent rows set where" },
    { "high_g", "High-G turn", "THROTTLE", Kind::Button, "Same as thrust and brake together" },
    { "autopilot", "Autopilot", "THROTTLE", Kind::Button, "Same as yaw left and right together" },
    { "look_x", "Look left/right", "CAMERA", Kind::Axis, "An axis for the camera, left and right" },
    { "look_y", "Look up/down", "CAMERA", Kind::Axis, "An axis for the camera, up and down" },
    { "look_up", "Look up", "CAMERA", Kind::Half, "A button, hat or half of an axis" },
    { "look_down", "Look down", "CAMERA", Kind::Half, "A button, hat or half of an axis" },
    { "look_left", "Look left", "CAMERA", Kind::Half, "A button, hat or half of an axis" },
    { "look_right", "Look right", "CAMERA", Kind::Half, "A button, hat or half of an axis" },
    { "change_view", "Change view", "CAMERA", Kind::Button, "Controller: the right stick's click" },
    { "fire_machine_gun", "Fire machine gun", "WEAPONS", Kind::Button, "Controller: A" },
    { "fire_missile", "Fire missile", "WEAPONS", Kind::Button, "Controller: B" },
    { "change_weapon", "Change weapon", "WEAPONS", Kind::Button, "Controller: Back" },
    { "switch_targets", "Switch target", "WEAPONS", Kind::Button, "Controller: Y" },
    { "switch_radar_map", "Radar / map", "OTHER", Kind::Button, "Controller: X" },
    { "pause", "Pause", "OTHER", Kind::Button, "Controller: Start" },
    { "wingman_up", "Wingman command (up)", "OTHER", Kind::Button, "Controller: d-pad up" },
    { "wingman_down", "Wingman command (down)", "OTHER", Kind::Button, "Controller: d-pad down" },
    { "wingman_left", "Filter map operations (left)", "OTHER", Kind::Button, "Controller: d-pad left" },
    { "wingman_right", "Filter map operations (right)", "OTHER", Kind::Button, "Controller: d-pad right" },
    { "gear", "Landing gear", "OTHER", Kind::Button, "Controller: the left stick's click" },
    { "menu_up", "Menu up", "MENUS", Kind::Button, "In menus, pauses, cutscenes and the hangar's display\nController: d-pad up" },
    { "menu_down", "Menu down", "MENUS", Kind::Button, "In menus, pauses, cutscenes and the hangar's display\nController: d-pad down" },
    { "menu_left", "Menu left", "MENUS", Kind::Button, "In menus, pauses, cutscenes and the hangar's display\nController: d-pad left" },
    { "menu_right", "Menu right", "MENUS", Kind::Button, "In menus, pauses, cutscenes and the hangar's display\nController: d-pad right" },
    { "menu_confirm", "Confirm", "MENUS", Kind::Button, "In menus, pauses, cutscenes and the hangar's display\nController: A" },
    { "menu_back", "Back", "MENUS", Kind::Button, "In menus, pauses, cutscenes and the hangar's display\nController: B" },
    { "menu_x", "Button X", "MENUS", Kind::Button, "In menus, pauses, cutscenes and the hangar's display\nController: X (the hangar's display)" },
    { "menu_y", "Button Y", "MENUS", Kind::Button, "In menus, pauses, cutscenes and the hangar's display\nController: Y (the hangar's description and zoom in)" },
    { "menu_previous", "Previous page", "MENUS", Kind::Button, "In menus, pauses, cutscenes and the hangar's display\nController: LB" },
    { "menu_next", "Next page", "MENUS", Kind::Button, "In menus, pauses, cutscenes and the hangar's display\nController: RB" },
};

bool IsMenuAction(std::string_view action)
{
    return action.starts_with("menu_");
}

// The buttons the button actions press on the pad.
uint16_t PadButtonOf(std::string_view action)
{
    struct { std::string_view name; uint16_t bit; } constexpr kMap[] = {
        { "change_view", kRightThumb }, { "fire_machine_gun", kA }, { "fire_missile", kB }, { "change_weapon", kBack },
        { "switch_targets", kY }, { "switch_radar_map", kX }, { "pause", kStart }, { "wingman_up", kDpadUp },
        { "wingman_down", kDpadDown }, { "wingman_left", kDpadLeft }, { "wingman_right", kDpadRight }, { "gear", kLeftThumb },
        { "menu_up", kDpadUp }, { "menu_down", kDpadDown }, { "menu_left", kDpadLeft }, { "menu_right", kDpadRight },
        { "menu_confirm", kA }, { "menu_back", kB }, { "menu_x", kX }, { "menu_y", kY }, { "menu_previous", kLeftShoulder }, { "menu_next", kRightShoulder },
    };
    for (const auto& m : kMap)
        if (m.name == action)
            return m.bit;
    return 0;
}

// SDL's gamepad buttons as the Xbox pad's bits (input.cpp's ReadGamepad).
uint16_t XboxBit(SDL_GamepadButton b)
{
    switch (b)
    {
    case SDL_GAMEPAD_BUTTON_DPAD_UP: return kDpadUp;
    case SDL_GAMEPAD_BUTTON_DPAD_DOWN: return kDpadDown;
    case SDL_GAMEPAD_BUTTON_DPAD_LEFT: return kDpadLeft;
    case SDL_GAMEPAD_BUTTON_DPAD_RIGHT: return kDpadRight;
    case SDL_GAMEPAD_BUTTON_START: return kStart;
    case SDL_GAMEPAD_BUTTON_BACK: return kBack;
    case SDL_GAMEPAD_BUTTON_LEFT_STICK: return kLeftThumb;
    case SDL_GAMEPAD_BUTTON_RIGHT_STICK: return kRightThumb;
    case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER: return kLeftShoulder;
    case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER: return kRightShoulder;
    case SDL_GAMEPAD_BUTTON_GUIDE: return kGuide;
    case SDL_GAMEPAD_BUTTON_SOUTH: return kA;
    case SDL_GAMEPAD_BUTTON_EAST: return kB;
    case SDL_GAMEPAD_BUTTON_WEST: return kX;
    case SDL_GAMEPAD_BUTTON_NORTH: return kY;
    default: return 0;
    }
}

// ---- Joysticks ------------------------------------------------------------------------------------------------------
struct Device
{
    SDL_Joystick* joystick;
    SDL_JoystickID id;
    std::string key;    // its GUID, with "#2" added for a second of the same
    std::string name;
};
std::vector<Device> s_devices;
SDL_JoystickID s_testStick = 0;   // AC6_TEST_JOYSTICK's virtual stick

void Open(SDL_JoystickID id)
{
    if (SDL_IsGamepad(id))
        return;   // a controller: player 1's is "pad", the others are players 2-4
    for (const Device& d : s_devices)
        if (d.id == id)
            return;
    SDL_Joystick* j = SDL_OpenJoystick(id);
    if (!j)
        return;
    char guid[64] = {};
    SDL_GUIDToString(SDL_GetJoystickGUID(j), guid, int(sizeof(guid)));
    std::string key = guid;
    int same = 1;
    for (const Device& d : s_devices)
        if (d.key == key || d.key.starts_with(key + "#"))
            same++;
    if (same > 1)
        key += std::format("#{}", same);
    const char* name = SDL_GetJoystickName(j);
    s_devices.push_back({ j, id, key, name ? name : "Joystick" });
    log::Info("Advanced controls: {} connected ({} axes, {} buttons, {} hats; {})", s_devices.back().name, SDL_GetNumJoystickAxes(j),
        SDL_GetNumJoystickButtons(j), SDL_GetNumJoystickHats(j), key);
}

const Device* FindDevice(std::string_view id)
{
    for (const Device& d : s_devices)
        if (d.key == id)
            return &d;
    for (const Device& d : s_devices)
        if (d.name == id)
            return &d;
    return nullptr;
}

// ---- Inputs ---------------------------------------------------------------------------------------------------------
enum class Mode : uint8_t { Plain, Reversed, Positive, Negative, Lever, LeverReversed };
struct Input
{
    enum Type : uint8_t { None, PadButton, PadAxis, JoyButton, JoyAxis, JoyHat } type = None;
    std::string device;   // joysticks: the key or name
    int index = 0;        // the button, axis or hat (pads: SDL's enum)
    uint8_t hat = 0;      // SDL_HAT_UP ...
    Mode mode = Mode::Plain;
};

Input Parse(std::string_view text)
{
    Input in;
    const size_t bar = text.rfind('|');
    if (bar == std::string_view::npos)
        return in;
    const std::string_view device = text.substr(0, bar);
    std::string name(text.substr(bar + 1));
    // The suffix.
    if (name.ends_with("~!"))
        in.mode = Mode::LeverReversed, name.resize(name.size() - 2);
    else if (name.ends_with('~'))
        in.mode = Mode::Lever, name.pop_back();
    else if (name.ends_with('!'))
        in.mode = Mode::Reversed, name.pop_back();
    else if (name.ends_with('+'))
        in.mode = Mode::Positive, name.pop_back();
    else if (name.ends_with('-'))
        in.mode = Mode::Negative, name.pop_back();
    if (device == "pad")
    {
        if (const SDL_GamepadButton b = SDL_GetGamepadButtonFromString(name.c_str()); b != SDL_GAMEPAD_BUTTON_INVALID)
            in.type = Input::PadButton, in.index = int(b);
        else if (const SDL_GamepadAxis a = SDL_GetGamepadAxisFromString(name.c_str()); a != SDL_GAMEPAD_AXIS_INVALID)
            in.type = Input::PadAxis, in.index = int(a);
        return in;
    }
    if (!device.starts_with("joy:"))
        return in;
    in.device = std::string(device.substr(4));
    auto number = [&](std::string_view prefix, int& out) {
        if (!name.starts_with(prefix))
            return false;
        const std::string digits = name.substr(prefix.size());
        if (digits.empty() || !std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; }))
            return false;
        out = std::atoi(digits.c_str());
        return true;
    };
    if (number("axis", in.index))
        in.type = Input::JoyAxis;
    else if (number("button", in.index))
        in.type = Input::JoyButton;
    else if (name.starts_with("hat"))
    {
        static const std::pair<std::string_view, uint8_t> kDirections[] = {
            { "up", SDL_HAT_UP }, { "down", SDL_HAT_DOWN }, { "left", SDL_HAT_LEFT }, { "right", SDL_HAT_RIGHT } };
        for (const auto& [word, bit] : kDirections)
            if (name.ends_with(word))
            {
                const std::string digits = name.substr(3, name.size() - 3 - word.size());
                if (!digits.empty() && std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; }))
                    in.type = Input::JoyHat, in.index = std::atoi(digits.c_str()), in.hat = bit;
            }
    }
    return in;
}

bool IsTrigger(const Input& in)
{
    return in.type == Input::PadAxis && (in.index == SDL_GAMEPAD_AXIS_LEFT_TRIGGER || in.index == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER);
}

// The raw value: an axis -1 to 1 (a pad's trigger 0 to 1), a button or hat 0 or 1. False when its device isn't there.
bool Raw(const Input& in, SDL_Gamepad* pad, float& out)
{
    out = 0.0f;
    switch (in.type)
    {
    case Input::PadButton:
        if (!pad)
            return false;
        out = SDL_GetGamepadButton(pad, SDL_GamepadButton(in.index)) ? 1.0f : 0.0f;
        return true;
    case Input::PadAxis:
        if (!pad)
            return false;
        out = std::clamp(SDL_GetGamepadAxis(pad, SDL_GamepadAxis(in.index)) / 32767.0f, -1.0f, 1.0f);
        return true;
    case Input::JoyButton:
    case Input::JoyAxis:
    case Input::JoyHat:
    {
        const Device* d = FindDevice(in.device);
        if (!d)
            return false;
        if (in.type == Input::JoyButton)
            out = SDL_GetJoystickButton(d->joystick, in.index) ? 1.0f : 0.0f;
        else if (in.type == Input::JoyHat)
            out = (SDL_GetJoystickHat(d->joystick, in.index) & in.hat) ? 1.0f : 0.0f;
        else
            out = std::clamp(SDL_GetJoystickAxis(d->joystick, in.index) / 32767.0f, -1.0f, 1.0f);
        return true;
    }
    default:
        return false;
    }
}

bool IsAxis(const Input& in)
{
    return in.type == Input::PadAxis || in.type == Input::JoyAxis;
}

// As an axis action takes it (-1 to 1), and as a half (0 to 1).
float AsAxis(const Input& in, float raw)
{
    if (!IsAxis(in) || IsTrigger(in))
        return 0.0f;
    return in.mode == Mode::Reversed ? -raw : in.mode == Mode::Plain ? raw : 0.0f;
}

float AsHalf(const Input& in, float raw)
{
    if (!IsAxis(in) || IsTrigger(in))
        return raw;
    switch (in.mode)
    {
    case Mode::Positive: return std::max(raw, 0.0f);
    case Mode::Negative: return std::max(-raw, 0.0f);
    case Mode::Lever: return (raw + 1.0f) / 2.0f;
    case Mode::LeverReversed: return (1.0f - raw) / 2.0f;
    default: return std::max(raw, 0.0f);   // (an axis bound whole to a half: its positive side)
    }
}

// ---- The tuning -----------------------------------------------------------------------------------------------------
struct Tune
{
    float deadzone = 0.05f, full = 1.0f, curve = 0.0f;
    bool invert = false;
};
struct Tuning
{
    Tune pitch, roll, yaw, look, throttle{ 0.02f, 1.0f, 0.0f, false };
    float detent = 0.5f, detentWidth = 0.04f;
};

// A percentage setting as a fraction (`def` a percentage too).
float Percent(const char* env, float def, float lo, float hi)
{
    const std::string v = settings::Get(env);
    char* end = nullptr;
    const double d = v.empty() ? def : std::strtod(v.c_str(), &end);
    return std::clamp(float(v.empty() || (end && *end == 0) ? d : def), lo, hi) / 100.0f;
}

Tune ReadTune(const char* name, Tune def)
{
    const std::string up = [&] { std::string s(name); for (char& c : s) c = char(std::toupper(uint8_t(c))); return s; }();
    const std::string dz = "AC6_ADV_" + up + "_DEADZONE", full = "AC6_ADV_" + up + "_FULL", curve = "AC6_ADV_" + up + "_CURVE",
        invert = "AC6_ADV_" + up + "_INVERT";
    Tune t;
    t.deadzone = Percent(dz.c_str(), def.deadzone * 100.0f, 0.0f, 50.0f);
    t.full = std::max(t.deadzone + 0.01f, Percent(full.c_str(), def.full * 100.0f, 50.0f, 100.0f));
    t.curve = Percent(curve.c_str(), def.curve * 100.0f, 0.0f, 100.0f);
    const std::string i = settings::Get(invert.c_str());
    t.invert = i == "1" || i == "true";
    return t;
}

const Tuning& Tuned()
{
    static Tuning t;
    static uint64_t generation = UINT64_MAX;
    if (const uint64_t now = settings::Generation(); now != generation)
    {
        generation = now;
        const Tuning def;
        t.pitch = ReadTune("pitch", def.pitch);
        t.roll = ReadTune("roll", def.roll);
        t.yaw = ReadTune("yaw", def.yaw);
        t.look = ReadTune("look", def.look);
        t.throttle = ReadTune("throttle", def.throttle);
        t.detent = Percent("AC6_ADV_LEVER_DETENT", def.detent * 100.0f, 0.0f, 100.0f);
        t.detentWidth = Percent("AC6_ADV_LEVER_DETENT_WIDTH", def.detentWidth * 100.0f, 0.0f, 20.0f);
    }
    return t;
}

const Tune& TuneOf(std::string_view action, const Tuning& t)
{
    if (action.starts_with("pitch")) return t.pitch;
    if (action.starts_with("roll")) return t.roll;
    if (action.starts_with("yaw")) return t.yaw;
    if (action.starts_with("look")) return t.look;
    return t.throttle;
}

// 0 to 1 through the dead zone, the full deflection and the curve (a blend of the straight line and its cube).
float Shape(float a, const Tune& t)
{
    if (a <= t.deadzone)
        return 0.0f;
    const float x = std::min(1.0f, (a - t.deadzone) / std::max(0.01f, t.full - t.deadzone));
    return (1.0f - t.curve) * x + t.curve * x * x * x;
}

// ---- The bindings ---------------------------------------------------------------------------------------------------
struct Bound
{
    const Action* action;
    std::vector<Input> inputs;
};
std::vector<Bound> s_bound;   // one per action, in kActions' order

std::filesystem::path FilePath()
{
    const std::filesystem::path p = settings::Path();
    return p.empty() ? std::filesystem::path("ac6_settings.toml") : p;
}

void Load()
{
    const auto bindings = ReadBindings();
    s_bound.clear();
    size_t count = 0;
    for (const Action& a : kActions)
    {
        Bound b{ &a, {} };
        if (const auto it = bindings.find(a.name); it != bindings.end())
        {
            for (const std::string& text : it->second)
            {
                if (const Input in = Parse(text); in.type != Input::None)
                    b.inputs.push_back(in);
                else
                    log::Warn("Advanced controls: {} = \"{}\" isn't an input I know: left out", a.name, text);
            }
        }
        count += b.inputs.size();
        s_bound.push_back(std::move(b));
    }
    log::Info("Advanced controls: {} binding(s) read", count);
}

void ReloadIfChanged()
{
    static uint64_t nextCheck = 0, generation = UINT64_MAX;
    static std::filesystem::file_time_type seen{};
    const uint64_t now = SDL_GetTicks();
    if (now < nextCheck && !s_bound.empty())
        return;
    nextCheck = now + 1000;
    std::error_code ec;
    const auto time = std::filesystem::last_write_time(FilePath(), ec);
    const uint64_t g = settings::Generation();
    if (!s_bound.empty() && (ec || time == seen) && g == generation)
        return;
    seen = time;
    generation = g;
    Load();
}

// The game's THROTTLE/YAW is REVERSE: [profile + 19588] = 1 (game/flight_input.cpp, input/dualsense.cpp).
uint32_t Word(uint32_t a)
{
    return a ? g_memory.As<be<uint32_t>>(a)->get() : 0;
}

bool Reverse()
{
    const uint32_t root = Word(0x826E4E54);
    if (!root)
        return false;
    const uint32_t index = Word(root + 112 + 0x243F8);
    return index < 4 && Word(root + 112 + index * 42976 + 19588) == 1;
}

// The game's CONTROL SCHEME is Type-B: [profile + 19572] = 1 (sub_82225D80, which maps the pad into the flight input
// block, reads it). In flight Type-B differs from Type-A in one thing: X changes the weapon
// and Back is the radar / map, the other way round from Type-A.
bool TypeB()
{
    const uint32_t root = Word(0x826E4E54);
    if (!root)
        return false;
    const uint32_t index = Word(root + 112 + 0x243F8);
    return index < 4 && Word(root + 112 + index * 42976 + 19572) == 1;
}

// ---- This frame -----------------------------------------------------------------------------------------------------
std::mutex s_mutex;
Flight s_flight;
std::array<float, 64> s_values{};   // each action's value now (kActions' order)
static_assert(std::size(s_values) >= 32);

// The values, from the bindings (the overlay's preview reads them in menus too).
struct Frame
{
    float pitch = 0, roll = 0, yawLeft = 0, yawRight = 0, thrust = 0, brake = 0, lookX = 0, lookY = 0;
    uint16_t buttons = 0;       // in flight: the flight button actions'
    uint16_t menuButtons = 0;   // outside flight: the menu actions', and the flight button actions' pressed on a joystick
    bool any = false;   // an action has a binding
};

Frame Evaluate(SDL_Gamepad* pad)
{
    const Tuning& t = Tuned();
    Frame f;
    std::array<bool, std::size(s_values)> onStick{};   // a button action pressed by a joystick's input (not the pad's)
    auto value = [&](size_t i) { return s_values[i]; };
    for (size_t i = 0; i < s_bound.size(); i++)
    {
        const Bound& b = s_bound[i];
        const Tune& tune = TuneOf(b.action->name, t);
        const bool lever = std::string_view(b.action->name) == "throttle_lever";
        float v = 0.0f;
        bool pressed = false;
        for (const Input& in : b.inputs)
        {
            float raw = 0.0f;
            if (!Raw(in, pad, raw))
                continue;
            f.any = true;
            const bool stick = in.type != Input::PadButton && in.type != Input::PadAxis;
            if (b.action->kind == Kind::Axis)
            {
                const float a = AsAxis(in, raw);
                const float s = std::copysign(Shape(std::fabs(a), tune), a);
                if (std::fabs(s) > std::fabs(v))
                    v = s;
            }
            else
            {
                const float h = AsHalf(in, raw);
                if (b.action->kind == Kind::Button)
                {
                    pressed = pressed || h > 0.5f;
                    onStick[i] = onStick[i] || (h > 0.5f && stick);
                }
                else
                    v = std::max(v, lever ? h : IsAxis(in) ? Shape(h, tune) : h);
            }
        }
        if (b.action->kind == Kind::Axis && tune.invert)
            v = -v;
        s_values[i] = b.action->kind == Kind::Button ? (pressed ? 1.0f : 0.0f) : v;
    }
    auto get = [&](std::string_view name) {
        for (size_t i = 0; i < s_bound.size(); i++)
            if (name == s_bound[i].action->name)
                return value(i);
        return 0.0f;
    };
    f.pitch = std::clamp(get("pitch") + get("pitch_up") - get("pitch_down"), -1.0f, 1.0f);
    f.roll = std::clamp(get("roll") + get("roll_right") - get("roll_left"), -1.0f, 1.0f);
    const float yaw = std::clamp(get("yaw") + get("yaw_right") - get("yaw_left"), -1.0f, 1.0f);
    f.yawLeft = std::max(-yaw, 0.0f), f.yawRight = std::max(yaw, 0.0f);
    f.lookX = std::clamp(get("look_x") + get("look_right") - get("look_left"), -1.0f, 1.0f);
    f.lookY = std::clamp(get("look_y") + get("look_up") - get("look_down"), -1.0f, 1.0f);
    // The lever: brake below the detent, thrust above it, none within its width. Each side through the throttle's shape
    // (its dead zone is the detent's width).
    f.thrust = get("thrust"), f.brake = get("brake");
    for (size_t i = 0; i < s_bound.size(); i++)
        if (std::string_view(s_bound[i].action->name) == "throttle_lever" && !s_bound[i].inputs.empty())
        {
            Tune side = t.throttle;
            side.deadzone = 0.0f;
            const float l = value(i), lo = t.detent - t.detentWidth / 2.0f, hi = t.detent + t.detentWidth / 2.0f;
            if (l > hi && hi < 1.0f)
                f.thrust = std::max(f.thrust, Shape((l - hi) / (1.0f - hi), side));
            if (l < lo && lo > 0.0f)
                f.brake = std::max(f.brake, Shape((lo - l) / lo, side));
        }
    if (get("high_g") > 0.5f)
        f.thrust = f.brake = 1.0f;
    if (get("autopilot") > 0.5f)
        f.yawLeft = f.yawRight = 1.0f;
    // (The bound button actions are what they say whatever the game's control scheme. The pad's buttons for them
    // are Type-A's. Under Type-B X and Back trade places.) Outside flight the menu actions press theirs, and so do the
    // flight button actions pressed on a joystick (a flight stick's fire button is A in the menus as on the controller):
    // a pad's bound buttons are the pad's own there already.
    const bool typeB = TypeB();
    for (size_t i = 0; i < s_bound.size(); i++)
        if (s_bound[i].action->kind == Kind::Button && value(i) > 0.5f)
        {
            const std::string_view name = s_bound[i].action->name;
            uint16_t bit = PadButtonOf(name);
            if (IsMenuAction(name))
            {
                f.menuButtons |= bit;
                continue;
            }
            if (typeB && (bit == kX || bit == kBack))
                bit = bit == kX ? kBack : kX;
            f.buttons |= bit;
            if (onStick[i])
                f.menuButtons |= bit;
        }
    return f;
}

bool s_masked = false;   // this frame's Mask took the bound pad inputs out

int16_t Stick(float v)
{
    return int16_t(std::clamp(std::lround(v * 32767.0f), -32768L, 32767L));
}

uint8_t Trigger(float v)
{
    return uint8_t(std::clamp(std::lround(std::ceil(v * 255.0f)), 0L, 255L));   // (up: a small input still shows)
}

// The larger of the two (the keyboard's or the pad's against the bindings').
int16_t Larger(int16_t a, int16_t b)
{
    return std::abs(int(b)) > std::abs(int(a)) ? b : a;
}

// ---- Capture --------------------------------------------------------------------------------------------------------
struct Capture
{
    bool on = false;
    Kind kind = Kind::Button;
    std::array<float, SDL_GAMEPAD_AXIS_COUNT> padAxes{};
    std::array<bool, SDL_GAMEPAD_BUTTON_COUNT> padButtons{};
    struct Stick
    {
        std::string key;
        std::vector<float> axes;
        std::vector<bool> buttons;
        std::vector<uint8_t> hats;
    };
    std::vector<Stick> sticks;
} s_capture;

SDL_Gamepad* s_pad = nullptr;   // player 1's, as the last Mask had it

SDL_Gamepad* Pad()
{
    return s_pad;
}

void Snapshot(Capture& c, SDL_Gamepad* pad)
{
    for (int a = 0; a < SDL_GAMEPAD_AXIS_COUNT; a++)
        c.padAxes[size_t(a)] = pad ? SDL_GetGamepadAxis(pad, SDL_GamepadAxis(a)) / 32767.0f : 0.0f;
    for (int b = 0; b < SDL_GAMEPAD_BUTTON_COUNT; b++)
        c.padButtons[size_t(b)] = pad && SDL_GetGamepadButton(pad, SDL_GamepadButton(b));
    c.sticks.clear();
    for (const Device& d : s_devices)
    {
        Capture::Stick s{ d.key, {}, {}, {} };
        for (int a = 0; a < SDL_GetNumJoystickAxes(d.joystick); a++)
            s.axes.push_back(SDL_GetJoystickAxis(d.joystick, a) / 32767.0f);
        for (int b = 0; b < SDL_GetNumJoystickButtons(d.joystick); b++)
            s.buttons.push_back(SDL_GetJoystickButton(d.joystick, b));
        for (int h = 0; h < SDL_GetNumJoystickHats(d.joystick); h++)
            s.hats.push_back(SDL_GetJoystickHat(d.joystick, h));
        c.sticks.push_back(std::move(s));
    }
}

// An axis moved from `base` to `now`, as the capture's kind takes it ("?" when it hasn't moved far enough, or for a
// trigger taken as an axis).
std::string AxisSuffix(Kind kind, float base, float now, bool trigger)
{
    const float d = now - base;
    if (std::fabs(d) < 0.5f)
        return "?";
    if (trigger)
        return kind == Kind::Axis ? "?" : "";
    if (kind == Kind::Axis)
        return d > 0.0f ? "" : "!";
    if (std::fabs(base) < 0.3f)
        return d > 0.0f ? "+" : "-";
    return base < 0.0f ? "~" : "~!";   // from one end to the other: a lever
}

} // namespace

const std::vector<Action>& Actions()
{
    return kActions;
}

bool Enabled()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<bool> value{ false };
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        const std::string v = settings::Get("AC6_CUSTOM_FLIGHT");
        value.store(v == "1" || v == "true", std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    return value.load(std::memory_order_relaxed);
}

void Init()
{
    // AC6_TEST_JOYSTICK=<delay>:<seconds>:a0=0.5,a2=-1,b3=1,h0=up[;...] (tests): a virtual flight stick, "AC6 Test Stick",
    // 8 axes, 16 buttons and a hat, given those values from <delay> seconds after the first flight ("@<delay>": after the
    // start) for <seconds> seconds. Several phases separated by ';'.
    if (const char* v = std::getenv("AC6_TEST_JOYSTICK"); v && *v)
    {
        SDL_VirtualJoystickDesc desc;
        SDL_INIT_INTERFACE(&desc);
        desc.type = SDL_JOYSTICK_TYPE_FLIGHT_STICK;
        desc.naxes = 8;
        desc.nbuttons = 16;
        desc.nhats = 1;
        desc.vendor_id = 0x1209;
        desc.product_id = 0x0AC6;
        desc.name = "AC6 Test Stick";
        SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");   // (it works while the window isn't focused)
        s_testStick = SDL_AttachVirtualJoystick(&desc);
        log::Info("Advanced controls: test stick {}", s_testStick ? "attached" : SDL_GetError());
    }
    int count = 0;
    if (SDL_JoystickID* ids = SDL_GetJoysticks(&count))
    {
        for (int i = 0; i < count; i++)
            Open(ids[i]);
        SDL_free(ids);
    }
}

void HandleEvent(const SDL_Event& e)
{
    if (e.type == SDL_EVENT_JOYSTICK_ADDED)
        Open(e.jdevice.which);
    else if (e.type == SDL_EVENT_JOYSTICK_REMOVED)
        for (auto it = s_devices.begin(); it != s_devices.end(); ++it)
            if (it->id == e.jdevice.which)
            {
                log::Info("Advanced controls: {} disconnected", it->name);
                SDL_CloseJoystick(it->joystick);
                s_devices.erase(it);
                break;
            }
}

void Mask(SDL_Gamepad* pad, PadState& state, bool flying, bool held)
{
    ReloadIfChanged();
    s_pad = pad;
    s_masked = Enabled() && flying && !held;
    if (!s_masked || !pad)
        return;
    for (const Bound& b : s_bound)
        for (const Input& in : b.inputs)
            if (in.type == Input::PadButton)
                state.buttons &= uint16_t(~XboxBit(SDL_GamepadButton(in.index)));
            else if (in.type == Input::PadAxis)
                switch (in.index)
                {
                case SDL_GAMEPAD_AXIS_LEFTX: state.thumbLX = 0; break;
                case SDL_GAMEPAD_AXIS_LEFTY: state.thumbLY = 0; break;
                case SDL_GAMEPAD_AXIS_RIGHTX: state.thumbRX = 0; break;
                case SDL_GAMEPAD_AXIS_RIGHTY: state.thumbRY = 0; break;
                case SDL_GAMEPAD_AXIS_LEFT_TRIGGER: state.leftTrigger = 0; break;
                case SDL_GAMEPAD_AXIS_RIGHT_TRIGGER: state.rightTrigger = 0; break;
                default: break;
                }
}

TriggerRole RoleOfTrigger(bool right)
{
    if (!Enabled())
        return TriggerRole::Unbound;
    ReloadIfChanged();
    const int axis = right ? SDL_GAMEPAD_AXIS_RIGHT_TRIGGER : SDL_GAMEPAD_AXIS_LEFT_TRIGGER;
    for (const Bound& b : s_bound)
        for (const Input& in : b.inputs)
            if (in.type == Input::PadAxis && in.index == axis)
            {
                const std::string_view name = b.action->name;
                if (name == "thrust" || name == "throttle_lever")
                    return TriggerRole::Thrust;
                if (name == "brake")
                    return TriggerRole::Brake;
                if (name == "yaw" || name == "yaw_left" || name == "yaw_right")
                    return TriggerRole::Yaw;
                return TriggerRole::Other;
            }
    return TriggerRole::Unbound;
}

void Add(SDL_Gamepad* pad, PadState& state, bool flying, bool held)
{
    // The test stick's values.
    if (s_testStick)
        if (const char* v = std::getenv("AC6_TEST_JOYSTICK"))
        {
            static uint64_t firstFlight = 0;
            static const uint64_t start = SDL_GetTicks();
            const uint64_t now = SDL_GetTicks();
            if (flying && !firstFlight)
                firstFlight = now;
            // Phases "<delay>:<seconds>:<values>" separated by ';', the first whose time it is applies, "@<delay>": from the
            // start, not the first flight.
            std::string list;
            for (std::string_view phases(v); !phases.empty() && list.empty();)
            {
                const size_t semi = phases.find(';');
                const std::string phase(phases.substr(0, semi));
                phases = semi == std::string_view::npos ? std::string_view() : phases.substr(semi + 1);
                const bool fromStart = !phase.empty() && phase[0] == '@';
                const uint64_t origin = fromStart ? start : firstFlight;
                const size_t c1 = phase.find(':'), c2 = c1 == std::string::npos ? c1 : phase.find(':', c1 + 1);
                if (c2 == std::string::npos || (!fromStart && !firstFlight))
                    continue;
                const double delay = std::atof(phase.c_str() + (fromStart ? 1 : 0)), seconds = std::atof(phase.c_str() + c1 + 1);
                if (now >= origin + uint64_t(delay * 1000.0) && now < origin + uint64_t((delay + seconds) * 1000.0))
                    list = phase.substr(c2 + 1);
            }
            if (SDL_Joystick* j = SDL_GetJoystickFromID(s_testStick))
            {
                for (int a = 0; a < 8; a++)
                    SDL_SetJoystickVirtualAxis(j, a, 0);
                for (int b = 0; b < 16; b++)
                    SDL_SetJoystickVirtualButton(j, b, false);
                SDL_SetJoystickVirtualHat(j, 0, SDL_HAT_CENTERED);
                for (std::string_view items = list; !items.empty();)
                {
                    const size_t comma = items.find(',');
                    const std::string item(items.substr(0, comma));
                    items = comma == std::string_view::npos ? std::string_view() : items.substr(comma + 1);
                    const size_t eq = item.find('=');
                    if (item.size() < 3 || eq == std::string::npos)
                        continue;
                    const int n = std::atoi(item.c_str() + 1);
                    const std::string value = item.substr(eq + 1);
                    if (item[0] == 'a')
                        SDL_SetJoystickVirtualAxis(j, n, int16_t(std::clamp(std::atof(value.c_str()), -1.0, 1.0) * 32767.0));
                    else if (item[0] == 'b')
                        SDL_SetJoystickVirtualButton(j, n, std::atof(value.c_str()) > 0.5);
                    else if (item[0] == 'h')
                        SDL_SetJoystickVirtualHat(j, n, value == "up" ? SDL_HAT_UP : value == "down" ? SDL_HAT_DOWN
                            : value == "left" ? SDL_HAT_LEFT : value == "right" ? SDL_HAT_RIGHT : SDL_HAT_CENTERED);
                }
            }
        }

    const Frame f = Evaluate(pad);
    // AC6_ADV_TRACE=1 (diagnostics): the values every 0.5 s.
    static const bool trace = [] { const char* t = std::getenv("AC6_ADV_TRACE"); return t && *t == '1'; }();
    if (trace)
    {
        static uint64_t next = 0;
        if (const uint64_t now = SDL_GetTicks(); now >= next)
        {
            next = now + 500;
            std::string axes;
            for (const Device& d : s_devices)
                for (int a = 0; a < std::min(4, SDL_GetNumJoystickAxes(d.joystick)); a++)
                    axes += std::format(" {:.2f}", SDL_GetJoystickAxis(d.joystick, a) / 32767.0f);
            log::Info("Advanced trace: on {} flying {} masked {} any {} | pitch {:.3f} roll {:.3f} yaw {:.3f}/{:.3f} thrust {:.3f} brake {:.3f} "
                "look {:.2f} {:.2f} buttons {:04X} menu {:04X} | axes{}", Enabled(), flying, s_masked, f.any, f.pitch, f.roll, f.yawLeft, f.yawRight,
                f.thrust, f.brake, f.lookX, f.lookY, f.buttons, f.menuButtons, axes);
        }
    }
    // Outside flight (menus, pauses, cutscenes), and in the hangar's display with the flight bindings (its X / Y / A
    // are the menus'): the menu buttons into the pad.
    if (Enabled() && (!flying || kbm::InHangarDisplay()) && !held)
        state.buttons |= f.menuButtons;
    Flight out;
    out.active = s_masked && f.any;
    out.passButtons = state.buttons;
    out.passLeftTrigger = state.leftTrigger, out.passRightTrigger = state.rightTrigger;
    out.passPitch = -state.thumbLY / 32767.0f, out.passRoll = state.thumbLX / 32767.0f;
    if (out.active)
    {
        out.pitch = f.pitch, out.roll = f.roll, out.yawLeft = f.yawLeft, out.yawRight = f.yawRight;
        out.thrust = f.thrust, out.brake = f.brake;
        // Into the pad as the game reads it: the stick (Xbox: up +, nose up is the stick down), the camera's stick, the
        // buttons. The triggers and bumpers as THROTTLE/YAW has them (NORMAL: the triggers thrust and brake, the bumpers
        // yaw, REVERSE: the other way round), a bumper pressed for any input (the sample's value then set to the input's).
        state.thumbLX = Larger(state.thumbLX, Stick(f.roll));
        state.thumbLY = Larger(state.thumbLY, Stick(-f.pitch));
        state.thumbRX = Larger(state.thumbRX, Stick(f.lookX));
        state.thumbRY = Larger(state.thumbRY, Stick(f.lookY));
        state.buttons |= f.buttons;
        const bool reverse = Reverse();
        const float lt = reverse ? f.yawLeft : f.brake, rt = reverse ? f.yawRight : f.thrust;
        const float lb = reverse ? f.brake : f.yawLeft, rb = reverse ? f.thrust : f.yawRight;
        state.leftTrigger = std::max(state.leftTrigger, Trigger(lt));
        state.rightTrigger = std::max(state.rightTrigger, Trigger(rt));
        if (lb > kFloor)
            state.buttons |= kLeftShoulder;
        if (rb > kFloor)
            state.buttons |= kRightShoulder;
    }
    std::lock_guard lock(s_mutex);
    s_flight = out;
}

Flight Current()
{
    std::lock_guard lock(s_mutex);
    return s_flight;
}

std::map<std::string, std::vector<std::string>> ReadBindings()
{
    std::map<std::string, std::vector<std::string>> out;
    std::error_code ec;
    if (!std::filesystem::exists(FilePath(), ec))
        return out;
    toml::parse_result parsed = toml::parse_file(platform::PathUtf8(FilePath()));
    if (!parsed)
        return out;
    if (const toml::table* s = parsed.table()["advanced_bindings"].as_table())
        for (auto&& [key, node] : *s)
            if (const toml::array* list = node.as_array())
            {
                std::vector<std::string>& inputs = out[std::string(key.str())];
                for (auto&& item : *list)
                    if (auto text = item.value<std::string>())
                        inputs.push_back(*text);
            }
    return out;
}

bool SaveBindings(const std::vector<std::pair<std::string, std::vector<std::string>>>& changes, std::string& error)
{
    auto quoted = [](const std::string& s) {
        std::string out = "\"";
        for (char c : s)
            out += c == '"' || c == '\\' ? std::string("\\") + c : std::string(1, c);
        return out + "\"";
    };
    std::vector<settings::TomlLine> lines;
    for (const auto& [action, inputs] : changes)
    {
        std::string list = "[";
        for (size_t i = 0; i < inputs.size(); i++)
            list += (i ? ", " : "") + quoted(inputs[i]);
        list += "]";
        lines.push_back({ "advanced_bindings", action, inputs.empty() ? std::string() : list, "[]" });
    }
    if (!settings::SaveLines(lines, error))
        return false;
    log::Info("Advanced controls: the settings overlay wrote {} binding(s)", changes.size());
    return true;
}

std::string InputLabel(const std::string& text)
{
    const Input in = Parse(text);
    std::string suffix;
    switch (in.mode)
    {
    case Mode::Reversed: suffix = " (REVERSED)"; break;
    case Mode::Positive: suffix = " +"; break;
    case Mode::Negative: suffix = " -"; break;
    case Mode::Lever: suffix = " (LEVER)"; break;
    case Mode::LeverReversed: suffix = " (LEVER, REVERSED)"; break;
    default: break;
    }
    std::string out;
    if (in.type == Input::PadButton || in.type == Input::PadAxis)
    {
        static const std::pair<std::string_view, std::string_view> kNames[] = {
            { "a", "A" }, { "b", "B" }, { "x", "X" }, { "y", "Y" }, { "back", "BACK" }, { "start", "START" }, { "guide", "GUIDE" },
            { "leftstick", "LEFT STICK CLICK" }, { "rightstick", "RIGHT STICK CLICK" }, { "leftshoulder", "LB" },
            { "rightshoulder", "RB" }, { "dpup", "D-PAD UP" }, { "dpdown", "D-PAD DOWN" }, { "dpleft", "D-PAD LEFT" },
            { "dpright", "D-PAD RIGHT" }, { "leftx", "LEFT STICK X" }, { "lefty", "LEFT STICK Y" }, { "rightx", "RIGHT STICK X" },
            { "righty", "RIGHT STICK Y" }, { "lefttrigger", "LT" }, { "righttrigger", "RT" }, { "touchpad", "TOUCHPAD" },
            { "misc1", "SHARE" }, { "paddle1", "PADDLE 1" }, { "paddle2", "PADDLE 2" }, { "paddle3", "PADDLE 3" },
            { "paddle4", "PADDLE 4" } };
        const char* sdl = in.type == Input::PadButton ? SDL_GetGamepadStringForButton(SDL_GamepadButton(in.index))
                                                      : SDL_GetGamepadStringForAxis(SDL_GamepadAxis(in.index));
        std::string name = sdl ? sdl : "?";
        for (const auto& [from, to] : kNames)
            if (from == name)
                name = to;
        out = "PAD " + name;
    }
    else if (in.type != Input::None)
    {
        const Device* d = FindDevice(in.device);
        std::string device = d ? d->name : "(NOT CONNECTED)";
        for (char& c : device)
            c = char(std::toupper(uint8_t(c)));
        if (in.type == Input::JoyAxis)
            out = std::format("{} AXIS {}", device, in.index + 1);
        else if (in.type == Input::JoyButton)
            out = std::format("{} BUTTON {}", device, in.index + 1);
        else
            out = std::format("{} HAT {} {}", device, in.index + 1, in.hat == SDL_HAT_UP ? "UP" : in.hat == SDL_HAT_DOWN ? "DOWN"
                : in.hat == SDL_HAT_LEFT ? "LEFT" : "RIGHT");
    }
    else
        return "? " + text;
    return out + suffix;
}

void BeginCapture(Kind kind)
{
    s_capture.on = true;
    s_capture.kind = kind;
    Snapshot(s_capture, Pad());
}

std::string PollCapture()
{
    if (!s_capture.on)
        return {};
    SDL_Gamepad* pad = Pad();
    Capture now;
    Snapshot(now, pad);
    const Kind kind = s_capture.kind;
    std::string found;
    // Buttons and hats: a press since the last poll (a button held when it began counts once let go and pressed again).
    if (kind != Kind::Axis)
    {
        for (int b = 0; b < SDL_GAMEPAD_BUTTON_COUNT && found.empty(); b++)
            if (now.padButtons[size_t(b)] && !s_capture.padButtons[size_t(b)])
                if (const char* name = SDL_GetGamepadStringForButton(SDL_GamepadButton(b)))
                    found = std::string("pad|") + name;
        for (const Capture::Stick& s : now.sticks)
            for (const Capture::Stick& was : s_capture.sticks)
                if (was.key == s.key && found.empty())
                {
                    for (size_t b = 0; b < s.buttons.size() && b < was.buttons.size() && found.empty(); b++)
                        if (s.buttons[b] && !was.buttons[b])
                            found = std::format("joy:{}|button{}", s.key, b);
                    for (size_t h = 0; h < s.hats.size() && h < was.hats.size() && found.empty(); h++)
                        for (const auto& [bit, word] : { std::pair{ SDL_HAT_UP, "up" }, std::pair{ SDL_HAT_DOWN, "down" },
                                                          std::pair{ SDL_HAT_LEFT, "left" }, std::pair{ SDL_HAT_RIGHT, "right" } })
                            if ((s.hats[h] & bit) && !(was.hats[h] & bit) && found.empty())
                                found = std::format("joy:{}|hat{}{}", s.key, h, word);
                }
        // (the buttons' state carried: a press is an edge)
        s_capture.padButtons = now.padButtons;
        for (Capture::Stick& was : s_capture.sticks)
            for (const Capture::Stick& s : now.sticks)
                if (was.key == s.key)
                    was.buttons = s.buttons, was.hats = s.hats;
    }
    // Axes: moved half their travel from where they were when it began.
    for (int a = 0; a < SDL_GAMEPAD_AXIS_COUNT && found.empty() && pad; a++)
    {
        const bool trigger = a == SDL_GAMEPAD_AXIS_LEFT_TRIGGER || a == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER;
        const std::string suffix = AxisSuffix(kind, s_capture.padAxes[size_t(a)], now.padAxes[size_t(a)], trigger);
        if (suffix != "?")
            if (const char* name = SDL_GetGamepadStringForAxis(SDL_GamepadAxis(a)))
                found = std::string("pad|") + name + suffix;
    }
    for (const Capture::Stick& s : now.sticks)
        for (const Capture::Stick& was : s_capture.sticks)
            if (was.key == s.key)
                for (size_t a = 0; a < s.axes.size() && a < was.axes.size() && found.empty(); a++)
                    if (const std::string suffix = AxisSuffix(kind, was.axes[a], s.axes[a], false); suffix != "?")
                        found = std::format("joy:{}|axis{}{}", s.key, a, suffix);
    if (!found.empty())
        s_capture.on = false;
    return found;
}

void EndCapture()
{
    s_capture.on = false;
}

float Value(const std::string& action, const std::vector<std::string>& inputs)
{
    const Action* a = nullptr;
    for (const Action& x : kActions)
        if (action == x.name)
            a = &x;
    if (!a)
        return 0.0f;
    const Tuning& t = Tuned();
    const Tune& tune = TuneOf(a->name, t);
    const bool lever = action == "throttle_lever";
    float v = 0.0f;
    for (const std::string& text : inputs)
    {
        const Input in = Parse(text);
        float raw = 0.0f;
        if (in.type == Input::None || !Raw(in, s_pad, raw))
            continue;
        if (a->kind == Kind::Axis)
        {
            const float x = AsAxis(in, raw), shaped = std::copysign(Shape(std::fabs(x), tune), x);
            if (std::fabs(shaped) > std::fabs(v))
                v = shaped;
        }
        else if (a->kind == Kind::Button)
            v = std::max(v, AsHalf(in, raw) > 0.5f ? 1.0f : 0.0f);
        else
            v = std::max(v, lever || !IsAxis(in) ? AsHalf(in, raw) : Shape(AsHalf(in, raw), tune));
    }
    return a->kind == Kind::Axis && tune.invert ? -v : v;
}

float RawValue(const std::string& action, const std::vector<std::string>& inputs)
{
    const Action* a = nullptr;
    for (const Action& x : kActions)
        if (action == x.name)
            a = &x;
    float v = 0.0f;
    for (const std::string& text : inputs)
    {
        const Input in = Parse(text);
        float raw = 0.0f;
        if (!a || in.type == Input::None || !Raw(in, s_pad, raw))
            continue;
        const float x = a->kind == Kind::Axis ? AsAxis(in, raw) : AsHalf(in, raw);
        if (std::fabs(x) > std::fabs(v))
            v = x;
    }
    return v;
}

std::vector<std::string> DeviceNames()
{
    std::vector<std::string> out;
    for (const Device& d : s_devices)
        out.push_back(d.name);
    return out;
}

} // namespace ac6::input::advanced
