// Keyboard and mouse controls, with AC7-style
// defaults. The keyboard and mouse are folded into player 1's pad (the state every layer
// of the game's input reads), so they work alongside a real controller or on their own:
//  - two key sets, menu and flight, switched by a flight heartbeat: the player's flight model ticking (air and ground
//    force steps, its master update, the ground-state maintainer, the control sampler: game/kbm_heartbeat.cpp) within
//    the last 300 ms. Pause and the menus stop the simulation, so the menu set takes over there.
//  - in flight the mouse steers (pitch / roll, a velocity model with smoothing, the stick shaped by a deadzone, a curve
//    and an anti-deadzone that clears the game's own). Holding the camera key (Left Alt) pans the camera with it instead.
//    With mouse_controls = "camera" it is the other way round: the mouse pans the camera, the camera key held steers.
//  - bindings and mouse feel in ac6_settings.toml's [mouse], [menu_keys] and [flight_keys] (an
//    ac6_input.toml's [mouse], [menu] and [flight] are moved there once), read again when the file changes (about once a
//    second: the settings overlay's Apply too).
// Keys are tracked from SDL's events and forgotten when the window loses focus. A key held across a key-set switch is
// ignored until it is released and pressed again (Escape is pause in flight and cancel in menus). Opt-in, AC6_KBM=1:
// without it the fixed keyboard map.
#include "input/kbm.h"
#include "platform/platform.h"
#include "input/input.h"
#include "log.h"
#include "settings.h"

#include <SDL3/SDL.h>

#define TOML_EXCEPTIONS 0
#include <toml++/toml.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ac6::input::kbm {

namespace {

using Clock = std::chrono::steady_clock;

int64_t NowMs()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
}

// ---- Keys ---------------------------------------------------------------------------------------------------------
// A key reference: a scancode, a mouse button (SDL_BUTTON_*), or the wheel (held for 90 ms after a notch).
struct Key
{
    enum Kind : uint8_t { Scancode, Mouse, WheelUp, WheelDown } kind = Scancode;
    uint32_t code = 0;
};

constexpr int kMaxScancodes = SDL_SCANCODE_COUNT;
std::array<bool, kMaxScancodes> s_keyDown{};    // from events, cleared on focus loss
std::array<bool, kMaxScancodes> s_keyLatched{}; // held across a key-set switch: ignored until released
std::array<bool, 8> s_mouseDown{}, s_mouseLatched{};
int64_t s_wheelUpUntil = 0, s_wheelDownUntil = 0;

bool Held(const Key& k)
{
    switch (k.kind)
    {
    case Key::Scancode: return k.code < kMaxScancodes && s_keyDown[k.code] && !s_keyLatched[k.code];
    case Key::Mouse: return k.code < s_mouseDown.size() && s_mouseDown[k.code] && !s_mouseLatched[k.code];
    case Key::WheelUp: return NowMs() < s_wheelUpUntil;
    case Key::WheelDown: return NowMs() < s_wheelDownUntil;
    }
    return false;
}

bool AnyHeld(const std::vector<Key>& keys)
{
    for (const Key& k : keys)
        if (Held(k))
            return true;
    return false;
}

// Key names (F1..F24, A..Z, 0..9, punctuation, navigation, numpad, modifiers, mouse).
bool ParseKey(std::string_view name, std::vector<Key>& out)
{
    auto sc = [&](SDL_Scancode c) { out.push_back({ Key::Scancode, uint32_t(c) }); return true; };
    auto mouse = [&](uint32_t b) { out.push_back({ Key::Mouse, b }); return true; };
    if (name.size() == 1)
    {
        const char c = name[0];
        if (c >= 'A' && c <= 'Z')
            return sc(SDL_Scancode(SDL_SCANCODE_A + (c - 'A')));
        if (c >= 'a' && c <= 'z')
            return sc(SDL_Scancode(SDL_SCANCODE_A + (c - 'a')));
        if (c >= '1' && c <= '9')
            return sc(SDL_Scancode(SDL_SCANCODE_1 + (c - '1')));
        if (c == '0')
            return sc(SDL_SCANCODE_0);
    }
    if (name.size() >= 2 && name[0] == 'F' && std::isdigit(uint8_t(name[1])))
    {
        const int n = std::atoi(std::string(name.substr(1)).c_str());
        if (n >= 1 && n <= 12)
            return sc(SDL_Scancode(SDL_SCANCODE_F1 + (n - 1)));
        if (n >= 13 && n <= 24)
            return sc(SDL_Scancode(SDL_SCANCODE_F13 + (n - 13)));
    }
    if (name.starts_with("Numpad") && name.size() == 7 && std::isdigit(uint8_t(name[6])))
        return sc(name[6] == '0' ? SDL_SCANCODE_KP_0 : SDL_Scancode(SDL_SCANCODE_KP_1 + (name[6] - '1')));
    static const std::unordered_map<std::string_view, SDL_Scancode> kNames = {
        { "Backtick", SDL_SCANCODE_GRAVE }, { "Minus", SDL_SCANCODE_MINUS }, { "Plus", SDL_SCANCODE_EQUALS },
        { "Comma", SDL_SCANCODE_COMMA }, { "Period", SDL_SCANCODE_PERIOD }, { "Semicolon", SDL_SCANCODE_SEMICOLON },
        { "Slash", SDL_SCANCODE_SLASH }, { "Backslash", SDL_SCANCODE_BACKSLASH }, { "LBracket", SDL_SCANCODE_LEFTBRACKET },
        { "RBracket", SDL_SCANCODE_RIGHTBRACKET }, { "Quote", SDL_SCANCODE_APOSTROPHE }, { "Escape", SDL_SCANCODE_ESCAPE },
        { "Esc", SDL_SCANCODE_ESCAPE }, { "Return", SDL_SCANCODE_RETURN }, { "Enter", SDL_SCANCODE_RETURN },
        { "Space", SDL_SCANCODE_SPACE }, { "Tab", SDL_SCANCODE_TAB }, { "Backspace", SDL_SCANCODE_BACKSPACE },
        { "Delete", SDL_SCANCODE_DELETE }, { "Insert", SDL_SCANCODE_INSERT }, { "Home", SDL_SCANCODE_HOME },
        { "End", SDL_SCANCODE_END }, { "PageUp", SDL_SCANCODE_PAGEUP }, { "PageDown", SDL_SCANCODE_PAGEDOWN },
        { "Left", SDL_SCANCODE_LEFT }, { "Right", SDL_SCANCODE_RIGHT }, { "Up", SDL_SCANCODE_UP }, { "Down", SDL_SCANCODE_DOWN },
        { "NumpadEnter", SDL_SCANCODE_KP_ENTER }, { "NumpadPlus", SDL_SCANCODE_KP_PLUS }, { "NumpadMinus", SDL_SCANCODE_KP_MINUS },
        { "NumpadStar", SDL_SCANCODE_KP_MULTIPLY }, { "NumpadSlash", SDL_SCANCODE_KP_DIVIDE },
        { "PrintScreen", SDL_SCANCODE_PRINTSCREEN }, { "Pause", SDL_SCANCODE_PAUSE }, { "CapsLock", SDL_SCANCODE_CAPSLOCK },
        { "NumLock", SDL_SCANCODE_NUMLOCKCLEAR }, { "ScrollLock", SDL_SCANCODE_SCROLLLOCK },
        { "LeftControl", SDL_SCANCODE_LCTRL }, { "LeftCtrl", SDL_SCANCODE_LCTRL }, { "RightControl", SDL_SCANCODE_RCTRL },
        { "RightCtrl", SDL_SCANCODE_RCTRL }, { "LeftAlt", SDL_SCANCODE_LALT }, { "RightAlt", SDL_SCANCODE_RALT },
        { "LeftShift", SDL_SCANCODE_LSHIFT }, { "RightShift", SDL_SCANCODE_RSHIFT },
    };
    if (auto it = kNames.find(name); it != kNames.end())
        return sc(it->second);
    if (name == "Shift") { sc(SDL_SCANCODE_LSHIFT); return sc(SDL_SCANCODE_RSHIFT); }
    if (name == "Control" || name == "Ctrl") { sc(SDL_SCANCODE_LCTRL); return sc(SDL_SCANCODE_RCTRL); }
    if (name == "Alt") { sc(SDL_SCANCODE_LALT); return sc(SDL_SCANCODE_RALT); }
    if (name == "LMB" || name == "Mouse1") return mouse(SDL_BUTTON_LEFT);
    if (name == "RMB" || name == "Mouse2") return mouse(SDL_BUTTON_RIGHT);
    if (name == "MMB" || name == "Mouse3") return mouse(SDL_BUTTON_MIDDLE);
    if (name == "Mouse4" || name == "MouseBack") return mouse(SDL_BUTTON_X1);
    if (name == "Mouse5" || name == "MouseForward") return mouse(SDL_BUTTON_X2);
    if (name == "MouseWheelUp") { out.push_back({ Key::WheelUp, 0 }); return true; }
    if (name == "MouseWheelDown") { out.push_back({ Key::WheelDown, 0 }); return true; }
    return false;
}

// ---- Actions ------------------------------------------------------------------------------------------------------
struct Effect
{
    uint16_t buttons = 0;
    int8_t lx = 0, ly = 0, rx = 0, ry = 0;   // -1 / +1: the stick pushed fully
    bool lt = false, rt = false;
};

struct Action
{
    const char* name;
    Effect effect;
    std::vector<std::string_view> defaults;
    std::vector<Key> keys;
};

std::vector<Action> MenuActions()
{
    return {
        { "up", { kDpadUp }, { "W", "Up" } },
        { "down", { kDpadDown }, { "S", "Down" } },
        { "left", { kDpadLeft }, { "A", "Left" } },
        { "right", { kDpadRight }, { "D", "Right" } },
        { "confirm", { kA }, { "Space", "Mouse1" } },
        { "cancel", { kB }, { "Escape", "Backspace", "Mouse2" } },
        { "start", { kStart }, { "Enter" } },
        { "back", { kBack }, { "Tab" } },
        { "x", { kX }, { "R" } },
        { "y", { kY }, { "F" } },
        { "previous", { kLeftShoulder }, { "Q" } },
        { "next", { kRightShoulder }, { "E" } },
    };
}

std::vector<Action> FlightActions()
{
    auto stick = [](int lx, int ly, int rx, int ry) { Effect e; e.lx = int8_t(lx); e.ly = int8_t(ly); e.rx = int8_t(rx); e.ry = int8_t(ry); return e; };
    Effect lt; lt.lt = true;
    Effect rt; rt.rt = true;
    Effect highG; highG.lt = highG.rt = true;
    // Pitch: the game's stick down (-y) is nose up, as on the pad, pitch_down = stick up.
    return {
        { "pitch_down", stick(0, 1, 0, 0), { "1" } },
        { "pitch_up", stick(0, -1, 0, 0), { "3" } },
        { "roll_left", stick(-1, 0, 0, 0), { "A" } },
        { "roll_right", stick(1, 0, 0, 0), { "D" } },
        { "yaw_left", { kLeftShoulder }, { "Q" } },
        { "yaw_right", { kRightShoulder }, { "E" } },
        { "accelerate", rt, { "W" } },
        { "decelerate", lt, { "S" } },
        { "fire_machine_gun", { kA }, { "Mouse1", "LeftControl" } },
        { "fire_missile", { kB }, { "Mouse2", "Space" } },
        { "change_weapon", { kBack }, { "MouseWheelUp", "MouseWheelDown", "C" } },
        { "camera_control", {}, { "LeftAlt" } },
        { "camera_up", stick(0, 0, 0, 1), { "Numpad8" } },
        { "camera_down", stick(0, 0, 0, -1), { "Numpad2" } },
        { "camera_left", stick(0, 0, -1, 0), { "Numpad4" } },
        { "camera_right", stick(0, 0, 1, 0), { "Numpad6" } },
        { "autopilot", { uint16_t(kLeftShoulder | kRightShoulder) }, { "Z", "X" } },
        { "high_g", highG, { "2" } },
        { "change_view", { kRightThumb }, { "V" } },
        { "switch_radar_map", { kX }, { "R" } },
        { "switch_targets", { kY }, { "Tab" } },
        { "pause", { kStart }, { "Escape" } },
        { "wingman_up", { kDpadUp }, { "Up" } },
        { "wingman_down", { kDpadDown }, { "Down" } },
        { "wingman_left", { kDpadLeft }, { "Left" } },
        { "wingman_right", { kDpadRight }, { "Right" } },
        { "gear", { kLeftThumb }, { "G" } },
    };
}

// A sensitivity setting of 1 is a factor of 1.5 (velocity: full stick at
// 1000 / 1.5 = 667 px/s, position and the camera: full deflection over 500 / 1.5 = 333 px). The settings are
// multiplied by this unit, and MouseConfig holds the result.
constexpr double kSensitivityUnit = 1.5;

struct MouseConfig
{
    bool steer = true;               // mode = "steer" | "off"
    bool cameraFirst = false;        // mouse_controls = "aircraft" | "camera": what it moves without the camera key
    bool velocity = true;            // steer_model = "velocity" | "position"
    double sensX = 1.5, sensY = 1.5;
    bool invertX = false, invertY = true;
    double smoothing = 0.035, curve = 1.0, deadzone = 0.0, antiDeadzone = 0.15, recenter = 3.0;
    // With mouse_controls = "camera": the camera's return to the centre (of a full deflection a second), how long the
    // mouse must be still before it starts (seconds), and where the camera stick starts: just inside the game's stick
    // dead zone (about 29% of full deflection), so the first mouse movements soon turn the view (the steering's
    // anti-deadzone leaves them deep inside it) but a small band around the centre still holds the camera there.
    double cameraRecenter = 0.75, cameraRecenterDelay = 2.0, cameraAntiDeadzone = 0.27;
};

struct Config
{
    std::vector<Action> menu = MenuActions(), flight = FlightActions();
    MouseConfig mouse;
};

Config s_config;

void ApplyDefaults(std::vector<Action>& actions)
{
    for (Action& a : actions)
    {
        a.keys.clear();
        for (std::string_view name : a.defaults)
            ParseKey(name, a.keys);
    }
}

// The file's section for a key set.
const char* SectionOf(std::string_view set)
{
    return set == "menu" ? "menu_keys" : "flight_keys";
}

// The settings file.
std::filesystem::path ConfigFilePath()
{
    const std::filesystem::path p = settings::Path();
    return p.empty() ? std::filesystem::path("ac6_settings.toml") : p;
}

void ReadActions(const toml::table& t, const char* section, std::vector<Action>& actions)
{
    const toml::table* s = t[section].as_table();
    if (!s)
        return;
    for (auto&& [key, node] : *s)
    {
        auto it = std::find_if(actions.begin(), actions.end(), [&](const Action& a) { return key.str() == a.name; });
        if (it == actions.end())
        {
            if (!(std::string_view(section) == "menu_keys" && key.str() == "instances"))
                log::Warn("[KBM] {}: unknown action '{}'", section, key.str());
            continue;
        }
        const toml::array* list = node.as_array();
        if (!list)
            continue;
        it->keys.clear();
        for (auto&& item : *list)
            if (auto name = item.value<std::string>())
                if (!ParseKey(*name, it->keys))
                    log::Warn("[KBM] {}.{}: unknown key '{}'", section, key.str(), *name);
    }
}

// Moves an ac6_input.toml (or the file AC6_INPUT_CONFIG names) into the settings file once: its [mouse], [menu] and
// [flight] become [mouse], [menu_keys] and [flight_keys], when the settings file has none of them. That file is then
// renamed with ".old" added to its name.
void MigrateInputFile()
{
    const char* v = std::getenv("AC6_INPUT_CONFIG");
    const std::filesystem::path old = v && *v ? std::filesystem::path(v) : std::filesystem::path("ac6_input.toml");
    std::error_code ec;
    if (!std::filesystem::exists(old, ec) || !std::filesystem::exists(ConfigFilePath(), ec))
        return;
    toml::parse_result settingsFile = toml::parse_file(platform::PathUtf8(ConfigFilePath()));
    if (!settingsFile)
        return;
    for (const char* section : { "mouse", "menu_keys", "flight_keys" })
        if (settingsFile.table()[section])
            return;   // moved already, or written there
    toml::parse_result parsed = toml::parse_file(platform::PathUtf8(old));
    if (!parsed)
    {
        log::Warn("[KBM] {}: {} (line {}); not moved into {}", platform::PathUtf8(old), parsed.error().description(),
            parsed.error().source().begin.line, platform::PathUtf8(ConfigFilePath()));
        return;
    }
    // Only what differs from the defaults is moved (they stay commented out in the template's form).
    const MouseConfig mouseDefaults;
    auto isDefault = [&](std::string_view section, std::string_view key, const toml::node& node) {
        if (section == "mouse")
        {
            const std::map<std::string_view, double> numbers = { { "sensitivity_x", mouseDefaults.sensX },
                { "sensitivity_y", mouseDefaults.sensY }, { "smoothing", mouseDefaults.smoothing }, { "curve_exponent", mouseDefaults.curve },
                { "deadzone", mouseDefaults.deadzone }, { "anti_deadzone", mouseDefaults.antiDeadzone }, { "recenter", mouseDefaults.recenter },
                { "camera_recenter", mouseDefaults.cameraRecenter },
                { "camera_recenter_delay", mouseDefaults.cameraRecenterDelay },
                { "camera_anti_deadzone", mouseDefaults.cameraAntiDeadzone } };
            if (const auto it = numbers.find(key); it != numbers.end())
                return node.is_number() && std::fabs(node.value<double>().value_or(-1.0) - it->second) < 1e-9;
            if (key == "invert_x" || key == "invert_y")
                return node.is_boolean() && *node.value<bool>() == (key == "invert_x" ? mouseDefaults.invertX : mouseDefaults.invertY);
            if (key == "mode" || key == "steer_model")
                return node.value<std::string>() == std::string(key == "mode" ? "steer" : "velocity");
            return false;
        }
        const toml::array* list = node.as_array();
        if (!list)
            return false;
        for (const Action& a : section == "menu" ? MenuActions() : FlightActions())
            if (key == a.name)
            {
                if (list->size() != a.defaults.size())
                    return false;
                for (size_t i = 0; i < a.defaults.size(); i++)
                    if ((*list)[i].value<std::string>() != std::string(a.defaults[i]))
                        return false;
                return true;
            }
        return false;
    };
    std::vector<settings::TomlLine> lines;
    for (const auto& [from, to] : { std::pair{ "mouse", "mouse" }, std::pair{ "menu", "menu_keys" }, std::pair{ "flight", "flight_keys" } })
        if (const toml::table* t = parsed.table()[from].as_table())
            for (auto&& [key, node] : *t)
            {
                if (isDefault(from, key.str(), node))
                    continue;
                std::ostringstream text;
                if (const toml::array* a = node.as_array())
                {
                    // ["A", "B"], as the template and the overlay write lists (toml++ would write [ 'A', 'B' ]).
                    text << '[';
                    for (size_t i = 0; i < a->size(); i++)
                        if (auto name = (*a)[i].value<std::string>())
                            text << (i ? ", \"" : "\"") << *name << '"';
                    text << ']';
                }
                else if (node.is_string())
                    text << '"' << *node.value<std::string>() << '"';
                else if (node.is_boolean())
                    text << (*node.value<bool>() ? "true" : "false");
                else if (node.is_number() && std::string_view(from) == "mouse" &&
                         (key.str() == "sensitivity_x" || key.str() == "sensitivity_y"))
                    text << std::format("{}", *node.value<double>() / kSensitivityUnit);   // (ac6_input.toml's 1.5 is 1 here)
                else if (node.is_integer())
                    text << *node.value<int64_t>();
                else if (node.is_floating_point())
                    text << *node.value<double>();
                else
                    continue;
                lines.push_back({ to, std::string(key.str()), text.str(), "" });
            }
    std::string error;
    if (!lines.empty() && !settings::SaveLines(lines, error))
    {
        log::Warn("[KBM] moving {} into {}: {}", platform::PathUtf8(old), platform::PathUtf8(ConfigFilePath()), error);
        return;
    }
    std::filesystem::rename(old, std::filesystem::path(old).concat(".old"), ec);
    log::Info("[KBM] moved {} line(s) of {} into {}{}", lines.size(), platform::PathUtf8(old), platform::PathUtf8(ConfigFilePath()),
        ec ? "" : " (the old file renamed .old)");
}

void Load()
{
    Config next;
    ApplyDefaults(next.menu);
    ApplyDefaults(next.flight);
    const std::filesystem::path path = ConfigFilePath();
    std::error_code missing;
    if (!std::filesystem::exists(path, missing))
    {
        s_config = next;   // the defaults
        return;
    }
    toml::parse_result parsed = toml::parse_file(platform::PathUtf8(path));
    if (!parsed)
    {
        log::Warn("[KBM] {}: {} (line {}); keeping the previous bindings", platform::PathUtf8(path), parsed.error().description(),
            parsed.error().source().begin.line);
        if (s_config.menu.front().keys.empty())
            s_config = next;
        return;
    }
    const toml::table& t = parsed.table();
    if (const toml::table* m = t["mouse"].as_table())
    {
        MouseConfig& c = next.mouse;
        c.steer = (*m)["mode"].value_or(std::string("steer")) != "off";
        c.velocity = (*m)["steer_model"].value_or(std::string("velocity")) != "position";
        c.sensX = (*m)["sensitivity_x"].value_or(c.sensX / kSensitivityUnit) * kSensitivityUnit;
        c.sensY = (*m)["sensitivity_y"].value_or(c.sensY / kSensitivityUnit) * kSensitivityUnit;
        c.invertX = (*m)["invert_x"].value_or(c.invertX);
        c.invertY = (*m)["invert_y"].value_or(c.invertY);
        c.smoothing = (*m)["smoothing"].value_or(c.smoothing);
        c.curve = (*m)["curve_exponent"].value_or(c.curve);
        c.deadzone = std::clamp((*m)["deadzone"].value_or(c.deadzone), 0.0, 0.95);
        c.antiDeadzone = std::clamp((*m)["anti_deadzone"].value_or(c.antiDeadzone), 0.0, 0.95);
        c.recenter = (*m)["recenter"].value_or(c.recenter);
        c.cameraRecenter = std::max(0.0, (*m)["camera_recenter"].value_or(c.cameraRecenter));
        c.cameraRecenterDelay = std::max(0.0, (*m)["camera_recenter_delay"].value_or(c.cameraRecenterDelay));
        c.cameraAntiDeadzone = std::clamp((*m)["camera_anti_deadzone"].value_or(c.cameraAntiDeadzone), 0.0, 0.95);
    }
    // The mouse settings the settings layer has (from the environment, the file at start or the overlay's Apply)
    // override the [mouse] table's.
    {
        MouseConfig& c = next.mouse;
        if (const std::string v = settings::Get("AC6_MOUSE_STEER_MODEL"); !v.empty())
            c.velocity = v != "position";
        if (const std::string v = settings::Get("AC6_MOUSE_CONTROLS"); !v.empty())
            c.cameraFirst = v == "camera";
        for (const auto& [env, value] : { std::pair{ "AC6_MOUSE_SENSITIVITY_X", &c.sensX }, std::pair{ "AC6_MOUSE_SENSITIVITY_Y", &c.sensY } })
            if (const std::string v = settings::Get(env); !v.empty() && std::atof(v.c_str()) > 0.0)
                *value = std::atof(v.c_str()) * kSensitivityUnit;
        for (const auto& [env, value] : { std::pair{ "AC6_MOUSE_INVERT_X", &c.invertX }, std::pair{ "AC6_MOUSE_INVERT_Y", &c.invertY } })
            if (const std::string v = settings::Get(env); !v.empty())
                *value = v == "1" || v == "true";
    }
    ReadActions(t, "menu_keys", next.menu);
    ReadActions(t, "flight_keys", next.flight);
    s_config = next;
    log::Info("[KBM] bindings from {}", platform::PathUtf8(path));
}

void ReloadIfChanged()
{
    static int64_t nextCheck = 0;
    static std::filesystem::file_time_type seen{};
    const int64_t now = NowMs();
    if (now < nextCheck)
        return;
    nextCheck = now + 1000;
    // Also when a live setting changed (the overlay's Apply: the mouse settings).
    static uint64_t generation = UINT64_MAX;
    const uint64_t g = settings::Generation();
    std::error_code ec;
    const auto time = std::filesystem::last_write_time(ConfigFilePath(), ec);
    if ((ec || time == seen) && g == generation)
        return;
    seen = time;
    generation = g;
    Load();
}

// ---- Flight heartbeat ---------------------------------------------------------------------------------------------
std::atomic<int64_t> s_flightBeat{ 0 };

bool FlightActive()
{
    return NowMs() - s_flightBeat.load(std::memory_order_relaxed) < 300;
}

// The hangar's display: the shown aircraft's control surfaces, airbrakes, rudder and weapon bay move with the pad as in
// flight, so the flight keys apply there.
std::atomic<int64_t> s_displayBeat{ 0 };

bool DisplayActive()
{
    return NowMs() - s_displayBeat.load(std::memory_order_relaxed) < 300;
}

// ---- Mouse --------------------------------------------------------------------------------------------------------
SDL_Window* s_window = nullptr;
float s_mouseDX = 0.0f, s_mouseDY = 0.0f;   // relative motion since the last update
bool s_relative = false;
double s_rateX = 0.0, s_rateY = 0.0, s_stickX = 0.0, s_stickY = 0.0, s_camX = 0.0, s_camY = 0.0;
int64_t s_camMoved = 0;   // when the mouse last moved the camera (ms)
int64_t s_lastPoll = 0, s_lastMotion = 0;
bool s_cursorHidden = false;

double Shape(double v, const MouseConfig& c)
{
    double a = std::fabs(v);
    if (a <= c.deadzone)
        return 0.0;
    a = (a - c.deadzone) / (1.0 - c.deadzone);
    a = std::pow(a, c.curve);
    if (c.antiDeadzone > 0.0)
        a = std::min(1.0, c.antiDeadzone + a * (1.0 - c.antiDeadzone));
    return std::copysign(a, v);
}

void SetRelative(bool on)
{
    if (on == s_relative || !s_window)
        return;
    SDL_SetWindowRelativeMouseMode(s_window, on);
    s_relative = on;
    s_rateX = s_rateY = s_stickX = s_stickY = s_camX = s_camY = 0.0;
    s_mouseDX = s_mouseDY = 0.0f;
}

void LatchHeld()
{
    for (int i = 0; i < kMaxScancodes; i++)
        s_keyLatched[i] = s_keyDown[i];
    for (size_t i = 0; i < s_mouseDown.size(); i++)
        s_mouseLatched[i] = s_mouseDown[i];
}

} // namespace

bool KeyCodes(std::string_view name, std::vector<uint32_t>& out)
{
    std::vector<Key> keys;
    if (!ParseKey(name, keys))
        return false;
    for (const Key& k : keys)
    {
        if (k.kind == Key::Scancode)
            out.push_back(k.code);
        else if (k.kind == Key::Mouse)
            out.push_back(kMouseCode + k.code);
        else
            return false;   // (the wheel)
    }
    return true;
}

// ---- The settings overlay's view of the bindings ---------------------------------------------

std::vector<BindingAction> BindingActions()
{
    std::vector<BindingAction> out;
    for (const auto& [section, actions] : { std::pair{ "menu", MenuActions() }, std::pair{ "flight", FlightActions() } })
        for (const Action& a : actions)
        {
            BindingAction b{ section, a.name, {} };
            for (std::string_view d : a.defaults)
                b.defaults.emplace_back(d);
            out.push_back(std::move(b));
        }
    return out;
}

std::string BindingsPath()
{
    return platform::PathUtf8(ConfigFilePath());
}

std::map<std::string, std::vector<std::string>> ReadBindings()
{
    std::map<std::string, std::vector<std::string>> out;
    for (const BindingAction& b : BindingActions())
        out[b.section + "." + b.name] = b.defaults;
    std::error_code ec;
    if (!std::filesystem::exists(ConfigFilePath(), ec))
        return out;   // (not written until something is saved: the defaults are its contents)
    toml::parse_result parsed = toml::parse_file(platform::PathUtf8(ConfigFilePath()));
    if (!parsed)
        return out;
    for (const char* section : { "menu", "flight" })
        if (const toml::table* s = parsed.table()[SectionOf(section)].as_table())
            for (auto&& [key, node] : *s)
            {
                auto it = out.find(std::string(section) + "." + std::string(key.str()));
                const toml::array* list = node.as_array();
                if (it == out.end() || !list)
                    continue;
                it->second.clear();
                for (auto&& item : *list)
                    if (auto name = item.value<std::string>())
                    {
                        std::vector<Key> parsedKeys;
                        if (ParseKey(*name, parsedKeys))
                            it->second.push_back(*name);
                    }
            }
    return out;
}

bool SaveBindings(const std::vector<std::pair<BindingAction, std::vector<std::string>>>& changes, std::string& error)
{
    auto list = [](const std::vector<std::string>& keys) {
        std::string out = "[";
        for (size_t i = 0; i < keys.size(); i++)
            out += (i ? ", \"" : "\"") + keys[i] + "\"";
        return out + "]";
    };
    // The defaults commented out again, as the settings file does a setting set back to its default.
    std::vector<settings::TomlLine> lines;
    for (const auto& [action, keys] : changes)
        lines.push_back({ SectionOf(action.section), action.name, keys == action.defaults ? std::string() : list(keys), list(action.defaults) });
    if (!settings::SaveLines(lines, error))
        return false;
    log::Info("[KBM] the settings overlay wrote {} binding(s) to {}", changes.size(), platform::PathUtf8(ConfigFilePath()));
    return true;
}

std::string KeyName(const SDL_Event& event)
{
    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN)
        switch (event.button.button)
        {
        case SDL_BUTTON_LEFT: return "Mouse1";
        case SDL_BUTTON_RIGHT: return "Mouse2";
        case SDL_BUTTON_MIDDLE: return "Mouse3";
        case SDL_BUTTON_X1: return "Mouse4";
        case SDL_BUTTON_X2: return "Mouse5";
        default: return {};
        }
    if (event.type == SDL_EVENT_MOUSE_WHEEL)
        return event.wheel.y > 0 ? "MouseWheelUp" : event.wheel.y < 0 ? "MouseWheelDown" : std::string();
    if (event.type != SDL_EVENT_KEY_DOWN)
        return {};
    const SDL_Scancode sc = event.key.scancode;
    if (sc >= SDL_SCANCODE_A && sc <= SDL_SCANCODE_Z)
        return std::string(1, char('A' + (sc - SDL_SCANCODE_A)));
    if (sc >= SDL_SCANCODE_1 && sc <= SDL_SCANCODE_9)
        return std::string(1, char('1' + (sc - SDL_SCANCODE_1)));
    if (sc == SDL_SCANCODE_0)
        return "0";
    if (sc >= SDL_SCANCODE_F1 && sc <= SDL_SCANCODE_F12)
        return "F" + std::to_string(1 + (sc - SDL_SCANCODE_F1));
    if (sc >= SDL_SCANCODE_F13 && sc <= SDL_SCANCODE_F24)
        return "F" + std::to_string(13 + (sc - SDL_SCANCODE_F13));
    if (sc >= SDL_SCANCODE_KP_1 && sc <= SDL_SCANCODE_KP_9)
        return "Numpad" + std::to_string(1 + (sc - SDL_SCANCODE_KP_1));
    if (sc == SDL_SCANCODE_KP_0)
        return "Numpad0";
    // One name per key, as the default file writes them (ParseKey takes the aliases too).
    static const std::pair<SDL_Scancode, const char*> kNames[] = {
        { SDL_SCANCODE_GRAVE, "Backtick" }, { SDL_SCANCODE_MINUS, "Minus" }, { SDL_SCANCODE_EQUALS, "Plus" },
        { SDL_SCANCODE_COMMA, "Comma" }, { SDL_SCANCODE_PERIOD, "Period" }, { SDL_SCANCODE_SEMICOLON, "Semicolon" },
        { SDL_SCANCODE_SLASH, "Slash" }, { SDL_SCANCODE_BACKSLASH, "Backslash" }, { SDL_SCANCODE_LEFTBRACKET, "LBracket" },
        { SDL_SCANCODE_RIGHTBRACKET, "RBracket" }, { SDL_SCANCODE_APOSTROPHE, "Quote" }, { SDL_SCANCODE_ESCAPE, "Escape" },
        { SDL_SCANCODE_RETURN, "Enter" }, { SDL_SCANCODE_SPACE, "Space" }, { SDL_SCANCODE_TAB, "Tab" },
        { SDL_SCANCODE_BACKSPACE, "Backspace" }, { SDL_SCANCODE_DELETE, "Delete" }, { SDL_SCANCODE_INSERT, "Insert" },
        { SDL_SCANCODE_HOME, "Home" }, { SDL_SCANCODE_END, "End" }, { SDL_SCANCODE_PAGEUP, "PageUp" },
        { SDL_SCANCODE_PAGEDOWN, "PageDown" }, { SDL_SCANCODE_LEFT, "Left" }, { SDL_SCANCODE_RIGHT, "Right" },
        { SDL_SCANCODE_UP, "Up" }, { SDL_SCANCODE_DOWN, "Down" }, { SDL_SCANCODE_KP_ENTER, "NumpadEnter" },
        { SDL_SCANCODE_KP_PLUS, "NumpadPlus" }, { SDL_SCANCODE_KP_MINUS, "NumpadMinus" }, { SDL_SCANCODE_KP_MULTIPLY, "NumpadStar" },
        { SDL_SCANCODE_KP_DIVIDE, "NumpadSlash" }, { SDL_SCANCODE_PRINTSCREEN, "PrintScreen" }, { SDL_SCANCODE_PAUSE, "Pause" },
        { SDL_SCANCODE_CAPSLOCK, "CapsLock" }, { SDL_SCANCODE_NUMLOCKCLEAR, "NumLock" }, { SDL_SCANCODE_SCROLLLOCK, "ScrollLock" },
        { SDL_SCANCODE_LCTRL, "LeftControl" }, { SDL_SCANCODE_RCTRL, "RightControl" }, { SDL_SCANCODE_LALT, "LeftAlt" },
        { SDL_SCANCODE_RALT, "RightAlt" }, { SDL_SCANCODE_LSHIFT, "LeftShift" }, { SDL_SCANCODE_RSHIFT, "RightShift" },
    };
    for (const auto& [code, name] : kNames)
        if (code == sc)
            return name;
    return {};
}

bool InFlight()
{
    return FlightActive();
}

void NotifyHangarDisplay()
{
    s_displayBeat.store(NowMs(), std::memory_order_relaxed);
}

bool InHangarDisplay()
{
    return DisplayActive();
}

bool FlightControls()
{
    return FlightActive() || DisplayActive();
}

bool Enabled()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_KBM"); return v && *v == '1'; }();
    return enabled;
}

void NotifyFlightStep()
{
    s_flightBeat.store(NowMs(), std::memory_order_relaxed);
}

void Init(SDL_Window* window)
{
    s_window = window;
    MigrateInputFile();   // (also with keyboard and mouse off: the settings overlay shows the bindings)
    if (!Enabled())
        return;
    Load();
}

void HandleEvent(const SDL_Event& e)
{
    switch (e.type)
    {
    case SDL_EVENT_KEY_DOWN:
        if (e.key.scancode < kMaxScancodes && !e.key.repeat)
            s_keyDown[e.key.scancode] = true;
        break;
    case SDL_EVENT_KEY_UP:
        if (e.key.scancode < kMaxScancodes)
            s_keyDown[e.key.scancode] = s_keyLatched[e.key.scancode] = false;
        break;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
        if (e.button.button < s_mouseDown.size())
            s_mouseDown[e.button.button] = true;
        s_lastMotion = NowMs();
        break;
    case SDL_EVENT_MOUSE_BUTTON_UP:
        if (e.button.button < s_mouseDown.size())
            s_mouseDown[e.button.button] = s_mouseLatched[e.button.button] = false;
        break;
    case SDL_EVENT_MOUSE_WHEEL:
        if (e.wheel.y > 0)
            s_wheelUpUntil = NowMs() + 90;
        else if (e.wheel.y < 0)
            s_wheelDownUntil = NowMs() + 90;
        s_lastMotion = NowMs();
        break;
    case SDL_EVENT_MOUSE_MOTION:
        s_mouseDX += e.motion.xrel;
        s_mouseDY += e.motion.yrel;
        if (e.motion.xrel != 0.0f || e.motion.yrel != 0.0f)
            s_lastMotion = NowMs();
        break;
    case SDL_EVENT_WINDOW_FOCUS_LOST:
        s_keyDown.fill(false);
        s_keyLatched.fill(false);
        s_mouseDown.fill(false);
        s_mouseLatched.fill(false);
        break;
    default:
        break;
    }
}

void Release()
{
    SetRelative(false);
    s_lastMotion = NowMs();
    if (s_cursorHidden)
    {
        SDL_ShowCursor();
        s_cursorHidden = false;
    }
}

void Apply(PadState& pad)
{
    // The idle cursor hides after AC6_CURSOR_HIDE_SECONDS (default 3, 0 = never) without motion, whatever the input.
    static const double hideAfter = [] { const char* v = std::getenv("AC6_CURSOR_HIDE_SECONDS"); return v && *v ? std::atof(v) : 3.0; }();
    const int64_t now = NowMs();
    const bool focused = s_window && (SDL_GetWindowFlags(s_window) & SDL_WINDOW_INPUT_FOCUS);
    if (!s_relative && hideAfter > 0.0)
    {
        const bool idle = focused && now - s_lastMotion > int64_t(hideAfter * 1000.0);
        if (idle != s_cursorHidden)
        {
            idle ? SDL_HideCursor() : SDL_ShowCursor();
            s_cursorHidden = idle;
        }
    }
    if (!Enabled())
        return;
    ReloadIfChanged();

    // Out of flight only after 10 polls in a row without the flight heartbeat: a hitch longer than the heartbeat's
    // 300 ms (a loading stall) misses it once, and switching sets there and back would latch every key held through it.
    static bool wasFlying = false;
    static uint32_t missed = 0;
    missed = FlightControls() ? 0 : missed + 1;
    const bool flying = wasFlying ? missed < 10 : missed == 0;
    if (flying != wasFlying)
    {
        LatchHeld();   // a key held across the switch means something else in the other set
        wasFlying = flying;
        log::Info("[KBM] {} keys", flying ? "flight" : "menu");
    }

    const std::vector<Action>& actions = flying ? s_config.flight : s_config.menu;
    uint16_t buttons = 0;
    bool lt = false, rt = false;
    int lx = 0, ly = 0, rx = 0, ry = 0;
    bool cameraKey = false;
    if (focused)
        for (const Action& a : actions)
            if (AnyHeld(a.keys))
            {
                if (std::string_view(a.name) == "camera_control")
                    cameraKey = true;
                buttons |= a.effect.buttons;
                lt |= a.effect.lt;
                rt |= a.effect.rt;
                lx += a.effect.lx, ly += a.effect.ly, rx += a.effect.rx, ry += a.effect.ry;
            }

    // The mouse: steering in flight (relative mode, the cursor hidden and held), released everywhere else.
    const MouseConfig& m = s_config.mouse;
    SetRelative(focused && flying && m.steer);
    double mx = 0.0, my = 0.0, camX = 0.0, camY = 0.0;
    if (s_relative)
    {
        const double dt = s_lastPoll ? std::clamp((now - s_lastPoll) / 1000.0, 0.001, 0.1) : 0.016;
        double dx = s_mouseDX, dy = s_mouseDY;
        if (now - s_lastPoll > 250)
            dx = dy = 0.0;   // a stale gap: start again
        if (m.invertX)
            dx = -dx;
        if (m.invertY)
            dy = -dy;
        // The camera with the camera key held, or without it when the mouse controls the camera.
        if (cameraKey != m.cameraFirst)
        {
            s_rateX = s_rateY = 0.0;
            s_camX = std::clamp(s_camX + dx * m.sensX / 500.0, -1.0, 1.0);
            // The camera looks where the mouse moves, up with the mouse up, whatever invert_y: that setting is the
            // nose's (the mouse as a stick). So invert_y is undone here.
            s_camY = std::clamp(s_camY + (m.invertY ? dy : -dy) * m.sensY / 500.0, -1.0, 1.0);
            if (dx != 0.0 || dy != 0.0)
                s_camMoved = now;
            if (m.cameraFirst)
            {
                // The mouse as the camera by default: once it has been still for camera_recenter_delay, the view drifts
                // back to the centre (camera_recenter of a full deflection a second), and the stick is shaped as the
                // steering's but starting at camera_anti_deadzone, so a small movement already turns the view.
                if (now - s_camMoved >= int64_t(m.cameraRecenterDelay * 1000.0) && m.cameraRecenter > 0.0)
                    for (double* c : { &s_camX, &s_camY })
                        *c = *c > 0.0 ? std::max(0.0, *c - m.cameraRecenter * dt) : std::min(0.0, *c + m.cameraRecenter * dt);
                MouseConfig shape = m;
                shape.antiDeadzone = m.cameraAntiDeadzone;
                camX = Shape(s_camX, shape), camY = Shape(s_camY, shape);
            }
            else
                camX = s_camX, camY = s_camY;
        }
        else
        {
            s_camX = s_camY = 0.0;
            if (m.velocity)
            {
                const double alpha = m.smoothing <= 0.0 ? 1.0 : 1.0 - std::exp(-dt / m.smoothing);
                s_rateX += (dx / dt - s_rateX) * alpha;
                s_rateY += (dy / dt - s_rateY) * alpha;
                if (std::fabs(s_rateX) < 2.0) s_rateX = 0.0;
                if (std::fabs(s_rateY) < 2.0) s_rateY = 0.0;
                s_stickX = std::clamp(s_rateX * m.sensX / 1000.0, -1.0, 1.0);
                s_stickY = std::clamp(s_rateY * m.sensY / 1000.0, -1.0, 1.0);
            }
            else
            {
                auto move = [&](double& stick, double d, double sens) {
                    stick += d * sens / 500.0;
                    if (d == 0.0 && m.recenter > 0.0)
                        stick = stick > 0.0 ? std::max(0.0, stick - m.recenter * dt) : std::min(0.0, stick + m.recenter * dt);
                    stick = std::clamp(stick, -1.0, 1.0);
                };
                move(s_stickX, dx, m.sensX);
                move(s_stickY, dy, m.sensY);
            }
            mx = Shape(s_stickX, m);
            my = Shape(s_stickY, m);
        }
    }
    s_mouseDX = s_mouseDY = 0.0f;
    s_lastPoll = now;

    // Into the pad: buttons and triggers added, an axis taken from the keys if pushed, else the mouse, else the pad's.
    pad.buttons |= buttons;
    if (lt) pad.leftTrigger = 255;
    if (rt) pad.rightTrigger = 255;
    auto axis = [](int keys) { return int16_t(std::clamp(keys, -1, 1) * 32767); };
    if (lx) pad.thumbLX = axis(lx); else if (mx != 0.0) pad.thumbLX = int16_t(mx * 32767.0);
    if (ly) pad.thumbLY = axis(ly); else if (my != 0.0) pad.thumbLY = int16_t(-my * 32767.0);
    if (rx) pad.thumbRX = axis(rx); else if (camX != 0.0) pad.thumbRX = int16_t(camX * 32767.0);
    if (ry) pad.thumbRY = axis(ry); else if (camY != 0.0) pad.thumbRY = int16_t(camY * 32767.0);
}

} // namespace ac6::input::kbm
