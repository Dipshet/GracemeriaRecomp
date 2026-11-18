#include "input/input.h"
#include "input/advanced.h"
#include "input/kbm.h"
#include "input/dualsense.h"
#include "game/frame_rate_fixes.h"
#include "game/game_modes.h"
#include "gpu/frame_rate.h"

#include "log.h"
#include "settings.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <mutex>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace ac6::input {

namespace {

constexpr uint32_t kUsers = 4;

std::mutex s_mutex;
PadState s_published[kUsers];
SDL_Gamepad* s_pads[kUsers] = {};
struct Rumble { bool pending = false; uint16_t left = 0, right = 0; } s_rumble[kUsers];

uint32_t s_frame = 0;
bool s_held = false;   // the settings overlay is open: idle pads, the mouse released
SDL_GamepadType s_padType = SDL_GAMEPAD_TYPE_UNKNOWN;   // player 1's controller (PromptStyle)
// What was down while the overlay was open stays up for the game until let go: the press that closed it (B, Escape)
// isn't the game's (in flight it would fire a missile). Buttons, and the triggers past a quarter.
uint16_t s_heldButtons[kUsers] = {};
bool s_heldTriggers[kUsers][2] = {};
bool s_wasHeld = false;   // the last Update was while the overlay was open
uint32_t s_autoStartPeriod = 0, s_autoStartHold = 4;
// Macro inputs above the button bits: the left stick pushed fully and the triggers pulled, for the whole period.
constexpr uint32_t kStickUp = 1u << 16, kStickDown = 1u << 17, kStickLeft = 1u << 18, kStickRight = 1u << 19, kLeftTrigger = 1u << 20,
    kRightTrigger = 1u << 21, kHoldPeriod = 1u << 22,   // "<combo>~": the buttons held for the whole period too
    kRStickUp = 1u << 23, kRStickDown = 1u << 24, kRStickLeft = 1u << 25, kRStickRight = 1u << 26;   // the right stick (the view)
std::vector<uint32_t> s_autoPresses{ kStart };   // one entry per period
std::vector<std::string> s_autoWaits{ "" };      // per entry: "<combo>@<mode>" repeats until the game enters that mode
bool s_autoLoop = true;                          // the default (Start only) repeats, a given list plays once
// The periods count from the last wait's end (frame 0 without waits).
size_t s_autoBase = 0;
uint32_t s_autoBaseFrame = 0;

// "start", "a", "start+a", "lup+rt", ... -> button mask and macro inputs (0 = unknown name).
uint32_t ParseButtons(std::string_view text)
{
    struct { std::string_view name; uint32_t mask; } constexpr kNames[] = {
        { "start", kStart }, { "back", kBack }, { "a", kA }, { "b", kB }, { "x", kX }, { "y", kY },
        { "up", kDpadUp }, { "down", kDpadDown }, { "left", kDpadLeft }, { "right", kDpadRight },
        { "lb", kLeftShoulder }, { "rb", kRightShoulder }, { "ls", kLeftThumb }, { "rs", kRightThumb },
        { "lup", kStickUp }, { "ldown", kStickDown }, { "lleft", kStickLeft }, { "lright", kStickRight },
        { "rup", kRStickUp }, { "rdown", kRStickDown }, { "rleft", kRStickLeft }, { "rright", kRStickRight },
        { "lt", kLeftTrigger }, { "rt", kRightTrigger }, { "none", 0 },
    };
    uint32_t mask = 0;
    if (text.ends_with('~'))
    {
        mask |= kHoldPeriod;
        text.remove_suffix(1);
    }
    while (!text.empty())
    {
        size_t plus = text.find('+');
        std::string_view name = text.substr(0, plus);
        bool found = false;
        for (auto& n : kNames)
            if (n.name.size() == name.size() &&
                std::equal(name.begin(), name.end(), n.name.begin(), [](char a, char b) { return std::tolower(uint8_t(a)) == b; }))
            {
                mask |= n.mask;
                found = true;
            }
        if (!found)
            log::Warn("AC6_AUTO_START: unknown button '{}'", name);
        text = plus == std::string_view::npos ? std::string_view() : text.substr(plus + 1);
    }
    return mask;
}

// Tests: AC6_CONTROL=<file> (from the environment only, never the settings file) drives player 1 from a harness that
// watches the screen: commands appended to the file, one a line, run in order:
//   press <combo> [frames]   the combo (the macro's names: a, left, lb+rb, lup, rt...) for <frames> (default 4), then 10
//                            frames with nothing, so the next press is a new one
//   wait <frames>            nothing for <frames>
//   until <mode> <combo> [n] the combo every n frames (default 60) until the game is in that mode (game/game_modes.h)
// When a command has run, <file>.done holds the number of commands done, the frame and the game's mode, one line: a
// harness waits for its count there, looks at the window and decides the next press.
struct ControlCommand
{
    uint32_t buttons = 0;   // macro inputs (ParseButtons)
    uint32_t frames = 0;    // pressed for (wait: idle for, until: the period)
    std::string untilMode;  // until: the mode that ends it
};
struct Control
{
    std::string path;
    uint64_t offset = 0;              // the file's bytes read so far
    std::string partial;              // a line not finished yet
    std::deque<ControlCommand> queue;
    uint32_t done = 0;
    uint32_t phase = 0;               // frames into the current command (pressed, then released)
    uint32_t checkFrame = 0;
};
Control* ControlChannel()
{
    static Control* control = [] () -> Control* {
        const char* v = std::getenv("AC6_CONTROL");
        if (!v || !*v || settings::SetByFile("AC6_CONTROL"))
            return nullptr;
        auto* c = new Control;
        c->path = v;
        log::Info("Control channel: commands from {}", c->path);
        return c;
    }();
    return control;
}

void ReadControl(Control& c, uint32_t frame)
{
    if (frame < c.checkFrame)
        return;
    c.checkFrame = frame + 5;
    std::ifstream in(c.path, std::ios::binary);
    if (!in)
        return;
    in.seekg(0, std::ios::end);
    const uint64_t size = uint64_t(in.tellg());
    if (size <= c.offset)
        return;
    in.seekg(std::streamoff(c.offset));
    std::string text(size_t(size - c.offset), '\0');
    in.read(text.data(), std::streamsize(text.size()));
    c.offset = size;
    c.partial += text;
    size_t end;
    while ((end = c.partial.find('\n')) != std::string::npos)
    {
        std::string line = c.partial.substr(0, end);
        c.partial.erase(0, end + 1);
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        const size_t space = line.find(' ');
        const std::string verb = line.substr(0, space), rest = space == std::string::npos ? "" : line.substr(space + 1);
        ControlCommand cmd;
        if (verb == "press")
        {
            const size_t second = rest.find(' ');
            cmd.buttons = ParseButtons(rest.substr(0, second));
            cmd.frames = second == std::string::npos ? 4u : uint32_t(std::max(1, std::atoi(rest.c_str() + second + 1)));
        }
        else if (verb == "wait")
            cmd.frames = uint32_t(std::max(1, std::atoi(rest.c_str())));
        else if (verb == "until")
        {
            const size_t a = rest.find(' '), b = a == std::string::npos ? a : rest.find(' ', a + 1);
            cmd.untilMode = rest.substr(0, a);
            cmd.buttons = a == std::string::npos ? 0u : ParseButtons(rest.substr(a + 1, b == std::string::npos ? b : b - a - 1));
            cmd.frames = b == std::string::npos ? 60u : uint32_t(std::max(10, std::atoi(rest.c_str() + b + 1)));
        }
        else
        {
            if (!line.empty())
                log::Warn("Control channel: unknown command '{}'", line);
            continue;
        }
        c.queue.push_back(cmd);
    }
}

// The current command's input into player 1's state, its end recorded in <file>.done.
void RunControl(Control& c, uint32_t frame, PadState& pad)
{
    ReadControl(c, frame);
    if (c.queue.empty())
        return;
    const ControlCommand& cmd = c.queue.front();
    if (!cmd.untilMode.empty())
    {
        // The combo for 4 frames of every period, until the mode comes.
        if (!game::ModeIs(cmd.untilMode))
        {
            if (c.phase++ % cmd.frames < 4)
                pad.buttons |= uint16_t(cmd.buttons);
            return;
        }
        c.queue.pop_front();
        c.phase = 0;
        c.done++;
        std::ofstream(c.path + ".done", std::ios::binary | std::ios::trunc) << c.done << ' ' << frame << ' ' << game::CurrentMode() << '\n';
        return;
    }
    const uint32_t gap = cmd.buttons ? 10u : 0u;
    if (c.phase < cmd.frames && cmd.buttons)
    {
        pad.buttons |= uint16_t(cmd.buttons);
        if (cmd.buttons & kStickUp) pad.thumbLY = 32767;
        if (cmd.buttons & kStickDown) pad.thumbLY = -32768;
        if (cmd.buttons & kStickLeft) pad.thumbLX = -32768;
        if (cmd.buttons & kStickRight) pad.thumbLX = 32767;
        if (cmd.buttons & kRStickUp) pad.thumbRY = 32767;
        if (cmd.buttons & kRStickDown) pad.thumbRY = -32768;
        if (cmd.buttons & kRStickLeft) pad.thumbRX = -32768;
        if (cmd.buttons & kRStickRight) pad.thumbRX = 32767;
        if (cmd.buttons & kLeftTrigger) pad.leftTrigger = 255;
        if (cmd.buttons & kRightTrigger) pad.rightTrigger = 255;
    }
    if (++c.phase < cmd.frames + gap)
        return;
    c.queue.pop_front();
    c.phase = 0;
    c.done++;
    std::ofstream(c.path + ".done", std::ios::binary | std::ios::trunc) << c.done << ' ' << frame << ' ' << game::CurrentMode() << '\n';
}

int16_t Axis(SDL_Gamepad* pad, SDL_GamepadAxis axis, bool invert)
{
    int v = SDL_GetGamepadAxis(pad, axis);
    if (invert)
        v = -v - 1;   // SDL: +down, Xbox: +up
    return int16_t(std::clamp(v, -32768, 32767));
}

uint8_t Trigger(SDL_Gamepad* pad, SDL_GamepadAxis axis)
{
    return uint8_t(std::clamp<int>(SDL_GetGamepadAxis(pad, axis), 0, 32767) * 255 / 32767);
}

void ReadGamepad(SDL_Gamepad* pad, PadState& s)
{
    struct { SDL_GamepadButton sdl; uint16_t xbox; } constexpr kMap[] = {
        { SDL_GAMEPAD_BUTTON_DPAD_UP, kDpadUp }, { SDL_GAMEPAD_BUTTON_DPAD_DOWN, kDpadDown },
        { SDL_GAMEPAD_BUTTON_DPAD_LEFT, kDpadLeft }, { SDL_GAMEPAD_BUTTON_DPAD_RIGHT, kDpadRight },
        { SDL_GAMEPAD_BUTTON_START, kStart }, { SDL_GAMEPAD_BUTTON_BACK, kBack },
        { SDL_GAMEPAD_BUTTON_LEFT_STICK, kLeftThumb }, { SDL_GAMEPAD_BUTTON_RIGHT_STICK, kRightThumb },
        { SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, kLeftShoulder }, { SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, kRightShoulder },
        { SDL_GAMEPAD_BUTTON_GUIDE, kGuide },
        { SDL_GAMEPAD_BUTTON_SOUTH, kA }, { SDL_GAMEPAD_BUTTON_EAST, kB },
        { SDL_GAMEPAD_BUTTON_WEST, kX }, { SDL_GAMEPAD_BUTTON_NORTH, kY },
    };
    s.connected = true;
    for (auto& m : kMap)
        if (SDL_GetGamepadButton(pad, m.sdl))
            s.buttons |= m.xbox;
    // AC6_TOUCHPAD_BACK=1 (live): a DualShock 4's or DualSense's touchpad click is Back too, the
    // Share / Create button staying Back.
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<bool> touchpadBack{ false };
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        const std::string v = settings::Get("AC6_TOUCHPAD_BACK");
        touchpadBack.store(v == "1" || v == "true", std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    if (touchpadBack.load(std::memory_order_relaxed) && SDL_GetGamepadButton(pad, SDL_GAMEPAD_BUTTON_TOUCHPAD))
        if (const SDL_GamepadType type = SDL_GetGamepadType(pad); type == SDL_GAMEPAD_TYPE_PS4 || type == SDL_GAMEPAD_TYPE_PS5)
            s.buttons |= kBack;
    s.leftTrigger = std::max(s.leftTrigger, Trigger(pad, SDL_GAMEPAD_AXIS_LEFT_TRIGGER));
    s.rightTrigger = std::max(s.rightTrigger, Trigger(pad, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER));
    s.thumbLX = Axis(pad, SDL_GAMEPAD_AXIS_LEFTX, false);
    s.thumbLY = Axis(pad, SDL_GAMEPAD_AXIS_LEFTY, true);
    s.thumbRX = Axis(pad, SDL_GAMEPAD_AXIS_RIGHTX, false);
    s.thumbRY = Axis(pad, SDL_GAMEPAD_AXIS_RIGHTY, true);
}

// The settings overlay's controller shortcut, L3 + R3 together (ui/overlay.cpp), never reaches the game: a stick click
// alone is held back for up to kChordMs to see whether the other one follows (two thumbs pressing "together" land up to
// about a tenth of a second apart). Without it the click goes on to the game, late by that much. A click released
// sooner still reaches the game, for kTapMs. Both down: neither reaches the game until both are let go.
constexpr uint64_t kChordMs = 120, kTapMs = 50;
struct StickChord
{
    uint16_t pending = 0;    // clicks held back, since `since`
    uint64_t since = 0;
    uint16_t passing = 0;    // clicks held past the window: the game's
    uint16_t tap = 0;        // clicks released inside the window: the game's until tapUntil
    uint64_t tapUntil = 0;
    bool chord = false;      // both were down: kept from the game until both are up
} s_chord[kUsers];

void FilterStickChord(StickChord& c, PadState& s, uint64_t now)
{
    constexpr uint16_t kBoth = kLeftThumb | kRightThumb;
    const uint16_t down = s.buttons & kBoth;
    s.buttons &= uint16_t(~kBoth);
    if (c.chord || down == kBoth)
    {
        c.chord = down != 0;
        c.pending = c.passing = c.tap = 0;
        return;
    }
    for (const uint16_t b : { uint16_t(kLeftThumb), uint16_t(kRightThumb) })
    {
        if (down & b)
        {
            if (!(c.pending & b) && !(c.passing & b))
            {
                c.pending |= b;
                c.since = now;
            }
            if ((c.pending & b) && now - c.since >= kChordMs)
            {
                c.pending &= uint16_t(~b);
                c.passing |= b;
            }
        }
        else
        {
            if (c.pending & b)
            {
                c.pending &= uint16_t(~b);
                c.tap |= b;
                c.tapUntil = now + kTapMs;
            }
            c.passing &= uint16_t(~b);
        }
    }
    if (c.tap && now >= c.tapUntil)
        c.tap = 0;
    s.buttons |= uint16_t(c.passing | c.tap);
}

void ReadKeyboard(PadState& s)
{
    const bool* keys = SDL_GetKeyboardState(nullptr);
    if (!keys)
        return;
    struct { SDL_Scancode key; uint16_t xbox; } constexpr kMap[] = {
        { SDL_SCANCODE_RETURN, kStart }, { SDL_SCANCODE_BACKSPACE, kBack },
        { SDL_SCANCODE_UP, kDpadUp }, { SDL_SCANCODE_DOWN, kDpadDown },
        { SDL_SCANCODE_LEFT, kDpadLeft }, { SDL_SCANCODE_RIGHT, kDpadRight },
        { SDL_SCANCODE_SPACE, kA }, { SDL_SCANCODE_LSHIFT, kB }, { SDL_SCANCODE_F, kX }, { SDL_SCANCODE_R, kY },
        { SDL_SCANCODE_Q, kLeftShoulder }, { SDL_SCANCODE_E, kRightShoulder },
        { SDL_SCANCODE_C, kLeftThumb }, { SDL_SCANCODE_V, kRightThumb },
    };
    for (auto& m : kMap)
        if (keys[m.key])
            s.buttons |= m.xbox;
    if (keys[SDL_SCANCODE_Z]) s.leftTrigger = 255;
    if (keys[SDL_SCANCODE_X]) s.rightTrigger = 255;
    auto stick = [&](SDL_Scancode neg, SDL_Scancode pos) -> int16_t {
        return int16_t((keys[pos] ? 32767 : 0) - (keys[neg] ? 32767 : 0));
    };
    if (int16_t v = stick(SDL_SCANCODE_A, SDL_SCANCODE_D)) s.thumbLX = v;
    if (int16_t v = stick(SDL_SCANCODE_S, SDL_SCANCODE_W)) s.thumbLY = v;
    if (int16_t v = stick(SDL_SCANCODE_J, SDL_SCANCODE_L)) s.thumbRX = v;
    if (int16_t v = stick(SDL_SCANCODE_K, SDL_SCANCODE_I)) s.thumbRY = v;
}

void OpenPad(SDL_JoystickID id)
{
    for (auto& slot : s_pads)
    {
        if (!slot)
        {
            slot = SDL_OpenGamepad(id);
            if (slot)
                log::Info("Gamepad connected: {}", SDL_GetGamepadName(slot) ? SDL_GetGamepadName(slot) : "?");
            return;
        }
    }
}

} // namespace

void Init()
{
    if (const char* v = std::getenv("AC6_AUTO_START"); v && *v)
    {
        // <period>[:<hold>[:<presses>]]: without <presses>, Start every period forever. <presses> is a
        // comma-separated sequence of button combos, one per period, played once, "<combo>*<n>" repeats
        // an entry (e.g. "60:4:start*25,a,left,a,a").
        std::string text(v);
        s_autoStartPeriod = std::max(2u, uint32_t(std::stoul(text)));
        if (auto colon = text.find(':'); colon != std::string::npos)
        {
            s_autoStartHold = std::max(1u, uint32_t(std::stoul(text.substr(colon + 1))));
            if (auto colon2 = text.find(':', colon + 1); colon2 != std::string::npos)
            {
                s_autoPresses.clear();
                s_autoWaits.clear();
                std::string_view list = std::string_view(text).substr(colon2 + 1);
                while (!list.empty())
                {
                    size_t comma = list.find(',');
                    std::string_view item = list.substr(0, comma);
                    uint32_t repeat = 1;
                    if (size_t star = item.find('*'); star != std::string_view::npos)
                    {
                        repeat = std::max(1, std::atoi(std::string(item.substr(star + 1)).c_str()));
                        item = item.substr(0, star);
                    }
                    // "<combo>@<mode>" ("@<mode>": no press): the combo every period until the game enters that mode
                    // (game/game_modes.cpp), then the next entry. The periods start again from there.
                    const size_t at = item.find('@');
                    s_autoPresses.insert(s_autoPresses.end(), repeat, at == 0 ? 0u : ParseButtons(item.substr(0, at)));
                    s_autoWaits.insert(s_autoWaits.end(), repeat, at == std::string_view::npos ? std::string() : std::string(item.substr(at + 1)));
                    list = comma == std::string_view::npos ? std::string_view() : list.substr(comma + 1);
                }
                if (s_autoPresses.empty())
                {
                    s_autoPresses.push_back(kStart);
                    s_autoWaits.push_back({});
                }
                s_autoLoop = false;
            }
        }
        s_autoStartHold = std::min(s_autoStartHold, s_autoStartPeriod - 1);
        log::Info("Auto press: player 1 presses {} combo(s) {}, {} of every {} frames", s_autoPresses.size(),
            s_autoLoop ? "repeatedly" : "once in order", s_autoStartHold, s_autoStartPeriod);
    }
    int count = 0;
    if (SDL_JoystickID* ids = SDL_GetGamepads(&count))
    {
        for (int i = 0; i < count; i++)
            OpenPad(ids[i]);
        SDL_free(ids);
    }
    advanced::Init();
}

void HandleEvent(const SDL_Event& event)
{
    kbm::HandleEvent(event);
    advanced::HandleEvent(event);
    if (event.type == SDL_EVENT_GAMEPAD_ADDED)
    {
        for (auto* pad : s_pads)
            if (pad && SDL_GetGamepadID(pad) == event.gdevice.which)
                return;
        OpenPad(event.gdevice.which);
    }
    else if (event.type == SDL_EVENT_GAMEPAD_REMOVED)
    {
        bool removed = false;
        for (auto& pad : s_pads)
            if (pad && SDL_GetGamepadID(pad) == event.gdevice.which)
            {
                SDL_CloseGamepad(pad);
                pad = nullptr;
                removed = true;
                log::Info("Gamepad disconnected");
            }
        if (!removed)
            return;
        // The pads left move down a slot each, so player 1 (the game's first player, and the slot the keyboard, the
        // Advanced controls and the DualSense triggers use) has a controller while any is connected.
        bool moved = false;
        uint32_t to = 0;
        for (uint32_t from = 0; from < kUsers; from++)
            if (s_pads[from])
            {
                if (from != to)
                {
                    s_pads[to] = s_pads[from];
                    s_pads[from] = nullptr;
                    moved = true;
                }
                to++;
            }
        // A pad that found no free slot earlier (a fifth) takes the one freed.
        int count = 0;
        if (SDL_JoystickID* ids = SDL_GetGamepads(&count))
        {
            for (int i = 0; i < count; i++)
            {
                bool open = false;
                for (auto* pad : s_pads)
                    open = open || (pad && SDL_GetGamepadID(pad) == ids[i]);
                if (!open)
                    OpenPad(ids[i]);
            }
            SDL_free(ids);
        }
        if (moved && s_pads[0])
            log::Info("Gamepads: player 1 is now {}", SDL_GetGamepadName(s_pads[0]) ? SDL_GetGamepadName(s_pads[0]) : "?");
    }
}

void Update()
{
    PadState next[kUsers];
    const uint64_t nowMs = SDL_GetTicks();
    for (uint32_t u = 0; u < kUsers; u++)
        if (s_pads[u])
        {
            ReadGamepad(s_pads[u], next[u]);
            FilterStickChord(s_chord[u], next[u], nowMs);
        }
    // Player 1 is always connected: the keyboard (and the macro) drive it even without a controller.
    next[0].connected = true;
    // Advanced controls: in flight, the controller's bound inputs out, then the bindings' in.
    const bool flying = kbm::InFlight();
    advanced::Mask(s_pads[0], next[0], flying, s_held);
    // The keyboard and mouse (input/kbm.cpp, AC6_KBM=1), or the fixed keyboard map. Apply also hides the idle cursor.
    if (s_held)
    {
        kbm::Release();
        if (!kbm::Enabled())
            ReadKeyboard(next[0]);   // (only for what stays up after closing, below)
    }
    else
    {
        kbm::Apply(next[0]);
        if (!kbm::Enabled())
            ReadKeyboard(next[0]);
    }
    if (s_autoStartPeriod)
    {
        size_t step = s_autoBase + (s_frame - s_autoBaseFrame) / s_autoStartPeriod;
        // A wait: the macro stays on it (pressing its combo every period) until its mode comes, then the periods
        // restart with the next entry.
        for (size_t i = s_autoBase; !s_autoLoop && i <= step && i < s_autoPresses.size(); i++)
        {
            if (s_autoWaits[i].empty())
                continue;
            s_autoBaseFrame += uint32_t(i - s_autoBase) * s_autoStartPeriod;
            s_autoBase = i;
            step = i;
            if (game::ModeIs(s_autoWaits[i]))
            {
                log::Info("Auto press step {}/{}: {} reached (frame {})", i + 1, s_autoPresses.size(), s_autoWaits[i], s_frame);
                s_autoBase = step = i + 1;
                s_autoBaseFrame = s_frame;
            }
            break;
        }
        if (s_autoLoop || step < s_autoPresses.size())
        {
            const uint32_t press = s_autoPresses[step % s_autoPresses.size()];
            // Buttons for the hold frames, the stick and triggers for the whole period.
            if (((s_frame - s_autoBaseFrame) % s_autoStartPeriod) < s_autoStartHold || (press & kHoldPeriod))
                next[0].buttons |= uint16_t(press);
            if (press & kStickUp) next[0].thumbLY = 32767;
            if (press & kStickDown) next[0].thumbLY = -32768;
            if (press & kStickLeft) next[0].thumbLX = -32768;
            if (press & kStickRight) next[0].thumbLX = 32767;
            if (press & kRStickUp) next[0].thumbRY = 32767;
            if (press & kRStickDown) next[0].thumbRY = -32768;
            if (press & kRStickLeft) next[0].thumbRX = -32768;
            if (press & kRStickRight) next[0].thumbRX = 32767;
            // AC6_AUTO_TRIGGER=<0-255>: a test's partial pull.
            static const uint8_t pull = uint8_t(std::getenv("AC6_AUTO_TRIGGER") ? std::atoi(std::getenv("AC6_AUTO_TRIGGER")) : 255);
            if (press & kLeftTrigger) next[0].leftTrigger = pull;
            if (press & kRightTrigger) next[0].rightTrigger = pull;
            static size_t logged = SIZE_MAX;
            if (!s_autoLoop && step != logged)
            {
                logged = step;
                log::Info("Auto press step {}/{}", step + 1, s_autoPresses.size());
            }
        }
    }
    if (Control* control = ControlChannel(); control && !s_held)
        RunControl(*control, s_frame, next[0]);
    advanced::Add(s_pads[0], next[0], flying, s_held);
    for (uint32_t u = 0; u < kUsers; u++)
    {
        PadState& pad = next[u];
        if (s_held)
        {
            s_heldButtons[u] = pad.buttons;
            s_heldTriggers[u][0] = pad.leftTrigger > 64;
            s_heldTriggers[u][1] = pad.rightTrigger > 64;
            pad = PadState{ pad.connected };
            continue;
        }
        if (s_wasHeld)
        {
            // Closed since the last update: what is down now came during the overlay (the press that closed it, B,
            // can come between two updates).
            s_heldButtons[u] |= pad.buttons;
            s_heldTriggers[u][0] = s_heldTriggers[u][0] || pad.leftTrigger > 64;
            s_heldTriggers[u][1] = s_heldTriggers[u][1] || pad.rightTrigger > 64;
        }
        s_heldButtons[u] &= pad.buttons;
        pad.buttons &= uint16_t(~s_heldButtons[u]);
        for (int t = 0; t < 2; t++)
        {
            uint8_t& trigger = t ? pad.rightTrigger : pad.leftTrigger;
            s_heldTriggers[u][t] = s_heldTriggers[u][t] && trigger > 16;
            if (s_heldTriggers[u][t])
                trigger = 0;
        }
    }
    s_wasHeld = s_held;
    // The macro counts the console's frames: with the frame rate unlocked (gpu/frame_rate.h) each frame counts as
    // game::FrameStepRatio() of them, so a macro presses at the same pace at any frame rate.
    static double clock = 0.0;
    clock += gpu::Unlocked() ? game::FrameStepRatio() : 1.0;
    s_frame = uint32_t(clock);

    Rumble rumble[kUsers];
    {
        std::lock_guard lock(s_mutex);
        for (uint32_t u = 0; u < kUsers; u++)
        {
            const PadState& prev = s_published[u];
            const bool changed = prev.buttons != next[u].buttons || prev.leftTrigger != next[u].leftTrigger ||
                prev.rightTrigger != next[u].rightTrigger || prev.thumbLX != next[u].thumbLX || prev.thumbLY != next[u].thumbLY ||
                prev.thumbRX != next[u].thumbRX || prev.thumbRY != next[u].thumbRY;
            next[u].packet = prev.packet + (changed ? 1 : 0);
            s_published[u] = next[u];
            rumble[u] = s_rumble[u];
            s_rumble[u].pending = false;   // the values stay for the periodic refresh
        }
    }
    // DualSense adaptive triggers: player 1's pad.
    dualsense::Update(s_pads[0], s_held);
    s_padType = s_pads[0] ? SDL_GetGamepadType(s_pads[0]) : SDL_GAMEPAD_TYPE_UNKNOWN;
    // Xbox vibration lasts until changed. SDL rumble has a duration, so it is sent for 2 s and refreshed every 0.5 s of
    // real time while it's on (counted in frames instead, a hitch or a frame rate under 30 would let it lapse).
    static uint64_t rumbleSent[kUsers] = {};
    for (uint32_t u = 0; u < kUsers; u++)
        if (s_pads[u] && (rumble[u].pending || ((rumble[u].left || rumble[u].right) && nowMs - rumbleSent[u] >= 500)))
        {
            SDL_RumbleGamepad(s_pads[u], rumble[u].left, rumble[u].right, 2000);
            rumbleSent[u] = nowMs;
        }
}

void SetHeld(bool held)
{
    s_held = held;
}

ButtonStyle PromptStyle()
{
    if (settings::Get("AC6_BUTTON_STYLE") != "playstation")
        return ButtonStyle::Xbox;
    const SDL_GamepadType type = s_padType;
    // PlayStation: the connected one's names (Select / Share / Create), else the DualSense's.
    return type == SDL_GAMEPAD_TYPE_PS3 ? ButtonStyle::PlayStation3
        : type == SDL_GAMEPAD_TYPE_PS4 ? ButtonStyle::PlayStation4 : ButtonStyle::PlayStation5;
}

PadState GetState(uint32_t user)
{
    if (user >= kUsers)
        return {};
    std::lock_guard lock(s_mutex);
    return s_published[user];
}

void SetVibration(uint32_t user, uint16_t left, uint16_t right)
{
    if (user >= kUsers)
        return;
    std::lock_guard lock(s_mutex);
    s_rumble[user] = { true, left, right };
}

} // namespace ac6::input
