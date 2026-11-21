#pragma once
// Keyboard and mouse controls (input/kbm.cpp).
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

struct SDL_Window;
union SDL_Event;

namespace ac6::input {
struct PadState;
}

namespace ac6::input::kbm {

bool Enabled();                          // AC6_KBM=1 (off by default: the fixed keyboard map)
void Init(SDL_Window* window);           // main thread
void HandleEvent(const SDL_Event& e);    // main thread
void Apply(PadState& pad);               // main thread, once per frame: player 1's pad plus the keyboard and mouse
void Release();                          // main thread, instead of Apply while the settings overlay is open: the cursor back
void NotifyFlightStep();                 // any thread: the player's flight model ticked
bool InFlight();                         // it ticked within the last 300 ms (flight: not menus, pause, cutscenes)

// The bindings as the settings overlay edits them. Key names are the file's ("Mouse1", "LeftControl").
struct BindingAction
{
    std::string section, name;            // "menu" / "flight", the file's key
    std::vector<std::string> defaults;
};
std::vector<BindingAction> BindingActions();   // the menu set then the flight set, in the file's order
// "section.name" -> its keys as the settings file has them (the defaults for one it leaves out, all of them when it can't
// be read). Names it doesn't know are left out.
std::map<std::string, std::vector<std::string>> ReadBindings();
// Writes those actions' keys into the settings file's [menu_keys] / [flight_keys] (each line
// replaced, its comment kept, the defaults commented out again). The game reads the file again within a second.
bool SaveBindings(const std::vector<std::pair<BindingAction, std::vector<std::string>>>& changes, std::string& error);
// The file's name for the key, mouse button or wheel notch of a key-down, button-down or wheel event ("" for others).
std::string KeyName(const SDL_Event& event);
// A key or mouse button by the file's names ("F5", "LeftControl", "Numpad1", "Mouse1"; "Shift" is either shift): its
// scancodes (SDL_SCANCODE_*) and mouse buttons (kMouseCode + SDL_BUTTON_*). False: no such key (the wheel's names
// included: it isn't held). Any thread.
constexpr uint32_t kMouseCode = 0x10000;
bool KeyCodes(std::string_view name, std::vector<uint32_t>& out);
std::string BindingsPath();

} // namespace ac6::input::kbm
