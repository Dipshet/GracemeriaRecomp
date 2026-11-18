#pragma once
// Controller input: SDL3 gamepads, plus the keyboard on player 1, presented to the
// game as Xbox 360 controllers through XamInputGetState & co.
//
// SDL is only touched on the main thread: Update() runs once per frame from the event pump and
// publishes a snapshot that the input imports (any guest thread) read.
//
// Keyboard and mouse (player 1): input/kbm.cpp with AC6_KBM=1, bindings in ac6_settings.toml's [menu_keys] and
// [flight_keys]. Without it a fixed map: Enter = Start, Backspace = Back, arrows = D-pad, WASD = left stick,
// IJKL = right stick, Space = A, Left Shift = B, F = X, R = Y, Q/E = LB/RB, Z/X = LT/RT, C/V = stick clicks.
//
// AC6_AUTO_START=<period>[:<hold>] presses Start on player 1 for <hold> frames every <period> frames
// (default hold 4), e.g. AC6_AUTO_START=60 to get through the intro without a controller.

#include <cstdint>

union SDL_Event;

namespace ac6::input {

// Xbox 360 button bits (XINPUT_GAMEPAD_*).
enum : uint16_t
{
    kDpadUp = 0x0001, kDpadDown = 0x0002, kDpadLeft = 0x0004, kDpadRight = 0x0008,
    kStart = 0x0010, kBack = 0x0020, kLeftThumb = 0x0040, kRightThumb = 0x0080,
    kLeftShoulder = 0x0100, kRightShoulder = 0x0200, kGuide = 0x0400,
    kA = 0x1000, kB = 0x2000, kX = 0x4000, kY = 0x8000,
};

struct PadState
{
    bool connected = false;
    uint32_t packet = 0;
    uint16_t buttons = 0;
    uint8_t leftTrigger = 0, rightTrigger = 0;
    int16_t thumbLX = 0, thumbLY = 0, thumbRX = 0, thumbRY = 0;
};

void Init();
void HandleEvent(const SDL_Event& event);   // main thread
void Update();                              // main thread, once per frame

PadState GetState(uint32_t user);           // any thread
// While the settings overlay is open (src/ui/overlay.cpp): the game sees idle pads, the mouse is released.
void SetHeld(bool held);                    // main thread
void SetVibration(uint32_t user, uint16_t left, uint16_t right);   // any thread

// The controller buttons to show: AC6_BUTTON_STYLE "playstation" (the Controls tab's Button style), else Xbox's.
// PlayStation's names for Back follow player 1's controller (Select / Share / Create, the DualSense's without one).
// Main thread (the settings overlay).
enum class ButtonStyle { Xbox, PlayStation3, PlayStation4, PlayStation5 };
ButtonStyle PromptStyle();

} // namespace ac6::input
