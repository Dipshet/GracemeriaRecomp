#pragma once
// The settings overlay: F4, or LS + RS (L3 + R3: both stick clicks) on a controller, opens it over the game.
// Dear ImGui runs on the window's thread at each event pump (the game's input held meanwhile, input::SetHeld). What it
// draws is handed to the renderer, which draws it over the frame it presents (gpu/renderer.cpp, DrawOverlay).
#include <cstdint>
#include <vector>

union SDL_Event;
struct SDL_Window;

namespace ac6::ui {

void Init(SDL_Window* window);              // the window's thread, once
bool HandleEvent(const SDL_Event& event);   // the window's thread: true when the overlay took the event (not the game's)
void Frame();                               // the window's thread, after each pump's events
bool IsOpen();
void SuppressStartupHint();                // replays: no "PRESS F4 OR LS + RS" note over the frames

// One frame of the overlay for the renderer: ImGui's vertices (window pixels, 20 bytes as ImDrawVert), 16-bit indices,
// the draw commands with their clip rectangles, and the texture changes since the renderer last took a frame.
// The texture ID of a draw command that shows the game's frame held under the overlay (the renderer's output target).
constexpr uint32_t kHeldFrameTexture = 0xFFFF0000u;

struct DrawVertex { float x, y, u, v; uint32_t color; };
struct DrawCommand { float clip[4]; uint32_t texture; uint32_t firstIndex, indexCount, vertexOffset; };
struct TextureUpdate { uint32_t id; uint32_t width = 0, height = 0; std::vector<uint8_t> rgba; bool destroy = false; };
struct DrawFrame
{
    float width = 0.0f, height = 0.0f;   // ImGui's display size (window units, the renderer scales to its image)
    std::vector<DrawVertex> vertices;
    std::vector<uint16_t> indices;
    std::vector<DrawCommand> commands;
    std::vector<TextureUpdate> textures;
};
// The renderer, once per presented frame: the textures to create, update or free (always), and the frame to draw
// (returns false when the overlay is closed).
bool TakeFrame(DrawFrame& out);

} // namespace ac6::ui
