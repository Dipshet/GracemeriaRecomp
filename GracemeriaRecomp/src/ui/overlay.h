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
// (returns false when nothing is drawn over the game).
bool TakeFrame(DrawFrame& out);

} // namespace ac6::ui

// Mods' drawing (mods/mods.h: the "draw" event): a layer over the game, under the settings overlay, built with the
// overlay's ImGui frame on the window's thread. The functions draw only on that thread while the mods' draw handlers
// run (Active); anywhere else they do nothing. Coordinates are window units (the mouse's), (0, 0) the top left;
// colours 0xAARRGGBB (alpha 0: transparent).
namespace ac6::ui::draw {

bool Active();
// The window, the game's picture in it (the letterboxed frame), and the game's 2D canvas (its 1280x720, 16:9 in the
// picture: where the game draws its HUD and menus).
void Size(float& width, float& height);
void View(float& x, float& y, float& width, float& height);
void Canvas(float& x, float& y, float& width, float& height);
// thickness <= 0: filled. A filled polygon is convex.
void Line(float x0, float y0, float x1, float y1, uint32_t color, float thickness);
void Rect(float x0, float y0, float x1, float y1, uint32_t color, float rounding, float thickness);
void Circle(float x, float y, float radius, uint32_t color, float thickness);
void Triangle(float x0, float y0, float x1, float y1, float x2, float y2, uint32_t color, float thickness);
void Polyline(const float* xy, int points, uint32_t color, bool closed, float thickness);
// Text at its top left, `size` its height in window units (0: 18), in the overlay's font (mono: its numbers' font).
void Text(float x, float y, const char* text, uint32_t color, float size, bool mono);
void TextSize(const char* text, float size, bool mono, float& width, float& height);
// Drawing clipped to a rectangle until PopClip. PopClips: the clips left pushed (after each mod's handlers).
void PushClip(float x0, float y0, float x1, float y1);
void PopClip();
void PopClips();

} // namespace ac6::ui::draw
