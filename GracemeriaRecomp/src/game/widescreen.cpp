// Wide and narrow outputs.
//
// With AC6_WIDESCREEN on (the default) an output of another shape than 16:9 is filled: the game's 1280x720 frame is
// rendered with more host pixels per guest pixel along one axis (gpu/video.h, WideFactor k = output aspect / 16:9: above 1
// wide, below 1 narrow). The game is then made to fill that frame correctly. Narrow outputs (16:10, 3:2, 4:3) are the
// same with the axes swapped: vert+ (the 16:9 horizontal view kept, more above and below: the vertical field of view
// widened by 1 / k in every screen projection), the 2D canvas full width in the middle, extended up and down, the bars
// above and below. The wide case:
//  - 3D, hor+: the game's one 16:9 aspect constant (which every camera's lens copies, and the front-end scenes read)
//    becomes the output's: every perspective matrix (two builders, vertical FOV) and every shader given the camera's
//    aspect (the cloud layer) follow. Not the AWACS picture-in-picture views, drawn into 208x144: their projections get
//    16:9 back in the builders' wrappers 0x82271300 / 0x82271350.
//  - 2D, kept 16:9 in the centre: every 2D pixel (HUD, Flash (swg) menus, text, fades) goes through the 2D layers,
//    each mapping a 1280x720 canvas with an ortho and a viewport (below): the canvas lands in the middle at its 16:9
//    proportions, with the HUD's markers still free to reach the screen edges, and the swg layer clipped to the 16:9
//    box.
//  - Untextured sprites covering the whole canvas (the global fade, flashes, wipes) are widened to the screen edges.
//  - Target markers (0x8224EC20: clip space to 1280x720 canvas pixels) are placed for the wide projection: x spans 640 k
//    canvas pixels either side of the centre. The off-screen arrow's aspect correction (640 / 360) takes k too.
//  - Bink movies (a full-viewport quad in clip space) are pillarboxed to 16:9.
//  - Outside missions (menus, briefing, hangar, debriefing: their screens are laid out for 16:9 and put things past its
//    edges) and in cutscenes (unless AC6_WIDESCREEN_CUTSCENES=1), black bars cover the sides: the 3D stays hor+ underneath,
//    so what remains is exactly the 16:9 view (Ace Combat: Assault Horizon bars its menus and cutscenes the same
//    way). Menus whose backdrop covers the canvas have it wrapped round to the sides instead
//    (AC6_WIDE_MENUS=0: barred).
#include "game/widescreen.h"

#include "endian.h"
#include "settings.h"
#include "gpu/frame_rate.h"
#include "gpu/renderer.h"
#include "gpu/video.h"
#include "heap.h"
#include "log.h"
#include "memory.h"
#include "ppc.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <format>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr float kAspect169 = 16.0f / 9.0f;

// k: the output's aspect over 16:9 (1 when the frame isn't wide).
float Wide()
{
    static const float k = [] {
        const float f = ac6::gpu::g_video.WideFactor();
        return std::fabs(f - 1.0f) > 1e-4f ? f : 1.0f;
    }();
    return k;
}

// How much farther than the 16:9 canvas the screen reaches along each axis (one of them is 1): across on a wide output
// (k), up and down on a narrow one (1 / k).
float Kx()
{
    return std::max(Wide(), 1.0f);
}

float Ky()
{
    return std::max(1.0f / Wide(), 1.0f);
}

float F(uint32_t a)
{
    return ac6::g_memory.As<ac6::be<float>>(a)->get();
}

void SetF(uint32_t a, float v)
{
    *ac6::g_memory.As<ac6::be<float>>(a) = v;
}

int32_t I(uint32_t a)
{
    return int32_t(ac6::g_memory.As<ac6::be<uint32_t>>(a)->get());
}

void SetI(uint32_t a, int32_t v)
{
    *ac6::g_memory.As<ac6::be<uint32_t>>(a) = uint32_t(v);
}

// The render view's current viewport size (RenderView_SetViewport 0x822713A0 stores it at view +80/+84, the view is
// [root + 0x370A0], root = [0x826E4E54]).
std::pair<float, float> ViewportSize()
{
    const uint32_t root = uint32_t(I(0x826E4E54));
    const uint32_t view = root ? uint32_t(I(root + 0x370A0)) : 0;
    if (!view)
        return { 0.0f, 0.0f };
    return { F(view + 80), F(view + 84) };
}

// The main view's projection (the last one built for the full 1280x720 viewport, widened), for the HUD (below).
uint8_t s_mainProjection[64];
bool s_haveMainProjection = false;

uint32_t RenderView()
{
    const uint32_t root = uint32_t(I(0x826E4E54));
    return root ? uint32_t(I(root + 0x370A0)) : 0;
}

// Inside one of the AWACS picture-in-picture renders (208x144): EffectScene::RenderView's 0x8209B3B0 and the mission
// managers' twins 0x821A3F68 / 0x821A4858 (wrapped below). Their cameras are whatever the mode set up (root + 0x310E0
// / + 0x338B0, or the campaign manager's own on the heap), so the render is what tells them apart.
thread_local int t_pip = 0;

// The screen's aspect at its source: the 16:9 constant 0x8206A0F4 (g_aspect_16_9), which every camera's lens copies
// (CAce6CameraManager::InitLens 0x8225DDE8 -> camera +216) and the briefing / debriefing / hangar / boot projections read.
// Patched to the output's aspect before the first lens is set up, so whatever reads a camera's aspect agrees with its
// projection: the cloud layer's vertex shader takes it as its own constant (Width_AspectRatio, VS c50.y: with
// only the projection widened, the clouds sit off the horizon when the view rolls).
void PatchAspectConstant()
{
    static bool done = false;
    if (done || Wide() == 1.0f)
        return;
    done = true;
    SetF(0x8206A0F4, kAspect169 * Wide());
    ac6::log::Info("Widescreen: the 16:9 aspect constant set to {:.4f}", kAspect169 * Wide());
}

// The aspect argument (f1 of 0x82271300, f2 of 0x82271350, whose f1 is the field of view in radians), when it is 16:9
// (from a camera set up before the constant was patched) or the patched constant: set to the output's aspect, except in
// the AWACS picture-in-picture renders, which keep 16:9 (their 208x144 picture). Returns whether this is the main
// view's projection (1280x720 viewport). The picture-in-picture is told by its render, not the view's viewport size:
// that stays 208x144 after it until the next pass sets one.
bool WidenAspect(PPCRegister& aspectRegister, bool* screen = nullptr)
{
    const float k = Wide();
    if (screen)
        *screen = false;
    PatchAspectConstant();
    const float aspect = float(aspectRegister.f64), wide = kAspect169 * k;
    const bool is169 = std::fabs(aspect - kAspect169) <= 1e-5f, isWide = std::fabs(aspect - wide) <= 1e-5f;
    if (!is169 && !isWide)
        return false;
    const auto [w, h] = ViewportSize();
    const bool pip = t_pip > 0;
    if (screen)
        *screen = !pip;
    if (k != 1.0f)
    {
        static std::set<std::pair<int, int>> seen;
        if (seen.size() < 16 && seen.insert({ int(w), pip ? -int(h) : int(h) }).second)
            ac6::log::Info("Widescreen: screen-aspect projection with a {}x{} viewport{}", w, h, pip ? " (picture-in-picture: 16:9)" : "");
        aspectRegister.f64 = double(pip ? kAspect169 : wide);
    }
    return !pip && w == 1280.0f && h == 720.0f;
}

// ---- The field of view option -----------------------------------------------------------
// The gameplay camera's vertical field of view, in place of the game's 46 (CAce6CameraManager::InitLens 0x8225DDE8 ->
// camera +212, about 74 horizontal at 16:9): AC6_FOV_OFFSET=<degrees> (0 to 40) adds to the 46, AC6_FOV=<degrees> (a
// legacy variable, 10 to 150, which wins when set) gives it directly. The game's own changes to it (camera modes,
// zooms) are kept as a ratio: tan(fov / 2) is scaled by tan(new fov / 2) / tan(23) in every screen-aspect projection
// that isn't the picture-in-picture's (the main view's passes, the 384x384 / 320x360 views that follow it, and the main
// camera's projection rebuilt after the picture-in-picture while the viewport is still its 208x144: that one is what
// the next frame's object jobs and HUD snapshot start from), and in the main camera's field of view handed to the
// object jobs and read by two HUD functions (mid-asm hooks below). Only in mission gameplay (the CModeTaskGame /
// CModeTaskGameTutorial tasks and the tutorial's other two, no cutscene playing), so cutscenes, replays, menus, the briefing and the hangar keep
// their framing, and never in the AWACS picture-in-picture.
constexpr double kGameFovDegrees = 46.0;
constexpr double kPi = 3.14159265358979323846;

double FovScale()
{
    // Follows FovDegrees (live: the settings overlay's Apply), logged when it changes.
    static thread_local double lastDegrees = -1.0, scale = 1.0;
    const double now = ac6::game::FovDegrees();
    if (now == lastDegrees)
        return scale;
    lastDegrees = now;
    scale = [] {
        const double degrees = ac6::game::FovDegrees();
        if (degrees == kGameFovDegrees)
            return 1.0;
        const double horizontal = 2.0 * std::atan(std::tan(degrees * kPi / 360.0) * kAspect169) * 180.0 / kPi;
        ac6::log::Info("FOV: gameplay vertical field of view {:.1f} degrees (the game's {:.0f}; {:.1f} horizontal at 16:9)", degrees,
            kGameFovDegrees, horizontal);
        return std::tan(degrees * kPi / 360.0) / std::tan(kGameFovDegrees * kPi / 360.0);
    }();
    return scale;
}

// The mode task running (the mode manager [0x8293B930] +8): its vtable, 0 between tasks.
uint32_t ModeTask()
{
    const uint32_t manager = uint32_t(I(0x8293B930));
    const uint32_t task = manager ? uint32_t(I(manager + 8)) : 0;
    return task ? uint32_t(I(task)) : 0;
}

bool FovActive()
{
    if (FovScale() == 1.0 || ac6::gpu::InCutscene())
        return false;
    static bool last = false;
    switch (ModeTask())
    {
    case 0:
        return last;   // between tasks: as before
    case 0x820642F4:   // CModeTaskGame
    case 0x8206474C:   // CModeTaskGameTutorial
    case 0x82066C24:   // CModeTaskGameTutorial_FromMissionMenu
    case 0x82066C84:   // CModeTaskGameTutorial_CampaignMission
        return last = true;
    default:
        return last = false;
    }
}

// tan(fov / 2)'s scale for a screen projection: 1 / k on a narrow output (vert+: the 16:9 horizontal view kept),
// times the option's in gameplay.
double ScreenFovScale()
{
    return double(Ky()) * (FovActive() ? FovScale() : 1.0);
}

// A field of view (radians) for the screen.
double ApplyFov(double radians)
{
    const double scale = ScreenFovScale();
    return scale == 1.0 ? radians : 2.0 * std::atan(std::tan(radians * 0.5) * scale);
}

// AC6_FOV_TRACE=1: the main view's vertical field of view (as the game asked for it, and as built) whenever it changes.
void TraceFov(double asked, double built)
{
    static const bool trace = [] { const char* v = std::getenv("AC6_FOV_TRACE"); return v && *v == '1'; }();
    static double last = -1.0;
    if (!trace || std::abs(built - last) < 0.05)
        return;
    last = built;
    const uint32_t root = uint32_t(I(0x826E4E54));
    const uint32_t camera = root ? uint32_t(I(root + 0x2F9A0)) : 0;
    ac6::log::Info("FOV trace: frame {}: the game's {:.2f} degrees, built {:.2f} (mode task {:08X}; camera {:08X} +400 = {}, +212 = {:.2f})",
        ac6::gpu::RendererFrame(), asked, built, ModeTask(), camera, camera ? I(camera + 400) : -1, camera ? F(camera + 212) : 0.0f);
}

void KeepMainProjection()
{
    if (const uint32_t view = RenderView())
    {
        std::memcpy(s_mainProjection, ac6::g_memory.Translate(view + 16), sizeof(s_mainProjection));
        s_haveMainProjection = true;
    }
}

} // namespace

// The lens set-up of every camera: the aspect constant patched first.
extern "C" PPC_FUNC(__imp__sub_8225DDE8);
PPC_FUNC(sub_8225DDE8)
{
    PatchAspectConstant();
    __imp__sub_8225DDE8(ctx, base);
}

extern "C" PPC_FUNC(__imp__sub_82271300);
PPC_FUNC(sub_82271300)
{
    bool screen = false;
    const bool main = WidenAspect(ctx.f1, &screen);
    // r3: the field of view in 65536ths of a turn (BuildPerspectiveAngle 0x82339868).
    const double asked = double(ctx.r3.s32) * (360.0 / 65536.0);
    if (screen && ScreenFovScale() != 1.0)
        ctx.r3.s64 = int32_t(std::lround(ApplyFov(double(ctx.r3.s32) * (2.0 * kPi / 65536.0)) * (65536.0 / (2.0 * kPi))));
    if (main)
        TraceFov(asked, double(ctx.r3.s32) * (360.0 / 65536.0));
    __imp__sub_82271300(ctx, base);
    if (main)
        KeepMainProjection();
}

extern "C" PPC_FUNC(__imp__sub_82271350);
PPC_FUNC(sub_82271350)
{
    bool screen = false;
    const bool main = WidenAspect(ctx.f2, &screen);
    const double asked = ctx.f1.f64 * 180.0 / kPi;
    if (screen && ScreenFovScale() != 1.0)
        ctx.f1.f64 = ApplyFov(ctx.f1.f64);   // f1: the field of view in radians
    if (main)
        TraceFov(asked, ctx.f1.f64 * 180.0 / kPi);
    __imp__sub_82271350(ctx, base);
    if (main)
        KeepMainProjection();
}

// The AWACS picture-in-picture renders (see t_pip).
extern "C" PPC_FUNC(__imp__sub_8209B3B0);
PPC_FUNC(sub_8209B3B0)
{
    t_pip++;
    __imp__sub_8209B3B0(ctx, base);
    t_pip--;
}

extern "C" PPC_FUNC(__imp__sub_821A3F68);
PPC_FUNC(sub_821A3F68)
{
    t_pip++;
    __imp__sub_821A3F68(ctx, base);
    t_pip--;
}

extern "C" PPC_FUNC(__imp__sub_821A4858);
PPC_FUNC(sub_821A4858)
{
    t_pip++;
    __imp__sub_821A4858(ctx, base);
    t_pip--;
}

// The 2D layers (0x82758DB8, 5 x 40 bytes): Draw2DLayers (0x821383D0) queues each layer's viewport (copy {x, y, w, h, min z,
// max z} at 0x82138530) and ortho (copy {l, r, t, b} at 0x821385A8) commands, r31 = the layer. Mid-asm hooks there:
//  - layer 3 (the swg Flash player: menus, briefing, pause, loading, also the radio text window, credits) keeps its
//    1280x720 ortho and is clipped to the centred 16:9 box, as on the console's 16:9 screen (its movies place things
//    past the canvas edges: gradient panels, icons). Ace Combat: Assault Horizon scissors its Flash layer the same way.
//  - the other full-width layers keep the full viewport and get the ortho widened about the centre.
//  - narrower layers (the radar's clip) keep their ortho and get the viewport moved toward the centre.
namespace {

constexpr uint32_t kLayers = 0x82758DB8, kLayerSize = 40, kSwgLayer = 3;

int LayerIndex(uint32_t layer)
{
    return layer >= kLayers && layer < kLayers + 5 * kLayerSize ? int((layer - kLayers) / kLayerSize) : -1;
}

} // namespace

void WideLayerViewport(PPCRegister& r4, PPCRegister& r31)
{
    const float k = Wide();
    if (k == 1.0f)
        return;
    const uint32_t vp = r4.u32;
    const int32_t x = I(vp), y = I(vp + 4), w = I(vp + 8), h = I(vp + 12);
    const bool swg = LayerIndex(r31.u32) == int(kSwgLayer);
    const bool fullX = x <= 0 && x + w >= 1280, fullY = y <= 0 && y + h >= 720;
    if (fullX && fullY && !swg)
        return;
    // Along each axis the layer spans (not the swg layer): left as it is (its ortho is extended). Otherwise the canvas
    // range is moved toward the centre, where the canvas is on the screen.
    if (!fullX || swg)
    {
        const double x0 = fullX ? 0.0 : x, x1 = fullX ? 1280.0 : double(x) + w;
        const int32_t h0 = int32_t(std::lround(640.0 + (x0 - 640.0) / Kx())), h1 = int32_t(std::lround(640.0 + (x1 - 640.0) / Kx()));
        SetI(vp, h0);
        SetI(vp + 8, std::max(1, h1 - h0));
    }
    if (!fullY || swg)
    {
        const double y0 = fullY ? 0.0 : y, y1 = fullY ? 720.0 : double(y) + h;
        const int32_t v0 = int32_t(std::lround(360.0 + (y0 - 360.0) / Ky())), v1 = int32_t(std::lround(360.0 + (y1 - 360.0) / Ky()));
        SetI(vp + 4, v0);
        SetI(vp + 12, std::max(1, v1 - v0));
    }
    // A clip rect (the radar's): the renderer places it by its canvas pixels, not the rounded guest pixels.
    if (!swg && !fullX && !fullY)
        ac6::gpu::NoteLayerClip(I(vp), I(vp + 4), I(vp + 8), I(vp + 12), x, y, w, h);
}

void WideLayerOrtho(PPCRegister& r4, PPCRegister& r31)
{
    const float k = Wide();
    const uint32_t rect = r4.u32;
    if (k == 1.0f || LayerIndex(r31.u32) == int(kSwgLayer))
        return;
    // {l, r, t, b}: an axis spanning the canvas is extended about its centre to where the screen reaches.
    const float l = F(rect), r = F(rect + 4), t = F(rect + 8), b = F(rect + 12);
    if (l <= 0.5f && r >= 1279.5f && Kx() != 1.0f)
    {
        SetF(rect, 640.0f - (640.0f - l) * Kx());
        SetF(rect + 4, 640.0f + (r - 640.0f) * Kx());
    }
    if (std::min(t, b) <= 0.5f && std::max(t, b) >= 719.5f && Ky() != 1.0f)
    {
        SetF(rect + 8, 360.0f + (t - 360.0f) * Ky());
        SetF(rect + 12, 360.0f + (b - 360.0f) * Ky());
    }
}

// Layer2D_AddSprite (r4 = the sprite description, copied): an untextured sprite ([+68] = -1) covering the whole canvas
// (position +0/+4 (z +8), size +24/+28, scale +32/+36, no rotation +40, top-left pivot +44) reaches the screen edges.
namespace {
std::pair<float, float> PipSpriteShift(uint32_t layer, uint32_t sprite);   // the HUD edges, below
bool Bars();                                                                  // the 16:9 screens, below

// The menus' backgrounds wrapped round to the sides. AC6_WIDE_MENUS=0: barred to 16:9.
bool WideMenus()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<bool> on{ true };
    if (const uint64_t now = ac6::settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        on.store(ac6::settings::Get("AC6_WIDE_MENUS") != "0", std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    return on.load(std::memory_order_relaxed);
}
// This 2D frame's backdrop covered the whole canvas and was wrapped: the bars stay off (Submit2DLayers reads and clears),
// and whether the last one's was (the hangar's tint goes full width only on a screen that wraps).
bool s_menuWrapped = false, s_lastWrapped = false;
// This 2D frame has pictures that end at the 16:9 box on a screen that wraps, and visible ones (frosted sides).
bool s_edgePictures = false, s_edgePicturesVisible = false;
// This 2D frame's thin upright lines on a screen that wraps (the briefing's vertical grid lines): where each is, and
// its sprite description as the game gave it. Submit2DLayers finds the regular sets among them and what continues each
// past the 16:9 box (s_uprightSets). The next frame adds those copies when the set's first line comes, so they sit at
// the lines' own place in the order (added at the frame's end they would lie over what dims the lines in the box, twice
// as bright there). One frame late: nothing to see of that in a steady grid, and a sweep's length lags by a frame.
struct UprightLine { float x; uint32_t texture, alphaBits; uint8_t description[128]; };
struct UprightSet { uint32_t texture; std::vector<float> at; std::vector<UprightLine> copies; bool added; };
std::vector<UprightLine> s_uprightLines;
std::vector<UprightSet> s_uprightSets;
uint32_t s_frames2D = 0;   // 2D frames submitted (the trace's clock)

// AC6_WIDE_MENUS_TRACE=1: what the wide menus do with the sprites, to the log.
bool WideMenusTrace()
{
    static const bool trace = [] { const char* v = std::getenv("AC6_WIDE_MENUS_TRACE"); return v && *v == '1'; }();
    return trace;
}
}

extern "C" PPC_FUNC(__imp__sub_820941F8);
PPC_FUNC(sub_820941F8)
{
    const float k = Wide();
    const uint32_t s = ctx.r4.u32, layer = ctx.r3.u32;   // (r3 doesn't survive the call)
    // The picture-in-picture pictures go out with the HUD (AC6_HUD_EDGES).
    const auto [pipX, pipY] = s ? PipSpriteShift(ctx.r3.u32, s) : std::pair<float, float>{ 0.0f, 0.0f };
    if (pipX != 0.0f || pipY != 0.0f)
    {
        const float px = F(s), py = F(s + 4);
        SetF(s, px + pipX);
        SetF(s + 4, py + pipY);
        __imp__sub_820941F8(ctx, base);
        SetF(s, px);
        SetF(s + 4, py);
        return;
    }
    float x = 0.0f, w = 0.0f, y = 0.0f, h = 0.0f;
    bool widen = false;
    if (k != 1.0f && s && I(s + 68) == -1 && I(s + 40) == 0 && I(s + 44) == 0)
    {
        x = F(s);
        w = F(s + 24);
        y = F(s + 4);
        h = F(s + 28);
        const float sx = F(s + 32), sy = F(s + 36);
        widen = sx > 0.0f && sy > 0.0f && x <= 0.5f && x + w * sx >= 1279.5f && y <= 0.5f && y + h * sy >= 719.5f;
        if (widen)
        {
            const float left = 640.0f - (640.0f - x) * Kx(), right = 640.0f + (x + w * sx - 640.0f) * Kx();
            const float top = 360.0f - (360.0f - y) * Ky(), bottom = 360.0f + (y + h * sy - 360.0f) * Ky();
            SetF(s, left);
            SetF(s + 24, (right - left) / sx);
            SetF(s + 4, top);
            SetF(s + 28, (bottom - top) / sy);
        }
    }
    // The global fade (DrawGlobalFade 0x8211D090, its sprite at 0x82A218F0) goes at the end of the swg layer, which is
    // clipped to 16:9: it moves to the front of layer 4 instead (the dialogs, full width), the same place in the order.
    const uint32_t swg = kLayers + kSwgLayer * kLayerSize, dialogs = swg + kLayerSize;
    const bool fade = widen && ctx.r3.u32 == swg && s == 0x82A218F0;
    if (fade)
        ctx.r3.u64 = dialogs;
    // A translucent picture over the whole canvas on a menu screen that wraps (the aircraft select's tint over the
    // hangar, 1280 x 720 at 55%, its outer ~150 pixels a vignette): stretched to the screen's edges, where its vignette
    // belongs, and moved to layer 1 (full width, drawn before the swg layer: still under the menu, layer 0 drew it
    // lighter, layer 2 is the radar's clipped one). Mirrored at the 16:9 box's edges instead, the vignette would make a
    // dark band there.
    bool overlay = false;
    float overlayX = 0.0f, overlayW = 0.0f;
    if (!widen && s && Kx() > 1.0f && layer == swg && I(s + 40) == 0 && I(s + 44) == 0 && s_lastWrapped && WideMenus() && Bars())
    {
        const float scale = F(s + 32), ox = F(s), oy = F(s + 4), ow = F(s + 24) * scale, oh = F(s + 28) * F(s + 36);
        if (scale > 0.0f && scale < 4.0f && ow > 400.0f && ox <= 0.5f && ox + ow >= 1279.5f && oy <= 0.5f && oy + oh >= 719.5f)
        {
            // A tint, not a picture fading in or out (the boot's Project Aces picture fades in over two seconds on the
            // wrapped black backdrop, translucent all the while: stretched into layer 1 it would stand at the left with
            // its last column smeared across the rest until opaque): its alpha hasn't changed for a quarter of a second
            // (by the clock: a fade steps with the game's ticks, and repeats a value over several frames at a high frame
            // rate), or, seen for the first time, it is well above nothing (a fade starts at nothing). And visible at all.
            using Clock = std::chrono::steady_clock;
            struct Seen { uint32_t texture = 0; float alpha = 0.0f; uint32_t frame = 0; Clock::time_point since; };
            static Seen seen[8];
            static uint32_t next = 0;
            const uint32_t texture = uint32_t(I(s + 68));
            const float alpha = F(s + 60);
            const Clock::time_point now = Clock::now();
            Seen* was = nullptr;
            for (Seen& e : seen)
                if (e.texture == texture)
                    was = &e;
            const bool recent = was && was->frame + 1 >= s_frames2D;
            if (!was)
                was = &seen[next++ % std::size(seen)];
            if (!recent)
                *was = { texture, alpha, s_frames2D, alpha > 0.25f ? now - std::chrono::seconds(1) : now };
            else if (std::fabs(was->alpha - alpha) >= 1.0e-4f)
            {
                was->alpha = alpha;
                was->since = now;
            }
            was->frame = s_frames2D;
            const bool steady = now - was->since >= std::chrono::milliseconds(250);
            if (WideMenusTrace() && alpha > 0.0f)
            {
                // (Each change of what is done with it.)
                static uint32_t lastTexture = 0, logged = 0;
                static int lastState = -1;
                const int state = alpha >= 0.999f ? 0 : steady ? 1 : 2;
                if ((texture != lastTexture || state != lastState) && logged++ < 2000)
                    ac6::log::Info("Wide menus trace: frame {} full-screen picture {:08X} alpha {}: {}", s_frames2D, texture, alpha,
                        state == 0 ? "opaque" : state == 1 ? "a tint: stretched" : "fading: left");
                lastTexture = texture;
                lastState = state;
            }
            overlay = alpha < 0.999f && alpha > 0.004f && steady;
        }
        if (overlay)
        {
            overlayX = ox;
            overlayW = F(s + 24);
            const float left = 640.0f - (640.0f - ox) * Kx(), right = 640.0f + (ox + ow - 640.0f) * Kx();
            SetF(s, left);
            SetF(s + 24, (right - left) / scale);
            ctx.r3.u64 = kLayers + 1 * kLayerSize;
        }
    }
    // Thin lines from a canvas edge on a menu screen that wraps (the briefing's grid lines and its cursor's, sprites of
    // a plain 64 x 64 texture scaled to one canvas pixel high and up to the canvas's width, which would end at the 16:9
    // box, the swg layer's clip, over the wrapped backdrop). A horizontal sprite at most 3
    // canvas pixels high that starts at the canvas's left edge or ends at its right one is drawn across the screen
    // instead: an end at a canvas edge goes to the screen's edge, and its length grows by k from there (the lines sweep
    // in from the left: the sweep then crosses the whole screen, one front). The part inside the box stays in the swg
    // layer, in its place in the order. The parts past the box go to layer 4 (full width), as the wrapped backdrop's.
    bool line = false, lineInBox = true;
    float lineX = 0.0f, lineScale = 0.0f, lineFrom = 0.0f, lineTo = 0.0f;
    // (Upright lines: at most 3 canvas pixels across, 16 or more high, visible: noted for Submit2DLayers.)
    if (!widen && !overlay && s && Kx() > 1.0f && layer == swg && I(s + 40) == 0 && I(s + 44) == 0 && s_lastWrapped && WideMenus() && Bars() &&
        F(s + 60) > 0.004f && s_uprightLines.size() < 128)
    {
        const float across = std::fabs(F(s + 24) * F(s + 32)), tall = std::fabs(F(s + 28) * F(s + 36));
        if (across > 0.0f && across <= 3.0f && tall >= 16.0f)
        {
            UprightLine& u = s_uprightLines.emplace_back();
            u.x = F(s);
            u.texture = uint32_t(I(s + 68));
            u.alphaBits = uint32_t(I(s + 60));
            std::memcpy(u.description, ac6::g_memory.Translate(s), sizeof(u.description));
            // The first line of a set the last frame found: its copies past the box now (layer 4).
            for (UprightSet& set : s_uprightSets)
            {
                if (set.added || set.texture != u.texture ||
                    std::none_of(set.at.begin(), set.at.end(), [&](float x) { return std::fabs(x - u.x) <= 0.75f; }))
                    continue;
                set.added = true;
                static const uint32_t scratch = ac6::g_memory.MapVirtual(ac6::g_heap.Alloc(128));
                for (const UprightLine& copy : set.copies)
                {
                    std::memcpy(ac6::g_memory.Translate(scratch), copy.description, sizeof(copy.description));
                    SetF(scratch, copy.x);
                    PPCContext call = ctx;
                    call.r3.u64 = dialogs;
                    call.r4.u64 = scratch;
                    __imp__sub_820941F8(call, base);
                }
            }
        }
    }
    if (!widen && !overlay && s && Kx() > 1.0f && layer == swg && I(s + 40) == 0 && I(s + 44) == 0 && s_lastWrapped && WideMenus() && Bars())
    {
        const float sx = F(s + 32), texels = F(s + 24), x0 = F(s), length = texels * sx, thick = std::fabs(F(s + 28) * F(s + 36));
        // (Within 8 canvas pixels of an edge: the lines that sweep in from the right keep their right end at 1280 only
        // roughly, 1280.1 at first and 1275.5 near the end of the sweep. A line across the whole canvas may reach past it.)
        const bool across = x0 <= 8.0f && x0 + length >= 1272.0f;
        const bool atLeft = across || (x0 <= 8.0f && x0 >= -8.0f), atRight = across || (x0 + length >= 1272.0f && x0 + length <= 1288.0f);
        if (WideMenusTrace() && sx > 0.0f && thick > 0.0f && thick <= 6.0f && length >= 400.0f && !((atLeft || atRight) && thick <= 3.0f))
        {
            static uint32_t logged = 0;
            if (logged++ < 24)
                ac6::log::Info("Wide menus trace: frame {} a long thin sprite left alone: at {}, {} length {} thick {} texture {:08X} alpha {}", s_frames2D, x0,
                    F(s + 4), length, thick, uint32_t(I(s + 68)), F(s + 60));
        }
        if (sx > 0.0f && texels > 0.0f && length > 0.0f && thick > 0.0f && thick <= 3.0f && (atLeft || atRight) && F(s + 60) > 0.0f)
        {
            const float screenLeft = 640.0f - 640.0f * Kx(), screenRight = 640.0f + 640.0f * Kx();
            line = true;
            lineX = x0;
            lineScale = sx;
            lineFrom = atLeft ? screenLeft : screenRight - length * Kx();
            lineTo = atRight ? screenRight : screenLeft + length * Kx();
            const float from = std::max(lineFrom, 0.0f), to = std::min(lineTo, 1280.0f);
            lineInBox = to > from;
            if (lineInBox)
            {
                SetF(s, from);
                SetF(s + 32, (to - from) / texels);
            }
            if (WideMenusTrace())
            {
                static uint32_t logged = 0;
                if (logged++ < 8)
                    ac6::log::Info("Wide menus trace: frame {} a line at {}, {} length {} texture {:08X}: drawn from {} to {}", s_frames2D, x0, F(s + 4),
                        length, uint32_t(I(s + 68)), lineFrom, lineTo);
            }
        }
    }
    const uint32_t head = fade ? uint32_t(I(dialogs + 32)) : 0, tail = fade ? uint32_t(I(dialogs + 36)) : 0;
    if (lineInBox)
        __imp__sub_820941F8(ctx, base);
    if (line)
    {
        // The parts past the box: left of it, right of it (the box's edges in layer 4's units, as the wrapped copies').
        const double h0 = std::lround(640.0 - 640.0 / Kx()), h1 = std::lround(640.0 + 640.0 / Kx());
        const float boxLeft = float(640.0 + (h0 - 640.0) * Kx()), boxRight = float(640.0 + (h1 - 640.0) * Kx());
        const float texels = F(s + 24);
        const float parts[2][2] = { { lineFrom, std::min(lineTo, boxLeft) }, { std::max(lineFrom, boxRight), lineTo } };
        for (const auto& part : parts)
        {
            if (part[1] - part[0] < 0.25f)
                continue;
            SetF(s, part[0]);
            SetF(s + 32, (part[1] - part[0]) / texels);
            PPCContext call = ctx;
            call.r3.u64 = dialogs;
            call.r4.u64 = s;
            __imp__sub_820941F8(call, base);
        }
        SetF(s, lineX);
        SetF(s + 32, lineScale);
    }
    // A row of tiles across the canvas on a screen that wraps (the briefing cursor's horizontal line, 21 tiles of a
    // 64 x 64 line texture turned a quarter, 64 apart from -32 to 1312, clipped at the box like the lines above). The tile
    // that straddles a canvas edge is added again past the box, a tile's length apart from the box's edge to the screen's
    // (layer 4, the texture runs the same along the line, so the copies need not keep the row's phase).
    if (!line && s && Kx() > 1.0f && layer == swg && I(s + 44) == 0 && s_lastWrapped && WideMenus() && Bars() && F(s + 60) > 0.0f)
    {
        // (The rotation is in 1/65536 of a turn: 16383 the quarter the cursor's tiles have. Either quarter counts.)
        const int32_t turn = I(s + 40) & 0x7FFF;
        const float across = std::fabs(F(s + 24) * F(s + 32)), along = std::fabs(F(s + 28) * F(s + 36)), x0 = F(s);
        if (std::abs(turn - 16384) <= 16 && across >= 8.0f && across <= 128.0f && along >= 8.0f && along <= 128.0f)
        {
            const bool leftTile = x0 < 0.5f && x0 + along > 0.5f, rightTile = x0 < 1279.5f && x0 + along > 1279.5f;
            if (leftTile || rightTile)
            {
                const double h0 = std::lround(640.0 - 640.0 / Kx()), h1 = std::lround(640.0 + 640.0 / Kx());
                const float boxLeft = float(640.0 + (h0 - 640.0) * Kx()), boxRight = float(640.0 + (h1 - 640.0) * Kx());
                const float screenLeft = 640.0f - 640.0f * Kx(), screenRight = 640.0f + 640.0f * Kx();
                if (WideMenusTrace())
                {
                    static uint32_t logged = 0;
                    if (logged++ < 4)
                        ac6::log::Info("Wide menus trace: frame {} a tile row's end at {}, {} ({} along, texture {:08X}): continued to the screen's {} edge",
                            s_frames2D, x0, F(s + 4), along, uint32_t(I(s + 68)), leftTile ? "left" : "right");
                }
                for (int k = 0; k < 32; k++)
                {
                    const float at = leftTile ? boxLeft - along * float(k + 1) : boxRight + along * float(k);
                    if (leftTile ? at + along <= screenLeft : at >= screenRight)
                        break;
                    SetF(s, at);
                    PPCContext call = ctx;
                    call.r3.u64 = dialogs;
                    call.r4.u64 = s;
                    __imp__sub_820941F8(call, base);
                }
                SetF(s, x0);
            }
        }
    }
    if (overlay)
    {
        SetF(s, overlayX);
        SetF(s + 24, overlayW);
    }
    if (fade && head)
    {
        // Appended (old tail +16 -> the new sprite): relinked as the head instead.
        const uint32_t added = uint32_t(I(dialogs + 36));
        SetI(tail + 16, 0);
        SetI(added + 16, int32_t(head));
        SetI(dialogs + 32, int32_t(added));
        SetI(dialogs + 36, int32_t(tail));
    }
    if (widen)
    {
        SetF(s, x);
        SetF(s + 24, w);
        SetF(s + 4, y);
        SetF(s + 28, h);
    }
    // The menus' backgrounds wrapped: on a wide screen outside missions, a swg sprite at least 64 x 64 that touches
    // the canvas's left or right edge and is a pattern (its texture stretched across at least 4 times: the
    // backdrop's gradients and bands, or a tile at most 400 wide: the backdrop's 320 x 180 tiles, not a picture: the
    // title's 800 / 640 x 720 art) is added again 1280 canvas pixels over, where it lies on the screen past the 16:9 box:
    // the image's wrapped continuation, its texture and colour as they are. The copies go to layer 4 (full width, the swg
    // layer is clipped to the box), each once, in the order added. When the whole canvas was covered so, the bars stay off.
    // AC6_WIDE_MENUS_TRACE=1: every distinct sprite reaching a canvas edge outside missions, once, with its layer,
    // size, scale, rotation, pivot, texture and colour (+48), to see what a screen's backdrop is made of.
    static const bool trace = [] { const char* v = std::getenv("AC6_WIDE_MENUS_TRACE"); return v && *v == '1'; }();
    // AC6_WIDE_MENUS_TRACE=2: every distinct long sprite outside missions (200 canvas pixels or more along one axis, by
    // layer, texture, rounded size and rotation), wherever it is, once, with the first place it was seen at (what a
    // screen's lines and cursors are made of).
    static const bool traceAll = [] { const char* v = std::getenv("AC6_WIDE_MENUS_TRACE"); return v && *v == '2'; }();
    if (traceAll && s && Kx() > 1.0f && Bars() &&
        std::max(std::fabs(F(s + 24) * F(s + 32)), std::fabs(F(s + 28) * F(s + 36))) >= 200.0f)
    {
        static std::vector<uint64_t> seenAll;
        uint64_t key = uint64_t(layer) * 0x9E3779B97F4A7C15ull ^ uint64_t(uint32_t(I(s + 68)));
        for (const int32_t part : { int32_t(std::lround(F(s + 24) * F(s + 32) / 8.0f)), int32_t(std::lround(F(s + 28) * F(s + 36) / 8.0f)), I(s + 40), I(s + 44) })
            key = (key ^ uint64_t(uint32_t(part))) * 0x100000001B3ull;
        if (seenAll.size() < 8192 && std::find(seenAll.begin(), seenAll.end(), key) == seenAll.end())
        {
            seenAll.push_back(key);
            ac6::log::Info("Wide menus trace: frame {} layer {} at {}, {} size {} x {} scale {} x {} rot {} pivot {} texture {:08X} uv {} {} colour {} {} {} {} flags {:X}",
                s_frames2D, LayerIndex(layer), F(s), F(s + 4), F(s + 24), F(s + 28), F(s + 32), F(s + 36), I(s + 40), I(s + 44), uint32_t(I(s + 68)),
                F(s + 16), F(s + 20), F(s + 48), F(s + 52), F(s + 56), F(s + 60), uint32_t(I(s + 64)));
        }
    }
    if (trace && s && Kx() > 1.0f && Bars())
    {
        static std::vector<uint64_t> seen;
        const float tx = F(s), tw = F(s + 24) * F(s + 32);
        const uint64_t key = (uint64_t(layer) << 40) ^ (uint64_t(uint32_t(I(s + 68))) << 8) ^ (uint64_t(uint32_t(int(tw))) << 24) ^
            uint64_t(uint32_t(int(tx)));
        if ((tx <= 0.5f || tx + tw >= 1279.5f) && seen.size() < 4096 && std::find(seen.begin(), seen.end(), key) == seen.end())
        {
            seen.push_back(key);
            ac6::log::Info("Wide menus trace: layer {} at {}, {} size {} x {} scale {} x {} rot {} pivot {} texture {:08X} colour {} {} {} {}",
                LayerIndex(layer), tx, F(s + 4), F(s + 24), F(s + 28), F(s + 32), F(s + 36), I(s + 40), I(s + 44), uint32_t(I(s + 68)),
                F(s + 48), F(s + 52), F(s + 56), F(s + 60));
        }
    }
    if (s && Kx() > 1.0f && layer == swg && I(s + 40) == 0 && I(s + 44) == 0 && WideMenus() && Bars())
    {
        const float sx0 = F(s), sy0 = F(s + 4), sw = F(s + 24) * F(s + 32), sh = F(s + 28) * F(s + 36);
        const bool left = sx0 <= 0.5f && sx0 + sw > 0.5f, right = sx0 + sw >= 1279.5f && sx0 < 1279.5f;
        // (A tile is drawn at its own size: the mission title card's smaller smoke pictures, 280 x 360 and 128 x 128,
        // zooming slowly, would be taken for tiles and wrapped to the other side. They get frosted glass, below.)
        const bool pattern = F(s + 32) >= 4.0f || (sw <= 400.0f && std::fabs(F(s + 32) - 1.0f) < 1.0e-3f);
        // The swg layer's box as WideLayerViewport rounds it to whole guest pixels, in layer 4's canvas units (the copies
        // end there: at the canvas's own edge a column would stay uncovered, a pixel past it would draw translucent
        // layers twice).
        const double h0 = std::lround(640.0 - 640.0 / Kx()), h1 = std::lround(640.0 + 640.0 / Kx());
        const float boxLeft = float(640.0 + (h0 - 640.0) * Kx()), boxRight = float(640.0 + (h1 - 640.0) * Kx());
        if (!overlay && sw >= 64.0f && sh >= 64.0f && pattern && (left || right))
        {
            const float screenLeft = 640.0f - 640.0f * Kx(), screenRight = 640.0f + 640.0f * Kx();
            static bool logged = false;
            if (!std::exchange(logged, true))
                ac6::log::Info("Widescreen menus: wrapping backgrounds (first: {} x {} at {}, {}, texture {:08X})", sw, sh, sx0,
                    sy0, uint32_t(I(s + 68)));
            // The sprite from canvas x `from` to `to` (layer 4's units), drawn `shift` over: its texture rectangle cut to
            // that part (the origin +16 and the size +24 are in texels, `to` may run a little past the sprite's end, as
            // the copies up to the box do).
            const float scaleX = F(s + 32), width = F(s + 24), origin = F(s + 16);
            auto piece = [&](float shift, float from, float to) {
                if (to - from < 0.25f)
                    return;
                SetF(s, from);
                SetF(s + 16, origin + (from - (sx0 + shift)) / scaleX);
                SetF(s + 24, (to - from) / scaleX);
                PPCContext call = ctx;
                call.r3.u64 = dialogs;
                call.r4.u64 = s;
                __imp__sub_820941F8(call, base);
                SetF(s, sx0);
                SetF(s + 16, origin);
                SetF(s + 24, width);
            };
            // The wrapped continuations: 1280 to the left where that reaches the screen, 1280 to the right likewise (a
            // right-edge sprite continues on the left, a left-edge one on the right). Each stops at the sprite's own
            // extent, and one that ends (starts) at the canvas's edge goes up to the box (above).
            // (Wherever a copy reaches the screen, not only from the edge the sprite touches. The pre-mission
            // screen's rolling band, 1280 x 128, stands 96 in from the left: it touches only the right edge, yet its
            // own last 96 pixels, cut by the swg layer's clip, and its continuation from 1376 belong on the right.)
            if (const float end = std::min(sx0 + sw - 1280.0f, sx0); end > screenLeft)
                piece(-1280.0f, sx0 - 1280.0f, std::fabs(end) <= 0.5f ? std::max(end, boxLeft) : end);
            if (const float start = std::max(sx0 + 1280.0f, sx0 + sw); start < screenRight)
                piece(1280.0f, std::fabs(start - 1280.0f) <= 0.5f ? std::min(start, boxRight) : start, sx0 + sw + 1280.0f);
            // A wide sprite's own reach past the box (the band's 1280 to 1376): the swg layer's clip cut it.
            // (Not a tile's: a row of tiles past an edge is that row's wrapped end too, and would be drawn twice.)
            if (sw >= 600.0f)
            {
                if (sx0 + sw > boxRight + 0.5f)
                    piece(0.0f, boxRight, std::min(sx0 + sw, screenRight));
                if (sx0 < boxLeft - 0.5f)
                    piece(0.0f, std::max(sx0, screenLeft), boxLeft);
            }
            if (left && right && sy0 <= 0.5f && sy0 + sh >= 719.5f)
                s_menuWrapped = true;
        }
        // A picture standing at one side of a screen that wraps (the mission title card's smoke, 800 x 720 at the
        // left, 640 x 720 and 280 x 360 at the right, over the wrapped backdrop, they end at the 16:9 box, the swg
        // layer's clip): such a screen gets frosted glass at its sides (the renderer's, of the picture in the box:
        // gpu/renderer.cpp PillarFrost), so the pictures end at a pane's edge instead of in the open. (Mirrored past the
        // box, the smoke would read as a mirror image.) Pictures that end about the canvas's edge (from 8
        // short of it to 32 past it: these zoom by a few pixels), of a picture's size (128 to 1000 across: not a glyph
        // or an icon parked off the canvas, not a full-screen picture). One that reaches further out (a panel sliding
        // in) doesn't count. A visible one starts it. While any is there it stays (they fade out before the backdrop
        // does: the glass would go with the screen still lit).
        // (A sprite's size, +24 / +28, is in texels: with its texture origin, +16 / +20, the texture rectangle drawn.)
        else if (const bool nearLeft = sx0 >= -32.0f && sx0 <= 8.0f && sx0 + sw < 1272.0f,
            nearRight = sx0 + sw >= 1272.0f && sx0 + sw <= 1312.0f && sx0 > 8.0f;
            !overlay && F(s + 32) < 4.0f && nearLeft != nearRight && s_lastWrapped && F(s + 32) > 0.0f && F(s + 24) > 0.0f &&
            sw >= 128.0f && sw <= 1000.0f && sh >= 128.0f)
        {
            s_edgePictures = true;
            if (F(s + 60) > 0.004f)
            {
                s_edgePicturesVisible = true;
                if (WideMenusTrace())
                {
                    static std::vector<uint32_t> seen;
                    const uint32_t texture = uint32_t(I(s + 68));
                    if (seen.size() < 256 && std::find(seen.begin(), seen.end(), texture) == seen.end())
                    {
                        seen.push_back(texture);
                        ac6::log::Info("Wide menus trace: frame {} picture {:08X} {} x {} at {}, {} alpha {}: ends at the {} edge (frosted sides)",
                            s_frames2D, texture, sw, sh, sx0, sy0, F(s + 60), nearLeft ? "left" : "right");
                    }
                }
            }
        }
    }
}

// ---- The HUD toward the screen edges (AC6_HUD_EDGES) ------------------------------------------------
// The mission HUD (CAce6HudImpl draw 0x8223DDB0) places every element at the screen centre plus offsets, so on a wide or
// narrow output it stays in the 16:9 box. With AC6_HUD_EDGES=<f> (1 = all the way, 0.5 halfway) the side panels move out
// by f x the room the screen adds on their side (640 (k - 1) across a wide screen, 360 (1/k - 1) up and down a narrow
// one). Done on what each element emits: the HUD renderer (interface vtable 0x820554E8 at HUD +0x28D20) records into a
// block per 2D layer at +76 + layer x 100, with four lists the frame's flush (0x820932B8) turns into layer primitives
// later: line strips (block +16: entries 16 bytes {count, ?, colour, vertex pointer}, vertices 16 bytes {x, y, ..}),
// polygons (+40: 24 bytes {?, count, ?, colour, ?, vertex pointer}, vertices {x, y}), sprites (+64: CAce62DSprite 112
// bytes, x +72 / +92, y +76 / +96) and layer rects (+80: 20 bytes {x, y, w, h, layer}, the radar's clip). Each list
// keeps its count at +8 of its header. After an element draws, the entries it added are moved:
//  - the panels on one side as a whole: the radar (0x82248E98, 0x82249110: left, down), the weapon panel (0x82238948:
//    right, down), the wingman orders (0x82239280: left, down).
//  - the mixed elements per panel, by where it sits: its primitives grouped by touching boxes (a label, its frame and
//    its digits), each group moved by where its box's centre is from the element's own centre (beyond 200 px across,
//    100 up / down, the move fading in over 40 / 30 more): the flight instruments (0x8223C660) only in the corners
//    (score, clock, the destroyed list: speed and altitude by the gun cross stay), the messages (0x8223A2E0) and the
//    score panel of other modes (0x8223BF58, 0x82237C90) on any side. The flight instruments place most of their output
//    from the boresight they are given (r5: it sways as the view turns and the aircraft manoeuvres) and the destroyed
//    list from the centre (r4: fixed). Each panel is measured from its own origin and grouped only with its own kind,
//    so the swaying HUD carries its panels without changing their move and never takes the list along. (Per primitive
//    by the screen's centre, a panel the swaying HUD carried across the threshold would be torn apart glyph by glyph
//    and its pieces would jump by the whole move.)
//  - the HUD draw's own output only where it draws on a picture-in-picture window (its name and bar), with the window.
//    The markers' element (0x8223EE08) is left alone: everything it draws is tied to the world, and moving what it drew
//    over a window's spot would drag world target boxes and labels along.
// Primitives covering most of the canvas (the orders' wipe, the radar's full-screen rect resets) are left alone, but for
// the radar's shapes inside its clip rect (the map's out-of-bounds area).
namespace {

double HudEdges()
{
    // Read again after a live change (the settings overlay's Apply).
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<double> f{ 0.0 };
    if (const uint64_t now = ac6::settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        const std::string v = ac6::settings::Get("AC6_HUD_EDGES");
        const double e = v.empty() ? 0.0 : std::atof(v.c_str());
        f.store(e > 0.0 ? std::min(e, 1.0) : 0.0, std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    return f.load(std::memory_order_relaxed);
}

uint32_t s_hudRenderer = 0;   // the HUD's renderer interface (HUD +0x28D20), set by the HUD draw's wrapper

constexpr uint32_t kHudBlocks = 1, kHudBlock0 = 76, kHudBlockSize = 100;

struct HudCounts { uint32_t lines[kHudBlocks], polys[kHudBlocks], sprites[kHudBlocks], rects[kHudBlocks]; };

HudCounts CountHud(uint32_t r)
{
    HudCounts c{};
    for (uint32_t i = 0; i < kHudBlocks; i++)
    {
        const uint32_t b = r + kHudBlock0 + i * kHudBlockSize;
        c.lines[i] = uint32_t(I(b + 16 + 8));
        c.polys[i] = uint32_t(I(b + 40 + 8));
        c.sprites[i] = uint32_t(I(b + 64 + 8));
        c.rects[i] = uint32_t(I(b + 80 + 8));
    }
    return c;
}

// Whole: a panel moved as one. Clipped: the same for a panel drawn inside its own clip rect (the radar), whose shapes
// may reach far past it (the map's out-of-bounds area at the closer zooms, over 1000 px across): they move too, only its
// full-screen rect resets stay. Split / Corners: by panels (below). InPip: on a picture-in-picture window.
enum class EdgeMode { Whole, Clipped, Split, Corners, InPip };

// The picture-in-picture windows on screen (the last picture sprites seen: centre and size, 16:9 canvas pixels).
struct PipWindow { float x, y, w, h; uint64_t frame; bool seen; };

// A window whose picture was drawn in the last two frames (closed ones stop counting).
bool PipOpen(const PipWindow& p)
{
    return p.seen && ac6::gpu::RendererFrame() <= p.frame + 2;
}
PipWindow s_pipWindows[2] = {};

// The panels' full move across and up / down, in whole canvas pixels: on the canvas's pixel grid (a fraction of a pixel
// left the radar's frame lines, which the renderer snaps to the grid, a part of a pixel off its background and its clip:
// uneven gaps). Rounded once, so the panels on either side stay as far from their edges (std::round: halves away from
// zero alike both ways).
std::pair<float, float> EdgeMove()
{
    return { std::round(float(HudEdges()) * 640.0f * (Kx() - 1.0f)), std::round(float(HudEdges()) * 360.0f * (Ky() - 1.0f)) };
}

// The move for a primitive's box: whole elements by their sides, split ones by where the box is.
std::pair<float, float> EdgeShift(EdgeMode mode, int sideX, int sideY, float x0, float x1, float y0, float y1)
{
    if ((x1 - x0 >= 1000.0f || y1 - y0 >= 600.0f) && mode != EdgeMode::Clipped)
        return { 0.0f, 0.0f };   // full-screen: not a panel
    if (mode == EdgeMode::InPip)
    {
        // Only what is drawn on a picture-in-picture window (its name, its bar): moved with the window.
        const float cx = (x0 + x1) * 0.5f, cy = (y0 + y1) * 0.5f;
        for (const PipWindow& p : s_pipWindows)
            if (PipOpen(p) && std::fabs(cx - p.x) <= p.w * 0.5f + 24.0f && std::fabs(cy - p.y) <= p.h * 0.5f + 30.0f)
                return EdgeShift(EdgeMode::Split, 0, 0, p.x, p.x, p.y, p.y);
        return { 0.0f, 0.0f };
    }
    const auto [dx, dy] = EdgeMove();
    if (mode != EdgeMode::Whole && mode != EdgeMode::Clipped)
    {
        const float cx = (x0 + x1) * 0.5f - 640.0f, cy = (y0 + y1) * 0.5f - 360.0f;
        sideX = cx < -200.0f ? -1 : cx > 200.0f ? 1 : 0;
        sideY = cy < -100.0f ? -1 : cy > 100.0f ? 1 : 0;
        if (mode == EdgeMode::Corners && (sideX == 0 || sideY == 0))
            return { 0.0f, 0.0f };
    }
    return { dx * float(sideX), dy * float(sideY) };
}

// The flight instruments' two origins for the call in progress (DrawHudElement), and the primitives of the last call by
// their box, with whether each was placed from the centre: what keeps its place on the screen while the boresight moves
// was (the destroyed list), what moves with it wasn't. Between moves, a primitive keeps what it was.
std::pair<float, float> s_plainCentre{ 640.0f, 360.0f }, s_plainBoresight{ 640.0f, 360.0f }, s_lastBoresight{ 640.0f, 360.0f };
std::unordered_map<uint64_t, bool> s_lastPlain;
// The destroyed list's primitives in the call in progress: the lists' counts at its start and end (mid-asm hooks
// DestroyedListBegin / End). 0: not drawn, 1: started, 2: drawn.
HudCounts s_listFrom{}, s_listTo{};
int s_listState = 0;

uint64_t BoxKey(float x0, float y0, float x1, float y1)
{
    auto q = [](float v) { return uint64_t(uint16_t(int32_t(std::lround(v * 4.0f)))); };
    return q(x0) | q(y0) << 16 | q(x1) << 32 | q(y1) << 48;
}

std::pair<float, float> PanelShift(EdgeMode mode, float x0, float x1, float y0, float y1, float centreX, float centreY)
{
    if (x1 - x0 >= 1000.0f || y1 - y0 >= 600.0f)
        return { 0.0f, 0.0f };
    auto ramp = [](float v, float from, float width) {
        const float t = std::clamp((std::fabs(v) - from) / width, 0.0f, 1.0f);
        return v < 0.0f ? -t : t;
    };
    const float cx = (x0 + x1) * 0.5f - centreX, cy = (y0 + y1) * 0.5f - centreY;
    float sx = ramp(cx, 200.0f, 40.0f), sy = ramp(cy, 100.0f, 30.0f);
    if (mode == EdgeMode::Corners)
    {
        const float w = std::fabs(sx) * std::fabs(sy);
        sx = cx < 0.0f ? -w : w;
        sy = cy < 0.0f ? -w : w;
    }
    // (Whole pixels also where a panel's move fades in, the same either side.)
    const auto [dx, dy] = EdgeMove();
    return { std::round(dx * sx), std::round(dy * sy) };
}

HudCounts s_pipLabelFrom{};   // the lists' counts where the HUD draw's picture-in-picture label block starts
bool s_pipLabelSeen = false;

enum HudList { kLines, kPolys, kSprites, kRects };

void MoveHudPanels(uint32_t r, const HudCounts& before, const HudCounts& after, EdgeMode mode);

void MoveHud(uint32_t r, const HudCounts& before, const HudCounts& after, EdgeMode mode, int sideX, int sideY)
{
    if (mode == EdgeMode::Split || mode == EdgeMode::Corners)
    {
        MoveHudPanels(r, before, after, mode);
        return;
    }
    // A whole panel moves only where it sits on its side (beyond 200 px of the centre across, 100 up / down): the radar
    // also draws the pause menu's map, right of the centre in the menu's frame, which would otherwise move out of it.
    if ((mode == EdgeMode::Whole || mode == EdgeMode::Clipped) && (sideX != 0 || sideY != 0))
    {
        float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
        auto add = [&](float a0, float a1, float b0, float b1) {
            if (a1 - a0 >= 1000.0f || b1 - b0 >= 600.0f)
                return;   // full-screen (the radar's rect resets): not where the panel is
            x0 = std::min(x0, a0), x1 = std::max(x1, a1), y0 = std::min(y0, b0), y1 = std::max(y1, b1);
        };
        const uint32_t b = r + kHudBlock0;
        struct List { uint32_t header, from, to, entrySize, countAt, pointerAt, vertexSize; };
        const List lists[] = {
            { b + 16, before.lines[0], after.lines[0], 16, 0, 12, 16 },
            { b + 40, before.polys[0], after.polys[0], 24, 4, 20, 8 },
        };
        for (const List& l : lists)
        {
            const uint32_t base = uint32_t(I(l.header));
            for (uint32_t e = l.from; e < l.to && e < l.from + 4096; e++)
            {
                const uint32_t entry = base + e * l.entrySize;
                const uint32_t count = uint32_t(I(entry + l.countAt)), vertices = uint32_t(I(entry + l.pointerAt));
                if (!vertices || count == 0 || count > 4096)
                    continue;
                float a0 = 1e9f, a1 = -1e9f, b0 = 1e9f, b1 = -1e9f;
                for (uint32_t v = 0; v < count; v++)
                {
                    const float x = F(vertices + v * l.vertexSize), y = F(vertices + v * l.vertexSize + 4);
                    a0 = std::min(a0, x), a1 = std::max(a1, x), b0 = std::min(b0, y), b1 = std::max(b1, y);
                }
                add(a0, a1, b0, b1);
            }
        }
        const uint32_t sprites = uint32_t(I(b + 64));
        for (uint32_t e = before.sprites[0]; e < after.sprites[0] && e < before.sprites[0] + 4096; e++)
        {
            const uint32_t q = sprites + e * 112;
            const float x = F(q + 72), y = F(q + 76);
            add(x, x + std::fabs(F(q + 84)), y, y + std::fabs(F(q + 88)));
        }
        // Where the panel's clip rect is, when it has one: its shapes can reach well past it (the radar's lines to incoming
        // missiles during an alert would pull the box toward the centre, leaving the radar in place while its bar,
        // drawn by its second part, moves).
        const uint32_t rects = uint32_t(I(b + 80));
        float r0 = 1e9f, r1 = -1e9f, s0 = 1e9f, s1 = -1e9f;
        for (uint32_t e = before.rects[0]; e < after.rects[0] && e < before.rects[0] + 256; e++)
        {
            const uint32_t q = rects + e * 20;
            const float a0 = float(I(q)), a1 = float(I(q) + I(q + 8)), b0 = float(I(q + 4)), b1 = float(I(q + 4) + I(q + 12));
            if (a1 - a0 < 1000.0f && b1 - b0 < 600.0f)
                r0 = std::min(r0, a0), r1 = std::max(r1, a1), s0 = std::min(s0, b0), s1 = std::max(s1, b1);
        }
        if (r0 <= r1)
            x0 = r0, x1 = r1, y0 = s0, y1 = s1;
        if (x0 <= x1)
        {
            const float cx = (x0 + x1) * 0.5f - 640.0f, cy = (y0 + y1) * 0.5f - 360.0f;
            if (sideX != 0 && !(sideX < 0 ? cx < -200.0f : cx > 200.0f))
                sideX = 0;
            if (sideY != 0 && !(sideY < 0 ? cy < -100.0f : cy > 100.0f))
                sideY = 0;
            if (sideX == 0 && sideY == 0)
                return;
        }
    }
    for (uint32_t i = 0; i < kHudBlocks; i++)
    {
        const uint32_t b = r + kHudBlock0 + i * kHudBlockSize;
        // Line strips and polygons: their vertices, moved together.
        struct List { HudList id; uint32_t header, from, to, entrySize, countAt, pointerAt, vertexSize; };
        const List lists[] = {
            { kLines, b + 16, before.lines[i], after.lines[i], 16, 0, 12, 16 },
            { kPolys, b + 40, before.polys[i], after.polys[i], 24, 4, 20, 8 },
        };
        for (const List& l : lists)
        {
            const uint32_t base = uint32_t(I(l.header));
            for (uint32_t e = l.from; e < l.to && e < l.from + 4096; e++)
            {
                const uint32_t entry = base + e * l.entrySize;
                const uint32_t count = uint32_t(I(entry + l.countAt)), vertices = uint32_t(I(entry + l.pointerAt));
                if (!vertices || count == 0 || count > 4096)
                    continue;
                float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
                for (uint32_t v = 0; v < count; v++)
                {
                    const float x = F(vertices + v * l.vertexSize), y = F(vertices + v * l.vertexSize + 4);
                    x0 = std::min(x0, x), x1 = std::max(x1, x), y0 = std::min(y0, y), y1 = std::max(y1, y);
                }
                const auto [dx, dy] = EdgeShift(mode, sideX, sideY, x0, x1, y0, y1);
                if (dx != 0.0f || dy != 0.0f)
                    for (uint32_t v = 0; v < count; v++)
                    {
                        SetF(vertices + v * l.vertexSize, F(vertices + v * l.vertexSize) + dx);
                        SetF(vertices + v * l.vertexSize + 4, F(vertices + v * l.vertexSize + 4) + dy);
                    }
            }
        }
        // Sprites: the top-left corner (+72 / +76, size +84 / +88) and the description's position (+92 / +96).
        const uint32_t sprites = uint32_t(I(b + 64));
        for (uint32_t e = before.sprites[i]; e < after.sprites[i] && e < before.sprites[i] + 4096; e++)
        {
            const uint32_t s = sprites + e * 112;
            const float x = F(s + 72), y = F(s + 76), w = std::fabs(F(s + 84)), h = std::fabs(F(s + 88));
            const auto [dx, dy] = EdgeShift(mode, sideX, sideY, x, x + w, y, y + h);
            SetF(s + 72, x + dx);
            SetF(s + 92, F(s + 92) + dx);
            SetF(s + 76, y + dy);
            SetF(s + 96, F(s + 96) + dy);
        }
        // Layer rects (the radar's clip), in whole pixels.
        const uint32_t rects = uint32_t(I(b + 80));
        for (uint32_t e = before.rects[i]; e < after.rects[i] && e < before.rects[i] + 256; e++)
        {
            const uint32_t q = rects + e * 20;
            const int32_t x = I(q), y = I(q + 4), w = I(q + 8), h = I(q + 12);
            const EdgeMode rectMode = mode == EdgeMode::Clipped ? EdgeMode::Whole : mode;   // its resets stay
            const auto [dx, dy] = EdgeShift(rectMode, sideX, sideY, float(x), float(x + w), float(y), float(y + h));
            SetI(q, x + int32_t(std::lround(dx)));
            SetI(q + 4, y + int32_t(std::lround(dy)));
        }
    }
}

// A split element's output by panels: every primitive's box, grouped where boxes touch (within 6 px, long ones such as
// the pitch ladder's lines, over 300 x 200 px, stay alone), each group moved by PanelShift of its box.
void MoveHudPanels(uint32_t r, const HudCounts& before, const HudCounts& after, EdgeMode mode)
{
    struct Item { HudList list; uint32_t address, count, vertexSize; float x0, x1, y0, y1; bool fromCentre = false; uint32_t index = 0; };
    std::vector<Item> items;
    const uint32_t b = r + kHudBlock0;
    // Line strips (entries 16 bytes: count +0, vertices +12, 16 bytes each) and polygons (24: +4, +20, 8).
    struct List { HudList id; uint32_t header, from, to, entrySize, countAt, pointerAt, vertexSize; };
    const List lists[] = {
        { kLines, b + 16, before.lines[0], after.lines[0], 16, 0, 12, 16 },
        { kPolys, b + 40, before.polys[0], after.polys[0], 24, 4, 20, 8 },
    };
    for (const List& l : lists)
    {
        const uint32_t base = uint32_t(I(l.header));
        for (uint32_t e = l.from; e < l.to && e < l.from + 4096; e++)
        {
            const uint32_t entry = base + e * l.entrySize;
            const uint32_t count = uint32_t(I(entry + l.countAt)), vertices = uint32_t(I(entry + l.pointerAt));
            if (!vertices || count == 0 || count > 4096)
                continue;
            Item it{ l.id, vertices, count, l.vertexSize, 1e9f, -1e9f, 1e9f, -1e9f };
            it.index = e;
            for (uint32_t v = 0; v < count; v++)
            {
                const float x = F(vertices + v * l.vertexSize), y = F(vertices + v * l.vertexSize + 4);
                it.x0 = std::min(it.x0, x), it.x1 = std::max(it.x1, x), it.y0 = std::min(it.y0, y), it.y1 = std::max(it.y1, y);
            }
            items.push_back(it);
        }
    }
    const uint32_t sprites = uint32_t(I(b + 64));
    for (uint32_t e = before.sprites[0]; e < after.sprites[0] && e < before.sprites[0] + 4096; e++)
    {
        const uint32_t s = sprites + e * 112;
        const float x = F(s + 72), y = F(s + 76), w = std::fabs(F(s + 84)), h = std::fabs(F(s + 88));
        items.push_back({ kSprites, s, 0, 0, x, x + w, y, y + h });
        items.back().index = e;
    }
    const uint32_t rects = uint32_t(I(b + 80));
    for (uint32_t e = before.rects[0]; e < after.rects[0] && e < before.rects[0] + 256; e++)
    {
        const uint32_t q = rects + e * 20;
        const float x = float(I(q)), y = float(I(q + 4));
        items.push_back({ kRects, q, 0, 0, x, x + float(I(q + 8)), y, y + float(I(q + 12)) });
        items.back().index = e;
    }
    // Moves an item's primitive (and its box).
    auto move = [](Item& it, float dx, float dy) {
        switch (it.list)
        {
        case kLines:
        case kPolys:
            for (uint32_t v = 0; v < it.count; v++)
            {
                SetF(it.address + v * it.vertexSize, F(it.address + v * it.vertexSize) + dx);
                SetF(it.address + v * it.vertexSize + 4, F(it.address + v * it.vertexSize + 4) + dy);
            }
            break;
        case kSprites:
            SetF(it.address + 72, F(it.address + 72) + dx);
            SetF(it.address + 92, F(it.address + 92) + dx);
            SetF(it.address + 76, F(it.address + 76) + dy);
            SetF(it.address + 96, F(it.address + 96) + dy);
            break;
        case kRects:
            SetI(it.address, I(it.address) + int32_t(std::lround(dx)));
            SetI(it.address + 4, I(it.address + 4) + int32_t(std::lround(dy)));
            break;
        }
        it.x0 += dx, it.x1 += dx, it.y0 += dy, it.y1 += dy;
    };
    // The flight instruments: which primitives were placed from the centre. The destroyed list is, and nothing else: what
    // was drawn between its start and its end (the mid-asm hooks). By how they moved since the last call (below) a new
    // or scrolled entry had nothing to go by, and a list passed by other things (an enemy's target box near it) was torn
    // apart, its pieces moved with the swaying HUD.
    if (mode == EdgeMode::Corners && s_listState != 1)
    {
        for (Item& it : items)
        {
            const uint32_t from = it.list == kLines ? s_listFrom.lines[0] : it.list == kPolys ? s_listFrom.polys[0] :
                it.list == kSprites ? s_listFrom.sprites[0] : s_listFrom.rects[0];
            const uint32_t to = it.list == kLines ? s_listTo.lines[0] : it.list == kPolys ? s_listTo.polys[0] :
                it.list == kSprites ? s_listTo.sprites[0] : s_listTo.rects[0];
            it.fromCentre = s_listState == 2 && it.index >= from && it.index < to;
        }
    }
    // (Without the hooks' marks: by how they moved since the last call.)
    else if (mode == EdgeMode::Corners)
    {
        const float dx = s_plainBoresight.first - s_lastBoresight.first, dy = s_plainBoresight.second - s_lastBoresight.second;
        const bool moved = std::fabs(dx) > 0.5f || std::fabs(dy) > 0.5f;
        std::unordered_map<uint64_t, bool> now;
        for (Item& it : items)
        {
            const uint64_t key = BoxKey(it.x0, it.y0, it.x1, it.y1);
            const auto stayed = s_lastPlain.find(key);
            const auto followed = moved ? s_lastPlain.find(BoxKey(it.x0 - dx, it.y0 - dy, it.x1 - dx, it.y1 - dy)) : s_lastPlain.end();
            if (moved && stayed != s_lastPlain.end() && followed == s_lastPlain.end())
                it.fromCentre = true;
            else if (moved && followed != s_lastPlain.end() && stayed == s_lastPlain.end())
                it.fromCentre = false;
            else if (stayed != s_lastPlain.end())
                it.fromCentre = stayed->second;
            else if (followed != s_lastPlain.end())
                it.fromCentre = followed->second;
            now[key] = it.fromCentre;
        }
        s_lastPlain = std::move(now);
        s_lastBoresight = s_plainBoresight;
    }
    // Groups: union-find over touching boxes of the same origin.
    const size_t n = items.size();
    std::vector<uint32_t> parent(n);
    for (size_t i = 0; i < n; i++)
        parent[i] = uint32_t(i);
    auto find = [&](uint32_t i) {
        while (parent[i] != i)
            i = parent[i] = parent[parent[i]];
        return i;
    };
    auto groupable = [](const Item& a) { return a.x1 - a.x0 <= 300.0f && a.y1 - a.y0 <= 200.0f; };
    constexpr float kGap = 6.0f;
    for (size_t i = 0; i < n; i++)
    {
        if (!groupable(items[i]))
            continue;
        for (size_t j = i + 1; j < n; j++)
            if (groupable(items[j]) && items[i].fromCentre == items[j].fromCentre && items[i].x0 <= items[j].x1 + kGap &&
                items[j].x0 <= items[i].x1 + kGap && items[i].y0 <= items[j].y1 + kGap && items[j].y0 <= items[i].y1 + kGap)
                parent[find(uint32_t(i))] = find(uint32_t(j));
    }
    struct Box { float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f; };
    // Each group's origin: the flight instruments' boresight or centre (the screen's for the other elements).
    const bool plain = mode == EdgeMode::Corners;
    static const bool trace = [] { const char* v = std::getenv("AC6_HUD_EDGES_TRACE"); return v && *v == '1'; }();
    if (trace && plain)
    {
        size_t fromCentre = 0;
        for (const Item& it : items)
            fromCentre += it.fromCentre;
        ac6::log::Info("HUD edges: frame {} boresight {:.1f} {:.1f}, centre {:.1f} {:.1f}; {} primitives, {} from the centre",
            ac6::gpu::RendererFrame(), s_plainBoresight.first, s_plainBoresight.second, s_plainCentre.first, s_plainCentre.second, n, fromCentre);
    }
    std::vector<Box> boxes(n);
    for (size_t i = 0; i < n; i++)
    {
        Box& g = boxes[find(uint32_t(i))];
        g.x0 = std::min(g.x0, items[i].x0), g.x1 = std::max(g.x1, items[i].x1);
        g.y0 = std::min(g.y0, items[i].y0), g.y1 = std::max(g.y1, items[i].y1);
    }
    // The destroyed list (the flight instruments' primitives from the centre) moves as one panel, by its top-left corner:
    // its header, which stays put while entries come and go below it. By its groups' centres it was moved less the more
    // entries it had (the move fades in from 100 px above the centre: four entries brought its centre down into the fade),
    // and its count and name columns, apart, moved by different amounts.
    Box list{};
    bool hasList = false;
    for (size_t i = 0; i < n; i++)
        if (plain && items[i].fromCentre)
        {
            list = hasList ? Box{ std::min(list.x0, items[i].x0), std::max(list.x1, items[i].x1), std::min(list.y0, items[i].y0),
                std::max(list.y1, items[i].y1) } : Box{ items[i].x0, items[i].x1, items[i].y0, items[i].y1 };
            hasList = true;
        }
    const auto listShift = hasList ? PanelShift(mode, list.x0, list.x0, list.y0, list.y0, s_plainCentre.first, s_plainCentre.second) :
        std::pair<float, float>{ 0.0f, 0.0f };
    if (trace && hasList)
        ac6::log::Info("HUD edges: the destroyed list at {:.1f} {:.1f} .. {:.1f} {:.1f}, moved {:.1f} {:.1f}", list.x0, list.y0, list.x1,
            list.y1, listShift.first, listShift.second);
    for (size_t i = 0; i < n; i++)
    {
        const Box& g = boxes[find(uint32_t(i))];
        const auto origin = !plain ? std::pair<float, float>{ 640.0f, 360.0f } : items[i].fromCentre ? s_plainCentre : s_plainBoresight;
        const auto [dx, dy] = plain && items[i].fromCentre ? listShift : PanelShift(mode, g.x0, g.x1, g.y0, g.y1, origin.first, origin.second);
        if (dx != 0.0f || dy != 0.0f)
            move(items[i], dx, dy);
    }
}

// An element's draw with its output moved (when the option is on and the screen isn't 16:9).
template <void (*Original)(PPCContext&, uint8_t*)>
void DrawHudElement(PPCContext& ctx, uint8_t* base, EdgeMode mode, int sideX, int sideY)
{
    const uint32_t r = s_hudRenderer;
    if (HudEdges() == 0.0 || Wide() == 1.0f || !r)
    {
        Original(ctx, base);
        return;
    }
    const HudCounts before = CountHud(r);
    if (mode == EdgeMode::Corners)
    {
        // The flight instruments (CAce6HudPlain::Draw, r4 = &centre, r5 = &boresight): most of their output is placed
        // from the boresight (it sways with the view and the manoeuvres), the destroyed list from the centre (fixed).
        const uint32_t centre = ctx.r4.u32, boresight = ctx.r5.u32;
        s_plainCentre = { F(centre), F(centre + 4) };
        s_plainBoresight = { F(boresight), F(boresight + 4) };
        s_listState = 0;
        Original(ctx, base);
    }
    else
        Original(ctx, base);
    const HudCounts after = CountHud(r);
    MoveHud(r, before, after, mode, sideX, sideY);
}

} // namespace

// CAce6HudPlain::Draw's destroyed list, its header (0x8223D884) to the end of its loop (0x8223DA94): the lists' counts
// at each end, so the element's move knows its primitives (only while the HUD edges option moves the element).
void DestroyedListBegin()
{
    if (s_hudRenderer && HudEdges() != 0.0 && Wide() != 1.0f)
    {
        s_listFrom = CountHud(s_hudRenderer);
        s_listState = 1;
    }
}

void DestroyedListEnd()
{
    if (s_listState == 1)
    {
        s_listTo = CountHud(s_hudRenderer);
        s_listState = 2;
    }
}

// The HUD draw (CAce6HudImpl slot 15, r3 = the HUD): its renderer interface for the element wrappers below.
static uint32_t s_hudInterface = 0;   // the same at any aspect (the pause map, below)
extern "C" PPC_FUNC(__imp__sub_8223DDB0);
PPC_FUNC(sub_8223DDB0)
{
    if (const uint32_t r = ctx.r3.u32 + 0x28D20; uint32_t(I(r)) == 0x820554E8)
        s_hudInterface = r;
    if (HudEdges() != 0.0 && Wide() != 1.0f)
    {
        const uint32_t r = ctx.r3.u32 + 0x28D20;
        if (uint32_t(I(r)) == 0x820554E8)
            s_hudRenderer = r;
        else
        {
            static bool warned = false;
            if (!warned)
                ac6::log::Warn("HUD edges: no HUD renderer at {:08X} (vtable {:08X}); off", r, uint32_t(I(r)));
            warned = true;
            s_hudRenderer = 0;
        }
    }
    // The picture-in-picture target's label (its type, name and damage bar: the HUD draw's last block, from 0x8223E2BC):
    // moved with the window where it sits on one. Only that block: the HUD draw's other output (0x82307938 /
    // 0x823072C8) is tied to the world, and moving what it drew over a window's spot would drag target boxes along.
    const uint32_t r = s_hudRenderer;
    if (HudEdges() == 0.0 || Wide() == 1.0f || !r)
    {
        __imp__sub_8223DDB0(ctx, base);
        return;
    }
    s_pipLabelSeen = false;
    __imp__sub_8223DDB0(ctx, base);
    if (s_pipLabelSeen)
        MoveHud(r, s_pipLabelFrom, CountHud(r), EdgeMode::InPip, 0, 0);
}

// The HUD draw at its picture-in-picture label's block (0x8223E2BC): what the lists hold so far.
void HudPipLabelBegin()
{
    if (const uint32_t r = s_hudRenderer; r && HudEdges() != 0.0 && Wide() != 1.0f)
    {
        s_pipLabelFrom = CountHud(r);
        s_pipLabelSeen = true;
    }
}

// The AWACS / missile picture-in-picture windows' quads (0x82274B20: the window at r3 +32 centre {x, y, z}, 0x82274C80:
// the one at +10224, r4 = four corners of 12 bytes {x, y, z}): moved out with the HUD, by the side of their centre.
namespace {

void MovePipQuad(uint32_t corners, float centreX, float centreY)
{
    if (HudEdges() == 0.0 || Wide() == 1.0f || !corners)
        return;
    const float x = centreX, y = centreY;
    const auto [dx, dy] = EdgeShift(EdgeMode::Split, 0, 0, x, x, y, y);
    for (uint32_t v = 0; v < 4; v++)
    {
        SetF(corners + v * 12, F(corners + v * 12) + dx);
        SetF(corners + v * 12 + 4, F(corners + v * 12 + 4) + dy);
    }
}

// The pictures themselves: layer-1 sprites (EffectScene::RenderView 0x8209AB0C / 0x8209AD24, labels Re2DWinL /
// Re2DWinR) of the textures 0x8018 / 0x801A, centred (pivot mode 4) on the window's position.
// The overlays drawn on them right after (CX360Fade helpers 0x8211E190 / 0x8211E2C8 / 0x8211E0A0: the noise texture
// 0xA030 and an untextured tint) are layer-1 sprites centred on the same point: moved with them.
std::pair<float, float> PipSpriteShift(uint32_t layer, uint32_t sprite)
{
    if (HudEdges() == 0.0 || Wide() == 1.0f || layer != kLayers + 1 * kLayerSize)
        return { 0.0f, 0.0f };
    const uint32_t texture = uint32_t(I(sprite + 68));
    const float x = F(sprite), y = F(sprite + 4);
    if (texture == 0x8018 || texture == 0x801A)
    {
        s_pipWindows[texture == 0x8018 ? 0 : 1] = { x, y, F(sprite + 24), F(sprite + 28), ac6::gpu::RendererFrame(), true };
        return EdgeShift(EdgeMode::Split, 0, 0, x, x, y, y);
    }
    if (I(sprite + 44) == 4)
        for (const PipWindow& p : s_pipWindows)
            if (PipOpen(p) && std::fabs(x - p.x) < 0.5f && std::fabs(y - p.y) < 0.5f)
                return EdgeShift(EdgeMode::Split, 0, 0, x, x, y, y);
    return { 0.0f, 0.0f };
}

} // namespace

extern "C" PPC_FUNC(__imp__sub_82274B20);
PPC_FUNC(sub_82274B20)
{
    const uint32_t window = ctx.r3.u32, corners = ctx.r4.u32;
    __imp__sub_82274B20(ctx, base);
    MovePipQuad(corners, F(window + 32), F(window + 36));
}

extern "C" PPC_FUNC(__imp__sub_82274C80);
PPC_FUNC(sub_82274C80)
{
    const uint32_t window = ctx.r3.u32, corners = ctx.r4.u32;
    __imp__sub_82274C80(ctx, base);
    MovePipQuad(corners, F(window + 10224), F(window + 10228));
}



#define AC6_HUD_EDGE_ELEMENT(addr, mode, sx, sy)                                                                       \
    extern "C" PPC_FUNC(__imp__sub_##addr);                                                                             \
    PPC_FUNC(sub_##addr) { DrawHudElement<__imp__sub_##addr>(ctx, base, mode, sx, sy); }


// CAce6HudRader also draws the pause menu's map: its image a sprite of texture 0x9205, 265 x 265 texels at 1:1, placed at
// (827.5, 99.5): half a pixel off the grid. At 720p each pixel samples halfway between two texels and the image's bright
// edge texels (its border) come out as a dim one-pixel line, as on the console. Above 720p the same place gives one
// bright pixel (a sample past the texture's edge) and a faint one. There such a sprite (that texture, at half pixels
// both ways) moves half a pixel left and up, onto the pixels it covers at 720p: its texels on the guest's pixels, the
// border one guest pixel thick (two host pixels at 1440p).
namespace {

constexpr uint32_t kPauseMapTexture = 0x9205;

void FitPauseMap(uint32_t r, const HudCounts& before, const HudCounts& after)
{
    const uint32_t sprites = uint32_t(I(r + kHudBlock0 + 64));
    for (uint32_t e = before.sprites[0]; e < after.sprites[0] && e < before.sprites[0] + 4096; e++)
    {
        const uint32_t sp = sprites + e * 112;
        const float x = F(sp + 72), y = F(sp + 76);
        auto half = [](float v) { return std::fabs(v - std::floor(v) - 0.5f) < 0.001f; };
        if (uint32_t(I(sp + 8)) != kPauseMapTexture || !half(x) || !half(y))
            continue;
        SetF(sp + 72, x - 0.5f);
        SetF(sp + 92, F(sp + 92) - 0.5f);
        SetF(sp + 76, y - 0.5f);
        SetF(sp + 96, F(sp + 96) - 0.5f);
    }
}

template <PPCFunc* Original>
void DrawRadar(PPCContext& ctx, uint8_t* base)
{
    const uint32_t r = s_hudInterface;
    const bool fit = r && std::min(ac6::gpu::g_video.frameScaleX, ac6::gpu::g_video.frameScaleY) > 1.001f;
    const HudCounts before = fit ? CountHud(r) : HudCounts{};
    DrawHudElement<Original>(ctx, base, EdgeMode::Clipped, -1, 1);
    if (fit)
        FitPauseMap(r, before, CountHud(r));
}

} // namespace

extern "C" PPC_FUNC(__imp__sub_82248E98);
PPC_FUNC(sub_82248E98) { DrawRadar<__imp__sub_82248E98>(ctx, base); }   // CAce6HudRader
extern "C" PPC_FUNC(__imp__sub_82249110);
PPC_FUNC(sub_82249110) { DrawRadar<__imp__sub_82249110>(ctx, base); }   // CAce6HudRader (its second part)
AC6_HUD_EDGE_ELEMENT(82238948, EdgeMode::Whole, 1, 1)    // CAce6HudWeapon

AC6_HUD_EDGE_ELEMENT(82239280, EdgeMode::Whole, -1, 1)   // CAce6HudOrder
AC6_HUD_EDGE_ELEMENT(8223BF58, EdgeMode::Split, 0, 0)    // CAce6HudMulti (score, clock)
AC6_HUD_EDGE_ELEMENT(82237C90, EdgeMode::Split, 0, 0)    // CAce6HudMulti (its second part)
AC6_HUD_EDGE_ELEMENT(8223A2E0, EdgeMode::Split, 0, 0)    // CAce6HudDisplay (messages)
AC6_HUD_EDGE_ELEMENT(8223C660, EdgeMode::Corners, 0, 0)  // CAce6HudPlain (score, clock, destroyed list, not speed / altitude)

// Hud_ClipToScreen (r3 = out {x, y}): x = (x / w + 1) * 640 on the 1280x720 canvas. With the wide projection the screen
// spans 640 k canvas pixels either side of the centre (narrow: 360 / k above and below).
extern "C" PPC_FUNC(__imp__sub_8224EC20);
PPC_FUNC(sub_8224EC20)
{
    const uint32_t out = ctx.r3.u32;
    __imp__sub_8224EC20(ctx, base);
    if (Wide() != 1.0f)
    {
        SetF(out, 640.0f + (F(out) - 640.0f) * Kx());
        SetF(out + 4, 360.0f + (F(out + 4) - 360.0f) * Ky());
    }
}

// ---- Bars: the 16:9 screens ----------------------------------------------------------------------------------------
namespace {

// The mode task running (the mode manager [0x8293B930] +8, its vtable = the screen): the mission ones are wide.
bool MissionTask()
{
    static bool last = false;
    const uint32_t task = ModeTask();
    if (!task)
        return last;   // between tasks: as before
    switch (task)
    {
    case 0x820642F4:   // CModeTaskGame
    case 0x8206474C:   // CModeTaskGameTutorial
    case 0x82066C24:   // CModeTaskGameTutorial_FromMissionMenu
    case 0x82066C84:   // CModeTaskGameTutorial_CampaignMission
    case 0x820646EC:   // CModeTaskGameReplay
    case 0x82066BC4:   // CModeTaskGameReplayMission
        return last = true;
    default:
        return last = false;
    }
}

// AC6_WIDESCREEN_CUTSCENES=1: in-engine cutscenes hor+ like gameplay, by default at 16:9 (barred), as they were framed.
// Live (the bars only cover the hor+ view): read again after a change.
bool CutscenesWide()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<bool> wide{ false };
    if (const uint64_t now = ac6::settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        wide.store(ac6::settings::Get("AC6_WIDESCREEN_CUTSCENES") == "1", std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    return wide.load(std::memory_order_relaxed);
}

bool Bars()
{
    if (Wide() == 1.0f)
        return false;
    const bool bars = ac6::gpu::InCutscene() ? !CutscenesWide() : !MissionTask();
    static bool logged = false;
    if (bars != logged)
    {
        ac6::log::Info("Widescreen: {} (frame {})", bars ? "16:9 with bars" : "wide", ac6::gpu::RendererFrame());
        logged = bars;
    }
    return bars;
}

} // namespace

// Submit2DLayers (0x821D0D78, at the end of the frame's 2D): with bars, two black sprites over the sides (narrow: above
// and below), added last to layer 4 (full screen), so they cover the 3D, the HUD layers and the swg layer outside the
// centred 16:9 box. The sprite descriptions are laid out as DrawGlobalFade's (0x8211D090): colour RGBA floats at +48.
extern "C" PPC_FUNC(__imp__sub_820941F8);
extern "C" PPC_FUNC(__imp__sub_821D0D78);
PPC_FUNC(sub_821D0D78)
{
    // (Not when this frame's backdrop covered the canvas and was wrapped to the sides.)
    const bool wrapped = std::exchange(s_menuWrapped, false);
    if (WideMenusTrace() && wrapped != s_lastWrapped)
        ac6::log::Info("Wide menus trace: frame {} backdrop wrapped: {}", s_frames2D, wrapped);
    s_frames2D++;
    s_lastWrapped = wrapped;
    // Frosted glass at the sides while the screen has pictures ending at the box: from the first visible one
    // to the last one there at all. The box as WideLayerViewport rounds it. AC6_PILLAR_FROST=0: none.
    {
        static bool frosted = false;
        static const bool allowed = [] { const char* v = std::getenv("AC6_PILLAR_FROST"); return !(v && *v == '0'); }();
        const bool pictures = std::exchange(s_edgePictures, false), visible = std::exchange(s_edgePicturesVisible, false);
        const bool now = allowed && wrapped && Kx() > 1.0f && pictures && (frosted || visible);
        if (now != frosted)
        {
            frosted = now;
            if (WideMenusTrace())
                ac6::log::Info("Wide menus trace: frame {} frosted sides: {}", s_frames2D, now);
            const double h0 = std::lround(640.0 - 640.0 / Kx()), h1 = std::lround(640.0 + 640.0 / Kx());
            ac6::gpu::SetPillarFrost(now ? float(h0 / 1280.0) : 0.0f, now ? float(h1 / 1280.0) : 0.0f);
        }
    }
    // The upright lines of a screen that wraps, continued to the sides. The briefing's vertical grid lines are thin
    // sprites every 100 canvas pixels from 36 to 1336, and at its start a second set every 120 from 25 to 1346,
    // sweeping in alternately from the top and the bottom. The swg layer's clip ends both at the 16:9 box, the last
    // line of each included, which leaves the screen beyond without lines. A set = four or more lines of one texture
    // and one alpha, evenly spaced (each gap within 2.5 pixels of the mean, 16 or more), from one side of the canvas to
    // the other. What continues it: its own lines past the box as they are, and more at its spacing to the screen's
    // edges, each a copy of the set's line two (or one) places in (the sweeping set alternates: a copy keeps its
    // neighbours' pattern). Kept for the next frame (above).
    {
        std::vector<UprightLine> lines = std::move(s_uprightLines);
        s_uprightLines.clear();
        s_uprightSets.clear();
        if (!lines.empty() && wrapped && Kx() > 1.0f && WideMenus())
        {
            const double h0 = std::lround(640.0 - 640.0 / Kx()), h1 = std::lround(640.0 + 640.0 / Kx());
            const float boxLeft = float(640.0 + (h0 - 640.0) * Kx()), boxRight = float(640.0 + (h1 - 640.0) * Kx());
            const float screenLeft = 640.0f - 640.0f * Kx(), screenRight = 640.0f + 640.0f * Kx();
            std::stable_sort(lines.begin(), lines.end(), [](const UprightLine& a, const UprightLine& b) {
                return std::tie(a.texture, a.alphaBits, a.x) < std::tie(b.texture, b.alphaBits, b.x);
            });
            for (size_t first = 0; first < lines.size();)
            {
                size_t last = first;
                while (last + 1 < lines.size() && lines[last + 1].texture == lines[first].texture && lines[last + 1].alphaBits == lines[first].alphaBits)
                    last++;
                const size_t count = last - first + 1;
                const float gap = count >= 2 ? (lines[last].x - lines[first].x) / float(count - 1) : 0.0f;
                // (And across the canvas: its first line within a gap of the left edge, its last within one of the right. A
                // table's column rules inside the box are a regular set too, and must not march out to the sides.)
                bool even = count >= 4 && gap >= 16.0f && lines[first].x <= gap + 8.0f && lines[last].x >= 1280.0f - gap - 8.0f;
                for (size_t i = first; even && i < last; i++)
                    even = std::fabs(lines[i + 1].x - lines[i].x - gap) <= 2.5f;
                if (even)
                {
                    UprightSet& set = s_uprightSets.emplace_back();
                    set.texture = lines[first].texture;
                    set.added = false;
                    auto add = [&](const UprightLine& from, float x) {
                        if (x <= screenLeft - 2.0f || x >= screenRight || (x >= boxLeft - 0.5f && x <= boxRight - 0.5f))
                            return;
                        set.copies.push_back(from);
                        set.copies.back().x = x;
                    };
                    for (size_t i = first; i <= last; i++)
                    {
                        set.at.push_back(lines[i].x);
                        add(lines[i], lines[i].x);   // (the game's own past the box, `add` leaves the ones inside it)
                    }
                    for (int k = 1; k < 64 && lines[first].x - gap * float(k) > screenLeft - 2.0f; k++)
                        add(lines[first + (k % 2)], lines[first].x - gap * float(k));
                    for (int k = 1; k < 64 && lines[last].x + gap * float(k) < screenRight; k++)
                        add(lines[last - (k % 2)], lines[last].x + gap * float(k));
                    if (WideMenusTrace())
                    {
                        static uint32_t logged = 0;
                        if (logged++ < 6)
                            ac6::log::Info("Wide menus trace: frame {} {} upright lines every {} from {} to {} (texture {:08X}): {} more past the box",
                                s_frames2D, count, gap, lines[first].x, lines[last].x, lines[first].texture, set.copies.size());
                    }
                }
                first = last + 1;
            }
        }
    }
    if (Bars() && !wrapped)
    {
        static const uint32_t sprites = ac6::g_memory.MapVirtual(ac6::g_heap.Alloc(2 * 128));
        // Wide: left and right of the canvas, narrow: above and below it.
        const bool wide = Kx() != 1.0f;
        const float edge = wide ? 640.0f * Kx() - 640.0f + 2.0f : 360.0f * Ky() - 360.0f + 2.0f;
        for (uint32_t i = 0; i < 2; i++)
        {
            const uint32_t s = sprites + i * 128;
            for (uint32_t w = 0; w < 6; w++)   // as the fade's: the template's position, 0, 0, 0
                SetI(s + w * 4, w < 3 ? I(0x8205CDB4 + w * 4) : 0);
            SetF(s, wide ? (i == 0 ? -edge : 1280.0f) : -2.0f);
            SetF(s + 4, wide ? -2.0f : (i == 0 ? -edge : 720.0f));
            SetF(s + 24, wide ? edge : 1284.0f);
            SetF(s + 28, wide ? 724.0f : edge);
            SetF(s + 32, 1.0f);
            SetF(s + 36, 1.0f);
            SetI(s + 40, 0);
            SetI(s + 44, 0);
            SetF(s + 48, 0.0f);
            SetF(s + 52, 0.0f);
            SetF(s + 56, 0.0f);
            SetF(s + 60, 1.0f);
            SetI(s + 64, 2081);
            SetI(s + 68, -1);
            SetI(s + 72, -1);
            SetI(s + 76, 0);
            PPCContext call = ctx;
            call.r3.u64 = kLayers + 4 * kLayerSize;
            call.r4.u64 = s;
            __imp__sub_820941F8(call, base);
        }
    }
    __imp__sub_821D0D78(ctx, base);
}

// Mid-asm hook at 0x82099460 (EffectScene::RenderView's snapshot for the HUD, S = r26): S+32 holds the render view's
// projection as it was last set, before this frame's own: the previous frame's last, which with an AWACS picture-in-
// picture on screen is that view's 16:9 one. With it the HUD would place its markers (and the flight HUD) for a 16:9
// projection on the wide view, further out the further from the centre. So the main view's projection is copied there
// instead.
void WideHudProjection(PPCRegister& r26)
{
    if ((Wide() != 1.0f || FovScale() != 1.0) && s_haveMainProjection)
        std::memcpy(ac6::g_memory.Translate(r26.u32 + 32), s_mainProjection, sizeof(s_mainProjection));
}

// Mid-asm hooks after the three render views' loads of the main camera's field of view (degrees) handed to their object
// jobs' set-up (0x820A15F8, f1): the jobs build the matrices the aircraft and other objects are drawn with from it, apart
// from the projection builders (without this the world would widen but the aircraft wouldn't).
void FovJobRead(PPCRegister& f1)
{
    if (ScreenFovScale() != 1.0 && t_pip == 0)
        f1.f64 = ApplyFov(f1.f64 * (kPi / 180.0)) * (180.0 / kPi);
}

// Mid-asm hooks at 0x8224A308 / 0x8224B924, after the loads of the main camera's field of view ([root + 0x2F9A0] +212,
// degrees) in the two HUD functions that size or place an element by it (0x8224A1B8: a size k / fov, 0x8224B7E0: an
// offset k / fov): the option applied, so they agree with the projection.
void FovHudRead(PPCRegister& f13)
{
    if (ScreenFovScale() != 1.0)
        f13.f64 = ApplyFov(f13.f64 * (kPi / 180.0)) * (180.0 / kPi);
}

// Mid-asm hook at 0x8223ECD8 (Hud_DrawOffscreenArrow): f0 = 640 / 360, the canvas's aspect, which the projected direction
// is corrected by. It becomes the wide screen's.
void WideArrowAspect(PPCRegister& f0)
{
    f0.f64 = f0.f64 * Wide();
}

// Mid-asm hook at 0x821D8F64 (the Bink frame quad's setup): r11 = the three clip-space vertices just written (x at +0,
// +16, +32, y at +4): narrowed to 16:9 in the middle of the wide frame (narrow: made shorter, cleared black around it).
void WideMovieQuad(PPCRegister& r11)
{
    if (Wide() != 1.0f)
        for (uint32_t v = 0; v < 3; v++)
        {
            SetF(r11.u32 + v * 16, F(r11.u32 + v * 16) / Kx());
            SetF(r11.u32 + v * 16 + 4, F(r11.u32 + v * 16 + 4) / Ky());
        }
}


namespace ac6::game {

double FovDegrees()
{
    // Read again after a live change (the settings overlay's Apply).
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<double> degrees{ kGameFovDegrees };
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        double d = kGameFovDegrees;
        if (const char* v = std::getenv("AC6_FOV"); v && *v && std::atof(v) >= 10.0 && std::atof(v) <= 150.0)
            d = std::atof(v);
        else if (const std::string offset = settings::Get("AC6_FOV_OFFSET"); !offset.empty())
            d = kGameFovDegrees + std::clamp(std::atof(offset.c_str()), 0.0, 40.0);
        degrees.store(d, std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    return degrees.load(std::memory_order_relaxed);
}

} // namespace ac6::game
