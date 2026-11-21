// The settings overlay, see overlay.h. Seven tabs (Game, Display, Graphics, Controls, Keybinds, Advanced controls,
// Debug), mostly generated from the settings table (src/settings.cpp: each setting's tab, group, label, offered values
// and default). Apply writes ac6_settings.toml in place (its comments kept). The live settings (kLiveKeys) take effect
// on Apply, the rest at the next start (Restart now starts the game again).
// The game keeps running under it (AC6_UI_PAUSE=1 holds it: gpu/renderer.cpp RendererHoldForOverlay). The look: the
// game's menu colours, glossy mid-2000s bars, Share Tech type (OFL, thirdparty/fonts), drawn here with ImGui's
// draw lists. Its navigation is its own (not ImGui's): up / down move the cursor, left / right change the value.
#include "ui/overlay.h"

#include "crash.h"
#include "fs/dlc.h"
#include "game/hud_colour.h"
#include "gpu/texture_mods.h"
#include "gpu/video.h"
#include "input/advanced.h"
#include "input/input.h"
#include "input/kbm.h"
#include "log.h"
#include "mods/mods.h"
#include "platform/platform.h"
#include "settings.h"
#include "update/updater.h"

#include <SDL3/SDL.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_sdl3.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <format>
#include <initializer_list>
#include <filesystem>
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <string_view>

// The fonts, embedded at build time (tools/cmake/embed_files.cmake).
const unsigned char* EmbeddedFileData(const char* name, size_t* size);

namespace ac6::ui {

namespace {

bool s_init = false;
std::atomic<bool> s_open{ false };
SDL_WindowID s_windowId = 0;

// The latest frame for the renderer, and the texture changes it hasn't taken yet.
std::mutex s_mutex;
DrawFrame s_frame;
bool s_frameReady = false;
std::vector<TextureUpdate> s_pendingTextures;
uint32_t s_nextTexture = 1;

ImFont* s_body = nullptr;      // Share Tech: labels and text
ImFont* s_mono = nullptr;      // headings, values, prompts: Share Tech too (not its Mono)
ImFont* s_numbers = nullptr;   // Share Tech Mono: numbers that change as you watch (a range's value, its typing field)

// The settings as the overlay edits them: the file's value when it opened, the value now ("" = the default), and the
// value the game runs with (its variable at the first opening: what the file and the environment gave at start).
struct Edit
{
    settings::Entry entry;
    std::string file, value, running;
    bool environment = false;   // set in the environment: the file's value doesn't apply
    std::string choices, def;   // the table's, or the display's (the monitor's modes, its peak brightness)
};
std::vector<Edit> s_edits;
bool s_runningKnown = false;

// Keybinds: the keyboard and mouse bindings, a tab of their own. Advanced controls: any
// controller's and flight stick's.
constexpr const char* kTabs[] = { "Game", "Display", "Graphics", "Controls", "Keybinds", "Advanced controls", "Debug" };
constexpr int kTabCount = int(std::size(kTabs));

// The Graphics tab's presets: the values they set (anything not listed: its default).
struct Preset
{
    const char* name;
    const char* help;
    std::initializer_list<std::pair<const char*, const char*>> values;
};
// Console, Performance, Enhanced. The presets set only the Graphics tab's settings (kPresetKeys), not the hidden
// Max anisotropy or the aircraft edges fix on the Debug tab.
const Preset kPresets[] = {
    { "Console", "Console default settings", {} },
    // Performance without MSAA, Enhanced with the high quality DoF and the 10-bit output.
    { "Performance", "Reduced resolution with FSR 3.1 Quality upscaling, no MSAA, 8x anisotropic filtering",
        { { "aa", "fsr" }, { "fsr_quality", "quality" }, { "msaa", "1" }, { "anisotropic_override", "8" } } },
    { "Enhanced", "FSR 3.1 Native anti-aliasing, 16x anisotropic filtering, doubled LOD, debanding, enhanced depth of field and bloom, ambient occlusion, planar reflection, 10-bit output",
        // (Bloom, ambient occlusion and the planar reflection on in Enhanced, off in the other two.)
        { { "aa", "fsr" }, { "anisotropic_override", "16" }, { "lod_scale", "2" }, { "deband", "1" }, { "dof_fix", "1" },
          { "bloom", "enhanced" }, { "ambient_occlusion", "1" }, { "planar_reflection", "1" },
          { "sdr_10bit", "1" } } },
};
constexpr const char* kPresetKeys[] = { "render_resolution", "aa", "fsr_quality", "msaa", "anisotropic_override", "lod_scale",
    "deband", "dof_fix", "sdr_10bit",
    // (Ambient occlusion, the enhanced bloom and the planar reflection: on in Enhanced only, the other effects off in
    // every preset.)
    "ambient_occlusion", "film_grain", "vignette", "chromatic_aberration", "motion_blur", "bloom", "planar_reflection" };

// A row of the current tab: a setting (its index in s_edits) or the preset row (-1).
struct Row
{
    int edit;
    const char* group;
};
std::vector<Row> s_rows;
constexpr int kPresetRow = -1;   // the Graphics tab's quality preset
constexpr int kDlcRow = -2;      // the Game tab's DLC counter (read only)
constexpr int kRenderResRow = -4;   // the Graphics tab's render resolution (read only)
constexpr int kQuitRow = -3;     // the Game tab's Quit to desktop (asks first)
constexpr int kTexturesRow = -5;   // the Debug tab's texture browser
constexpr int kDlcFolderRow = -6;   // the Game tab's Open DLC folder (an action)
constexpr int kUpdateRow = -7;      // the Game tab's Updates: the version, a check, the update (actions)
constexpr int kModRow0 = -5000;     // rows -5000 - i: the Game tab's MODS, mod i of s_mods (read only)

// The mods (mods/mods.h: the Lua mods, then the plugins), read with the rows.
std::vector<ac6::mods::ModStatus> s_mods;

bool IsModRow(int edit) { return edit <= kModRow0 && edit > kModRow0 - int(s_mods.size()); }
const ac6::mods::ModStatus& ModAt(int edit) { return s_mods[size_t(kModRow0 - edit)]; }

// A mod row's value: on (with its hooks), off, or failed (red).
std::string ModText(const ac6::mods::ModStatus& m, bool* problem = nullptr)
{
    // (Turned off on purpose: disabled in its mod.toml, left out by AC6_LUA_DISABLE, a plugin declining.)
    const bool failed = !m.loaded && !m.error.empty() && m.error.find("disabled") == std::string::npos &&
        m.error.find("left out") == std::string::npos && m.error.find("declined") == std::string::npos;
    if (problem)
        *problem = failed;
    if (!m.loaded)
        return failed ? "FAILED" : "OFF";
    const std::string kind = m.plugin ? "PLUGIN" : m.native ? "ON · NATIVE" : "ON";
    return m.hooks ? std::format("{} · {} HOOK{}", kind, m.hooks, m.hooks == 1 ? "" : "S") : kind;
}

// Open DLC folder: the system's file manager, off the main thread.
std::atomic<bool> s_dlcFolderFailed{ false };

const std::filesystem::path& DlcFolderPath()
{
    static const std::filesystem::path folder = [] {
        std::error_code ec;
        std::filesystem::path path = std::filesystem::absolute(fs::DlcFolder(), ec);
        return ec ? fs::DlcFolder() : path.make_preferred();
    }();
    return folder;
}

std::string DlcFolderText() { return reinterpret_cast<const char*>(DlcFolderPath().u8string().c_str()); }

void OpenDlcFolder()
{
    std::thread([] {
        std::error_code ec;
        std::filesystem::create_directories(DlcFolderPath(), ec);
        const std::string folder = DlcFolderText();
        const bool opened = SDL_OpenURL(folder.c_str());
        s_dlcFolderFailed = !opened;
        if (opened)
            log::Info("Settings overlay: opened the DLC folder {}", folder);
        else
            log::Warn("Settings overlay: can't open the DLC folder {}: {}", folder, SDL_GetError());
    }).detach();
}

// The texture browser (gpu/texture_mods.h): the last frame's textures, one per content hash (cube maps left out). A
// snapshot: taken when the overlay opens and on Refresh (R / the controller's Y, or a click on REFRESH), not every
// frame.
bool s_browsing = false;   // Enter on the row: the arrows move between the thumbnails
int s_texCursor = 0, s_texColumns = 6, s_texScroll = 0;
bool s_texFollow = true;   // the grid scrolled to keep the chosen one in view (keys), the wheel scrolls it freely
std::vector<gpu::mods::FrameTexture> s_texSnapshot;
bool s_texTaken = false;

void RefreshTextures()
{
    s_texSnapshot.clear();
    for (const gpu::mods::FrameTexture& t : gpu::mods::FrameTextures())
        if (!t.cube && std::none_of(s_texSnapshot.begin(), s_texSnapshot.end(), [&](const auto& o) { return o.hash == t.hash; }))
            s_texSnapshot.push_back(t);
    s_texTaken = true;
    s_texCursor = s_texSnapshot.empty() ? 0 : std::clamp(s_texCursor, 0, int(s_texSnapshot.size()) - 1);
    s_texFollow = true;
}

const std::vector<gpu::mods::FrameTexture>& BrowserTextures()
{
    if (!s_texTaken)
        RefreshTextures();
    return s_texSnapshot;
}
const char* GuestFormatName(uint32_t format)
{
    switch (format)
    {
    case 2: return "8";
    case 6: return "8888";
    case 18: return "DXT1";
    case 19: return "DXT3";
    case 20: return "DXT5";
    case 49: return "DXN";
    default: return "?";
    }
}

// The DLC the game has: "19 OF 20 (1 DAMAGED)" (loaded of found, and of those loaded the damaged ones), OFF (AC6_DLC=0
// at start), SCANNING (not done yet). `problem`: a package not loaded or damaged (the count is drawn red).
std::string DlcCount(bool* problem = nullptr)
{
    if (problem)
        *problem = false;
    if (settings::Get("AC6_DLC") == "0")
        return "OFF";
    if (!fs::DlcScanFinished())
        return "SCANNING";
    const auto& packages = fs::DlcPackages();
    const size_t usable = size_t(std::count_if(packages.begin(), packages.end(), [](const fs::DlcPackage& p) { return p.Usable(); }));
    const size_t damaged = size_t(std::count_if(packages.begin(), packages.end(), [](const fs::DlcPackage& p) { return p.Usable() && !p.damage.empty(); }));
    if (problem)
        *problem = usable != packages.size() || damaged;
    return std::format("{} OF {}", usable, packages.size()) + (damaged ? std::format(" ({} DAMAGED)", damaged) : std::string());
}

int s_tab = 0;
int s_cursor = 0;            // the selected row
int s_footer = -1;           // >= 0: the cursor is on that footer button instead (Revert, Apply, Restart now)
bool s_scrollToCursor = false;
int s_editing = -1;          // the text or number row being typed into
int s_dragRow = -1;          // the range row whose blocks the mouse is dragging
bool s_editStart = false;
bool s_endEditing = false;   // a controller's A or B ends the typing (it has no Enter or Escape)
std::string s_fileError;     // the settings file's parse error at the last load ("" when it parses)
char s_text[512] = {};
bool s_gamepad = false;      // the controller was used last: its prompts
bool s_hintDone = false;     // the start-up hint (DrawStartupHint) shown out, or the overlay opened
bool s_updateHintDone = false;   // the update notice (DrawUpdateHint) shown out, or the overlay opened
std::string s_status;
bool s_statusWarn = false;
// The confirmation up, if any: Quit to desktop's, Restart now's.
enum Confirm { kNoConfirm, kConfirmQuit, kConfirmRestart };
Confirm s_confirm = kNoConfirm;
// The HUD colour's picker: hue, saturation and value kept apart, so the hue survives a grey or black.
bool s_picker = false;
bool s_pickerAuto = false;       // left on the row's default: Auto, or the game's green
float s_pickerH = 0.0f, s_pickerS = 0.0f, s_pickerV = 0.0f;
std::string s_pickerKey;         // hud_colour, hud_friendly or hud_enemy
int s_confirmFocus = 1;      // the confirmation's buttons: 0 Confirm, 1 Cancel

enum Footer { kRevert, kApply, kRestart, kFooterCount };

// --- Drawing ------------------------------------------------------------------------------------------------------

ImU32 Rgb(uint32_t hex, float a = 1.0f)
{
    return IM_COL32((hex >> 16) & 255, (hex >> 8) & 255, hex & 255, int(std::lround(std::clamp(a, 0.0f, 1.0f) * 255.0f)));
}

// The layout is drawn for 1920x1080 and scaled to the window (centred, 16:9, the backdrop covers the whole window).
float u = 1.0f;
ImVec2 o;
// The text at 100% is this much larger than the layout's own size.
constexpr float kTextBase = 1.1f;
ImVec2 P(float x, float y) { return ImVec2(o.x + x * u, o.y + y * u); }
// Text is drawn in `ut`: u times the text size (the settings text size). The boxes around it keep
// their sizes.
float ut = 1.0f;
// Where to draw text of `size` (layout units) so it grows about the point it was laid out around at u: a box's middle
// `middle` units below `top`, the text drawn `offset` units below it at the layout's own size (ut equal to u).
float TextTop(float top, float middle, float offset)
{
    return top + middle * u - (middle - offset) * ut;
}

// A vertical gradient through stops (position 0..1, colour).
void VGradient(ImDrawList* dl, ImVec2 a, ImVec2 b, std::initializer_list<std::pair<float, ImU32>> stops)
{
    const auto* s = stops.begin();
    for (size_t i = 0; i + 1 < stops.size(); i++)
    {
        const float y0 = a.y + (b.y - a.y) * s[i].first, y1 = a.y + (b.y - a.y) * s[i + 1].first;
        if (y1 > y0)
            dl->AddRectFilledMultiColor(ImVec2(a.x, y0), ImVec2(b.x, y1), s[i].second, s[i].second, s[i + 1].second, s[i + 1].second);
    }
}

void HGradient(ImDrawList* dl, ImVec2 a, ImVec2 b, ImU32 left, ImU32 right)
{
    dl->AddRectFilledMultiColor(a, b, left, right, right, left);
}

// The glossy bar of the game's menus: a light upper half over a darker lower half.
void Gloss(ImDrawList* dl, ImVec2 a, ImVec2 b, uint32_t top, uint32_t upper, uint32_t lower, uint32_t bottom, float alpha = 1.0f)
{
    VGradient(dl, a, b, { { 0.0f, Rgb(top, alpha) }, { 0.49f, Rgb(upper, alpha) }, { 0.51f, Rgb(lower, alpha) }, { 1.0f, Rgb(bottom, alpha) } });
}

// Boxes' outlines: square corners, 1.5 units thick.
float Outline()
{
    return std::max(1.0f, 1.5f * u);
}

float TextWidth(ImFont* font, float size, std::string_view text, float tracking = 0.0f)
{
    float w = 0.0f;
    for (const char* p = text.data(); p < text.data() + text.size();)
    {
        unsigned int c = 0;
        const int n = std::max(1, ImTextCharFromUtf8(&c, p, text.data() + text.size()));
        w += font->CalcTextSizeA(size, FLT_MAX, 0.0f, p, p + n).x + tracking * size;
        p += n;
    }
    return text.empty() ? 0.0f : w - tracking * size;
}

float Text(ImDrawList* dl, ImFont* font, float size, ImVec2 pos, ImU32 color, std::string_view text, float tracking = 0.0f,
    ImU32 shadow = 0)
{
    float x = pos.x;
    for (const char* p = text.data(); p < text.data() + text.size();)
    {
        unsigned int c = 0;
        const int n = std::max(1, ImTextCharFromUtf8(&c, p, text.data() + text.size()));
        if (shadow)
            dl->AddText(font, size, ImVec2(x, pos.y + std::max(1.0f, u)), shadow, p, p + n);
        dl->AddText(font, size, ImVec2(x, pos.y), color, p, p + n);
        x += font->CalcTextSizeA(size, FLT_MAX, 0.0f, p, p + n).x + tracking * size;
        p += n;
    }
    return x - pos.x - (text.empty() ? 0.0f : tracking * size);
}

std::string Upper(std::string_view s)
{
    std::string out(s);
    for (size_t i = 0; i < out.size(); i++)
    {
        char& c = out[i];
        if (c >= 'a' && c <= 'z')
            c = char(c - 'a' + 'A');
        // Latin-1's lower case letters (UTF-8 C3 A0..BE but the division sign C3 B7) to their capitals, C3 80..9E: FRANÇAIS,
        // ESPAÑOL.
        else if (uint8_t(c) == 0xC3 && i + 1 < out.size())
        {
            const uint8_t next = uint8_t(out[i + 1]);
            if (next >= 0xA0 && next <= 0xBE && next != 0xB7)
                out[i + 1] = char(next - 0x20);
            i++;
        }
        // Cyrillic (a language pack's name): а..п (D0 B0..BF) to А..П (D0 90..9F), р..я (D1 80..8F) to Р..Я (D0 A0..AF),
        // ѐ..џ (D1 90..9F) to Ѐ..Џ (D0 80..8F).
        else if ((uint8_t(c) == 0xD0 || uint8_t(c) == 0xD1) && i + 1 < out.size())
        {
            const uint8_t next = uint8_t(out[i + 1]);
            if (uint8_t(c) == 0xD0 && next >= 0xB0 && next <= 0xBF)
                out[i + 1] = char(next - 0x20);
            else if (uint8_t(c) == 0xD1 && next >= 0x80 && next <= 0x8F)
                c = char(0xD0), out[i + 1] = char(next + 0x20);
            else if (uint8_t(c) == 0xD1 && next >= 0x90 && next <= 0x9F)
                c = char(0xD0), out[i + 1] = char(next - 0x10);
            i++;
        }
        // Greek: α..ο (CE B1..BF) to Α..Ο (CE 91..9F), π..ω (CF 80..89, but final ς) to Π..Ω (CE A0..A9).
        else if ((uint8_t(c) == 0xCE || uint8_t(c) == 0xCF) && i + 1 < out.size())
        {
            const uint8_t next = uint8_t(out[i + 1]);
            if (uint8_t(c) == 0xCE && next >= 0xB1 && next <= 0xBF)
                out[i + 1] = char(next - 0x20);
            else if (uint8_t(c) == 0xCF && next >= 0x80 && next <= 0x89 && next != 0x82)
                c = char(0xCE), out[i + 1] = char(next + 0x20);
            i++;
        }
        else if (uint8_t(c) >= 0xC0 && i + 1 < out.size())
            i++;   // (another two-byte letter: its second byte isn't a letter of its own)
    }
    return out;
}

float KeyCapWidth(std::string_view label)
{
    return std::max(26.0f * u, TextWidth(s_mono, 12.0f * ut, label) + 12.0f * u);
}

// The middle of Share Tech's capitals in a line of text drawn at `y` (a key cap or face button beside it is centred on
// it).
float CapsMiddle(float y, float size)
{
    return y + 0.41f * size;
}

// A glyph's ink (its texels at least half covered) in its X0..Y1 coordinates: the quad less the atlas's padding around
// the letter. The padding isn't even, so centring on the quad (or on the 0.41 ratio) would put key caps' letters a
// pixel low. False when the atlas's pixels aren't there or the glyph has no ink.
bool GlyphInk(ImFont* font, const ImFontGlyph* g, ImVec2& ink0, ImVec2& ink1)
{
    ImTextureData* tex = font->OwnerAtlas ? font->OwnerAtlas->TexData : nullptr;
    if (!g || !g->Visible || !tex || !tex->Pixels)
        return false;
    const int u0 = int(std::lround(g->U0 * tex->Width)), u1 = int(std::lround(g->U1 * tex->Width));
    const int v0 = int(std::lround(g->V0 * tex->Height)), v1 = int(std::lround(g->V1 * tex->Height));
    if (u1 <= u0 || v1 <= v0 || u0 < 0 || v0 < 0 || u1 > tex->Width || v1 > tex->Height)
        return false;
    int ix0 = u1, ix1 = u0 - 1, iy0 = v1, iy1 = v0 - 1;
    for (int y = v0; y < v1; y++)
        for (int x = u0; x < u1; x++)
        {
            const unsigned char* px = static_cast<const unsigned char*>(tex->GetPixelsAt(x, y));
            if ((tex->BytesPerPixel == 4 ? px[3] : px[0]) < 128)
                continue;
            ix0 = std::min(ix0, x), ix1 = std::max(ix1, x), iy0 = std::min(iy0, y), iy1 = std::max(iy1, y);
        }
    if (ix1 < ix0)
        return false;
    const float sx = (g->X1 - g->X0) / float(u1 - u0), sy = (g->Y1 - g->Y0) / float(v1 - v0);
    ink0 = ImVec2(g->X0 + float(ix0 - u0) * sx, g->Y0 + float(iy0 - v0) * sy);
    ink1 = ImVec2(g->X0 + float(ix1 + 1 - u0) * sx, g->Y0 + float(iy1 + 1 - v0) * sy);
    return true;
}

// Where to draw `label` for its ink to be centred on `centre`, on whole pixels: across,
// from the first letter's ink to the last's (not the advance widths with their side bearings). Up and down, the middle
// of an H's ink at this size (the 0.41 ratio would put key caps' letters a pixel low).
ImVec2 InkCentred(ImFont* font, float size, std::string_view label, ImVec2 centre, ImVec2* inkCentre = nullptr)
{
    float left = 0.0f, right = TextWidth(font, size, label), capsMiddle = 0.41f * size;
    if (ImFontBaked* baked = font->GetFontBaked(size); baked && !label.empty())
    {
        unsigned int first = 0, last = 0;
        ImTextCharFromUtf8(&first, label.data(), label.data() + label.size());
        size_t lastAt = label.size() - 1;
        while (lastAt > 0 && (static_cast<unsigned char>(label[lastAt]) & 0xC0) == 0x80)
            lastAt--;
        ImTextCharFromUtf8(&last, label.data() + lastAt, label.data() + label.size());
        ImVec2 a, b;
        if (GlyphInk(font, baked->FindGlyphNoFallback(ImWchar(first)), a, b))
            left = a.x;
        if (GlyphInk(font, baked->FindGlyphNoFallback(ImWchar(last)), a, b))
            right = TextWidth(font, size, label.substr(0, lastAt)) + b.x;
        if (GlyphInk(font, baked->FindGlyphNoFallback(ImWchar('H')), a, b))
            capsMiddle = (a.y + b.y) / 2;
    }
    const ImVec2 at(std::round(centre.x - (left + right) / 2), std::round(centre.y - capsMiddle));
    if (inkCentre)   // where the ink's middle lands, drawn at whole pixels (up to half a pixel from `centre`)
        *inkCentre = ImVec2(at.x + (left + right) / 2, at.y + capsMiddle);
    return at;
}

// A keyboard key cap ("F4", "LT") centred on `middle` (a label's CapsMiddle): returns its width. Its outline is a frame
// of whole pixels around the fill (a stroked outline over it leaves a pale line inside).
float KeyCap(ImDrawList* dl, float x, float middle, std::string_view label)
{
    const float size = 12.0f * ut, h = 22.0f * u;
    const float w = KeyCapWidth(label);
    const float x0 = std::round(x), y0 = std::round(middle - h / 2), x1 = std::round(x + w), y1 = std::round(middle + h / 2);
    const float t = std::max(1.0f, std::round(Outline()));
    dl->AddRectFilled(ImVec2(x0, y0 + std::max(1.0f, std::round(u))), ImVec2(x1, y1 + std::max(1.0f, std::round(u))), Rgb(0x000000, 0.6f));
    dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), Rgb(0x5A6390));
    dl->AddRectFilledMultiColor(ImVec2(x0 + t, y0 + t), ImVec2(x1 - t, y1 - t), Rgb(0xD8DEF0), Rgb(0xD8DEF0), Rgb(0x9AA3C2), Rgb(0x9AA3C2));
    Text(dl, s_mono, size, InkCentred(s_mono, size, label, ImVec2((x0 + x1) / 2, (y0 + y1) / 2)), Rgb(0x10143F), label);
    return w;
}

// Whether the prompts show PlayStation's buttons (AC6_BUTTON_STYLE).
bool PlayStation() { return input::PromptStyle() != input::ButtonStyle::Xbox; }

// A controller button's name in the prompts' style: the Xbox name as given, or PlayStation's.
std::string ButtonName(std::string_view xbox)
{
    if (!PlayStation())
        return std::string(xbox);
    const input::ButtonStyle style = input::PromptStyle();
    if (xbox == "A") return "Cross";
    if (xbox == "B") return "Circle";
    if (xbox == "X") return "Square";
    if (xbox == "Y") return "Triangle";
    if (xbox == "LB") return "L1";
    if (xbox == "RB") return "R1";
    if (xbox == "LT") return "L2";
    if (xbox == "RT") return "R2";
    if (xbox == "LS") return "L3";
    if (xbox == "RS") return "R3";
    if (xbox == "Start") return style == input::ButtonStyle::PlayStation3 ? "Start" : "Options";
    if (xbox == "Back")
        return style == input::ButtonStyle::PlayStation3 ? "Select" : style == input::ButtonStyle::PlayStation4 ? "Share" : "Create";
    return std::string(xbox);
}

// A text naming controller buttons by their Xbox names ("LB and RB together"), in the prompts' style: each word that is
// one.
std::string ButtonText(std::string_view text)
{
    if (!PlayStation())
        return std::string(text);
    std::string out;
    size_t i = 0;
    while (i < text.size())
    {
        size_t j = i;
        while (j < text.size() && std::isalnum(uint8_t(text[j])))
            j++;
        if (j > i)
        {
            out += ButtonName(text.substr(i, j - i));
            i = j;
        }
        else
            out += text[i++];
    }
    return out;
}

// PlayStation's face buttons: a dark button, its symbol in its colour, drawn (the fonts have no
// such glyphs): A the cross, B the circle, X the square, Y the triangle.
float PlayStationButton(ImDrawList* dl, ImVec2 pos, char letter)
{
    const float r = std::max(4.0f, std::round(13.0f * u));
    const ImVec2 c(std::round(pos.x + 13.0f * u), std::round(pos.y + 13.0f * u));
    dl->AddCircleFilled(c, r, Rgb(0x22263A), 48);
    dl->AddCircle(c, r - u, Rgb(0xFFFFFF, 0.35f), 48, 2.0f * u);
    const float s = 0.42f * r, t = std::max(1.5f, 2.0f * u);
    switch (letter)
    {
    case 'A':   // cross, blue
        dl->AddLine(ImVec2(c.x - s, c.y - s), ImVec2(c.x + s, c.y + s), Rgb(0x8DB4F0), t);
        dl->AddLine(ImVec2(c.x - s, c.y + s), ImVec2(c.x + s, c.y - s), Rgb(0x8DB4F0), t);
        break;
    case 'B':   // circle, red
        dl->AddCircle(c, s * 1.05f, Rgb(0xF07070), 32, t);
        break;
    case 'X':   // square, pink
        dl->AddRect(ImVec2(c.x - s * 0.9f, c.y - s * 0.9f), ImVec2(c.x + s * 0.9f, c.y + s * 0.9f), Rgb(0xE59AD8), 0.0f, 0, t);
        break;
    default:    // triangle, green
        dl->AddTriangle(ImVec2(c.x, c.y - s * 1.05f), ImVec2(c.x + s * 1.1f, c.y + s * 0.8f), ImVec2(c.x - s * 1.1f, c.y + s * 0.8f),
            Rgb(0x4CC9AE), t);
        break;
    }
    return 26.0f * u;
}

float FaceButton(ImDrawList* dl, ImVec2 pos, char letter)
{
    if (PlayStation())
        return PlayStationButton(dl, pos, letter);
    // The letter is placed on whole pixels first, then the circle is centred on the letter's ink, rounded to the
    // nearest half pixel (a pixel's corner or middle), which keeps it round: a centre between pixel rows draws it 24
    // pixels wide and 26 tall at 1080p. Centring the letter on the circle instead can't centre a letter an odd number
    // of pixels wide or tall, and the rounding moves it up to a pixel right and down. The radius is a whole number of
    // pixels.
    const float r = std::max(4.0f, std::round(13.0f * u));
    const char text[2] = { letter, 0 };
    const float size = 14.0f * ut;
    ImVec2 ink;
    const ImVec2 textAt = InkCentred(s_mono, size, text, ImVec2(std::round(pos.x + 13.0f * u), std::round(pos.y + 13.0f * u)), &ink);
    const ImVec2 c(std::round(ink.x * 2.0f) / 2.0f, std::round(ink.y * 2.0f) / 2.0f);
    // Flat: one colour, the outline, the letter centred on its ink, no shadow under the
    // button, no gloss on it, no shadow on the letter.
    uint32_t fill = 0x3E9E34;
    if (letter == 'B')
        fill = 0xC23A2E;
    else if (letter == 'X')
        fill = 0x2F68C4;
    else if (letter == 'Y')
        fill = 0xCF9A1C;
    dl->AddCircleFilled(c, r, Rgb(fill), 48);
    // The outline 2 units, inside the edge, not half outside.
    dl->AddCircle(c, r - u, Rgb(0xFFFFFF, 0.35f), 48, 2.0f * u);
    Text(dl, s_mono, size, textAt, Rgb(0xFFFFFF), text);
    return 26.0f * u;   // as Prompt measures it
}

// A prompt: the controller's button or the key, then what it does. Returns its width (only that with no draw list).
float Prompt(ImDrawList* dl, ImVec2 pos, char button, std::string_view key, std::string_view action)
{
    if (!dl)
        return (s_gamepad && button ? 26.0f * u : KeyCapWidth(key)) + 8.0f * u + TextWidth(s_mono, 15.0f * ut, action, 0.06f);
    float x = pos.x;
    const float middle = CapsMiddle(pos.y + 4.0f * u, 15.0f * u), textY = middle - 0.41f * 15.0f * ut;
    if (s_gamepad && button)
        x += FaceButton(dl, ImVec2(pos.x, middle - 13.0f * u), button);
    else
        x += KeyCap(dl, pos.x, middle, key);
    x += 8.0f * u;
    x += Text(dl, s_mono, 15.0f * ut, ImVec2(x, textY), Rgb(0xC9D0EA), action, 0.06f);
    return x - pos.x;
}

// The menus' panel: a dark navy body, a glossy title bar, a periwinkle frame.
void Panel(ImDrawList* dl, ImVec2 a, ImVec2 b, std::string_view title, std::string_view right = {})
{
    dl->AddRectFilled(ImVec2(a.x + 2 * u, a.y + 6 * u), ImVec2(b.x + 2 * u, b.y + 6 * u), Rgb(0x000000, 0.35f));
    VGradient(dl, a, b, { { 0.0f, Rgb(0x2C388C, 0.60f) }, { 0.06f, Rgb(0x101440, 0.82f) }, { 1.0f, Rgb(0x090B2A, 0.86f) } });
    const float th = 34.0f * u;
    Gloss(dl, a, ImVec2(b.x, a.y + th), 0x6577D8, 0x3242A6, 0x26338E, 0x1C2674);
    dl->AddLine(ImVec2(a.x, a.y + th), ImVec2(b.x, a.y + th), Rgb(0x0B0E30), u);
    dl->AddLine(ImVec2(a.x + u, a.y + u), ImVec2(b.x - u, a.y + u), Rgb(0xC8D7FF, 0.35f), u);
    Text(dl, s_mono, 15.0f * ut, ImVec2(a.x + 18 * u, TextTop(a.y, 17, 8)), Rgb(0xEEF2FF), Upper(title), 0.16f, Rgb(0x000000));
    if (!right.empty())
    {
        const float size = 12.0f * ut;
        Text(dl, s_mono, size, ImVec2(b.x - 18 * u - TextWidth(s_mono, size, right, 0.1f), TextTop(a.y, 17, 11)), Rgb(0xC9D3FF), right, 0.1f);
    }
    dl->AddRect(a, b, Rgb(0x4A5AB0), 0.0f, 0, Outline());
}

void Triangle(ImDrawList* dl, ImVec2 c, float w, float h, bool right, ImU32 color)
{
    if (right)
        dl->AddTriangleFilled(ImVec2(c.x - w / 2, c.y - h / 2), ImVec2(c.x + w / 2, c.y), ImVec2(c.x - w / 2, c.y + h / 2), color);
    else
        dl->AddTriangleFilled(ImVec2(c.x + w / 2, c.y - h / 2), ImVec2(c.x + w / 2, c.y + h / 2), ImVec2(c.x - w / 2, c.y), color);
}

float BadgeWidth(std::string_view text)
{
    return TextWidth(s_mono, 11.0f * ut, text, 0.1f) + 12.0f * u;
}

float Badge(ImDrawList* dl, ImVec2 pos, std::string_view text, bool live)
{
    const float size = 11.0f * ut;
    const float w = BadgeWidth(text), h = 17.0f * u;
    dl->AddRectFilled(pos, ImVec2(pos.x + w, pos.y + h), live ? Rgb(0x3C78B4, 0.25f) : Rgb(0x8C6E1E, 0.25f));
    dl->AddRect(pos, ImVec2(pos.x + w, pos.y + h), live ? Rgb(0x4D86B8) : Rgb(0x8A7430), 0.0f, 0, Outline());
    Text(dl, s_mono, size, ImVec2(pos.x + 6.0f * u, TextTop(pos.y, 8.5f, 2.5f)), live ? Rgb(0xA9E6FF) : Rgb(0xF2D27A), text, 0.1f);
    return w;
}

// The text size at which `text` fits in `width` (smaller than `size` only when it wouldn't: a long value or label at a large
// text size).
float Fit(ImFont* font, float size, std::string_view text, float width, float tracking = 0.0f)
{
    const float w = TextWidth(font, size, text, tracking);
    return w > width && width > 0.0f ? size * width / w : size;
}

// Text wrapped to `width`, with a new line at each '\n'. A line that starts with "Default:" (the help texts' last) is
// drawn in the notes' yellow. Returns the height used (with no draw list it only measures).
float Paragraph(ImDrawList* dl, ImVec2 pos, float width, float size, ImU32 color, std::string_view text)
{
    float y = pos.y;
    if (text.empty())
        return 0.0f;
    for (size_t at = 0;;)
    {
        const size_t newline = text.find('\n', at);
        const std::string_view line = text.substr(at, newline == std::string_view::npos ? std::string_view::npos : newline - at);
        const ImU32 lineColor = line.starts_with("Default:") ? Rgb(0xF2D27A) : color;
        const char* begin = line.data();
        const char* const end = begin + line.size();
        if (begin == end)
            y += size * 1.5f;   // an empty line
        while (begin < end)
        {
            const char* stop = s_body->CalcWordWrapPosition(size, begin, end, width);
            if (stop == begin)
                stop = begin + 1;
            if (dl)   // none: only measured
                dl->AddText(s_body, size, ImVec2(pos.x, y), lineColor, begin, stop);
            y += size * 1.5f;
            begin = stop;
            while (begin < end && *begin == ' ')
                begin++;
        }
        if (newline == std::string_view::npos)
            break;
        at = newline + 1;
    }
    return y - pos.y;
}

// --- The settings -------------------------------------------------------------------------------------------------

// "value=Label|value=Label" -> the values and labels.
std::vector<std::pair<std::string, std::string>> Choices(const char* list)
{
    std::vector<std::pair<std::string, std::string>> out;
    for (std::string_view rest = list ? list : ""; !rest.empty();)
    {
        const size_t bar = rest.find('|');
        const std::string_view item = rest.substr(0, bar);
        const size_t eq = item.find('=');
        if (eq != std::string_view::npos)
            out.push_back({ std::string(item.substr(0, eq)), std::string(item.substr(eq + 1)) });
        rest = bar == std::string_view::npos ? std::string_view() : rest.substr(bar + 1);
    }
    return out;
}

// The value in effect: the edited one, or the default.
std::string Effective(const Edit& e)
{
    return e.value.empty() ? e.def : e.value;
}

std::string RunningEffective(const Edit& e)
{
    return e.running.empty() ? e.def : e.running;
}

bool IsOn(const std::string& v)
{
    return v == "1" || v == "true" || v == "on" || v == "yes";
}

// Same value, as numbers when both are ("1" and "1.0").
bool SameValue(const std::string& a, const std::string& b)
{
    if (a == b)
        return true;
    char* ea = nullptr;
    char* eb = nullptr;
    const double da = std::strtod(a.c_str(), &ea), db = std::strtod(b.c_str(), &eb);
    return !a.empty() && !b.empty() && *ea == 0 && *eb == 0 && da == db;
}

// Sets a value. The default is stored as "" (its line in the file is commented out again).
void SetValue(Edit& e, std::string value)
{
    if (SameValue(value, e.def))
        value.clear();
    e.value = std::move(value);
}

bool Unsaved(const Edit& e)
{
    return e.value != e.file;
}

// The settings that take effect on Apply without a restart. The rest apply at the next start (the display's resolution
// among them: the renderer's sizes are made at start).
constexpr const char* kLiveKeys[] = { "fullscreen", "refresh", "vsync", "frame_rate", "cutscene_unlock", "menu_unlock", "mode",
    "hdr_tonemap", "hdr_paper_white", "hdr_ui_brightness", "hdr_peak", "hdr_exposure", "hdr_highlights", "hdr_shadows", "hdr_contrast", "hdr_saturation",
    "hdr_gamut_compression", "hdr_compression", "hdr_scene", "hdr_hue_correction", "hdr_clouds", "hdr_cloud_light", "hdr_cloud_edge", "hdr_aircraft_light", "hdr_afterburner", "hdr_effects", "hdr_lit_highlights", "hdr_stripes", "sdr_10bit", "audio", "hud_edges", "hud_custom", "hud_colour", "hud_friendly", "hud_enemy", "fov_offset", "cutscene_widescreen",
    "overlay_text_size",
    // The Graphics tab's that take effect while running. Anti-aliasing between Off, SMAA
    // and TAA only (NeedsRestart).
    "deband", "anisotropic_override", "lod_scale", "centroid", "dof_fix", "wide_blurs", "aa", "ambient_occlusion", "shadow_detail", "film_grain", "vignette",
    "chromatic_aberration", "motion_blur", "bloom", "captures",
    // The mouse, then the controller's settings.
    "steer_model", "sensitivity_x", "sensitivity_y", "invert_x", "invert_y",
    "touchpad_back", "adaptive_triggers", "custom_flight", "analog_yaw",
    // Advanced controls' axes.
    "pitch_deadzone", "pitch_full", "pitch_curve", "pitch_invert", "roll_deadzone", "roll_full", "roll_curve", "roll_invert",
    "yaw_deadzone", "yaw_full", "yaw_curve", "yaw_invert", "throttle_deadzone", "throttle_full", "throttle_curve", "lever_detent",
    "lever_detent_width", "look_deadzone", "look_full", "look_curve", "look_invert",
    // Rotation and frame smoothing and the attitude trace, the button style, the texture dump and replacements.
    "attitude_smoothing", "attitude_trace", "frame_smoothing", "button_style", "dump_textures", "texture_replacements",
    // The FPS counter.
    "fps_counter" };

bool IsLive(const Edit& e)
{
    for (const char* key : kLiveKeys)
        if (std::string_view(key) == e.entry.key)
            return true;
    return false;
}

bool NeedsRestart(const Edit& e)
{
    if (e.environment || SameValue(Effective(e), RunningEffective(e)))
        return false;
    // Anti-aliasing: live between Off, SMAA and TAA. FSR on or off at the next start (its context, the world surfaces'
    // sizes), and TAA turned on in a game started without it gets its motion vectors then.
    if (std::string_view(e.entry.key) == "aa")
    {
        const std::string now = Effective(e), start = RunningEffective(e);
        auto has = [](const std::string& v, const char* what) { return v.find(what) != std::string::npos; };
        return has(now, "fsr") != has(start, "fsr") || (has(now, "taa") && !has(start, "taa"));
    }
    return !IsLive(e);
}

// What the live settings were last given (their variables' text, "" = the default), by key.
std::vector<std::pair<std::string, std::string>> s_pushed;

// Each live setting's applied value (the file's, after Apply) to the game (settings::SetLive) when it changed: changes
// take effect when they are applied, not while they are being picked. Not those set in the
// environment (the environment's value applies).
void PushLive()
{
    // A file that doesn't parse reads as all defaults: pushing them would reset the running game's live settings.
    if (!s_fileError.empty())
        return;
    for (const Edit& e : s_edits)
    {
        if (!IsLive(e) || e.environment)
            continue;
        const std::string text = e.file.empty() ? std::string() : settings::VariableText(e.entry, e.file);
        auto it = std::find_if(s_pushed.begin(), s_pushed.end(), [&](const auto& p) { return p.first == e.entry.key; });
        if (it == s_pushed.end())
            it = s_pushed.insert(s_pushed.end(), { e.entry.key, settings::Get(e.entry.env) });
        if (it->second == text)
            continue;
        it->second = text;
        settings::SetLive(e.entry.env, text);
        log::Info("Settings overlay: {} = {} (live)", e.entry.key, text.empty() ? "(default)" : text);
    }
}

bool IsHudColour(const Edit& e)
{
    const std::string_view key = e.entry.key;
    return key == "hud_colour" || key == "hud_friendly" || key == "hud_enemy";
}

uint32_t PickerColour()
{
    float r, g, b;
    ImGui::ColorConvertHSVtoRGB(s_pickerH, s_pickerS, s_pickerV, r, g, b);
    auto byte = [](float x) { return uint32_t(std::lround(std::clamp(x, 0.0f, 1.0f) * 255.0f)); };
    return byte(r) << 16 | byte(g) << 8 | byte(b);
}

void SetPicker(uint32_t rgb)
{
    ImGui::ColorConvertRGBtoHSV(float((rgb >> 16) & 255) / 255.0f, float((rgb >> 8) & 255) / 255.0f, float(rgb & 255) / 255.0f, s_pickerH,
        s_pickerS, s_pickerV);
}

Edit* Find(std::string_view key);

uint32_t ColourOf(std::string_view key)
{
    const Edit* e = Find(key);
    const std::string v = e ? Effective(*e) : std::string();
    char* end = nullptr;
    const unsigned long rgb = std::strtoul(v.c_str(), &end, 16);
    return v.size() == 6 && !*end ? uint32_t(rgb) : game::kHudAuto;
}

// The colourblind preset's colours, when Custom HUD colours is set to one (false otherwise).
bool RowPreset(game::HudPalette& palette)
{
    const Edit* custom = Find("hud_custom");
    return custom && game::HudPreset(Effective(*custom), palette);
}

// The three colours as the rows stand, with the picker's in place of its row's while it's open.
game::HudPalette RowPalette()
{
    if (game::HudPalette preset; RowPreset(preset))
        return preset;
    uint32_t hud = ColourOf("hud_colour"), friendly = ColourOf("hud_friendly"), enemy = ColourOf("hud_enemy");
    if (s_picker)
    {
        const uint32_t picked = s_pickerAuto ? game::kHudAuto : PickerColour();
        (s_pickerKey == "hud_colour" ? hud : s_pickerKey == "hud_friendly" ? friendly : enemy) = picked;
    }
    return game::ResolveHudPalette(hud, friendly, enemy);
}

uint32_t PaletteColour(const game::HudPalette& p, std::string_view key)
{
    return key == "hud_colour" ? p.hud : key == "hud_friendly" ? p.friendly : p.enemy;
}

Edit* Find(std::string_view key)
{
    for (Edit& e : s_edits)
        if (key == e.entry.key)
            return &e;
    return nullptr;
}

// The preset the Graphics settings match, or -1 (custom).
int CurrentPreset()
{
    for (int p = 0; p < int(std::size(kPresets)); p++)
    {
        bool match = true;
        for (const char* key : kPresetKeys)
            if (const Edit* e = Find(key))
            {
                std::string want = e->def;
                for (const auto& [k, v] : kPresets[p].values)
                    if (std::string_view(k) == key)
                        want = v;
                match = match && SameValue(Effective(*e), want);
            }
        if (match)
            return p;
    }
    return -1;
}

void ApplyPreset(int p)
{
    for (const char* key : kPresetKeys)
        if (Edit* e = Find(key))
        {
            std::string want;
            for (const auto& [k, v] : kPresets[p].values)
                if (std::string_view(k) == key)
                    want = v;
            SetValue(*e, want);
        }
}

std::string FormatNumber(double x, double step);

// --- The display ----------------------------------------------------------------------------------------------------

// The primary monitor's modes (the one gpu/video.cpp picks from), for the resolution and refresh rate choices.
struct DisplayMode
{
    int w, h;
    float hz;
};
std::vector<DisplayMode> s_modes;
int s_desktopW = 0, s_desktopH = 0;
std::string s_modesMonitor = "?";   // the Monitor value s_modes were read for

// The display a Monitor value names ("" the primary, "n" the n-th SDL lists, gpu/video.cpp ChosenDisplay).
SDL_DisplayID MonitorDisplay(const std::string& value)
{
    SDL_DisplayID chosen = SDL_GetPrimaryDisplay();
    int count = 0;
    SDL_DisplayID* ids = SDL_GetDisplays(&count);
    if (const int n = std::atoi(value.c_str()); ids && n >= 1 && n <= count)
        chosen = ids[n - 1];
    SDL_free(ids);
    return chosen;
}

// "=Primary|1=1: NAME (3440x1440)|..." for the connected monitors.
std::string MonitorChoices()
{
    std::string out = "=Primary";
    int count = 0;
    SDL_DisplayID* ids = SDL_GetDisplays(&count);
    for (int i = 0; ids && i < count; i++)
    {
        std::string name = SDL_GetDisplayName(ids[i]) ? SDL_GetDisplayName(ids[i]) : "Monitor";
        if (name.size() > 12)   // (the value column holds about 28 characters between its arrows)
            name = name.substr(0, 11) + ".";
        for (char& c : name)
            if (c == '|' || c == '=')
                c = ' ';
        const SDL_DisplayMode* desktop = SDL_GetDesktopDisplayMode(ids[i]);
        out += desktop ? std::format("|{}={}: {} ({}x{})", i + 1, i + 1, name, desktop->w, desktop->h) : std::format("|{}={}: {}", i + 1, i + 1, name);
    }
    SDL_free(ids);
    return out;
}

void ReadModes(SDL_DisplayID display)
{
    s_modes.clear();
    if (const SDL_DisplayMode* desktop = display ? SDL_GetDesktopDisplayMode(display) : nullptr)
        s_desktopW = desktop->w, s_desktopH = desktop->h;
    int count = 0;
    if (SDL_DisplayMode** modes = display ? SDL_GetFullscreenDisplayModes(display, &count) : nullptr)
    {
        for (int i = 0; i < count; i++)
            s_modes.push_back({ modes[i]->w, modes[i]->h, modes[i]->refresh_rate });
        SDL_free(modes);
    }
}

// Smallest to largest (right: larger), then Native: "1280x720=1280 x 720|...|native=Native" (SDL lists the largest first).
std::string ResolutionChoices()
{
    std::string out;
    std::vector<std::pair<int, int>> seen;
    for (auto m = s_modes.rbegin(); m != s_modes.rend(); ++m)
        if (std::find(seen.begin(), seen.end(), std::pair{ m->w, m->h }) == seen.end())
        {
            seen.push_back({ m->w, m->h });
            out += std::format("{}x{}={} x {}|", m->w, m->h, m->w, m->h);
        }
    return out + "native=Native";
}

// The rates of the resolution chosen (native: the desktop's size), lowest to highest, then Highest:
// "60=60 Hz|...|144=144 Hz|0=Highest".
std::string RefreshChoices(const std::string& resolution)
{
    int w = s_desktopW, h = s_desktopH;
    if (resolution != "native")
        std::sscanf(resolution.c_str(), "%dx%d", &w, &h);
    std::vector<float> rates;
    for (const DisplayMode& m : s_modes)
        if (m.w == w && m.h == h && m.hz > 0.0f)
            rates.push_back(float(std::round(m.hz * 100.0) / 100.0));
    std::sort(rates.begin(), rates.end());
    rates.erase(std::unique(rates.begin(), rates.end()), rates.end());
    std::string out;
    for (float r : rates)
    {
        const std::string hz = FormatNumber(r, 0.01);
        out += std::format("{}={} Hz|", hz, hz);
    }
    return out + "0=Highest";
}

SDL_Window* Window()
{
    return s_windowId ? SDL_GetWindowFromID(s_windowId) : nullptr;
}

// The system's HDR for the window's display (Windows: HDR on in the display settings).
bool HdrAvailable()
{
    SDL_Window* w = Window();
    return w && SDL_GetBooleanProperty(SDL_GetWindowProperties(w), SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN, false);
}

// Its peak brightness in nits as the system reports it (0: unknown, as gpu/video.cpp reads it for AC6_HDR_PEAK's default).
float DisplayPeakNits()
{
    SDL_Window* w = Window();
    if (!w || !HdrAvailable())
        return 0.0f;
    const SDL_PropertiesID props = SDL_GetWindowProperties(w);
    const float white = SDL_GetFloatProperty(props, SDL_PROP_WINDOW_SDR_WHITE_LEVEL_FLOAT, 1.0f);
    const float headroom = SDL_GetFloatProperty(props, SDL_PROP_WINDOW_HDR_HEADROOM_FLOAT, 1.0f);
    return headroom > 1.0f ? headroom * white * 80.0f : 0.0f;
}

// --- The keyboard and mouse bindings (the Keybinds tab) ------------------------------------
// Each action of the settings file's [flight_keys] (input/kbm.cpp) is a row: its keys as key caps. Enter / A
// or a click waits for a key, mouse button or wheel notch: one the action hasn't is added, one it has is removed (Escape
// clears all its keys, the controller's B or 5 s without one cancels). Del / X: the action's defaults. Apply writes
// them with the settings (the game reads them again within a second). Greyed while Keyboard and mouse is off.
struct BindingInfo
{
    const char* set;     // "menu" / "flight"
    const char* name;    // the file's key
    const char* label;
    const char* group;
    const char* pad;     // what it is on the controller
};
// The flight keys only: the menus' (up, down, confirm, ...) aren't listed (the settings
// file's [menu_keys] still sets them).
const BindingInfo kBindingInfo[] = {
    { "flight", "pitch_up", "Pitch up", "FLIGHT", "the left stick down (nose up)" },
    { "flight", "pitch_down", "Pitch down", "FLIGHT", "the left stick up (nose down)" },
    { "flight", "roll_left", "Roll left", "FLIGHT", "the left stick left" },
    { "flight", "roll_right", "Roll right", "FLIGHT", "the left stick right" },
    { "flight", "yaw_left", "Yaw left", "FLIGHT", "LB" },
    { "flight", "yaw_right", "Yaw right", "FLIGHT", "RB" },
    { "flight", "accelerate", "Accelerate", "FLIGHT", "RT" },
    { "flight", "decelerate", "Decelerate", "FLIGHT", "LT" },
    { "flight", "high_g", "High-G turn", "FLIGHT", "LT and RT together" },
    { "flight", "autopilot", "Autopilot", "FLIGHT", "LB and RB together" },
    { "flight", "fire_machine_gun", "Fire machine gun", "WEAPON", "A" },
    { "flight", "fire_missile", "Fire missile", "WEAPON", "B" },
    { "flight", "change_weapon", "Change weapon", "WEAPON", "Back" },
    { "flight", "switch_targets", "Switch target", "WEAPON", "Y" },
    { "flight", "camera_control", "Free look (hold)", "CAMERA", "no button: while held, the mouse moves the camera" },
    { "flight", "camera_up", "Look up", "CAMERA", "the right stick up" },
    { "flight", "camera_down", "Look down", "CAMERA", "the right stick down" },
    { "flight", "camera_left", "Look left", "CAMERA", "the right stick left" },
    { "flight", "camera_right", "Look right", "CAMERA", "the right stick right" },
    { "flight", "change_view", "Change view", "CAMERA", "the right stick's click" },
    { "flight", "switch_radar_map", "Radar / map", "OTHER", "X" },
    { "flight", "pause", "Pause", "OTHER", "Start" },
    { "flight", "wingman_up", "Wingman command (up)", "OTHER", "the d-pad up" },
    { "flight", "wingman_down", "Wingman command (down)", "OTHER", "the d-pad down" },
    { "flight", "wingman_left", "Filter map operations (left)", "OTHER", "the d-pad left" },
    { "flight", "wingman_right", "Filter map operations (right)", "OTHER", "the d-pad right" },
    { "flight", "gear", "Landing gear", "OTHER", "the left stick's click" },
};
constexpr size_t kMaxBindingKeys = 4;

struct Binding
{
    input::kbm::BindingAction action;
    const BindingInfo* info;
    std::vector<std::string> file, value;   // the file's keys, and as edited
};
std::vector<Binding> s_bindings;
constexpr int kBindingRow0 = -1000;   // rows -1000 - i: binding i
int s_capture = -1;                   // the binding waiting for a key
uint64_t s_captureUntil = 0;          // SDL_GetTicks() when it gives up

bool IsBindingRow(int edit)
{
    return edit <= kBindingRow0 && edit > kBindingRow0 - int(s_bindings.size());   // (not Advanced controls' rows)
}

Binding& BindingAt(int edit)
{
    return s_bindings[size_t(kBindingRow0 - edit)];
}

void LoadBindings()
{
    const auto actions = input::kbm::BindingActions();
    const auto keys = input::kbm::ReadBindings();
    s_bindings.clear();
    s_capture = -1;
    for (const BindingInfo& info : kBindingInfo)
        for (const input::kbm::BindingAction& a : actions)
            if (a.section == info.set && a.name == info.name)
            {
                Binding b{ a, &info, a.defaults, a.defaults };
                if (const auto it = keys.find(a.section + "." + a.name); it != keys.end())
                    b.file = b.value = it->second;
                s_bindings.push_back(std::move(b));
            }
}

bool BindingsUnsaved()
{
    return std::any_of(s_bindings.begin(), s_bindings.end(), [](const Binding& b) { return b.value != b.file; });
}

// A key's name as shown: "LeftControl" -> "LEFT CONTROL", "Mouse1" -> "MOUSE 1", "F12" stays.
std::string KeyLabel(std::string_view name)
{
    const bool functionKey = name.size() >= 2 && name[0] == 'F' && std::isdigit(uint8_t(name[1]));
    std::string out;
    for (size_t i = 0; i < name.size(); i++)
    {
        const char c = name[i], prev = i ? name[i - 1] : 0;
        if (i && !functionKey && ((std::isupper(uint8_t(c)) && std::islower(uint8_t(prev))) ||
                                  (std::isdigit(uint8_t(c)) && std::isalpha(uint8_t(prev)))))
            out += ' ';
        out += c;
    }
    return Upper(out);
}

std::string BindingText(const std::vector<std::string>& keys)
{
    std::string out;
    for (const std::string& k : keys)
        out += (out.empty() ? "" : " · ") + KeyLabel(k);
    return out.empty() ? std::string("UNBOUND") : out;
}

// Why the bindings can't be changed now (keyboard and mouse off), or null.
const char* BindingsDisabled();

// Other actions of the same set using its keys: "FIRE MISSILE (SPACE)".
std::string BindingConflicts(const Binding& b)
{
    std::string out;
    for (const Binding& other : s_bindings)
        if (&other != &b && other.action.section == b.action.section)
            for (const std::string& k : b.value)
                if (std::find(other.value.begin(), other.value.end(), k) != other.value.end())
                    out += std::format("{}{} ({})", out.empty() ? "" : ", ", Upper(other.info->label), KeyLabel(k));
    return out;
}

// --- Advanced controls' bindings (the Advanced controls tab) ---------------------------------
// Each action of input/advanced.cpp is a row: its inputs. Enter / A or a click waits for an input (a button, a hat, or an
// axis moved half its travel): one the action hasn't is added, one it has removed. Esc clears all, Del / X too, 6 s without
// one cancels (the controller's B is an input here). Apply writes them into [advanced_bindings].
struct AdvBinding
{
    const input::advanced::Action* action;
    std::vector<std::string> file, value;
};
std::vector<AdvBinding> s_adv;
constexpr int kAdvRow0 = -3000;   // rows -3000 - i: Advanced controls' action i
constexpr size_t kMaxAdvInputs = 4;
int s_advCapture = -1;            // the action waiting for an input
uint64_t s_advCaptureUntil = 0;

bool IsAdvRow(int edit)
{
    return edit <= kAdvRow0 && edit > kAdvRow0 - int(s_adv.size());
}

AdvBinding& AdvAt(int edit)
{
    return s_adv[size_t(kAdvRow0 - edit)];
}

void LoadAdv()
{
    const auto bindings = input::advanced::ReadBindings();
    s_adv.clear();
    s_advCapture = -1;
    input::advanced::EndCapture();
    for (const input::advanced::Action& a : input::advanced::Actions())
    {
        AdvBinding b{ &a, {}, {} };
        if (const auto it = bindings.find(a.name); it != bindings.end())
            b.file = b.value = it->second;
        s_adv.push_back(std::move(b));
    }
}

bool AdvUnsaved()
{
    return std::any_of(s_adv.begin(), s_adv.end(), [](const AdvBinding& b) { return b.value != b.file; });
}

// With no inputs (the game's own control): nothing for an axis or a half, DEFAULT for a button.
std::string AdvText(const std::vector<std::string>& inputs, bool button)
{
    std::string out;
    for (const std::string& in : inputs)
        out += (out.empty() ? "" : " · ") + input::advanced::InputLabel(in);
    return out.empty() ? std::string(button ? "DEFAULT" : "") : out;
}

// What an axis action is moved to when its input is taken.
const char* AdvDirection(std::string_view action)
{
    return action == "pitch" ? "NOSE UP" : action == "roll" ? "ROLL RIGHT" : action == "yaw" ? "YAW RIGHT" : action == "look_x" ? "LOOK RIGHT"
        : action == "look_y" ? "LOOK UP" : "";
}

// Other actions using its inputs: "FIRE MISSILE (PAD B)".
std::string AdvConflicts(const AdvBinding& b)
{
    std::string out;
    for (const AdvBinding& other : s_adv)
        if (&other != &b)
            for (const std::string& in : b.value)
                if (std::find(other.value.begin(), other.value.end(), in) != other.value.end())
                    out += std::format("{}{} ({})", out.empty() ? "" : ", ", Upper(other.action->label), input::advanced::InputLabel(in));
    return out;
}

// --- Advanced controls' fine tuning: a window of its own per axis group, as the HUD colours'
// picker, with the response curve drawn. Its rows are the group's settings, edited as any (Apply writes them).
struct TuneGroup
{
    const char* label;
    const char* group;    // the tab's heading
    const char* action;   // the axis the curve shows the input of (nullptr: no curve)
    bool axis;            // -1 to 1 (else 0 to 1)
    std::initializer_list<const char*> keys;
};
const TuneGroup kTuneGroups[] = {
    { "Pitch tuning", "PITCH", "pitch", true, { "pitch_deadzone", "pitch_full", "pitch_curve", "pitch_invert" } },
    { "Roll tuning", "ROLL", "roll", true, { "roll_deadzone", "roll_full", "roll_curve", "roll_invert" } },
    { "Yaw tuning", "YAW", "yaw", true, { "yaw_deadzone", "yaw_full", "yaw_curve", "yaw_invert" } },
    { "Throttle tuning", "THROTTLE", "thrust", false,
      { "throttle_deadzone", "throttle_full", "throttle_curve", "lever_detent", "lever_detent_width" } },
    { "Camera tuning", "CAMERA", "look_x", true, { "look_deadzone", "look_full", "look_curve", "look_invert" } },
    // The RenoDX-style HDR grading: its rows in a window of their own instead of in the tab's list.
    { "HDR grading", "HDR GRADING", nullptr, false,
      { "hdr_exposure", "hdr_highlights", "hdr_shadows", "hdr_contrast", "hdr_saturation", "hdr_hue_correction",
        "hdr_gamut_compression", "hdr_compression" } },
};
constexpr int kHdrTuneGroup = 5;   // (kTuneGroups' HDR entry)
constexpr int kTuneRow0 = -4000;   // rows -4000 - g: group g's fine tuning
int s_tuning = -1;                 // the group whose window is open
int s_tuneCursor = 0;

bool IsTuneRow(int edit)
{
    return edit <= kTuneRow0 && edit > kTuneRow0 - int(std::size(kTuneGroups));
}

const TuneGroup& TuneAt(int edit)
{
    return kTuneGroups[size_t(kTuneRow0 - edit)];
}

// The group's settings as rows (their indices in s_edits).
std::vector<Row> TuneRows(const TuneGroup& g)
{
    std::vector<Row> rows;
    for (const char* key : g.keys)
        for (int i = 0; i < int(s_edits.size()); i++)
            if (std::string_view(s_edits[size_t(i)].entry.key) == key)
                rows.push_back({ i, g.group });
    return rows;
}

float TuneNumber(std::string_view key)
{
    for (const Edit& e : s_edits)
        if (key == e.entry.key)
        {
            const std::string v = Effective(e);
            return e.entry.kind == settings::ValueKind::Bool ? (v == "1" || v == "true" ? 1.0f : 0.0f) : float(std::atof(v.c_str()));
        }
    return 0.0f;
}

// The response as input/advanced.cpp shapes it, from the window's values: 0 to 1 in, 0 to 1 out.
float TuneShape(const TuneGroup& g, float a)
{
    const std::string prefix = std::string(g.keys.begin()[0]).substr(0, std::string(g.keys.begin()[0]).find('_'));
    const float dz = TuneNumber(prefix + "_deadzone") / 100.0f, full = std::max(dz + 0.01f, TuneNumber(prefix + "_full") / 100.0f),
        curve = TuneNumber(prefix + "_curve") / 100.0f;
    if (a <= dz)
        return 0.0f;
    const float x = std::min(1.0f, (a - dz) / std::max(0.01f, full - dz));
    return (1.0f - curve) * x + curve * x * x * x;
}

bool TuneUnsaved(const TuneGroup& g);

void LoadEdits()
{
    LoadBindings();
    LoadAdv();
    s_edits.clear();
    s_modesMonitor = "?";
    for (const settings::Entry& e : settings::Entries())
    {
        const std::string file = settings::FileValue(e);
        Edit edit{ e, file, file };
        if (const char* v = std::getenv(e.env))
            edit.running = v;
        edit.environment = settings::FromEnvironment(e);
        edit.choices = e.choices ? e.choices : "";
        edit.def = e.def;
        if (std::string_view(e.key) == "monitor")
            edit.choices = MonitorChoices();
        else if (std::string_view(e.key) == "hdr_peak" && DisplayPeakNits() > 0.0f)
            edit.def = FormatNumber(std::round(DisplayPeakNits()), 1.0);
        s_edits.push_back(std::move(edit));
    }
    // An older file's output.mode = "deep" (10-bit SDR) is graphics.sdr_10bit: offered as a change to apply.
    if (Edit* mode = Find("mode"); mode && mode->value == "deep")
    {
        mode->value.clear();
        if (Edit* deep = Find("sdr_10bit"))
            SetValue(*deep, "1");
    }
    // A refresh rate none of the monitor's modes has (an older file's, set on a 0-500 slider): offered as Highest.
    if (Edit* refresh = Find("refresh"); refresh && !refresh->value.empty() && !SameValue(refresh->value, "0"))
    {
        const double hz = std::atof(refresh->value.c_str());
        bool reported = false;
        for (const DisplayMode& m : s_modes)
            reported = reported || std::fabs(m.hz - hz) < 0.6;
        if (!reported)
            refresh->value.clear();
    }
    s_runningKnown = true;
    s_fileError = settings::FileError();
}

// HDR chosen and the display offering it: the output is HDR's 16-bit float, the bit-depth upgrade shows on (its own
// value is kept for SDR).
bool HdrOn()
{
    const Edit* mode = Find("mode");
    return mode && Effective(*mode) == "hdr" && HdrAvailable();
}

// Advanced controls' rows other than its switch while the switch is off, or null.
const char* AdvancedDisabled()
{
    const Edit* on = Find("custom_flight");
    return on && Effective(*on) != "1" ? "ONLY WITH ADVANCED CONTROLS ON" : nullptr;
}

const char* BindingsDisabled()
{
    // Keyboard and mouse as it is set (the environment's when its variable is set there).
    const Edit* kbm = Find("keyboard_mouse");
    return kbm && (kbm->environment ? kbm->running : Effective(*kbm)) != "1" ? "ONLY WITH KEYBOARD AND MOUSE ON (THE CONTROLS TAB)" : nullptr;
}

// Why a row can't be changed now (it is greyed out), or null.
const char* Disabled(const Edit& e)
{
    const std::string_view key = e.entry.key;
    if (std::string_view(e.entry.tab) == "Advanced controls" && key != "custom_flight")
        if (const char* why = AdvancedDisabled())
            return why;
    if (key == "check_updates" && update::Get().state == update::State::Off)
        return "ONLY IN A RELEASE DOWNLOAD: A BUILD FROM SOURCE DOESN'T UPDATE ITSELF";
    // The capture rows: with the captures' switch on.
    if (key == "frame_ring" || key == "frame_ring_height" || key == "pipeline_ring")
        if (const Edit* on = Find("captures"); on && Effective(*on) != "1")
            return "ONLY WITH CAPTURES ON";
    // The touchpad as Back: with a DualShock 4 or DualSense connected.
    if (key == "touchpad_back")
    {
        int count = 0;
        bool playStation = false;
        if (SDL_JoystickID* pads = SDL_GetGamepads(&count))
        {
            for (int i = 0; i < count; i++)
                if (const SDL_GamepadType t = SDL_GetGamepadTypeForID(pads[i]); t == SDL_GAMEPAD_TYPE_PS4 || t == SDL_GAMEPAD_TYPE_PS5)
                    playStation = true;
            SDL_free(pads);
        }
        if (!playStation)
            return "NO DUALSHOCK 4 OR DUALSENSE CONNECTED";
    }
    // The adaptive triggers: with a DualSense connected.
    if (key == "adaptive_triggers")
    {
        int count = 0;
        bool dualSense = false;
        if (SDL_JoystickID* pads = SDL_GetGamepads(&count))
        {
            for (int i = 0; i < count; i++)
                dualSense = dualSense || SDL_GetGamepadTypeForID(pads[i]) == SDL_GAMEPAD_TYPE_PS5;
            SDL_free(pads);
        }
        if (!dualSense)
            return "NO DUALSENSE CONNECTED";
    }
    // The mouse settings apply with keyboard and mouse on.
    if (std::string_view(e.entry.section) == "mouse")
        if (const char* why = BindingsDisabled())
            return why;
    // FSR's quality picks the scene's size: the render resolution doesn't apply with it.
    if (key == "render_resolution")
        if (const Edit* aa = Find("aa"); aa && Effective(*aa).find("fsr") != std::string::npos)
            return "DISABLED WITH FSR: FSR QUALITY PICKS THE RENDER RESOLUTION";
    // Analog yaw: Advanced controls, when on, gives the triggers' rudder its analog yaw itself.
    if (key == "analog_yaw")
        if (const Edit* adv = Find("custom_flight"); adv && Effective(*adv) == "1")
            return "YAW IS ANALOG WITH ADVANCED CONTROLS";
    // Motion blur reads the motion vectors temporal AA and FSR make.
    if (key == "motion_blur")
        if (const Edit* aa = Find("aa"); !aa || (Effective(*aa).find("taa") == std::string::npos && Effective(*aa).find("fsr") == std::string::npos))
            return "ONLY WITH TAA OR FSR 3.1 ANTI-ALIASING";
    // FSR quality only with FSR.
    if (key == "fsr_quality")
        if (const Edit* aa = Find("aa"); !aa || Effective(*aa).find("fsr") == std::string::npos)
            return "ONLY WITH FSR 3.1 ANTI-ALIASING";
    if (key == "refresh")
    {
        const Edit* f = Find("fullscreen");
        if (!f || Effective(*f) != "exclusive")
            return "ONLY IN EXCLUSIVE FULLSCREEN: WINDOWED AND BORDERLESS RUN AT THE DESKTOP'S REFRESH RATE";
    }
    if (key == "mode" || key.starts_with("hdr_"))
    {
        if (!HdrAvailable())
            return "NO HDR DISPLAY: TURN ON HDR FOR THIS DISPLAY IN THE SYSTEM'S DISPLAY SETTINGS";
        if (const Edit* mode = Find("mode"); key != "mode" && mode && Effective(*mode) != "hdr")
            return "HDR IS OFF";
        // The grading rows apply with RenoDRT or PsychoV-30 tone mapping. Gamut compression and compression apply with
        // PsychoV-30 only.
        const Edit* tm = Find("hdr_tonemap");
        const std::string tonemap = tm ? Effective(*tm) : "psychov30";
        if ((key == "hdr_gamut_compression" || key == "hdr_compression") && tonemap != "psychov30")
            return "ONLY WITH PSYCHOV30 TONE MAPPING";
        if (key == "hdr_exposure" || key == "hdr_highlights" || key == "hdr_shadows" || key == "hdr_contrast" || key == "hdr_saturation" ||
            key == "hdr_scene" || key == "hdr_hue_correction" || key == "hdr_effects" || key == "hdr_lit_highlights" || key == "hdr_clouds" ||
            key == "hdr_cloud_light" || key == "hdr_cloud_edge" || key == "hdr_aircraft_light" || key == "hdr_afterburner")
            if (tonemap != "renodrt" && tonemap != "psychov30")
                return "ONLY WITH RENODRT OR PSYCHOV30 TONE MAPPING";
    }
    if (key == "hud_edges")
        if (const Edit* aspect = Find("widescreen"); aspect && Effective(*aspect) == "0")
            return "ONLY WITH THE AUTO ASPECT RATIO: DOES NOTHING AT 16:9";
    if (IsHudColour(e))
    {
        if (game::HudPalette preset; RowPreset(preset))
            return "SET BY THE COLOURBLIND OPTION IN CUSTOM HUD COLOURS";
        if (const Edit* custom = Find("hud_custom"); custom && Effective(*custom) != "custom")
            return "ONLY WITH CUSTOM HUD COLOURS SET TO CUSTOM";
    }
    if (key == "cutscene_widescreen")
        if (const Edit* aspect = Find("widescreen"); aspect && Effective(*aspect) == "0")
            return "ONLY WITH THE AUTO ASPECT RATIO: AT 16:9 EVERYTHING IS 16:9";
    if (key == "sdr_10bit" && HdrOn())
        return "SDR ONLY\n"
               "HDR OUTPUT IS ALREADY AT 16-BIT";
    return nullptr;
}

bool AnyUnsaved()
{
    for (const Edit& e : s_edits)
        if (Unsaved(e))
            return true;
    return BindingsUnsaved() || AdvUnsaved();
}

bool AnyRestart()
{
    for (const Edit& e : s_edits)
        if (NeedsRestart(e))
            return true;
    return false;
}

void Status(std::string text, bool warn = false)
{
    s_status = std::move(text);
    s_statusWarn = warn;
}

bool SaveEdits()
{
    std::vector<std::pair<settings::Entry, std::string>> changes;
    for (const Edit& e : s_edits)
        if (Unsaved(e))
            changes.push_back({ e.entry, e.value });
    // The bindings, into the same file.
    std::vector<std::pair<input::kbm::BindingAction, std::vector<std::string>>> bindings;
    for (const Binding& b : s_bindings)
        if (b.value != b.file)
            bindings.push_back({ b.action, b.value });
    std::vector<std::pair<std::string, std::vector<std::string>>> adv;   // Advanced controls'
    for (const AdvBinding& b : s_adv)
        if (b.value != b.file)
            adv.push_back({ b.action->name, b.value });
    if (changes.empty() && bindings.empty() && adv.empty())
    {
        Status("Nothing to apply");
        return true;
    }
    std::string error;
    if ((!changes.empty() && !settings::Save(changes, error)) || (!bindings.empty() && !input::kbm::SaveBindings(bindings, error)) ||
        (!adv.empty() && !input::advanced::SaveBindings(adv, error)))
    {
        Status("Not saved: " + error, true);
        log::Error("Settings overlay: {}", error);
        return false;
    }
    for (Edit& e : s_edits)
        e.file = e.value;
    for (Binding& b : s_bindings)
        b.file = b.value;
    for (AdvBinding& b : s_adv)
        b.file = b.value;
    const size_t count = changes.size() + bindings.size() + adv.size();
    Status(std::format("Saved {} setting{} to {}.{}", count, count == 1 ? "" : "s",
        platform::PathUtf8(settings::Path().filename()), AnyRestart() ? " Restart the game for them to take effect." : ""));
    log::Info("Settings overlay: saved {} setting(s), {} binding(s), {} advanced binding(s)", changes.size(), bindings.size(), adv.size());
    return true;
}

void Revert()
{
    for (Edit& e : s_edits)
        e.value = e.file;
    for (Binding& b : s_bindings)
        b.value = b.file;
    for (AdvBinding& b : s_adv)
        b.value = b.file;
    s_capture = -1;
    s_advCapture = -1;
    input::advanced::EndCapture();
    Status("Changes reverted.");
}

void Restart()
{
    if (AnyUnsaved() && !SaveEdits())
        return;
    log::Info("Settings overlay: starting the game again");
    settings::ForgetApplied();
    if (!platform::StartAgain())
    {
        Status("Couldn't restart the game: please start it again yourself", true);
        return;
    }
    crash::Shutdown(0, "restarting for the settings");
}

void SetOpen(bool open)
{
    if (open == s_open.load())
        return;
    if (open)
    {
        s_hintDone = true;   // (the start-up hint and the update notice: their job is done)
        s_updateHintDone = true;
    }
    // Unapplied changes are kept while it is closed.
    if (open && (!s_runningKnown || !AnyUnsaved()))
    {
        if (s_runningKnown)
        {
            // Refresh the file's values, keep the running ones.
            std::vector<std::string> running;
            for (const Edit& e : s_edits)
                running.push_back(e.running);
            LoadEdits();
            for (size_t i = 0; i < running.size() && i < s_edits.size(); i++)
                s_edits[i].running = running[i];
        }
        else
            LoadEdits();
    }
    if (open)
    {
        s_texTaken = false;   // (the texture browser's snapshot: the frame at this opening)
        s_footer = -1;
        s_editing = -1;
        s_confirm = kNoConfirm;
        s_picker = false;
        s_scrollToCursor = true;
        Status("");
        if (!s_fileError.empty())
            Status(std::format("Error in the settings file ({}): none of its settings apply.",
                               s_fileError.substr(0, s_fileError.find(':'))), true);   // (the whole error: in the log)
    }
    s_open = open;
    input::SetHeld(open);
    // ImGui sees no events while closed: a key down when it closed (Escape) would still be down at the next opening, and
    // that frame's long time step would make it a held, repeating press (it would close again at once).
    if (s_init)
    {
        ImGui::GetIO().ClearInputKeys();
        ImGui::GetIO().ClearInputMouse();
    }
    log::Info("Settings overlay {}", open ? "opened" : "closed");
}

// The 3D scene's size the settings give (the Render resolution row), as gpu/video.cpp and
// the renderer work it out: the output (Resolution: native is the monitor's desktop, another size at least 640x360),
// the frame in it (its largest 16:9, or all of it with the auto aspect ratio when it isn't 16:9), then the render scale,
// or with FSR the quality's ratio (the render scale doesn't apply then), each raised to the minimum: half the console's
// resolution on either axis (0.5 output pixels a guest pixel, 640x360 for a 16:9 frame). False when the desktop's size
// isn't known. `asked`: the size before raising (equal to the result when not raised).
struct RenderSize
{
    int w = 0, h = 0, askedW = 0, askedH = 0;
    bool raised = false;
};

// A setting's value as the game takes it: the environment's when its variable is set there (it overrides the file).
std::string InEffect(const Edit& e)
{
    return e.environment ? e.running : Effective(e);
}

bool ComputeRenderSize(RenderSize& out)
{
    const Edit* resolution = Find("resolution");
    const Edit* widescreen = Find("widescreen");
    const Edit* scale = Find("render_resolution");
    const Edit* aa = Find("aa");
    const Edit* quality = Find("fsr_quality");
    int ow = 0, oh = 0;
    const std::string res = resolution ? InEffect(*resolution) : "native";
    if (res == "native" || res.empty())
        ow = s_desktopW, oh = s_desktopH;
    else if (std::sscanf(res.c_str(), "%dx%d", &ow, &oh) == 2)
        ow = std::clamp(ow, 640, 16384), oh = std::clamp(oh, 360, 16384);
    if (ow <= 0 || oh <= 0)
        return false;
    constexpr double kW = 1280.0, kH = 720.0;
    double sx = std::min(ow / kW, oh / kH), sy = sx, fw = std::round(kW * sx), fh = std::round(kH * sy);
    const int64_t shape = int64_t(ow) * 720 - int64_t(oh) * 1280;
    if ((!widescreen || InEffect(*widescreen) != "0") && (shape > 720 || shape < -1280))
        fw = ow, fh = oh, sx = ow / kW, sy = oh / kH;
    const double floor = 0.5 / std::min(sx, sy);
    double f = 1.0;
    if (aa && InEffect(*aa).find("fsr") != std::string::npos)
    {
        const std::string q = quality ? InEffect(*quality) : "";
        f = q == "quality" ? 1.0 / 1.5 : q == "balanced" ? 1.0 / 1.7 : q == "performance" ? 0.5 : q == "ultra" ? 1.0 / 3.0 : 1.0;
        out.askedW = int(std::lround(kW * sx * f)), out.askedH = int(std::lround(kH * sy * f));
        out.raised = f < floor;
        f = std::min(1.0, std::max(f, floor));
        out.w = int(std::lround(kW * sx * f)), out.h = int(std::lround(kH * sy * f));
        return true;
    }
    f = scale ? std::atof(InEffect(*scale).c_str()) : 1.0;
    if (!(f >= 0.25 && f <= 4.0))
        f = 1.0;
    out.askedW = std::max(1, int(std::lround(fw * f))), out.askedH = std::max(1, int(std::lround(fh * f)));
    out.raised = f < floor;
    f = std::max(f, floor);
    out.w = std::max(1, int(std::lround(fw * f))), out.h = std::max(1, int(std::lround(fh * f)));
    return true;
}

// Raised to the minimum: what the rows say (the Render resolution, Render scale, Anti-aliasing and FSR quality rows).
std::string RenderSizeNote()
{
    RenderSize r;
    if (!ComputeRenderSize(r) || !r.raised)
        return {};
    return std::format("BELOW THE MINIMUM RENDER RESOLUTION (640X360 AT 16:9)\n"
                       "{}X{} IS RAISED TO {}X{}",
        r.askedW, r.askedH, r.w, r.h);
}

void BuildRows()
{
    // The resolutions of the monitor chosen (read again when the choice changes).
    if (const Edit* monitor = Find("monitor"); monitor && Effective(*monitor) != s_modesMonitor)
    {
        s_modesMonitor = Effective(*monitor);
        ReadModes(MonitorDisplay(s_modesMonitor));
        if (Edit* resolution = Find("resolution"))
            resolution->choices = ResolutionChoices();
    }
    if (Edit* refresh = Find("refresh"))
        if (const Edit* resolution = Find("resolution"))
            refresh->choices = RefreshChoices(Effective(*resolution));
    s_rows.clear();
    if (std::string_view(kTabs[s_tab]) == "Graphics")
        s_rows.push_back({ -1, "PRESET" });
    // Grouped in the order the groups first appear, the general ones first (GENERAL leads the Game tab, whose HUD settings
    // sit in the file's display section).
    std::vector<std::string_view> groups;
    for (const Edit& e : s_edits)
        if (std::string_view(e.entry.tab) == kTabs[s_tab] && std::find(groups.begin(), groups.end(), e.entry.group) == groups.end())
            groups.push_back(e.entry.group);
    std::stable_partition(groups.begin(), groups.end(), [](std::string_view g) { return g == "GENERAL"; });
    // (The Debug tab's Rendering and Capture and logs last.)
    std::stable_partition(groups.begin(), groups.end(), [](std::string_view g) { return g != "RENDERING" && g != "CAPTURE AND LOGS"; });
    for (std::string_view g : groups)
        for (int i = 0; i < int(s_edits.size()); i++)
            if (std::string_view(s_edits[size_t(i)].entry.tab) == kTabs[s_tab] && g == s_edits[size_t(i)].entry.group)
                s_rows.push_back({ i, s_edits[size_t(i)].entry.group });
    // Extended field of view closes GENERAL: its setting is in the file's display section, ahead of the others.
    for (size_t r = 0; r < s_rows.size(); r++)
        if (s_rows[r].edit >= 0 && std::string_view(s_edits[size_t(s_rows[r].edit)].entry.key) == "fov_offset")
        {
            size_t end = r + 1;
            while (end < s_rows.size() && std::string_view(s_rows[end].group) == "GENERAL")
                end++;
            std::rotate(s_rows.begin() + std::ptrdiff_t(r), s_rows.begin() + std::ptrdiff_t(r + 1), s_rows.begin() + std::ptrdiff_t(end));
            break;
        }
    for (size_t r = 0; r < s_rows.size(); r++)
        if (s_rows[r].edit >= 0 && std::string_view(s_edits[size_t(s_rows[r].edit)].entry.key) == "dlc")
        {
            s_rows.insert(s_rows.begin() + std::ptrdiff_t(r + 1), { kDlcRow, s_rows[r].group });
            s_rows.insert(s_rows.begin() + std::ptrdiff_t(r + 2), { kDlcFolderRow, s_rows[r].group });
            break;
        }
    for (size_t r = 0; r < s_rows.size(); r++)
        if (s_rows[r].edit >= 0 && std::string_view(s_edits[size_t(s_rows[r].edit)].entry.key) == "check_updates")
        {
            s_rows.insert(s_rows.begin() + std::ptrdiff_t(r + 1), { kUpdateRow, s_rows[r].group });
            break;
        }
    for (size_t r = 0; r < s_rows.size(); r++)
        if (s_rows[r].edit >= 0 && std::string_view(s_edits[size_t(s_rows[r].edit)].entry.key) == "render_resolution")
        {
            s_rows.insert(s_rows.begin() + std::ptrdiff_t(r), { kRenderResRow, s_rows[r].group });   // above Render scale
            break;
        }
    for (size_t r = 0; r < s_rows.size(); r++)
        if (s_rows[r].edit >= 0 && std::string_view(s_edits[size_t(s_rows[r].edit)].entry.key) == "texture_replacements")
        {
            s_rows.insert(s_rows.begin() + std::ptrdiff_t(r), { kTexturesRow, s_rows[r].group });   // above Texture replacements
            break;
        }
    for (size_t r = 0; r < s_rows.size(); r++)
        if (s_rows[r].edit >= 0 && std::string_view(s_edits[size_t(s_rows[r].edit)].entry.key) == "hdr_ui_brightness")
        {
            // (in the HDR section, after the UI brightness)
            s_rows.insert(s_rows.begin() + std::ptrdiff_t(r + 1), { kTuneRow0 - kHdrTuneGroup, s_rows[r].group });
            break;
        }
    if (std::string_view(kTabs[s_tab]) == "Keybinds")
        for (size_t i = 0; i < s_bindings.size(); i++)
            s_rows.push_back({ kBindingRow0 - int(i), s_bindings[i].info->group });
    if (std::string_view(kTabs[s_tab]) == "Advanced controls")
    {
        std::vector<Row> settingsRows = std::move(s_rows);
        s_rows.clear();
        auto settingsOf = [&](std::string_view group) {
            for (const Row& r : settingsRows)
                if (group == r.group)
                    s_rows.push_back(r);
        };
        settingsOf("ADVANCED CONTROLS");
        for (const char* group : { "PITCH", "ROLL", "YAW", "THROTTLE", "CAMERA", "WEAPONS", "OTHER" })
        {
            for (size_t i = 0; i < s_adv.size(); i++)
                if (std::string_view(s_adv[i].action->group) == group)
                    s_rows.push_back({ kAdvRow0 - int(i), s_adv[i].action->group });
            for (int g = 0; g < int(std::size(kTuneGroups)); g++)
                if (std::string_view(kTuneGroups[g].group) == group)
                    s_rows.push_back({ kTuneRow0 - g, kTuneGroups[g].group });   // its settings in a window
        }
    }
    if (std::string_view(kTabs[s_tab]) == "Game")
    {
        s_mods = mods::Status();
        for (size_t i = 0; i < s_mods.size(); i++)
            s_rows.push_back({ kModRow0 - int(i), "MODS" });
        s_rows.push_back({ kQuitRow, "EXIT" });
    }
    s_cursor = std::clamp(s_cursor, 0, std::max(0, int(s_rows.size()) - 1));
}

// The Updates row's value: what the updater is doing or found.
std::string UpdateText(bool* problem = nullptr)
{
    const update::Status st = update::Get();
    if (problem)
        *problem = st.state == update::State::Failed;
    switch (st.state)
    {
    case update::State::Off: return std::format("{} (SOURCE BUILD)", update::Version());
    case update::State::Idle: return "CHECK NOW";
    case update::State::Checking: return "CHECKING";
    case update::State::UpToDate: return std::format("{}: UP TO DATE", update::Version());
    case update::State::Available: return std::format("UPDATE TO {}", st.latest);
    case update::State::Downloading:
        return st.progress >= 0.0f ? std::format("DOWNLOADING {:.0f}%", st.progress * 100.0f) : std::string("DOWNLOADING");
    case update::State::Installing: return "INSTALLING";
    case update::State::Installed: return "RESTART TO FINISH";
    case update::State::Failed: return "FAILED: CHECK AGAIN";
    }
    return {};
}

// A row's value as shown.
std::string ValueText(const Row& row)
{
    if (row.edit >= 0 && IsHudColour(s_edits[size_t(row.edit)]))
    {
        const Edit& e = s_edits[size_t(row.edit)];
        if (game::HudPalette preset; RowPreset(preset))
            return std::format("#{:06X}", PaletteColour(preset, e.entry.key));
        const std::string v = Upper(Effective(e));
        if (!v.empty())
            return "#" + v;
        if (std::string_view(e.entry.key) == "hud_colour")
            return "GREEN";
        return std::format("AUTO #{:06X}", PaletteColour(RowPalette(), e.entry.key));
    }
    if (row.edit == kDlcRow)
        return DlcCount();
    if (row.edit == kDlcFolderRow)
        return s_dlcFolderFailed ? "CAN'T OPEN" : "";
    if (row.edit == kUpdateRow)
        return UpdateText();
    if (row.edit == kTexturesRow)
        return std::format("{} IN USE", BrowserTextures().size());
    if (IsModRow(row.edit))
        return ModText(ModAt(row.edit));
    if (IsBindingRow(row.edit))
        return BindingText(BindingAt(row.edit).value);
    if (IsAdvRow(row.edit))
        return AdvText(AdvAt(row.edit).value, AdvAt(row.edit).action->kind == input::advanced::Kind::Button);
    if (IsTuneRow(row.edit) && !TuneAt(row.edit).action)
        return std::string();   // (the HDR grading: no summary of its values on the row)
    if (IsTuneRow(row.edit))
    {
        const TuneGroup& g = TuneAt(row.edit);
        const std::string prefix = std::string(g.keys.begin()[0]).substr(0, std::string(g.keys.begin()[0]).find('_'));
        std::string out = std::format("DEAD ZONE {:.0f} · FULL {:.0f} · CURVE {:.0f}", TuneNumber(prefix + "_deadzone"),
            TuneNumber(prefix + "_full"), TuneNumber(prefix + "_curve"));
        if (g.axis && TuneNumber(prefix + "_invert") > 0.5f)
            out += " · INVERTED";
        return out;
    }
    if (row.edit == kRenderResRow)
    {
        RenderSize r;
        return ComputeRenderSize(r) ? std::format("{} X {}", r.w, r.h) : std::string("?");   // capital X as the values' caps
    }
    if (row.edit == kQuitRow)
        return {};
    if (row.edit == kPresetRow)
    {
        const int p = CurrentPreset();
        return p < 0 ? "CUSTOM" : Upper(kPresets[p].name);
    }
    const Edit& e = s_edits[size_t(row.edit)];
    if (std::string_view(e.entry.key) == "sdr_10bit" && HdrOn())
        return "ON";
    const std::string v = Effective(e);
    for (const auto& [value, label] : Choices(e.choices.c_str()))
        if (SameValue(value, v))
            return label == "日本語" && !s_body->IsGlyphInFont(0x65E5) ? std::string("JAPANESE") : Upper(label);
    if (e.entry.kind == settings::ValueKind::Bool)
        return IsOn(v) ? "ON" : "OFF";
    if (e.entry.kind == settings::ValueKind::Text && v.empty())
        return "(NONE)";
    return v;
}

// A number as the setting's text: no more decimals than its step has (0.05 -> 2), no float noise.
std::string FormatNumber(double x, double step)
{
    int decimals = 0;
    for (double s = step; decimals < 4 && std::fabs(s - std::round(s)) > 1e-9; s *= 10.0)
        decimals++;
    std::string out = std::format("{:.{}f}", x, decimals);
    if (out.find('.') != std::string::npos)
    {
        while (out.back() == '0')
            out.pop_back();
        if (out.back() == '.')
            out.pop_back();
    }
    return out == "-0" ? "0" : out;
}

bool HasChoices(const Edit& e)
{
    return !e.choices.empty();
}

bool IsRange(const Edit& e)
{
    return e.entry.kind == settings::ValueKind::Number && e.entry.max > e.entry.min && !HasChoices(e);
}

bool IsText(const Edit& e)
{
    return !HasChoices(e) && e.entry.kind != settings::ValueKind::Bool && !IsRange(e);
}

// A choice row's value as a position: its index (-1: a value not in the list, a custom preset) and the count (0: not a
// choice row: a range or text).
std::pair<int, int> ChoicePosition(const Row& row)
{
    if (row.edit == kDlcRow || row.edit == kDlcFolderRow || row.edit == kQuitRow || row.edit == kRenderResRow || row.edit == kTexturesRow ||
        row.edit == kUpdateRow || IsBindingRow(row.edit) || IsAdvRow(row.edit) || IsTuneRow(row.edit) || IsModRow(row.edit))
        return { -1, 0 };
    if (row.edit == kPresetRow)
        return { CurrentPreset(), int(std::size(kPresets)) };
    const Edit& e = s_edits[size_t(row.edit)];
    const std::string v = Effective(e);
    const auto choices = Choices(e.choices.c_str());
    if (!choices.empty())
    {
        int at = -1;
        for (int i = 0; i < int(choices.size()); i++)
            if (SameValue(choices[size_t(i)].first, v))
                at = i;
        return { at, int(choices.size()) };
    }
    if (e.entry.kind == settings::ValueKind::Bool)
        return { IsOn(v) ? 1 : 0, 2 };
    return { -1, 0 };
}

// Left / right on a row (dir -1 / +1): the previous / next value. A range moves by `steps` of its step (held: faster).
// Left / right stop at the ends (wrap: A / Enter cycles through them).
void Step(const Row& row, int dir, int steps = 1, bool wrap = false)
{
    if (row.edit >= 0 && Disabled(s_edits[size_t(row.edit)]))
        return;
    if (const auto [at, count] = ChoicePosition(row); count > 0)
    {
        int next = at < 0 ? (dir > 0 ? 0 : count - 1) : at + dir;
        next = wrap ? (next + count) % count : std::clamp(next, 0, count - 1);
        if (next == at)
            return;
        if (row.edit == kPresetRow)
            ApplyPreset(next);
        else if (Edit& e = s_edits[size_t(row.edit)]; HasChoices(e))
            SetValue(e, Choices(e.choices.c_str())[size_t(next)].first);
        else
            SetValue(e, next ? "1" : "0");
        return;
    }
    if (row.edit < 0)
        return;
    Edit& e = s_edits[size_t(row.edit)];
    const std::string v = Effective(e);
    if (IsRange(e))
    {
        const double step = e.entry.step > 0.0 ? e.entry.step : (e.entry.max - e.entry.min) / 100.0;
        const double x = std::clamp(std::round((std::atof(v.c_str()) + dir * steps * step) / step) * step, e.entry.min, e.entry.max);
        SetValue(e, FormatNumber(x, step));
    }
}

bool IsProfileName(const Edit& e)
{
    return std::string_view(e.entry.key) == "profile_name";
}

// Typing the profile name: a gamertag's characters only.
int ProfileNameFilter(ImGuiInputTextCallbackData* data)
{
    return settings::ProfileNameChar(data->EventChar) ? 0 : 1;
}

void StartEditing(int edit)
{
    s_editing = edit;
    s_editStart = true;
    const Edit& e = s_edits[size_t(edit)];
    std::snprintf(s_text, sizeof(s_text), "%s", (IsProfileName(e) ? settings::CleanProfileName(Effective(e)) : Effective(e)).c_str());
}

void Activate(const Row& row)
{
    if (row.edit == kDlcRow || row.edit == kRenderResRow || IsModRow(row.edit))
        return;
    if (row.edit == kDlcFolderRow)
    {
        OpenDlcFolder();
        return;
    }
    if (row.edit == kUpdateRow)
    {
        switch (update::Get().state)
        {
        case update::State::Idle:
        case update::State::UpToDate:
        case update::State::Failed: update::Check(); break;
        case update::State::Available: update::Install(); break;
        case update::State::Installed: Restart(); break;
        default: break;   // (busy, or a source build)
        }
        return;
    }
    if (row.edit == kTexturesRow)
    {
        s_browsing = !BrowserTextures().empty();
        return;
    }
    if (IsBindingRow(row.edit))
    {
        if (BindingsDisabled())
            return;
        s_capture = kBindingRow0 - row.edit;
        s_captureUntil = SDL_GetTicks() + 5000;
        Status(std::format("{}: press a key or mouse button to add it. Esc: clear all. B or 5 s: cancel.",
            BindingAt(row.edit).info->label));
        return;
    }
    if (IsTuneRow(row.edit))
    {
        if (TuneAt(row.edit).action && AdvancedDisabled())
            return;
        s_tuning = kTuneRow0 - row.edit;
        s_tuneCursor = 0;
        return;
    }
    if (IsAdvRow(row.edit))
    {
        if (AdvancedDisabled())
            return;
        const AdvBinding& b = AdvAt(row.edit);
        s_advCapture = kAdvRow0 - row.edit;
        s_advCaptureUntil = SDL_GetTicks() + 6000;
        input::advanced::BeginCapture(b.action->kind);
        Status(b.action->kind == input::advanced::Kind::Axis
                ? std::format("{}: move an axis to {}. Esc: clear all. 6 s: cancel.", b.action->label,
                      Upper(AdvDirection(b.action->name)))
                : std::format("{}: press a button or hat, or move an axis or trigger to add it. Esc: clear all. 6 s: cancel.",
                      b.action->label));
        return;
    }
    if (row.edit == kQuitRow)
    {
        s_confirm = kConfirmQuit;
        s_confirmFocus = 1;   // Cancel first
        return;
    }
    if (row.edit >= 0 && Disabled(s_edits[size_t(row.edit)]))
        return;
    if (row.edit >= 0 && IsHudColour(s_edits[size_t(row.edit)]))
    {
        s_pickerKey = s_edits[size_t(row.edit)].entry.key;
        s_pickerAuto = ColourOf(s_pickerKey) == game::kHudAuto;
        SetPicker(PaletteColour(RowPalette(), s_pickerKey));
        s_picker = true;
        return;
    }
    // Text, and the number of a range: typed. (Not a range from a controller: left and right change it, and there's
    // nothing to type with.)
    if (row.edit >= 0 && IsRange(s_edits[size_t(row.edit)]) && s_gamepad)
        return;
    if (row.edit >= 0 && (IsText(s_edits[size_t(row.edit)]) || IsRange(s_edits[size_t(row.edit)])))
        StartEditing(row.edit);
    else
        Step(row, 1, 1, true);
}

// Quit game, confirmed: as closing the game's window.
void Quit()
{
    log::Info("Settings overlay: quitting the game");
    crash::Shutdown(0, "quit from the settings overlay");
}

// The picker's OK.
void PickerAccept()
{
    s_picker = false;
    if (Edit* e = Find(s_pickerKey))
        SetValue(*e, s_pickerAuto ? std::string() : std::format("{:06X}", PickerColour()));
}

// The row's default in the picker: the game's green, or Auto.
void PickerDefault()
{
    s_pickerAuto = true;
    const bool open = s_picker;
    s_picker = true;
    SetPicker(PaletteColour(RowPalette(), s_pickerKey));
    s_picker = open;
}

void ConfirmPress(int button)
{
    const Confirm confirm = s_confirm;
    s_confirm = kNoConfirm;
    if (button != 0)
        return;
    if (confirm == kConfirmQuit)
        Quit();
    else if (confirm == kConfirmRestart)
        Restart();
}

// The footer's buttons shown: Restart now only when a change needs it (one to a setting that isn't live).
int FooterCount()
{
    return AnyRestart() ? kFooterCount : kRestart;
}

void FooterPress(int button)
{
    if (button == kRevert)
        Revert();
    else if (button == kApply)
        SaveEdits();
    else if (button == kRestart && AnyRestart())
    {
        s_confirm = kConfirmRestart;   // asks first
        s_confirmFocus = 1;
    }
}

void SwitchTab(int tab)
{
    s_tab = (tab + kTabCount) % kTabCount;
    s_cursor = 0;
    s_footer = -1;
    s_editing = -1;
    s_scrollToCursor = true;
}

// repeat: held, it presses again (moving and changing values, not closing or applying).
bool Pressed(std::initializer_list<ImGuiKey> keys, bool repeat = false)
{
    for (ImGuiKey k : keys)
        if (ImGui::IsKeyPressed(k, repeat))
            return true;
    return false;
}

void HandleKeys()
{
    const ImGuiIO& io = ImGui::GetIO();
    for (int k = ImGuiKey_GamepadStart; k <= ImGuiKey_GamepadRStickDown; k++)
        if (ImGui::IsKeyPressed(ImGuiKey(k), false))
            s_gamepad = true;
    if (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f || ImGui::IsAnyMouseDown())
        s_gamepad = false;
    for (int k = ImGuiKey_NamedKey_BEGIN; k < ImGuiKey_GamepadStart; k++)
        if (ImGui::IsKeyPressed(ImGuiKey(k), false))
            s_gamepad = false;
    // AC6_UI_GAMEPAD=1 (tests: screenshots without a controller): the controller's prompts whatever was used.
    static const bool forceGamepad = [] { const char* v = std::getenv("AC6_UI_GAMEPAD"); return v && *v == '1'; }();
    if (forceGamepad)
        s_gamepad = true;
    if (s_editing >= 0)
    {
        if (Pressed({ ImGuiKey_GamepadFaceDown, ImGuiKey_GamepadFaceRight }))
            s_endEditing = true;   // the value as typed
        return;   // the text field has the keyboard
    }
    if (s_capture >= 0)
    {
        if (SDL_GetTicks() > s_captureUntil)
        {
            s_capture = -1;
            Status("No key pressed: unchanged.");
        }
        return;   // the binding waits for its key (HandleEvent)
    }
    if (s_advCapture >= 0)
    {
        // Advanced controls: an input moved or pressed (input/advanced.cpp watches them all), or the time up.
        if (s_advCapture < int(s_adv.size()))
            if (const std::string found = input::advanced::PollCapture(); !found.empty())
            {
                AdvBinding& b = s_adv[size_t(s_advCapture)];
                s_advCapture = -1;
                const std::string label = input::advanced::InputLabel(found);
                if (const auto it = std::find(b.value.begin(), b.value.end(), found); it != b.value.end())
                {
                    b.value.erase(it);
                    Status(std::format("{}: {} removed.", b.action->label, label));
                }
                else if (b.value.size() >= kMaxAdvInputs)
                    Status(std::format("{}: {} inputs at most. Remove one first", b.action->label, kMaxAdvInputs), true);
                else
                {
                    b.value.push_back(found);
                    Status(std::format("{}: {} added.", b.action->label, label));
                }
                return;
            }
        if (SDL_GetTicks() > s_advCaptureUntil)
        {
            s_advCapture = -1;
            input::advanced::EndCapture();
            Status("Nothing moved or pressed: unchanged.");
        }
        return;
    }
    if (s_tuning >= 0)
    {
        // Up / down a setting, left / right its value (held: faster), Del / X its default. Enter / A, Esc / B close.
        const std::vector<Row> rows = TuneRows(kTuneGroups[size_t(s_tuning)]);
        if (rows.empty())
        {
            s_tuning = -1;
            return;
        }
        s_tuneCursor = std::clamp(s_tuneCursor, 0, int(rows.size()) - 1);
        if (Pressed({ ImGuiKey_UpArrow, ImGuiKey_W, ImGuiKey_GamepadDpadUp, ImGuiKey_GamepadLStickUp }, true))
            s_tuneCursor = (s_tuneCursor + int(rows.size()) - 1) % int(rows.size());
        if (Pressed({ ImGuiKey_DownArrow, ImGuiKey_S, ImGuiKey_GamepadDpadDown, ImGuiKey_GamepadLStickDown }, true))
            s_tuneCursor = (s_tuneCursor + 1) % int(rows.size());
        float held = 0.0f;
        for (ImGuiKey k : { ImGuiKey_LeftArrow, ImGuiKey_RightArrow, ImGuiKey_A, ImGuiKey_D, ImGuiKey_GamepadDpadLeft,
                 ImGuiKey_GamepadDpadRight, ImGuiKey_GamepadLStickLeft, ImGuiKey_GamepadLStickRight })
            held = std::max(held, ImGui::GetKeyData(k)->DownDuration);
        const Row& row = rows[size_t(s_tuneCursor)];
        if (Pressed({ ImGuiKey_LeftArrow, ImGuiKey_A, ImGuiKey_GamepadDpadLeft, ImGuiKey_GamepadLStickLeft }, true))
            Step(row, -1, held > 1.6f ? 10 : 1);
        if (Pressed({ ImGuiKey_RightArrow, ImGuiKey_D, ImGuiKey_GamepadDpadRight, ImGuiKey_GamepadLStickRight }, true))
            Step(row, 1, held > 1.6f ? 10 : 1);
        if (Pressed({ ImGuiKey_Delete, ImGuiKey_GamepadFaceLeft }))
            s_edits[size_t(row.edit)].value.clear();
        if (Pressed({ ImGuiKey_Enter, ImGuiKey_KeypadEnter, ImGuiKey_Space, ImGuiKey_GamepadFaceDown, ImGuiKey_Escape,
                ImGuiKey_GamepadFaceRight }))
            s_tuning = -1;
        return;
    }
    if (s_picker)
    {
        // Left / right saturation, up / down brightness, Q / E or the shoulder buttons hue. Held: faster.
        float held = 0.0f;
        for (ImGuiKey k : { ImGuiKey_LeftArrow, ImGuiKey_RightArrow, ImGuiKey_UpArrow, ImGuiKey_DownArrow, ImGuiKey_A, ImGuiKey_D, ImGuiKey_W,
                 ImGuiKey_S, ImGuiKey_Q, ImGuiKey_E, ImGuiKey_GamepadDpadLeft, ImGuiKey_GamepadDpadRight, ImGuiKey_GamepadDpadUp,
                 ImGuiKey_GamepadDpadDown, ImGuiKey_GamepadLStickLeft, ImGuiKey_GamepadLStickRight, ImGuiKey_GamepadLStickUp,
                 ImGuiKey_GamepadLStickDown, ImGuiKey_GamepadL1, ImGuiKey_GamepadR1 })
            held = std::max(held, ImGui::GetKeyData(k)->DownDuration);
        const float step = held > 0.6f ? 0.03f : 0.01f;
        const float h0 = s_pickerH, s0 = s_pickerS, v0 = s_pickerV;
        if (Pressed({ ImGuiKey_LeftArrow, ImGuiKey_A, ImGuiKey_GamepadDpadLeft, ImGuiKey_GamepadLStickLeft }, true))
            s_pickerS = std::max(0.0f, s_pickerS - step);
        if (Pressed({ ImGuiKey_RightArrow, ImGuiKey_D, ImGuiKey_GamepadDpadRight, ImGuiKey_GamepadLStickRight }, true))
            s_pickerS = std::min(1.0f, s_pickerS + step);
        if (Pressed({ ImGuiKey_UpArrow, ImGuiKey_W, ImGuiKey_GamepadDpadUp, ImGuiKey_GamepadLStickUp }, true))
            s_pickerV = std::min(1.0f, s_pickerV + step);
        if (Pressed({ ImGuiKey_DownArrow, ImGuiKey_S, ImGuiKey_GamepadDpadDown, ImGuiKey_GamepadLStickDown }, true))
            s_pickerV = std::max(0.0f, s_pickerV - step);
        if (Pressed({ ImGuiKey_Q, ImGuiKey_GamepadL1 }, true))
            s_pickerH = std::fmod(s_pickerH - step + 1.0f, 1.0f);
        if (Pressed({ ImGuiKey_E, ImGuiKey_GamepadR1 }, true))
            s_pickerH = std::fmod(s_pickerH + step, 1.0f);
        if (h0 != s_pickerH || s0 != s_pickerS || v0 != s_pickerV)
            s_pickerAuto = false;
        if (Pressed({ ImGuiKey_Delete, ImGuiKey_GamepadFaceLeft }))
            PickerDefault();
        if (Pressed({ ImGuiKey_Enter, ImGuiKey_KeypadEnter, ImGuiKey_Space, ImGuiKey_GamepadFaceDown }))
            PickerAccept();
        else if (Pressed({ ImGuiKey_Escape, ImGuiKey_GamepadFaceRight }))
            s_picker = false;
        return;
    }
    if (s_confirm != kNoConfirm)
    {
        // Left / right between Confirm and Cancel, Enter / A presses, Esc / B cancels.
        if (Pressed({ ImGuiKey_LeftArrow, ImGuiKey_A, ImGuiKey_GamepadDpadLeft, ImGuiKey_GamepadLStickLeft }))
            s_confirmFocus = 0;
        if (Pressed({ ImGuiKey_RightArrow, ImGuiKey_D, ImGuiKey_GamepadDpadRight, ImGuiKey_GamepadLStickRight }))
            s_confirmFocus = 1;
        if (Pressed({ ImGuiKey_Enter, ImGuiKey_KeypadEnter, ImGuiKey_Space, ImGuiKey_GamepadFaceDown }))
            ConfirmPress(s_confirmFocus);
        else if (Pressed({ ImGuiKey_Escape, ImGuiKey_GamepadFaceRight }))
            s_confirm = kNoConfirm;
        return;
    }

    // The texture browser: the arrows between the thumbnails, Enter dumps the one chosen, Esc / B back to the list.
    if (s_browsing)
    {
        if (Pressed({ ImGuiKey_R, ImGuiKey_GamepadFaceUp }))
        {
            RefreshTextures();
            Status(std::format("Textures refreshed: {} in the last frame.", BrowserTextures().size()));
        }
        const auto& list = BrowserTextures();
        if (list.empty() || s_rows.empty() || s_rows[size_t(s_cursor)].edit != kTexturesRow)
            s_browsing = false;
        else
        {
            const int n = int(list.size());
            const int before = s_texCursor;
            if (Pressed({ ImGuiKey_LeftArrow, ImGuiKey_A, ImGuiKey_GamepadDpadLeft, ImGuiKey_GamepadLStickLeft }, true))
                s_texCursor = std::max(0, s_texCursor - 1);
            if (Pressed({ ImGuiKey_RightArrow, ImGuiKey_D, ImGuiKey_GamepadDpadRight, ImGuiKey_GamepadLStickRight }, true))
                s_texCursor = std::min(n - 1, s_texCursor + 1);
            if (Pressed({ ImGuiKey_UpArrow, ImGuiKey_W, ImGuiKey_GamepadDpadUp, ImGuiKey_GamepadLStickUp }, true))
                s_texCursor = std::max(0, s_texCursor - s_texColumns);
            if (Pressed({ ImGuiKey_DownArrow, ImGuiKey_S, ImGuiKey_GamepadDpadDown, ImGuiKey_GamepadLStickDown }, true))
                s_texCursor = std::min(n - 1, s_texCursor + s_texColumns);
            s_texCursor = std::clamp(s_texCursor, 0, n - 1);
            if (s_texCursor != before)
                s_texFollow = true;
            if (Pressed({ ImGuiKey_Enter, ImGuiKey_KeypadEnter, ImGuiKey_Space, ImGuiKey_GamepadFaceDown }))
            {
                const auto& t = list[size_t(s_texCursor)];
                gpu::mods::RequestDump(t);
                Status(std::format("Texture {:016X}: dumped to dump/textures/{:016X}.dds next to the game.", t.hash, t.hash));
            }
            if (Pressed({ ImGuiKey_Escape, ImGuiKey_GamepadFaceRight }))
                s_browsing = false;
            return;
        }
    }

    // Tabs: Q / E (PgUp / PgDn, Ctrl+(Shift+)Tab too), the controller's LT / RT.
    if (Pressed({ ImGuiKey_GamepadL2, ImGuiKey_PageUp, ImGuiKey_Q }) || (io.KeyCtrl && io.KeyShift && Pressed({ ImGuiKey_Tab })))
        SwitchTab(s_tab - 1);
    else if (Pressed({ ImGuiKey_GamepadR2, ImGuiKey_PageDown, ImGuiKey_E }) || (io.KeyCtrl && !io.KeyShift && Pressed({ ImGuiKey_Tab })))
        SwitchTab(s_tab + 1);
    if (Pressed({ ImGuiKey_Escape, ImGuiKey_GamepadFaceRight }))
    {
        SetOpen(false);
        return;
    }
    if (Pressed({ ImGuiKey_GamepadFaceUp }) || (io.KeyCtrl && Pressed({ ImGuiKey_S })))
        SaveEdits();

    // Moving: WASD or the arrows. S isn't down with Ctrl (Ctrl+S applies).
    const bool up = Pressed({ ImGuiKey_UpArrow, ImGuiKey_W, ImGuiKey_GamepadDpadUp, ImGuiKey_GamepadLStickUp }, true);
    const bool down = Pressed({ ImGuiKey_DownArrow, ImGuiKey_GamepadDpadDown, ImGuiKey_GamepadLStickDown }, true) ||
        (!io.KeyCtrl && Pressed({ ImGuiKey_S }, true));
    const bool left = Pressed({ ImGuiKey_LeftArrow, ImGuiKey_A, ImGuiKey_GamepadDpadLeft, ImGuiKey_GamepadLStickLeft }, true);
    const bool right = Pressed({ ImGuiKey_RightArrow, ImGuiKey_D, ImGuiKey_GamepadDpadRight, ImGuiKey_GamepadLStickRight }, true);
    const bool accept = Pressed({ ImGuiKey_Enter, ImGuiKey_KeypadEnter, ImGuiKey_Space, ImGuiKey_GamepadFaceDown });
    const bool reset = Pressed({ ImGuiKey_Delete, ImGuiKey_GamepadFaceLeft });
    if (s_footer >= 0)
    {
        if (up && !s_rows.empty())
        {
            s_footer = -1;
            s_cursor = int(s_rows.size()) - 1;
            s_scrollToCursor = true;
        }
        if (left)
            s_footer = std::max(0, s_footer - 1);
        if (right)
            s_footer = std::min(FooterCount() - 1, s_footer + 1);
        if (accept)
            FooterPress(s_footer);
        return;
    }
    if (s_rows.empty())
        return;
    if (up && s_cursor > 0)
    {
        s_cursor--;
        s_scrollToCursor = true;
    }
    if (down)
    {
        if (s_cursor + 1 < int(s_rows.size()))
            s_cursor++;
        else
            s_footer = kApply;
        s_scrollToCursor = true;
    }
    const Row& row = s_rows[size_t(s_cursor)];
    if (left || right)
    {
        float held = 0.0f;
        for (ImGuiKey k : { ImGuiKey_LeftArrow, ImGuiKey_RightArrow, ImGuiKey_A, ImGuiKey_D, ImGuiKey_GamepadDpadLeft,
                 ImGuiKey_GamepadDpadRight, ImGuiKey_GamepadLStickLeft, ImGuiKey_GamepadLStickRight })
            held = std::max(held, ImGui::GetKeyData(k)->DownDuration);
        Step(row, left ? -1 : 1, held > 1.6f ? 50 : held > 0.6f ? 10 : 1);
    }
    if (accept)
        Activate(row);
    if (reset && row.edit >= 0 && !Disabled(s_edits[size_t(row.edit)]))
        s_edits[size_t(row.edit)].value.clear();
    else if (reset && row.edit == kPresetRow)
        ApplyPreset(0);
    else if (reset && IsBindingRow(row.edit) && !BindingsDisabled())
        BindingAt(row.edit).value = BindingAt(row.edit).action.defaults;   // its defaults
    else if (reset && IsAdvRow(row.edit) && !AdvancedDisabled())
        AdvAt(row.edit).value.clear();   // none: the game's
}

// --- The screen ---------------------------------------------------------------------------------------------------

constexpr float kListX = 56.0f, kListW = 820.0f, kTop = 166.0f, kBottom = 978.0f, kRowH = 44.0f, kValueW = 300.0f;
constexpr float kBlocksW = 215.0f;   // a range's track (dragged), then its number
constexpr float kNumberX = 228.0f;

void DrawBackdrop(ImDrawList* dl, ImVec2 size)
{
    // The menus' tint over the held frame, their column bands and scanlines.
    VGradient(dl, ImVec2(0, 0), size, { { 0.0f, IM_COL32(18, 10, 38, 235) }, { 0.42f, IM_COL32(22, 30, 96, 209) },
        { 0.50f, IM_COL32(24, 34, 104, 204) }, { 0.75f, IM_COL32(14, 16, 56, 230) }, { 1.0f, IM_COL32(8, 8, 36, 242) } });
    for (const auto [from, to] : { std::pair{ 0.475f, 0.482f }, std::pair{ 0.64f, 0.647f } })
        dl->AddRectFilled(ImVec2(size.x * from, 0), ImVec2(size.x * to, size.y), IM_COL32(0, 0, 0, 46));
    const float pitch = std::max(3.0f, std::round(3.0f * u));
    for (float y = 0; y < size.y; y += pitch)
        dl->AddRectFilled(ImVec2(0, y), ImVec2(size.x, y + std::max(1.0f, std::round(u))), IM_COL32(0, 0, 0, 56));
}

// A gear's outline: `teeth` trapezoid teeth from radius `inner` to `outer`, turned by `phase` (radians), and its hub.
void Gear(ImDrawList* dl, ImVec2 c, float outer, float inner, int teeth, float phase, ImU32 color, float thickness)
{
    constexpr float kPi = 3.14159265f;
    const float half = kPi / float(teeth);
    std::vector<ImVec2> points;
    auto at = [&](float r, float angle) { return ImVec2(c.x + r * std::cos(angle), c.y + r * std::sin(angle)); };
    for (int i = 0; i < teeth; i++)
    {
        const float t = phase + float(i) * 2.0f * half;
        points.push_back(at(inner, t - 0.62f * half));
        points.push_back(at(outer, t - 0.38f * half));
        points.push_back(at(outer, t + 0.38f * half));
        points.push_back(at(inner, t + 0.62f * half));
    }
    dl->AddPolyline(points.data(), int(points.size()), color, ImDrawFlags_Closed, thickness);
    dl->AddCircle(c, inner * 0.42f, color, 0, thickness);
}

// The breadcrumb's level marks: `count` of U+300B (》, the double angle bracket of Japanese text), drawn (Share Tech has no
// CJK glyphs) as two nested chevrons each, at a text `size`. Returns the width used.
float Crumbs(ImDrawList* dl, ImVec2 pos, float size, int count, ImU32 color)
{
    const float w = size * 0.62f, h = size * 0.72f, gap = size * 0.38f, thickness = std::max(1.0f, size * 0.09f);
    const float cy = pos.y + size * 0.55f;
    for (int i = 0; i < count; i++)
    {
        const float x = pos.x + float(i) * (w + gap);
        for (const float inset : { 0.0f, w * 0.42f })
        {
            const ImVec2 points[3] = { ImVec2(x + inset, cy - h / 2), ImVec2(x + inset + w * 0.58f, cy), ImVec2(x + inset, cy + h / 2) };
            dl->AddPolyline(points, 3, color, ImDrawFlags_None, thickness);
        }
    }
    return float(count) * (w + gap);
}

void DrawHeader(ImDrawList* dl)
{
    // Our emblem: two gears (the settings) in a square.
    const ImVec2 a = P(56, 34), b = P(110, 88);
    const ImU32 line = Rgb(0xA7B2D8);
    dl->AddRect(ImVec2(a.x + u, a.y + u), ImVec2(b.x - u, b.y - u), line, 0.0f, 0, 2.5f * u);
    // Meshing (not overlapping): the same tooth size on both (pitch radius 13.25 for 8 teeth, 9.9 for 6), the
    // small one up-right of the large one at 45 degrees, a large gear's tooth there and a small gear's gap facing it, the
    // centres a tooth's tip and a little clearance past the other's root.
    // The pair spans a 43.8-unit square (its middle 6.9 right of and above the large gear's centre): centred in the
    // square, scaled to just fit it (about two units to spare each side).
    constexpr float k = 0.98f;
    const float bx = 83.0f - 6.9f * k, by = 61.0f + 6.9f * k;
    Gear(dl, P(bx, by), 15.0f * k * u, 11.5f * k * u, 8, 0.0f, line, 2.0f * u);
    Gear(dl, P(bx + 17.6f * k, by - 17.6f * k), 11.2f * k * u, 8.6f * k * u, 6, 3.14159265f / 4.0f, line, 2.0f * u);
    // The breadcrumb: 》 SETTINGS, 》》 the tab (U+300B, as the game's Japanese menus mark their levels). At the
    // layout's own size whatever the text size.
    const float size = 15.0f * u;
    for (int level = 1; level <= 2; level++)
    {
        const ImVec2 at = P(124, 40 + 20.0f * float(level - 1));
        const float x = at.x + Crumbs(dl, at, size, level, line) + size * 0.35f;
        Text(dl, s_mono, size, ImVec2(x, at.y), line, level == 1 ? std::string("SETTINGS") : Upper(kTabs[s_tab]), 0.06f);
    }

    // Right: closing (F4, or B / LS + RS).
    const std::string_view close = "CLOSE";
    float x = o.x + 1864.0f * u - TextWidth(s_mono, 15.0f * ut, close, 0.06f);
    const float middle = CapsMiddle(o.y + 44 * u, 15.0f * u);
    Text(dl, s_mono, 15.0f * ut, ImVec2(x, middle - 0.41f * 15.0f * ut), Rgb(0xC9D0EA), close, 0.06f);
    x -= 8.0f * u;
    if (s_gamepad)
    {
        x -= 26.0f * u;
        FaceButton(dl, ImVec2(x, middle - 13.0f * u), 'B');
    }
    else
    {
        x -= KeyCapWidth("F4");
        KeyCap(dl, x, middle, "F4");
    }
}

void DrawTabs(ImDrawList* dl)
{
    const float y1 = 150.0f, y0 = 104.0f;
    dl->AddRectFilled(P(56, y1), P(1864, y1 + 2), Rgb(0x4F63D8));
    float x = 56.0f;
    // Each tab's label is centred in its box (the raised, selected one too). The key caps
    // stay put (moving with the selected tab beside them, they would jump), half way between a raised
    // and an unselected tab's middles: 0.75 units off either. (The label's 1-unit shadow lowers its capitals 1.5 units to
    // the eye.)
    auto labelTop = [&](float top, float bottom) { return (top + bottom) / 2 - 0.41f * 18.0f * ut - 1.5f * u; };
    const float capMiddle = (P(0, y0 - 1.5f).y + P(0, y1).y) / 2;
    x += KeyCap(dl, P(x, 0).x, capMiddle, s_gamepad ? ButtonName("LT") : "Q") / u + 10.0f;
    const float size = 18.0f * ut;
    for (int t = 0; t < kTabCount; t++)
    {
        const std::string label = Upper(kTabs[t]);
        const float w = TextWidth(s_mono, size, label, 0.14f) / u + 56.0f;
        const bool on = t == s_tab;
        const ImVec2 a = P(x, on ? y0 - 3 : y0), b = P(x + w, y1);
        ImGui::SetCursorScreenPos(a);
        ImGui::PushID(t);
        if (ImGui::InvisibleButton("##tab", ImVec2(b.x - a.x, b.y - a.y)))
            SwitchTab(t);
        const bool hover = ImGui::IsItemHovered();
        ImGui::PopID();
        if (on)
            Gloss(dl, a, b, 0x9AAEFF, 0x5266DA, 0x3B4DC4, 0x2D3DAE);
        else
            Gloss(dl, a, b, hover ? 0x3A46A8 : 0x2A3488, hover ? 0x242E80 : 0x1A2266, 0x141B55, 0x10164A);
        dl->AddLine(ImVec2(a.x + u, a.y + u), ImVec2(b.x - u, a.y + u), Rgb(0xC8D7FF, on ? 0.6f : 0.28f), u);
        dl->AddRect(a, ImVec2(b.x, b.y + u), on ? Rgb(0x8FA2FF) : Rgb(0x3D4BA8), 0.0f, 0, Outline());
        Text(dl, s_mono, size, ImVec2(a.x + 28 * u, labelTop(a.y, b.y)), on ? Rgb(0xFFFFFF) : Rgb(0x8EA2F0), label, 0.14f,
            on ? Rgb(0x10164A) : 0);
        x += w + 4.0f;
    }
    KeyCap(dl, P(x + 6.0f, 0).x, capMiddle, s_gamepad ? ButtonName("RT") : "E");
}

// A typed value's field (a slider's number, a text setting) drawn as the text field typing in it is, so the two match
// at every size (at 720p and 150% text a fixed 24-unit box would be shorter than its text): the
// text 16 x the text unit, 4 units above and below it, 8 to its left.
constexpr float kFieldText = 16.0f;
ImVec2 FieldPadding()
{
    return ImVec2(8.0f * u, 4.0f * u);
}

float FieldHeight()
{
    return kFieldText * ut + 2.0f * FieldPadding().y;
}

// The field from x0 to x1 centred on cy, and its text where the text field puts it (smaller, centred, when too long).
void DrawField(ImDrawList* dl, float x0, float x1, float cy, ImFont* font, std::string_view value, bool selected)
{
    const ImVec2 n0(x0, cy - FieldHeight() / 2), n1(x1, cy + FieldHeight() / 2);
    dl->AddRectFilled(n0, n1, Rgb(0x05071E, selected ? 0.8f : 0.5f));
    dl->AddRect(n0, n1, selected ? Rgb(0x6A78C0) : Rgb(0x2E3A86), 0.0f, 0, Outline());
    const ImVec2 pad = FieldPadding();
    const float full = kFieldText * ut, size = Fit(font, full, value, n1.x - n0.x - 2.0f * pad.x);
    Text(dl, font, size, ImVec2(n0.x + pad.x, n0.y + pad.y + (full - size) / 2), Rgb(0xE4E9FA), value);
}

// How many parts a slider's ticks cut its range into: as near ten as can be with each
// part a whole number of the setting's steps and the default on a tick (render resolution: 7, every 0.5x, the idle
// cursor: 10, every 3 s, peak brightness: 12, every 300 nits). When none puts the default on a tick (paper white's
// 203 nits), the nearest ten with whole steps, else ten.
int SliderDivisions(const Edit& e)
{
    const double range = e.entry.max - e.entry.min, step = e.entry.step > 0.0 ? e.entry.step : 0.0;
    const bool hasDefault = !e.def.empty();
    const double offset = std::atof(e.def.c_str()) - e.entry.min;
    auto whole = [](double x) { return std::abs(x - std::round(x)) < 1e-6; };
    constexpr int kOrder[] = { 10, 9, 11, 8, 12, 7, 6, 5, 4 };
    for (int pass = 0; pass < 2; pass++)
        for (int n : kOrder)
        {
            const double part = range / n;
            if (step > 0.0 && !whole(part / step))
                continue;
            if (pass == 0 && (!hasDefault || !whole(offset / part)))
                continue;
            return n;
        }
    return 10;
}

void DrawRow(ImDrawList* dl, int index, ImVec2 a, ImVec2 b, bool selected)
{
    const Row& row = s_rows[size_t(index)];
    const Edit* e = row.edit >= 0 ? &s_edits[size_t(row.edit)] : nullptr;
    const Binding* bind = IsBindingRow(row.edit) ? &BindingAt(row.edit) : nullptr;
    const AdvBinding* adv = IsAdvRow(row.edit) ? &AdvAt(row.edit) : nullptr;
    const TuneGroup* tune = IsTuneRow(row.edit) ? &TuneAt(row.edit) : nullptr;
    const bool unsaved = (e && Unsaved(*e)) || (bind && bind->value != bind->file) || (adv && adv->value != adv->file) ||
        (tune && TuneUnsaved(*tune));
    const float cy = (a.y + b.y) / 2;
    if (selected)
    {
        HGradient(dl, a, ImVec2(a.x + (b.x - a.x) * 0.55f, b.y), Rgb(0xBED2FF, 0.32f), Rgb(0x7896FF, 0.16f));
        HGradient(dl, ImVec2(a.x + (b.x - a.x) * 0.55f, a.y), b, Rgb(0x7896FF, 0.16f), Rgb(0x7896FF, 0.0f));
        dl->AddLine(a, ImVec2(b.x, a.y), Rgb(0xFFFFFF, 0.22f), u);
        dl->AddLine(ImVec2(a.x, b.y - u), ImVec2(b.x, b.y - u), Rgb(0x000000, 0.4f), u);
        Triangle(dl, ImVec2(a.x + 22.5f * u, cy), 9.0f * u, 12.0f * u, true, Rgb(0xDCE5FF));
    }
    const std::string label = e ? e->entry.label : bind ? bind->info->label : adv ? adv->action->label : tune ? tune->label : row.edit == kDlcRow ? "DLC loaded" : row.edit == kDlcFolderRow ? "Open DLC folder" : row.edit == kUpdateRow ? "Updates" : row.edit == kQuitRow ? "Quit to desktop"
        : row.edit == kRenderResRow ? "Render resolution" : row.edit == kTexturesRow ? "Textures on screen"
        : IsModRow(row.edit) ? ModAt(row.edit).name : "Quality preset";
    float x = a.x + 40.0f * u;
    const ImU32 labelColor = selected ? Rgb(0xFFFFFF) : Rgb(0xB7C0E3);
    // ENV, or RESTART on a changed setting that needs one (no LIVE badge).
    const char* badge = !e ? nullptr : e->environment ? "ENV" : NeedsRestart(*e) ? "RESTART" : nullptr;
    const float vx0 = b.x - (22.0f + kValueW) * u, vx1 = b.x - 22.0f * u;
    // The label shrinks when it and its marks would reach the value.
    const float marks = 10.0f * u + (unsaved ? 12.0f * u : 0.0f) + (badge ? BadgeWidth(badge) + 6.0f * u : 0.0f);
    const float size = Fit(s_body, 17.0f * ut, label, vx0 - 12.0f * u - marks - x);
    if (selected)
        Text(dl, s_body, size, ImVec2(x, cy - size / 2 - u), Rgb(0xA0BEFF, 0.25f), label);
    x += Text(dl, s_body, size, ImVec2(x, cy - size / 2 - u), labelColor, label) + 10.0f * u;
    if (unsaved)
    {
        dl->AddCircleFilled(ImVec2(x, cy), 3.5f * u, Rgb(0xF2D27A));
        x += 12.0f * u;
    }
    if (badge)
        x += Badge(dl, ImVec2(x, cy - 8.5f * u), badge, std::string_view(badge) != "RESTART") + 6.0f * u;

    const std::string value = ValueText(row);
    if (row.edit == kQuitRow)
        return;   // an action: no value
    if (tune)
    {
        const float size = Fit(s_mono, 14.0f * ut, value, vx1 - vx0), w = TextWidth(s_mono, size, value);
        Text(dl, s_mono, size, ImVec2((vx0 + vx1) / 2 - w / 2, cy - size / 2 - u), Rgb(0xE4E9FA), value);
        return;
    }
    if (adv)
    {
        // Its inputs as text. Waiting for one: the request, amber, with a live bar of its value under it.
        const bool waiting = s_advCapture == kAdvRow0 - row.edit;
        const std::string shown = !waiting ? value
            : adv->action->kind == input::advanced::Kind::Axis ? std::format("MOVE AN AXIS: {}", AdvDirection(adv->action->name))
                                                                : std::string("PRESS OR MOVE AN INPUT");
        const float size = Fit(s_mono, 15.0f * ut, shown, vx1 - vx0), w = TextWidth(s_mono, size, shown);
        Text(dl, s_mono, size, ImVec2((vx0 + vx1) / 2 - w / 2, cy - size / 2 - 3.0f * u),
            waiting ? Rgb(0xF2D27A) : adv->value.empty() ? Rgb(0x7F8CC0) : Rgb(0xE4E9FA), shown);
        if (!adv->value.empty() && !waiting)
        {
            const float v = input::advanced::Value(adv->action->name, adv->value), y = cy + 11.0f * u;
            const bool axis = adv->action->kind == input::advanced::Kind::Axis;
            dl->AddRectFilled(ImVec2(vx0, y), ImVec2(vx1, y + 3.0f * u), Rgb(0x2A3488));
            const float mid = axis ? (vx0 + vx1) / 2 : vx0, end = mid + (axis ? (vx1 - vx0) / 2 : (vx1 - vx0)) * v;
            dl->AddRectFilled(ImVec2(std::min(mid, end), y), ImVec2(std::max(mid, end), y + 3.0f * u), Rgb(0xF2D27A));
        }
        return;
    }
    if (bind)
    {
        // Its keys as key caps, centred (as text when they don't fit). Waiting for a key: the request, amber.
        if (s_capture == kBindingRow0 - row.edit)
        {
            const std::string_view ask = "KEY OR MOUSE, ESC TO CLEAR";
            const float size = Fit(s_mono, 15.0f * ut, ask, vx1 - vx0), w = TextWidth(s_mono, size, ask);
            Text(dl, s_mono, size, ImVec2((vx0 + vx1) / 2 - w / 2, cy - size / 2 - u), Rgb(0xF2D27A), ask);
            return;
        }
        float total = 0.0f;
        for (const std::string& k : bind->value)
            total += KeyCapWidth(KeyLabel(k)) + 6.0f * u;
        if (bind->value.empty() || total - 6.0f * u > vx1 - vx0)
        {
            const float size = Fit(s_mono, 15.0f * ut, value, vx1 - vx0), w = TextWidth(s_mono, size, value);
            Text(dl, s_mono, size, ImVec2((vx0 + vx1) / 2 - w / 2, cy - size / 2 - u),
                bind->value.empty() ? Rgb(0x7F8CC0) : Rgb(0xE4E9FA), value);
            return;
        }
        float kx = (vx0 + vx1) / 2 - (total - 6.0f * u) / 2;
        for (const std::string& k : bind->value)
            kx += KeyCap(dl, kx, cy, KeyLabel(k)) + 6.0f * u;
        return;
    }
    if (row.edit == kTexturesRow)
    {
        // Read only: the count, centred like a choice's value. Enter browses them (the side panel).
        const float vsize = Fit(s_mono, 16.0f * ut, value, vx1 - vx0), w = TextWidth(s_mono, vsize, value);
        Text(dl, s_mono, vsize, ImVec2((vx0 + vx1) / 2 - w / 2, cy - vsize / 2 - u), Rgb(0xA9B4DC), value);
        return;
    }
    if (row.edit == kRenderResRow)
    {
        // Read only, centred like a choice's value, without arrows, amber when raised to the minimum.
        const bool raised = !RenderSizeNote().empty();
        const float vsize = Fit(s_numbers, 16.0f * ut, value, vx1 - vx0), w = TextWidth(s_numbers, vsize, value);
        Text(dl, s_numbers, vsize, ImVec2((vx0 + vx1) / 2 - w / 2, cy - vsize / 2 - u), raised ? Rgb(0xF2D27A) : Rgb(0xA9B4DC), value);
        return;
    }
    if (row.edit == kDlcRow || row.edit == kDlcFolderRow || row.edit == kUpdateRow || IsModRow(row.edit))
    {
        // Read only: the count, centred like a choice's value, without arrows, red when a package didn't load or is
        // damaged. Open DLC folder: nothing, unless it didn't open. Updates: its state, red when it failed. A mod: its
        // state, red when it failed.
        bool problem = false;
        if (row.edit == kDlcRow)
            DlcCount(&problem);
        else if (row.edit == kUpdateRow)
            UpdateText(&problem);
        else if (IsModRow(row.edit))
            ModText(ModAt(row.edit), &problem);
        else
            problem = s_dlcFolderFailed;
        const float vsize = Fit(s_mono, 16.0f * ut, value, vx1 - vx0), w = TextWidth(s_mono, vsize, value);
        Text(dl, s_mono, vsize, ImVec2((vx0 + vx1) / 2 - w / 2, cy - vsize / 2 - u), problem ? Rgb(0xFF5A4E) : Rgb(0xA9B4DC), value);
        return;
    }
    if (e && IsHudColour(*e))
    {
        // The colour itself, and its name or number. Enter or a click opens the picker.
        const uint32_t rgb = PaletteColour(RowPalette(), e->entry.key);
        const float size = Fit(s_numbers, 16.0f * ut, value, vx1 - vx0 - 60.0f * u), w = TextWidth(s_numbers, size, value);
        const float total = 34.0f * u + 12.0f * u + w, sx = (vx0 + vx1) / 2 - total / 2;
        dl->AddRectFilled(ImVec2(sx, cy - 10.0f * u), ImVec2(sx + 34.0f * u, cy + 10.0f * u), Rgb(rgb));
        dl->AddRect(ImVec2(sx, cy - 10.0f * u), ImVec2(sx + 34.0f * u, cy + 10.0f * u), Rgb(0xDCE5FF, selected ? 1.0f : 0.5f), 0.0f, 0, Outline());
        Text(dl, s_numbers, size, ImVec2(sx + 46.0f * u, cy - size / 2 - u), Rgb(0xE4E9FA), value);
        return;
    }
    float vsize = 16.0f * ut;
    const bool editing = e && s_editing == row.edit;
    if (e && IsRange(*e))
    {
        // A thin track, filled to the value, with a bar for a handle and ticks under it, one on the default
        // (SliderDivisions).
        const double v = std::atof(Effective(*e).c_str());
        const float f = float(std::clamp((v - e->entry.min) / (e->entry.max - e->entry.min), 0.0, 1.0));
        const float t0 = vx0 + 2.0f * u, t1 = vx0 + kBlocksW * u - 2.0f * u, hx = t0 + f * (t1 - t0);
        const int divisions = SliderDivisions(*e);
        for (int i = 0; i <= divisions; i++)
        {
            const float tx = std::round(t0 + (t1 - t0) * float(i) / float(divisions));
            dl->AddRectFilled(ImVec2(tx, cy + 6.0f * u), ImVec2(tx + std::max(1.0f, u), cy + 10.0f * u),
                tx <= hx ? Rgb(0x7E95F0, 0.8f) : Rgb(0x3D4BA8, 0.8f));
        }
        dl->AddRectFilled(ImVec2(t0, cy - 1.5f * u), ImVec2(t1, cy + 1.5f * u), Rgb(0x2E3A86));
        HGradient(dl, ImVec2(t0, cy - 1.5f * u), ImVec2(hx, cy + 1.5f * u), Rgb(0x6A80E0), Rgb(0xD8E2FF));
        dl->AddRectFilled(ImVec2(hx - 2.5f * u, cy - 10.0f * u), ImVec2(hx + 2.5f * u, cy + 10.0f * u),
            selected ? Rgb(0xFFFFFF) : Rgb(0xC8D4FF));
        if (editing)
            return;   // the number field is drawn over the number
        // The number, in a field-like box: clicking it (or A / Enter) types a value.
        DrawField(dl, vx0 + kNumberX * u, vx1, cy, s_numbers, value, selected);
        return;
    }
    if (editing)
        return;   // the text field is drawn over it
    if (e && IsText(*e))
    {
        // In a field-like box as a slider's number: Enter / A or a click types in it.
        DrawField(dl, vx0, vx1, cy, s_mono, value, selected);
        return;
    }
    // The arrows: dim at the ends (left / right don't wrap).
    const auto [at, count] = ChoicePosition(row);
    // Under the value, a tick for each choice, the chosen one lit and taller (after Cyberpunk 2077's
    // dashes). The value and arrows a little higher for them. Ticks closer together when there
    // are many (the resolutions), none when they'd still not fit.
    float pitch = 6.0f * u;
    if (count > 1 && pitch * float(count - 1) > 200.0f * u)
        pitch = 200.0f * u / float(count - 1);
    const bool ticks = count > 1 && pitch >= 3.0f * u;
    const float vy = ticks ? cy - 3.0f * u * ut / (kTextBase * u) : cy;   // higher with larger text
    Triangle(dl, ImVec2(vx0 + 6.0f * u, vy), 8.0f * u, 10.0f * u, false, Rgb(0x7F92E0, at == 0 ? 0.25f : 1.0f));
    Triangle(dl, ImVec2(vx1 - 6.0f * u, vy), 8.0f * u, 10.0f * u, true, Rgb(0x7F92E0, at == count - 1 ? 0.25f : 1.0f));
    vsize = Fit(s_mono, vsize, value, vx1 - vx0 - 40.0f * u);
    const float w = TextWidth(s_mono, vsize, value);
    Text(dl, s_mono, vsize, ImVec2((vx0 + vx1) / 2 - w / 2, vy - vsize / 2 - u), Rgb(0xE4E9FA), value);
    if (ticks)
    {
        const float tw = std::max(1.0f, std::round(2.0f * u));
        float tx = std::round((vx0 + vx1) / 2 - pitch * float(count - 1) / 2 - tw / 2);
        for (int i = 0; i < count; i++, tx += pitch)
        {
            const float x0 = std::round(tx);
            if (i == at)
                dl->AddRectFilled(ImVec2(x0, cy + 9.0f * u), ImVec2(x0 + tw, cy + 17.0f * u), Rgb(0xA9E6FF));
            else
                dl->AddRectFilled(ImVec2(x0, cy + 11.0f * u), ImVec2(x0 + tw, cy + 15.0f * u), Rgb(0x5A6BC8, 0.85f));
        }
    }
}

void DrawList(ImDrawList* dl)
{
    const ImVec2 a = P(kListX, kTop), b = P(kListX + kListW, kBottom);
    const std::string tab = Upper(kTabs[s_tab]);
    Panel(dl, a, b, tab);   // (no preset name at the right on the Graphics tab: the preset row says it)

    ImGui::SetCursorScreenPos(ImVec2(a.x + u, a.y + 35.0f * u));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
    ImGui::BeginChild("##list", ImVec2(b.x - a.x - 2.0f * u, b.y - a.y - 36.0f * u), 0,
        ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoNavInputs | ImGuiWindowFlags_NoNavFocus);
    ImDrawList* cl = ImGui::GetWindowDrawList();
    const ImGuiIO& io = ImGui::GetIO();
    const bool mouseMoved = io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f;
    const float width = ImGui::GetContentRegionAvail().x;
    const char* group = nullptr;
    for (int i = 0; i < int(s_rows.size()); i++)
    {
        const Row& row = s_rows[size_t(i)];
        if (!group || std::string_view(group) != row.group)
        {
            group = row.group;
            const ImVec2 g = ImGui::GetCursorScreenPos();
            ImGui::Dummy(ImVec2(width, 38.0f * u));
            const float size = 13.0f * ut;
            const float tw = Text(cl, s_mono, size, ImVec2(g.x + 22.0f * u, TextTop(g.y, 24.5f, 18)), Rgb(0x8EA2F0), group, 0.22f);
            const float ly = g.y + 25.5f * u;
            HGradient(cl, ImVec2(g.x + 34.0f * u + tw, ly), ImVec2(g.x + width - 22.0f * u, ly + u), Rgb(0x4A5AB0), Rgb(0x4A5AB0, 0.0f));
        }
        ImGui::PushID(i);
        const ImVec2 r0 = ImGui::GetCursorScreenPos();
        const bool clicked = ImGui::InvisibleButton("##row", ImVec2(width, kRowH * u));
        const ImVec2 r1(r0.x + width, r0.y + kRowH * u);
        const float blocks0 = r1.x - (22.0f + kValueW) * u;
        Edit* const rangeEdit = row.edit >= 0 && IsRange(s_edits[size_t(row.edit)]) ? &s_edits[size_t(row.edit)] : nullptr;
        if (rangeEdit && !Disabled(*rangeEdit) && ImGui::IsItemActivated() && io.MousePos.x >= blocks0 && io.MousePos.x < blocks0 + kBlocksW * u && s_editing < 0)
            s_dragRow = i;
        if (s_dragRow == i && rangeEdit)
        {
            if (ImGui::IsItemActive())
            {
                const double f = std::clamp((io.MousePos.x - blocks0) / (kBlocksW * u), 0.0f, 1.0f);
                const double step = rangeEdit->entry.step > 0.0 ? rangeEdit->entry.step : 0.01;
                const double v = std::round((rangeEdit->entry.min + f * (rangeEdit->entry.max - rangeEdit->entry.min)) / step) * step;
                SetValue(*rangeEdit, FormatNumber(std::clamp(v, rangeEdit->entry.min, rangeEdit->entry.max), step));
                s_cursor = i;
                s_footer = -1;
            }
            else
                s_dragRow = -1;
        }
        if (ImGui::IsItemHovered() && mouseMoved && s_editing < 0 && s_capture < 0 && s_advCapture < 0)
        {
            s_cursor = i;
            s_footer = -1;
        }
        if (clicked && s_editing < 0 && s_capture < 0 && s_advCapture < 0)
        {
            s_cursor = i;
            s_footer = -1;
            const float vx0 = r1.x - (22.0f + kValueW) * u;
            if (rangeEdit)
            {
                // The track is dragged (above), the number is typed.
                if (io.MousePos.x >= vx0 + kNumberX * u && !Disabled(*rangeEdit))
                    StartEditing(row.edit);
            }
            else if (row.edit != kQuitRow && row.edit != kDlcFolderRow && row.edit != kUpdateRow && row.edit != kTexturesRow && !IsBindingRow(row.edit) &&
                !IsAdvRow(row.edit) && !IsTuneRow(row.edit) &&
                !(row.edit >= 0 && IsHudColour(s_edits[size_t(row.edit)])) && io.MousePos.x >= vx0 &&
                io.MousePos.x < vx0 + 40.0f * u)
                Step(row, -1);
            else
                Activate(row);
        }
        if (s_scrollToCursor && i == s_cursor && s_footer < 0)
        {
            if (i == 0)
                ImGui::SetScrollY(0.0f);
            else
                ImGui::ScrollToItem(ImGuiScrollFlags_KeepVisibleEdgeY);
            s_scrollToCursor = false;
        }
        DrawRow(cl, i, r0, r1, i == s_cursor && s_footer < 0);
        if ((row.edit >= 0 && Disabled(s_edits[size_t(row.edit)])) || (IsBindingRow(row.edit) && BindingsDisabled()) ||
            ((IsAdvRow(row.edit) || (IsTuneRow(row.edit) && TuneAt(row.edit).action)) && AdvancedDisabled()))
        {
            // Greyed out: a gradient as the selection's, the other way: lightest at the
            // label, darkest over the value.
            const float gx0 = r0.x, gm = gx0 + (r1.x - gx0) * 0.45f;   // from the row's left end
            // The shade's opacity: 43% at the left end, 63% at 45% of the row, 86% at the right end.
            HGradient(cl, ImVec2(gx0, r0.y), ImVec2(gm, r1.y), Rgb(0x090B2A, 0.43f), Rgb(0x090B2A, 0.63f));
            HGradient(cl, ImVec2(gm, r0.y), r1, Rgb(0x090B2A, 0.63f), Rgb(0x090B2A, 0.86f));
        }
        if (row.edit >= 0 && s_editing == row.edit)
        {
            const bool number = IsRange(s_edits[size_t(row.edit)]);
            const float fx = r1.x - (22.0f + kValueW) * u + (number ? kNumberX * u : 0.0f);
            ImGui::SetNextItemWidth((number ? kValueW - kNumberX : kValueW) * u);
            // As the field drawn when not typing (DrawField): its font, size, padding and place.
            ImGui::PushFont(number ? s_numbers : s_mono, kFieldText * ut);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, FieldPadding());
            ImGui::SetCursorScreenPos(ImVec2(fx, (r0.y + r1.y - ImGui::GetFrameHeight()) / 2));
            if (s_editStart)
            {
                ImGui::SetKeyboardFocusHere();
                s_editStart = false;
            }
            // The profile name: letters, digits and spaces, 15 at most.
            const bool profile = IsProfileName(s_edits[size_t(row.edit)]);
            ImGui::InputText("##text", s_text, profile ? settings::kProfileNameMax + 1 : sizeof(s_text),
                ImGuiInputTextFlags_AutoSelectAll | (number ? ImGuiInputTextFlags_CharsScientific : 0) |
                    (profile ? ImGuiInputTextFlags_CallbackCharFilter : 0),
                profile ? ProfileNameFilter : nullptr);
            if (ImGui::IsItemDeactivated() || s_endEditing)
            {
                s_endEditing = false;
                Edit& e = s_edits[size_t(row.edit)];
                if (IsRange(e))
                {
                    char* end = nullptr;
                    const double x = std::strtod(s_text, &end);
                    while (end && *end == ' ')
                        end++;
                    if (end != s_text && end && *end == 0 && std::isfinite(x))
                    {
                        const double clamped = std::clamp(x, e.entry.min, e.entry.max);
                        SetValue(e, FormatNumber(clamped, 0.0001));
                        if (clamped != x)
                            Status(std::format("{} is kept between {} and {}.", e.entry.label, FormatNumber(e.entry.min, 0.0001),
                                FormatNumber(e.entry.max, 0.0001)), true);
                    }
                    else if (*s_text)
                        Status(std::format("\"{}\" isn't a number.", s_text), true);
                }
                else if (profile)
                    SetValue(e, settings::CleanProfileName(s_text));   // the ends' spaces trimmed, none left: the default
                else
                    SetValue(e, s_text);
                s_editing = -1;
            }
            ImGui::PopStyleVar();
            ImGui::PopFont();
            ImGui::SetCursorScreenPos(ImVec2(r0.x, r1.y));
            ImGui::Dummy(ImVec2(0, 0));
        }
        ImGui::PopID();
    }
    ImGui::Dummy(ImVec2(width, 12.0f * u));
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
}

// The texture browser in the side panel: the last frame's textures as thumbnails, drawn by their
// descriptors over a checkerboard (their alpha shows), the chosen one's hash, size and format, and whether it is
// replaced or dumped.
void DrawTextureBrowser(ImDrawList* dl, ImVec2 a, ImVec2 b)
{
    const auto& list = BrowserTextures();
    Panel(dl, a, b, s_browsing ? "TEXTURES (ENTER: DUMP, ESC: BACK)" : "TEXTURES ON SCREEN");
    // REFRESH in the title bar at its right, a prompt like the footer's (key cap R / the Y button), its middle on the bar's.
    // Also R / Y while browsing, a click on it.
    {
        const ImGuiIO& io = ImGui::GetIO();
        const float w = Prompt(nullptr, ImVec2(), 'Y', "R", "REFRESH");
        const ImVec2 at(b.x - 18.0f * u - w, a.y + 17.0f * u - 4.0f * u - 0.41f * 15.0f * u);
        const ImVec2 r0(at.x - 6.0f * u, a.y + 2.0f * u), r1(b.x - 12.0f * u, a.y + 32.0f * u);
        const bool hover = io.MousePos.x >= r0.x && io.MousePos.x < r1.x && io.MousePos.y >= r0.y && io.MousePos.y < r1.y;
        if (hover)
            dl->AddRectFilled(r0, r1, Rgb(0xFFFFFF, 0.12f));
        Prompt(dl, at, 'Y', "R", "REFRESH");
        if (hover && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        {
            RefreshTextures();
            Status(std::format("Textures refreshed: {} in the last frame.", BrowserTextures().size()));
        }
    }
    dl->PushClipRect(a, b, true);
    const float pad = 22.0f * u, cell = 112.0f * u, gap = 10.0f * u;
    float y = a.y + 52.0f * u;
    if (list.empty())
    {
        Paragraph(dl, ImVec2(a.x + pad, y), b.x - a.x - 2 * pad, 18.0f * ut, Rgb(0xC9D0EA),
            "No textures yet: they are listed from the last frame the game drew\n"
            "Open the settings during a mission or a menu");
        dl->PopClipRect();
        return;
    }
    s_texCursor = std::clamp(s_texCursor, 0, int(list.size()) - 1);
    const gpu::mods::FrameTexture& sel = list[size_t(s_texCursor)];
    const std::string info = std::format("{:016X}  ·  {} x {}  ·  {}  ·  {} level{}{}{}", sel.hash, sel.width, sel.height,
        GuestFormatName(sel.format), sel.levels, sel.levels == 1 ? "" : "s", sel.replaced ? "  ·  REPLACED" : "",
        gpu::mods::Dumped(sel.hash) ? "  ·  DUMPED" : "");
    Text(dl, s_numbers, 15.0f * ut, ImVec2(a.x + pad, y), Rgb(0xEEF2FF), info);
    y += 26.0f * u;
    const std::string help = s_browsing
        ? std::format("Enter / {}: dump it to dump/textures/<hash>.dds next to the game\n"
                      "A replacement goes in mods/<mod>/textures/<hash>.dds (any resolution, BC1-5 or 8-bit RGBA)", ButtonName("A"))
        : std::format("Enter / {}: browse the {} textures of the frame the settings opened on (R / {}: refresh)\n"
                      "Dump textures (above) saves every texture the game uses", ButtonName("A"), list.size(), ButtonName("Y"));
    y += Paragraph(dl, ImVec2(a.x + pad, y), b.x - a.x - 2 * pad, 14.0f * ut, Rgb(0x9FB0E6), help) + 12.0f * u;
    // The grid: as many columns as fit (a scroll bar's width kept at the right), scrolled by the mouse wheel over it, or to keep
    // the chosen one in view when the keys move it.
    const float barW = 8.0f * u;
    s_texColumns = std::max(1, int((b.x - a.x - 2 * pad - barW - gap + gap) / (cell + gap)));
    const int rowsVisible = std::max(1, int((b.y - y - pad + gap) / (cell + gap)));
    const int rowsAll = (int(list.size()) + s_texColumns - 1) / s_texColumns;
    const ImGuiIO& io = ImGui::GetIO();
    if (io.MousePos.x >= a.x && io.MousePos.x < b.x && io.MousePos.y >= y && io.MousePos.y < b.y && io.MouseWheel != 0.0f)
    {
        s_texScroll -= int(io.MouseWheel > 0.0f ? std::ceil(io.MouseWheel) : std::floor(io.MouseWheel));
        s_texFollow = false;
    }
    if (s_texFollow)
    {
        const int selRow = s_texCursor / s_texColumns;
        if (selRow < s_texScroll)
            s_texScroll = selRow;
        if (selRow >= s_texScroll + rowsVisible)
            s_texScroll = selRow - rowsVisible + 1;
    }
    s_texScroll = std::clamp(s_texScroll, 0, std::max(0, rowsAll - rowsVisible));
    if (rowsAll > rowsVisible)
    {
        // The scroll bar: where the rows shown are among all.
        const float bx = b.x - pad - barW, by0 = y, by1 = y + rowsVisible * (cell + gap) - gap;
        dl->AddRectFilled(ImVec2(bx, by0), ImVec2(bx + barW, by1), Rgb(0x2A2E46));
        const float t0 = by0 + (by1 - by0) * float(s_texScroll) / float(rowsAll);
        const float t1 = by0 + (by1 - by0) * float(s_texScroll + rowsVisible) / float(rowsAll);
        dl->AddRectFilled(ImVec2(bx, t0), ImVec2(bx + barW, t1), Rgb(0x8E9AD0));
    }
    for (int i = s_texScroll * s_texColumns; i < int(list.size()) && i < (s_texScroll + rowsVisible) * s_texColumns; i++)
    {
        const int r = i / s_texColumns - s_texScroll, c = i % s_texColumns;
        const ImVec2 c0(a.x + pad + c * (cell + gap), y + r * (cell + gap)), c1(c0.x + cell, c0.y + cell);
        // A checkerboard under it.
        const float q = cell / 8;
        for (int cy = 0; cy < 8; cy++)
            for (int cx = 0; cx < 8; cx++)
                dl->AddRectFilled(ImVec2(c0.x + cx * q, c0.y + cy * q), ImVec2(c0.x + (cx + 1) * q, c0.y + (cy + 1) * q),
                    (cx + cy) & 1 ? Rgb(0x3A3F5C) : Rgb(0x2A2E46));
        // The texture, its aspect kept.
        const gpu::mods::FrameTexture& t = list[size_t(i)];
        const float aspect = float(t.width) / float(std::max(1u, t.height));
        const float w = aspect >= 1.0f ? cell : cell * aspect, h = aspect >= 1.0f ? cell / aspect : cell;
        const ImVec2 i0(c0.x + (cell - w) / 2, c0.y + (cell - h) / 2);
        dl->AddImage(ImTextureRef(ImTextureID(gpu::mods::kGameTextureFlag | (t.descriptor & 0xFFFFFFu))), i0, ImVec2(i0.x + w, i0.y + h));
        const bool chosen = i == s_texCursor;
        dl->AddRect(c0, c1, chosen ? Rgb(0xDCE5FF) : t.replaced ? Rgb(0x6FD38A) : Rgb(0x4A5AB0), 0.0f, 0, chosen ? 3.0f * u : Outline());
        if (io.MousePos.x >= c0.x && io.MousePos.x < c1.x && io.MousePos.y >= c0.y && io.MousePos.y < c1.y)
        {
            if (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f)
                s_texCursor = i;
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
            {
                s_texCursor = i;
                if (s_browsing)
                {
                    gpu::mods::RequestDump(t);
                    Status(std::format("Texture {:016X}: dumped to dump/textures/{:016X}.dds next to the game.", t.hash, t.hash));
                }
                s_browsing = true;
            }
        }
    }
    dl->PopClipRect();
}

void DrawSide(ImDrawList* dl)
{
    const float x0 = kListX + kListW + 24.0f, x1 = 1864.0f;
    if (s_footer < 0 && !s_rows.empty() && s_rows[size_t(s_cursor)].edit == kTexturesRow)
    {
        DrawTextureBrowser(dl, P(x0, kTop), P(x1, kBottom));
        return;
    }
    s_browsing = false;
    // The selected setting: what it does, and why it can't be changed now when it can't: the whole column.
    const ImVec2 ha = P(x0, kTop), hb = P(x1, kBottom);
    std::string title = "SETTINGS", text, note;
    if (s_footer >= 0)
    {
        title = s_footer == kRevert ? "REVERT" : s_footer == kApply ? "APPLY" : "RESTART NOW";
        text = s_footer == kRevert ? "Undoes any unapplied changes"
            : s_footer == kApply   ? "Saves the changes to ac6_settings.toml"
                                   : "Applies the changes, then restarts the game\n"
                                     "Unsaved progress in the game is lost";
    }
    else if (!s_rows.empty())
    {
        const Row& row = s_rows[size_t(s_cursor)];
        if (row.edit == kDlcFolderRow)
        {
            title = "OPEN DLC FOLDER";
            text = std::format("{} the folder the game reads DLC from: {}\n"
                               "Put your extracted DLC files in the folder\n"
                               "The game loads DLC at its next start",
                s_dlcFolderFailed ? "Couldn't open" : "Opens", DlcFolderText());
        }
        else if (row.edit == kUpdateRow)
        {
            const update::Status st = update::Get();
            title = "UPDATES";
            switch (st.state)
            {
            case update::State::Off:
                text = std::format("This is version {}, built from source\n"
                                   "Use a release version from GitHub for auto update", update::Version());
                break;
            case update::State::Idle:
            case update::State::Checking:
                text = std::format("This is version {}\n"
                                   "Enter / {}, or a click to check for a new release on GitHub", update::Version(),
                    ButtonText("A"));
                break;
            case update::State::UpToDate:
                text = std::format("Version {} is the newest release\n"
                                   "Enter / {}, or a click to check for a new release on GitHub", update::Version(),
                    ButtonText("A"));
                break;
            case update::State::Available:
                text = std::format("Version {} is out (this is {})\n"
                                   "Enter / {}, or a click to update to the latest version", st.latest, update::Version(), ButtonText("A"));
                break;
            case update::State::Downloading:
            case update::State::Installing:
                text = std::format("Downloading and installing version {}", st.latest);
                break;
            case update::State::Installed:
                text = std::format("Version {} is installed\n"
                                   "Enter / {}, or a click to restart the game now\n"
                                   "Unsaved progress in the game is lost", st.latest, ButtonText("A"));
                break;
            case update::State::Failed:
                text = std::format("The update failed: {}\n"
                                   "Nothing was changed\n"
                                   "Enter / {}, or a click to try again", st.error,
                    ButtonText("A"));
                break;
            }
        }
        else if (row.edit == kQuitRow)
        {
            title = "QUIT TO DESKTOP";
            text = "Closes the game\n"
                   "Progress since the game last saved is lost";
            if (AnyUnsaved())
                text += "\n"
                        "Unapplied settings are lost too";
        }
        else if (row.edit == kRenderResRow)
        {
            title = "RENDER RESOLUTION";
            text = "The resolution the 3D scene renders at: the resolution times the render scale (with FSR 3.1, set by FSR quality instead)";
            note = RenderSizeNote();
        }
        else if (row.edit == kDlcRow)
        {
            title = "DLC LOADED";
            const std::string count = DlcCount();
            if (count == "OFF" || count == "SCANNING")
                text = count == "OFF" ? "DLC is off (Use DLC)" : "The DLC folder is still being checked";
            else
            {
                // How many loaded, and the ones that didn't with why (dlc_report.txt, next to the log, has the details).
                const auto& packages = fs::DlcPackages();
                std::string failed, damaged;
                size_t usable = 0, listed = 0;
                for (const fs::DlcPackage& p : packages)
                {
                    if (p.Usable())
                    {
                        usable++;
                        if (!p.damage.empty())
                            damaged += std::format(" {}: {}.", !p.label.empty() ? p.label : !p.displayName.empty() ? p.displayName : p.name,
                                p.damage.front());
                        continue;
                    }
                    if (listed++ < 4)
                        failed += std::format(" {}: {}.", !p.label.empty() ? p.label : !p.displayName.empty() ? p.displayName : p.name,
                            p.errors.empty() ? "unusable" : p.errors.front());
                }
                if (packages.empty())
                    text = "No DLC was found in the dlc folder";
                else if (usable == packages.size())
                    text = std::format("{} DLC package{} found in the dlc folder, all loaded", packages.size(),
                        packages.size() == 1 ? "" : "s");
                else
                    text = std::format("{} of the {} DLC packages in the dlc folder next to the game loaded. Not loaded:{}{}", usable,
                        packages.size(), failed, listed > 4 ? " ..." : "");
                if (!damaged.empty())
                    text += " Loaded though damaged (what's damaged may look wrong, the rest works):" + damaged;
                text += " Details: dlc_report.txt, next to the log.";
            }
        }
        else if (IsModRow(row.edit))
        {
            // What it is, who made it, what it has in place, and why it's off.
            const mods::ModStatus& m = ModAt(row.edit);
            title = Upper(m.name);
            if (m.plugin)
                text = std::format("A plugin: native code the game started from mods/{}", m.id);
            else
            {
                text = !m.description.empty() ? m.description : std::format("A Lua mod, in mods/{}", m.id);
                if (!m.version.empty() || !m.author.empty())
                    text += std::format("\n{}{}{}", m.version.empty() ? "" : "Version " + m.version,
                        !m.version.empty() && !m.author.empty() ? ", " : "", m.author.empty() ? "" : "by " + m.author);
                if (m.native)
                    text += "\nIt runs native code (a DLL it loaded)";
            }
            if (m.loaded)
                text += std::format("\n{} hook{} on the game's functions", m.hooks, m.hooks == 1 ? "" : "s");
            text += "\nMods run with the game's full access: use only mods you trust";
            // (The error's first line: a script's has its traceback after it, in ac6_runtime.log.)
            std::string error = m.error.substr(0, m.error.find('\n'));
            if (error.size() > 160)
                error = error.substr(0, 157) + "...";
            if (!error.empty())
                note = (m.loaded ? "LAST ERROR: " : "") + Upper(error);
        }
        else if (IsTuneRow(row.edit) && !TuneAt(row.edit).action)
        {
            title = Upper(TuneAt(row.edit).label);
            text = std::format("Enter / {}, or a click to open the HDR grading window", ButtonName("A"));
        }
        else if (IsTuneRow(row.edit))
        {
            const TuneGroup& g = TuneAt(row.edit);
            title = Upper(g.label);
            text = std::format("Enter / {}, or a click to open the tuning window\n"
                               "Dead zone, full deflection and curve{}, with a graph of the response and the live input{}",
                ButtonName("A"), g.axis ? ", and invert" : ", and the lever's detent and its width",
                std::string_view(g.group) == "THROTTLE" ? "\nThe game's throttle isn't linear: a third of the way holds about 625 km/h, "
                                                          "full goes past 1300 (cruise 600)\nThe curve can even it out" : "");
        }
        else if (IsAdvRow(row.edit))
        {
            const AdvBinding& b = AdvAt(row.edit);
            title = Upper(b.action->label);
            const bool axis = b.action->kind == input::advanced::Kind::Axis;
            text = std::format("{}\n"
                               "Enter / {}, or a click, then {} to add it, up to {}\n"
                               "Using one it already has removes it\n"
                               "Esc or Del / {} clears all\n"
                               "Without bindings, the game's own control is used{}",
                b.action->help, ButtonName("A"),
                axis ? std::format("move an axis to {}", Upper(AdvDirection(b.action->name)))
                     : std::string("press a button or hat, or move an axis or trigger"),
                kMaxAdvInputs, ButtonName("X"),
                b.value.empty() ? "" : std::format("\nNow: {:.0f}%", input::advanced::Value(b.action->name, b.value) * 100.0f));
            if (const char* why = AdvancedDisabled())
                note = why;
            else if (const std::string clash = AdvConflicts(b); !clash.empty())
                note = "ALSO ON " + clash;
        }
        else if (IsBindingRow(row.edit))
        {
            const Binding& b = BindingAt(row.edit);
            title = Upper(b.info->label);
            text = std::format("{}, controller: {}\n"
                               "Enter / {}, or a click, then press a key, mouse button or wheel notch to add it, up to {}\n"
                               "Pressing one it already has removes it, Esc clears all its keys\n"
                               "Del / {} restores the default\n"
                               "Default: {}",
                b.action.section == "menu" ? "In menus" : "In flight", ButtonText(b.info->pad), ButtonName("A"), kMaxBindingKeys,
                ButtonName("X"), BindingText(b.action.defaults));
            if (const char* why = BindingsDisabled())
                note = why;
            else if (const std::string clash = BindingConflicts(b); !clash.empty())
                note = "ALSO ON " + clash;
        }
        else if (row.edit == kPresetRow)
        {
            const int p = CurrentPreset();
            title = "QUALITY PRESET";
            text = p < 0 ? "The Graphics settings below don't match a preset\n"
                           "Left / right picks one: Console, Performance, Enhanced"
                         : kPresets[p].help;
        }
        else
        {
            const Edit& e = s_edits[size_t(row.edit)];
            title = Upper(e.entry.label);
            text = e.entry.help;
            if (Disabled(e))
                note = Disabled(e);
            else if (e.environment)
                note = std::format("SET BY {} IN THE ENVIRONMENT", e.entry.env);
            else if (const std::string_view key = e.entry.key;
                     (key == "render_resolution" || key == "aa" || key == "fsr_quality" || key == "resolution") &&
                     !RenderSizeNote().empty())
                note = RenderSizeNote();   // the render size raised to the minimum
            else if (const std::string_view key = e.entry.key; key == "hdr_paper_white" || key == "hdr_peak" || key == "hdr_tonemap")
            {
                // Paper white at or above the peak (the peak as the output pass gets it, in 50-nit steps): the
                // highlights aren't raised and there's no tone mapping, the SDR picture at paper white.
                const Edit* paper = Find("hdr_paper_white");
                const Edit* peak = Find("hdr_peak");
                if (paper && peak)
                {
                    const double paperNits = std::atof(Effective(*paper).c_str());
                    const double peakNits = std::round(std::atof(Effective(*peak).c_str()) / 50.0) * 50.0;
                    if (peakNits <= paperNits)
                        note = std::format("PAPER WHITE ({}) IS AT OR ABOVE THE PEAK ({}): HIGHLIGHTS AREN'T RAISED AND "
                                           "THERE'S NO TONE MAPPING", Effective(*paper), Effective(*peak));
                }
            }
        }
    }
    Panel(dl, ha, hb, title);
    dl->PushClipRect(ha, hb, true);
    float y = ha.y + 52.0f * u;
    if (!text.empty())
        y += Paragraph(dl, ImVec2(ha.x + 22.0f * u, y), hb.x - ha.x - 44.0f * u, 18.0f * ut, Rgb(0xC9D0EA), text) + 10.0f * u;
    if (!note.empty())
        Paragraph(dl, ImVec2(ha.x + 22.0f * u, y), hb.x - ha.x - 44.0f * u, 15.0f * ut, Rgb(0xF2D27A), note);
    dl->PopClipRect();
}

float ButtonWidth(std::string_view label)
{
    return TextWidth(s_mono, 16.0f * ut, label, 0.12f) + 52.0f * u;
}

bool Button(ImDrawList* dl, int id, ImVec2 a, std::string_view label, bool primary, bool focused, bool glow)
{
    const float size = 16.0f * ut;
    const ImVec2 b(a.x + ButtonWidth(label), a.y + 42.0f * u);
    ImGui::SetCursorScreenPos(a);
    ImGui::PushID(id);
    const bool clicked = ImGui::InvisibleButton("##button", ImVec2(b.x - a.x, b.y - a.y));
    const bool hover = ImGui::IsItemHovered();
    ImGui::PopID();
    dl->AddRectFilled(ImVec2(a.x, a.y + 2 * u), ImVec2(b.x, b.y + 4 * u), Rgb(0x000000, 0.5f));
    if (primary)
        Gloss(dl, a, b, 0xC2D0FF, 0x7189F2, 0x526BDF, 0x3F57CA);
    else if (glow)
        Gloss(dl, a, b, 0xF5E2A0, 0xC9A23A, 0xA8841E, 0x8A6A12);
    else
        Gloss(dl, a, b, 0x56649F, 0x2C3672, 0x222B62, 0x1B2252);
    dl->AddLine(ImVec2(a.x + u, a.y + u), ImVec2(b.x - u, a.y + u), Rgb(0xFFFFFF, 0.35f), u);
    dl->AddRect(a, b, primary ? Rgb(0xB4C2FF) : glow ? Rgb(0xF2D27A) : Rgb(0x6A78C0), 0.0f, 0, Outline());
    if (focused || hover)
        dl->AddRect(ImVec2(a.x - 3 * u, a.y - 3 * u), ImVec2(b.x + 3 * u, b.y + 3 * u), Rgb(0xDCE5FF, focused ? 1.0f : 0.5f), 0.0f, 0, 2.0f * u);
    const bool dark = primary || glow;
    Text(dl, s_mono, size, ImVec2(a.x + 26.0f * u, TextTop(a.y, 21, 12)), dark ? Rgb(0x0B0F35) : Rgb(0xE8ECFA), label, 0.12f,
        dark ? Rgb(0xFFFFFF, 0.4f) : Rgb(0x000000));
    return clicked;
}

// The footer as the game's menus have it: the buttons (Revert, Apply, Restart now) at the
// left, the prompts at the right, the status between them.
void DrawFooter(ImDrawList* dl)
{
    const float y = 1004.0f * u + o.y;
    const char* labels[kFooterCount] = { "REVERT", "APPLY", "RESTART NOW" };
    float bx = 56.0f * u + o.x;
    const bool restart = AnyRestart();
    const int count = FooterCount();
    if (s_footer >= count)
        s_footer = count - 1;   // Restart now gone (its change undone)
    for (int i = 0; i < count; i++)
    {
        if (Button(dl, 100 + i, ImVec2(bx, y - 8.0f * u), labels[i], i == kApply && AnyUnsaved(), s_footer == i, i == kRestart && restart))
            FooterPress(i);
        bx += ButtonWidth(labels[i]) + 12.0f * u;
    }

    // Right-aligned prompts.
    struct Item { char button; const char* key; const char* action; };
    constexpr Item kPrompts[] = { { 'A', "ENTER", "CHANGE" }, { 'B', "ESC", "CLOSE" }, { 'X', "DEL", "DEFAULT" }, { 'Y', "CTRL+S", "APPLY" } };
    float total = 0.0f;
    for (const Item& p : kPrompts)
        total += Prompt(nullptr, {}, p.button, p.key, p.action) + 26.0f * u;
    const float px = 1864.0f * u + o.x - total + 26.0f * u;
    float x = px;
    for (const Item& p : kPrompts)
        x += Prompt(dl, ImVec2(x, y), p.button, p.key, p.action) + 26.0f * u;

    // The status between the buttons and the prompts (cut short there).
    if (!s_status.empty())
    {
        dl->PushClipRect(ImVec2(bx, y - 20.0f * u), ImVec2(px - 16.0f * u, y + 40.0f * u), true);
        Text(dl, s_mono, 14.0f * ut, ImVec2(bx + 12.0f * u, CapsMiddle(y + 4.0f * u, 15.0f * u) - 0.41f * 14.0f * ut),
            s_statusWarn ? Rgb(0xF2D27A) : Rgb(0xA9E6FF), s_status, 0.04f);
        dl->PopClipRect();
    }
}

// Quit to desktop's and Restart now's confirmation: the screen dimmed, a panel in the middle, Confirm and Cancel.
void DrawConfirm(ImDrawList* dl, ImVec2 size)
{
    dl->AddRectFilled(ImVec2(0, 0), size, Rgb(0x04051A, 0.72f));
    const bool restart = s_confirm == kConfirmRestart;
    std::string text;
    if (restart)
        text = "Restart the game now? Your changes are applied first, then the game starts again with them. Progress "
               "since the game last saved is lost.";
    else
    {
        text = "Quit the game? Progress since the game last saved is lost.";
        if (AnyUnsaved())
            text += " Settings changes you haven't applied are lost too.";
    }
    // As tall as its text needs (larger text sizes wrap it more), at least 270 units, in the middle of the screen.
    const float width = 600.0f * u, textH = Paragraph(nullptr, {}, width - 44.0f * u, 18.0f * ut, 0, text);
    const float h = std::max(270.0f * u, 52.0f * u + textH + 24.0f * u + 64.0f * u);
    const ImVec2 a(o.x + 660.0f * u, o.y + 515.0f * u - h / 2), b(a.x + width, a.y + h);
    dl->AddRectFilled(a, b, Rgb(0x0B1140));   // opaque: the list mustn't show through
    Panel(dl, a, b, restart ? "RESTART NOW" : "QUIT TO DESKTOP");
    Paragraph(dl, ImVec2(a.x + 22.0f * u, a.y + 52.0f * u), width - 44.0f * u, 18.0f * ut, Rgb(0xC9D0EA), text);
    const char* labels[] = { "CONFIRM", "CANCEL" };
    float bx = b.x - 22.0f * u - ButtonWidth(labels[0]) - 12.0f * u - ButtonWidth(labels[1]);
    for (int i = 0; i < 2; i++)
    {
        if (Button(dl, 200 + i, ImVec2(bx, b.y - 64.0f * u), labels[i], false, s_confirmFocus == i, i == 0))
        {
            ConfirmPress(i);
            return;
        }
        bx += ButtonWidth(labels[i]) + 12.0f * u;
    }
}

bool TuneUnsaved(const TuneGroup& g)
{
    for (const Row& r : TuneRows(g))
        if (Unsaved(s_edits[size_t(r.edit)]))
            return true;
    return false;
}

// Advanced controls' fine tuning window: the group's settings at the left, the response curve at the right (the
// input across, the output up, the straight line faint, the dead zone shaded, the input now as a dot).
void DrawTuning(ImDrawList* dl, ImVec2 size)
{
    const TuneGroup& g = kTuneGroups[size_t(s_tuning)];
    const std::vector<Row> rows = TuneRows(g);
    dl->AddRectFilled(ImVec2(0, 0), size, Rgb(0x04051A, 0.72f));
    const float width = 980.0f * u, height = g.action ? 560.0f * u : (60.0f + 52.0f * float(rows.size()) + 170.0f) * u;
    const ImVec2 a(o.x + 960.0f * u - width / 2, o.y + 540.0f * u - height / 2), b(a.x + width, a.y + height);
    dl->AddRectFilled(a, b, Rgb(0x0B1140));
    Panel(dl, a, b, g.label);
    const ImGuiIO& io = ImGui::GetIO();

    // The settings.
    const float lx = a.x + 24.0f * u, lw = g.action ? 470.0f * u : width - 48.0f * u, rowH = 52.0f * u;
    float y = a.y + 60.0f * u;
    for (int i = 0; i < int(rows.size()); i++)
    {
        const Row& row = rows[size_t(i)];
        const Edit& e = s_edits[size_t(row.edit)];
        const ImVec2 r0(lx, y), r1(lx + lw, y + rowH - 6.0f * u);
        ImGui::SetCursorScreenPos(r0);
        ImGui::PushID(400 + i);
        const bool clicked = ImGui::InvisibleButton("##tune", ImVec2(r1.x - r0.x, r1.y - r0.y));
        ImGui::PopID();
        const bool selected = i == s_tuneCursor;
        const float vx0 = r1.x - 200.0f * u, cy = (r0.y + r1.y) / 2;
        if (clicked)
        {
            s_tuneCursor = i;
            if (io.MousePos.x >= vx0 && io.MousePos.x < vx0 + 40.0f * u)
                Step(row, -1);
            else if (io.MousePos.x >= r1.x - 40.0f * u)
                Step(row, 1);
        }
        if (selected)
            HGradient(dl, r0, r1, Rgb(0xBED2FF, 0.30f), Rgb(0x7896FF, 0.05f));
        dl->AddRect(r0, r1, Rgb(0x3D4BA8), 0.0f, 0, Outline());
        Text(dl, s_body, 17.0f * ut, ImVec2(r0.x + 14.0f * u, cy - 9.0f * ut - u), selected ? Rgb(0xFFFFFF) : Rgb(0xB7C0E3), e.entry.label);
        std::string v = ValueText(row);
        if (IsRange(e) && g.action)   // (the controls' percentages, not the HDR group's factors)
            v += "%";
        const float vs = 17.0f * ut, vw = TextWidth(s_numbers, vs, v);
        Text(dl, s_numbers, vs, ImVec2((vx0 + r1.x) / 2 - vw / 2, cy - vs / 2 - u), Unsaved(e) ? Rgb(0xF2D27A) : Rgb(0xE4E9FA), v);
        Triangle(dl, ImVec2(vx0 + 14.0f * u, cy), 8.0f * u, 10.0f * u, false, Rgb(0x8EA2F0));
        Triangle(dl, ImVec2(r1.x - 14.0f * u, cy), 8.0f * u, 10.0f * u, true, Rgb(0x8EA2F0));
        y += rowH;
    }
    if (s_tuneCursor >= 0 && s_tuneCursor < int(rows.size()))
    {
        const Edit& e = s_edits[size_t(rows[size_t(s_tuneCursor)].edit)];
        const char* help = e.entry.help;
        // (Why it doesn't apply now, as the list shows it.)
        if (const char* why = Disabled(e); why && !g.action)
        {
            Text(dl, s_mono, 14.0f * ut, ImVec2(lx, y + 8.0f * u), Rgb(0xF2D27A), why);
            y += 24.0f * u;
        }
        // The help, wrapped to the list's width.
        std::string line, text = help;
        if (!text.empty())
            text[0] = char(std::toupper(uint8_t(text[0])));
        float hy = y + 8.0f * u;
        const float hs = 15.0f * ut;
        std::string word;
        auto flush = [&] { Text(dl, s_body, hs, ImVec2(lx, hy), Rgb(0x8E9AD0), line); hy += hs * 1.35f; line.clear(); };
        for (size_t i = 0; i <= text.size(); i++)
        {
            if (i == text.size() || text[i] == ' ')
            {
                const std::string next = line.empty() ? word : line + " " + word;
                if (TextWidth(s_body, hs, next) > lw && !line.empty())
                {
                    flush();
                    line = word;
                }
                else
                    line = next;
                word.clear();
            }
            else
                word += text[i];
        }
        if (!line.empty())
            flush();
    }

    if (!g.action)
    {
        // (No curve: the HDR group.)
        Text(dl, s_mono, 14.0f * ut, ImVec2(lx, b.y - 52.0f * u), Rgb(0x8E9AD0),
            s_gamepad ? ButtonText("D-PAD: CHOOSE AND CHANGE   X: DEFAULT   A / B: DONE")
                      : "ARROWS: CHOOSE AND CHANGE   DEL: DEFAULT   ENTER / ESC: DONE");
        const char* doneLabel = "DONE";
        if (Button(dl, 230, ImVec2(b.x - 22.0f * u - ButtonWidth(doneLabel), b.y - 60.0f * u), doneLabel, true, false, false))
            s_tuning = -1;
        return;
    }
    // The curve.
    const float side = 400.0f * u;
    const ImVec2 g0(b.x - 24.0f * u - side, a.y + 60.0f * u), g1(g0.x + side, g0.y + side);
    dl->AddRectFilled(g0, g1, Rgb(0x060A2C));
    dl->AddRect(g0, g1, Rgb(0x6A78C0), 0.0f, 0, Outline());
    auto at = [&](float x, float yv) {
        // x, y: -1 to 1 for an axis, 0 to 1 otherwise
        const float fx = g.axis ? (x + 1.0f) / 2.0f : x, fy = g.axis ? (yv + 1.0f) / 2.0f : yv;
        return ImVec2(g0.x + fx * side, g1.y - fy * side);
    };
    for (int k = 1; k < 4; k++)
    {
        const float t = float(k) / 4.0f;
        dl->AddLine(ImVec2(g0.x + t * side, g0.y), ImVec2(g0.x + t * side, g1.y), Rgb(0x2A3488, k == 2 && g.axis ? 1.0f : 0.5f), u);
        dl->AddLine(ImVec2(g0.x, g0.y + t * side), ImVec2(g1.x, g0.y + t * side), Rgb(0x2A3488, k == 2 && g.axis ? 1.0f : 0.5f), u);
    }
    const std::string prefix = std::string(g.keys.begin()[0]).substr(0, std::string(g.keys.begin()[0]).find('_'));
    const float dz = TuneNumber(prefix + "_deadzone") / 100.0f;
    if (dz > 0.0f)
        dl->AddRectFilled(at(g.axis ? -dz : 0.0f, g.axis ? -1.0f : 0.0f), at(dz, 1.0f), Rgb(0x7896FF, 0.10f));
    dl->AddLine(at(g.axis ? -1.0f : 0.0f, g.axis ? -1.0f : 0.0f), at(1.0f, 1.0f), Rgb(0x8EA2F0, 0.30f), u);
    const bool invert = g.axis && TuneNumber(prefix + "_invert") > 0.5f;
    std::vector<ImVec2> points;
    constexpr int kSteps = 96;
    for (int k = 0; k <= kSteps; k++)
    {
        const float x = g.axis ? -1.0f + 2.0f * float(k) / kSteps : float(k) / kSteps;
        float yv = std::copysign(TuneShape(g, std::fabs(x)), x);
        if (invert)
            yv = -yv;
        points.push_back(at(x, yv));
    }
    dl->AddPolyline(points.data(), int(points.size()), Rgb(0xF2D27A), 0, 2.5f * u);
    // The input now: the group's axis (or, for the throttle, thrust, else the lever).
    std::string action = g.action;
    const AdvBinding* bound = nullptr;
    for (const AdvBinding& ab : s_adv)
        if (action == ab.action->name)
            bound = &ab;
    if (std::string_view(g.group) == "THROTTLE" && (!bound || bound->value.empty()))
        for (const AdvBinding& ab : s_adv)
            if (std::string_view(ab.action->name) == "throttle_lever")
                bound = &ab;
    std::string note = "NOTHING BOUND TO SHOW";
    if (bound && !bound->value.empty())
    {
        const float raw = input::advanced::RawValue(bound->action->name, bound->value);
        float out = std::copysign(TuneShape(g, std::fabs(raw)), raw);
        if (invert)
            out = -out;
        const ImVec2 dot = at(g.axis ? raw : std::fabs(raw), g.axis ? out : std::fabs(out));
        dl->AddCircleFilled(dot, 6.0f * u, Rgb(0xFFFFFF));
        dl->AddCircle(dot, 6.0f * u, Rgb(0x000000), 16, 1.5f * u);
        note = std::format("{}: IN {:.0f}%  OUT {:.0f}%", Upper(bound->action->label), raw * 100.0f, out * 100.0f);
    }
    Text(dl, s_mono, 14.0f * ut, ImVec2(g0.x, g1.y + 10.0f * u), Rgb(0xB7C0E3), note);
    Text(dl, s_mono, 13.0f * ut, ImVec2(g0.x, g1.y + 34.0f * u), Rgb(0x7F8CC0), "INPUT ACROSS, OUTPUT UP");

    Text(dl, s_mono, 14.0f * ut, ImVec2(lx, b.y - 52.0f * u), Rgb(0x8E9AD0),
        s_gamepad ? ButtonText("D-PAD: CHOOSE AND CHANGE   X: DEFAULT   A / B: DONE")
                  : "ARROWS: CHOOSE AND CHANGE   DEL: DEFAULT   ENTER / ESC: DONE");
    const char* label = "DONE";
    if (Button(dl, 230, ImVec2(b.x - 22.0f * u - ButtonWidth(label), b.y - 60.0f * u), label, true, false, false))
        s_tuning = -1;
}

// The HUD colour's picker: a saturation and value square, a hue bar, the colour, its number and four starting points.
void DrawPicker(ImDrawList* dl, ImVec2 size)
{
    dl->AddRectFilled(ImVec2(0, 0), size, Rgb(0x04051A, 0.72f));
    const float width = 760.0f * u, height = 520.0f * u;
    const ImVec2 a(o.x + 960.0f * u - width / 2, o.y + 540.0f * u - height / 2), b(a.x + width, a.y + height);
    dl->AddRectFilled(a, b, Rgb(0x0B1140));
    const bool isHud = s_pickerKey == "hud_colour", isFriendly = s_pickerKey == "hud_friendly";
    Panel(dl, a, b, isHud ? "HUD COLOUR" : isFriendly ? "FRIENDLY COLOUR" : "ENEMY/ALERT COLOUR");
    const ImGuiIO& io = ImGui::GetIO();
    const float side = 340.0f * u;
    const ImVec2 sq(a.x + 24.0f * u, a.y + 56.0f * u), sqEnd(sq.x + side, sq.y + side);
    const ImVec2 hue(sqEnd.x + 18.0f * u, sq.y), hueEnd(hue.x + 34.0f * u, sqEnd.y);

    ImGui::SetCursorScreenPos(sq);
    ImGui::InvisibleButton("##sv", ImVec2(side, side));
    if (ImGui::IsItemActive())
    {
        s_pickerS = std::clamp((io.MousePos.x - sq.x) / side, 0.0f, 1.0f);
        s_pickerV = 1.0f - std::clamp((io.MousePos.y - sq.y) / side, 0.0f, 1.0f);
        s_pickerAuto = false;
    }
    ImGui::SetCursorScreenPos(hue);
    ImGui::InvisibleButton("##hue", ImVec2(hueEnd.x - hue.x, side));
    if (ImGui::IsItemActive())
    {
        s_pickerH = std::clamp((io.MousePos.y - hue.y) / side, 0.0f, 0.9999f);
        s_pickerAuto = false;
    }

    float r, g, bl;
    ImGui::ColorConvertHSVtoRGB(s_pickerH, 1.0f, 1.0f, r, g, bl);
    const ImU32 pure = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, bl, 1.0f));
    dl->AddRectFilledMultiColor(sq, sqEnd, Rgb(0xFFFFFF), pure, pure, Rgb(0xFFFFFF));
    dl->AddRectFilledMultiColor(sq, sqEnd, Rgb(0x000000, 0.0f), Rgb(0x000000, 0.0f), Rgb(0x000000), Rgb(0x000000));
    dl->AddRect(sq, sqEnd, Rgb(0x6A78C0), 0.0f, 0, Outline());
    for (int i = 0; i < 6; i++)
    {
        float r0, g0, b0, r1, g1, b1;
        ImGui::ColorConvertHSVtoRGB(float(i) / 6.0f, 1.0f, 1.0f, r0, g0, b0);
        ImGui::ColorConvertHSVtoRGB(float(i + 1) / 6.0f, 1.0f, 1.0f, r1, g1, b1);
        const ImU32 top = ImGui::ColorConvertFloat4ToU32(ImVec4(r0, g0, b0, 1.0f)), bottom = ImGui::ColorConvertFloat4ToU32(ImVec4(r1, g1, b1, 1.0f));
        dl->AddRectFilledMultiColor(ImVec2(hue.x, hue.y + side * float(i) / 6.0f), ImVec2(hueEnd.x, hue.y + side * float(i + 1) / 6.0f), top, top, bottom, bottom);
    }
    dl->AddRect(hue, hueEnd, Rgb(0x6A78C0), 0.0f, 0, Outline());
    const ImVec2 mark(sq.x + s_pickerS * side, sq.y + (1.0f - s_pickerV) * side);
    dl->AddCircle(mark, 8.0f * u, Rgb(0x000000), 24, 3.0f * u);
    dl->AddCircle(mark, 8.0f * u, Rgb(0xFFFFFF), 24, 1.5f * u);
    const float hy = hue.y + s_pickerH * side;
    dl->AddRectFilled(ImVec2(hue.x - 4.0f * u, hy - 3.0f * u), ImVec2(hueEnd.x + 4.0f * u, hy + 3.0f * u), Rgb(0x000000));
    dl->AddRectFilled(ImVec2(hue.x - 3.0f * u, hy - 1.5f * u), ImVec2(hueEnd.x + 3.0f * u, hy + 1.5f * u), Rgb(0xFFFFFF));

    // The colour, its number, the starting points.
    const uint32_t rgb = PickerColour();
    const float rx = hueEnd.x + 24.0f * u, rw = b.x - 24.0f * u - rx;
    dl->AddRectFilled(ImVec2(rx, sq.y), ImVec2(rx + rw, sq.y + 110.0f * u), Rgb(rgb));
    dl->AddRect(ImVec2(rx, sq.y), ImVec2(rx + rw, sq.y + 110.0f * u), Rgb(0x6A78C0), 0.0f, 0, Outline());
    const std::string number = !s_pickerAuto ? std::format("#{:06X}", rgb) : isHud ? "DEFAULT" : std::format("AUTO #{:06X}", rgb);
    const float size1 = Fit(s_numbers, 20.0f * ut, number, rw);
    Text(dl, s_numbers, size1, ImVec2(rx, sq.y + 124.0f * u), Rgb(0xE4E9FA), number);
    using Start = std::pair<uint32_t, const char*>;
    constexpr Start kHudStarts[] = { { game::kGameHud, "GREEN" }, { 0xFFB000, "AMBER" }, { 0x00E5FF, "CYAN" }, { 0xF2F2F2, "WHITE" } };
    constexpr Start kFriendlyStarts[] = { { game::kGameFriendly, "BLUE" }, { 0x00E0C0, "TEAL" }, { 0xF2F2F2, "WHITE" }, { 0xE040FF, "VIOLET" } };
    constexpr Start kEnemyStarts[] = { { game::kGameEnemy, "RED" }, { 0xFF00C0, "MAGENTA" }, { 0xFFD000, "YELLOW" }, { 0xFF7A00, "ORANGE" } };
    const Start* starts = isHud ? kHudStarts : isFriendly ? kFriendlyStarts : kEnemyStarts;
    float cy = sq.y + 170.0f * u;
    for (int k = 0; k < 4; k++)
    {
        const auto& [colour, name] = starts[k];
        const ImVec2 c0(rx, cy), c1(rx + rw, cy + 34.0f * u);
        ImGui::SetCursorScreenPos(c0);
        ImGui::PushID(name);
        if (ImGui::InvisibleButton("##start", ImVec2(rw, 34.0f * u)))
        {
            SetPicker(colour);
            s_pickerAuto = false;
        }
        const bool hover = ImGui::IsItemHovered();
        ImGui::PopID();
        dl->AddRectFilled(ImVec2(c0.x, c0.y + 4.0f * u), ImVec2(c0.x + 34.0f * u, c1.y - 4.0f * u), Rgb(colour));
        Text(dl, s_mono, 15.0f * ut, ImVec2(c0.x + 44.0f * u, c0.y + 8.0f * u), hover ? Rgb(0xFFFFFF) : Rgb(0xB7C0E3), name);
        cy += 40.0f * u;
    }

    const std::string reset = isHud ? "DEFAULT" : "AUTO";   // (the game's green: "DEFAULT")
    Text(dl, s_mono, 14.0f * ut, ImVec2(sq.x, sqEnd.y + 12.0f * u), Rgb(0x8E9AD0),
        s_gamepad ? ButtonText("D-PAD: SATURATION AND BRIGHTNESS   LB / RB: HUE   X: " + reset)
                  : "ARROWS: SATURATION AND BRIGHTNESS   Q / E: HUE   DEL: " + reset);

    // All three as they'd be, and a warning when two of them are hard to tell apart.
    const game::HudPalette palette = RowPalette();
    const std::pair<uint32_t, const char*> trio[] = { { palette.hud, "HUD" }, { palette.friendly, "FRIENDLY" }, { palette.enemy, "ENEMY/ALERT" } };
    float tx = sq.x;
    const float ty = b.y - 54.0f * u;
    for (const auto& [colour, name] : trio)
    {
        dl->AddRectFilled(ImVec2(tx, ty), ImVec2(tx + 22.0f * u, ty + 22.0f * u), Rgb(colour));
        dl->AddRect(ImVec2(tx, ty), ImVec2(tx + 22.0f * u, ty + 22.0f * u), Rgb(0x6A78C0), 0.0f, 0, Outline());
        Text(dl, s_mono, 14.0f * ut, ImVec2(tx + 30.0f * u, ty + 2.0f * u), Rgb(0xB7C0E3), name);
        tx += 30.0f * u + TextWidth(s_mono, 14.0f * ut, name) + 22.0f * u;
    }
    if (game::HudColoursClose(palette.hud, palette.friendly) || game::HudColoursClose(palette.hud, palette.enemy) ||
        game::HudColoursClose(palette.friendly, palette.enemy))
        Text(dl, s_mono, 14.0f * ut, ImVec2(sq.x, ty + 28.0f * u), Rgb(0xF2D27A), "TWO OF THESE ARE HARD TO TELL APART");
    const char* labels[] = { "OK", "CANCEL" };
    float bx = b.x - 22.0f * u - ButtonWidth(labels[0]) - 12.0f * u - ButtonWidth(labels[1]);
    for (int i = 0; i < 2; i++)
    {
        if (Button(dl, 210 + i, ImVec2(bx, b.y - 60.0f * u), labels[i], i == 0, false, false))
        {
            if (i == 0)
                PickerAccept();
            else
                s_picker = false;
            return;
        }
        bx += ButtonWidth(labels[i]) + 12.0f * u;
    }
}

// The settings text size (overlay_text_size, applied): 1, 1.25 or 1.5 (the variable takes 0.75 to 1.5:
// larger would break the layout),
// or Auto (unset): the text at least as large, in the system's scaled pixels, as on a 1080p screen at 100%: 1080 over the
// canvas's height divided by the display's scale, 1 to 1.5. A 1440p or 4K desktop gets 1, a 1080p laptop at 125% 1.25, a
// Steam Deck (800 lines, the canvas 720) 1.5.
float TextSize()
{
    const std::string v = settings::Get("AC6_UI_TEXT_SIZE");
    if (!v.empty() && v != "auto")
    {
        const float x = float(std::atof(v.c_str()));
        if (x > 0.0f)
            return std::clamp(x, 0.75f, 1.5f);
    }
    SDL_Window* w = Window();
    const float scale = w ? SDL_GetWindowDisplayScale(w) : 0.0f;
    return std::clamp(1080.0f * (scale > 0.0f ? scale : 1.0f) / (1080.0f * u), 1.0f, 1.5f);
}

void DrawUi()
{
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    u = std::min(vp->Size.x / 1920.0f, vp->Size.y / 1080.0f);
    o = ImVec2(vp->Pos.x + (vp->Size.x - 1920.0f * u) / 2, vp->Pos.y + (vp->Size.y - 1080.0f * u) / 2);
    ut = u * kTextBase * TextSize();
    constexpr ImGuiWindowFlags kFullScreen = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoNavInputs | ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoScrollWithMouse;
    ImGui::SetNextWindowPos(vp->Pos);
    ImGui::SetNextWindowSize(vp->Size);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin("##settings", nullptr, kFullScreen);
    ImGui::PopStyleVar();
    ImGui::PushFont(s_body, 17.0f * ut);
    BuildRows();
    HandleKeys();
    PushLive();
    if (!s_open.load())
    {
        ImGui::PopFont();
        ImGui::End();
        return;
    }
    BuildRows();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    DrawBackdrop(dl, vp->Size);
    DrawHeader(dl);
    // Under a confirmation, the rest takes no clicks (disabled items aren't hovered or pressed).
    const bool confirming = s_confirm != kNoConfirm || s_picker || s_tuning >= 0;
    ImGui::BeginDisabled(confirming);
    DrawTabs(dl);
    DrawList(dl);
    DrawSide(dl);
    DrawFooter(dl);
    ImGui::EndDisabled();
    // In a window of their own: the list is a child window above this one, and would take the mouse wherever they overlap.
    if (confirming)
    {
        ImGui::SetNextWindowPos(vp->Pos);
        ImGui::SetNextWindowSize(vp->Size);
        ImGui::Begin("##modal", nullptr, kFullScreen);
        if (s_picker)
            DrawPicker(ImGui::GetForegroundDrawList(), vp->Size);
        else if (s_tuning >= 0)
            DrawTuning(ImGui::GetForegroundDrawList(), vp->Size);
        else
            DrawConfirm(ImGui::GetForegroundDrawList(), vp->Size);
        ImGui::End();
    }
    ImGui::PopFont();
    ImGui::End();
}

void ApplyStyle()
{
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding = 0.0f;
    s.WindowBorderSize = 0.0f;
    s.ChildBorderSize = 0.0f;
    s.FrameRounding = 0.0f;
    s.FrameBorderSize = 1.5f;
    s.FramePadding = ImVec2(8.0f, 4.0f);
    s.ScrollbarSize = 10.0f;
    s.ScrollbarRounding = 0.0f;
    ImVec4* c = s.Colors;
    auto rgb = [](uint32_t hex, float a = 1.0f) { return ImVec4(((hex >> 16) & 255) / 255.0f, ((hex >> 8) & 255) / 255.0f, (hex & 255) / 255.0f, a); };
    c[ImGuiCol_Text] = rgb(0xE4E9FA);
    c[ImGuiCol_FrameBg] = rgb(0x05071E);
    c[ImGuiCol_FrameBgHovered] = rgb(0x1A2266);
    c[ImGuiCol_FrameBgActive] = rgb(0x10164A);
    c[ImGuiCol_Border] = rgb(0x8FA2FF);
    c[ImGuiCol_TextSelectedBg] = rgb(0x5266DA, 0.6f);
    c[ImGuiCol_ScrollbarBg] = rgb(0x000000, 0.0f);
    c[ImGuiCol_ScrollbarGrab] = rgb(0x4A5AB0);
    c[ImGuiCol_ScrollbarGrabHovered] = rgb(0x6A78C0);
    c[ImGuiCol_ScrollbarGrabActive] = rgb(0x8FA2FF);
}

// The draw data and texture requests, for the renderer.
void Publish(ImDrawData* data)
{
    std::lock_guard lock(s_mutex);
    if (data->Textures)
        for (ImTextureData* tex : *data->Textures)
        {
            if (tex->Status == ImTextureStatus_WantCreate || tex->Status == ImTextureStatus_WantUpdates)
            {
                // The whole texture each time (the atlas changes rarely: new glyphs).
                TextureUpdate t;
                t.id = tex->Status == ImTextureStatus_WantCreate ? s_nextTexture++ : uint32_t(tex->GetTexID());
                t.width = uint32_t(tex->Width);
                t.height = uint32_t(tex->Height);
                const uint8_t* pixels = static_cast<const uint8_t*>(tex->GetPixels());
                t.rgba.assign(pixels, pixels + size_t(tex->Width) * tex->Height * 4);
                s_pendingTextures.push_back(std::move(t));
                tex->SetTexID(ImTextureID(s_pendingTextures.back().id));
                tex->SetStatus(ImTextureStatus_OK);
            }
            else if (tex->Status == ImTextureStatus_WantDestroy && tex->UnusedFrames > 0)
            {
                TextureUpdate t;
                t.id = uint32_t(tex->GetTexID());
                t.destroy = true;
                s_pendingTextures.push_back(std::move(t));
                tex->SetTexID(ImTextureID_Invalid);
                tex->SetStatus(ImTextureStatus_Destroyed);
            }
        }
    DrawFrame& f = s_frame;
    f.width = data->DisplaySize.x;
    f.height = data->DisplaySize.y;
    f.vertices.clear();
    f.indices.clear();
    f.commands.clear();
    for (const ImDrawList* list : data->CmdLists)
    {
        const uint32_t vertexBase = uint32_t(f.vertices.size()), indexBase = uint32_t(f.indices.size());
        static_assert(sizeof(ImDrawVert) == sizeof(DrawVertex) && sizeof(ImDrawIdx) == 2);
        f.vertices.resize(vertexBase + size_t(list->VtxBuffer.Size));
        std::memcpy(f.vertices.data() + vertexBase, list->VtxBuffer.Data, size_t(list->VtxBuffer.Size) * sizeof(ImDrawVert));
        f.indices.insert(f.indices.end(), list->IdxBuffer.Data, list->IdxBuffer.Data + list->IdxBuffer.Size);
        for (const ImDrawCmd& cmd : list->CmdBuffer)
        {
            if (cmd.UserCallback || cmd.ElemCount == 0)
                continue;
            f.commands.push_back({ { cmd.ClipRect.x - data->DisplayPos.x, cmd.ClipRect.y - data->DisplayPos.y, cmd.ClipRect.z - data->DisplayPos.x,
                                       cmd.ClipRect.w - data->DisplayPos.y },
                uint32_t(cmd.GetTexID()), indexBase + cmd.IdxOffset, cmd.ElemCount, vertexBase + cmd.VtxOffset });
        }
    }
    s_frameReady = true;
}

ImFont* LoadFont(const char* name, float size)
{
    size_t bytes = 0;
    const unsigned char* data = EmbeddedFileData(name, &bytes);
    if (!data)
        return nullptr;
    ImFontConfig config;
    config.FontDataOwnedByAtlas = false;   // (the embedded copy)
    std::snprintf(config.Name, sizeof(config.Name), "%s", name);
    return ImGui::GetIO().Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(data), int(bytes), size, &config);
}

// Share Tech has only Latin letters, but the Language setting names Japanese in its own script (日本語) and a language
// pack by its own name (Русский, 한국어, 简体中文...). The letters Share Tech lacks come from the system's fonts (Windows:
// Segoe UI for Cyrillic, Greek and accented Latin first, as the Japanese fonts' Cyrillic is full width, then a Japanese,
// a Korean and a Chinese font, nothing is embedded or shipped), merged into Share Tech in that order. Without a Japanese
// font the value reads JAPANESE.
void MergeSystemFonts()
{
    const char* windows = std::getenv("WINDIR");
    if (!windows)
        return;
    const std::initializer_list<const char*> groups[] = {
        { "segoeui.ttf" },
        { "YuGothM.ttc", "meiryo.ttc", "msgothic.ttc" },
        { "malgun.ttf", "gulim.ttc" },
        { "msyh.ttc", "simsun.ttc" },
    };
    for (const std::initializer_list<const char*>& group : groups)
        for (const char* file : group)
        {
            const std::filesystem::path path = std::filesystem::path(windows) / "Fonts" / file;
            std::error_code ec;
            if (!std::filesystem::exists(path, ec))
                continue;
            ImFontConfig config;
            config.MergeMode = true;
            if (ImGui::GetIO().Fonts->AddFontFromFileTTF(platform::PathUtf8(path).c_str(), 18.0f, &config))
                break;
        }
}

// The FPS counter (the Debug tab's setting, AC6_FPS_COUNTER, live): the presented frames (the swap chain's count) over
// half-second windows, and their mean frame time, in the top right corner.
bool FpsCounterOn()
{
    const std::string v = settings::Get("AC6_FPS_COUNTER");
    return v == "1" || v == "true";
}

// How to open the overlay, at the start: shown from the first frame on screen for kHintSeconds, fading out over the last
// one, centred at the bottom, and gone for good once the overlay has been opened.
constexpr double kHintSeconds = 10.0;

float HintAlpha()
{
    using Clock = std::chrono::steady_clock;
    static Clock::time_point start{};
    if (s_hintDone)
        return 0.0f;
    // Counted from the first frame shown (the window stays black while the game loads).
    if (gpu::g_video.frameCount == 0)
        return 1.0f;
    if (start == Clock::time_point{})
        start = Clock::now();
    const double t = std::chrono::duration<double>(Clock::now() - start).count();
    if (t >= kHintSeconds)
    {
        s_hintDone = true;
        return 0.0f;
    }
    return float(std::min(1.0, kHintSeconds - t));
}

void DrawStartupHint(ImDrawList* dl, ImVec2 pos, ImVec2 size, float alpha)
{
    const float scale = std::min(size.x / 1920.0f, size.y / 1080.0f), textSize = 24.0f * scale, pad = 10.0f * scale;
    const std::string text = std::format("PRESS F4 OR {} TO OPEN THE SETTINGS", ButtonText("LS + RS"));
    const float w = TextWidth(s_body, textSize, text, 0.06f), h = textSize + 2.0f * pad;
    // Centred at the bottom.
    const ImVec2 a(pos.x + (size.x - w) * 0.5f - pad, pos.y + size.y - 24.0f * scale - h), b(a.x + w + 2.0f * pad, a.y + h);
    dl->AddRectFilled(a, b, Rgb(0x04051A, 0.7f * alpha));
    Text(dl, s_body, textSize, ImVec2(a.x + pad, a.y + pad), Rgb(0xE4E9FA, alpha), text, 0.06f);
}

// A new release found at the start: said once, after the start-up hint, for kHintSeconds, until the overlay is opened.

float UpdateHintAlpha()
{
    using Clock = std::chrono::steady_clock;
    static Clock::time_point start{};
    if (s_updateHintDone || !s_hintDone || update::Get().state != update::State::Available)
        return 0.0f;
    if (start == Clock::time_point{})
        start = Clock::now();
    const double t = std::chrono::duration<double>(Clock::now() - start).count();
    if (t >= kHintSeconds)
    {
        s_updateHintDone = true;
        return 0.0f;
    }
    return float(std::min(1.0, kHintSeconds - t));
}

void DrawUpdateHint(ImDrawList* dl, ImVec2 pos, ImVec2 size, float alpha)
{
    const float scale = std::min(size.x / 1920.0f, size.y / 1080.0f), textSize = 24.0f * scale, pad = 10.0f * scale;
    const std::string text = std::format("VERSION {} IS OUT: PRESS F4 OR {} TO UPDATE", update::Get().latest, ButtonText("LS + RS"));
    const float w = TextWidth(s_body, textSize, text, 0.06f), h = textSize + 2.0f * pad;
    const ImVec2 a(pos.x + (size.x - w) * 0.5f - pad, pos.y + size.y - 24.0f * scale - h), b(a.x + w + 2.0f * pad, a.y + h);
    dl->AddRectFilled(a, b, Rgb(0x04051A, 0.7f * alpha));
    Text(dl, s_body, textSize, ImVec2(a.x + pad, a.y + pad), Rgb(0xE4E9FA, alpha), text, 0.06f);
}

void DrawFpsCounter(ImDrawList* dl, ImVec2 pos, ImVec2 size)
{
    using Clock = std::chrono::steady_clock;
    static Clock::time_point since = Clock::now();
    static uint64_t frames = gpu::g_video.frameCount;
    static float fps = 0.0f, ms = 0.0f;
    const Clock::time_point now = Clock::now();
    const double seconds = std::chrono::duration<double>(now - since).count();
    if (seconds >= 0.5)
    {
        const uint64_t count = gpu::g_video.frameCount;
        const uint64_t n = count >= frames ? count - frames : 0;
        fps = float(double(n) / seconds);
        ms = n ? float(seconds * 1000.0 / double(n)) : 0.0f;
        frames = count;
        since = now;
    }
    const float scale = std::min(size.x / 1920.0f, size.y / 1080.0f), textSize = 20.0f * scale;
    const std::string text = std::format("{:.0f} FPS  {:.1f} MS", fps, ms);
    const float w = TextWidth(s_numbers, textSize, text), pad = 8.0f * scale;
    const ImVec2 b(pos.x + size.x - 16.0f * scale, pos.y + 16.0f * scale + textSize + 2.0f * pad);
    const ImVec2 a(b.x - w - 2.0f * pad, pos.y + 16.0f * scale);
    dl->AddRectFilled(a, b, Rgb(0x04051A, 0.65f));
    Text(dl, s_numbers, textSize, ImVec2(a.x + pad, a.y + pad), Rgb(0xE4E9FA), text);
}

} // namespace

// --- Mods' drawing (overlay.h ac6::ui::draw) ----------------------------------------------------------------------------
namespace draw {

namespace {

thread_local bool t_active = false;   // this thread is running the mods' draw handlers
ImDrawList* s_list = nullptr;         // their layer (the background list)
int s_clips = 0;                      // clips pushed and not popped

// 0xAARRGGBB as ImGui's colour (0xAABBGGRR).
ImU32 Color(uint32_t argb)
{
    return IM_COL32((argb >> 16) & 0xFF, (argb >> 8) & 0xFF, argb & 0xFF, argb >> 24);
}

ImFont* FontOf(bool mono) { return mono ? s_numbers : s_body; }

} // namespace

bool Active() { return t_active && s_list; }

void Size(float& width, float& height)
{
    const ImVec2 size = ImGui::GetIO().DisplaySize;
    width = size.x;
    height = size.y;
}

void View(float& x, float& y, float& width, float& height)
{
    float w = 0.0f, h = 0.0f;
    Size(w, h);
    const float fw = float(std::max(1u, gpu::g_video.frameWidth)), fh = float(std::max(1u, gpu::g_video.frameHeight));
    const float scale = std::min(w / fw, h / fh);
    width = fw * scale;
    height = fh * scale;
    x = (w - width) / 2;
    y = (h - height) / 2;
}

void Canvas(float& x, float& y, float& width, float& height)
{
    float vx = 0.0f, vy = 0.0f, vw = 0.0f, vh = 0.0f;
    View(vx, vy, vw, vh);
    // (A picture wider or narrower than 16:9, AC6_WIDESCREEN: the game keeps its 2D at 16:9, centred.)
    const float unit = std::min(vw / 16.0f, vh / 9.0f);
    width = 16.0f * unit;
    height = 9.0f * unit;
    x = vx + (vw - width) / 2;
    y = vy + (vh - height) / 2;
}

void Line(float x0, float y0, float x1, float y1, uint32_t color, float thickness)
{
    if (Active())
        s_list->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), Color(color), thickness > 0.0f ? thickness : 1.0f);
}

void Rect(float x0, float y0, float x1, float y1, uint32_t color, float rounding, float thickness)
{
    if (!Active())
        return;
    if (thickness <= 0.0f)
        s_list->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), Color(color), rounding);
    else
        s_list->AddRect(ImVec2(x0, y0), ImVec2(x1, y1), Color(color), rounding, 0, thickness);
}

void Circle(float x, float y, float radius, uint32_t color, float thickness)
{
    if (!Active())
        return;
    if (thickness <= 0.0f)
        s_list->AddCircleFilled(ImVec2(x, y), radius, Color(color));
    else
        s_list->AddCircle(ImVec2(x, y), radius, Color(color), 0, thickness);
}

void Triangle(float x0, float y0, float x1, float y1, float x2, float y2, uint32_t color, float thickness)
{
    if (!Active())
        return;
    if (thickness <= 0.0f)
        s_list->AddTriangleFilled(ImVec2(x0, y0), ImVec2(x1, y1), ImVec2(x2, y2), Color(color));
    else
        s_list->AddTriangle(ImVec2(x0, y0), ImVec2(x1, y1), ImVec2(x2, y2), Color(color), thickness);
}

void Polyline(const float* xy, int points, uint32_t color, bool closed, float thickness)
{
    if (!Active() || !xy || points < 2)
        return;
    std::vector<ImVec2> p(static_cast<size_t>(points));
    for (int i = 0; i < points; i++)
        p[size_t(i)] = ImVec2(xy[2 * i], xy[2 * i + 1]);
    if (closed && thickness <= 0.0f)
        s_list->AddConvexPolyFilled(p.data(), points, Color(color));
    else
        s_list->AddPolyline(p.data(), points, Color(color), closed ? ImDrawFlags_Closed : 0, thickness > 0.0f ? thickness : 1.0f);
}

void Text(float x, float y, const char* text, uint32_t color, float size, bool mono)
{
    if (Active() && text && *text)
        s_list->AddText(FontOf(mono), size > 0.0f ? size : 18.0f, ImVec2(x, y), Color(color), text);
}

void TextSize(const char* text, float size, bool mono, float& width, float& height)
{
    width = height = 0.0f;
    if (!Active() || !text)
        return;
    const float s = size > 0.0f ? size : 18.0f;
    const ImVec2 v = FontOf(mono)->CalcTextSizeA(s, FLT_MAX, 0.0f, text);
    width = v.x;
    height = v.y;
}

void PushClip(float x0, float y0, float x1, float y1)
{
    if (!Active())
        return;
    s_list->PushClipRect(ImVec2(x0, y0), ImVec2(x1, y1), true);
    s_clips++;
}

void PopClip()
{
    if (!Active() || s_clips == 0)
        return;
    s_list->PopClipRect();
    s_clips--;
}

void PopClips()
{
    while (Active() && s_clips > 0)
        PopClip();
}

} // namespace draw

namespace {

// The mods' layer (their "draw" handlers): ImGui's background list, under the overlay's window and the FPS counter.
// Inside a frame (between NewFrame and Render), on the window's thread: the game's main thread, a game thread.
void DrawMods()
{
    if (!mods::WantsDraw())
        return;
    draw::s_list = ImGui::GetBackgroundDrawList();
    draw::s_clips = 0;
    draw::t_active = true;
    struct End
    {
        ~End()
        {
            draw::PopClips();
            draw::t_active = false;
            draw::s_list = nullptr;
        }
    } end;
    mods::OnDraw();
}

} // namespace

void Init(SDL_Window* window)
{
    if (s_init || !window)
        return;
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;   // nothing written next to the game
    io.LogFilename = nullptr;
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures | ImGuiBackendFlags_RendererHasVtxOffset;
    io.BackendRendererName = "ac6_plume";
    s_body = LoadFont("ShareTech", 18.0f);
    if (s_body)
        MergeSystemFonts();
    s_mono = LoadFont("ShareTechMono", 18.0f);
    if (!s_body || !s_mono)
    {
        log::Warn("Settings overlay: the embedded fonts are missing; ImGui's own font instead");
        s_body = s_mono = io.Fonts->AddFontDefault();
    }
    // Share Tech for all text, its Mono only for the numbers that change as you watch (they'd shift as their digits'
    // widths do).
    s_numbers = s_mono;
    s_mono = s_body;
    // AC6_UI_FONT=<.ttf file> (dev: trying a typeface): it for all text but the changing numbers.
    if (const std::filesystem::path file = platform::EnvironmentPath("AC6_UI_FONT"); !file.empty())
    {
        if (ImFont* f = io.Fonts->AddFontFromFileTTF(platform::PathUtf8(file).c_str(), 18.0f))
        {
            s_body = s_mono = f;
            log::Info("Settings overlay: font {} (AC6_UI_FONT)", platform::PathUtf8(file));
        }
        else
            log::Warn("Settings overlay: AC6_UI_FONT={} can't be loaded", platform::PathUtf8(file));
    }
    ImGui_ImplSDL3_InitForOther(window);
    s_windowId = SDL_GetWindowID(window);
    ApplyStyle();
    s_init = true;
    log::Info("Settings overlay ready (F4, or LS + RS / L3 + R3 on a controller)");
}

bool HandleEvent(const SDL_Event& event)
{
    if (!s_init)
        return false;
    if (event.type == SDL_EVENT_KEY_DOWN && event.key.scancode == SDL_SCANCODE_F4 && !event.key.repeat)
    {
        SetOpen(!s_open.load());
        return true;
    }
    // Both stick clicks together on a controller, LS + RS (L3 + R3): input::Update keeps them from the game (a click alone
    // reaches it a moment later). Not Back + Start: Start also opens the game's pause menu.
    if (event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN &&
        (event.gbutton.button == SDL_GAMEPAD_BUTTON_LEFT_STICK || event.gbutton.button == SDL_GAMEPAD_BUTTON_RIGHT_STICK))
        if (SDL_Gamepad* pad = SDL_GetGamepadFromID(event.gbutton.which))
        {
            const SDL_GamepadButton other = event.gbutton.button == SDL_GAMEPAD_BUTTON_LEFT_STICK ? SDL_GAMEPAD_BUTTON_RIGHT_STICK
                                                                                                 : SDL_GAMEPAD_BUTTON_LEFT_STICK;
            if (SDL_GetGamepadButton(pad, other))
            {
                s_gamepad = true;
                SetOpen(!s_open.load());
                return true;
            }
        }
    if (!s_open.load())
        return false;
    if (s_advCapture >= 0 && s_advCapture < int(s_adv.size()) && event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat &&
        event.key.scancode == SDL_SCANCODE_ESCAPE)
    {
        AdvBinding& b = s_adv[size_t(s_advCapture)];
        s_advCapture = -1;
        input::advanced::EndCapture();
        b.value.clear();
        Status(std::format("{}: all inputs cleared (the game's).", b.action->label));
        return true;
    }
    if (s_capture >= 0 && s_capture < int(s_bindings.size()))
    {
        if (event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN && event.gbutton.button == SDL_GAMEPAD_BUTTON_EAST)
        {
            s_capture = -1;
            Status("Cancelled: unchanged.");
            return true;
        }
        if ((event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) || event.type == SDL_EVENT_MOUSE_BUTTON_DOWN ||
            (event.type == SDL_EVENT_MOUSE_WHEEL && event.wheel.y != 0.0f))
        {
            Binding& b = s_bindings[size_t(s_capture)];
            s_capture = -1;
            const std::string name = input::kbm::KeyName(event);
            if (name == "Escape")
            {
                // Escape clears the action: unbound until a key is added or Del brings its defaults.
                b.value.clear();
                Status(std::format("{}: all keys cleared.", b.info->label));
            }
            else if (name.empty())
                Status("That key can't be bound.", true);
            else if (const auto it = std::find(b.value.begin(), b.value.end(), name); it != b.value.end())
            {
                b.value.erase(it);
                Status(std::format("{}: {} removed.", b.info->label, KeyLabel(name)));
            }
            else if (b.value.size() >= kMaxBindingKeys)
                Status(std::format("{}: {} keys at most. Remove one first", b.info->label, kMaxBindingKeys), true);
            else
            {
                b.value.push_back(name);
                Status(std::format("{}: {} added.", b.info->label, KeyLabel(name)));
            }
            return true;
        }
    }
    // Keys posted to the window while it has no keyboard focus (as automation tools do) arrive with no window ID.
    // ImGui's backend takes only its own window's events, so they are given its ID.
    if ((event.type == SDL_EVENT_KEY_DOWN || event.type == SDL_EVENT_KEY_UP) && event.key.windowID == 0)
    {
        SDL_Event copy = event;
        copy.key.windowID = s_windowId;
        ImGui_ImplSDL3_ProcessEvent(&copy);
    }
    else if (event.type == SDL_EVENT_TEXT_INPUT && event.text.windowID == 0)
    {
        SDL_Event copy = event;
        copy.text.windowID = s_windowId;
        ImGui_ImplSDL3_ProcessEvent(&copy);
    }
    else
        ImGui_ImplSDL3_ProcessEvent(&event);
    // While open, the keyboard, the mouse and the controllers' buttons and sticks are the overlay's. Devices coming and
    // going, the window's events and quitting still go on.
    switch (event.type)
    {
    case SDL_EVENT_KEY_DOWN: case SDL_EVENT_KEY_UP: case SDL_EVENT_TEXT_INPUT: case SDL_EVENT_TEXT_EDITING:
    case SDL_EVENT_MOUSE_MOTION: case SDL_EVENT_MOUSE_BUTTON_DOWN: case SDL_EVENT_MOUSE_BUTTON_UP: case SDL_EVENT_MOUSE_WHEEL:
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN: case SDL_EVENT_GAMEPAD_BUTTON_UP: case SDL_EVENT_GAMEPAD_AXIS_MOTION:
        return true;
    default:
        return false;
    }
}

void Frame()
{
    if (!s_init)
        return;
    if (!s_open.load())
    {
        // (The FPS counter and the start-up hint alone, without the backend's new frame: it would read the gamepads for
        // the overlay's keys and show the mouse cursor during play.)
        const float hint = HintAlpha(), updateHint = UpdateHintAlpha();
        if (FpsCounterOn() || hint > 0.0f || updateHint > 0.0f || mods::WantsDraw())
            if (SDL_Window* window = Window())
            {
                int w = 0, h = 0, pw = 0, ph = 0;
                SDL_GetWindowSize(window, &w, &h);
                SDL_GetWindowSizeInPixels(window, &pw, &ph);
                if (w > 0 && h > 0)
                {
                    ImGuiIO& io = ImGui::GetIO();
                    io.DisplaySize = ImVec2(float(w), float(h));
                    io.DisplayFramebufferScale = ImVec2(float(pw) / float(w), float(ph) / float(h));
                    io.DeltaTime = 1.0f / 60.0f;
                    ImGui::NewFrame();
                    DrawMods();
                    const ImGuiViewport* vp = ImGui::GetMainViewport();
                    if (FpsCounterOn())
                        DrawFpsCounter(ImGui::GetForegroundDrawList(), vp->Pos, vp->Size);
                    if (hint > 0.0f)
                        DrawStartupHint(ImGui::GetForegroundDrawList(), vp->Pos, vp->Size, hint);
                    else if (updateHint > 0.0f)
                        DrawUpdateHint(ImGui::GetForegroundDrawList(), vp->Pos, vp->Size, updateHint);
                    ImGui::Render();
                    Publish(ImGui::GetDrawData());
                    return;
                }
            }
        std::lock_guard lock(s_mutex);
        s_frameReady = false;
        return;
    }
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
    DrawMods();   // (under the overlay)
    DrawUi();
    ImGui::Render();
    Publish(ImGui::GetDrawData());
    if (!s_open.load())
    {
        std::lock_guard lock(s_mutex);
        s_frameReady = false;   // (closed during the frame: its texture changes still go)
    }
}

bool IsOpen()
{
    return s_open.load();
}

void SuppressStartupHint()
{
    s_hintDone = true;
}

bool TakeFrame(DrawFrame& out)
{
    std::lock_guard lock(s_mutex);
    out.textures = std::move(s_pendingTextures);
    s_pendingTextures.clear();
    if (!s_frameReady)
        return false;
    out.width = s_frame.width;
    out.height = s_frame.height;
    out.vertices = s_frame.vertices;
    out.indices = s_frame.indices;
    out.commands = s_frame.commands;
    return true;
}

} // namespace ac6::ui
