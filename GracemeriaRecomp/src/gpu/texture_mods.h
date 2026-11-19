#pragma once
// Texture dumps and replacements.
//
// Dump: with AC6_DUMP_TEXTURES=1 (settings modding.dump_textures, live) every texture the game uploads is written once, as
// the GPU gets it (untiled, byte order fixed, its mips), to dump/textures/<hash>.dds next to the executable: <hash> is the
// renderer's content hash of the guest data (16 hex digits), the same whatever is drawn around it.
// Replace: mods/<any mod>/textures/<hash>[anything].dds (next to the executable) replaces the texture with that hash, at
// the file's own size (higher resolutions too) and format (BC1-5, 8-bit RGBA / BGRA, 8-bit luminance, mips taken from
// the file, or made for 8-bit colour). AC6_TEXTURE_MODS=0 (modding.texture_replacements) turns replacements off (live).
// The browser (the settings overlay's Debug tab): the textures the last frame used, drawn by their descriptors.
// F5 (RequestReload) reads the mods folder and every replacement file again at the next frame, so a modder sees an
// edited file in the game at once.
//
// The files are the player's: nothing is shipped or downloaded. A mod ships its own art.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ac6::gpu::mods {

// A texture the last frame used (the overlay's browser).
struct FrameTexture
{
    uint64_t hash = 0;                 // the content hash (the dump's and a replacement's name)
    uint32_t width = 0, height = 0;    // the game's size
    uint32_t levels = 0;               // the game's mip levels
    uint32_t format = 0;               // the guest format (xenos)
    uint32_t descriptor = 0;           // the host image's (the replacement's when replaced)
    bool replaced = false, cube = false;
    uint32_t fetch[6] = {};            // the fetch constant (a dump on request decodes from it)
};

// The overlay's thumbnails: an ImGui texture id of this flag | descriptor draws that game texture.
constexpr uint32_t kGameTextureFlag = 0xFE000000u;

// Overlay side (main thread).
std::vector<FrameTexture> FrameTextures();   // the last frame presented while the game ran
void RequestDump(const FrameTexture& texture);   // written by the renderer at its next present (the held frame's too)
bool Dumped(uint64_t hash);                       // a dump of it exists (this run or before)
std::filesystem::path DumpDirectory();
std::filesystem::path ModsDirectory();
size_t ReplacementCount();
void RequestReload();   // F5: the replacements read again from their files at the next frame

// Renderer side.
bool DumpOn();
bool ReplaceOn();
std::optional<std::filesystem::path> FindReplacement(uint64_t hash);
// The settings or the mods changed: the replacement index read again. True when the renderer should check its textures
// again (replacements on / off, files added).
bool RefreshIndex();
void PublishFrame(std::vector<FrameTexture>&& textures);
std::vector<FrameTexture> TakeDumpRequests();
void NoteDumped(uint64_t hash);

// An image as DDS holds it: tightly packed levels (blocks for BC formats). Formats: plume's RenderFormat values, as int.
struct Image
{
    int format = 0;
    uint32_t width = 0, height = 0;
    std::vector<std::vector<uint8_t>> levels;
};
bool ReadDds(const std::filesystem::path& path, Image& out, std::string& error);
// Written on a background thread (the call returns at once).
void WriteDdsAsync(std::filesystem::path path, Image image);
// Bytes of a level (w x h texels) in a format, 0 if unknown.
uint64_t LevelBytes(int format, uint32_t w, uint32_t h);
// 8-bit colour: the mip chain made by 2x2 box filtering, to `levels` levels in all.
void MakeMips(Image& image, uint32_t levels);

} // namespace ac6::gpu::mods
