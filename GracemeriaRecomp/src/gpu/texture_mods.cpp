// Texture dumps and replacements: see gpu/texture_mods.h.
#include "gpu/texture_mods.h"
#include "game/localization.h"
#include "log.h"
#include "platform/platform.h"
#include "settings.h"

#include <plume_render_interface_types.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <format>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

const unsigned char* EmbeddedFileData(const char* name, size_t* size);   // (generated: tools/cmake/embed_files.cmake)

namespace ac6::gpu::mods {

using plume::RenderFormat;

namespace {

std::mutex s_mutex;
std::vector<FrameTexture> s_frame;                   // the last frame's textures
std::vector<FrameTexture> s_requests;                // dumps asked for from the overlay
std::unordered_set<uint64_t> s_dumped;               // dumps written (or found) this run
std::unordered_map<uint64_t, std::filesystem::path> s_index;   // replacements by hash
bool s_indexRead = false;

std::atomic<uint64_t> s_generation{ UINT64_MAX };
std::atomic<bool> s_dumpOn{ false }, s_replaceOn{ true };
std::atomic<bool> s_reload{ false };   // F5: everything read again

void ReadSettings()
{
    if (const uint64_t now = settings::Generation(); s_generation.load(std::memory_order_acquire) != now)
    {
        const std::string d = settings::Get("AC6_DUMP_TEXTURES"), r = settings::Get("AC6_TEXTURE_MODS");
        s_dumpOn.store(d == "1" || d == "true", std::memory_order_relaxed);
        s_replaceOn.store(r != "0" && r != "false", std::memory_order_relaxed);
        s_generation.store(now, std::memory_order_release);
    }
}

// "0123456789ABCDEF..." at the start of a file name: its hash.
bool ParseHash(const std::string& stem, uint64_t& hash)
{
    if (stem.size() < 16)
        return false;
    hash = 0;
    for (size_t i = 0; i < 16; i++)
    {
        const char c = stem[i];
        const int v = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
        if (v < 0)
            return false;
        hash = hash << 4 | uint64_t(v);
    }
    return stem.size() == 16 || !std::isxdigit(uint8_t(stem[16]));
}

// The PlayStation face buttons in the game's prompts (assets/button_glyphs/playstation, embedded): replacements of
// the game's 40x40 glyph textures, by their hashes, while the Button style setting is PlayStation. A mod's or language
// pack's file for the same hash wins.
constexpr const char* kEmbeddedPrefix = "embedded:";
constexpr const char* kPlayStationGlyphs[] = {
    "567C603ACEC3BEF6",   // X: square
    "5815009FFD003F54",   // B: circle
    "62B5B1271CAFB882",   // Y: triangle
    "B30926707688AFA7",   // A: cross
};

bool PlayStationGlyphs()
{
    return settings::Get("AC6_BUTTON_STYLE") == "playstation";
}

// The chosen language pack's textures first, then mods/<mod>/textures/**.dds with `includeMods`, the mods in name order
// (the first one with a hash wins).
std::unordered_map<uint64_t, std::filesystem::path> ScanMods(bool includeMods)
{
    std::unordered_map<uint64_t, std::filesystem::path> index;
    std::error_code ec;
    const std::filesystem::path root = ModsDirectory();
    std::vector<std::filesystem::path> mods;
    if (includeMods && std::filesystem::is_directory(root, ec))
    {
        for (const auto& entry : std::filesystem::directory_iterator(root, ec))
            if (entry.is_directory(ec))
                mods.push_back(entry.path());
        std::sort(mods.begin(), mods.end());
    }
    if (const localization::Pack* pack = localization::Active())
        mods.insert(mods.begin(), pack->dir);
    for (const auto& mod : mods)
    {
        const std::filesystem::path textures = mod / "textures";
        if (!std::filesystem::is_directory(textures, ec))
            continue;
        for (auto it = std::filesystem::recursive_directory_iterator(textures, ec); it != std::filesystem::recursive_directory_iterator();
             it.increment(ec))
        {
            if (ec || !it->is_regular_file(ec))
                continue;
            std::string ext = platform::PathUtf8(it->path().extension());
            std::transform(ext.begin(), ext.end(), ext.begin(), [](char c) { return char(std::tolower(uint8_t(c))); });
            uint64_t hash;
            if (ext == ".dds" && ParseHash(platform::PathUtf8(it->path().stem()), hash))
                index.emplace(hash, it->path());
        }
    }
    if (PlayStationGlyphs())
        for (const char* name : kPlayStationGlyphs)
        {
            uint64_t hash;
            if (ParseHash(name, hash))
                index.emplace(hash, std::filesystem::path(std::string(kEmbeddedPrefix) + "PlayStationGlyph_" + name));
        }
    return index;
}

// --- DDS ---------------------------------------------------------------------------------------------------------
constexpr uint32_t FourCC(char a, char b, char c, char d) { return uint32_t(uint8_t(a)) | uint32_t(uint8_t(b)) << 8 | uint32_t(uint8_t(c)) << 16 | uint32_t(uint8_t(d)) << 24; }
constexpr uint32_t kDdpfAlpha = 0x1, kDdpfFourCC = 0x4, kDdpfRgb = 0x40, kDdpfLuminance = 0x20000;

#pragma pack(push, 1)
struct DdsPixelFormat { uint32_t size, flags, fourCC, bits, r, g, b, a; };
struct DdsHeader
{
    uint32_t size, flags, height, width, pitchOrLinearSize, depth, mipCount, reserved1[11];
    DdsPixelFormat pf;
    uint32_t caps, caps2, caps3, caps4, reserved2;
};
struct DdsHeader10 { uint32_t dxgiFormat, dimension, miscFlag, arraySize, miscFlags2; };
#pragma pack(pop)
static_assert(sizeof(DdsHeader) == 124 && sizeof(DdsPixelFormat) == 32);

uint32_t BlockBytes(int format)
{
    switch (RenderFormat(format))
    {
    case RenderFormat::BC1_UNORM: case RenderFormat::BC4_UNORM: return 8;
    case RenderFormat::BC2_UNORM: case RenderFormat::BC3_UNORM: case RenderFormat::BC5_UNORM: case RenderFormat::BC7_UNORM: return 16;
    default: return 0;
    }
}
uint32_t PixelBytes(int format)
{
    switch (RenderFormat(format))
    {
    case RenderFormat::B8G8R8A8_UNORM: case RenderFormat::R8G8B8A8_UNORM: return 4;
    case RenderFormat::R8_UNORM: return 1;
    default: return 0;
    }
}

struct Job { std::filesystem::path path; Image image; };
std::mutex s_jobMutex;
std::condition_variable s_jobCv;
std::deque<Job> s_jobs;

bool WriteDds(const std::filesystem::path& path, const Image& image)
{
    DdsHeader h{};
    h.size = 124;
    h.flags = 0x1 | 0x2 | 0x4 | 0x1000 | (image.levels.size() > 1 ? 0x20000 : 0);
    h.width = image.width;
    h.height = image.height;
    h.mipCount = uint32_t(image.levels.size());
    h.pf.size = 32;
    h.caps = 0x1000 | (image.levels.size() > 1 ? 0x400008 : 0);
    switch (RenderFormat(image.format))
    {
    case RenderFormat::BC1_UNORM: h.pf.flags = kDdpfFourCC; h.pf.fourCC = FourCC('D', 'X', 'T', '1'); break;
    case RenderFormat::BC2_UNORM: h.pf.flags = kDdpfFourCC; h.pf.fourCC = FourCC('D', 'X', 'T', '3'); break;
    case RenderFormat::BC3_UNORM: h.pf.flags = kDdpfFourCC; h.pf.fourCC = FourCC('D', 'X', 'T', '5'); break;
    case RenderFormat::BC4_UNORM: h.pf.flags = kDdpfFourCC; h.pf.fourCC = FourCC('A', 'T', 'I', '1'); break;
    case RenderFormat::BC5_UNORM: h.pf.flags = kDdpfFourCC; h.pf.fourCC = FourCC('A', 'T', 'I', '2'); break;
    case RenderFormat::B8G8R8A8_UNORM:
        h.pf.flags = kDdpfRgb | kDdpfAlpha; h.pf.bits = 32; h.pf.r = 0x00FF0000; h.pf.g = 0x0000FF00; h.pf.b = 0x000000FF; h.pf.a = 0xFF000000;
        break;
    case RenderFormat::R8G8B8A8_UNORM:
        h.pf.flags = kDdpfRgb | kDdpfAlpha; h.pf.bits = 32; h.pf.r = 0x000000FF; h.pf.g = 0x0000FF00; h.pf.b = 0x00FF0000; h.pf.a = 0xFF000000;
        break;
    case RenderFormat::R8_UNORM: h.pf.flags = kDdpfLuminance; h.pf.bits = 8; h.pf.r = 0xFF; break;
    default: return false;
    }
    h.flags |= BlockBytes(image.format) ? 0x80000 : 0x8;
    h.pitchOrLinearSize = BlockBytes(image.format) ? uint32_t(LevelBytes(image.format, image.width, image.height))
                                                   : image.width * PixelBytes(image.format);
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    const std::filesystem::path temp = std::filesystem::path(path).concat(".part");
    {
        std::ofstream f(temp, std::ios::binary);
        if (!f)
            return false;
        f.write("DDS ", 4);
        f.write(reinterpret_cast<const char*>(&h), sizeof(h));
        for (const auto& level : image.levels)
            f.write(reinterpret_cast<const char*>(level.data()), std::streamsize(level.size()));
        if (!f)
            return false;
    }
    std::filesystem::rename(temp, path, ec);
    return !ec;
}

void WriterLoop()
{
    for (;;)
    {
        Job job;
        {
            std::unique_lock lock(s_jobMutex);
            s_jobCv.wait(lock, [] { return !s_jobs.empty(); });
            job = std::move(s_jobs.front());
            s_jobs.pop_front();
        }
        if (!WriteDds(job.path, job.image))
            log::Warn("Texture dump: writing {} failed", platform::PathUtf8(job.path));
    }
}

} // namespace

std::filesystem::path DumpDirectory() { return platform::ExecutableDirectory() / "dump" / "textures"; }
std::filesystem::path ModsDirectory() { return platform::ExecutableDirectory() / "mods"; }

bool DumpOn()
{
    ReadSettings();
    return s_dumpOn.load(std::memory_order_relaxed);
}

bool ReplaceOn()
{
    ReadSettings();
    return s_replaceOn.load(std::memory_order_relaxed);
}

bool RefreshIndex()
{
    static uint64_t seen = UINT64_MAX;
    static bool replacing = true;
    const uint64_t now = settings::Generation();
    const bool reload = s_reload.exchange(false);
    if (!reload && seen == now && s_indexRead)
        return false;
    seen = now;
    const bool on = ReplaceOn();
    auto index = ScanMods(on);   // (a language pack's textures either way)
    bool changed = on != replacing;
    replacing = on;
    {
        std::lock_guard lock(s_mutex);
        if (index.size() != s_index.size())
            changed = true;
        else
            for (const auto& [hash, path] : index)
                if (auto it = s_index.find(hash); it == s_index.end() || it->second != path)
                {
                    changed = true;
                    break;
                }
        s_index = std::move(index);
        if (!s_indexRead || changed || reload)
            log::Info("Texture mods: {} replacement(s) in {}{}{}", s_index.size(), platform::PathUtf8(ModsDirectory()),
                on ? "" : " (off: AC6_TEXTURE_MODS=0)", reload ? " (F5: all read again)" : "");
        s_indexRead = true;
    }
    // (A reload has the renderer check every texture again, and a replaced one reads its file again.)
    return changed || reload;
}

void RequestReload()
{
    s_reload.store(true);
}

std::optional<std::filesystem::path> FindReplacement(uint64_t hash)
{
    std::lock_guard lock(s_mutex);
    if (auto it = s_index.find(hash); it != s_index.end())
        return it->second;
    return std::nullopt;
}

size_t ReplacementCount()
{
    std::lock_guard lock(s_mutex);
    return s_index.size();
}

void PublishFrame(std::vector<FrameTexture>&& textures)
{
    std::lock_guard lock(s_mutex);
    s_frame = std::move(textures);
}

std::vector<FrameTexture> FrameTextures()
{
    std::lock_guard lock(s_mutex);
    return s_frame;
}

void RequestDump(const FrameTexture& texture)
{
    std::lock_guard lock(s_mutex);
    s_requests.push_back(texture);
}

std::vector<FrameTexture> TakeDumpRequests()
{
    std::lock_guard lock(s_mutex);
    return std::exchange(s_requests, {});
}

void NoteDumped(uint64_t hash)
{
    std::lock_guard lock(s_mutex);
    s_dumped.insert(hash);
}

bool Dumped(uint64_t hash)
{
    {
        std::lock_guard lock(s_mutex);
        if (s_dumped.contains(hash))
            return true;
    }
    std::error_code ec;
    return std::filesystem::exists(DumpDirectory() / std::format("{:016X}.dds", hash), ec);
}

uint64_t LevelBytes(int format, uint32_t w, uint32_t h)
{
    if (const uint32_t block = BlockBytes(format))
        return uint64_t(std::max(1u, (w + 3) / 4)) * std::max(1u, (h + 3) / 4) * block;
    return uint64_t(w) * h * PixelBytes(format);
}

void MakeMips(Image& image, uint32_t levels)
{
    const uint32_t bpp = PixelBytes(image.format);
    if (bpp == 0 || image.levels.empty())
        return;
    image.levels.resize(1);
    uint32_t w = image.width, h = image.height;
    while (image.levels.size() < levels && (w > 1 || h > 1))
    {
        const uint32_t nw = std::max(1u, w / 2), nh = std::max(1u, h / 2);
        const std::vector<uint8_t>& src = image.levels.back();
        std::vector<uint8_t> dst(size_t(nw) * nh * bpp);
        for (uint32_t y = 0; y < nh; y++)
            for (uint32_t x = 0; x < nw; x++)
                for (uint32_t c = 0; c < bpp; c++)
                {
                    const uint32_t x0 = std::min(2 * x, w - 1), x1 = std::min(2 * x + 1, w - 1), y0 = std::min(2 * y, h - 1),
                                   y1 = std::min(2 * y + 1, h - 1);
                    const uint32_t sum = src[(size_t(y0) * w + x0) * bpp + c] + src[(size_t(y0) * w + x1) * bpp + c] +
                        src[(size_t(y1) * w + x0) * bpp + c] + src[(size_t(y1) * w + x1) * bpp + c];
                    dst[(size_t(y) * nw + x) * bpp + c] = uint8_t((sum + 2) / 4);
                }
        image.levels.push_back(std::move(dst));
        w = nw;
        h = nh;
    }
}

bool ReadDds(const std::filesystem::path& path, Image& out, std::string& error)
{
    std::vector<uint8_t> data;
    if (const std::string name = platform::PathUtf8(path); name.starts_with(kEmbeddedPrefix))
    {
        size_t size = 0;
        const unsigned char* bytes = EmbeddedFileData(name.c_str() + std::strlen(kEmbeddedPrefix), &size);
        if (!bytes)
        {
            error = "not embedded";
            return false;
        }
        data.assign(bytes, bytes + size);
    }
    else
    {
        std::ifstream f(path, std::ios::binary);
        if (!f)
        {
            error = "can't open";
            return false;
        }
        data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    if (data.size() < 128 || std::memcmp(data.data(), "DDS ", 4) != 0)
    {
        error = "not a DDS file";
        return false;
    }
    DdsHeader h;
    std::memcpy(&h, data.data() + 4, sizeof(h));
    size_t at = 128;
    RenderFormat format = RenderFormat::UNKNOWN;
    enum class Convert { None, Rgb24 } convert = Convert::None;
    const DdsPixelFormat& pf = h.pf;
    if (pf.flags & kDdpfFourCC)
    {
        switch (pf.fourCC)
        {
        case FourCC('D', 'X', 'T', '1'): format = RenderFormat::BC1_UNORM; break;
        case FourCC('D', 'X', 'T', '2'): case FourCC('D', 'X', 'T', '3'): format = RenderFormat::BC2_UNORM; break;
        case FourCC('D', 'X', 'T', '4'): case FourCC('D', 'X', 'T', '5'): format = RenderFormat::BC3_UNORM; break;
        case FourCC('A', 'T', 'I', '1'): case FourCC('B', 'C', '4', 'U'): format = RenderFormat::BC4_UNORM; break;
        case FourCC('A', 'T', 'I', '2'): case FourCC('B', 'C', '5', 'U'): format = RenderFormat::BC5_UNORM; break;
        case FourCC('D', 'X', '1', '0'):
        {
            if (data.size() < 148)
            {
                error = "truncated DX10 header";
                return false;
            }
            DdsHeader10 h10;
            std::memcpy(&h10, data.data() + 128, sizeof(h10));
            at = 148;
            switch (h10.dxgiFormat)
            {
            case 71: case 72: format = RenderFormat::BC1_UNORM; break;
            case 74: case 75: format = RenderFormat::BC2_UNORM; break;
            case 77: case 78: format = RenderFormat::BC3_UNORM; break;
            case 80: format = RenderFormat::BC4_UNORM; break;
            case 83: format = RenderFormat::BC5_UNORM; break;
            case 97: case 98: case 99: format = RenderFormat::BC7_UNORM; break;   // (sRGB read as stored, as BC1-3's)
            case 87: case 88: case 91: format = RenderFormat::B8G8R8A8_UNORM; break;
            case 28: case 29: format = RenderFormat::R8G8B8A8_UNORM; break;
            case 61: format = RenderFormat::R8_UNORM; break;
            default: error = std::format("DXGI format {} not supported", h10.dxgiFormat); return false;
            }
            break;
        }
        default: error = "compression not supported (DXT1/3/5, ATI1/2 and DX10 BC1-5, BC7 / 8-bit RGBA are)"; return false;
        }
    }
    else if ((pf.flags & kDdpfRgb) && pf.bits == 32)
        format = pf.r == 0x000000FF ? RenderFormat::R8G8B8A8_UNORM : RenderFormat::B8G8R8A8_UNORM;
    else if ((pf.flags & kDdpfRgb) && pf.bits == 24)
    {
        format = RenderFormat::B8G8R8A8_UNORM;
        convert = Convert::Rgb24;
    }
    else if ((pf.flags & kDdpfLuminance) && pf.bits == 8)
        format = RenderFormat::R8_UNORM;
    else
    {
        error = "pixel format not supported";
        return false;
    }
    out.format = int(format);
    out.width = h.width;
    out.height = h.height;
    out.levels.clear();
    const uint32_t count = std::max(1u, (h.flags & 0x20000) ? h.mipCount : 1u);
    uint32_t w = h.width, ht = h.height;
    for (uint32_t level = 0; level < count && (level == 0 || w > 1 || ht > 1 || level == 0); level++)
    {
        const uint64_t srcBytes = convert == Convert::Rgb24 ? uint64_t(w) * ht * 3 : LevelBytes(int(format), w, ht);
        if (at + srcBytes > data.size())
        {
            if (level == 0)
            {
                error = "truncated";
                return false;
            }
            break;
        }
        std::vector<uint8_t> level_(size_t(LevelBytes(int(format), w, ht)));
        if (convert == Convert::Rgb24)
            for (size_t i = 0; i < size_t(w) * ht; i++)
            {
                // 24-bit: the masks name the byte order. B, G, R in memory is the usual one.
                const uint8_t* s = data.data() + at + i * 3;
                const bool rgb = pf.r == 0x0000FF;
                level_[i * 4 + 0] = rgb ? s[2] : s[0];
                level_[i * 4 + 1] = s[1];
                level_[i * 4 + 2] = rgb ? s[0] : s[2];
                level_[i * 4 + 3] = 0xFF;
            }
        else
            std::memcpy(level_.data(), data.data() + at, level_.size());
        at += srcBytes;
        out.levels.push_back(std::move(level_));
        if (w == 1 && ht == 1)
            break;
        w = std::max(1u, w / 2);
        ht = std::max(1u, ht / 2);
    }
    // R8G8B8A8 is made B8G8R8A8 (the game's 8-bit colour textures are), so their channel mapping stays the game's.
    if (format == RenderFormat::R8G8B8A8_UNORM)
    {
        for (auto& level : out.levels)
            for (size_t i = 0; i + 3 < level.size(); i += 4)
                std::swap(level[i], level[i + 2]);
        out.format = int(RenderFormat::B8G8R8A8_UNORM);
    }
    return true;
}

void WriteDdsAsync(std::filesystem::path path, Image image)
{
    static std::once_flag started;
    std::call_once(started, [] { std::thread(WriterLoop).detach(); });
    {
        std::lock_guard lock(s_jobMutex);
        s_jobs.push_back({ std::move(path), std::move(image) });
    }
    s_jobCv.notify_one();
}

} // namespace ac6::gpu::mods
