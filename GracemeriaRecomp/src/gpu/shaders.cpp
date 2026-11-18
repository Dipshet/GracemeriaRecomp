#include "gpu/shaders.h"
#include "gpu/capture.h"
#include "gpu/renderer.h"

#include "endian.h"
#include "fast_hash.h"
#include "gpu/dxc.h"
#include "gpu/video.h"
#include "host_cpu.h"
#include "gpu/xenos/shader_recompiler.h"
#include "gpu/xenos/spirv_recompiler.h"
#include "log.h"
#include "memory.h"
#include "platform/platform.h"
#include "settings.h"

#define XXH_INLINE_ALL
#include "xxhash.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <thread>
#include <unordered_set>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <unordered_map>

namespace ac6::gpu {

extern const char* const kXenosShaderCommon;   // shader_common.h, embedded at build time
extern const uint64_t kShaderCacheKey;          // hash of the translator's sources (tools/cmake/source_key.cmake)

namespace {

// Guest D3D shader objects (AC6, June 2007 XDK, the container's own flags give the stage: bit 0 set =
// vertex):
//   vertex shader (object type 6): microcode pointer at +0x20, copy of the container's virtual part at +0x368
//   pixel shader  (object type 7): microcode pointer at +0x18, copy of the container's virtual part at +0x28
constexpr uint32_t kVsMicrocode = 0x20, kVsContainer = 0x368;
constexpr uint32_t kPsMicrocode = 0x18, kPsContainer = 0x28;

uint32_t U32(uint32_t guest) { return *g_memory.As<be<uint32_t>>(guest); }

std::mutex s_mutex;
FastMap<uint64_t, std::unique_ptr<HostShader>> s_byHash;
// Shaders by object and state (vertex shaders: the container record D3D loads and the patched microcode it
// loaded, which the game alternates per draw for objects drawn with several vertex declarations), so a
// bound object is recognised without rebuilding and hashing its container. Each entry
// keeps what it was built from: the container's sizes and microcode address, and the generation its
// microcode pages were write-watched from.
struct ObjectShader
{
    uint32_t object = 0;
    uint64_t state = 0;
    HostShader* shader = nullptr;
    uint32_t microcode = 0, virtualSize = 0, physicalSize = 0;
    uint64_t watchGeneration = 0;
    uint64_t checkedFrame = UINT64_MAX;   // the memory was last compared in this frame
};
FastMap<uint64_t, ObjectShader> s_byObject;
std::atomic<uint64_t> s_stateEpoch{ 0 };   // bumped with any object's state: invalidates the per-thread memo

uint64_t FastHash(const uint8_t* data, size_t size)
{
    uint64_t h = 0x9E3779B97F4A7C15ull ^ size;
    size_t i = 0;
    for (; i + 8 <= size; i += 8)
    {
        uint64_t w;
        std::memcpy(&w, data + i, 8);
        h = (h ^ w) * 0xFF51AFD7ED558CCDull;
        h ^= h >> 32;
    }
    for (; i < size; i++)
        h = (h ^ data[i]) * 0x100000001B3ull;
    return h;
}

FastMap<uint32_t, uint32_t> s_vsVariant;   // vertex shader object -> record D3D loaded last
// Vertex shader object -> the patched microcode D3D loaded last (offset into the physical part, bytes).
struct PatchedMicrocode { uint32_t offset = 0; std::vector<uint8_t> bytes; uint64_t key = 0; };
FastMap<uint32_t, PatchedMicrocode> s_vsPatched;

void SetPatched(uint32_t object, uint32_t offset, std::vector<uint8_t> bytes)
{
    const uint64_t key = FastHash(bytes.data(), bytes.size()) ^ (uint64_t(offset) << 40) ^ 1;
    s_vsPatched[object] = { offset, std::move(bytes), key };
    s_stateEpoch.fetch_add(1, std::memory_order_release);
}

// The object's current state (s_mutex held).
uint64_t ObjectState(uint32_t object, bool pixel)
{
    if (pixel)
        return 1;
    uint64_t state = 2;
    if (auto it = s_vsVariant.find(object); it != s_vsVariant.end())
        state ^= uint64_t(it->second) << 2;
    if (auto it = s_vsPatched.find(object); it != s_vsPatched.end())
        state ^= it->second.key * 0x9E3779B97F4A7C15ull;
    return state;
}

uint64_t Hash(const uint8_t* data, size_t size)
{
    uint64_t h = 0xCBF29CE484222325ull;   // FNV-1a
    for (size_t i = 0; i < size; i++)
        h = (h ^ data[i]) * 0x100000001B3ull;
    return h;
}

// Copies of one shader. The game compiled some shader sources more than once, into different effect and map packages,
// and the copies differ only in the container's debug block (the compile time and the source path, between the header
// and the constant table). The renderer's lists name the copies that were seen drawing. Here the others take a listed
// copy's hash, so a list's fix reaches every copy: keyed by the code (the container from the constant table on, the
// microcode included, ^1 for a pixel shader). Only groups whose listed copies all get the same treatment are here.
struct CodeTwin { uint64_t code, listed; };
constexpr CodeTwin kCodeTwins[] = {
    { 0x88D6D41EE3C374E6ull, 0xCDB30131360ACAAAull },   // Map_HDR/psMapCTF2_Detail_Expfog_Ocean: the ground and the sea
    { 0x2F3905B4054D205Bull, 0x3ADEF3EC4BA4640Dull },   // Map/psMapCTF2_Detail_Expfog_Ocean
    { 0x48277E24935528D7ull, 0x5776EB8756E408E9ull },   // Effect/Cloud_p
    { 0xC52F24803B0C6ED4ull, 0xA8D7F765A3393F89ull },   // Effect/TranslucentEdge: smoke
    { 0xC52F24803B0C6ED4ull, 0xABB89761D796D312ull },
    { 0xC52F24803B0C6ED4ull, 0x2CF84218D7CB5AA2ull },
    { 0xE0377879EB4C45E1ull, 0x757D23F71F52C060ull },   // Effect/TranslucentEdgeDetail: smoke
    { 0xE0377879EB4C45E1ull, 0x10FBAD402151CCB3ull },
    { 0xE0ACA2C6B456E504ull, 0x7305CA060F4A2E57ull },   // Effect/AddDetail: fire
    { 0x48B294DF9461A267ull, 0x228BE7B164C30800ull },   // Effect/TranslucentCustomFog: trails
    { 0xA48466281D3F4780ull, 0x3A245E21E97A7428ull },   // Effect/ReductionBufferSynthesis (vertex shader)
    { 0xDAFACDF714BBDC97ull, 0xE92849A92A4685C9ull },   // PostEffect/vsPosition
    { 0xEBE0747B21D36838ull, 0x93A6F18170BA6E59ull },   // PostEffect/vsPositionTexcoord
    { 0xEBE0747B21D36838ull, 0x1E69729DE8C04E0Full },
    { 0xEBE0747B21D36838ull, 0x7DB6E665461263CAull },
    { 0xEBE0747B21D36838ull, 0x283D9FA4E34F3016ull },
};

// The shader's source name from the container's debug block (at the header's third word: a time, the path's length,
// the path to the game's .updb file): its folder and name, as "Effect/AfterBurner". Empty without one.
std::string SourceName(const std::vector<uint8_t>& c)
{
    auto be32 = [&c](size_t at) { return at + 4 <= c.size() ? __builtin_bswap32(*reinterpret_cast<const uint32_t*>(c.data() + at)) : 0u; };
    const uint32_t block = be32(12), length = be32(size_t(block) + 4);
    if (!block || length == 0 || length > 512 || size_t(block) + 8 + length > c.size())
        return {};
    std::string path(reinterpret_cast<const char*>(c.data() + block + 8), length);
    path.resize(std::strlen(path.c_str()));
    std::replace(path.begin(), path.end(), '\\', '/');
    if (path.ends_with(".updb"))
        path.resize(path.size() - 5);
    const size_t last = path.rfind('/');
    const size_t folder = last == std::string::npos || last == 0 ? std::string::npos : path.rfind('/', last - 1);
    return folder == std::string::npos ? path : path.substr(folder + 1);
}

// A container's hash as the shader lists and caches know it (before the vertex shader record's bits): its own, or for
// an unlisted copy of a listed shader, the listed copy's.
uint64_t ShaderHash(const uint8_t* container, size_t size, bool pixel)
{
    const uint64_t own = Hash(container, size) ^ (pixel ? 1 : 0);
    static std::mutex mutex;
    static std::unordered_map<uint64_t, uint64_t> known;
    std::lock_guard lock(mutex);
    if (const auto it = known.find(own); it != known.end())
        return it->second;
    uint64_t result = own;
    const bool listed = std::any_of(std::begin(kCodeTwins), std::end(kCodeTwins), [own](const CodeTwin& t) { return t.listed == own; });
    const uint32_t constantTable = size >= 20 ? __builtin_bswap32(*reinterpret_cast<const uint32_t*>(container + 16)) : 0;
    if (!listed && constantTable && constantTable < size)
    {
        const uint64_t code = Hash(container + constantTable, size - constantTable) ^ (pixel ? 1 : 0);
        for (const CodeTwin& t : kCodeTwins)
            if (t.code == code)
            {
                result = t.listed;
                log::Info("Shader {:016X}: a copy of {:016X} (the same code), known by its hash", own, result);
                break;
            }
    }
    known.emplace(own, result);
    return result;
}

bool DumpEnabled()
{
    static const bool enabled = [] {
        const char* v = std::getenv("AC6_DUMP_SHADERS");
        return v && *v && *v != '0';
    }();
    return enabled;
}

void Dump(const HostShader& s, const std::string& hlsl, const std::string& errors)
{
    std::filesystem::path dir = std::filesystem::path(log::Directory()) / "shaders";
    std::filesystem::create_directories(dir);
    std::string stem = std::format("{:016X}_{}", s.hash, s.pixel ? "ps" : "vs");
    std::ofstream(dir / (stem + ".bin"), std::ios::binary).write(reinterpret_cast<const char*>(s.container.data()), s.container.size());
    if (!hlsl.empty())
        std::ofstream(dir / (stem + ".hlsl"), std::ios::binary) << hlsl;
    if (!errors.empty())
        std::ofstream(dir / (stem + ".errors.txt"), std::ios::binary) << errors;
    if (!s.spirv.empty())
        std::ofstream(dir / (stem + ".spv"), std::ios::binary).write(reinterpret_cast<const char*>(s.spirv.data()), s.spirv.size() * 4);
}

// Rebuilds the container bytes: [virtual part][physical part].
bool BuildContainer(uint32_t object, bool pixel, std::vector<uint8_t>& out)
{
    const uint32_t virt = object + (pixel ? kPsContainer : kVsContainer);
    auto* header = g_memory.As<xenos::ShaderContainer>(virt);
    if ((header->flags & 0xFFFFFF00u) != 0x102A1100u || ((header->flags & 1u) != 0) == pixel)
        return false;
    const uint32_t virtualSize = header->virtualSize, physicalSize = header->physicalSize;
    const uint32_t microcode = *g_memory.As<be<uint32_t>>(object + (pixel ? kPsMicrocode : kVsMicrocode));
    if (virtualSize > 0x10000 || physicalSize > 0x40000 || !microcode)
        return false;
    out.resize(virtualSize + physicalSize);
    std::memcpy(out.data(), g_memory.Translate(virt), virtualSize);
    std::memcpy(out.data() + virtualSize, g_memory.Translate(microcode), physicalSize);
    // The microcode D3D actually loaded (patched for the vertex declaration) replaces the original.
    if (!pixel)
        if (auto it = s_vsPatched.find(object); it != s_vsPatched.end() &&
            uint64_t(it->second.offset) + it->second.bytes.size() <= physicalSize)
            std::memcpy(out.data() + virtualSize + it->second.offset, it->second.bytes.data(), it->second.bytes.size());
    return true;
}

// AC6_SHADER_PATCH=<file>: text replacements applied to the generated HLSL of chosen shaders (bisecting a
// shader's inputs in a replay). Each line is <hash prefix>|<from>|<to>, with \t and \n escapes in from/to.
void ApplyPatches(uint64_t hash, std::string& hlsl)
{
    static const std::vector<std::array<std::string, 3>> patches = [] {
        std::vector<std::array<std::string, 3>> out;
        const char* v = std::getenv("AC6_SHADER_PATCH");
        if (!v || !*v)
            return out;
        std::ifstream f(v);
        std::string line;
        auto unescape = [](std::string t) {
            std::string r;
            for (size_t i = 0; i < t.size(); i++)
                if (t[i] == '\\' && i + 1 < t.size() && (t[i + 1] == 't' || t[i + 1] == 'n'))
                    r += t[++i] == 't' ? '\t' : '\n';
                else
                    r += t[i];
            return r;
        };
        while (std::getline(f, line))
        {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
                line.pop_back();
            const size_t a = line.find('|'), b = line.find('|', a + 1);
            if (line.empty() || line[0] == '#' || a == std::string::npos || b == std::string::npos)
                continue;
            out.push_back({ line.substr(0, a), unescape(line.substr(a + 1, b - a - 1)), unescape(line.substr(b + 1)) });
        }
        log::Info("Shader patches: {} from {}", out.size(), v);
        return out;
    }();
    const std::string hex = std::format("{:016X}", hash);
    for (const auto& [prefix, from, to] : patches)
    {
        if (!hex.starts_with(prefix) || from.empty())
            continue;
        uint32_t count = 0;
        for (size_t pos = hlsl.find(from); pos != std::string::npos; pos = hlsl.find(from, pos + to.size()), count++)
            hlsl.replace(pos, from.size(), to);
        if (count)
            log::Info("Shader {}: patched {} occurrence(s) of \"{}\"", hex, count, from.substr(0, 60));
        else
            log::Warn("Shader {}: patch \"{}\" matched nothing", hex, from.substr(0, 60));
    }
}

// Persistent shader cache: the translator's outputs per shader hash in
// <exe dir>/shadercache/<hash>_<vs|ps>.bin, so a shader is translated and compiled once, not on every
// run (a mission's first minute meets a few hundred shaders). Entries
// carry kShaderCacheKey: a changed translator ignores them. Off with AC6_SHADER_CACHE=0, and while
// shaders are dumped or patched.
constexpr uint32_t kCacheMagic = 0x53364341;   // "AC6S". Bump kCacheFormat when the entry layout changes
constexpr uint32_t kCacheFormat = 21;   // bumped whenever the entry layout or a translation's output changes

uint64_t CacheKey()
{
    // Every switch a translation depends on (the EDRAM swizzle emulation decides the un-swizzle shaders' identity
    // fetches).
    return kShaderCacheKey ^ kCacheFormat ^ (HlslBackend() ? 0x100 : 0) ^ (UniformConstants() ? 0 : 0x200) ^
        (EdramSwizzle() && !HlslBackend() ? 0x400 : 0);
}

// Masked execution of pixel shader jumps and predicated fetches. AC6_MASKED_EXEC=0 translates them as branches
// instead, for comparisons (the cache is off then).
bool MaskedExecEnabled()
{
    static const bool enabled = [] {
        const char* v = std::getenv("AC6_MASKED_EXEC");
        return !v || *v != '0';
    }();
    return enabled;
}

// AC6_SCENE_FLOAT_MIN=0: the float scene's shaders keep their minimums against white (the effects
// composite's clamp). Those shaders are then translated at each start, not cached.
bool KeepWhiteMin()
{
    static const bool keep = [] { const char* v = std::getenv("AC6_SCENE_FLOAT_MIN"); return v && *v == '0'; }();
    return keep;
}

bool CacheEnabled()
{
    static const bool enabled = [] {
        const char* v = std::getenv("AC6_SHADER_CACHE");
        const char* patch = std::getenv("AC6_SHADER_PATCH");
        return (!v || *v != '0') && !(patch && *patch) && !DumpEnabled() && MaskedExecEnabled();
    }();
    return enabled;
}

std::filesystem::path CachePath(const HostShader& s)
{
    return std::filesystem::path(log::Directory()) / "shadercache" / std::format("{:016X}_{}.bin", s.hash, s.pixel ? "ps" : "vs");
}

// The cache entries on disk, listed once: a shader first met where it is drawn is
// rarely cached, and looking for its file on the render thread can cost milliseconds on some disks while the cache
// thread is creating files in the same directory. Entries written since are added as they are queued.
std::mutex& s_cacheIndexMutex = *new std::mutex;
std::unordered_set<uint64_t>& s_cacheIndex = *new std::unordered_set<uint64_t>;

uint64_t CacheIndexKey(uint64_t hash, bool pixel)
{
    return hash * 2 + (pixel ? 1 : 0);
}

bool InCacheIndex(const HostShader& s)
{
    static std::once_flag listed;
    std::call_once(listed, [] {
        std::unordered_set<uint64_t> found;
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(std::filesystem::path(log::Directory()) / "shadercache", ec))
        {
            // <16 hex digits>_vs.bin / _ps.bin
            const std::string name = platform::PathUtf8(entry.path().filename());
            if (name.size() != 23 || (name.substr(16) != "_vs.bin" && name.substr(16) != "_ps.bin"))
                continue;
            char* end = nullptr;
            const uint64_t hash = std::strtoull(name.substr(0, 16).c_str(), &end, 16);
            if (end && *end == 0)
                found.insert(CacheIndexKey(hash, name[17] == 'p'));
        }
        std::lock_guard lock(s_cacheIndexMutex);
        s_cacheIndex.merge(found);
        log::Info("Shader cache: {} entries on disk", s_cacheIndex.size());
    });
    std::lock_guard lock(s_cacheIndexMutex);
    return s_cacheIndex.contains(CacheIndexKey(s.hash, s.pixel));
}

bool LoadCached(HostShader& s, bool& usesMemExport)
{
    if (!InCacheIndex(s))
        return false;
    const std::filesystem::path path = CachePath(s);
    std::error_code ec;
    const uintmax_t bytes = std::filesystem::file_size(path, ec);
    std::ifstream f(path, std::ios::binary);
    if (ec || !f || bytes % 4)
        return false;
    std::vector<uint32_t> words(bytes / 4);
    if (!f.read(reinterpret_cast<char*>(words.data()), std::streamsize(bytes)))
        return false;
    size_t at = 0;
    auto get = [&](uint32_t& v) { if (at >= words.size()) return false; v = words[at++]; return true; };
    uint32_t magic, keyLow, keyHigh, flags, inputs, overrides, spirvWords;
    if (!get(magic) || magic != kCacheMagic || !get(keyLow) || !get(keyHigh) ||
        (uint64_t(keyHigh) << 32 | keyLow) != CacheKey() || !get(s.specConstantsMask) || !get(s.indexedFetchStreams) ||
        !get(s.constantFirst) || !get(s.constantEnd) || s.constantEnd > 256 || !get(flags) || !get(inputs) || inputs > 64)
        return false;
    usesMemExport = flags & 1;
    s.vertexInputs.clear();
    for (uint32_t i = 0; i < inputs; i++)
    {
        uint32_t v[7];
        for (uint32_t& x : v)
            if (!get(x))
                return false;
        s.vertexInputs.push_back({ v[0], v[1], int32_t(v[2]), v[3], v[4], v[5] != 0, v[6] != 0 });
    }
    if (!get(overrides) || overrides > 16)
        return false;
    s.samplerOverrides.resize(overrides);
    for (uint32_t& o : s.samplerOverrides)
        if (!get(o))
            return false;
    if (!get(spirvWords) || spirvWords == 0 || at + spirvWords != words.size())
        return false;
    s.spirv.assign(words.begin() + at, words.end());
    return true;
}

// Cache files are written by a thread of their own: creating a file can take up to ~40 ms
// on the render thread, longer than translating the shader. Entries are written under a temporary name and
// renamed, so a file is complete or absent, `skipIfExists` leaves an existing one alone.
struct FileWrite
{
    std::filesystem::path path;
    std::vector<uint8_t> bytes;
    bool skipIfExists = false;
};
std::mutex& s_writeMutex = *new std::mutex;
std::condition_variable& s_writeCv = *new std::condition_variable;
std::deque<FileWrite>& s_writes = *new std::deque<FileWrite>;

void WriteFileAsync(FileWrite write)
{
    static const bool started = [] {
        std::thread([] {
            platform::SetCurrentThreadPriority(platform::kPriorityBelowNormal);
            for (;;)
            {
                FileWrite w;
                {
                    std::unique_lock lock(s_writeMutex);
                    s_writeCv.wait(lock, [] { return !s_writes.empty(); });
                    w = std::move(s_writes.front());
                    s_writes.pop_front();
                }
                std::error_code ec;
                if (w.skipIfExists && std::filesystem::exists(w.path, ec))
                    continue;
                std::filesystem::create_directories(w.path.parent_path(), ec);
                const std::filesystem::path temp = std::filesystem::path(w.path).concat(".tmp");
                {
                    std::ofstream f(temp, std::ios::binary);
                    f.write(reinterpret_cast<const char*>(w.bytes.data()), std::streamsize(w.bytes.size()));
                    if (!f)
                        continue;
                }
                std::filesystem::rename(temp, w.path, ec);
            }
        }).detach();
        return true;
    }();
    (void)started;
    std::lock_guard lock(s_writeMutex);
    s_writes.push_back(std::move(write));
    s_writeCv.notify_one();
}

void StoreCached(const HostShader& s, bool usesMemExport)
{
    std::vector<uint32_t> words = { kCacheMagic, uint32_t(CacheKey()), uint32_t(CacheKey() >> 32), s.specConstantsMask,
        s.indexedFetchStreams, s.constantFirst, s.constantEnd, usesMemExport ? 1u : 0u, uint32_t(s.vertexInputs.size()) };
    for (const ShaderVertexInput& in : s.vertexInputs)
        words.insert(words.end(), { in.location, in.fetchSlot, uint32_t(in.offset), in.stride, in.format, uint32_t(in.isSigned), uint32_t(in.isInteger) });
    words.push_back(uint32_t(s.samplerOverrides.size()));
    words.insert(words.end(), s.samplerOverrides.begin(), s.samplerOverrides.end());
    words.push_back(uint32_t(s.spirv.size()));
    words.insert(words.end(), s.spirv.begin(), s.spirv.end());
    FileWrite w{ CachePath(s) };
    {
        std::lock_guard lock(s_cacheIndexMutex);
        s_cacheIndex.insert(CacheIndexKey(s.hash, s.pixel));
    }
    w.bytes.resize(words.size() * 4);
    std::memcpy(w.bytes.data(), words.data(), w.bytes.size());
    WriteFileAsync(std::move(w));
}

ShaderCompileStats s_compileStats;   // s_mutex. Shaders compiled where they were needed (not warmed up)

// Shader warm-up: the shader objects AC6 creates from its shader package as it loads
// it, waiting to be translated and compiled (or loaded from the cache) by background threads before
// their first draw. All under s_mutex. The condition variable signals new jobs and finished ones. Never
// destroyed: the workers may still wait on them as the process exits.
struct WarmJob
{
    uint64_t hash = 0;
    bool pixel = false;
    uint32_t variant = 0;
    std::vector<uint8_t> container;
};
std::deque<WarmJob>& s_warmQueue = *new std::deque<WarmJob>;
std::unordered_set<uint64_t>& s_warmSeen = *new std::unordered_set<uint64_t>;      // queued at some point
std::unordered_set<uint64_t>& s_warmRunning = *new std::unordered_set<uint64_t>;   // being compiled now
std::condition_variable& s_warmCv = *new std::condition_variable;
uint64_t s_warmDone = 0, s_warmCompiled = 0;

// The game's EDRAM un-swizzles: pixel shaders that read an image of a 1x surface aliasing a
// promoted 4x one (or the reverse) through frac(abs(vPos) * 0.25) -> the middle two of each 4 pixels swapped.
// Their fetches of that image (the slots below) sample 1:1 instead, since the alias copy doesn't swizzle: the
// scene restore (colour and depth), a mask pass, and the world compositor (its slot 4, its other fetches are plain).
uint32_t EdramUnswizzleSlots(const HostShader& s)
{
    if (!s.pixel || EdramSwizzle())
        return 0;
    switch (s.idHash)
    {
    case 0x1F54F7BFE2B54D54ull: return (1u << 1) | (1u << 2);
    case 0x073B5B1A70553837ull: return 1u << 0;
    case 0x8E36B699AF21554Aull: return 1u << 4;
    default: return 0;
    }
}

// AC6_LOG_XXH3=1: each shader's microcode hashed as Xenia keys it (XXH3-64 of the microcode as in guest memory), so
// a shader can be matched with Xenia's.
void LogMicrocodeXxh3(const HostShader& s)
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_LOG_XXH3"); return v && *v == '1'; }();
    if (!enabled || s.container.size() < sizeof(xenos::ShaderContainer))
        return;
    const uint32_t virtualSize = reinterpret_cast<const xenos::ShaderContainer*>(s.container.data())->virtualSize;
    if (virtualSize >= s.container.size())
        return;
    log::Info("Shader {:016X} ({}): microcode XXH3 {:016X}", s.hash, s.pixel ? "pixel" : "vertex",
        XXH3_64bits(s.container.data() + virtualSize, s.container.size() - virtualSize));
}

// Translation options set while running: High quality DoF (AC6_DOF_FIX, off by default), round bloom (AC6_WIDE_BLURS,
// on), HDR from the scene (the game's grade pass keeps values above 1) and the game's lighting in float (the shaders
// holding the game's code constants, 240/255 and 241/255, keep their colours above 1 and decode without the emitter
// codes). A pixel shader they change (OptionUse) is looked up per option state: its hash carries the state, so a change
// translates the other version at its next draw (and new pipelines for it: a short hitch) and keeps both. The disk
// cache's key holds no option, so shaders using the first three skip it. Those using only the lighting in float are
// cached with its state in their hash (Translate).
constexpr uint32_t kOptDof = 1, kOptBloom = 2, kOptHdrScene = 4, kOptSceneFloat = 8;

uint32_t ShaderOptions()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<uint32_t> value{ UINT32_MAX };
    static std::atomic<bool> hdrScene{ false };   // (follows the output mode too, not only the settings)
    const bool hdr = HdrSceneActive();
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now ||
        hdrScene.load(std::memory_order_relaxed) != hdr)
    {
        hdrScene.store(hdr, std::memory_order_relaxed);
        auto on = [](const char* env, bool byDefault) {
            const std::string v = settings::Get(env);
            return v.empty() ? byDefault : v != "0" && v != "false";
        };
        const uint32_t options = (on("AC6_DOF_FIX", false) ? kOptDof : 0u) | (on("AC6_WIDE_BLURS", true) ? kOptBloom : 0u) |
            (hdr ? kOptHdrScene : 0u) | (hdr && SceneFloatActive() ? kOptSceneFloat : 0u);
        if (value.exchange(options, std::memory_order_relaxed) != options)
            s_stateEpoch.fetch_add(1, std::memory_order_release);   // the per-thread memo looks again
        generation.store(now, std::memory_order_release);
    }
    return value.load(std::memory_order_relaxed);
}

// The DoF passes' microcode XXH3 (below) and the bloom blurs' hashes (SquareOffsetSlots).
uint64_t MicrocodeXxh3(const uint8_t* container, size_t size)
{
    if (size < sizeof(xenos::ShaderContainer))
        return 0;
    const uint32_t virtualSize = reinterpret_cast<const xenos::ShaderContainer*>(container)->virtualSize;
    return virtualSize < size ? XXH3_64bits(container + virtualSize, size - virtualSize) : 0;
}

uint32_t OptionUseBase(const uint8_t* container, size_t size, uint64_t plainHash)
{
    if (plainHash == 0x818FCD6341FBF2C8ull || plainHash == 0x3B7049BE4C307E6Aull)
        return kOptBloom;
    // The game's final grade (levels, gamma, gain, vignette on the canvas, its saturate is the game's
    // clip): with HDR from the scene it keeps values above 1.
    if (plainHash == 0x6C6D041CAB12E202ull)
        return kOptHdrScene;
    switch (MicrocodeXxh3(container, size))
    {
    case 0x85D733927E3BBA9Cull: case 0xD050DFA6F58567F6ull: case 0x6328F9C40913C82Cull: case 0x5BD20F9D0D911687ull:
        return kOptDof;
    default: return 0;
    }
}

uint32_t OptionUse(const uint8_t* container, size_t size, uint64_t plainHash, bool pixel)
{
    if (!pixel || HlslBackend())
        return 0;
    uint32_t use = OptionUseBase(container, size, plainHash);
    // The exposure ladder's first stage and the bright pass read the 320x180 downsample, which the downsample
    // (23E9D611FA16091E) encodes as the game does, emitter codes included, also in float: they keep the game's decoding
    // (read as values, the sun's 94 would become 1 and its glare would halve).
    if (plainHash == 0x556E84DFDE2A6164ull || plainHash == 0x15E29341C450CAE5ull)
        return use;
    // The game's lighting in float: its code scale 240/255 or its decoders' emitter test 241/255 among the
    // shader's data (big-endian, as the container holds them).
    static constexpr uint8_t kScale[4] = { 0x3F, 0x70, 0xF0, 0xF1 }, kTest[4] = { 0x3F, 0x71, 0x71, 0x71 };
    const uint8_t* end = container + size;
    if (std::search(container, end, kScale, kScale + 4) != end || std::search(container, end, kTest, kTest + 4) != end)
        use |= kOptSceneFloat;
    return use;
}

// The game's code constants a shader holds: the lit encode's 10/255, the decoders' 241/255.
void NoteCodeKinds(HostShader& shader)
{
    static constexpr uint8_t kEncode[4] = { 0x3D, 0x20, 0xA0, 0xA1 }, kTest[4] = { 0x3F, 0x71, 0x71, 0x71 };
    const uint8_t* begin = shader.container.data();
    const uint8_t* end = begin + shader.container.size();
    shader.litEncode = std::search(begin, end, kEncode, kEncode + 4) != end;
    shader.decoder = std::search(begin, end, kTest, kTest + 4) != end;
}

// A shader's hashes for the options set now (GetShaderLocked and the warm-up alike): the lookup hash carries the state of
// the options it uses (bits 58-62), idHash, what the renderer's lists know it by (without the float lighting's state).
struct OptionHashes { uint64_t hash, idHash; uint32_t optionUse, options; };
OptionHashes ApplyOptions(const uint8_t* container, size_t size, uint64_t plainHash, bool pixel, uint32_t options)
{
    OptionHashes h{ plainHash, plainHash, OptionUse(container, size, plainHash, pixel), 0 };
    h.options = options & h.optionUse;
    if (h.optionUse & ~kOptSceneFloat)
        h.idHash ^= uint64_t(8u | (h.options & ~kOptSceneFloat)) << 59;
    if (h.optionUse)
        h.hash ^= uint64_t(16u | h.options) << 58;
    return h;
}

// The depth of field above 1x (High quality DoF): its passes by microcode XXH3:
// the two circle of confusion (CoC) pre-blurs snap to guest texels, the two 13-tap gathers (by this renderer's hashes 3C12F298E423B5B3
// horizontal, AFE2B74E7691A45D vertical) are densified along their axis.
// With AC6_DOF_FIX off (the default) they are translated unchanged.
struct DofSlots { uint32_t snap = 0, x = 0, y = 0; };

// Screen blurs with texel offsets in the instructions: on a wide output their offsets are made square
// on screen. The bloom's horizontal 15-tap Gaussian (160x90) and its 13-tap diamond blur (320x180). AC6_WIDE_BLURS=0: off.
uint32_t SquareOffsetSlots(const HostShader& s)
{
    // The bloom blurs (OptionUse) when round bloom is on (the state it is translated for: s.options).
    return (s.optionUse & kOptBloom) && (s.options & kOptBloom) ? ~0u : 0u;
}
DofSlots DofFixSlots(const HostShader& s)
{
    // Off by default (High quality DoF), the state it is translated for.
    if (!(s.optionUse & kOptDof) || !(s.options & kOptDof))
        return {};
    switch (MicrocodeXxh3(s.container.data(), s.container.size()))
    {
    case 0x85D733927E3BBA9Cull: case 0xD050DFA6F58567F6ull: return { ~0u, 0, 0 };
    case 0x6328F9C40913C82Cull: return { 0, ~0u, 0 };
    case 0x5BD20F9D0D911687ull: return { 0, 0, ~0u };
    default: return {};
    }
}

// A float constant's register by its name in the shader's constant table, when it has at least `count` registers.
// UINT32_MAX: none.
uint32_t FloatRegister(const HostShader& s, std::string_view wanted, uint32_t count)
{
    const size_t size = s.container.size();
    if (size < sizeof(xenos::ShaderContainer))
        return UINT32_MAX;
    const auto container = reinterpret_cast<const xenos::ShaderContainer*>(s.container.data());
    const uint64_t tableAt = uint64_t(container->constantTableOffset) + 4;   // past ConstantTableContainer::size
    if (tableAt + sizeof(xenos::ConstantTable) > size)
        return UINT32_MAX;
    const uint8_t* table = s.container.data() + tableAt;
    const auto header = reinterpret_cast<const xenos::ConstantTable*>(table);
    for (uint32_t i = 0; i < header->constants; i++)
    {
        const uint64_t infoAt = uint64_t(header->constantInfo) + i * sizeof(xenos::ConstantInfo);
        if (tableAt + infoAt + sizeof(xenos::ConstantInfo) > size)
            break;
        const auto info = reinterpret_cast<const xenos::ConstantInfo*>(table + infoAt);
        if (info->registerSet != xenos::RegisterSet::Float4 || info->registerCount < count || tableAt + info->name >= size)
            continue;
        const char* name = reinterpret_cast<const char*>(table + info->name);
        if (std::string_view(name, strnlen(name, size - tableAt - info->name)) == wanted)
            return info->registerIndex;
    }
    return UINT32_MAX;
}

// The matrix a vertex shader transforms its position with (TAA's motion vectors), by the names in its
// constant table: the object shaders' ACE_mWorldViewProj (c218: the object's world x view x projection), else the
// scenery's WorldViewProjectionMatrix (c120), else ViewProjectionMatrix (c116: world-space vertices). Row vectors, four
// registers. UINT32_MAX: none of them.
uint32_t PositionMatrix(const HostShader& s)
{
    const size_t size = s.container.size();
    if (s.pixel || size < sizeof(xenos::ShaderContainer))
        return UINT32_MAX;
    const auto container = reinterpret_cast<const xenos::ShaderContainer*>(s.container.data());
    const uint64_t tableAt = uint64_t(container->constantTableOffset) + 4;   // past ConstantTableContainer::size
    if (tableAt + sizeof(xenos::ConstantTable) > size)
        return UINT32_MAX;
    const uint8_t* table = s.container.data() + tableAt;
    const auto header = reinterpret_cast<const xenos::ConstantTable*>(table);
    constexpr std::string_view kNames[3] = { "ACE_mWorldViewProj", "WorldViewProjectionMatrix", "ViewProjectionMatrix" };
    uint32_t found[3] = { UINT32_MAX, UINT32_MAX, UINT32_MAX };
    for (uint32_t i = 0; i < header->constants; i++)
    {
        const uint64_t infoAt = uint64_t(header->constantInfo) + i * sizeof(xenos::ConstantInfo);
        if (tableAt + infoAt + sizeof(xenos::ConstantInfo) > size)
            break;
        const auto info = reinterpret_cast<const xenos::ConstantInfo*>(table + infoAt);
        if (info->registerSet != xenos::RegisterSet::Float4 || info->registerCount < 4 || tableAt + info->name >= size)
            continue;
        const char* name = reinterpret_cast<const char*>(table + info->name);
        const std::string_view view(name, strnlen(name, size - tableAt - info->name));
        for (uint32_t k = 0; k < 3; k++)
            if (view == kNames[k])
                found[k] = info->registerIndex;
    }
    for (uint32_t k = 0; k < 3; k++)
        if (found[k] != UINT32_MAX)
            return found[k];
    return UINT32_MAX;
}

// The game's light inputs: float4 constants by name, their colour (xyz) scaled per kind by the
// relighting (renderer.cpp Relight). Names from the game's shaders' constant tables.
void NoteLightConstants(HostShader& s)
{
    s.lightConstants.clear();
    s.cloud = false;
    const size_t size = s.container.size();
    if (size < sizeof(xenos::ShaderContainer))
        return;
    const auto container = reinterpret_cast<const xenos::ShaderContainer*>(s.container.data());
    const uint64_t tableAt = uint64_t(container->constantTableOffset) + 4;
    if (tableAt + sizeof(xenos::ConstantTable) > size)
        return;
    const uint8_t* table = s.container.data() + tableAt;
    const auto header = reinterpret_cast<const xenos::ConstantTable*>(table);
    // Kinds:
    //   0  objects' sun: the aircraft's and objects' directional light and its specular
    //   1  objects' ambient
    //   2  clouds' sun
    //   3  clouds' ambient
    //   4  particles' sun (smoke and effects, lit in their vertex shaders)
    //   5  particles' ambient
    //   6  fog
    //   7  the map's specular: the city's and the sea's glints (the map's diffuse light is baked)
    //   8  objects' light cap, ACE_vCommonParam3.x: the aircraft's pixel shaders and the objects' lit vertex shaders
    //      hold the light's brightest channel to it (light x min(1, cap / brightest)). Scaled with the objects' sun, x
    //      only.
    //   9  objects' specular colour: its own factor, the objects' sun's by default
    //   10 clouds' rim light, SpriteLightVector_LightAngle.w (how far the view faces the sun): the cloud puffs' pixel
    //      shader 5776EB8756E408E9 and the flat clouds' vertex shader 8BFEF687FB1D2665 multiply their outline colour by
    //      it and read it nowhere else. Its xyz, the light's direction, is left alone.
    static const std::pair<std::string_view, uint8_t> kNames[] = {
        { "ACE_vDirCol1", 0 }, { "ACE_vSpecularCol", 9 }, { "ACE_vAmbCol", 1 }, { "ACE_vCommonParam3", 8 },
        { "CloudColorDiffuse_NONE", 2 }, { "CloudColorAmbient_NONE", 3 }, { "CloudColorUpperAmbient_NONE", 3 },
        { "CloudColorLowerAmbient_NONE", 3 },
        { "DiffuseColor_Altitude", 4 }, { "AmbientUpperColor_Altitude", 5 }, { "AmbientLowerColor_Height", 5 },
        { "ACE_vFogColor", 6 }, { "FogColor_Param", 6 }, { "CloudFogColor_Param", 6 }, { "CustomFogColor_Param", 6 },
        { "ACE_vMapPartsSpecularCol", 7 }, { "ACE_vMapMeshSpecularCol", 7 },
        { "SpriteLightVector_LightAngle", 10 },
    };
    for (uint32_t i = 0; i < header->constants; i++)
    {
        const uint64_t infoAt = uint64_t(header->constantInfo) + i * sizeof(xenos::ConstantInfo);
        if (tableAt + infoAt + sizeof(xenos::ConstantInfo) > size)
            break;
        const auto info = reinterpret_cast<const xenos::ConstantInfo*>(table + infoAt);
        if (info->registerSet != xenos::RegisterSet::Float4 || tableAt + info->name >= size)
            continue;
        const char* name = reinterpret_cast<const char*>(table + info->name);
        const std::string_view view(name, strnlen(name, size - tableAt - info->name));
        if (view.starts_with("Cloud"))
            s.cloud = true;
        for (const auto& [n, kind] : kNames)
            if (view == n)
                s.lightConstants.push_back({ uint16_t(info->registerIndex), kind });
    }
}

bool Translate(HostShader& s, bool createModule = true)
{
    LogMicrocodeXxh3(s);
    s.drawFlags = !HlslBackend();
    // As the SPIR-V backend declares it (SpirvRecompiler::DeclareInterface): pixel shaders writing colour 0 at most.
    if (s.pixel && s.drawFlags && s.container.size() >= sizeof(xenos::ShaderContainer))
    {
        const auto container = reinterpret_cast<const xenos::ShaderContainer*>(s.container.data());
        if (uint64_t(container->shaderOffset) + sizeof(xenos::PixelShader) <= s.container.size())
        {
            const uint32_t outputs = reinterpret_cast<const xenos::PixelShader*>(s.container.data() + container->shaderOffset)->outputs;
            s.motionOutput = !(outputs & (xenos::PIXEL_SHADER_OUTPUT_COLOR1 | xenos::PIXEL_SHADER_OUTPUT_COLOR2 | xenos::PIXEL_SHADER_OUTPUT_COLOR3));
        }
    }
    s.positionMatrix = PositionMatrix(s);
    if (!s.pixel)
    {
        s.inverseViewProjectionMatrix = FloatRegister(s, "InverseWorldViewProjectionMatrix", 4);
        s.inverseViewMatrix = FloatRegister(s, "InverseViewMatrix", 2);
        s.viewMatrix = FloatRegister(s, "ViewMatrix", 4);
        s.worldInverseViewMatrix = FloatRegister(s, "WorldInverseViewMatrix", 2);
    }
    NoteLightConstants(s);
    s.seaReflection = s.pixel && (s.idHash == 0xCDB30131360ACAAAull || s.idHash == 0x3ADEF3EC4BA4640Dull);   // ground and sea
    s.aoTerrain = s.seaReflection || (s.pixel && s.idHash == 0x1F6C07D3A32CB1ACull);   // the terrain: no AO on it
    s.source = SourceName(s.container);
    // (The flat cloud layers, Effect/BGFlatCloud and HiFlatCloud, live in the Effect folder too but are the sky's.)
    s.effect = s.source.starts_with("Effect/") && s.source.find("Cloud") == std::string::npos;
    bool cachedMemExport = false;
    // Shaders with another option's state in their hash are not cached. The game's lighting in float marks dozens of
    // lit and effect shaders: those are cached, their hash carrying its state, else every start would translate them
    // at their first use (not with AC6_SCENE_FLOAT_MIN=0).
    const bool cacheable = CacheEnabled() && !(s.optionUse & ~kOptSceneFloat) && !((s.optionUse & kOptSceneFloat) && KeepWhiteMin());
    if (cacheable && LoadCached(s, cachedMemExport))
    {
        s.ok = true;
        if (cachedMemExport)
            log::Warn("Shader {:016X} ({}) uses memory export, which isn't translated", s.hash, s.pixel ? "pixel" : "vertex");
        if (createModule && g_video.Ready())
            MakeShaderModule(s);
        static std::atomic<uint32_t> loaded{ 0 };
        if (const uint32_t n = ++loaded; n % 500 == 1)
            log::Info("Shader cache: {} shader(s) loaded so far", n);
        return false;
    }
    auto start = std::chrono::steady_clock::now();
    if (!HlslBackend())
    {
        xenos::SpirvRecompiler recompiler;
        recompiler.allowMaskedExecution = MaskedExecEnabled();
        recompiler.identityFetchSlots = EdramUnswizzleSlots(s);
        const DofSlots dof = DofFixSlots(s);
        recompiler.guestTexelSnapSlots = dof.snap;
        recompiler.densifyXSlots = dof.x;
        recompiler.densifyYSlots = dof.y;
        recompiler.squareOffsetSlots = SquareOffsetSlots(s);
        // The map's pixel shaders (the ground and the sea): texture 0 is an atlas page.
        recompiler.tileAtlasSlots = s.pixel && (s.idHash == 0xCDB30131360ACAAAull || s.idHash == 0x3ADEF3EC4BA4640Dull) ? 1u : 0u;
        recompiler.seaReflection = s.seaReflection;
        recompiler.rangeSaturate = (s.optionUse & kOptHdrScene) && (s.options & kOptHdrScene);   // HDR from the scene
        recompiler.sceneFloat = (s.optionUse & kOptSceneFloat) && (s.options & kOptSceneFloat);   // lighting in float
        recompiler.liftWhiteMin = !KeepWhiteMin();   // AC6_SCENE_FLOAT_MIN
        // The sun's shaders (its disc and glow): their 8-bit
        // emitter codes decoded at the draw when the scene is float (the per-draw flag, no option, the SDR path unchanged).
        recompiler.emitterDecode = s.pixel && (s.idHash == 0xBE0A9DB65D5CB28Cull || s.idHash == 0x80390F4960C3D4A9ull);
        // The game's light inputs, scaled where the shader reads them.
        recompiler.lightRegisters.clear();
        for (const auto& lc : s.lightConstants)
            recompiler.lightRegisters[lc.reg] = lc.kind;
        recompiler.uniformConstants = UniformConstants();
        recompiler.recompile(s.container.data(), s.variant);
        // AC6_SCENE_FLOAT_LOG=1: the clamps the game's lighting in float lifted, per shader.
        static const bool sceneFloatLog = [] { const char* v = std::getenv("AC6_SCENE_FLOAT_LOG"); return v && *v == '1'; }();
        if (sceneFloatLog && recompiler.sceneFloat)
            log::Info("Scene float: ps {:016X}: {} clamp(s) lifted", s.idHash, recompiler.unclampedCount());
        s.specConstantsMask = recompiler.specConstantsMask;
        s.indexedFetchStreams = recompiler.indexedFetchStreams;
        s.samplerOverrides = recompiler.samplerOverrides;
        s.textureSlots = recompiler.textureSlots;
        s.constantFirst = recompiler.constantEnd ? recompiler.constantFirst : 0;
        s.constantEnd = recompiler.constantEnd;
        if (recompiler.usesMemExport)
            log::Warn("Shader {:016X} ({}) uses memory export, which isn't translated", s.hash, s.pixel ? "pixel" : "vertex");
        if (!recompiler.warnings.empty())
            log::Warn("Shader {:016X} ({}): {}", s.hash, s.pixel ? "pixel" : "vertex", recompiler.warnings.substr(0, 400));
        for (const xenos::VertexInput& in : recompiler.vertexInputs)
            s.vertexInputs.push_back({ in.location, in.fetchSlot, in.offset, in.stride, in.format, in.isSigned, in.isInteger });
        s.spirv = std::move(recompiler.spirv);
        s.ok = !s.spirv.empty();
        if (s.ok && createModule && g_video.Ready())
            MakeShaderModule(s);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        log::Info("Shader {:016X} ({}): {} bytes of microcode -> {} bytes SPIR-V in {:.2f} ms", s.hash, s.pixel ? "pixel" : "vertex",
            s.container.size(), s.spirv.size() * 4, ms);
        if (DumpEnabled())
            Dump(s, "", "");
        if (s.ok && cacheable)
            StoreCached(s, recompiler.usesMemExport);
        return true;
    }
    xenos::ShaderRecompiler recompiler;
    recompiler.allowMaskedExecution = MaskedExecEnabled();
    recompiler.recompile(s.container.data(), kXenosShaderCommon, s.variant);
    ApplyPatches(s.hash, recompiler.out);
    s.specConstantsMask = recompiler.specConstantsMask;
    s.indexedFetchStreams = recompiler.indexedFetchStreams;
    s.samplerOverrides = recompiler.samplerOverrides;
    if (recompiler.usesMemExport)
        log::Warn("Shader {:016X} ({}) uses memory export, which isn't translated", s.hash, s.pixel ? "pixel" : "vertex");
    for (const xenos::VertexInput& in : recompiler.vertexInputs)
        s.vertexInputs.push_back({ in.location, in.fetchSlot, in.offset, in.stride, in.format, in.isSigned, in.isInteger });

    std::string errors;
    s.ok = CompileHlslToSpirv(recompiler.out, s.pixel, s.spirv, errors);
    if (s.ok && createModule && g_video.Ready())
        MakeShaderModule(s);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (s.ok)
        log::Info("Shader {:016X} ({}): {} bytes of microcode -> {} bytes SPIR-V in {:.1f} ms", s.hash, s.pixel ? "pixel" : "vertex",
            s.container.size(), s.spirv.size() * 4, ms);
    else
        log::Error("Shader {:016X} ({}) failed to compile: {}", s.hash, s.pixel ? "pixel" : "vertex", errors.substr(0, 400));
    if (DumpEnabled() || !s.ok)
        Dump(s, recompiler.out, errors);
    if (s.ok && cacheable)
        StoreCached(s, recompiler.usesMemExport);
    return true;
}

bool WarmupEnabled()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_SHADER_WARMUP"); return !v || *v != '0'; }();
    return enabled;
}

void WarmWorker()
{
    std::unique_lock lock(s_mutex);
    for (;;)
    {
        s_warmCv.wait(lock, [] { return !s_warmQueue.empty(); });
        WarmJob job = std::move(s_warmQueue.front());
        s_warmQueue.pop_front();
        if (s_byHash.contains(job.hash))
            continue;
        s_warmRunning.insert(job.hash);
        lock.unlock();
        auto shader = std::make_unique<HostShader>();
        // As GetShaderLocked will look it up: with the options set now (the lit and effect shaders carry one).
        const OptionHashes oh = ApplyOptions(job.container.data(), job.container.size(), job.hash, job.pixel, ShaderOptions());
        shader->hash = oh.hash;
        shader->idHash = oh.idHash;
        shader->codeScale = (oh.optionUse & kOptSceneFloat) != 0;
        shader->optionUse = oh.optionUse;
        shader->options = oh.options;
        shader->pixel = job.pixel;
        shader->variant = job.variant;
        shader->container = std::move(job.container);
        NoteCodeKinds(*shader);
        const bool compiled = Translate(*shader, false);
        // Its pipeline library too, where pipelines are linked from them.
        if (shader->drawFlags)
            RendererPrepareShaderLibrary(*shader);
        // The pipeline list's pipelines made of it are built ahead of their draws: its module now.
        if (shader->ok && !shader->shader && g_video.Ready() && RendererPipelineListWants(oh.hash))
            MakeShaderModule(*shader);
        lock.lock();
        s_warmRunning.erase(job.hash);
        if (auto& slot = s_byHash[oh.hash]; !slot)
        {
            slot = std::move(shader);
            RendererShaderReady(*slot);
        }
        s_warmDone++;
        s_warmCompiled += compiled;
        static uint64_t logged = 0;
        if (s_warmQueue.empty() && s_warmRunning.empty() && s_warmDone >= logged + 256)
        {
            logged = s_warmDone;
            log::Info("Shader warm-up: {} shader(s) ready so far ({} compiled, the rest from the cache)", s_warmDone, s_warmCompiled);
        }
        s_warmCv.notify_all();
    }
}

// Guest containers of the shaders the warm-up can't see coming: vertex shaders D3D
// patches for the bound vertex declaration (their objects hold templates) and other records. Each one
// compiled where a draw needed it is kept in <exe dir>/shadercache/guest/<hash>_<vs|ps>.bin ([magic,
// pixel, variant][container]), independent of the translator, and every later warm-up queues them too:
// after a translator change they are rebuilt at boot, not in the middle of a mission.
constexpr uint32_t kGuestMagic = 0x47364341;   // "AC6G"

std::filesystem::path GuestDirectory()
{
    return std::filesystem::path(log::Directory()) / "shadercache" / "guest";
}

void StoreGuestContainer(const HostShader& s)
{
    if (!CacheEnabled() || !WarmupEnabled())
        return;
    FileWrite w{ GuestDirectory() / std::format("{:016X}_{}.bin", s.hash, s.pixel ? "ps" : "vs"), {}, true };
    const uint32_t header[3] = { kGuestMagic, s.pixel ? 1u : 0u, s.variant };
    w.bytes.resize(sizeof(header) + s.container.size());
    std::memcpy(w.bytes.data(), header, sizeof(header));
    std::memcpy(w.bytes.data() + sizeof(header), s.container.data(), s.container.size());
    WriteFileAsync(std::move(w));
}

// Queues the stored guest containers (on a thread of its own: the directory is read at boot).
void QueueGuestContainers()
{
    std::error_code ec;
    uint32_t queued = 0;
    for (const auto& entry : std::filesystem::directory_iterator(GuestDirectory(), ec))
    {
        if (entry.path().extension() != ".bin")
            continue;
        std::ifstream f(entry.path(), std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        uint32_t header[3];
        if (bytes.size() <= sizeof(header))
            continue;
        std::memcpy(header, bytes.data(), sizeof(header));
        if (header[0] != kGuestMagic)
            continue;
        WarmJob job{ 0, header[1] != 0, header[2], std::vector<uint8_t>(bytes.begin() + sizeof(header), bytes.end()) };
        job.hash = ShaderHash(job.container.data(), job.container.size(), job.pixel) ^ (uint64_t(job.variant) << 56);
        std::lock_guard lock(s_mutex);
        if (s_byHash.contains(job.hash) || !s_warmSeen.insert(job.hash).second)
            continue;
        s_warmQueue.push_back(std::move(job));
        s_warmCv.notify_one();
        queued++;
    }
    if (queued)
        log::Info("Shader warm-up: {} stored guest shader(s) queued", queued);
}

} // namespace

// The translator's backend: SPIR-V written directly, or HLSL compiled by DXC
// (AC6_SHADER_BACKEND=hlsl, also whenever AC6_SHADER_PATCH edits the generated HLSL).
bool HlslBackend()
{
    static const bool hlsl = [] {
        const char* v = std::getenv("AC6_SHADER_BACKEND");
        const char* patch = std::getenv("AC6_SHADER_PATCH");
        return (v && std::strcmp(v, "hlsl") == 0) || (patch && *patch);
    }();
    return hlsl;
}

void MakeShaderModule(HostShader& s)
{
    struct Module { std::vector<uint32_t> spirv; std::shared_ptr<plume::RenderShader> shader; };
    static std::mutex mutex;
    static FastMap<uint64_t, std::vector<Module>> modules;   // by code hash (the SPIR-V compared too)
    const uint64_t hash = Hash(reinterpret_cast<const uint8_t*>(s.spirv.data()), s.spirv.size() * 4) ^ (s.pixel ? 1 : 0);
    s.codeHash = hash;
    static const bool shared = [] { const char* v = std::getenv("AC6_SHARED_MODULES"); return !v || *v != '0'; }();
    if (!shared)
        s.codeHash = 0;
    std::lock_guard lock(mutex);
    auto& list = modules[hash];
    for (const Module& m : list)
        if (shared && m.spirv == s.spirv)
        {
            s.shader = m.shader;
            return;
        }
    // AC6_SHADER_SALT=<n> (tests): the module's generator word (its header's third, read by nothing)
    // changed by n, so the driver's own cache knows none of the shaders or pipelines: a cold driver cache on demand.
    static const uint32_t salt = [] { const char* v = std::getenv("AC6_SHADER_SALT"); return v ? uint32_t(std::strtoul(v, nullptr, 0)) : 0u; }();
    std::shared_ptr<plume::RenderShader> shader;
    if (salt && s.spirv.size() > 5)
    {
        std::vector<uint32_t> salted = s.spirv;
        salted[2] ^= salt;
        shader = g_video.device->createShader(salted.data(), salted.size() * 4, "main", plume::RenderShaderFormat::SPIRV);
    }
    else
        shader = g_video.device->createShader(s.spirv.data(), s.spirv.size() * 4, "main", plume::RenderShaderFormat::SPIRV);
    // AC6_GPU_NAMES=1: the module named after the first shader it was made for (others with the
    // same SPIR-V share it).
    static const bool names = [] { const char* v = std::getenv("AC6_GPU_NAMES"); return v && *v == '1'; }();
    if (names && shader)
        shader->setName(std::format("{} {:016X}", s.pixel ? "ps" : "vs", s.hash));
    list.push_back({ s.spirv, shader });
    s.shader = std::move(shader);
}

void NoteShaderObjectCreated(uint32_t object)
{
    const uint32_t type = U32(object) & 0xF;   // D3D resource type (sub_821E40B8): 6 vertex shader, 7 pixel shader
    if ((type != 6 && type != 7) || !WarmupEnabled())
        return;
    const bool pixel = type == 7;
    std::vector<uint8_t> container;
    std::lock_guard lock(s_mutex);
    if (!BuildContainer(object, pixel, container))
        return;
    // As GetShaderLocked hashes it: record 0, the object's own microcode (nothing loaded or patched yet).
    const uint64_t hash = ShaderHash(container.data(), container.size(), pixel);
    if (s_byHash.contains(hash) || !s_warmSeen.insert(hash).second)
        return;
    static const bool started = [] {
        const uint32_t threads = std::clamp(HostProcessors(), 3u, 34u) - 2;
        for (uint32_t i = 0; i < threads; i++)
            std::thread([] {
                platform::SetCurrentThreadPriority(platform::kPriorityBelowNormal);
                WarmWorker();
            }).detach();
        log::Info("Shader warm-up: {} thread(s)", threads);
        std::thread(QueueGuestContainers).detach();
        return true;
    }();
    (void)started;
    s_warmQueue.push_back({ hash, pixel, 0, std::move(container) });
    s_warmCv.notify_one();
}

HostShader* GetShaderLocked(uint32_t object, bool pixel);

HostShader* GetShader(uint32_t object, bool pixel)
{
    if (!object)
        return nullptr;
    // Fastest path, without the lock: this thread already found the object's shader this frame and no
    // object has changed version since (about 2500 calls a frame, most of them repeats).
    struct Memo { uint32_t object = 0; bool pixel = false; uint64_t frame = UINT64_MAX, epoch = 0; HostShader* shader = nullptr; };
    thread_local std::array<Memo, 64> memo;
    Memo& m = memo[(object >> 4) & 63];
    const uint64_t frame = RendererFrame(), epoch = s_stateEpoch.load(std::memory_order_acquire);
    if (m.object == object && m.pixel == pixel && m.frame == frame && m.epoch == epoch)
        return m.shader;
    HostShader* found = GetShaderLocked(object, pixel);
    if (found)
        m = { object, pixel, frame, epoch, found };
    return found;
}

HostShader* GetShaderLocked(uint32_t object, bool pixel)
{
    const uint32_t virt = object + (pixel ? kPsContainer : kVsContainer);
    const auto* header = g_memory.As<xenos::ShaderContainer>(virt);
    const uint32_t virtualSize = header->virtualSize, physicalSize = header->physicalSize;
    const uint32_t microcode = U32(object + (pixel ? kPsMicrocode : kVsMicrocode));
    const uint32_t options = ShaderOptions();
    uint64_t state;
    {
        std::lock_guard lock(s_mutex);
        state = ObjectState(object, pixel);
        if (auto it = s_byObject.find(uint64_t(object) ^ state * 0xC2B2AE3D27D4EB4Full); it != s_byObject.end())
        {
            ObjectShader& e = it->second;
            const uint64_t frame = RendererFrame();
            if (e.object == object && e.state == state && e.microcode == microcode && e.virtualSize == virtualSize &&
                e.physicalSize == physicalSize && virtualSize <= e.shader->container.size() &&
                e.shader->options == (options & e.shader->optionUse))   // translated for the options set now
            {
                // Bound some 2500 times a frame: the memory itself is compared once per frame.
                if (e.checkedFrame == frame)
                    return e.shader;
                if (g_memory.Unchanged(microcode, physicalSize, e.watchGeneration) &&
                    std::memcmp(e.shader->container.data(), g_memory.Translate(virt), virtualSize) == 0)
                {
                    e.checkedFrame = frame;
                    return e.shader;
                }
            }
        }
    }
    // Watched before it is read, so a write from here on is seen at the next call.
    const uint64_t watchGeneration = g_memory.Watch(microcode, physicalSize);
    std::vector<uint8_t> container;
    if (!BuildContainer(object, pixel, container))
    {
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            log::Warn("Guest shader object {:08X} has no recognisable container", object);
        }
        return nullptr;
    }
    std::unique_lock lock(s_mutex);
    uint32_t variant = 0;
    if (!pixel)
        if (auto it = s_vsVariant.find(object); it != s_vsVariant.end())
            variant = it->second;
    uint64_t hash = ShaderHash(container.data(), container.size(), pixel) ^ (uint64_t(variant) << 56);
    // A shader the live translation options change: its hash carries their state.
    const OptionHashes oh = ApplyOptions(container.data(), container.size(), hash, pixel, options);
    const uint32_t optionUse = oh.optionUse;
    const uint64_t idHash = oh.idHash;
    hash = oh.hash;
    // Warm-up: a shader still queued is compiled here and now. One a warm-up thread is
    // compiling is waited for.
    if (!s_byHash.contains(hash) && s_warmSeen.contains(hash))
    {
        if (auto it = std::find_if(s_warmQueue.begin(), s_warmQueue.end(), [hash](const WarmJob& j) { return j.hash == hash; });
            it != s_warmQueue.end())
            s_warmQueue.erase(it);
        s_warmCv.wait(lock, [hash] { return !s_warmRunning.contains(hash); });
    }
    auto& slot = s_byHash[hash];
    bool made = false;   // its module made here: the pipeline list hears of it
    if (!slot)
    {
        made = true;
        slot = std::make_unique<HostShader>();
        slot->hash = hash;
        slot->idHash = idHash;
        slot->codeScale = (optionUse & kOptSceneFloat) != 0;
        slot->pixel = pixel;
        slot->variant = variant;
        slot->optionUse = optionUse;
        slot->options = options & optionUse;
        slot->container = std::move(container);
        NoteCodeKinds(*slot);
        const auto start = std::chrono::steady_clock::now();
        if (Translate(*slot))
        {
            s_compileStats.count++;
            s_compileStats.ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
            if (slot->ok)
                StoreGuestContainer(*slot);
        }
    }
    // Warmed up: the shader module is made at first use. Thousands made up front would slow the driver: the
    // mission start's texture uploads would take 350 ms instead of 60.
    if (slot->ok && !slot->shader && g_video.Ready())
    {
        MakeShaderModule(*slot);
        made = true;
    }
    if (made)
        RendererShaderReady(*slot);
    s_byObject[uint64_t(object) ^ state * 0xC2B2AE3D27D4EB4Full] = { object, state, slot.get(), microcode, virtualSize, physicalSize,
        watchGeneration };
    return slot.get();
}

ShaderCompileStats GetShaderCompileStats()
{
    std::lock_guard lock(s_mutex);
    return s_compileStats;
}

void NoteVertexShaderLoad(uint32_t object, uint32_t variant, uint32_t writePointer)
{
    // The loader's record: container +0x18 + 8 * variant holds its offset. Microcode offset at +0,
    // size in bytes at +4 (the record's physicalOffset / size).
    const uint32_t container = object + kVsContainer;
    const uint32_t record = container + U32(container + 0x18 + variant * 8);
    const uint32_t offset = U32(record), size = U32(record + 4);
    const uint32_t packet = writePointer + 12;   // IM_LOAD_IMMEDIATE header (type 3, opcode 0x2B)
    if (size == 0 || size > 0x40000 || (U32(packet) & 0xC0007F00u) != 0xC0002B00u)
    {
        static uint32_t warned = 0;
        if (warned++ < 8)
            log::Warn("Vertex shader load {:08X}: no IM_LOAD_IMMEDIATE at {:08X} (header {:08X})", object, packet, U32(packet));
        return;
    }
    std::vector<uint8_t> bytes(size);
    std::memcpy(bytes.data(), g_memory.Translate(writePointer + 24), size);
    const uint32_t microcode = U32(object + kVsMicrocode);
    static uint32_t reported = 0;
    if (std::memcmp(bytes.data(), g_memory.Translate(microcode + offset), size) != 0 && reported++ < 8)
        log::Info("Vertex shader {:08X}: loaded microcode differs from the object's (D3D patched it)", object);
    static const uint64_t traceFrame = [] { const char* v = std::getenv("AC6_DRAW_TRACE"); return v ? std::stoull(v) : UINT64_MAX; }();
    if (RendererFrame() == traceFrame)
        log::Info("Vertex shader load: object {:08X} record {} ({} bytes, hash {:016X})", object, variant, size, Hash(bytes.data(), bytes.size()));
    capture::OnVertexShaderLoad(object, offset, bytes);
    std::lock_guard lock(s_mutex);
    if (auto it = s_vsPatched.find(object); it != s_vsPatched.end() && it->second.offset == offset && it->second.bytes == bytes)
        return;
    SetPatched(object, offset, std::move(bytes));
}

void ReplayVertexShaderLoad(uint32_t object, uint32_t offset, std::vector<uint8_t> bytes)
{
    std::lock_guard lock(s_mutex);
    SetPatched(object, offset, std::move(bytes));
}

uint64_t VertexShaderStateEpoch()
{
    return s_stateEpoch.load(std::memory_order_acquire);
}

std::vector<uint8_t> ExportVertexShaderState()
{
    std::lock_guard lock(s_mutex);
    std::vector<uint8_t> out;
    auto put = [&out](uint32_t v) { out.insert(out.end(), reinterpret_cast<uint8_t*>(&v), reinterpret_cast<uint8_t*>(&v) + 4); };
    put(uint32_t(s_vsVariant.size()));
    for (auto& [object, variant] : s_vsVariant)
    {
        put(object);
        put(variant);
    }
    put(uint32_t(s_vsPatched.size()));
    for (auto& [object, patched] : s_vsPatched)
    {
        put(object);
        put(patched.offset);
        put(uint32_t(patched.bytes.size()));
        out.insert(out.end(), patched.bytes.begin(), patched.bytes.end());
    }
    return out;
}

void ImportVertexShaderState(const uint8_t* data, size_t size)
{
    std::lock_guard lock(s_mutex);
    size_t at = 0;
    auto get = [&]() { uint32_t v = 0; if (at + 4 <= size) std::memcpy(&v, data + at, 4); at += 4; return v; };
    s_vsVariant.clear();
    s_vsPatched.clear();
    s_byObject.clear();
    for (uint32_t n = get(), i = 0; i < n; i++)
    {
        const uint32_t object = get();
        s_vsVariant[object] = get();
    }
    for (uint32_t n = get(), i = 0; i < n && at < size; i++)
    {
        const uint32_t object = get(), offset = get(), bytes = get();
        SetPatched(object, offset, std::vector<uint8_t>(data + at, data + std::min<size_t>(size, at + bytes)));
        at += bytes;
    }
}

void NoteVertexShaderPatchedInPlace(uint32_t object, uint32_t variant)
{
    capture::OnVertexShaderPatchedInPlace(object, variant);
    std::lock_guard lock(s_mutex);
    s_vsPatched.erase(object);
    s_vsVariant[object] = variant;
    s_stateEpoch.fetch_add(1, std::memory_order_release);
}

// AC6_UNIFORM_CONSTANTS=0: the SPIR-V backend's constants through buffer addresses (push
// constants) instead of uniform buffers.
bool UniformConstants()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_UNIFORM_CONSTANTS"); return !(v && *v == '0') && !HlslBackend(); }();
    return enabled;
}

bool EdramSwizzle()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_EDRAM_SWIZZLE"); return (v && *v == '1') || HlslBackend(); }();
    return enabled;
}

void NoteVertexShaderVariant(uint32_t object, uint32_t variant)
{
    capture::OnVertexShaderVariant(object, variant);
    std::lock_guard lock(s_mutex);
    auto [it, inserted] = s_vsVariant.try_emplace(object, variant);
    if (!inserted && it->second == variant)
        return;
    it->second = variant;
    s_stateEpoch.fetch_add(1, std::memory_order_release);
    static std::set<std::pair<uint32_t, uint32_t>> seen;
    if (variant != 0 && seen.size() < 32 && seen.insert({ object, variant }).second)
        log::Info("Vertex shader {:08X} uses record {}", object, variant);
}

} // namespace ac6::gpu
