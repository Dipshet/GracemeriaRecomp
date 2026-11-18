// The game's data packages: see game/packages.h.
#include "game/packages.h"
#include "game/localization.h"
#include "plugin/ac6_plugin.h"
#include "plugin/plugins.h"
#include "game/inflate.h"
#include "game/game_modes.h"
#include "gpu/renderer.h"
#include "cpu/guest_thread.h"
#include "endian.h"
#include "fs/vfs.h"
#include "heap.h"
#include "kernel/function.h"
#include "log.h"
#include "memory.h"
#include "platform/platform.h"
#include "settings.h"

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <format>
#include <fstream>
#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>


namespace ac6::game {

namespace {

// DATA.TBL: u32 count, u32 archives, then 16-byte entries (big-endian): u8 archive (0: DATA00.PAC, 1: DATA01.PAC), u8 mode
// (1 packed, 2 stored: the decompressor's mode), u16 0, u32 offset, u32 packed size, u32 size (unpacked).
struct Package { uint8_t archive = 0, mode = 0; uint32_t offset = 0, packed = 0, size = 0; };

// The run step's results (traced): 1 more to do, 2 wants more input than it has, 3 the stream ended.
constexpr uint32_t kRunMore = 1;

uint32_t Be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }

bool ReadFile(const std::string& guestPath, std::vector<uint8_t>& out, uint64_t offset = 0, uint64_t length = UINT64_MAX)
{
    const fs::Resolution r = fs::Resolve(guestPath);
    if (!r.device || !r.entry)
        return false;
    const uint64_t n = std::min<uint64_t>(length, r.entry->size > offset ? r.entry->size - offset : 0);
    out.resize(size_t(n));
    return r.device->Read(*r.entry, offset, out.data(), n) == n;
}

bool ReadTable(std::vector<Package>& packages)
{
    std::vector<uint8_t> tbl;
    if (!ReadFile("game:\\DATA.TBL", tbl) || tbl.size() < 8)
        return false;
    const uint32_t count = Be32(tbl.data());
    packages.clear();
    for (uint32_t i = 0; i < count && 8 + 16 * (i + 1) <= tbl.size(); i++)
    {
        const uint8_t* e = tbl.data() + 8 + 16 * i;
        packages.push_back({ e[0], e[1], Be32(e + 4), Be32(e + 8), Be32(e + 12) });
    }
    return true;
}

// pi's binary fraction, computed: every package is XORed with 8 of its bytes. The game computes pi with Machin's
// formula (pi = 16 atan(1/5) - 4 atan(1/239)) in base 2^32 at each package's start (0x822CF018). Here once, in fixed
// point with guard words.
const std::vector<uint8_t>& PiFraction()
{
    static const std::vector<uint8_t> bytes = [] {
        constexpr size_t kBytes = 8 * 256 + 16, kLimbs = kBytes / 4 + 4;   // limb 0: the integer part, most significant first
        auto divSmall = [](std::vector<uint32_t>& x, uint32_t d) {
            uint64_t rem = 0;
            bool nonzero = false;
            for (uint32_t& l : x)
            {
                const uint64_t cur = rem << 32 | l;
                l = uint32_t(cur / d);
                rem = cur % d;
                nonzero |= l != 0;
            }
            return nonzero;
        };
        auto addSub = [](std::vector<uint32_t>& x, const std::vector<uint32_t>& y, bool sub) {
            int64_t carry = 0;
            for (size_t i = x.size(); i-- > 0;)
            {
                const int64_t cur = int64_t(x[i]) + (sub ? -int64_t(y[i]) : int64_t(y[i])) + carry;
                x[i] = uint32_t(uint64_t(cur));
                carry = cur >> 32;
            }
        };
        auto mulSmall = [](std::vector<uint32_t> x, uint32_t m) {
            uint64_t carry = 0;
            for (size_t i = x.size(); i-- > 0;)
            {
                const uint64_t cur = uint64_t(x[i]) * m + carry;
                x[i] = uint32_t(cur);
                carry = cur >> 32;
            }
            return x;
        };
        auto arctanInv = [&](uint32_t x) {   // atan(1/x) = 1/x - 1/(3x^3) + 1/(5x^5) - ...
            std::vector<uint32_t> term(kLimbs, 0);
            term[0] = 1;
            divSmall(term, x);
            std::vector<uint32_t> sum = term;
            for (uint32_t k = 1; divSmall(term, x * x); k++)
            {
                std::vector<uint32_t> t = term;
                divSmall(t, 2 * k + 1);
                addSub(sum, t, k % 2 == 1);
            }
            return sum;
        };
        std::vector<uint32_t> pi = mulSmall(arctanInv(5), 16);
        addSub(pi, mulSmall(arctanInv(239), 4), true);
        std::vector<uint8_t> out;
        for (size_t i = 1; i < pi.size() && out.size() < kBytes; i++)
            for (int b = 3; b >= 0; b--)
                out.push_back(uint8_t(pi[i] >> (8 * b)));
        out.resize(kBytes);
        return out;
    }();
    return bytes;
}

// Packages unpacked natively: the bytes XORed with pi at 8 * (index & 0xFF) + 4 (the game's key, 0x822CF018), then
// raw deflate (mode 1, the game's 0x822CF2F8, game/inflate.cpp here) or stored (mode 2). No guest code: Alloc and Free
// do nothing.
struct Decoder
{
    bool Alloc(uint32_t, uint32_t) { return true; }
    void Free() {}

    // Package `index` unpacked into `unpacked` (its table size, zeros past a stream that ends short of it).
    bool Unpack(uint32_t index, const Package& pk, std::vector<uint8_t>& unpacked, std::string& why)
    {
        std::vector<uint8_t> packed;
        if (!ReadFile(std::format("game:\\DATA{:02}.PAC", pk.archive), packed, pk.offset, pk.packed) || packed.size() != pk.packed)
        {
            why = std::format("not read (archive {}, offset {:#x}, {} bytes)", pk.archive, pk.offset, pk.packed);
            return false;
        }
        const uint8_t* key = PiFraction().data() + 8 * (index & 0xFF) + 4;
        for (size_t j = 0; j < packed.size(); j++)
            packed[j] ^= key[j & 7];
        if (pk.mode == 2)
        {
            unpacked = std::move(packed);
            unpacked.resize(pk.size, 0);
            return true;
        }
        if (pk.mode != 1)
        {
            why = std::format("unknown mode {}", pk.mode);
            return false;
        }
        unpacked.assign(pk.size, 0);
        const size_t made = Inflate(packed.data(), packed.size(), unpacked.data(), unpacked.size());
        if (made == SIZE_MAX)
        {
            why = std::format("inflate failed ({} -> {} bytes)", pk.packed, pk.size);
            return false;
        }
        return true;
    }
};

struct Args
{
    std::filesystem::path folder;
    int first = 0, count = -1;
    bool items = false;
    int result = 1;
};

// An item's kind from its first four bytes: printable letters and digits, lower case ("ntxr", "ndxr"), else "bin".
std::string Kind(const uint8_t* p, size_t size)
{
    std::string k;
    for (size_t i = 0; i < 4 && i < size; i++)
    {
        const char c = char(p[i]);
        if (std::isalnum(uint8_t(c)))
            k += char(std::tolower(uint8_t(c)));
        else if (c != ' ')
            return "bin";
    }
    return k.empty() ? "bin" : k;
}

// An FHM's items written into `dir`: <item>.<kind>, a nested FHM as the folder <item>.
uint32_t WriteItems(const std::filesystem::path& dir, const uint8_t* data, size_t size)
{
    if (size < 0x14 || std::memcmp(data, "FHM ", 4) != 0)
        return 0;
    const uint32_t n = Be32(data + 0x10);
    if (0x14 + 16ull * n > size)
        return 0;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    uint32_t written = 0;
    for (uint32_t k = 0; k < n; k++)
    {
        const uint32_t offset = Be32(data + 0x14 + 4 * k), bytes = Be32(data + 0x14 + 4 * (n + k));
        if (!bytes || uint64_t(offset) + bytes > size)
            continue;
        const uint8_t* item = data + offset;
        if (bytes >= 4 && std::memcmp(item, "FHM ", 4) == 0)
        {
            written += WriteItems(dir / std::to_string(k), item, bytes);
            continue;
        }
        std::ofstream f(dir / std::format("{}.{}", k, Kind(item, bytes)), std::ios::binary);
        f.write(reinterpret_cast<const char*>(item), bytes);
        written++;
    }
    return written;
}

void Extract(void* p)
{
    Args& args = *static_cast<Args*>(p);
    std::vector<Package> packages;
    if (!ReadTable(packages))
    {
        log::Error("Packages: DATA.TBL not found on the disc");
        return;
    }
    uint32_t maxPacked = 0, maxSize = 0;
    for (const Package& pk : packages)
    {
        maxPacked = std::max(maxPacked, pk.packed);
        maxSize = std::max(maxSize, pk.size);
    }
    Decoder decoder;
    if (!decoder.Alloc(maxPacked, maxSize))
    {
        log::Error("Packages: no guest memory for the buffers ({} / {} MB)", maxPacked >> 20, maxSize >> 20);
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(args.folder, ec);
    const int last = args.count < 0 ? int(packages.size()) : std::min(int(packages.size()), args.first + args.count);
    log::Info("Packages: {} in DATA.TBL; extracting {} to {}", packages.size(), last - args.first, platform::PathUtf8(args.folder));
    std::printf("Packages: %zu in DATA.TBL; extracting %d to %s\n", packages.size(), last - args.first, platform::PathUtf8(args.folder).c_str());
    const auto start = std::chrono::steady_clock::now();
    int done = 0, failed = 0;
    uint64_t bytes = 0;
    for (int i = args.first; i < last; i++)
    {
        std::vector<uint8_t> unpacked;
        std::string why;
        if (!decoder.Unpack(uint32_t(i), packages[size_t(i)], unpacked, why))
        {
            log::Warn("Packages: {}: {}", i, why);
            failed++;
            continue;
        }
        {
            std::ofstream f(args.folder / std::format("{:04}.bin", i), std::ios::binary);
            f.write(reinterpret_cast<const char*>(unpacked.data()), std::streamsize(unpacked.size()));
        }
        if (args.items)
            WriteItems(args.folder / std::format("{:04}", i), unpacked.data(), unpacked.size());
        done++;
        bytes += unpacked.size();
        if (done % 50 == 0)
            std::printf("  %d of %d (%llu MB)\n", done, last - args.first, (unsigned long long)(bytes >> 20));
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    log::Info("Packages: {} extracted ({} MB), {} failed, in {:.1f} s", done, bytes >> 20, failed, seconds);
    std::printf("Packages: %d extracted (%llu MB), %d failed, in %.1f s\n", done, (unsigned long long)(bytes >> 20), failed, seconds);
    args.result = failed ? 2 : 0;
}

// ---- Package mods ---------------------------------------------------------------------------------
// mods/<mod>/packages/<index>.bin replaces a whole package (unpacked: an FHM), mods/<mod>/packages/<index>/<item>.<kind>
// one of its items (<item>/ a folder for a nested FHM's items): the names --extract-packages --items writes. Mods in name
// order: the first to name a package or an item wins. The packages are rebuilt once, before the game starts (the
// originals unpacked natively, the replaced items put in, each item at its original offset's alignment), and handed to
// the game when it unpacks them: the decoder's begin / run hooks below, DATA.TBL's sizes raised for packages that grew
// (PatchPackageTable, from NtReadFile).
struct ModNode
{
    std::filesystem::path file;             // replaces this node whole (a package or an item)
    std::map<uint32_t, ModNode> children;   // its items, by number
};

// "12", "12.ntxr", "0009": the number before the first '.'.
bool ParseNumber(const std::string& name, uint32_t& value)
{
    const std::string digits = name.substr(0, name.find('.'));
    if (digits.empty() || digits.size() > 9 || !std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; }))
        return false;
    value = uint32_t(std::stoul(digits));
    return true;
}

uint32_t CountFiles(const ModNode& node)
{
    uint32_t n = node.file.empty() ? 0 : 1;
    for (const auto& [k, child] : node.children)
        n += CountFiles(child);
    return n;
}

std::filesystem::path FirstFile(const ModNode& node)
{
    if (!node.file.empty())
        return node.file;
    for (const auto& [k, child] : node.children)
        if (auto f = FirstFile(child); !f.empty())
            return f;
    return {};
}

void ScanItems(const std::filesystem::path& dir, ModNode& node)
{
    std::error_code ec;
    std::vector<std::filesystem::path> entries;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec))
        entries.push_back(e.path());
    std::sort(entries.begin(), entries.end());
    for (const auto& path : entries)
    {
        uint32_t k;
        if (!ParseNumber(platform::PathUtf8(path.filename()), k))
            continue;
        ModNode& child = node.children[k];
        if (std::filesystem::is_directory(path, ec))
        {
            if (child.file.empty())
                ScanItems(path, child);
        }
        else if (child.file.empty() && child.children.empty())
            child.file = path;
    }
}

// The chosen language pack's packages first (they win over a mod's), then the mods' in name order when `includeMods`
// is on.
std::map<uint32_t, ModNode> ScanMods(bool includeMods)
{
    std::map<uint32_t, ModNode> mods;
    std::error_code ec;
    std::vector<std::filesystem::path> folders;
    if (includeMods)
    {
        const std::filesystem::path root = platform::ExecutableDirectory() / "mods";
        for (const auto& e : std::filesystem::directory_iterator(root, ec))
            if (e.is_directory(ec))
                folders.push_back(e.path() / "packages");
        std::sort(folders.begin(), folders.end());
    }
    if (const localization::Pack* pack = localization::Active())
        folders.insert(folders.begin(), pack->dir / "packages");
    for (const auto& packages : folders)
    {
        std::vector<std::filesystem::path> entries;
        for (const auto& e : std::filesystem::directory_iterator(packages, ec))
            entries.push_back(e.path());
        std::sort(entries.begin(), entries.end());
        for (const auto& path : entries)
        {
            uint32_t index;
            if (!ParseNumber(platform::PathUtf8(path.filename()), index))
                continue;
            ModNode& node = mods[index];
            if (std::filesystem::is_directory(path, ec))
            {
                if (node.file.empty())
                    ScanItems(path, node);
            }
            else if (node.file.empty() && node.children.empty())
                node.file = path;
        }
    }
    return mods;
}

// An item's alignment: the largest power of two its original offset is a multiple of, 16 to 64 KB (textures 4 KB and up).
uint32_t Alignment(uint32_t offset)
{
    uint32_t a = 16;
    while (a < 0x10000 && offset && offset % (a * 2) == 0)
        a *= 2;
    return a;
}

void Put32(uint8_t* p, uint32_t v)
{
    p[0] = uint8_t(v >> 24);
    p[1] = uint8_t(v >> 16);
    p[2] = uint8_t(v >> 8);
    p[3] = uint8_t(v);
}

// `data` with the mod's replacements: the node's file whole, or an FHM laid out again with its replaced items.
bool Rebuild(const uint8_t* data, size_t size, const ModNode& node, std::vector<uint8_t>& out, std::string& why)
{
    if (!node.file.empty())
    {
        std::ifstream f(node.file, std::ios::binary);
        if (!f)
        {
            why = "can't read " + platform::PathUtf8(node.file);
            return false;
        }
        out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        return true;
    }
    if (node.children.empty())
    {
        out.assign(data, data + size);
        return true;
    }
    if (size < 0x14 || std::memcmp(data, "FHM ", 4) != 0)
    {
        why = "items given for something that isn't an FHM";
        return false;
    }
    const uint32_t n = Be32(data + 0x10);
    if (0x14 + 16ull * n > size)
    {
        why = "a damaged FHM";
        return false;
    }
    std::vector<uint32_t> offsets(n), sizes(n), order(n);
    size_t headerEnd = 0x14 + 16ull * n, first = SIZE_MAX;
    for (uint32_t k = 0; k < n; k++)
    {
        offsets[k] = Be32(data + 0x14 + 4 * k);
        sizes[k] = Be32(data + 0x14 + 4 * (n + k));
        order[k] = k;
        if (sizes[k] && offsets[k] >= headerEnd)
            first = std::min<size_t>(first, offsets[k]);
    }
    for (const auto& [k, child] : node.children)
        if (k >= n)
        {
            why = std::format("item {} named, the FHM has {}", k, n);
            return false;
        }
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return offsets[a] < offsets[b]; });
    const size_t header = first != SIZE_MAX ? first : (headerEnd + 15) & ~size_t(15);
    out.assign(data, data + std::min(header, size));
    std::vector<uint32_t> newOffsets = offsets, newSizes = sizes;
    for (uint32_t k : order)
    {
        std::vector<uint8_t> item;
        const auto it = node.children.find(k);
        if (it != node.children.end())
        {
            if (uint64_t(offsets[k]) + sizes[k] > size || !Rebuild(data + offsets[k], sizes[k], it->second, item, why))
            {
                why = std::format("item {}: {}", k, why.empty() ? "out of range" : why);
                return false;
            }
        }
        else if (sizes[k] && uint64_t(offsets[k]) + sizes[k] <= size)
            item.assign(data + offsets[k], data + offsets[k] + sizes[k]);
        if (item.empty())
        {
            newSizes[k] = 0;   // empty (an empty original, or a mod's 0-byte file): the offset kept, nothing to read there
            continue;
        }
        const uint32_t align = Alignment(offsets[k]);
        out.resize((out.size() + align - 1) & ~size_t(align - 1));
        newOffsets[k] = uint32_t(out.size());
        newSizes[k] = uint32_t(item.size());
        out.insert(out.end(), item.begin(), item.end());
    }
    out.resize((out.size() + 15) & ~size_t(15));
    for (uint32_t k = 0; k < n; k++)
    {
        Put32(out.data() + 0x14 + 4 * k, newOffsets[k]);
        Put32(out.data() + 0x14 + 4 * (n + k), newSizes[k]);
    }
    return true;
}

std::unordered_map<uint32_t, std::vector<uint8_t>> s_built;   // package index -> its modded bytes (filled before the game)
std::unordered_map<uint32_t, uint32_t> s_tableSizes;          // package index -> its size in DATA.TBL as the game reads it
std::unordered_map<uint32_t, uint32_t> s_packedSizes;         // package index -> its packed size: DLC files reuse low numbers
std::mutex s_activeMutex;
std::unordered_map<uint32_t, uint32_t> s_active;              // decoder state -> the modded package it is unpacking

struct BuildArgs
{
    std::map<uint32_t, ModNode> mods;
    std::atomic<bool> done{ false };
};

void BuildMods(void* p)
{
    BuildArgs& args = *static_cast<BuildArgs*>(p);
    // The start waits for `done`: set on every way out, a throw included.
    struct Done { std::atomic<bool>& done; ~Done() { done = true; } } finished{ args.done };
    std::vector<Package> packages;
    if (!ReadTable(packages))
    {
        log::Error("Package mods: DATA.TBL not found");
        args.done = true;
        return;
    }
    uint32_t maxPacked = 0, maxSize = 0;
    for (const auto& [index, node] : args.mods)
        if (index < packages.size() && node.file.empty())
        {
            maxPacked = std::max(maxPacked, packages[index].packed);
            maxSize = std::max(maxSize, packages[index].size);
        }
    Decoder decoder;
    if (maxSize && !decoder.Alloc(maxPacked, maxSize))
    {
        log::Error("Package mods: no guest memory for the decoder ({} / {} MB)", maxPacked >> 20, maxSize >> 20);
        decoder.Free();
        args.done = true;
        return;
    }
    for (const auto& [index, node] : args.mods)
    {
        if (index >= packages.size())
        {
            log::Warn("Package mods: package {} named; DATA.TBL has {}", index, packages.size());
            continue;
        }
        std::vector<uint8_t> original, built;
        std::string why;
        try
        {
            if (node.file.empty() && !decoder.Unpack(index, packages[index], original, why))
            {
                log::Warn("Package mods: {}: the original {}", index, why);
                continue;
            }
            if (!Rebuild(original.data(), original.size(), node, built, why))
            {
                log::Warn("Package mods: {}: {} (the game's is used)", index, why);
                continue;
            }
        }
        catch (const std::exception& e)
        {
            log::Warn("Package mods: {}: {} (the game's is used)", index, e.what());
            continue;
        }
        log::Info("Package mods: {}: {} file(s) from {}; {} -> {} bytes", index, CountFiles(node), platform::PathUtf8(FirstFile(node).parent_path()),
            packages[index].size, built.size());
        s_tableSizes[index] = std::max<uint32_t>(packages[index].size, uint32_t(built.size()));
        s_packedSizes[index] = packages[index].packed;
        s_built[index] = std::move(built);
    }
    decoder.Free();
    args.done = true;
}

} // namespace

int ExtractPackages(const std::filesystem::path& folder, int first, int count, bool items)
{
    Args args{ folder, first, count, items };
    GuestThreadParams params;
    params.hostEntry = Extract;
    params.hostArg = &args;
    GuestThread::RunOnCurrentThread(params);
    return args.result;
}

void LoadPackageMods()
{
    const std::string v = settings::Get("AC6_PACKAGE_MODS");
    BuildArgs args;
    args.mods = ScanMods(v != "0" && v != "false");   // (a language pack's packages either way)
    if (args.mods.empty())
        return;
    const auto start = std::chrono::steady_clock::now();
    // On a guest thread of its own, waited for: before the game reads DATA.TBL.
    GuestThreadParams params;
    params.hostEntry = BuildMods;
    params.hostArg = &args;
    GuestThread* thread = GuestThread::Start(params);   // (created held: Resume launches it)
    if (!thread)
    {
        log::Error("Package mods: no thread to build them on");
        return;
    }
    thread->Resume();
    while (!args.done.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    log::Info("Package mods: {} of {} package(s) ready in {:.2f} s", s_built.size(), args.mods.size(),
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}

void PatchPackageTable(uint8_t* data, uint64_t offset, uint64_t length)
{
    for (const auto& [index, size] : s_tableSizes)
    {
        const uint64_t at = 8 + 16ull * index + 12;   // the entry's size
        for (uint32_t b = 0; b < 4; b++)
            if (at + b >= offset && at + b < offset + length)
                data[at + b - offset] = uint8_t(size >> (24 - 8 * b));
    }
}

namespace packages_hooks {

// AC6_PACKAGE_LOG=1 (or a file path): every package the game unpacks, with the frame and the game's mode, to
// package_log.tsv next to the executable: what a catalogue of the packages and their contents can be built from.
void LogPackage(int32_t index, uint64_t packedSize, uint64_t size)
{
    static const std::string setting = settings::Get("AC6_PACKAGE_LOG");
    if (setting.empty() || setting == "0" || setting == "false")
        return;
    static std::mutex mutex;
    static std::ofstream file;
    std::lock_guard lock(mutex);
    static std::vector<Package> table;
    if (!file.is_open())
    {
        const std::filesystem::path path =
            setting == "1" || setting == "true" ? platform::ExecutableDirectory() / "package_log.tsv" : std::filesystem::path(setting);
        file.open(path, std::ios::binary | std::ios::app);
        file << "frame\tpackage\tsource\tpacked\tsize\tmode\n";
        ReadTable(table);
    }
    // The source: "disc" when the sizes are DATA.TBL's for that index. DLC packages are numbered in their own tables.
    const bool disc = index >= 0 && size_t(index) < table.size() && table[size_t(index)].packed == packedSize &&
                      table[size_t(index)].size == size;
    const char* mode = CurrentMode();
    file << std::format("{}\t{}\t{}\t{}\t{}\t{}\n", gpu::RendererFrame(), index, disc ? "disc" : "other", packedSize, size,
                        *mode ? mode : "(boot)");
    file.flush();
}

void OnBegin(uint32_t state, int32_t index, uint32_t result, uint64_t packedSize, uint64_t size)
{
    if (result)
    {
        LogPackage(index, packedSize, size);
        plugin::Event(plugin::kPackageUnpacked, index);   // (for plugins, such as one adding aircraft)
    }
    if (s_built.empty())
        return;
    std::lock_guard lock(s_activeMutex);
    // The number alone isn't enough: DLC paints are numbered slot + 1, in their own series.
    const auto mod = index >= 0 ? s_packedSizes.find(uint32_t(index)) : s_packedSizes.end();
    if (result && mod != s_packedSizes.end() && mod->second == packedSize)
        s_active[state] = uint32_t(index);
    else
        s_active.erase(state);
}

// A modded package's run step: the game's reads counted until all of its input is in, then the rebuilt package written to
// the output and the rest of the buffer zeroed, finishing as a whole stream does: the last step returns 1 with all of the
// output made (+40 = +48) and all of the input taken (+8 = +32), so the game's next call, to the decoder itself, returns
// 0 (done). Returning 3 (a stream that ended short) crashes the game right after (a stale pointer in sub_82122398). The
// XOR's position (+22400) is put at the end too, so that call scrambles nothing: in stored mode the input buffer is the
// output (the game reads into it), which is also why nothing is written before the reads end.
bool OnRun(PPCContext& ctx)
{
    if (s_built.empty())   // filled before the game starts: no lock needed
        return false;
    const uint32_t state = ctx.r3.u32;
    uint32_t index;
    {
        std::lock_guard lock(s_activeMutex);
        const auto it = s_active.find(state);
        if (it == s_active.end())
            return false;
        index = it->second;
    }
    auto field = [&](uint32_t o) -> be<uint64_t>& { return *g_memory.As<be<uint64_t>>(state + o); };
    const uint64_t total = field(32).get();
    const uint64_t available = std::min<uint64_t>(field(16).get() + ctx.r4.u64, total);
    field(16) = available;
    field(8) = available;
    if (available < total)
    {
        ctx.r3.u64 = kRunMore;
        return true;
    }
    const std::vector<uint8_t>& data = s_built.at(index);
    const uint32_t out = g_memory.As<be<uint32_t>>(state + 60)->get();
    const uint64_t size = field(48).get();
    const uint64_t n = std::min<uint64_t>(size, data.size());
    std::memcpy(g_memory.Translate(out), data.data(), size_t(n));
    if (size > n)
        std::memset(g_memory.Translate(out + uint32_t(n)), 0, size_t(size - n));
    field(40) = size;
    field(8) = total;
    field(16) = total;
    field(22400) = total;
    {
        std::lock_guard lock(s_activeMutex);
        s_active.erase(state);
    }
    static std::mutex logMutex;
    static std::set<uint32_t> logged;
    {
        std::lock_guard lock(logMutex);
        if (logged.insert(index).second)
            log::Info("Package mods: {} given to the game ({} bytes into its {}-byte buffer)", index, n, size);
    }
    ctx.r3.u64 = kRunMore;
    return true;
}

} // namespace packages_hooks

} // namespace ac6::game

// The decoder's begin and run: modded packages handed over (above), the rest as the game's.
extern "C" PPC_FUNC(__imp__sub_822CF248);
PPC_FUNC(sub_822CF248)
{
    const uint32_t state = ctx.r3.u32;
    const int32_t index = int32_t(ctx.r10.u32);
    const uint64_t packed = ctx.r6.u64, size = ctx.r7.u64;
    __imp__sub_822CF248(ctx, base);
    ac6::game::packages_hooks::OnBegin(state, index, ctx.r3.u32, packed, size);
}

extern "C" PPC_FUNC(__imp__sub_822CF510);
PPC_FUNC(sub_822CF510)
{
    if (!ac6::game::packages_hooks::OnRun(ctx))
        __imp__sub_822CF510(ctx, base);
}
