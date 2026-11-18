// DLC discovery, checks and the report. See dlc.h.
#include "fs/dlc.h"

#include "fs/stfs.h"
#include "log.h"
#include "platform/platform.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <format>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <set>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace ac6::fs {

namespace {

// --- Known-good packages ----------------------------------------------------------------------------

struct KnownFile
{
    const char* package;
    const char* label;
    const char* file;
    uint64_t size;
    const char* sha1;
};

constexpr KnownFile kKnownFiles[] = {
#include "fs/dlc_manifest.inc"
};

struct KnownPackage
{
    std::string name, label;
    std::vector<const KnownFile*> files;
};

std::string Upper(std::string_view s)
{
    std::string out(s);
    for (char& c : out)
        if (c >= 'a' && c <= 'z')
            c = char(c - 'a' + 'A');
    return out;
}

const std::vector<KnownPackage>& KnownPackages()
{
    static const std::vector<KnownPackage> packages = [] {
        std::vector<KnownPackage> out;
        for (const KnownFile& f : kKnownFiles)
        {
            if (out.empty() || out.back().name != f.package)
                out.push_back({ f.package, f.label, {} });
            out.back().files.push_back(&f);
        }
        return out;
    }();
    return packages;
}

const KnownPackage* KnownByName(std::string_view name)
{
    const std::string key = Upper(name);
    for (const KnownPackage& k : KnownPackages())
        if (k.name == key || (key.size() == 40 && k.name.starts_with(key)))
            return &k;
    return nullptr;
}

// The known package that a set of (lower-case) file names belongs to: by a file only that package has.
const KnownPackage* KnownByContents(const std::set<std::string>& names)
{
    static const std::map<std::string, const KnownPackage*> unique = [] {
        std::map<std::string, int> count;
        for (const KnownFile& f : kKnownFiles)
            count[ToLower(f.file)]++;
        std::map<std::string, const KnownPackage*> out;
        for (const KnownPackage& k : KnownPackages())
            for (const KnownFile* f : k.files)
                if (count[ToLower(f->file)] == 1)
                    out[ToLower(f->file)] = &k;
        return out;
    }();
    for (const std::string& n : names)
        if (auto it = unique.find(n); it != unique.end())
            return it->second;
    return nullptr;
}

// --- Helpers -------------------------------------------------------------------------------------------

std::string Utf8(const std::filesystem::path& p) { return reinterpret_cast<const char*>(p.u8string().c_str()); }

bool IsHex(std::string_view s)
{
    return !s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return std::isxdigit(uint8_t(c)); });
}

// Content file names are the content id in hex: 40 digits, 42 as the marketplace writes them.
bool IsPackageName(std::string_view s) { return s.size() >= 40 && s.size() <= 42 && IsHex(s); }

std::string Bytes(uint64_t n)
{
    return n >= (1u << 20) ? std::format("{:.1f} MB", n / 1048576.0) : std::format("{:.1f} KB", n / 1024.0);
}

const char* ContentTypeName(uint32_t type)
{
    switch (type)
    {
    case 0x00001: return "a saved game";
    case 0x00002: return "marketplace content";
    case 0x07000: return "a Games on Demand title";
    case 0x10000: return "a profile";
    case 0x20000: return "a gamer picture";
    case 0x30000: return "a dashboard theme";
    case 0x80000: return "a game demo";
    case 0x90000: return "a video";
    case 0xA0000: return "a game title";
    case 0xB0000: return "a title update";
    case 0xD0000: return "an arcade title";
    default: return "content of another kind";
    }
}

// An archive that should have been extracted, by its signature.
const char* ArchiveKind(const uint8_t* head, size_t n)
{
    if (n >= 4 && std::memcmp(head, "PK\x03\x04", 4) == 0) return "a zip archive";
    if (n >= 4 && std::memcmp(head, "Rar!", 4) == 0) return "a RAR archive";
    if (n >= 6 && std::memcmp(head, "7z\xBC\xAF\x27\x1C", 6) == 0) return "a 7-Zip archive";
    return nullptr;
}

struct FoundFile
{
    std::string rel;        // backslash-separated, from the package root
    std::string key;        // lower-case rel
    std::filesystem::path path;
    uint64_t size = 0;
    std::array<uint8_t, 20> sha1{};
    bool read = false;      // read in full (sha1 valid)
    bool zero = false;      // every byte is zero
    uint8_t head[8]{};      // its first bytes
    size_t headSize = 0;
    std::string readError;
};

void Digest(FoundFile& f, const std::vector<uint8_t>& data)
{
    f.sha1 = Sha1(data.data(), data.size());
    f.headSize = std::min(data.size(), sizeof(f.head));
    std::memcpy(f.head, data.data(), f.headSize);
    f.zero = !data.empty() && std::all_of(data.begin(), data.end(), [](uint8_t b) { return b == 0; });
    f.read = true;
}

void ReadHostFile(FoundFile& f)
{
    platform::HostFile h;
    if (!h.Open(f.path, platform::HostFile::Mode::Read))
    {
        f.readError = platform::ErrorText(h.Error());
        return;
    }
    std::vector<uint8_t> data(size_t(f.size));
    size_t done = 0;
    while (done < data.size())
    {
        uint32_t got = 0;
        if (!h.ReadAt(data.data() + done, uint32_t(std::min<size_t>(data.size() - done, 1u << 24)), done, got))
        {
            f.readError = platform::ErrorText(h.Error());
            break;
        }
        if (got == 0)
        {
            f.readError = std::format("only {} of its {} bytes could be read", done, data.size());
            break;
        }
        done += got;
    }
    h.Close();
    if (f.readError.empty())
        Digest(f, data);
}

// Checks that apply to every package's files, then the comparison with the known-good dump.
void CheckFiles(DlcPackage& p, const std::vector<FoundFile>& files, const KnownPackage* known, bool verify)
{
    if (files.empty())
    {
        p.errors.push_back("it has no files");
        return;
    }
    std::set<std::string> reported;   // one reason per file
    auto report = [&](const FoundFile& f, std::string reason) {
        if (reported.insert(f.key).second)
            p.errors.push_back(std::move(reason));
    };
    for (const FoundFile& f : files)
    {
        if (!f.readError.empty())
            report(f, std::format("'{}' can't be read ({})", f.rel, f.readError));
        else if (f.size == 0)
            report(f, std::format("'{}' is empty (0 bytes)", f.rel));
        else if (f.zero)
            report(f, std::format("'{}' is all zeros: an incomplete copy or download", f.rel));
        else if (const char* kind = ArchiveKind(f.head, f.headSize))
            report(f, std::format("'{}' is {}: extract it into the dlc folder", f.rel, kind));
        // Colour packs carry a name table. The other packs (aircraft, missions) have none, so it is
        // only checked where it is present.
        else if (f.key == "dlcolorname.dat" && (f.headSize < 3 || std::memcmp(f.head, "DCN", 3) != 0))
            report(f, "'dlcolorname.dat' is damaged (it doesn't start with its DCN header)");
    }

    if (!known)
        return;
    std::map<std::string, const FoundFile*> byKey;
    for (const FoundFile& f : files)
        byKey[f.key] = &f;
    std::set<std::string> expected;
    for (const KnownFile* k : known->files)
    {
        const std::string key = ToLower(k->file);
        expected.insert(key);
        auto it = byKey.find(key);
        if (it == byKey.end())
        {
            // Somewhere deeper, perhaps (an extra folder level from the extraction).
            std::string deeper;
            for (const FoundFile& f : files)
                if (f.key.ends_with("\\" + key))
                    deeper = f.rel;
            if (!deeper.empty())
                p.errors.push_back(std::format("'{}' is at '{}': the game only reads the package's top level", k->file, deeper));
            else
                p.errors.push_back(std::format("'{}' is missing ({} bytes)", k->file, k->size));
            continue;
        }
        const FoundFile& f = *it->second;
        if (!f.read || reported.count(f.key))
            continue;   // reported above
        if (f.size != k->size)
            p.errors.push_back(std::format("'{}' is {} bytes, the known-good file {}{}", f.rel, f.size, k->size,
                f.size < k->size ? " (truncated)" : ""));
        else if (verify && Hex(f.sha1.data(), 20) != Upper(k->sha1))
            p.damage.push_back(std::format("'{}' is damaged: its contents differ from the known-good file (SHA-1 {}, expected {})",
                f.rel, Hex(f.sha1.data(), 20), Upper(k->sha1)));
    }
    for (const FoundFile& f : files)
        if (!expected.count(f.key))
        {
            bool deeperCopy = false;
            for (const std::string& e : expected)
                deeperCopy |= f.key.ends_with("\\" + e);
            if (!deeperCopy)
                p.warnings.push_back(std::format("'{}' isn't part of this package", f.rel));
        }
}

// --- Devices ------------------------------------------------------------------------------------------------

// An extracted package, read-only: the files are read in place (positional reads, opened on first use).
struct DlcFolderDevice : Device
{
    std::vector<std::filesystem::path> paths;
    std::vector<std::unique_ptr<platform::HostFile>> handles;   // opened on first use
    std::mutex mutex;

    const char* Kind() const override { return "DLC folder"; }
    bool ReadOnly() const override { return true; }
    const char* FileSystemName() const override { return "STFS"; }

    uint64_t Read(const Entry& entry, uint64_t offset, void* destination, uint64_t length) override
    {
        if (entry.directory || offset >= entry.size)
            return 0;
        platform::HostFile* h;
        {
            std::lock_guard lock(mutex);
            std::unique_ptr<platform::HostFile>& slot = handles[entry.offset];
            if (!slot)
            {
                auto file = std::make_unique<platform::HostFile>();
                if (!file->Open(paths[entry.offset], platform::HostFile::Mode::Read))
                {
                    log::Error("DLC: can't open '{}' ({})", Utf8(paths[entry.offset]), platform::ErrorText(file->Error()));
                    return 0;
                }
                slot = std::move(file);
            }
            h = slot.get();
        }
        const uint64_t n = std::min(length, entry.size - offset);
        uint64_t done = 0;
        while (done < n)
        {
            const uint64_t at = offset + done;
            uint32_t got = 0;
            const bool ok = h->ReadAt(static_cast<uint8_t*>(destination) + done, uint32_t(std::min<uint64_t>(n - done, 1u << 30)), at, got);
            if (!ok || !got)
            {
                log::Error("DLC: reading '{}' at {:#x} failed ({})", Utf8(paths[entry.offset]), at,
                    ok ? std::string("end of file") : platform::ErrorText(h->Error()));
                break;
            }
            done += got;
        }
        return done;
    }
};

std::shared_ptr<Device> MakeFolderDevice(const std::filesystem::path& folder, const std::vector<FoundFile>& files)
{
    auto device = std::make_shared<DlcFolderDevice>();
    device->root.directory = true;
    device->root.attributes = 0x10;
    for (const FoundFile& f : files)
    {
        Entry* dir = &device->root;
        std::string_view rest = f.rel;
        for (size_t slash; (slash = rest.find('\\')) != std::string_view::npos; rest.remove_prefix(slash + 1))
        {
            const std::string part(rest.substr(0, slash));
            Entry* next = nullptr;
            for (auto& c : dir->children)
                if (c->directory && c->key == ToLower(part))
                    next = c.get();
            if (!next)
            {
                auto e = std::make_unique<Entry>();
                e->name = part;
                e->key = ToLower(part);
                e->directory = true;
                e->attributes = 0x10;
                e->parent = dir;
                next = e.get();
                dir->children.push_back(std::move(e));
            }
            dir = next;
        }
        auto e = std::make_unique<Entry>();
        e->name = std::string(rest);
        e->key = ToLower(e->name);
        e->size = f.size;
        e->offset = device->paths.size();
        e->parent = dir;
        dir->children.push_back(std::move(e));
        device->paths.push_back(f.path);
        device->handles.push_back(nullptr);
        device->totalBytes += f.size;
    }
    device->Finish();
    if (const platform::PathInfo info = platform::StatPath(folder); info.exists)
        device->timestamp = info.write;
    device->sectorSize = 0x1000;
    return device;
}

// --- The scan ---------------------------------------------------------------------------------------------

struct Scan
{
    uint32_t titleId = 0;
    bool verify = true;
    std::vector<DlcPackage> packages;
    std::vector<std::string> notes;   // per-root lines and stray files
};

void ScanContainer(Scan& scan, const std::filesystem::path& path, const StfsHeader& h)
{
    DlcPackage p;
    p.path = path;
    p.container = true;
    p.displayName = h.displayName;
    const std::string fileName = Utf8(path.filename());
    p.name = IsPackageName(fileName) ? Upper(fileName) : Hex(h.contentId.data(), 20);

    if (h.contentType != 2)
    {
        p.errors.push_back(std::format("it is {} (content type {:#x}), not DLC{}", ContentTypeName(h.contentType), h.contentType,
            h.contentType == 0xB0000 ? ": the runtime doesn't use title updates" : ""));
        scan.packages.push_back(std::move(p));
        return;
    }
    if (h.descriptorType != 0)
    {
        p.errors.push_back(std::format("it is an SVOD container (descriptor type {}), which DLC never is", h.descriptorType));
        scan.packages.push_back(std::move(p));
        return;
    }

    StfsOpenResult open = OpenStfs(path, h);
    p.errors = std::move(open.errors);
    p.damage = std::move(open.damage);
    p.warnings = std::move(open.warnings);
    std::vector<FoundFile> files;
    if (open.device)
    {
        std::vector<std::pair<const Entry*, std::string>> stack{ { &open.device->root, std::string() } };
        while (!stack.empty())
        {
            auto [dir, prefix] = stack.back();
            stack.pop_back();
            for (auto& c : dir->children)
            {
                const std::string rel = prefix.empty() ? c->name : prefix + "\\" + c->name;
                if (c->directory)
                {
                    stack.push_back({ c.get(), rel });
                    continue;
                }
                FoundFile f;
                f.rel = rel;
                f.key = ToLower(rel);
                f.size = c->size;
                std::vector<uint8_t> data(size_t(c->size));
                // A short read is a truncated container, reported by OpenStfs.
                if (open.device->Read(*c, 0, data.data(), data.size()) == data.size())
                    Digest(f, data);
                files.push_back(std::move(f));
            }
        }
    }
    std::set<std::string> keys;
    for (const FoundFile& f : files)
        keys.insert(f.key);
    const KnownPackage* known = KnownByName(p.name);
    if (!known)
        known = KnownByContents(keys);
    if (known && known->name != p.name)
    {
        p.warnings.push_back(std::format("its files are the known package {}: given to the game under that name", known->name));
        p.name = known->name;
    }

    if (h.titleId != scan.titleId)
    {
        if (known)
            p.warnings.push_back(std::format("it was made for title {:08X} (this disc is {:08X}), but its files are the known package", h.titleId, scan.titleId));
        else
            p.errors.push_back(std::format("it is for another title ({:08X}; this disc is {:08X}): the game wouldn't find its files", h.titleId, scan.titleId));
    }
    if (h.licenseBits != 0xFFFFFFFF)
        p.warnings.push_back(std::format("its header licenses {:#010x} only ({} entr{}); the game is told every bit is licensed",
            h.licenseBits, h.licenseEntries, h.licenseEntries == 1 ? "y" : "ies"));
    // A container's blocks are verified by its own hash tree. The known-good comparison adds file names
    // and sizes, and catches a container rebuilt from damaged files.
    CheckFiles(p, files, known, scan.verify);
    p.known = known != nullptr;
    if (known)
        p.label = known->label;
    else
        p.warnings.push_back("not a known package: checked against its own hashes only");
    p.files = uint32_t(files.size());
    for (const FoundFile& f : files)
        p.bytes += f.size;
    if (p.errors.empty() || (!scan.verify && open.device))
        p.device = std::move(open.device);
    scan.packages.push_back(std::move(p));
}

// A container that can't be read whole (out of memory on a size its damaged table claims, a read that throws) is
// reported as unusable instead of ending the scan: the main menu waits for the scan, and the other packages still list.
void ScanContainerChecked(Scan& scan, const std::filesystem::path& path, const StfsHeader& h)
{
    try
    {
        ScanContainer(scan, path, h);
    }
    catch (const std::exception& e)
    {
        DlcPackage p;
        p.path = path;
        p.container = true;
        p.displayName = h.displayName;
        const std::string fileName = Utf8(path.filename());
        p.name = IsPackageName(fileName) ? Upper(fileName) : Hex(h.contentId.data(), 20);
        p.errors.push_back(std::format("it couldn't be read ({}): damaged", e.what()));
        scan.packages.push_back(std::move(p));
    }
}

void ScanFolder(Scan& scan, const std::filesystem::path& folder)
{
    std::error_code ec;
    DlcPackage p;
    p.path = folder;
    std::filesystem::path dir = folder;

    // An extra level from the extraction (a folder holding only one folder): look inside.
    for (int depth = 0; depth < 3; depth++)
    {
        std::vector<std::filesystem::path> subdirs;
        bool hasFiles = false;
        for (auto& e : std::filesystem::directory_iterator(dir, std::filesystem::directory_options::skip_permission_denied, ec))
        {
            if (e.is_regular_file(ec))
                hasFiles = true;
            else if (e.is_directory(ec))
                subdirs.push_back(e.path());
        }
        if (hasFiles || subdirs.size() != 1)
            break;
        p.warnings.push_back(std::format("its files are in the subfolder '{}' (used as the package)", Utf8(std::filesystem::relative(subdirs[0], folder, ec))));
        dir = subdirs[0];
    }

    // A folder named after the content may hold the raw container itself.
    std::vector<std::filesystem::path> containers;
    std::vector<FoundFile> files;
    for (auto it = std::filesystem::recursive_directory_iterator(dir, std::filesystem::directory_options::skip_permission_denied, ec);
         !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec))
    {
        if (!it->is_regular_file(ec))
            continue;
        FoundFile f;
        f.path = it->path();
        f.rel = Utf8(std::filesystem::relative(f.path, dir, ec));
        std::replace(f.rel.begin(), f.rel.end(), '/', '\\');
        f.key = ToLower(f.rel);
        f.size = it->file_size(ec);
#if defined(_WIN32)
        const DWORD attributes = GetFileAttributesW(f.path.c_str());
        if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & (FILE_ATTRIBUTE_OFFLINE | 0x00040000 | 0x00400000)))
            p.warnings.push_back(std::format("'{}' is a cloud placeholder (OneDrive or similar), not stored on this PC", f.rel));
#endif
        files.push_back(std::move(f));
    }
    if (ec)
        p.errors.push_back(std::format("the folder can't be listed ({})", ec.message()));
    for (FoundFile& f : files)
    {
        StfsHeader h;
        std::string error;
        if (f.rel.find('\\') == std::string::npos && ReadStfsHeader(f.path, h, error))
            containers.push_back(f.path);
    }
    if (!containers.empty())
    {
        for (auto& c : containers)
        {
            StfsHeader h;
            std::string error;
            ReadStfsHeader(c, h, error);
            ScanContainerChecked(scan, c, h);
        }
        return;
    }

    const std::string folderName = Utf8(folder.filename());
    p.name = Upper(folderName.substr(0, 42));
    for (FoundFile& f : files)
        ReadHostFile(f);
    std::set<std::string> keys;
    for (const FoundFile& f : files)
        keys.insert(f.key);
    const KnownPackage* known = KnownByName(p.name);
    if (!known)
        known = KnownByContents(keys);
    if (known && known->name != p.name)
    {
        p.warnings.push_back(std::format("its files are the known package {}: given to the game under that name", known->name));
        p.name = known->name;
    }
    else if (!known && !IsPackageName(folderName))
        p.warnings.push_back("the folder isn't named after its content (40-42 hex digits)");
    CheckFiles(p, files, known, scan.verify);
    p.known = known != nullptr;
    if (known)
        p.label = known->label;
    else
        p.warnings.push_back("not a known package: its files can't be verified");
    p.files = uint32_t(files.size());
    for (const FoundFile& f : files)
        p.bytes += f.size;
    const bool readable = std::none_of(files.begin(), files.end(), [](const FoundFile& f) { return !f.readError.empty(); });
    if (p.errors.empty() || (!scan.verify && readable && !files.empty()))
        p.device = MakeFolderDevice(dir, files);
    scan.packages.push_back(std::move(p));
}

bool ParseHex8(std::string_view s, uint32_t& out)
{
    if (s.size() != 8 || !IsHex(s))
        return false;
    out = uint32_t(std::stoul(std::string(s), nullptr, 16));
    return true;
}

void ScanRoot(Scan& scan, const std::filesystem::path& root, const std::string& why)
{
    std::error_code ec;
    if (!std::filesystem::is_directory(root, ec))
    {
        scan.notes.push_back(std::format("scanned {} ({}): no such folder", Utf8(root), why));
        return;
    }
    const size_t before = scan.packages.size();
    size_t strays = 0;
    std::string firstStray;
    for (auto it = std::filesystem::recursive_directory_iterator(root, std::filesystem::directory_options::skip_permission_denied, ec);
         !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec))
    {
        const std::filesystem::path path = it->path();
        const std::string name = Utf8(path.filename());
        if (it->is_directory(ec))
        {
            // Xenia's layout: <title id>/00000002/<package>, with the headers under <title id>/Headers.
            uint32_t type = 0, title = 0;
            const bool anchored = ParseHex8(Utf8(path.parent_path().filename()), type) &&
                ParseHex8(Utf8(path.parent_path().parent_path().filename()), title) && type == 2;
            if (ToLower(name) == "headers")
            {
                it.disable_recursion_pending();
                continue;
            }
            if (anchored || IsPackageName(name))
            {
                ScanFolder(scan, path);
                it.disable_recursion_pending();
            }
            continue;
        }
        if (!it->is_regular_file(ec))
            continue;
        StfsHeader h;
        std::string error;
        if (ReadStfsHeader(path, h, error))
        {
            ScanContainerChecked(scan, path, h);
            continue;
        }
        if (!error.empty())
        {
            DlcPackage p;
            p.path = path;
            p.container = true;
            p.name = Upper(name.substr(0, 42));
            p.errors.push_back(error);
            scan.packages.push_back(std::move(p));
            continue;
        }
        uint8_t head[8]{};
        std::ifstream in(path, std::ios::binary);
        in.read(reinterpret_cast<char*>(head), sizeof(head));
        if (const char* kind = ArchiveKind(head, size_t(in.gcount())))
            scan.notes.push_back(std::format("'{}' is {}: extract it into the dlc folder", Utf8(std::filesystem::relative(path, root, ec)), kind));
        else if (IsPackageName(name))
        {
            DlcPackage p;
            p.path = path;
            p.container = true;
            p.name = Upper(name);
            p.errors.push_back(std::format("it is named like a package but isn't an STFS container (it starts with {}): "
                "damaged, or not a raw container", Hex(head, size_t(in.gcount()))));
            scan.packages.push_back(std::move(p));
        }
        else if (ToLower(platform::PathUtf8(path.extension())) != ".header" && ToLower(name) != "desktop.ini" && ToLower(name) != "thumbs.db")
        {
            if (!strays++)
                firstStray = Utf8(std::filesystem::relative(path, root, ec));
        }
    }
    if (ec)
        scan.notes.push_back(std::format("scanned {} ({}): listing stopped ({})", Utf8(root), why, ec.message()));
    scan.notes.push_back(std::format("scanned {} ({}): {} package(s){}", Utf8(root), why, scan.packages.size() - before,
        strays ? std::format(", {} other file(s) ignored (first: '{}')", strays, firstStray) : std::string()));
}

void Report(const Scan& scan, double seconds)
{
    std::vector<std::string> lines;
    size_t usable = 0;
    for (const DlcPackage& p : scan.packages)
        usable += p.Usable();
    lines.push_back(std::format("DLC: {} package(s) found, {} mounted, {} not mounted ({:.2f} s{})", scan.packages.size(), usable,
        scan.packages.size() - usable, seconds, scan.verify ? "" : ", AC6_DLC_VERIFY=0: contents not checked"));
    for (const std::string& n : scan.notes)
        lines.push_back("DLC: " + n);
    for (const DlcPackage& p : scan.packages)
    {
        std::string what = p.label.empty() ? std::string() : p.label + " ";
        if (!p.displayName.empty())
            what += "\"" + p.displayName + "\" ";
        lines.push_back(std::format("DLC {} {} {}({}, {} file(s), {}{}): {}",
            !p.Usable() ? "[NOT MOUNTED]" : !p.damage.empty() ? "[ok, DAMAGED]" : "[ok]         ", p.name, what,
            p.container ? "container" : "folder", p.files, Bytes(p.bytes), p.known ? ", a known package" : "", Utf8(p.path)));
        for (const std::string& e : p.errors)
            lines.push_back("      " + std::string(p.Usable() ? "ignored: " : "reason: ") + e);
        for (const std::string& d : p.damage)
            lines.push_back("      damaged: " + d + (p.Usable() ? " (mounted anyway)" : ""));
        for (const std::string& w : p.warnings)
            lines.push_back("      note: " + w);
    }
    for (const std::string& l : lines)
    {
        if (l.find("[NOT MOUNTED]") != std::string::npos || l.starts_with("      reason"))
            log::Error("{}", l);
        else if (l.starts_with("      ") || l.find("[ok, DAMAGED]") != std::string::npos)
            log::Warn("{}", l);
        else
            log::Info("{}", l);
    }
    std::ofstream out(std::filesystem::path(log::Directory()) / "dlc_report.txt", std::ios::binary);
    for (const std::string& l : lines)
        out << l << "\r\n";
}

std::mutex s_mutex;
std::shared_future<void> s_done;
std::vector<DlcPackage> s_packages;

} // namespace

std::filesystem::path DlcFolder()
{
    if (std::filesystem::path env = platform::EnvironmentPath("AC6_DLC_DIR"); !env.empty())
        return env;
    const auto iso = FindIso();
    return iso.empty() ? platform::ExecutableDirectory() / "dlc" : iso.parent_path() / "dlc";
}

void StartDlcScan(uint32_t titleId)
{
    std::lock_guard lock(s_mutex);
    if (s_done.valid())
        return;
    s_done = std::async(std::launch::async, [titleId] {
        const auto start = std::chrono::steady_clock::now();
        Scan scan;
        scan.titleId = titleId;
        if (const char* v = std::getenv("AC6_DLC"); v && std::string_view(v) == "0")
        {
            log::Info("DLC: disabled (AC6_DLC=0)");
            return;
        }
        if (const char* v = std::getenv("AC6_DLC_VERIFY"); v && std::string_view(v) == "0")
            scan.verify = false;
        std::vector<std::pair<std::filesystem::path, std::string>> roots;
        if (std::filesystem::path env = platform::EnvironmentPath("AC6_DLC_DIR"); !env.empty())
            roots.push_back({ env, "AC6_DLC_DIR" });
        else
        {
            const auto iso = FindIso();
            if (!iso.empty())
                roots.push_back({ iso.parent_path() / "dlc", "next to the disc image" });
            std::error_code ec;
            const auto exe = platform::ExecutableDirectory() / "dlc";
            if (iso.empty() || !std::filesystem::equivalent(exe, iso.parent_path() / "dlc", ec))
                if (iso.empty() || std::filesystem::is_directory(exe, ec))
                    roots.push_back({ exe, "next to the executable" });
        }
        for (auto& [root, why] : roots)
        {
            try
            {
                ScanRoot(scan, root, why);
            }
            catch (const std::exception& e)
            {
                log::Error("DLC: scanning {} stopped ({}): the packages found so far are used", Utf8(root), e.what());
            }
        }

        std::stable_sort(scan.packages.begin(), scan.packages.end(), [](const DlcPackage& a, const DlcPackage& b) { return a.name < b.name; });
        // One package per name: the first usable copy is the one mounted.
        for (size_t i = 0; i < scan.packages.size();)
        {
            size_t j = i;
            while (j < scan.packages.size() && scan.packages[j].name == scan.packages[i].name)
                j++;
            size_t chosen = i;
            for (size_t k = i; k < j; k++)
                if (scan.packages[k].Usable())
                {
                    chosen = k;
                    break;
                }
            for (size_t k = i; k < j; k++)
                if (k != chosen)
                {
                    scan.packages[k].device.reset();
                    scan.packages[k].errors.insert(scan.packages[k].errors.begin(),
                        std::format("a duplicate: the same package is also at {}{}", Utf8(scan.packages[chosen].path),
                            scan.packages[chosen].Usable() ? " (that copy is used)" : ""));
                }
            i = j;
        }
        Report(scan, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        s_packages = std::move(scan.packages);
    }).share();
}

const std::vector<DlcPackage>& DlcPackages()
{
    std::shared_future<void> done;
    {
        std::lock_guard lock(s_mutex);
        done = s_done;
    }
    if (done.valid())
        done.wait();
    return s_packages;
}

bool DlcScanFinished()
{
    std::shared_future<void> done;
    {
        std::lock_guard lock(s_mutex);
        done = s_done;
    }
    return !done.valid() || done.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

const DlcPackage* FindDlc(std::string_view name)
{
    const std::string key = Upper(name);
    for (const DlcPackage& p : DlcPackages())
        if (p.name == key && p.Usable())
            return &p;
    for (const DlcPackage& p : DlcPackages())
        if (p.name == key)
            return &p;
    return nullptr;
}

} // namespace ac6::fs
