#pragma once
// Virtual filesystem behind the kernel's file imports.
//
// Two kinds of devices:
//   - read-only devices (the disc image, DLC containers and folders) expose a tree of entries and positional reads,
//   - host folders (save containers) are writable and resolve to real paths, matched
//     case-insensitively against what is on disk (the console's filesystems are case-insensitive).
// Devices are mounted under guest device names ("game:", "d:", "\Device\Cdrom0", save roots).
// Guest paths are resolved against that table.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ac6::fs {

struct Entry
{
    std::string name;            // as stored on the device
    std::string key;             // lower-case name, for lookups
    bool directory = false;
    uint8_t attributes = 0;      // FILE_ATTRIBUTE_* (0x10 = directory)
    uint64_t offset = 0;         // device-specific data location
    uint64_t size = 0;
    Entry* parent = nullptr;
    std::vector<std::unique_ptr<Entry>> children;   // sorted by key once the device is built

    const Entry* Find(std::string_view lowerName) const;
    std::string Path() const;   // backslash-separated, from the device root
};

struct Device
{
    virtual ~Device() = default;
    virtual const char* Kind() const = 0;
    virtual bool ReadOnly() const = 0;
    // Reads up to `length` bytes at `offset` of a file entry. Returns the number of bytes read.
    virtual uint64_t Read(const Entry& entry, uint64_t offset, void* destination, uint64_t length) = 0;
    virtual bool IsHostFolder() const { return false; }
    // Reported by NtQueryVolumeInformationFile: discs are GDFX, content packages STFS.
    virtual const char* FileSystemName() const { return ReadOnly() ? "GDFX" : "STFS"; }

    Entry root;
    uint64_t timestamp = 0;      // FILETIME used for every entry (discs carry no per-file times)
    uint32_t sectorSize = 2048;
    uint64_t totalBytes = 0;

    const Entry* Resolve(std::string_view path) const;   // relative to root, '\' or '/' separators
    void Finish();                                       // sorts children (call after building the tree)
};

// A writable host folder (save containers). Read() is unused: files are opened as host files.
struct HostFolderDevice : Device
{
    std::filesystem::path rootPath;

    explicit HostFolderDevice(std::filesystem::path root);
    const char* Kind() const override { return "host folder"; }
    bool ReadOnly() const override { return false; }
    bool IsHostFolder() const override { return true; }
    uint64_t Read(const Entry&, uint64_t, void*, uint64_t) override { return 0; }

    // Maps a device path to a host path, reusing the on-disk spelling of existing components.
    std::filesystem::path Map(std::string_view devicePath) const;
};

void Mount(std::string_view deviceName, std::shared_ptr<Device> device);
void Unmount(std::string_view deviceName);

struct Resolution
{
    std::shared_ptr<Device> device;
    const Entry* entry = nullptr;     // null if not found
    bool parentFound = false;         // the containing directory exists
    std::string devicePath;           // path inside the device

    // Host folders only: the real path, whether it exists and whether it is a directory.
    std::filesystem::path hostPath;
    bool hostExists = false;
    bool hostDirectory = false;
};

// Resolves a guest path such as "game:\DATA.TBL" or "\Device\Cdrom0\DATA.TBL". `base`, if given,
// is a directory the path is relative to (open-by-root-handle).
Resolution Resolve(std::string_view guestPath, const Resolution* base = nullptr);

// Disc images.
std::shared_ptr<Device> OpenIso(const std::filesystem::path& path);
// The disc images to try: AC6_ISO, else every *.iso (any name, sorted) in the first folder that has one, from the
// executable's folder up through its parents.
std::vector<std::filesystem::path> FindIsos();
// The image in use (SetIso, by main.cpp once it has picked one), else the first candidate.
std::filesystem::path FindIso();
void SetIso(const std::filesystem::path& path);

std::string ToLower(std::string_view s);

// Read statistics per device path (logged at shutdown).
void NoteRead(const std::string& path, uint64_t bytes);
void DumpStats();

} // namespace ac6::fs
