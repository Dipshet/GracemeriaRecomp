// Xbox 360 disc images (GDFX), read through a memory mapping.
// The directory parsing follows UnleashedRecomp (UnleashedRecomp/install/iso_file_system.cpp, GPL-3.0,
// itself referenced from Xenia's disc image device, BSD). This version keeps the directory tree.
#include "fs/vfs.h"

#include "log.h"

#include <algorithm>
#include <cstring>
#include <stack>

#include <memory_mapped_file.h>

namespace ac6::fs {

namespace {

constexpr uint64_t kSectorSize = 2048;
constexpr char kMagic[] = "MICROSOFT*XBOX*MEDIA";
// Where the game partition starts in the image formats in use (plain XGD, XGD2 and XGD3 rips).
constexpr uint64_t kPartitionOffsets[] = { 0x00000000, 0x0000FB20, 0x00020600, 0x02080000, 0x0FD90000 };
constexpr uint8_t kAttributeDirectory = 0x10;

struct IsoDevice : Device
{
    bool volumeFound = false;   // the GDFX volume header was there
    bool cut = false;           // the directory or a file lies past the image's end
    uint64_t dataEnd = 0;       // the end of the furthest file or directory
    MemoryMappedFile file;

    const char* Kind() const override { return "disc image"; }
    bool ReadOnly() const override { return true; }

    uint64_t Read(const Entry& entry, uint64_t offset, void* destination, uint64_t length) override
    {
        if (entry.directory || offset >= entry.size)
            return 0;
        uint64_t n = std::min(length, entry.size - offset);
        uint64_t at = entry.offset + offset;
        if (at + n > file.size())
            n = at < file.size() ? file.size() - at : 0;
        std::memcpy(destination, file.data() + at, n);
        return n;
    }

    bool Parse()
    {
        const uint8_t* data = file.data();
        const uint64_t size = file.size();

        uint64_t partition = UINT64_MAX;
        for (uint64_t candidate : kPartitionOffsets)
        {
            uint64_t at = candidate + 32 * kSectorSize;
            if (at + sizeof(kMagic) - 1 <= size && std::memcmp(data + at, kMagic, sizeof(kMagic) - 1) == 0)
                partition = candidate;
        }
        if (partition == UINT64_MAX)
            return false;
        volumeFound = true;

        const uint64_t volume = partition + 32 * kSectorSize;
        uint32_t rootSector, rootSize;
        std::memcpy(&rootSector, data + volume + 20, 4);
        std::memcpy(&rootSize, data + volume + 24, 4);
        if (rootSize < 13 || rootSize > 32 * 1024 * 1024)
            return false;

        root.directory = true;
        root.attributes = kAttributeDirectory;

        // Each directory is a binary tree of entries (little-endian):
        //   u16 left, u16 right (in dwords from the directory start), u32 sector, u32 size, u8 attributes,
        //   u8 name length, name.
        struct Step { Entry* directory; uint64_t base; uint64_t offset; uint64_t limit; };
        std::stack<Step> work;
        work.push({ &root, partition + uint64_t(rootSector) * kSectorSize, 0, rootSize });
        while (!work.empty())
        {
            Step step = work.top();
            work.pop();
            const uint64_t at = step.base + step.offset;
            if (at + 14 > size)
            {
                cut = true;   // the directory lies past the image's end
                return false;
            }
            if (step.offset + 14 > step.limit)
                return false;

            uint16_t left, right;
            uint32_t sector, length;
            std::memcpy(&left, data + at, 2);
            std::memcpy(&right, data + at + 2, 2);
            std::memcpy(&sector, data + at + 4, 4);
            std::memcpy(&length, data + at + 8, 4);
            const uint8_t attributes = data[at + 12];
            const uint8_t nameLength = data[at + 13];
            if (left == 0xFFFF && right == 0xFFFF)
                continue;   // an empty directory's sector is all 0xFF
            if (at + 14 + nameLength > size)
            {
                cut = true;
                return false;
            }

            auto entry = std::make_unique<Entry>();
            entry->name.assign(reinterpret_cast<const char*>(data + at + 14), nameLength);
            entry->key = ToLower(entry->name);
            entry->attributes = attributes;
            entry->directory = (attributes & kAttributeDirectory) != 0;
            entry->offset = partition + uint64_t(sector) * kSectorSize;
            dataEnd = std::max(dataEnd, entry->offset + length);
            entry->size = entry->directory ? 0 : length;
            entry->parent = step.directory;

            if (left)
                work.push({ step.directory, step.base, uint64_t(left) * 4, step.limit });
            if (right)
                work.push({ step.directory, step.base, uint64_t(right) * 4, step.limit });
            if (entry->directory && length > 0)
                work.push({ entry.get(), entry->offset, 0, length });

            step.directory->children.push_back(std::move(entry));
        }
        totalBytes = size - partition;
        cut = dataEnd > size;
        Finish();
        return true;
    }
};

uint64_t FileTimeOf(const std::filesystem::path& path)
{
    std::error_code ec;
    auto t = std::filesystem::last_write_time(path, ec);
    if (ec)
        return 0;
    // file_time_type on Windows counts 100 ns units since 1601, like FILETIME.
    return static_cast<uint64_t>(t.time_since_epoch().count());
}

} // namespace

std::shared_ptr<Device> OpenIso(const std::filesystem::path& path)
{
    constexpr double kGB = 1024.0 * 1024.0 * 1024.0;
    auto device = std::make_shared<IsoDevice>();
    if (!device->file.open(path))
    {
        log::Error("Cannot open disc image {}", reinterpret_cast<const char*>(path.u8string().c_str()));
        return nullptr;
    }
    if (!device->Parse())
    {
        if (device->cut)
            log::Error("{} is incomplete: it ends ({:.2f} GB) before its own directory. Copy or dump it again",
                reinterpret_cast<const char*>(path.u8string().c_str()), double(device->file.size()) / kGB);
        else if (device->volumeFound)
            log::Error("{} is damaged: its directory can't be read", reinterpret_cast<const char*>(path.u8string().c_str()));
        else
            log::Error("{} is not an Xbox 360 disc image (no GDFX volume found)", reinterpret_cast<const char*>(path.u8string().c_str()));
        return nullptr;
    }
    // Files past the image's end: a copy or a download cut short. (A trimmed image ends at its last file and passes.)
    if (device->cut)
    {
        log::Error("{} is incomplete: its files need {:.2f} GB, the image has {:.2f} GB. Copy or dump it again",
            reinterpret_cast<const char*>(path.u8string().c_str()), double(device->dataEnd) / kGB, double(device->file.size()) / kGB);
        return nullptr;
    }
    device->timestamp = FileTimeOf(path);

    size_t files = 0, dirs = 0;
    std::stack<const Entry*> walk;
    walk.push(&device->root);
    while (!walk.empty())
    {
        const Entry* e = walk.top();
        walk.pop();
        for (auto& c : e->children)
        {
            (c->directory ? dirs : files)++;
            if (c->directory)
                walk.push(c.get());
        }
    }
    log::Info("Disc image {}: {} files, {} directories", reinterpret_cast<const char*>(path.filename().u8string().c_str()), files, dirs);
    return device;
}

} // namespace ac6::fs
