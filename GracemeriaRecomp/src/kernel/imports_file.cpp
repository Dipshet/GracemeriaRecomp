// File imports over the virtual filesystem (fs/vfs.h).
//
// Two kinds of files:
//   - disc files (read-only image entries).
//   - host-folder files (mounted save containers, fs/save_store.h). Handles opened for writing work on
//     the file's staging copy, shared by every writable handle of that file. Closing the last one commits
//     it atomically over the real file (the previous version kept in the container's backup folder) when
//     any handle changed it and none failed. A failed copy, write, resize or flush is never committed: the
//     previous version stays.
//
// I/O completes synchronously: the status block is filled, the caller's event (if any) is signalled and
// the call returns the final status, which is valid NT behaviour for any request.
#include "fs/save_store.h"
#include "game/packages.h"
#include "fs/vfs.h"
#include "kernel/dispatcher.h"
#include "kernel/function.h"
#include "kernel/object.h"
#include "log.h"
#include "memory.h"
#include "platform/platform.h"

#include <algorithm>
#include <atomic>
#include <format>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace ac6::kernel {

namespace {

constexpr uint32_t X_STATUS_NO_MORE_FILES         = 0x80000006;
constexpr uint32_t X_STATUS_END_OF_FILE           = 0xC0000011;
constexpr uint32_t X_STATUS_OBJECT_NAME_NOT_FOUND = 0xC0000034;
constexpr uint32_t X_STATUS_OBJECT_NAME_COLLISION = 0xC0000035;
constexpr uint32_t X_STATUS_OBJECT_PATH_NOT_FOUND = 0xC000003A;
constexpr uint32_t X_STATUS_INVALID_INFO_CLASS    = 0xC0000003;
constexpr uint32_t X_STATUS_INFO_LENGTH_MISMATCH  = 0xC0000004;
constexpr uint32_t X_STATUS_FILE_IS_A_DIRECTORY   = 0xC00000BA;
constexpr uint32_t X_STATUS_NOT_A_DIRECTORY       = 0xC0000103;
constexpr uint32_t X_STATUS_DIRECTORY_NOT_EMPTY   = 0xC0000101;
constexpr uint32_t X_STATUS_DISK_FULL             = 0xC000007F;
constexpr uint32_t X_STATUS_DEVICE_DATA_ERROR     = 0xC000009C;

// Create dispositions, options, access bits and the "information" results of a create.
constexpr uint32_t X_FILE_SUPERSEDE = 0, X_FILE_OPEN = 1, X_FILE_CREATE = 2, X_FILE_OPEN_IF = 3, X_FILE_OVERWRITE = 4, X_FILE_OVERWRITE_IF = 5;
constexpr uint32_t X_FILE_DIRECTORY_FILE = 0x1, X_FILE_NON_DIRECTORY_FILE = 0x40, X_FILE_DELETE_ON_CLOSE = 0x1000;
constexpr uint32_t kWriteAccess = 0x40000000 /* GENERIC_WRITE */ | 0x10000000 /* GENERIC_ALL */ | 0x2 /* FILE_WRITE_DATA */ |
    0x4 /* FILE_APPEND_DATA */;
constexpr uint32_t X_FILE_SUPERSEDED = 0, X_FILE_OPENED = 1, X_FILE_CREATED = 2, X_FILE_OVERWRITTEN = 3;

constexpr uint32_t X_FILE_ATTRIBUTE_READONLY = 0x01, X_FILE_ATTRIBUTE_DIRECTORY = 0x10, X_FILE_ATTRIBUTE_NORMAL = 0x80;

struct XIoStatusBlock
{
    be<uint32_t> status;
    be<uint32_t> information;
};

// What a query needs to know about a file or directory, whichever device it is on.
struct FileStat
{
    bool directory = false;
    uint64_t size = 0;
    uint64_t allocation = 0;
    uint32_t attributes = X_FILE_ATTRIBUTE_NORMAL;
    uint64_t creation = 0, access = 0, write = 0;
    uint64_t index = 0;
};

struct HostListing
{
    std::string name;
    FileStat stat;
};

FileStat StatHostPath(const std::filesystem::path& path)
{
    FileStat s;
    const platform::PathInfo info = platform::StatPath(path);
    if (!info.exists)
        return s;
    s.directory = info.directory;
    s.size = info.size;
    s.allocation = (s.size + 0x3FFF) & ~uint64_t(0x3FFF);
    s.attributes = s.directory ? X_FILE_ATTRIBUTE_DIRECTORY : X_FILE_ATTRIBUTE_NORMAL;
    s.creation = info.creation;
    s.access = info.access;
    s.write = info.write;
    return s;
}

// The writable handles of each host file (by its staging path): the first open makes the staging copy, the last close
// commits or discards it.
struct Staging
{
    int handles = 0;
    bool changed = false;   // a handle wrote, truncated or created it
    bool failed = false;    // a handle's write, resize or flush failed
};
std::mutex s_stagingMutex;
std::map<std::filesystem::path, Staging> s_staging;

struct File : KernelObject
{
    XDispatcherHeader header{};    // files are waitable: signalled whenever no I/O is in flight (always)
    fs::Resolution where;
    std::mutex mutex;
    uint64_t position = 0;
    size_t directoryCursor = 0;
    std::string directoryMask;

    // Host-folder files.
    bool host = false;
    bool hostDirectory = false;
    platform::HostFile hostFile;
    bool writable = false;         // works on `staged`, committed over `target` at the last writable close
    bool mustCommit = false;       // written, truncated or newly created
    bool failed = false;           // a write, resize or flush failed: the staging copy is never committed
    bool deleteOnClose = false;
    std::filesystem::path target, staged, container;
    std::vector<HostListing> listing;

    explicit File(fs::Resolution r) : where(std::move(r)) { InitializeHeader(header, kObjEventNotification, 1); }
    ~File() override { CloseHost(); }
    uint32_t DispatcherHeader() override { return g_memory.MapVirtual(&header); }
    const char* TypeName() const override { return "file"; }
    const fs::Entry& entry() const { return *where.entry; }
    bool IsDirectory() const { return host ? hostDirectory : entry().directory; }

    FileStat Stat()
    {
        if (!host)
        {
            const fs::Entry& e = entry();
            const fs::Device& d = *where.device;
            FileStat s;
            s.directory = e.directory;
            s.size = e.size;
            s.allocation = (e.size + d.sectorSize - 1) / d.sectorSize * d.sectorSize;
            s.attributes = (e.attributes & ~X_FILE_ATTRIBUTE_NORMAL) | (e.directory ? X_FILE_ATTRIBUTE_DIRECTORY : 0) |
                (d.ReadOnly() ? X_FILE_ATTRIBUTE_READONLY : 0);
            if (!s.attributes)
                s.attributes = X_FILE_ATTRIBUTE_NORMAL;
            s.creation = s.access = s.write = d.timestamp;
            s.index = e.offset;
            return s;
        }
        if (hostFile.IsOpen())
        {
            FileStat s;
            if (const platform::PathInfo info = hostFile.Info(); info.exists)
            {
                s.size = info.size;
                s.allocation = (s.size + 0x3FFF) & ~uint64_t(0x3FFF);
                s.creation = info.creation;
                s.access = info.access;
                s.write = info.write;
                s.index = info.index;
            }
            return s;
        }
        return StatHostPath(target);
    }

    void CloseHost()
    {
        if (!host)
            return;
        if (hostFile.IsOpen())
        {
            if (writable && mustCommit && !deleteOnClose && !failed && (!hostFile.Flush() || fs::FaultInjected("flush")))
            {
                failed = true;
                log::Error("Flushing '{}' failed: it isn't saved", where.devicePath);
            }
            hostFile.Close();
        }
        std::error_code ec;
        // The shared staging copy: add this handle's changed and failed flags, and if this is the last writable
        // handle, decide whether the copy is committed or discarded.
        bool last = true, changed = mustCommit, broken = failed;
        if (writable && !staged.empty())
        {
            std::lock_guard lock(s_stagingMutex);
            if (auto it = s_staging.find(staged); it != s_staging.end())
            {
                Staging& s = it->second;
                s.changed |= mustCommit;
                s.failed |= failed;
                last = --s.handles <= 0;
                changed = s.changed;
                broken = s.failed;
                if (last)
                    s_staging.erase(it);
            }
        }
        if (deleteOnClose)
        {
            if (writable && last)
                std::filesystem::remove(staged, ec);
            if (hostDirectory)
                std::filesystem::remove(target, ec);
            else
                fs::DeleteWithBackup(container, target);
        }
        else if (writable && last)
        {
            if (changed && broken)
                log::Error("'{}' isn't saved: a write to it failed. The previous version is kept; the failed copy is in {}",
                    where.devicePath, reinterpret_cast<const char*>(staged.u8string().c_str()));
            else if (changed)
                fs::CommitStaged(container, staged, target);
            else
                std::filesystem::remove(staged, ec);
        }
        host = false;
    }
};

std::string GuestString(const XAnsiString* s)
{
    return s && s->buffer ? std::string(g_memory.As<char>(s->buffer), s->length) : std::string();
}

void Complete(XIoStatusBlock* io, uint32_t status, uint32_t information, uint32_t event = 0)
{
    if (io)
    {
        io->status = status;
        io->information = information;
    }
    if (event)
        if (KernelObject* e = LookupHandle(event); e && e->DispatcherHeader())
            SetEvent(e->DispatcherHeader());
}

struct XFileNetworkOpenInformation
{
    be<uint64_t> creationTime, lastAccessTime, lastWriteTime, changeTime;
    be<uint64_t> allocationSize, endOfFile;
    be<uint32_t> fileAttributes;
    be<uint32_t> pad;
};
static_assert(sizeof(XFileNetworkOpenInformation) == 0x38);

void FillNetworkOpen(XFileNetworkOpenInformation& info, const FileStat& s)
{
    info.creationTime = s.creation;
    info.lastAccessTime = s.access;
    info.lastWriteTime = info.changeTime = s.write;
    info.allocationSize = s.allocation;
    info.endOfFile = s.size;
    info.fileAttributes = s.attributes;
    info.pad = 0;
}

bool MaskMatches(std::string_view mask, std::string_view name)
{
    // Case-insensitive '*' / '?' glob ("*" and "*.*" match everything).
    if (mask.empty() || mask == "*" || mask == "*.*")
        return true;
    size_t m = 0, n = 0, star = std::string_view::npos, mark = 0;
    auto lower = [](char c) { return c >= 'A' && c <= 'Z' ? char(c - 'A' + 'a') : c; };
    while (n < name.size())
    {
        if (m < mask.size() && (mask[m] == '?' || lower(mask[m]) == lower(name[n])))
        {
            m++;
            n++;
        }
        else if (m < mask.size() && mask[m] == '*')
        {
            star = m++;
            mark = n;
        }
        else if (star != std::string_view::npos)
        {
            m = star + 1;
            n = ++mark;
        }
        else
        {
            return false;
        }
    }
    while (m < mask.size() && mask[m] == '*')
        m++;
    return m == mask.size();
}

// Resolves an object-attributes path, honouring a directory root handle.
fs::Resolution ResolveAttributes(const XObjectAttributes* attributes, std::string& path)
{
    path = attributes ? GuestString(g_memory.As<XAnsiString>(attributes->name)) : std::string();
    if (attributes && attributes->rootDirectory && attributes->rootDirectory != 0xFFFFFFFD)
        if (auto* dir = LookupHandleAs<File>(attributes->rootDirectory); dir && dir->IsDirectory())
            return fs::Resolve(path, &dir->where);
    // 0xFFFFFFFD is the object-namespace root: the path is absolute ("game:\...").
    return fs::Resolve(path);
}

uint32_t Fail(const char* api, const std::string& path, uint32_t access, uint32_t disposition, uint32_t status, XIoStatusBlock* io)
{
    log::Info("{}('{}', access {:08X}, disposition {}) -> {:08X}", api, path, access, disposition, status);
    Complete(io, status, 0);
    return status;
}

uint32_t OpenHost(const char* api, const std::string& path, fs::Resolution r, be<uint32_t>* handle, uint32_t access,
    XIoStatusBlock* io, uint32_t disposition, uint32_t options)
{
    const bool exists = r.hostExists, isDirectory = r.hostDirectory;
    auto& device = static_cast<fs::HostFolderDevice&>(*r.device);
    std::error_code ec;
    uint32_t information = X_FILE_OPENED;

    auto file = [&]() -> File* {
        auto* f = CreateKernelObject<File>(r);
        if (!f)
            return nullptr;
        f->host = true;
        f->target = r.hostPath;
        f->container = device.rootPath;
        f->deleteOnClose = (options & X_FILE_DELETE_ON_CLOSE) != 0;
        return f;
    };

    // Directories.
    if ((options & X_FILE_DIRECTORY_FILE) || (exists && isDirectory))
    {
        if (exists && !isDirectory)
            return Fail(api, path, access, disposition, X_STATUS_NOT_A_DIRECTORY, io);
        if (exists && (options & X_FILE_NON_DIRECTORY_FILE))
            return Fail(api, path, access, disposition, X_STATUS_FILE_IS_A_DIRECTORY, io);
        if (exists && disposition == X_FILE_CREATE)
            return Fail(api, path, access, disposition, X_STATUS_OBJECT_NAME_COLLISION, io);
        if (!exists)
        {
            if (!r.parentFound)
                return Fail(api, path, access, disposition, X_STATUS_OBJECT_PATH_NOT_FOUND, io);
            if (disposition != X_FILE_CREATE && disposition != X_FILE_OPEN_IF)
                return Fail(api, path, access, disposition, X_STATUS_OBJECT_NAME_NOT_FOUND, io);
            std::filesystem::create_directory(r.hostPath, ec);
            information = X_FILE_CREATED;
        }
        File* f = file();
        if (!f)
            return X_STATUS_NO_MEMORY;
        f->hostDirectory = true;
        *handle = CreateHandle(f);
        log::Info("{}('{}') -> {:08X} (directory{})", api, path, uint32_t(*handle), information == X_FILE_CREATED ? ", created" : "");
        Complete(io, X_STATUS_SUCCESS, information);
        return X_STATUS_SUCCESS;
    }

    // Files.
    if (!r.parentFound)
        return Fail(api, path, access, disposition, X_STATUS_OBJECT_PATH_NOT_FOUND, io);
    bool truncate = false;
    switch (disposition)
    {
    case X_FILE_OPEN:
        if (!exists)
            return Fail(api, path, access, disposition, X_STATUS_OBJECT_NAME_NOT_FOUND, io);
        break;
    case X_FILE_CREATE:
        if (exists)
            return Fail(api, path, access, disposition, X_STATUS_OBJECT_NAME_COLLISION, io);
        information = X_FILE_CREATED;
        break;
    case X_FILE_OPEN_IF:
        information = exists ? X_FILE_OPENED : X_FILE_CREATED;
        break;
    case X_FILE_OVERWRITE:
        if (!exists)
            return Fail(api, path, access, disposition, X_STATUS_OBJECT_NAME_NOT_FOUND, io);
        truncate = true;
        information = X_FILE_OVERWRITTEN;
        break;
    case X_FILE_OVERWRITE_IF:
        truncate = true;
        information = exists ? X_FILE_OVERWRITTEN : X_FILE_CREATED;
        break;
    case X_FILE_SUPERSEDE:
    default:
        truncate = true;
        information = exists ? X_FILE_SUPERSEDED : X_FILE_CREATED;
        break;
    }

    File* f = file();
    if (!f)
        return X_STATUS_NO_MEMORY;
    const bool created = information == X_FILE_CREATED;
    f->writable = (access & kWriteAccess) || truncate || created;
    if (f->writable)
    {
        // Work on the file's staging copy. The real file is only replaced at the last writable close. The first writable
        // handle copies the file into it. A copy that fails or comes up short fails the open, so the game never reads
        // (and never saves back) an empty, stale or partial copy.
        f->staged = fs::StagingPathFor(f->container, f->target);
        std::lock_guard lock(s_stagingMutex);
        Staging& s = s_staging[f->staged];
        const bool shared = s.handles > 0;
        if (!shared && exists && !truncate)
        {
            std::filesystem::copy_file(f->target, f->staged, std::filesystem::copy_options::overwrite_existing, ec);
            std::error_code sizeError;
            if (!ec && std::filesystem::file_size(f->staged, sizeError) != std::filesystem::file_size(f->target, sizeError))
                ec = std::make_error_code(std::errc::io_error);
            if (!ec && fs::FaultInjected("copy"))
                ec = std::make_error_code(std::errc::permission_denied);
            if (ec)
            {
                s_staging.erase(f->staged);
                f->host = false;   // nothing to commit or discard
                f->writable = false;
                RetainObject(f);
                ReleaseObject(f);  // destroys it
                log::Error("{}('{}'): copying the file for writing failed ({}): not opened; the file is untouched", api, path,
                    ec.message());
                Complete(io, X_STATUS_ACCESS_DENIED, 0);
                return X_STATUS_ACCESS_DENIED;
            }
        }
        f->hostFile.Open(f->staged, truncate || (!exists && !shared) ? platform::HostFile::Mode::ReadWriteTruncate
                                                                     : platform::HostFile::Mode::ReadWrite);
        if (f->hostFile.IsOpen())
            s.handles++;
        else if (!shared)
            s_staging.erase(f->staged);
        f->mustCommit = truncate || created;
    }
    else
    {
        f->hostFile.Open(f->target, platform::HostFile::Mode::Read);
    }
    if (!f->hostFile.IsOpen())
    {
        const uint32_t error = f->hostFile.Error();
        f->host = false;   // nothing to commit
        RetainObject(f);
        ReleaseObject(f);  // destroys it
        log::Error("{}('{}'): host open failed (error {})", api, path, error);
        Complete(io, X_STATUS_ACCESS_DENIED, 0);
        return X_STATUS_ACCESS_DENIED;
    }
    *handle = CreateHandle(f);
    log::Info("{}('{}') -> {:08X} ({}{})", api, path, uint32_t(*handle), f->writable ? "write" : "read",
        information == X_FILE_CREATED ? ", created" : truncate ? ", truncated" : "");
    Complete(io, X_STATUS_SUCCESS, information);
    return X_STATUS_SUCCESS;
}

uint32_t OpenCommon(const char* api, be<uint32_t>* handle, uint32_t access, XObjectAttributes* attributes,
    XIoStatusBlock* io, uint32_t disposition, uint32_t options)
{
    if (handle)
        *handle = 0;
    std::string path;
    fs::Resolution r = ResolveAttributes(attributes, path);
    if (!r.device)
        return Fail(api, path, access, disposition, X_STATUS_OBJECT_PATH_NOT_FOUND, io);
    if (r.device->IsHostFolder())
        return OpenHost(api, path, std::move(r), handle, access, io, disposition, options);

    uint32_t status = X_STATUS_SUCCESS;
    if (!r.entry)
        status = r.parentFound ? X_STATUS_OBJECT_NAME_NOT_FOUND : X_STATUS_OBJECT_PATH_NOT_FOUND;
    else if (disposition == X_FILE_CREATE)
        status = X_STATUS_OBJECT_NAME_COLLISION;
    // Content packages (DLC) open even when write access is requested: the game opens them for read/write
    // (dlcolorname.dat: GENERIC_READ | GENERIC_WRITE) but only reads them. Writes to them fail.
    else if (((access & kWriteAccess) && std::string_view(r.device->FileSystemName()) != "STFS") ||
             disposition == X_FILE_SUPERSEDE || disposition == X_FILE_OVERWRITE || disposition == X_FILE_OVERWRITE_IF)
        status = X_STATUS_ACCESS_DENIED;
    else if ((options & X_FILE_DIRECTORY_FILE) && !r.entry->directory)
        status = X_STATUS_NOT_A_DIRECTORY;
    else if ((options & X_FILE_NON_DIRECTORY_FILE) && r.entry->directory)
        status = X_STATUS_FILE_IS_A_DIRECTORY;
    if (status != X_STATUS_SUCCESS)
        return Fail(api, path, access, disposition, status, io);

    auto* file = CreateKernelObject<File>(std::move(r));
    if (!file)
        return X_STATUS_NO_MEMORY;
    *handle = CreateHandle(file);
    log::Info("{}('{}') -> {:08X} ({} bytes)", api, path, uint32_t(*handle), file->entry().size);
    Complete(io, X_STATUS_SUCCESS, X_FILE_OPENED);
    return X_STATUS_SUCCESS;
}

// FILE_USE_FILE_POINTER_POSITION (-2) or no offset: the current position. -1 (FILE_WRITE_TO_END_OF_FILE):
// the end.
uint64_t ResolveOffset(File& file, const be<int64_t>* byteOffset)
{
    if (!byteOffset || int64_t(*byteOffset) == -2)
        return file.position;
    if (int64_t(*byteOffset) == -1)
        return file.Stat().size;
    return uint64_t(int64_t(*byteOffset));
}

} // namespace

uint32_t NtCreateFile(be<uint32_t>* handle, uint32_t desiredAccess, XObjectAttributes* attributes, XIoStatusBlock* io,
    be<uint64_t>* allocationSize, uint32_t fileAttributes, uint32_t shareAccess, uint32_t createDisposition, uint32_t createOptions)
{
    return OpenCommon("NtCreateFile", handle, desiredAccess, attributes, io, createDisposition, createOptions);
}

uint32_t NtOpenFile(be<uint32_t>* handle, uint32_t desiredAccess, XObjectAttributes* attributes, XIoStatusBlock* io,
    uint32_t shareAccess, uint32_t openOptions)
{
    return OpenCommon("NtOpenFile", handle, desiredAccess, attributes, io, X_FILE_OPEN, openOptions);
}

uint32_t NtReadFile(uint32_t handle, uint32_t event, uint32_t apcRoutine, uint32_t apcContext, XIoStatusBlock* io,
    uint8_t* buffer, uint32_t length, be<int64_t>* byteOffset)
{
    auto* file = LookupHandleAs<File>(handle);
    if (!file)
        return X_STATUS_INVALID_HANDLE;
    if (file->IsDirectory())
        return X_STATUS_INVALID_PARAMETER;
    if (apcRoutine)
    {
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            log::Warn("NtReadFile with an APC completion routine {:08X}: APCs are not delivered yet", apcRoutine);
        }
    }

    std::lock_guard lock(file->mutex);
    const uint64_t offset = ResolveOffset(*file, byteOffset);
    // ReadFile writes the buffer from the kernel, where a page not committed yet or write-watched can't fault into the
    // handler.
    g_memory.PrepareHostIo(g_memory.MapVirtual(buffer), length);
    g_memory.PrepareHostWrite(g_memory.MapVirtual(buffer), length);
    uint64_t n = 0;
    uint32_t status = X_STATUS_SUCCESS;
    if (file->host)
    {
        uint32_t got = 0;
        if (!file->hostFile.ReadAt(buffer, length, offset, got))
            status = X_STATUS_UNSUCCESSFUL;
        n = got;
        if (length && n == 0 && status == X_STATUS_SUCCESS)
            status = X_STATUS_END_OF_FILE;
    }
    else
    {
        if (length && offset >= file->entry().size)
            status = X_STATUS_END_OF_FILE;
        else if (length)
        {
            n = file->where.device->Read(file->entry(), offset, buffer, length);
            // Inside the file's size a read comes back whole, unless the game image is damaged or cut short: an error,
            // which the game takes to its disc-error screen (XamShowDirtyDiscErrorUI), instead of a success with part
            // of the data.
            if (n < std::min<uint64_t>(length, file->entry().size - offset))
            {
                static std::atomic<uint32_t> reported{ 0 };
                if (reported.fetch_add(1) < 8)
                    log::Error("NtReadFile('{}'): {} of {} bytes at offset {:#x} inside the file ({:#x} bytes): the game "
                               "image is damaged or incomplete", file->where.devicePath, n, length, offset, file->entry().size);
                status = X_STATUS_DEVICE_DATA_ERROR;
            }
        }
    }
    if (n < length)
    {
        // Short reads are legitimate at the end of a file, but a read the game expects to fill (e.g. at an
        // offset computed wrongly) shows up here first.
        static std::atomic<uint32_t> reported{ 0 };
        if (reported.fetch_add(1) < 32)
            log::Warn("NtReadFile('{}'): {} of {} bytes at offset {:#x} (file size {:#x}, offset argument {})", file->where.devicePath, n,
                length, offset, file->host ? 0 : file->entry().size, byteOffset ? std::format("{:#x}", uint64_t(int64_t(*byteOffset))) : "none");
    }
    // Mods' packages that grew: their sizes raised in DATA.TBL as the game reads it (game/packages.h).
    if (n && fs::ToLower(file->where.devicePath) == "data.tbl")
        game::PatchPackageTable(buffer, offset, n);
    file->position = offset + n;
    fs::NoteRead(file->where.devicePath, n);
    Complete(io, status, static_cast<uint32_t>(n), event);
    return status;
}

uint32_t NtWriteFile(uint32_t handle, uint32_t event, uint32_t apcRoutine, uint32_t apcContext, XIoStatusBlock* io,
    const uint8_t* buffer, uint32_t length, be<int64_t>* byteOffset)
{
    auto* file = LookupHandleAs<File>(handle);
    if (!file)
        return X_STATUS_INVALID_HANDLE;
    if (!file->host || !file->writable)
    {
        log::Warn("NtWriteFile to '{}' ({} bytes): not opened for writing", file->where.devicePath, length);
        Complete(io, X_STATUS_ACCESS_DENIED, 0, event);
        return X_STATUS_ACCESS_DENIED;
    }
    std::lock_guard lock(file->mutex);
    const uint64_t offset = ResolveOffset(*file, byteOffset);
    uint32_t wrote = 0;
    uint32_t status = X_STATUS_SUCCESS;
    g_memory.PrepareHostIo(g_memory.MapVirtual(buffer), length);   // (WriteFile reads it from the kernel)
    if (!file->hostFile.WriteAt(buffer, length, offset, wrote) || wrote != length ||
        (fs::FaultInjected("write") && (wrote = length / 2, true)))
    {
        log::Error("NtWriteFile to '{}' failed (error {}, {} of {} bytes): it won't be saved", file->where.devicePath,
            file->hostFile.Error(), wrote, length);
        status = X_STATUS_DISK_FULL;
        file->failed = true;   // never committed: the previous version stays
    }
    else
        file->mustCommit = true;
    file->position = offset + wrote;
    Complete(io, status, wrote, event);
    return status;
}

uint32_t NtFlushBuffersFile(uint32_t handle, XIoStatusBlock* io)
{
    auto* file = LookupHandleAs<File>(handle);
    if (!file)
        return X_STATUS_INVALID_HANDLE;
    if (file->host && file->hostFile.IsOpen())
        file->hostFile.Flush();
    Complete(io, X_STATUS_SUCCESS, 0);
    return X_STATUS_SUCCESS;
}

uint32_t NtQueryInformationFile(uint32_t handle, XIoStatusBlock* io, uint8_t* info, uint32_t length, uint32_t infoClass)
{
    auto* file = LookupHandleAs<File>(handle);
    if (!file)
        return X_STATUS_INVALID_HANDLE;
    const FileStat s = file->Stat();

    auto fill = [&](auto value) -> uint32_t {
        if (length < sizeof(value))
        {
            Complete(io, X_STATUS_INFO_LENGTH_MISMATCH, 0);
            return X_STATUS_INFO_LENGTH_MISMATCH;
        }
        std::memcpy(info, &value, sizeof(value));
        Complete(io, X_STATUS_SUCCESS, sizeof(value));
        return X_STATUS_SUCCESS;
    };

    switch (infoClass)
    {
    case 4:   // FileBasicInformation
    {
        struct { be<uint64_t> creation, access, write, change; be<uint32_t> attributes, pad; } v{};
        v.creation = s.creation;
        v.access = s.access;
        v.write = v.change = s.write;
        v.attributes = s.attributes;
        return fill(v);
    }
    case 5:   // FileStandardInformation
    {
        struct { be<uint64_t> allocationSize, endOfFile; be<uint32_t> links; uint8_t deletePending, directory, pad[2]; } v{};
        v.allocationSize = s.allocation;
        v.endOfFile = s.size;
        v.links = 1;
        v.deletePending = file->deleteOnClose ? 1 : 0;
        v.directory = s.directory ? 1 : 0;
        return fill(v);
    }
    case 6:   // FileInternalInformation
    {
        struct { be<uint64_t> index; } v{};
        v.index = s.index;
        return fill(v);
    }
    case 14:  // FilePositionInformation
    {
        struct { be<uint64_t> position; } v{};
        std::lock_guard lock(file->mutex);
        v.position = file->position;
        return fill(v);
    }
    case 20:  // FileEndOfFileInformation
    {
        struct { be<uint64_t> endOfFile; } v{};
        v.endOfFile = s.size;
        return fill(v);
    }
    case 34:  // FileNetworkOpenInformation
    {
        XFileNetworkOpenInformation v{};
        FillNetworkOpen(v, s);
        return fill(v);
    }
    default:
        log::Warn("NtQueryInformationFile: unsupported class {} ({} bytes)", infoClass, length);
        Complete(io, X_STATUS_INVALID_INFO_CLASS, 0);
        return X_STATUS_INVALID_INFO_CLASS;
    }
}

uint32_t NtSetInformationFile(uint32_t handle, XIoStatusBlock* io, const uint8_t* info, uint32_t length, uint32_t infoClass)
{
    auto* file = LookupHandleAs<File>(handle);
    if (!file)
        return X_STATUS_INVALID_HANDLE;
    auto done = [&](uint32_t status) {
        Complete(io, status, 0);
        return status;
    };
    std::lock_guard lock(file->mutex);
    switch (infoClass)
    {
    case 14:  // FilePositionInformation
        if (length < 8)
            return done(X_STATUS_INFO_LENGTH_MISMATCH);
        file->position = *reinterpret_cast<const be<uint64_t>*>(info);
        return done(X_STATUS_SUCCESS);

    case 4:   // FileBasicInformation (times/attributes): accepted, host times are kept
    case 19:  // FileAllocationInformation: space is allocated on demand
        return done(file->host ? X_STATUS_SUCCESS : X_STATUS_ACCESS_DENIED);

    case 20:  // FileEndOfFileInformation
    {
        if (!file->host || !file->writable || length < 8)
            return done(X_STATUS_ACCESS_DENIED);
        if (!file->hostFile.Resize(uint64_t(*reinterpret_cast<const be<uint64_t>*>(info))))
        {
            file->failed = true;   // never committed
            return done(X_STATUS_DISK_FULL);
        }
        file->mustCommit = true;
        return done(X_STATUS_SUCCESS);
    }

    case 13:  // FileDispositionInformation: { BOOLEAN DeleteFile }
        if (!file->host || length < 1)
            return done(X_STATUS_ACCESS_DENIED);
        if (info[0] && file->hostDirectory)
        {
            std::error_code ec;
            if (!std::filesystem::is_empty(file->target, ec))
                return done(X_STATUS_DIRECTORY_NOT_EMPTY);
        }
        file->deleteOnClose = info[0] != 0;
        return done(X_STATUS_SUCCESS);

    case 10:  // FileRenameInformation: { BOOLEAN ReplaceIfExists; HANDLE RootDirectory; ANSI_STRING FileName }
    {
        if (!file->host || length < 16)
            return done(X_STATUS_ACCESS_DENIED);
        const bool replace = info[0] != 0;
        const auto* name = reinterpret_cast<const XAnsiString*>(info + 8);
        const fs::Resolution to = fs::Resolve(GuestString(name));
        if (!to.device || to.device != file->where.device)
            return done(X_STATUS_ACCESS_DENIED);
        if (to.hostExists && !replace)
            return done(X_STATUS_OBJECT_NAME_COLLISION);
        std::error_code ec;
        if (std::filesystem::exists(file->target, ec))
        {
            if (to.hostExists)
                fs::DeleteWithBackup(file->container, to.hostPath);
            std::filesystem::rename(file->target, to.hostPath, ec);
            if (ec)
                return done(X_STATUS_ACCESS_DENIED);
        }
        log::Info("Renamed '{}' -> '{}'", file->where.devicePath, to.devicePath);
        file->target = to.hostPath;   // a pending staged write now commits under the new name
        file->where.devicePath = to.devicePath;
        return done(X_STATUS_SUCCESS);
    }

    default:
        log::Warn("NtSetInformationFile('{}'): unsupported class {} ({} bytes)", file->where.devicePath, infoClass, length);
        return done(X_STATUS_ACCESS_DENIED);
    }
}

uint32_t NtQueryVolumeInformationFile(uint32_t handle, XIoStatusBlock* io, uint8_t* info, uint32_t length, uint32_t infoClass)
{
    auto* file = LookupHandleAs<File>(handle);
    if (!file)
        return X_STATUS_INVALID_HANDLE;
    const fs::Device& d = *file->where.device;
    switch (infoClass)
    {
    case 1:   // FileFsVolumeInformation (label left empty)
    {
        struct { be<uint64_t> creation; be<uint32_t> serial, labelLength; uint8_t supportsObjects, pad[3]; } v{};
        v.creation = d.timestamp;
        v.serial = 0xAC600006;
        if (length < sizeof(v))
            return X_STATUS_INFO_LENGTH_MISMATCH;
        std::memcpy(info, &v, sizeof(v));
        Complete(io, X_STATUS_SUCCESS, sizeof(v));
        return X_STATUS_SUCCESS;
    }
    case 3:   // FileFsSizeInformation
    {
        struct { be<uint64_t> totalUnits, availableUnits; be<uint32_t> sectorsPerUnit, bytesPerSector; } v{};
        v.totalUnits = d.totalBytes / d.sectorSize;
        v.availableUnits = d.ReadOnly() ? 0 : v.totalUnits / 2;
        // A save folder reports its drive's real free space, capped at half the reported total (so the game's own
        // "not enough space" check works on a nearly full drive).
        if (const auto* folder = dynamic_cast<const fs::HostFolderDevice*>(&d))
        {
            std::error_code ec;
            const std::filesystem::space_info space = std::filesystem::space(folder->rootPath, ec);
            if (!ec)
                v.availableUnits = std::min<uint64_t>(v.availableUnits, space.available / d.sectorSize);
        }
        v.sectorsPerUnit = 1;
        v.bytesPerSector = d.sectorSize;
        if (length < sizeof(v))
            return X_STATUS_INFO_LENGTH_MISMATCH;
        std::memcpy(info, &v, sizeof(v));
        Complete(io, X_STATUS_SUCCESS, sizeof(v));
        return X_STATUS_SUCCESS;
    }
    case 5:   // FileFsAttributeInformation
    {
        struct { be<uint32_t> attributes; be<int32_t> maxComponent; be<uint32_t> nameLength; char name[4]; } v{};
        v.attributes = 0;
        const bool stfs = std::string_view(d.FileSystemName()) == "STFS";
        v.maxComponent = stfs ? 42 : 255;
        v.nameLength = 4;
        std::memcpy(v.name, d.FileSystemName(), 4);
        if (length < sizeof(v))
            return X_STATUS_INFO_LENGTH_MISMATCH;
        std::memcpy(info, &v, sizeof(v));
        Complete(io, X_STATUS_SUCCESS, sizeof(v));
        return X_STATUS_SUCCESS;
    }
    default:
        log::Warn("NtQueryVolumeInformationFile: unsupported class {} ({} bytes)", infoClass, length);
        Complete(io, X_STATUS_INVALID_INFO_CLASS, 0);
        return X_STATUS_INVALID_INFO_CLASS;
    }
}

uint32_t NtQueryDirectoryFile(uint32_t handle, uint32_t event, uint32_t apcRoutine, uint32_t apcContext, XIoStatusBlock* io,
    uint8_t* info, uint32_t length, XAnsiString* fileMask, uint32_t restartScan)
{
    auto* file = LookupHandleAs<File>(handle);
    if (!file || !file->IsDirectory())
        return X_STATUS_INVALID_HANDLE;

    std::lock_guard lock(file->mutex);
    if (restartScan || (fileMask && file->directoryCursor == 0))
    {
        file->directoryCursor = 0;
        file->directoryMask = fileMask ? GuestString(fileMask) : std::string();
        if (file->host)
        {
            // Snapshot the host directory at the start of each scan.
            file->listing.clear();
            std::error_code ec;
            for (auto& e : std::filesystem::directory_iterator(file->target, ec))
                file->listing.push_back({ reinterpret_cast<const char*>(e.path().filename().u8string().c_str()), StatHostPath(e.path()) });
            std::sort(file->listing.begin(), file->listing.end(), [](auto& a, auto& b) { return fs::ToLower(a.name) < fs::ToLower(b.name); });
        }
    }

    // A FILE_DIRECTORY_INFORMATION entry with its ANSI name.
    struct Header
    {
        be<uint32_t> nextEntryOffset, fileIndex;
        be<uint64_t> creation, access, write, change, endOfFile, allocationSize;
        be<uint32_t> attributes, nameLength;
    };
    static_assert(sizeof(Header) == 0x40);

    const size_t count = file->host ? file->listing.size() : file->entry().children.size();
    const fs::Device& d = *file->where.device;
    uint32_t written = 0, lastHeader = UINT32_MAX;
    while (file->directoryCursor < count)
    {
        std::string name;
        FileStat s;
        if (file->host)
        {
            name = file->listing[file->directoryCursor].name;
            s = file->listing[file->directoryCursor].stat;
        }
        else
        {
            const fs::Entry& c = *file->entry().children[file->directoryCursor];
            name = c.name;
            s.directory = c.directory;
            s.size = c.size;
            s.allocation = (c.size + d.sectorSize - 1) / d.sectorSize * d.sectorSize;
            s.attributes = (c.directory ? X_FILE_ATTRIBUTE_DIRECTORY : 0) | X_FILE_ATTRIBUTE_READONLY;
            s.creation = s.access = s.write = d.timestamp;
        }
        if (!MaskMatches(file->directoryMask, name))
        {
            file->directoryCursor++;
            continue;
        }
        const uint32_t size = static_cast<uint32_t>(sizeof(Header) + name.size());
        const uint32_t at = (written + 7) & ~7u;
        if (at + size > length)
            break;
        Header h{};
        h.fileIndex = static_cast<uint32_t>(file->directoryCursor);
        h.creation = s.creation;
        h.access = s.access;
        h.write = h.change = s.write;
        h.endOfFile = s.size;
        h.allocationSize = s.allocation;
        h.attributes = s.attributes;
        h.nameLength = static_cast<uint32_t>(name.size());
        std::memcpy(info + at, &h, sizeof(h));
        std::memcpy(info + at + sizeof(h), name.data(), name.size());
        if (lastHeader != UINT32_MAX)
            reinterpret_cast<Header*>(info + lastHeader)->nextEntryOffset = at - lastHeader;
        lastHeader = at;
        written = at + size;
        file->directoryCursor++;
        // One entry per call, as the console's kernel returns them: the game's FindNextFile reads only
        // the first entry of the buffer, so packing more would skip files (DLC skins).
        break;
    }
    const uint32_t status = written ? X_STATUS_SUCCESS : X_STATUS_NO_MORE_FILES;
    Complete(io, status, written, event);
    return status;
}

uint32_t NtQueryFullAttributesFile(XObjectAttributes* attributes, XFileNetworkOpenInformation* info)
{
    std::string path;
    fs::Resolution r = ResolveAttributes(attributes, path);
    if (r.device && r.device->IsHostFolder() && r.hostExists)
    {
        FillNetworkOpen(*info, StatHostPath(r.hostPath));
        return X_STATUS_SUCCESS;
    }
    if (!r.device || !r.entry)
    {
        log::Info("NtQueryFullAttributesFile('{}') -> not found", path);
        return r.parentFound ? X_STATUS_OBJECT_NAME_NOT_FOUND : X_STATUS_OBJECT_PATH_NOT_FOUND;
    }
    FileStat s;
    const fs::Device& d = *r.device;
    s.directory = r.entry->directory;
    s.size = r.entry->size;
    s.allocation = (s.size + d.sectorSize - 1) / d.sectorSize * d.sectorSize;
    s.attributes = (s.directory ? X_FILE_ATTRIBUTE_DIRECTORY : 0) | X_FILE_ATTRIBUTE_READONLY;
    s.creation = s.access = s.write = d.timestamp;
    FillNetworkOpen(*info, s);
    return X_STATUS_SUCCESS;
}

} // namespace ac6::kernel

AC6_IMPORT(NtCreateFile)
AC6_IMPORT(NtOpenFile)
AC6_IMPORT(NtReadFile)
AC6_IMPORT(NtWriteFile)
AC6_IMPORT(NtFlushBuffersFile)
AC6_IMPORT(NtQueryInformationFile)
AC6_IMPORT(NtSetInformationFile)
AC6_IMPORT(NtQueryVolumeInformationFile)
AC6_IMPORT(NtQueryDirectoryFile)
AC6_IMPORT(NtQueryFullAttributesFile)
