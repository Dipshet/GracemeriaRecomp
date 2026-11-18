#include "fs/vfs.h"

#include "log.h"
#include "platform/platform.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <mutex>

namespace ac6::fs {

namespace {

std::mutex s_mountMutex;
std::map<std::string, std::shared_ptr<Device>, std::less<>> s_mounts;   // lower-case device name

bool IsSeparator(char c) { return c == '\\' || c == '/'; }

// Splits "a\b/c" into components, dropping empty ones and ".".
std::vector<std::string_view> Components(std::string_view path)
{
    std::vector<std::string_view> parts;
    size_t i = 0;
    while (i < path.size())
    {
        while (i < path.size() && IsSeparator(path[i]))
            i++;
        size_t start = i;
        while (i < path.size() && !IsSeparator(path[i]))
            i++;
        if (i > start && path.substr(start, i - start) != ".")
            parts.push_back(path.substr(start, i - start));
    }
    return parts;
}

} // namespace

std::string ToLower(std::string_view s)
{
    std::string out(s);
    for (char& c : out)
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
    return out;
}

const Entry* Entry::Find(std::string_view lowerName) const
{
    auto it = std::lower_bound(children.begin(), children.end(), lowerName,
        [](const std::unique_ptr<Entry>& e, std::string_view k) { return e->key < k; });
    return it != children.end() && (*it)->key == lowerName ? it->get() : nullptr;
}

std::string Entry::Path() const
{
    if (!parent)
        return {};
    std::string up = parent->Path();
    return up.empty() ? name : up + "\\" + name;
}

const Entry* Device::Resolve(std::string_view path) const
{
    const Entry* e = &root;
    for (std::string_view part : Components(path))
    {
        if (part == "..")
        {
            e = e->parent ? e->parent : e;
            continue;
        }
        if (!e->directory)
            return nullptr;
        e = e->Find(ToLower(part));
        if (!e)
            return nullptr;
    }
    return e;
}

void Device::Finish()
{
    std::vector<Entry*> stack{ &root };
    while (!stack.empty())
    {
        Entry* e = stack.back();
        stack.pop_back();
        std::sort(e->children.begin(), e->children.end(),
            [](const std::unique_ptr<Entry>& a, const std::unique_ptr<Entry>& b) { return a->key < b->key; });
        for (auto& c : e->children)
            if (c->directory)
                stack.push_back(c.get());
    }
}

HostFolderDevice::HostFolderDevice(std::filesystem::path folder) : rootPath(std::move(folder))
{
    root.directory = true;
    root.attributes = 0x10;
    sectorSize = 0x4000;   // console HDD cluster size
    totalBytes = 20ull << 30;
}

std::filesystem::path HostFolderDevice::Map(std::string_view devicePath) const
{
    std::filesystem::path path = rootPath;
    std::error_code ec;
    for (std::string_view part : Components(devicePath))
    {
        if (part == "..")
        {
            if (path != rootPath)
                path = path.parent_path();
            continue;
        }
        const std::string key = ToLower(part);
        std::filesystem::path match;
        if (std::filesystem::is_directory(path, ec))
            for (auto& e : std::filesystem::directory_iterator(path, ec))
                if (ToLower(reinterpret_cast<const char*>(e.path().filename().u8string().c_str())) == key)
                {
                    match = e.path();
                    break;
                }
        path = match.empty() ? path / std::filesystem::path(std::u8string(part.begin(), part.end())) : match;
    }
    return path;
}

void Unmount(std::string_view deviceName)
{
    std::lock_guard lock(s_mountMutex);
    if (s_mounts.erase(ToLower(deviceName)))
        log::Info("Unmounted '{}'", deviceName);
}

void Mount(std::string_view deviceName, std::shared_ptr<Device> device)
{
    std::lock_guard lock(s_mountMutex);
    s_mounts[ToLower(deviceName)] = std::move(device);
    log::Info("Mounted {} as '{}'", s_mounts[ToLower(deviceName)]->Kind(), deviceName);
}

Resolution Resolve(std::string_view guestPath, const Resolution* base)
{
    Resolution r;
    std::string_view rest = guestPath;

    if (base && base->device && base->entry)
    {
        r.device = base->device;
        r.devicePath = base->devicePath;
    }
    else
    {
        // "\??\game:\x" and "\Device\..." forms, otherwise "name:\x".
        if (rest.starts_with("\\??\\"))
            rest.remove_prefix(4);
        std::string mountName;
        if (size_t colon = rest.find(':'); colon != std::string_view::npos && colon < rest.find_first_of("\\/"))
        {
            mountName = ToLower(rest.substr(0, colon));
            rest.remove_prefix(colon + 1);
        }
        else if (ToLower(rest.substr(0, 15)) == "\\device\\cdrom0\\" || ToLower(rest) == "\\device\\cdrom0")
        {
            mountName = "\\device\\cdrom0";
            rest.remove_prefix(std::min<size_t>(rest.size(), 14));
        }
        std::lock_guard lock(s_mountMutex);
        auto it = s_mounts.find(mountName);
        if (it == s_mounts.end())
            return r;
        r.device = it->second;
    }

    std::string combined = r.devicePath;
    for (std::string_view part : Components(rest))
    {
        if (!combined.empty())
            combined += '\\';
        combined += part;
    }
    r.devicePath = combined;
    if (r.device->IsHostFolder())
    {
        auto& host = static_cast<HostFolderDevice&>(*r.device);
        std::error_code ec;
        r.hostPath = host.Map(combined);
        r.hostExists = std::filesystem::exists(r.hostPath, ec);
        r.hostDirectory = r.hostExists && std::filesystem::is_directory(r.hostPath, ec);
        r.parentFound = std::filesystem::is_directory(r.hostPath.parent_path(), ec) || combined.empty();
        return r;
    }
    r.entry = r.device->Resolve(combined);

    size_t slash = combined.find_last_of('\\');
    const Entry* parent = r.device->Resolve(slash == std::string::npos ? std::string_view{} : std::string_view(combined).substr(0, slash));
    r.parentFound = parent && parent->directory;
    return r;
}

namespace {
struct ReadStats { uint64_t reads = 0, bytes = 0; };
std::mutex s_statsMutex;
std::map<std::string, ReadStats> s_stats;
}

void NoteRead(const std::string& path, uint64_t bytes)
{
    std::lock_guard lock(s_statsMutex);
    auto& st = s_stats[path];
    st.reads++;
    st.bytes += bytes;
}

void DumpStats()
{
    std::lock_guard lock(s_statsMutex);
    log::Info("--- file reads ---");
    for (auto& [path, st] : s_stats)
        log::Info("  {:40} {:8} reads {:10.1f} MB", path, st.reads, st.bytes / 1048576.0);
}

namespace {
std::filesystem::path s_iso;   // the image main.cpp picked
}

std::vector<std::filesystem::path> FindIsos()
{
    if (std::filesystem::path env = platform::EnvironmentPath("AC6_ISO"); !env.empty())
        return { env };
    // The executable's folder first, then its parents (the development layout keeps the image at the
    // repository root, above build/).
    std::error_code ec;
    for (auto dir = platform::ExecutableDirectory(); !dir.empty(); dir = dir.parent_path())
    {
        std::vector<std::filesystem::path> found;
        for (auto& entry : std::filesystem::directory_iterator(dir, ec))
            if (entry.is_regular_file(ec) && ToLower(platform::PathUtf8(entry.path().extension())) == ".iso")
                found.push_back(entry.path());
        if (!found.empty())
        {
            std::sort(found.begin(), found.end());
            return found;
        }
        if (dir == dir.root_path())
            break;
    }
    return {};
}

std::filesystem::path FindIso()
{
    if (!s_iso.empty())
        return s_iso;
    const auto isos = FindIsos();
    return isos.empty() ? std::filesystem::path() : isos.front();
}

void SetIso(const std::filesystem::path& path) { s_iso = path; }

} // namespace ac6::fs
