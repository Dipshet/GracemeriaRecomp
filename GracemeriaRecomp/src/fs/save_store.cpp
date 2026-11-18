#include "fs/save_store.h"

#include "fs/vfs.h"
#include "log.h"
#include "platform/platform.h"
#include "settings.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <fstream>
#include <mutex>

namespace ac6::fs {

namespace {

constexpr int kBackups = 3;
std::mutex s_mutex;   // serialises commits (rename + backup rotation) across threads

std::string Utf8(const std::filesystem::path& p) { return reinterpret_cast<const char*>(p.u8string().c_str()); }

std::filesystem::path Sibling(const std::filesystem::path& content, const char* suffix)
{
    return content.parent_path() / (content.filename().native() + std::filesystem::path(suffix).native());
}

} // namespace

std::filesystem::path SaveRoot()
{
    static const std::filesystem::path root = [] {
        if (std::filesystem::path env = platform::EnvironmentPath("AC6_SAVE_DIR"); !env.empty())
            return env;
        return platform::ExecutableDirectory() / "saves";
    }();
    return root;
}

std::filesystem::path ContentPath(uint64_t xuid, uint32_t contentType, const std::string& fileName)
{
    return SaveRoot() / std::format("{:016X}", xuid) / std::format("{:08X}", contentType) /
        std::filesystem::path(std::u8string(fileName.begin(), fileName.end()));
}

std::vector<ContentEntry> ListContent(uint64_t xuid, uint32_t contentType)
{
    std::vector<ContentEntry> out;
    std::error_code ec;
    const auto dir = SaveRoot() / std::format("{:016X}", xuid) / std::format("{:08X}", contentType);
    for (auto& e : std::filesystem::directory_iterator(dir, ec))
    {
        if (!e.is_directory(ec))
            continue;
        std::string name = Utf8(e.path().filename());
        if (name.ends_with(".backup") || name.ends_with(".staging"))
            continue;
        out.push_back({ name, ReadContentMeta(e.path()) });
    }
    return out;
}

bool WriteContentMeta(const std::filesystem::path& content, const std::string& displayName)
{
    std::ofstream out(Sibling(content, ".meta"), std::ios::binary | std::ios::trunc);
    out << displayName;
    out.close();
    return bool(out);
}

bool FaultInjected(const char* kind)
{
    static const std::string faults = [] {
        const char* v = std::getenv("AC6_SAVE_FAULT");
        return v && *v && !settings::SetByFile("AC6_SAVE_FAULT") ? "," + std::string(v) + "," : std::string();
    }();
    return !faults.empty() && faults.find("," + std::string(kind) + ",") != std::string::npos;
}

std::string ReadContentMeta(const std::filesystem::path& content)
{
    std::ifstream f(Sibling(content, ".meta"), std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), {});
}

void WriteThumbnail(const std::filesystem::path& content, const void* data, size_t size)
{
    std::ofstream(Sibling(content, ".thumb"), std::ios::binary | std::ios::trunc).write(static_cast<const char*>(data), size);
}

bool TrashContent(const std::filesystem::path& content)
{
    std::lock_guard lock(s_mutex);
    std::error_code ec;
    if (!std::filesystem::exists(content, ec))
        return false;
    const auto stamp = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const auto trash = SaveRoot() / ".trash" / std::format("{}-{}", stamp, Utf8(content.filename()));
    std::filesystem::create_directories(trash, ec);
    // The container first: it is the save. If it can't move (a file in it open: a scan, Explorer), nothing has.
    std::filesystem::rename(content, trash / "content", ec);
    if (ec)
    {
        log::Error("Save container {} couldn't be moved to the trash ({}): not deleted", Utf8(content), ec.message());
        std::filesystem::remove(trash, ec);
        return false;
    }
    for (const char* suffix : { ".meta", ".thumb", ".backup" })
    {
        const auto from = Sibling(content, suffix);
        if (std::filesystem::exists(from, ec))
        {
            std::filesystem::rename(from, trash / (std::string("content") + suffix), ec);
            if (ec)
                log::Warn("Save container {}: its {} stayed ({})", Utf8(content), suffix, ec.message());
        }
    }
    log::Info("Save container {} moved to {}", Utf8(content), Utf8(trash));
    return true;
}

std::filesystem::path ContainerOf(const std::filesystem::path& hostPath)
{
    // <root>/<XUID>/<type>/<container>/...: the container is three levels below the save root.
    std::error_code ec;
    const auto relative = std::filesystem::relative(hostPath, SaveRoot(), ec);
    auto it = relative.begin();
    std::filesystem::path container = SaveRoot();
    for (int i = 0; i < 3 && it != relative.end(); i++, ++it)
        container /= *it;
    return container;
}

std::filesystem::path StagingPathFor(const std::filesystem::path& content, const std::filesystem::path& target)
{
    std::error_code ec;
    const auto relative = std::filesystem::relative(target, content, ec);
    auto staged = Sibling(content, ".staging") / relative;
    staged += ".tmp";
    std::filesystem::create_directories(staged.parent_path(), ec);
    return staged;
}

namespace {

void RotateBackups(const std::filesystem::path& content, const std::filesystem::path& target)
{
    std::error_code ec;
    if (!std::filesystem::exists(target, ec))
        return;
    const auto relative = std::filesystem::relative(target, content, ec);
    const auto base = Sibling(content, ".backup") / relative;
    std::filesystem::create_directories(base.parent_path(), ec);
    auto numbered = [&](int n) { auto p = base; p += std::format(".{}", n); return p; };
    std::filesystem::remove(numbered(kBackups), ec);
    for (int n = kBackups - 1; n >= 1; n--)
        if (std::filesystem::exists(numbered(n), ec))
            std::filesystem::rename(numbered(n), numbered(n + 1), ec);
    std::filesystem::copy_file(target, numbered(1), std::filesystem::copy_options::overwrite_existing, ec);
    if (ec)
        log::Warn("Save backup of {} failed: {}", Utf8(target), ec.message());
}

} // namespace

bool CommitStaged(const std::filesystem::path& content, const std::filesystem::path& staged, const std::filesystem::path& target)
{
    std::lock_guard lock(s_mutex);
    std::error_code ec;
    std::filesystem::create_directories(target.parent_path(), ec);
    // The current version copied aside first. It becomes backup 1 only once the new one is in place, so a failed commit
    // leaves the backups as they were.
    const auto relative = std::filesystem::relative(target, content, ec);
    const auto base = Sibling(content, ".backup") / relative;
    auto numbered = [&](int n) { auto p = base; p += std::format(".{}", n); return p; };
    auto pending = base;
    pending += ".pending";
    bool kept = false;
    if (std::filesystem::exists(target, ec))
    {
        std::filesystem::create_directories(base.parent_path(), ec);
        std::filesystem::copy_file(target, pending, std::filesystem::copy_options::overwrite_existing, ec);
        kept = !ec;
        if (ec)
            log::Warn("Save backup of {} failed: {}", Utf8(target), ec.message());
    }
    // The replace, retried for a few seconds: an antivirus, a sync or backup tool or Explorer's preview can hold the
    // file for a moment.
    static std::atomic<int> busy{ 0 };
    bool replaced = false;
    for (int attempt = 0; attempt < 12 && !replaced; attempt++)
    {
        if (attempt)
            std::this_thread::sleep_for(std::chrono::milliseconds(attempt < 4 ? 50 : 250));
        if (FaultInjected("replace") || (FaultInjected("replace-busy") && busy.fetch_add(1) < 3))
            continue;
        replaced = platform::ReplaceFileDurably(staged, target);
    }
    if (!replaced)
    {
        std::filesystem::remove(pending, ec);
        log::Error("Save commit of {} failed (error {}); the previous version is untouched, the new data stays in {}",
            Utf8(target), platform::LastError(), Utf8(staged));
        return false;
    }
    if (kept)
    {
        std::filesystem::remove(numbered(kBackups), ec);
        for (int n = kBackups - 1; n >= 1; n--)
            if (std::filesystem::exists(numbered(n), ec))
                std::filesystem::rename(numbered(n), numbered(n + 1), ec);
        std::filesystem::rename(pending, numbered(1), ec);
    }
    log::Info("Saved {} ({} bytes)", Utf8(std::filesystem::relative(target, SaveRoot(), ec)), std::filesystem::file_size(target, ec));
    return true;
}

void CleanStaging(const std::filesystem::path& content)
{
    std::lock_guard lock(s_mutex);
    const auto staging = Sibling(content, ".staging");
    std::error_code ec;
    if (!std::filesystem::is_directory(staging, ec))
        return;
    for (auto it = std::filesystem::recursive_directory_iterator(staging, ec); !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec))
        if (it->is_regular_file(ec))
        {
            log::Warn("Save container {} has an uncommitted staged file {}", Utf8(content), Utf8(it->path()));
            return;
        }
    std::filesystem::remove_all(staging, ec);
}

bool DeleteWithBackup(const std::filesystem::path& content, const std::filesystem::path& target)
{
    std::lock_guard lock(s_mutex);
    RotateBackups(content, target);
    std::error_code ec;
    const bool removed = std::filesystem::remove(target, ec);
    log::Info("Deleted {} (backed up)", Utf8(target));
    return removed;
}

} // namespace ac6::fs
