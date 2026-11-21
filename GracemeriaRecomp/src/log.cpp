#include "log.h"

#include <chrono>
#include <cstdio>
#include <mutex>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

#include "platform/platform.h"

namespace ac6::log {

static std::mutex s_mutex;
static FILE* s_file = nullptr;
static std::filesystem::path s_directory = ".";
static const auto s_start = std::chrono::steady_clock::now();

void Init(const std::filesystem::path& path)
{
    s_directory = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
    std::lock_guard lock(s_mutex);
#if defined(_WIN32)
    s_file = _wfopen(path.c_str(), L"w");   // (any folder name: the path's wide form)
#else
    s_file = fopen(path.c_str(), "w");
#endif
}

const std::filesystem::path& Directory() { return s_directory; }

double Seconds()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - s_start).count();
}

uint32_t ThreadId()
{
    return platform::CurrentThreadId();
}

void Write(Level level, std::string_view message)
{
    static constexpr const char* kTags[] = { "trace", "info", "warn", "ERROR" };
    std::string line = std::format("[{:10.6f} {:5}] {:5} {}\n", Seconds(), ThreadId(), kTags[int(level)], message);

    std::lock_guard lock(s_mutex);
    if (level != Level::Trace)
        fwrite(line.data(), 1, line.size(), stdout);
    if (s_file)
    {
        fwrite(line.data(), 1, line.size(), s_file);
        // Info and above reach the disk at once: a hard process exit must not lose the last lines.
        if (level >= Level::Info)
            fflush(s_file);
    }
}

void FlushToDisk()
{
    std::lock_guard lock(s_mutex);
    if (!s_file)
        return;
    fflush(s_file);
#if defined(_WIN32)
    _commit(_fileno(s_file));
#else
    fsync(fileno(s_file));
#endif
}

} // namespace ac6::log
