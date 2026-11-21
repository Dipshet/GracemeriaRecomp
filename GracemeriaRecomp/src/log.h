#pragma once
#include <filesystem>
// Minimal thread-safe logger: console + ac6_runtime.log next to the executable.

#include <format>
#include <string>
#include <string_view>

namespace ac6::log {

enum class Level { Trace, Info, Warn, Error };

void Init(const std::filesystem::path& path);
const std::filesystem::path& Directory();   // folder of the log file (for diagnostic dumps)
void Write(Level level, std::string_view message);
// The log file's lines so far onto the disk itself, not only the system's cache: for what must survive the machine
// stopping (a GPU hang the driver can't recover from freezes it).
void FlushToDisk();
double Seconds(); // since start
uint32_t ThreadId();

template<typename... Args>
void Info(std::format_string<Args...> fmt, Args&&... args) { Write(Level::Info, std::format(fmt, std::forward<Args>(args)...)); }
template<typename... Args>
void Warn(std::format_string<Args...> fmt, Args&&... args) { Write(Level::Warn, std::format(fmt, std::forward<Args>(args)...)); }
template<typename... Args>
void Error(std::format_string<Args...> fmt, Args&&... args) { Write(Level::Error, std::format(fmt, std::forward<Args>(args)...)); }
template<typename... Args>
void Trace(std::format_string<Args...> fmt, Args&&... args) { Write(Level::Trace, std::format(fmt, std::forward<Args>(args)...)); }

} // namespace ac6::log
