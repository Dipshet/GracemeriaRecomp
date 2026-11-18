// Win32 implementation of platform/platform.h.
#include "platform/platform.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string>
#include <thread>
#include <utility>

#include <windows.h>
#include <timeapi.h>

namespace ac6::platform {

namespace {

// Recompiled code nests deeply on the host stack.
constexpr SIZE_T kHostStackReserve = 16 * 1024 * 1024;

struct ThreadStart
{
    void (*entry)(void*);
    void* arg;
};

DWORD WINAPI ThreadProc(void* p)
{
    const ThreadStart start = *static_cast<ThreadStart*>(p);
    delete static_cast<ThreadStart*>(p);
    start.entry(start.arg);
    return 0;
}

uint64_t FileTimeValue(const FILETIME& t) { return (uint64_t(t.dwHighDateTime) << 32) | t.dwLowDateTime; }

} // namespace

// --- Threads -------------------------------------------------------------------------------------------------------

uint32_t CurrentThreadId() { return uint32_t(GetCurrentThreadId()); }

void SetCurrentThreadPriority(int priority) { ::SetThreadPriority(GetCurrentThread(), priority); }

uint32_t ProcessorCount()
{
    static const uint32_t count = [] {
        DWORD_PTR process = 0, system = 0;
        if (GetProcessAffinityMask(GetCurrentProcess(), &process, &system) && process)
            return uint32_t(std::popcount(uint64_t(process)));
        return std::max(1u, std::thread::hardware_concurrency());
    }();
    return count;
}

HostThread* CreateHeldThread(void (*entry)(void*), void* arg, uint32_t& threadId)
{
    auto* start = new ThreadStart{ entry, arg };
    DWORD tid = 0;
    HANDLE h = CreateThread(nullptr, kHostStackReserve, ThreadProc, start, CREATE_SUSPENDED | STACK_SIZE_PARAM_IS_A_RESERVATION, &tid);
    if (!h)
    {
        delete start;
        return nullptr;
    }
    threadId = tid;
    return reinterpret_cast<HostThread*>(h);
}

void Launch(HostThread* thread) { ResumeThread(reinterpret_cast<HANDLE>(thread)); }

HostThread* AdoptCurrentThread()
{
    HANDLE h = nullptr;
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &h, 0, FALSE, DUPLICATE_SAME_ACCESS);
    return reinterpret_cast<HostThread*>(h);
}

void CloseThread(HostThread* thread) { CloseHandle(reinterpret_cast<HANDLE>(thread)); }

void SetThreadPriority(HostThread* thread, int priority) { ::SetThreadPriority(reinterpret_cast<HANDLE>(thread), priority); }

void SetThreadPriority(void* nativeHandle, int priority) { ::SetThreadPriority(static_cast<HANDLE>(nativeHandle), priority); }

void SetIdealProcessor(HostThread* thread, uint32_t processor)
{
    const DWORD hostCount = std::max<DWORD>(1, GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));
    SetThreadIdealProcessor(reinterpret_cast<HANDLE>(thread), processor % hostCount);
}

// --- Waits ---------------------------------------------------------------------------------------------------------

void WaitOnAddress(const void* address, uint32_t compare, uint32_t ms)
{
    ::WaitOnAddress(const_cast<void*>(address), &compare, sizeof(compare), ms);
}

void WakeOne(const void* address) { WakeByAddressSingle(const_cast<void*>(address)); }
void WakeAll(const void* address) { WakeByAddressAll(const_cast<void*>(address)); }

Event::Event() : handle(CreateEventW(nullptr, FALSE, FALSE, nullptr)) {}
Event::~Event() { CloseHandle(handle); }
void Event::Set() { SetEvent(handle); }
void Event::Wait() { WaitForSingleObject(handle, INFINITE); }

uint64_t SystemTime100ns()
{
    FILETIME ft;
    GetSystemTimePreciseAsFileTime(&ft);
    return FileTimeValue(ft);
}

void YieldThread() { SwitchToThread(); }

bool SleepPrecise(uint64_t hundredNs)
{
    // High-resolution waitable timers take the NT convention directly (negative = relative).
    thread_local HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    LARGE_INTEGER due;
    due.QuadPart = -int64_t(hundredNs);
    if (!timer || !SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE))
        return false;
    WaitForSingleObject(timer, INFINITE);
    return true;
}

void RaiseTimerResolution() { timeBeginPeriod(1); }

// --- Paths and the environment ------------------------------------------------------------------------------------

std::filesystem::path ExecutablePath()
{
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return std::filesystem::path(std::wstring(buf, n));
}

std::filesystem::path ExecutableDirectory() { return ExecutablePath().parent_path(); }

std::filesystem::path EnvironmentPath(const char* name)
{
    const std::wstring wide(name, name + std::strlen(name));
    const wchar_t* v = _wgetenv(wide.c_str());
    return v && *v ? std::filesystem::path(v) : std::filesystem::path();
}

void SetEnvironment(const char* name, const char* value) { _putenv_s(name, value); }
void UnsetEnvironment(const char* name) { _putenv_s(name, ""); }   // an empty value removes it

bool StartAgain()
{
    SetEnvironment("AC6_RESTARTED_FROM", std::to_string(GetCurrentProcessId()).c_str());
    // CreateProcessW may write to the command line it is given: a copy.
    std::wstring commandLine = GetCommandLineW();
    STARTUPINFOW startup{ sizeof(startup) };
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(nullptr, commandLine.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &startup, &process))
        return false;
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return true;
}

bool WaitForRestart()
{
    const char* from = std::getenv("AC6_RESTARTED_FROM");
    if (!from)
        return false;
    if (HANDLE previous = OpenProcess(SYNCHRONIZE, FALSE, static_cast<DWORD>(std::strtoul(from, nullptr, 10))))
    {
        WaitForSingleObject(previous, 15000);
        CloseHandle(previous);
    }
    UnsetEnvironment("AC6_RESTARTED_FROM");
    return true;
}

std::string PathUtf8(const std::filesystem::path& path)
{
    const std::u8string s = path.u8string();
    return std::string(s.begin(), s.end());
}

// --- Host files ----------------------------------------------------------------------------------------------------

PathInfo StatPath(const std::filesystem::path& path)
{
    PathInfo info;
    WIN32_FILE_ATTRIBUTE_DATA data{};
    if (!GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &data))
        return info;
    info.exists = true;
    info.directory = (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
    info.readOnly = (data.dwFileAttributes & FILE_ATTRIBUTE_READONLY) != 0;
    info.size = info.directory ? 0 : (uint64_t(data.nFileSizeHigh) << 32) | data.nFileSizeLow;
    info.creation = FileTimeValue(data.ftCreationTime);
    info.access = FileTimeValue(data.ftLastAccessTime);
    info.write = FileTimeValue(data.ftLastWriteTime);
    return info;
}

HostFile& HostFile::operator=(HostFile&& other) noexcept
{
    if (this != &other)
    {
        Close();
        handle = std::exchange(other.handle, INVALID_HANDLE_VALUE);
        error = other.error;
    }
    return *this;
}

bool HostFile::Open(const std::filesystem::path& path, Mode mode)
{
    Close();
    const DWORD access = mode == Mode::Read ? GENERIC_READ : GENERIC_READ | GENERIC_WRITE;
    const DWORD disposition = mode == Mode::Read ? OPEN_EXISTING : mode == Mode::ReadWrite ? OPEN_ALWAYS : CREATE_ALWAYS;
    handle = CreateFileW(path.c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, disposition,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE)
    {
        error = GetLastError();
        return false;
    }
    return true;
}

bool HostFile::IsOpen() const { return handle != INVALID_HANDLE_VALUE; }

void HostFile::Close()
{
    if (handle != INVALID_HANDLE_VALUE)
        CloseHandle(handle);
    handle = INVALID_HANDLE_VALUE;
}

bool HostFile::ReadAt(void* buffer, uint32_t size, uint64_t offset, uint32_t& done)
{
    OVERLAPPED ov{};
    ov.Offset = DWORD(offset);
    ov.OffsetHigh = DWORD(offset >> 32);
    DWORD got = 0;
    const bool ok = ReadFile(handle, buffer, size, &got, &ov) || GetLastError() == ERROR_HANDLE_EOF;
    if (!ok)
        error = GetLastError();
    done = got;
    return ok;
}

bool HostFile::WriteAt(const void* buffer, uint32_t size, uint64_t offset, uint32_t& done)
{
    OVERLAPPED ov{};
    ov.Offset = DWORD(offset);
    ov.OffsetHigh = DWORD(offset >> 32);
    DWORD wrote = 0;
    const bool ok = WriteFile(handle, buffer, size, &wrote, &ov);
    if (!ok)
        error = GetLastError();
    done = wrote;
    return ok;
}

bool HostFile::Size(uint64_t& size)
{
    LARGE_INTEGER s{};
    if (!GetFileSizeEx(handle, &s))
    {
        error = GetLastError();
        return false;
    }
    size = uint64_t(s.QuadPart);
    return true;
}

bool HostFile::Resize(uint64_t size)
{
    FILE_END_OF_FILE_INFO eof{};
    eof.EndOfFile.QuadPart = int64_t(size);
    if (!SetFileInformationByHandle(handle, FileEndOfFileInfo, &eof, sizeof(eof)))
    {
        error = GetLastError();
        return false;
    }
    return true;
}

bool HostFile::Flush() { return FlushFileBuffers(handle) != 0; }

PathInfo HostFile::Info()
{
    PathInfo s;
    BY_HANDLE_FILE_INFORMATION info{};
    if (!GetFileInformationByHandle(handle, &info))
        return s;
    s.exists = true;
    s.directory = (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
    s.size = (uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
    s.creation = FileTimeValue(info.ftCreationTime);
    s.access = FileTimeValue(info.ftLastAccessTime);
    s.write = FileTimeValue(info.ftLastWriteTime);
    s.index = (uint64_t(info.nFileIndexHigh) << 32) | info.nFileIndexLow;
    return s;
}

uint32_t LastError() { return GetLastError(); }

std::string ErrorText(uint32_t code)
{
    char* text = nullptr;
    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code,
        MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US), reinterpret_cast<char*>(&text), 0, nullptr);
    std::string s = text ? text : "";
    LocalFree(text);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == '.'))
        s.pop_back();
    return std::format("Windows error {}{}{}", code, s.empty() ? "" : ": ", s);
}

bool ReplaceFileDurably(const std::filesystem::path& from, const std::filesystem::path& to)
{
    return MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
}

// --- User interface ------------------------------------------------------------------------------------------------

void ShowErrorBox(const char* title, const char* message)
{
    MessageBoxA(nullptr, message, title, MB_OK | MB_ICONERROR);
}

} // namespace ac6::platform
