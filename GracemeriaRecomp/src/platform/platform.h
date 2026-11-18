#pragma once
// Host operating system services beyond the C++ library: threads, address waits, events,
// sleeps, host files and paths, message boxes. platform_win32.cpp implements them with Win32, platform_linux.cpp
// with POSIX / Linux. Guest memory (memory.cpp), the crash handler (crash_*.cpp) and the profiler keep their own
// per-platform code: they are too tied to their OS's fault and context model to share an interface.

#include <cstdint>
#include <filesystem>
#include <string>

#if !defined(_WIN32)
#include <condition_variable>
#include <mutex>
#endif

namespace ac6::platform {

// --- Threads -------------------------------------------------------------------------------------------------------
// Thread priorities, the Win32 scale: -2 lowest, -1 below normal, 0 normal, 1 above normal, 2 highest. Linux has no
// unprivileged way to raise a thread: there only the lower ones apply (as nice values).
constexpr int kPriorityLowest = -2, kPriorityBelowNormal = -1, kPriorityNormal = 0, kPriorityAboveNormal = 1,
    kPriorityHighest = 2;

uint32_t CurrentThreadId();                         // the OS's thread id (the one the log and guest thread ids use)
void SetCurrentThreadPriority(int priority);
uint32_t ProcessorCount();                          // processors this process may run on (its affinity mask)

// A host thread for guest code (cpu/guest_thread.cpp): a 16 MB stack, created held and started by Launch().
struct HostThread;
HostThread* CreateHeldThread(void (*entry)(void*), void* arg, uint32_t& threadId);
void Launch(HostThread* thread);
HostThread* AdoptCurrentThread();                   // a handle on the calling thread
void CloseThread(HostThread* thread);
void SetThreadPriority(HostThread* thread, int priority);
void SetIdealProcessor(HostThread* thread, uint32_t processor);   // a hint (Windows), nothing on Linux
// Native handle for std::thread::native_handle() users (nothing on Linux: those threads lower their own priority).
void SetThreadPriority(void* nativeHandle, int priority);

// --- Waits ---------------------------------------------------------------------------------------------------------
// Sleeps while the 32-bit word at `address` equals `compare`, up to `ms` milliseconds (kInfinite: no limit). Spurious
// returns are allowed: callers check their condition again.
constexpr uint32_t kInfinite = 0xFFFFFFFF;
void WaitOnAddress(const void* address, uint32_t compare, uint32_t ms);
void WakeOne(const void* address);
void WakeAll(const void* address);

// An auto-reset event: Set() before Wait() is never lost.
class Event
{
public:
    Event();
    ~Event();
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    void Set();
    void Wait();

private:
#if defined(_WIN32)
    void* handle = nullptr;
#else
    std::mutex mutex;
    std::condition_variable cv;
    bool signaled = false;
#endif
};

// --- Time ----------------------------------------------------------------------------------------------------------
// The wall clock as a FILETIME (100 ns since 1601-01-01 UTC).
uint64_t SystemTime100ns();

void YieldThread();
// A precise relative sleep in 100 ns units (the NT convention's magnitude). False if unavailable (then the caller sleeps
// another way).
bool SleepPrecise(uint64_t hundredNs);
// Finer timer and sleep granularity for the process (Windows: 1 ms, nothing to do on Linux).
void RaiseTimerResolution();

// --- Paths and the environment ------------------------------------------------------------------------------------
std::filesystem::path ExecutablePath();
std::filesystem::path ExecutableDirectory();
// An environment variable as a path (Windows: read as UTF-16, so any path works), empty if unset.
std::filesystem::path EnvironmentPath(const char* name);
void SetEnvironment(const char* name, const char* value);
void UnsetEnvironment(const char* name);
// Starts another copy of this program with its command line and the current environment (the settings overlay's
// restart). False when it couldn't. The copy is told this process's ID (AC6_RESTARTED_FROM). WaitForRestart, called
// at the copy's start, waits up to 15 seconds for this process to end, so only one game at a time uses the saves, the
// log and the window.
bool StartAgain();
bool WaitForRestart();   // true when this copy was started again (StartAgain)
std::string PathUtf8(const std::filesystem::path& path);

// --- Host files ----------------------------------------------------------------------------------------------------
// Times are FILETIME units (100 ns since 1601), as the guest kernel reports them.
struct PathInfo
{
    bool exists = false, directory = false, readOnly = false;
    uint64_t size = 0;
    uint64_t creation = 0, access = 0, write = 0;
    uint64_t index = 0;   // a file identity (open files only)
};
PathInfo StatPath(const std::filesystem::path& path);

class HostFile
{
public:
    enum class Mode
    {
        Read,               // existing file, read only (others may read, write and delete it meanwhile)
        ReadWrite,          // existing file or a new one, read and write
        ReadWriteTruncate,  // a new or emptied file, read and write
    };
    HostFile() = default;
    ~HostFile() { Close(); }
    HostFile(const HostFile&) = delete;
    HostFile& operator=(const HostFile&) = delete;
    HostFile(HostFile&& other) noexcept { *this = std::move(other); }
    HostFile& operator=(HostFile&& other) noexcept;

    bool Open(const std::filesystem::path& path, Mode mode);
    bool IsOpen() const;
    void Close();
    // Positional I/O. Reading past the end is not an error (`done` < `size`).
    bool ReadAt(void* buffer, uint32_t size, uint64_t offset, uint32_t& done);
    bool WriteAt(const void* buffer, uint32_t size, uint64_t offset, uint32_t& done);
    bool Size(uint64_t& size);
    bool Resize(uint64_t size);
    bool Flush();
    PathInfo Info();
    // The last failure's OS error code (ErrorText gives its text).
    uint32_t Error() const { return error; }

private:
#if defined(_WIN32)
    void* handle = reinterpret_cast<void*>(intptr_t(-1));
#else
    int fd = -1;
#endif
    uint32_t error = 0;
};

uint32_t LastError();                        // the calling thread's last OS error code
std::string ErrorText(uint32_t code);        // "Windows error <code>: <message>" (Linux: "error <code>: <message>")
// Replaces `to` with `from` atomically, durably (written through to the disk).
bool ReplaceFileDurably(const std::filesystem::path& from, const std::filesystem::path& to);

// --- User interface ------------------------------------------------------------------------------------------------
void ShowErrorBox(const char* title, const char* message);

#if !defined(_WIN32)
// --- Faults (Linux, Windows uses vectored exception handlers) -------------------------------------------------------
// SIGSEGV and SIGBUS go to the fault handlers in priority order (lowest first). The first returning true resumes the
// faulting instruction. A fault none handles goes to the crash handler.
struct Fault
{
    int signal = 0;
    void* address = nullptr;   // the data address
    bool write = false;
    void* context = nullptr;   // the ucontext_t
};
using FaultHandler = bool (*)(const Fault& fault);
void AddFaultHandler(int priority, FaultHandler handler);
void SetCrashHandler(void (*handler)(const Fault& fault));
#endif

} // namespace ac6::platform
