// Linux implementation of platform/platform.h.
#include "platform/platform.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <ucontext.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace ac6::platform {

namespace {

// Recompiled code nests deeply on the host stack.
constexpr size_t kHostStackSize = 16 * 1024 * 1024;

long Futex(const void* address, int op, uint32_t value, const timespec* timeout)
{
    return syscall(SYS_futex, address, op, value, timeout, nullptr, 0);
}

// Win32 priorities as nice values. Raising a thread above 0 needs privileges a game doesn't have: those fail quietly.
void SetNice(pid_t tid, int priority)
{
    const int nice = priority <= kPriorityLowest ? 10 : priority == kPriorityBelowNormal ? 5 : priority == kPriorityNormal ? 0
        : priority == kPriorityAboveNormal ? -5 : -10;
    setpriority(PRIO_PROCESS, id_t(tid), nice);
}

uint64_t FileTime(const timespec& t)
{
    // FILETIME: 100 ns since 1601-01-01.
    return (uint64_t(t.tv_sec) + 11644473600ull) * 10000000ull + uint64_t(t.tv_nsec) / 100;
}

PathInfo FromStat(const struct stat& st)
{
    PathInfo info;
    info.exists = true;
    info.directory = S_ISDIR(st.st_mode);
    info.readOnly = !(st.st_mode & S_IWUSR);
    info.size = info.directory ? 0 : uint64_t(st.st_size);
    info.creation = FileTime(st.st_mtim);   // (no creation time in stat)
    info.access = FileTime(st.st_atim);
    info.write = FileTime(st.st_mtim);
    info.index = uint64_t(st.st_ino);
    return info;
}

} // namespace

// --- Threads -------------------------------------------------------------------------------------------------------

struct HostThread
{
    pid_t tid = 0;
    void (*entry)(void*) = nullptr;
    void* arg = nullptr;
    std::atomic<uint32_t> ready{ 0 };   // the new thread has published its tid
    std::atomic<uint32_t> gate{ 0 };    // 0 held, 1 launched, 2 closed before launch
    std::atomic<int> refs{ 1 };         // the creator's handle (+1 for a thread not yet past the gate)

    void Release()
    {
        if (refs.fetch_sub(1, std::memory_order_acq_rel) == 1)
            delete this;
    }
};

namespace {

void* ThreadMain(void* p)
{
    auto* t = static_cast<HostThread*>(p);
    t->tid = pid_t(syscall(SYS_gettid));
    t->ready.store(1, std::memory_order_release);
    Futex(&t->ready, FUTEX_WAKE_PRIVATE, INT_MAX, nullptr);
    uint32_t gate;
    while ((gate = t->gate.load(std::memory_order_acquire)) == 0)
        Futex(&t->gate, FUTEX_WAIT_PRIVATE, 0, nullptr);
    void (*entry)(void*) = t->entry;
    void* arg = t->arg;
    t->Release();
    if (gate == 1)
        entry(arg);
    return nullptr;
}

} // namespace

uint32_t CurrentThreadId() { return uint32_t(syscall(SYS_gettid)); }

void SetCurrentThreadPriority(int priority) { SetNice(pid_t(syscall(SYS_gettid)), priority); }

uint32_t ProcessorCount()
{
    static const uint32_t count = [] {
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) == 0 && CPU_COUNT(&set) > 0)
            return uint32_t(CPU_COUNT(&set));
        return std::max(1u, std::thread::hardware_concurrency());
    }();
    return count;
}

HostThread* CreateHeldThread(void (*entry)(void*), void* arg, uint32_t& threadId)
{
    auto* t = new HostThread;
    t->entry = entry;
    t->arg = arg;
    t->refs = 2;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, kHostStackSize);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_t thread;
    const int rc = pthread_create(&thread, &attr, ThreadMain, t);
    pthread_attr_destroy(&attr);
    if (rc != 0)
    {
        delete t;
        return nullptr;
    }
    while (!t->ready.load(std::memory_order_acquire))
        Futex(&t->ready, FUTEX_WAIT_PRIVATE, 0, nullptr);
    threadId = uint32_t(t->tid);
    return t;
}

void Launch(HostThread* thread)
{
    uint32_t held = 0;
    if (thread->gate.compare_exchange_strong(held, 1, std::memory_order_acq_rel))
        Futex(&thread->gate, FUTEX_WAKE_PRIVATE, INT_MAX, nullptr);
}

HostThread* AdoptCurrentThread()
{
    auto* t = new HostThread;
    t->tid = pid_t(syscall(SYS_gettid));
    t->gate = 1;
    return t;
}

void CloseThread(HostThread* thread)
{
    uint32_t held = 0;
    if (thread->gate.compare_exchange_strong(held, 2, std::memory_order_acq_rel))
        Futex(&thread->gate, FUTEX_WAKE_PRIVATE, INT_MAX, nullptr);
    thread->Release();
}

void SetThreadPriority(HostThread* thread, int priority) { SetNice(thread->tid, priority); }

// A pthread_t has no tid to renice: the threads that want a lower priority set their own.
void SetThreadPriority(void*, int) {}

void SetIdealProcessor(HostThread*, uint32_t) {}

// --- Waits ---------------------------------------------------------------------------------------------------------

void WaitOnAddress(const void* address, uint32_t compare, uint32_t ms)
{
    if (ms == kInfinite)
    {
        Futex(address, FUTEX_WAIT_PRIVATE, compare, nullptr);
        return;
    }
    const timespec timeout{ time_t(ms / 1000), long(ms % 1000) * 1000000 };
    Futex(address, FUTEX_WAIT_PRIVATE, compare, &timeout);
}

void WakeOne(const void* address) { Futex(address, FUTEX_WAKE_PRIVATE, 1, nullptr); }
void WakeAll(const void* address) { Futex(address, FUTEX_WAKE_PRIVATE, INT_MAX, nullptr); }

Event::Event() = default;
Event::~Event() = default;

void Event::Set()
{
    {
        std::lock_guard lock(mutex);
        signaled = true;
    }
    cv.notify_one();
}

void Event::Wait()
{
    std::unique_lock lock(mutex);
    cv.wait(lock, [this] { return signaled; });
    signaled = false;
}

uint64_t SystemTime100ns()
{
    timespec t{};
    clock_gettime(CLOCK_REALTIME, &t);
    return FileTime(t);
}

void YieldThread() { sched_yield(); }

bool SleepPrecise(uint64_t hundredNs)
{
    timespec t{ time_t(hundredNs / 10000000), long(hundredNs % 10000000) * 100 };
    while (clock_nanosleep(CLOCK_MONOTONIC, 0, &t, &t) == EINTR)
        ;
    return true;
}

void RaiseTimerResolution() {}

// --- Paths and the environment ------------------------------------------------------------------------------------

std::filesystem::path ExecutablePath()
{
    std::error_code ec;
    return std::filesystem::read_symlink("/proc/self/exe", ec);
}

std::filesystem::path ExecutableDirectory() { return ExecutablePath().parent_path(); }

std::filesystem::path EnvironmentPath(const char* name)
{
    const char* v = std::getenv(name);
    return v && *v ? std::filesystem::path(v) : std::filesystem::path();
}

void SetEnvironment(const char* name, const char* value) { setenv(name, value, 1); }
void UnsetEnvironment(const char* name) { unsetenv(name); }

bool StartAgain()
{
    // The command line from /proc (NUL-separated), for the same executable.
    std::ifstream in("/proc/self/cmdline", std::ios::binary);
    std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<std::string> args;
    for (size_t at = 0; at < all.size();)
    {
        const size_t end = all.find('\0', at);
        args.push_back(all.substr(at, end == std::string::npos ? std::string::npos : end - at));
        at = end == std::string::npos ? all.size() : end + 1;
    }
    if (args.empty())
        return false;
    const std::string exe = ExecutablePath().string();
    SetEnvironment("AC6_RESTARTED_FROM", std::to_string(getpid()).c_str());
    const pid_t pid = fork();
    if (pid < 0)
        return false;
    if (pid == 0)
    {
        std::vector<char*> argv;
        for (std::string& a : args)
            argv.push_back(a.data());
        argv.push_back(nullptr);
        execv(exe.c_str(), argv.data());
        _exit(127);
    }
    return true;
}

bool WaitForRestart()
{
    const char* from = std::getenv("AC6_RESTARTED_FROM");
    if (!from)
        return false;
    // The copy is the previous one's child: its parent changes when that one has ended.
    const pid_t previous = static_cast<pid_t>(std::strtol(from, nullptr, 10));
    for (int i = 0; i < 1500 && getppid() == previous; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    UnsetEnvironment("AC6_RESTARTED_FROM");
    return true;
}

std::string PathUtf8(const std::filesystem::path& path) { return path.string(); }

// --- Host files ----------------------------------------------------------------------------------------------------

PathInfo StatPath(const std::filesystem::path& path)
{
    struct stat st{};
    if (stat(path.c_str(), &st) != 0)
        return {};
    PathInfo info = FromStat(st);
    info.index = 0;
    return info;
}

HostFile& HostFile::operator=(HostFile&& other) noexcept
{
    if (this != &other)
    {
        Close();
        fd = std::exchange(other.fd, -1);
        error = other.error;
    }
    return *this;
}

bool HostFile::Open(const std::filesystem::path& path, Mode mode)
{
    Close();
    const int flags = mode == Mode::Read ? O_RDONLY : mode == Mode::ReadWrite ? O_RDWR | O_CREAT : O_RDWR | O_CREAT | O_TRUNC;
    fd = open(path.c_str(), flags | O_CLOEXEC, 0644);
    if (fd < 0)
    {
        error = uint32_t(errno);
        return false;
    }
    return true;
}

bool HostFile::IsOpen() const { return fd >= 0; }

void HostFile::Close()
{
    if (fd >= 0)
        close(fd);
    fd = -1;
}

bool HostFile::ReadAt(void* buffer, uint32_t size, uint64_t offset, uint32_t& done)
{
    done = 0;
    while (done < size)
    {
        const ssize_t n = pread(fd, static_cast<uint8_t*>(buffer) + done, size - done, off_t(offset + done));
        if (n < 0 && errno == EINTR)
            continue;
        if (n < 0)
        {
            error = uint32_t(errno);
            return false;
        }
        if (n == 0)
            break;
        done += uint32_t(n);
    }
    return true;
}

bool HostFile::WriteAt(const void* buffer, uint32_t size, uint64_t offset, uint32_t& done)
{
    done = 0;
    while (done < size)
    {
        const ssize_t n = pwrite(fd, static_cast<const uint8_t*>(buffer) + done, size - done, off_t(offset + done));
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
        {
            error = n < 0 ? uint32_t(errno) : uint32_t(ENOSPC);
            return false;
        }
        done += uint32_t(n);
    }
    return true;
}

bool HostFile::Size(uint64_t& size)
{
    struct stat st{};
    if (fstat(fd, &st) != 0)
    {
        error = uint32_t(errno);
        return false;
    }
    size = uint64_t(st.st_size);
    return true;
}

bool HostFile::Resize(uint64_t size)
{
    if (ftruncate(fd, off_t(size)) != 0)
    {
        error = uint32_t(errno);
        return false;
    }
    return true;
}

bool HostFile::Flush() { return fsync(fd) == 0; }

PathInfo HostFile::Info()
{
    struct stat st{};
    return fstat(fd, &st) == 0 ? FromStat(st) : PathInfo{};
}

uint32_t LastError() { return uint32_t(errno); }

std::string ErrorText(uint32_t code) { return std::format("error {}: {}", code, std::strerror(int(code))); }

bool ReplaceFileDurably(const std::filesystem::path& from, const std::filesystem::path& to)
{
    // Written through: the new file's data before the rename, the directory entry after it.
    if (const int f = open(from.c_str(), O_RDONLY | O_CLOEXEC); f >= 0)
    {
        fsync(f);
        close(f);
    }
    if (rename(from.c_str(), to.c_str()) != 0)
        return false;
    if (const int d = open(to.parent_path().empty() ? "." : to.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC); d >= 0)
    {
        fsync(d);
        close(d);
    }
    return true;
}

// --- User interface ------------------------------------------------------------------------------------------------

void ShowErrorBox(const char* title, const char* message)
{
    std::fprintf(stderr, "%s: %s\n", title, message);
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, title, message, nullptr);
}

// --- Faults --------------------------------------------------------------------------------------------------------

namespace {

struct RegisteredHandler
{
    int priority;
    FaultHandler handler;
};
constexpr size_t kMaxFaultHandlers = 8;
RegisteredHandler s_faultHandlers[kMaxFaultHandlers];
std::atomic<size_t> s_faultHandlerCount{ 0 };
void (*s_crashHandler)(const Fault&) = nullptr;
std::mutex s_faultInstallMutex;

void OnFault(int signal, siginfo_t* info, void* context)
{
    Fault fault;
    fault.signal = signal;
    fault.address = info->si_addr;
    fault.context = context;
    // x86-64 page fault error code: bit 1 set for a write.
    fault.write = (static_cast<ucontext_t*>(context)->uc_mcontext.gregs[REG_ERR] & 2) != 0;
    const size_t count = s_faultHandlerCount.load(std::memory_order_acquire);
    for (size_t i = 0; i < count; i++)
        if (s_faultHandlers[i].handler(fault))
            return;
    if (s_crashHandler)
        s_crashHandler(fault);
    // No crash handler (or it returned): the default action on the retried instruction.
    ::signal(signal, SIG_DFL);
}

void InstallFaultSignals()
{
    static bool installed = false;
    if (installed)
        return;
    installed = true;
    struct sigaction action{};
    action.sa_sigaction = OnFault;
    action.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigemptyset(&action.sa_mask);
    sigaction(SIGSEGV, &action, nullptr);
    sigaction(SIGBUS, &action, nullptr);
}

} // namespace

void AddFaultHandler(int priority, FaultHandler handler)
{
    std::lock_guard lock(s_faultInstallMutex);
    size_t count = s_faultHandlerCount.load(std::memory_order_relaxed);
    if (count == kMaxFaultHandlers)
        return;
    // Inserted in order while nothing reads past the published count, then published.
    size_t at = count;
    while (at > 0 && s_faultHandlers[at - 1].priority > priority)
    {
        s_faultHandlers[at] = s_faultHandlers[at - 1];
        at--;
    }
    s_faultHandlers[at] = { priority, handler };
    s_faultHandlerCount.store(count + 1, std::memory_order_release);
    InstallFaultSignals();
}

void SetCrashHandler(void (*handler)(const Fault& fault))
{
    std::lock_guard lock(s_faultInstallMutex);
    s_crashHandler = handler;
    InstallFaultSignals();
}

} // namespace ac6::platform
