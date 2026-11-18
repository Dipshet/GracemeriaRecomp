// The Windows half of the crash handler (crash.h): structured exceptions, unwinding by the image's
// unwind tables, the write watches.
#include "crash.h"
#include "crash_internal.h"

#include "cpu/guest_thread.h"
#include "log.h"
#include "memory.h"
#include "ppc.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <format>
#include <mutex>
#include <string>
#include <thread>

#include <windows.h>
#include <tlhelp32.h>

namespace ac6::crash {

using detail::DumpGuestContext;

// The guest function whose host code contains `host`: the unwind tables give the exact start of the
// containing host function, which works for recompiled functions and runtime hooks alike.
uint32_t GuestFunctionAt(uintptr_t host)
{
    DWORD64 imageBase = 0;
    PRUNTIME_FUNCTION entry = RtlLookupFunctionEntry(host, &imageBase, nullptr);
    if (!entry)
        return 0;
    return detail::GuestAtFunctionStart(imageBase + entry->BeginAddress);
}

namespace {

std::string DescribeHostStack(CONTEXT c);

LONG WINAPI CrashHandler(EXCEPTION_POINTERS* info)
{
    auto code = info->ExceptionRecord->ExceptionCode;
    // Ignore benign/debug exceptions (thread naming, C++ exceptions handled elsewhere, debugger output).
    if (code == 0x406D1388 || code == 0xE06D7363 || code == DBG_PRINTEXCEPTION_C || code == 0x4001000A)
        return EXCEPTION_CONTINUE_SEARCH;
    if (code != EXCEPTION_ACCESS_VIOLATION && code != EXCEPTION_ILLEGAL_INSTRUCTION && code != EXCEPTION_BREAKPOINT &&
        code != EXCEPTION_INT_DIVIDE_BY_ZERO && code != EXCEPTION_STACK_OVERFLOW && code != EXCEPTION_PRIV_INSTRUCTION)
        return EXCEPTION_CONTINUE_SEARCH;

    uintptr_t rip = info->ContextRecord->Rip;
    uint32_t guestFn = GuestFunctionAt(rip);
    auto exeBase = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
    log::Error("=== CRASH: exception {:08X} at host {:#x} (llvm-symbolizer --obj=ac6.exe {:#x}); nearest recompiled sub_{:08X} ===",
        static_cast<uint32_t>(code), rip, rip - exeBase + 0x140000000ull, guestFn);
    if (code == EXCEPTION_ACCESS_VIOLATION)
    {
        auto addr = reinterpret_cast<uint8_t*>(info->ExceptionRecord->ExceptionInformation[1]);
        bool write = info->ExceptionRecord->ExceptionInformation[0] == 1;
        if (g_memory.IsGuest(addr))
            log::Error("  {} guest address {:08X} (not backed)", write ? "write to" : "read from", g_memory.MapVirtual(addr));
        else
            log::Error("  {} host address {}", write ? "write to" : "read from", static_cast<void*>(addr));
    }
    DumpGuestContext();
    log::Error("  host stack:{}", DescribeHostStack(*info->ContextRecord));
    Shutdown(2, "crash");
}

// Host stack of a (suspended) thread as a compact chain: recompiled functions as sub_X, other frames
// in ac6.exe as symbolizer addresses, everything else as "ext". Innermost first.
std::string DescribeHostStack(CONTEXT c)
{
    auto exeBase = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
    std::string out;
    uint32_t lastGuest = 0;
    bool lastExt = false;
    for (int frame = 0; frame < 48 && c.Rip; frame++)
    {
        if (uint32_t guest = GuestFunctionAt(c.Rip))
        {
            if (guest != lastGuest)
                out += std::format(" sub_{:08X}", guest);
            lastGuest = guest;
            lastExt = false;
        }
        else
        {
            DWORD64 imageBase = 0;
            RtlLookupFunctionEntry(c.Rip, &imageBase, nullptr);
            if (imageBase == exeBase)
                out += std::format(" exe:{:#x}", c.Rip - exeBase + 0x140000000ull);
            else if (!lastExt)
                out += " ext";
            lastExt = imageBase != exeBase;
            lastGuest = 0;
        }

        DWORD64 imageBase = 0;
        PRUNTIME_FUNCTION entry = RtlLookupFunctionEntry(c.Rip, &imageBase, nullptr);
        if (!entry)
        {
            c.Rip = *reinterpret_cast<DWORD64*>(c.Rsp);   // leaf function
            c.Rsp += 8;
        }
        else
        {
            void* handlerData = nullptr;
            DWORD64 establisher = 0;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, c.Rip, entry, &c, &handlerData, &establisher, nullptr);
        }
    }
    return out;
}

BOOL WINAPI ConsoleHandler(DWORD type)
{
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT)
        Shutdown(1, "interrupted");
    return FALSE;
}

} // namespace

void detail::DumpGuestThreads()
{
    log::Info("--- guest threads ---");
    const DWORD self = GetCurrentThreadId();
    GuestThread::ForEachRunning([&](GuestThread& t) {
        PPCContext* ctx = t.context;
        std::string stack;
        HANDLE h = reinterpret_cast<HANDLE>(t.hostHandle);
        if (t.threadId != self && h && SuspendThread(h) != DWORD(-1))
        {
            CONTEXT c{};
            c.ContextFlags = CONTEXT_FULL;
            if (GetThreadContext(h, &c))
                stack = DescribeHostStack(c);
            ResumeThread(h);
        }
        log::Info("  thread {:5} kthread {:08X} entry {:08X} lr={:08X} r1={:08X} r3={:08X} |{}", t.threadId,
            t.DispatcherHeader(), t.params.function, ctx ? GuestLinkRegister(*ctx) : 0, ctx ? ctx->r1.u32 : 0,
            ctx ? ctx->r3.u32 : 0, stack);
    });
}

namespace {

// --- Write watch (diagnostics): logs every write to a guest range, with the value written -----------
// A write faults on the read-only pages. The handler notes the writer, makes the page writable,
// single-steps the instruction, logs the new value (writes inside the exact range only) and
// re-protects the page on the trap.
std::atomic<uint32_t> s_watchBegin{ 0 }, s_watchSize{ 0 };
std::atomic<uint32_t> s_watchExact{ 0 }, s_watchExactSize{ 0 };
std::atomic<uint32_t> s_watchLogged{ 0 };
thread_local uint8_t* t_reprotect = nullptr;
thread_local uint32_t t_hit = 0;            // watched word written by the instruction being stepped
thread_local std::string t_hitWriter;
std::mutex s_watchMutex;

LONG CALLBACK WatchHandler(EXCEPTION_POINTERS* info)
{
    const auto* record = info->ExceptionRecord;
    if (record->ExceptionCode == EXCEPTION_SINGLE_STEP && t_reprotect)
    {
        if (t_hit)
        {
            log::Info("Write watch: {:08X} = {:08X} by {}", t_hit, uint32_t(*g_memory.As<be<uint32_t>>(t_hit)), t_hitWriter);
            t_hit = 0;
        }
        DWORD old = 0;
        VirtualProtect(t_reprotect, 0x1000, PAGE_READONLY, &old);
        t_reprotect = nullptr;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    const uint32_t size = s_watchSize.load();
    if (!size || record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->ExceptionInformation[0] != 1)
        return EXCEPTION_CONTINUE_SEARCH;
    auto* addr = reinterpret_cast<uint8_t*>(record->ExceptionInformation[1]);
    if (!g_memory.IsGuest(addr))
        return EXCEPTION_CONTINUE_SEARCH;
    const uint32_t guest = g_memory.MapVirtual(addr), begin = s_watchBegin.load();
    if (guest < begin || guest - begin >= size)
        return EXCEPTION_CONTINUE_SEARCH;
    if (guest - s_watchExact.load() < s_watchExactSize.load() && s_watchLogged++ < 2000)
    {
        std::lock_guard lock(s_watchMutex);
        PPCContext* ctx = GetPPCContext();
        t_hit = guest & ~3u;
        t_hitWriter = std::format("sub_{:08X} lr={:08X} thread {} (host stack:{})", GuestFunctionAt(info->ContextRecord->Rip),
            ctx ? GuestLinkRegister(*ctx) : 0, GetCurrentThreadId(), DescribeHostStack(*info->ContextRecord));
    }
    uint8_t* page = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(addr) & ~uintptr_t(0xFFF));
    DWORD old = 0;
    VirtualProtect(page, 0x1000, PAGE_READWRITE, &old);
    t_reprotect = page;
    info->ContextRecord->EFlags |= 0x100;   // trap after this instruction
    return EXCEPTION_CONTINUE_EXECUTION;
}

} // namespace

void ArmWriteWatch(uint32_t guest, uint32_t size)
{
    DWORD old = 0;
    s_watchBegin = guest & ~0xFFFu;
    s_watchExact = guest;
    s_watchExactSize = size;
    const uint32_t bytes = ((guest + size + 0xFFF) & ~0xFFFu) - (guest & ~0xFFFu);
    if (!VirtualProtect(g_memory.Translate(guest & ~0xFFFu), bytes, PAGE_READONLY, &old))
    {
        log::Warn("Write watch: can't protect {:08X}+{:#x} (error {})", guest, size, GetLastError());
        return;
    }
    s_watchSize = bytes;
    log::Info("Write watch armed on {:08X}+{:#x}", guest & ~0xFFFu, bytes);
}

// --- Hardware write watch (diagnostics): AC6_WATCH_HW=<hex guest address> ---------------------------
// A debug-register data breakpoint (DR0, 4 bytes, writes) on the host address of one guest word, in
// every thread of the process (re-applied every 50 ms, so new threads are covered too). No slowdown,
// so it catches writes that only happen with normal timing. Every write is logged with its value and
// writer, up to 2000.
namespace {
std::atomic<uintptr_t> s_hwWatchHost{ 0 };
std::atomic<uint32_t> s_hwWatchGuest{ 0 }, s_hwWatchLogged{ 0 };
bool s_hwWatchReads = false;   // AC6_WATCH_HW=<address>:rw also catches reads

LONG CALLBACK HardwareWatchHandler(EXCEPTION_POINTERS* info)
{
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP || !(info->ContextRecord->Dr6 & 1) || !s_hwWatchHost.load())
        return EXCEPTION_CONTINUE_SEARCH;
    info->ContextRecord->Dr6 = 0;
    const uint32_t guest = s_hwWatchGuest.load();
    // Read watches: only the game's own accesses (the runtime's copies would fill the log).
    if (s_hwWatchReads && !GuestFunctionAt(info->ContextRecord->Rip))
        return EXCEPTION_CONTINUE_EXECUTION;
    // AC6_WATCH_HW_FROM=<seconds>: only the writes from then on.
    static const double from = [] { const char* v = std::getenv("AC6_WATCH_HW_FROM"); return v ? std::atof(v) : 0.0; }();
    if (log::Seconds() < from)
        return EXCEPTION_CONTINUE_EXECUTION;
    if (s_hwWatchLogged++ < 2000)
    {
        PPCContext* ctx = GetPPCContext();
        log::Info("HW watch: {:08X} = {:08X} by sub_{:08X} lr={:08X} thread {} (host stack:{})", guest, uint32_t(*g_memory.As<be<uint32_t>>(guest)),
            GuestFunctionAt(info->ContextRecord->Rip), ctx ? GuestLinkRegister(*ctx) : 0, GetCurrentThreadId(), DescribeHostStack(*info->ContextRecord));
    }
    return EXCEPTION_CONTINUE_EXECUTION;
}

void ApplyHardwareWatch(DWORD threadId, uintptr_t host)
{
    HANDLE thread = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME, FALSE, threadId);
    if (!thread)
        return;
    if (SuspendThread(thread) != DWORD(-1))
    {
        CONTEXT c{};
        c.ContextFlags = CONTEXT_DEBUG_REGISTERS;
        if (GetThreadContext(thread, &c) && c.Dr0 != host)
        {
            c.Dr0 = host;
            // L0, R/W0 = write (1) or read/write (3), LEN0 = 4 bytes
            c.Dr7 = (c.Dr7 & ~0xF0003ull) | 1 | ((s_hwWatchReads ? 3ull : 1ull) << 16) | (3ull << 18);
            SetThreadContext(thread, &c);
        }
        ResumeThread(thread);
    }
    CloseHandle(thread);
}
} // namespace

void StartHardwareWatch()
{
    const char* v = std::getenv("AC6_WATCH_HW");
    if (!v || !*v)
        return;
    const uint32_t guest = uint32_t(std::stoul(v, nullptr, 16)) & ~3u;
    s_hwWatchReads = std::string_view(v).ends_with(":rw");
    const uintptr_t host = reinterpret_cast<uintptr_t>(g_memory.Translate(guest));
    s_hwWatchGuest = guest;
    s_hwWatchHost = host;
    AddVectoredExceptionHandler(1, HardwareWatchHandler);
    log::Info("HW watch armed on {:08X} (host {:#x})", guest, host);
    std::thread([host] {
        const DWORD self = GetCurrentThreadId(), process = GetCurrentProcessId();
        for (;;)
        {
            HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            if (snapshot != INVALID_HANDLE_VALUE)
            {
                THREADENTRY32 entry{ sizeof(entry) };
                for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry))
                    if (entry.th32OwnerProcessID == process && entry.th32ThreadID != self)
                        ApplyHardwareWatch(entry.th32ThreadID, host);
                CloseHandle(snapshot);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }).detach();
}

std::string HostStack()
{
    CONTEXT c{};
    RtlCaptureContext(&c);
    return DescribeHostStack(c);
}

void detail::InstallPlatform()
{
    _set_invalid_parameter_handler([](const wchar_t*, const wchar_t*, const wchar_t*, unsigned, uintptr_t) {
        FatalNonException("CRT invalid parameter");
    });
    AddVectoredExceptionHandler(1, WatchHandler);
    // Top-level filter: only genuine crashes reach it (lazy commits are resolved by memory.cpp first).
    SetUnhandledExceptionFilter(CrashHandler);
    SetConsoleCtrlHandler(ConsoleHandler, TRUE);
}

} // namespace ac6::crash
