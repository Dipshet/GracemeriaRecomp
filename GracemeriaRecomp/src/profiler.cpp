// Sampling profiler (AC6_PROFILE=<seconds>): every millisecond each thread of the process is suspended
// and its instruction pointer attributed to a recompiled guest function (sub_X), a host function of the
// executable (exe:<address of its start>, for llvm-symbolizer) or a DLL. Each interval the busiest
// threads and their top entries go to the log.
#include "profiler.h"

#include "crash.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <format>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)
// Windows only: it suspends threads to read their instruction pointers. Linux has perf.
namespace ac6::profiler {

void Stop() {}

void Start()
{
    if (const char* v = std::getenv("AC6_PROFILE"); v && *v)
        log::Warn("AC6_PROFILE: the sampling profiler is Windows-only (use perf on Linux)");
}

} // namespace ac6::profiler
#else
#include <windows.h>
#include <tlhelp32.h>

namespace ac6::profiler {

namespace {

uint32_t s_stackDepth = 0;   // AC6_PROFILE_STACK: executable frames reported beyond the first caller
std::atomic<bool> s_stop{ false }, s_sampling{ false };   // Stop(): no thread is left suspended at exit
bool s_lines = false;        // AC6_PROFILE_LINES=1: exact addresses (for source lines) instead of function starts

std::string Where(uintptr_t rip, uintptr_t exeBase)
{
    if (const uint32_t guest = crash::GuestFunctionAt(rip))
        return std::format("sub_{:08X}", guest);
    DWORD64 imageBase = 0;
    if (PRUNTIME_FUNCTION entry = RtlLookupFunctionEntry(rip, &imageBase, nullptr); entry && imageBase == exeBase)
        return std::format("exe:{:#x}", (s_lines ? rip : imageBase + entry->BeginAddress) - exeBase + 0x140000000ull);
    // Leaf functions have no unwind entry: still attributable to the executable by address.
    if (s_lines && rip >= exeBase && rip < exeBase + 0x8000000)
        return std::format("exe:{:#x}", rip - exeBase + 0x140000000ull);
    HMODULE module = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(rip), &module))
    {
        wchar_t path[MAX_PATH];
        const DWORD n = GetModuleFileNameW(module, path, MAX_PATH);
        std::wstring name(path, n);
        name = name.substr(name.find_last_of(L"\\/") + 1);
        return std::string(name.begin(), name.end());
    }
    return "?";
}

// A thread's stack bounds, from its TEB (the stack walk below must not leave them: unwinding a thread stopped at
// an arbitrary instruction can follow a bad frame, and a read outside the stack crashes the profiler).
struct ThreadBasicInformation
{
    LONG exitStatus;
    PVOID tebBaseAddress;
    PVOID clientId[2];   // CLIENT_ID
    ULONG_PTR affinityMask;
    LONG priority, basePriority;
};
using NtQueryInformationThreadFn = LONG(NTAPI*)(HANDLE, ULONG, PVOID, ULONG, PULONG);

const NT_TIB* ThreadTib(HANDLE h)
{
    static const auto query = reinterpret_cast<NtQueryInformationThreadFn>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationThread"));
    ThreadBasicInformation info{};
    if (!query || query(h, 0, &info, sizeof(info), nullptr) != 0)
        return nullptr;
    return static_cast<const NT_TIB*>(info.tebBaseAddress);
}

void Run(double interval)
{
    const DWORD self = GetCurrentThreadId(), process = GetCurrentProcessId();
    const auto exeBase = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
    const auto* exeNt = reinterpret_cast<const IMAGE_NT_HEADERS*>(exeBase + reinterpret_cast<const IMAGE_DOS_HEADER*>(exeBase)->e_lfanew);
    const uintptr_t exeEnd = exeBase + exeNt->OptionalHeader.SizeOfImage;
    // A stack word that is a return address into the executable: inside it, right after a call instruction.
    auto isReturnAddress = [&](uintptr_t a) {
        if (a < exeBase + 0x1000 || a >= exeEnd)
            return false;
        const auto* p = reinterpret_cast<const uint8_t*>(a);
        return p[-5] == 0xE8 ||                                              // call rel32
            (p[-6] == 0xFF && (p[-5] & 0x38) == 0x10) ||                     // call [rip+disp32] / [reg+disp32]
            (p[-2] == 0xFF && (p[-1] & 0xF8) == 0xD0) ||                     // call reg
            (p[-3] == 0x41 && p[-2] == 0xFF && (p[-1] & 0xF8) == 0xD0) ||    // call r8-r15
            (p[-3] == 0xFF && (p[-2] & 0xF8) == 0x50);                       // call [reg+disp8]
    };
    std::unordered_map<DWORD, std::unordered_map<std::string, uint32_t>> hits;
    struct Sampled { HANDLE handle; const NT_TIB* tib; };
    std::unordered_map<DWORD, Sampled> handles;
    auto windowStart = std::chrono::steady_clock::now();
    uint32_t ticks = 0;
    for (;;)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ticks++;
        if (ticks % 500 == 1)   // pick up new threads now and then
        {
            HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            THREADENTRY32 entry{ sizeof(entry) };
            for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry))
                if (entry.th32OwnerProcessID == process && entry.th32ThreadID != self && !handles.contains(entry.th32ThreadID))
                    if (HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, entry.th32ThreadID))
                        handles.emplace(entry.th32ThreadID, Sampled{ h, ThreadTib(h) });
            CloseHandle(snapshot);
        }
        s_sampling = true;
        if (s_stop)
        {
            s_sampling = false;
            return;
        }
        for (auto& [id, sampled] : handles)
        {
            const HANDLE h = sampled.handle;
            if (SuspendThread(h) == DWORD(-1))
                continue;
            CONTEXT c{};
            c.ContextFlags = CONTEXT_FULL;
            const bool got = GetThreadContext(h, &c);
            // In a DLL (waiting, the driver): find the first caller in the executable while the thread is
            // still suspended (the stack is stable), so waits are attributed to what waited. With
            // AC6_PROFILE_STACK=<n>, n further executable frames follow (up to the first guest function).
            // The stack is scanned for return addresses instead of unwound: RtlVirtualUnwind on a thread stopped at
            // an arbitrary instruction can follow a bad frame and read outside the stack. Only the thread's own
            // stack is read, from its TEB's bounds, and at most 16 KB of it.
            uintptr_t callers[9] = {};
            uint32_t callerCount = 0;
            const uintptr_t stackLow = sampled.tib ? reinterpret_cast<uintptr_t>(sampled.tib->StackLimit) : 0;
            const uintptr_t stackHigh = sampled.tib ? reinterpret_cast<uintptr_t>(sampled.tib->StackBase) : 0;
            const bool inExe = got && c.Rip >= exeBase && c.Rip < exeEnd;
            const uint32_t want = inExe ? s_stackDepth : 1 + s_stackDepth;
            if (got && want && c.Rsp >= stackLow && c.Rsp < stackHigh && !(inExe && crash::GuestFunctionAt(c.Rip)))
            {
                const uintptr_t end = std::min<uintptr_t>(stackHigh, c.Rsp + (16u << 10));
                for (uintptr_t at = c.Rsp & ~uintptr_t(7); at + 8 <= end && callerCount < want; at += 8)
                {
                    const uintptr_t value = *reinterpret_cast<const uintptr_t*>(at);
                    if (!isReturnAddress(value))
                        continue;
                    callers[callerCount++] = value;
                    if (crash::GuestFunctionAt(value))
                        break;
                }
            }
            ResumeThread(h);
            if (got)
            {
                std::string where = Where(c.Rip, exeBase);
                for (uint32_t k = 0; k < callerCount; k++)
                    where += "<" + Where(callers[k], exeBase);
                hits[id][where]++;
            }
        }
        s_sampling = false;
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - windowStart).count();
        if (elapsed < interval)
            continue;
        // Report the threads that were busy (not mostly waiting in the kernel) with their top entries.
        std::vector<std::pair<uint32_t, DWORD>> threads;
        for (auto& [id, m] : hits)
        {
            uint32_t total = 0, waiting = 0;
            for (auto& [where, n] : m)
            {
                total += n;
                if (where.starts_with("ntdll.dll") || where.starts_with("win32u.dll"))
                    waiting += n;
            }
            threads.push_back({ total - waiting, id });
        }
        std::sort(threads.rbegin(), threads.rend());
        log::Info("Profile: {:.1f} s, {} samples per thread", elapsed, ticks);
        for (size_t t = 0; t < threads.size() && t < 6 && threads[t].first > ticks / 50; t++)
        {
            auto& m = hits[threads[t].second];
            uint32_t total = 0;
            std::vector<std::pair<uint32_t, std::string>> top;
            for (auto& [where, n] : m)
            {
                total += n;
                top.push_back({ n, where });
            }
            std::sort(top.rbegin(), top.rend());
            std::string line;
            for (size_t i = 0; i < top.size() && i < (s_lines ? 160u : 24u); i++)
                line += std::format(" {} {:.1f}%", top[i].second, 100.0 * top[i].first / total);
            log::Info("Profile thread {} ({:.0f}% outside ntdll/win32u):{}", threads[t].second, 100.0 * threads[t].first / total, line);
        }
        hits.clear();
        ticks = 0;
        windowStart = std::chrono::steady_clock::now();
    }
}

} // namespace

void Stop()
{
    s_stop = true;
    while (s_sampling)
        std::this_thread::yield();
}

void Start()
{
    const char* v = std::getenv("AC6_PROFILE");
    if (!v || !*v)
        return;
    const double interval = std::max(1.0, std::atof(v));
    if (const char* l = std::getenv("AC6_PROFILE_LINES"))
        s_lines = *l == '1';
    if (const char* d = std::getenv("AC6_PROFILE_STACK"))
        s_stackDepth = std::clamp(std::atoi(d), 0, 8);
    // Suspending threads while DLLs load at startup can deadlock (the unwinder needs locks the suspended thread
    // may hold): sampling starts after AC6_PROFILE_DELAY seconds (default 15).
    const char* d = std::getenv("AC6_PROFILE_DELAY");
    const double delay = d && *d ? std::atof(d) : 15.0;
    std::thread([interval, delay] {
        std::this_thread::sleep_for(std::chrono::duration<double>(delay));
        Run(interval);
    }).detach();
    log::Info("Profiler: sampling every thread each millisecond, report every {} s", interval);
}

} // namespace ac6::profiler
#endif
