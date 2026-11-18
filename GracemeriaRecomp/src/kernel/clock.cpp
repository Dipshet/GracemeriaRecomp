#include "kernel/clock.h"

#include "kernel/dispatcher.h"
#include "kernel/xtypes.h"
#include "loader.h"
#include "log.h"
#include "memory.h"
#include "platform/platform.h"

#include <chrono>
#include <cstring>
#include <thread>

#include <x86intrin.h>

namespace ac6::kernel {

namespace {

uint64_t s_timebaseFrequency = 0;
uint64_t s_bootTime = 0;
std::chrono::steady_clock::time_point s_bootClock;

struct XTimeStampBundle
{
    be<uint64_t> interruptTime;   // 100 ns since boot
    be<uint64_t> systemTime;      // FILETIME
    be<uint32_t> tickCount;       // ms since boot
    be<uint32_t> padding;
};

void UpdateTimeStampBundle(XTimeStampBundle* bundle)
{
    auto sinceBoot = std::chrono::steady_clock::now() - s_bootClock;
    bundle->interruptTime = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(sinceBoot).count() / 100);
    bundle->systemTime = SystemTime100ns();
    bundle->tickCount = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(sinceBoot).count());
}

} // namespace

void InitClock()
{
    platform::RaiseTimerResolution();
    s_bootTime = SystemTime100ns();
    s_bootClock = std::chrono::steady_clock::now();

    // TSC ticks over a measured interval of the steady clock (QPC on Windows, CLOCK_MONOTONIC on Linux).
    const auto clock0 = std::chrono::steady_clock::now();
    uint64_t tsc0 = __rdtsc();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto clock1 = std::chrono::steady_clock::now();
    uint64_t tsc1 = __rdtsc();
    double seconds = std::chrono::duration<double>(clock1 - clock0).count();
    s_timebaseFrequency = static_cast<uint64_t>(double(tsc1 - tsc0) / seconds);
    log::Info("Guest time base (host TSC): {:.3f} MHz", s_timebaseFrequency / 1e6);
}

uint64_t TimebaseFrequency() { return s_timebaseFrequency; }
uint64_t BootTime100ns() { return s_bootTime; }

void InitVariableImports()
{
    if (uint32_t v = FindVariableImport(g_module, "XboxKrnlVersion"))
    {
        auto* version = g_memory.As<be<uint16_t>>(v);
        version[0] = 2;       // major
        version[1] = 0;       // minor
        version[2] = 17559;   // build (final console kernel)
        version[3] = 0;       // qfe
    }
    if (uint32_t v = FindVariableImport(g_module, "VdGpuClockInMHz"))
        *g_memory.As<be<uint32_t>>(v) = 500;
    if (uint32_t v = FindVariableImport(g_module, "VdHSIOCalibrationLock"))
    {
        // A kernel-owned critical section (the D3D library takes it while training EDRAM): zeroed memory
        // would read as "held" (LockCount 0), so initialise it as the kernel does.
        auto* cs = g_memory.As<XRtlCriticalSection>(v);
        InitializeHeader(cs->header, kObjEventSynchronization, 0);
        cs->lockCount = -1;
    }
    if (uint32_t v = FindVariableImport(g_module, "ExLoadedCommandLine"))
        std::strcpy(g_memory.As<char>(v), "\"default.xex\"");
    if (uint32_t v = FindVariableImport(g_module, "KeTimeStampBundle"))
    {
        auto* bundle = g_memory.As<XTimeStampBundle>(v);
        UpdateTimeStampBundle(bundle);
        std::thread([bundle] {
            platform::SetCurrentThreadPriority(platform::kPriorityHighest);
            for (;;)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                UpdateTimeStampBundle(bundle);
            }
        }).detach();
    }
}

} // namespace ac6::kernel
