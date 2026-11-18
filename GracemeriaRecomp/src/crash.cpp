// The platform-independent half of the crash handler (crash.h): guest state dumps, the hang watchdog, shutdown. The
// faults, host stacks and write watches are crash_win32.cpp's / crash_linux.cpp's.
#include "crash.h"
#include "crash_internal.h"

#include "cpu/guest_thread.h"
#include "fs/vfs.h"
#include "input/dualsense.h"
#include "kernel/dispatcher.h"
#include "kernel/trace.h"
#include "log.h"
#include "platform/platform.h"
#include "profiler.h"
#include "memory.h"
#include "ppc.h"

#include <algorithm>
#include <csignal>
#include <exception>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <format>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <mutex>
#include <vector>

#if defined(AC6_PGO_GENERATE)
extern "C" int __llvm_profile_write_file(void);   // the profile runtime's
#endif

namespace ac6::crash {

namespace {

std::unordered_map<uintptr_t, uint32_t> s_hostToGuest; // host function start -> guest address (recompiled or hooked)
std::atomic<bool> s_shuttingDown{ false };
bool s_showFatalErrors = false;

void BuildFunctionIndex()
{
    for (size_t i = 0; PPCFuncMappings[i].guest != 0; i++)
        if (PPCFuncMappings[i].host)
            s_hostToGuest.emplace(reinterpret_cast<uintptr_t>(PPCFuncMappings[i].host), PPCFuncMappings[i].guest);
}

} // namespace

uint32_t detail::GuestAtFunctionStart(uintptr_t start)
{
    auto it = s_hostToGuest.find(start);
    return it != s_hostToGuest.end() ? it->second : 0;
}

namespace {
thread_local const char* t_task = nullptr;
thread_local void (*t_taskOnFault)(void*) = nullptr;
thread_local void* t_taskContext = nullptr;
}

void SetThreadTask(const char* task, void (*onFault)(void*), void* context)
{
    t_task = task;
    t_taskOnFault = onFault;
    t_taskContext = context;
}

void detail::DumpGuestContext()
{
    // What the thread was doing (SetThreadTask), and that task's own note of the fault. Both are cleared first, so a
    // fault inside the task's handler doesn't report the task again.
    if (const char* task = std::exchange(t_task, nullptr))
    {
        log::Error("  while {}", task);
        if (auto* onFault = std::exchange(t_taskOnFault, nullptr))
            onFault(t_taskContext);
    }
    PPCContext* ctx = GetPPCContext();
    if (!ctx)
    {
        log::Error("  (no guest context on this thread)");
        return;
    }
    // The recompiler's register-as-local options (config/AC6.toml) leave registers out of
    // the context: only those still in it are shown.
    log::Error("  r1={:08X} r13={:08X}", ctx->r1.u32, ctx->r13.u32);
#ifndef PPC_CONFIG_SKIP_LR
    log::Error("  lr={:08X}", static_cast<uint32_t>(ctx->lr));
#endif
#ifndef PPC_CONFIG_CTR_AS_LOCAL
    log::Error("  ctr={:08X}", ctx->ctr.u32);
#endif
    log::Error("  r3={:08X} r4={:08X} r5={:08X} r6={:08X} r7={:08X} r8={:08X} r9={:08X} r10={:08X}",
        ctx->r3.u32, ctx->r4.u32, ctx->r5.u32, ctx->r6.u32, ctx->r7.u32, ctx->r8.u32, ctx->r9.u32, ctx->r10.u32);
#ifndef PPC_CONFIG_NON_ARGUMENT_AS_LOCAL
    log::Error("  r11={:08X} r12={:08X}", ctx->r11.u32, ctx->r12.u32);
#endif
#ifndef PPC_CONFIG_NON_VOLATILE_AS_LOCAL
    log::Error("  r14-r31: {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X}",
        ctx->r14.u32, ctx->r15.u32, ctx->r16.u32, ctx->r17.u32, ctx->r18.u32, ctx->r19.u32, ctx->r20.u32, ctx->r21.u32,
        ctx->r22.u32, ctx->r23.u32, ctx->r24.u32, ctx->r25.u32, ctx->r26.u32, ctx->r27.u32, ctx->r28.u32, ctx->r29.u32,
        ctx->r30.u32, ctx->r31.u32);
#endif
    // Guest call chain from the stack back chain: each frame's word 0 links to the caller's frame, and
    // a function saves its return address at -8 from that caller frame (stw r12,-8(r1) before stwu).
    std::string chain;
    uint32_t sp = ctx->r1.u32;
    for (int i = 0; i < 32; i++)
    {
        if (!g_memory.IsBacked(sp))
            break;
        const uint32_t next = *g_memory.As<be<uint32_t>>(sp);
        if (next <= sp || next - sp > 0x10000 || !g_memory.IsBacked(next - 8))
            break;
        chain += std::format(" {:08X}", uint32_t(*g_memory.As<be<uint32_t>>(next - 8)));
        sp = next;
    }
    log::Error("  guest return addresses (innermost first; leaf frames may be missing):{}", chain);
    // Memory the registers point at (32 bytes each, big-endian words): the data a crash was walking.
    const PPCRegister* regs[] = { &ctx->r3, &ctx->r4, &ctx->r5, &ctx->r6, &ctx->r7, &ctx->r8, &ctx->r9, &ctx->r10,
#ifndef PPC_CONFIG_NON_ARGUMENT_AS_LOCAL
        &ctx->r11,
#endif
#ifndef PPC_CONFIG_NON_VOLATILE_AS_LOCAL
        &ctx->r14, &ctx->r15, &ctx->r16, &ctx->r17, &ctx->r18, &ctx->r19, &ctx->r20, &ctx->r21, &ctx->r22, &ctx->r23, &ctx->r24,
        &ctx->r25, &ctx->r26, &ctx->r27, &ctx->r28, &ctx->r29, &ctx->r30, &ctx->r31,
#endif
    };
    const char* names[] = { "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10",
#ifndef PPC_CONFIG_NON_ARGUMENT_AS_LOCAL
        "r11",
#endif
#ifndef PPC_CONFIG_NON_VOLATILE_AS_LOCAL
        "r14", "r15", "r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23", "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31",
#endif
    };
    for (size_t i = 0; i < std::size(regs); i++)
    {
        const uint32_t address = regs[i]->u32 & ~3u;
        if (address < 0x10000 || !g_memory.IsBacked(address) || !g_memory.IsBacked(address + 31))
            continue;
        std::string words;
        for (uint32_t w = 0; w < 8; w++)
            words += std::format(" {:08X}", uint32_t(*g_memory.As<be<uint32_t>>(address + w * 4)));
        log::Error("  [{}={:08X}]{}", names[i], address, words);
    }
}

// Failures that bypass the exception filter (std::terminate, abort, CRT invalid parameters) are
// logged and turned into the normal shutdown, so they leave a reason in the log.
void detail::FatalNonException(const char* what)
{
    log::Error("=== FATAL: {} on host thread {} ===", what, platform::CurrentThreadId());
    DumpGuestContext();
    log::Error("  host stack:{}", HostStack());
    Shutdown(3, what);
}

void Install()
{
    BuildFunctionIndex();
    std::set_terminate([] { detail::FatalNonException("std::terminate"); });
    signal(SIGABRT, [](int) { detail::FatalNonException("abort()"); });
    detail::InstallPlatform();
}

namespace {
std::atomic<uint64_t> s_frames{ 0 };
}

void NoteFrame()
{
    s_frames++;
}

void StartWatchdog()
{
    // Hang report: once frames have started, a stall of AC6_HANG_SECONDS (default 8) without a present
    // logs every guest thread's stack and the kernel waits, once per stall.
    const char* h = std::getenv("AC6_HANG_SECONDS");
    const double hangSeconds = h && *h ? std::atof(h) : 8.0;
    if (hangSeconds > 0)
        std::thread([hangSeconds] {
            uint64_t last = 0;
            auto since = std::chrono::steady_clock::now();
            bool reported = false;
            for (;;)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                const uint64_t frames = s_frames.load();
                const auto now = std::chrono::steady_clock::now();
                if (frames != last)
                {
                    if (reported)
                        log::Info("Frames resumed after the stall (frame {})", frames);
                    last = frames;
                    since = now;
                    reported = false;
                    continue;
                }
                if (frames && !reported && now - since > std::chrono::duration<double>(hangSeconds))
                {
                    reported = true;
                    log::Error("=== HANG: no frame presented for {:.0f} s (last frame {}) ===", hangSeconds, frames);
                    detail::DumpGuestThreads();
                    kernel::DumpWaits();
                }
            }
        }).detach();

    const char* v = std::getenv("AC6_RUN_SECONDS");
    if (!v || !*v)
        return;
    double seconds = std::atof(v);
    std::thread([seconds] {
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
        Shutdown(0, "AC6_RUN_SECONDS elapsed");
    }).detach();
}

void ShowFatalErrors(bool enable) { s_showFatalErrors = enable; }

void ShowFatalError(const char* message)
{
    // Background runs (AC6_BACKGROUND=1: tests and replays) leave the error in the log instead of a message
    // box in front of whoever is using the machine.
    if (const char* v = std::getenv("AC6_BACKGROUND"); v && *v == '1')
        return;
    platform::ShowErrorBox("ACE COMBAT 6: FIRES OF LIBERATION", message);
}

void Shutdown(int code, const char* reason)
{
    if (!s_shuttingDown.exchange(true))
    {
        profiler::Stop();
        log::Info("Shutting down: {}", reason);
        input::dualsense::Clear();   // a DualSense's triggers back to normal
        detail::DumpGuestThreads();
        kernel::DumpWaits();
        fs::DumpStats();
        kernel::WriteImportSummary();
#if defined(AC6_PGO_GENERATE)
        // std::_Exit skips the profile runtime's atexit hook: written here (an instrumented build).
        __llvm_profile_write_file();
#endif
        if (code != 0 && s_showFatalErrors)
            ShowFatalError(std::format("The game stopped ({}). Details are in ac6_runtime.log next to the executable.", reason).c_str());
    }
    std::_Exit(code);
}

} // namespace ac6::crash
