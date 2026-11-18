// The Linux half of the crash handler (crash.h): signals, unwinding through the executable's
// .eh_frame (libgcc's unwinder), the software write watch. The hardware watch (debug registers) isn't ported.
#include "crash.h"
#include "crash_internal.h"

#include "cpu/guest_thread.h"
#include "log.h"
#include "memory.h"
#include "platform/platform.h"
#include "ppc.h"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <format>
#include <mutex>
#include <string>

#include <link.h>
#include <sys/mman.h>
#include <ucontext.h>
#include <unwind.h>

// libgcc's frame-description lookup (exported by libgcc_s, not declared by <unwind.h>): the start of the function
// containing `pc`, from the unwind tables, as Windows' RtlLookupFunctionEntry.
struct dwarf_eh_bases
{
    void* tbase;
    void* dbase;
    void* func;
};
extern "C" const void* _Unwind_Find_FDE(void* pc, dwarf_eh_bases* bases);

namespace ac6::crash {

using detail::DumpGuestContext;

namespace {

// The executable's load address (it is position-independent): symbolizer offsets are relative to it.
uintptr_t ExecutableBase()
{
    static const uintptr_t base = [] {
        uintptr_t found = 0;
        dl_iterate_phdr([](dl_phdr_info* info, size_t, void* out) {
            *static_cast<uintptr_t*>(out) = info->dlpi_addr;   // the first object is the executable
            return 1;
        }, &found);
        return found;
    }();
    return base;
}

uintptr_t FunctionStart(uintptr_t pc)
{
    dwarf_eh_bases bases{};
    return _Unwind_Find_FDE(reinterpret_cast<void*>(pc), &bases) ? reinterpret_cast<uintptr_t>(bases.func) : 0;
}

// A host stack as a compact chain: recompiled functions as sub_X, other frames as symbolizer offsets. Innermost
// first. From a signal handler the unwinder walks through the signal frame into the interrupted code.
struct StackWalk
{
    std::string out;
    uint32_t lastGuest = 0;
    int frames = 0;
};

_Unwind_Reason_Code WalkFrame(_Unwind_Context* context, void* arg)
{
    auto& walk = *static_cast<StackWalk*>(arg);
    const uintptr_t pc = _Unwind_GetIP(context);
    if (!pc || ++walk.frames > 64)
        return _URC_END_OF_STACK;
    if (uint32_t guest = GuestFunctionAt(pc - 1))
    {
        if (guest != walk.lastGuest)
            walk.out += std::format(" sub_{:08X}", guest);
        walk.lastGuest = guest;
    }
    else
    {
        walk.out += std::format(" exe:{:#x}", pc - ExecutableBase());
        walk.lastGuest = 0;
    }
    return _URC_NO_REASON;
}

std::string DescribeThisStack()
{
    StackWalk walk;
    _Unwind_Backtrace(WalkFrame, &walk);
    return walk.out;
}

const char* SignalName(int signal)
{
    switch (signal)
    {
    case SIGSEGV: return "SIGSEGV";
    case SIGBUS: return "SIGBUS";
    case SIGILL: return "SIGILL";
    case SIGFPE: return "SIGFPE";
    case SIGTRAP: return "SIGTRAP";
    default: return "signal";
    }
}

void ReportCrash(int signal, void* address, bool write, bool hasAddress, void* context)
{
    const uintptr_t rip = context ? uintptr_t(static_cast<ucontext_t*>(context)->uc_mcontext.gregs[REG_RIP]) : 0;
    log::Error("=== CRASH: {} at host {:#x} (llvm-symbolizer --obj=ac6 {:#x}); nearest recompiled sub_{:08X} ===",
        SignalName(signal), rip, rip - ExecutableBase(), GuestFunctionAt(rip));
    if (hasAddress)
    {
        if (g_memory.IsGuest(address))
            log::Error("  {} guest address {:08X} (not backed)", write ? "write to" : "read from", g_memory.MapVirtual(address));
        else
            log::Error("  {} host address {}", write ? "write to" : "read from", address);
    }
    DumpGuestContext();
    log::Error("  host stack:{}", DescribeThisStack());
    Shutdown(2, "crash");
}

void OnCrashFault(const platform::Fault& fault)
{
    ReportCrash(fault.signal, fault.address, fault.write, true, fault.context);
}

void OnCrashSignal(int signal, siginfo_t* info, void* context)
{
    ReportCrash(signal, info->si_addr, false, false, context);
}

void OnInterrupt(int)
{
    Shutdown(1, "interrupted");
}

// --- Write watch (diagnostics), as on Windows: the write faults on the read-only page, which is made writable for one
// single-stepped instruction (the trap flag), then protected again on the SIGTRAP.
std::atomic<uint32_t> s_watchBegin{ 0 }, s_watchSize{ 0 };
std::atomic<uint32_t> s_watchExact{ 0 }, s_watchExactSize{ 0 };
std::atomic<uint32_t> s_watchLogged{ 0 };
thread_local uint8_t* t_reprotect = nullptr;
thread_local uint32_t t_hit = 0;
thread_local std::string t_hitWriter;
std::mutex s_watchMutex;

bool WatchFault(const platform::Fault& fault)
{
    const uint32_t size = s_watchSize.load();
    if (!size || !fault.write || !g_memory.IsGuest(fault.address))
        return false;
    const uint32_t guest = g_memory.MapVirtual(fault.address), begin = s_watchBegin.load();
    if (guest < begin || guest - begin >= size)
        return false;
    auto* uc = static_cast<ucontext_t*>(fault.context);
    if (guest - s_watchExact.load() < s_watchExactSize.load() && s_watchLogged++ < 2000)
    {
        std::lock_guard lock(s_watchMutex);
        PPCContext* ctx = GetPPCContext();
        t_hit = guest & ~3u;
        t_hitWriter = std::format("sub_{:08X} lr={:08X} thread {} (host stack:{})", GuestFunctionAt(uintptr_t(uc->uc_mcontext.gregs[REG_RIP])),
            ctx ? GuestLinkRegister(*ctx) : 0, platform::CurrentThreadId(), DescribeThisStack());
    }
    auto* page = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(fault.address) & ~uintptr_t(0xFFF));
    mprotect(page, 0x1000, PROT_READ | PROT_WRITE);
    t_reprotect = page;
    uc->uc_mcontext.gregs[REG_EFL] |= 0x100;   // trap after this instruction
    return true;
}

void OnTrap(int signal, siginfo_t* info, void* context)
{
    auto* uc = static_cast<ucontext_t*>(context);
    if (!t_reprotect)
    {
        ReportCrash(signal, info->si_addr, false, false, context);
        return;
    }
    uc->uc_mcontext.gregs[REG_EFL] &= ~0x100;
    if (t_hit)
    {
        log::Info("Write watch: {:08X} = {:08X} by {}", t_hit, uint32_t(*g_memory.As<be<uint32_t>>(t_hit)), t_hitWriter);
        t_hit = 0;
    }
    mprotect(t_reprotect, 0x1000, PROT_READ);
    t_reprotect = nullptr;
}

void SetSignal(int signal, void (*handler)(int, siginfo_t*, void*))
{
    struct sigaction action{};
    action.sa_sigaction = handler;
    action.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigemptyset(&action.sa_mask);
    sigaction(signal, &action, nullptr);
}

} // namespace

uint32_t GuestFunctionAt(uintptr_t host)
{
    const uintptr_t start = FunctionStart(host);
    return start ? detail::GuestAtFunctionStart(start) : 0;
}

void detail::DumpGuestThreads()
{
    // (Other threads' host stacks would need them stopped: only their guest state here.)
    log::Info("--- guest threads ---");
    GuestThread::ForEachRunning([&](GuestThread& t) {
        PPCContext* ctx = t.context;
        log::Info("  thread {:5} kthread {:08X} entry {:08X} lr={:08X} r1={:08X} r3={:08X}", t.threadId,
            t.DispatcherHeader(), t.params.function, ctx ? GuestLinkRegister(*ctx) : 0, ctx ? ctx->r1.u32 : 0,
            ctx ? ctx->r3.u32 : 0);
    });
}

void ArmWriteWatch(uint32_t guest, uint32_t size)
{
    s_watchBegin = guest & ~0xFFFu;
    s_watchExact = guest;
    s_watchExactSize = size;
    const uint32_t bytes = ((guest + size + 0xFFF) & ~0xFFFu) - (guest & ~0xFFFu);
    if (mprotect(g_memory.Translate(guest & ~0xFFFu), bytes, PROT_READ) != 0)
    {
        log::Warn("Write watch: can't protect {:08X}+{:#x} (error {})", guest, size, errno);
        return;
    }
    s_watchSize = bytes;
    log::Info("Write watch armed on {:08X}+{:#x}", guest & ~0xFFFu, bytes);
}

void StartHardwareWatch()
{
    if (const char* v = std::getenv("AC6_WATCH_HW"); v && *v)
        log::Warn("AC6_WATCH_HW: the hardware write watch is Windows-only");
}

std::string HostStack()
{
    return DescribeThisStack();
}

void detail::InstallPlatform()
{
    platform::AddFaultHandler(-1, WatchFault);   // before guest memory's (diagnostics only, when armed)
    platform::SetCrashHandler(OnCrashFault);
    SetSignal(SIGILL, OnCrashSignal);
    SetSignal(SIGFPE, OnCrashSignal);
    SetSignal(SIGTRAP, OnTrap);
    signal(SIGINT, OnInterrupt);
    signal(SIGTERM, OnInterrupt);
}

} // namespace ac6::crash
