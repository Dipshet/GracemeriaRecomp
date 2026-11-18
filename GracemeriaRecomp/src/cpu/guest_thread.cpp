// Guest threads, see guest_thread.h. Block layout started from UnleashedRecomp (GPL-3.0).
#include "cpu/guest_thread.h"

#include "heap.h"
#include "kernel/dispatcher.h"
#include "loader.h"
#include "log.h"
#include "memory.h"
#include "platform/platform.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <mutex>

namespace ac6 {

namespace {

// PCR fields the game reads (r13-relative): +0 static TLS data, +0x100 current KTHREAD,
// +0x10C processor number, +0x150 "in DPC" flag (must stay 0 for Get/SetLastError).
constexpr uint32_t kPcrSize = 0xAB0;
constexpr uint32_t kPcrTls = 0x000;
constexpr uint32_t kPcrCurrentThread = 0x100;
constexpr uint32_t kPcrProcessor = 0x10C;
constexpr uint32_t kMinStack = 0x10000;
constexpr uint32_t kStackSlack = 0x100;          // room above the initial r1 for the callee's back-chain stores

thread_local GuestThread* t_current = nullptr;
std::mutex s_registryMutex;
std::vector<GuestThread*> s_running;

constexpr uint32_t AlignUp(uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); }

void StoreBE32(uint8_t* p, uint32_t v) { *reinterpret_cast<be<uint32_t>*>(p) = v; }

void HostThreadProc(void* arg)
{
    auto* thread = static_cast<GuestThread*>(arg);
    thread->Run();
    kernel::ReleaseObject(thread);   // the running reference
}

} // namespace

GuestThread::GuestThread(const GuestThreadParams& p)
    : params(p)
{
    kernel::InitializeHeader(kthread.header, kernel::kObjThread, 0);
}

GuestThread::~GuestThread()
{
    if (hostHandle)
        platform::CloseThread(hostHandle);
}

uint32_t GuestThread::DispatcherHeader()
{
    return g_memory.MapVirtual(&kthread.header);
}

GuestThread* GuestThread::Current() { return t_current; }

void GuestThread::LockRegistry() { s_registryMutex.lock(); }
void GuestThread::UnlockRegistry() { s_registryMutex.unlock(); }
const std::vector<GuestThread*>& GuestThread::Running() { return s_running; }

GuestThread* GuestThread::FromKThread(uint32_t kthread)
{
    return dynamic_cast<GuestThread*>(kernel::ObjectFromHeader(kthread));
}

uint32_t GuestThread::ProcessorFromFlags(uint32_t flags)
{
    const auto mask = static_cast<uint8_t>(flags >> 24);
    return mask == 0 ? 0 : 7 - std::countl_zero(mask);
}

uint32_t GuestThread::Run()
{
    t_current = this;

    const TlsInfo& tls = g_module.tls;
    const uint32_t slotCount = tls.slotCount ? tls.slotCount : 64;
    const uint32_t tlsDataSize = AlignUp(std::max(tls.dataSize, tls.rawDataSize), 16);
    uint32_t stackSize = params.stackSize ? params.stackSize : (g_module.defaultStackSize ? g_module.defaultStackSize : 0x40000);
    stackSize = AlignUp(std::max(stackSize, kMinStack), 0x1000);
    const uint32_t slotsSize = AlignUp(slotCount * 4, 16);
    const uint32_t total = kPcrSize + slotsSize + tlsDataSize + stackSize;

    auto* block = static_cast<uint8_t*>(g_heap.Alloc(total));
    if (!block)
    {
        log::Error("Out of user heap for a {:#x}-byte guest thread block", total);
        kernel::SignalObject(DispatcherHeader());
        return exitCode = 0;
    }
    std::memset(block, 0, total);
    uint8_t* slots = block + kPcrSize;
    uint8_t* tlsData = slots + slotsSize;
    if (tls.rawDataAddress && tls.rawDataSize)
        std::memcpy(tlsData, g_memory.Translate(tls.rawDataAddress), tls.rawDataSize);

    StoreBE32(block + kPcrTls, g_memory.MapVirtual(tlsData));
    StoreBE32(block + kPcrCurrentThread, g_memory.MapVirtual(&kthread));
    // The processor number follows the affinity once one is set (it may have been set while the thread
    // was created suspended), else the creation flags.
    pcr = block;
    const uint32_t mask = affinity.load();
    block[kPcrProcessor] = static_cast<uint8_t>(mask ? std::countr_zero(mask) : ProcessorFromFlags(params.flags));
    kthread.threadId = threadId;
    tlsSlots = g_memory.MapVirtual(slots);

    PPCContext ctx{};
    ctx.r1.u64 = g_memory.MapVirtual(block + total - kStackSlack);
    ctx.r13.u64 = g_memory.MapVirtual(block);
    ctx.fpscr.loadFromHost();
    SetPPCContext(ctx);
    context = &ctx;
    {
        std::lock_guard lock(s_registryMutex);
        s_running.push_back(this);
    }

    uint32_t entry = params.startup ? params.startup : params.function;
    PPCFunc* func = params.hostEntry ? nullptr : g_memory.Function(entry);
    if (params.hostEntry)
    {
        params.hostEntry(params.hostArg);
    }
    else if (!func)
    {
        log::Error("No recompiled function at {:08X}", entry);
    }
    else
    {
        if (params.startup)
        {
            ctx.r3.u64 = params.function;
            ctx.r4.u64 = params.value;
        }
        else
        {
            ctx.r3.u64 = params.value;
        }
        try
        {
            func(ctx, g_memory.base);
            exitCode = ctx.r3.u32;
        }
        catch (const GuestThreadExit& e)
        {
            exitCode = e.exitCode;
        }
    }

    {
        std::lock_guard lock(s_registryMutex);
        std::erase(s_running, this);
    }
    context = nullptr;
    g_ppcContext = nullptr;
    tlsSlots = 0;
    kernel::SignalObject(DispatcherHeader());
    pcr = nullptr;
    g_heap.Free(block);
    return exitCode;
}

GuestThread* GuestThread::Start(const GuestThreadParams& params)
{
    auto* thread = kernel::CreateKernelObject<GuestThread>(params);
    if (!thread)
        return nullptr;
    kernel::RetainObject(thread);           // the running reference, dropped by HostThreadProc
    kernel::RegisterObjectHeader(thread);

    // Created suspended. One suspension stands for "not launched yet" and is undone by the caller's
    // Resume() once it holds a handle. A guest CREATE_SUSPENDED adds a second one.
    thread->suspendCount = (params.flags & 1) ? 2 : 1;
    uint32_t tid = 0;
    platform::HostThread* h = platform::CreateHeldThread(HostThreadProc, thread, tid);
    if (!h)
    {
        log::Error("CreateThread failed for guest thread {:08X}", params.function);
        kernel::ReleaseObject(thread);
        return nullptr;
    }
    thread->hostHandle = h;
    thread->threadId = tid;
    if (uint32_t mask = params.flags >> 24)   // 1 << processor (the game's CreateThread wrapper builds it)
        thread->SetAffinity(mask);
    return thread;
}

uint32_t GuestThread::RunOnCurrentThread(const GuestThreadParams& params)
{
    auto* thread = kernel::CreateKernelObject<GuestThread>(params);
    kernel::RetainObject(thread);
    kernel::RegisterObjectHeader(thread);
    thread->hostHandle = platform::AdoptCurrentThread();
    thread->threadId = platform::CurrentThreadId();
    return thread->Run();
}

int32_t GuestThread::Resume()
{
    int32_t previous = suspendCount.load();
    while (previous > 0 && !suspendCount.compare_exchange_weak(previous, previous - 1))
        ;
    if (previous == 1 && hostHandle)
        platform::Launch(hostHandle);
    return previous;
}

void GuestThread::SetPriority(int32_t priority)
{
    // Console priorities: small increments (-2..2) plus the "time critical"/"idle" ends (around
    // +-15..17). The ends map to HIGHEST/LOWEST rather than TIME_CRITICAL/IDLE so a host with fewer
    // cores than the game has threads can't starve anything.
    basePriority = priority;
    int host = priority >= 15 ? platform::kPriorityHighest
        : priority <= -15 ? platform::kPriorityLowest
        : std::clamp(priority, -2, 2);
    if (hostHandle)
        platform::SetThreadPriority(hostHandle, host);
}

uint32_t GuestThread::SetAffinity(uint32_t mask)
{
    // The console's six hardware threads map onto host logical processors as an ideal-processor hint.
    // On the console the thread also moves to an allowed processor, which the game sees as the PCR's
    // processor number: it picks per-processor data with it (the render workers' matrix stacks at
    // 0x827A6240 + n * 8208). Workers left on one processor number would share one stack and race on its
    // depth until it holds float data, which crashes missions.
    uint32_t previous = affinity.exchange(mask);
    if (uint8_t* block = pcr.load(); block && mask && !(mask & (1u << block[kPcrProcessor])))
    {
        block[kPcrProcessor] = static_cast<uint8_t>(std::countr_zero(mask));
        static std::atomic<uint32_t> logged{ 0 };
        if (logged++ < 32)
            log::Info("Thread {} moves to processor {} (affinity {:02X})", threadId, std::countr_zero(mask), mask);
    }
    if (mask && hostHandle)
        platform::SetIdealProcessor(hostHandle, std::countr_zero(mask));
    return previous;
}

} // namespace ac6

namespace ac6::kernel {

KernelObject* CurrentThreadObject() { return GuestThread::Current(); }

} // namespace ac6::kernel
