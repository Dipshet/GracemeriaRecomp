#pragma once
// Guest threads: one host thread per guest thread.
//
// Each thread has a block in guest memory: PCR (r13 points at it), dynamic TLS slots, the XEX's static
// TLS data, and the stack. The KTHREAD lives in the thread's kernel object instead (physical heap), so
// it outlives the thread: handles and ObReferenceObjectByHandle pointers stay valid after exit, and its
// dispatcher header is signalled when the thread terminates.
// The block layout started from UnleashedRecomp (UnleashedRecomp/cpu/guest_thread.cpp, GPL-3.0).
// AC6's own r13 accesses set the offsets used here (listed in guest_thread.cpp).

#include "kernel/object.h"
#include "platform/platform.h"
#include "ppc.h"

#include <atomic>
#include <cstdint>
#include <vector>

namespace ac6 {

struct GuestThreadParams
{
    uint32_t function = 0;   // start routine
    uint32_t value = 0;      // its argument
    uint32_t flags = 0;      // ExCreateThread flags: bit 0 = create suspended, bits 24-31 = processor mask
    uint32_t stackSize = 0;  // 0 = module default
    uint32_t startup = 0;    // XAPI startup routine (called as startup(function, value)), 0 = none
    // Runtime-owned threads (interrupt sources, ...): a host body run with a full guest context, so it
    // can call guest callbacks (GuestCall). Replaces `function`.
    void (*hostEntry)(void*) = nullptr;
    void* hostArg = nullptr;
};

// Thrown by ExTerminateThread to unwind the guest thread back to its host entry.
struct GuestThreadExit
{
    uint32_t exitCode;
};

struct GuestThread : kernel::KernelObject
{
    kernel::XKThread kthread{};   // guest-visible: PCR+0x100 points here
    GuestThreadParams params;
    uint32_t threadId = 0;
    uint32_t tlsSlots = 0;        // guest address of this thread's dynamic TLS slot array
    platform::HostThread* hostHandle = nullptr;
    std::atomic<int32_t> suspendCount{ 0 };
    uint32_t exitCode = 0;
    int32_t basePriority = 0;
    std::atomic<uint32_t> affinity{ 0 };
    std::atomic<uint8_t*> pcr{ nullptr };   // this thread's PCR block while it runs
    PPCContext* context = nullptr;   // while running (diagnostics)

    explicit GuestThread(const GuestThreadParams& params);
    ~GuestThread() override;

    uint32_t DispatcherHeader() override;
    const char* TypeName() const override { return "thread"; }

    // Creates the guest thread on a new host thread, held: the caller's Resume() starts it (a second Resume() is
    // needed when flags bit 0 asks for a suspended thread).
    static GuestThread* Start(const GuestThreadParams& params);
    // Runs a guest function as a guest thread on the calling host thread (the entry point).
    static uint32_t RunOnCurrentThread(const GuestThreadParams& params);

    static GuestThread* Current();
    static GuestThread* FromKThread(uint32_t kthread);

    int32_t Resume();   // returns the previous suspend count
    void SetPriority(int32_t priority);
    uint32_t SetAffinity(uint32_t mask);

    uint32_t Run();     // body: runs on the host thread that owns this guest thread
    static uint32_t ProcessorFromFlags(uint32_t flags);

    // Calls fn for every running guest thread (under the registry lock).
    template<typename F> static void ForEachRunning(F&& fn);
    static void LockRegistry();
    static void UnlockRegistry();
    static const std::vector<GuestThread*>& Running();
};

template<typename F>
void GuestThread::ForEachRunning(F&& fn)
{
    LockRegistry();
    for (GuestThread* t : Running())
        fn(*t);
    UnlockRegistry();
}

} // namespace ac6
