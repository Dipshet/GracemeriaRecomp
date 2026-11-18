// Thread imports: creation, termination, suspension, priority/affinity, sleeping, TLS.
#include "cpu/guest_thread.h"
#include "kernel/dispatcher.h"
#include "kernel/function.h"
#include "kernel/object.h"
#include "loader.h"
#include "log.h"
#include "memory.h"

#include <bitset>
#include <mutex>

namespace ac6::kernel {

namespace {

constexpr uint32_t kMaxTlsSlots = 64;
std::mutex s_tlsMutex;
std::bitset<kMaxTlsSlots> s_tlsUsed;

uint32_t TlsSlotCount()
{
    return std::min<uint32_t>(g_module.tls.slotCount ? g_module.tls.slotCount : kMaxTlsSlots, kMaxTlsSlots);
}

be<uint32_t>* TlsSlot(uint32_t index)
{
    GuestThread* thread = GuestThread::Current();
    if (!thread || !thread->tlsSlots || index >= TlsSlotCount())
        return nullptr;
    return g_memory.As<be<uint32_t>>(thread->tlsSlots) + index;
}

} // namespace

uint32_t ExCreateThread(be<uint32_t>* handle, uint32_t stackSize, be<uint32_t>* threadId, uint32_t startup,
    uint32_t startAddress, uint32_t startContext, uint32_t creationFlags)
{
    GuestThread* thread = GuestThread::Start({ startAddress, startContext, creationFlags, stackSize, startup });
    if (!thread)
        return X_STATUS_NO_MEMORY;
    log::Info("ExCreateThread: {:08X}({:08X}) stack {:#x} flags {:08X} -> thread {}", startAddress, startContext,
        stackSize, creationFlags, thread->threadId);
    if (handle)
        *handle = CreateHandle(thread);
    if (threadId)
        *threadId = thread->threadId;
    thread->Resume();   // launch (stays suspended if the guest asked for CREATE_SUSPENDED)
    return X_STATUS_SUCCESS;
}

void ExTerminateThread(uint32_t exitCode)
{
    throw GuestThreadExit{ exitCode };
}

uint32_t NtResumeThread(uint32_t handle, be<uint32_t>* suspendCount)
{
    auto* thread = LookupHandleAs<GuestThread>(handle);
    if (!thread)
        return X_STATUS_INVALID_HANDLE;
    int32_t previous = thread->Resume();
    if (suspendCount)
        *suspendCount = previous;
    return X_STATUS_SUCCESS;
}

uint32_t KeResumeThread(uint32_t kthread)
{
    auto* thread = GuestThread::FromKThread(kthread);
    return thread ? thread->Resume() : 0;
}

int32_t KeSetBasePriorityThread(uint32_t kthread, int32_t priority)
{
    auto* thread = GuestThread::FromKThread(kthread);
    if (!thread)
        return 0;
    int32_t previous = thread->basePriority;
    thread->SetPriority(priority);
    return previous;
}

int32_t KeQueryBasePriorityThread(uint32_t kthread)
{
    auto* thread = GuestThread::FromKThread(kthread);
    return thread ? thread->basePriority : 0;
}

uint32_t KeSetAffinityThread(uint32_t kthread, uint32_t affinity, be<uint32_t>* previousAffinity)
{
    auto* thread = GuestThread::FromKThread(kthread);
    if (!thread)
        return X_STATUS_INVALID_HANDLE;
    uint32_t previous = thread->SetAffinity(affinity);
    if (previousAffinity)
        *previousAffinity = previous;
    return X_STATUS_SUCCESS;
}

uint32_t KeDelayExecutionThread(uint32_t waitMode, uint32_t alertable, be<int64_t>* interval)
{
    DelayExecution(Timeout::FromGuest(interval));
    return X_STATUS_SUCCESS;
}

uint32_t KeGetCurrentProcessType()
{
    return 1; // title process
}

uint32_t KeTlsAlloc()
{
    std::lock_guard lock(s_tlsMutex);
    for (uint32_t i = 0; i < TlsSlotCount(); i++)
    {
        if (!s_tlsUsed[i])
        {
            s_tlsUsed[i] = true;
            return i;
        }
    }
    log::Error("KeTlsAlloc: all {} TLS slots in use", TlsSlotCount());
    return 0xFFFFFFFF;
}

uint32_t KeTlsFree(uint32_t index)
{
    std::lock_guard lock(s_tlsMutex);
    if (index >= TlsSlotCount() || !s_tlsUsed[index])
        return 0;
    s_tlsUsed[index] = false;
    return 1;
}

uint32_t KeTlsGetValue(uint32_t index)
{
    be<uint32_t>* slot = TlsSlot(index);
    return slot ? uint32_t(*slot) : 0;
}

uint32_t KeTlsSetValue(uint32_t index, uint32_t value)
{
    be<uint32_t>* slot = TlsSlot(index);
    if (!slot)
        return 0;
    *slot = value;
    return 1;
}

} // namespace ac6::kernel

AC6_IMPORT(ExCreateThread)
AC6_IMPORT(ExTerminateThread)
AC6_IMPORT(NtResumeThread)
AC6_IMPORT(KeResumeThread)
AC6_IMPORT(KeSetBasePriorityThread)
AC6_IMPORT(KeQueryBasePriorityThread)
AC6_IMPORT(KeSetAffinityThread)
AC6_IMPORT(KeDelayExecutionThread)
AC6_IMPORT(KeGetCurrentProcessType)
AC6_IMPORT(KeTlsAlloc)
AC6_IMPORT(KeTlsFree)
AC6_IMPORT(KeTlsGetValue)
AC6_IMPORT(KeTlsSetValue)
