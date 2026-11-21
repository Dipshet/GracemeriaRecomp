// Synchronisation imports: critical sections, events, semaphores, mutants, waits, spin locks, object
// references. Everything waitable goes through the dispatcher (kernel/dispatcher.h).
#include "kernel/dispatcher.h"
#include "kernel/function.h"
#include "kernel/object.h"
#include "log.h"
#include "memory.h"

#include <atomic>
#include <thread>
#include <vector>

#include <immintrin.h>

namespace ac6::kernel {

namespace {

// --- Handle-based dispatcher objects --------------------------------------------------------------

struct Event : KernelObject
{
    XDispatcherHeader header{};
    Event(uint8_t type, bool signalled) { InitializeHeader(header, type, signalled ? 1 : 0); }
    uint32_t DispatcherHeader() override { return g_memory.MapVirtual(&header); }
    const char* TypeName() const override { return "event"; }
};

struct Semaphore : KernelObject
{
    XKSemaphore semaphore{};
    Semaphore(int32_t count, int32_t limit)
    {
        InitializeHeader(semaphore.header, kObjSemaphore, count);
        semaphore.limit = limit;
    }
    uint32_t DispatcherHeader() override { return g_memory.MapVirtual(&semaphore.header); }
    const char* TypeName() const override { return "semaphore"; }
};

struct Mutant : KernelObject
{
    XKMutant mutant{};
    explicit Mutant(bool initialOwner)
    {
        InitializeHeader(mutant.header, kObjMutant, initialOwner ? 0 : 1);
        mutant.ownerThread = initialOwner ? CurrentKThread() : 0;
    }
    uint32_t DispatcherHeader() override { return g_memory.MapVirtual(&mutant.header); }
    const char* TypeName() const override { return "mutant"; }
};

void WarnIfNamed(const XObjectAttributes* attributes, const char* what)
{
    if (attributes && attributes->name)
    {
        auto* name = g_memory.As<XAnsiString>(attributes->name);
        std::string_view text(g_memory.As<char>(name->buffer), name->length);
        log::Warn("Named {} '{}' created unnamed (named objects not implemented)", what, text);
    }
}

uint32_t CreateHandleFor(KernelObject* obj, be<uint32_t>* handle)
{
    if (!obj)
        return X_STATUS_NO_MEMORY;
    *handle = CreateHandle(obj);
    return X_STATUS_SUCCESS;
}

// Header of a handle's object. False for an invalid handle. Objects without a header give 0, which waits treat as
// signalled.
bool HeaderFromHandle(uint32_t handle, uint32_t& header)
{
    KernelObject* obj = LookupHandle(handle);
    if (!obj)
        return false;
    header = obj->DispatcherHeader();
    return true;
}

// --- Spin locks and IRQL --------------------------------------------------------------------------

thread_local uint32_t t_irql = 0;

void SpinAcquire(be<uint32_t>* lock)
{
    // The lock word holds the owner's KTHREAD (non-zero) while held. Host threads can be preempted
    // while holding one, so back off to yielding instead of burning the core.
    std::atomic_ref<uint32_t> word(*reinterpret_cast<uint32_t*>(lock));
    const uint32_t owner = __builtin_bswap32(CurrentKThread() | 1);
    for (uint32_t spins = 0;; spins++)
    {
        uint32_t expected = 0;
        if (word.load(std::memory_order_relaxed) == 0 && word.compare_exchange_weak(expected, owner, std::memory_order_acquire))
            return;
        if (spins < 64)
            _mm_pause();
        else
            std::this_thread::yield();
    }
}

bool SpinTryAcquire(be<uint32_t>* lock)
{
    std::atomic_ref<uint32_t> word(*reinterpret_cast<uint32_t*>(lock));
    uint32_t expected = 0;
    return word.compare_exchange_strong(expected, __builtin_bswap32(CurrentKThread() | 1), std::memory_order_acquire);
}

void SpinRelease(be<uint32_t>* lock)
{
    std::atomic_ref<uint32_t>(*reinterpret_cast<uint32_t*>(lock)).store(0, std::memory_order_release);
}

// Atomic add on a big-endian guest int32. Returns the new value.
int32_t AtomicAddBE(be<int32_t>& value, int32_t delta)
{
    std::atomic_ref<uint32_t> word(reinterpret_cast<uint32_t&>(value));
    uint32_t old = word.load(std::memory_order_relaxed);
    for (;;)
    {
        int32_t next = static_cast<int32_t>(__builtin_bswap32(old)) + delta;
        if (word.compare_exchange_weak(old, __builtin_bswap32(static_cast<uint32_t>(next)), std::memory_order_acq_rel))
            return next;
    }
}

bool CompareExchangeBE(be<int32_t>& value, int32_t expected, int32_t desired)
{
    std::atomic_ref<uint32_t> word(reinterpret_cast<uint32_t&>(value));
    uint32_t e = __builtin_bswap32(static_cast<uint32_t>(expected));
    return word.compare_exchange_strong(e, __builtin_bswap32(static_cast<uint32_t>(desired)), std::memory_order_acq_rel);
}

constexpr uint32_t kCriticalSectionSpins = 1024;

} // namespace

// --- Critical sections (the classic NT algorithm over the embedded synchronisation event) -----------

void RtlInitializeCriticalSection(XRtlCriticalSection* cs)
{
    // Threads can already wait on a critical section the game hasn't initialised yet: entered while its memory was still
    // zero (a lock count of 0 reads as held), they queued on its event. The stream system's poll thread does this when
    // the game sets up the stream records more than its first 5 ms sleep after starting it: initialised over it, 
    // the lock read free and the thread waited for good, and so did the soundtrack it serves (the campaign stuck
    // on its loading screen). The lock goes to the first waiter instead, as if the initialisation had come first 
    // and that thread had found it free; the others stay queued behind it.
    const uint32_t header = g_memory.MapVirtual(&cs->header);
    const size_t waiters = WaiterCount(header);
    InitializeHeader(cs->header, kObjEventSynchronization, 0);
    cs->recursionCount = 0;
    cs->owningThread = 0;
    if (waiters == 0)
    {
        cs->lockCount = -1;
        return;
    }
    cs->lockCount = int32_t(waiters) - 1;   // the holder and the rest still waiting
    log::Info("RtlInitializeCriticalSection({:08X}): {} thread(s) already waiting on it, the lock handed to the first",
        header, waiters);
    SetEvent(header);
}

uint32_t RtlInitializeCriticalSectionAndSpinCount(XRtlCriticalSection* cs, uint32_t spinCount)
{
    RtlInitializeCriticalSection(cs);
    return X_STATUS_SUCCESS;
}

void RtlEnterCriticalSection(XRtlCriticalSection* cs)
{
    const uint32_t thread = CurrentKThread();
    if (cs->owningThread != thread)
    {
        // Brief spin for a free lock before queueing.
        for (uint32_t i = 0; i < kCriticalSectionSpins; i++)
        {
            if (cs->lockCount == -1 && CompareExchangeBE(cs->lockCount, -1, 0))
            {
                cs->owningThread = thread;
                cs->recursionCount = 1;
                return;
            }
            _mm_pause();
        }
    }
    if (AtomicAddBE(cs->lockCount, 1) != 0)
    {
        if (cs->owningThread == thread)
        {
            cs->recursionCount = cs->recursionCount + 1;
            return;
        }
        WaitForObject(g_memory.MapVirtual(&cs->header), Timeout::Infinite(), false);
    }
    cs->owningThread = thread;
    cs->recursionCount = 1;
}

uint32_t RtlTryEnterCriticalSection(XRtlCriticalSection* cs)
{
    const uint32_t thread = CurrentKThread();
    if (CompareExchangeBE(cs->lockCount, -1, 0))
    {
        cs->owningThread = thread;
        cs->recursionCount = 1;
        return 1;
    }
    if (cs->owningThread == thread)
    {
        AtomicAddBE(cs->lockCount, 1);
        cs->recursionCount = cs->recursionCount + 1;
        return 1;
    }
    return 0;
}

void RtlLeaveCriticalSection(XRtlCriticalSection* cs)
{
    int32_t recursion = cs->recursionCount - 1;
    cs->recursionCount = recursion;
    if (recursion != 0)
    {
        AtomicAddBE(cs->lockCount, -1);
        return;
    }
    cs->owningThread = 0;
    if (AtomicAddBE(cs->lockCount, -1) >= 0)
        SetEvent(g_memory.MapVirtual(&cs->header));   // hand over to one waiter
}

// --- Events -----------------------------------------------------------------------------------------

uint32_t NtCreateEvent(be<uint32_t>* handle, XObjectAttributes* attributes, uint32_t eventType, uint32_t initialState)
{
    WarnIfNamed(attributes, "event");
    // EVENT_TYPE: 0 = notification (manual reset), 1 = synchronization (auto reset): same as the header type.
    return CreateHandleFor(CreateKernelObject<Event>(eventType ? kObjEventSynchronization : kObjEventNotification, initialState != 0), handle);
}

uint32_t NtSetEvent(uint32_t handle, be<int32_t>* previousState)
{
    uint32_t header;
    if (!HeaderFromHandle(handle, header) || !header)
        return X_STATUS_INVALID_HANDLE;
    int32_t previous = SetEvent(header);
    if (previousState)
        *previousState = previous;
    return X_STATUS_SUCCESS;
}

uint32_t NtClearEvent(uint32_t handle)
{
    uint32_t header;
    if (!HeaderFromHandle(handle, header) || !header)
        return X_STATUS_INVALID_HANDLE;
    ResetEvent(header);
    return X_STATUS_SUCCESS;
}

uint32_t NtPulseEvent(uint32_t handle, be<int32_t>* previousState)
{
    uint32_t header;
    if (!HeaderFromHandle(handle, header) || !header)
        return X_STATUS_INVALID_HANDLE;
    int32_t previous = PulseEvent(header);
    if (previousState)
        *previousState = previous;
    return X_STATUS_SUCCESS;
}

int32_t KeSetEvent(uint32_t event, uint32_t increment, uint32_t wait)
{
    return SetEvent(event);
}

int32_t KeResetEvent(uint32_t event)
{
    return ResetEvent(event);
}

// --- Semaphores -------------------------------------------------------------------------------------

uint32_t NtCreateSemaphore(be<uint32_t>* handle, XObjectAttributes* attributes, int32_t initialCount, int32_t maximumCount)
{
    WarnIfNamed(attributes, "semaphore");
    return CreateHandleFor(CreateKernelObject<Semaphore>(initialCount, maximumCount), handle);
}

uint32_t NtReleaseSemaphore(uint32_t handle, int32_t releaseCount, be<int32_t>* previousCount)
{
    uint32_t header;
    if (!HeaderFromHandle(handle, header) || !header)
        return X_STATUS_INVALID_HANDLE;
    int32_t previous = 0;
    uint32_t status = ReleaseSemaphore(header, releaseCount, &previous);
    if (previousCount)
        *previousCount = previous;
    return status;
}

void KeInitializeSemaphore(XKSemaphore* semaphore, int32_t count, int32_t limit)
{
    InitializeHeader(semaphore->header, kObjSemaphore, count);
    semaphore->limit = limit;
}

int32_t KeReleaseSemaphore(uint32_t semaphore, uint32_t increment, int32_t adjustment, uint32_t wait)
{
    int32_t previous = 0;
    if (ReleaseSemaphore(semaphore, adjustment, &previous) != X_STATUS_SUCCESS)
        log::Warn("KeReleaseSemaphore({:08X}, {}) exceeds the limit", semaphore, adjustment);
    return previous;
}

// --- Mutants ----------------------------------------------------------------------------------------

uint32_t NtCreateMutant(be<uint32_t>* handle, XObjectAttributes* attributes, uint32_t initialOwner)
{
    WarnIfNamed(attributes, "mutant");
    return CreateHandleFor(CreateKernelObject<Mutant>(initialOwner != 0), handle);
}

uint32_t NtReleaseMutant(uint32_t handle, be<int32_t>* previousCount)
{
    uint32_t header;
    if (!HeaderFromHandle(handle, header) || !header)
        return X_STATUS_INVALID_HANDLE;
    int32_t previous = 0;
    uint32_t status = ReleaseMutant(header, false, &previous);
    if (previousCount)
        *previousCount = previous;
    return status;
}

// --- Waits ------------------------------------------------------------------------------------------

uint32_t KeWaitForSingleObject(uint32_t object, uint32_t waitReason, uint32_t waitMode, uint32_t alertable, be<int64_t>* timeout)
{
    return WaitForObject(object, Timeout::FromGuest(timeout), alertable != 0);
}

uint32_t KeWaitForMultipleObjects(uint32_t count, be<uint32_t>* objects, uint32_t waitType, uint32_t waitReason,
    uint32_t waitMode, uint32_t alertable, be<int64_t>* timeout, uint32_t waitBlocks)
{
    std::vector<uint32_t> headers(objects, objects + count);
    // WAIT_TYPE: 0 = WaitAll, 1 = WaitAny.
    return WaitForObjects(headers, waitType == 0, Timeout::FromGuest(timeout), alertable != 0);
}

uint32_t NtWaitForSingleObjectEx(uint32_t handle, uint32_t waitMode, uint32_t alertable, be<int64_t>* timeout)
{
    uint32_t header;
    if (!HeaderFromHandle(handle, header))
        return X_STATUS_INVALID_HANDLE;
    return WaitForObject(header, Timeout::FromGuest(timeout), alertable != 0);
}

uint32_t NtWaitForMultipleObjectsEx(uint32_t count, be<uint32_t>* handles, uint32_t waitType, uint32_t waitMode,
    uint32_t alertable, be<int64_t>* timeout)
{
    std::vector<uint32_t> headers(count);
    for (uint32_t i = 0; i < count; i++)
        if (!HeaderFromHandle(handles[i], headers[i]))
            return X_STATUS_INVALID_HANDLE;
    return WaitForObjects(headers, waitType == 0, Timeout::FromGuest(timeout), alertable != 0);
}

uint32_t NtSignalAndWaitForSingleObjectEx(uint32_t signalHandle, uint32_t waitHandle, uint32_t waitMode,
    uint32_t alertable, be<int64_t>* timeout)
{
    uint32_t signal, wait;
    if (!HeaderFromHandle(signalHandle, signal) || !HeaderFromHandle(waitHandle, wait) || !signal)
        return X_STATUS_INVALID_HANDLE;
    return SignalAndWait(signal, wait, Timeout::FromGuest(timeout), alertable != 0);
}

// --- Spin locks, IRQL, critical regions ------------------------------------------------------------

uint32_t KfAcquireSpinLock(be<uint32_t>* lock)
{
    uint32_t old = t_irql;
    t_irql = 2;
    SpinAcquire(lock);
    return old;
}

void KfReleaseSpinLock(be<uint32_t>* lock, uint32_t oldIrql)
{
    SpinRelease(lock);
    t_irql = oldIrql;
}

void KeAcquireSpinLockAtRaisedIrql(be<uint32_t>* lock) { SpinAcquire(lock); }
void KeReleaseSpinLockFromRaisedIrql(be<uint32_t>* lock) { SpinRelease(lock); }
uint32_t KeTryToAcquireSpinLockAtRaisedIrql(be<uint32_t>* lock) { return SpinTryAcquire(lock) ? 1 : 0; }

uint32_t KeRaiseIrqlToDpcLevel()
{
    uint32_t old = t_irql;
    t_irql = 2;
    return old;
}

void KfLowerIrql(uint32_t irql) { t_irql = irql; }

void KeEnterCriticalRegion() {}
void KeLeaveCriticalRegion() {}
void KeLockL2() {}
void KeUnlockL2() {}

// --- Objects and handles ----------------------------------------------------------------------------

uint32_t ObReferenceObjectByHandle(uint32_t handle, uint32_t objectType, be<uint32_t>* object)
{
    KernelObject* obj = LookupHandle(handle);
    if (!obj)
    {
        if (object)
            *object = 0;
        return X_STATUS_INVALID_HANDLE;
    }
    RetainObject(obj);
    RegisterObjectHeader(obj);
    if (object)
        *object = ObjectAddress(obj);
    return X_STATUS_SUCCESS;
}

void ObDereferenceObject(uint32_t object)
{
    if (KernelObject* obj = ObjectFromHeader(object))
        ReleaseObject(obj);
}

uint32_t NtClose(uint32_t handle)
{
    return CloseHandle(handle);
}

uint32_t NtDuplicateObject(uint32_t sourceHandle, be<uint32_t>* targetHandle, uint32_t options)
{
    uint32_t out = 0;
    uint32_t status = DuplicateHandle(sourceHandle, out);
    if (status == X_STATUS_SUCCESS)
    {
        if (targetHandle)
            *targetHandle = out;
        if (options & 1)   // DUPLICATE_CLOSE_SOURCE
            CloseHandle(sourceHandle);
    }
    return status;
}

} // namespace ac6::kernel

AC6_IMPORT(RtlInitializeCriticalSection)
AC6_IMPORT(RtlInitializeCriticalSectionAndSpinCount)
AC6_IMPORT(RtlEnterCriticalSection)
AC6_IMPORT(RtlTryEnterCriticalSection)
AC6_IMPORT(RtlLeaveCriticalSection)
AC6_IMPORT(NtCreateEvent)
AC6_IMPORT(NtSetEvent)
AC6_IMPORT(NtClearEvent)
AC6_IMPORT(NtPulseEvent)
AC6_IMPORT(KeSetEvent)
AC6_IMPORT(KeResetEvent)
AC6_IMPORT(NtCreateSemaphore)
AC6_IMPORT(NtReleaseSemaphore)
AC6_IMPORT(KeInitializeSemaphore)
AC6_IMPORT(KeReleaseSemaphore)
AC6_IMPORT(NtCreateMutant)
AC6_IMPORT(NtReleaseMutant)
AC6_IMPORT(KeWaitForSingleObject)
AC6_IMPORT(KeWaitForMultipleObjects)
AC6_IMPORT(NtWaitForSingleObjectEx)
AC6_IMPORT(NtWaitForMultipleObjectsEx)
AC6_IMPORT(NtSignalAndWaitForSingleObjectEx)
AC6_IMPORT(KfAcquireSpinLock)
AC6_IMPORT(KfReleaseSpinLock)
AC6_IMPORT(KeAcquireSpinLockAtRaisedIrql)
AC6_IMPORT(KeReleaseSpinLockFromRaisedIrql)
AC6_IMPORT(KeTryToAcquireSpinLockAtRaisedIrql)
AC6_IMPORT(KeRaiseIrqlToDpcLevel)
AC6_IMPORT(KfLowerIrql)
AC6_IMPORT(KeEnterCriticalRegion)
AC6_IMPORT(KeLeaveCriticalRegion)
AC6_IMPORT(KeLockL2)
AC6_IMPORT(KeUnlockL2)
AC6_IMPORT(ObReferenceObjectByHandle)
AC6_IMPORT(ObDereferenceObject)
AC6_IMPORT(NtClose)
AC6_IMPORT(NtDuplicateObject)
