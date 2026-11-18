#include "kernel/dispatcher.h"

#include "host_cpu.h"

#include "log.h"
#include "memory.h"
#include "platform/platform.h"
#include "ppc.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <immintrin.h>
#include <format>
#include <string>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ac6::kernel {

namespace {

using Clock = std::chrono::steady_clock;

struct Waiter
{
    std::span<const uint32_t> objects;
    bool waitAll = false;
    uint32_t thread = 0;
    // Index of the satisfying object (wait-any) or 0 (wait-all), written under s_lock, read by the waiter
    // without it (it sleeps on this word: WaitOnAddress).
    std::atomic<int32_t> satisfied{ -1 };
    bool sleeping = false;    // (under s_lock) set once the waiter is about to sleep: wakers then signal it
};

std::mutex s_lock;
std::unordered_map<uint32_t, std::vector<Waiter*>> s_waitLists;   // header -> FIFO of waiters

// Waiters this thread satisfied while holding s_lock, woken once it is released. A waiter woken under the lock
// would block on the lock at once and sleep again until the waker let go (two kernel wake-ups per hand-over, and
// the game's threads hand work over thousands of times a second). The woken waiter doesn't take the lock again:
// its waker settled everything under it. (The waiter may be gone by the time it is woken: platform::WakeOne
// doesn't touch the address, and a wait word reused meanwhile only gets a spurious wake, which its waiter checks
// for.)
thread_local std::vector<std::atomic<int32_t>*> t_wakes;

void FlushWakes()
{
    for (std::atomic<int32_t>* word : t_wakes)
        platform::WakeOne(word);
    t_wakes.clear();
}

// s_lock, held for the scope. The threads satisfied meanwhile are woken after it is released.
struct Locked
{
    std::unique_lock<std::mutex> lock{ s_lock };
    ~Locked()
    {
        if (lock.owns_lock())
            lock.unlock();
        FlushWakes();
    }
};

XDispatcherHeader& Header(uint32_t address) { return *g_memory.As<XDispatcherHeader>(address); }

bool IsSignaled(uint32_t address, uint32_t thread)
{
    if (address == 0)
        return true;
    auto& h = Header(address);
    switch (h.type)
    {
    case kObjMutant:
        return h.signalState > 0 || reinterpret_cast<XKMutant&>(h).ownerThread == thread;
    case kObjEventNotification:
    case kObjEventSynchronization:
    case kObjSemaphore:
    case kObjThread:
    case kObjTimerNotification:
    case kObjTimerSynchronization:
        return h.signalState > 0;
    default:
    {
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            log::Warn("Wait on dispatcher header {:08X} with unknown type {}", address, h.type);
        }
        return h.signalState > 0;
    }
    }
}

// The side effect of a satisfied wait.
void Acquire(uint32_t address, uint32_t thread)
{
    if (address == 0)
        return;
    auto& h = Header(address);
    switch (h.type)
    {
    case kObjEventSynchronization:
    case kObjTimerSynchronization:
        h.signalState = 0;
        break;
    case kObjSemaphore:
        h.signalState = h.signalState - 1;
        break;
    case kObjMutant:
    {
        auto& m = reinterpret_cast<XKMutant&>(h);
        m.header.signalState = m.header.signalState - 1;
        if (m.header.signalState == 0)
        {
            m.ownerThread = thread;
            m.abandoned = 0;
        }
        break;
    }
    default:
        break;
    }
}

// Acquires the objects if the wait can be satisfied and gives the satisfying index (0 for wait-all). The caller
// publishes it in w.satisfied.
bool TrySatisfy(Waiter& w, int32_t& index)
{
    if (w.waitAll)
    {
        for (uint32_t o : w.objects)
            if (!IsSignaled(o, w.thread))
                return false;
        for (uint32_t o : w.objects)
            Acquire(o, w.thread);
        index = 0;
        return true;
    }
    for (size_t i = 0; i < w.objects.size(); i++)
    {
        if (IsSignaled(w.objects[i], w.thread))
        {
            Acquire(w.objects[i], w.thread);
            index = static_cast<int32_t>(i);
            return true;
        }
    }
    return false;
}

void Enqueue(Waiter& w)
{
    for (uint32_t o : w.objects)
        if (o)
            s_waitLists[o].push_back(&w);
}

void Dequeue(Waiter& w)
{
    for (uint32_t o : w.objects)
    {
        auto it = s_waitLists.find(o);
        if (it == s_waitLists.end())
            continue;
        std::erase(it->second, &w);
        if (it->second.empty())
            s_waitLists.erase(it);
    }
}

// Called with the lock held after an object's state changed: satisfy queued waits in FIFO order.
void WakeWaiters(uint32_t address)
{
    auto it = s_waitLists.find(address);
    if (it == s_waitLists.end())
        return;
    std::vector<Waiter*> queue = it->second;   // satisfying a waiter edits the lists
    for (Waiter* w : queue)
    {
        if (!IsSignaled(address, w->thread))
        {
            if (Header(address).type != kObjMutant)
                break;   // nothing left to hand out
            continue;    // a mutant is still "signalled" for its owner only
        }
        int32_t index = -1;
        if (TrySatisfy(*w, index))
        {
            // Off the lists first, `satisfied` published last: a waiter that isn't asleep returns as soon as it sees
            // it (without the lock) and its Waiter, on its stack, is gone. Dequeuing after publishing would read a
            // dead Waiter.
            Dequeue(*w);
            const bool sleeping = w->sleeping;
            std::atomic<int32_t>* word = &w->satisfied;
            word->store(index, std::memory_order_release);
            if (sleeping)
                t_wakes.push_back(word);
        }
    }
}

Clock::time_point Deadline(const Timeout& t)
{
    int64_t relative = t.value < 0 ? -t.value : std::max<int64_t>(0, t.value - static_cast<int64_t>(SystemTime100ns()));
    return Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::nanoseconds(relative * 100));
}

} // namespace

uint64_t SystemTime100ns()
{
    return platform::SystemTime100ns();
}

uint32_t CurrentKThread()
{
    PPCContext* ctx = GetPPCContext();
    return ctx ? uint32_t(*g_memory.As<be<uint32_t>>(ctx->r13.u32 + 0x100)) : 0u;
}

void InitializeHeader(XDispatcherHeader& header, uint8_t type, int32_t signalState)
{
    Locked lock;
    header.type = type;
    header.absolute = 0;
    header.size = 0;
    header.inserted = 0;
    header.signalState = signalState;
    // An empty list points at itself, as the console's initialisers leave it.
    uint32_t self = g_memory.MapVirtual(&header.waitListFlink);
    header.waitListFlink = self;
    header.waitListBlink = self;
}

namespace {

// Waits with s_lock already held. Returns with it held, except when the wait is satisfied after sleeping (then
// nothing is left to do under it).
uint32_t WaitLocked(std::unique_lock<std::mutex>& lock, std::span<const uint32_t> headers, bool waitAll, Timeout timeout)
{
    Waiter w;
    w.objects = headers;
    w.waitAll = waitAll;
    w.thread = CurrentKThread();

    if (int32_t index = -1; TrySatisfy(w, index))
        return X_STATUS_WAIT_0 + index;
    if (!timeout.infinite && timeout.value == 0)
        return X_STATUS_TIMEOUT;

    Enqueue(w);
    const Clock::time_point deadline = timeout.infinite ? Clock::time_point::max() : Deadline(timeout);
    // Straight to sleep: no spinning first (spinning gains nothing measurable, and on laptops it spends the CPU's
    // power budget).
    w.sleeping = true;
    lock.unlock();
    FlushWakes();   // wakes the threads satisfied above (a SignalAndWait's partner) before this thread sleeps
    for (;;)
    {
        const int32_t satisfied = w.satisfied.load(std::memory_order_acquire);
        if (satisfied >= 0)
            return X_STATUS_WAIT_0 + satisfied;
        uint32_t ms = platform::kInfinite;
        if (!timeout.infinite)
        {
            const auto left = deadline - Clock::now();
            if (left <= Clock::duration::zero())
                break;
            ms = uint32_t(std::min<int64_t>(std::chrono::ceil<std::chrono::milliseconds>(left).count(), platform::kInfinite - 1));
        }
        platform::WaitOnAddress(&w.satisfied, uint32_t(-1), ms);   // while unsatisfied
    }
    // Timed out, unless a waker got there first.
    lock.lock();
    if (w.satisfied.load(std::memory_order_relaxed) >= 0)
        return X_STATUS_WAIT_0 + w.satisfied.load(std::memory_order_relaxed);
    Dequeue(w);
    return X_STATUS_TIMEOUT;
}

// Signal operations with s_lock held.
int32_t SetEventLocked(uint32_t address)
{
    auto& h = Header(address);
    int32_t previous = h.signalState;
    h.signalState = 1;
    WakeWaiters(address);
    return previous;
}

uint32_t ReleaseSemaphoreLocked(uint32_t address, int32_t count, int32_t* previous)
{
    auto& s = *g_memory.As<XKSemaphore>(address);
    int32_t current = s.header.signalState;
    if (previous)
        *previous = current;
    if (count <= 0 || int64_t(current) + count > int32_t(s.limit))
        return X_STATUS_SEMAPHORE_LIMIT_EXCEEDED;
    s.header.signalState = current + count;
    WakeWaiters(address);
    return X_STATUS_SUCCESS;
}

uint32_t ReleaseMutantLocked(uint32_t address, bool abandoned, int32_t* previous)
{
    auto& m = *g_memory.As<XKMutant>(address);
    int32_t current = m.header.signalState;
    if (previous)
        *previous = current;
    if (!abandoned && m.ownerThread != CurrentKThread())
        return X_STATUS_MUTANT_NOT_OWNED;
    m.header.signalState = abandoned ? 1 : current + 1;
    if (abandoned)
        m.abandoned = 1;
    if (m.header.signalState == 1)
    {
        m.ownerThread = 0;
        WakeWaiters(address);
    }
    return X_STATUS_SUCCESS;
}

} // namespace

uint32_t WaitForObjects(std::span<const uint32_t> headers, bool waitAll, Timeout timeout, bool alertable)
{
    Locked locked;
    return WaitLocked(locked.lock, headers, waitAll, timeout);
}

uint32_t SignalAndWait(uint32_t signalHeader, uint32_t waitHeader, Timeout timeout, bool alertable)
{
    Locked locked;
    uint32_t status = X_STATUS_SUCCESS;
    switch (Header(signalHeader).type)
    {
    case kObjEventNotification:
    case kObjEventSynchronization:
        SetEventLocked(signalHeader);
        break;
    case kObjSemaphore:
        status = ReleaseSemaphoreLocked(signalHeader, 1, nullptr);
        break;
    case kObjMutant:
        status = ReleaseMutantLocked(signalHeader, false, nullptr);
        break;
    default:
        return X_STATUS_OBJECT_TYPE_MISMATCH;
    }
    if (status != X_STATUS_SUCCESS)
        return status;
    return WaitLocked(locked.lock, std::span(&waitHeader, 1), false, timeout);
}

uint32_t WaitForObject(uint32_t header, Timeout timeout, bool alertable)
{
    return WaitForObjects(std::span(&header, 1), false, timeout, alertable);
}

int32_t SetEvent(uint32_t address)
{
    Locked lock;
    return SetEventLocked(address);
}

int32_t ResetEvent(uint32_t address)
{
    Locked lock;
    auto& h = Header(address);
    int32_t previous = h.signalState;
    h.signalState = 0;
    return previous;
}

int32_t PulseEvent(uint32_t address)
{
    Locked lock;
    auto& h = Header(address);
    int32_t previous = h.signalState;
    h.signalState = 1;
    WakeWaiters(address);
    h.signalState = 0;
    return previous;
}

uint32_t ReleaseSemaphore(uint32_t address, int32_t count, int32_t* previous)
{
    Locked lock;
    return ReleaseSemaphoreLocked(address, count, previous);
}

uint32_t ReleaseMutant(uint32_t address, bool abandoned, int32_t* previous)
{
    Locked lock;
    return ReleaseMutantLocked(address, abandoned, previous);
}

void SignalObject(uint32_t address)
{
    Locked lock;
    Header(address).signalState = 1;
    WakeWaiters(address);
}

void DumpWaits()
{
    // try_lock: a dump from a crash handler must not deadlock on a lock the crashing thread holds.
    std::unique_lock lock(s_lock, std::try_to_lock);
    if (!lock.owns_lock())
    {
        log::Info("--- waits: dispatcher lock busy ---");
        return;
    }
    log::Info("--- waits ---");
    std::vector<Waiter*> seen;
    for (auto& [header, waiters] : s_waitLists)
    {
        for (Waiter* w : waiters)
        {
            if (std::find(seen.begin(), seen.end(), w) != seen.end())
                continue;
            seen.push_back(w);
            std::string objects;
            for (uint32_t o : w->objects)
            {
                if (!o)
                    continue;
                auto& h = Header(o);
                objects += std::format(" {:08X}[type {} signal {}", o, h.type, int32_t(h.signalState));
                if (h.type == kObjMutant)
                    objects += std::format(" owner {:08X}", uint32_t(reinterpret_cast<XKMutant&>(h).ownerThread));
                objects += "]";
            }
            log::Info("  kthread {:08X} waits {} on{}", w->thread, w->waitAll ? "all" : "any", objects);
        }
    }
}

void DelayExecution(Timeout interval)
{
    if (interval.infinite)
    {
        for (;;)
            std::this_thread::sleep_for(std::chrono::hours(1));
    }
    if (interval.value == 0)
    {
        platform::YieldThread();
        return;
    }
    // Relative intervals (negative, the NT convention) on a high-resolution timer, absolute ones by the clock.
    if (interval.value > 0 || !platform::SleepPrecise(uint64_t(-interval.value)))
        std::this_thread::sleep_until(Deadline(interval));
}

} // namespace ac6::kernel
