// Kernel timers (NtCreateTimer / NtSetTimerEx / NtCancelTimer). A timer is a dispatcher header of type
// 8 (notification) or 9 (synchronization). One host scheduler thread keeps the armed timers ordered by
// due time and signals each through the dispatcher when it expires, re-arming periodic ones. Each arm
// gets a new generation number, so re-setting or cancelling a timer just makes its queued entry stale.
//
// The game's XAM notification pump (sub_82204BF8) is a periodic 10 ms timer. Without it that thread
// exits immediately and the main thread waits forever for it to start.
#include "kernel/dispatcher.h"
#include "kernel/function.h"
#include "kernel/object.h"
#include "log.h"
#include "memory.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ac6::kernel {

namespace {

using Clock = std::chrono::steady_clock;

struct Timer : KernelObject
{
    XDispatcherHeader header{};
    uint64_t generation = 0;     // guarded by the scheduler mutex
    bool armed = false;
    explicit Timer(uint8_t type) { InitializeHeader(header, type, 0); }
    uint32_t DispatcherHeader() override { return g_memory.MapVirtual(&header); }
    const char* TypeName() const override { return "timer"; }
};

struct Entry
{
    Clock::time_point due;
    Timer* timer;                // holds a reference while queued
    uint64_t generation;
    int64_t periodMs;
    bool operator>(const Entry& o) const { return due > o.due; }
};

class Scheduler
{
public:
    // Arms (or re-arms) a timer. Returns whether it was armed before.
    bool Set(Timer* timer, Clock::time_point due, int64_t periodMs)
    {
        std::lock_guard lock(m_mutex);
        EnsureThread();
        bool wasArmed = timer->armed;
        timer->armed = true;
        RetainObject(timer);
        m_queue.push({ due, timer, ++timer->generation, periodMs });
        m_wake.notify_one();
        return wasArmed;
    }

    bool Cancel(Timer* timer)
    {
        std::lock_guard lock(m_mutex);
        bool wasArmed = timer->armed;
        timer->armed = false;
        ++timer->generation;    // the queued entry, if any, becomes stale
        return wasArmed;
    }

private:
    void EnsureThread()
    {
        if (!m_started)
        {
            m_started = true;
            std::thread([this] { Run(); }).detach();
        }
    }

    void Run()
    {
        std::unique_lock lock(m_mutex);
        for (;;)
        {
            if (m_queue.empty())
            {
                m_wake.wait(lock);
                continue;
            }
            Entry next = m_queue.top();
            if (next.timer->generation != next.generation)
            {
                m_queue.pop();
                ReleaseLater(next.timer, lock);
                continue;
            }
            if (Clock::now() < next.due)
            {
                m_wake.wait_until(lock, next.due);
                continue;
            }
            m_queue.pop();
            uint32_t header = next.timer->DispatcherHeader();
            if (next.periodMs > 0)
            {
                // Periodic: the queue keeps the reference. Skip missed periods instead of bursting.
                Clock::time_point due = next.due + std::chrono::milliseconds(next.periodMs);
                if (due < Clock::now())
                    due = Clock::now() + std::chrono::milliseconds(next.periodMs);
                m_queue.push({ due, next.timer, next.generation, next.periodMs });
                lock.unlock();
                SignalObject(header);
                lock.lock();
            }
            else
            {
                next.timer->armed = false;
                lock.unlock();
                SignalObject(header);
                ReleaseObject(next.timer);
                lock.lock();
            }
        }
    }

    // Dropping the last reference destroys the object, which must not happen under the scheduler lock.
    void ReleaseLater(Timer* timer, std::unique_lock<std::mutex>& lock)
    {
        lock.unlock();
        ReleaseObject(timer);
        lock.lock();
    }

    std::mutex m_mutex;
    std::condition_variable m_wake;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> m_queue;
    bool m_started = false;
};

Scheduler& GetScheduler()
{
    static Scheduler* s = new Scheduler();   // never destroyed: its thread outlives static teardown
    return *s;
}

// NT due time: negative = relative 100 ns units, positive = absolute system time (100 ns since 1601).
Clock::time_point DueTime(int64_t value)
{
    int64_t relative100ns = value < 0 ? -value : value - static_cast<int64_t>(SystemTime100ns());
    if (relative100ns < 0)
        relative100ns = 0;
    return Clock::now() + std::chrono::nanoseconds(relative100ns * 100);
}

} // namespace

uint32_t NtCreateTimer(be<uint32_t>* handle, XObjectAttributes* attributes, uint32_t timerType)
{
    if (attributes && attributes->name)
        log::Warn("Named timer created unnamed (named objects not implemented)");
    // TIMER_TYPE: 0 = notification (stays signalled), 1 = synchronization (auto reset).
    auto* timer = CreateKernelObject<Timer>(timerType ? kObjTimerSynchronization : kObjTimerNotification);
    if (!timer)
        return X_STATUS_NO_MEMORY;
    *handle = CreateHandle(timer);
    return X_STATUS_SUCCESS;
}

uint32_t NtSetTimerEx(uint32_t handle, be<int64_t>* dueTime, uint32_t apcRoutine, uint32_t apcMode,
    uint32_t apcContext, uint32_t resume, int32_t periodMs, uint8_t* previousState)
{
    auto* timer = LookupHandleAs<Timer>(handle);
    if (!timer || !dueTime)
        return timer ? X_STATUS_INVALID_PARAMETER : X_STATUS_INVALID_HANDLE;
    if (apcRoutine)
        log::Warn("NtSetTimerEx: timer APC {:08X} not called (no APCs yet)", apcRoutine);

    // Setting a timer clears its signal state, as KeSetTimer does.
    ResetEvent(timer->DispatcherHeader());
    bool wasArmed = GetScheduler().Set(timer, DueTime(static_cast<int64_t>(*dueTime)), periodMs > 0 ? periodMs : 0);
    if (previousState)
        *previousState = wasArmed ? 1 : 0;
    return X_STATUS_SUCCESS;
}

uint32_t NtCancelTimer(uint32_t handle, uint8_t* currentState)
{
    auto* timer = LookupHandleAs<Timer>(handle);
    if (!timer)
        return X_STATUS_INVALID_HANDLE;
    bool wasArmed = GetScheduler().Cancel(timer);
    if (currentState)
        *currentState = wasArmed ? 1 : 0;
    return X_STATUS_SUCCESS;
}

} // namespace ac6::kernel

AC6_IMPORT(NtCreateTimer)
AC6_IMPORT(NtSetTimerEx)
AC6_IMPORT(NtCancelTimer)
