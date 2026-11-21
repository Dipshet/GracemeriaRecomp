#pragma once
// The wait/signal dispatcher. Guest dispatcher headers (in guest memory) are the single source of
// truth for object state, exactly as on the console: events, semaphores, mutants, timers and threads
// are all "a header with a type and a signal state". All state changes and waits happen under one
// global lock. Every blocked thread has its own waiter record, queued FIFO on each object it waits on.
// Whoever changes an object's state satisfies queued waits directly (performing their acquire side
// effects) and wakes only those threads. That gives correct pulse semantics, wait-any/wait-all,
// fairness, and no stolen wake-ups.

#include "kernel/xtypes.h"

#include <cstdint>
#include <span>

namespace ac6::kernel {

// Timeouts use the NT convention: a pointer to a 64-bit count of 100 ns units, negative = relative,
// positive = absolute system time, null = infinite, 0 = poll.
struct Timeout
{
    bool infinite = true;
    int64_t value = 0;

    static Timeout Infinite() { return {}; }
    static Timeout FromGuest(const be<int64_t>* t) { return t ? Timeout{ false, static_cast<int64_t>(*t) } : Timeout{}; }
    static Timeout FromMilliseconds(uint32_t ms) { return ms == X_INFINITE ? Timeout{} : Timeout{ false, -int64_t(ms) * 10000 }; }
};

// Initialises a header the way the game's inline initialisers do.
void InitializeHeader(XDispatcherHeader& header, uint8_t type, int32_t signalState);

// Waits on dispatcher headers (guest addresses). Returns X_STATUS_WAIT_0 + index (wait-any),
// X_STATUS_WAIT_0 (wait-all) or X_STATUS_TIMEOUT. A header address of 0 counts as always signalled.
uint32_t WaitForObjects(std::span<const uint32_t> headers, bool waitAll, Timeout timeout, bool alertable);
uint32_t WaitForObject(uint32_t header, Timeout timeout, bool alertable);

// Atomically signals one object (event: set, semaphore: release 1, mutant: release) and waits on
// another, as NtSignalAndWaitForSingleObjectEx does: nobody can run between the signal and the wait
// being queued, which condition-variable code built on it relies on.
uint32_t SignalAndWait(uint32_t signalHeader, uint32_t waitHeader, Timeout timeout, bool alertable);

// State changes. Each returns the previous signal state (or an NTSTATUS for the release functions).
int32_t SetEvent(uint32_t header);
int32_t ResetEvent(uint32_t header);
int32_t PulseEvent(uint32_t header);
uint32_t ReleaseSemaphore(uint32_t header, int32_t count, int32_t* previous);
uint32_t ReleaseMutant(uint32_t header, bool abandoned, int32_t* previous);
// Sets the signal state to 1 and satisfies waiters (thread termination, timer expiry).
void SignalObject(uint32_t header);

// Sleeps the calling thread (KeDelayExecutionThread semantics, 0 yields).
void DelayExecution(Timeout interval);

// The KTHREAD guest address of the calling guest thread (mutant ownership, critical sections).
uint32_t CurrentKThread();

// Diagnostics: logs every blocked wait and the state of the objects involved (thread dump).
void DumpWaits();
// How many threads wait on that object now.
size_t WaiterCount(uint32_t header);

// Host clock for absolute timeouts: current system time in 100 ns units since 1601 (FILETIME).
uint64_t SystemTime100ns();

} // namespace ac6::kernel
