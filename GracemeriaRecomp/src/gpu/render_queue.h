#pragma once
// The renderer's command queue. The game's render thread is the frame's critical path, and
// translating its draws would be half of that thread's time. The renderer's entry points (draws, clears, resolves,
// tiling, present) only take a snapshot of the guest device state the call reads and queue it. A worker
// thread runs the calls in order against the snapshot, overlapped with the game issuing the frame's next draws.
// Guest data the GPU would read (vertices, indices, textures) is still read by the worker, as the console GPU
// reads it after the draw call. The queue is drained wherever the game waits for the GPU (present included),
// so no read happens after the game was told the GPU is done.
//
// Each call is a payload in an arena (a ring of chunks) with the function that runs it. Producers may be on
// several threads (the game presents on its main thread and draws on its render thread). They are serialised.

#include "platform/platform.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>

namespace ac6::gpu {

class RenderQueue
{
public:
    using Invoke = void (*)(void* payload);

    RenderQueue();

    // A call of `bytes` of payload, run by `invoke`. The caller fills the payload and calls Commit. Between
    // the two the queue is locked: Begin / Commit pairs from different threads don't interleave.
    void* Begin(uint32_t bytes, Invoke invoke);
    void Commit();
    // Waits until every committed call has run. Not from the worker.
    void WaitIdle();
    bool OnWorker() const { return std::this_thread::get_id() == workerId; }
    bool Threaded() const { return threaded; }

private:
    struct alignas(16) Header
    {
        Invoke invoke;    // null: a skipped block at a chunk's end
        uint32_t size;    // bytes to the next header
    };
    static constexpr uint64_t kChunkBytes = 1u << 20;
    static constexpr uint32_t kChunks = 8;    // a ring: positions only grow. Small enough to stay in cache

    void Run();

    std::mutex producer;
    bool threaded = true;
    std::atomic<uint8_t*> chunks[kChunks] = {};
    uint64_t writePos = 0;                 // producer: next header
    void* pending = nullptr;               // synchronous mode: the payload being filled
    Invoke pendingInvoke = nullptr;
    alignas(64) std::atomic<uint64_t> committed{ 0 };   // bytes the worker may run
    std::atomic<uint32_t> workerSleeping{ 0 };
    alignas(64) std::atomic<uint64_t> done{ 0 };        // bytes the worker has run
    // WaitIdle's waiters, any number of threads at once (the present, the D3D GPU waits): each sleeps on the condition
    // until `done` reaches its target. The worker wakes them all when it has any.
    std::atomic<uint32_t> idleWaiters{ 0 };
    std::mutex idleMutex;
    std::condition_variable idleCondition;
    platform::Event workEvent;
    std::thread::id workerId;
    std::thread worker;
};

extern RenderQueue* g_renderQueue;


} // namespace ac6::gpu
