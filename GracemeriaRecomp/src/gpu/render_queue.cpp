#include "gpu/render_queue.h"

#include "host_cpu.h"
#include "log.h"

#include <chrono>
#include <cstdlib>
#include <immintrin.h>
#include <new>
#include <vector>

namespace ac6::gpu {

RenderQueue* g_renderQueue = nullptr;

RenderQueue::RenderQueue()
{
    // AC6_RENDER_WORKER=0 runs every call on the calling thread, when it is committed.
    if (const char* v = std::getenv("AC6_RENDER_WORKER"); v && *v == '0')
        threaded = false;
    log::Info("Renderer calls: {}", threaded ? "queued to a worker thread" : "on the calling thread");
    if (!threaded)
        return;
    // Events (auto-reset), as the deferred command list: a Set before the wait is never lost.
    worker = std::thread([this] { Run(); });
    workerId = worker.get_id();
    platform::SetThreadPriority(reinterpret_cast<void*>(worker.native_handle()), platform::kPriorityAboveNormal);
    worker.detach();   // lives as long as the process (shutdown is std::_Exit)
}

void* RenderQueue::Begin(uint32_t bytes, Invoke invoke)
{
    producer.lock();
    if (!threaded)
    {
        thread_local std::vector<uint8_t> scratch;
        scratch.resize(bytes + 16);
        pending = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(scratch.data()) + 15) & ~uintptr_t(15));
        pendingInvoke = invoke;
        return pending;
    }
    const uint64_t need = (sizeof(Header) + bytes + 15) & ~uint64_t(15);
    if (need > kChunkBytes)
    {
        log::Error("Renderer call of {} bytes exceeds the queue's chunk", bytes);
        std::abort();
    }
    uint64_t offset = writePos % kChunkBytes;
    if (offset + need > kChunkBytes)
    {
        uint8_t* chunk = chunks[(writePos / kChunkBytes) % kChunks].load(std::memory_order_relaxed);
        *reinterpret_cast<Header*>(chunk + offset) = { nullptr, uint32_t(kChunkBytes - offset) };
        writePos += kChunkBytes - offset;
        offset = 0;
    }
    if (offset == 0)
    {
        // Entering a chunk: the worker must be past it (a frame is ~2 MB of calls, this only waits when the
        // worker falls a whole ring behind).
        committed.store(writePos, std::memory_order_seq_cst);
        while (writePos - done.load(std::memory_order_acquire) > (kChunks - 1) * kChunkBytes)
        {
            if (workerSleeping.load(std::memory_order_seq_cst) && workerSleeping.exchange(0, std::memory_order_seq_cst))
                workEvent.Set();
            _mm_pause();
        }
    }
    const uint64_t index = (writePos / kChunkBytes) % kChunks;
    uint8_t* chunk = chunks[index].load(std::memory_order_relaxed);
    if (!chunk)
    {
        chunk = static_cast<uint8_t*>(::operator new(kChunkBytes, std::align_val_t(64)));
        chunks[index].store(chunk, std::memory_order_release);
    }
    *reinterpret_cast<Header*>(chunk + offset) = { invoke, uint32_t(need) };
    writePos += need;
    return chunk + offset + sizeof(Header);
}

void RenderQueue::Commit()
{
    if (!threaded)
    {
        pendingInvoke(pending);
        producer.unlock();
        return;
    }
    committed.store(writePos, std::memory_order_seq_cst);
    // The flag is cleared by whoever wakes the worker: one wake-up per sleep, not per call.
    if (workerSleeping.load(std::memory_order_seq_cst) && workerSleeping.exchange(0, std::memory_order_seq_cst))
        workEvent.Set();
    producer.unlock();
}

void RenderQueue::WaitIdle()
{
    if (!threaded)
        return;
    const uint64_t target = [this] {
        std::lock_guard lock(producer);
        return writePos;
    }();
    // No spinning first: it gives no measurable gain and costs laptops.
    if (done.load(std::memory_order_acquire) >= target)
        return;
    idleWaiters.fetch_add(1, std::memory_order_seq_cst);
    {
        std::unique_lock lock(idleMutex);
        idleCondition.wait(lock, [&] { return done.load(std::memory_order_seq_cst) >= target; });
    }
    idleWaiters.fetch_sub(1, std::memory_order_relaxed);
}

void RenderQueue::Run()
{
    uint64_t readPos = 0;
    for (;;)
    {
        uint64_t end = committed.load(std::memory_order_acquire);
        if (readPos == end)
        {
            // Nothing to run: sleep until the producers commit (no spinning first).
            if (end == readPos)
            {
                workerSleeping.store(1, std::memory_order_seq_cst);
                if (committed.load(std::memory_order_seq_cst) == readPos)
                    workEvent.Wait();
                workerSleeping.store(0, std::memory_order_seq_cst);
                continue;
            }
        }
        while (readPos != end)
        {
            uint8_t* chunk = chunks[(readPos / kChunkBytes) % kChunks].load(std::memory_order_acquire);
            auto* header = reinterpret_cast<Header*>(chunk + readPos % kChunkBytes);
            if (header->invoke)
                header->invoke(reinterpret_cast<uint8_t*>(header) + sizeof(Header));
            readPos += header->size;
            // Published per call: WaitIdle and a producer a ring behind wait on it.
            done.store(readPos, std::memory_order_seq_cst);
            // (Taking the lock orders this wake after a waiter's check of `done`: none is lost.)
            if (idleWaiters.load(std::memory_order_seq_cst))
            {
                { std::lock_guard lock(idleMutex); }
                idleCondition.notify_all();
            }
        }
    }
}

} // namespace ac6::gpu
