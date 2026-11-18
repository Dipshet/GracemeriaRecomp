// There is no console GPU, so the Xbox D3D library's waits on GPU progress (ring-buffer space, fences) would spin
// forever. They all step through one helper, sub_821E6AC8 ("keep waiting?"), which returns 0 to give up. The D3D
// library still runs (the renderer hooks its calls, not CreateDevice), so the helper is hooked: it always gives up,
// after draining the renderer's queued calls when the wait is for GPU progress (KeepWaitingForGpu below).
#include "gpu/renderer.h"
#include "kernel/function.h"
#include "log.h"
#include "ppc.h"

#include <atomic>

namespace ac6::gpu {

namespace {
std::atomic<uint64_t> s_abandonedWaits{ 0 };
// Inside the ring-space wait (sub_821E51F8): the ring's packets aren't the renderer's input.
thread_local bool t_ringWait = false;
// Inside the swap (sub_821F03B0): its waits limit how many frames the GPU may fall behind (one wait a frame, looping
// while 15 swaps are outstanding). The present itself keeps the renderer at most a frame behind.
thread_local bool t_swapWait = false;
}

void SetRingWait(bool waiting) { t_ringWait = waiting; }
void SetSwapWait(bool waiting) { t_swapWait = waiting; }

// The renderer's queued calls are what the GPU would still be working on: a wait for GPU
// progress (a fence, a resource the GPU may still read) drains them before it gives up. The ring-space waits,
// ~100 a frame as the ring's segments are reused, wait on the ring alone, and the swap's throttle on frames
// in flight doesn't need it.
uint32_t KeepWaitingForGpu(uint32_t waitState)
{
    if (!t_ringWait && !t_swapWait)
        RendererWaitIdle();
    if (s_abandonedWaits.fetch_add(1) == 0)
        log::Warn("No GPU: abandoning D3D GPU waits (sub_821E6AC8)");
    return 0;
}

} // namespace ac6::gpu

AC6_HOOK(sub_821E6AC8, ac6::gpu::KeepWaitingForGpu)

extern "C" PPC_FUNC(__imp__sub_821E51F8);   // wait until the GPU has read the ring up to an address (segment reuse)
PPC_FUNC(sub_821E51F8)
{
    ac6::gpu::SetRingWait(true);
    __imp__sub_821E51F8(ctx, base);
    ac6::gpu::SetRingWait(false);
}
