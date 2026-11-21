#include "gpu/deferred_commands.h"
#include "gpu/render_queue.h"

#include "log.h"

#include <chrono>
#include <cstring>
#include <immintrin.h>

namespace ac6::gpu {

using namespace plume;

DeferredCommandList::DeferredCommandList()
{
    // Events (auto-reset, workEvent / finishEvent), not WaitOnAddress: a worker in WaitOnAddress whose flag a
    // waker has cleared could miss its wake-up and sleep for good (a frame would stall every few minutes of a mission).
    // A Set before the wait leaves the event set, so no wake-up can be lost. A stale one only costs a spurious pass.
    worker = std::thread([this] { Run(); });
    worker.detach();   // lives as long as the process (shutdown is std::_Exit)
}

DeferredCommandList::~DeferredCommandList() = default;

void DeferredCommandList::Start(RenderCommandList* list)
{
    // The worker is idle (Finish returned): it picks the target up with the first call it replays.
    target = list;
    lastFramebuffer = lastGraphicsLayout = kUnknown;
}

void* DeferredCommandList::Allocate(uint64_t bytes, Invoke invoke)
{
    const uint64_t need = (sizeof(Header) + bytes + 15) & ~uint64_t(15);
    if (need > kChunkBytes)
    {
        log::Error("Deferred command of {} bytes exceeds the arena chunk", bytes);
        std::abort();
    }
    uint64_t offset = writePos % kChunkBytes;
    if (offset + need > kChunkBytes)
    {
        // The rest of this chunk is skipped: a data block to its end.
        uint8_t* chunk = chunks[(writePos / kChunkBytes) % kChunks].load(std::memory_order_relaxed);
        *reinterpret_cast<Header*>(chunk + offset) = { nullptr, uint32_t(kChunkBytes - offset) };
        writePos += kChunkBytes - offset;
        offset = 0;
    }
    if (offset == 0)
    {
        // Entering a chunk: it must be replayed already (a frame is ~0.5 MB of calls, this only waits if the
        // worker falls a whole ring behind).
        while (writePos - replayed.load(std::memory_order_acquire) > (kChunks - 1) * kChunkBytes)
        {
            Publish();
            _mm_pause();
        }
    }
    const uint64_t index = (writePos / kChunkBytes) % kChunks;
    uint8_t* chunk = chunks[index].load(std::memory_order_relaxed);
    if (!chunk)
    {
        chunk = new uint8_t[kChunkBytes];
        chunks[index].store(chunk, std::memory_order_release);
    }
    *reinterpret_cast<Header*>(chunk + offset) = { invoke, uint32_t(need) };
    writePos += need;
    return chunk + offset + sizeof(Header);
}

void DeferredCommandList::Publish()
{
    published = writePos;
    committed.store(writePos, std::memory_order_seq_cst);
    // The flag is cleared by whoever wakes the worker: one wake-up per sleep, not per call.
    if (workerSleeping.load(std::memory_order_seq_cst) && workerSleeping.exchange(0, std::memory_order_seq_cst))
        workEvent.Set();
}

void DeferredCommandList::Finish()
{
    // Data blocks only (no call) are published too: the worker skips them.
    Publish();
    // No spinning first.
    while (replayed.load(std::memory_order_acquire) != writePos)
    {
        finishSleeping.store(1, std::memory_order_seq_cst);
        if (replayed.load(std::memory_order_seq_cst) != writePos)
            finishEvent.Wait();
        finishSleeping.store(0, std::memory_order_relaxed);
    }
}

void DeferredCommandList::Run()
{
    uint64_t readPos = 0;
    for (;;)
    {
        uint64_t end = committed.load(std::memory_order_acquire);
        if (readPos == end)
        {
            // Nothing to replay: sleep until the recorder publishes (no spinning first).
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
                header->invoke(*target, reinterpret_cast<uint8_t*>(header) + sizeof(Header));
            readPos += header->size;
        }
        replayed.store(readPos, std::memory_order_seq_cst);
        if (finishSleeping.load(std::memory_order_seq_cst) && finishSleeping.exchange(0, std::memory_order_seq_cst))
            finishEvent.Set();
    }
}

// --- The calls ---------------------------------------------------------------------------------------------

void DeferredCommandList::barriers(RenderBarrierStages stages, const RenderBufferBarrier* bufferBarriers, uint32_t bufferBarriersCount,
    const RenderTextureBarrier* textureBarriers, uint32_t textureBarriersCount)
{
    const auto* b = Copy(bufferBarriers, bufferBarriersCount);
    const auto* t = Copy(textureBarriers, textureBarriersCount);
    Record([=](RenderCommandList& cl) { cl.barriers(stages, b, b ? bufferBarriersCount : 0, t, t ? textureBarriersCount : 0); });
}

void DeferredCommandList::dispatch(uint32_t x, uint32_t y, uint32_t z)
{
    Record([=](RenderCommandList& cl) { cl.dispatch(x, y, z); });
}

void DeferredCommandList::traceRays(uint32_t, uint32_t, uint32_t, RenderBufferReference, const RenderShaderBindingGroupsInfo&)
{
    log::Error("Deferred command list: traceRays is not supported");
}

void DeferredCommandList::drawInstanced(uint32_t vertexCountPerInstance, uint32_t instanceCount, uint32_t startVertexLocation, uint32_t startInstanceLocation)
{
    Record([=](RenderCommandList& cl) { cl.drawInstanced(vertexCountPerInstance, instanceCount, startVertexLocation, startInstanceLocation); });
}

void DeferredCommandList::drawIndexedInstanced(uint32_t indexCountPerInstance, uint32_t instanceCount, uint32_t startIndexLocation,
    int32_t baseVertexLocation, uint32_t startInstanceLocation)
{
    Record([=](RenderCommandList& cl) {
        cl.drawIndexedInstanced(indexCountPerInstance, instanceCount, startIndexLocation, baseVertexLocation, startInstanceLocation);
    });
}

void DeferredCommandList::setPipeline(const RenderPipeline* pipeline)
{
    Record([=](RenderCommandList& cl) { cl.setPipeline(pipeline); });
}

int64_t PendingPipeline::Wait() const
{
    if (ready.load(std::memory_order_acquire))
        return 0;
    const auto start = std::chrono::steady_clock::now();
    while (!ready.load(std::memory_order_acquire))
        ready.wait(false, std::memory_order_acquire);
    return (std::chrono::steady_clock::now() - start).count();
}

void DeferredCommandList::setPendingPipeline(const PendingPipeline* pending)
{
    Record([pending, this](RenderCommandList& cl) {
        if (const int64_t ns = pending->Wait())
            pipelineWaitNs.fetch_add(ns, std::memory_order_relaxed);
        cl.setPipeline(pending->pipeline.get());
    });
}

void DeferredCommandList::setComputePipelineLayout(const RenderPipelineLayout* pipelineLayout)
{
    Record([=](RenderCommandList& cl) { cl.setComputePipelineLayout(pipelineLayout); });
}

void DeferredCommandList::setComputePushConstants(uint32_t rangeIndex, const void* data, uint32_t offset, uint32_t size)
{
    if (!size)
        log::Error("Deferred command list: push constants need an explicit size");
    const auto* d = Copy(static_cast<const uint8_t*>(data), size);
    Record([=](RenderCommandList& cl) { cl.setComputePushConstants(rangeIndex, d, offset, size); });
}

void DeferredCommandList::setComputeDescriptorSet(RenderDescriptorSet* descriptorSet, uint32_t setIndex)
{
    Record([=](RenderCommandList& cl) { cl.setComputeDescriptorSet(descriptorSet, setIndex); });
}

void DeferredCommandList::setGraphicsPipelineLayout(const RenderPipelineLayout* pipelineLayout)
{
    if (pipelineLayout == lastGraphicsLayout)
        return;
    lastGraphicsLayout = pipelineLayout;
    Record([=](RenderCommandList& cl) { cl.setGraphicsPipelineLayout(pipelineLayout); });
}

void DeferredCommandList::setGraphicsPushConstants(uint32_t rangeIndex, const void* data, uint32_t offset, uint32_t size)
{
    if (!size)
        log::Error("Deferred command list: push constants need an explicit size");
    const auto* d = Copy(static_cast<const uint8_t*>(data), size);
    Record([=](RenderCommandList& cl) { cl.setGraphicsPushConstants(rangeIndex, d, offset, size); });
}

void DeferredCommandList::setGraphicsDescriptorSet(RenderDescriptorSet* descriptorSet, uint32_t setIndex)
{
    Record([=](RenderCommandList& cl) { cl.setGraphicsDescriptorSet(descriptorSet, setIndex); });
}

void DeferredCommandList::setGraphicsDescriptorSetDynamic(RenderDescriptorSet* descriptorSet, uint32_t setIndex, const uint32_t* dynamicOffsets,
    uint32_t dynamicOffsetCount)
{
    const auto* offsets = Copy(dynamicOffsets, dynamicOffsetCount);
    Record([=](RenderCommandList& cl) { cl.setGraphicsDescriptorSetDynamic(descriptorSet, setIndex, offsets, dynamicOffsetCount); });
}

void DeferredCommandList::setGraphicsRootDescriptor(RenderBufferReference bufferReference, uint32_t rootDescriptorIndex)
{
    Record([=](RenderCommandList& cl) { cl.setGraphicsRootDescriptor(bufferReference, rootDescriptorIndex); });
}

void DeferredCommandList::setRaytracingPipelineLayout(const RenderPipelineLayout*)
{
    log::Error("Deferred command list: ray tracing is not supported");
}

void DeferredCommandList::setRaytracingPushConstants(uint32_t, const void*, uint32_t, uint32_t)
{
    log::Error("Deferred command list: ray tracing is not supported");
}

void DeferredCommandList::setRaytracingDescriptorSet(RenderDescriptorSet*, uint32_t)
{
    log::Error("Deferred command list: ray tracing is not supported");
}

void DeferredCommandList::setIndexBuffer(const RenderIndexBufferView* view)
{
    const auto* v = Copy(view, 1);
    Record([=](RenderCommandList& cl) { cl.setIndexBuffer(v); });
}

void DeferredCommandList::setVertexBuffers(uint32_t startSlot, const RenderVertexBufferView* views, uint32_t viewCount, const RenderInputSlot* inputSlots)
{
    const auto* v = Copy(views, viewCount);
    const auto* s = Copy(inputSlots, viewCount);
    Record([=](RenderCommandList& cl) { cl.setVertexBuffers(startSlot, v, viewCount, s); });
}

void DeferredCommandList::setViewports(const RenderViewport* viewports, uint32_t count)
{
    const auto* v = Copy(viewports, count);
    Record([=](RenderCommandList& cl) { cl.setViewports(v, count); });
}

void DeferredCommandList::setScissors(const RenderRect* scissorRects, uint32_t count)
{
    const auto* r = Copy(scissorRects, count);
    Record([=](RenderCommandList& cl) { cl.setScissors(r, count); });
}

void DeferredCommandList::setFramebuffer(const RenderFramebuffer* framebuffer)
{
    // plume continues the active render pass for the same framebuffer, and with none active only sets it.
    if (framebuffer == lastFramebuffer)
        return;
    lastFramebuffer = framebuffer;
    Record([=](RenderCommandList& cl) { cl.setFramebuffer(framebuffer); });
}

void DeferredCommandList::custom(CustomCall fn, const void* data, uint32_t size)
{
    const auto* d = Copy(static_cast<const uint8_t*>(data), size);
    Record([=](RenderCommandList& cl) { fn(cl, d); });
    lastFramebuffer = kUnknown;
    lastGraphicsLayout = kUnknown;
}

void DeferredCommandList::runRenderPass()
{
    Record([](RenderCommandList& cl) { cl.runRenderPass(); });
}

void DeferredCommandList::setDepthBias(float depthBias, float depthBiasClamp, float slopeScaledDepthBias)
{
    Record([=](RenderCommandList& cl) { cl.setDepthBias(depthBias, depthBiasClamp, slopeScaledDepthBias); });
}

void DeferredCommandList::setBlendConstants(const float constants[4])
{
    const float c0 = constants[0], c1 = constants[1], c2 = constants[2], c3 = constants[3];
    Record([=](RenderCommandList& cl) {
        const float c[4] = { c0, c1, c2, c3 };
        cl.setBlendConstants(c);
    });
}

void DeferredCommandList::clearColor(uint32_t attachmentIndex, RenderColor colorValue, const RenderRect* clearRects, uint32_t clearRectsCount)
{
    const auto* r = Copy(clearRects, clearRectsCount);
    Record([=](RenderCommandList& cl) { cl.clearColor(attachmentIndex, colorValue, r, r ? clearRectsCount : 0); });
}

void DeferredCommandList::clearDepthStencil(bool clearDepth, bool clearStencil, float depthValue, uint32_t stencilValue, const RenderRect* clearRects,
    uint32_t clearRectsCount)
{
    const auto* r = Copy(clearRects, clearRectsCount);
    Record([=](RenderCommandList& cl) { cl.clearDepthStencil(clearDepth, clearStencil, depthValue, stencilValue, r, r ? clearRectsCount : 0); });
}

void DeferredCommandList::copyBufferRegion(RenderBufferReference dstBuffer, RenderBufferReference srcBuffer, uint64_t size)
{
    Record([=](RenderCommandList& cl) { cl.copyBufferRegion(dstBuffer, srcBuffer, size); });
}

void DeferredCommandList::copyTextureRegion(const RenderTextureCopyLocation& dstLocation, const RenderTextureCopyLocation& srcLocation, uint32_t dstX,
    uint32_t dstY, uint32_t dstZ, const RenderBox* srcBox)
{
    const auto* box = Copy(srcBox, 1);
    Record([=, dst = dstLocation, src = srcLocation](RenderCommandList& cl) { cl.copyTextureRegion(dst, src, dstX, dstY, dstZ, box); });
}

void DeferredCommandList::copyBuffer(const RenderBuffer* dstBuffer, const RenderBuffer* srcBuffer)
{
    Record([=](RenderCommandList& cl) { cl.copyBuffer(dstBuffer, srcBuffer); });
}

void DeferredCommandList::copyTexture(const RenderTexture* dstTexture, const RenderTexture* srcTexture)
{
    Record([=](RenderCommandList& cl) { cl.copyTexture(dstTexture, srcTexture); });
}

void DeferredCommandList::resolveTexture(const RenderTexture* dstTexture, const RenderTexture* srcTexture)
{
    Record([=](RenderCommandList& cl) { cl.resolveTexture(dstTexture, srcTexture); });
}

void DeferredCommandList::resolveTextureRegion(const RenderTexture* dstTexture, uint32_t dstX, uint32_t dstY, const RenderTexture* srcTexture,
    const RenderRect* srcRect, RenderResolveMode resolveMode)
{
    const auto* r = Copy(srcRect, 1);
    Record([=](RenderCommandList& cl) { cl.resolveTextureRegion(dstTexture, dstX, dstY, srcTexture, r, resolveMode); });
}

void DeferredCommandList::buildBottomLevelAS(const RenderAccelerationStructure*, RenderBufferReference, const RenderBottomLevelASBuildInfo&)
{
    log::Error("Deferred command list: acceleration structures are not supported");
}

void DeferredCommandList::buildTopLevelAS(const RenderAccelerationStructure*, RenderBufferReference, RenderBufferReference, const RenderTopLevelASBuildInfo&)
{
    log::Error("Deferred command list: acceleration structures are not supported");
}

void DeferredCommandList::discardTexture(const RenderTexture* texture)
{
    Record([=](RenderCommandList& cl) { cl.discardTexture(texture); });
}

void DeferredCommandList::resetQueryPool(const RenderQueryPool* queryPool, uint32_t queryFirstIndex, uint32_t queryCount)
{
    Record([=](RenderCommandList& cl) { cl.resetQueryPool(queryPool, queryFirstIndex, queryCount); });
}

void DeferredCommandList::writeTimestamp(const RenderQueryPool* queryPool, uint32_t queryIndex)
{
    Record([=](RenderCommandList& cl) { cl.writeTimestamp(queryPool, queryIndex); });
}

void DeferredCommandList::writeBreadcrumb(const RenderBuffer* buffer, uint64_t reachedOffset, uint64_t doneOffset, uint32_t value)
{
    Record([=](RenderCommandList& cl) { cl.writeBreadcrumb(buffer, reachedOffset, doneOffset, value); });
}

} // namespace ac6::gpu
