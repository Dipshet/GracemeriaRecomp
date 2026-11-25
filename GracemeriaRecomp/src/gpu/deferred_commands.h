#pragma once
// Deferred command recording. Recording Vulkan commands (the driver's and plume's work) is a large part of the
// renderer's host time. Draws record into this list instead: each call is stored as a small closure in an arena, and a
// worker thread replays the calls, in order, onto the frame's real command list while the renderer goes on
// translating. The frame's end waits for the worker before the list is closed and submitted. The commands and their
// order are the same, so the frames are too.
//
// Objects the calls refer to must outlive the replay: the renderer only frees GPU objects once the frame's
// GPU work has completed (its retire lists), which is after the replay. Pointer arguments (barriers, push
// constants, rectangles, vertex buffer views) are copied into the arena.

#include "platform/platform.h"

#include <plume_render_interface.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <type_traits>

namespace ac6::gpu {

// A pipeline being built on another thread. A draw records its bind before the pipeline
// exists. The replay waits for it there, while the renderer goes on translating draws.
struct PendingPipeline
{
    std::unique_ptr<plume::RenderPipeline> pipeline;
    std::atomic<bool> ready{ false };

    void Publish(std::unique_ptr<plume::RenderPipeline> built)
    {
        pipeline = std::move(built);
        ready.store(true, std::memory_order_release);
        ready.notify_all();
    }
    // Returns the nanoseconds waited.
    int64_t Wait() const;
};

class DeferredCommandList final : public plume::RenderCommandList
{
public:
    DeferredCommandList();
    ~DeferredCommandList() override;

    // A new frame recorded into `target` (begun already). The worker is idle between frames.
    void Start(plume::RenderCommandList* target);
    // Waits until every call recorded so far has been replayed onto the target.
    void Finish();

    void begin() override {}
    void end() override {}
    void barriers(plume::RenderBarrierStages stages, const plume::RenderBufferBarrier* bufferBarriers, uint32_t bufferBarriersCount,
        const plume::RenderTextureBarrier* textureBarriers, uint32_t textureBarriersCount) override;
    void dispatch(uint32_t x, uint32_t y, uint32_t z) override;
    void traceRays(uint32_t, uint32_t, uint32_t, plume::RenderBufferReference, const plume::RenderShaderBindingGroupsInfo&) override;
    void drawInstanced(uint32_t vertexCountPerInstance, uint32_t instanceCount, uint32_t startVertexLocation, uint32_t startInstanceLocation) override;
    void drawIndexedInstanced(uint32_t indexCountPerInstance, uint32_t instanceCount, uint32_t startIndexLocation, int32_t baseVertexLocation,
        uint32_t startInstanceLocation) override;
    void setPipeline(const plume::RenderPipeline* pipeline) override;
    // Binds `pending`'s pipeline once it is built (the replay waits for it).
    void setPendingPipeline(const PendingPipeline* pending);
    // Nanoseconds the replay has spent waiting for pipelines, in all.
    int64_t PipelineWaitNs() const { return pipelineWaitNs.load(std::memory_order_relaxed); }
    void setComputePipelineLayout(const plume::RenderPipelineLayout* pipelineLayout) override;
    void setComputePushConstants(uint32_t rangeIndex, const void* data, uint32_t offset, uint32_t size) override;
    void setComputeDescriptorSet(plume::RenderDescriptorSet* descriptorSet, uint32_t setIndex) override;
    void setGraphicsPipelineLayout(const plume::RenderPipelineLayout* pipelineLayout) override;
    void setGraphicsPushConstants(uint32_t rangeIndex, const void* data, uint32_t offset, uint32_t size) override;
    void setGraphicsDescriptorSet(plume::RenderDescriptorSet* descriptorSet, uint32_t setIndex) override;
    void setGraphicsDescriptorSetDynamic(plume::RenderDescriptorSet* descriptorSet, uint32_t setIndex, const uint32_t* dynamicOffsets,
        uint32_t dynamicOffsetCount) override;
    void setGraphicsRootDescriptor(plume::RenderBufferReference bufferReference, uint32_t rootDescriptorIndex) override;
    void setRaytracingPipelineLayout(const plume::RenderPipelineLayout*) override;
    void setRaytracingPushConstants(uint32_t, const void*, uint32_t, uint32_t) override;
    void setRaytracingDescriptorSet(plume::RenderDescriptorSet*, uint32_t) override;
    void setIndexBuffer(const plume::RenderIndexBufferView* view) override;
    void setVertexBuffers(uint32_t startSlot, const plume::RenderVertexBufferView* views, uint32_t viewCount, const plume::RenderInputSlot* inputSlots) override;
    void setViewports(const plume::RenderViewport* viewports, uint32_t count) override;
    void setScissors(const plume::RenderRect* scissorRects, uint32_t count) override;
    void setFramebuffer(const plume::RenderFramebuffer* framebuffer) override;
    void runRenderPass() override;
    void setDepthBias(float depthBias, float depthBiasClamp, float slopeScaledDepthBias) override;
    void setBlendConstants(const float constants[4]) override;
    void setLineWidth(float width) override;
    void clearColor(uint32_t attachmentIndex, plume::RenderColor colorValue, const plume::RenderRect* clearRects, uint32_t clearRectsCount) override;
    void clearDepthStencil(bool clearDepth, bool clearStencil, float depthValue, uint32_t stencilValue, const plume::RenderRect* clearRects,
        uint32_t clearRectsCount) override;
    void copyBufferRegion(plume::RenderBufferReference dstBuffer, plume::RenderBufferReference srcBuffer, uint64_t size) override;
    void copyTextureRegion(const plume::RenderTextureCopyLocation& dstLocation, const plume::RenderTextureCopyLocation& srcLocation, uint32_t dstX,
        uint32_t dstY, uint32_t dstZ, const plume::RenderBox* srcBox) override;
    void copyBuffer(const plume::RenderBuffer* dstBuffer, const plume::RenderBuffer* srcBuffer) override;
    void copyTexture(const plume::RenderTexture* dstTexture, const plume::RenderTexture* srcTexture) override;
    void resolveTexture(const plume::RenderTexture* dstTexture, const plume::RenderTexture* srcTexture) override;
    void resolveTextureRegion(const plume::RenderTexture* dstTexture, uint32_t dstX, uint32_t dstY, const plume::RenderTexture* srcTexture,
        const plume::RenderRect* srcRect, plume::RenderResolveMode resolveMode) override;
    void buildBottomLevelAS(const plume::RenderAccelerationStructure*, plume::RenderBufferReference, const plume::RenderBottomLevelASBuildInfo&) override;
    void buildTopLevelAS(const plume::RenderAccelerationStructure*, plume::RenderBufferReference, plume::RenderBufferReference,
        const plume::RenderTopLevelASBuildInfo&) override;
    void discardTexture(const plume::RenderTexture* texture) override;
    void resetQueryPool(const plume::RenderQueryPool* queryPool, uint32_t queryFirstIndex, uint32_t queryCount) override;
    void writeTimestamp(const plume::RenderQueryPool* queryPool, uint32_t queryIndex) override;
    void writeBreadcrumb(const plume::RenderBuffer* buffer, uint64_t reachedOffset, uint64_t doneOffset, uint32_t value) override;

    // A call on the target list itself in its place (a library recording its own Vulkan commands):
    // `fn` gets the target and a copy of the `size` bytes at `data`. The list skips binds that repeat its last
    // framebuffer and graphics pipeline layout, and `fn` may change those, so both are unknown afterwards.
    using CustomCall = void (*)(plume::RenderCommandList& target, const void* data);
    void custom(CustomCall fn, const void* data, uint32_t size);

private:
    using Invoke = void (*)(plume::RenderCommandList&, void*);
    struct alignas(16) Header
    {
        Invoke invoke;    // null: a data block, skipped
        uint32_t size;    // bytes to the next header
    };
    static constexpr uint64_t kChunkBytes = 1u << 20;
    static constexpr uint32_t kChunks = 8;    // a ring: positions only grow, chunk = position / size % kChunks. Small enough to stay in cache
    static constexpr uint64_t kPublishBytes = 16u << 10;

    template <class F>
    void Record(F&& f)
    {
        using T = std::decay_t<F>;
        static_assert(std::is_trivially_destructible_v<T>, "recorded calls capture plain values");
        void* payload = Allocate(sizeof(T), [](plume::RenderCommandList& cl, void* p) { (*static_cast<T*>(p))(cl); });
        new (payload) T(std::forward<F>(f));
        // Published in batches: each publication is a locked store to a line the worker is reading.
        if (writePos - published >= kPublishBytes)
            Publish();
    }
    template <class T>
    const T* Copy(const T* data, uint32_t count)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        if (!data || !count)
            return nullptr;
        void* p = Allocate(sizeof(T) * count, nullptr);
        std::memcpy(p, data, sizeof(T) * count);
        return static_cast<const T*>(p);
    }
    void* Allocate(uint64_t bytes, Invoke invoke);   // a header and `bytes` of payload (not yet published)
    void Publish();
    void Run();

    std::atomic<uint8_t*> chunks[kChunks] = {};
    uint64_t writePos = 0;                  // recorder: next header
    uint64_t published = 0;                 // recorder: last position published
    // Calls that repeat the state the target list already has: not recorded (the renderer binds framebuffer
    // and pipeline layout for every draw). Each frame's list keeps its own state, so these start unknown.
    static inline const void* const kUnknown = reinterpret_cast<const void*>(~uintptr_t(0));
    const void* lastFramebuffer = kUnknown;
    const void* lastGraphicsLayout = kUnknown;
    // On lines of their own: the recorder writes one, the worker the other.
    alignas(64) std::atomic<uint64_t> committed{ 0 };   // bytes the worker may replay
    std::atomic<uint32_t> workerSleeping{ 0 };
    alignas(64) std::atomic<uint64_t> replayed{ 0 };    // bytes the worker has replayed
    std::atomic<uint32_t> finishSleeping{ 0 };
    platform::Event workEvent;     // the worker sleeps on it (auto-reset)
    platform::Event finishEvent;   // Finish sleeps on it
    plume::RenderCommandList* target = nullptr;
    std::atomic<int64_t> pipelineWaitNs{ 0 };
    std::thread worker;
};

} // namespace ac6::gpu
