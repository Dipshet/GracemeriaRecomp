//
// plume
//
// Copyright (c) 2024 renderbag and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file for details.
//

#pragma once

#include "plume_render_interface.h"

#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

#if defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#elif defined(__ANDROID__)
#define VK_USE_PLATFORM_ANDROID_KHR
#elif defined(__linux__) && !defined(PLUME_SDL_VULKAN_ENABLED)
#define VK_USE_PLATFORM_XLIB_KHR
#elif defined(__APPLE__)
#define VK_USE_PLATFORM_METAL_EXT
#include "plume_apple.h"
#endif

// For VK_KHR_portability_subset
#define VK_ENABLE_BETA_EXTENSIONS

#include <volk.h>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#endif

#include <vk_mem_alloc.h>

#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace plume {
    // AC6 runtime addition: the backend's error messages go to this callback when one is set (stderr otherwise; the
    // windowed build has no console). VulkanDeviceLost() is true once a submit, a fence wait, a present or an acquire
    // returned VK_ERROR_DEVICE_LOST; NoteVulkanResult records it for the caller's own Vulkan calls.
    void SetVulkanErrorCallback(void (*callback)(const char *message));
    bool VulkanDeviceLost();
    void NoteVulkanResult(VkResult res);
    // AC6: crash diagnostics. VulkanBufferMarkers: breadcrumbs are buffer markers (they also say what finished), not
    // transfer fills. VulkanDeviceFaultReport: after a lost device, the driver's description of the fault
    // (VK_EXT_device_fault) as log lines; empty where the driver offers none.
    struct RenderDevice;
    bool VulkanBufferMarkers(RenderDevice *device);
    std::vector<std::string> VulkanDeviceFaultReport(RenderDevice *device);

    struct VulkanCommandQueue;
    struct VulkanDevice;
    struct VulkanInterface;
    struct VulkanPool;
    struct VulkanQueue;

    struct VulkanBuffer : RenderBuffer {
        VkBuffer vk = VK_NULL_HANDLE;
        VulkanDevice *device = nullptr;
        VulkanPool *pool = nullptr;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VmaAllocationInfo allocationInfo = {};
        RenderBufferDesc desc;
        RenderBarrierStages barrierStages = RenderBarrierStage::NONE;

        VulkanBuffer() = default;
        VulkanBuffer(VulkanDevice *device, VulkanPool *pool, const RenderBufferDesc &desc);
        ~VulkanBuffer() override;
        void *map(uint32_t subresource, const RenderRange *readRange) override;
        void unmap(uint32_t subresource, const RenderRange *writtenRange) override;
        std::unique_ptr<RenderBufferFormattedView> createBufferFormattedView(RenderFormat format) override;
        void setName(const std::string &name) override;
        uint64_t getDeviceAddress() const override;
    };

    struct VulkanBufferFormattedView : RenderBufferFormattedView {
        VkBufferView vk = VK_NULL_HANDLE;
        VulkanBuffer *buffer = nullptr;

        VulkanBufferFormattedView(VulkanBuffer *buffer, RenderFormat format);
        ~VulkanBufferFormattedView() override;
    };

    struct VulkanTexture : RenderTexture {
        VkImage vk = VK_NULL_HANDLE;
        VkImageView imageView = VK_NULL_HANDLE;
        VkFormat imageFormat = VK_FORMAT_UNDEFINED;
        VkImageSubresourceRange imageSubresourceRange = {};
        VulkanDevice *device = nullptr;
        VulkanPool *pool = nullptr;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VmaAllocationInfo allocationInfo = {};
        RenderTextureLayout textureLayout = RenderTextureLayout::UNKNOWN;
        RenderBarrierStages barrierStages = RenderBarrierStage::NONE;
        bool ownership = false;
        RenderTextureDesc desc;

        VulkanTexture() = default;
        VulkanTexture(VulkanDevice *device, VulkanPool *pool, const RenderTextureDesc &desc);
        VulkanTexture(VulkanDevice *device, VkImage image);
        ~VulkanTexture() override;
        void createImageView(VkFormat format);
        std::unique_ptr<RenderTextureView> createTextureView(const RenderTextureViewDesc &desc) const override;
        void setName(const std::string &name) override;
        void fillSubresourceRange();
    };

    struct VulkanTextureView : RenderTextureView {
        VkImageView vk = VK_NULL_HANDLE;
        const VulkanTexture *texture = nullptr;
        RenderTextureViewDesc desc;

        VulkanTextureView(const VulkanTexture *texture, const RenderTextureViewDesc &desc);
        ~VulkanTextureView() override;
    };

    struct VulkanAccelerationStructure : RenderAccelerationStructure {
        VkAccelerationStructureKHR vk = VK_NULL_HANDLE;
        VulkanDevice *device = nullptr;
        RenderAccelerationStructureType type = RenderAccelerationStructureType::UNKNOWN;

        VulkanAccelerationStructure(VulkanDevice *device, const RenderAccelerationStructureDesc &desc);
        ~VulkanAccelerationStructure() override;
    };

    struct VulkanDescriptorSetLayout {
        VkDescriptorSetLayout vk = VK_NULL_HANDLE;
        std::vector<VkDescriptorSetLayoutBinding> setBindings;
        std::vector<uint32_t> descriptorIndexBases;
        std::vector<uint32_t> descriptorBindingIndices;
        VulkanDevice *device = nullptr;

        VulkanDescriptorSetLayout(VulkanDevice *device, const RenderDescriptorSetDesc &descriptorSetDesc);
        ~VulkanDescriptorSetLayout();
    };

    struct VulkanPipelineLayout : RenderPipelineLayout {
        VkPipelineLayout vk = VK_NULL_HANDLE;
        std::vector<VkPushConstantRange> pushConstantRanges;
        std::vector<VulkanDescriptorSetLayout *> descriptorSetLayouts;
        VulkanDevice *device = nullptr;

        VulkanPipelineLayout(VulkanDevice *device, const RenderPipelineLayoutDesc &desc);
        ~VulkanPipelineLayout() override;
    };

    struct VulkanShader : RenderShader {
        VkShaderModule vk = VK_NULL_HANDLE;
        uint64_t serial = 0;   // AC6: never reused, unlike module handles (pipeline library keys)
        std::string entryPointName;
        VulkanDevice *device = nullptr;
        RenderShaderFormat format = RenderShaderFormat::UNKNOWN;

        VulkanShader(VulkanDevice *device, const void *data, uint64_t size, const char *entryPointName, RenderShaderFormat format);
        ~VulkanShader() override;
        virtual void setName(const std::string &name) override;
    };

    struct VulkanSampler : RenderSampler {
        VkSampler vk = VK_NULL_HANDLE;
        VulkanDevice *device = nullptr;

        VulkanSampler(VulkanDevice *device, const RenderSamplerDesc &desc);
        ~VulkanSampler();
    };

    struct VulkanPipeline : RenderPipeline {
        enum class Type {
            Unknown,
            Compute,
            Graphics,
            Raytracing
        };

        VulkanDevice *device = nullptr;
        Type type = Type::Unknown;

        VulkanPipeline(VulkanDevice *device, Type type);
        virtual ~VulkanPipeline() override;
    };

    struct VulkanComputePipeline : VulkanPipeline {
        VkPipeline vk = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

        VulkanComputePipeline(VulkanDevice *device, const RenderComputePipelineDesc &desc);
        ~VulkanComputePipeline() override;
        void setName(const std::string &name) override;
        RenderPipelineProgram getProgram(const std::string &name) const override;
    };

    struct VulkanGraphicsPipeline : VulkanPipeline {
        // AC6: rasterizer and depth/stencil state the Vulkan 1.3 path sets when the pipeline is bound.
        struct DynamicValues {
            VkCullModeFlags cullMode = VK_CULL_MODE_NONE;
            VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            VkBool32 depthTest = VK_FALSE, depthWrite = VK_FALSE, stencilTest = VK_FALSE, depthBiasEnable = VK_FALSE, depthClamp = VK_FALSE;
            VkCompareOp depthCompare = VK_COMPARE_OP_ALWAYS;
            VkStencilOp stencilOps[2][3] = {};   // front/back: fail, pass, depth fail
            VkCompareOp stencilCompare[2] = { VK_COMPARE_OP_ALWAYS, VK_COMPARE_OP_ALWAYS };
            uint32_t stencilCompareMask = 0, stencilWriteMask = 0, stencilReference = 0;
            bool setDepthBiasValues = false;
            float depthBias[3] = {};   // constant, clamp, slope
        };

        VkPipeline vk = VK_NULL_HANDLE;
        VkRenderPass renderPass = VK_NULL_HANDLE;
        bool usesDynamicState = false;
        DynamicValues dynamicValues;

        VulkanGraphicsPipeline(VulkanDevice *device, const RenderGraphicsPipelineDesc &desc);
        ~VulkanGraphicsPipeline() override;
        void setName(const std::string &name) override;
        RenderPipelineProgram getProgram(const std::string &name) const override;
        static VkRenderPass createRenderPass(VulkanDevice *device, const VkFormat *renderTargetFormat, uint32_t renderTargetCount, VkFormat depthTargetFormat, VkSampleCountFlagBits sampleCount);
        void createDynamic(VulkanDevice *device, const RenderGraphicsPipelineDesc &desc, const VkPipelineShaderStageCreateInfo *stages, uint32_t stageCount,
            const VkPipelineVertexInputStateCreateInfo &vertexInput, const VkPipelineInputAssemblyStateCreateInfo &inputAssembly,
            const VkPipelineRasterizationStateCreateInfo &rasterization, const VkPipelineMultisampleStateCreateInfo &multisampling,
            const VkPipelineColorBlendStateCreateInfo &colorBlend, const VkFormat *renderTargetFormats);
    };

    struct VulkanRaytracingPipeline : VulkanPipeline {
        VkPipeline vk = VK_NULL_HANDLE;
        std::unordered_map<std::string, RenderPipelineProgram> nameProgramMap;
        uint32_t groupCount = 0;
        uint32_t descriptorSetCount = 0;

        VulkanRaytracingPipeline(VulkanDevice *device, const RenderRaytracingPipelineDesc &desc, const RenderPipeline *previousPipeline);
        ~VulkanRaytracingPipeline() override;
        void setName(const std::string &name) override;
        RenderPipelineProgram getProgram(const std::string &name) const override;
    };

    struct VulkanDescriptorSet : RenderDescriptorSet {
        VkDescriptorSet vk = VK_NULL_HANDLE;
        VulkanDescriptorSetLayout *setLayout = nullptr;
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        VulkanDevice *device = nullptr;

        VulkanDescriptorSet(VulkanDevice *device, const RenderDescriptorSetDesc &desc);
        ~VulkanDescriptorSet() override;
        void setBuffer(uint32_t descriptorIndex, const RenderBuffer *buffer, uint64_t bufferSize, const RenderBufferStructuredView *bufferStructuredView, const RenderBufferFormattedView *bufferFormattedView) override;
        void setTexture(uint32_t descriptorIndex, const RenderTexture *texture, RenderTextureLayout textureLayout, const RenderTextureView *textureView) override;
        void setSampler(uint32_t descriptorIndex, const RenderSampler *sampler) override;
        void setAccelerationStructure(uint32_t descriptorIndex, const RenderAccelerationStructure *accelerationStructure) override;
        void setDescriptor(uint32_t descriptorIndex, const VkDescriptorBufferInfo *bufferInfo, const VkDescriptorImageInfo *imageInfo, const VkBufferView *texelBufferView, void *pNext);
        static VkDescriptorPool createDescriptorPool(VulkanDevice *device, const std::unordered_map<VkDescriptorType, uint32_t> &typeCounts, bool lastRangeIsBoundless);
    };

    struct VulkanSwapChain : RenderSwapChain {
        RenderSwapChainDesc desc;
        VkSwapchainKHR vk = VK_NULL_HANDLE;
        VulkanCommandQueue *commandQueue = nullptr;
        VkSurfaceKHR surface = VK_NULL_HANDLE;
#if defined(__APPLE__)
        std::unique_ptr<CocoaWindow> windowWrapper;
#endif
        uint64_t presentCount = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        VkSwapchainCreateInfoKHR createInfo = {};
        VkSurfaceFormatKHR pickedSurfaceFormat = {};
        VkPresentModeKHR createdPresentMode = VK_PRESENT_MODE_FIFO_KHR;
        VkPresentModeKHR requiredPresentMode = VK_PRESENT_MODE_FIFO_KHR;
        VkCompositeAlphaFlagBitsKHR pickedAlphaFlag = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        std::vector<VulkanTexture> textures;
        uint64_t currentPresentId = 0;
        bool immediatePresentModeSupported = false;
        bool mailboxPresentModeSupported = false;

        VulkanSwapChain(VulkanCommandQueue *commandQueue, const RenderSwapChainDesc &desc);
        ~VulkanSwapChain() override;
        bool present(uint32_t textureIndex, RenderCommandSemaphore **waitSemaphores, uint32_t waitSemaphoreCount) override;
        void wait() override;
        bool resize() override;
        bool needsResize() const override;
        void setVsyncEnabled(bool vsyncEnabled) override;
        bool isVsyncEnabled() const override;
        uint32_t getWidth() const override;
        uint32_t getHeight() const override;
        RenderTexture *getTexture(uint32_t textureIndex) override;
        uint32_t getTextureCount() const override;
        bool acquireTexture(RenderCommandSemaphore *signalSemaphore, uint32_t *textureIndex) override;
        RenderWindow getWindow() const override;
        bool isEmpty() const override;
        uint32_t getRefreshRate() const override;
        RenderColorSpace getColorSpace() const override;
        bool setFormat(RenderFormat format, RenderColorSpace colorSpace) override;
        void getWindowSize(uint32_t &dstWidth, uint32_t &dstHeight) const;
        void releaseSwapChain();
        void releaseImageViews();
    };

    struct VulkanFramebuffer : RenderFramebuffer {
        VulkanDevice *device = nullptr;
        VkFramebuffer vk = VK_NULL_HANDLE;
        VkRenderPass renderPass = VK_NULL_HANDLE;
        std::vector<const VulkanTexture *> colorAttachments;
        const VulkanTexture *depthAttachment = nullptr;
        std::unique_ptr<VulkanTextureView> depthAttachmentView = nullptr;
        std::unique_ptr<VulkanTextureView> depthResolveView = nullptr;
        bool depthAttachmentReadOnly = false;
        // AC6: dynamic rendering (Vulkan 1.3 path): the attachments' views and the depth format.
        std::vector<VkImageView> colorViews;
        VkImageView depthView = VK_NULL_HANDLE;
        VkFormat depthFormat = VK_FORMAT_UNDEFINED;
        uint32_t width = 0;
        uint32_t height = 0;

        VulkanFramebuffer(VulkanDevice *device, const RenderFramebufferDesc &desc);
        ~VulkanFramebuffer() override;
        uint32_t getWidth() const override;
        uint32_t getHeight() const override;
        bool contains(const VulkanTexture *attachment) const;
    };

    struct VulkanQueryPool : RenderQueryPool {
        VulkanDevice *device = nullptr;
        std::vector<uint64_t> results;
        VkQueryPool vk = VK_NULL_HANDLE;

        VulkanQueryPool(VulkanDevice *device, uint32_t queryCount);
        virtual ~VulkanQueryPool() override;
        virtual void queryResults() override;
        virtual const uint64_t *getResults() const override;
        virtual uint32_t getCount() const override;
    };

    struct VulkanCommandList : RenderCommandList {
        VkCommandBuffer vk = VK_NULL_HANDLE;
        VkCommandPool commandPool = VK_NULL_HANDLE;
        VulkanCommandQueue *queue = nullptr;
        const VulkanFramebuffer *targetFramebuffer = nullptr;
        const VulkanPipelineLayout *activeComputePipelineLayout = nullptr;
        const VulkanPipelineLayout *activeGraphicsPipelineLayout = nullptr;
        const VulkanPipelineLayout *activeRaytracingPipelineLayout = nullptr;
        VkRenderPass activeRenderPass = VK_NULL_HANDLE;
        bool renderingActive = false;   // AC6: inside vkCmdBeginRendering (Vulkan 1.3 path)
        VulkanGraphicsPipeline::DynamicValues appliedDynamic;
        bool appliedDynamicValid = false;
        void applyDynamicState(const VulkanGraphicsPipeline::DynamicValues &values);

        VulkanCommandList(VulkanCommandQueue *queue);
        ~VulkanCommandList() override;
        void begin() override;
        void end() override;
        void barriers(RenderBarrierStages stages, const RenderBufferBarrier *bufferBarriers, uint32_t bufferBarriersCount, const RenderTextureBarrier *textureBarriers, uint32_t textureBarriersCount) override;
        void dispatch(uint32_t threadGroupCountX, uint32_t threadGroupCountY, uint32_t threadGroupCountZ) override;
        void traceRays(uint32_t width, uint32_t height, uint32_t depth, RenderBufferReference shaderBindingTable, const RenderShaderBindingGroupsInfo &shaderBindingGroupsInfo) override;
        void drawInstanced(uint32_t vertexCountPerInstance, uint32_t instanceCount, uint32_t startVertexLocation, uint32_t startInstanceLocation) override;
        void drawIndexedInstanced(uint32_t indexCountPerInstance, uint32_t instanceCount, uint32_t startIndexLocation, int32_t baseVertexLocation, uint32_t startInstanceLocation) override;
        void setPipeline(const RenderPipeline *pipeline) override;
        void setComputePipelineLayout(const RenderPipelineLayout *pipelineLayout) override;
        void setComputePushConstants(uint32_t rangeIndex, const void *data, uint32_t offset = 0, uint32_t size = 0) override;
        void setComputeDescriptorSet(RenderDescriptorSet *descriptorSet, uint32_t setIndex) override;
        void setGraphicsPipelineLayout(const RenderPipelineLayout *pipelineLayout) override;
        void setGraphicsPushConstants(uint32_t rangeIndex, const void *data, uint32_t offset = 0, uint32_t size = 0) override;
        void setGraphicsDescriptorSet(RenderDescriptorSet *descriptorSet, uint32_t setIndex) override;
        void setGraphicsDescriptorSetDynamic(RenderDescriptorSet *descriptorSet, uint32_t setIndex, const uint32_t *dynamicOffsets, uint32_t dynamicOffsetCount) override;
        void setGraphicsRootDescriptor(RenderBufferReference bufferReference, uint32_t rootDescriptorIndex) override;
        void setRaytracingPipelineLayout(const RenderPipelineLayout *pipelineLayout) override;
        void setRaytracingPushConstants(uint32_t rangeIndex, const void *data, uint32_t offset = 0, uint32_t size = 0) override;
        void setRaytracingDescriptorSet(RenderDescriptorSet *descriptorSet, uint32_t setIndex) override;
        void setIndexBuffer(const RenderIndexBufferView *view) override;
        void setVertexBuffers(uint32_t startSlot, const RenderVertexBufferView *views, uint32_t viewCount, const RenderInputSlot *inputSlots) override;
        void setViewports(const RenderViewport *viewports, uint32_t count) override;
        void setScissors(const RenderRect *scissorRects, uint32_t count) override;
        void setFramebuffer(const RenderFramebuffer *framebuffer) override;
        void runRenderPass() override;
        void setDepthBias(float depthBias, float depthBiasClamp, float slopeScaledDepthBias) override;
        void setBlendConstants(const float constants[4]) override;
        void clearColor(uint32_t attachmentIndex, RenderColor colorValue, const RenderRect *clearRects, uint32_t clearRectsCount) override;
        void clearDepthStencil(bool clearDepth, bool clearStencil, float depthValue, uint32_t stencilValue, const RenderRect *clearRects, uint32_t clearRectsCount) override;
        void copyBufferRegion(RenderBufferReference dstBuffer, RenderBufferReference srcBuffer, uint64_t size) override;
        void copyTextureRegion(const RenderTextureCopyLocation &dstLocation, const RenderTextureCopyLocation &srcLocation, uint32_t dstX, uint32_t dstY, uint32_t dstZ, const RenderBox *srcBox) override;
        void copyBuffer(const RenderBuffer *dstBuffer, const RenderBuffer *srcBuffer) override;
        void copyTexture(const RenderTexture *dstTexture, const RenderTexture *srcTexture) override;
        void resolveTexture(const RenderTexture *dstTexture, const RenderTexture *srcTexture) override;
        void resolveTextureRegion(const RenderTexture *dstTexture, uint32_t dstX, uint32_t dstY, const RenderTexture *srcTexture, const RenderRect *srcRect, RenderResolveMode resolveMode) override;
        void buildBottomLevelAS(const RenderAccelerationStructure *dstAccelerationStructure, RenderBufferReference scratchBuffer, const RenderBottomLevelASBuildInfo &buildInfo) override;
        void buildTopLevelAS(const RenderAccelerationStructure *dstAccelerationStructure, RenderBufferReference scratchBuffer, RenderBufferReference instancesBuffer, const RenderTopLevelASBuildInfo &buildInfo) override;
        void discardTexture(const RenderTexture* texture) override;
        void resetQueryPool(const RenderQueryPool *queryPool, uint32_t queryFirstIndex, uint32_t queryCount) override;
        void writeTimestamp(const RenderQueryPool *queryPool, uint32_t queryIndex) override;
        void writeBreadcrumb(const RenderBuffer *buffer, uint64_t reachedOffset, uint64_t doneOffset, uint32_t value) override;
        // AC6: a breadcrumb made inside a render pass without VK_AMD_buffer_marker: written as the pass ends.
        VkBuffer pendingBreadcrumbBuffer = VK_NULL_HANDLE;
        uint64_t pendingBreadcrumbOffset = 0;
        uint32_t pendingBreadcrumbValue = 0;
        void checkActiveRenderPass();
        void endActiveRenderPass();
        // AC6: barriers wait until the next command needs them, so consecutive ones are one vkCmdPipelineBarrier.
        void flushBarriers();
        std::vector<VkImageMemoryBarrier> pendingImageBarriers;
        std::vector<VkBufferMemoryBarrier> pendingBufferBarriers;
        VkPipelineStageFlags pendingSrcStages = 0, pendingDstStages = 0;
        void setDescriptorSet(VkPipelineBindPoint bindPoint, const VulkanPipelineLayout *pipelineLayout, const RenderDescriptorSet *descriptorSet, uint32_t setIndex);
    };

    struct VulkanCommandFence : RenderCommandFence {
        VkFence vk = VK_NULL_HANDLE;
        VulkanDevice *device = nullptr;

        VulkanCommandFence(VulkanDevice *device);
        ~VulkanCommandFence() override;
    };

    struct VulkanCommandSemaphore : RenderCommandSemaphore {
        VkSemaphore vk = VK_NULL_HANDLE;
        VulkanDevice *device = nullptr;

        VulkanCommandSemaphore(VulkanDevice *device);
        ~VulkanCommandSemaphore() override;
    };

    struct VulkanCommandQueue : RenderCommandQueue {
        VulkanQueue *queue = nullptr;
        VulkanDevice *device = nullptr;
        uint32_t familyIndex = 0;
        uint32_t queueIndex = 0;
        std::unordered_set<VulkanSwapChain *> swapChains;
        RenderCommandListType type = RenderCommandListType::UNKNOWN;

        VulkanCommandQueue(VulkanDevice *device, RenderCommandListType type);
        ~VulkanCommandQueue() override;
        std::unique_ptr<RenderCommandList> createCommandList() override;
        std::unique_ptr<RenderSwapChain> createSwapChain(const RenderSwapChainDesc &desc) override;
        void executeCommandLists(const RenderCommandList **commandLists, uint32_t commandListCount, RenderCommandSemaphore **waitSemaphores, uint32_t waitSemaphoreCount, RenderCommandSemaphore **signalSemaphores, uint32_t signalSemaphoreCount, RenderCommandFence *signalFence) override;
        void waitForCommandFence(RenderCommandFence *fence) override;
    };

    struct VulkanPool : RenderPool {
        VmaPool vk = VK_NULL_HANDLE;
        VulkanDevice *device = nullptr;

        VulkanPool(VulkanDevice *device, const RenderPoolDesc &desc);
        ~VulkanPool() override;
        std::unique_ptr<RenderBuffer> createBuffer(const RenderBufferDesc &desc) override;
        std::unique_ptr<RenderTexture> createTexture(const RenderTextureDesc &desc) override;
    };
    
    struct VulkanQueue {
        VkQueue vk;
        std::unique_ptr<std::mutex> mutex;
        std::unordered_set<const VulkanCommandQueue *> virtualQueues;
    };

    struct VulkanQueueFamily {
        std::vector<VulkanQueue> queues;

        void add(VulkanCommandQueue *virtualQueue);
        void remove(VulkanCommandQueue *virtualQueue);
    };

    struct VulkanDevice : RenderDevice {
        VkDevice vk = VK_NULL_HANDLE;
        VulkanInterface *renderInterface = nullptr;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkPhysicalDeviceProperties physicalDeviceProperties = {};
        VmaAllocator allocator = VK_NULL_HANDLE;
        uint32_t queueFamilyIndices[3] = {};
        std::vector<VulkanQueueFamily> queueFamilies;
        RenderDeviceCapabilities capabilities;
        RenderDeviceDescription description;
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtPipelineProperties = {};
        VkPhysicalDeviceSampleLocationsPropertiesEXT sampleLocationProperties = {};
        std::unique_ptr<RenderBuffer> nullBuffer;
        bool loadStoreOpNoneSupported = false;
        bool nullDescriptorSupported = false;
        bool bufferMarkerSupported = false;   // AC6: VK_AMD_buffer_marker (writeBreadcrumb)
        bool breadcrumbsSynchronous = false;  // AC6: AC6_GPU_BREADCRUMBS=sync (writeBreadcrumb)
        bool deviceFaultSupported = false;    // AC6: VK_EXT_device_fault (VulkanDeviceFaultReport)
        // AC6: the Vulkan 1.3 path: dynamic rendering with the rasterizer and depth/stencil state dynamic, depth clamp
        // dynamic too where VK_EXT_extended_dynamic_state3 allows, and pipelines linked from libraries
        // (VK_EXT_graphics_pipeline_library with fast linking): per shader, compiled once, ahead of time.
        bool dynamicRenderingSupported = false;
        bool timelineSemaphoreSupported = false;   // AC6
        bool depthClampDynamicSupported = false;
        bool pipelineLibrariesSupported = false;
        std::mutex pipelineLibraryMutex;
        std::unordered_map<std::string, VkPipeline> pipelineLibraries;
        uint64_t pipelineLibrariesBuilt = 0;   // AC6: under pipelineLibraryMutex (diagnostics)
        VkPipeline pipelineLibrary(const std::string &key, const std::function<VkPipeline()> &create);
        VkPipeline preRasterizationLibrary(const VulkanPipelineLayout *layout, const VulkanShader *shader, const RenderSpecConstant *specConstants,
            uint32_t specConstantsCount, bool depthClamp);
        VkPipeline fragmentShaderLibrary(const VulkanPipelineLayout *layout, const VulkanShader *shader, const RenderSpecConstant *specConstants,
            uint32_t specConstantsCount);
        void prepareShaderLibrary(const RenderPipelineLayout *pipelineLayout, const RenderShader *shader, bool pixel,
            const RenderSpecConstant *specConstants, uint32_t specConstantsCount) override;
        bool usesShaderLibraries() const override { return pipelineLibrariesSupported; }

        VulkanDevice(VulkanInterface *renderInterface, const std::string &preferredDeviceName);
        ~VulkanDevice() override;
        std::unique_ptr<RenderDescriptorSet> createDescriptorSet(const RenderDescriptorSetDesc &desc) override;
        std::unique_ptr<RenderShader> createShader(const void *data, uint64_t size, const char *entryPointName, RenderShaderFormat format) override;
        std::unique_ptr<RenderSampler> createSampler(const RenderSamplerDesc &desc) override;
        std::unique_ptr<RenderPipeline> createComputePipeline(const RenderComputePipelineDesc &desc) override;
        std::unique_ptr<RenderPipeline> createGraphicsPipeline(const RenderGraphicsPipelineDesc &desc) override;
        std::unique_ptr<RenderPipeline> createRaytracingPipeline(const RenderRaytracingPipelineDesc &desc, const RenderPipeline *previousPipeline) override;
        std::unique_ptr<RenderCommandQueue> createCommandQueue(RenderCommandListType type) override;
        std::unique_ptr<RenderBuffer> createBuffer(const RenderBufferDesc &desc) override;
        std::unique_ptr<RenderTexture> createTexture(const RenderTextureDesc &desc) override;
        std::unique_ptr<RenderAccelerationStructure> createAccelerationStructure(const RenderAccelerationStructureDesc &desc) override;
        std::unique_ptr<RenderPool> createPool(const RenderPoolDesc &desc) override;
        std::unique_ptr<RenderPipelineLayout> createPipelineLayout(const RenderPipelineLayoutDesc &desc) override;
        std::unique_ptr<RenderCommandFence> createCommandFence() override;
        std::unique_ptr<RenderCommandSemaphore> createCommandSemaphore() override;
        std::unique_ptr<RenderFramebuffer> createFramebuffer(const RenderFramebufferDesc &desc) override;
        std::unique_ptr<RenderQueryPool> createQueryPool(uint32_t queryCount) override;
        void setBottomLevelASBuildInfo(RenderBottomLevelASBuildInfo &buildInfo, const RenderBottomLevelASMesh *meshes, uint32_t meshCount, bool preferFastBuild, bool preferFastTrace) override;
        void setTopLevelASBuildInfo(RenderTopLevelASBuildInfo &buildInfo, const RenderTopLevelASInstance *instances, uint32_t instanceCount, bool preferFastBuild, bool preferFastTrace) override;
        void setShaderBindingTableInfo(RenderShaderBindingTableInfo &tableInfo, const RenderShaderBindingGroups &groups, const RenderPipeline *pipeline, RenderDescriptorSet **descriptorSets, uint32_t descriptorSetCount) override;
        const RenderDeviceCapabilities &getCapabilities() const override;
        const RenderDeviceDescription &getDescription() const override;
        RenderSampleCounts getSampleCountsSupported(RenderFormat format) const override;
        void release();
        bool isValid() const;
        bool beginCapture() override;
        bool endCapture() override;
    };

    struct VulkanInterface : RenderInterface {
        VkInstance instance = VK_NULL_HANDLE;
        VkApplicationInfo appInfo = {};
        RenderInterfaceCapabilities capabilities;
        std::vector<std::string> deviceNames;

#   if PLUME_SDL_VULKAN_ENABLED
        VulkanInterface(RenderWindow sdlWindow);
#   else
        VulkanInterface();
#   endif

        ~VulkanInterface() override;
        std::unique_ptr<RenderDevice> createDevice(const std::string &preferredDeviceName) override;
        const RenderInterfaceCapabilities &getCapabilities() const override;
        const std::vector<std::string> &getDeviceNames() const override;
        bool isValid() const;
    };
};
