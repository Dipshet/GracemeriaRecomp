// AMD FSR 3.1 in place of the renderer's temporal AA, see fsr.h.
//
// AMD's Vulkan backend (thirdparty/FidelityFX-SDK, built as ac6_ffx) records FSR's compute passes straight into the
// frame's VkCommandBuffer: with deferred recording that happens on the replay worker, through a custom
// call in the list's order. The backend binds its own compute pipelines, layouts and descriptor sets there, so plume's
// record of what is bound starts over afterwards. External images are passed in the states their layouts are (shader
// read, or general for the output) and the backend puts them back in those states at the end of its dispatch.
#include "gpu/fsr.h"

#include "gpu/deferred_commands.h"
#include "gpu/video.h"
#include "log.h"

#include <plume_vulkan.h>

#include "gpu/ffx_compat.h"   // what the SDK's headers take from MSVC
#include <FidelityFX/host/ffx_fsr3upscaler.h>
#include <FidelityFX/host/backends/vk/ffx_vk.h>

#include <algorithm>
#include <utility>
#include <cstdlib>
#include <format>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

// The backend's per-job hook (ffx_vk.cpp, AC6).
extern void (*ac6FfxJobHook)(VkCommandBuffer commandBuffer, const wchar_t* label);

namespace ac6::gpu::fsr {

namespace {

struct State
{
    bool failed = false;
    bool backend = false;
    bool context = false;
    VkDeviceContext device{};
    std::vector<uint8_t> scratch;
    FfxInterface backendInterface{};
    FfxFsr3UpscalerContext fsr{};
    // The images the upscaler shares with other effects (frame generation): dilated depth, dilated motion vectors, the
    // reconstructed previous nearest depth. The application makes them (as AMD's FSR 3 component does), in a backend
    // context of their own.
    uint32_t sharedContext = 0;
    FfxResourceInternal shared[3] = {};
    uint32_t maxRenderWidth = 0, maxRenderHeight = 0, upscaleWidth = 0, upscaleHeight = 0;
    bool hdr = false;   // made for a high dynamic range colour image (a float scene)
};
State& s_fsr = *new State;

void Message(FfxMsgType type, const wchar_t* message)
{
    std::string text;
    for (const wchar_t* c = message; c && *c; c++)
        text += *c < 128 ? char(*c) : '?';
    if (type == FFX_MESSAGE_TYPE_ERROR)
        log::Error("FSR: {}", text);
    else
        log::Warn("FSR: {}", text);
}

FfxSurfaceFormat SurfaceFormat(plume::RenderFormat format)
{
    switch (format)
    {
    case plume::RenderFormat::B8G8R8A8_UNORM: return FFX_SURFACE_FORMAT_B8G8R8A8_UNORM;
    case plume::RenderFormat::R8G8B8A8_UNORM: return FFX_SURFACE_FORMAT_R8G8B8A8_UNORM;
    case plume::RenderFormat::R16G16B16A16_FLOAT: return FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    case plume::RenderFormat::R16G16_FLOAT: return FFX_SURFACE_FORMAT_R16G16_FLOAT;
    case plume::RenderFormat::R32_FLOAT: return FFX_SURFACE_FORMAT_R32_FLOAT;
    case plume::RenderFormat::R8_UNORM: return FFX_SURFACE_FORMAT_R8_UNORM;   // the reactive mask
    case plume::RenderFormat::R8G8_UNORM: return FFX_SURFACE_FORMAT_R8G8_UNORM;
    case plume::RenderFormat::R16_FLOAT: return FFX_SURFACE_FORMAT_R16_FLOAT;
    case plume::RenderFormat::R16_UNORM: return FFX_SURFACE_FORMAT_R16_UNORM;
    case plume::RenderFormat::R32G32_FLOAT: return FFX_SURFACE_FORMAT_R32G32_FLOAT;
    case plume::RenderFormat::R10G10B10A2_UNORM: return FFX_SURFACE_FORMAT_R10G10B10A2_UNORM;
    default:
    {
        // The backend would make its view of the image with no format (undefined: a driver may read anything).
        static bool logged = false;
        if (!std::exchange(logged, true))
            log::Error("FSR: an image format with no FidelityFX format ({})", int(format));
        return FFX_SURFACE_FORMAT_UNKNOWN;
    }
    }
}

FfxResource Resource(const Image& image, bool output, const wchar_t* name)
{
    FfxResourceDescription description{};
    description.type = FFX_RESOURCE_TYPE_TEXTURE2D;
    description.format = SurfaceFormat(image.format);
    description.width = image.width;
    description.height = image.height;
    description.depth = 1;
    description.mipCount = 1;
    description.flags = FFX_RESOURCE_FLAGS_NONE;
    description.usage = output ? FFX_RESOURCE_USAGE_UAV : FFX_RESOURCE_USAGE_READ_ONLY;
    const VkImage vk = static_cast<const plume::VulkanTexture*>(image.texture)->vk;
    return ffxGetResourceVK(reinterpret_cast<void*>(vk), description, name,
        output ? FFX_RESOURCE_STATE_UNORDERED_ACCESS : FFX_RESOURCE_STATE_COMPUTE_READ);
}

// The backend's device function lookup. It asks for some functions by their extension names (vkGetBufferMemoryRequirements2KHR)
// when the device merely supports the extension. Promoted to core (Vulkan 1.1+), they are there under the core name only
// unless the extension was enabled, which plume doesn't do: the backend would call a null pointer.
PFN_vkVoidFunction DeviceProcAddr(VkDevice device, const char* name)
{
    if (PFN_vkVoidFunction function = vkGetDeviceProcAddr(device, name))
        return function;
    const std::string_view view(name);
    for (std::string_view suffix : { "KHR", "EXT" })
        if (view.ends_with(suffix))
            return vkGetDeviceProcAddr(device, std::string(view.substr(0, view.size() - suffix.size())).c_str());
    return nullptr;
}

// The context for these sizes: made once, made again (after the GPU has finished with the old one) when the render size
// grows past what it was made for, the output size changes or the colour image changes between 8-bit and float.
bool EnsureContext(plume::RenderDevice& renderDevice, uint32_t renderWidth, uint32_t renderHeight, uint32_t upscaleWidth, uint32_t upscaleHeight,
    bool hdr)
{
    if (s_fsr.context && renderWidth <= s_fsr.maxRenderWidth && renderHeight <= s_fsr.maxRenderHeight && upscaleWidth == s_fsr.upscaleWidth &&
        upscaleHeight == s_fsr.upscaleHeight && hdr == s_fsr.hdr)
        return true;
    auto& device = static_cast<plume::VulkanDevice&>(renderDevice);
    FfxInterface& backend = s_fsr.backendInterface;
    if (s_fsr.context)
    {
        // Every frame that used the old context done: the present thread idle and the frames in flight finished
        // (vkDeviceWaitIdle would race the present thread's submits on the same queue).
        g_video.WaitGpuIdle();
        for (FfxResourceInternal& resource : s_fsr.shared)
            backend.fpDestroyResource(&backend, resource, s_fsr.sharedContext);
        backend.fpDestroyBackendContext(&backend, s_fsr.sharedContext);
        ffxFsr3UpscalerContextDestroy(&s_fsr.fsr);
        s_fsr.context = false;
    }
    if (!s_fsr.backend)
    {
        // Two effect contexts: the upscaler's and its shared images'.
        s_fsr.device = { device.vk, device.physicalDevice, DeviceProcAddr };
        s_fsr.scratch.resize(ffxGetScratchMemorySizeVK(device.physicalDevice, 2));
        if (ffxGetInterfaceVK(&s_fsr.backendInterface, ffxGetDeviceVK(&s_fsr.device), s_fsr.scratch.data(), s_fsr.scratch.size(), 2) != FFX_OK)
        {
            log::Error("FSR: the Vulkan backend could not be made; the scene is scaled up without anti-aliasing");
            s_fsr.failed = true;
            return false;
        }
        s_fsr.backend = true;
    }
    FfxFsr3UpscalerContextDescription description{};
    description.flags = FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED | FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE;
    // A float scene (HDR from the scene): values above 1, which FSR compresses for its own accumulation.
    if (hdr)
        description.flags |= FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE;
    // AC6_FSR_AUTOEXP=0 (profiling): FSR's own exposure estimate off (the image is LDR, its exposure then 1).
    static const bool autoExposure = [] { const char* v = std::getenv("AC6_FSR_AUTOEXP"); return !(v && *v == '0'); }();
    if (autoExposure)
        description.flags |= FFX_FSR3UPSCALER_ENABLE_AUTO_EXPOSURE;
    static const bool debugChecks = [] { const char* v = std::getenv("AC6_FSR_CHECKS"); return v && *v == '1'; }();
    if (debugChecks)
        description.flags |= FFX_FSR3UPSCALER_ENABLE_DEBUG_CHECKING;
    description.maxRenderSize = { renderWidth, renderHeight };
    description.maxUpscaleSize = { upscaleWidth, upscaleHeight };
    description.fpMessage = Message;
    description.backendInterface = s_fsr.backendInterface;
    if (const FfxErrorCode error = ffxFsr3UpscalerContextCreate(&s_fsr.fsr, &description); error != FFX_OK)
    {
        log::Error("FSR: the context could not be made (error {:X}); the scene is scaled up without anti-aliasing", uint32_t(error));
        s_fsr.failed = true;
        return false;
    }
    FfxFsr3UpscalerSharedResourceDescriptions shared{};
    bool sharedMade = ffxFsr3UpscalerGetSharedResourceDescriptions(&s_fsr.fsr, &shared) == FFX_OK &&
        backend.fpCreateBackendContext(&backend, FFX_EFFECT_SHAREDRESOURCES, nullptr, &s_fsr.sharedContext) == FFX_OK;
    FfxCreateResourceDescription* descriptions[3] = { &shared.dilatedDepth, &shared.dilatedMotionVectors, &shared.reconstructedPrevNearestDepth };
    for (uint32_t i = 0; i < 3 && sharedMade; i++)
        sharedMade = backend.fpCreateResource(&backend, descriptions[i], s_fsr.sharedContext, &s_fsr.shared[i]) == FFX_OK;
    if (!sharedMade)
    {
        log::Error("FSR: its shared images could not be made; the scene is scaled up without anti-aliasing");
        s_fsr.failed = true;
        return false;
    }
    s_fsr.context = true;
    s_fsr.maxRenderWidth = renderWidth;
    s_fsr.maxRenderHeight = renderHeight;
    s_fsr.upscaleWidth = upscaleWidth;
    s_fsr.upscaleHeight = upscaleHeight;
    s_fsr.hdr = hdr;
    log::Info("FSR 3.1: context for {}x{} -> {}x{}{}", renderWidth, renderHeight, upscaleWidth, upscaleHeight, hdr ? " (a float scene)" : "");
    return true;
}

struct Call
{
    FfxFsr3UpscalerContext* context;
    FfxFsr3UpscalerDispatchDescription dispatch;
};

// AC6_FSR_PROFILE=1: the GPU time of each of FSR's passes, from a timestamp before each of the backend's jobs
// (ac6FfxJobHook, an AC6 hook in ffx_vk.cpp), read four dispatches later and logged every 600.
struct Profiler
{
    static constexpr uint32_t kSlots = 4, kMaxStamps = 48;
    VkDevice device = VK_NULL_HANDLE;
    double nsPerTick = 1.0;
    VkQueryPool pools[kSlots] = {};
    std::vector<std::string> labels[kSlots];
    bool pending[kSlots] = {};
    uint32_t slot = 0;
    std::vector<std::pair<std::string, double>> totals;   // in order of first appearance
    uint32_t dispatches = 0;
};
Profiler* s_profiler = nullptr;

bool Profiling()
{
    static const bool on = [] { const char* v = std::getenv("AC6_FSR_PROFILE"); return v && *v == '1'; }();
    return on;
}

void ProfileHook(VkCommandBuffer commandBuffer, const wchar_t* label)
{
    Profiler& p = *s_profiler;
    auto& labels = p.labels[p.slot];
    if (labels.size() >= Profiler::kMaxStamps)
        return;
    std::string name = label ? "" : "(end)";
    for (const wchar_t* c = label; c && *c; c++)
        name += *c < 128 ? char(*c) : '?';
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, p.pools[p.slot], uint32_t(labels.size()));
    labels.push_back(name.empty() ? "(unnamed)" : name);
}

// Before a dispatch: the oldest slot's times taken, the slot reset for this one.
void ProfileBegin(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandBuffer commandBuffer)
{
    if (!s_profiler)
    {
        s_profiler = new Profiler;
        s_profiler->device = device;
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        s_profiler->nsPerTick = properties.limits.timestampPeriod;
        VkQueryPoolCreateInfo info{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        info.queryCount = Profiler::kMaxStamps;
        for (VkQueryPool& pool : s_profiler->pools)
            vkCreateQueryPool(device, &info, nullptr, &pool);
    }
    Profiler& p = *s_profiler;
    p.slot = (p.slot + 1) % Profiler::kSlots;
    auto& labels = p.labels[p.slot];
    if (p.pending[p.slot] && labels.size() > 1)
    {
        uint64_t stamps[Profiler::kMaxStamps] = {};
        if (vkGetQueryPoolResults(device, p.pools[p.slot], 0, uint32_t(labels.size()), sizeof(stamps), stamps, sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS)
        {
            for (size_t i = 0; i + 1 < labels.size(); i++)
            {
                const double ns = double(stamps[i + 1] - stamps[i]) * p.nsPerTick;
                auto it = std::find_if(p.totals.begin(), p.totals.end(), [&](const auto& t) { return t.first == labels[i]; });
                if (it == p.totals.end())
                    p.totals.push_back({ labels[i], ns });
                else
                    it->second += ns;
            }
            if (++p.dispatches == 600)
            {
                double all = 0.0;
                for (const auto& [label, ns] : p.totals)
                    all += ns;
                std::string line;
                for (const auto& [label, ns] : p.totals)
                    line += std::format(" {} {:.3f}", label, ns / 1e6 / p.dispatches);
                log::Info("FSR profile ({} dispatches, ms each): all {:.3f};{}", p.dispatches, all / 1e6 / p.dispatches, line);
                p.totals.clear();
                p.dispatches = 0;
            }
        }
    }
    labels.clear();
    p.pending[p.slot] = true;
    vkCmdResetQueryPool(commandBuffer, p.pools[p.slot], 0, Profiler::kMaxStamps);
}

// On the list itself (the replay worker's, with deferred recording).
void Run(plume::RenderCommandList& target, const void* data)
{
    Call call;
    std::memcpy(&call, data, sizeof(call));
    auto& list = static_cast<plume::VulkanCommandList&>(target);
    // plume holds barriers back for its next command (an AC6 change): the layouts the backend is told must be there first.
    list.endActiveRenderPass();
    list.flushBarriers();
    call.dispatch.commandList = ffxGetCommandListVK(list.vk);
    // FFX's compute after the passes that wrote its inputs (the scene, depth and motion: colour and depth attachments,
    // copies), and the passes after it (the copy back: a fragment shader, copies) after its writes. plume's barriers
    // name graphics and copy stages only, and FFX's own are compute to compute.
    VkMemoryBarrier before{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    before.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
        VK_ACCESS_SHADER_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(list.vk, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
        1, &before, 0, nullptr, 0, nullptr);
    if (Profiling())
    {
        ProfileBegin(s_fsr.device.vkDevice, s_fsr.device.vkPhysicalDevice, list.vk);
        ac6FfxJobHook = ProfileHook;
    }
    if (const FfxErrorCode error = ffxFsr3UpscalerContextDispatch(call.context, &call.dispatch); error != FFX_OK)
    {
        static uint32_t reported = 0;
        if (reported++ < 4)
            log::Error("FSR: dispatch failed (error {:X})", uint32_t(error));
    }
    ac6FfxJobHook = nullptr;
    VkMemoryBarrier after{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(list.vk, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
        1, &after, 0, nullptr, 0, nullptr);
    list.activeComputePipelineLayout = nullptr;
    list.activeGraphicsPipelineLayout = nullptr;
    list.appliedDynamicValid = false;
}

} // namespace

} // namespace ac6::gpu::fsr

// The backend's interface points at the frame generation swapchain's configuration (its FrameInterpolationSwapchain
// sources, which only frame generation needs and which aren't built): never called for the upscaler.
FFX_API FfxErrorCode ffxSetFrameGenerationConfigToSwapchainVK(FfxFrameGenerationConfig const*)
{
    return FFX_ERROR_INVALID_ARGUMENT;
}

namespace ac6::gpu::fsr {

bool Requested()
{
    static const bool requested = [] { const char* v = std::getenv("AC6_AA"); return v && std::string_view(v).find("fsr") != std::string_view::npos; }();
    return requested;
}

float RenderRatio()
{
    static const float ratio = [] {
        if (const char* v = std::getenv("AC6_FSR_SCALE"); v && *v)
            return std::clamp(float(std::atof(v)), 0.33f, 1.0f);
        const char* q = std::getenv("AC6_FSR_QUALITY");
        const std::string_view mode = q ? q : "";
        if (mode == "quality")
            return 1.0f / 1.5f;
        if (mode == "balanced")
            return 1.0f / 1.7f;
        if (mode == "performance")
            return 0.5f;
        if (mode == "ultra" || mode == "ultra_performance")
            return 1.0f / 3.0f;
        if (!mode.empty() && mode != "native")
            log::Warn("AC6_FSR_QUALITY={}: expected native, quality, balanced, performance or ultra", mode);
        return 1.0f;
    }();
    return ratio;
}

void Jitter(uint64_t frame, int32_t renderWidth, int32_t outputWidth, float& x, float& y)
{
    const int32_t phases = ffxFsr3UpscalerGetJitterPhaseCount(renderWidth, outputWidth);
    x = y = 0.0f;
    ffxFsr3UpscalerGetJitterOffset(&x, &y, int32_t(frame % uint64_t(std::max(phases, 1))), phases);
}

bool Failed()
{
    return s_fsr.failed;
}

bool Dispatch(plume::RenderDevice& device, plume::RenderCommandList& cl, DeferredCommandList* deferred, const Frame& frame)
{
    if (s_fsr.failed || !frame.color.texture || !frame.depth.texture || !frame.motion.texture || !frame.output.texture)
        return false;
    // Made for render sizes up to the output's: an upscaled frame and a native one (the first after a load, before the scene's
    // surfaces are known) share it, with no remake (and its wait for the GPU) between.
    if (!EnsureContext(device, std::max(frame.color.width, frame.output.width), std::max(frame.color.height, frame.output.height),
            frame.output.width, frame.output.height, frame.color.format == plume::RenderFormat::R16G16B16A16_FLOAT))
        return false;
    Call call{};
    call.context = &s_fsr.fsr;
    FfxFsr3UpscalerDispatchDescription& d = call.dispatch;
    d.color = Resource(frame.color, false, L"AC6_Color");
    d.depth = Resource(frame.depth, false, L"AC6_Depth");
    d.motionVectors = Resource(frame.motion, false, L"AC6_MotionVectors");
    if (frame.reactive.texture)
        d.reactive = Resource(frame.reactive, false, L"AC6_Reactive");
    d.output = Resource(frame.output, true, L"AC6_Output");
    FfxInterface& backend = s_fsr.backendInterface;
    d.dilatedDepth = backend.fpGetResource(&backend, s_fsr.shared[0]);
    d.dilatedMotionVectors = backend.fpGetResource(&backend, s_fsr.shared[1]);
    d.reconstructedPrevNearestDepth = backend.fpGetResource(&backend, s_fsr.shared[2]);
    d.jitterOffset = { frame.jitterX, frame.jitterY };
    d.motionVectorScale = { frame.motionScaleX, frame.motionScaleY };
    d.renderSize = { frame.color.width, frame.color.height };
    d.upscaleSize = { frame.output.width, frame.output.height };
    d.enableSharpening = frame.sharpness > 0.0f;
    d.sharpness = frame.sharpness;
    d.frameTimeDelta = frame.frameTimeMs;
    d.preExposure = 1.0f;
    d.reset = frame.reset;
    d.cameraNear = frame.cameraNear;
    d.cameraFar = 1.0e7f;   // unused with an infinite far plane
    d.cameraFovAngleVertical = frame.fovY;
    d.viewSpaceToMetersFactor = 1.0f;
    d.flags = frame.debugView ? FFX_FSR3UPSCALER_DISPATCH_DRAW_DEBUG_VIEW : 0;
    if (deferred)
        deferred->custom(Run, &call, sizeof(call));
    else
        Run(cl, &call);
    return true;
}

} // namespace ac6::gpu::fsr
