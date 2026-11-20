#include "gpu/video.h"

#include "crash.h"
#include "game/game_modes.h"
#include "gpu/breadcrumbs.h"
#include "gpu/capture.h"
#include "gpu/deferred_commands.h"
#include "gpu/frame_rate.h"
#include "gpu/renderer.h"
#include "input/input.h"
#include "input/kbm.h"
#include "log.h"
#include "platform/platform.h"
#include "settings.h"
#include "system_info.h"
#include "ui/overlay.h"
#include "plume_vulkan.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <format>
#include <string_view>
#include <thread>

namespace plume {
extern std::unique_ptr<RenderInterface> CreateVulkanInterface();
}

namespace ac6::gpu {

Video g_video;

namespace {

constexpr plume::RenderFormat kSwapChainFormat = plume::RenderFormat::B8G8R8A8_UNORM;
constexpr uint32_t kSwapChainImages = 3;

} // namespace

// AC6_MONITOR=<n>: the monitor to play on, the n-th SDL lists (the settings overlay's numbers). The
// primary by default, or when there is no n-th.
SDL_DisplayID ChosenDisplay()
{
    const SDL_DisplayID primary = SDL_GetPrimaryDisplay();
    const std::string v = settings::Get("AC6_MONITOR");
    if (v.empty())
        return primary;
    int count = 0;
    SDL_DisplayID* ids = SDL_GetDisplays(&count);
    const int n = std::atoi(v.c_str());
    SDL_DisplayID chosen = primary;
    if (ids && n >= 1 && n <= count)
        chosen = ids[n - 1];
    else
        log::Warn("AC6_MONITOR={}: no such monitor ({} connected): the primary", v, count);
    SDL_free(ids);
    return chosen;
}

// The monitor's modes are logged, AC6_RESOLUTION picks one (any of them, not only multiples of 1280x720) and
// AC6_REFRESH its refresh rate. Returns the mode for exclusive fullscreen (null: none, borderless or a window then).
const SDL_DisplayMode* Video::ChooseDisplayMode()
{
    display = ChosenDisplay();
    const SDL_DisplayMode* desktop = display ? SDL_GetDesktopDisplayMode(display) : nullptr;
    int count = 0;
    SDL_DisplayMode** modes = display ? SDL_GetFullscreenDisplayModes(display, &count) : nullptr;
    if (desktop)
        refreshHz = desktop->refresh_rate > 0.0f ? desktop->refresh_rate : 60.0f;
    std::string list;
    for (int i = 0; i < count; i++)
        list += std::format("{}{}x{}@{:.2f}", i ? ", " : "", modes[i]->w, modes[i]->h, modes[i]->refresh_rate);
    if (desktop)
        log::Info("Display: {}, desktop {}x{} at {:.2f} Hz; modes: {}", SDL_GetDisplayName(display) ? SDL_GetDisplayName(display) : "?",
            desktop->w, desktop->h, desktop->refresh_rate, list);
    const SDL_DisplayMode* chosen = nullptr;
    const char* resolution = std::getenv("AC6_RESOLUTION");
    if (!resolution || !*resolution)
        resolution = nativeByDefault ? "native" : nullptr;
    if (const char* v = resolution)
    {
        int w = 0, h = 0;
        if (std::string_view(v) == "native" && desktop)
        {
            w = desktop->w, h = desktop->h;
            nativeSize = true;
        }
        else if (std::sscanf(v, "%dx%d", &w, &h) != 2)
            w = h = 0;
        float wantHz = 0.0f;
        if (const char* r = std::getenv("AC6_REFRESH"); r && *r)
            wantHz = float(std::atof(r));
        for (int i = 0; i < count; i++)
        {
            const SDL_DisplayMode* m = modes[i];
            if (m->w != w || m->h != h)
                continue;
            // The refresh rate asked for (the nearest), else the highest.
            if (!chosen || (wantHz > 0.0f ? std::fabs(m->refresh_rate - wantHz) < std::fabs(chosen->refresh_rate - wantHz)
                                          : m->refresh_rate > chosen->refresh_rate))
                chosen = m;
        }
        if (chosen)
        {
            outputWidth = uint32_t(chosen->w);
            outputHeight = uint32_t(chosen->h);
            refreshHz = chosen->refresh_rate;
        }
        else if (w > 0 && h > 0)
        {
            // Any size: the window's and the frame's. Exclusive fullscreen needs a mode (Init).
            // At least 640x360.
            outputWidth = uint32_t(std::clamp(w, 640, 16384));
            outputHeight = uint32_t(std::clamp(h, 360, 16384));
            log::Info("AC6_RESOLUTION={}: not one of the monitor's modes: a window that size (or borderless, the frame scaled)", v);
        }
        else
            log::Warn("AC6_RESOLUTION={}: expected <width>x<height> or native", v);
    }
    // The largest 16:9 frame inside the output: the guest's 1280x720 times one scale for both axes.
    // AC6_RENDER_RESOLUTION=<width>x<height> renders at another size than the output's (any: supersampling, tests).
    uint32_t fitWidth = outputWidth, fitHeight = outputHeight;
    if (const char* v = std::getenv("AC6_RENDER_RESOLUTION"); v && *v)
    {
        int w = 0, h = 0;
        if (std::sscanf(v, "%dx%d", &w, &h) == 2 && w > 0 && h > 0)
            fitWidth = uint32_t(w), fitHeight = uint32_t(h);
        else
            log::Warn("AC6_RENDER_RESOLUTION={}: expected <width>x<height>", v);
    }
    renderScale = std::min(float(fitWidth) / float(kGuestWidth), float(fitHeight) / float(kGuestHeight));
    renderWidth = std::max(1u, uint32_t(std::lround(kGuestWidth * double(renderScale))));
    renderHeight = std::max(1u, uint32_t(std::lround(kGuestHeight * double(renderScale))));
    renderScaleX = renderScaleY = renderScale;
    // AC6_WIDESCREEN (the aspect ratio: auto, the default, 0 = 16:9): an output of
    // another shape than 16:9 is filled instead of pillarboxed or letterboxed: wider ones hor+, narrower ones (16:10, 3:2,
    // 4:3) vert+ (game/widescreen.cpp).
    static const bool widescreen = [] { const char* v = std::getenv("AC6_WIDESCREEN"); return !v || *v != '0'; }();
    const int64_t shape = int64_t(fitWidth) * kGuestHeight - int64_t(fitHeight) * kGuestWidth;
    if (widescreen && (shape > int64_t(kGuestHeight) || shape < -int64_t(kGuestWidth)))
    {
        renderWidth = fitWidth;
        renderHeight = fitHeight;
        renderScaleX = float(fitWidth) / float(kGuestWidth);
        renderScaleY = float(fitHeight) / float(kGuestHeight);
        renderScale = std::min(renderScaleX, renderScaleY);
    }
    // The frame (the UI's size and the output target's), and its 3D at AC6_RENDER_SCALE times that.
    frameWidth = renderWidth;
    frameHeight = renderHeight;
    frameScaleX = renderScaleX;
    frameScaleY = renderScaleY;
    // Not with FSR (AC6_AA=fsr), whose quality setting picks the scene's size (the overlay greys the row).
    const char* aa = std::getenv("AC6_AA");
    const bool fsr = aa && std::string_view(aa).find("fsr") != std::string_view::npos;
    if (const char* v = std::getenv("AC6_RENDER_SCALE"); v && *v && fsr)
        log::Info("AC6_RENDER_SCALE={}: not with FSR (AC6_FSR_QUALITY picks the scene's size)", v);
    else if (const char* v = std::getenv("AC6_RENDER_SCALE"); v && *v && !std::getenv("AC6_RENDER_RESOLUTION"))
    {
        double scale = std::atof(v);
        // The 3D at least half the console's resolution on either axis (0.5 host pixels per guest pixel: 640x360 at
        // 16:9): smaller, guest images under a host pixel break surface creation.
        const double floorScale = 0.5 / std::min(double(frameScaleX), double(frameScaleY));
        if (scale >= 0.25 && scale <= 4.0 && scale < floorScale)
        {
            log::Info("AC6_RENDER_SCALE={}: below the minimum (the 3D at half the console's resolution): {:.3f}", v, floorScale);
            scale = floorScale;
        }
        if (scale >= 0.25 && scale <= 4.0)
        {
            if (std::fabs(scale - 1.0) > 1e-6)
            {
                renderWidth = std::max(1u, uint32_t(std::lround(frameWidth * scale)));
                renderHeight = std::max(1u, uint32_t(std::lround(frameHeight * scale)));
                renderScaleX = float(renderWidth) / float(kGuestWidth);
                renderScaleY = float(renderHeight) / float(kGuestHeight);
                renderScale = std::min(renderScaleX, renderScaleY);
            }
        }
        else
            log::Warn("AC6_RENDER_SCALE={}: expected 0.25 to 4", v);
    }
    // The kept copy of the mode outlives SDL's array.
    static SDL_DisplayMode kept;
    if (chosen)
        kept = *chosen;
    SDL_free(modes);
    return chosen ? &kept : nullptr;
}

bool Video::Init()
{
#if defined(__linux__)
    // The Vulkan surface is an Xlib one (plume): X11, or XWayland on Wayland desktops. (SDL_VIDEO_DRIVER overrides it.)
    SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "x11");
#endif
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD | SDL_INIT_EVENTS))
    {
        log::Error("SDL_Init failed: {}", SDL_GetError());
        return false;
    }
    input::Init();
    if (const char* v = std::getenv("AC6_BACKGROUND"); v && *v == '1')
        background = true;
    if (const char* v = std::getenv("AC6_FRAME_PACING"); v && *v)
        framePacing = *v != '0';
    if (const char* v = std::getenv("AC6_ASYNC_PRESENT"); v && *v == '0')
        asyncPresent = false;
    log::Info("Frame pacing: {}", framePacing ? "the game's present interval" : "off (host pace)");
    if (background)
    {
        SDL_SetHint(SDL_HINT_WINDOW_ACTIVATE_WHEN_SHOWN, "0");
        SDL_SetHint(SDL_HINT_WINDOW_ACTIVATE_WHEN_RAISED, "0");
    }
    const SDL_DisplayMode* mode = ChooseDisplayMode();
    // AC6_FULLSCREEN=1 (exclusive) | borderless (the default) | 0 (a window, background runs' default: tests and replays
    // don't cover the screen).
    const char* fullscreenEnv = std::getenv("AC6_FULLSCREEN");
    if (!fullscreenEnv || !*fullscreenEnv)
        fullscreenEnv = background ? "0" : "borderless";
    fullscreenNow = fullscreenEnv;
    refreshNow = settings::Get("AC6_REFRESH");
    const bool exclusive = fullscreenEnv && *fullscreenEnv == '1' && mode;
    // Exclusive fullscreen at a size that isn't one of the monitor's modes: borderless, the frame scaled.
    const bool borderless = fullscreenEnv && (std::string_view(fullscreenEnv) == "borderless" || (*fullscreenEnv == '1' && !mode));
    if (fullscreenEnv && *fullscreenEnv == '1' && !mode)
        log::Info("Fullscreen: exclusive needs one of the monitor's modes; borderless instead");
    // On the monitor chosen (centred), fullscreen there too (SDL takes the window's display).
    {
        const SDL_PropertiesID create = SDL_CreateProperties();
        SDL_SetStringProperty(create, SDL_PROP_WINDOW_CREATE_TITLE_STRING, "ACE COMBAT 6: FIRES OF LIBERATION");
        SDL_SetNumberProperty(create, SDL_PROP_WINDOW_CREATE_X_NUMBER, SDL_WINDOWPOS_CENTERED_DISPLAY(display));
        SDL_SetNumberProperty(create, SDL_PROP_WINDOW_CREATE_Y_NUMBER, SDL_WINDOWPOS_CENTERED_DISPLAY(display));
        SDL_SetNumberProperty(create, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, int(outputWidth));
        SDL_SetNumberProperty(create, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, int(outputHeight));
        SDL_SetBooleanProperty(create, SDL_PROP_WINDOW_CREATE_RESIZABLE_BOOLEAN, true);
        window = SDL_CreateWindowWithProperties(create);
        SDL_DestroyProperties(create);
    }
    if (window)
        input::kbm::Init(window);
    if (!window)
    {
        log::Error("SDL_CreateWindow failed: {}", SDL_GetError());
        return false;
    }
    // A window the desktop's size: maximised (its frame and the taskbar stay, the picture is fitted into it).
    if (!exclusive && !borderless && nativeSize)
        SDL_MaximizeWindow(window);
    ReadDisplayInfo();
    if (exclusive || borderless)
    {
        // Exclusive: the chosen mode (resolution and refresh rate). Borderless: the desktop's.
        if (!SDL_SetWindowFullscreenMode(window, exclusive ? mode : nullptr) || !SDL_SetWindowFullscreen(window, true))
            log::Warn("Fullscreen failed: {}", SDL_GetError());
        SDL_SyncWindow(window);
    }
    log::Info("Output {}x{} at {:.2f} Hz ({}), guest frame rendered at {}x{} (scale {:.4f} x {:.4f}{}){}", outputWidth, outputHeight, refreshHz,
        exclusive ? "exclusive fullscreen" : borderless ? "borderless fullscreen" : "window", renderWidth, renderHeight, renderScaleX, renderScaleY,
        WideFactor() != 1.0f ? std::format(", {}: {:.4f}x 16:9", WideFactor() > 1.0f ? "wide" : "narrow", WideFactor()) : std::string(),
        frameWidth != renderWidth || frameHeight != renderHeight ? std::format("; the UI and the frame at {}x{} (AC6_RENDER_SCALE)", frameWidth, frameHeight)
                                                                 : std::string());

    // The Vulkan backend's errors into the log (they go to stderr otherwise, which the windowed build doesn't have).
    plume::SetVulkanErrorCallback([](const char* message) {
        static std::atomic<uint32_t> count{ 0 };
        const uint32_t n = count++;
        if (n < 100)
            log::Error("Vulkan: {}", message);
        else if (n == 100)
            log::Error("Vulkan: further errors aren't logged");
    });
    renderInterface = plume::CreateVulkanInterface();
    if (!renderInterface)
    {
        log::Error("No Vulkan: the game runs without a window image (headless)");
        return false;
    }
    // AC6_GPU=<part of a name> picks that GPU (e.g. a laptop's integrated one). Otherwise plume prefers a discrete
    // GPU. The GPUs found are logged.
    std::string preferred;
    {
        const char* want = std::getenv("AC6_GPU");
        std::string list;
        for (const std::string& name : renderInterface->getDeviceNames())
        {
            list += (list.empty() ? "" : ", ") + name;
            std::string lowerName = name, lowerWant = want ? want : "";
            for (auto* t : { &lowerName, &lowerWant })
                for (char& c : *t)
                    c = char(std::tolower(uint8_t(c)));
            if (want && *want && preferred.empty() && lowerName.find(lowerWant) != std::string::npos)
                preferred = name;
        }
        log::Info("Vulkan GPUs: {}", list);
        if (want && *want && preferred.empty())
            log::Warn("AC6_GPU={}: no GPU's name contains it", want);
    }
    device = renderInterface->createDevice(preferred);
    if (!device)
    {
        log::Error("No usable Vulkan device: the game runs headless");
        renderInterface.reset();
        return false;
    }
    {
        const plume::RenderDeviceDescription& d = device->getDescription();
        log::Info("Vulkan device: {} (driver {}, {} MB video memory)", d.name,
            system_info::DriverVersion(uint32_t(d.vendor), d.driverVersion), d.dedicatedVideoMemory / (1024 * 1024));
    }
    // The renderer needs Vulkan 1.2 (buffer device addresses, descriptor indexing, timeline semaphores). On an older
    // driver the first of those calls is to a function the driver doesn't have. The game stops with a message instead.
    if (const auto* vk = dynamic_cast<const plume::VulkanDevice*>(device.get()); vk && vk->physicalDeviceProperties.apiVersion < VK_API_VERSION_1_2)
    {
        const uint32_t v = vk->physicalDeviceProperties.apiVersion;
        log::Error("{} offers Vulkan {}.{}.{}: the game needs Vulkan 1.2 or newer (update the graphics driver)", device->getDescription().name,
            VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v), VK_API_VERSION_PATCH(v));
        static const std::string reason = std::format("the graphics driver offers Vulkan {}.{}, the game needs 1.2 or newer. "
            "Update the graphics driver", VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v));
        crash::Shutdown(1, reason.c_str());
    }
    if (const auto* vk = dynamic_cast<const plume::VulkanDevice*>(device.get()))
        log::Info("Vulkan paths: {}, {}", vk->dynamicRenderingSupported ? "dynamic rendering" : "render passes",
            vk->pipelineLibrariesSupported ? "pipeline libraries" : "whole pipelines");
    // The game's largest surface (the hangar's 1280x1440, two 720-line halves) at the render scale must fit the GPU's
    // image size limit: render scale 4 at a 4K output asks 17280 lines of a GPU that makes 16384.
    if (const uint64_t limit = device->getCapabilities().maxTextureSize; limit > 0)
    {
        constexpr uint32_t kLargestGuestHeight = 2 * kGuestHeight;
        const uint64_t needX = uint64_t(std::ceil(double(renderScaleX) * kGuestWidth)), needY = uint64_t(std::ceil(double(renderScaleY) * kLargestGuestHeight));
        if (needX > limit || needY > limit)
        {
            const double fit = std::min(double(limit) / double(needX), double(limit) / double(needY));
            const uint32_t oldWidth = renderWidth, oldHeight = renderHeight;
            renderWidth = std::max(1u, uint32_t(std::floor(renderWidth * fit)));
            renderHeight = std::max(1u, uint32_t(std::floor(renderHeight * fit)));
            renderScaleX = float(renderWidth) / float(kGuestWidth);
            renderScaleY = float(renderHeight) / float(kGuestHeight);
            renderScale = std::min(renderScaleX, renderScaleY);
            log::Warn("Render scale: the 3D at {}x{} instead of {}x{}: the GPU's images are at most {} pixels", renderWidth,
                renderHeight, oldWidth, oldHeight, limit);
        }
    }

#if defined(_WIN32)
    auto hwnd = static_cast<HWND>(SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
    renderWindow = hwnd;
#else
    const SDL_PropertiesID properties = SDL_GetWindowProperties(window);
    renderWindow = {};
    renderWindow.display = static_cast<Display*>(SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, nullptr));
    renderWindow.window = ::Window(SDL_GetNumberProperty(properties, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0));
    if (!renderWindow.display || !renderWindow.window)
    {
        log::Error("No X11 window (SDL video driver {}): the game runs headless", SDL_GetCurrentVideoDriver() ? SDL_GetCurrentVideoDriver() : "none");
        device.reset();
        renderInterface.reset();
        return false;
    }
#endif

    queue = device->createCommandQueue(plume::RenderCommandListType::DIRECT);
    CreateSwapChain();
    if (!swapChain || !swapChain->resize())
    {
        log::Error("Swap chain creation failed: the game runs headless");
        device.reset();
        return false;
    }
    for (uint32_t i = 0; i < kFramesInFlight; i++)
    {
        commandLists[i] = queue->createCommandList();
        fences[i] = device->createCommandFence();
        acquireSemaphores[i] = device->createCommandSemaphore();
    }
    // The frame counter costs an extra submission per frame: made only where something uses it (AC6_POOL_HAZARD).
    static const bool wantCounter = [] { const char* v = std::getenv("AC6_POOL_HAZARD"); return v && *v == '1'; }();
    if (auto* vd = dynamic_cast<plume::VulkanDevice*>(device.get()); wantCounter && vd && vd->timelineSemaphoreSupported)
    {
        VkSemaphoreTypeCreateInfo type{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, nullptr, VK_SEMAPHORE_TYPE_TIMELINE, 0 };
        VkSemaphoreCreateInfo info{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type };
        VkSemaphore semaphore = VK_NULL_HANDLE;
        if (vkCreateSemaphore(vd->vk, &info, nullptr, &semaphore) == VK_SUCCESS)
            timeline = semaphore;
    }
    CreateFramebuffers();
    if (const char* v = std::getenv("AC6_DEFERRED_RECORDING"); !v || *v != '0')
        deferred = std::make_unique<DeferredCommandList>();
    log::Info("Command recording: {}", deferred ? "deferred to a worker thread" : "on the calling thread");
    log::Info("Window {}x{}, swap chain {}x{}", outputWidth, outputHeight, swapChain->getWidth(), swapChain->getHeight());
    return true;
}

// HDR on in the settings while the display's HDR is off in the system: the output is SDR until it is turned on
// there. A driver may offer the scRGB swap chain all the same (the fallback in CreateSwapChain only catches the ones
// that don't), and the HDR picture then goes to an SDR desktop: white at 2.5 times the desktop's, the rest clipped.
// The setting itself is left as it is: the output follows the system's switch, also while running (the display's
// state is read again when it changes, PumpEvents, ApplyOutputChange then makes the swap chain again).
// Windows only: elsewhere what SDL reports of the display's HDR is not relied on. AC6_HDR_FORCE=1: the HDR
// output whatever the system reports.
bool Video::HdrHeldBack() const
{
#ifdef _WIN32
    static const bool force = [] { const char* v = std::getenv("AC6_HDR_FORCE"); return v && *v == '1'; }();
    return !force && settings::Get("AC6_OUTPUT") == "hdr" && !displayHdr.load(std::memory_order_relaxed);
#else
    return false;
#endif
}

// AC6_OUTPUT=hdr, or AC6_SDR_10BIT=1 (AC6_OUTPUT=deep is taken too): "hdr", "deep" or "" (8-bit SDR).
std::string Video::WantedOutput() const
{
    const std::string v = settings::Get("AC6_OUTPUT");
    return v == "hdr" && !HdrHeldBack() ? "hdr" : v == "deep" || settings::Get("AC6_SDR_10BIT") == "1" ? "deep" : "";
}

// The swap chain for the output asked for: at start, and after a live change (ApplyOutputChange).
void Video::CreateSwapChain()
{
    // AC6_OUTPUT=hdr | deep: HDR a 16-bit float swap chain in scRGB, deep a 10-bit one in sRGB
    // (a float swap chain is always scRGB to Windows, whatever its colour space: a float deep picture comes out broken,
    // or as HDR where only scRGB is offered). Either only in the colour space asked for, else the 8-bit one.
    // VSync's queue kept short: each present waits until the one before it is on the screen (VK_KHR_present_wait), so the
    // game can't run ahead of the display. A pace a hair faster than the display's refresh (60 against 59.94 Hz) fills
    // the swap chain's queue in about 26 s otherwise, two frames more latency from then on. AC6_PRESENT_WAIT=0: off.
    presentWait = device->getCapabilities().presentWait && settings::Get("AC6_PRESENT_WAIT") != "0";
    auto withWait = [this](plume::RenderSwapChainDesc desc) {
        desc.enablePresentWait = presentWait;
        desc.maxFrameLatency = presentWait ? 2 : 0;
        return desc;
    };
    {
        const std::string want = WantedOutput();
        if (HdrHeldBack())
            log::Info("Output: HDR is on in the settings but off for this display in the system: SDR until it is turned on there");
        if (want == "hdr" || want == "deep")
        {
            const bool hdr = want == "hdr";
            const plume::RenderFormat format = hdr ? plume::RenderFormat::R16G16B16A16_FLOAT : plume::RenderFormat::R10G10B10A2_UNORM;
            plume::RenderSwapChainDesc desc(renderWindow, format, kSwapChainImages);
            desc.colorSpace = hdr ? plume::RenderColorSpace::EXTENDED_SRGB_LINEAR : plume::RenderColorSpace::SRGB_NONLINEAR;
            swapChain = queue->createSwapChain(withWait(desc));
            if (swapChain && swapChain->resize() && swapChain->getColorSpace() == desc.colorSpace)
            {
                backBufferFormat = format;
                outputMode = hdr ? Output::Hdr : Output::SdrDeep;
                log::Info("Output: {}", hdr ? "HDR, scRGB (16-bit float swap chain)" : "SDR, 10 bits per colour (10:10:10:2 swap chain)");
            }
            else
            {
                log::Warn("AC6_OUTPUT={}: the display or driver offers no {} swap chain{}: 8-bit SDR", want,
                    hdr ? "scRGB (16-bit float)" : "10-bit sRGB", hdr ? " (Windows HDR off?)" : "");
                swapChain.reset();
            }
        }
    }
    if (!swapChain)
    {
        swapChain = queue->createSwapChain(withWait(plume::RenderSwapChainDesc(renderWindow, kSwapChainFormat, kSwapChainImages)));
        backBufferFormat = kSwapChainFormat;
        outputMode = Output::Sdr8;
    }
    // AC6_VSYNC=0: present without waiting for the display (measuring the frame rate past its refresh, live).
    vsync = settings::Get("AC6_VSYNC") != "0";
    if (swapChain && !vsync)
    {
        swapChain->setVsyncEnabled(false);
        log::Info("Vsync off (AC6_VSYNC=0)");
    }
    outputWanted = WantedOutput();
    log::Info("Present wait: {}", presentWait ? "on (a present waits for the one before it to reach the screen)"
                                               : device->getCapabilities().presentWait ? "off (AC6_PRESENT_WAIT=0)"
                                                                                       : "not offered by the driver");
}

// The output switched while running (the settings overlay: HDR on / off, 10-bit SDR): the swap
// chain built again for it. The render thread, at a present's start (before the window's image is used). The GPU is
// waited for first. True when it switched (the renderer then builds its window pipelines again).
bool Video::ApplyOutputChange()
{
    if (!queue || imageAcquired || WantedOutput() == outputWanted)
        return false;
    WaitGpuIdle();
    std::lock_guard lock(swapMutex);
    const std::string want = WantedOutput();
    outputWanted = want;
    // The same swap chain made again in the format asked for (plume setFormat: its surface kept, the old swap chain
    // handed over, as a resize: destroying it and its surface for a new one leaves the window black after an HDR off ->
    // on -> off), else 8-bit SDR.
    framebuffers.clear();
    auto set = [&](plume::RenderFormat format, plume::RenderColorSpace colorSpace, Output mode) {
        if (!swapChain->setFormat(format, colorSpace))
            return false;
        backBufferFormat = format;
        outputMode = mode;
        return true;
    };
    const Output before = outputMode;
    bool ok = want == "hdr"  ? set(plume::RenderFormat::R16G16B16A16_FLOAT, plume::RenderColorSpace::EXTENDED_SRGB_LINEAR, Output::Hdr)
            : want == "deep" ? set(plume::RenderFormat::R10G10B10A2_UNORM, plume::RenderColorSpace::SRGB_NONLINEAR, Output::SdrDeep)
                             : false;
    if (!ok)
    {
        if (!want.empty())
            log::Warn("Output: the display or driver offers no {} swap chain now: 8-bit SDR", want == "hdr" ? "scRGB (16-bit float)" : "10-bit sRGB");
        ok = set(kSwapChainFormat, plume::RenderColorSpace::SRGB_NONLINEAR, Output::Sdr8);
    }
    CreateFramebuffers();
    if (!ok)
    {
        log::Error("Output: the swap chain couldn't be made again for the output asked for ({})", want.empty() ? "sdr" : want);
        return false;
    }
    log::Info("Output switched: {} -> {}{}", before == Output::Hdr ? "HDR" : before == Output::SdrDeep ? "10-bit SDR" : "8-bit SDR",
        outputMode == Output::Hdr ? "HDR" : outputMode == Output::SdrDeep ? "10-bit SDR" : "8-bit SDR",
        HdrHeldBack() ? " (HDR is on in the settings, off for this display in the system)" : "");
    return true;
}

// Everything submitted done (the present thread's too).
void Video::WaitGpuIdle()
{
    WaitPresentIdle();
    for (uint32_t s = 0; s < kFramesInFlight; s++)
        if (submitted[s])
        {
            queue->waitForCommandFence(fences[s].get());
            submitted[s] = false;
        }
}

// The display mode and refresh rate while running (the settings overlay): the window's thread, at
// its event pump, after a live change. The output's size (the resolution) stays the one chosen at start.
void Video::ApplyWindowSettings()
{
    static uint64_t seen = 0;
    const uint64_t generation = settings::Generation();
    if (!window || generation == seen)
        return;
    seen = generation;
    std::string fullscreen = settings::Get("AC6_FULLSCREEN");
    if (fullscreen.empty())
        fullscreen = background ? "0" : "borderless";
    const std::string refresh = settings::Get("AC6_REFRESH");
    if (fullscreen == fullscreenNow && refresh == refreshNow)
        return;
    fullscreenNow = fullscreen;
    refreshNow = refresh;
    // Exclusive: the monitor's mode of the output's size at the rate asked for (the nearest, none: the highest).
    const SDL_DisplayMode* mode = nullptr;
    if (fullscreen == "1")
    {
        const float wantHz = float(std::atof(refresh.c_str()));
        int count = 0;
        SDL_DisplayMode** modes = SDL_GetFullscreenDisplayModes(SDL_GetDisplayForWindow(window), &count);
        for (int i = 0; i < count; i++)
        {
            const SDL_DisplayMode* m = modes[i];
            if (uint32_t(m->w) != outputWidth || uint32_t(m->h) != outputHeight)
                continue;
            if (!mode || (wantHz > 0.0f ? std::fabs(m->refresh_rate - wantHz) < std::fabs(mode->refresh_rate - wantHz)
                                        : m->refresh_rate > mode->refresh_rate))
                mode = m;
        }
        static SDL_DisplayMode chosen;   // (SDL keeps the pointer it is given)
        if (mode)
        {
            chosen = *mode;
            mode = &chosen;
        }
        SDL_free(modes);
    }
    if (fullscreen == "1" || fullscreen == "borderless")
    {
        // Exclusive needs one of the monitor's modes: borderless instead.
        if (!SDL_SetWindowFullscreenMode(window, mode) || !SDL_SetWindowFullscreen(window, true))
            log::Warn("Fullscreen failed: {}", SDL_GetError());
        if (mode)
            refreshHz = mode->refresh_rate;
    }
    else
    {
        SDL_SetWindowFullscreen(window, false);
        if (nativeSize)
            SDL_MaximizeWindow(window);
    }
    if (!mode)
        if (const SDL_DisplayMode* desktop = SDL_GetDesktopDisplayMode(SDL_GetDisplayForWindow(window)); desktop && desktop->refresh_rate > 0.0f)
            refreshHz = desktop->refresh_rate;
    SDL_SyncWindow(window);
    log::Info("Display mode: {}{} (live)", mode ? "exclusive fullscreen" : fullscreen == "0" ? "window" : "borderless fullscreen",
        mode ? std::format(" at {:.2f} Hz", mode->refresh_rate) : std::string());
}

// The window's display as the system reports it: HDR on, its peak (AC6_HDR_PEAK's default), and, out of exclusive
// fullscreen, its refresh rate (vsync's pace, the Match display cap). At start, and when the window moves to another
// display or its HDR state changes (PumpEvents).
void Video::ReadDisplayInfo()
{
    const SDL_PropertiesID props = SDL_GetWindowProperties(window);
    displayHdr = SDL_GetBooleanProperty(props, SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN, false);
    // AC6_DISPLAY_HDR=0 | 1 (tests): the display's HDR state as if the system reported that.
    if (const char* v = std::getenv("AC6_DISPLAY_HDR"); v && (*v == '0' || *v == '1'))
        displayHdr = *v == '1';
    const float white = SDL_GetFloatProperty(props, SDL_PROP_WINDOW_SDR_WHITE_LEVEL_FLOAT, 1.0f);
    const float headroom = SDL_GetFloatProperty(props, SDL_PROP_WINDOW_HDR_HEADROOM_FLOAT, 1.0f);
    // scRGB 1.0 is 80 nits. The headroom is over SDR white.
    displayPeakNits = displayHdr && headroom > 1.0f ? headroom * white * 80.0f : 0.0f;
    const SDL_DisplayID on = SDL_GetDisplayForWindow(window);
    if (fullscreenNow != "1")
        if (const SDL_DisplayMode* desktop = on ? SDL_GetDesktopDisplayMode(on) : nullptr; desktop && desktop->refresh_rate > 0.0f)
            refreshHz = desktop->refresh_rate;
    log::Info("Display: {} ({:.2f} Hz), HDR {}{}", on && SDL_GetDisplayName(on) ? SDL_GetDisplayName(on) : "?", refreshHz,
        displayHdr ? "on" : "off",
        displayHdr ? std::format(" (SDR white {:.0f} nits, peak {:.0f} nits)", white * 80.0f, displayPeakNits) : std::string());
}

void Video::CreateFramebuffers()
{
    framebuffers.clear();
    for (uint32_t i = 0; i < swapChain->getTextureCount(); i++)
    {
        const plume::RenderTexture* color = swapChain->getTexture(i);
        plume::RenderFramebufferDesc desc;
        desc.colorAttachments = &color;
        desc.colorAttachmentsCount = 1;
        framebuffers.push_back(device->createFramebuffer(desc));
    }
    while (releaseSemaphores.size() < swapChain->getTextureCount())
        releaseSemaphores.push_back(device->createCommandSemaphore());
}

bool GpuTiming()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_RENDER_STATS"); return v && *v && *v != '0'; }();
    return enabled;
}

bool Video::GpuPassTiming() const
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_GPU_PASSES"); return GpuTiming() && v && (*v == '1' || *v == '2'); }();
    return enabled;
}

bool Video::GpuDrawTiming() const
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_GPU_PASSES"); return GpuTiming() && v && *v == '2'; }();
    return enabled;
}

// Timestamps are written at the bottom of the pipe: a pass's time is from the work before it completing to its own
// completing, so passes that overlap on the GPU share the time unevenly (the totals still add up to the frame's).
void Video::MarkGpuPass(std::string label)
{
    if (!GpuPassTiming() || !Ready())
        return;
    plume::RenderCommandList& cl = Commands();
    auto& passes = gpuPasses[slot];
    if (!gpuTimers[slot] || passes.size() >= kMaxGpuPasses)
        return;
    passes.push_back({ std::move(label) });
    cl.writeTimestamp(gpuTimers[slot].get(), uint32_t(passes.size()));
}

void Video::CountGpuPassDraw()
{
    if (GpuPassTiming() && !gpuPasses[slot].empty())
        gpuPasses[slot].back().draws++;
}

std::vector<std::string> Video::TakeGpuPasses(uint32_t top)
{
    std::vector<std::string> lines;
    if (!GpuPassTiming() || !gpuPassFrames)
        return lines;
    std::sort(gpuPassTotals.begin(), gpuPassTotals.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
    double total = 0;
    for (const auto& [label, t] : gpuPassTotals)
        total += t.ns;
    for (uint32_t i = 0; i < gpuPassTotals.size() && i < top; i++)
    {
        const auto& [label, t] = gpuPassTotals[i];
        lines.push_back(std::format("{:6.3f} ms {:4.1f}% {:5.1f}x {:6.1f} draws  {}", t.ns / 1e6 / gpuPassFrames, 100.0 * t.ns / total,
            double(t.count) / gpuPassFrames, double(t.draws) / gpuPassFrames, label));
    }
    lines.push_back(std::format("{:6.3f} ms in all, {} frames, {} kinds of pass", total / 1e6 / gpuPassFrames, gpuPassFrames, gpuPassTotals.size()));
    gpuPassTotals.clear();
    gpuPassFrames = 0;
    gpuPassDump = !GpuDrawTiming();   // one frame's ~1000 draws in order would flood the log
    return lines;
}

void Video::BeginFrame()
{
    commandLists[slot]->begin();
    if (deferred)
        deferred->Start(commandLists[slot].get());
    recordingEpoch++;
    frameOpen = true;
    if (GpuTiming())
    {
        // The slot's previous frame has completed (its fence was waited for before the slot came round).
        // With AC6_GPU_PASSES: query 0 starts the frame, 1..n the marked passes, kMaxGpuPasses + 1 ends it.
        const uint32_t queries = GpuPassTiming() ? kMaxGpuPasses + 2 : 2;
        auto& passes = gpuPasses[slot];
        if (!gpuTimers[slot])
            gpuTimers[slot] = device->createQueryPool(queries);
        else if (gpuTimed[slot])
        {
            gpuTimers[slot]->queryResults();
            const uint64_t* ns = gpuTimers[slot]->getResults();
            gpuNs += int64_t(ns[queries - 1] - ns[0]);
            gpuFrames++;
            if (GpuPassTiming())
            {
                // The frame's start to the first pass, then each pass to the next (the last one to the frame's end).
                auto add = [&](const std::string& label, int64_t delta, uint32_t draws) {
                    auto it = std::find_if(gpuPassTotals.begin(), gpuPassTotals.end(), [&](const auto& t) { return t.first == label; });
                    if (it == gpuPassTotals.end())
                        it = gpuPassTotals.insert(gpuPassTotals.end(), { label, {} });
                    it->second.ns += double(delta);
                    it->second.count++;
                    it->second.draws += draws;
                };
                gpuPassFrames++;
                const size_t n = passes.size();
                add("(frame start)", int64_t((n ? ns[1] : ns[queries - 1]) - ns[0]), 0);
                for (size_t i = 0; i < n; i++)
                {
                    const int64_t delta = int64_t((i + 1 < n ? ns[i + 2] : ns[queries - 1]) - ns[i + 1]);
                    add(passes[i].label, delta, passes[i].draws);
                    if (gpuPassDump)
                        log::Info("GPU pass {:3}: {:6.3f} ms {:5} draws  {}", i, delta / 1e6, passes[i].draws, passes[i].label);
                }
                gpuPassDump = false;
            }
        }
        passes.clear();
        plume::RenderCommandList& cl = Commands();
        cl.resetQueryPool(gpuTimers[slot].get(), 0, queries);
        cl.writeTimestamp(gpuTimers[slot].get(), 0);
        gpuTimed[slot] = false;
    }
}

double Video::TakeGpuFrameMs()
{
    const double ms = gpuFrames ? gpuNs / 1e6 / gpuFrames : 0.0;
    gpuNs = 0;
    gpuFrames = 0;
    return ms;
}

plume::RenderCommandList& Video::Commands()
{
    if (!frameOpen)
        BeginFrame();
    if (deferred)
        return *deferred;
    return *commandLists[slot];
}

// The back buffer is acquired when the frame first asks for it (its final blit), not when it starts: the
// present thread may still be presenting the frame before.
void Video::AcquireImage()
{
    if (imageAcquired || acquireFailed)
        return;
    // Minimised: no image and no present. Windows holds each present to a minimised window for about a second, so the
    // game would run at 1 FPS, on some systems for a while after the window comes back too. The frame draws into
    // the stand-in.
    if (windowMinimized.load(std::memory_order_acquire))
    {
        acquireFailed = true;
        return;
    }
    bool resize;
    {
        std::lock_guard lock(swapMutex);
        // AC6_VSYNC (live: the settings overlay) and AC6_FAST_BOOT=1 (no vsync through the boot logos,
        // game/game_modes.cpp): the swap chain is rebuilt each way.
        static uint64_t seen = UINT64_MAX;
        static bool bootVsyncOff = false;
        if (const uint64_t generation = settings::Generation(); generation != seen || game::FastBoot() != bootVsyncOff)
        {
            seen = generation;
            bootVsyncOff = game::FastBoot();
            const bool want = !bootVsyncOff && settings::Get("AC6_VSYNC") != "0";
            if (want != vsync)
            {
                vsync = want;
                swapChain->setVsyncEnabled(want);
                if (!bootVsyncOff)
                    log::Info("Vsync {}", want ? "on" : "off");
            }
        }
        resize = swapChain->needsResize() || rebuildSwapChain.exchange(false);
    }
    if (resize)
    {
        // The old images' framebuffers go: nothing may still use them.
        WaitPresentIdle();
        for (uint32_t s = 0; s < kFramesInFlight; s++)
            if (submitted[s] && s != slot)
            {
                queue->waitForCommandFence(fences[s].get());
                submitted[s] = false;
            }
        std::lock_guard lock(swapMutex);
        swapChain->resize();
        CreateFramebuffers();
    }
    std::unique_lock lock(swapMutex);
    if (!swapChain->acquireTexture(acquireSemaphores[slot].get(), &swapChainIndex))
    {
        // Out of date (the display, the output's mode changed under it): made again once. A submission waiting on an
        // acquire that didn't happen would never run.
        static uint32_t failures = 0;
        if (failures++ < 8)
            log::Warn("Swap chain: acquiring an image failed: made again");
        lock.unlock();
        WaitGpuIdle();
        lock.lock();
        swapChain->resize();
        CreateFramebuffers();
        if (!swapChain->acquireTexture(acquireSemaphores[slot].get(), &swapChainIndex))
        {
            // No image this frame: it draws into the stand-in and is submitted without the acquire's semaphore and
            // without a present (a submission waiting on an acquire that didn't happen would never run).
            static uint32_t again = 0;
            if (again++ < 8)
                log::Error("Swap chain: acquiring an image failed again: this frame isn't shown");
            acquireFailed = true;
            return;
        }
    }
    imageAcquired = true;
}

bool Video::NoWindowImage() const
{
    return acquireFailed || swapChain->getWidth() == 0 || swapChain->getHeight() == 0;
}

void Video::EnsureStandIn()
{
    if (standIn && standInFormat == backBufferFormat)
        return;
    standInFramebuffer.reset();
    standIn = device->createTexture(plume::RenderTextureDesc::ColorTarget(kStandInSize, kStandInSize, backBufferFormat));
    if (!standIn)
        crash::Shutdown(1, "out of video memory");
    const plume::RenderTexture* color = standIn.get();
    plume::RenderFramebufferDesc desc;
    desc.colorAttachments = &color;
    desc.colorAttachmentsCount = 1;
    standInFramebuffer = device->createFramebuffer(desc);
    standInFormat = backBufferFormat;
}

plume::RenderTexture* Video::BackBuffer()
{
    AcquireImage();
    if (NoWindowImage())
    {
        EnsureStandIn();
        return standIn.get();
    }
    return swapChain->getTexture(swapChainIndex);
}

plume::RenderFramebuffer* Video::BackBufferFramebuffer()
{
    AcquireImage();
    if (NoWindowImage())
    {
        EnsureStandIn();
        return standInFramebuffer.get();
    }
    return framebuffers[swapChainIndex].get();
}
uint32_t Video::BackBufferWidth() const { return NoWindowImage() ? kStandInSize : swapChain->getWidth(); }
uint32_t Video::BackBufferHeight() const { return NoWindowImage() ? kStandInSize : swapChain->getHeight(); }

void Video::Present(uint32_t vblanks, bool waitAll, bool pumpEvents)
{
    if (!Ready())
    {
        Pace(vblanks);
        frameCount++;
        if (pumpEvents)
            PumpEvents();
        return;
    }
    plume::RenderCommandList& cl = Commands();
    if (GpuTiming() && gpuTimers[slot])
    {
        cl.writeTimestamp(gpuTimers[slot].get(), GpuPassTiming() ? kMaxGpuPasses + 1 : 1);
        gpuTimed[slot] = true;
    }
    if (imageAcquired)   // (the window's image itself, even when the frame drew into the stand-in)
        cl.barriers(plume::RenderBarrierStage::NONE,
            plume::RenderTextureBarrier(swapChain->getTexture(swapChainIndex), plume::RenderTextureLayout::PRESENT));
    if (deferred)
        deferred->Finish();
    commandLists[slot]->end();
    frameOpen = false;
    const PresentJob job{ slot, swapChainIndex, imageAcquired, vblanks, timeline ? timelineIssued.load(std::memory_order_relaxed) + 1 : 0 };
    if (timeline)
        timelineIssued.store(job.timeline, std::memory_order_release);
    imageAcquired = false;
    acquireFailed = false;

    // Unpaced frames are handed to the present thread (the frame before must have been handed over: the
    // queue and the slots are used by one thread at a time). Paced ones keep their present where the
    // game's frame ends, and readbacks need the frame done anyway.
    using Clock = std::chrono::steady_clock;
    WaitPresentIdle();
    if (asyncPresent && !waitAll && !(framePacing && vblanks))
    {
        if (!presentThreadStarted)
        {
            presentThreadStarted = true;
            std::thread([this] { PresentLoop(); }).detach();
        }
        std::lock_guard lock(presentMutex);
        presentJob = job;
        presentQueued = true;
        presentCv.notify_all();
    }
    else
        Execute(job);
    submitted[job.slot] = true;
    const auto t2 = Clock::now();
    // The next frame reuses the other slot: wait for the frame recorded there (handed over already, above,
    // and for this one too when the caller reads results back). Each submission is waited for once (plume
    // resets the fence).
    const uint32_t next = (slot + 1) % kFramesInFlight;
    for (uint32_t s : { next, slot })
        if (submitted[s] && (s == next || waitAll))
        {
            queue->waitForCommandFence(fences[s].get());
            submitted[s] = false;
        }
    slot = next;
    lastGpuWaitNs = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t2).count();
    frameCount++;
    if (pumpEvents)
        PumpEvents();
}

uint64_t Video::FramesCompleted() const
{
    if (!timeline)
        return 0;
    uint64_t value = 0;
    vkGetSemaphoreCounterValue(static_cast<plume::VulkanDevice*>(device.get())->vk, static_cast<VkSemaphore>(timeline), &value);
    return value;
}

void Video::WaitFrameCompleted(uint64_t frame) const
{
    if (!timeline)
        return;
    VkSemaphore semaphore = static_cast<VkSemaphore>(timeline);
    VkSemaphoreWaitInfo info{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, nullptr, 0, 1, &semaphore, &frame };
    plume::NoteVulkanResult(vkWaitSemaphores(static_cast<plume::VulkanDevice*>(device.get())->vk, &info, UINT64_MAX));
}

std::string Video::DescribeMemory(plume::RenderBuffer* buffer) const
{
    const auto* vb = dynamic_cast<const plume::VulkanBuffer*>(buffer);
    const auto* vd = dynamic_cast<const plume::VulkanDevice*>(device.get());
    if (!vb || !vd)
        return "unknown";
    VkMemoryPropertyFlags flags = 0;
    vmaGetMemoryTypeProperties(vd->allocator, vb->allocationInfo.memoryType, &flags);
    std::string s = "type " + std::to_string(vb->allocationInfo.memoryType) + ":";
    for (auto [bit, name] : { std::pair{ VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "device local" }, { VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, "host visible" },
             { VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "coherent" }, { VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "cached" } })
        if (flags & bit)
            s += std::string(s.back() == ':' ? " " : ", ") + name;
    return s;
}

void Video::Execute(const PresentJob& job)
{
    const plume::RenderCommandList* lists = commandLists[job.slot].get();
    plume::RenderCommandSemaphore* wait = acquireSemaphores[job.slot].get();
    plume::RenderCommandSemaphore* signal = releaseSemaphores[job.image].get();
    const uint32_t semaphores = job.hasImage ? 1 : 0;
    using Clock = std::chrono::steady_clock;
    auto ns = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count(); };
    queue->executeCommandLists(&lists, 1, &wait, semaphores, &signal, semaphores, fences[job.slot].get());
    if (job.timeline)
    {
        // An empty batch after the frame's: its signal waits for everything submitted before it.
        auto* vq = static_cast<plume::VulkanCommandQueue*>(queue.get());
        VkSemaphore semaphore = static_cast<VkSemaphore>(timeline);
        VkTimelineSemaphoreSubmitInfo values{ VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
        values.signalSemaphoreValueCount = 1;
        values.pSignalSemaphoreValues = &job.timeline;
        VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO, &values };
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &semaphore;
        std::lock_guard lock(*vq->queue->mutex);
        plume::NoteVulkanResult(vkQueueSubmit(vq->queue->vk, 1, &submit, VK_NULL_HANDLE));
    }
    CheckDeviceLost();
    const auto t0 = Clock::now();
    Pace(job.vblanks);   // the GPU works on the frame meanwhile
    const auto t1 = Clock::now();
    NotePresent();
    if (job.hasImage)
    {
        std::lock_guard lock(swapMutex);
        if (!swapChain->present(job.image, &signal, 1))
        {
            static std::atomic<uint32_t> failures{ 0 };
            if (failures++ < 8)
                log::Warn("Swap chain: presenting failed (out of date?)");
        }
        else if (presentWait && vsync)
        {
            // Until the last frame is on the screen: the next frame starts no earlier than that (up to 100 ms).
            const auto w0 = Clock::now();
            swapChain->wait();
            const int64_t waited = ns(w0, Clock::now());
            static int64_t total = 0, most = 0;
            static uint32_t count = 0;
            total += waited;
            most = std::max(most, waited);
            if (++count == 1200)
            {
                log::Info("Present wait: {:.2f} ms a frame on average, {:.2f} ms at most (the last 1200 frames)", total / 1200.0 / 1e6,
                    most / 1e6);
                total = most = 0;
                count = 0;
            }
        }
        CheckDeviceLost();
    }
    lastPaceNs = ns(t0, t1);
    lastSwapNs = ns(t1, Clock::now());
}

// A lost device (a driver reset after a GPU hang or crash, a removed GPU) never comes back: every later submit fails and
// the window would stay frozen. The game stops with a message instead, and the log says where.
void Video::CheckDeviceLost()
{
    if (!plume::VulkanDeviceLost())
        return;
    const plume::RenderDeviceDescription& d = device->getDescription();
    log::Error("The Vulkan device was lost (VK_ERROR_DEVICE_LOST) with the renderer at frame {}: the GPU driver reset or "
               "crashed. GPU: {}, driver {}", RendererFrame(), d.name, system_info::DriverVersion(uint32_t(d.vendor), d.driverVersion));
    breadcrumbs::Report(device.get());
    crash::Shutdown(1, "the graphics driver stopped responding");
}

void Video::PresentLoop()
{
    std::unique_lock lock(presentMutex);
    for (;;)
    {
        presentCv.wait(lock, [this] { return presentQueued; });
        const PresentJob job = presentJob;
        lock.unlock();
        Execute(job);
        lock.lock();
        presentQueued = false;
        presentCv.notify_all();
    }
}

void Video::WaitPresentIdle()
{
    std::unique_lock lock(presentMutex);
    presentCv.wait(lock, [this] { return !presentQueued; });
}

// The console shows a frame for the swap's interval of vblanks (sub_821EFE78 encodes the device's
// D3DPRESENT_INTERVAL into the swap packet, the GPU holds the flip). Presents here are synchronous, so
// the same hold is a wait before presenting: the game's frame loop runs at 60 / vblanks Hz.
void Video::Pace(uint32_t vblanks)
{
    if (!framePacing || !vblanks)
        return;
    using Clock = std::chrono::steady_clock;
    constexpr auto kVBlank = std::chrono::nanoseconds(1'000'000'000 / 60);
    // An unlocked mission frame (gpu/frame_rate.h) lasts 1 / AC6_FRAME_RATE instead of its vblanks. (The rate can change
    // between the frame's swap and here, the settings overlay's live changes: to 0, uncapped, as well.)
    const uint32_t rate = FrameRateTarget();
    // AC6_PACE_TEST_HZ=<rate> (tests): unlocked frames paced at that rate exactly, past the cap's rules (a pace a
    // little faster than the display's real refresh, as a "60 Hz" monitor's 59.9x against a pace of 60).
    static const double testHz = [] { const char* v = std::getenv("AC6_PACE_TEST_HZ"); return v ? std::atof(v) : 0.0; }();
    const auto step = Unlocked() && testHz > 0.0 ? std::chrono::nanoseconds(int64_t(1e9 / testHz))
        : Unlocked() && rate ? std::chrono::nanoseconds(1'000'000'000 / rate) : kVBlank * vblanks;
    const auto now = Clock::now();
    if (nextPresent.time_since_epoch().count() == 0 || now > nextPresent + step)
        nextPresent = now;   // first frame, or fell behind (loading): don't burst to catch up
    else
    {
        // Sleep on the high-resolution timer to just short of the deadline, then yield the rest: the timer wakes up to
        // ~0.5 ms late on Windows, tens of microseconds on Linux. (Sleeping to 2 ms before it and yielding the rest
        // would burn up to 2 ms of a core a frame, all of it on Linux, where sched_yield returns at once.)
#if defined(_WIN32)
        constexpr auto kTail = std::chrono::microseconds(500);
#else
        constexpr auto kTail = std::chrono::microseconds(100);
#endif
        if (const auto sleep = nextPresent - kTail - now; sleep > Clock::duration::zero())
        {
            const auto hundredNs = std::chrono::duration_cast<std::chrono::duration<int64_t, std::ratio<1, 10'000'000>>>(sleep);
            if (!platform::SleepPrecise(uint64_t(hundredNs.count())))
                std::this_thread::sleep_until(nextPresent - std::chrono::milliseconds(2));
        }
        while (Clock::now() < nextPresent)
            std::this_thread::yield();
    }
    nextPresent += step;
}

void Video::PumpEvents()
{
    if (window)
    {
        ui::Init(window);   // the settings overlay (src/ui/overlay.cpp), on the window's thread
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            // Moved to another monitor, or its HDR switched in the system: what the game takes from the display read again.
            if (event.type == SDL_EVENT_WINDOW_DISPLAY_CHANGED || event.type == SDL_EVENT_WINDOW_HDR_STATE_CHANGED)
                ReadDisplayInfo();
            if (event.type == SDL_EVENT_QUIT || event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED)
                crash::Shutdown(0, "window closed");
            if (event.type == SDL_EVENT_WINDOW_MINIMIZED && !windowMinimized.exchange(true))
                log::Info("Window minimised: frames aren't shown until it's restored");
            if ((event.type == SDL_EVENT_WINDOW_RESTORED || event.type == SDL_EVENT_WINDOW_MAXIMIZED) && windowMinimized.exchange(false))
            {
                rebuildSwapChain = true;   // (out of whatever state the minimised window left it in)
                log::Info("Window restored");
            }
            if (ui::HandleEvent(event))
            {
                // The overlay's (F4, and everything while it is open). Releases still reach the game's input: a key
                // or a mouse button let go while the overlay was open isn't held once it closes.
                if (event.type == SDL_EVENT_KEY_UP || event.type == SDL_EVENT_MOUSE_BUTTON_UP)
                    input::HandleEvent(event);
                continue;
            }
            // F11 / F12 only with the captures' switch on (off by default, no capture by accident).
            const bool captureKey = event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat &&
                (event.key.scancode == SDL_SCANCODE_F12 || event.key.scancode == SDL_SCANCODE_F11);
            if (captureKey && settings::Get("AC6_CAPTURES") != "1" && settings::Get("AC6_CAPTURES") != "true")
            {
                static bool told = false;
                if (!told)
                    log::Info("F11 / F12: captures are off (the Debug tab's Captures, AC6_CAPTURES=1)");
                told = true;
            }
            else if (captureKey && event.key.scancode == SDL_SCANCODE_F12)
                capture::RequestCapture(2);
            else if (captureKey)
                RequestRingDump();   // the frame ring
            input::HandleEvent(event);
        }
        ui::Frame();
        ApplyWindowSettings();
    }
    input::Update();   // once per game frame, on the main thread
}

} // namespace ac6::gpu
