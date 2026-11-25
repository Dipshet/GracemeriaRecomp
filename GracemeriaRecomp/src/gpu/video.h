#pragma once
// Host video: the SDL3 window and the plume (Vulkan) device, swap chain and per-frame command list.
//
// The game's own swap call ends the host frame, so host FPS == game FPS. Paced frames
// are submitted and presented right there. Unpaced ones on a present thread, so the game's main thread
// goes on while the driver presents. The window lives on the host main thread, which is
// also the game's main thread. SDL events are pumped at each present.

#include <plume_render_interface.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>

struct SDL_Window;
struct SDL_DisplayMode;

namespace ac6::gpu {

class DeferredCommandList;

// The fixed guest resolution.
constexpr uint32_t kGuestWidth = 1280;
constexpr uint32_t kGuestHeight = 720;

struct Video
{
    // Show the window without taking the focus (AC6_BACKGROUND=1, and replays): test runs started from
    // a terminal don't take the focus away.
    bool background = false;
    // Hold each present to the game's present interval: on by default in
    // AceCombat6.exe, off in ac6.exe so test runs go at the host's pace, AC6_FRAME_PACING=0/1 overrides.
    bool framePacing = false;
    SDL_Window* window = nullptr;
    // The display mode: AC6_RESOLUTION=<width>x<height> (any: the window's size and the frame's,
    // exclusive fullscreen takes it only as one of the monitor's modes, else borderless) or native (the desktop's),
    // AC6_REFRESH=<Hz> (default: that resolution's highest), AC6_FULLSCREEN=1 (exclusive) or borderless (the window the
    // desktop's size, the frame scaled into it). Default: native and borderless (a window in AC6_BACKGROUND runs,
    // 1280x720 in replays). The guest frame is made at the largest 16:9 size that fits the output (frameWidth x
    // frameHeight, filled with AC6_WIDESCREEN) and letterboxed into it. Its 3D is rendered at AC6_RENDER_SCALE=<f>
    // times that (renderWidth x renderHeight, renderScale = renderWidth / 1280), the UI and HUD at the frame's size.
    // AC6_RENDER_RESOLUTION=<width>x<height> (tests) makes the whole frame another size.
    uint32_t outputWidth = kGuestWidth, outputHeight = kGuestHeight;
    float refreshHz = 60.0f;          // the chosen mode's (or the desktop's) refresh rate: vsync's pace
    // Without AC6_RESOLUTION the output is the desktop's size (a window maximised). Replays keep the 1280x720 window.
    bool nativeByDefault = true;
    bool nativeSize = false;          // the output is the desktop's size
    // The window's display as the system reports it (SDL's HDR properties, at start and when they change): HDR on, and
    // its peak in nits (0: unknown). AC6_HDR_PEAK's default. The HDR output follows displayHdr (WantedOutput, read by
    // the render thread).
    std::atomic<bool> displayHdr{ false };
    float displayPeakNits = 0.0f;
    uint32_t renderWidth = kGuestWidth, renderHeight = kGuestHeight;
    uint32_t frameWidth = kGuestWidth, frameHeight = kGuestHeight;   // the UI's and the output target's
    float frameScaleX = 1.0f, frameScaleY = 1.0f;                   // their host pixels per guest pixel
    float renderScale = 1.0f;         // host pixels per guest pixel (vertical, the 16:9 case: both axes)
    // Wide or narrow output (AC6_WIDESCREEN=1): an output of another shape than 16:9 is filled,
    // the guest's 1280x720 frame rendered with more host pixels per guest pixel along one axis (wider: across, hor+,
    // narrower: down, vert+). The game extends its projection and keeps its 2D at 16:9 to match (game/widescreen.cpp).
    // WideFactor() = the output aspect over 16:9 (above 1 wide, below 1 narrow, 1 when 16:9 or not filled).
    float renderScaleX = 1.0f, renderScaleY = 1.0f;
    float WideFactor() const { return renderScaleX / renderScaleY; }
    std::unique_ptr<plume::RenderInterface> renderInterface;
    std::unique_ptr<plume::RenderDevice> device;
    std::unique_ptr<plume::RenderCommandQueue> queue;
    // Frames the GPU has finished, counted by a timeline semaphore each frame's submission signals: any thread can
    // ask. Made only with AC6_POOL_HAZARD=1 (an extra submission a frame). FramesIssued() is the last value handed to a
    // submission, 0 without the semaphore.
    uint64_t FramesIssued() const { return timelineIssued.load(std::memory_order_acquire); }
    uint64_t FramesCompleted() const;
    void WaitFrameCompleted(uint64_t frame) const;
    bool HasFrameCounter() const { return timeline != nullptr; }
    enum class Output { Sdr8, SdrDeep, Hdr };
    std::unique_ptr<plume::RenderSwapChain> swapChain;
    Output outputMode = Output::Sdr8;
    plume::RenderFormat backBufferFormat = plume::RenderFormat::B8G8R8A8_UNORM;
    // Two frames in flight: the CPU records frame N+1 while the GPU draws frame N.
    // Each slot has its command list, fence and acquire semaphore. Present submits the current slot and then
    // waits for the other one, the frame before, whose resources the next frame reuses.
    static constexpr uint32_t kFramesInFlight = 2;
    std::unique_ptr<plume::RenderCommandList> commandLists[kFramesInFlight];
    std::unique_ptr<plume::RenderCommandFence> fences[kFramesInFlight];
    std::unique_ptr<plume::RenderCommandSemaphore> acquireSemaphores[kFramesInFlight];
    bool submitted[kFramesInFlight] = {};
    // GPU time per frame (AC6_RENDER_STATS=1): timestamps at the start and end of each frame's commands.
    std::unique_ptr<plume::RenderQueryPool> gpuTimers[kFramesInFlight];
    bool gpuTimed[kFramesInFlight] = {};
    int64_t gpuNs = 0;
    uint32_t gpuFrames = 0;
    // The average GPU time per frame since the last call (ms, 0 without AC6_RENDER_STATS).
    double TakeGpuFrameMs();
    // GPU time per pass (AC6_GPU_PASSES=1 with AC6_RENDER_STATS): a timestamp where the renderer
    // starts a pass (MarkGpuPass), the time to the next one is that pass's, summed by label over the stats window.
    // =2: a timestamp before every game draw too, so the totals are by draw (its shaders and target
    // size): which shaders cost the GPU most. The timestamps between draws serialise them, so the times are inflated
    // for small draws. The ranking is what counts.
    struct GpuPass
    {
        std::string label;
        uint32_t draws = 0;
    };
    static constexpr uint32_t kMaxGpuPasses = 4096;
    std::vector<GpuPass> gpuPasses[kFramesInFlight];
    struct GpuPassTotal
    {
        double ns = 0;
        uint64_t count = 0, draws = 0;
    };
    std::vector<std::pair<std::string, GpuPassTotal>> gpuPassTotals;
    uint32_t gpuPassFrames = 0;
    bool gpuPassDump = false;        // the next frame's passes are logged in order (after each TakeGpuPasses)
    bool GpuPassTiming() const;
    bool GpuDrawTiming() const;      // AC6_GPU_PASSES=2: every game draw timed too, labelled by its shaders
    void MarkGpuPass(std::string label);
    void CountGpuPassDraw();
    // The passes since the last call, the most expensive first, as log lines (empty without AC6_GPU_PASSES).
    std::vector<std::string> TakeGpuPasses(uint32_t top);
    // The Vulkan memory type a buffer was placed in, e.g. "type 3: device local, host visible, coherent" (logs).
    std::string DescribeMemory(plume::RenderBuffer* buffer) const;
    uint32_t slot = 0;               // the frame being recorded
    std::vector<std::unique_ptr<plume::RenderCommandSemaphore>> releaseSemaphores;
    // Calls are recorded here and replayed onto commandLists[slot] by a worker. Null with
    // AC6_DEFERRED_RECORDING=0 (recorded straight into the list).
    std::unique_ptr<DeferredCommandList> deferred;
    std::vector<std::unique_ptr<plume::RenderFramebuffer>> framebuffers;

    bool frameOpen = false;          // commandList is recording
    uint64_t recordingEpoch = 0;     // counts command list recordings (state caches key on it)
    bool imageAcquired = false;      // this frame has a back buffer (acquired when first asked for)
    uint32_t swapChainIndex = 0;     // acquired back buffer (valid while imageAcquired)
    // No window image to draw into this frame (both acquires failed, or a minimised window's 0x0 swap chain): the frame
    // draws into the stand-in, a small offscreen image in the back buffer's format, and isn't presented.
    bool acquireFailed = false;
    // The window minimised (the event pump's): frames aren't acquired or presented meanwhile, and the swap chain is made
    // again when it's restored. Windows holds each present to a minimised window for about a second.
    std::atomic<bool> windowMinimized{ false };
    std::atomic<bool> rebuildSwapChain{ false };
    std::unique_ptr<plume::RenderTexture> standIn;
    std::unique_ptr<plume::RenderFramebuffer> standInFramebuffer;
    plume::RenderFormat standInFormat = plume::RenderFormat::UNKNOWN;
    static constexpr uint32_t kStandInSize = 64;
    bool NoWindowImage() const;
    void EnsureStandIn();
    uint64_t frameCount = 0;
    // Time the last Present spent pacing, in the swap chain's present, and waiting for the GPU (ns).
    std::atomic<int64_t> lastPaceNs = 0, lastSwapNs = 0, lastGpuWaitNs = 0;
    bool asyncPresent = true;        // AC6_ASYNC_PRESENT=0: every frame presented on the calling thread

    // Creates the window and the device. Returns false (and logs) if no Vulkan device is usable. The game
    // then runs headless.
    bool Init();
    const SDL_DisplayMode* ChooseDisplayMode();
    bool Ready() const { return device != nullptr; }

    // The command list for the current frame (begins the frame on first use).
    plume::RenderCommandList& Commands();
    // Ends the frame: transitions the back buffer, submits, presents, waits for the previous frame (all
    // frames with `waitAll`: readbacks), pumps window events. With frame pacing on, the present waits until
    // `vblanks` 60 Hz periods have passed since the last one. Afterwards `slot` is the next frame's, and the
    // GPU is done with everything that slot used before.
    // pumpEvents: pump the window's events and input (false when the renderer's worker presents: the window's
    // thread pumps them).
    void Present(uint32_t vblanks = 0, bool waitAll = false, bool pumpEvents = true);
    void PumpEvents();   // the window's events and the game's input, once a frame on the window's thread

    // Back buffer of the current frame, for the final copy/blit.
    plume::RenderTexture* BackBuffer();
    plume::RenderFramebuffer* BackBufferFramebuffer();
    uint32_t BackBufferWidth() const;
    uint32_t BackBufferHeight() const;
    // The output: AC6_OUTPUT=hdr (a float swap chain in scRGB, the picture at AC6_HDR_PAPER_WHITE
    // nits), AC6_SDR_10BIT=1 for deep (a 10:10:10:2 SDR swap chain in sRGB: more than 8 bits to the display), default
    // the 8-bit swap chain.
    Output OutputMode() const { return outputMode; }
    plume::RenderFormat BackBufferFormat() const { return backBufferFormat; }

    // The display mode applied: "0" a window, "borderless", "1" exclusive (the window's thread: Alt+Enter, ui/overlay.cpp).
    const std::string& DisplayModeNow() const { return fullscreenNow; }
private:
    struct PresentJob
    {
        uint32_t slot = 0, image = 0;
        bool hasImage = false;
        uint32_t vblanks = 0;
        uint64_t timeline = 0;   // the value its submission signals
        uint32_t breadcrumbs = 0;   // the GPU breadcrumbs made up to its end (gpu/breadcrumbs.h)
    };
    void* timeline = nullptr;    // VkSemaphore (timeline)
    std::atomic<uint64_t> timelineIssued{ 0 };
    void BeginFrame();
    void AcquireImage();
    void CreateFramebuffers();
    // Live changes (the settings overlay).
    std::string WantedOutput() const;
    bool HdrHeldBack() const;               // HDR on in the settings, off for the display in the system
    void CreateSwapChain();
    void ApplyWindowSettings();
    void ReadDisplayInfo();                 // the window's display: HDR, its peak, its refresh rate (when not exclusive)
    uint32_t display = 0;                   // the monitor chosen (an SDL_DisplayID: AC6_MONITOR, else the primary)
    plume::RenderWindow renderWindow{};
    std::string outputWanted;               // the output the swap chain was made for (WantedOutput)
    std::string fullscreenNow, refreshNow;  // the display mode and refresh rate applied
    bool vsync = true;
    bool presentWait = false;               // VK_KHR_present_wait on (CreateSwapChain)
    void Pace(uint32_t vblanks);
    void Execute(const PresentJob& job);   // submits the frame, paces, presents
    void CheckDeviceLost();                // a lost device: stops the game with a message
    void PresentLoop();
    void WaitPresentIdle();
public:
    // Applies a live change of the output (on the render thread): the swap chain built again. True when it was.
    bool ApplyOutputChange();
    void WaitGpuIdle();
private:

    std::chrono::steady_clock::time_point nextPresent{};
    // The present thread's one job (a frame waits for the one before it to be handed over).
    std::mutex presentMutex;
    std::condition_variable presentCv;
    bool presentQueued = false, presentThreadStarted = false;
    PresentJob presentJob;
    std::mutex swapMutex;   // acquire (the recording thread) and present (the present thread)
};

extern Video g_video;

} // namespace ac6::gpu
