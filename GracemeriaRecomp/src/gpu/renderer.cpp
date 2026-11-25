#include "gpu/renderer.h"
#include "game/cockpit.h"
#include "game/draw_distance.h"
#include "game/ui_layers.h"
#include "gpu/breadcrumbs.h"
#include "gpu/capture.h"
#include "gpu/deferred_commands.h"
#include "gpu/fsr.h"
#include "gpu/render_queue.h"
#include "gpu/texture_mods.h"

#include "crash.h"
#include "endian.h"
#include "fast_hash.h"
#include "heap.h"
#include "kernel/clock.h"
#include "gpu/internal_shaders.h"
#include "../../../thirdparty/smaa/AreaTex.h"
#include "../../../thirdparty/smaa/SearchTex.h"
#include "gpu/shaders.h"
#include "gpu/video.h"
#include "host_cpu.h"
#include "gpu/xenos.h"
#include "gpu/xenos/shader_common.h"
#include "log.h"
#include "memory.h"
#include "platform/platform.h"
#include "simd.h"
#include "settings.h"
#include "ui/overlay.h"

#include <plume_render_interface_builders.h>
#include <plume_vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// The embedded files (tools/cmake/embed_files.cmake): the shipped pipeline list.
const unsigned char* EmbeddedFileData(const char* name, size_t* size);

namespace ac6::gpu {

using namespace plume;

namespace {

// --- Guest device layout (AC6, June 2007 XDK) ----------
constexpr uint32_t kDevVsConstants = 0x780;     // 256 x float4, big-endian
constexpr uint32_t kDevPsConstants = 0x1780;    // 256 x float4 (the shaders use 224)
constexpr uint32_t kDevTextureFetch = 0x480;    // 26 x 24 bytes (sampler s at +s*24)
constexpr uint32_t kDevVertexFetch = 0x6F0;     // 18 x 8 bytes, stream s at +(17 - s) * 8
[[maybe_unused]] constexpr uint32_t kDevStreamStride = 12520;    // u8 per stream: stride / 4
[[maybe_unused]] constexpr uint32_t kDevVertexDecl = 0x2E24;
constexpr uint32_t kDevPixelShader = 0x318C;
constexpr uint32_t kDevVertexShader = 0x3190;

// GPU register shadows the D3D library keeps up to date (found through the render-state setters):
constexpr uint32_t kDevColorMask = 10460;     // bits 0-3: RT0 write mask
constexpr uint32_t kDevAlphaRef = 10500;      // float, 0..1
constexpr uint32_t kDevDepthControl = 10548;  // bit 1 z enable, bit 2 z write, bits 4-6 z func
// RB_STENCILREFMASK (0x210D) and RB_STENCILREFMASK_BF (0x210C): ref bits 0-7, mask 8-15, write mask 16-23.
constexpr uint32_t kDevStencilRefMask = 0x2900, kDevStencilRefMaskBack = 0x28FC;
constexpr uint32_t kDevBlendConstant = 0x28E0;  // RB_BLEND_RED/GREEN/BLUE/ALPHA (0x2105-0x2108), floats
constexpr uint32_t kDevBlendControl = 10552;  // RT0: src 0-4, op 5-7, dst 8-12, alpha src 16-20, op 21-23, dst 24-28
constexpr uint32_t kDevColorControl = 10556;  // bits 0-2 alpha func, bit 3 alpha test enable
constexpr uint32_t kBlendDisabled = 0x00010001;   // ONE / ZERO / ADD for colour and alpha

// Vertex declaration object: element count at +24, 12-byte elements from +52.
[[maybe_unused]] constexpr uint32_t kDeclCount = 24, kDeclElements = 52;

// Render targets (SetRenderTarget 0x821DD260 / SetDepthStencilSurface 0x821DD5C8) and the index buffer.
constexpr uint32_t kDevIndexBuffer = 0x308C;    // D3DIndexBuffer*
constexpr uint32_t kDevRenderTargets = 0x3090;  // 4 x D3DSurface*
constexpr uint32_t kDevDepthStencil = 0x30A0;   // D3DSurface*
constexpr uint32_t kDevViewport = 0x2908;       // PA_CL_VPORT_X/Y/Z SCALE/OFFSET (0x210F-0x2114), floats
constexpr uint32_t kDevVteControl = 0x294C;     // PA_CL_VTE_CNTL (0x2206)
// Register shadows (Xbox D3D flushes them from these blocks, sub_821DEA48): 0x2000 +0x2880, 0x2100 +0x28CC,
// 0x2180 +0x2920, 0x2200 +0x2934, 0x2280 +0x2964, 0x2300 +0x29B8, 0x2380 +0x2A50.
constexpr uint32_t kDevClipControl = 0x2944;    // PA_CL_CLIP_CNTL (0x2204): bit 16 CLIP_DISABLE, bit 19 DX_CLIP_SPACE_DEF
constexpr uint32_t kDevScModeControl = 0x2948;  // PA_SU_SC_MODE_CNTL (0x2205)
constexpr uint32_t kDevContextMisc = 0x2924;    // SQ_CONTEXT_MISC (0x2181): bits 2-3 sc_sample_cntl
constexpr uint32_t kDevVtxControl = 0x29C0;    // PA_SU_VTX_CNTL: bit 0 pix_center (0 Direct3D: centres at .0, 1 OpenGL: .5)
constexpr uint32_t kDevPointSize = 0x2964;      // PA_SU_POINT_SIZE (0x2280): height 0-15, width 16-31 (12.4 radius)
constexpr uint32_t kDevPointMinMax = 0x2968;    // PA_SU_POINT_MINMAX (0x2281): min 0-15, max 16-31 (12.4 radius)
constexpr uint32_t kDevInterpolatorControl = 0x2928;   // SQ_INTERPOLATOR_CNTL (0x2182): bits 16-31 sampling pattern (1 = centre)
constexpr uint32_t kDevPolyOffset = 0x2A50;     // PA_SU_POLY_OFFSET_FRONT/BACK_SCALE/OFFSET (0x2380-0x2383)

// D3DSurface: +0x18 RB_SURFACE_INFO (pitch, MSAA), +0x1C RB_COLOR_INFO / RB_DEPTH_INFO (EDRAM base,
// format in bits 16-19), +0x24 size: width = (v >> 18) + 1, height = ((v & 0xFFFF) >> 3) + 1.
constexpr uint32_t kSurfaceFormatInfo = 0x1C, kSurfaceSize = 0x24;
constexpr uint32_t kSurfaceInfo = 0x18;   // RB_SURFACE_INFO: pitch bits 0-13, MSAA bits 16-17
// D3DIndexBuffer: +0 common (bit 31 = 32-bit indices), +0x18 physical address. D3DTexture: fetch
// constant at +0x1C.
constexpr uint32_t kIndexBufferAddress = 0x18, kTextureObjectFetch = 0x1C;
// The worker's device image: the guest device's layout, as much as the calls read.
constexpr uint32_t kDeviceImageBytes = 0x4000;
uint32_t s_deviceImage = 0;
// Presents whose frame's guest data the worker has read (vertex copies done), and presents the game has issued
// (RendererFrame: the game's frame count, which the worker's runs behind by at most one).
std::atomic<uint64_t> s_dataReleased{ 0 }, s_issuedFrames{ 0 };

// Resolve flags (D3DDevice_Resolve 0x821E2BB8): source in bits 0-2 (0-3 colour target, 4 depth),
// 0x100 clear the colour target afterwards, 0x200 clear depth/stencil.
constexpr uint32_t kResolveDepth = 4, kResolveClearColor = 0x100, kResolveClearDepth = 0x200;
// Clear flags (D3DDevice_Clear): 1 << n = colour target n, 0x10 depth, 0x20 stencil (ClearInternal
// 0x821E1CF0 drops 0x30 when no depth surface is bound).
constexpr uint32_t kClearTargets = 0x0F, kClearDepth = 0x10, kClearStencil = 0x20;

// D24S8 and D24FS8 alike: float depth plus the 8-bit stencil.
constexpr RenderFormat kDepthFormat = RenderFormat::D32_FLOAT_S8_UINT;



// Shader-visible "shared constants" block (XenosRecomp's layout).
// Layout shared with xenos/shader_common.h and the translator: 32 slots per kind, indexed by hardware
// fetch constant (pixel samplers 0-15, vertex samplers 16+).
constexpr uint32_t kFetchSlots = 32;
constexpr uint32_t kUsedFetchSlots = 20;   // 16 pixel + 4 vertex samplers, 20+ overlap the vertex fetch constants
struct SharedConstants
{
    uint32_t texture2DIndices[kFetchSlots];
    uint32_t texture3DIndices[kFetchSlots];
    uint32_t textureCubeIndices[kFetchSlots];
    uint32_t samplerIndices[kFetchSlots];
    uint32_t booleans;
    uint32_t swappedTexcoords;
    float halfPixelOffsetX, halfPixelOffsetY;
    float alphaThreshold;
    uint32_t vertexIndexBase;          // guest vertex index of SV_VertexID 0 (the shaders' initial r0.x)
    float depthBias;                   // the constant polygon offset, added to the depth in the vertex shaders
    uint32_t zero;                     // always 0: the SPIR-V shaders OR values with it (SpirvRecompiler::Unknowable)
    uint64_t streamAddress[16];        // whole guest stream buffers for vertex fetches at computed indices
    uint32_t streamDwords[16];
    float pixelPosScale;               // guest pixels per host pixel of the render target (0.5 for promoted 4x MSAA)
    uint32_t overrideSamplers[16];     // fetches with their own filters: pixel shader 0-7, vertex shader 8-15
    // Point sprites: the per-vertex diameter's clamp range (pixels), NDC units per pixel,
    // and the diameter for vertex shaders that don't export one.
    float pointDiameterMin, pointDiameterMax, pointNdcPerPixelX, pointNdcPerPixelY;
    float pointConstantDiameterX, pointConstantDiameterY;
    uint32_t interpolatorCentroid;     // pixel shader inputs interpolated at the centroid (bit = register)
    uint32_t drawFlags;                // the draw's SPEC_CONSTANT_* bits, for SPIR-V backend shaders
    float resolutionScale;             // host pixels per guest pixel at the internal resolution
    uint32_t scaledTextures;           // fetch slots sampling images the game rendered (at the internal resolution)
    float resolutionScaleY;            // the vertical scale (resolutionScale is the horizontal one)
    uint64_t fallbackAddress;          // always-mapped memory computed-index fetches read outside their stream
    // TAA's motion vectors: the draw's clip position to last frame's (zero: none known), rows.
    float motion[16];
    // Rendered images at another scale than the draw's (FSR upscaling: the scene at the render
    // scale, the rest at the output's): those fetch slots, and that scale.
    uint32_t altScaledTextures;
    float altScaleX, altScaleY;
    // The relighting factors per kind (shaders.cpp NoteLightConstants' kinds 0-9), applied by
    // the shaders where they read those registers, 1 each without HDR from the scene.
    float relight[10];
    // The map's tile filter: the most taps along a footprint (0: off, the game's fetches).
    uint32_t tileTaps;
    // The clouds' rim light (NoteLightConstants' kind 10), 1 without HDR from the scene.
    float cloudRim;
    // The sea's reflection of the scene, for the map's pixel shaders: the image (0: none), the 3 x 3 matrix K that maps
    // a pixel position to its place in the image ((x, y, 1) x K = (u w, v w, w), sea[0-8] are K's rows) and the
    // strength (sea[9]).
    uint32_t seaImage;
    float sea[10];
    // The 2D snaps' host grid above the guest's size: a guest pixel boundary k is at host k x (px, py) + (ox, oy).
    float hostGridX, hostGridY, hostGridOX, hostGridOY;
    // The viewport the 2D is drawn with (whole host pixels), NDC to host pixels: host = ndc x (X, Y) + (OX, OY), y up.
    float hostViewX, hostViewY, hostViewOX, hostViewOY;
    // The 2D line draws' segments (primitive IDs 0-63) of strips drawn counterclockwise: their box's inside on their left.
    uint32_t lineFlip[2];
    uint32_t pad1072[2];
};
static_assert(offsetof(SharedConstants, vertexIndexBase) == 532 && offsetof(SharedConstants, depthBias) == 536 &&
    offsetof(SharedConstants, streamAddress) == 544 &&
    offsetof(SharedConstants, streamDwords) == 672 && offsetof(SharedConstants, pixelPosScale) == 736 &&
    offsetof(SharedConstants, overrideSamplers) == 740 && offsetof(SharedConstants, pointDiameterMin) == 804 &&
    offsetof(SharedConstants, pointConstantDiameterX) == 820 && offsetof(SharedConstants, interpolatorCentroid) == 828 &&
    offsetof(SharedConstants, drawFlags) == 832 && offsetof(SharedConstants, resolutionScale) == 836 &&
    offsetof(SharedConstants, scaledTextures) == 840 && offsetof(SharedConstants, resolutionScaleY) == 844 &&
    offsetof(SharedConstants, fallbackAddress) == 848 && offsetof(SharedConstants, motion) == 856 &&
    offsetof(SharedConstants, altScaledTextures) == 920 && offsetof(SharedConstants, altScaleX) == 924 &&
    offsetof(SharedConstants, relight) == 932 && offsetof(SharedConstants, tileTaps) == 972 && offsetof(SharedConstants, cloudRim) == 976 &&
    offsetof(SharedConstants, seaImage) == 980 && offsetof(SharedConstants, sea) == 984 && offsetof(SharedConstants, hostGridX) == 1024 && offsetof(SharedConstants, hostViewX) == 1040 && offsetof(SharedConstants, lineFlip) == 1056 && sizeof(SharedConstants) <= 1072);

constexpr uint32_t kNullTexture2D = 0, kNullTexture3D = 1, kNullTextureCube = 2;
// The uniform buffers' ranges: 256 VS and 224 PS float registers, the shared block. Each draw's
// allocations are at least this big, so a range never reaches past the upload ring.
constexpr uint32_t kConstantBlockBytes[3] = { 256 * 16, 224 * 16, 1072 };
constexpr uint32_t kTextureDescriptors = 65536, kSamplerDescriptors = 1024;
constexpr uint32_t kTargetWidth = kGuestWidth, kTargetHeight = kGuestHeight;
constexpr RenderFormat kTargetFormat = RenderFormat::B8G8R8A8_UNORM;

// --- Per-frame upload ring (host visible, device addressable) ---------------------------------------
struct UploadRing
{
    std::unique_ptr<RenderBuffer> buffer;
    uint8_t* memory = nullptr;
    uint64_t deviceAddress = 0;
    uint64_t size = 0, used = 0;   // per frame
    uint64_t base = 0;             // the current frame's part: one per frame in flight

    void StartFrame(uint32_t slot)
    {
        base = uint64_t(slot) * size;
        used = 0;
    }

    struct Allocation { uint8_t* memory; uint64_t offset; uint64_t deviceAddress; };

    // Returns memory == nullptr when the frame's data doesn't fit (the caller drops that work): wrapping
    // would overwrite data the frame's recorded commands still read.
    Allocation Allocate(uint64_t bytes, uint64_t alignment = 256)
    {
        const uint64_t at = (used + alignment - 1) & ~(alignment - 1);
        if (at + bytes > size)
        {
            static uint32_t warned = 0;
            if (warned++ < 8)
                log::Error("Upload ring full ({} MB, {} bytes requested): work dropped this frame", size >> 20, bytes);
            return { nullptr, 0, 0 };
        }
        used = at + bytes;
        return { memory + base + at, base + at, deviceAddress + base + at };
    }
};

// --- Host textures for guest textures -----------------------------------------------------------------
struct HostTexture
{
    std::unique_ptr<RenderTexture> texture;
    std::unique_ptr<RenderTextureView> view;
    uint32_t descriptor = 0;
    uint64_t contentHash = 0;
    uint64_t lastFrame = 0;
    uint32_t levels = 0;   // mip levels the host image was created with
    // Write tracking: the generation its guest pages were watched from (0: not
    // watched), and textures the CPU rewrites all the time, which are hashed instead of watched.
    uint64_t watchGeneration = 0;
    uint8_t changeStreak = 0;
    uint16_t unchangedChecks = 0;
    bool dynamic = false;
    bool replaced = false;                 // the image is a mod's replacement (gpu/texture_mods.h)
    uint64_t recordedFrame = UINT64_MAX;   // listed for the overlay's texture browser in that frame
    uint64_t bytes = 0;                    // the image's size (its upload's), counted in s_r.textureBytes
};

// A host image standing in for a guest render surface (one per D3DSurface object, no EDRAM) or for
// the destination texture of a resolve (one per guest base address). Resolve destinations never go
// through guest memory: textures that point at them sample the host image directly.
struct HostImage
{
    std::unique_ptr<RenderTexture> texture;
    std::unique_ptr<RenderTextureView> view;
    uint32_t descriptor = 0;
    RenderFormat format = RenderFormat::UNKNOWN;
    uint32_t width = 0, height = 0;
    bool depth = false;
    bool fresh = false;          // created, not yet cleared
    float resX = 0.0f, resY = 0.0f;   // host pixels per guest pixel it was made at (0: the renderer's, the UI image's differ)
    uint32_t edramBase = UINT32_MAX;   // render surfaces: EDRAM base (RB_*_INFO bits 0-11)
    uint32_t scale = 1;                // render surfaces: host pixels per guest pixel (2 for 4x MSAA surfaces)
    uint64_t colorVersion = 0;         // colour surfaces: the EDRAM write version this image holds
    uint64_t depthVersion = 0;         // depth surfaces: the EDRAM write version this image holds
    RenderTextureLayout layout = RenderTextureLayout::UNKNOWN;
    // Resolve targets as textures: one view per fetch-constant swizzle (view, descriptor), as uploads have.
    FastMap<uint32_t, std::pair<std::unique_ptr<RenderTextureView>, uint32_t>> swizzled;
    // Multisampled render surfaces: the sample count, and the single-sampled copy that
    // everything reading the surface uses (colour averaged, depth and stencil from sample 0), made when it
    // is read after the surface was drawn.
    uint32_t samples = 1;
    std::unique_ptr<HostImage> single;
    bool singleCurrent = false;
    uint32_t guestFormat = 0;      // render surfaces: RB_COLOR_INFO / RB_DEPTH_INFO format (depth: 0 D24S8, 1 D24FS8)
    // A float image holding 8_8_8_8 data (HDR from the scene: the canvas and its resolves): sampled with
    // the B8G8R8A8 images' component order (FetchComponentMapping).
    bool bgraOrder = false;
    uint64_t depthAliasSeq = 0;    // colour surfaces: the depth write at their EDRAM base they were filled from
    std::unique_ptr<HostImage> stencilCopy;   // single-sampled depth images: the stencil as R8
    // Depth images: a view of the stencil, read by depth resolves, multisampled ones also a
    // view of their samples' depth (single-sampled ones' is `view`).
    std::unique_ptr<RenderTextureView> msDepthView, stencilView;
    uint32_t msDepthDescriptor = 0, stencilDescriptor = 0;
    // Resolve targets whose mip levels the game renders and resolves itself: the mip
    // chain, a sampled-only image beside the level-0 one (which stays a render target), with level 0
    // copied in at every level-0 resolve. The levels written so far (bit = level). Textures sample the
    // levels written contiguously from 0.
    std::unique_ptr<HostImage> mipChain;
    uint32_t mipLevels = 1;        // mip chains: levels the image has
    uint32_t mipWritten = 0;
    bool swizzleTrace = false;     // AC6_TRACE_SWIZZLE: written by an EDRAM swizzle alias pass, or resolved from one
    // Resolve swaps: a full-size colour resolve gives the surface's image to its target and the
    // target's old image to the surface. The surface's contents are then the target's (staleHolder) until they are
    // copied back (RestoreStale: the copy the resolve skipped) or overwritten whole (a clear, a full-screen opaque
    // draw), when nothing need come back. staleOf: the target's side of the link.
    HostImage* staleHolder = nullptr;
    HostImage* staleOf = nullptr;
};

struct Renderer
{
    bool ready = false;
    std::unique_ptr<RenderPipelineLayout> layout;
    std::unique_ptr<RenderDescriptorSet> textureSet;
    // Set 4: the draw's VS, PS and shared constants as dynamic uniform buffers into the upload ring.
    std::unique_ptr<RenderDescriptorSet> constantSet;
    std::unique_ptr<RenderDescriptorSet> samplerSet;
    std::unique_ptr<RenderTexture> nullTextures[3];
    std::unique_ptr<RenderTextureView> nullViews[3];
    std::unique_ptr<RenderSampler> samplers[2];   // 0 = linear clamp, 1 = linear wrap
    // Samplers made from texture fetch constants, by their sampling bits (see GetSampler).
    FastMap<uint64_t, uint32_t> samplerIndices;
    std::map<std::array<uint32_t, 12>, uint32_t> samplersByState;   // a sampler's index by its state (see GetSampler)
    std::vector<std::unique_ptr<RenderSampler>> fetchSamplers;
    uint32_t nextDescriptor = 3;

    UploadRing upload;

    std::unique_ptr<RenderTexture> target;      // stands in for the guest back buffer
    std::unique_ptr<RenderTextureView> targetView;
    std::unique_ptr<RenderFramebuffer> targetFramebuffer;
    uint32_t targetDescriptor = 0;
    RenderTextureLayout targetLayout = RenderTextureLayout::UNKNOWN;
    // The internal resolution: host pixels per guest pixel of everything the game renders
    // (render surfaces, resolve targets), and the guest frame's host size (the target above).
    float resX = 1.0f, resY = 1.0f;   // host pixels per guest pixel on each axis
    RenderFramebuffer* markedFramebuffer = nullptr;   // AC6_GPU_PASSES: the draws' pass last marked
    RenderFramebuffer* crumbFramebuffer = nullptr;    // the draws' target of the last breadcrumb (gpu/breadcrumbs.h)
    uint32_t targetWidth = kGuestWidth, targetHeight = kGuestHeight;
    // Descriptors of rendered images: the scale they were rendered at (0: not rendered).
    std::vector<std::array<float, 2>> scaledDescriptors = std::vector<std::array<float, 2>>(kTextureDescriptors);
    // The descriptors last given to a fetch that reads a depth image as k_24_8: a shadow map (ShadowDetail).
    std::vector<bool> shadowDescriptors = std::vector<bool>(kTextureDescriptors);
    std::set<uint32_t> swizzleTraceDescriptors;   // AC6_TRACE_SWIZZLE

    std::unique_ptr<RenderShader> copyVs, copyPs, depthCopyPs, gammaPs, gammaHdrPs;
    // SMAA on the scene image: its passes, SMAA's precomputed lookup textures, its targets, a point sampler.
    std::unique_ptr<RenderShader> smaaEdgePs, smaaWeightPs, smaaBlendPs;
    std::unique_ptr<RenderPipeline> smaaEdgePipeline, smaaWeightPipeline, smaaBlendPipeline;
    // SMAA's blend, TAA and the sharpening pass into a float image (with HDR from the scene the scene image is
    // R16G16B16A16_FLOAT).
    std::unique_ptr<RenderPipeline> smaaBlendPipelineFloat, taaPipelineFloat, sharpenPipelineFloat;
    std::unique_ptr<RenderTexture> smaaArea, smaaSearch;
    std::unique_ptr<RenderTextureView> smaaAreaView, smaaSearchView;
    uint32_t smaaAreaDescriptor = 0, smaaSearchDescriptor = 0;
    std::unique_ptr<RenderSampler> pointSampler;
    // The output pass: into the window's format, and into the target's for screenshots.
    std::unique_ptr<RenderShader> outputPs;
    std::unique_ptr<RenderPipeline> outputPipeline, outputTargetPipeline;
    // AC6_HDR_SIM=1 (tests): the HDR output pass rendered on screenshot frames into this float image
    // (scRGB: 1 = 80 nits) and dumped as raw_hdr_out_<frame>.bin, without an HDR display.
    std::unique_ptr<RenderPipeline> outputSimPipeline;
    HostImage hdrSimImage;
    std::unique_ptr<RenderBuffer> simReadback;
    uint64_t simDumpFrame = ~0ull;
    // The settings overlay (src/ui/overlay.cpp): Dear ImGui's triangles over the window's image, and its textures by id.
    std::unique_ptr<RenderShader> imguiVs, imguiPs;
    std::unique_ptr<RenderPipeline> imguiPipeline, imguiTargetPipeline;   // into the window's format, the target's (screenshots)
    struct OverlayTexture { std::unique_ptr<RenderTexture> texture; std::unique_ptr<RenderTextureView> view; uint32_t descriptor = 0, width = 0, height = 0; };
    std::unordered_map<uint32_t, OverlayTexture> overlayTextures;
    HostImage outputShot;
    // The frame before its 2D layers: the HDR output keeps what they drew at paper white.
    HostImage preUi, preUiOut;   // as drawn, and through the gamma ramp (at present)
    // HDR from the scene: the frame and the frame before its UI through the gamma ramp into float
    // images (values above 1 kept), for the output pass. The frames they were drawn.
    HostImage hdrFrame, hdrPreUi;
    uint64_t hdrFrameFrame = ~0ull, hdrPreUiFrame = ~0ull;
    // The HDR frames taken by the output pass as the game drew them: the frame it holds for, its texture word
    // (the frame | the frame before the UI << 16) and its ramp word (bit 31 | the ramp's image, 0: no ramp).
    uint64_t hdrDirectFrame = ~0ull;
    uint32_t hdrDirectTextures = 0, hdrDirectRamp = 0;
    HostImage rampImage;
    std::unique_ptr<RenderShader> gammaLutPs;
    std::unique_ptr<RenderPipeline> gammaLutPipeline;
    std::unique_ptr<RenderPipeline> copyHdrPipeline, gammaHdrPipeline;   // into float images
    // Every additive effect's draw (fire, explosions, flares, glows), into a layer per sample count (the scene
    // surface is multisampled, the effects surface not), composed after the composite with the lit highlights compressed.
    HostImage hdrGlow, hdrGlowMs, hdrComposeSource;
    uint64_t hdrGlowFrame = ~0ull, hdrGlowMsFrame = ~0ull;
    // The scene's lights added into the effects pass's layer before its first cloud is drawn.
    std::unique_ptr<RenderPipeline> glowMergePipeline;
    std::unique_ptr<RenderShader> glowMergePs;
    // The sun's disc on screen this frame (the compose pass softens it there only): the frame, and
    // the union of its draws' boxes, 0-1 across the scene (x0, y0, x1, y1).
    uint64_t sunFrame = ~0ull;
    float sunBox[4] = {};
    // The cloud particles' draws again into a float layer (their contribution, alpha blended over nothing), for
    // the cloud lift.
    HostImage hdrSky, hdrSkyMs;
    uint64_t hdrSkyFrame = ~0ull, hdrSkyMsFrame = ~0ull;
    std::unique_ptr<RenderShader> hdrComposePs;
    std::unique_ptr<RenderPipeline> hdrComposePipeline;
    // The UI at the frame's size while the 3D is rendered at AC6_RENDER_SCALE times it: the frame's
    // target as the UI starts, scaled into this image, which the UI then draws into and the present shows.
    HostImage uiImage;
    uint64_t uiImageFrame = ~0ull;
    uint32_t sceneWidth = 0, sceneHeight = 0;   // the rendered frame's size (the output target's: targetWidth)
    float uiResX = 1.0f, uiResY = 1.0f;         // the UI's host pixels per guest pixel
    float sceneResX = 1.0f, sceneResY = 1.0f;   // the rendered frame's (resX / resY outside the UI's draws)
    uint64_t preUiFrame = ~0ull, preUiOutFrame = ~0ull;
    uint64_t sceneDrawFrame = ~0ull, allUiFrame = ~0ull;   // a non-UI draw into the frame's target, a frame all UI
    // Temporal AA: its pass (colour + history), the scene surface it jitters, the history, last
    // frame's view-projection.
    std::unique_ptr<RenderShader> taaPs, sharpenPs;
    // Frosted glass at the sides of a 16:9 picture: the small images and the frame's copy with it.
    std::unique_ptr<RenderShader> pillarFrostPs;
    std::map<uint32_t, std::unique_ptr<RenderPipeline>> pillarFrostPipelines;
    HostImage frostA, frostB, frostFrame;
    std::unique_ptr<RenderPipeline> taaPipeline, sharpenPipeline;
    // FSR 3.1 in TAA's place: on, its motion completion pass, every pixel's motion and its result.
    bool fsr = false;
    std::unique_ptr<RenderShader> motionCompletePs;
    std::unique_ptr<RenderPipeline> motionCompletePipeline;
    HostImage fsrMotion, fsrOutput, fsrReactive;
    HostImage fsrAutoReactive;   // AC6_FSR_LATE: FSR's own reactive mask (the effects layer)
    std::unique_ptr<RenderFramebuffer> fsrMotionFramebuffer;   // fsrMotion and fsrReactive, the motion pass's targets
    const RenderTexture* fsrMotionFramebufferOf[2] = {};
    // FSR upscaling: the scene's render surfaces (the guest's surface objects, as the last scene
    // resolve had them, that frame) at the render scale, everything else at the frame's. The scene's colour resolve at the
    // render size (FSR's input), the mip bias of the scene's textures.
    bool upscaling = false;
    bool lateUpscale = false;   // AC6_FSR_LATE: the 3D frame at the render size, FSR before its first 2D layer
    bool lateNative = false;    // AC6_FSR_LATE at the native ratio: FSR on the finished 3D frame, in place
    uint64_t lateFsrFrame = ~0ull;
    uint64_t fsrRanFrame = ~0ull;   // the last frame FSR ran (its jitter only while it does)
    float worldResX = 1.0f, worldResY = 1.0f;
    uint32_t worldColorSurface = 0, worldDepthSurface = 0;
    uint64_t worldFrame = ~0ull;
    // The frame the scene image's post steps (ambient occlusion to FSR) last ran in: once a frame.
    uint64_t scenePostFrame = ~0ull;
    HostImage fsrColor;
    float worldMipBias = 0.0f;
    const HostImage* taaSurface = nullptr;
    // The frame the scene was last resolved from it: a screen without a 3D scene draws its 2D into
    // the same surface, and must not be jittered (TaaSurfaceLive).
    uint64_t taaSurfaceFrame = 0;
    HostImage taaHistory[2], taaColor, taaSharpened;
    uint32_t taaHistoryIndex = 0;
    double taaPrevVp[16] = {};
    bool taaHavePrev = false;
    uint64_t taaLastFrame = 0;
    double taaVp[16] = {};
    uint64_t taaVpFrame = ~0ull;
    // This frame's view-projections in c116-c119 among the scene's perspective draws, with how many draws used each:
    // the world's is the one most draws use.
    struct TaaVpCandidate { double m[16]; uint32_t draws; };
    std::vector<TaaVpCandidate> taaVpCandidates;
    // A scene with no world camera in c116-c119 (the hangar's shaders take each object's object x camera): the draws'
    // position matrices, of which the one most use (the static building's) stands in for the camera.
    std::vector<TaaVpCandidate> taaObjectCandidates;
    uint64_t taaObjectFrame = ~0ull;
    // TAA's motion vectors: the scene surface's second colour target (each pixel's movement since
    // last frame), its single-sampled copy, the framebuffers with it, and each object's c116-c119 transform by draw
    // (this frame's and last frame's, keyed by what the draw draws, instances in drawing order).
    HostImage motion, motionSingle;
    std::map<std::pair<const RenderTexture*, const RenderTexture*>, std::unique_ptr<RenderFramebuffer>> motionFramebuffers;
    // The late 3D pass's MSAA: the multisampled copies of the composite target and its depth the
    // late draws go to, what they copy, the effects composite's target this frame.
    HostImage lateColor, lateDepth;
    HostImage* lateSource = nullptr;
    HostImage* lateDepthSource = nullptr;
    bool lateActive = false;
    // Only the late draws' screen rectangle is loaded and resolved: what is loaded so far this pass,
    // each late mesh's own points (rebuilt from one capture of its clip positions: lateBoundsGs), by what the draw draws.
    RenderRect lateLoaded{};
    struct LateMesh
    {
        std::vector<std::array<double, 3>> points;   // the mesh in the space its transform takes to clip space
        uint64_t captureFrame = ~0ull;                 // pending: the frame captured, its slots and transform's inverse
        uint32_t base = 0, count = 0;
        double inverse[16] = {};
        bool unusable = false;                         // no capture possible (not a triangle list, a degenerate transform)
    };
    FastMap<uint64_t, LateMesh> lateMeshes;
    std::unique_ptr<RenderShader> lateBoundsGs;
    std::unique_ptr<RenderShader> line2dGs;   // the 2D's lines above the guest's size as rectangles (internal_shaders.hlsl LINE_2D)
    std::unique_ptr<RenderBuffer> lateCapture;           // this frame's captured clip positions
    std::unique_ptr<RenderBuffer> lateReadback[4];       // copied at the pass's end, read four frames later
    uint32_t lateCaptureUsed = 0;
    uint64_t lateCaptureFrame = ~0ull;
    const HostImage* compositeTarget = nullptr;
    uint64_t compositeFrame = ~0ull;
    uint64_t motionClearedFrame = ~0ull;
    uint32_t motionStats[3] = {};   // this frame's draws writing motion, with a transform, paired (AC6_TAA_DEBUG=7)
    bool motionClearPending = false;
    struct MotionTransforms
    {
        uint64_t frame = ~0ull;
        std::unordered_map<uint64_t, std::vector<std::array<double, 16>>> byObject;
    } motionCurrent, motionPrevious;
    // Volumetric smoke: the harvesting geometry shader and the volume pass, the particle records
    // (one buffer per frame in flight, and one spare), the frame's camera and lighting as the smoke draws had them, the
    // noise volume.
    std::unique_ptr<RenderShader> volHarvestGs, volVs, volPs, volComposePs;
    // The cloudscape: the cloud puffs as one field. Its shaders, the top-down map (four levels side
    // by side), the puffs' bounds on screen, the marched image. The frame it ran in.
    std::unique_ptr<RenderShader> volCloudVs, volCloudMapPs, volCloudBoundsPs, volCloudMarchPs, volCloudTemporalPs, volCloudOverPs;
    std::unique_ptr<RenderShader> volCloudScenePs, volCloudDepthPs, volCloudDepthOutPs, volCloudGatePs;
    // The cloudscape and the game's effects pass: the frame the game's half-size copies were drawn in and their
    // target, the pass's targets once its first draw came, whether that began with the background cloud sprites, and the
    // frame the gate (those sprites faded by the clouds, the clouds' depth for the rest) ran in.
    uint64_t fxArmedFrame = ~0ull, fxTargetFrame = ~0ull, fxGateFrame = ~0ull, cloudEarlyFrame = ~0ull;
    HostImage fxBackup;   // the pass's target as the game's scene copy left it, before the sprites
    HostImage* fxColor = nullptr;
    HostImage* fxDepth = nullptr;
    uint32_t fxWidth = 0, fxHeight = 0, fxSamples = 1;
    bool fxSprites = false;
    // The clouds into the scene before its resolve: the scene's colour surface as the last scene resolve had it
    // (its draws give the frame's camera), and the last frame's harvest (the frame, its records' count and camera:
    // cloudRecFrame, cloudRecCount and cloudRecVp, further down).
    const HostImage* cloudSceneSurface = nullptr;
    // Ambient occlusion (AC6_AO): the passes' shaders and pipelines (by shader and target format),
    // XeGTAO's view-space depth levels and its two working images (the visibility and the edges) and the copy of the scene
    // the apply pass reads.
    std::unique_ptr<RenderShader> aoDepthPs, aoDepthMipPs, aoMainPs, aoDenoisePs, aoApplyPs, aoSubjectPs;
    std::map<std::pair<const RenderShader*, uint32_t>, std::unique_ptr<RenderPipeline>> aoPipelines;
    HostImage aoDepth[5], aoWork[2], aoCopy;
    // The last frame the map (the terrain) was drawn in: no occlusion in scenes without it (RunAmbientOcclusion).
    uint64_t aoMapFrame = ~0ull;
    // The terrain's depth alone: the map's draws again, into a depth image of the scene's. A pixel of the scene
    // with that depth is terrain. The frame it was last drawn in and the scene depth it stands for.
    HostImage aoTerrainDepth;
    uint64_t aoTerrainFrame = ~0ull;
    const HostImage* aoTerrainFor = nullptr;
    // The hangar: the occlusion on the aircraft alone. The aircraft is what is drawn into the scene after the
    // game puts its kept surroundings back (the second half of its frame), which the renderer sees as a
    // depth alias into the scene's depth: the frame and the image of the last one. With it the frame's noted draws are
    // the aircraft's (aoSubject), not a terrain's, and the depth made of them is what the occlusion reads.
    uint64_t aoAliasFrame = ~0ull;
    const HostImage* aoAliasDepth = nullptr;
    bool aoSubject = false;
    HostImage aoSubjectDepth;
    // The game's shadow blend as its shadow pass last set it (Draw): colour added, and the share of the destination
    // taken away.
    float aoShadow[4] = { 9.0f / 255.0f, 11.0f / 255.0f, 12.0f / 255.0f, 76.0f / 255.0f };
    // The projections of the frame's draws into the scene (NoteAoProjection): the one most draws use is the scene's.
    struct AoProjection { float tanX, tanY, clipZ; uint32_t draws, width, height; };
    std::vector<AoProjection> aoProjections;
    uint64_t aoProjectionsFrame = ~0ull;
    // The sea's reflection of the scene (AC6_SSR=1): the pass's shader here, the rest after the lens effects' members:
    // the pipelines (by the image's format), the image, the frame it was made in and the camera it was made with (its
    // view-projection, the jitter in pixels), and the cameras of this frame's sea draws by their target (the scene's is
    // the pass's).
    std::unique_ptr<RenderShader> waterSsrPs;
    // The enhanced bloom (bloom.hlsl): the passes' shaders and pipelines (by shader, format and blend),
    // the bright image and the pyramid's levels and results, the bright pass's threshold and offset as the game's
    // bright pass last had them, and the composite's bloom and star scales taken from its draw (RunDraw).
    std::unique_ptr<RenderShader> bloomBrightPs, bloomDownPs, bloomUpPs, bloomApplyPs;
    std::map<std::tuple<const RenderShader*, uint32_t, bool>, std::unique_ptr<RenderPipeline>> bloomPipelines;
    HostImage bloomLevels[7], bloomResults[7];
    float bloomBright[2] = { 0.5f, 1.0f }, bloomScales[2] = {};
    uint64_t bloomScalesFrame = ~0ull;
    // Light from the game's emitters (emitter_light.hlsl): the passes' shaders and pipelines, the light image and its
    // depth (the emitters' draws issued again), its ten levels, the copy of the scene the apply pass reads, and the
    // frame the emitters' draws were last noted in.
    std::unique_ptr<RenderShader> lightDown0Ps, lightDown0bPs, lightDownPs, lightApplyPs;
    std::map<std::pair<const RenderShader*, uint32_t>, std::unique_ptr<RenderPipeline>> lightPipelines;
    // (Ten levels, emitter_light.hlsl kLevels: the light's colour and distance, and where it is on screen.)
    HostImage lightColor, lightDepth, lightLevels[10], lightPlaces[10], lightCopy;
    uint64_t lightNotesFrame = ~0ull;
    // Object motion blur (motion_blur.hlsl): the pass's shader and pipelines (by the scene's format),
    // the copy of the scene it reads, and every pixel's motion made for it (the motion completion pass's shader with a
    // pipeline of its own: FSR's is only made when FSR is asked for at start).
    std::unique_ptr<RenderShader> motionBlurPs;
    std::map<uint32_t, std::unique_ptr<RenderPipeline>> motionBlurPipelines;
    std::unique_ptr<RenderPipeline> blurMotionPipeline;
    HostImage motionBlurCopy, blurMotion;
    // Optional lens effects (lens_fx.hlsl): the pass's shader, its pipelines by the picture's format,
    // the copy of the picture it reads. The last frame the scene's composite drew in (a frame with a 3D picture) and
    // the last frame the pass ran in (once, before the frame's first 2D layer draw).
    std::unique_ptr<RenderShader> lensFxPs;
    std::map<uint32_t, std::unique_ptr<RenderPipeline>> lensFxPipelines;
    HostImage lensFxCopy;
    uint64_t sceneCompositeFrame = ~0ull, lensFxFrame = ~0ull;
    std::map<uint32_t, std::unique_ptr<RenderPipeline>> waterSsrPipelines;
    HostImage seaReflection;
    uint64_t seaReflectionFrame = ~0ull;
    double seaReflectionVp[16] = {};
    float seaReflectionJitter[2] = {};
    const HostImage* seaReflectionSurface = nullptr;
    const HostImage* seaSceneSurface = nullptr;
    struct SeaView { const HostImage* target; double vp[16]; };
    std::vector<SeaView> seaViews;
    uint64_t seaViewsFrame = ~0ull;
    // The planar reflection (AC6_SSR=planar): the blur's shader and pipelines (by format), the mirrored scene's colour
    // and depth, the blur's intermediate image, and the frame the draws were noted in.
    std::unique_ptr<RenderShader> planarBlurPs;
    std::map<uint32_t, std::unique_ptr<RenderPipeline>> planarBlurPipelines;
    HostImage planarColor, planarDepth, planarBlur, planarSky, planarGlowAcross, planarGlow, planarLights;
    const HostImage* planarSceneDepth = nullptr;   // the scene's depth (the effects' image shares it)
    // The game's exposure as its scene composite last had it (the adapted luminance's 1 x 1, the band's low and high):
    // the planar reflection's glow passes its lights through the game's bright pass with it.
    uint32_t planarLuminance = 0;
    float planarBand[2] = { 0.25f, 0.5f };
    uint64_t planarNotesFrame = ~0ull;
    uint64_t cloudRecFrame = ~0ull;
    uint32_t cloudRecCount = 0;
    double cloudRecVp[16] = {};
    // cloudImage and cloudFront: the march's picture and the view depth it sees the cloud at (its two targets, their
    // framebuffer), cloudAccum: the picture blended into the last frames', this frame's and the last's in turn.
    HostImage cloudMap, cloudBounds, cloudImage, cloudFront, cloudAccum[2];
    HostImage cloudCoarse;   // the clearance map: where no puff is within a kilometre
    std::unique_ptr<RenderFramebuffer> cloudFramebuffer;
    const RenderTexture* cloudFramebufferOf[2] = {};
    uint64_t cloudscapeFrame = ~0ull;
    double cloudPrevVp[16] = {};          // the view-projection the last result was made with
    double cloudPrevCamera[3] = {};
    uint64_t cloudHistoryFrame = ~0ull;   // the frame cloudAccum's other image is of
    std::map<uint64_t, std::unique_ptr<RenderPipeline>> volPipelines;
    HostImage volImage;   // the volumes at a fraction of the resolution (premultiplied), composited onto the frame
    std::unique_ptr<RenderBuffer> volParticles[Video::kFramesInFlight + 1];
    uint32_t volCount = 0;
    std::unique_ptr<RenderBuffer> volDump;   // AC6_VOL_DUMP: the frame's records, read back
    uint32_t volDumpCount = 0;
    uint64_t volDumpFrame = 0;
    uint64_t volFrame = ~0ull;
    const HostImage* volTarget = nullptr;   // the effects pass: smoke drawn elsewhere (another view) isn't harvested
    bool volComposited = false;
    double volVp[16] = {};
    bool volHaveVp = false;
    uint64_t volVpFrame = ~0ull;   // the frame a smoke or fire draw last gave the camera
    float volFog[4] = {}, volSun[4] = {};
    // The cloud puffs' vertex constants c80-c98 as their draw had them (the cloudscape): fog, colours,
    // altitude bands, the light vector.
    float volDepthA = 0.0f, volDepthB = 1.0f;   // the depth buffer's z = a + b / (view depth), from the frame's camera
    float volCloud[19][4] = {};
    float volCloudFade[4] = {};   // c50, DebugZ_FadeDistance
    uint64_t volCloudFrame = ~0ull;
    std::unique_ptr<RenderTexture> volNoise;
    std::unique_ptr<RenderTextureView> volNoiseView;
    uint32_t volNoiseDescriptor = 0;
    std::unique_ptr<RenderPipeline> copyPipeline, depthCopyPipeline, depthResamplePipeline, gammaPipeline;
    std::unique_ptr<RenderShader> depthResamplePs, depthSwizzlePs, colorSwizzlePs;
    std::unique_ptr<RenderPipeline> depthSwizzlePipeline;
    std::map<RenderFormat, std::unique_ptr<RenderPipeline>> colorSwizzlePipelines;
    // Writing single-sampled contents into multisampled surfaces, by sample count:
    // depth (stencil cleared), one pass per stencil bit, colour by format.
    std::unique_ptr<RenderShader> depthLoadPs, colorLoadPs, stencilBitPs, texelCopyPs;
    std::map<RenderFormat, std::unique_ptr<RenderPipeline>> texelCopyPipelines;   // colour resolves as draws
    std::map<uint32_t, std::unique_ptr<RenderPipeline>> depthLoadPipelines, stencilBitPipelines;
    std::map<std::pair<RenderFormat, uint32_t>, std::unique_ptr<RenderPipeline>> colorLoadPipelines;
    // EDRAM aliasing of depth: the surface that last wrote depth at each EDRAM base, and a version that
    // advances with each write there (see SyncAliasedDepth).
    FastMap<uint32_t, std::pair<struct HostImage*, uint64_t>> edramDepth;
    FastMap<uint32_t, std::pair<struct HostImage*, uint64_t>> edramColor;
    // The order of colour and depth writes at each EDRAM base.
    FastMap<uint32_t, uint64_t> edramColorSeq, edramDepthSeq;
    uint64_t edramSeq = 0;
    std::unique_ptr<RenderShader> depthToColorPs;
    std::unique_ptr<RenderPipeline> depthToColorPipeline;

    FastMap<uint32_t, HostImage> surfaces;   // by guest D3DSurface address
    // HDR from the scene: the canvas's float twins (by owner), used from the frame's scene composite
    // (the frame hdrCanvasFrame) to its end. Before it the canvas holds the game's encoded scene, clipped at 1 as on the
    // console (8_8_8_8), and shares its image with the effects surface over the same EDRAM.
    FastMap<uint32_t, HostImage> hdrSurfaces;
    uint64_t hdrCanvasFrame = ~0ull;
    // Their resolves' float twins (by physical address, as `resolved`): a target can be resolved from the 8-bit canvas
    // (or the world) and from the float twin in one frame (A631B000 in missions, 05BEB000 in the hangar). Textures read
    // the one written last (hdrResolvedCurrent).
    FastMap<uint32_t, HostImage> hdrResolved;
    FastMap<uint32_t, bool> hdrResolvedCurrent;
    // Surfaces over the same EDRAM with the same host image: the guest surface whose image they
    // share, by EDRAM base, kind, format, samples and size, and each surface's owner (itself when not shared).
    FastMap<uint64_t, uint32_t> sharedSurfaces;
    FastMap<uint32_t, uint32_t> surfaceOwners;
    FastMap<uint32_t, HostImage> resolved;   // by physical base address of the destination
    // The stencil of depth resolves (R8, the 24_8 texel's X byte), by the same address: a depth resolve
    // read as 8_8_8_8 sees it.
    FastMap<uint32_t, HostImage> resolvedStencil;
    std::unique_ptr<RenderBuffer> stencilStaging;   // stencil aspect -> R8 image, through a buffer
    uint64_t stencilStagingSize = 0;
    std::map<std::pair<const RenderTexture*, const RenderTexture*>, std::unique_ptr<RenderFramebuffer>> framebuffers;
    // Depth resolves from multisampled surfaces: the R32F target and the R8 stencil one beside it.
    std::map<std::pair<const RenderTexture*, const RenderTexture*>, std::unique_ptr<RenderFramebuffer>> depthResolveFramebuffers;
    std::unique_ptr<RenderShader> depthCopyMsPs, depthCopySsPs;
    std::unique_ptr<RenderPipeline> depthCopyMsPipeline, depthCopySsPipeline;
    // Resources replaced mid-frame stay alive until the frame's commands have executed.
    std::vector<std::unique_ptr<RenderTexture>> retiredTextures;
    std::vector<std::unique_ptr<RenderBuffer>> retiredStaging;   // large-texture staging buffers of the frame
    std::vector<std::unique_ptr<RenderTextureView>> retiredViews;
    std::vector<std::unique_ptr<RenderFramebuffer>> retiredFramebuffers;
    std::vector<uint32_t> retiredDescriptors;   // of images retired this frame: free again once its GPU work is done
    std::vector<uint32_t> freeDescriptors;      // reused before new ones
    uint64_t textureBytes = 0;                  // the guest textures' images (HostTexture::bytes)
    // The resources retired while recording each frame in flight: released once its GPU work is done.
    struct InFlight
    {
        std::vector<std::unique_ptr<RenderTexture>> textures;
        std::vector<std::unique_ptr<RenderBuffer>> staging;
        std::vector<std::unique_ptr<RenderTextureView>> views;
        std::vector<std::unique_ptr<RenderFramebuffer>> framebuffers;
        std::vector<uint32_t> descriptors;
    } inFlight[Video::kFramesInFlight];

    // Predicated tiling (BeginTiling/EndTiling): the frame is drawn once at full size, so surfaces
    // bound while tiling are as large as the union of the tiles.
    bool tiling = false;
    uint32_t tilingWidth = 0, tilingHeight = 0;

    // Vertex data copied at submission, as plain records (a closure allocated per stream per draw would cost 6% of the
    // render thread), and diagnostics run then.
    struct VertexCopy
    {
        uint8_t* dst;
        const uint8_t* src;
        uint32_t vertexCount, stride, slot;
        uint64_t validBytes;   // within the fetch constant's size: the console reads zeros past it
        bool rect;
        const std::vector<RenderInputElement>* elements;   // rect lists: the layout (cached, outlives the frame)
        bool point = false;   // point lists drawn as sprites: each vertex six times
    };
    std::vector<VertexCopy> vertexCopies;
    // Vertex copies made when their draw is translated, on the worker while the game goes on,
    // instead of at submission: their sources are watched, and only those written since are copied again then.
    // A source rewritten two frames running (CPU-skinned meshes, particles) is copied at submission alone for a
    // while: a write fault per page each frame would cost more.
    struct SourceWatch { uint8_t streak = 0; uint64_t staticAgainFrame = 0; };
    FastMap<uint64_t, SourceWatch> vertexSources;   // by guest address << 32 | bytes
    struct EarlyCopy { VertexCopy copy; uint32_t guest, bytes; uint64_t generation; SourceWatch* source; };
    std::vector<EarlyCopy> earlyCopies;
    std::vector<std::function<void()>> pendingCopies;   // diagnostics, run after the copies
    // Whole streams for computed-index fetches, swapped at submission (split over threads: ~10 MB a frame).
    struct SwapFill { uint32_t* dst; const uint32_t* src; uint32_t dwords; };
    std::vector<SwapFill> swapFills;
    // Computed-index streams uploaded whole this frame (guest address, bytes): AC6_POOL_HAZARD=1 counts the game's
    // writes to them before the GPU is done with the frame.
    std::vector<std::pair<uint32_t, uint32_t>> wholeStreams;
    // Large computed-index streams (the particle and effect pools, ~10 MB a frame in missions, a few pages of
    // them written per frame) are kept across frames, a buffer per frame in flight, and only the pages
    // written since that buffer was last brought up to date are swapped again.
    struct PersistentStream
    {
        uint32_t start = 0, end = 0;   // physical addresses
        uint32_t wantStart = 0;        // a draw sliced it from lower: rebuilt from there next frame
        std::unique_ptr<RenderBuffer> buffers[Video::kFramesInFlight];
        uint32_t* memory[Video::kFramesInFlight] = {};
        uint64_t address[Video::kFramesInFlight] = {};
        uint64_t generation[Video::kFramesInFlight] = {};   // watch generation it is up to date with (0: never)
        uint64_t lastFrame = UINT64_MAX;
        uint8_t busyFrames = 0;        // consecutive updates rewriting most of it
        bool dynamic = false;          // rewritten every frame: uploaded whole like the small streams
        bool vertex = false;           // also drawn from as a vertex buffer (the static vertex cache)
        uint8_t writtenFrames = 0;     // vertex: consecutive updates finding any page written
    };
    FastMap<uint32_t, PersistentStream> persistentStreams;   // by end address
    // Vertex streams drawn from on consecutive frames: from the third, they are kept like the
    // computed-index streams above, instead of copied every frame. By end address.
    struct VertexStreamSeen { uint64_t lastFrame = UINT64_MAX; uint32_t frames = 0; };
    FastMap<uint32_t, VertexStreamSeen> vertexStreamsSeen;
    std::vector<PersistentStream*> streamsToUpdate;                     // used this frame
    // Whole streams for vertex fetches at computed indices, uploaded once per frame. Particle buffers
    // are slices of one pool whose fetch constants all run to the pool's end, so uploads are keyed by
    // the end address: (lowest start uploaded, its device address).
    FastMap<uint32_t, std::pair<uint32_t, uint64_t>> indexedStreams;
    std::vector<std::function<void()>> nextFrameChecks, dueChecks;   // diagnostics run one frame later
    uint32_t device = 0;                 // last guest device seen
    HostImage* frontBuffer = nullptr;    // last display-sized colour resolve
    uint64_t frontBufferFrame = UINT64_MAX;

    std::unique_ptr<RenderBuffer> readback;      // screenshots
    std::unique_ptr<RenderBuffer> rawReadback;   // AC6_DUMP_RESOLVES: raw texels of small float targets
    std::string rawName;
    uint32_t rawWidth = 0, rawHeight = 0;
    RenderFormat rawFormat = RenderFormat::UNKNOWN;
    // AC6_RESOLVE_STATS=<frame>: a raw copy of every resolve destination of that frame, in order.
    struct ResolveCapture { uint32_t index; std::string name; std::unique_ptr<RenderBuffer> buffer; uint32_t width, height; RenderFormat format; };
    std::vector<ResolveCapture> resolveCaptures;
    // Frame capture (gpu/capture.h): readbacks of the resolve targets at the start of the captured frames.
    struct ResolvedSnapshot { uint32_t kind, address; uint32_t width, height; RenderFormat format; std::unique_ptr<RenderBuffer> buffer; };
    std::vector<ResolvedSnapshot> resolvedSnapshots;
    // The pipeline ring's (AC6_PIPELINE_RING): per frame slot, read when the slot is free again (no wait).
    std::vector<ResolvedSnapshot> ringSnapshots[Video::kFramesInFlight];
    uint64_t ringSnapshotFrame[Video::kFramesInFlight] = {};
    uint64_t resolveStatsFrame = UINT64_MAX, resolveStatsFrames = 1;
    uint32_t resolveIndex = 0;
    std::set<uint64_t> screenshotFrames;
    bool screenshotPending = false;
    // The frame ring: the frame scaled to the ring's size, one readback per frame in flight, the
    // frames kept (the newest last), their frame numbers.
    HostImage ringImage;
    std::unique_ptr<RenderBuffer> ringReadback[Video::kFramesInFlight];
    bool ringPending[Video::kFramesInFlight] = {};
    uint64_t ringPendingFrame[Video::kFramesInFlight] = {};
    std::deque<std::pair<uint64_t, std::vector<uint8_t>>> ringFrames;
    std::vector<std::vector<uint8_t>> ringSpare;   // buffers of frames dropped, reused
    uint32_t ringWidth = 0, ringHeight = 0;
    std::chrono::steady_clock::time_point ringLast{};
    std::atomic<bool> ringDump{ false };
    std::string screenshotName;
    // AC6_DUMP_RESOLVES: the resolve targets / surfaces still to show and capture, one per frame.
    std::vector<std::pair<std::string, HostImage*>> dumpQueue;
    bool rectList = false;   // the draw being translated is a rectangle list (never culled)
    HostShader nullPixelShader;   // for draws without a pixel shader (depth-only passes, shadow maps)
    uint64_t dumpFrame = UINT64_MAX;

    FastMap<uint64_t, std::unique_ptr<PendingPipeline>> pipelines;   // ready, or being built
    // Bumped by anything that can change what a fetch constant samples within a frame without the constant
    // changing: resolves (a resolve target made or re-made, its mip levels, its stencil), restored captures.
    uint64_t textureEpoch = 0;
    FastMap<uint64_t, HostTexture> textures;
    std::vector<mods::FrameTexture> frameTextures;   // the textures this frame used (the overlay's browser)
    std::unordered_set<uint32_t> mippedReplacements;   // descriptors of replacements with mips: sampled through them
    uint64_t frame = 0;
    RenderFormat windowFormat = RenderFormat::UNKNOWN;   // what CreateWindowPipelines made them for
    bool windowOutputActive = false;
    uint64_t targetHeldFrame = UINT64_MAX - 1;   // the frame the output target holds as presented (the overlay's hold)
};

Renderer s_r;

// The frame number the per-frame noise is seeded from (volumetric jitter, TAA jitter, dithering): the live frame's,
// also in a replay of a capture that carries it (gpu/capture.cpp), so the replay draws the same.
std::atomic<uint64_t> s_frameSeedBase{ 0 };
uint64_t SeedFrame()
{
    return s_r.frame + s_frameSeedBase.load(std::memory_order_relaxed);
}

// AC6_GPU_FAULT_TEST=<frame>: a test of the device-lost report (gpu/breadcrumbs.h). From that frame the GPU reads through
// an unmapped address (0x1000), so it faults and the device is lost: in a draw of its own at the end of each frame
// (whatever the output path), and where it reads guest vertices at computed indices.
bool FaultTestDue()
{
    static const uint64_t faultFrame = [] { const char* v = std::getenv("AC6_GPU_FAULT_TEST"); return v && *v ? std::strtoull(v, nullptr, 10) : UINT64_MAX; }();
    if (s_r.frame < faultFrame)
        return false;
    static bool logged = false;
    if (!std::exchange(logged, true))
        log::Warn("AC6_GPU_FAULT_TEST: the GPU reads an unmapped address from frame {} (a test crash)", s_r.frame);
    return true;
}

uint64_t FaultTestAddress(uint64_t address)
{
    return FaultTestDue() ? 0x1000 : address;
}

// AC6_GPU_PASSES: an image in a pass label, "3840x2160 f24 x2" (size, RenderFormat, samples).
std::string PassImage(const HostImage* image)
{
    if (!image)
        return "-";
    return std::format("{}x{} f{}{}", image->width, image->height, int(image->format), image->samples > 1 ? std::format(" x{}", image->samples) : "");
}

// Starts a pass: a GPU crash breadcrumb at the caller's place (gpu/breadcrumbs.h), and with AC6_GPU_PASSES a timed pass
// (the label is made only then). The draws' passes start where their framebuffer changes.
template <class Label>
void MarkPass(Label&& label, std::source_location where = std::source_location::current())
{
    if (g_video.Ready())
        breadcrumbs::Mark(g_video.Commands(), s_r.frame, 0, 0, where);
    s_r.crumbFramebuffer = nullptr;
    if (!g_video.GpuPassTiming())
        return;
    g_video.MarkGpuPass(label());
    s_r.markedFramebuffer = nullptr;
}

// Guest pixels to host pixels at the internal resolution: rounded alike everywhere, so the
// edges of neighbouring rectangles meet. At scale 1 the identity.
int32_t ToHostX(double guest)
{
    return int32_t(std::lround(guest * s_r.resX));
}

int32_t ToHostY(double guest)
{
    return int32_t(std::lround(guest * s_r.resY));
}

// The scale an image was made at (host pixels per guest pixel: its own, else the rendered frame's).
std::array<float, 2> ImageScale(const HostImage& image)
{
    return { image.resX > 0.0f ? image.resX : s_r.sceneResX, image.resY > 0.0f ? image.resY : s_r.sceneResY };
}

// FSR upscaling: a guest render surface of the scene (as the last scene resolve, last frame or this
// one, had them), made and drawn at the render scale.
bool WorldSurface(uint32_t surface)
{
    return s_r.upscaling && surface && (surface == s_r.worldColorSurface || surface == s_r.worldDepthSurface) &&
        s_r.worldFrame != ~0ull && s_r.frame <= s_r.worldFrame + 1;
}

// The tree impostor prologue's surface: every frame the
// game renders each tree model near the camera into a 256x256 corner of a 384x384 surface and resolves it, then draws its
// mip levels from it and resolves each (8 trees in mission 01: 40 draws, 41 resolves). The textures are 256x256 by design
// and seen as small distant trees (forests), so AC6_IMPOSTOR_NATIVE=1 keeps the surface at the console's size instead of
// the internal resolution's. Off by default: the chain is bound by its 81 dependent steps (~8 us each), not by pixels,
// and at 1x it costs the same (under 1 ms a frame either way). The only 384x384 render surface.
uint32_t U32(uint32_t guest);

bool NativeSurface(uint32_t surface)
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_IMPOSTOR_NATIVE"); return v && *v == '1'; }();
    if (!enabled || !surface)
        return false;
    const uint32_t size = U32(surface + kSurfaceSize);
    return (size >> 18) + 1 == 384 && ((size & 0xFFFF) >> 3) + 1 == 384;
}

// The renderer's guest-to-host scale switched for a scope (the scene's draws, clears and resolves at the render scale).
struct ScopedScale
{
    float x, y;
    bool on;
    ScopedScale(bool active, float scaleX, float scaleY) : x(s_r.resX), y(s_r.resY), on(active)
    {
        if (on)
            s_r.resX = scaleX, s_r.resY = scaleY;
    }
    ~ScopedScale()
    {
        if (on)
            s_r.resX = x, s_r.resY = y;
    }
};

// Concurrency check: draws come from the game's render thread, present from the thread that calls the
// D3D swap. Counts overlaps (logged with the render stats).
std::atomic<int> s_inDraw{ 0 }, s_inPresent{ 0 };
std::atomic<uint64_t> s_overlaps{ 0 };

// Per-frame cost counters, logged every 120 frames with AC6_RENDER_STATS=1.
struct Stats
{
    uint64_t draws = 0, hashedBytes = 0, uploadedBytes = 0, uploads = 0, deferredUploads = 0, stagedUploads = 0;
    uint64_t vertexCopies = 0, vertexBytes = 0, repeatedVertexBytes = 0;
    uint64_t earlyBytes = 0, earlyKeptBytes = 0, swapBytes = 0;   // early copies made / found unchanged, stream swaps
    uint64_t cachedBytes = 0;                                       // vertex data drawn from kept streams
    uint64_t constantBlocks[3] = {}, constantSharedReused = 0;      // VS/PS/shared constant blocks uploaded, shared reused
    uint64_t constantBytes = 0;                                     // VS/PS float constants written
    uint64_t swapSmallBytes = 0, swapLargeBytes = 0, swapPageBytes = 0;   // stream swaps: small/large whole streams, kept pages
    uint64_t resolveSwaps = 0, resolveRestores = 0;                 // resolves swapped, swapped surfaces copied back
    int64_t drawNs = 0, textureNs = 0, presentNs = 0, vertexNs = 0;
    std::chrono::steady_clock::time_point windowStart = std::chrono::steady_clock::now();
} s_stats;
// Pipelines created and the time it took (not reset with the window above: the hitch log takes deltas).
std::atomic<uint64_t> s_pipelinesCreated{ 0 };
std::atomic<int64_t> s_pipelineNs{ 0 };

bool StatsEnabled()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_RENDER_STATS"); return v && *v && *v != '0'; }();
    return enabled;
}

// The constant dirty masks since each file was last uploaded (NoteDrawDirty), or-ed over draw cores whose draws
// returned before uploading. A draw no draw core noted (capture replays) uploads both.
thread_local uint64_t t_vsDirty = ~0ull, t_psDirty = ~0ull;
thread_local bool t_dirtyNoted = false;
// Set around BindTargets by callers that overwrite the colour target whole.
thread_local bool t_overwritesColor = false;
thread_local bool t_uiDraw = false;      // the queued draw is the 2D layers' (RunDraw)

// The settings overlay held the game since the last frame (RendererHoldForOverlay): that frame's time isn't a hitch.
std::atomic<bool> s_heldForOverlay{ false };

// Frames over 50 ms are logged with what they spent the time on (always on: a player's log shows what a
// stutter was).
void LogHitch()
{
    struct Snapshot
    {
        std::chrono::steady_clock::time_point at;
        int64_t drawNs, textureNs, presentNs;
        uint64_t uploadedBytes, faults;
        ShaderCompileStats shaders;
        uint64_t pipelines;
        int64_t pipelineNs, pipelineWaitNs;
    };
    static Snapshot last{ std::chrono::steady_clock::now() };
    static uint32_t logged = 0;
    const Snapshot now{ std::chrono::steady_clock::now(), s_stats.drawNs, s_stats.textureNs, s_stats.presentNs, s_stats.uploadedBytes,
        g_memory.WatchFaults(), GetShaderCompileStats(), s_pipelinesCreated.load(), s_pipelineNs.load(),
        g_video.deferred ? g_video.deferred->PipelineWaitNs() : 0 };
    const double ms = std::chrono::duration<double, std::milli>(now.at - last.at).count();
    // The stats window resets every 120 frames: a delta across the reset is meaningless, skip it.
    const bool comparable = now.drawNs >= last.drawNs && now.presentNs >= last.presentNs;
    const bool held = s_heldForOverlay.exchange(false);
    if (ms > 50 && comparable && logged < 200 && !held)
    {
        logged++;
        log::Info("Hitch: frame {} took {:.0f} ms: draws {:.1f} ms (textures {:.1f} ms, {:.1f} MB uploaded), present {:.1f} ms, "
                  "{} shader(s) compiled in {:.0f} ms, {} pipeline(s) created in {:.0f} ms (waited for {:.0f} ms), {} write fault(s); "
                  "last present: pacing {:.1f} ms, "
                  "swap {:.1f} ms, GPU {:.1f} ms",
            s_r.frame, ms, (now.drawNs - last.drawNs) / 1e6, (now.textureNs - last.textureNs) / 1e6,
            (now.uploadedBytes - last.uploadedBytes) / 1048576.0, (now.presentNs - last.presentNs) / 1e6,
            now.shaders.count - last.shaders.count, (now.shaders.ns - last.shaders.ns) / 1e6, now.pipelines - last.pipelines,
            (now.pipelineNs - last.pipelineNs) / 1e6, (now.pipelineWaitNs - last.pipelineWaitNs) / 1e6, now.faults - last.faults,
            g_video.lastPaceNs / 1e6, g_video.lastSwapNs / 1e6, g_video.lastGpuWaitNs / 1e6);
    }
    last = now;
}

// One row of a texture upload, untiled and byte-swapped into `out` (the upload buffer). The row is built
// in cached memory and written out in one go: the upload buffer is write-combined, and swapping in place
// would read it back at ~110 MB/s (~1 s for a mission start's 100 MB of textures).
// Tiled rows interleave in 16-byte pieces (bank and row-pair bits only change above them), so the blocks
// of an aligned 16-byte piece of a row (8 bytes for 1-byte blocks: x bit 3 moves the bank) lie together:
// one address per piece instead of per block (at ~30 ns a block a 1024x1024 DXT1 would take 2 ms).
template <uint32_t B>
void UntileRow(uint8_t* row, const uint8_t* src, uint32_t blocksX, uint32_t y, uint32_t pitchBlocks, uint32_t ox, uint32_t oy)
{
    constexpr uint32_t kRun = (B == 1 ? 8 : 16) / B;   // blocks per piece
    uint32_t x = 0;
    if (ox % kRun == 0)
        for (; x + kRun <= blocksX; x += kRun)
            std::memcpy(row + x * B, src + uint64_t(xenos::TiledOffset2D(x + ox, y + oy, pitchBlocks, B)) * B, kRun * B);
    for (; x < blocksX; x++)
        std::memcpy(row + x * B, src + uint64_t(xenos::TiledOffset2D(x + ox, y + oy, pitchBlocks, B)) * B, B);
}

void ConvertRow(uint8_t* out, const uint8_t* src, uint32_t blocksX, uint32_t y, bool tiled, uint32_t pitchBlocks, uint32_t ox,
    uint32_t oy, uint32_t bytesPerBlock, uint32_t endian)
{
    thread_local std::vector<uint8_t> row;
    const size_t bytes = size_t(blocksX) * bytesPerBlock;
    if (row.size() < bytes)
        row.resize(bytes);
    if (!tiled)
        std::memcpy(row.data(), src + uint64_t(y) * pitchBlocks * bytesPerBlock, bytes);
    else if (bytesPerBlock == 4)
        UntileRow<4>(row.data(), src, blocksX, y, pitchBlocks, ox, oy);
    else if (bytesPerBlock == 8)
        UntileRow<8>(row.data(), src, blocksX, y, pitchBlocks, ox, oy);
    else if (bytesPerBlock == 16)
        UntileRow<16>(row.data(), src, blocksX, y, pitchBlocks, ox, oy);
    else if (bytesPerBlock == 2)
        UntileRow<2>(row.data(), src, blocksX, y, pitchBlocks, ox, oy);
    else if (bytesPerBlock == 1)
        UntileRow<1>(row.data(), src, blocksX, y, pitchBlocks, ox, oy);
    else
        for (uint32_t x = 0; x < blocksX; x++)
            std::memcpy(row.data() + x * bytesPerBlock, src + uint64_t(xenos::TiledOffset2D(x + ox, y + oy, pitchBlocks, bytesPerBlock)) * bytesPerBlock,
                bytesPerBlock);
    xenos::SwapEndian(row.data(), bytes, endian);
    std::memcpy(out, row.data(), bytes);
}

// Rows of large texture uploads are converted on up to eight threads (the pool's and the caller): a mission start
// converts ~100 MB at once.
class RowPool
{
public:
    explicit RowPool(uint32_t maxThreads = 8) : maxThreads(maxThreads) {}

    // Calls f(i) for i in [0, count), split over the pool's threads and the caller.
    void Run(uint32_t count, const std::function<void(uint32_t)>& f)
    {
        Start();
        {
            std::lock_guard lock(mutex);
            job = &f;
            total = count;
            next = 0;
            active = uint32_t(threads.size());
            generation++;
        }
        wake.notify_all();
        Work(f, count);
        std::unique_lock lock(mutex);
        finished.wait(lock, [&] { return active == 0; });
        job = nullptr;
    }

private:
    static constexpr uint32_t kChunk = 8;   // rows per claim
    uint32_t maxThreads;
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable wake, finished;
    const std::function<void(uint32_t)>* job = nullptr;
    std::atomic<uint32_t> next{ 0 };
    uint32_t total = 0, active = 0;
    uint64_t generation = 0;

    void Work(const std::function<void(uint32_t)>& f, uint32_t count)
    {
        for (uint32_t at; (at = next.fetch_add(kChunk)) < count;)
            for (uint32_t i = at; i < std::min(count, at + kChunk); i++)
                f(i);
    }

    void Start()
    {
        if (!threads.empty())
            return;
        const uint32_t n = std::clamp(HostProcessors(), 2u, std::max(maxThreads, 2u)) - 1;
        for (uint32_t i = 0; i < n; i++)
            threads.emplace_back([this] {
                uint64_t seen = 0;
                for (;;)
                {
                    const std::function<void(uint32_t)>* f;
                    uint32_t count;
                    {
                        std::unique_lock lock(mutex);
                        wake.wait(lock, [&] { return generation != seen; });
                        seen = generation;
                        f = job;
                        count = total;
                    }
                    Work(*f, count);
                    std::lock_guard lock(mutex);
                    if (--active == 0)
                        finished.notify_one();
                }
            });
    }
} s_rowPool;
// The frame's vertex data, swapped at submission (on the presenting thread, while the render thread may be
// converting textures through s_rowPool).
RowPool s_copyPool(5);

void RunVertexCopy(const Renderer::VertexCopy& c);

// The vertex copies made when draws are translated can run on threads of their own:
// they are a fifth of the renderer worker's time. They are handed over in batches of ~128 KB (a wake-up per
// copy cost more than the copies). Submission waits for them before it checks which sources were written
// since. Off unless AC6_ASYNC_VERTEX_COPIES=1: no gain on an 8-thread machine (the threads compete with the
// game's), and on a 16-thread one UI elements flicker for single frames in the main menu and the briefing
// (cause unknown).
class AsyncCopier
{
public:
    static bool Enabled()
    {
        static const bool enabled = [] {
            const char* v = std::getenv("AC6_ASYNC_VERTEX_COPIES");
            return v && *v == '1';
        }();
        return enabled;
    }

    void Push(const Renderer::VertexCopy& copy)
    {
        batch.push_back(copy);
        batchBytes += uint64_t(copy.vertexCount) * copy.stride;
        if (batchBytes >= (128u << 10))
            Flush();
    }

    void Wait()
    {
        Flush();
        std::unique_lock lock(mutex);
        idle.wait(lock, [this] { return pending == 0; });
    }

private:
    void Flush()
    {
        if (batch.empty())
            return;
        std::call_once(started, [this] {
            const uint32_t threads = std::clamp(HostProcessors() / 6, 1u, 3u);
            for (uint32_t i = 0; i < threads; i++)
                std::thread([this] { Run(); }).detach();
            log::Info("Vertex copies: {} thread(s) of their own", threads);
        });
        {
            std::lock_guard lock(mutex);
            jobs.push_back(std::move(batch));
            pending++;
        }
        batch = {};
        batchBytes = 0;
        work.notify_one();
    }

    void Run()
    {
        std::unique_lock lock(mutex);
        for (;;)
        {
            work.wait(lock, [this] { return !jobs.empty(); });
            const std::vector<Renderer::VertexCopy> copies = std::move(jobs.front());
            jobs.pop_front();
            lock.unlock();
            for (const Renderer::VertexCopy& copy : copies)
                RunVertexCopy(copy);
            lock.lock();
            if (--pending == 0)
                idle.notify_all();
        }
    }

    std::vector<Renderer::VertexCopy> batch;   // the renderer worker's
    uint64_t batchBytes = 0;
    std::once_flag started;
    std::mutex mutex;
    std::condition_variable work, idle;
    std::deque<std::vector<Renderer::VertexCopy>> jobs;
    uint32_t pending = 0;
};
AsyncCopier& s_asyncCopier = *new AsyncCopier;   // never destroyed: its threads outlive exit

// Uploads from 256 KB on are split over the pool (a mission start's 1024x1024 DXT1s are 0.5 MB each).
void ConvertRows(uint32_t rows, uint64_t bytes, const std::function<void(uint32_t)>& f)
{
    if (bytes < (256u << 10))
        for (uint32_t y = 0; y < rows; y++)
            f(y);
    else
        s_rowPool.Run(rows, f);
}

// Stats timers count TSC ticks: two clock reads a draw would cost ~2% of the renderer worker.
double NsPerTick()
{
    static const double ns = [] {
        if (const uint64_t frequency = kernel::TimebaseFrequency())
            return 1e9 / double(frequency);
        // Replays don't start the guest clock, so the TSC is measured here (else the hitch log's times are 1e9 too large).
        const auto clock0 = std::chrono::steady_clock::now();
        const uint64_t tsc0 = __rdtsc();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - clock0).count();
        return seconds * 1e9 / double(std::max<uint64_t>(1, __rdtsc() - tsc0));
    }();
    return ns;
}

struct ScopedNs
{
    int64_t& total;
    uint64_t start = __rdtsc();
    explicit ScopedNs(int64_t& t) : total(t) {}
    ~ScopedNs() { total += int64_t(double(__rdtsc() - start) * NsPerTick()); }
};

uint32_t U32(uint32_t guest) { return *g_memory.As<be<uint32_t>>(guest); }

uint64_t HashBytes(const void* data, size_t size, uint64_t h = 0xCBF29CE484222325ull)
{
    auto* p = static_cast<const uint8_t*>(data);
    size_t i = 0;
    for (; i + 8 <= size; i += 8)
    {
        uint64_t v;
        std::memcpy(&v, p + i, 8);
        h = (h ^ v) * 0x100000001B3ull;
        h ^= h >> 29;
    }
    for (; i < size; i++)
        h = (h ^ p[i]) * 0x100000001B3ull;
    return h;
}

// Content hash for large guest buffers (textures): eight independent lanes, so the multiplies overlap
// and it runs near memory bandwidth (one serial chain would cost ~18 ms per frame for ~100 MB in missions).
uint64_t HashLarge(const void* data, size_t size)
{
    auto* p = static_cast<const uint8_t*>(data);
    uint64_t lanes[8];
    for (uint32_t l = 0; l < 8; l++)
        lanes[l] = 0xCBF29CE484222325ull + l * 0x9E3779B97F4A7C15ull;
    size_t i = 0;
    for (; i + 64 <= size; i += 64)
        for (uint32_t l = 0; l < 8; l++)
        {
            uint64_t v;
            std::memcpy(&v, p + i + l * 8, 8);
            lanes[l] = (lanes[l] ^ v) * 0x100000001B3ull;
            lanes[l] ^= lanes[l] >> 29;
        }
    uint64_t h = HashBytes(p + i, size - i, size);
    for (uint32_t l = 0; l < 8; l++)
        h = (h ^ lanes[l]) * 0x100000001B3ull;
    return h;
}

void TransitionTarget(RenderCommandList& cl, RenderTextureLayout layout)
{
    if (s_r.targetLayout != layout)
    {
        // Copies read it too (screenshots): the stages the next use waits for include them.
        cl.barriers(RenderBarrierStage::GRAPHICS | RenderBarrierStage::COPY, RenderTextureBarrier(s_r.target.get(), layout));
        s_r.targetLayout = layout;
    }
}

// --- Formats -------------------------------------------------------------------------------------------

RenderFormat TextureFormat(uint32_t format)
{
    switch (format)
    {
    case xenos::kFormat8: return RenderFormat::R8_UNORM;
    case xenos::kFormat8888: return RenderFormat::B8G8R8A8_UNORM;
    case xenos::kFormatDXT1: return RenderFormat::BC1_UNORM;
    case xenos::kFormatDXT2_3: return RenderFormat::BC2_UNORM;
    case xenos::kFormatDXT4_5: return RenderFormat::BC3_UNORM;
    case 49: return RenderFormat::BC5_UNORM;   // DXN
    default: return RenderFormat::UNKNOWN;
    }
}

RenderSwizzle Swizzle(uint32_t code)
{
    switch (code)
    {
    case 0: return RenderSwizzle::R;
    case 1: return RenderSwizzle::G;
    case 2: return RenderSwizzle::B;
    case 3: return RenderSwizzle::A;
    case 4: return RenderSwizzle::ZERO;
    default: return RenderSwizzle::ONE;
    }
}

// A texture fetch constant's swizzle (dword 3) as a view's component mapping. The swizzle names guest
// components. In B8G8R8A8 images (8_8_8_8 textures, uploaded or resolved) guest X is in blue and Z in red,
// so X and Z swap places for them.
RenderComponentMapping FetchComponentMapping(uint32_t swizzle, RenderFormat hostFormat)
{
    const bool bgra = hostFormat == RenderFormat::B8G8R8A8_UNORM;
    auto component = [bgra](uint32_t code) { return Swizzle(bgra && code < 3 ? 2 - code : code); };
    return RenderComponentMapping(component((swizzle >> 1) & 7), component((swizzle >> 4) & 7), component((swizzle >> 7) & 7),
        component((swizzle >> 10) & 7));
}

// The images a resolve target's textures read: its float twin when the HDR canvas resolved into it
// last, else the resolve targets.
FastMap<uint32_t, HostImage>& ResolvedImages(uint32_t address)
{
    if (!s_r.hdrResolved.empty())
        if (auto it = s_r.hdrResolvedCurrent.find(address); it != s_r.hdrResolvedCurrent.end() && it->second)
            return s_r.hdrResolved;
    return s_r.resolved;
}

// Host vertex format for a vfetch (GPU format, signed, integer). Streams are copied with each 32-bit
// word swapped, which is the GPU's 8in32 endian swap. Component X is then in the low bits, and the
// instruction's swizzle (set up by D3D for D3DCOLOR and friends) orders the components.
RenderFormat VertexFormat(uint32_t format, bool isSigned, bool isInteger)
{
    auto pick = [&](RenderFormat unorm, RenderFormat snorm, RenderFormat uint, RenderFormat sint) {
        return isInteger ? (isSigned ? sint : uint) : (isSigned ? snorm : unorm);
    };
    switch (format)
    {
    case 6: return pick(RenderFormat::R8G8B8A8_UNORM, RenderFormat::R8G8B8A8_SNORM, RenderFormat::R8G8B8A8_UINT, RenderFormat::R8G8B8A8_SINT);
    case 7: case 16: case 17: return RenderFormat::R32_UINT;   // packed, unpacked in the shader
    case 25: return pick(RenderFormat::R16G16_UNORM, RenderFormat::R16G16_SNORM, RenderFormat::R16G16_UINT, RenderFormat::R16G16_SINT);
    case 26: return pick(RenderFormat::R16G16B16A16_UNORM, RenderFormat::R16G16B16A16_SNORM, RenderFormat::R16G16B16A16_UINT,
        RenderFormat::R16G16B16A16_SINT);
    case 31: return RenderFormat::R16G16_FLOAT;
    case 32: return RenderFormat::R16G16B16A16_FLOAT;
    case 33: return isSigned ? RenderFormat::R32_SINT : RenderFormat::R32_UINT;
    case 34: return isSigned ? RenderFormat::R32G32_SINT : RenderFormat::R32G32_UINT;
    case 35: return isSigned ? RenderFormat::R32G32B32A32_SINT : RenderFormat::R32G32B32A32_UINT;
    case 36: return RenderFormat::R32_FLOAT;
    case 37: return RenderFormat::R32G32_FLOAT;
    case 38: return RenderFormat::R32G32B32A32_FLOAT;
    case 57: return RenderFormat::R32G32B32_FLOAT;
    default: return RenderFormat::UNKNOWN;
    }
}

bool IsFloatFormat(RenderFormat f)
{
    return f == RenderFormat::R32_FLOAT || f == RenderFormat::R32G32_FLOAT || f == RenderFormat::R32G32B32_FLOAT ||
        f == RenderFormat::R32G32B32A32_FLOAT;
}

uint32_t FormatBytes(RenderFormat f)
{
    switch (f)
    {
    case RenderFormat::R32_FLOAT: case RenderFormat::B8G8R8A8_UNORM: case RenderFormat::R8G8B8A8_UNORM:
    case RenderFormat::R32_UINT: case RenderFormat::R16G16_FLOAT: return 4;
    case RenderFormat::R32G32_FLOAT: case RenderFormat::R16G16B16A16_FLOAT: return 8;
    case RenderFormat::R32G32B32_FLOAT: return 12;
    case RenderFormat::R32G32B32A32_FLOAT: return 16;
    default: return 0;
    }
}

RenderBlend Blend(uint32_t v)
{
    switch (v)
    {
    case 0: return RenderBlend::ZERO;
    case 1: return RenderBlend::ONE;
    case 4: return RenderBlend::SRC_COLOR;
    case 5: return RenderBlend::INV_SRC_COLOR;
    case 6: return RenderBlend::SRC_ALPHA;
    case 7: return RenderBlend::INV_SRC_ALPHA;
    case 8: return RenderBlend::DEST_COLOR;
    case 9: return RenderBlend::INV_DEST_COLOR;
    case 10: return RenderBlend::DEST_ALPHA;
    case 11: return RenderBlend::INV_DEST_ALPHA;
    case 12: return RenderBlend::BLEND_FACTOR;
    case 13: return RenderBlend::INV_BLEND_FACTOR;
    case 14: return RenderBlend::BLEND_FACTOR_ALPHA;
    case 15: return RenderBlend::INV_BLEND_FACTOR_ALPHA;
    case 16: return RenderBlend::SRC_ALPHA_SAT;
    default: return RenderBlend::ONE;
    }
}

// Whether RB_BLENDCONTROL uses the blend constant (factors 12-15: constant colour / alpha and inverses).
bool UsesBlendConstant(uint32_t control)
{
    if (control == kBlendDisabled)
        return false;
    for (const uint32_t shift : { 0u, 8u, 16u, 24u })
        if (const uint32_t f = (control >> shift) & 0x1F; f >= 12 && f <= 15)
            return true;
    return false;
}

RenderBlendOperation BlendOp(uint32_t v)
{
    switch (v)
    {
    case 1: return RenderBlendOperation::SUBTRACT;
    case 2: return RenderBlendOperation::MIN;
    case 3: return RenderBlendOperation::MAX;
    case 4: return RenderBlendOperation::REV_SUBTRACT;
    default: return RenderBlendOperation::ADD;
    }
}

// --- Copy (present) shader --------------------------------------------------------------------------------

// The renderer's own shaders (internal_shaders.hlsl), compiled at build time.
const InternalShaderBlob* FindInternalShader(std::string_view name)
{
    for (unsigned i = 0; i < kInternalShaderCount; i++)
        if (name == kInternalShaders[i].name)
            return &kInternalShaders[i];
    log::Error("Internal shader {} missing", name);
    return nullptr;
}

// The output pass's settings: AC6_DEBAND=<threshold in 8-bit steps> (1 = 1.5 steps), radius
// AC6_DEBAND_RADIUS=<pixels at 1080p> (16), dithering with any of them or a deep / HDR output (AC6_DITHER=0: none, 1: also
// on a plain 8-bit output), AC6_HDR_PAPER_WHITE=<nits> (203) for the picture's white on an HDR output.
struct OutputSettings
{
    bool active = false;
    uint32_t flags = 0;
    float paperScale = 203.0f / 80.0f;
    uint32_t grade0 = 0, grade1 = 0;   // the HDR grading (internal_shaders.hlsl PushConstants)
    uint32_t stripes = 0;   // the brightness stripes: 0 none, 1 the display's brightness, 2 the game's values
};

// AC6_HDR_SIM=1 (tests only): everything as with an HDR output (the output pass's HDR flags, HDR from
// the scene, the frame before the UI) while the window stays SDR. The HDR output pass is rendered on screenshot frames into
// a float image and dumped (raw_hdr_out_<frame>.bin), so its nits can be measured without an HDR display.
bool HdrSim()
{
    static const bool on = [] { const char* v = std::getenv("AC6_HDR_SIM"); return v && *v == '1'; }();
    return on;
}

// HDR from the game's scene: AC6_HDR_SCENE=1 (default) with an HDR output, 0 the SDR picture's
// inverse, force: the float canvas even with an SDR output (tests: the picture should stay as it is).
bool HdrSceneOn()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<int> setting{ 1 };   // 0 off, 1 with HDR, 2 always
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        const std::string v = settings::Get("AC6_HDR_SCENE");
        setting.store(v == "force" ? 2 : v == "0" || v == "false" ? 0 : 1, std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    const int s = setting.load(std::memory_order_relaxed);
    return s == 2 || (s == 1 && (g_video.OutputMode() == Video::Output::Hdr || HdrSim()));
}

// The game's lighting in float: with HDR from the scene, AC6_HDR_SCENE_FLOAT (default on, 0 = the 8-bit scene):
// the game's colour surfaces at the frame's size and its 1/2, 1/4 and 1/8 (the scene,
// the canvas, the effects surface, the downsample, the bloom) float for the whole frame, the shaders writing the game's
// code range unclamped and its decoders reading values (shaders.cpp kOptSceneFloat).
bool SceneFloatOn()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<bool> setting{ true };
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        const std::string v = settings::Get("AC6_HDR_SCENE_FLOAT");
        setting.store(!(v == "0" || v == "false"), std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    return setting.load(std::memory_order_relaxed) && HdrSceneOn();
}

// An image of the scene's size: the frame's, or with FSR upscaling the world's surfaces' (the frame's
// size at the scale they were made at).
bool SceneSized(const HostImage& image)
{
    if (image.width == s_r.sceneWidth && image.height == s_r.sceneHeight)
        return true;
    if (!s_r.upscaling || image.resX <= 0.0f || image.resY <= 0.0f || s_r.sceneResX <= 0.0f || s_r.sceneResY <= 0.0f ||
        (image.resX == s_r.sceneResX && image.resY == s_r.sceneResY))
        return false;
    return std::abs(int32_t(image.width) - int32_t(std::lround(double(s_r.sceneWidth) * image.resX / s_r.sceneResX))) <= 1 &&
        std::abs(int32_t(image.height) - int32_t(std::lround(double(s_r.sceneHeight) * image.resY / s_r.sceneResY))) <= 1;
}

// HDR from the scene's effects and lit highlights, live: AC6_HDR_EFFECTS 0-4 (default 1, 0 = as the game draws them,
// 8-bit, at most white), how much of every additive effect is added again in float, AC6_HDR_LIT_HIGHLIGHTS 0-100
// (default 100), how much of the lit surfaces' range above white is kept (the game's lit codes reach 1.5, which the
// grade lifts further, so sunlit roofs can outshine explosions).
float EffectsGlow()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<float> value{ 1.0f };
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        const std::string v = settings::Get("AC6_HDR_EFFECTS");
        value.store(v.empty() ? 1.0f : std::clamp(float(std::atof(v.c_str())), 0.0f, 4.0f), std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    return value.load(std::memory_order_relaxed);
}
// The clouds' brightness with HDR from the scene, live: AC6_HDR_CLOUDS 1-6 (default 1: as the game's SDR picture has
// them), the light factor for the clouds' sunlit parts (AC7 + RenoDX at the same paper white puts sunlit clouds at
// 180-320 nits, AC6's come to 40-90).
float CloudLift()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<float> value{ 1.0f };
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        const std::string v = settings::Get("AC6_HDR_CLOUDS");
        value.store(v.empty() ? 1.0f : std::clamp(float(std::atof(v.c_str())), 1.0f, 6.0f), std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    return value.load(std::memory_order_relaxed);
}
// The relighting prototype, with HDR from the scene: the game's light inputs scaled per kind before
// each draw (HostShader::lightConstants), as Unreal's light units would set them, per kind: AC6_RELIGHT_OBJECT_SUN,
// _OBJECT_AMBIENT, _CLOUD_SUN, _CLOUD_AMBIENT, _PARTICLE_SUN, _PARTICLE_AMBIENT, _FOG, _MAP_SPECULAR (default 1 each =
// off). Live (settings: the overlay's Cloud light and Aircraft light are the clouds' and the objects' sun),
// with defaults: the clouds' sun kRelightCloudSun, the objects' sun kRelightObjectSun (1 for the rest).
constexpr float kRelightCloudSun = 1.0f, kRelightObjectSun = 1.5f;
struct RelightFactors { float k[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }; float cloudRim = 1.5f; bool on = false; };
const RelightFactors& Relight()
{
    thread_local uint64_t generation = UINT64_MAX;
    thread_local RelightFactors f;
    if (const uint64_t now = settings::Generation(); generation != now)
    {
        RelightFactors r;
        const char* names[8] = { "AC6_RELIGHT_OBJECT_SUN", "AC6_RELIGHT_OBJECT_AMBIENT", "AC6_RELIGHT_CLOUD_SUN", "AC6_RELIGHT_CLOUD_AMBIENT",
            "AC6_RELIGHT_PARTICLE_SUN", "AC6_RELIGHT_PARTICLE_AMBIENT", "AC6_RELIGHT_FOG", "AC6_RELIGHT_MAP_SPECULAR" };
        r.k[0] = kRelightObjectSun;
        r.k[2] = kRelightCloudSun;
        for (int i = 0; i < 8; i++)
            if (const std::string v = settings::Get(names[i]); !v.empty())
                r.k[i] = std::clamp(float(std::atof(v.c_str())), 0.0f, 32.0f);
        // (8: the objects' light cap, with their sun, 9: their specular, AC6_RELIGHT_OBJECT_SPECULAR, else their sun's.)
        r.k[8] = r.k[0];
        const std::string spec = settings::Get("AC6_RELIGHT_OBJECT_SPECULAR");
        r.k[9] = !spec.empty() ? std::clamp(float(std::atof(spec.c_str())), 0.0f, 32.0f) : r.k[0];
        // The clouds' rim light (AC6_RELIGHT_CLOUD_RIM, the settings file's hdr_cloud_edge, 1 = the game's, default
        // 1.5, chosen for Cloud light at 1).
        if (const std::string rim = settings::Get("AC6_RELIGHT_CLOUD_RIM"); !rim.empty())
            r.cloudRim = std::clamp(float(std::atof(rim.c_str())), 0.0f, 32.0f);
        for (int i = 0; i < 10; i++)
            r.on |= r.k[i] != 1.0f;
        r.on |= r.cloudRim != 1.0f;
        f = r;
        generation = now;
    }
    return f;
}

// The auto exposure (experimental): the scene composite's vertex shader (6B6FBBD8930EFA4D) turns the
// adapted luminance L (the exposure ladder's 1x1) into exposure = L < lo ? lo / L : L > hi ? hi / L : 1, lo and hi the xy of its
// CONST_VS_24_27 (register c106, 0.25 and 0.5 in the scenes seen): a band, exposure 1 inside it, the scenes above it
// metered already (day flight sits at 0.47-0.54). AC6_HDR_EXPOSURE_BAND 0-1 (read at start, HDR from the scene only, 0 = the
// game's band) closes the band from below, lo towards hi, 1 meters every scene to hi (exposure = hi / L), as Unreal's eye
// adaptation does with a fixed key (the game's own adaptation speed): the dim scenes brought up, the day's unchanged.
float ExposureBand()
{
    static const float k = [] {
        const char* v = std::getenv("AC6_HDR_EXPOSURE_BAND");
        return v && *v ? std::clamp(float(std::atof(v)), 0.0f, 1.0f) : 0.0f;
    }();
    return k;
}

// The scene's ceiling, live: AC6_HDR_SCENE_CEILING (the canvas's 2.2-encoded units before the grade,
// default 3, 0 = none). The sun's disc decodes at 100x white beside its glare's 2-3: a hard-edged ball at the
// display's peak, where SDR's clip merges disc and glare. Applied in the compose pass, after the game's bloom (which
// spreads the disc's light into the glare) and before the effects' float glow (explosions, afterburners): what lies
// above a knee at 0.6 x the ceiling is spread over a small Gaussian and rolled off towards the ceiling, the colour
// kept (internal_shaders.hlsl HDR_COMPOSE).
float SceneCeiling()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<float> value{ 3.0f };
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        const std::string v = settings::Get("AC6_HDR_SCENE_CEILING");
        value.store(v.empty() ? 3.0f : std::clamp(float(std::atof(v.c_str())), 0.0f, 200.0f), std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    return value.load(std::memory_order_relaxed);
}

// The sun's own ceiling, live: AC6_HDR_SUN_CEILING (the same units, default kSunCeilingDefault): the
// top of the sun's soft pass over its box. With the frame's ceiling for the sun too, the sun and the aircraft's sunlit
// panels come out alike (on a 1020-nit display: the wing 810 nits, the sun 860). Above it the sun's core
// reaches nearer the display's peak, the knee (0.6 x the frame's ceiling) the same as outside its box.
constexpr float kSunCeilingDefault = 6.0f;
float SunCeiling()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<float> value{ kSunCeilingDefault };
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        const std::string v = settings::Get("AC6_HDR_SUN_CEILING");
        value.store(v.empty() ? kSunCeilingDefault : std::clamp(float(std::atof(v.c_str())), 0.0f, 200.0f), std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    return value.load(std::memory_order_relaxed);
}

// The afterburners' glow, live: AC6_HDR_AFTERBURNER 0-6 (default kAfterburnerDefault), the weight of
// their draws (VS 7186B5F32D48D10B, additive into the scene: the nozzles' rings and flames) in the effects' float layer,
// on top of Effects brightness: their rings peak at 260-310 nits on a 460-nit display, AC7's at about 440:
// 3 brings them to 370-400 at Effects brightness 1.6 (the effects' own ceiling and the peak level them off).
constexpr float kAfterburnerDefault = 3.0f;
float AfterburnerGlow()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<float> value{ kAfterburnerDefault };
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        const std::string v = settings::Get("AC6_HDR_AFTERBURNER");
        value.store(v.empty() ? kAfterburnerDefault : std::clamp(float(std::atof(v.c_str())), 0.0f, 6.0f), std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    return value.load(std::memory_order_relaxed);
}

float LitHighlights()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<float> value{ 1.0f };
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        // (Default 100: the float scene's clouds and glints above white kept. The scene's ceiling rolls off the
        // top. 50 halves them.)
        const std::string v = settings::Get("AC6_HDR_LIT_HIGHLIGHTS");
        value.store(v.empty() ? 1.0f : std::clamp(float(std::atof(v.c_str())), 0.0f, 100.0f) / 100.0f, std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    return value.load(std::memory_order_relaxed);
}

// The HDR grading (RenoDRT's and PsychoV-30's sliders, as RenoDX's mods offer them):
// AC6_HDR_EXPOSURE 0.5-2 (1), AC6_HDR_HIGHLIGHTS / _SHADOWS / _CONTRAST / _SATURATION 0-100 (50 = as is),
// AC6_HDR_GAMUT_COMPRESSION 0-100 (100) and AC6_HDR_COMPRESSION 0-2 (1, 0 = automatic): PsychoV-30's,
// AC6_HDR_HUE_CORRECTION 0-100 (50: the afterburners keep their colour): both. Live.
void HdrGrade(uint32_t& grade0, uint32_t& grade1)
{
    const auto value = [](const char* env, double def, double lo, double hi) {
        const std::string v = settings::Get(env);
        return std::clamp(!v.empty() ? std::atof(v.c_str()) : def, lo, hi);
    };
    const auto byte = [](double v) { return uint32_t(std::lround(v)) & 255u; };
    grade0 = byte((value("AC6_HDR_EXPOSURE", 1.0, 0.5, 2.0) - 0.5) * 100.0) | byte(value("AC6_HDR_HIGHLIGHTS", 50.0, 0.0, 100.0)) << 8 |
        byte(value("AC6_HDR_SHADOWS", 50.0, 0.0, 100.0)) << 16 | byte(value("AC6_HDR_CONTRAST", 50.0, 0.0, 100.0)) << 24;
    grade1 = byte(value("AC6_HDR_SATURATION", 50.0, 0.0, 100.0)) | byte(value("AC6_HDR_GAMUT_COMPRESSION", 100.0, 0.0, 100.0)) << 8 |
        byte(value("AC6_HDR_COMPRESSION", 1.0, 0.0, 2.0) * 100.0) << 16 | byte(value("AC6_HDR_HUE_CORRECTION", 50.0, 0.0, 100.0)) << 25;
}

OutputSettings Output(uint32_t targetHeight)
{
    OutputSettings o;
    const std::string debandSetting = settings::Get("AC6_DEBAND");   // (live)
    const char* deband = debandSetting.c_str();
    const char* radius = std::getenv("AC6_DEBAND_RADIUS");
    const char* dither = std::getenv("AC6_DITHER");
    const std::string paper = settings::Get("AC6_HDR_PAPER_WHITE");   // (live)
    double threshold = deband && *deband ? std::atof(deband) : 0.0;
    if (threshold == 1.0)
        threshold = 1.5;
    const auto mode = g_video.OutputMode();
    if (threshold > 0.0)
    {
        const double r = (radius && *radius ? std::atof(radius) : 16.0) * targetHeight / 1080.0;
        o.flags |= 1u | (uint32_t(std::clamp(long(std::lround(r)), 1l, 255l)) << 8) |
            (uint32_t(std::clamp(long(std::lround(threshold * 16.0)), 1l, 255l)) << 16);
    }
    const bool ditherOn = !(dither && *dither == '0') && (threshold > 0.0 || mode != Video::Output::Sdr8 || (dither && *dither == '1'));
    if (ditherOn)
        o.flags |= 2u | (mode == Video::Output::SdrDeep ? 8u : 0u);
    if (mode == Video::Output::Hdr || HdrSim())
    {
        // AC6_HDR_PEAK=<nits> (default: the display's, else 1000, 0: none): the brightest highlights' target.
        const std::string peak = settings::Get("AC6_HDR_PEAK");   // (live)
        const double peakNits = !peak.empty() ? std::atof(peak.c_str()) : g_video.displayPeakNits > 0.0f ? g_video.displayPeakNits : 1000.0;
        o.flags |= 4u | (uint32_t(std::clamp(long(std::lround(peakNits / 50.0)), 0l, 255l)) << 24);
        // AC6_HDR_TONEMAP=psychov30 (the default) | renodrt. Any other value gets PsychoV-30.
        const std::string tonemap = settings::Get("AC6_HDR_TONEMAP");   // (live)
        o.flags |= (tonemap == "renodrt" ? 2u : 3u) << 4;
        o.paperScale = float((!paper.empty() ? std::atof(paper.c_str()) : 203.0) / 80.0);
        HdrGrade(o.grade0, o.grade1);
        // AC6_HDR_STRIPES=output | scene: stripes over the picture showing its brightness (live).
        const std::string stripes = settings::Get("AC6_HDR_STRIPES");
        o.stripes = stripes == "output" || stripes == "1" ? 1u : stripes == "scene" ? 2u : 0u;
        // AC6_HDR_EMITTER_CAP=0: the stray emitter codes as decoded (the afterburners' sparkles).
        if (settings::Get("AC6_HDR_EMITTER_CAP") == "0")
            o.stripes |= 256u;
        // AC6_HDR_UI_BRIGHTNESS=<nits> (live, default 250): the HUD's and menus' white, apart from the picture's paper
        // white. Bits 16-31: the nits.
        const std::string uiNits = settings::Get("AC6_HDR_UI_BRIGHTNESS");
        o.stripes |= uint32_t(std::clamp(uiNits.empty() ? 250.0 : std::atof(uiNits.c_str()), 80.0, 1000.0)) << 16;
    }
    o.active = o.flags != 0 || mode != Video::Output::Sdr8;
    return o;
}

const OutputSettings& OutputConfig()
{
    // Built again after a live change (the settings overlay: HDR, its tone mapping, paper white and peak) or an output
    // switch.
    static OutputSettings o;
    static uint64_t generation = UINT64_MAX;
    static Video::Output mode = Video::Output::Sdr8;
    static float peak = -1.0f;
    if (const uint64_t now = settings::Generation(); now != generation || g_video.OutputMode() != mode || g_video.displayPeakNits != peak)
    {
        generation = now;
        mode = g_video.OutputMode();
        peak = g_video.displayPeakNits;
        o = Output(g_video.renderHeight);
    }
    return o;
}

// The frame before its UI: kept for an HDR output (or AC6_HDR_UI_DEBUG=1, its screenshot view).
bool UiSplit()
{
    static const bool debug = [] { const char* v = std::getenv("AC6_HDR_UI_DEBUG"); return v && *v == '1'; }();
    return g_video.OutputMode() == Video::Output::Hdr || debug || HdrSim();
}

// The output pass's flag for a frame that is all UI (bit 7).
uint32_t AllUiFlag()
{
    return s_r.allUiFrame == s_r.frame ? 128u : 0u;
}

// The descriptor of the frame before its UI (preUiOut) for the output pass, when this frame's was taken (0: none,
// the whole frame is the scene).
uint32_t PreUiDescriptor()
{
    return s_r.preUiOutFrame == s_r.frame && s_r.preUiOut.texture && s_r.preUiOut.descriptor < 65536 ? s_r.preUiOut.descriptor : 0;
}

// AC6_AA=smaa | taa | smaa+taa: SMAA and / or temporal AA on the scene image. Off by
// default (the picture stays the console's). Live: their pipelines are built at start unless FSR
// was asked for, and the passes run while the setting asks for them. FSR on or off still takes a restart (its context
// and the world surfaces' sizes), and so do TAA's motion vectors (the world pass's targets: TaaAtStart).
uint32_t LiveAa()   // bit 0 SMAA, bit 1 TAA, bit 2 FSR
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<uint32_t> value{ 0 };
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        const std::string v = settings::Get("AC6_AA");
        value.store((v.find("smaa") != std::string::npos ? 1u : 0u) | (v.find("taa") != std::string::npos ? 2u : 0u) |
                (v.find("fsr") != std::string::npos ? 4u : 0u), std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    return value.load(std::memory_order_relaxed);
}

bool TaaAtStart()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_AA"); return v && std::string_view(v).find("taa") != std::string_view::npos; }();
    return enabled;
}

bool SmaaOn()
{
    return s_r.smaaBlendPipeline && !s_r.fsr && (LiveAa() & 1);
}

// FSR asked for and running (not failed at its start: the scene is then scaled up as it is, with nothing to undo a jitter).
bool FsrRunning()
{
    return s_r.fsr && !fsr::Failed();
}

// The scene draws jittered for FSR: only while it runs (it ran last frame or this one). A screen where it can't (the
// hangar has no scene depth for it) would otherwise show the raw jitter, the picture shaking by a pixel.
bool FsrJitters()
{
    return FsrRunning() && s_r.fsrRanFrame != ~0ull && s_r.frame - s_r.fsrRanFrame <= 1;
}

// FSR's reactive mask: the scene's translucent draws (smoke, fire, trails, glows, the game's sprites) write how much they
// cover each pixel into the motion image's alpha (which then has four channels), and FSR takes more of those pixels from
// this frame. Without it they keep the motion of what is behind them and FSR smears them along it. AC6_FSR_REACTIVE=0: off.
bool FsrReactive()
{
    static const bool on = [] { const char* v = std::getenv("AC6_FSR_REACTIVE"); return !v || *v != '0'; }();
    return on && FsrRunning();
}

bool TaaOn()
{
    return s_r.taaPipeline && !s_r.fsr && (LiveAa() & 2);
}

// Shadow maps at the render scale (AC6_SHADOW_DETAIL, live). The game draws its character shadow maps (cutscenes:
// 1024 x 1024 depth, resolved and read as k_24_8) and filters them with a 4 x 4 grid of point samples weighted by
// getWeights2D, in its own texels. Its surfaces are drawn at the render scale anyway, so the shadow map already holds
// that many texels: with this on, the filter's samples and weights take the image's own texels, a finer grid (2x at
// 1440p, 3x at 2160p), and the steps along the shadows' edges shrink with it. Off: the game's texels, as on the console.
bool ShadowDetail()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<bool> on{ false };
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        const std::string text = settings::Get("AC6_SHADOW_DETAIL");
        on.store(text == "1" || text == "true" || text == "on", std::memory_order_release);
        generation.store(now, std::memory_order_release);
    }
    return on.load(std::memory_order_acquire);
}

// Ambient occlusion on the scene (settings graphics.ambient_occlusion, AC6_AO): off by default (not
// the console's picture). Intel's XeGTAO: 1 High ("1", "on", "high", "gtao"), 2 Low ("low": its visibility at half the
// scene's resolution). Live: read again when a setting changes.
int AoMode()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<int> mode{ 0 };
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        const std::string text = settings::Get("AC6_AO");
        mode.store(text == "low" ? 2 :
            text == "1" || text == "on" || text == "true" || text == "high" || text == "gtao" || text == "xegtao" ? 1 : 0,
            std::memory_order_release);
        generation.store(now, std::memory_order_release);
    }
    return mode.load(std::memory_order_acquire);
}

// AC6_AO_TERRAIN: what the terrain (the map's own surface: ground, hills, the sea) and
// the trees on it take of the ambient occlusion. 0: none, 1 (the default): what objects cast on them, not
// what they cast on themselves or each other, 2: all of it.
int AoTerrainMode()
{
    static const int mode = [] { const char* v = std::getenv("AC6_AO_TERRAIN"); return v && *v ? std::clamp(std::atoi(v), 0, 2) : 1; }();
    return mode;
}

// Optional lens effects (settings graphics.film_grain / vignette / chromatic_aberration,
// AC6_FILM_GRAIN / AC6_VIGNETTE / AC6_CHROMATIC_ABERRATION, each 0 to 1, 0 by default). Live.
struct LensFx
{
    float grain = 0.0f, vignette = 0.0f, aberration = 0.0f;
    bool Any() const { return grain > 0.0f || vignette > 0.0f || aberration > 0.0f; }
};

LensFx LensFxSettings()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static LensFx current;
    static std::mutex mutex;
    std::lock_guard lock(mutex);
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        auto read = [](const char* name) {
            const std::string v = settings::Get(name);
            return v.empty() ? 0.0f : std::clamp(float(std::atof(v.c_str())), 0.0f, 1.0f);
        };
        current = { read("AC6_FILM_GRAIN"), read("AC6_VIGNETTE"), read("AC6_CHROMATIC_ABERRATION") };
        generation.store(now, std::memory_order_release);
    }
    return current;
}

// The enhanced bloom (settings graphics.bloom, AC6_BLOOM: "console", the default, or "enhanced"). Live.
bool BloomEnhanced()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<bool> enhanced{ false };
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        enhanced.store(settings::Get("AC6_BLOOM") == "enhanced", std::memory_order_release);
        generation.store(now, std::memory_order_release);
    }
    return enhanced.load(std::memory_order_acquire);
}
// The game's scene composite (psHDRFinal) and bright pass (psHDRBrightPassFilter), by their hashes without the option
// bits (the top five).
constexpr uint64_t kOptionFree = (1ull << 59) - 1;
constexpr uint64_t kBloomCompositePs = 0xD02C9AE3A213CABBull & kOptionFree, kBloomBrightPassPs = 0x15E29341C450CAE5ull & kOptionFree;

// AC6_EMITTER_LIGHT=<strength> (an experiment, 0 = off, the default): explosions, fires, flares and
// afterburners light what is near them.
float EmitterLight()
{
    static const float strength = [] { const char* v = std::getenv("AC6_EMITTER_LIGHT"); return v && *v ? std::max(0.0f, float(std::atof(v))) : 0.0f; }();
    return strength;
}

// Object motion blur (settings graphics.motion_blur, AC6_MOTION_BLUR, 0 to 1, 0 by default): only
// with temporal AA or FSR (their motion vectors). Live.
float MotionBlurStrength()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<float> strength{ 0.0f };
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        const std::string v = settings::Get("AC6_MOTION_BLUR");
        strength.store(v.empty() ? 0.0f : std::clamp(float(std::atof(v.c_str())), 0.0f, 1.0f), std::memory_order_release);
        generation.store(now, std::memory_order_release);
    }
    return strength.load(std::memory_order_acquire);
}

// The sea's planar reflection (the setting graphics.planar_reflection, AC6_PLANAR_REFLECTION, or AC6_SSR=planar): the
// scene's draws issued again, mirrored in the sea's plane, at a fraction of the scene's size, then blurred
// (RunPlanarReflection). Read at start: its shaders and pipelines are made then.
bool PlanarSea()
{
    static const bool on = [] {
        const char* v = std::getenv("AC6_SSR");
        const std::string setting = settings::Get("AC6_PLANAR_REFLECTION");
        return (v && std::string_view(v) == "planar") || setting == "1" || setting == "true" || setting == "on";
    }();
    return on;
}

// The sea reflects the scene: AC6_SSR=1 (screen-space reflections, an experiment) or the planar reflection.
bool WaterSsr()
{
    static const bool on = [] { const char* v = std::getenv("AC6_SSR"); return (v && *v && *v != '0') || PlanarSea(); }();
    return on;
}

// AC6_VOLUMETRICS=1: the game's smoke as lit volumes instead of billboards, =debug: both. Off by
// default: nothing built, no geometry shader stage in the layout.
int VolumetricsMode()
{
    static const int mode = [] {
        const char* v = std::getenv("AC6_VOLUMETRICS");
        if (!v || !*v || *v == '0')
            return 0;
        return std::string_view(v) == "debug" ? 2 : 1;
    }();
    return mode;
}

float VolSetting(const char* name, float fallback)
{
    const char* v = std::getenv(name);
    return v && *v ? float(std::atof(v)) : fallback;
}

constexpr uint32_t kSpecVolHarvest = 1u << 30;   // a pipeline key bit: the draw's pipeline gets the harvesting stage
constexpr uint32_t kSpecLateBounds = 1u << 29;   // and the late mesh capture stage
constexpr uint32_t kSpecLine2D = 1u << 28;       // and the 2D lines' rectangles
constexpr uint32_t kVolCapacity = 16384;         // particle records per frame
constexpr uint32_t kVolRecord = 80;              // bytes a record (volumetrics.hlsl)

// The effect draws made volumes, by pixel shader: their kind (volumetrics.hlsl VOL_*), -1 for the others.
// Smoke: the "TranslucentEdge" family's inputs (0: colour add in TEXCOORD3, fog in TEXCOORD1.z) and
// "TranslucentEdgeDetail"'s (1: add in TEXCOORD4, fog in TEXCOORD3.x). Their vertex shaders TankSmoke,
// SimpleExplosionSmoke, ImpactSand (0), RisingSmokeNeo, ImpactSmoke (1). The rest of the effects buffer:
// SimpleExplosionFlame (2) and RisingSmokeNeo's AddDetail fire (3), additive. The
// cloud puffs, Cloud_p (4). The missile trails, MissileSmokeBelt's ribbons (5) and their additive variant (6).
// AC6_VOL_KINDS=<smoke,fire,cloud,trail> (all): which become volumes. The others stay the game's billboards.
int VolumetricLayout(uint64_t psHash)
{
    static const uint32_t kinds = [] {
        const char* v = std::getenv("AC6_VOL_KINDS");
        if (!v || !*v)
            return 0x7Fu;
        const std::string_view list(v);
        uint32_t mask = 0;
        if (list.find("smoke") != std::string_view::npos)
            mask |= 0x03;
        if (list.find("fire") != std::string_view::npos)
            mask |= 0x0C;
        if (list.find("cloud") != std::string_view::npos)
            mask |= 0x10;
        if (list.find("trail") != std::string_view::npos)
            mask |= 0x60;
        return mask;
    }();
    int kind = -1;
    switch (psHash)
    {
    case 0xA8D7F765A3393F89ull:
    case 0xABB89761D796D312ull:
    case 0x2CF84218D7CB5AA2ull:
        kind = 0;
        break;
    case 0x757D23F71F52C060ull:
    case 0x10FBAD402151CCB3ull:
        kind = 1;
        break;
    case 0x244BBBF8D6249424ull:
        kind = 2;
        break;
    case 0x7305CA060F4A2E57ull:
        kind = 3;
        break;
    case 0x5776EB8756E408E9ull:
        kind = 4;
        break;
    case 0x228BE7B164C30800ull:
        kind = 5;
        break;
    case 0xE13C9024369C2A1Dull:
        kind = 6;
        break;
    }
    return kind >= 0 && (kinds >> kind) & 1 ? kind : -1;
}

// The scene surface while the scene is drawn in it: resolved this frame or the last. The surface
// stays the one the last scene used. A menu screen with no 3D scene (after the hangar: the pre-mission screen) draws its
// 2D layers into it, and those must not be jittered, with nothing to resolve the jitter: the clip of the swg layer (the
// game's Flash menus) is in whole pixels and doesn't move with its contents, so on some frames a pixel column at the
// 16:9 box's edge would be covered by neither the menu nor its wrapped copy (a one-pixel black line with temporal AA).
bool TaaSurfaceLive()
{
    return s_r.taaSurface && s_r.frame - s_r.taaSurfaceFrame <= 1;
}

// The sub-pixel jitter of a frame's scene draws (host pixels, -0.5..0.5): Halton (2, 3), 8 positions. AC6_TAA_JITTER=0:
// none (checking the reprojection, AC6_TAA_DEBUG=3).
std::pair<float, float> TaaJitter(uint64_t frame)
{
    static const bool jitter = [] { const char* v = std::getenv("AC6_TAA_JITTER"); return !(v && *v == '0'); }();
    if (!jitter)
        return { 0.0f, 0.0f };
    // FSR upscaling: FSR's sequence, as many phases as the upscale needs (render pixels).
    if (s_r.upscaling)
    {
        float x = 0.0f, y = 0.0f;
        fsr::Jitter(frame, int32_t(std::lround(kGuestWidth * double(s_r.worldResX))), int32_t(std::lround(kGuestWidth * double(s_r.sceneResX))), x, y);
        return { x, y };
    }
    if (s_r.lateUpscale)
    {
        float x = 0.0f, y = 0.0f;
        fsr::Jitter(frame, int32_t(s_r.sceneWidth), int32_t(s_r.targetWidth), x, y);
        return { x, y };
    }
    auto halton = [](uint32_t index, uint32_t base) {
        float f = 1.0f, r = 0.0f;
        for (uint32_t i = index; i > 0; i /= base)
        {
            f /= float(base);
            r += f * float(i % base);
        }
        return r;
    };
    const uint32_t i = uint32_t(frame % 8) + 1;
    return { halton(i, 2) - 0.5f, halton(i, 3) - 0.5f };
}

std::unique_ptr<RenderShader> InternalShader(std::string_view name)
{
    const InternalShaderBlob* blob = FindInternalShader(name);
    if (!blob)
        return nullptr;
    return g_video.device->createShader(blob->data, blob->size, "main", RenderShaderFormat::SPIRV);
}

// --- Textures -------------------------------------------------------------------------------------------

// AC6_ANISO_OVERRIDE (live): 0 = as the game asks.
uint32_t AnisoOverride()
{
    static std::atomic<uint64_t> generation{ UINT64_MAX };
    static std::atomic<uint32_t> value{ 0 };
    if (const uint64_t now = settings::Generation(); generation.load(std::memory_order_acquire) != now)
    {
        const std::string v = settings::Get("AC6_ANISO_OVERRIDE");
        value.store(v.empty() ? 0u : std::clamp(uint32_t(std::max(0, std::atoi(v.c_str()))), 0u, 16u), std::memory_order_relaxed);
        generation.store(now, std::memory_order_release);
    }
    return value.load(std::memory_order_relaxed);
}

// The sampler a texture fetch constant asks for. Bit positions as Xbox D3D's sampler-state setters write
// them (table 0x82691798): dword 0 bits 10-12/13-15/16-18 = address mode U/V/W, dword 3 bits 19-20 =
// mag filter, 21-22 = min filter, 23-24 = mip filter, 25-27 = anisotropy, dword 4 bits 12-21 = signed LOD
// bias (1/32 steps), dword 5 bits 0-1 = border colour.
// `instructionFilters`: a fetch instruction's own filter fields (mag, min, mip 2 bits each, anisotropy 3,
// 3 and 7 keep the fetch constant's), which replace the constant's. `maxAniso`: the most anisotropy allowed.
// `upscaled`: a texture of the scene's draws with FSR upscaling, sampled finer by s_r.worldMipBias.
// `overridable`: AC6_ANISO_OVERRIDE may raise its anisotropy (not the sea's shore mask).
uint32_t GetSampler(const xenos::TextureFetch& fetch, uint32_t instructionFilters = 0x1FF, uint32_t maxAniso = 16, bool upscaled = false,
    bool overridable = true, bool levelQuery = false)
{
    const uint32_t address = (fetch.dw[0] >> 10) & 0x1FF;
    uint32_t filters = (fetch.dw[3] >> 19) & 0x1FF;
    for (const auto [shift, bits, keep] : { std::tuple{ 0u, 3u, 3u }, { 2u, 3u, 3u }, { 4u, 3u, 3u }, { 6u, 7u, 7u } })
        if (const uint32_t v = (instructionFilters >> shift) & bits; v != keep)
            filters = (filters & ~(bits << shift)) | v << shift;
    const uint32_t lodBias = (fetch.dw[4] >> 12) & 0x3FF, border = fetch.dw[5] & 3;
    // AC6_ANISO_OVERRIDE (live): in the key, so a change makes new samplers.
    // An instruction that turns anisotropy off itself (field 0, 7 leaves it to the fetch constant) means it: the override
    // doesn't raise it (the sea's far fetch and its level query, whose footprints must stay inside a map tile).
    // (`raise` is in the key: a slot's own sampler with the same filters does take the override.)
    const bool raise = overridable && ((instructionFilters >> 6) & 7) != 0;
    const uint32_t anisoOverride = AnisoOverride();
    const uint64_t key = address | (uint64_t(filters) << 9) | (uint64_t(lodBias) << 18) | (uint64_t(border) << 28) |
        (uint64_t(std::min(maxAniso, 31u)) << 30) | (uint64_t(upscaled) << 35) | (uint64_t(raise) << 36) |
        (uint64_t(anisoOverride) << 37) | (uint64_t(levelQuery) << 42);
    if (auto it = s_r.samplerIndices.find(key); it != s_r.samplerIndices.end())
        return it->second;
    auto addressMode = [](uint32_t mode) {
        switch (mode)
        {
        case 0: return RenderTextureAddressMode::WRAP;
        case 1: return RenderTextureAddressMode::MIRROR;
        case 2: case 4: return RenderTextureAddressMode::CLAMP;      // clamp to last texel / half way to border
        case 6: return RenderTextureAddressMode::BORDER;
        default: return RenderTextureAddressMode::MIRROR_ONCE;      // mirror-then-clamp variants
        }
    };
    RenderSamplerDesc desc;
    desc.addressU = addressMode(address & 7);
    desc.addressV = addressMode((address >> 3) & 7);
    desc.addressW = addressMode((address >> 6) & 7);
    desc.magFilter = (filters & 3) == 0 ? RenderFilter::NEAREST : RenderFilter::LINEAR;
    desc.minFilter = ((filters >> 2) & 3) == 0 ? RenderFilter::NEAREST : RenderFilter::LINEAR;
    const uint32_t mip = (filters >> 4) & 3;
    desc.mipmapMode = mip == 1 ? RenderMipmapMode::LINEAR : RenderMipmapMode::NEAREST;
    if (mip == 2)
        desc.maxLOD = 0.0f;   // base map only
    // AC6_ANISO_OVERRIDE=<n>: at least n:1 on every linearly filtered texture with mipmaps, beyond
    // what the game asks (it asks on few), still under the draw's cap (the sea's 4:1, AC6_MAX_ANISO).
    const uint32_t aniso = (filters >> 6) & 7;
    uint32_t ratio = aniso >= 2 && aniso <= 5 ? 1u << (aniso - 1) : 1u;
    if (raise && anisoOverride >= 2 && desc.minFilter == RenderFilter::LINEAR && mip != 2)
        ratio = std::max(ratio, anisoOverride);
    if (ratio = std::min(ratio, maxAniso); ratio >= 2)
    {
        desc.anisotropyEnabled = true;
        desc.maxAnisotropy = ratio;
    }
    desc.mipLODBias = float(int32_t(lodBias << 22) >> 22) / 32.0f + (upscaled ? s_r.worldMipBias : 0.0f);
    // A level query's sampler (getCompTexLOD): the sea's shader reads its map with 8:1 anisotropy up
    // to level 2.6 and plain trilinear past it, because the map's textures are atlases of 256-texel tiles with 8 texels
    // of border and four mip levels: a footprint that reaches further than the border samples the next tile. At level L
    // an n:1 footprint reaches 2^L x (1/2 + 1/n) texels from the pixel: 3.8 at 2.6, and 8 only at 3.7 (8:1) or 3.8 (16:1).
    // So the query answers up to a level less (AC6_TERRAIN_SHARP_REACH, 1, 0: as the console), by what the ratio in use
    // leaves inside the border: the ground and the sea keep their sharp fetch twice as far out, with nothing from the
    // next tile (no dots with 1, dots again with 2).
    if (levelQuery)
    {
        static const float reach = [] { const char* v = std::getenv("AC6_TERRAIN_SHARP_REACH"); return v && *v ? std::clamp(float(std::atof(v)), 0.0f, 1.0f) : 1.0f; }();
        const float n = float(std::clamp(std::max(8u, overridable ? anisoOverride : 0u), 1u, std::max(maxAniso, 1u)));
        desc.mipLODBias -= std::clamp(std::log2(8.0f / (0.5f + 1.0f / n)) - 2.6f, 0.0f, reach);
    }
    desc.borderColor = border == 1 ? RenderBorderColor::OPAQUE_WHITE : border == 0 ? RenderBorderColor::TRANSPARENT_BLACK
                                                                                  : RenderBorderColor::OPAQUE_BLACK;
    // Samplers are kept for the session, and the key above has inputs that often come to the same sampler (each value
    // of the live anisotropy override on a texture that doesn't take it, bias variants): those share one, so the table
    // holds only distinct samplers.
    const std::array<uint32_t, 12> state = { uint32_t(desc.minFilter), uint32_t(desc.magFilter), uint32_t(desc.mipmapMode),
        uint32_t(desc.addressU), uint32_t(desc.addressV), uint32_t(desc.addressW), std::bit_cast<uint32_t>(desc.mipLODBias),
        desc.anisotropyEnabled ? desc.maxAnisotropy : 0u, uint32_t(desc.borderColor), std::bit_cast<uint32_t>(desc.minLOD),
        std::bit_cast<uint32_t>(desc.maxLOD), 0u };
    if (auto it = s_r.samplersByState.find(state); it != s_r.samplersByState.end())
    {
        s_r.samplerIndices.emplace(key, it->second);
        return it->second;
    }
    if (s_r.fetchSamplers.size() + 3 >= kSamplerDescriptors)   // the last slot: SMAA's point sampler
    {
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            log::Error("Sampler table full ({} samplers): new sampler states use sampler 0 (linear, clamped)", s_r.fetchSamplers.size());
        }
        return 0;
    }
    const uint32_t index = uint32_t(2 + s_r.fetchSamplers.size());
    s_r.samplersByState.emplace(state, index);
    s_r.fetchSamplers.push_back(g_video.device->createSampler(desc));
    s_r.samplerSet->setSampler(index, s_r.fetchSamplers.back().get());
    s_r.samplerIndices.emplace(key, index);
    return index;
}

// A texture, or the game stops with a message: an image that couldn't be made (out of video memory, past the GPU's
// limits) can't be drawn into or sampled, and the next view or barrier of it would crash in the driver.
std::unique_ptr<RenderTexture> MakeTexture(const RenderTextureDesc& desc, const char* what)
{
    std::unique_ptr<RenderTexture> texture = g_video.device->createTexture(desc);
    if (!texture)
    {
        log::Error("Could not make {} ({}x{}x{}, {} level(s), format {}, {} sample(s)): out of video memory, or past the "
                   "GPU's limits", what, desc.width, desc.height, desc.depth, desc.mipLevels, int(desc.format),
                   uint32_t(desc.multisampling.sampleCount));
        crash::Shutdown(1, "out of video memory");
    }
    return texture;
}

uint32_t AllocateDescriptor()
{
    if (!s_r.freeDescriptors.empty())
    {
        const uint32_t index = s_r.freeDescriptors.back();
        s_r.freeDescriptors.pop_back();
        s_r.scaledDescriptors[index] = {};   // (a rendered image's scale, from the image that had it before)
        s_r.swizzleTraceDescriptors.erase(index);
        return index;
    }
    if (s_r.nextDescriptor < kTextureDescriptors)
        return s_r.nextDescriptor++;
    static bool warned = false;
    if (!warned)
    {
        warned = true;
        log::Error("Texture descriptors used up ({}): new images are sampled as the null texture", kTextureDescriptors);
    }
    return kNullTexture2D;
}

// A descriptor no longer used by what's recorded from now on: reused once the frame being recorded has run on the GPU
// (draws recorded earlier in it may still sample the old image through it).
void RetireDescriptor(uint32_t index)
{
    if (index > kNullTextureCube)
        s_r.retiredDescriptors.push_back(index);
}

// A descriptor's image. The null textures' slots (0-2) keep theirs: an image that got no descriptor of its own
// (AllocateDescriptor gives kNullTexture2D when they're used up) is sampled as the null texture, and the draws that
// sample "no texture" keep seeing nothing.
void SetDescriptor(uint32_t index, const RenderTexture* texture, const RenderTextureView* view)
{
    if (index <= kNullTextureCube)
        return;
    s_r.textureSet->setTexture(index, texture, RenderTextureLayout::SHADER_READ, view);
}

uint32_t GetTextureImpl(RenderCommandList& cl, const xenos::TextureFetch& fetch);

// --- Texture dumps and replacements (gpu/texture_mods.h) ---------------------------------------
// A guest 2D texture decoded as the upload decodes it (untiled, byte order fixed, its separate mip levels), tightly
// packed, for a dump. The layout is GetTextureImpl's: keep the two in step.
bool DecodeTexture(const xenos::TextureFetch& fetch, mods::Image& out)
{
    const uint32_t format = fetch.Format();
    const RenderFormat hostFormat = TextureFormat(format);
    if (fetch.Type() != 2 || fetch.Dimension() != 1 || hostFormat == RenderFormat::UNKNOWN || !fetch.BaseAddress())
        return false;
    const uint32_t width = fetch.Width(), height = fetch.Height();
    const uint32_t bw = xenos::BlockWidth(format), bh = xenos::BlockHeight(format);
    const uint32_t bytesPerBlock = xenos::BitsPerPixel(format) * bw * bh / 8;
    const uint32_t blocksX = (width + bw - 1) / bw;
    const uint32_t pitchBlocks = std::max(fetch.Pitch() / bw, blocksX);
    const uint8_t* base = g_memory.As<uint8_t>(PhysicalToGuest(fetch.BaseAddress()));
    const bool packedMips = (fetch.dw[5] >> 11) & 1;
    auto packedOffset = [bw, bh](uint32_t w, uint32_t h, uint32_t tile, uint32_t& ox, uint32_t& oy) {
        const uint32_t lw = std::bit_width(std::max(w, 1u) - 1), lh = std::bit_width(std::max(h, 1u) - 1);
        ox = oy = 0;
        if (std::min(lw, lh) > 4)
            return;
        if (tile < 3)
            (lw > lh ? oy : ox) = 16 >> tile;
        else
            (lw > lh ? ox : oy) = 16 >> (tile - 2);
        ox /= bw;
        oy /= bh;
    };
    uint32_t baseOffsetX = 0, baseOffsetY = 0;
    if (packedMips && fetch.Tiled())
        packedOffset(std::bit_ceil(width), std::bit_ceil(height), 0, baseOffsetX, baseOffsetY);
    out.format = int(hostFormat);
    out.width = width;
    out.height = height;
    out.levels.clear();
    auto level = [&](const uint8_t* src, uint32_t w, uint32_t h, bool tiled, uint32_t pitch, uint32_t ox, uint32_t oy) {
        const uint32_t bx = (w + bw - 1) / bw, by = (h + bh - 1) / bh, rowBytes = bx * bytesPerBlock;
        std::vector<uint8_t> data(size_t(rowBytes) * by);
        for (uint32_t y = 0; y < by; y++)
            ConvertRow(data.data() + size_t(y) * rowBytes, src, bx, y, tiled, pitch, ox, oy, bytesPerBlock, fetch.Endian());
        out.levels.push_back(std::move(data));
    };
    level(base, width, height, fetch.Tiled(), pitchBlocks, baseOffsetX, baseOffsetY);
    if (fetch.Tiled() && fetch.MipAddress() && std::has_single_bit(width) && std::has_single_bit(height))
    {
        const uint8_t* mipBase = g_memory.As<uint8_t>(PhysicalToGuest(fetch.MipAddress()));
        uint64_t mipBytes = 0;
        uint32_t packedBase = 0;
        for (uint32_t l = 1; l <= fetch.MaxMip(); l++)
        {
            const uint32_t w = std::max(1u, width >> l), h = std::max(1u, height >> l);
            const uint32_t alignedBlocksX = ((w + bw - 1) / bw + 31) & ~31u, alignedBlocksY = ((h + bh - 1) / bh + 31) & ~31u;
            if (packedMips && !packedBase && std::min(w, h) <= 16)
                packedBase = l;
            if (packedBase)
            {
                const uint32_t tw = std::max(1u, width >> packedBase);
                uint32_t ox, oy;
                packedOffset(w, h, l - packedBase, ox, oy);
                level(mipBase + mipBytes, w, h, true, ((tw + bw - 1) / bw + 31) & ~31u, ox, oy);
                continue;
            }
            level(mipBase + mipBytes, w, h, true, alignedBlocksX, 0, 0);
            mipBytes += (uint64_t(alignedBlocksX) * alignedBlocksY * bytesPerBlock + 4095) & ~uint64_t(4095);
        }
    }
    return true;
}

// A texture written to dump/textures/<hash>.dds once (this run, or found there already).
void DumpGuestTexture(const xenos::TextureFetch& fetch, uint64_t hash)
{
    if (!hash || mods::Dumped(hash))
        return;
    mods::Image image;
    if (!DecodeTexture(fetch, image))
        return;
    mods::NoteDumped(hash);
    static uint32_t logged = 0;
    if (logged++ < 8)
        log::Info("Texture dump: {:016X} ({}x{}, {} level(s), format {}) to {}", hash, image.width, image.height, image.levels.size(),
            fetch.Format(), platform::PathUtf8(mods::DumpDirectory()));
    mods::WriteDdsAsync(mods::DumpDirectory() / std::format("{:016X}.dds", hash), std::move(image));
}

// The dumps asked for from the overlay's browser (at a present: the held frame's too, the guest data is as it was).
void ServiceDumpRequests()
{
    for (const mods::FrameTexture& r : mods::TakeDumpRequests())
    {
        xenos::TextureFetch fetch;
        std::memcpy(fetch.dw, r.fetch, sizeof(fetch.dw));
        if (mods::Dumped(r.hash))
            continue;
        DumpGuestTexture(fetch, r.hash);
    }
}

// A replacement's image in place of the guest texture's: the file's size, levels and format. Sampled through the guest
// fetch constant's swizzle as the original's format maps it (so a file made from a dump looks as the dump). 0 if the file
// can't be used (the guest texture is uploaded as usual).
uint32_t ReplaceTexture(RenderCommandList& cl, HostTexture& t, const xenos::TextureFetch& fetch, RenderFormat original,
    const std::filesystem::path& path, uint64_t hash, bool guestHasMips)
{
    mods::Image image;
    std::string error;
    if (!mods::ReadDds(path, image, error) || image.levels.empty() || !image.width || !image.height)
    {
        static std::set<std::string> warned;
        if (warned.insert(platform::PathUtf8(path)).second)
            log::Warn("Texture mod {}: {} (the game's texture is used)", platform::PathUtf8(path), error.empty() ? "empty" : error);
        return 0;
    }
    // A mip chain made for a file without one (8-bit colour, BC files keep theirs): a larger texture than the game's needs
    // it to stay smooth further away, and the draws sample it through its mips (mippedReplacements), whatever the game's
    // sampler asks (the F-16C's paint, one level, "base map only").
    (void)guestHasMips;
    if (image.levels.size() == 1)
        mods::MakeMips(image, uint32_t(std::bit_width(std::max(image.width, image.height))));
    const RenderFormat format = RenderFormat(image.format);
    const bool block = format == RenderFormat::BC1_UNORM || format == RenderFormat::BC2_UNORM || format == RenderFormat::BC3_UNORM ||
        format == RenderFormat::BC4_UNORM || format == RenderFormat::BC5_UNORM || format == RenderFormat::BC7_UNORM;
    const uint32_t bw = block ? 4 : 1;
    const uint32_t bytesPerBlock = format == RenderFormat::BC1_UNORM || format == RenderFormat::BC4_UNORM ? 8 : block ? 16
        : format == RenderFormat::R8_UNORM ? 1 : 4;
    const uint32_t levels = uint32_t(image.levels.size());
    // The staging buffer: rows padded to 256 bytes, levels 512-aligned.
    uint64_t total = 0;
    for (uint32_t l = 0; l < levels; l++)
    {
        const uint32_t w = std::max(1u, image.width >> l), h = std::max(1u, image.height >> l);
        const uint32_t rowBytes = (w + bw - 1) / bw * bytesPerBlock, rows = (h + bw - 1) / bw;
        total = ((total + 511) & ~uint64_t(511)) + uint64_t((rowBytes + 255) & ~255u) * rows;
    }
    if (t.texture)
    {
        s_r.retiredViews.push_back(std::move(t.view));
        s_r.retiredTextures.push_back(std::move(t.texture));
        RetireDescriptor(t.descriptor);
        t.descriptor = 0;   // a frame in flight samples the old image through the old descriptor
        s_r.textureBytes -= t.bytes;
        t.bytes = 0;
    }
    t.texture = MakeTexture(RenderTextureDesc::Texture2D(image.width, image.height, levels, format), "a replacement texture");
    RenderTextureViewDesc viewDesc = RenderTextureViewDesc::Texture2D(format);
    viewDesc.componentMapping = FetchComponentMapping(fetch.dw[3], original);
    t.view = t.texture->createTextureView(viewDesc);
    t.descriptor = AllocateDescriptor();
    SetDescriptor(t.descriptor, t.texture.get(), t.view.get());
    t.levels = levels;
    t.replaced = true;
    if (levels > 1)
        s_r.mippedReplacements.insert(t.descriptor);
    auto staging = g_video.device->createBuffer(RenderBufferDesc::UploadBuffer(total));
    uint8_t* memory = static_cast<uint8_t*>(staging->map());
    cl.barriers(RenderBarrierStage::COPY, RenderTextureBarrier(t.texture.get(), RenderTextureLayout::COPY_DEST));
    uint64_t at = 0;
    for (uint32_t l = 0; l < levels; l++)
    {
        const uint32_t w = std::max(1u, image.width >> l), h = std::max(1u, image.height >> l);
        const uint32_t rowBytes = (w + bw - 1) / bw * bytesPerBlock, rows = (h + bw - 1) / bw, rowPitch = (rowBytes + 255) & ~255u;
        at = (at + 511) & ~uint64_t(511);
        const std::vector<uint8_t>& src = image.levels[l];
        for (uint32_t y = 0; y < rows && size_t(y + 1) * rowBytes <= src.size(); y++)
            std::memcpy(memory + at + uint64_t(y) * rowPitch, src.data() + size_t(y) * rowBytes, rowBytes);
        cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(t.texture.get(), l, 0),
            RenderTextureCopyLocation::PlacedFootprint(staging.get(), format, w, h, 1, rowPitch * bw / bytesPerBlock, at));
        at += uint64_t(rowPitch) * rows;
    }
    cl.barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(t.texture.get(), RenderTextureLayout::SHADER_READ));
    staging->unmap();
    s_r.retiredStaging.push_back(std::move(staging));
    static std::set<uint64_t> logged;
    if (logged.insert(hash).second)
        log::Info("Texture mod: {:016X} ({}x{}) replaced by {} ({}x{}, {} level(s))", hash, fetch.Width(), fetch.Height(), platform::PathUtf8(path),
            image.width, image.height, levels);
    return t.descriptor;
}

const struct DumpTexture
{
    uint32_t address = 0;
    std::set<uint64_t> frames;
} dumpTexture = [] {
    DumpTexture d;
    if (const char* v = std::getenv("AC6_DUMP_TEXTURE"))
        if (const char* at = std::strchr(v, '@'))
        {
            d.address = uint32_t(std::stoul(std::string(v, at), nullptr, 16));
            std::stringstream list(at + 1);
            for (std::string f; std::getline(list, f, ',');)
                d.frames.insert(std::stoull(f));
        }
    return d;
}();

// Returns the descriptor index of the host texture for a guest 2D texture fetch constant, uploading the
// guest data when it is new or changed.
// AC6_DUMP_TEXTURE=<physical address hex>@<frame>[,<frame>...]: reads back that texture's host image
// (level 0) the first time it is bound in those frames, written at present like a resolve capture (index
// 900, resolve_<frame>_900.bmp with AC6_RESOLVE_STATS_BMP=1), and logs every upload of it: compares what
// the GPU samples with guest memory.
uint32_t GetTexture(RenderCommandList& cl, const xenos::TextureFetch& fetch)
{
    const uint32_t descriptor = GetTextureImpl(cl, fetch);
    static uint64_t dumped = UINT64_MAX;
    if (dumpTexture.frames.contains(s_r.frame) && dumped != s_r.frame && (fetch.BaseAddress() & 0x1FFFFFFF) == (dumpTexture.address & 0x1FFFFFFF))
        for (auto& [key, t] : s_r.textures)
            if (t.texture && t.descriptor == descriptor)
            {
                dumped = s_r.frame;
                const RenderFormat format = TextureFormat(fetch.Format());
                const uint32_t bpp = format == RenderFormat::B8G8R8A8_UNORM || format == RenderFormat::R8G8B8A8_UNORM ? 4 : 0;
                log::Info("Texture dump {:08X}: descriptor {} hash {:016X} format {} {}x{}", fetch.BaseAddress(), descriptor, t.contentHash,
                    int(format), fetch.Width(), fetch.Height());
                if (!bpp)
                    break;
                auto buffer = g_video.device->createBuffer(RenderBufferDesc::ReadbackBuffer(uint64_t(fetch.Width()) * fetch.Height() * bpp));
                cl.barriers(RenderBarrierStage::COPY, RenderTextureBarrier(t.texture.get(), RenderTextureLayout::COPY_SOURCE));
                cl.copyTextureRegion(RenderTextureCopyLocation::PlacedFootprint(buffer.get(), format, fetch.Width(), fetch.Height(), 1, fetch.Width()),
                    RenderTextureCopyLocation::Subresource(t.texture.get()));
                cl.barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(t.texture.get(), RenderTextureLayout::SHADER_READ));
                s_r.resolveCaptures.push_back({ 900, std::format("texture {:08X}", fetch.BaseAddress()), std::move(buffer), fetch.Width(), fetch.Height(),
                    format == RenderFormat::R8G8B8A8_UNORM ? RenderFormat::B8G8R8A8_UNORM : format });
                break;
            }
    return descriptor;
}

uint32_t GetTextureImpl(RenderCommandList& cl, const xenos::TextureFetch& fetch)
{
    // Resolve destinations live on the GPU only. They are sampled through the fetch constant's
    // swizzle like uploaded textures (the luminance chain relies on it to replicate its one channel).
    auto& resolvedImages = ResolvedImages(fetch.BaseAddress() & 0x1FFFFFFF);   // (the HDR canvas's twins)
    if (fetch.Type() == 2 && fetch.BaseAddress())
        if (auto it = resolvedImages.find(fetch.BaseAddress() & 0x1FFFFFFF); it != resolvedImages.end() && it->second.texture)
        {
            // A depth resolve holds 24_8 texels on the console: read as 8_8_8_8, component X is the stencil.
            // The motion blur pass rebuilds the cockpit's depth that way (stencil 120-129 = the cockpit),
            // which keeps the canopy glass drawn after it behind the instrument panel.
            HostImage* chosen = &it->second;
            if (fetch.Format() == 6)
                if (auto st = s_r.resolvedStencil.find(fetch.BaseAddress() & 0x1FFFFFFF); st != s_r.resolvedStencil.end() && st->second.texture)
                    chosen = &st->second;
            HostImage& image = *chosen;
            const uint32_t sw = fetch.dw[3] & 0x1FFE;
            // Mip levels the game resolved: the chain's levels written from 0 on, up to the
            // fetch constant's last level (the passes making the levels sample level 0 alone, with max mip 0).
            if (const uint32_t levels = image.mipChain && image.mipChain->texture
                    ? std::min(uint32_t(std::countr_one(image.mipWritten)), fetch.MaxMip() + 1) : 0; levels > 1)
            {
                HostImage& chain = *image.mipChain;
                auto& [view, descriptor] = chain.swizzled[sw | levels << 16];
                if (!view)
                {
                    RenderTextureViewDesc viewDesc = RenderTextureViewDesc::Texture2D(chain.format);
                    viewDesc.mipLevels = levels;
                    viewDesc.componentMapping = FetchComponentMapping(sw, image.bgraOrder ? RenderFormat::B8G8R8A8_UNORM : chain.format);
                    view = chain.texture->createTextureView(viewDesc);
                    if (!descriptor)
                        descriptor = AllocateDescriptor();
                    SetDescriptor(descriptor, chain.texture.get(), view.get());
                }
                s_r.scaledDescriptors[descriptor] = ImageScale(chain);
                return descriptor;
            }
            auto& [view, descriptor] = image.swizzled[sw];
            if (!view)
            {
                RenderTextureViewDesc viewDesc = RenderTextureViewDesc::Texture2D(image.format);
                viewDesc.componentMapping = FetchComponentMapping(sw, image.bgraOrder ? RenderFormat::B8G8R8A8_UNORM : image.format);
                view = image.texture->createTextureView(viewDesc);
                if (!descriptor)
                    descriptor = AllocateDescriptor();
                SetDescriptor(descriptor, image.texture.get(), view.get());
            }
            s_r.scaledDescriptors[descriptor] = ImageScale(image);   // rendered: at its scale
            s_r.shadowDescriptors[descriptor] = fetch.Format() == 22;   // k_24_8: a depth image read as a shadow map
            if (image.swizzleTrace)
                s_r.swizzleTraceDescriptors.insert(descriptor);
            return descriptor;
        }

    const uint32_t format = fetch.Format();
    const RenderFormat hostFormat = TextureFormat(format);
    const bool cube = fetch.Dimension() == 3;
    if (fetch.Type() != 2 || (fetch.Dimension() != 1 && !cube) || hostFormat == RenderFormat::UNKNOWN || !fetch.BaseAddress())
    {
        static std::set<uint32_t> seen;
        if (fetch.Type() == 2 && fetch.BaseAddress() && seen.size() < 64 && seen.insert(format | (fetch.Dimension() << 8)).second)
            log::Warn("Unsupported texture: format {} dimension {} ({}x{}) at {:08X}", format, fetch.Dimension(), fetch.Width(), fetch.Height(), fetch.BaseAddress());
        return cube ? kNullTextureCube : kNullTexture2D;
    }

    // Content is checked once per frame: hashing every use would cost gigabytes per frame in missions. Checked
    // before working out the layout, which most uses (already checked this frame) don't need.
    const uint64_t key = HashBytes(fetch.dw, 12) ^ (uint64_t(fetch.dw[3] & 0x1FFE) << 48);
    HostTexture& t = s_r.textures[key];
    if (t.texture && t.lastFrame == s_r.frame)
        return t.descriptor;
    // Listed for the overlay's texture browser once a frame, as the function leaves (gpu/texture_mods.h).
    struct Record
    {
        HostTexture* t;
        const xenos::TextureFetch& fetch;
        ~Record()
        {
            if (!t || !t->texture || t->recordedFrame == s_r.frame || !t->contentHash)
                return;
            t->recordedFrame = s_r.frame;
            mods::FrameTexture f;
            f.hash = t->contentHash;
            f.width = fetch.Width();
            f.height = fetch.Height();
            f.levels = fetch.MaxMip() + 1;
            f.format = fetch.Format();
            f.descriptor = t->descriptor;
            f.replaced = t->replaced;
            f.cube = fetch.Dimension() == 3;
            std::memcpy(f.fetch, fetch.dw, sizeof(f.fetch));
            s_r.frameTextures.push_back(f);
        }
    } record{ &t, fetch };

    const uint32_t width = fetch.Width(), height = fetch.Height();
    const uint32_t bw = xenos::BlockWidth(format), bh = xenos::BlockHeight(format);
    const uint32_t bytesPerBlock = xenos::BitsPerPixel(format) * bw * bh / 8;
    const uint32_t blocksX = (width + bw - 1) / bw, blocksY = (height + bh - 1) / bh;
    const uint32_t pitchBlocks = std::max(fetch.Pitch() / bw, blocksX);
    // Tiled surfaces are laid out in 32x32-block macro tiles.
    const uint32_t rowsAllocated = fetch.Tiled() ? (blocksY + 31) & ~31u : blocksY;
    const uint32_t pitchAllocated = fetch.Tiled() ? (pitchBlocks + 31) & ~31u : pitchBlocks;
    // A cube map is six faces, each a 2D surface of the same layout, 4 KB aligned.
    const uint32_t layers = cube ? 6 : 1;
    const uint64_t faceBytes = uint64_t(pitchAllocated) * rowsAllocated * bytesPerBlock;
    const uint64_t layerStride = cube ? (faceBytes + 4095) & ~uint64_t(4095) : faceBytes;
    const uint64_t guestBytes = layerStride * (layers - 1) + faceBytes;
    const uint8_t* base = g_memory.As<uint8_t>(PhysicalToGuest(fetch.BaseAddress()));

    // Mip levels, laid out as the game's D3D computes them (sub_821DF998 / sub_821DF878): from the fetch
    // constant's mip address, level n of the power-of-two base size >> n, each padded to 32x32 blocks and
    // 4 KB, one after another. Once a level is 16 texels or less in either dimension the rest share a
    // packed tail, which isn't decoded: sampling stops at the last separate level. Only power-of-two 2D
    // textures, whose Vulkan mip sizes match. Without mips, minified textures would sample single texels: the
    // terrain atlas's alpha (its water mask) turns to noise and darkens the ground.
    // Packed mips (fetch dword 5 bit 11): every level from the first one of 16 texels or less in a
    // dimension shares one tiled 32x32-block tile, each at an offset (the base level too when it is
    // that small). Offsets as in Xenia's TextureInfo::GetPackedTileOffset.
    const bool packedMips = (fetch.dw[5] >> 11) & 1;
    auto packedOffset = [bw, bh](uint32_t w, uint32_t h, uint32_t tile, uint32_t& ox, uint32_t& oy) {
        const uint32_t lw = std::bit_width(std::max(w, 1u) - 1), lh = std::bit_width(std::max(h, 1u) - 1);
        ox = oy = 0;
        if (std::min(lw, lh) > 4)
            return;
        if (tile < 3)
            (lw > lh ? oy : ox) = 16 >> tile;
        else
            (lw > lh ? ox : oy) = 16 >> (tile - 2);
        ox /= bw;
        oy /= bh;
    };
    uint32_t baseOffsetX = 0, baseOffsetY = 0;
    if (packedMips && fetch.Tiled() && !cube)
        packedOffset(std::bit_ceil(width), std::bit_ceil(height), 0, baseOffsetX, baseOffsetY);
    struct MipLevel { uint32_t width, height; const uint8_t* src; uint32_t pitchBlocks, offsetX, offsetY; };
    // Inline (a fetch has at most 15 levels): this runs for every textured fetch of every draw.
    struct MipList
    {
        MipLevel items[16];
        uint32_t count = 0;
        void push_back(const MipLevel& m) { if (count < 16) items[count++] = m; }
        size_t size() const { return count; }
        const MipLevel& operator[](size_t i) const { return items[i]; }
        const MipLevel* begin() const { return items; }
        const MipLevel* end() const { return items + count; }
    } mips;
    uint64_t mipBytes = 0;
    if (!cube && fetch.Tiled() && fetch.MipAddress() && std::has_single_bit(width) && std::has_single_bit(height))
    {
        const uint8_t* mipBase = g_memory.As<uint8_t>(PhysicalToGuest(fetch.MipAddress()));
        uint32_t packedBase = 0;   // first level in the packed tile (0: not reached)
        for (uint32_t level = 1; level <= fetch.MaxMip(); level++)
        {
            const uint32_t w = std::max(1u, width >> level), h = std::max(1u, height >> level);
            const uint32_t alignedBlocksX = ((w + bw - 1) / bw + 31) & ~31u, alignedBlocksY = ((h + bh - 1) / bh + 31) & ~31u;
            if (packedMips && !packedBase && std::min(w, h) <= 16)
                packedBase = level;
            if (packedBase)
            {
                // The tile is laid out like the first packed level's surface.
                const uint32_t tw = std::max(1u, width >> packedBase);
                uint32_t ox, oy;
                packedOffset(w, h, level - packedBase, ox, oy);
                mips.push_back({ w, h, mipBase + mipBytes, ((tw + bw - 1) / bw + 31) & ~31u, ox, oy });
                if (level == fetch.MaxMip())
                    mipBytes += (uint64_t(alignedBlocksX) * alignedBlocksY * bytesPerBlock + 4095) & ~uint64_t(4095);
                continue;
            }
            mips.push_back({ w, h, mipBase + mipBytes, alignedBlocksX, 0, 0 });
            mipBytes += (uint64_t(alignedBlocksX) * alignedBlocksY * bytesPerBlock + 4095) & ~uint64_t(4095);
        }
        if (packedBase && mipBytes == 0)
            mipBytes = 4096;
    }

    t.lastFrame = s_r.frame;
    // Pages watched since the last check and not written: unchanged without reading them (hashing
    // every texture every frame would cost ~9 ms in missions).
    const uint32_t baseGuest = g_memory.MapVirtual(base), mipGuest = mipBytes ? g_memory.MapVirtual(mips[0].src) : 0;
    static const bool watch = [] { const char* v = std::getenv("AC6_TEXTURE_WATCH"); return !v || *v != '0'; }();
    if (!watch)
        t.dynamic = true;
    if (t.texture && t.watchGeneration && g_memory.Unchanged(baseGuest, uint32_t(guestBytes), t.watchGeneration) &&
        (!mipBytes || g_memory.Unchanged(mipGuest, uint32_t(mipBytes), t.watchGeneration)))
        return t.descriptor;
    ScopedNs timer(s_stats.textureNs);
    // Watched before hashing, so a write during or after the hash is seen at the next check.
    t.watchGeneration = 0;
    if (!t.dynamic)
    {
        t.watchGeneration = g_memory.Watch(baseGuest, uint32_t(guestBytes));
        if (mipBytes)
            g_memory.Watch(mipGuest, uint32_t(mipBytes));
    }
    s_stats.hashedBytes += guestBytes + mipBytes;
    // The pipeline ring: a texture read without being watched.
    if (t.dynamic && capture::RingEnabled())
    {
        capture::OnRendererRead(baseGuest, uint32_t(guestBytes));
        if (mipBytes)
            capture::OnRendererRead(mipGuest, uint32_t(mipBytes));
    }
    const uint64_t hash = HashLarge(base, guestBytes) ^ (mipBytes ? HashLarge(mips[0].src, mipBytes) * 31 : 0);
    if (t.texture && t.contentHash == hash)
    {
        // A texture rewritten at most now and then goes back to being watched.
        t.changeStreak = 0;
        if (t.dynamic && ++t.unchangedChecks >= 120)
            t.dynamic = false;
        return t.descriptor;
    }
    // Rewritten at three checks in a row (movie frames, render-to-memory): a fault per page per frame
    // would cost more than hashing, so it is just hashed from now on.
    if (t.texture && ++t.changeStreak >= 3 && !t.dynamic)
    {
        t.dynamic = true;
        t.watchGeneration = 0;
    }
    t.unchangedChecks = 0;
    // A mod's replacement for this content (gpu/texture_mods.h): its image instead of the upload. (The index has the
    // mods only with replacements on, the language pack's and the button glyphs either way.)
    if (!cube)
        if (const auto path = mods::FindReplacement(hash))
            if (const uint32_t descriptor = ReplaceTexture(cl, t, fetch, hostFormat, *path, hash, mips.size() > 0))
            {
                t.contentHash = hash;
                if (mods::DumpOn())
                    DumpGuestTexture(fetch, hash);
                return descriptor;
            }
    // Linear (host) size of the upload: rows padded to 256 bytes, every level 512-aligned.
    const uint32_t rowBytes = blocksX * bytesPerBlock;
    const uint32_t rowPitch = (rowBytes + 255) & ~255u;
    uint64_t linearBytes = layers * ((uint64_t(rowPitch) * blocksY + 511) & ~uint64_t(511));
    for (const MipLevel& m : mips)
        linearBytes += (uint64_t((((m.width + bw - 1) / bw) * bytesPerBlock + 255) & ~255u) * ((m.height + bh - 1) / bh) + 511) & ~uint64_t(511);
    // Large textures get a staging buffer of their own, released once the frame's GPU work has completed,
    // instead of a share of the frame's upload ring: the mission terrain is six 21 MB tiles (4096x4096 DXT5
    // with mips) that together exceed the ring, and the ones deferred frame after frame would sample the null
    // texture, whose zero alpha sends their land down the terrain shader's water path (dark, sky-coloured
    // land across the city).
    const bool staged = linearBytes > (4ull << 20);
    // A level-load burst of new textures can exceed the frame's upload ring: past the point that leaves
    // 32 MB for the frame's draws, uploads wait for the next frame (the texture samples as before, or
    // as the null texture until then) instead of the ring dropping the rest of the frame's work.
    if (!staged && s_r.upload.used + linearBytes + (32ull << 20) > s_r.upload.size)
    {
        s_stats.deferredUploads++;
        static uint32_t deferred = 0;
        if (deferred++ < 4)
            log::Info("Texture upload deferred to the next frame (upload ring {} of {} MB used)", s_r.upload.used >> 20, s_r.upload.size >> 20);
        if (t.texture)
            return t.descriptor;
        record.t = nullptr;   // (erased below)
        s_r.textures.erase(key);
        return cube ? kNullTextureCube : kNullTexture2D;
    }

    // The cache key covers the fetch constant's address, format and size, not its mip fields (dwords 4-5): a
    // texture first bound without mips (while it loads) and later with them needs a new image. Copying the
    // mips into levels the image doesn't have would write past its end on the GPU, over whatever the driver
    // placed next to it (a block of noise on the player's paint texture, speckles on city textures).
    const uint32_t levels = uint32_t(1 + mips.size());
    if (t.texture && (t.levels != levels || t.replaced))
    {
        t.replaced = false;
        s_r.retiredViews.push_back(std::move(t.view));
        s_r.retiredTextures.push_back(std::move(t.texture));
        // A new descriptor: a frame still in flight samples the old texture through the old one.
        RetireDescriptor(t.descriptor);
        t.descriptor = 0;
        s_r.textureBytes -= t.bytes;
        t.bytes = 0;
    }
    if (!t.texture)
    {
        RenderTextureDesc desc = RenderTextureDesc::Texture2D(width, height, levels, hostFormat);
        if (cube)
        {
            desc.arraySize = 6;
            desc.flags = RenderTextureFlag::CUBE;
        }
        t.texture = MakeTexture(desc, "a game texture");
        t.bytes = linearBytes;
        s_r.textureBytes += t.bytes;
        RenderTextureViewDesc viewDesc = cube ? RenderTextureViewDesc::TextureCube(hostFormat) : RenderTextureViewDesc::Texture2D(hostFormat);
        viewDesc.componentMapping = FetchComponentMapping(fetch.dw[3], hostFormat);
        t.view = t.texture->createTextureView(viewDesc);
        if (!t.descriptor)
            t.descriptor = AllocateDescriptor();
        SetDescriptor(t.descriptor, t.texture.get(), t.view.get());
        t.levels = levels;
    }
    t.contentHash = hash;
    s_stats.uploads++;
    if (!cube && mods::DumpOn())
        DumpGuestTexture(fetch, hash);
    if (dumpTexture.address && (fetch.BaseAddress() & 0x1FFFFFFF) == (dumpTexture.address & 0x1FFFFFFF))
        log::Info("Texture {:08X} uploaded in frame {} (hash {:016X}, descriptor {})", fetch.BaseAddress(), s_r.frame, hash, t.descriptor);
    s_stats.uploadedBytes += guestBytes;

    // Untile / copy into the upload ring (or the texture's staging buffer) as a linear block image, then
    // fix the byte order.
    std::unique_ptr<RenderBuffer> staging;
    uint8_t* stagingMemory = nullptr;
    uint64_t stagingUsed = 0;
    if (staged)
    {
        staging = g_video.device->createBuffer(RenderBufferDesc::UploadBuffer(linearBytes));
        stagingMemory = static_cast<uint8_t*>(staging->map());
        s_stats.stagedUploads++;
    }
    RenderBuffer* const uploadBuffer = staged ? staging.get() : s_r.upload.buffer.get();
    auto allocate = [&](uint64_t bytes) -> UploadRing::Allocation {
        if (!staged)
            return s_r.upload.Allocate(bytes, 512);
        const uint64_t at = (stagingUsed + 511) & ~uint64_t(511);
        stagingUsed = at + bytes;
        return { stagingMemory + at, at, 0 };
    };
    MarkPass([&] { return std::format("texture upload {}x{} f{}{}", width, height, int(hostFormat), layers > 1 ? std::format(" x{} layers", layers) : ""); });
    cl.barriers(RenderBarrierStage::COPY, RenderTextureBarrier(t.texture.get(), RenderTextureLayout::COPY_DEST));
    for (uint32_t layer = 0; layer < layers; layer++)
    {
        const uint8_t* src = base + layer * layerStride;
        auto a = allocate(uint64_t(rowPitch) * blocksY);
        if (!a.memory)
        {
            t.contentHash = 0;   // retried next frame
            break;
        }
        ConvertRows(blocksY, uint64_t(rowPitch) * blocksY, [&](uint32_t y) {
            ConvertRow(a.memory + uint64_t(y) * rowPitch, src, blocksX, y, fetch.Tiled(), pitchBlocks, baseOffsetX, baseOffsetY,
                bytesPerBlock, fetch.Endian());
        });
        cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(t.texture.get(), 0, layer),
            RenderTextureCopyLocation::PlacedFootprint(uploadBuffer, hostFormat, width, height, 1, rowPitch * bw / bytesPerBlock, a.offset));
    }
    for (size_t level = 0; level < mips.size() && t.contentHash; level++)
    {
        const MipLevel& m = mips[level];
        const uint32_t mbx = (m.width + bw - 1) / bw, mby = (m.height + bh - 1) / bh;
        const uint32_t mRowBytes = mbx * bytesPerBlock, mRowPitch = (mRowBytes + 255) & ~255u;
        auto a = allocate(uint64_t(mRowPitch) * mby);
        if (!a.memory)
        {
            t.contentHash = 0;
            break;
        }
        ConvertRows(mby, uint64_t(mRowPitch) * mby, [&](uint32_t y) {
            ConvertRow(a.memory + uint64_t(y) * mRowPitch, m.src, mbx, y, true, m.pitchBlocks, m.offsetX, m.offsetY, bytesPerBlock,
                fetch.Endian());
        });
        cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(t.texture.get(), uint32_t(level + 1), 0),
            RenderTextureCopyLocation::PlacedFootprint(uploadBuffer, hostFormat, m.width, m.height, 1, mRowPitch * bw / bytesPerBlock, a.offset));
    }
    cl.barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(t.texture.get(), RenderTextureLayout::SHADER_READ));
    if (staging)
    {
        staging->unmap();
        s_r.retiredStaging.push_back(std::move(staging));
    }
    return t.descriptor;
}

// --- Render surfaces --------------------------------------------------------------------------------------

// EDRAM colour formats (RB_COLOR_INFO bits 16-19) -> host formats.
RenderFormat ColorTargetFormat(uint32_t colorFormat)
{
    switch (colorFormat)
    {
    case 0: case 1: return RenderFormat::B8G8R8A8_UNORM;     // 8_8_8_8 (and its gamma variant)
    case 4: case 6: return RenderFormat::R16G16_FLOAT;       // 16_16 (fixed -32..32), 16_16_FLOAT
    case 14: return RenderFormat::R32_FLOAT;                 // 32_FLOAT
    case 15: return RenderFormat::R32G32_FLOAT;              // 32_32_FLOAT
    default: return RenderFormat::R16G16B16A16_FLOAT;        // 2_10_10_10 variants, 16_16_16_16(_FLOAT)
    }
}

void Transition(RenderCommandList& cl, HostImage& image, RenderTextureLayout layout)
{
    if (image.layout == layout)
        return;
    cl.barriers(RenderBarrierStage::GRAPHICS | RenderBarrierStage::COPY, RenderTextureBarrier(image.texture.get(), layout));
    image.layout = layout;
}

// Two images' transitions in one barrier (a copy's source and destination: one wait instead of two).
void Transition(RenderCommandList& cl, HostImage& a, RenderTextureLayout layoutA, HostImage& b, RenderTextureLayout layoutB)
{
    RenderTextureBarrier barriers[2];
    uint32_t count = 0;
    if (a.layout != layoutA)
        barriers[count++] = RenderTextureBarrier(a.texture.get(), layoutA);
    if (b.layout != layoutB && &b != &a)
        barriers[count++] = RenderTextureBarrier(b.texture.get(), layoutB);
    if (count)
        cl.barriers(RenderBarrierStage::GRAPHICS | RenderBarrierStage::COPY, barriers, count);
    a.layout = layoutA;
    b.layout = layoutB;
}

void RetireImage(HostImage& image)
{
    for (auto it = s_r.framebuffers.begin(); it != s_r.framebuffers.end();)
    {
        if (it->first.first == image.texture.get() || it->first.second == image.texture.get())
        {
            s_r.retiredFramebuffers.push_back(std::move(it->second));
            it = s_r.framebuffers.erase(it);
        }
        else
        {
            ++it;
        }
    }
    for (auto it = s_r.depthResolveFramebuffers.begin(); it != s_r.depthResolveFramebuffers.end();)
    {
        if (it->first.first == image.texture.get() || it->first.second == image.texture.get())
        {
            s_r.retiredFramebuffers.push_back(std::move(it->second));
            it = s_r.depthResolveFramebuffers.erase(it);
        }
        else
        {
            ++it;
        }
    }
    for (auto it = s_r.motionFramebuffers.begin(); it != s_r.motionFramebuffers.end();)
    {
        if (it->first.first == image.texture.get() || it->first.second == image.texture.get() || &image == &s_r.motion)
        {
            s_r.retiredFramebuffers.push_back(std::move(it->second));
            it = s_r.motionFramebuffers.erase(it);
        }
        else
        {
            ++it;
        }
    }
    s_r.retiredViews.push_back(std::move(image.view));
    s_r.retiredViews.push_back(std::move(image.msDepthView));
    s_r.retiredViews.push_back(std::move(image.stencilView));
    for (auto& [swizzle, entry] : image.swizzled)
        s_r.retiredViews.push_back(std::move(entry.first));
    s_r.retiredTextures.push_back(std::move(image.texture));
    // New descriptors for the image made next, as for uploaded textures: draws recorded earlier
    // in the frame sample the old image through the old ones. Rewritten in place, they would have those draws read
    // the new, still empty image when the frame runs (the exposure's feedback would read 0).
    for (auto& [swizzle, entry] : image.swizzled)
        RetireDescriptor(entry.second);
    image.swizzled.clear();
    RetireDescriptor(image.descriptor);
    RetireDescriptor(image.msDepthDescriptor);
    RetireDescriptor(image.stencilDescriptor);
    image.descriptor = 0;
    image.msDepthDescriptor = 0;
    image.stencilDescriptor = 0;
}

// The guest textures' images over their budget: the ones unused longest (and not for the last 600 frames) are let go,
// oldest first, until the images are under 90% of it. Uploaded again if used again. The budget: AC6_TEXTURE_BUDGET_MB,
// else a quarter of the GPU's video memory, at least 1 GB (a mission's textures are some 270 MB, so only long sessions
// across many missions on small GPUs reach it).
void EvictTextures()
{
    static const uint64_t budget = [] {
        if (const char* v = std::getenv("AC6_TEXTURE_BUDGET_MB"); v && *v)
            return uint64_t(std::max(16, std::atoi(v))) << 20;
        return std::max<uint64_t>(1ull << 30, g_video.device->getDescription().dedicatedVideoMemory / 4);
    }();
    constexpr uint64_t kMinUnusedFrames = 600;
    if (s_r.frame % 60 != 0 || s_r.textureBytes <= budget)
        return;
    std::vector<std::pair<uint64_t, uint64_t>> old;   // last frame, key
    for (const auto& [key, t] : s_r.textures)
        if (t.texture && t.lastFrame + kMinUnusedFrames < s_r.frame)
            old.push_back({ t.lastFrame, key });
    std::sort(old.begin(), old.end());
    const uint64_t before = s_r.textureBytes;
    size_t evicted = 0;
    for (const auto& [last, key] : old)
    {
        if (s_r.textureBytes <= budget / 10 * 9)
            break;
        auto it = s_r.textures.find(key);
        HostTexture& t = it->second;
        s_r.retiredViews.push_back(std::move(t.view));
        s_r.retiredTextures.push_back(std::move(t.texture));
        RetireDescriptor(t.descriptor);
        s_r.textureBytes -= t.bytes;
        s_r.textures.erase(it);
        evicted++;
    }
    static uint32_t passes = 0;
    if (evicted && (passes++ < 10 || passes % 100 == 0))   // (a budget under a mission's textures evicts every second)
        log::Info("Textures: {} unused ones let go, {} MB -> {} MB (budget {} MB)", evicted, before >> 20, s_r.textureBytes >> 20, budget >> 20);
}

// A rendered image's size in guest pixels (render surfaces: before the 4x MSAA promotion): ToHost's exact inverse
// (its rounding is under half a host pixel, less than half a guest pixel).
uint32_t GuestWidth(const HostImage& image)
{
    return uint32_t(std::lround(double(image.width) / (double(image.scale) * (image.resX > 0.0f ? image.resX : s_r.resX))));
}

uint32_t GuestHeight(const HostImage& image)
{
    return uint32_t(std::lround(double(image.height) / (double(image.scale) * (image.resY > 0.0f ? image.resY : s_r.resY))));
}

// AC6_MS_DEPTH_RESOLVE=0: depth resolves from a multisampled surface go through its single-sampled copy.
bool MsDepthResolve()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_MS_DEPTH_RESOLVE"); return !(v && *v == '0'); }();
    return enabled;
}

// AC6_RESOLVE_SWAP=0: full-size colour resolves are copied. =2 (a check): swapped, and
// always copied back.
int ResolveSwapMode()
{
    static const int mode = [] { const char* v = std::getenv("AC6_RESOLVE_SWAP"); return v && *v ? std::atoi(v) : 1; }();
    return mode;
}

bool ResolveSwaps()
{
    return ResolveSwapMode() != 0;
}

void UnlinkStale(HostImage& surface)
{
    if (surface.staleHolder)
        surface.staleHolder->staleOf = nullptr;
    surface.staleHolder = nullptr;
}

// A swapped surface's contents back from its resolve target's image: the copy its resolve skipped,
// before anything reads the surface or draws into it without covering it whole.
void RestoreStale(RenderCommandList& cl, HostImage& surface)
{
    HostImage* holder = surface.staleHolder;
    if (!holder)
        return;
    UnlinkStale(surface);
    if (!holder->texture || !surface.texture)
        return;
    MarkPass([&] { return "resolve swap restore " + PassImage(&surface); });
    static const uint64_t traceFrame = [] { const char* v = std::getenv("AC6_DRAW_TRACE"); return v ? std::stoull(v) : UINT64_MAX; }();
    if (s_r.frame == traceFrame)
        log::Info("Resolve swap restore: {}", PassImage(&surface));
    Transition(cl, *holder, RenderTextureLayout::COPY_SOURCE);
    Transition(cl, surface, RenderTextureLayout::COPY_DEST);
    const RenderBox box(0, 0, int32_t(surface.width), int32_t(surface.height));
    cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(surface.texture.get()),
        RenderTextureCopyLocation::Subresource(holder->texture.get()), 0, 0, 0, &box);
    // Out of the copy layout: a transition to the same layout records no barrier, and a later copy into the surface
    // (a replay's snapshot) would then follow this one unsynchronised.
    Transition(cl, *holder, RenderTextureLayout::SHADER_READ);
    Transition(cl, surface, RenderTextureLayout::COLOR_WRITE);
    s_stats.resolveRestores++;
}

// (Re)creates `image` as a colour or depth target of at least the given size (a remade one with new descriptors:
// RetireImage).
// `exact`: that size, also smaller than it was (FSR upscaling's images at the render scale).
void EnsureImage(HostImage& image, uint32_t width, uint32_t height, RenderFormat format, bool depth, uint32_t samples = 1, bool storage = false,
    bool exact = false)
{
    // At a low internal resolution a small guest image can come to under a host pixel.
    width = std::max(width, 1u);
    height = std::max(height, 1u);
    // Past the GPU's image size limit (the render scale is kept under it for the game's known surfaces at start):
    // clamped, drawn and read clipped, instead of an image the GPU can't make.
    if (const uint32_t limit = uint32_t(std::min<uint64_t>(g_video.device->getCapabilities().maxTextureSize, UINT32_MAX)); limit && (width > limit || height > limit))
    {
        static uint32_t warned = 0;
        if (warned++ < 4)
            log::Warn("A {}x{} image is past the GPU's limit ({}): clamped, drawn clipped", width, height, limit);
        width = std::min(width, limit);
        height = std::min(height, limit);
    }
    if (image.texture && image.format == format && image.samples == samples && image.width >= width && image.height >= height &&
        (!exact || (image.width == width && image.height == height)))
        return;
    if (image.texture)
    {
        if (!exact)
        {
            width = std::max(width, image.width);
            height = std::max(height, image.height);
        }
        // A swapped surface made again starts over, as any remade image: ResolveImpl brings a
        // surface back before its holder can be remade.
        UnlinkStale(image);
        if (image.staleOf)
        {
            log::Warn("Resolve swap: a resolve target holding a surface's contents was made again");
            UnlinkStale(*image.staleOf);
        }
        RetireImage(image);
    }
    const RenderMultisampling multisampling(samples);
    image.resX = s_r.resX;
    image.resY = s_r.resY;
    if (depth)
        image.texture = MakeTexture(RenderTextureDesc::DepthTarget(width, height, format, multisampling), "a depth surface");
    else
    {
        // `storage`: also a storage image (FSR's output).
        RenderTextureDesc desc = RenderTextureDesc::ColorTarget(width, height, format, multisampling);
        if (storage)
            desc.flags |= RenderTextureFlag::STORAGE;
        image.texture = MakeTexture(desc, "a colour surface");
    }
    // Multisampled images are only drawn to: shaders read their single-sampled copy (Readable).
    if (samples == 1)
    {
        image.view = image.texture->createTextureView(RenderTextureViewDesc::Texture2D(format));
        if (!image.descriptor)
            image.descriptor = AllocateDescriptor();
        SetDescriptor(image.descriptor, image.texture.get(), image.view.get());
    }
    else if (depth && s_r.depthCopyMsPipeline)
    {
        // Depth resolves read the samples themselves: the depth here, the stencil below.
        image.msDepthView = image.texture->createTextureView(RenderTextureViewDesc::Texture2D(format));
        if (!image.msDepthDescriptor)
            image.msDepthDescriptor = AllocateDescriptor();
        SetDescriptor(image.msDepthDescriptor, image.texture.get(), image.msDepthView.get());
    }
    // The stencil (an R8_UINT view of the depth/stencil image) for one-pass depth resolves.
    if (depth && (samples > 1 ? bool(s_r.depthCopyMsPipeline) : bool(s_r.depthCopySsPipeline)))
    {
        image.stencilView = image.texture->createTextureView(RenderTextureViewDesc::Texture2D(RenderFormat::R8_UINT));
        if (!image.stencilDescriptor)
            image.stencilDescriptor = AllocateDescriptor();
        SetDescriptor(image.stencilDescriptor, image.texture.get(), image.stencilView.get());
    }
    image.samples = samples;
    image.singleCurrent = false;
    image.format = format;
    image.width = width;
    image.height = height;
    image.depth = depth;
    image.fresh = true;
    image.layout = RenderTextureLayout::UNKNOWN;
}

// The R8 image beside a depth resolve that holds its stencil (read as the resolve's 8_8_8_8 texels: the cockpit's
// depth that the effects pass and the motion blur rebuild from it). Made and scaled as the depth resolve is: with FSR
// upscaling the scene's resolves are at the render scale, into images of exactly that size. Kept at the frame's size,
// the render-size stencil went into its top-left corner and was read across the whole image: the cockpit drawn small in
// the corner of the effects pass's depth, the clouds cut along that shape.
void EnsureResolvedStencil(HostImage& image, uint32_t width, uint32_t height, bool worldResolve, bool nativeResolve)
{
    EnsureImage(image, width, height, RenderFormat::R8_UNORM, false, 1, false, worldResolve);
    if (s_r.upscaling || nativeResolve)
    {
        image.resX = s_r.resX;   // the scale its contents are at (sampled at it)
        image.resY = s_r.resY;
    }
}

// The mip chain of resolve target `image`, (re)made for a width x height texture of
// `levels` levels in the image's format, level 0 copied from the image when new.
HostImage& EnsureMipChain(RenderCommandList& cl, HostImage& image, uint32_t width, uint32_t height, uint32_t levels)
{
    if (!image.mipChain)
        image.mipChain = std::make_unique<HostImage>();
    HostImage& chain = *image.mipChain;
    // A chain at a render scale under 1 is smaller than the guest's: no more levels than its size has.
    levels = std::min(levels, uint32_t(std::bit_width(std::max({ width, height, 1u }))));
    if (chain.texture && chain.format == image.format && chain.width == width && chain.height == height && chain.mipLevels == levels)
        return chain;
    if (chain.texture)
        RetireImage(chain);   // its views are made again on next use, with new descriptors
    chain.texture = MakeTexture(RenderTextureDesc::Texture2D(width, height, levels, image.format), "a mip chain");
    chain.format = image.format;
    chain.width = width;
    chain.height = height;
    chain.mipLevels = levels;
    chain.resX = image.resX;   // at the level-0 image's scale (the tree impostors' 1)
    chain.resY = image.resY;
    chain.layout = RenderTextureLayout::UNKNOWN;
    image.mipWritten = 0;
    if (image.width >= width && image.height >= height)
    {
        Transition(cl, image, RenderTextureLayout::COPY_SOURCE);
        Transition(cl, chain, RenderTextureLayout::COPY_DEST);
        const RenderBox box(0, 0, int32_t(width), int32_t(height));
        cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(chain.texture.get()),
            RenderTextureCopyLocation::Subresource(image.texture.get()), 0, 0, 0, &box);
        image.mipWritten = 1;
    }
    Transition(cl, chain, RenderTextureLayout::SHADER_READ);
    return chain;
}

// The MSAA override: AC6_MSAA=<samples> draws the game's multisampled surfaces (the 2x scene
// colour and depth, 4x ones while tiling) with that many samples: 0 or 1 single-sampled, 2, 4, 8, 16.
// Unset or `console`, the game's own count. Clamped to the highest count the GPU supports for colour and depth
// targets. 4x surfaces outside tiling (the effects buffer) are drawn at twice the size either way: their MSAA is the
// mixed-resolution trick, not anti-aliasing. Returns 0 for the game's own count. A GPU without 2x (Vulkan only
// guarantees 1 and 4: llvmpipe) draws the 2x surfaces with 4 samples, as asking it for 2 leaves them black.
uint32_t MsaaOverride()
{
    static const uint32_t samples = [] {
        const uint32_t supported = uint32_t(g_video.device->getSampleCountsSupported(kDepthFormat)) &
            uint32_t(g_video.device->getSampleCountsSupported(RenderFormat::B8G8R8A8_UNORM)) &
            uint32_t(g_video.device->getSampleCountsSupported(RenderFormat::R16G16B16A16_FLOAT));
        const char* v = std::getenv("AC6_MSAA");
        if (!v || !*v || std::strcmp(v, "console") == 0)
        {
            if (supported & 2)
                return 0u;
            const uint32_t fallback = (supported & 4) ? 4u : 1u;
            log::Info("MSAA: the GPU has no 2x (supports mask {:X}): {} sample(s) for the game's 2x surfaces", supported, fallback);
            return fallback;
        }
        uint32_t want = uint32_t(std::max(1, std::atoi(v)));
        want = want >= 16 ? 16 : want >= 8 ? 8 : want >= 4 ? 4 : want >= 2 ? 2 : 1;
        uint32_t samples = want;
        while (samples > 1 && !(supported & samples))
            samples >>= 1;
        log::Info("MSAA: {} sample(s) for the game's multisampled surfaces (asked {}, GPU supports mask {:X})", samples, want, supported);
        return samples;
    }();
    return samples;
}

// AC6_SHARE_SURFACES=0: every guest surface has its own image, filled from the others over its EDRAM when bound.
bool ShareSurfaces()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_SHARE_SURFACES"); return !(v && *v == '0') && !EdramSwizzle(); }();
    return enabled;
}

// The surface whose image `surface` uses.
uint32_t SurfaceOwner(uint32_t surface)
{
    auto it = s_r.surfaceOwners.find(surface);
    return it != s_r.surfaceOwners.end() ? it->second : surface;
}

HostImage* GetSurface(uint32_t surface, bool depth)
{
    // Render surfaces are made at the rendered frame's scale, also while a UI draw has switched the scale to the UI's
    // (made at the UI's, the frame's target would grow to the UI's size after its first UI draw).
    // The scene's own surfaces with FSR upscaling at the render scale, exactly.
    // The tree impostors' surface at the console's size.
    const bool world = WorldSurface(surface), native = !world && NativeSurface(surface);
    struct SceneScale
    {
        float x = s_r.resX, y = s_r.resY;
        SceneScale(bool world, bool native)
        {
            s_r.resX = world ? s_r.worldResX : native ? 1.0f : s_r.sceneResX;
            s_r.resY = world ? s_r.worldResY : native ? 1.0f : s_r.sceneResY;
        }
        ~SceneScale() { s_r.resX = x, s_r.resY = y; }
    } sceneScale(world, native);
    if (!surface)
        return nullptr;
    const uint32_t size = U32(surface + kSurfaceSize);
    uint32_t width = (size >> 18) + 1, height = ((size & 0xFFFF) >> 3) + 1;
    if (s_r.tiling)
    {
        width = std::max(width, s_r.tilingWidth);
        height = std::max(height, s_r.tilingHeight);
    }
    // 4x MSAA surfaces (RB_SURFACE_INFO bits 16-17 = 2) are drawn at twice the size, one host pixel per
    // sample: a 640x360 4x surface holds exactly the EDRAM of a 1280x720 one, and the game relies on that
    // (the half-resolution effects buffer is seen sample for sample by the full-size surface over it).
    const uint32_t msaa = (U32(surface + kSurfaceInfo) >> 16) & 3;
    const uint32_t scale = !s_r.tiling && msaa == 2 ? 2 : 1;
    // At the internal resolution.
    width = uint32_t(ToHostX(double(width) * scale));
    height = uint32_t(ToHostY(double(height) * scale));
    // The other MSAA surfaces (2x, and 4x while tiling) are drawn multisampled.
    const uint32_t samples = scale == 1 && (msaa == 1 || msaa == 2) ? (MsaaOverride() ? MsaaOverride() : 1u << msaa) : 1u;
    RenderFormat format = depth ? kDepthFormat : ColorTargetFormat((U32(surface + kSurfaceFormatInfo) >> 16) & 0xF);
    // HDR from the scene: from the frame's scene composite on, the frame-sized
    // canvas (8_8_8_8, single-sampled on the console too: not the world's 2x surface) is a float twin, so the composite's
    // decoded scene x exposure + bloom keeps its values above 1 through the post chain. Its resolves follow its format.
    bool hdrCanvas = !depth && s_r.hdrCanvasFrame == s_r.frame && HdrSceneOn() && format == RenderFormat::B8G8R8A8_UNORM &&
        msaa == 0 && samples == 1 && width == s_r.sceneWidth && height == s_r.sceneHeight;
    // The game's lighting in float: the frame's colour surfaces and their 1/2, 1/4, 1/8 for the whole
    // frame (the scene, the canvas, the effects surface, the downsample, the bloom), twins as the canvas's.
    if (!hdrCanvas && !depth && format == RenderFormat::B8G8R8A8_UNORM && SceneFloatOn())
        for (uint32_t k = 1; k <= 8 && !hdrCanvas; k *= 2)
            hdrCanvas = std::abs(int32_t(width * k) - int32_t(s_r.sceneWidth)) <= int32_t(k) &&
                std::abs(int32_t(height * k) - int32_t(s_r.sceneHeight)) <= int32_t(k);
    // The world's surfaces with FSR upscaling: the frame's size at the render scale, float too. The test above compares
    // with the frame's size at the frame's scale, which misses them, and the shaders write the game's code range
    // unclamped either way: left 8-bit, the world would clip at white while everything after it is float.
    if (!hdrCanvas && !depth && world && format == RenderFormat::B8G8R8A8_UNORM && SceneFloatOn() && s_r.sceneResX > 0.0f && s_r.sceneResY > 0.0f)
        hdrCanvas = std::abs(int32_t(width) - int32_t(std::lround(double(s_r.sceneWidth) * s_r.resX / s_r.sceneResX))) <= 1 &&
            std::abs(int32_t(height) - int32_t(std::lround(double(s_r.sceneHeight) * s_r.resY / s_r.sceneResY))) <= 1;
    if (hdrCanvas)
        format = RenderFormat::R16G16B16A16_FLOAT;
    // Surfaces over the same EDRAM whose host images would match (format, samples, size: e.g. a 4x MSAA 640x360
    // surface promoted over a 1280x720 one) share one image, as they share the EDRAM on the
    // console: binding one after the other has written copies nothing (copying the whole image across costs 0.6 ms a
    // frame at 4K). Images that differ still alias through copies and resampling (SyncAliasedDepth/Color).
    const uint32_t edramBase = U32(surface + kSurfaceFormatInfo) & 0xFFF;
    uint32_t owner = surface;
    if (ShareSurfaces())
    {
        // The guest's EDRAM layout too: 1x and promoted 4x surfaces hold a pixel's samples alike, a 2x
        // one (or a 4x one drawn multisampled while tiling) doesn't. With AC6_MSAA=0 the hangar's 2x scene surface
        // would otherwise share the 1x surfaces' image: the game fills its samples through a 1280x1440 1x surface,
        // which isn't emulated, so the scene relies on its own image keeping its last frame, and a shared image would
        // hold the final composite instead (the post effects would feed back into the hangar).
        const uint64_t layout = msaa == 0 || scale == 2 ? 0 : msaa;
        const uint64_t key = uint64_t(edramBase) | uint64_t(depth) << 12 | uint64_t(format) << 13 | uint64_t(samples) << 21 |
            uint64_t(width) << 26 | uint64_t(height) << 42 | layout << 58;
        auto [it, added] = s_r.sharedSurfaces.try_emplace(key, surface);
        if (!added && it->second != surface)
        {
            // The owner's image changed since (its guest surface was set up again): this surface takes over.
            auto& images = hdrCanvas ? s_r.hdrSurfaces : s_r.surfaces;
            auto o = images.find(it->second);
            if (o == images.end() || !o->second.texture || o->second.format != format || o->second.samples != samples ||
                o->second.width != width || o->second.height != height || o->second.depth != depth || o->second.edramBase != edramBase)
                it->second = surface;
        }
        owner = it->second;
        auto [was, first] = s_r.surfaceOwners.try_emplace(surface, owner);
        if (was->second != owner || (first && owner != surface))
        {
            was->second = owner;
            if (owner != surface)
                log::Info("Render surface {:08X} shares {:08X}'s image (EDRAM base {})", surface, owner, edramBase);
        }
    }
    HostImage& image = hdrCanvas ? s_r.hdrSurfaces[owner] : s_r.surfaces[owner];
    const bool created = !image.texture || image.format != format || image.samples != samples || image.width < width || image.height < height ||
        (world && (image.width != width || image.height != height));
    EnsureImage(image, width, height, format, depth, samples, false, world);
    image.edramBase = edramBase;
    image.scale = scale;
    image.guestFormat = (U32(surface + kSurfaceFormatInfo) >> 16) & 0xF;
    image.bgraOrder = hdrCanvas;
    if (created)
        log::Info("Render surface {:08X}: {} {}x{} format {} (guest {}, MSAA {}, {} sample(s), EDRAM base {})", surface, depth ? "depth" : "colour",
            image.width, image.height, int(format), (U32(surface + kSurfaceFormatInfo) >> 16) & 0xF, msaa, samples, image.edramBase);
    return &image;
}

RenderFramebuffer* GetFramebuffer(HostImage* color, HostImage* depth)
{
    const auto key = std::make_pair<const RenderTexture*, const RenderTexture*>(color ? color->texture.get() : nullptr,
        depth ? depth->texture.get() : nullptr);
    auto& slot = s_r.framebuffers[key];
    if (!slot)
    {
        const RenderTexture* attachment = key.first;
        RenderFramebufferDesc desc;
        desc.colorAttachments = color ? &attachment : nullptr;
        desc.colorAttachmentsCount = color ? 1 : 0;
        desc.depthAttachment = key.second;
        slot = g_video.device->createFramebuffer(desc);
    }
    return slot.get();
}

// The scene surface with TAA's motion image as its second colour target.
RenderFramebuffer* GetMotionFramebuffer(HostImage* color, HostImage* depth)
{
    const auto key = std::make_pair<const RenderTexture*, const RenderTexture*>(color->texture.get(), depth ? depth->texture.get() : nullptr);
    auto& slot = s_r.motionFramebuffers[key];
    if (!slot)
    {
        const RenderTexture* attachments[2] = { key.first, s_r.motion.texture.get() };
        RenderFramebufferDesc desc;
        desc.colorAttachments = attachments;
        desc.colorAttachmentsCount = 2;
        desc.depthAttachment = key.second;
        slot = g_video.device->createFramebuffer(desc);
    }
    return slot.get();
}

struct Targets
{
    HostImage* color = nullptr;
    HostImage* depth = nullptr;
    HostImage* motion = nullptr;   // TAA's motion image, the second colour target
    uint32_t width = 0, height = 0;
    uint32_t scale = 1;     // host pixels per guest pixel
    uint32_t samples = 1;   // host samples per pixel
};

// TAA's motion vectors: with TAA and the SPIR-V backend's shaders. AC6_TAA_MOTION=0: the camera's
// reprojection only.
bool MotionVectors()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_TAA_MOTION"); return !(v && *v == '0') && !HlslBackend(); }();
    return enabled && ((s_r.taaPipeline && TaaAtStart()) || FsrRunning());   // TAA's at start only
}
// What a motion pixel holds where no draw gave it its motion (spirv_recompiler.cpp kMotionUnknown).
constexpr float kMotionUnknown = 30000.0f;

// Graphics state bound through the command list once and kept: every pass uses the
// same pipeline layout and the same four bindless sets, which stay bound across pipelines and render passes,
// so they are bound once per command list. Pipeline, viewport and scissor are skipped when unchanged.
// Every bind goes through these.
struct BoundState
{
    uint64_t epoch = UINT64_MAX;
    bool sets = false, hasViewport = false, hasScissor = false;
    const void* pipeline = nullptr;   // the RenderPipeline, or the PendingPipeline bound before it was built
    bool hasConstants = false;
    uint32_t constantOffsets[3] = {};  // set 4's
    RenderViewport viewport{};
    RenderRect scissor{};
} s_bound;

void RefreshBound()
{
    if (s_bound.epoch != g_video.recordingEpoch)
        s_bound = BoundState{ g_video.recordingEpoch };
}

void BindCommon(RenderCommandList& cl)
{
    RefreshBound();
    cl.setGraphicsPipelineLayout(s_r.layout.get());   // plume-side only: no Vulkan call
    if (s_bound.sets)
        return;
    cl.setGraphicsDescriptorSet(s_r.textureSet.get(), 0);
    cl.setGraphicsDescriptorSet(s_r.textureSet.get(), 1);
    cl.setGraphicsDescriptorSet(s_r.textureSet.get(), 2);
    cl.setGraphicsDescriptorSet(s_r.samplerSet.get(), 3);
    s_bound.sets = true;
}

void SetPipeline(RenderCommandList& cl, const RenderPipeline* pipeline)
{
    RefreshBound();
    if (s_bound.pipeline == pipeline)
        return;
    cl.setPipeline(pipeline);
    s_bound.pipeline = pipeline;
}

// A draw's pipeline (GetPipeline): bound now if it is built, else, on the deferred list, bound by the replay
// once it is. Any other list waits for it here.
void SetPipeline(RenderCommandList& cl, const PendingPipeline* pending)
{
    if (pending->ready.load(std::memory_order_acquire))
        return SetPipeline(cl, pending->pipeline.get());
    RefreshBound();
    if (s_bound.pipeline == pending)
        return;
    if (&cl == static_cast<RenderCommandList*>(g_video.deferred.get()))
    {
        g_video.deferred->setPendingPipeline(pending);
        s_bound.pipeline = pending;
        return;
    }
    pending->Wait();
    SetPipeline(cl, pending->pipeline.get());
}

void SetViewport(RenderCommandList& cl, const RenderViewport& viewport)
{
    RefreshBound();
    if (s_bound.hasViewport && std::memcmp(&s_bound.viewport, &viewport, sizeof(viewport)) == 0)
        return;
    cl.setViewports(viewport);
    s_bound.viewport = viewport;
    s_bound.hasViewport = true;
}

void SetScissor(RenderCommandList& cl, const RenderRect& scissor)
{
    RefreshBound();
    if (s_bound.hasScissor && std::memcmp(&s_bound.scissor, &scissor, sizeof(scissor)) == 0)
        return;
    cl.setScissors(scissor);
    s_bound.scissor = scissor;
    s_bound.hasScissor = true;
}

bool SwizzleTrace()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_TRACE_SWIZZLE"); return v && *v == '1'; }();
    return enabled;
}

// A rectangle the next alias passes are limited to (the late pass's partial loads).
const RenderRect* s_aliasScissor = nullptr;

// Draws `source` into `target` (bound as its framebuffer) through `pipeline` over the whole target, or over
// s_aliasScissor when it is set.
void DrawAliasPass(RenderCommandList& cl, HostImage& source, HostImage& target, RenderPipeline* pipeline)
{
    if (SwizzleTrace() && (pipeline == s_r.depthSwizzlePipeline.get() || s_r.colorSwizzlePipelines[target.format].get() == pipeline))
        target.swizzleTrace = true;
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(target.width), float(target.height)));
    SetScissor(cl, s_aliasScissor ? *s_aliasScissor : RenderRect(0, 0, int32_t(target.width), int32_t(target.height)));
    SetPipeline(cl, pipeline);
    BindCommon(cl);
    // offsetX: the internal resolution's scale for the EDRAM swizzles (their sample grid is the guest's).
    const int32_t push[4] = { int32_t(source.descriptor), 0, s_r.resX != 1.0f || s_r.resY != 1.0f ? std::bit_cast<int32_t>(s_r.resX) : 0,
        s_r.resX != 1.0f || s_r.resY != 1.0f ? std::bit_cast<int32_t>(s_r.resY) : 0 };
    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    cl.drawInstanced(3, 1, 0, 0);
}

// The stencil of a single-sampled depth image (a rectangle of it) into an R8 image: buffer copies of the
// stencil aspect are tightly packed bytes.
void CopyStencil(RenderCommandList& cl, HostImage& src, int32_t sx, int32_t sy, int32_t w, int32_t h, HostImage& dst, int32_t dx, int32_t dy)
{
    cl.setFramebuffer(nullptr);
    const uint64_t bytes = (uint64_t(w) * h + 255) & ~uint64_t(255);
    if (s_r.stencilStagingSize < bytes)
    {
        if (s_r.stencilStaging)
            s_r.retiredStaging.push_back(std::move(s_r.stencilStaging));
        s_r.stencilStaging = g_video.device->createBuffer(RenderBufferDesc::DefaultBuffer(bytes));
        s_r.stencilStagingSize = bytes;
        // Declared as written by copies from the start, so the barrier after its first copy waits for it.
        const RenderBufferBarrier toWrite(s_r.stencilStaging.get(), RenderBufferAccess::WRITE);
        cl.barriers(RenderBarrierStage::COPY, &toWrite, 1, nullptr, 0);
    }
    Transition(cl, src, RenderTextureLayout::COPY_SOURCE);
    const RenderBox box(sx, sy, sx + w, sy + h);
    cl.copyTextureRegion(RenderTextureCopyLocation::PlacedFootprint(s_r.stencilStaging.get(), RenderFormat::R8_UINT, uint32_t(w), uint32_t(h), 1,
        uint32_t(w)), RenderTextureCopyLocation::Subresource(src.texture.get()), 0, 0, 0, &box);
    const RenderBufferBarrier toRead(s_r.stencilStaging.get(), RenderBufferAccess::READ);
    cl.barriers(RenderBarrierStage::COPY, &toRead, 1, nullptr, 0);
    Transition(cl, dst, RenderTextureLayout::COPY_DEST);
    cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(dst.texture.get()),
        RenderTextureCopyLocation::PlacedFootprint(s_r.stencilStaging.get(), RenderFormat::R8_UNORM, uint32_t(w), uint32_t(h), 1, uint32_t(w)),
        uint32_t(dx), uint32_t(dy), 0);
    const RenderBufferBarrier toWrite(s_r.stencilStaging.get(), RenderBufferAccess::WRITE);
    cl.barriers(RenderBarrierStage::COPY, &toWrite, 1, nullptr, 0);
    Transition(cl, dst, RenderTextureLayout::SHADER_READ);
}

// --- Multisampled surfaces ----------------------------------------------------------
// 2x MSAA surfaces are drawn multisampled. Everything else sees them through a single-sampled copy, as the
// console's resolves do: colour averaged over the samples, depth and stencil from sample 0.

RenderFramebuffer* GetDepthResolveFramebuffer(HostImage& image, HostImage& single)
{
    // Keyed like other framebuffers (a depth image is never a colour attachment), so RetireImage drops it.
    const auto key = std::make_pair<const RenderTexture*, const RenderTexture*>(single.texture.get(), image.texture.get());
    auto& slot = s_r.framebuffers[key];
    if (!slot)
    {
        RenderFramebufferDesc desc;
        desc.depthAttachment = image.texture.get();
        desc.depthResolveAttachment = single.texture.get();
        slot = g_video.device->createFramebuffer(desc);
    }
    return slot.get();
}

// The image to read a surface through: the surface itself, or its single-sampled copy brought up to date.
HostImage& Readable(RenderCommandList& cl, HostImage& image)
{
    if (image.staleHolder)
        RestoreStale(cl, image);
    if (image.samples == 1)
        return image;
    if (!image.single)
        image.single = std::make_unique<HostImage>();
    HostImage& single = *image.single;
    EnsureImage(single, image.width, image.height, image.format, image.depth);
    single.resX = image.resX;   // the image's scale, whatever the draw's
    single.resY = image.resY;
    single.edramBase = image.edramBase;
    single.scale = image.scale;
    if (image.singleCurrent && !single.fresh)
        return single;
    MarkPass([&] { return "MSAA resolve " + PassImage(&image); });
    if (image.fresh)
    {
        // Never drawn: reads as cleared, like the surface will be when first bound.
        if (image.depth)
        {
            Transition(cl, single, RenderTextureLayout::DEPTH_WRITE);
            cl.setFramebuffer(GetFramebuffer(nullptr, &single));
            cl.clearDepthStencil(true, true, 0.0f, 0);
        }
        else
        {
            Transition(cl, single, RenderTextureLayout::COLOR_WRITE);
            cl.setFramebuffer(GetFramebuffer(&single, nullptr));
            cl.clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
        }
    }
    else if (image.depth)
    {
        Transition(cl, image, RenderTextureLayout::DEPTH_WRITE);
        Transition(cl, single, RenderTextureLayout::DEPTH_WRITE);
        cl.setFramebuffer(GetDepthResolveFramebuffer(image, single));
        cl.runRenderPass();
    }
    else
    {
        Transition(cl, image, RenderTextureLayout::RESOLVE_SOURCE);
        Transition(cl, single, RenderTextureLayout::RESOLVE_DEST);
        const RenderRect rect(0, 0, int32_t(image.width), int32_t(image.height));
        cl.resolveTextureRegion(single.texture.get(), 0, 0, image.texture.get(), &rect);
    }
    cl.setFramebuffer(nullptr);
    single.fresh = false;
    image.singleCurrent = true;
    return single;
}

// A multisampled colour image's samples as a texture (the HDR layers, read by the compose pass as
// drawn). Multisampled colour images have no view otherwise (shaders read their single-sampled copy).
uint32_t SamplesDescriptor(HostImage& image)
{
    if (!image.view)
    {
        image.view = image.texture->createTextureView(RenderTextureViewDesc::Texture2D(image.format));
        if (!image.descriptor)
            image.descriptor = AllocateDescriptor();
        SetDescriptor(image.descriptor, image.texture.get(), image.view.get());
    }
    return image.descriptor;
}

RenderPipeline* DepthLoadPipeline(uint32_t samples)
{
    auto& pipeline = s_r.depthLoadPipelines[samples];
    if (!pipeline)
    {
        RenderGraphicsPipelineDesc desc;
        desc.pipelineLayout = s_r.layout.get();
        desc.vertexShader = s_r.copyVs.get();
        desc.pixelShader = s_r.depthLoadPs.get();
        desc.renderTargetCount = 0;
        desc.depthTargetFormat = kDepthFormat;
        desc.depthEnabled = true;
        desc.depthWriteEnabled = true;
        desc.depthFunction = RenderComparisonFunction::ALWAYS;
        // The stencil is cleared here. The bit passes set it.
        RenderStencilFaceDesc face;
        face.compareFunction = RenderComparisonFunction::ALWAYS;
        face.failOp = face.passOp = face.depthFailOp = RenderStencilOp::REPLACE;
        desc.stencilEnabled = true;
        desc.stencilFrontFace = desc.stencilBackFace = face;
        desc.stencilReference = 0;
        desc.stencilWriteMask = 0xFF;
        desc.multisampling.sampleCount = samples;
        pipeline = g_video.device->createGraphicsPipeline(desc);
    }
    return pipeline.get();
}

RenderPipeline* StencilBitPipeline(uint32_t samples, uint32_t bit)
{
    auto& pipeline = s_r.stencilBitPipelines[samples * 8 + bit];
    if (!pipeline)
    {
        RenderGraphicsPipelineDesc desc;
        desc.pipelineLayout = s_r.layout.get();
        desc.vertexShader = s_r.copyVs.get();
        desc.pixelShader = s_r.stencilBitPs.get();
        desc.renderTargetCount = 0;
        desc.depthTargetFormat = kDepthFormat;
        desc.depthEnabled = false;
        desc.depthWriteEnabled = false;
        RenderStencilFaceDesc face;
        face.compareFunction = RenderComparisonFunction::ALWAYS;
        face.failOp = face.passOp = face.depthFailOp = RenderStencilOp::REPLACE;
        desc.stencilEnabled = true;
        desc.stencilFrontFace = desc.stencilBackFace = face;
        desc.stencilReference = 0xFF;
        desc.stencilWriteMask = 1u << bit;
        desc.multisampling.sampleCount = samples;
        pipeline = g_video.device->createGraphicsPipeline(desc);
    }
    return pipeline.get();
}

// AC6_DRAW_COPIES=1: colour resolves drawn instead of copied with vkCmdCopyImage, to compare the two on a
// given GPU.
bool DrawCopies()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_DRAW_COPIES"); return v && *v == '1'; }();
    return enabled;
}

// A colour resolve drawn instead of copied: the texels through a shader into the destination as a
// render target, for the formats a shader moves exactly (8-bit UNORM and 16-bit floats: no 32-bit denormals to flush).
RenderPipeline* TexelCopyPipeline(RenderFormat format)
{
    if (!s_r.texelCopyPs || (format != RenderFormat::B8G8R8A8_UNORM && format != RenderFormat::R16G16_FLOAT &&
        format != RenderFormat::R16G16B16A16_FLOAT))
        return nullptr;
    auto& pipeline = s_r.texelCopyPipelines[format];
    if (!pipeline)
    {
        RenderGraphicsPipelineDesc desc;
        desc.pipelineLayout = s_r.layout.get();
        desc.vertexShader = s_r.copyVs.get();
        desc.pixelShader = s_r.texelCopyPs.get();
        desc.renderTargetFormat[0] = format;
        desc.renderTargetBlend[0] = RenderBlendDesc::Copy();
        desc.renderTargetCount = 1;
        pipeline = g_video.device->createGraphicsPipeline(desc);
    }
    return pipeline.get();
}

RenderPipeline* ColorLoadPipeline(RenderFormat format, uint32_t samples)
{
    auto& pipeline = s_r.colorLoadPipelines[{ format, samples }];
    if (!pipeline)
    {
        RenderGraphicsPipelineDesc desc;
        desc.pipelineLayout = s_r.layout.get();
        desc.vertexShader = s_r.copyVs.get();
        desc.pixelShader = s_r.colorLoadPs.get();
        desc.renderTargetFormat[0] = format;
        desc.renderTargetBlend[0] = RenderBlendDesc::Copy();
        desc.renderTargetCount = 1;
        desc.multisampling.sampleCount = samples;
        pipeline = g_video.device->createGraphicsPipeline(desc);
    }
    return pipeline.get();
}

// A multisampled surface takes its single-sampled copy's contents in every sample (after EDRAM aliasing or a
// capture restore wrote the copy).
void LoadSamples(RenderCommandList& cl, HostImage& single, HostImage& image, bool withStencil = true);

void WriteSamples(RenderCommandList& cl, HostImage& image)
{
    MarkPass([&] { return "MSAA write samples " + PassImage(&image); });
    LoadSamples(cl, *image.single, image);
    image.fresh = false;
    image.singleCurrent = true;
}

// Every sample of the multisampled `image` from the single-sampled `single`: the colour, or the depth and, with
// `withStencil`, the stencil (without it the stencil is cleared).
void LoadSamples(RenderCommandList& cl, HostImage& single, HostImage& image, bool withStencil)
{
    if (image.depth && !withStencil)
    {
        Transition(cl, single, RenderTextureLayout::SHADER_READ);
        Transition(cl, image, RenderTextureLayout::DEPTH_WRITE);
        cl.setFramebuffer(GetFramebuffer(nullptr, &image));
        DrawAliasPass(cl, single, image, DepthLoadPipeline(image.samples));
    }
    else if (image.depth)
    {
        if (!single.stencilCopy)
            single.stencilCopy = std::make_unique<HostImage>();
        HostImage& stencil = *single.stencilCopy;
        EnsureImage(stencil, single.width, single.height, RenderFormat::R8_UNORM, false);
        stencil.fresh = false;
        CopyStencil(cl, single, 0, 0, int32_t(image.width), int32_t(image.height), stencil, 0, 0);
        Transition(cl, single, RenderTextureLayout::SHADER_READ);
        Transition(cl, image, RenderTextureLayout::DEPTH_WRITE);
        cl.setFramebuffer(GetFramebuffer(nullptr, &image));
        DrawAliasPass(cl, single, image, DepthLoadPipeline(image.samples));
        for (uint32_t bit = 0; bit < 8; bit++)
        {
            SetPipeline(cl, StencilBitPipeline(image.samples, bit));
            const uint32_t push[4] = { stencil.descriptor, 1u << bit, 0, 0 };
            cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
            cl.drawInstanced(3, 1, 0, 0);
        }
    }
    else
    {
        Transition(cl, single, RenderTextureLayout::SHADER_READ);
        Transition(cl, image, RenderTextureLayout::COLOR_WRITE);
        cl.setFramebuffer(GetFramebuffer(&image, nullptr));
        DrawAliasPass(cl, single, image, ColorLoadPipeline(image.format, image.samples));
    }
}

// The same EDRAM seen by a 1x surface and a 4x MSAA one (drawn at twice the size, so the host images have
// the same size): their texels are related by the console's sample interleave.
// The EDRAM sample interleave between a 1x surface and a promoted 4x one is only emulated with
// AC6_EDRAM_SWIZZLE=1: the game un-swizzles it itself in the passes that read it (three pixel
// shaders, EdramUnswizzleSlots in shaders.cpp), and those sample 1:1, so the alias is a plain copy. Swizzle and
// un-swizzle can't both be exact at a fractional internal resolution (a guest sample there spans one or two host
// pixels): the frame would come out in 2-pixel stair steps.
bool MsaaAlias(const HostImage& a, const HostImage& b)
{
    return EdramSwizzle() && a.scale != b.scale && a.width == b.width && a.height == b.height;
}

// EDRAM aliasing without EDRAM: depth surfaces that share an EDRAM base share their contents on the
// console (missions draw half-resolution effects against the scene depth through a 4x MSAA 640x360
// surface over the 1280x720 one, and restore passes bind other surfaces there). When a depth surface is
// bound whose base was written last by another surface, it is filled from that one, resampled.
void CopyAliasedDepth(RenderCommandList& cl, HostImage& source, HostImage& depth);

// `discard`: the surface is bound to be cleared whole (depth and stencil), so it only takes the version: the copy would
// be overwritten (the scene's 4K MSAA depth: 0.8 ms of the frame at 4K).
void SyncAliasedDepth(RenderCommandList& cl, HostImage& surface, bool discard)
{
    auto it = s_r.edramDepth.find(surface.edramBase);
    if (it == s_r.edramDepth.end() || it->second.first == &surface || surface.depthVersion == it->second.second)
        return;
    HostImage& writer = *it->second.first;
    surface.depthVersion = it->second.second;
    if (!writer.texture || !surface.texture || discard)
        return;
    {
        static const uint64_t traceFrame = [] { const char* v = std::getenv("AC6_DRAW_TRACE"); return v ? std::stoull(v) : UINT64_MAX; }();
        auto name = [](const HostImage& image) {
            for (const auto& [address, entry] : s_r.surfaces)
                if (&entry == &image)
                    return address;
            return 0u;
        };
        if (s_r.frame == traceFrame)
            log::Info("Alias sync: depth base {} {:08X} {}x{} scale {} samples {} <- {:08X} {}x{} scale {} samples {} ({})", surface.edramBase,
                name(surface), surface.width, surface.height, surface.scale, surface.samples, name(writer), writer.width, writer.height, writer.scale,
                writer.samples, MsaaAlias(writer, surface) ? "MSAA swizzle, no stencil"
                    : writer.width == surface.width && writer.height == surface.height ? "copy with stencil" : "resample, no stencil");
    }
    // Multisampled surfaces take part through their single-sampled copies. The receiving
    // one's copy is brought up to date first: the passes that resample keep its own stencil.
    MarkPass([&] { return "alias depth " + PassImage(&writer) + " -> " + PassImage(&surface); });
    s_r.aoAliasFrame = s_r.frame;   // (the hangar's surroundings put back)
    s_r.aoAliasDepth = &surface;
    HostImage& source = Readable(cl, writer);
    HostImage& depth = surface.samples > 1 ? Readable(cl, surface) : surface;
    CopyAliasedDepth(cl, source, depth);
    if (surface.samples > 1)
        WriteSamples(cl, surface);
}

void CopyAliasedDepth(RenderCommandList& cl, HostImage& source, HostImage& depth)
{
    if (MsaaAlias(source, depth) && source.format == depth.format)
    {
        Transition(cl, source, RenderTextureLayout::SHADER_READ);
        Transition(cl, depth, RenderTextureLayout::DEPTH_WRITE);
        cl.setFramebuffer(GetFramebuffer(nullptr, &depth));
        depth.fresh = false;
        DrawAliasPass(cl, source, depth, s_r.depthSwizzlePipeline.get());
        return;
    }
    if (source.width == depth.width && source.height == depth.height && source.format == depth.format)
    {
        // Same size: a straight copy, which carries the stencil too (the cockpit's stencil mask).
        Transition(cl, source, RenderTextureLayout::COPY_SOURCE);
        Transition(cl, depth, RenderTextureLayout::COPY_DEST);
        cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(depth.texture.get()), RenderTextureCopyLocation::Subresource(source.texture.get()));
        depth.fresh = false;
        return;
    }
    Transition(cl, source, RenderTextureLayout::SHADER_READ);
    Transition(cl, depth, RenderTextureLayout::DEPTH_WRITE);
    cl.setFramebuffer(GetFramebuffer(nullptr, &depth));
    if (depth.fresh)
    {
        cl.clearDepthStencil(true, true, 0.0f, 0);
        depth.fresh = false;
    }
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(depth.width), float(depth.height)));
    SetScissor(cl, RenderRect(0, 0, int32_t(depth.width), int32_t(depth.height)));
    SetPipeline(cl, s_r.depthResamplePipeline.get());
    BindCommon(cl);
    const int32_t push[4] = { int32_t(source.descriptor), int32_t(depth.width | (depth.height << 16)), int32_t(source.width), int32_t(source.height) };
    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    cl.drawInstanced(3, 1, 0, 0);
}

// The colour side of EDRAM aliasing: a colour surface bound or resolved after another one with the same
// EDRAM base, format and host size wrote there receives its contents.
void CopyAliasedColor(RenderCommandList& cl, HostImage& source, HostImage& color);

// Fills an 8_8_8_8 colour image with the EDRAM bits of a depth/stencil image.
void DepthToColor(RenderCommandList& cl, HostImage& depth, uint32_t depthFormat, HostImage& color)
{
    if (!depth.stencilCopy)
        depth.stencilCopy = std::make_unique<HostImage>();
    HostImage& stencil = *depth.stencilCopy;
    EnsureImage(stencil, depth.width, depth.height, RenderFormat::R8_UNORM, false);
    stencil.fresh = false;
    CopyStencil(cl, depth, 0, 0, int32_t(depth.width), int32_t(depth.height), stencil, 0, 0);
    Transition(cl, depth, RenderTextureLayout::SHADER_READ);
    Transition(cl, color, RenderTextureLayout::COLOR_WRITE);
    cl.setFramebuffer(GetFramebuffer(&color, nullptr));
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(color.width), float(color.height)));
    SetScissor(cl, RenderRect(0, 0, int32_t(color.width), int32_t(color.height)));
    SetPipeline(cl, s_r.depthToColorPipeline.get());
    BindCommon(cl);
    const uint32_t push[4] = { depth.descriptor, stencil.descriptor, depthFormat == 1 ? 1u : 0u, depth.width / depth.scale };
    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    cl.drawInstanced(3, 1, 0, 0);
    color.fresh = false;
}

// `discard`: as SyncAliasedDepth's, for a colour clear.
void SyncAliasedColor(RenderCommandList& cl, HostImage& surface, bool discard)
{
    // Depth written at this EDRAM base after any colour: the colour surface sees the depth's bits (the game
    // reads its hangar aircraft shadow maps back that way, resolving the depth through a 64x64 colour surface).
    if (surface.format == RenderFormat::B8G8R8A8_UNORM && surface.scale == 1)
        if (auto d = s_r.edramDepth.find(surface.edramBase); d != s_r.edramDepth.end() && d->second.first && d->second.first->texture)
        {
            const uint64_t depthSeq = s_r.edramDepthSeq[surface.edramBase];
            if (depthSeq > s_r.edramColorSeq[surface.edramBase] && surface.depthAliasSeq != depthSeq)
            {
                surface.depthAliasSeq = depthSeq;
                if (discard)
                    return;
                HostImage& writer = *d->second.first;
                MarkPass([&] { return "alias depth to colour " + PassImage(&writer) + " -> " + PassImage(&surface); });
                HostImage& depth = Readable(cl, writer);
                HostImage& color = surface.samples > 1 ? Readable(cl, surface) : surface;
                DepthToColor(cl, depth, writer.guestFormat, color);
                if (surface.samples > 1)
                    WriteSamples(cl, surface);
                return;
            }
        }
    auto it = s_r.edramColor.find(surface.edramBase);
    if (it == s_r.edramColor.end() || it->second.first == &surface || surface.colorVersion == it->second.second)
        return;
    HostImage& writer = *it->second.first;
    surface.colorVersion = it->second.second;
    if (discard)
        return;
    // (Formats: the same, or the HDR canvas's float and an 8_8_8_8 surface over the same EDRAM: the
    // missions' half-resolution effects surface, seen by the canvas.)
    const auto convertible = [](RenderFormat f) { return f == RenderFormat::B8G8R8A8_UNORM || f == RenderFormat::R16G16B16A16_FLOAT; };
    if (!writer.texture || !surface.texture || (writer.format != surface.format && !(convertible(writer.format) && convertible(surface.format))) ||
        writer.width != surface.width || writer.height != surface.height)
        return;
    MarkPass([&] { return "alias colour " + PassImage(&writer) + " -> " + PassImage(&surface); });
    HostImage& source = Readable(cl, writer);
    HostImage& color = surface.samples > 1 ? Readable(cl, surface) : surface;
    CopyAliasedColor(cl, source, color);
    if (surface.samples > 1)
        WriteSamples(cl, surface);
}

void CopyAliasedColor(RenderCommandList& cl, HostImage& source, HostImage& color)
{
    if (MsaaAlias(source, color))
    {
        auto& pipeline = s_r.colorSwizzlePipelines[color.format];
        if (!pipeline)
        {
            RenderGraphicsPipelineDesc desc;
            desc.pipelineLayout = s_r.layout.get();
            desc.vertexShader = s_r.copyVs.get();
            desc.pixelShader = s_r.colorSwizzlePs.get();
            desc.renderTargetFormat[0] = color.format;
            desc.renderTargetBlend[0] = RenderBlendDesc::Copy();
            desc.renderTargetCount = 1;
            pipeline = g_video.device->createGraphicsPipeline(desc);
        }
        Transition(cl, source, RenderTextureLayout::SHADER_READ);
        Transition(cl, color, RenderTextureLayout::COLOR_WRITE);
        cl.setFramebuffer(GetFramebuffer(&color, nullptr));
        color.fresh = false;
        DrawAliasPass(cl, source, color, pipeline.get());
        return;
    }
    // Another format (the HDR canvas's float and an 8_8_8_8 surface): texel for texel through a draw.
    if (source.format != color.format)
    {
        if (RenderPipeline* const copy = TexelCopyPipeline(color.format))
        {
            Transition(cl, source, RenderTextureLayout::SHADER_READ);
            Transition(cl, color, RenderTextureLayout::COLOR_WRITE);
            cl.setFramebuffer(GetFramebuffer(&color, nullptr));
            s_r.markedFramebuffer = nullptr;
            SetViewport(cl, RenderViewport(0.0f, 0.0f, float(color.width), float(color.height)));
            SetScissor(cl, s_aliasScissor ? *s_aliasScissor : RenderRect(0, 0, int32_t(color.width), int32_t(color.height)));
            SetPipeline(cl, copy);
            BindCommon(cl);
            const int32_t push[4] = { int32_t(source.descriptor), 0, 0, 0 };
            cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
            cl.drawInstanced(3, 1, 0, 0);
            color.fresh = false;
        }
        return;
    }
    Transition(cl, source, RenderTextureLayout::COPY_SOURCE);
    Transition(cl, color, RenderTextureLayout::COPY_DEST);
    cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(color.texture.get()), RenderTextureCopyLocation::Subresource(source.texture.get()));
    color.fresh = false;
}

void NoteColorWrite(HostImage* color)
{
    if (!color || color->edramBase == UINT32_MAX)
        return;
    s_r.edramColorSeq[color->edramBase] = ++s_r.edramSeq;
    auto& [writer, version] = s_r.edramColor[color->edramBase];
    if (writer != color || color->colorVersion != version)
    {
        writer = color;
        color->colorVersion = ++version;
    }
}

// Records that `depth` wrote its EDRAM base (draws with depth writes, clears).
void NoteDepthWrite(HostImage* depth)
{
    if (!depth || depth->edramBase == UINT32_MAX)
        return;
    s_r.edramDepthSeq[depth->edramBase] = ++s_r.edramSeq;
    auto& [writer, version] = s_r.edramDepth[depth->edramBase];
    if (writer != depth || depth->depthVersion != version)
    {
        writer = depth;
        depth->depthVersion = ++version;
    }
}

// The late 3D pass: the game draws the own aircraft's canopy glass into the composite target after
// the effects composite, outside the world pass and so outside the
// scene's MSAA and TAA: its edges would be 1-pixel stair steps. With an anti-aliasing mode on, the draws into that target
// after the composite go to a multisampled copy of it (its colour, and its depth in every sample), resolved back into it
// when the pass ends: at the next other bind, a resolve or the present. Depth the late draws write stays in the copy (the
// composite's depth is read by nothing after). AC6_LATE_MSAA=<2|4|8> (4), 0 off.
uint32_t LateMsaaSamples()
{
    static const uint32_t samples = [] {
        const char* v = std::getenv("AC6_LATE_MSAA");
        const int n = v && *v ? std::atoi(v) : 4;
        return n == 2 || n == 4 || n == 8 ? uint32_t(n) : 0u;
    }();
    return TaaOn() || SmaaOn() || s_r.fsr ? samples : 0;
}

void EndLatePass(RenderCommandList& cl)
{
    if (!s_r.lateActive)
        return;
    s_r.lateActive = false;
    HostImage& target = *s_r.lateSource;
    MarkPass([&] { return "late pass MSAA resolve " + PassImage(&target); });
    cl.setFramebuffer(nullptr);
    s_r.markedFramebuffer = nullptr;
    Transition(cl, s_r.lateColor, RenderTextureLayout::RESOLVE_SOURCE);
    Transition(cl, target, RenderTextureLayout::RESOLVE_DEST);
    const RenderRect rect = s_r.lateLoaded;   // only what was loaded (and so drawn)
    if (rect.right > rect.left && rect.bottom > rect.top)
        cl.resolveTextureRegion(target.texture.get(), uint32_t(rect.left), uint32_t(rect.top), s_r.lateColor.texture.get(), &rect);
    NoteColorWrite(&target);
    // This frame's mesh captures, to read four frames on.
    if (s_r.lateCaptureFrame == s_r.frame && s_r.lateCaptureUsed)
    {
        RenderBuffer* readback = s_r.lateReadback[s_r.frame % 4].get();
        cl.barriers(RenderBarrierStage::COPY, RenderBufferBarrier(s_r.lateCapture.get(), RenderBufferAccess::READ));
        cl.barriers(RenderBarrierStage::COPY, RenderBufferBarrier(s_r.lateCapture.get(), RenderBufferAccess::READ));
        cl.copyBufferRegion(RenderBufferReference(readback), RenderBufferReference(s_r.lateCapture.get()), uint64_t(s_r.lateCaptureUsed) * 16);
    }
}

bool Invert4(const double m[16], double out[16]);
RenderViewport ReadViewport(uint32_t device, const Targets& t);

// AC6_LATE_RECT=0: the late pass loads and resolves the whole target.
bool LateRectWanted()
{
    static const bool wanted = [] {
        const char* aa = std::getenv("AC6_AA");
        const char* rect = std::getenv("AC6_LATE_RECT");
        const char* late = std::getenv("AC6_LATE_MSAA");
        return aa && *aa && std::string_view(aa) != "off" && !(rect && *rect == '0') && !(late && *late == '0');
    }();
    return wanted;
}

// The late meshes captured four frames ago: their points, from the clip positions and that frame's transform (row vectors,
// as the game's shaders use them: clip = point x M, so point = clip x inverse(M)).
void LateReadCaptures()
{
    for (auto& [key, mesh] : s_r.lateMeshes)
    {
        if (mesh.captureFrame == ~0ull || mesh.captureFrame + 4 > s_r.frame)
            continue;
        if (mesh.captureFrame + 4 < s_r.frame)
        {
            mesh.captureFrame = ~0ull;   // its readback was overwritten: capture again
            continue;
        }
        RenderBuffer* readback = s_r.lateReadback[mesh.captureFrame % 4].get();
        const auto* clip = static_cast<const float*>(readback->map());
        mesh.points.clear();
        bool bad = false;
        for (uint32_t i = 0; i < mesh.count && !bad; i++)
        {
            const float* c = clip + (mesh.base + i) * 4;
            double point[4] = {};
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 4; k++)
                    point[j] += double(c[k]) * mesh.inverse[k * 4 + j];
            if (c[3] == 0.0f && c[0] == 0.0f && c[1] == 0.0f)
                continue;   // an empty slot
            bad = !(std::fabs(point[3]) > 1e-9) || !std::isfinite(point[0] + point[1] + point[2]);
            if (!bad)
                mesh.points.push_back({ point[0] / point[3], point[1] / point[3], point[2] / point[3] });
        }
        readback->unmap();
        if (bad || mesh.points.empty())
        {
            mesh.points.clear();
            mesh.unusable = true;
        }
        mesh.captureFrame = ~0ull;
        static uint32_t logged = 0;
        if (logged++ < 8)
            log::Info("Late pass: mesh {:016X} captured, {} of {} slots used{}", key, mesh.points.size(), mesh.count,
                mesh.unusable ? " (unusable: the whole target)" : "");
    }
}

// A late draw's rectangle on its target (host pixels, a few pixels of margin): its mesh's points through this draw's
// transform, mapped by its viewport (guest pixels) to the target. False when unknown (not captured yet, or a point
// behind the camera): the whole target then. `capture` asks for this mesh's capture this frame (its slots).
bool LateDrawRect(uint32_t device, const HostShader& vs, const Targets& t, uint64_t key, uint32_t triangleVertices, RenderRect& rect,
    uint32_t& captureBase, uint32_t& captureEnd)
{
    captureBase = captureEnd = 0;
    const uint32_t reg = vs.positionMatrix;
    if (!s_r.lateBoundsGs || reg == UINT32_MAX || !t.color)
        return false;
    double m[16];
    for (uint32_t i = 0; i < 16; i++)
        m[i] = double(std::bit_cast<float>(U32(device + kDevVsConstants + reg * 16 + i * 4)));
    Renderer::LateMesh& mesh = s_r.lateMeshes[key];
    if (mesh.points.empty())
    {
        constexpr uint32_t kCapacity = 16384;   // vertices a frame
        if (!mesh.unusable && mesh.captureFrame == ~0ull && triangleVertices && Invert4(m, mesh.inverse))
        {
            if (s_r.lateCaptureFrame != s_r.frame)
            {
                s_r.lateCaptureFrame = s_r.frame;
                s_r.lateCaptureUsed = 0;
            }
            if (s_r.lateCaptureUsed + triangleVertices <= kCapacity)
            {
                mesh.captureFrame = s_r.frame;
                mesh.base = s_r.lateCaptureUsed;
                mesh.count = triangleVertices;
                captureBase = mesh.base;
                captureEnd = mesh.base + triangleVertices;
                s_r.lateCaptureUsed += triangleVertices;
            }
        }
        else if (!triangleVertices)
            mesh.unusable = true;
        return false;
    }
    double x0 = 1e30, y0 = 1e30, x1 = -1e30, y1 = -1e30;
    for (const auto& p : mesh.points)
    {
        double c[4];
        for (int j = 0; j < 4; j++)
            c[j] = p[0] * m[j] + p[1] * m[4 + j] + p[2] * m[8 + j] + m[12 + j];
        if (!(c[3] > 1e-5))
            return false;   // through the camera's plane: its projection is unbounded
        x0 = std::min(x0, c[0] / c[3]);
        x1 = std::max(x1, c[0] / c[3]);
        y0 = std::min(y0, c[1] / c[3]);
        y1 = std::max(y1, c[1] / c[3]);
    }
    // NDC to guest pixels by the draw's viewport (y down), to host pixels by the target's scale.
    const RenderViewport vp = ReadViewport(device, t);
    const double kx = double(t.color->width) / std::max(1u, GuestWidth(*t.color)), ky = double(t.color->height) / std::max(1u, GuestHeight(*t.color));
    const double gx0 = vp.x + (0.5 + 0.5 * x0) * vp.width, gx1 = vp.x + (0.5 + 0.5 * x1) * vp.width;
    const double gy0 = vp.y + (0.5 - 0.5 * y1) * vp.height, gy1 = vp.y + (0.5 - 0.5 * y0) * vp.height;
    constexpr int32_t kMargin = 4;
    const int32_t w = int32_t(t.color->width), h = int32_t(t.color->height);
    rect.left = std::clamp(int32_t(std::floor(gx0 * kx)) - kMargin, 0, w);
    rect.top = std::clamp(int32_t(std::floor(gy0 * ky)) - kMargin, 0, h);
    rect.right = std::clamp(int32_t(std::ceil(gx1 * kx)) + kMargin, 0, w);
    rect.bottom = std::clamp(int32_t(std::ceil(gy1 * ky)) + kMargin, 0, h);
    return true;
}

// The late copies' samples loaded from the target over `area` (all of it when null).
void LateLoad(RenderCommandList& cl, const Targets& t, const RenderRect* area)
{
    s_aliasScissor = area;
    LoadSamples(cl, *t.color, s_r.lateColor);
    if (t.depth)
        LoadSamples(cl, *t.depth, s_r.lateDepth, false);
    s_aliasScissor = nullptr;
}

// A game draw bound to `t`: when it is a late draw, `t` becomes the multisampled copies (made at the pass's first draw).
// Late draws are the 3D ones (`threeD`: a vertex shader with a named world-view-projection) right after the composite.
// The game goes on using the target for 2D passes and the HUD, and the first of those ends the late pass.
bool BeginLateDraw(RenderCommandList& cl, Targets& t, bool threeD, const RenderRect* bounds)
{
    const uint32_t samples = LateMsaaSamples();
    if (!samples || !t.color || t.color != s_r.compositeTarget || s_r.compositeFrame != s_r.frame || t.samples != 1)
        return false;
    if (!threeD)
    {
        s_r.compositeTarget = nullptr;
        if (s_r.lateActive)
        {
            EndLatePass(cl);
            cl.setFramebuffer(GetFramebuffer(t.color, t.depth));
        }
        return false;
    }
    if (s_r.lateActive && (t.color != s_r.lateSource || t.depth != s_r.lateDepthSource))
        EndLatePass(cl);
    // Only the draws' rectangles are loaded: the pass's first draw's, then what each later one adds
    // (its parts outside what is loaded, which no late draw has touched yet), all of it for a draw whose rectangle isn't
    // known.
    const RenderRect full(0, 0, int32_t(t.color->width), int32_t(t.color->height));
    const RenderRect need = bounds ? *bounds : full;
    if (!s_r.lateActive)
    {
        MarkPass([&] { return "late pass MSAA load " + PassImage(t.color); });
        EnsureImage(s_r.lateColor, t.color->width, t.color->height, t.color->format, false, samples);
        if (t.depth)
            EnsureImage(s_r.lateDepth, t.depth->width, t.depth->height, kDepthFormat, true, samples);
        if (need.right > need.left && need.bottom > need.top)
            LateLoad(cl, t, &need);
        s_r.lateLoaded = need;
        s_r.lateActive = true;
        s_r.lateSource = t.color;
        s_r.lateDepthSource = t.depth;
    }
    else
    {
        const RenderRect have = s_r.lateLoaded;
        const RenderRect all(std::min(have.left, need.left), std::min(have.top, need.top), std::max(have.right, need.right),
            std::max(have.bottom, need.bottom));
        if (all.left != have.left || all.top != have.top || all.right != have.right || all.bottom != have.bottom)
        {
            MarkPass([&] { return "late pass MSAA load more " + PassImage(t.color); });
            // The ring around what is loaded: above, below, then left and right of it.
            const RenderRect parts[4] = { RenderRect(all.left, all.top, all.right, have.top), RenderRect(all.left, have.bottom, all.right, all.bottom),
                RenderRect(all.left, have.top, have.left, have.bottom), RenderRect(have.right, have.top, all.right, have.bottom) };
            for (const RenderRect& part : parts)
                if (part.right > part.left && part.bottom > part.top)
                    LateLoad(cl, t, &part);
            s_r.lateLoaded = all;
        }
    }
    t.color = &s_r.lateColor;
    t.depth = t.depth ? &s_r.lateDepth : nullptr;
    t.samples = samples;
    Transition(cl, s_r.lateColor, RenderTextureLayout::COLOR_WRITE);
    if (t.depth)
        Transition(cl, s_r.lateDepth, RenderTextureLayout::DEPTH_WRITE);
    cl.setFramebuffer(GetFramebuffer(t.color, t.depth));
    s_r.markedFramebuffer = nullptr;
    return true;
}

// Binds the guest device's current render target 0 and depth/stencil surface. `discard`: kClearTargets and/or
// kClearDepth | kClearStencil when the caller clears those whole right after (their aliased contents aren't copied in).
// `keepSingles`: the caller keeps multisampled surfaces' single-sampled copies up to date itself (resolve clears).
// `motion`: the game's draws, which the scene surface's motion image goes with.
bool BindTargets(RenderCommandList& cl, uint32_t device, Targets& t, uint32_t discard = 0, bool keepSingles = false, bool motion = false)
{
    t.color = GetSurface(U32(device + kDevRenderTargets), false);
    t.depth = GetSurface(U32(device + kDevDepthStencil), true);
    // The late pass ends at anything else.
    if (s_r.lateActive && !(motion && t.color == s_r.lateSource))
        EndLatePass(cl);
    if (t.color && t.depth && (t.depth->width < t.color->width || t.depth->height < t.color->height))
    {
        static uint32_t warned = 0;
        if (warned++ < 4)
            log::Warn("Depth surface {}x{} smaller than colour target {}x{}: drawing without depth", t.depth->width, t.depth->height,
                t.color->width, t.color->height);
        t.depth = nullptr;
    }
    if (t.color && t.depth && t.depth->samples != t.color->samples)
    {
        static uint32_t warned = 0;
        if (warned++ < 4)
            log::Warn("Depth surface has {} sample(s), colour target {}: drawing without depth", t.depth->samples, t.color->samples);
        t.depth = nullptr;
    }
    if (!t.color && !t.depth)
        return false;
    t.width = t.color ? t.color->width : t.depth->width;
    t.scale = t.color ? t.color->scale : t.depth->scale;
    t.samples = t.color ? t.color->samples : t.depth->samples;
    t.height = t.color ? t.color->height : t.depth->height;
    // A swapped surface gets its contents back, unless the caller overwrites it whole: it then
    // drops the link once that is recorded (a draw that isn't recorded leaves it for the next use).
    if (t.color && t.color->staleHolder && !(discard & kClearTargets) && !t_overwritesColor)
        RestoreStale(cl, *t.color);
    if (t.depth)
        SyncAliasedDepth(cl, *t.depth, (discard & (kClearDepth | kClearStencil)) == (kClearDepth | kClearStencil));
    if (t.color)
        SyncAliasedColor(cl, *t.color, (discard & kClearTargets) != 0);
    // Fresh images are cleared whole below: their copies go out of date whatever the caller does.
    const bool keepColor = keepSingles && t.color && !t.color->fresh, keepDepth = keepSingles && t.depth && !t.depth->fresh;
    if (t.color)
        Transition(cl, *t.color, RenderTextureLayout::COLOR_WRITE);
    if (t.depth)
        Transition(cl, *t.depth, RenderTextureLayout::DEPTH_WRITE);
    // Bound to be drawn: their single-sampled copies go out of date.
    if (t.color && !keepColor)
        t.color->singleCurrent = false;
    if (t.depth && !keepDepth)
        t.depth->singleCurrent = false;
    // TAA's motion vectors: the scene surface's draws also write their motion, into an image cleared
    // to "unknown" at the frame's first and after the game clears the surface.
    t.motion = nullptr;
    if (motion && t.color && t.color == s_r.taaSurface && MotionVectors())
    {
        EnsureImage(s_r.motion, t.color->width, t.color->height, FsrReactive() ? RenderFormat::R16G16B16A16_FLOAT : RenderFormat::R16G16_FLOAT,
            false, t.color->samples);
        Transition(cl, s_r.motion, RenderTextureLayout::COLOR_WRITE);
        t.motion = &s_r.motion;
    }
    RenderFramebuffer* framebuffer = t.motion ? GetMotionFramebuffer(t.color, t.depth) : GetFramebuffer(t.color, t.depth);
    if (framebuffer != s_r.crumbFramebuffer)
    {
        const HostImage* target = t.color ? t.color : t.depth;
        breadcrumbs::Mark(cl, s_r.frame, target ? target->width : 0, target ? target->height : 0);
        s_r.crumbFramebuffer = framebuffer;
    }
    if (g_video.GpuPassTiming() && framebuffer != s_r.markedFramebuffer)
    {
        g_video.MarkGpuPass(std::format("draw {} + {}", PassImage(t.color), PassImage(t.depth)));
        s_r.markedFramebuffer = framebuffer;
    }
    cl.setFramebuffer(framebuffer);
    if (t.color && t.color->fresh)
    {
        cl.clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
        t.color->fresh = false;
    }
    if (t.motion && (s_r.motionClearedFrame != s_r.frame || s_r.motionClearPending || s_r.motion.fresh))
    {
        cl.clearColor(1, RenderColor(kMotionUnknown, kMotionUnknown, 0.0f, 0.0f));
        s_r.motionClearedFrame = s_r.frame;
        s_r.motionClearPending = false;
        s_r.motion.fresh = false;
    }
    if (t.depth && t.depth->fresh)
    {
        cl.clearDepthStencil(true, true, 0.0f, 0);
        t.depth->fresh = false;
    }
    return true;
}

// The viewport from the PA_CL_VPORT registers when the viewport transform is on (VTE scale/offset
// bits 0-5), otherwise the whole target.
RenderViewport ReadViewport(uint32_t device, const Targets& t)
{
    const uint32_t vte = U32(device + kDevVteControl);
    auto f = [&](uint32_t i) { return std::bit_cast<float>(U32(device + kDevViewport + i * 4)); };
    RenderViewport vp(0.0f, 0.0f, float(t.width), float(t.height));
    if ((vte & 0x3F) == 0x3F)
    {
        const float xs = std::fabs(f(0)), xo = f(1), ys = std::fabs(f(2)), yo = f(3), zs = f(4), zo = f(5);
        vp = RenderViewport(xo - xs, yo - ys, 2.0f * xs, 2.0f * ys, zo, zo + zs);
    }
    static std::set<std::array<uint32_t, 7>> seen;
    std::array<uint32_t, 7> key{ vte };
    for (uint32_t i = 0; i < 6; i++)
        key[i + 1] = U32(device + kDevViewport + i * 4);
    if (seen.size() < 24 && seen.insert(key).second)
        log::Info("Viewport: vte {:X} -> {} {} {}x{} z {}..{} (target {}x{})", vte, vp.x, vp.y, vp.width, vp.height, vp.minDepth, vp.maxDepth,
            t.width, t.height);
    return vp;
}

// Ambient occlusion's camera: what the passes need of it is the projection alone (the field of view
// and the depth's scale), and every perspective draw into the scene carries it in the matrix its vertex shader
// transforms the position with (HostShader::positionMatrix, by the constant table's names, c116 where none is named),
// whatever world or view is multiplied in: row vectors, clip x, y and w are the position along three vectors whose
// lengths are the projection's x and y scales and 1 (times the object's scale, which their ratios lose), and clip z is
// the constant [14] (the game's reversed depth with an infinite far plane). The frame's projection is the one most
// draws into targets of the scene's size have (the cockpit and the inset cameras have their own). Not by the surface the
// scene is resolved from: the hangar draws its world into another surface than the one its scene resolve reads, and has
// no world view-projection at c116, which TAA's camera is read from.
void NoteAoProjection(uint32_t device, const HostShader& vs, const HostShader& ps, const Targets& targets)
{
    if ((!AoMode() && EmitterLight() <= 0.0f) || !targets.color)
        return;
    if (ps.seaReflection)
        s_r.aoMapFrame = s_r.frame;   // (the map's pixel shaders: a scene with terrain)
    const uint32_t reg = vs.positionMatrix != UINT32_MAX ? vs.positionMatrix : 116;
    float m[16];
    for (uint32_t i = 0; i < 16; i++)
        m[i] = std::bit_cast<float>(U32(device + kDevVsConstants + reg * 16 + i * 4));
    const float sx = std::sqrt(m[0] * m[0] + m[4] * m[4] + m[8] * m[8]), sy = std::sqrt(m[1] * m[1] + m[5] * m[5] + m[9] * m[9]);
    const float sw = std::sqrt(m[3] * m[3] + m[7] * m[7] + m[11] * m[11]), sz = std::sqrt(m[2] * m[2] + m[6] * m[6] + m[10] * m[10]);
    // A perspective with a constant clip z: w follows the position, z doesn't.
    if (!(sw > 1e-6f) || !(sx > 1e-6f) || !(sy > 1e-6f) || !(std::fabs(m[14]) > 1e-9f) || sz > 1e-4f * sw || !std::isfinite(sx + sy + sw + m[14]))
        return;
    const float tanX = sw / sx, tanY = sw / sy, clipZ = std::fabs(m[14]);
    if (tanX < 0.01f || tanX > 20.0f || tanY < 0.01f || tanY > 20.0f)
        return;
    if (s_r.aoProjectionsFrame != s_r.frame)
    {
        s_r.aoProjections.clear();
        s_r.aoProjectionsFrame = s_r.frame;
    }
    for (auto& p : s_r.aoProjections)
        if (p.width == targets.width && p.height == targets.height && std::fabs(p.tanX - tanX) < 1e-3f * tanX &&
            std::fabs(p.tanY - tanY) < 1e-3f * tanY && std::fabs(p.clipZ - clipZ) < 1e-3f * clipZ)
        {
            p.draws++;
            return;
        }
    if (s_r.aoProjections.size() < 64)
        s_r.aoProjections.push_back({ tanX, tanY, clipZ, 1, targets.width, targets.height });
}

// The sea's reflection of the scene at a draw of the map's pixel shaders: the draw's camera is noted
// for the frame's reflection pass (RunSeaReflection), and, where last frame's reflection is of this target, the shader
// gets the image and its pixel's way into it. A pixel of the sea lies on the sea's plane, and a plane seen by two
// cameras relates their pictures by a 3 x 3 matrix: pixel -> NDC (A), NDC -> the plane's (x, z, 1) through this draw's
// camera (the inverse of M: the plane's coordinates to clip x, y, w), the plane -> last frame's clip (its M), clip -> the
// image's uv (B), each with its frame's jitter. K = A x inverse(M now) x M then x B, row vectors as the game's.
void SeaReflectionAtDraw(uint32_t device, const HostShader& vs, const HostShader& ps, const Targets& targets, SharedConstants& sc)
{
    sc.seaImage = 0;
    std::memset(sc.sea, 0, sizeof(sc.sea));
    if (!ps.seaReflection || !WaterSsr() || !targets.color)
        return;
    // The map's vertex shaders (vsMapCstCT_Ocean) add the patch's place to the vertex and transform the world position
    // with ACE_mWorldViewProj: the world's view-projection.
    const uint32_t reg = vs.positionMatrix != UINT32_MAX ? vs.positionMatrix : 218;
    double vp[16];
    for (uint32_t i = 0; i < 16; i++)
        vp[i] = double(std::bit_cast<float>(U32(device + kDevVsConstants + reg * 16 + i * 4)));
    if (std::fabs(vp[3]) + std::fabs(vp[7]) + std::fabs(vp[11]) < 1e-6)
        return;   // not a perspective
    if (s_r.seaViewsFrame != s_r.frame)
    {
        s_r.seaViews.clear();
        s_r.seaViewsFrame = s_r.frame;
    }
    if (std::none_of(s_r.seaViews.begin(), s_r.seaViews.end(), [&](const Renderer::SeaView& v) { return v.target == targets.color; }) &&
        s_r.seaViews.size() < 16)
    {
        s_r.seaViews.push_back({ targets.color, {} });
        std::memcpy(s_r.seaViews.back().vp, vp, sizeof(vp));
    }
    if (s_r.seaReflectionFrame + 1 != s_r.frame || targets.color != s_r.seaReflectionSurface || !s_r.seaReflection.texture ||
        s_r.seaReflection.layout != RenderTextureLayout::SHADER_READ)
        return;
    static const double sea = [] { const char* v = std::getenv("AC6_SSR_SEA"); return v && *v ? std::atof(v) : 0.0; }();
    static const float strength = [] { const char* v = std::getenv("AC6_SSR_STRENGTH"); return v && *v ? float(std::atof(v)) : 1.0f; }();
    auto plane = [](const double m[16], double out[9]) {
        const double rows[3][3] = { { m[0], m[1], m[3] }, { m[8], m[9], m[11] },
            { sea * m[4] + m[12], sea * m[5] + m[13], sea * m[7] + m[15] } };
        std::memcpy(out, rows, sizeof(rows));
    };
    auto multiply = [](const double a[9], const double b[9], double out[9]) {
        double r[9];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
        std::memcpy(out, r, sizeof(r));
    };
    double now[9], then[9], inverse[9];
    plane(vp, now);
    plane(s_r.seaReflectionVp, then);
    const double det = now[0] * (now[4] * now[8] - now[5] * now[7]) - now[1] * (now[3] * now[8] - now[5] * now[6]) +
        now[2] * (now[3] * now[7] - now[4] * now[6]);
    if (std::fabs(det) < 1e-30)
        return;
    inverse[0] = (now[4] * now[8] - now[5] * now[7]) / det;
    inverse[1] = (now[2] * now[7] - now[1] * now[8]) / det;
    inverse[2] = (now[1] * now[5] - now[2] * now[4]) / det;
    inverse[3] = (now[5] * now[6] - now[3] * now[8]) / det;
    inverse[4] = (now[0] * now[8] - now[2] * now[6]) / det;
    inverse[5] = (now[2] * now[3] - now[0] * now[5]) / det;
    inverse[6] = (now[3] * now[7] - now[4] * now[6]) / det;
    inverse[7] = (now[1] * now[6] - now[0] * now[7]) / det;
    inverse[8] = (now[0] * now[4] - now[1] * now[3]) / det;
    // This draw's jitter (as Draw applies it to the scene's viewport), in pixels.
    float jx = 0.0f, jy = 0.0f;
    if ((TaaOn() || FsrJitters()) && targets.color == s_r.taaSurface && TaaSurfaceLive() && !t_uiDraw)
    {
        const auto jitter = TaaJitter(SeedFrame());
        jx = jitter.first;
        jy = jitter.second;
    }
    const double w = double(targets.width), h = double(targets.height);
    const double a[9] = { 2.0 / w, 0.0, 0.0, 0.0, -2.0 / h, 0.0, -1.0 - 2.0 * jx / w, 1.0 + 2.0 * jy / h, 1.0 };
    const double b[9] = { 0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.5 + s_r.seaReflectionJitter[0], 0.5 + s_r.seaReflectionJitter[1], 1.0 };
    double k[9];
    multiply(a, inverse, k);
    multiply(k, then, k);
    multiply(k, b, k);
    double largest = 0.0;
    for (const double v : k)
        largest = std::max(largest, std::fabs(v));
    if (!(largest > 0.0) || !std::isfinite(largest))
        return;
    for (int i = 0; i < 9; i++)
        sc.sea[i] = float(k[i] / largest);
    sc.sea[9] = strength;
    sc.seaImage = s_r.seaReflection.descriptor;
}

// --- Pipelines ------------------------------------------------------------------------------------------

struct VertexLayout
{
    std::vector<RenderInputElement> elements;
    std::vector<RenderInputSlot> slots;
    uint32_t streamMask = 0;
    uint32_t strides[16] = {};
    uint64_t hash = 0;
};

bool BuildVertexLayout(const HostShader& vs, VertexLayout& out);

// A draw of the terrain to issue again, depth only, for the ambient occlusion (Draw notes them,
// RunAmbientOcclusion issues them): its pipeline for a depth target alone and what the draw had bound (the frame's
// upload ring holds the constants, vertices and indices until the frame ends).
struct AoTerrainDraw
{
    const PendingPipeline* pipeline = nullptr;
    RenderViewport viewport{};
    uint32_t width = 0, height = 0;
    uint32_t constantOffsets[3] = {};
    uint64_t addresses[3] = {};
    const VertexLayout* layout = nullptr;
    RenderVertexBufferView views[16];
    uint32_t viewCount = 0;
    uint32_t indexOffset = 0, indexCount = 0, drawCount = 0;
    bool blendConstant = false;
    float blend[4] = {};
};
std::vector<AoTerrainDraw> s_aoTerrainDraws;
// The emitters' draws, to issue again into the light image.
std::vector<AoTerrainDraw> s_lightDraws;
// The scene's draws to issue again mirrored in the sea (the planar reflection): the draw, its copy of the vertex
// constants (whose position matrix RunPlanarReflection rewrites), that matrix as the game set it, and the scene's
// viewport without the frame's jitter.
struct PlanarDraw
{
    AoTerrainDraw draw;
    float* matrix = nullptr;
    double m[16] = {};
    // The other camera matrices the vertex shader reads (HostShader::inverseViewProjectionMatrix and the rest): their
    // place in the copy of the constants (null: not read) and their values as the game set them.
    float* inverseVp = nullptr;
    float* inverseView = nullptr;
    float* view = nullptr;
    float* worldInverseView = nullptr;
    double inverseVpValue[16] = {}, inverseViewValue[8] = {}, viewValue[16] = {}, worldInverseViewValue[8] = {};
    float* plane = nullptr;   // vertex constant 255 in that copy: the sea's plane in clip space (SPEC_CONSTANT_PLANAR_CLIP)
    RenderViewport viewport{};
    bool depthWrite = false;   // (those that don't are the sky's layers, the clouds, the glows: drawn twice)
};
std::vector<PlanarDraw> s_planarDraws;

// The layout depends on the shader alone: built once per shader.
const VertexLayout* CachedVertexLayout(const HostShader& vs)
{
    static FastMap<const HostShader*, std::unique_ptr<VertexLayout>> cache;
    auto [it, inserted] = cache.try_emplace(&vs);
    if (inserted)
    {
        auto layout = std::make_unique<VertexLayout>();
        if (BuildVertexLayout(vs, *layout))
            it->second = std::move(layout);
    }
    return it->second.get();
}

// The input layout of a vertex shader, from its vfetch instructions: stream s = vertex fetch constant
// 95 - s (D3D's SetStreamSource numbering), byte offset and stride from the instruction.
bool BuildVertexLayout(const HostShader& vs, VertexLayout& out)
{
    for (const ShaderVertexInput& in : vs.vertexInputs)
    {
        const RenderFormat format = VertexFormat(in.format, in.isSigned, in.isInteger);
        const uint32_t stream = 95 - in.fetchSlot;
        if (format == RenderFormat::UNKNOWN || in.fetchSlot > 95 || stream >= 16 || in.offset < 0)
        {
            // (Under a lock: the pipeline list's builders make layouts too.)
            static std::set<uint32_t> seen;
            static std::mutex seenMutex;
            std::lock_guard lock(seenMutex);
            if (seen.size() < 64 && seen.insert(in.format | (in.fetchSlot << 8)).second)
                log::Warn("Unsupported vertex fetch: format {} fetch constant {} offset {}", in.format, in.fetchSlot, in.offset);
            continue;
        }
        out.elements.push_back(RenderInputElement("TEXCOORD", in.location, in.location, format, stream, uint32_t(in.offset) * 4));
        out.streamMask |= 1u << stream;
        if (in.stride)
            out.strides[stream] = in.stride * 4;
    }
    for (uint32_t s = 0; s < 16; s++)
        if (out.streamMask & (1u << s))
            out.slots.emplace_back(s, out.strides[s]);
    // Hash explicit fields (the structs have uninitialised padding).
    uint64_t h = HashBytes(out.strides, sizeof(out.strides));
    for (auto& e : out.elements)
    {
        const uint32_t fields[4] = { e.location, uint32_t(e.format), e.slotIndex, e.alignedByteOffset };
        h = HashBytes(fields, sizeof(fields), h);
    }
    out.hash = h;
    return true;
}

// AC6_GPU_NAMES=1: Vulkan objects named for GPU tools (Nsight): the game's pipelines and shader
// modules after its shaders' hashes.
bool GpuNames()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_GPU_NAMES"); return v && *v == '1'; }();
    return enabled;
}

// AC6_CACHED_UPLOADS=1: the upload ring and the kept vertex streams in host-cached memory (the
// CPU writes them through its caches, the GPU reads them coherently) instead of the upload heap's write-combined or
// uncached type. On a Vega iGPU the byte swaps into the latter run at an IPC of 0.01: 4.4 -> 0.9 ms of copies
// a frame, but the GPU takes 0.4 ms longer reading the ring's constants and vertices. =2: only the kept vertex streams
// (what the byte swaps write): the same 4.4 -> 0.8 ms, the GPU's time unchanged, so the default on integrated GPUs
// (unified memory: the Vega, the Steam Deck). =0: neither, the default on a discrete GPU, which would read system
// memory over PCIe.
int CachedUploads()
{
    static const int mode = [] {
        if (const char* v = std::getenv("AC6_CACHED_UPLOADS"))
            return std::atoi(v);
        return g_video.device->getCapabilities().uma ? 2 : 0;
    }();
    return mode;
}

// `stream`: a kept vertex stream's buffer (else the upload ring).
std::unique_ptr<RenderBuffer> CreateUploadBuffer(uint64_t size, RenderBufferFlags flags, bool stream)
{
    if (CachedUploads() == 1 || (CachedUploads() == 2 && stream))
    {
        RenderBufferDesc desc = RenderBufferDesc::UploadBuffer(size, flags);
        desc.heapType = RenderHeapType::HOST_CACHED_UPLOAD;
        auto buffer = g_video.device->createBuffer(desc);
        if (buffer && static_cast<plume::VulkanBuffer*>(buffer.get())->vk != VK_NULL_HANDLE)
            return buffer;
        static bool warned = false;
        if (!warned)
        {
            log::Warn("AC6_CACHED_UPLOADS: no host-cached coherent memory type; using the upload heap");
            warned = true;
        }
    }
    return g_video.device->createBuffer(RenderBufferDesc::UploadBuffer(size, flags));
}

// A large computed-index stream's persistent copy for this frame's draws (null: upload it whole this frame).
Renderer::PersistentStream* UsePersistentStream(uint32_t start, uint32_t end)
{
    auto& p = s_r.persistentStreams[end];
    if (p.dynamic)
        return nullptr;
    if (p.buffers[0] && start < p.start)
    {
        // Earlier draws of this frame may already read the current buffers: grow next frame.
        p.wantStart = std::min(p.wantStart ? p.wantStart : p.start, start);
        return nullptr;
    }
    if (!p.buffers[0] || (p.wantStart && p.lastFrame != s_r.frame))
    {
        for (auto& buffer : p.buffers)
            if (buffer)
                s_r.retiredStaging.push_back(std::move(buffer));
        p.start = p.wantStart ? std::min(p.wantStart, start) : start;
        p.end = end;
        p.wantStart = 0;
        for (uint32_t k = 0; k < Video::kFramesInFlight; k++)
        {
            p.buffers[k] = CreateUploadBuffer(p.end - p.start, RenderBufferFlag::VERTEX | RenderBufferFlag::DEVICE_ADDRESSABLE, true);
            p.memory[k] = static_cast<uint32_t*>(p.buffers[k]->map());
            p.address[k] = p.buffers[k]->getDeviceAddress();
            p.generation[k] = 0;
        }
        static bool loggedMemory = false;
        if (!loggedMemory)
        {
            log::Info("Kept vertex stream memory: {}", g_video.DescribeMemory(p.buffers[0].get()));
            loggedMemory = true;
        }
        log::Info("Indexed vertex stream {:08X}-{:08X} kept across frames ({} KB)", p.start, p.end, (p.end - p.start) >> 10);
    }
    if (p.lastFrame != s_r.frame)
    {
        p.lastFrame = s_r.frame;
        s_r.streamsToUpdate.push_back(&p);
    }
    return &p;
}

// A vertex stream's kept copy for this frame's draws, or null (copy it per draw): the
// static meshes are most of a mission's vertex data, 4-7 MB a frame that would be swapped again every frame from
// memory long out of the cache (on a 16 MB L3, the biggest share of the renderer worker's DRAM reads). Kept from
// the third consecutive frame a stream is drawn from. One whose pages are written two updates running (skinned,
// animated) goes back to the copies.
Renderer::PersistentStream* UseVertexStream(uint32_t start, uint32_t end)
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_VERTEX_CACHE"); return !v || *v != '0'; }();
    if (!enabled)
        return nullptr;
    auto& seen = s_r.vertexStreamsSeen[end];
    if (seen.lastFrame != s_r.frame)
    {
        seen.frames = seen.lastFrame + 1 == s_r.frame ? seen.frames + 1 : 1;
        seen.lastFrame = s_r.frame;
    }
    if (seen.frames < 3)
        return nullptr;
    Renderer::PersistentStream* p = UsePersistentStream(start, end);
    if (p)
        p->vertex = true;
    return p;
}

// At submission: brings each persistent stream's buffer for this frame up to date, queuing the pages written
// since it last was (all of them the first time) as swap fills. The pages are watched again first, so a
// write during or after the copy is seen next time.
void UpdatePersistentStreams()
{
    const uint32_t slot = g_video.slot;
    constexpr uint32_t kPage = 4096;
    for (Renderer::PersistentStream* p : s_r.streamsToUpdate)
    {
        const uint32_t guest = PhysicalToGuest(p->start), bytes = p->end - p->start;
        const uint64_t since = p->generation[slot];
        const uint64_t generation = g_memory.Watch(guest, bytes);
        if (!p->vertex)
            s_r.wholeStreams.push_back({ guest, bytes });   // AC6_POOL_HAZARD: their writes are counted too
        const auto* src = g_memory.As<uint32_t>(guest);
        uint32_t dirty = 0, run = UINT32_MAX;
        auto flush = [&](uint32_t at, uint32_t to) {
            s_stats.swapPageBytes += to - at;
            for (uint32_t piece = at; piece < to; piece += (128u << 10))
            {
                const uint32_t n = std::min(to, piece + (128u << 10)) - piece;
                s_r.swapFills.push_back({ p->memory[slot] + piece / 4, src + piece / 4, n / 4 });
            }
        };
        for (uint32_t offset = 0; offset < bytes; offset += kPage)
        {
            // Page boundaries of guest memory, not of the stream (it needn't start on one).
            const uint32_t pageEnd = std::min(bytes, ((guest + offset) & ~(kPage - 1)) + kPage - guest);
            const bool written = !since || !g_memory.Unchanged(guest + offset, pageEnd - offset, since);
            if (written)
            {
                dirty += pageEnd - offset;
                if (run == UINT32_MAX)
                    run = offset;
            }
            else if (run != UINT32_MAX)
            {
                flush(run, offset);
                run = UINT32_MAX;
            }
            offset = pageEnd - kPage;   // the loop adds kPage
        }
        if (run != UINT32_MAX)
            flush(run, bytes);
        p->generation[slot] = generation;
        // A pool rewritten nearly whole every frame costs a write fault per 64 KB block on top of the copy:
        // after a few such frames it is uploaded whole each frame instead.
        if (since && dirty > bytes / 2)
        {
            if (++p->busyFrames >= 8)
            {
                p->dynamic = true;
                log::Info("Indexed vertex stream {:08X}-{:08X} rewritten every frame: uploaded whole", p->start, p->end);
            }
        }
        else
            p->busyFrames = 0;
        // A kept vertex stream written two updates running costs a write fault per 64 KB block each frame on top
        // of its swaps: copied per draw again (where a source written every frame isn't watched).
        if (p->vertex && !p->dynamic && since && dirty)
        {
            if (++p->writtenFrames >= 2)
                p->dynamic = true;
        }
        else
            p->writtenFrames = 0;
    }
    s_r.streamsToUpdate.clear();
    // Kept vertex streams not drawn from for 10000 frames go (their buffers once the frames in flight are done
    // with them): a mission keeps ~15 MB, and a shorter age would remake streams drawn now and then (effects, the
    // inset camera) every few seconds at uncapped frame rates.
    constexpr uint64_t kUnusedFrames = 10000;
    if (s_r.frame % 600 == 599)
    {
        for (auto it = s_r.persistentStreams.begin(); it != s_r.persistentStreams.end();)
            if (it->second.vertex && it->second.lastFrame + kUnusedFrames < s_r.frame)
            {
                for (auto& buffer : it->second.buffers)
                    if (buffer)
                        s_r.retiredStaging.push_back(std::move(buffer));
                it = s_r.persistentStreams.erase(it);
            }
            else
                ++it;
        for (auto it = s_r.vertexStreamsSeen.begin(); it != s_r.vertexStreamsSeen.end();)
            it = it->second.lastFrame + kUnusedFrames < s_r.frame ? s_r.vertexStreamsSeen.erase(it) : std::next(it);
    }
}

RenderComparisonFunction CompareFunction(uint32_t v)
{
    constexpr RenderComparisonFunction kFunctions[8] = { RenderComparisonFunction::NEVER, RenderComparisonFunction::LESS,
        RenderComparisonFunction::EQUAL, RenderComparisonFunction::LESS_EQUAL, RenderComparisonFunction::GREATER,
        RenderComparisonFunction::NOT_EQUAL, RenderComparisonFunction::GREATER_EQUAL, RenderComparisonFunction::ALWAYS };
    return kFunctions[v & 7];
}

// Pipelines built off the render thread. A pipeline whose shader libraries weren't built
// at boot (vertex shaders D3D patches for a vertex declaration, first met where they are drawn) takes the
// driver 5-70 ms. A mission's first frame meets 36 of them, 237 ms one after the other on the render thread.
// Each is a job for these threads and the draw records its bind at once: the replay waits for the
// pipeline, the render thread goes on, and a frame's new pipelines are built side by side. The commands and
// the pipelines are the same either way. AC6_ASYNC_PIPELINES=0 builds them where they are needed.
// A second set of threads, below the game's priority, builds the pipelines of the pipeline list ahead of their draws.
class PipelineBuilder
{
public:
    explicit PipelineBuilder(bool ahead) : ahead(ahead) {}

    void Push(std::function<void()> job)
    {
        std::call_once(started, [this] {
            // AC6_PIPELINE_THREADS / AC6_PIPELINE_AHEAD_THREADS override the counts.
            const char* v = std::getenv(ahead ? "AC6_PIPELINE_AHEAD_THREADS" : "AC6_PIPELINE_THREADS");
            const uint32_t threads = v && std::atoi(v) > 0 ? uint32_t(std::atoi(v)) : std::clamp(HostProcessors(), 4u, 16u) - 2;
            for (uint32_t i = 0; i < threads; i++)
                std::thread([this] {
                    if (ahead)
                        platform::SetCurrentThreadPriority(platform::kPriorityBelowNormal);
                    Run();
                }).detach();
            log::Info("Pipeline builder{}: {} thread(s)", ahead ? " (ahead of the draws)" : "", threads);
        });
        {
            std::lock_guard lock(mutex);
            jobs.push_back(std::move(job));
            outstanding++;
        }
        cv.notify_one();
    }

    // Called by the thread that finishes the last job queued (set before the first Push).
    std::function<void()> whenIdle;

private:
    void Run()
    {
        for (;;)
        {
            std::function<void()> job;
            {
                std::unique_lock lock(mutex);
                cv.wait(lock, [this] { return !jobs.empty(); });
                job = std::move(jobs.front());
                jobs.pop_front();
            }
            job();
            bool idle;
            {
                std::lock_guard lock(mutex);
                idle = --outstanding == 0;
            }
            if (idle && whenIdle)
                whenIdle();
        }
    }

    const bool ahead;
    std::once_flag started;
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::function<void()>> jobs;   // first needed, first built
    uint64_t outstanding = 0;                 // queued or being built
};
PipelineBuilder& s_pipelineBuilder = *new PipelineBuilder(false);   // never destroyed: its threads outlive exit
PipelineBuilder& s_pipelineAhead = *new PipelineBuilder(true);

bool AsyncPipelines()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_ASYNC_PIPELINES"); return !v || *v != '0'; }();
    return enabled && g_video.deferred != nullptr;
}

// A blend control used instead of the guest's for the pipelines asked for while it is set (the clouds'
// occluders drawn into their float layer), 0 = the guest's.
thread_local uint32_t t_blendOverride = 0;
// The pipeline asked for has no stencil test (the hangar's aircraft drawn again into a depth image
// of its own, whose stencil holds nothing of the scene's).
thread_local bool t_noStencil = false;
// The pipeline asked for writes depth whatever is there (the emitters' draws issued again into a
// light image, whose depth is to hold the emitters' own).
thread_local bool t_depthAlwaysWrite = false;
// The pipeline asked for draws the scene mirrored in the sea (the planar reflection): the faces' winding turns
// around, and the near and far planes always clip (the sea's plane is the near plane there).
thread_local bool t_mirror = false;

// The constant polygon offset applied by the vertex shaders rather than as Vulkan's depth bias:
// the SPIR-V backend's shaders only. AC6_DEPTH_BIAS_VS=0: Vulkan's bias.
bool DepthBiasInShader()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_DEPTH_BIAS_VS"); return !(v && *v == '0') && !HlslBackend(); }();
    return enabled;
}

// AC6_HIGHLIGHT_VS=<prefix>[,..]: draws with those vertex shaders ignore depth and culling (finding
// where a draw actually lands).
const std::vector<std::string>& HighlightedShaders()
{
    static const std::vector<std::string> highlight = [] {
        std::vector<std::string> list;
        if (const char* v = std::getenv("AC6_HIGHLIGHT_VS"))
        {
            std::stringstream ss(v);
            for (std::string item; std::getline(ss, item, ',');)
                if (!item.empty()) list.push_back(item);
        }
        return list;
    }();
    return highlight;
}

// What a draw's pipeline is made from besides its two shaders (its vertex layout is its vertex shader's): plain words,
// hashed and compared whole, and kept in the pipeline list, so a pipeline met in one run can be
// built before its first draw in the next. Read from the device's render state by ReadPipelineState, BuildPipeline
// makes the pipeline from it. A change of its layout or of the values it holds (plume's enumerations) needs a new
// kPipelineRecordVersion.
struct PipelineState
{
    uint32_t topology = 0;                  // RenderPrimitiveTopology
    uint32_t spec = 0;                      // the shaders' specialization constant: 0 where both read the draw flags
    uint32_t geometry = 0;                  // 0: none, 1: the smoke harvest's stage, 2: the late meshes' bounds, 3: 2D line rectangles
    uint32_t colorFormat = 0;               // RenderFormat, UNKNOWN: no colour target
    uint32_t motion = 0;                    // TAA's motion image: 0 not bound, 1 bound but not written, 2 written
    uint32_t blend[8] = {};                 // enabled, source, destination, operation, the alpha's three, the write mask
    uint32_t blendConstant = 0;             // reads the blend constant (set per draw)
    uint32_t depthTarget = 0, depthEnabled = 0, depthWrite = 0, depthFunction = 0;
    uint32_t cull = 0, clockwiseFront = 0;
    uint32_t slopeBias = 0;                 // a float's bits, per host pixel (the guest's x the draw's vertical resolution scale)
    int32_t constantBias = 0;
    uint32_t stencil = 0, stencilBits = 0, stencilRefMask = 0;
    uint32_t depthClip = 0, samples = 0, alphaToCoverage = 0;
};
static_assert(sizeof(PipelineState) == 28 * 4, "PipelineState is hashed and stored whole: no padding");

// False: nothing to draw (both faces culled).
bool ReadPipelineState(uint32_t device, const HostShader& vs, const HostShader& ps, RenderPrimitiveTopology topology, uint32_t specConstants,
    const Targets& targets, PipelineState& st)
{
    st = PipelineState{};
    st.topology = uint32_t(topology);
    st.geometry = (specConstants & kSpecVolHarvest) ? 1 : (specConstants & kSpecLateBounds) ? 2 : (specConstants & kSpecLine2D) ? 3 : 0;
    specConstants &= ~(kSpecVolHarvest | kSpecLateBounds | kSpecLine2D);
    RenderBlendDesc blend = RenderBlendDesc::Copy();
    const uint32_t control = t_blendOverride ? t_blendOverride : U32(device + kDevBlendControl);
    if (control != kBlendDisabled)
    {
        // Alpha factors can't use colour terms.
        auto alphaOnly = [](RenderBlend b) {
            switch (b)
            {
            case RenderBlend::SRC_COLOR: return RenderBlend::SRC_ALPHA;
            case RenderBlend::INV_SRC_COLOR: return RenderBlend::INV_SRC_ALPHA;
            case RenderBlend::DEST_COLOR: return RenderBlend::DEST_ALPHA;
            case RenderBlend::INV_DEST_COLOR: return RenderBlend::INV_DEST_ALPHA;
            default: return b;
            }
        };
        blend.blendEnabled = true;
        blend.srcBlend = Blend(control & 0x1F);
        blend.blendOp = BlendOp((control >> 5) & 7);
        blend.dstBlend = Blend((control >> 8) & 0x1F);
        blend.srcBlendAlpha = alphaOnly(Blend((control >> 16) & 0x1F));
        blend.blendOpAlpha = BlendOp((control >> 21) & 7);
        blend.dstBlendAlpha = alphaOnly(Blend((control >> 24) & 0x1F));
    }
    blend.renderTargetWriteMask = &ps == &s_r.nullPixelShader ? 0 : uint8_t(U32(device + kDevColorMask) & 0xF);

    // Depth (RB_DEPTHCONTROL: bit 1 z enable, bit 2 write, bits 4-6 function). As in D3D9, the write bit
    // only takes effect with z enabled: a stencil-only full-screen pass in missions sets write without
    // enable, and writing there would replace the whole scene depth with its own (no soft particles, clouds).
    const uint32_t depthControl = U32(device + kDevDepthControl);
    const bool depthTest = targets.depth && ((depthControl & 2) || t_depthAlwaysWrite);
    const bool depthWrite = depthTest && ((depthControl & 4) || t_depthAlwaysWrite);
    RenderComparisonFunction depthFunction = depthTest && !t_depthAlwaysWrite ? CompareFunction(depthControl >> 4) : RenderComparisonFunction::ALWAYS;
    // AC6_HIGHLIGHT_KEEP=depth,cull,alpha,blend keeps those states as the game set them (bisecting
    // which one hides a draw).
    static const std::string highlightKeep = [] { const char* v = std::getenv("AC6_HIGHLIGHT_KEEP"); return std::string(v ? v : ""); }();
    bool highlighted = false;
    for (auto& prefix : HighlightedShaders())
        highlighted |= std::format("{:016X}", vs.hash).starts_with(prefix);
    if (highlighted && highlightKeep.find("depth") == std::string::npos)
        depthFunction = RenderComparisonFunction::ALWAYS;

    // Culling (PA_SU_SC_MODE_CNTL: bit 0 cull front, bit 1 cull back, bit 2 front face is clockwise).
    // Rectangle lists are expanded here with their own winding, so they're never culled.
    // AC6_CULL=0 disables culling, AC6_FLIP_WINDING=1 swaps the face orientation (diagnostics).
    static const bool cullEnabled = [] { const char* v = std::getenv("AC6_CULL"); return !v || *v != '0'; }();
    static const bool flipWinding = [] { const char* v = std::getenv("AC6_FLIP_WINDING"); return v && *v == '1'; }();
    const uint32_t scMode = U32(device + kDevScModeControl);
    RenderCullMode cull = RenderCullMode::NONE;
    if (cullEnabled && topology >= RenderPrimitiveTopology::TRIANGLE_LIST && !s_r.rectList)
    {
        if ((scMode & 3) == 3)
            return false;   // both faces culled: nothing to draw
        cull = (scMode & 1) ? RenderCullMode::FRONT : (scMode & 2) ? RenderCullMode::BACK : RenderCullMode::NONE;
    }
    const bool clockwiseFront = (((scMode >> 2) & 1) != flipWinding) != t_mirror;
    // (Mirrored, what writes no depth is drawn from both sides: the effects build some quads on the screen, after the
    // projection, where the mirror doesn't turn their winding round, as the afterburners' plumes.)
    if (t_mirror && !(depthControl & 4))
        cull = RenderCullMode::NONE;
    if (highlighted)
    {
        if (highlightKeep.find("cull") == std::string::npos)
            cull = RenderCullMode::NONE;
        if (highlightKeep.find("blend") == std::string::npos)
        {
            blend.blendEnabled = false;
            blend.renderTargetWriteMask = 0xF;
        }
        if (highlightKeep.find("alpha") == std::string::npos)
            specConstants &= ~uint32_t(SPEC_CONSTANT_ALPHA_TEST);
    }
    // Polygon offset (PA_SU_SC_MODE_CNTL bits 11/12 enable it for front/back faces, PA_SU_POLY_OFFSET holds
    // front scale, front offset, back scale, back offset). The scale is taken to be in 1/16 units (the
    // game's decal passes use 48, i.e. a slope factor of 3). The offset is in depth units, converted to
    // Vulkan's constant factor for a 24-bit mantissa. Terrain overlays drawn without it z-fight the
    // ground under them.
    float slopeBias = 0.0f;
    int32_t constantBias = 0;
    if (targets.depth && (scMode & 0x1800))
    {
        const uint32_t base = (scMode & 0x800) ? 0 : 8;
        slopeBias = std::bit_cast<float>(U32(device + kDevPolyOffset + base)) / 16.0f;
        // With the SPIR-V backend the offset is the vertex shaders' (SharedConstants::depthBias).
        if (!DepthBiasInShader())
            constantBias = int32_t(std::lround(std::bit_cast<float>(U32(device + kDevPolyOffset + base + 4)) * 16777216.0f));
    }
    // Stencil (RB_DEPTHCONTROL bit 0, front function/fail/pass/zfail in bits 8-19, the back face's in bits
    // 20-31 when bit 7 is set, reference and masks from RB_STENCILREFMASK). In cockpit view the cockpit and
    // the player's aircraft stamp the stencil and the world draws only where it is lower: the cockpit has
    // its own projection, so depth alone would let the world and the canopy draw over it.
    const bool stencil = !t_noStencil && targets.depth && (depthControl & 1) && !(highlighted && highlightKeep.find("depth") == std::string::npos);
    if (stencil)
    {
        // Vulkan's reference and masks are shared by both faces here (plume). The console has a separate
        // back-face set, which AC6 keeps equal.
        static uint32_t warned = 0;
        if ((depthControl & 0x80) && (U32(device + kDevStencilRefMaskBack) & 0xFFFFFF) != (U32(device + kDevStencilRefMask) & 0xFFFFFF) && warned++ < 4)
            log::Warn("Back-face stencil reference/masks {:06X} differ from the front's {:06X}", U32(device + kDevStencilRefMaskBack) & 0xFFFFFF,
                U32(device + kDevStencilRefMask) & 0xFFFFFF);
    }
    // Depth clipping as the console does it: PA_CL_CLIP_CNTL bit 16 (CLIP_DISABLE) clear
    // means the near and far planes clip. Only then is depth clamped instead. plume clamps unless asked, and
    // with clamping the near terrain's triangles reaching behind the camera are lost: no depth under the
    // ground nearest the camera, so the stencil shadow volumes (z-fail) count nothing there.
    // AC6_DEPTH_CLIP=0: never clip at the near and far planes (diagnostics: depth clamped instead).
    static const bool depthClipOff = [] { const char* v = std::getenv("AC6_DEPTH_CLIP"); return v && *v == '0'; }();

    // Shaders that read the flags from the shared constants specialise on nothing: one pipeline
    // library per shader, and one pipeline whatever the flags (but for the motion image's write mask).
    st.spec = vs.drawFlags && ps.drawFlags ? 0 : specConstants;
    st.colorFormat = uint32_t(targets.color ? targets.color->format : RenderFormat::UNKNOWN);
    // TAA's motion image, written by the draws the draw flags say (Draw: what covers what's behind).
    // With FSR's reactive mask (a four-channel motion image) 3-5: present, opaque (motion and no reactive) or translucent
    // (reactive only, the largest kept).
    const bool reactiveImage = targets.motion && targets.motion->format == RenderFormat::R16G16B16A16_FLOAT;
    st.motion = !targets.motion ? 0 : (specConstants & SPEC_CONSTANT_MOTION) ? (reactiveImage ? 4 : 2) :
        (specConstants & SPEC_CONSTANT_REACTIVE) && reactiveImage ? 5 : reactiveImage ? 3 : 1;
    const uint32_t blendFields[8] = { blend.blendEnabled, uint32_t(blend.srcBlend), uint32_t(blend.dstBlend), uint32_t(blend.blendOp),
        uint32_t(blend.srcBlendAlpha), uint32_t(blend.dstBlendAlpha), uint32_t(blend.blendOpAlpha), blend.renderTargetWriteMask };
    std::memcpy(st.blend, blendFields, sizeof(st.blend));
    // The blend constant (RB_BLEND_RGBA) is set per draw: the aircraft's ground shadow is a stencilled
    // full-screen pass blending constant colour + destination * (1 - constant alpha).
    st.blendConstant = UsesBlendConstant(control);
    st.depthTarget = targets.depth != nullptr;
    st.depthEnabled = depthTest || depthWrite;
    st.depthWrite = depthWrite;
    st.depthFunction = uint32_t(depthFunction);
    st.cull = uint32_t(cull);
    st.clockwiseFront = clockwiseFront;
    // (Per host pixel, in the state the pipeline is found by: a scale applied only when the pipeline is built would
    // stay its first draw's. With FSR upscaling the frame before FSR first runs is at the full size, so pipelines built
    // then, or ahead from the list, would bias the smaller scene's slopes by too much.)
    st.slopeBias = std::bit_cast<uint32_t>(slopeBias == 0.0f ? 0.0f : slopeBias * s_r.resY);
    st.constantBias = constantBias;
    st.stencil = stencil;
    st.stencilBits = stencil ? (depthControl & 0x80 ? depthControl >> 8 : (depthControl >> 8 & 0xFFF) | (depthControl << 4 & 0xFFF000)) : 0;
    st.stencilRefMask = stencil ? U32(device + kDevStencilRefMask) & 0xFFFFFF : 0;
    st.depthClip = t_mirror || (!depthClipOff && !(U32(device + kDevClipControl) & (1u << 16)));
    st.samples = targets.samples;
    // Alpha to mask (RB_COLORCONTROL bit 4) is alpha to coverage on a multisampled target.
    st.alphaToCoverage = targets.samples > 1 && (U32(device + kDevColorControl) & 0x10);
    return true;
}

// By the shaders' SPIR-V, not their guest hashes: alike-translating shaders share pipelines.
uint64_t PipelineKey(const PipelineState& st, const HostShader& vs, const HostShader& ps, const VertexLayout& layout)
{
    const uint64_t key = HashBytes(&st, sizeof(st), (vs.codeHash ? vs.codeHash : vs.hash) * 31 + (ps.codeHash ? ps.codeHash : ps.hash));
    return (key ^ layout.hash) * 0x9E3779B97F4A7C15ull;
}

// The pipeline itself (on any thread: the shaders, the pipeline layout and the internal stages are kept for good).
std::unique_ptr<RenderPipeline> BuildPipeline(const PipelineState& st, const HostShader& vs, const HostShader& ps, const VertexLayout& layout)
{
    RenderGraphicsPipelineDesc desc;
    desc.pipelineLayout = s_r.layout.get();
    desc.vertexShader = vs.shader.get();
    desc.pixelShader = ps.shader.get();
    if (st.geometry == 1)
        desc.geometryShader = s_r.volHarvestGs.get();
    else if (st.geometry == 2)
        desc.geometryShader = s_r.lateBoundsGs.get();
    else if (st.geometry == 3)
        desc.geometryShader = s_r.line2dGs.get();
    desc.primitiveTopology = RenderPrimitiveTopology(st.topology);
    desc.cullMode = RenderCullMode(st.cull);
    // Bit 2 set = clockwise front faces, in Vulkan's framebuffer orientation as the vertex shaders
    // (with -fvk-invert-y) deliver it: the opposite turns the models inside out.
    desc.frontFace = st.clockwiseFront ? RenderFrontFace::CLOCKWISE : RenderFrontFace::COUNTER_CLOCKWISE;
    if (RenderFormat(st.colorFormat) != RenderFormat::UNKNOWN)
    {
        RenderBlendDesc blend = RenderBlendDesc::Copy();
        blend.blendEnabled = st.blend[0] != 0;
        blend.srcBlend = RenderBlend(st.blend[1]);
        blend.dstBlend = RenderBlend(st.blend[2]);
        blend.blendOp = RenderBlendOperation(st.blend[3]);
        blend.srcBlendAlpha = RenderBlend(st.blend[4]);
        blend.dstBlendAlpha = RenderBlend(st.blend[5]);
        blend.blendOpAlpha = RenderBlendOperation(st.blend[6]);
        blend.renderTargetWriteMask = uint8_t(st.blend[7]);
        desc.renderTargetFormat[0] = RenderFormat(st.colorFormat);
        desc.renderTargetBlend[0] = blend;
        desc.renderTargetCount = 1;
    }
    if (st.motion)
    {
        desc.renderTargetFormat[1] = st.motion >= 3 ? RenderFormat::R16G16B16A16_FLOAT : RenderFormat::R16G16_FLOAT;
        desc.renderTargetBlend[1] = RenderBlendDesc::Copy();
        desc.renderTargetBlend[1].renderTargetWriteMask = st.motion == 2 ? 0x3 : st.motion == 4 ? 0xF : st.motion == 5 ? 0x8 : 0;
        if (st.motion == 5)
        {
            desc.renderTargetBlend[1].blendEnabled = true;
            desc.renderTargetBlend[1].srcBlendAlpha = RenderBlend::ONE;
            desc.renderTargetBlend[1].dstBlendAlpha = RenderBlend::ONE;
            desc.renderTargetBlend[1].blendOpAlpha = RenderBlendOperation::MAX;
        }
        desc.renderTargetCount = 2;
    }
    desc.dynamicBlendConstantsEnabled = st.blendConstant != 0;
    desc.dynamicLineWidthEnabled = desc.primitiveTopology == RenderPrimitiveTopology::LINE_LIST ||
        desc.primitiveTopology == RenderPrimitiveTopology::LINE_STRIP;   // (LineWidth, set per draw)
    desc.depthClipEnabled = st.depthClip != 0;
    desc.multisampling.sampleCount = st.samples;
    desc.alphaToCoverageEnabled = st.alphaToCoverage != 0;
    if (st.depthTarget)
    {
        desc.depthTargetFormat = kDepthFormat;
        desc.depthEnabled = st.depthEnabled != 0;
        desc.depthWriteEnabled = st.depthWrite != 0;
        desc.depthFunction = RenderComparisonFunction(st.depthFunction);
        // Per host pixel in Vulkan: at the internal resolution a pixel spans 1/res of the guest's.
        desc.slopeScaledDepthBias = std::bit_cast<float>(st.slopeBias);
        desc.depthBias = st.constantBias;
        if (st.stencil)
        {
            constexpr RenderStencilOp kOps[8] = { RenderStencilOp::KEEP, RenderStencilOp::ZERO, RenderStencilOp::REPLACE,
                RenderStencilOp::INCREMENT_AND_CLAMP, RenderStencilOp::DECREMENT_AND_CLAMP, RenderStencilOp::INVERT,
                RenderStencilOp::INCREMENT_AND_WRAP, RenderStencilOp::DECREMENT_AND_WRAP };
            auto face = [&](uint32_t bits) {
                RenderStencilFaceDesc f;
                f.compareFunction = CompareFunction(bits);
                f.failOp = kOps[(bits >> 3) & 7];
                f.passOp = kOps[(bits >> 6) & 7];
                f.depthFailOp = kOps[(bits >> 9) & 7];
                return f;
            };
            desc.stencilEnabled = true;
            desc.stencilFrontFace = face(st.stencilBits & 0xFFF);
            desc.stencilBackFace = face(st.stencilBits >> 12);
            desc.stencilReference = st.stencilRefMask & 0xFF;
            desc.stencilReadMask = (st.stencilRefMask >> 8) & 0xFF;
            desc.stencilWriteMask = (st.stencilRefMask >> 16) & 0xFF;
        }
    }
    desc.inputSlots = layout.slots.data();
    desc.inputSlotsCount = uint32_t(layout.slots.size());
    desc.inputElements = layout.elements.data();
    desc.inputElementsCount = uint32_t(layout.elements.size());
    const RenderSpecConstant spec(0, st.spec);
    desc.specConstants = &spec;
    desc.specConstantsCount = 1;
    return g_video.device->createGraphicsPipeline(desc);
}

// The pipeline list. A driver without pipeline libraries (AMD's on Windows, older ones) compiles
// every pipeline whole where it is first drawn: mission 01 meets 238 of them, 12 ms each at the median and 132 ms at
// most on a cold driver cache, the replay waiting for them (the frames stall). Which pipelines the game draws with
// can't be told ahead (the render state is the draw's), but it is the same from run to run: each pipeline built for a
// draw is added to <log dir>/shadercache/pipelines.bin as its two guest shaders' hashes and its state (no code: the
// list survives a new translator, when every pipeline is new to the driver again), and at the next start each one is
// built in the background as soon as both its shaders are translated (the boot warm-up): a draw then finds its
// pipeline made. Where pipelines are linked from libraries only those with a geometry stage are compiled
// whole, and only those are built ahead. AC6_PIPELINE_LIST=0: no list (nothing read, written or built ahead),
// AC6_PIPELINE_AHEAD=0 / 1: build nothing / everything ahead.
// The port ships a list too (shipped_pipelines.bin, embedded, made with tools/pipelines/merge.py from recorded lists), so
// a first run has them as well: its pipelines are built ahead after the player's own, and never written to the
// player's file. AC6_PIPELINE_SHIPPED=0: not the shipped list.
struct PipelineRecord
{
    uint64_t vs = 0, ps = 0;   // HostShader::hash: the guest shaders' (with the vertex shader's record and the options' state)
    PipelineState state;
};
static_assert(sizeof(PipelineRecord) == 128, "PipelineRecord is hashed and stored whole: no padding");
constexpr uint32_t kPipelineRecordVersion = 1;

class PipelineList
{
public:
    // Once the renderer is ready (the pipeline layout and the internal stages exist).
    void Load()
    {
        const char* v = std::getenv("AC6_PIPELINE_LIST");
        if ((v && *v == '0') || !HighlightedShaders().empty())
            return;
        const char* a = std::getenv("AC6_PIPELINE_AHEAD");
        aheadNone = a && *a == '0';
        aheadAll = a ? *a == '1' : !g_video.device->usesShaderLibraries();
        path = std::filesystem::path(log::Directory()) / "shadercache" / "pipelines.bin";
        std::vector<uint8_t> bytes;
        {
            std::ifstream f(path, std::ios::binary);
            bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        }
        // Each entry stands alone ([tag][record][hash]): a torn or foreign stretch is skipped byte by byte.
        size_t at = 0;
        bool clean = true;
        while (at + sizeof(Entry) <= bytes.size())
        {
            Entry e;
            std::memcpy(&e, bytes.data() + at, sizeof(e));
            if (e.tag != kTag || e.hash != HashBytes(&e.record, sizeof(e.record)))
            {
                at++;
                clean = false;
                continue;
            }
            at += sizeof(e);
            if (known.insert(e.hash).second)
                records.push_back(e.record);
            else
                clean = false;
        }
        clean &= at == bytes.size();
        // Records whose build ahead ended in a fault (OnFault below) are left to their draws from then on.
        skipPath = path;
        skipPath.replace_extension(".skip");
        {
            std::ifstream f(skipPath, std::ios::binary);
            for (uint64_t hash; f.read(reinterpret_cast<char*>(&hash), sizeof(hash));)
                skipped.insert(hash);
        }
        if (!skipped.empty())
            log::Warn("Pipeline list: {} pipeline(s) are not built ahead: an earlier run faulted building them ({})", skipped.size(),
                platform::PathUtf8(skipPath.filename()));
        // Kept to a size: the oldest go (they come back if they are drawn again).
        constexpr size_t kMost = 32768, kKept = 24576;
        if (records.size() > kMost)
        {
            records.erase(records.begin(), records.end() - kKept);
            known.clear();
            for (const PipelineRecord& r : records)
                known.insert(HashBytes(&r, sizeof(r)));
            clean = false;
        }
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        if (!clean)
        {
            const std::filesystem::path temp = std::filesystem::path(path).concat(".tmp");
            {
                std::ofstream f(temp, std::ios::binary);
                for (const PipelineRecord& r : records)
                    WriteEntry(f, r);
            }
            std::filesystem::rename(temp, path, ec);
        }
        const size_t own = records.size();
        static const bool shippedList = [] { const char* s = std::getenv("AC6_PIPELINE_SHIPPED"); return !(s && *s == '0'); }();
        size_t shipped = 0;
        if (shippedList)
        {
            size_t size = 0;
            const unsigned char* data = EmbeddedFileData("ShippedPipelines", &size);
            for (size_t at = 0; data && at + sizeof(Entry) <= size; at += sizeof(Entry))
            {
                Entry e;
                std::memcpy(&e, data + at, sizeof(e));
                if (e.tag == kTag && e.hash == HashBytes(&e.record, sizeof(e.record)) && known.insert(e.hash).second)
                {
                    records.push_back(e.record);
                    shipped++;
                }
            }
        }
        started.assign(records.size(), 0);
        for (uint32_t i = 0; i < records.size(); i++)
        {
            byShader.emplace(records[i].vs, i);
            byShader.emplace(records[i].ps, i);
            toBuild += !aheadNone && (aheadAll || records[i].state.geometry) && !skipped.contains(HashBytes(&records[i], sizeof(PipelineRecord)));
        }
        if (s_r.nullPixelShader.ok)
            ready.emplace(s_r.nullPixelShader.hash, &s_r.nullPixelShader);
        s_pipelineAhead.whenIdle = [this] { Report(); };
        log::Info("Pipeline list: {} pipeline(s) known from earlier runs and {} more from the shipped list ({})", own, shipped,
            aheadNone ? "none built ahead: AC6_PIPELINE_AHEAD=0"
            : aheadAll ? "built ahead as their shaders are ready" : "linked from libraries at their draws; those with a geometry stage built ahead");
        active.store(true, std::memory_order_release);
    }

    // Whether a pipeline of the list is made of this shader and is built ahead: its module is then wanted early.
    bool Wants(uint64_t shaderHash)
    {
        if (!active.load(std::memory_order_acquire) || aheadNone)
            return false;
        std::lock_guard lock(mutex);
        return byShader.contains(shaderHash);
    }

    // A shader has its module: the list's pipelines made of it and of another shader that is ready are built.
    void ShaderReady(const HostShader& s)
    {
        if (!active.load(std::memory_order_acquire) || aheadNone)
            return;
        std::lock_guard lock(mutex);
        if (!ready.emplace(s.hash, &s).second)
            return;
        const auto [first, last] = byShader.equal_range(s.hash);
        for (auto it = first; it != last; ++it)
        {
            const uint32_t i = it->second;
            const PipelineRecord& r = records[i];
            if (started[i] || (!aheadAll && !r.state.geometry) || (!skipped.empty() && skipped.contains(HashBytes(&r, sizeof(r)))))
                continue;
            const auto vs = ready.find(r.vs), ps = ready.find(r.ps);
            if (vs == ready.end() || ps == ready.end())
                continue;
            started[i] = 1;
            queued++;
            s_pipelineAhead.Push([this, r, vs = vs->second, ps = ps->second] { BuildAhead(r, *vs, *ps); });
        }
    }

    // A draw makes the pipeline `key` its own: the one built ahead (or being built: the draw's bind waits for it), or null.
    std::unique_ptr<PendingPipeline> Take(uint64_t key)
    {
        if (!active.load(std::memory_order_acquire) || aheadNone)
            return nullptr;
        std::lock_guard lock(mutex);
        drawn.insert(key);
        const auto it = ahead.find(key);
        if (it == ahead.end())
            return nullptr;
        std::unique_ptr<PendingPipeline> pending = std::move(it->second);
        ahead.erase(it);
        taken++;
        return pending;
    }

    // A pipeline was built for a draw: into the list, if it is new to it.
    void Note(const PipelineRecord& r)
    {
        if (!active.load(std::memory_order_acquire))
            return;
        {
            std::lock_guard lock(mutex);
            if (!known.insert(HashBytes(&r, sizeof(r))).second)
                return;
        }
        std::lock_guard lock(fileMutex);
        if (!file.is_open())
            file.open(path, std::ios::binary | std::ios::app);
        WriteEntry(file, r);
        file.flush();
    }

    // For the render statistics (AC6_RENDER_STATS), empty without a list.
    std::string Stats()
    {
        if (!active.load(std::memory_order_acquire))
            return {};
        std::lock_guard lock(mutex);
        return std::format("pipeline list: {} known from earlier runs, {} of {} built ahead in {:.0f} ms of the builders' time, {} of them taken by draws",
            records.size(), built, toBuild, builtNs / 1e6, taken);
    }

private:
    struct Entry
    {
        uint64_t tag = 0;
        PipelineRecord record;
        uint64_t hash = 0;
    };
    static constexpr uint64_t kTag = 0x0050363643410000ull | kPipelineRecordVersion << 8 | sizeof(PipelineRecord) >> 3;

    static void WriteEntry(std::ofstream& f, const PipelineRecord& r)
    {
        const Entry e{ kTag, r, HashBytes(&r, sizeof(r)) };
        f.write(reinterpret_cast<const char*>(&e), sizeof(e));
    }

    void BuildAhead(const PipelineRecord& r, const HostShader& vs, const HostShader& ps)
    {
        // (A list from elsewhere, or damaged past its hashes: only what a draw could have asked for.)
        const PipelineState& st = r.state;
        if (vs.pixel || !ps.pixel || !vs.ok || !ps.ok || !vs.shader || !ps.shader || st.geometry > 2 ||
            (st.geometry == 1 && !s_r.volHarvestGs) || (st.geometry == 2 && !s_r.lateBoundsGs) || st.samples == 0 || st.samples > 8 ||
            st.topology > uint32_t(RenderPrimitiveTopology::TRIANGLE_FAN) || st.motion > 5 || st.cull > uint32_t(RenderCullMode::BACK) ||
            st.depthFunction > uint32_t(RenderComparisonFunction::ALWAYS) || st.colorFormat >= uint32_t(RenderFormat::MAX) ||
            std::max({ st.blend[1], st.blend[2], st.blend[4], st.blend[5] }) > uint32_t(RenderBlend::INV_BLEND_FACTOR_ALPHA) ||
            std::max(st.blend[3], st.blend[6]) > uint32_t(RenderBlendOperation::MAX))
            return;
        VertexLayout layout;
        if (!BuildVertexLayout(vs, layout))
            return;
        const uint64_t key = PipelineKey(st, vs, ps, layout);
        PendingPipeline* pending;
        {
            std::lock_guard lock(mutex);
            if (drawn.contains(key) || ahead.contains(key))
                return;   // a draw was first (it builds its own), or another record of the same pipeline
            auto& slot = ahead[key];
            slot = std::make_unique<PendingPipeline>();
            pending = slot.get();
        }
        // A fault in the driver here would come back at every start: the fault's report notes the record (OnFault) and
        // later runs leave it to its draw.
        thread_local FaultContext t_fault;
        t_fault = { this, HashBytes(&r, sizeof(r)) };
        crash::SetThreadTask("building a pipeline of the pipeline list ahead of its draws (it is skipped from the next run on)",
            [](void* context) {
                const auto* c = static_cast<const FaultContext*>(context);
                std::ofstream f(c->list->skipPath, std::ios::binary | std::ios::app);
                f.write(reinterpret_cast<const char*>(&c->record), sizeof(c->record));
            }, &t_fault);
        // AC6_PIPELINE_AHEAD_FAULT=<n> (tests): the n-th build ahead faults, as a driver might.
        static const uint64_t faultAt = [] { const char* v = std::getenv("AC6_PIPELINE_AHEAD_FAULT"); return v ? std::strtoull(v, nullptr, 10) : 0ull; }();
        static std::atomic<uint64_t> builds{ 0 };
        if (++builds == faultAt)
            *static_cast<volatile uint32_t*>(nullptr) = 0;
        const auto start = std::chrono::steady_clock::now();
        std::unique_ptr<RenderPipeline> pipeline = BuildPipeline(st, vs, ps, layout);
        const int64_t ns = (std::chrono::steady_clock::now() - start).count();
        crash::SetThreadTask(nullptr);
        if (pipeline && GpuNames())
            pipeline->setName(std::format("vs {:016X} ps {:016X} (ahead)", vs.hash, ps.hash));
        pending->Publish(std::move(pipeline));
        std::lock_guard lock(mutex);
        built++;
        builtNs += ns;
        slowestNs = std::max(slowestNs, ns);
    }

    // The ahead threads have nothing left to build (at that moment: shaders met later bring more, a few at a time: not each
    // of those is logged).
    void Report()
    {
        std::lock_guard lock(mutex);
        if (built == reported || (reported && built < reported + 32) || reports >= 64)
            return;
        reports++;
        reported = built;
        log::Info("Pipeline list: {} pipeline(s) built ahead of their draws so far, {:.0f} ms of the builders' time (the slowest {:.1f} ms); "
                  "{} taken by draws; {} of the {} to build wait for shaders not met in this run",
            built, builtNs / 1e6, slowestNs / 1e6, taken, toBuild - queued, toBuild);
    }

    struct FaultContext
    {
        PipelineList* list = nullptr;
        uint64_t record = 0;
    };

    std::atomic<bool> active{ false };
    bool aheadAll = false, aheadNone = false;
    std::filesystem::path path, skipPath;
    std::unordered_set<uint64_t> skipped;                        // records' hashes: not built ahead (set before `active`)
    std::mutex mutex;
    std::vector<PipelineRecord> records;                         // as loaded
    std::vector<uint8_t> started;                                // per record: queued for building
    std::unordered_set<uint64_t> known;                          // the records' hashes, with those noted in this run
    std::unordered_multimap<uint64_t, uint32_t> byShader;        // shader hash -> records made of it
    std::unordered_map<uint64_t, const HostShader*> ready;       // shaders that have their module
    std::unordered_map<uint64_t, std::unique_ptr<PendingPipeline>> ahead;   // by pipeline key: built, or being built
    std::unordered_set<uint64_t> drawn;                          // keys the draws have taken or built themselves
    uint64_t toBuild = 0;                                        // records built ahead once their shaders are ready
    uint64_t queued = 0, built = 0, taken = 0, reported = 0, reports = 0;
    int64_t builtNs = 0, slowestNs = 0;
    std::mutex fileMutex;
    std::ofstream file;
};
PipelineList& s_pipelineList = *new PipelineList;   // never destroyed: the builders' threads outlive exit

PendingPipeline* GetPipelineUncached(uint32_t device, const HostShader& vs, const HostShader& ps, const VertexLayout& layout,
    RenderPrimitiveTopology topology, uint32_t specConstants, const Targets& targets);

// Consecutive draws mostly share their pipeline: what it is made from is compared with the
// draw before's (the shaders, the layout, the targets and the render state words it reads) before its key is
// hashed and looked up.
PendingPipeline* GetPipeline(uint32_t device, const HostShader& vs, const HostShader& ps, const VertexLayout& layout,
    RenderPrimitiveTopology topology, uint32_t specConstants, const Targets& targets)
{
    struct Inputs
    {
        const void* vs;
        const void* ps;
        const void* layout;
        const void* color;
        uint32_t topology, specConstants, colorFormat, depth, samples, rectList, motion;
        uint32_t words[11];
    };
    static Inputs last;
    static PendingPipeline* lastPipeline = nullptr;
    static bool lastValid = false;
    static const bool enabled = [] { const char* v = std::getenv("AC6_PIPELINE_REUSE"); return (!v || *v != '0') && !std::getenv("AC6_HIGHLIGHT_VS"); }();
    Inputs in;
    std::memset(&in, 0, sizeof(in));
    in.vs = &vs;
    in.ps = &ps;
    in.layout = &layout;
    in.color = targets.color;
    in.topology = uint32_t(topology);
    in.specConstants = specConstants;
    in.colorFormat = targets.color ? uint32_t(targets.color->format) : 0;
    in.depth = targets.depth != nullptr;
    in.samples = targets.samples;
    in.rectList = s_r.rectList;
    in.motion = (targets.motion != nullptr ? 1u : 0u) | (t_noStencil ? 2u : 0u) | (t_depthAlwaysWrite ? 4u : 0u) | (t_mirror ? 8u : 0u);
    const uint32_t offsets[11] = { kDevBlendControl, kDevColorMask, kDevDepthControl, kDevScModeControl, kDevPolyOffset, kDevPolyOffset + 4,
        kDevPolyOffset + 8, kDevPolyOffset + 12, kDevStencilRefMask, kDevClipControl, kDevColorControl };
    for (uint32_t i = 0; i < 11; i++)
        in.words[i] = U32(device + offsets[i]);
    if (t_blendOverride)
        in.words[0] = t_blendOverride;
    if (enabled && lastValid && std::memcmp(&in, &last, sizeof(in)) == 0)
        return lastPipeline;
    PendingPipeline* pipeline = GetPipelineUncached(device, vs, ps, layout, topology, specConstants, targets);
    last = in;
    lastPipeline = pipeline;
    lastValid = true;
    return pipeline;
}

PendingPipeline* GetPipelineUncached(uint32_t device, const HostShader& vs, const HostShader& ps, const VertexLayout& layout,
    RenderPrimitiveTopology topology, uint32_t specConstants, const Targets& targets)
{
    PipelineState st;
    if (!ReadPipelineState(device, vs, ps, topology, specConstants, targets, st))
        return nullptr;
    const uint64_t key = PipelineKey(st, vs, ps, layout);
    auto& slot = s_r.pipelines[key];
    if (slot)
        return slot.get();
    // Built ahead from the pipeline list, or being built: the bind waits for it as for any other.
    if ((slot = s_pipelineList.Take(key)))
        return slot.get();
    slot = std::make_unique<PendingPipeline>();

    // Everything the job points to outlives the build (the shaders and the vertex layouts are kept for good).
    auto build = [pending = slot.get(), st, vs = &vs, ps = &ps, layout = &layout, key] {
        const auto pipelineStart = std::chrono::steady_clock::now();
        std::unique_ptr<RenderPipeline> pipeline = BuildPipeline(st, *vs, *ps, *layout);
        const int64_t pipelineNs = (std::chrono::steady_clock::now() - pipelineStart).count();
        // AC6_GPU_NAMES=1: named after its shaders, as the log knows them, for GPU tools.
        if (pipeline && GpuNames())
            pipeline->setName(std::format("vs {:016X} ps {:016X}", vs->hash, ps->hash));
        s_pipelinesCreated++;
        s_pipelineNs += pipelineNs;
        log::Info("Pipeline {:016X}: vs {:016X} ps {:016X} topology {} blend {} {}/{} colour {} depth {}/{}/{} in {:.2f} ms", key, vs->hash, ps->hash,
            st.topology, st.blend[0] != 0, st.blend[1], st.blend[2], st.colorFormat, st.depthTarget && st.depthEnabled,
            st.depthTarget && st.depthWrite, st.depthFunction, pipelineNs / 1e6);
        pending->Publish(std::move(pipeline));
        s_pipelineList.Note({ vs->hash, ps->hash, st });
    };
    if (AsyncPipelines())
        s_pipelineBuilder.Push(std::move(build));
    else
        build();
    return slot.get();
}

// Per-channel statistics of raw texels (AC6_DUMP_RESOLVES / AC6_RESOLVE_STATS): what pictures can't show
// (alpha, values outside 0..1, NaN, infinity, exact 0 and 1 counts).
std::string TexelStats(const uint8_t* p, RenderFormat format, uint32_t width, uint32_t height)
{
    const bool half = format == RenderFormat::R16G16B16A16_FLOAT, bgra8 = format == RenderFormat::B8G8R8A8_UNORM;
    const uint32_t channels = half || bgra8 ? 4 : 1, count = width * height;
    auto value = [&](uint32_t texel, uint32_t c) {
        if (bgra8)
            return p[texel * 4 + (c == 3 ? 3 : 2 - c)] / 255.0f;
        if (!half)
        {
            float f;
            std::memcpy(&f, p + texel * 4, 4);
            return f;
        }
        uint16_t h;
        std::memcpy(&h, p + texel * 8 + c * 2, 2);
        const uint32_t sign = (h >> 15) & 1, exponent = (h >> 10) & 0x1F, mantissa = h & 0x3FF;
        float f = exponent == 0 ? std::ldexp(float(mantissa), -24)
            : exponent == 31 ? (mantissa ? NAN : INFINITY) : std::ldexp(float(mantissa | 0x400), int(exponent) - 25);
        return sign ? -f : f;
    };
    std::string text;
    for (uint32_t c = 0; c < channels; c++)
    {
        float lo = INFINITY, hi = -INFINITY;
        double sum = 0;
        uint32_t nans = 0, infs = 0, zeros = 0, ones = 0;
        for (uint32_t i = 0; i < count; i++)
        {
            const float v = value(i, c);
            zeros += v == 0.0f;
            ones += v == 1.0f;
            if (std::isnan(v)) { nans++; continue; }
            if (std::isinf(v)) { infs++; continue; }
            lo = std::min(lo, v);
            hi = std::max(hi, v);
            sum += v;
        }
        text += std::format(" [{}: min {} max {} mean {:.5g} nan {} inf {} =0 {} =1 {}]", "rgba"[c], lo, hi,
            count > nans + infs ? sum / (count - nans - infs) : 0.0, nans, infs, zeros, ones);
    }
    return text;
}

// A BMP of raw texels (B8G8R8A8 as is, R32F as grey scaled to its maximum), top-down.
void WriteTexelsBmp(const std::filesystem::path& path, const uint8_t* p, RenderFormat format, uint32_t width, uint32_t height)
{
    std::vector<uint8_t> bgra(size_t(width) * height * 4, 255);
    if (format == RenderFormat::B8G8R8A8_UNORM)
        std::memcpy(bgra.data(), p, bgra.size());
    else if (format == RenderFormat::R32_FLOAT)
    {
        float top = 0.0f;
        for (size_t i = 0; i < size_t(width) * height; i++)
        {
            float v;
            std::memcpy(&v, p + i * 4, 4);
            if (std::isfinite(v))
                top = std::max(top, v);
        }
        for (size_t i = 0; i < size_t(width) * height; i++)
        {
            float v;
            std::memcpy(&v, p + i * 4, 4);
            const uint8_t g = top > 0.0f && std::isfinite(v) ? uint8_t(std::clamp(v / top, 0.0f, 1.0f) * 255.0f) : 0;
            bgra[i * 4] = bgra[i * 4 + 1] = bgra[i * 4 + 2] = g;
        }
    }
    else
        return;
    for (size_t i = 0; i < size_t(width) * height; i++)
        bgra[i * 4 + 3] = 255;   // opaque, so viewers show the colour channels
    std::ofstream f(path, std::ios::binary);
    const uint32_t imageSize = width * height * 4;
    auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    f.write("BM", 2); u32(54 + imageSize); u32(0); u32(54);
    u32(40); u32(width); u32(uint32_t(-int32_t(height))); u16(1); u16(32); u32(0); u32(imageSize); u32(2835); u32(2835); u32(0); u32(0);
    f.write(reinterpret_cast<const char*>(bgra.data()), imageSize);
}

// AC6_RING=<seconds>: the last seconds of frames kept at 60 a second (AC6_RING_HEIGHT=<pixels>, 720: the frame scaled to
// that height). F11 writes them out.
double RingSeconds()
{
    // (Nothing kept with the captures' switch off at the start.)
    static const double seconds = [] {
        const char* on = std::getenv("AC6_CAPTURES");
        if (!on || (std::string_view(on) != "1" && std::string_view(on) != "true"))
            return 0.0;
        const char* v = std::getenv("AC6_RING");
        return v && *v ? std::clamp(std::atof(v), 0.0, 60.0) : 0.0;
    }();
    return seconds;
}

void WriteBgraBmp(const std::filesystem::path& path, const uint8_t* bgra, uint32_t width, uint32_t height)
{
    std::ofstream f(path, std::ios::binary);
    const uint32_t imageSize = width * height * 4;
    auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    f.write("BM", 2); u32(54 + imageSize); u32(0); u32(54);
    u32(40); u32(width); u32(uint32_t(-int32_t(height))); u16(1); u16(32); u32(0); u32(imageSize); u32(2835); u32(2835); u32(0); u32(0);
    f.write(reinterpret_cast<const char*>(bgra), imageSize);
}

// The frames held, written on a thread of their own (the game goes on): frame_<i>_<frame number>.bmp, oldest first.
void DumpRing()
{
    if (s_r.ringFrames.empty())
    {
        log::Info("Frame ring: nothing kept yet");
        return;
    }
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &t);
#else
    localtime_r(&t, &local);
#endif
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &local);
    const std::filesystem::path dir = std::filesystem::path(log::Directory()) / std::format("ring_{}", stamp);
    auto frames = std::make_shared<std::deque<std::pair<uint64_t, std::vector<uint8_t>>>>(std::move(s_r.ringFrames));
    s_r.ringFrames.clear();
    const uint32_t w = s_r.ringWidth, h = s_r.ringHeight;
    log::Info("Frame ring: writing {} frame(s) of {}x{} to {}", frames->size(), w, h, platform::PathUtf8(dir));
    std::thread([frames, dir, w, h] {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        uint32_t i = 0;
        for (const auto& [frame, pixels] : *frames)
            WriteBgraBmp(dir / std::format("frame_{:04}_{}.bmp", i++, frame), pixels.data(), w, h);
        log::Info("Frame ring: {} frame(s) written to {}", frames->size(), platform::PathUtf8(dir));
    }).detach();
}

void WriteScreenshot(const std::string& name)
{
    const uint32_t rowPitch = s_r.targetWidth * 4;
    auto* pixels = static_cast<const uint8_t*>(s_r.readback->map());
    std::filesystem::path path = std::filesystem::path(log::Directory()) / (name + ".bmp");
    std::ofstream f(path, std::ios::binary);
    const uint32_t imageSize = rowPitch * s_r.targetHeight;
    auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    f.write("BM", 2); u32(54 + imageSize); u32(0); u32(54);
    u32(40); u32(s_r.targetWidth); u32(uint32_t(-int32_t(s_r.targetHeight))); u16(1); u16(32); u32(0); u32(imageSize); u32(2835); u32(2835); u32(0); u32(0);
    f.write(reinterpret_cast<const char*>(pixels), imageSize);   // BGRA, top-down
    s_r.readback->unmap();
    log::Info("Screenshot {}", platform::PathUtf8(path));
}

} // namespace

bool RendererSeaReflects()
{
    return WaterSsr();
}

// (Defined with the other passes over the scene, below, used by the draws.)
void RunSeaReflection(RenderCommandList& cl, HostImage& image, const HostImage* surface);
void RunPlanarReflection(RenderCommandList& cl, HostImage& image, const HostImage* surface);
void RunLensFx(RenderCommandList& cl, HostImage& image, const LensFx& fx);
HostImage* MotionForBlur(RenderCommandList& cl);
void RunEnhancedBloom(RenderCommandList& cl, uint32_t device, const Targets& targets, const SharedConstants& sc);
void RunEmitterLight(RenderCommandList& cl, HostImage& image);
void RunMotionBlur(RenderCommandList& cl, HostImage& image, HostImage& motion);

// --- Public API -----------------------------------------------------------------------------------------

std::atomic<bool> s_shotRequest{ false };   // RequestScreenshot

void RequestScreenshot()
{
    s_shotRequest.store(true);
}

// NoteLayerClip's rects: {guest x, y, w, h, canvas x, y, w, h}, the latest few (the radar's, a frame or two of them).
std::mutex& s_layerClipMutex = *new std::mutex;
std::vector<std::array<int32_t, 8>>& s_layerClips = *new std::vector<std::array<int32_t, 8>>;

void NoteLayerClip(int32_t gx, int32_t gy, int32_t gw, int32_t gh, int32_t cx, int32_t cy, int32_t cw, int32_t ch)
{
    std::lock_guard lock(s_layerClipMutex);
    for (auto& c : s_layerClips)
        if (c[0] == gx && c[1] == gy && c[2] == gw && c[3] == gh)
        {
            c = { gx, gy, gw, gh, cx, cy, cw, ch };
            return;
        }
    if (s_layerClips.size() >= 16)
        s_layerClips.erase(s_layerClips.begin());
    s_layerClips.push_back({ gx, gy, gw, gh, cx, cy, cw, ch });
}

// The canvas rect noted for a guest viewport, if any.
bool LayerClip(const RenderViewport& v, std::array<int32_t, 4>& canvas)
{
    const int32_t x = int32_t(std::lround(v.x)), y = int32_t(std::lround(v.y)), w = int32_t(std::lround(v.width)),
        h = int32_t(std::lround(v.height));
    std::lock_guard lock(s_layerClipMutex);
    for (const auto& c : s_layerClips)
        if (c[0] == x && c[1] == y && c[2] == w && c[3] == h)
        {
            canvas = { c[4], c[5], c[6], c[7] };
            return true;
        }
    return false;
}

void RequestRingDump()
{
    if (RingSeconds() > 0.0)
        s_r.ringDump = true;
    else
        log::Info("F11: the frame ring is off (AC6_RING=<seconds> keeps the last seconds of frames)");
}

// The pipelines that draw into the window's image (its format: HDR, 10-bit or 8-bit SDR): the output pass's (when one
// of its settings is on) and the settings overlay's. At start, and again when the output changes while running (the
// overlay's live changes, the GPU waited for first).
void CreateWindowPipelines()
{
    RenderDevice& device = *g_video.device;
    s_r.outputPipeline.reset();
    s_r.outputTargetPipeline.reset();
    s_r.outputSimPipeline.reset();
    s_r.imguiPipeline.reset();
    s_r.imguiTargetPipeline.reset();
    // The output pass: only when one of its settings is on.
    if (OutputConfig().active)
    {
        if (!s_r.outputPs)
            s_r.outputPs = InternalShader("outputPs");
        if (s_r.outputPs)
        {
            RenderGraphicsPipelineDesc outputDesc;
            outputDesc.pipelineLayout = s_r.layout.get();
            outputDesc.vertexShader = s_r.copyVs.get();
            outputDesc.renderTargetBlend[0] = RenderBlendDesc::Copy();
            outputDesc.renderTargetCount = 1;
            outputDesc.pixelShader = s_r.outputPs.get();
            outputDesc.renderTargetFormat[0] = g_video.BackBufferFormat();
            s_r.outputPipeline = device.createGraphicsPipeline(outputDesc);
            outputDesc.renderTargetFormat[0] = kTargetFormat;
            s_r.outputTargetPipeline = device.createGraphicsPipeline(outputDesc);
            if (HdrSim())
            {
                outputDesc.renderTargetFormat[0] = RenderFormat::R16G16B16A16_FLOAT;
                s_r.outputSimPipeline = device.createGraphicsPipeline(outputDesc);
            }
            log::Info("Output pass: flags {:X}{}", OutputConfig().flags,
                g_video.OutputMode() == Video::Output::Hdr ? std::format(", paper white {:.0f} nits, peak {} nits, tone mapping {}, grading {:08X} {:06X}",
                    OutputConfig().paperScale * 80.0f, (OutputConfig().flags >> 24) * 50,
                    ((OutputConfig().flags >> 4) & 3) == 2 ? "RenoDRT" : "PsychoV-30", OutputConfig().grade0, OutputConfig().grade1) : std::string());
        }
    }

    // The settings overlay (src/ui/overlay.cpp): ImGui's vertices (position, texture coordinates, 8-bit colour: 20 bytes),
    // alpha blended into the window's image.
    if (s_r.imguiVs && s_r.imguiPs)
    {
        const RenderInputSlot slot(0, sizeof(ui::DrawVertex));
        const RenderInputElement elements[] = {
            RenderInputElement("TEXCOORD", 0, 0, RenderFormat::R32G32_FLOAT, 0, 0),
            RenderInputElement("TEXCOORD", 1, 1, RenderFormat::R32G32_FLOAT, 0, 8),
            RenderInputElement("TEXCOORD", 2, 2, RenderFormat::R8G8B8A8_UNORM, 0, 16),
        };
        RenderGraphicsPipelineDesc uiDesc;
        uiDesc.pipelineLayout = s_r.layout.get();
        uiDesc.vertexShader = s_r.imguiVs.get();
        uiDesc.pixelShader = s_r.imguiPs.get();
        uiDesc.inputSlots = &slot;
        uiDesc.inputSlotsCount = 1;
        uiDesc.inputElements = elements;
        uiDesc.inputElementsCount = uint32_t(std::size(elements));
        uiDesc.renderTargetFormat[0] = g_video.BackBufferFormat();
        uiDesc.renderTargetBlend[0] = RenderBlendDesc::AlphaBlend();
        uiDesc.renderTargetCount = 1;
        s_r.imguiPipeline = device.createGraphicsPipeline(uiDesc);
        uiDesc.renderTargetFormat[0] = kTargetFormat;
        s_r.imguiTargetPipeline = device.createGraphicsPipeline(uiDesc);
    }
    s_r.windowFormat = g_video.BackBufferFormat();
    s_r.windowOutputActive = OutputConfig().active;
}

bool InitRenderer()
{
    // The worker's copy of the guest device, at a guest address so the translation reads it
    // as it reads the device.
    if (void* image = s_deviceImage ? nullptr : g_heap.AllocPhysical(kDeviceImageBytes, 4096))
    {
        std::memset(image, 0, kDeviceImageBytes);
        s_deviceImage = g_memory.MapVirtual(image);
    }
    if (!g_renderQueue)
        g_renderQueue = new RenderQueue;
    if (!g_video.Ready())
        return false;
    RenderDevice& device = *g_video.device;
    breadcrumbs::Init(device);

    // Layout: sets 0-2 = the bindless texture heap (2D/3D/cube views of one set), set 3 = samplers, set 4 = the
    // constants as uniform buffers (SPIR-V backend), 28 bytes of push constants (VS/PS/shared constant addresses, as
    // XenosRecomp's shaders expect, and internal_shaders.hlsl PushConstants).
    RenderPipelineLayoutBuilder layoutBuilder;
    layoutBuilder.begin(false, true);
    RenderDescriptorSetBuilder setBuilder;
    setBuilder.begin();
    setBuilder.addTexture(0, kTextureDescriptors);
    setBuilder.end(true, kTextureDescriptors);
    s_r.textureSet = setBuilder.create(&device);
    layoutBuilder.addDescriptorSet(setBuilder);
    layoutBuilder.addDescriptorSet(setBuilder);
    layoutBuilder.addDescriptorSet(setBuilder);
    setBuilder.begin();
    setBuilder.addSampler(0, kSamplerDescriptors);
    setBuilder.end(true, kSamplerDescriptors);
    s_r.samplerSet = setBuilder.create(&device);
    layoutBuilder.addDescriptorSet(setBuilder);
    // Set 4: the SPIR-V backend's constants, each draw's from its offset in the upload ring.
    setBuilder.begin();
    for (uint32_t i = 0; i < 3; i++)
        setBuilder.addDynamicConstantBuffer(i);
    setBuilder.end();
    s_r.constantSet = setBuilder.create(&device);
    layoutBuilder.addDescriptorSet(setBuilder);
    RenderShaderStageFlags pushStages = RenderShaderStageFlag::VERTEX | RenderShaderStageFlag::PIXEL;
    if ((VolumetricsMode() || LateRectWanted()) && device.getCapabilities().geometryShader)
        pushStages |= RenderShaderStageFlag::GEOMETRY;   // the smoke harvest, the late mesh capture
    layoutBuilder.addPushConstant(0, 4, 28, pushStages);   // internal_shaders.hlsl PushConstants
    layoutBuilder.end();
    s_r.layout = layoutBuilder.create(&device);

    for (uint32_t i = 0; i < 3; i++)
    {
        RenderTextureDesc desc = RenderTextureDesc::Texture2D(1, 1, 1, RenderFormat::R8_UNORM);
        RenderTextureViewDesc viewDesc = RenderTextureViewDesc::Texture2D(RenderFormat::R8_UNORM);
        if (i == kNullTexture3D)
        {
            desc = RenderTextureDesc::Texture3D(1, 1, 1, 1, RenderFormat::R8_UNORM);
            viewDesc = RenderTextureViewDesc::Texture3D(RenderFormat::R8_UNORM);
        }
        else if (i == kNullTextureCube)
        {
            desc.arraySize = 6;
            desc.flags = RenderTextureFlag::CUBE;
            viewDesc = RenderTextureViewDesc::TextureCube(RenderFormat::R8_UNORM);
        }
        viewDesc.componentMapping = RenderComponentMapping(RenderSwizzle::ZERO, RenderSwizzle::ZERO, RenderSwizzle::ZERO, RenderSwizzle::ZERO);
        s_r.nullTextures[i] = MakeTexture(desc, "a null texture");
        s_r.nullViews[i] = s_r.nullTextures[i]->createTextureView(viewDesc);
        s_r.textureSet->setTexture(i, s_r.nullTextures[i].get(), RenderTextureLayout::SHADER_READ, s_r.nullViews[i].get());
    }

    RenderSamplerDesc clampDesc;
    clampDesc.addressU = clampDesc.addressV = clampDesc.addressW = RenderTextureAddressMode::CLAMP;
    s_r.samplers[0] = device.createSampler(clampDesc);
    s_r.samplers[1] = device.createSampler(RenderSamplerDesc());
    s_r.samplerSet->setSampler(0, s_r.samplers[0].get());
    s_r.samplerSet->setSampler(1, s_r.samplers[1].get());
    // The last slot: point, clamped (SMAA, the enhanced bloom's exposure, the planar glow, the lens effects, the motion
    // blur). Made here whatever the anti-aliasing: a sampler slot never written is undefined to sample through.
    {
        RenderSamplerDesc pointDesc;
        pointDesc.minFilter = pointDesc.magFilter = RenderFilter::NEAREST;
        pointDesc.addressU = pointDesc.addressV = pointDesc.addressW = RenderTextureAddressMode::CLAMP;
        s_r.pointSampler = device.createSampler(pointDesc);
        s_r.samplerSet->setSampler(kSamplerDescriptors - 1, s_r.pointSampler.get());
    }

    s_r.upload.size = 128ull << 20;
    s_r.upload.buffer = CreateUploadBuffer(s_r.upload.size * Video::kFramesInFlight,
        RenderBufferFlag::VERTEX | RenderBufferFlag::INDEX | RenderBufferFlag::CONSTANT | RenderBufferFlag::DEVICE_ADDRESSABLE, false);
    s_r.upload.memory = static_cast<uint8_t*>(s_r.upload.buffer->map());
    s_r.constantSet->setBuffer(0, s_r.upload.buffer.get(), kConstantBlockBytes[0]);
    s_r.constantSet->setBuffer(1, s_r.upload.buffer.get(), kConstantBlockBytes[1]);
    s_r.constantSet->setBuffer(2, s_r.upload.buffer.get(), kConstantBlockBytes[2]);
    s_r.upload.deviceAddress = s_r.upload.buffer->getDeviceAddress();
    log::Info("Upload ring memory: {}", g_video.DescribeMemory(s_r.upload.buffer.get()));
    g_memory.SetGpuFrameQuery([] { return g_video.FramesCompleted(); });

    s_r.resX = g_video.renderScaleX;
    s_r.resY = g_video.renderScaleY;
    s_r.targetWidth = g_video.frameWidth;
    s_r.targetHeight = g_video.frameHeight;
    s_r.sceneWidth = g_video.renderWidth;
    s_r.sceneHeight = g_video.renderHeight;
    s_r.uiResX = g_video.frameScaleX;
    s_r.uiResY = g_video.frameScaleY;
    s_r.sceneResX = g_video.renderScaleX;
    s_r.sceneResY = g_video.renderScaleY;
    s_r.target = MakeTexture(RenderTextureDesc::ColorTarget(s_r.targetWidth, s_r.targetHeight, kTargetFormat), "the frame's target");
    s_r.targetView = s_r.target->createTextureView(RenderTextureViewDesc::Texture2D(kTargetFormat));
    const RenderTexture* color = s_r.target.get();
    RenderFramebufferDesc fbDesc;
    fbDesc.colorAttachments = &color;
    fbDesc.colorAttachmentsCount = 1;
    s_r.targetFramebuffer = device.createFramebuffer(fbDesc);
    s_r.targetDescriptor = AllocateDescriptor();
    SetDescriptor(s_r.targetDescriptor, s_r.target.get(), s_r.targetView.get());

    s_r.copyVs = InternalShader("copyVs");
    s_r.copyPs = InternalShader("copyPs");
    if (!s_r.copyVs || !s_r.copyPs)
        return false;
    RenderGraphicsPipelineDesc copyDesc;
    copyDesc.pipelineLayout = s_r.layout.get();
    copyDesc.vertexShader = s_r.copyVs.get();
    copyDesc.pixelShader = s_r.copyPs.get();
    copyDesc.renderTargetFormat[0] = RenderFormat::B8G8R8A8_UNORM;
    copyDesc.renderTargetBlend[0] = RenderBlendDesc::Copy();
    copyDesc.renderTargetCount = 1;
    s_r.copyPipeline = device.createGraphicsPipeline(copyDesc);

    s_r.gammaPs = InternalShader("gammaPs");
    if (!s_r.gammaPs)
        return false;
    s_r.pillarFrostPs = InternalShader("pillarFrostPs");
    copyDesc.pixelShader = s_r.gammaPs.get();
    s_r.gammaPipeline = device.createGraphicsPipeline(copyDesc);
    // Into float images (HDR from the scene): the UI image's copy, the ramp continued past 1.
    s_r.gammaHdrPs = InternalShader("gammaHdrPs");
    if (s_r.gammaHdrPs)
    {
        RenderGraphicsPipelineDesc hdrDesc = copyDesc;
        hdrDesc.renderTargetFormat[0] = RenderFormat::R16G16B16A16_FLOAT;
        hdrDesc.pixelShader = s_r.copyPs.get();
        s_r.copyHdrPipeline = device.createGraphicsPipeline(hdrDesc);
        {
            // (A multisampled float layer's picture added to another.)
            s_r.glowMergePs = InternalShader("glowMergePs");
            RenderGraphicsPipelineDesc addDesc = hdrDesc;
            addDesc.pixelShader = s_r.glowMergePs ? s_r.glowMergePs.get() : nullptr;
            addDesc.renderTargetBlend[0].blendEnabled = true;
            addDesc.renderTargetBlend[0].srcBlend = addDesc.renderTargetBlend[0].dstBlend = RenderBlend::ONE;
            addDesc.renderTargetBlend[0].srcBlendAlpha = addDesc.renderTargetBlend[0].dstBlendAlpha = RenderBlend::ONE;
            if (s_r.glowMergePs)
                s_r.glowMergePipeline = device.createGraphicsPipeline(addDesc);
        }
        hdrDesc.pixelShader = s_r.gammaHdrPs.get();
        s_r.gammaHdrPipeline = device.createGraphicsPipeline(hdrDesc);
        // The ramp as an image for the output pass.
        s_r.gammaLutPs = InternalShader("gammaLutPs");
        if (s_r.gammaLutPs)
        {
            hdrDesc.pixelShader = s_r.gammaLutPs.get();
            s_r.gammaLutPipeline = device.createGraphicsPipeline(hdrDesc);
        }
        // The canvas composed after the scene composite: its lit highlights compressed, the effects' glow added.
        s_r.hdrComposePs = InternalShader("hdrComposePs");
        if (s_r.hdrComposePs)
        {
            hdrDesc.pixelShader = s_r.hdrComposePs.get();
            s_r.hdrComposePipeline = device.createGraphicsPipeline(hdrDesc);
        }
    }

    if (WaterSsr())
        s_r.waterSsrPs = InternalShader("waterSsrPs");
    if (PlanarSea())
        s_r.planarBlurPs = InternalShader("planarBlurPs");
    s_r.lensFxPs = InternalShader("lensFxPs");
    s_r.motionBlurPs = InternalShader("motionBlurPs");
    s_r.bloomBrightPs = InternalShader("bloomBrightPs");
    s_r.bloomDownPs = InternalShader("bloomDownPs");
    s_r.bloomUpPs = InternalShader("bloomUpPs");
    s_r.bloomApplyPs = InternalShader("bloomApplyPs");
    if (EmitterLight() > 0.0f)
    {
        s_r.lightDown0Ps = InternalShader("lightDown0Ps");
        s_r.lightDown0bPs = InternalShader("lightDown0bPs");
        s_r.lightDownPs = InternalShader("lightDownPs");
        s_r.lightApplyPs = InternalShader("lightApplyPs");
    }
    if (!s_r.motionCompletePs)
        s_r.motionCompletePs = InternalShader("motionCompletePs");
    // (Whatever the setting says now: it can change while running.)
    {
        s_r.aoDepthPs = InternalShader("aoDepthPs");
        s_r.aoDepthMipPs = InternalShader("aoDepthMipPs");
        s_r.aoMainPs = InternalShader("aoMainPs");
        s_r.aoDenoisePs = InternalShader("aoDenoisePs");
        s_r.aoApplyPs = InternalShader("aoApplyPs");
        s_r.aoSubjectPs = InternalShader("aoSubjectPs");
    }
    // Temporal AA: two targets, the scene image's format and the float history. Built unless FSR is
    // on (TAA can be turned on while running).
    if (!fsr::Requested())
    {
        s_r.taaPs = InternalShader("taaPs");
        if (s_r.taaPs)
        {
            RenderGraphicsPipelineDesc taaDesc = copyDesc;
            taaDesc.pixelShader = s_r.taaPs.get();
            taaDesc.renderTargetFormat[0] = RenderFormat::B8G8R8A8_UNORM;
            taaDesc.renderTargetFormat[1] = RenderFormat::R16G16B16A16_FLOAT;
            taaDesc.renderTargetBlend[1] = RenderBlendDesc::Copy();
            taaDesc.renderTargetCount = 2;
            s_r.taaPipeline = device.createGraphicsPipeline(taaDesc);
            taaDesc.renderTargetFormat[0] = RenderFormat::R16G16B16A16_FLOAT;
            s_r.taaPipelineFloat = device.createGraphicsPipeline(taaDesc);
            s_r.sharpenPs = InternalShader("sharpenPs");
            if (s_r.sharpenPs)
            {
                RenderGraphicsPipelineDesc sharpenDesc = copyDesc;
                sharpenDesc.pixelShader = s_r.sharpenPs.get();
                s_r.sharpenPipeline = device.createGraphicsPipeline(sharpenDesc);
                sharpenDesc.renderTargetFormat[0] = RenderFormat::R16G16B16A16_FLOAT;
                s_r.sharpenPipelineFloat = device.createGraphicsPipeline(sharpenDesc);
            }
            log::Info("Temporal AA {}(camera reprojection, variance clipping, motion-adaptive blend, sharpening)",
                TaaAtStart() ? "on the scene image " : "ready ");
        }
    }
    // FSR 3.1 (AC6_AA=fsr): the motion completion pass here, FSR's context at its first frame.
    if (fsr::Requested())
    {
        s_r.motionCompletePs = InternalShader("motionCompletePs");
        if (s_r.motionCompletePs)
        {
            RenderGraphicsPipelineDesc motionDesc = copyDesc;
            motionDesc.pixelShader = s_r.motionCompletePs.get();
            motionDesc.renderTargetFormat[0] = RenderFormat::R16G16_FLOAT;
            motionDesc.renderTargetFormat[1] = RenderFormat::R8_UNORM;   // the reactive mask
            motionDesc.renderTargetBlend[1] = RenderBlendDesc::Copy();
            motionDesc.renderTargetCount = 2;
            s_r.motionCompletePipeline = device.createGraphicsPipeline(motionDesc);
            s_r.fsr = s_r.motionCompletePipeline != nullptr;
            // FSR's sharpening in its copy back: TAA's sharpening pass.
            if (!s_r.sharpenPipeline)
            {
                s_r.sharpenPs = InternalShader("sharpenPs");
                if (s_r.sharpenPs)
                {
                    RenderGraphicsPipelineDesc sharpenDesc = copyDesc;
                    sharpenDesc.pixelShader = s_r.sharpenPs.get();
                    s_r.sharpenPipeline = device.createGraphicsPipeline(sharpenDesc);
                    sharpenDesc.renderTargetFormat[0] = RenderFormat::R16G16B16A16_FLOAT;
                    s_r.sharpenPipelineFloat = device.createGraphicsPipeline(sharpenDesc);
                }
            }
            // Upscaling: the scene's surfaces at the render ratio of the frame's scale, the textures
            // their draws sample a mip level finer (log2 of the ratio, less one: AMD's recommendation).
            float ratio = fsr::RenderRatio();
            // At least half the console's resolution on either axis (0.5 host pixels per guest pixel), as
            // AC6_RENDER_SCALE: Ultra performance at 1280x800 would ask 427x240.
            if (const float floor = 0.5f / std::min(s_r.sceneResX, s_r.sceneResY); ratio < floor)
            {
                log::Info("FSR 3.1: the quality's ratio {:.3f} is below the minimum render size (half the console's): {:.3f}",
                    ratio, std::min(1.0f, floor));
                ratio = std::min(1.0f, floor);
            }
            s_r.upscaling = s_r.fsr && ratio < 0.999f && !fsr::Late();
            s_r.worldResX = s_r.sceneResX * ratio;
            s_r.worldResY = s_r.sceneResY * ratio;
            s_r.worldMipBias = s_r.upscaling ? std::log2(ratio) - 1.0f : 0.0f;
            // AC6_FSR_LATE: the whole 3D frame is already at the render size (video.cpp), the world's textures sampled
            // finer as with the world's upscaling.
            s_r.lateUpscale = s_r.fsr && fsr::Late() && (s_r.sceneWidth < s_r.targetWidth || s_r.sceneHeight < s_r.targetHeight);
            s_r.lateNative = s_r.fsr && fsr::Late() && !s_r.lateUpscale;
            if (s_r.lateNative)
                log::Info("FSR 3.1 at the end of the 3D frame at {}x{} (anti-aliasing only)", s_r.sceneWidth, s_r.sceneHeight);
            if (s_r.lateUpscale)
            {
                const float lateRatio = float(s_r.sceneWidth) / float(s_r.targetWidth);
                s_r.worldMipBias = std::log2(lateRatio) - 1.0f;
                log::Info("FSR 3.1 upscaling at the end of the 3D frame: {}x{} for {}x{} (mip bias {:.2f})", s_r.sceneWidth,
                    s_r.sceneHeight, s_r.targetWidth, s_r.targetHeight, s_r.worldMipBias);
            }
            if (s_r.upscaling)
                log::Info("FSR 3.1 upscaling: the scene at {}x{} for {}x{} (ratio {:.3f}, mip bias {:.2f})",
                    std::lround(kGuestWidth * double(s_r.worldResX)), std::lround(kGuestHeight * double(s_r.worldResY)),
                    std::lround(kGuestWidth * double(s_r.sceneResX)), std::lround(kGuestHeight * double(s_r.sceneResY)), ratio, s_r.worldMipBias);
            else
                log::Info("FSR 3.1 on the scene image (AMD FidelityFX SDK 1.1.4, native anti-aliasing)");
        }
    }

    // The 2D's lines above the guest's size as rectangles (AC6_LINE_RECTS=0: wide lines).
    // (It reads the host grid from the shared block's uniform buffer.)
    if (device.getCapabilities().geometryShader && !HlslBackend() && UniformConstants() &&
        [] { const char* v = std::getenv("AC6_LINE_RECTS"); return !(v && *v == '0'); }())
        s_r.line2dGs = InternalShader("line2dGs");
    // The late pass's rectangle: the mesh capture stage and its buffers.
    if (LateRectWanted() && device.getCapabilities().geometryShader && UniformConstants())
    {
        s_r.lateBoundsGs = InternalShader("lateBoundsGs");
        if (s_r.lateBoundsGs)
        {
            s_r.lateCapture = device.createBuffer(RenderBufferDesc::DefaultBuffer(16384 * 16,
                RenderBufferFlag::STORAGE | RenderBufferFlag::UNORDERED_ACCESS | RenderBufferFlag::DEVICE_ADDRESSABLE));
            for (auto& readback : s_r.lateReadback)
                readback = device.createBuffer(RenderBufferDesc::ReadbackBuffer(16384 * 16));
        }
        log::Info("Late pass: {}", s_r.lateBoundsGs ? "the draws' rectangles only" : "the whole target (no capture shader)");
    }
    else if (LateRectWanted())
        log::Info("Late pass: the whole target ({})", !device.getCapabilities().geometryShader ? "no geometry shaders" : "the shader constants read through buffer addresses");
    // Volumetric smoke: geometry shaders (writing buffers) and the constants as uniform buffers.
    if (VolumetricsMode())
    {
        if (!device.getCapabilities().geometryShader || !UniformConstants())
            log::Warn("Volumetric smoke off: {}", !device.getCapabilities().geometryShader ? "no geometry shaders" : "the shader constants read through buffer addresses");
        else
        {
            s_r.volHarvestGs = InternalShader("volHarvestGs");
            s_r.volVs = InternalShader("volVs");
            s_r.volPs = InternalShader("volPs");
            s_r.volComposePs = InternalShader("volComposePs");
            s_r.volCloudVs = InternalShader("volCloudVs");
            s_r.volCloudMapPs = InternalShader("volCloudMapPs");
            s_r.volCloudBoundsPs = InternalShader("volCloudBoundsPs");
            s_r.volCloudMarchPs = InternalShader("volCloudMarchPs");
            s_r.volCloudTemporalPs = InternalShader("volCloudTemporalPs");
            s_r.volCloudOverPs = InternalShader("volCloudOverPs");
            s_r.volCloudScenePs = InternalShader("volCloudScenePs");
            s_r.volCloudDepthPs = InternalShader("volCloudDepthPs");
            s_r.volCloudDepthOutPs = InternalShader("volCloudDepthOutPs");
            s_r.volCloudGatePs = InternalShader("volCloudGatePs");
            for (auto& buffer : s_r.volParticles)
                buffer = device.createBuffer(RenderBufferDesc::DefaultBuffer(uint64_t(kVolCapacity) * kVolRecord,
                    RenderBufferFlag::STORAGE | RenderBufferFlag::UNORDERED_ACCESS | RenderBufferFlag::DEVICE_ADDRESSABLE));
            if (!s_r.volHarvestGs || !s_r.volVs || !s_r.volPs || !s_r.volComposePs)
            {
                s_r.volHarvestGs.reset();
                s_r.volVs.reset();
                log::Warn("Volumetric smoke off: shaders missing");
            }
            else
                log::Info("Volumetric smoke{}", VolumetricsMode() == 2 ? " (debug: the billboards kept)" : "");
        }
    }

    s_r.imguiVs = InternalShader("imguiVs");
    s_r.imguiPs = InternalShader("imguiPs");
    CreateWindowPipelines();
    if (!s_r.imguiPipeline)
        log::Warn("Settings overlay: no pipeline (its shaders are missing); F4 shows nothing");

    // SMAA: built unless FSR is on (it can be turned on while running).
    if (!fsr::Requested())
    {
        s_r.smaaEdgePs = InternalShader("smaaEdgePs");
        s_r.smaaWeightPs = InternalShader("smaaWeightPs");
        s_r.smaaBlendPs = InternalShader("smaaBlendPs");
        if (s_r.smaaEdgePs && s_r.smaaWeightPs && s_r.smaaBlendPs)
        {
            RenderGraphicsPipelineDesc smaaDesc = copyDesc;
            smaaDesc.pixelShader = s_r.smaaEdgePs.get();
            smaaDesc.renderTargetFormat[0] = RenderFormat::R8G8_UNORM;
            s_r.smaaEdgePipeline = device.createGraphicsPipeline(smaaDesc);
            smaaDesc.pixelShader = s_r.smaaWeightPs.get();
            smaaDesc.renderTargetFormat[0] = RenderFormat::R8G8B8A8_UNORM;
            s_r.smaaWeightPipeline = device.createGraphicsPipeline(smaaDesc);
            smaaDesc.pixelShader = s_r.smaaBlendPs.get();
            smaaDesc.renderTargetFormat[0] = RenderFormat::B8G8R8A8_UNORM;
            s_r.smaaBlendPipeline = device.createGraphicsPipeline(smaaDesc);
            smaaDesc.renderTargetFormat[0] = RenderFormat::R16G16B16A16_FLOAT;
            s_r.smaaBlendPipelineFloat = device.createGraphicsPipeline(smaaDesc);
            log::Info("SMAA {}(high preset)", (LiveAa() & 1) ? "on the scene image " : "ready ");
        }
        else
            log::Warn("SMAA: shaders missing; off");
    }

    s_r.depthCopyPs = InternalShader("depthCopyPs");
    if (!s_r.depthCopyPs)
        return false;
    copyDesc.pixelShader = s_r.depthCopyPs.get();
    copyDesc.renderTargetFormat[0] = RenderFormat::R32_FLOAT;
    s_r.depthCopyPipeline = device.createGraphicsPipeline(copyDesc);
    if (MsDepthResolve())
    {
        s_r.depthCopyMsPs = InternalShader("depthCopyMsPs");
        if (!s_r.depthCopyMsPs)
            return false;
        RenderGraphicsPipelineDesc msDesc = copyDesc;
        msDesc.pixelShader = s_r.depthCopyMsPs.get();
        msDesc.renderTargetFormat[1] = RenderFormat::R8_UNORM;
        msDesc.renderTargetBlend[1] = RenderBlendDesc::Copy();
        msDesc.renderTargetCount = 2;
        s_r.depthCopyMsPipeline = device.createGraphicsPipeline(msDesc);
        s_r.depthCopySsPs = InternalShader("depthCopySsPs");
        if (!s_r.depthCopySsPs)
            return false;
        msDesc.pixelShader = s_r.depthCopySsPs.get();
        s_r.depthCopySsPipeline = device.createGraphicsPipeline(msDesc);
    }

    s_r.depthResamplePs = InternalShader("depthResamplePs");
    if (!s_r.depthResamplePs)
        return false;
    RenderGraphicsPipelineDesc resampleDesc;
    resampleDesc.pipelineLayout = s_r.layout.get();
    resampleDesc.vertexShader = s_r.copyVs.get();
    resampleDesc.pixelShader = s_r.depthResamplePs.get();
    resampleDesc.renderTargetCount = 0;
    resampleDesc.depthTargetFormat = kDepthFormat;
    resampleDesc.depthEnabled = true;
    resampleDesc.depthWriteEnabled = true;
    resampleDesc.depthFunction = RenderComparisonFunction::ALWAYS;
    s_r.depthResamplePipeline = device.createGraphicsPipeline(resampleDesc);

    s_r.depthSwizzlePs = InternalShader("depthSwizzlePs");
    s_r.colorSwizzlePs = InternalShader("colorSwizzlePs");
    if (!s_r.depthSwizzlePs || !s_r.colorSwizzlePs)
        return false;
    resampleDesc.pixelShader = s_r.depthSwizzlePs.get();
    s_r.depthSwizzlePipeline = device.createGraphicsPipeline(resampleDesc);

    s_r.depthToColorPs = InternalShader("depthToColorPs");
    if (!s_r.depthToColorPs)
        return false;
    copyDesc.pixelShader = s_r.depthToColorPs.get();
    copyDesc.renderTargetFormat[0] = RenderFormat::B8G8R8A8_UNORM;
    s_r.depthToColorPipeline = device.createGraphicsPipeline(copyDesc);

    s_r.depthLoadPs = InternalShader("depthLoadPs");
    if (DrawCopies())
        s_r.texelCopyPs = InternalShader("texelCopyPs");
    s_r.colorLoadPs = InternalShader("colorLoadPs");
    s_r.stencilBitPs = InternalShader("stencilBitPs");
    if (!s_r.depthLoadPs || !s_r.colorLoadPs || !s_r.stencilBitPs)
        return false;

    // AC6_SCREENSHOT=300,900,...: those frames written to <log dir>/frame_<n>.bmp, at the frame's target size.
    if (const char* frames = std::getenv("AC6_SCREENSHOT"))
    {
        std::stringstream list(frames);
        for (std::string item; std::getline(list, item, ',');)
            if (!item.empty())
                s_r.screenshotFrames.insert(std::stoull(item));
    }
    if (const char* v = std::getenv("AC6_DUMP_RESOLVES"))
        s_r.dumpFrame = std::stoull(v);
    if (const char* v = std::getenv("AC6_RESOLVE_STATS"))   // <first frame>[:<frames>]
    {
        const std::string text(v);
        s_r.resolveStatsFrame = std::stoull(text);
        if (const size_t colon = text.find(':'); colon != std::string::npos)
            s_r.resolveStatsFrames = std::stoull(text.substr(colon + 1));
    }
    // (Also with the control channel, whose "shot" asks for one at any frame: RequestScreenshot.)
    if (!s_r.screenshotFrames.empty() || s_r.dumpFrame != UINT64_MAX || std::getenv("AC6_CONTROL"))
    {
        s_r.readback = device.createBuffer(RenderBufferDesc::ReadbackBuffer(uint64_t(s_r.targetWidth) * s_r.targetHeight * 4));
        s_r.rawReadback = device.createBuffer(RenderBufferDesc::ReadbackBuffer(uint64_t(s_r.targetWidth) * s_r.targetHeight * 8 * 2));
    }

    // Depth-only draws bind no pixel shader. They get an empty one with colour writes off.
    {
        HostShader& n = s_r.nullPixelShader;
        n.hash = 0x0000000000000001ull;
        n.idHash = n.hash;
        n.pixel = true;
        n.drawFlags = true;   // reads nothing
        if (const InternalShaderBlob* blob = FindInternalShader("nullPs"))
        {
            n.spirv.assign(reinterpret_cast<const uint32_t*>(blob->data), reinterpret_cast<const uint32_t*>(blob->data) + blob->size / 4);
            n.shader = device.createShader(blob->data, blob->size, "main", RenderShaderFormat::SPIRV);
            n.ok = true;
        }
    }

    s_r.ready = true;
    s_pipelineList.Load();
    log::Info("Renderer ready: guest target {}x{}, upload ring {} MB, {} byte swaps", s_r.targetWidth, s_r.targetHeight, s_r.upload.size >> 20,
        simd::PathName());
    return true;
}

void ClearImpl(uint32_t device, uint32_t flags, const float color[4], float z, uint32_t stencil)
{
    if (!s_r.ready)
        return;
    s_r.device = device;
    const uint32_t clearTarget = U32(device + kDevRenderTargets) ? U32(device + kDevRenderTargets) : U32(device + kDevDepthStencil);
    ScopedScale worldScale(WorldSurface(clearTarget), s_r.worldResX, s_r.worldResY);
    ScopedScale nativeScale(!WorldSurface(clearTarget) && NativeSurface(clearTarget), 1.0f, 1.0f);
    static std::set<uint32_t> seenFlags;
    if (seenFlags.size() < 16 && seenFlags.insert(flags).second)
        log::Info("Clear flags {:X} (z {}, stencil {})", flags, z, stencil);
    RenderCommandList& cl = g_video.Commands();
    {
        static const uint64_t traceFrame = [] { const char* v = std::getenv("AC6_DRAW_TRACE"); return v ? std::stoull(v) : UINT64_MAX; }();
        if (s_r.frame == traceFrame)
            log::Info("Clear: rt {:08X} ds {:08X} flags {:X}", U32(device + kDevRenderTargets), U32(device + kDevDepthStencil), flags);
    }
    Targets t;
    // A clear covers the whole surface (as on the console: the bound surfaces' EDRAM): what it overwrites isn't
    // brought in first.
    if (!BindTargets(cl, device, t, flags))
        return;
    if ((flags & kClearTargets) && t.color)
    {
        cl.clearColor(0, RenderColor(color[0], color[1], color[2], color[3]));
        NoteColorWrite(t.color);
        UnlinkStale(*t.color);
        if (t.color == s_r.taaSurface)
            s_r.motionClearPending = true;
    }
    if ((flags & (kClearDepth | kClearStencil)) && t.depth)
    {
        cl.clearDepthStencil((flags & kClearDepth) != 0, (flags & kClearStencil) != 0, z, stencil);
        NoteDepthWrite(t.depth);
    }
}

// (AC6_FSR_LATE: run by BeginUiImage, below.)
bool RunFsr(RenderCommandList& cl, HostImage& image, HostImage* input, HostImage* lateOut, bool late);

namespace {

// Whether a draw overwrites its colour target whole, asked only when that target is a swapped surface
// (AC6_RESOLVE_SWAP): the post chain's full-screen passes. Their vertex shaders, read in their HLSL: x and y straight
// from stream 0's first two floats (16-byte vertices), z a literal or c100.x, w 1. Their pixel shaders kill nothing
// but through the alpha test. Then: a rectangle of exactly the clip-space square's corners, a viewport over the
// target, and no state that can leave a pixel alone (blending, a partial write mask, the alpha test, stencil, a depth
// test that can fail, z outside the clip range).
bool OverwritesColor(uint32_t device, uint32_t primType, uint32_t firstVertex, uint32_t vertexCount, bool indexed,
    const HostShader& vs, const HostShader& ps)
{
    if (ResolveSwapMode() != 1 || primType != xenos::kPrimRectList || vertexCount != 3 || indexed)
        return false;
    float z = 0.0f;
    switch (vs.hash)
    {
    case 0x3A245E21E97A7428ull: z = 0.0f; break;   // the world compositor's
    case 0xD34AEB8CB26FA231ull: z = std::bit_cast<float>(0x3F7FBE77u); break;
    case 0xF67ACC9D6858BE31ull:
    case 0x6B6FBBD8930EFA4Dull:
    // The bloom, blur and exposure chains'.
    case 0x1E69729DE8C04E0Full:
    case 0x93A6F18170BA6E59ull:
    case 0x86AD041FA74D78D6ull:
    case 0x283D9FA4E34F3016ull:
    case 0x7DB6E665461263CAull:
    case 0xE92849A92A4685C9ull: z = std::bit_cast<float>(U32(device + kDevVsConstants + 100 * 16)); break;
    default: return false;
    }
    switch (ps.idHash)
    {
    case 0x8E36B699AF21554Aull: case 0xD02C9AE3A213CABBull: case 0x3C3AA92EFC4631A8ull: case 0x6C6D041CAB12E202ull:
    case 0x9825C7C374A8B834ull: case 0x3B7049BE4C307E6Aull: case 0x818FCD6341FBF2C8ull: case 0xD551A4748CFC27F2ull:
    case 0xF115E7B164AEBF81ull: case 0x3B72A8A6EBFFAB75ull: case 0x15E29341C450CAE5ull: case 0x49404B23F14C314Bull: break;
    default: return false;
    }
    HostImage* color = GetSurface(U32(device + kDevRenderTargets), false);
    if (!color || !color->staleHolder || color->samples != 1)
        return false;
    if (!(z >= 0.0f && z <= 1.0f))
        return false;
    if (U32(device + kDevBlendControl) != kBlendDisabled || (U32(device + kDevColorMask) & 0xF) != 0xF)
        return false;
    if (U32(device + kDevColorControl) & 0x18)   // alpha test, alpha to mask
        return false;
    const uint32_t depthControl = U32(device + kDevDepthControl);
    if ((depthControl & 1) || ((depthControl & 2) && ((depthControl >> 4) & 7) != 7))
        return false;
    const uint32_t fetch = device + kDevVertexFetch + 17 * 8;   // stream 0
    const xenos::VertexFetch vf{ { U32(fetch), U32(fetch + 4) } };
    if (!vf.Address())
        return false;
    const uint32_t base = PhysicalToGuest(vf.Address()) + firstVertex * 16;
    uint32_t corners = 0;
    for (uint32_t v = 0; v < 3; v++)
    {
        const uint32_t x = U32(base + v * 16), y = U32(base + v * 16 + 4);
        if ((x != 0xBF800000u && x != 0x3F800000u) || (y != 0xBF800000u && y != 0x3F800000u))
            return false;
        corners |= 1u << ((x >> 31) | (y >> 31) << 1);
    }
    if (std::popcount(corners) != 3)
        return false;
    Targets t;
    t.width = color->width;
    t.height = color->height;
    const RenderViewport vp = ReadViewport(device, t);
    const float gw = float(GuestWidth(*color)), gh = float(GuestHeight(*color));
    return vp.x <= 0.0f && vp.y <= 0.0f && vp.x + vp.width >= gw && vp.y + vp.height >= gh;
}

// Swaps a vertex stream's words to little-endian into the upload ring. Rect lists also get their fourth
// corner. The three corners arrive as v0, v1, v2 with the right angle at v1 (top-left, top-right,
// bottom-right on screen), so the missing corner is v3 = v0 + v2 - v1 for float elements (v1's value
// otherwise): triangles (v0, v1, v2), (v0, v2, v3).
void RunVertexCopy(const Renderer::VertexCopy& c)
{
    auto* words = reinterpret_cast<uint32_t*>(c.dst);
    if (c.point)
    {
        // Each vertex six times. Expanded in cached memory and written to the ring in one sequential copy:
        // the ring is write-combined GPU memory, where small or overlapping writes are slow (a per-word loop
        // costs ~0.6 us per vertex, 6.7 ms per frame in a particle-heavy scene).
        const uint32_t dw = c.stride / 4, used = std::min<uint32_t>(dw, 64);
        thread_local std::unique_ptr<uint32_t[]> swapped, expanded;
        thread_local size_t swappedCapacity = 0, expandedCapacity = 0;
        const size_t inWords = size_t(c.vertexCount) * dw, outWords = inWords * 6;
        if (swappedCapacity < inWords + 64)
        {
            swappedCapacity = (inWords + 64) * 2;
            swapped.reset(new uint32_t[swappedCapacity]);
        }
        if (expandedCapacity < outWords + 64)
        {
            expandedCapacity = (outWords + 64) * 2;
            expanded.reset(new uint32_t[expandedCapacity]);
        }
        const size_t validWords = std::min<uint64_t>(inWords, c.validBytes / 4);
        simd::ByteSwap32(swapped.get(), reinterpret_cast<const uint32_t*>(c.src), validWords);
        std::memset(swapped.get() + validWords, 0, (inWords - validWords) * 4);
        const uint32_t* in = swapped.get();
        uint32_t* out = expanded.get();
        if (dw == 1)
        {
            // One word per vertex (the common particle layout): 8 vertices become 48 words, six permutes.
            uint32_t v = 0;
#if defined(__AVX2__)
            const __m256i p0 = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 1, 1), p1 = _mm256_setr_epi32(1, 1, 1, 1, 2, 2, 2, 2);
            const __m256i p2 = _mm256_setr_epi32(2, 2, 3, 3, 3, 3, 3, 3), p3 = _mm256_setr_epi32(4, 4, 4, 4, 4, 4, 5, 5);
            const __m256i p4 = _mm256_setr_epi32(5, 5, 5, 5, 6, 6, 6, 6), p5 = _mm256_setr_epi32(6, 6, 7, 7, 7, 7, 7, 7);
            for (; v + 8 <= c.vertexCount; v += 8, in += 8, out += 48)
            {
                const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), _mm256_permutevar8x32_epi32(x, p0));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 8), _mm256_permutevar8x32_epi32(x, p1));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 16), _mm256_permutevar8x32_epi32(x, p2));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 24), _mm256_permutevar8x32_epi32(x, p3));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 32), _mm256_permutevar8x32_epi32(x, p4));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 40), _mm256_permutevar8x32_epi32(x, p5));
            }
#endif
            for (; v < c.vertexCount; v++, in++)
                for (uint32_t k = 0; k < 6; k++)
                    *out++ = *in;
        }
        else if (dw >= 2 && dw <= 16)
        {
            // Whole 32-byte stores, each overlapped by the next copy (both buffers have slack).
            for (uint32_t v = 0; v < c.vertexCount; v++, in += dw)
            {
#if defined(__AVX2__)
                const __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in));
                const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + 8));
                for (uint32_t k = 0; k < 6; k++, out += dw)
                {
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), x0);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 8), x1);
                }
#else
                __m128i x[4];
                for (uint32_t q = 0; q < 4; q++)
                    x[q] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + q * 4));
                for (uint32_t k = 0; k < 6; k++, out += dw)
                    for (uint32_t q = 0; q < 4; q++)
                        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + q * 4), x[q]);
#endif
            }
        }
        else
        {
            for (uint32_t v = 0; v < c.vertexCount; v++, in += dw)
                for (uint32_t k = 0; k < 6; k++, out += dw)
                {
                    std::memcpy(out, in, used * 4);
                    if (used < dw)
                        std::memset(out + used, 0, size_t(dw - used) * 4);
                }
        }
        simd::CopyToGpu(words, expanded.get(), outWords * 4);
        return;
    }
    if (!c.rect)
    {
        const uint32_t n = c.vertexCount * c.stride / 4;
        const uint32_t valid = uint32_t(std::min<uint64_t>(n, c.validBytes / 4));
        simd::ByteSwap32(words, reinterpret_cast<const uint32_t*>(c.src), valid);
        if (valid < n)
            std::memset(words + valid, 0, size_t(n - valid) * 4);
        return;
    }
    for (uint32_t r = 0; r + 3 <= c.vertexCount; r += 3)
    {
        uint32_t v[4][64];
        const uint32_t dw = std::min<uint32_t>(c.stride / 4, 64);
        for (uint32_t k = 0; k < 3; k++)
            for (uint32_t i = 0; i < dw; i++)
                v[k][i] = uint64_t(r + k) * c.stride + (i + 1) * 4 <= c.validBytes
                    ? __builtin_bswap32(reinterpret_cast<const uint32_t*>(c.src + uint64_t(r + k) * c.stride)[i]) : 0;
        std::memcpy(v[3], v[1], dw * 4);
        for (auto& e : *c.elements)
        {
            if (e.slotIndex != c.slot || !IsFloatFormat(e.format))
                continue;
            for (uint32_t k = 0; k < FormatBytes(e.format) / 4; k++)
            {
                const uint32_t i = e.alignedByteOffset / 4 + k;
                float f0, f1, f2;
                std::memcpy(&f0, &v[0][i], 4);
                std::memcpy(&f1, &v[1][i], 4);
                std::memcpy(&f2, &v[2][i], 4);
                float f3 = f0 + f2 - f1;
                std::memcpy(&v[3][i], &f3, 4);
            }
        }
        for (uint32_t k : { 0u, 1u, 2u, 0u, 2u, 3u })
        {
            std::memcpy(words, v[k], dw * 4);
            words += dw;
        }
    }
}

// Index lists, resized for every draw: without the zero-fill of each newly used element (every element
// is written before it is read).
template <class T>
struct NoInitAllocator : std::allocator<T>
{
    template <class U>
    struct rebind { using other = NoInitAllocator<U>; };
    template <class U, class... Args>
    void construct(U* p, Args&&... args) { if constexpr (sizeof...(Args) > 0) ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...); }
};
using IndexList = std::vector<uint32_t, NoInitAllocator<uint32_t>>;

// The first restart index (0xFFFFFFFF) at or after `from`, or `n`.
uint32_t NextRestart(const uint32_t* in, uint32_t from, uint32_t n)
{
#if defined(__AVX2__)
    const __m256i restart = _mm256_set1_epi32(-1);
    for (; from + 8 <= n; from += 8)
        if (const uint32_t hit = uint32_t(_mm256_movemask_ps(_mm256_castsi256_ps(
                _mm256_cmpeq_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + from)), restart)))))
            return from + std::countr_zero(hit);
#endif
    while (from < n && in[from] != 0xFFFFFFFFu)
        from++;
    return from;
}

// Writes `count` indices less `bias` (restarts, all ones, stay all ones) to the upload ring.
void UploadIndices(uint32_t* out, const uint32_t* in, size_t count, uint32_t bias)
{
    size_t k = 0;
#if defined(__AVX2__)
    const __m256i vbias = _mm256_set1_epi32(int32_t(bias)), all = _mm256_set1_epi32(-1);
    for (; k + 8 <= count; k += 8)
    {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + k));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + k), _mm256_or_si256(_mm256_sub_epi32(v, vbias), _mm256_cmpeq_epi32(v, all)));
    }
#endif
    for (; k < count; k++)
        out[k] = in[k] == 0xFFFFFFFFu ? 0xFFFFFFFFu : in[k] - bias;
}

// Inverts a 4x4 matrix (Gauss-Jordan elimination with partial pivoting). False when it is singular.
bool Invert4(const double m[16], double out[16])
{
    double a[4][8];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 8; c++)
            a[r][c] = c < 4 ? m[r * 4 + c] : (c - 4 == r ? 1.0 : 0.0);
    for (int c = 0; c < 4; c++)
    {
        int pivot = c;
        for (int r = c + 1; r < 4; r++)
            if (std::fabs(a[r][c]) > std::fabs(a[pivot][c]))
                pivot = r;
        if (std::fabs(a[pivot][c]) < 1e-12)
            return false;
        std::swap(a[c], a[pivot]);
        const double d = a[c][c];
        for (int k = 0; k < 8; k++)
            a[c][k] /= d;
        for (int r = 0; r < 4; r++)
            if (r != c)
            {
                const double f = a[r][c];
                for (int k = 0; k < 8; k++)
                    a[r][k] -= f * a[c][k];
            }
    }
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            out[r * 4 + c] = a[r][c + 4];
    return true;
}

// TAA's motion vectors: a scene draw's clip position to last frame's, inverse(this frame's
// transform) x last frame's (row vectors), into `out`, zeros when none is known (no transform, or the object wasn't drawn
// the frame before). Objects are what the draw draws (`object`: its shader, vertices and counts). An object drawn several
// times a frame (instances of a model) pairs with last frame's instance nearest it on screen (its clip-space origin), so
// a changed drawing order still pairs them.
void MotionTransform(uint64_t object, const double m[16], bool transform, float out[16])
{
    auto& current = s_r.motionCurrent;
    auto& previous = s_r.motionPrevious;
    if (current.frame != s_r.frame)
    {
        std::swap(previous, current);
        current.byObject.clear();
        current.frame = s_r.frame;
        std::memset(s_r.motionStats, 0, sizeof(s_r.motionStats));
    }
    std::memset(out, 0, 16 * sizeof(float));
    s_r.motionStats[0]++;
    if (!transform)
        return;
    s_r.motionStats[1]++;
    std::array<double, 16> matrix;
    std::memcpy(matrix.data(), m, sizeof(matrix));
    current.byObject[object].push_back(matrix);
    if (previous.frame + 1 != s_r.frame)
        return;
    const auto it = previous.byObject.find(object);
    if (it == previous.byObject.end())
        return;
    s_r.motionStats[2]++;
    const std::array<double, 16>* last = &it->second[0];
    if (it->second.size() > 1)
    {
        double best = 1e300;
        for (const auto& candidate : it->second)
        {
            double d = 0.0;
            for (int i = 12; i < 16; i++)
                d += (candidate[i] - m[i]) * (candidate[i] - m[i]);
            if (d < best)
            {
                best = d;
                last = &candidate;
            }
        }
    }
    // Most draws pair the same two transforms as the draw before (the world's, frame to frame).
    static std::array<double, 16> cachedThis{}, cachedLast{};
    static float cachedOut[16] = {};
    static bool cached = false;
    if (cached && cachedThis == matrix && cachedLast == *last)
    {
        std::memcpy(out, cachedOut, sizeof(cachedOut));
        return;
    }
    double inverse[16];
    if (!Invert4(m, inverse))
        return;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < 4; k++)
                sum += inverse[i * 4 + k] * (*last)[k * 4 + j];
            out[i * 4 + j] = float(sum);
        }
    cachedThis = matrix;
    cachedLast = *last;
    std::memcpy(cachedOut, out, sizeof(cachedOut));
    cached = true;
}

// Volumetric smoke. The frame's records start empty. Each smoke draw reserves its particles' slots
// (in draw order) and leaves the camera, fog colour and sun it was drawn with.
RenderBuffer* VolParticles()
{
    return s_r.volParticles[s_r.frame % std::size(s_r.volParticles)].get();
}

void VolBeginFrame()
{
    if (s_r.volFrame == s_r.frame)
        return;
    s_r.volFrame = s_r.frame;
    s_r.volCount = 0;
    s_r.volComposited = false;
    s_r.volTarget = nullptr;
}

void VolNoteConstants(uint32_t device, int kind)
{
    auto reg = [&](uint32_t r, uint32_t c) { return std::bit_cast<float>(U32(device + kDevVsConstants + r * 16 + c * 4)); };
    // The camera from the kinds whose vertex shaders take ViewProjectionMatrix (c116): smoke and fire. The clouds' take
    // a view (c112) and a projection (c40), the trails' a world-view-projection (c120). The records of all are
    // unprojected with the one camera (checked once against the clouds' own below).
    if (kind == 4)
    {
        if (s_r.volCloudFrame != s_r.frame)
        {
            s_r.volCloudFrame = s_r.frame;
            for (uint32_t r = 0; r < 19; r++)
                for (uint32_t c = 0; c < 4; c++)
                    s_r.volCloud[r][c] = reg(80 + r, c);
            for (uint32_t c = 0; c < 4; c++)
            {
                s_r.volCloudFade[c] = reg(50, c);
                s_r.volSun[c] = reg(98, c);
            }
        }
        // Their view x projection (V x P: rows, as c116) is the smoke's camera (0.0002 apart in mission 01), so
        // a frame with clouds and no smoke still has its camera.
        double cloudVp[16];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < 4; k++)
                    sum += double(reg(112 + i, k)) * double(reg(40 + k, j));
                cloudVp[i * 4 + j] = sum;
            }
        if (std::fabs(cloudVp[3]) + std::fabs(cloudVp[7]) + std::fabs(cloudVp[11]) > 1e-6 && std::fabs(cloudVp[14]) > 1e-6 &&
            s_r.volVpFrame != s_r.frame)
        {
            std::memcpy(s_r.volVp, cloudVp, sizeof(cloudVp));
            s_r.volHaveVp = true;
        }
        static uint32_t checked = 0;
        if (s_r.volHaveVp && s_r.volVpFrame == s_r.frame && checked++ < 3)
        {
            double view[16], projection[16], worst[2] = {};
            for (uint32_t i = 0; i < 16; i++)
            {
                view[i] = double(reg(112 + i / 4, i % 4));
                projection[i] = double(reg(40 + i / 4, i % 4));
            }
            // V x P and P x V (whichever order the registers hold them in).
            for (int order = 0; order < 2; order++)
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                    {
                        double sum = 0.0;
                        for (int k = 0; k < 4; k++)
                            sum += order ? projection[i * 4 + k] * view[k * 4 + j] : view[i * 4 + k] * projection[k * 4 + j];
                        worst[order] = std::max(worst[order], std::fabs(sum - s_r.volVp[i * 4 + j]) / std::max(1e-3, std::fabs(s_r.volVp[i * 4 + j])));
                    }
            log::Info("Volumetric clouds: their view and projection against the smoke's camera: worst relative difference {:.4f} "
                "(V x P), {:.4f} (P x V)", worst[0], worst[1]);
        }
        return;
    }
    if (kind > 3)
        return;
    double m[16];
    for (uint32_t i = 0; i < 16; i++)
        m[i] = double(reg(116 + i / 4, i % 4));
    // ViewProjectionMatrix as a perspective has it (the world's: reversed depth, constant clip z).
    if (std::fabs(m[3]) + std::fabs(m[7]) + std::fabs(m[11]) > 1e-6 && std::fabs(m[14]) > 1e-6)
    {
        std::memcpy(s_r.volVp, m, sizeof(m));
        s_r.volHaveVp = true;
        s_r.volVpFrame = s_r.frame;
    }
    if (kind > 1)
        return;
    static uint32_t logged = 0;
    for (uint32_t c = 0; c < 4; c++)
    {
        // FogColor_Param as the smoke's pixel shaders read it (their own c97), SpriteLightVector_LightAngle (VS c98).
        s_r.volFog[c] = std::bit_cast<float>(U32(device + kDevPsConstants + 97 * 16 + c * 4));
        s_r.volSun[c] = reg(98, c);
    }
    if (logged++ < 2)
        log::Info("Volumetric smoke: fog colour PS {:.3f} {:.3f} {:.3f}, VS {:.3f} {:.3f} {:.3f}; sun {:.3f} {:.3f} {:.3f} {:.3f}", s_r.volFog[0],
            s_r.volFog[1], s_r.volFog[2], reg(97, 0), reg(97, 1), reg(97, 2), s_r.volSun[0], s_r.volSun[1], s_r.volSun[2], s_r.volSun[3]);
}

// A tiling 64^3 noise volume (Perlin-Worley-like: value-noise octaves over inverted cellular noise), made once.
void VolEnsureNoise(RenderCommandList& cl)
{
    if (s_r.volNoise)
        return;
    constexpr int n = 64;
    auto hash = [](int x, int y, int z, int seed) {
        uint32_t h = uint32_t(x) * 73856093u ^ uint32_t(y) * 19349663u ^ uint32_t(z) * 83492791u ^ uint32_t(seed) * 2654435761u;
        h ^= h >> 13;
        h *= 0x5BD1E995u;
        h ^= h >> 15;
        return float(h & 0xFFFFFF) / float(0x1000000);
    };
    auto value = [&](float x, float y, float z, int period, int seed) {
        const int x0 = int(std::floor(x)), y0 = int(std::floor(y)), z0 = int(std::floor(z));
        const float fx = x - x0, fy = y - y0, fz = z - z0;
        auto s = [](float t) { return t * t * (3.0f - 2.0f * t); };
        auto at = [&](int i, int j, int k) {
            return hash(((x0 + i) % period + period) % period, ((y0 + j) % period + period) % period, ((z0 + k) % period + period) % period, seed);
        };
        const float sx = s(fx), sy = s(fy), sz = s(fz);
        auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };
        return lerp(lerp(lerp(at(0, 0, 0), at(1, 0, 0), sx), lerp(at(0, 1, 0), at(1, 1, 0), sx), sy),
            lerp(lerp(at(0, 0, 1), at(1, 0, 1), sx), lerp(at(0, 1, 1), at(1, 1, 1), sx), sy), sz);
    };
    auto worley = [&](float x, float y, float z, int period, int seed) {
        const int cx = int(std::floor(x)), cy = int(std::floor(y)), cz = int(std::floor(z));
        float best = 4.0f;
        for (int k = -1; k <= 1; k++)
            for (int j = -1; j <= 1; j++)
                for (int i = -1; i <= 1; i++)
                {
                    const int gx = cx + i, gy = cy + j, gz = cz + k;
                    const int wx = (gx % period + period) % period, wy = (gy % period + period) % period, wz = (gz % period + period) % period;
                    const float px = gx + hash(wx, wy, wz, seed), py = gy + hash(wx, wy, wz, seed + 1), pz = gz + hash(wx, wy, wz, seed + 2);
                    best = std::min(best, (px - x) * (px - x) + (py - y) * (py - y) + (pz - z) * (pz - z));
                }
        return 1.0f - std::min(std::sqrt(best), 1.0f);
    };
    // Two channels: the Perlin-Worley blend (the shapes), the cellular noise alone (the cloudscape's billows).
    std::vector<uint8_t> bytes(size_t(n) * n * n * 2);
    for (int z = 0; z < n; z++)
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
            {
                float v = 0.0f, amplitude = 0.5f, total = 0.0f;
                for (int o = 0; o < 4; o++)
                {
                    const int period = 4 << o;
                    const float k = float(period) / n;
                    v += amplitude * value(x * k, y * k, z * k, period, o * 7);
                    total += amplitude;
                    amplitude *= 0.5f;
                }
                v /= total;
                float w = 0.0f;
                for (int o = 0; o < 3; o++)
                {
                    const int period = 4 << o;
                    const float k = float(period) / n;
                    w += worley(x * k, y * k, z * k, period, 100 + o * 11) * (o == 0 ? 0.625f : o == 1 ? 0.25f : 0.125f);
                }
                // Perlin-Worley: the value noise remapped by the cellular noise (billowy edges).
                const float pw = std::clamp((v - (w - 1.0f)) / (1.0f - (w - 1.0f)), 0.0f, 1.0f);
                bytes[((size_t(z) * n + y) * n + x) * 2] = uint8_t(std::lround(std::clamp(pw * 0.6f + w * 0.4f, 0.0f, 1.0f) * 255.0f));
                bytes[((size_t(z) * n + y) * n + x) * 2 + 1] = uint8_t(std::lround(std::clamp(w, 0.0f, 1.0f) * 255.0f));
            }
    // With its mip levels (box filtered: the noise tiles): the cloudscape reads the level a sample's size on screen
    // asks for (far clouds' fine noise averaged, not aliased).
    constexpr uint32_t levels = 7;
    std::vector<std::vector<uint8_t>> mips(levels);
    mips[0] = std::move(bytes);
    for (uint32_t l = 1; l < levels; l++)
    {
        const int m = n >> l, big = m * 2;
        mips[l].resize(size_t(m) * m * m * 2);
        for (int z = 0; z < m; z++)
            for (int y = 0; y < m; y++)
                for (int x = 0; x < m; x++)
                    for (int c = 0; c < 2; c++)
                    {
                        uint32_t sum = 0;
                        for (int k = 0; k < 8; k++)
                            sum += mips[l - 1][((size_t(z * 2 + (k >> 2)) * big + (y * 2 + ((k >> 1) & 1))) * big + (x * 2 + (k & 1))) * 2 + c];
                        mips[l][((size_t(z) * m + y) * m + x) * 2 + c] = uint8_t((sum + 4) / 8);
                    }
    }
    UploadRing::Allocation allocations[levels];
    for (uint32_t l = 0; l < levels; l++)
    {
        allocations[l] = s_r.upload.Allocate(mips[l].size(), 512);
        if (!allocations[l].memory)
            return;
        std::memcpy(allocations[l].memory, mips[l].data(), mips[l].size());
    }
    s_r.volNoise = MakeTexture(RenderTextureDesc::Texture3D(n, n, n, levels, RenderFormat::R8G8_UNORM), "the smoke noise");
    s_r.volNoiseView = s_r.volNoise->createTextureView(RenderTextureViewDesc::Texture3D(RenderFormat::R8G8_UNORM));
    s_r.volNoiseDescriptor = AllocateDescriptor();
    SetDescriptor(s_r.volNoiseDescriptor, s_r.volNoise.get(), s_r.volNoiseView.get());
    cl.barriers(RenderBarrierStage::COPY, RenderTextureBarrier(s_r.volNoise.get(), RenderTextureLayout::COPY_DEST));
    for (uint32_t l = 0; l < levels; l++)
    {
        const uint32_t m = uint32_t(n) >> l;
        cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(s_r.volNoise.get(), l),
            RenderTextureCopyLocation::PlacedFootprint(s_r.upload.buffer.get(), RenderFormat::R8G8_UNORM, m, m, m, m, allocations[l].offset));
    }
    cl.barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(s_r.volNoise.get(), RenderTextureLayout::SHADER_READ));
}

// Before the effects composite (outside its render pass): the records written by the smoke draws become readable.
void VolBeforeComposite(RenderCommandList& cl)
{
    cl.setFramebuffer(nullptr);
    s_r.markedFramebuffer = nullptr;
    // plume's first barrier on a buffer waits for nothing (its previous stages unknown): the second waits for graphics.
    cl.barriers(RenderBarrierStage::GRAPHICS, RenderBufferBarrier(VolParticles(), RenderBufferAccess::READ));
    cl.barriers(RenderBarrierStage::GRAPHICS, RenderBufferBarrier(VolParticles(), RenderBufferAccess::READ));
    VolEnsureNoise(cl);
}

constexpr uint32_t kVolParamRows = 53;
constexpr uint32_t kCloudMapTexels = 512, kCloudMapLevels = 4, kCloudCoarseTexels = 128;

// AC6_VOL_CLOUDSCAPE=0: the cloud puffs each a volume of its own instead of the cloudscape.
bool CloudscapeWanted()
{
    static const bool on = [] { const char* v = std::getenv("AC6_VOL_CLOUDSCAPE"); return !(v && *v == '0'); }();
    return on;
}

// AC6_CLOUD_AMORTIZE=1: the cloudscape's march over every pixel of its target each frame. By default (2) over one in
// each 2 x 2, another each frame (a quarter of the cost, the temporal pass keeps the rest from the last frames).
uint32_t CloudMarchStep()
{
    static const uint32_t step = VolSetting("AC6_CLOUD_AMORTIZE", 2.0f) >= 2.0f ? 2 : 1;
    return step;
}

// The sub-pixel jitter the scene (and its depth) was drawn with this frame, in its pixels (TAA, FSR), for the passes that
// read the scene's depth while drawing into the resolved, steady picture.
std::pair<float, float> SceneJitter()
{
    static const bool on = [] { const char* v = std::getenv("AC6_VOL_JITTER"); return !(v && *v == '0'); }();
    if (!on || !(TaaOn() || FsrJitters()) || !TaaSurfaceLive())
        return { 0.0f, 0.0f };
    return TaaJitter(SeedFrame());
}

// AC6_CLOUD_EARLY=0: the cloudscape over the resolved scene at the effects composite instead of into the scene's
// surface before its resolve.
bool CloudscapeEarly()
{
    static const bool on = [] { const char* v = std::getenv("AC6_CLOUD_EARLY"); return !(v && *v == '0'); }();
    return on && CloudscapeWanted() && s_r.volCloudScenePs != nullptr;
}

// A perspective draw into the scene gave this view-projection (c116-c119): the frame's camera is the one most of them
// use (SceneReprojection's vote), kept here too when neither TAA nor FSR is on, for the early cloudscape.
void NoteSceneVp(const double m[16])
{
    if (s_r.taaVpFrame != s_r.frame)
        s_r.taaVpCandidates.clear();
    s_r.taaVpFrame = s_r.frame;
    auto& candidates = s_r.taaVpCandidates;
    auto same = [&](const Renderer::TaaVpCandidate& c) { return std::memcmp(c.m, m, 16 * sizeof(double)) == 0; };
    if (!candidates.empty() && same(candidates.back()))
        candidates.back().draws++;
    else if (auto it = std::find_if(candidates.begin(), candidates.end(), same); it != candidates.end())
        it->draws++;
    else if (candidates.size() < 256)
    {
        candidates.push_back({ {}, 1 });
        std::memcpy(candidates.back().m, m, 16 * sizeof(double));
    }
}

// The frame's camera, lighting and tuning for the volume passes (volumetrics.hlsl's P rows), for a target that size.
// history: the cloudscape has a usable last frame, cloudDone: the cloudscape's result is made (row 37's third texture is
// then this frame's result, for the volumes behind clouds. Otherwise the last frame's, for the temporal pass).
// viewVp: the camera the passes look through (null: the frame's harvest's, volVp), recordVp, recordFrame: the camera and
// the frame the records were harvested in (null: the same. The cloudscape before the scene's resolve uses the
// last frame's records with this frame's camera).
UploadRing::Allocation VolUploadParams(uint32_t width, uint32_t height, bool floatScene = false, bool history = false, bool cloudDone = true,
    double* cameraOut = nullptr, const double* viewVp = nullptr, const double* recordVp = nullptr, uint64_t recordFrame = ~0ull)
{
    const UploadRing::Allocation none{};
    if (!viewVp)
        viewVp = s_r.volVp;
    if (!recordVp)
        recordVp = viewVp;
    if (recordFrame == ~0ull)
        recordFrame = s_r.frame;
    double inverse[16], recordInverse[16];
    if (!Invert4(viewVp, inverse) || !Invert4(recordVp, recordInverse))
        return none;
    // Camera: clip = [x y z 1] VP (rows). The camera is where clip = (0, 0, z, 0), the inverse's third row.
    auto unproject = [&](double x, double y, double z, double out[3]) {
        double h[4];
        for (int c = 0; c < 4; c++)
            h[c] = x * inverse[c] + y * inverse[4 + c] + z * inverse[8 + c] + inverse[12 + c];
        for (int c = 0; c < 3; c++)
            out[c] = h[c] / h[3];
    };
    double camera[3], centre[3], right[3], up[3], forward[3];
    for (int c = 0; c < 3; c++)
        camera[c] = inverse[8 + c] / inverse[11];
    unproject(0.0, 0.0, 0.5, centre);
    unproject(1.0, 0.0, 0.5, right);
    unproject(0.0, 1.0, 0.5, up);
    auto normalize = [](double v[3]) {
        const double l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        for (int c = 0; c < 3; c++)
            v[c] = l > 0.0 ? v[c] / l : 0.0;
    };
    for (int c = 0; c < 3; c++)
    {
        right[c] -= centre[c];
        up[c] -= centre[c];
    }
    // A target pixel's angle (radians, small): the view's half width at that depth over its distance, over half the width.
    const double pixelAngle = std::sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]) /
        std::max(1e-9, std::sqrt((centre[0] - camera[0]) * (centre[0] - camera[0]) + (centre[1] - camera[1]) * (centre[1] - camera[1]) +
            (centre[2] - camera[2]) * (centre[2] - camera[2]))) * 2.0 / std::max(1u, width);
    normalize(right);
    normalize(up);
    for (int c = 0; c < 3; c++)
        forward[c] = centre[c] - camera[c];   // through the screen centre
    normalize(forward);
    // The records' camera: its eye, and its axes through the screen's centre.
    double recordEye[3], recordRight[3], recordUp[3], recordForward[3];
    {
        auto at = [&](double x, double y, double out[3]) {
            double h[4];
            for (int c = 0; c < 4; c++)
                h[c] = x * recordInverse[c] + y * recordInverse[4 + c] + 0.5 * recordInverse[8 + c] + recordInverse[12 + c];
            for (int c = 0; c < 3; c++)
                out[c] = h[c] / h[3];
        };
        double middle[3];
        for (int c = 0; c < 3; c++)
            recordEye[c] = recordInverse[8 + c] / recordInverse[11];
        at(0.0, 0.0, middle);
        at(1.0, 0.0, recordRight);
        at(0.0, 1.0, recordUp);
        for (int c = 0; c < 3; c++)
        {
            recordRight[c] -= middle[c];
            recordUp[c] -= middle[c];
            recordForward[c] = middle[c] - recordEye[c];
        }
        normalize(recordRight);
        normalize(recordUp);
        normalize(recordForward);
    }
    // SpriteLightVector (c98) is in view space: x right + y up - z forward is the same world vector from every camera.
    // Read as a world vector the light would turn with the camera.
    // (The view of the frame it was read in: the records'.)
    double sun[3];
    for (int c = 0; c < 3; c++)
        sun[c] = s_r.volSun[0] * recordRight[c] + s_r.volSun[1] * recordUp[c] - s_r.volSun[2] * recordForward[c];
    normalize(sun);
    static const float sunSign = VolSetting("AC6_VOL_SUN_SIGN", 1.0f);
    static const float sizeScale = VolSetting("AC6_VOL_SIZE", 1.1f), density = VolSetting("AC6_VOL_DENSITY", 16.0f);
    static const float steps = VolSetting("AC6_VOL_STEPS", 5.0f), shadow = VolSetting("AC6_VOL_SHADOW", 1.5f);
    static const float noiseScale = VolSetting("AC6_VOL_NOISE", 0.6f), detail = VolSetting("AC6_VOL_DETAIL", 0.6f);
    static const float erosion = VolSetting("AC6_VOL_FALLOFF", 1.0f), floor = VolSetting("AC6_VOL_SHADOW_FLOOR", 0.55f);
    // AC6_VOL_TRAIL_SIZE: the trails' tube radius over the ribbon's half width. AC6_VOL_GLOW: the additive kinds' glow
    // over their billboards' mean. AC6_VOL_TRAIL_NOISE: the trails' noise cell (world units). AC6_VOL_SHOW=<mask>: the
    // kinds drawn (diagnostics, all by default).
    static const float trailSize = VolSetting("AC6_VOL_TRAIL_SIZE", 0.8f), glow = VolSetting("AC6_VOL_GLOW", 4.0f);
    static const float trailNoise = VolSetting("AC6_VOL_TRAIL_NOISE", 12.0f);
    static const uint32_t show = uint32_t(VolSetting("AC6_VOL_SHOW", 127.0f));
    // The cloud puffs are the cloudscape's when it ran this frame.
    const uint32_t shown = s_r.cloudscapeFrame == s_r.frame ? show & ~0x10u : show;
    auto params = s_r.upload.Allocate(kVolParamRows * 16, 16);
    if (!params.memory)
        return none;
    float* f = reinterpret_cast<float*>(params.memory);
    for (int i = 0; i < 16; i++)
    {
        f[i] = float(inverse[i]);
        f[16 + i] = float(viewVp[i]);
    }
    const float block[] = {
        float(camera[0]), float(camera[1]), float(camera[2]), float(double(SeedFrame() % (1ull << 24)) / 60.0),
        float(sun[0] * sunSign), float(sun[1] * sunSign), float(sun[2] * sunSign), sizeScale,
        s_r.volFog[0], s_r.volFog[1], s_r.volFog[2], density,
        float(right[0]), float(right[1]), float(right[2]), std::clamp(steps, 2.0f, 64.0f),
        float(up[0]), float(up[1]), float(up[2]), shadow,
        float(width), float(height), 1.0f / float(width), 1.0f / float(height),
        noiseScale, detail, erosion, floor,
        std::bit_cast<float>(uint32_t(recordFrame & 0xFFFFFF)), VolSetting("AC6_VOL_VIEW", 0.0f), 0.0f, 0.0f,   // the records' frame tag, a debug view
        float(forward[0]), float(forward[1]), float(forward[2]), 0.0f,
        trailSize, glow, trailNoise, std::bit_cast<float>(shown),
    };
    std::memcpy(f + 32, block, sizeof(block));
    if (cameraOut)
        std::copy(camera, camera + 3, cameraOut);
    // The scene's depth as a view depth: clip z is a x (view depth) + b along the view (two points on its axis), so
    // the depth buffer's z is a + b / (view depth).
    {
        double zw[2][2];
        for (int k = 0; k < 2; k++)
        {
            const double along = k ? 4.0 : 1.0;   // the point unprojected, and one four times as far from the eye
            double q[3];
            for (int c = 0; c < 3; c++)
                q[c] = camera[c] + (centre[c] - camera[c]) * along;
            zw[k][0] = q[0] * viewVp[2] + q[1] * viewVp[6] + q[2] * viewVp[10] + viewVp[14];
            zw[k][1] = q[0] * viewVp[3] + q[1] * viewVp[7] + q[2] * viewVp[11] + viewVp[15];
        }
        s_r.volDepthA = 0.0f;
        s_r.volDepthB = 1.0f;
        if (std::fabs(zw[1][1] - zw[0][1]) > 1e-9)
        {
            s_r.volDepthA = float((zw[1][0] - zw[0][0]) / (zw[1][1] - zw[0][1]));
            s_r.volDepthB = float(zw[0][0] - double(s_r.volDepthA) * zw[0][1]);
        }
    }
    // The cloudscape's rows (volumetrics.hlsl): the puffs' constants (c80-c98 as their draw had them), the map's levels
    // around the camera (each centred on a whole texel of its own: the map doesn't swim as the camera moves), the tuning.
    const bool relight = HdrSceneOn() && Relight().on;
    const float sunK = relight ? Relight().k[2] : 1.0f, ambientK = relight ? Relight().k[3] : 1.0f, fogK = relight ? Relight().k[6] : 1.0f;
    const auto& cc = s_r.volCloud;
    static const float extents[kCloudMapLevels] = { VolSetting("AC6_CLOUD_EXTENT0", 4000.0f), VolSetting("AC6_CLOUD_EXTENT1", 12000.0f),
        VolSetting("AC6_CLOUD_EXTENT2", 32000.0f), VolSetting("AC6_CLOUD_EXTENT3", 64000.0f) };
    float centres[kCloudMapLevels][2];
    for (uint32_t l = 0; l < kCloudMapLevels; l++)
    {
        const double texel = 2.0 * extents[l] / kCloudMapTexels;
        centres[l][0] = float(std::round(camera[0] / texel) * texel);
        centres[l][1] = float(std::round(camera[2] / texel) * texel);
    }
    static const float cloudSize = VolSetting("AC6_CLOUD_SIZE", 1.0f), cloudCoverage = VolSetting("AC6_CLOUD_COVERAGE", 3.0f);
    static const float cloudTop = VolSetting("AC6_CLOUD_TOP", 0.75f), cloudBottom = VolSetting("AC6_CLOUD_BOTTOM", 1.0f);
    static const float cloudDensity = VolSetting("AC6_CLOUD_DENSITY", 0.03f), cloudNoise = VolSetting("AC6_CLOUD_NOISE", 1100.0f);
    static const float cloudDetailSize = VolSetting("AC6_CLOUD_DETAIL_SIZE", 500.0f), cloudDetail = VolSetting("AC6_CLOUD_DETAIL", 0.8f);
    static const float cloudSteps = VolSetting("AC6_CLOUD_STEPS", 128.0f), cloudStep = VolSetting("AC6_CLOUD_STEP", 25.0f);
    static const float cloudGrowth = VolSetting("AC6_CLOUD_STEP_GROWTH", 0.002f), cloudStride = VolSetting("AC6_CLOUD_STRIDE", 4.0f);
    static const float cloudSun = VolSetting("AC6_CLOUD_SUN", 0.7f), cloudAmbient = VolSetting("AC6_CLOUD_AMBIENT", 0.7f);
    static const float cloudPhase = VolSetting("AC6_CLOUD_PHASE", 0.6f), cloudShadow = VolSetting("AC6_CLOUD_SHADOW", 0.024f);
    static const float cloudView = VolSetting("AC6_CLOUD_VIEW", 0.0f), cloudGain = VolSetting("AC6_CLOUD_GAIN", 240.0f / 255.0f);
    static const float cloudPhaseCap = VolSetting("AC6_CLOUD_PHASE_CAP", 3.0f), cloudSoft = VolSetting("AC6_CLOUD_SOFT", 2.0f);
    const float cloudRows[] = {
        cc[0][0] * fogK, cc[0][1] * fogK, cc[0][2] * fogK, cc[0][3],                                   // 18: CloudFogColor_Param
        cc[1][0] * ambientK, cc[1][1] * ambientK, cc[1][2] * ambientK, 0.0f,                           // 19: CloudColorAmbient
        cc[2][0] * sunK, cc[2][1] * sunK, cc[2][2] * sunK, 0.0f,                                       // 20: CloudColorDiffuse
        cc[3][0] * ambientK, cc[3][1] * ambientK, cc[3][2] * ambientK, 0.0f,                           // 21: CloudColorUpperAmbient
        cc[4][0] * ambientK, cc[4][1] * ambientK, cc[4][2] * ambientK, 0.0f,                           // 22: CloudColorLowerAmbient
        cc[5][0], cc[5][1], cc[7][2], cc[7][3],                                                        // 23: the alpha band, the ambient band
        extents[0], extents[1], extents[2], extents[3],                                                // 24
        centres[0][0], centres[0][1], centres[1][0], centres[1][1],                                    // 25
        centres[2][0], centres[2][1], centres[3][0], centres[3][1],                                    // 26
        float(kCloudMapTexels), 1.0f / float(kCloudMapTexels * kCloudMapLevels), 1.0f / float(kCloudMapTexels), cc[5][1] - 800.0f,   // 27
        std::bit_cast<float>(s_r.cloudMap.descriptor), std::bit_cast<float>(s_r.cloudBounds.descriptor),
        std::bit_cast<float>(s_r.cloudCoarse.descriptor), 0.0f,                                        // 28
        cloudSize, cloudCoverage, cloudTop, cloudBottom,                                               // 29
        cloudDensity, cloudNoise, cloudDetailSize, cloudDetail,                                        // 30
        std::clamp(cloudSteps, 8.0f, 512.0f), cloudStep, cloudGrowth, cloudStride,                     // 31
        cloudSun, cloudAmbient, cloudPhase, cloudShadow,                                               // 32
        cloudView, cloudGain, cloudPhaseCap, cloudSoft,                                                // 33
        float(sun[0] * sunSign), float(sun[1] * sunSign), float(sun[2] * sunSign), floatScene ? 1.0f : 0.0f,   // 34
        VolSetting("AC6_CLOUD_FINE_SIZE", 150.0f), VolSetting("AC6_CLOUD_FINE", 0.35f), VolSetting("AC6_CLOUD_BASE_NOISE", 1.2f),
        VolSetting("AC6_CLOUD_SHARP", 3.0f),                                                           // 35
        VolSetting("AC6_CLOUD_BASE_AMBIENT", 0.7f), std::clamp(VolSetting("AC6_CLOUD_TEMPORAL", CloudMarchStep() > 1 ? 0.3f : 0.12f), 0.01f, 1.0f),
        history ? 1.0f : 0.0f, s_r.cloudscapeFrame == s_r.frame ? 1.0f : 0.0f,                         // 36
        std::bit_cast<float>(s_r.cloudImage.descriptor), std::bit_cast<float>(s_r.cloudFront.descriptor),
        std::bit_cast<float>(s_r.cloudAccum[(s_r.frame & 1) ^ (cloudDone ? 0 : 1)].descriptor), 0.0f,   // 37
        float(s_r.cloudPrevVp[0]), float(s_r.cloudPrevVp[1]), float(s_r.cloudPrevVp[2]), float(s_r.cloudPrevVp[3]),       // 38
        float(s_r.cloudPrevVp[4]), float(s_r.cloudPrevVp[5]), float(s_r.cloudPrevVp[6]), float(s_r.cloudPrevVp[7]),       // 39
        float(s_r.cloudPrevVp[8]), float(s_r.cloudPrevVp[9]), float(s_r.cloudPrevVp[10]), float(s_r.cloudPrevVp[11]),     // 40
        float(s_r.cloudPrevVp[12]), float(s_r.cloudPrevVp[13]), float(s_r.cloudPrevVp[14]), float(s_r.cloudPrevVp[15]),   // 41
        VolSetting("AC6_CLOUD_THICK", 2.0f), float(pixelAngle) * VolSetting("AC6_CLOUD_LOD", 1.0f) * 64.0f,
        VolSetting("AC6_CLOUD_WARP_SIZE", 2400.0f), VolSetting("AC6_CLOUD_WARP", 250.0f),              // 42
        VolSetting("AC6_CLOUD_THRESHOLD", 0.6f), VolSetting("AC6_CLOUD_OCCLUSION", 0.3f), float(CloudMarchStep()),
        float(s_r.frame & 3),                                                                          // 43
        VolSetting("AC6_VOL_LOD", 0.25f), VolSetting("AC6_VOL_LIGHT_SIZE", 1e9f), s_r.volDepthA, s_r.volDepthB,   // 44
        VolSetting("AC6_CLOUD_CLEARANCE", 1000.0f), float(kCloudCoarseTexels), SceneJitter().first, SceneJitter().second,   // 45
        float(recordInverse[0]), float(recordInverse[1]), float(recordInverse[2]), float(recordInverse[3]),       // 46
        float(recordInverse[4]), float(recordInverse[5]), float(recordInverse[6]), float(recordInverse[7]),       // 47
        float(recordInverse[8]), float(recordInverse[9]), float(recordInverse[10]), float(recordInverse[11]),     // 48
        float(recordInverse[12]), float(recordInverse[13]), float(recordInverse[14]), float(recordInverse[15]),   // 49
        float(recordEye[0]), float(recordEye[1]), float(recordEye[2]), 0.0f,                           // 50
        float(recordForward[0]), float(recordForward[1]), float(recordForward[2]), 0.0f,               // 51
    };
    static_assert(18 * 4 + sizeof(cloudRows) / sizeof(float) <= kVolParamRows * 4);
    std::memcpy(f + 18 * 4, cloudRows, sizeof(cloudRows));
    return params;
}

// The cloudscape (volumetrics.hlsl): before the effects composite, into the scene as it reads it:
// the clouds under what the game draws in its effects pass (flashes, tracers, afterburners), as the game's own cloud
// sprites mostly are. The composite (ReductionBufferSynthesisPS) writes its whole target from two pictures of the scene: its
// fetch slot 4 (its target's own resolve, taken just before: the picture where no effect is) and slot 1 (the scene's
// resolve, blended with the effects where they are). The clouds go over both. The puffs' map, their bounds on screen,
// the march, and the image over those.
//
// sceneSurface: the scene's own colour surface, at its resolve: the clouds go into it before the resolve, each
// sample tested against the scene's depth (the anti-aliasing after then has aircraft against cloud, not against the sky
// with the cloud laid behind: a row of sky along the aircraft). The frame's puffs aren't drawn yet then (the effects pass
// comes after): the last frame's records are used (they are places in the world), with this frame's camera from the
// scene's draws. Without it (the first frame, AC6_CLOUD_EARLY=0): at the effects composite, as above.
void RunCloudscape(RenderCommandList& cl, uint32_t device, HostImage* sceneSurface = nullptr)
{
    const bool early = sceneSurface != nullptr;
    if (!CloudscapeWanted() || !s_r.volCloudVs || !s_r.volCloudMapPs || !s_r.volCloudBoundsPs || !s_r.volCloudMarchPs || !s_r.volHaveVp ||
        s_r.cloudscapeFrame == s_r.frame)
        return;
    double viewVp[16], recordVp[16];
    uint32_t recordCount = 0;
    uint64_t recordFrame = s_r.frame;
    RenderBuffer* records = nullptr;
    HostImage* depthSurface = nullptr;
    if (early)
    {
        // Last frame's harvest, and this frame's camera.
        if (!CloudscapeEarly() || s_r.cloudRecFrame + 1 != s_r.frame || !s_r.cloudRecCount || s_r.taaVpFrame != s_r.frame ||
            s_r.taaVpCandidates.empty() || !sceneSurface->texture)
            return;
        const auto world = std::max_element(s_r.taaVpCandidates.begin(), s_r.taaVpCandidates.end(),
            [](const Renderer::TaaVpCandidate& a, const Renderer::TaaVpCandidate& b) { return a.draws < b.draws; });
        std::memcpy(viewVp, world->m, sizeof(viewVp));
        std::memcpy(recordVp, s_r.cloudRecVp, sizeof(recordVp));
        recordCount = s_r.cloudRecCount;
        recordFrame = s_r.frame - 1;
        records = s_r.volParticles[recordFrame % std::size(s_r.volParticles)].get();
        depthSurface = GetSurface(U32(device + kDevDepthStencil), true);
        if (!records || !depthSurface || !depthSurface->texture || depthSurface->width != sceneSurface->width ||
            depthSurface->height != sceneSurface->height || depthSurface->samples != sceneSurface->samples)
            return;
        cl.setFramebuffer(nullptr);
        s_r.markedFramebuffer = nullptr;
        VolEnsureNoise(cl);
    }
    else
    {
        if (s_r.volCloudFrame != s_r.frame || !s_r.volCount)
            return;
        std::memcpy(viewVp, s_r.volVp, sizeof(viewVp));
        std::memcpy(recordVp, s_r.volVp, sizeof(recordVp));
        recordCount = s_r.volCount;
        records = VolParticles();
    }
    if (!s_r.volNoise)
        return;
    static const uint32_t divisor = uint32_t(std::clamp(VolSetting("AC6_VOL_RES", 2.0f), 1.0f, 4.0f));
    HostImage* scenes[2] = {};
    for (uint32_t i = 0; i < 2 && !early; i++)
    {
        const uint32_t slot = i ? 1 : 4;
        const uint32_t address = xenos::ReadTextureFetch(g_memory.As<uint8_t>(device + kDevTextureFetch + slot * 24)).BaseAddress() & 0x1FFFFFFF;
        auto& images = ResolvedImages(address);
        if (auto it = images.find(address); it != images.end() && it->second.texture)
            scenes[i] = &it->second;
    }
    if (early)
        scenes[0] = scenes[1] = sceneSurface;
    auto depthIt = s_r.resolved.find(0x05F83000);
    const bool haveDepth = depthIt != s_r.resolved.end() && depthIt->second.texture && depthIt->second.layout == RenderTextureLayout::SHADER_READ;
    if (!scenes[0] || !scenes[1] || (!early && !haveDepth))
        return;
    HostImage& scene = *scenes[0];
    static uint32_t logged = 0;
    if (logged++ < 4)
        log::Info("Cloudscape frame {} ({}): scene {}x{} x{} format {}, resolved depth {}, {} record(s) of frame {}", s_r.frame,
            early ? "into the scene before its resolve" : "over the resolved scene", scene.width, scene.height, scene.samples, int(scene.format),
            haveDepth, recordCount, recordFrame);
    // The pipelines: the puffs into the map and the bounds (MAX), the march (its picture and its depth), the temporal
    // blend, the result over the scene (premultiplied).
    enum : uint64_t { kMap = 1, kBounds = 2, kMarch = 3, kTemporal = 4, kOver = 5, kScene = 6 };
    constexpr RenderFormat kFloat = RenderFormat::R16G16B16A16_FLOAT, kDepthFloat = RenderFormat::R32_FLOAT;
    auto cloudPipeline = [&](uint64_t kind, RenderFormat format) -> RenderPipeline* {
        auto& pipeline = s_r.volPipelines[(kind << 40) | uint64_t(format)];
        if (!pipeline)
        {
            RenderGraphicsPipelineDesc desc;
            desc.pipelineLayout = s_r.layout.get();
            desc.vertexShader = kind <= kBounds ? s_r.volCloudVs.get() : s_r.copyVs.get();
            desc.pixelShader = kind == kMap ? s_r.volCloudMapPs.get() : kind == kBounds ? s_r.volCloudBoundsPs.get()
                : kind == kMarch ? s_r.volCloudMarchPs.get() : kind == kTemporal ? s_r.volCloudTemporalPs.get() : s_r.volCloudOverPs.get();
            desc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
            desc.cullMode = RenderCullMode::NONE;
            desc.renderTargetFormat[0] = format;
            desc.renderTargetCount = 1;
            RenderBlendDesc blend;
            blend.renderTargetWriteMask = 0xF;
            if (kind <= kBounds)
            {
                blend.blendEnabled = true;
                blend.srcBlend = blend.dstBlend = blend.srcBlendAlpha = blend.dstBlendAlpha = RenderBlend::ONE;
                blend.blendOp = blend.blendOpAlpha = RenderBlendOperation::MAX;
            }
            else if (kind == kOver)
            {
                blend.blendEnabled = true;
                blend.srcBlend = RenderBlend::ONE;
                blend.dstBlend = RenderBlend::INV_SRC_ALPHA;
                blend.blendOp = RenderBlendOperation::ADD;
                blend.srcBlendAlpha = RenderBlend::ZERO;   // the scene's alpha kept
                blend.dstBlendAlpha = RenderBlend::ONE;
                blend.blendOpAlpha = RenderBlendOperation::ADD;
            }
            desc.renderTargetBlend[0] = blend;
            if (kind == kMarch)
            {
                desc.renderTargetFormat[1] = kDepthFloat;
                desc.renderTargetCount = 2;
                RenderBlendDesc plain;
                plain.renderTargetWriteMask = 0xF;
                desc.renderTargetBlend[1] = plain;
            }
            pipeline = g_video.device->createGraphicsPipeline(desc);
        }
        return pipeline.get();
    };
    RenderPipeline* const map = cloudPipeline(kMap, kFloat);
    RenderPipeline* const bounds = cloudPipeline(kBounds, kFloat);
    RenderPipeline* const march = cloudPipeline(kMarch, kFloat);
    RenderPipeline* const temporal = cloudPipeline(kTemporal, kFloat);
    RenderPipeline* overs[2] = {};
    if (!early)
    {
        overs[0] = cloudPipeline(kOver, scenes[0]->format);
        overs[1] = cloudPipeline(kOver, scenes[1]->format);
    }
    else
    {
        // Into the scene's surface: its samples, its depth tested (nearer is greater: the game's depth is reversed), not
        // written.
        auto& pipeline = s_r.volPipelines[(uint64_t(kScene) << 40) | uint64_t(scene.format) | (uint64_t(scene.samples) << 16) |
            (uint64_t(depthSurface->format) << 24)];
        if (!pipeline)
        {
            RenderGraphicsPipelineDesc desc;
            desc.pipelineLayout = s_r.layout.get();
            desc.vertexShader = s_r.copyVs.get();
            desc.pixelShader = s_r.volCloudScenePs.get();
            desc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
            desc.cullMode = RenderCullMode::NONE;
            desc.renderTargetFormat[0] = scene.format;
            desc.renderTargetCount = 1;
            RenderBlendDesc blend;
            blend.renderTargetWriteMask = 0xF;
            blend.blendEnabled = true;
            blend.srcBlend = RenderBlend::ONE;
            blend.dstBlend = RenderBlend::INV_SRC_ALPHA;
            blend.blendOp = RenderBlendOperation::ADD;
            blend.srcBlendAlpha = RenderBlend::ZERO;   // the scene's alpha kept
            blend.dstBlendAlpha = RenderBlend::ONE;
            blend.blendOpAlpha = RenderBlendOperation::ADD;
            desc.renderTargetBlend[0] = blend;
            desc.depthTargetFormat = depthSurface->format;
            desc.depthEnabled = true;
            desc.depthWriteEnabled = false;
            desc.depthFunction = RenderComparisonFunction::GREATER;
            desc.multisampling.sampleCount = scene.samples;
            pipeline = g_video.device->createGraphicsPipeline(desc);
        }
        overs[0] = overs[1] = pipeline.get();
    }
    if (!map || !bounds || !march || !temporal || !overs[0] || !overs[1])
        return;
    const uint32_t width = std::max(1u, scene.width / divisor), height = std::max(1u, scene.height / divisor);
    HostImage& result = s_r.cloudAccum[s_r.frame & 1];
    HostImage& last = s_r.cloudAccum[(s_r.frame & 1) ^ 1];
    // The last frame's result is usable if it is the last frame's, that size, and the camera didn't jump (a cut).
    const uint32_t marchWidth = (width + CloudMarchStep() - 1) / CloudMarchStep(), marchHeight = (height + CloudMarchStep() - 1) / CloudMarchStep();
    bool history = s_r.cloudHistoryFrame + 1 == s_r.frame && last.texture && last.width == width && last.height == height &&
        s_r.cloudImage.texture && s_r.cloudImage.width == marchWidth && s_r.cloudImage.height == marchHeight;
    EnsureImage(s_r.cloudMap, kCloudMapTexels * kCloudMapLevels, kCloudMapTexels, kFloat, false);
    EnsureImage(s_r.cloudCoarse, kCloudCoarseTexels, kCloudCoarseTexels, kFloat, false);
    EnsureImage(s_r.cloudBounds, width, height, kFloat, false, 1, false, true);
    EnsureImage(s_r.cloudImage, marchWidth, marchHeight, kFloat, false, 1, false, true);
    EnsureImage(s_r.cloudFront, marchWidth, marchHeight, kDepthFloat, false, 1, false, true);
    EnsureImage(result, width, height, kFloat, false, 1, false, true);
    EnsureImage(last, width, height, kFloat, false, 1, false, true);
    if (s_r.cloudFramebufferOf[0] != s_r.cloudImage.texture.get() || s_r.cloudFramebufferOf[1] != s_r.cloudFront.texture.get())
    {
        if (s_r.cloudFramebuffer)
            s_r.retiredFramebuffers.push_back(std::move(s_r.cloudFramebuffer));
        s_r.cloudFramebufferOf[0] = s_r.cloudImage.texture.get();
        s_r.cloudFramebufferOf[1] = s_r.cloudFront.texture.get();
        s_r.cloudFramebuffer = g_video.device->createFramebuffer(RenderFramebufferDesc(s_r.cloudFramebufferOf, 2));
    }
    double camera[3] = {};
    {
        // A cut: the camera moved more than an aircraft does in a frame.
        double inverse[16];
        if (Invert4(viewVp, inverse))
        {
            double moved = 0.0;
            for (int c = 0; c < 3; c++)
                moved += (inverse[8 + c] / inverse[11] - s_r.cloudPrevCamera[c]) * (inverse[8 + c] / inverse[11] - s_r.cloudPrevCamera[c]);
            if (moved > 400.0 * 400.0)
                history = false;
        }
    }
    s_r.cloudscapeFrame = s_r.frame;
    const auto params = VolUploadParams(width, height, scene.format == kFloat, history, false, camera, viewVp, recordVp, recordFrame);
    if (!params.memory)
    {
        s_r.cloudscapeFrame = ~0ull;
        return;
    }
    const uint64_t particles = records->getDeviceAddress();
    uint32_t push[7] = { uint32_t(params.deviceAddress), uint32_t(params.deviceAddress >> 32), uint32_t(particles), uint32_t(particles >> 32),
        haveDepth ? depthIt->second.descriptor : kNullTexture2D, s_r.volNoiseDescriptor, 0 };
    cl.setFramebuffer(nullptr);
    s_r.markedFramebuffer = nullptr;
    // The map: each level's square of the atlas.
    MarkPass([] { return std::string("cloudscape map"); });
    Transition(cl, s_r.cloudMap, RenderTextureLayout::COLOR_WRITE);
    cl.setFramebuffer(GetFramebuffer(&s_r.cloudMap, nullptr));
    cl.clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
    SetPipeline(cl, map);
    BindCommon(cl);
    for (uint32_t l = 0; l < kCloudMapLevels; l++)
    {
        SetViewport(cl, RenderViewport(float(l * kCloudMapTexels), 0.0f, float(kCloudMapTexels), float(kCloudMapTexels)));
        SetScissor(cl, RenderRect(int32_t(l * kCloudMapTexels), 0, int32_t((l + 1) * kCloudMapTexels), int32_t(kCloudMapTexels)));
        push[6] = l;
        cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
        cl.drawInstanced(6, recordCount, 0, 0);
    }
    // The clearance map (the same shaders, mode 5).
    Transition(cl, s_r.cloudCoarse, RenderTextureLayout::COLOR_WRITE);
    cl.setFramebuffer(GetFramebuffer(&s_r.cloudCoarse, nullptr));
    cl.clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(kCloudCoarseTexels), float(kCloudCoarseTexels)));
    SetScissor(cl, RenderRect(0, 0, int32_t(kCloudCoarseTexels), int32_t(kCloudCoarseTexels)));
    SetPipeline(cl, map);
    BindCommon(cl);
    push[6] = 5;
    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    cl.drawInstanced(6, recordCount, 0, 0);
    // The bounds.
    MarkPass([] { return std::string("cloudscape bounds"); });
    Transition(cl, s_r.cloudBounds, RenderTextureLayout::COLOR_WRITE);
    cl.setFramebuffer(GetFramebuffer(&s_r.cloudBounds, nullptr));
    cl.clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(width), float(height)));
    SetScissor(cl, RenderRect(0, 0, int32_t(width), int32_t(height)));
    SetPipeline(cl, bounds);
    BindCommon(cl);
    push[6] = 4;
    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    cl.drawInstanced(6, recordCount, 0, 0);
    // The march: its picture and its depth.
    MarkPass([] { return std::string("cloudscape march"); });
    cl.setFramebuffer(nullptr);
    Transition(cl, s_r.cloudMap, RenderTextureLayout::SHADER_READ);
    Transition(cl, s_r.cloudCoarse, RenderTextureLayout::SHADER_READ);
    Transition(cl, s_r.cloudBounds, RenderTextureLayout::SHADER_READ);
    Transition(cl, s_r.cloudImage, RenderTextureLayout::COLOR_WRITE);
    Transition(cl, s_r.cloudFront, RenderTextureLayout::COLOR_WRITE);
    cl.setFramebuffer(s_r.cloudFramebuffer.get());
    cl.clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
    cl.clearColor(1, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(marchWidth), float(marchHeight)));
    SetScissor(cl, RenderRect(0, 0, int32_t(marchWidth), int32_t(marchHeight)));
    SetPipeline(cl, march);
    BindCommon(cl);
    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    cl.drawInstanced(3, 1, 0, 0);
    // Blended into the last frames'.
    MarkPass([] { return std::string("cloudscape temporal"); });
    cl.setFramebuffer(nullptr);
    Transition(cl, s_r.cloudImage, RenderTextureLayout::SHADER_READ);
    Transition(cl, s_r.cloudFront, RenderTextureLayout::SHADER_READ);
    Transition(cl, last, RenderTextureLayout::SHADER_READ);
    Transition(cl, result, RenderTextureLayout::COLOR_WRITE);
    cl.setFramebuffer(GetFramebuffer(&result, nullptr));
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(width), float(height)));
    SetScissor(cl, RenderRect(0, 0, int32_t(width), int32_t(height)));
    SetPipeline(cl, temporal);
    BindCommon(cl);
    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    cl.drawInstanced(3, 1, 0, 0);
    cl.setFramebuffer(nullptr);
    Transition(cl, result, RenderTextureLayout::SHADER_READ);
    std::memcpy(s_r.cloudPrevVp, viewVp, sizeof(s_r.cloudPrevVp));
    std::copy(camera, camera + 3, s_r.cloudPrevCamera);
    s_r.cloudHistoryFrame = s_r.frame;
    const float depthA = s_r.volDepthA, depthB = s_r.volDepthB;
    // Over the scene's two pictures. Where a resolve swap left a surface's contents in one of these images (staleOf),
    // they are copied back to that surface first, so the clouds go into the resolve alone.
    MarkPass([] { return std::string("cloudscape over the scene"); });
    static const float soft = VolSetting("AC6_CLOUD_OCCLUDE", 200.0f);
    if (early)
    {
        HostImage& target = *sceneSurface;
        const auto [jitterX, jitterY] = SceneJitter();
        Transition(cl, target, RenderTextureLayout::COLOR_WRITE);
        Transition(cl, *depthSurface, RenderTextureLayout::DEPTH_WRITE);
        cl.setFramebuffer(GetFramebuffer(&target, depthSurface));
        SetViewport(cl, RenderViewport(0.0f, 0.0f, float(target.width), float(target.height)));
        SetScissor(cl, RenderRect(0, 0, int32_t(target.width), int32_t(target.height)));
        SetPipeline(cl, overs[0]);
        BindCommon(cl);
        const uint32_t scenePush[7] = { result.descriptor, s_r.cloudFront.descriptor, haveDepth ? depthIt->second.descriptor : kNullTexture2D,
            std::bit_cast<uint32_t>(depthA), std::bit_cast<uint32_t>(depthB), std::bit_cast<uint32_t>(soft),
            uint32_t(std::clamp((jitterX + 1.0f) * 32767.0f, 0.0f, 65535.0f)) | uint32_t(std::clamp((jitterY + 1.0f) * 32767.0f, 0.0f, 65535.0f)) << 16 };
        cl.setGraphicsPushConstants(0, scenePush, 0, sizeof(scenePush));
        cl.drawInstanced(3, 1, 0, 0);
        cl.setFramebuffer(nullptr);
        s_r.markedFramebuffer = nullptr;
        target.singleCurrent = false;
        NoteColorWrite(&target);
        // Only with AC6_CLOUD_SCENE_DEPTH=1 (off by default): the clouds' depth into the scene's resolved depth, where
        // a cloud covers at least AC6_CLOUD_DEPTH of the pixel (0.5, 0: off). The game's effects pass (its half-size
        // depth is made from this one: the background cloud sprites, tracers, particles) then has the clouds in front
        // of what is behind them, and TAA and FSR reproject the clouds by their own depth. It cuts the background cloud
        // sprites on a hard line along each cloud, so by default the effects pass gets the clouds' depth after those
        // sprites instead (RunCloudGate).
        static const float depthShare = VolSetting("AC6_CLOUD_SCENE_DEPTH", 0.0f) > 0.0f ? VolSetting("AC6_CLOUD_DEPTH", 0.5f) : 0.0f;
        s_r.cloudEarlyFrame = s_r.frame;
        if (haveDepth && depthShare > 0.0f && s_r.volCloudDepthPs && depthIt->second.format == RenderFormat::R32_FLOAT)
        {
            HostImage& resolvedDepth = depthIt->second;
            auto& pipeline = s_r.volPipelines[(7ull << 40) | uint64_t(resolvedDepth.format)];
            if (!pipeline)
            {
                RenderGraphicsPipelineDesc desc;
                desc.pipelineLayout = s_r.layout.get();
                desc.vertexShader = s_r.copyVs.get();
                desc.pixelShader = s_r.volCloudDepthPs.get();
                desc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
                desc.cullMode = RenderCullMode::NONE;
                desc.renderTargetFormat[0] = resolvedDepth.format;
                desc.renderTargetCount = 1;
                RenderBlendDesc blend;
                blend.renderTargetWriteMask = 0xF;
                blend.blendEnabled = true;
                blend.srcBlend = blend.dstBlend = blend.srcBlendAlpha = blend.dstBlendAlpha = RenderBlend::ONE;
                blend.blendOp = blend.blendOpAlpha = RenderBlendOperation::MAX;
                desc.renderTargetBlend[0] = blend;
                pipeline = g_video.device->createGraphicsPipeline(desc);
            }
            if (pipeline)
            {
                MarkPass([] { return std::string("cloudscape depth"); });
                if (resolvedDepth.staleOf)
                    RestoreStale(cl, *resolvedDepth.staleOf);
                Transition(cl, resolvedDepth, RenderTextureLayout::COLOR_WRITE);
                cl.setFramebuffer(GetFramebuffer(&resolvedDepth, nullptr));
                SetViewport(cl, RenderViewport(0.0f, 0.0f, float(target.width), float(target.height)));
                SetScissor(cl, RenderRect(0, 0, int32_t(target.width), int32_t(target.height)));
                SetPipeline(cl, pipeline.get());
                BindCommon(cl);
                const uint32_t depthPush[7] = { result.descriptor, s_r.cloudFront.descriptor, 0, std::bit_cast<uint32_t>(depthA),
                    std::bit_cast<uint32_t>(depthB), std::bit_cast<uint32_t>(depthShare), scenePush[6] };
                cl.setGraphicsPushConstants(0, depthPush, 0, sizeof(depthPush));
                cl.drawInstanced(3, 1, 0, 0);
                cl.setFramebuffer(nullptr);
                s_r.markedFramebuffer = nullptr;
                Transition(cl, resolvedDepth, RenderTextureLayout::SHADER_READ);
            }
        }
        return;
    }
    for (uint32_t i = 0; i < 2; i++)
    {
        HostImage& target = *scenes[i];
        if (i && &target == scenes[0])
            break;
        if (target.staleOf)
            RestoreStale(cl, *target.staleOf);
        Transition(cl, target, RenderTextureLayout::COLOR_WRITE);
        cl.setFramebuffer(GetFramebuffer(&target, nullptr));
        SetViewport(cl, RenderViewport(0.0f, 0.0f, float(target.width), float(target.height)));
        SetScissor(cl, RenderRect(0, 0, int32_t(target.width), int32_t(target.height)));
        SetPipeline(cl, overs[i]);
        BindCommon(cl);
        const auto [jitterX, jitterY] = SceneJitter();
        const uint32_t overPush[7] = { result.descriptor, s_r.cloudFront.descriptor, depthIt->second.descriptor, std::bit_cast<uint32_t>(depthA),
            std::bit_cast<uint32_t>(depthB), std::bit_cast<uint32_t>(soft),
            uint32_t(std::clamp((jitterX + 1.0f) * 32767.0f, 0.0f, 65535.0f)) | uint32_t(std::clamp((jitterY + 1.0f) * 32767.0f, 0.0f, 65535.0f)) << 16 };
        cl.setGraphicsPushConstants(0, overPush, 0, sizeof(overPush));
        cl.drawInstanced(3, 1, 0, 0);
        cl.setFramebuffer(nullptr);
        Transition(cl, target, RenderTextureLayout::SHADER_READ);
    }
    s_r.markedFramebuffer = nullptr;
}

// The cloudscape against the game's effects pass. After the scene's resolves
// the game draws ReductionBufferHalfRes: the scene's colour (alpha 0) and the half-size depth into the effects pass's
// target, which is the composite's own. The effects are then blended over that copy, the background cloud sprites
// (BGCloud) first. Once those sprites are drawn (before the pass's first other draw, or its resolve):
//  - the copy as it was before them is blended back by the clouds' alpha: the sprites are far behind the cloudscape's
//    clouds, and fade behind them as the clouds thicken (cut by depth, they would end on a hard line),
//  - the clouds' depth goes into the pass's depth (where a cloud covers at least AC6_CLOUD_DEPTH of the pixel, 0.5, 0
//    off): tracers, particles and the rest are behind or in front of the clouds as they are of the scene.
// `restore`: the draw's targets to bind again after (null: none).
void RunCloudGate(RenderCommandList& cl, const Targets* restore)
{
    s_r.fxGateFrame = s_r.frame;
    HostImage* const color = s_r.fxColor;
    HostImage* const depth = s_r.fxDepth;
    if (!color || !color->texture || !s_r.volCloudGatePs || !s_r.volCloudDepthOutPs)
        return;
    const HostImage& result = s_r.cloudAccum[s_r.frame & 1];
    if (!result.texture || result.layout != RenderTextureLayout::SHADER_READ || s_r.cloudFront.layout != RenderTextureLayout::SHADER_READ)
        return;
    static const float share = VolSetting("AC6_CLOUD_DEPTH", 0.5f), fade = VolSetting("AC6_CLOUD_SPRITE_FADE", 1.0f);
    const bool haveDepth = depth && depth->texture;
    auto gatePipeline = [&](bool depthOnly) -> RenderPipeline* {
        auto& pipeline = s_r.volPipelines[((depthOnly ? 9ull : 8ull) << 40) | uint64_t(color->format) | (uint64_t(s_r.fxSamples) << 16) |
            (uint64_t(haveDepth ? depth->format : RenderFormat::UNKNOWN) << 24)];
        if (!pipeline)
        {
            RenderGraphicsPipelineDesc desc;
            desc.pipelineLayout = s_r.layout.get();
            desc.vertexShader = s_r.copyVs.get();
            desc.pixelShader = depthOnly ? s_r.volCloudDepthOutPs.get() : s_r.volCloudGatePs.get();
            desc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
            desc.cullMode = RenderCullMode::NONE;
            desc.renderTargetFormat[0] = color->format;
            desc.renderTargetCount = 1;
            RenderBlendDesc blend;
            if (depthOnly)
                blend.renderTargetWriteMask = 0;
            else
            {
                blend.renderTargetWriteMask = 0xF;
                blend.blendEnabled = true;
                blend.srcBlend = RenderBlend::SRC_ALPHA;
                blend.dstBlend = RenderBlend::INV_SRC_ALPHA;
                blend.blendOp = RenderBlendOperation::ADD;
                blend.srcBlendAlpha = RenderBlend::ZERO;
                blend.dstBlendAlpha = RenderBlend::INV_SRC_ALPHA;
                blend.blendOpAlpha = RenderBlendOperation::ADD;
            }
            desc.renderTargetBlend[0] = blend;
            if (haveDepth)
            {
                desc.depthTargetFormat = depth->format;
                desc.depthEnabled = depthOnly;
                desc.depthWriteEnabled = depthOnly;
                desc.depthFunction = depthOnly ? RenderComparisonFunction::GREATER : RenderComparisonFunction::ALWAYS;
            }
            desc.multisampling.sampleCount = s_r.fxSamples;
            pipeline = g_video.device->createGraphicsPipeline(desc);
        }
        return pipeline.get();
    };
    const bool sprites = s_r.fxSprites && fade > 0.0f && s_r.fxBackup.texture && s_r.fxBackup.layout == RenderTextureLayout::SHADER_READ;
    RenderPipeline* const blendBack = sprites ? gatePipeline(false) : nullptr;
    RenderPipeline* const depthOnly = share > 0.0f && haveDepth ? gatePipeline(true) : nullptr;
    if (!blendBack && !depthOnly)
        return;
    static uint32_t logged = 0;
    if (logged++ < 3)
        log::Info("Cloud gate frame {}: effects pass {}x{} x{} format {}, sprites faded {}, depth {}", s_r.frame, s_r.fxWidth, s_r.fxHeight,
            s_r.fxSamples, int(color->format), blendBack != nullptr, depthOnly != nullptr);
    MarkPass([] { return std::string("cloudscape gate"); });
    Transition(cl, *color, RenderTextureLayout::COLOR_WRITE);
    if (haveDepth)
        Transition(cl, *depth, RenderTextureLayout::DEPTH_WRITE);
    cl.setFramebuffer(GetFramebuffer(color, haveDepth ? depth : nullptr));
    s_r.markedFramebuffer = nullptr;
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(s_r.fxWidth), float(s_r.fxHeight)));
    SetScissor(cl, RenderRect(0, 0, int32_t(s_r.fxWidth), int32_t(s_r.fxHeight)));
    const uint32_t noJitter = 32767u | (32767u << 16);
    if (blendBack)
    {
        SetPipeline(cl, blendBack);
        BindCommon(cl);
        const uint32_t push[7] = { result.descriptor, s_r.fxBackup.descriptor, 0, 0, 0, std::bit_cast<uint32_t>(fade), noJitter };
        cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
        cl.drawInstanced(3, 1, 0, 0);
        NoteColorWrite(color);
    }
    if (depthOnly)
    {
        SetPipeline(cl, depthOnly);
        BindCommon(cl);
        const uint32_t push[7] = { result.descriptor, s_r.cloudFront.descriptor, 0, std::bit_cast<uint32_t>(s_r.volDepthA),
            std::bit_cast<uint32_t>(s_r.volDepthB), std::bit_cast<uint32_t>(share), noJitter };
        cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
        cl.drawInstanced(3, 1, 0, 0);
        NoteDepthWrite(depth);
    }
    if (restore)
    {
        cl.setFramebuffer(GetFramebuffer(restore->color, restore->depth));
        s_r.markedFramebuffer = nullptr;
    }
}

// AC6_HDR_GLOW_OCCLUSION=0: the effects' HDR light shows through the clouds (clouds drawn later don't hide it).
bool GlowOcclusion()
{
    static const bool on = [] { const char* v = std::getenv("AC6_HDR_GLOW_OCCLUSION"); return !(v && *v == '0'); }();
    return on;
}

// Before a cloud's draw into a single-sampled scene target (the effects pass): the scene's own lights
// (the multisampled layer: aircraft lights, afterburners, the sun's glow, drawn in the world pass) are added into that
// target's layer, once a frame, so the clouds drawn from here on hide them as they hide the effects pass's own. The
// game's picture does the same: the effects composite lays the effects pass over the scene by its alpha. The
// multisampled layer is then spent (the compose pass adds the one layer).
void GlowUnderClouds(RenderCommandList& cl, uint32_t device, const Targets& targets)
{
    if (!GlowOcclusion() || !targets.color || targets.color->samples != 1 || s_r.hdrGlowMsFrame != s_r.frame || s_r.hdrCanvasFrame == s_r.frame ||
        !HdrSceneOn() || !s_r.hdrComposePipeline || !s_r.glowMergePipeline || EffectsGlow() <= 0.0f || !SceneSized(*targets.color))
        return;
    const uint32_t blendControl = U32(device + kDevBlendControl);
    if (blendControl == kBlendDisabled || Blend(blendControl & 0x1F) != RenderBlend::SRC_ALPHA || Blend((blendControl >> 8) & 0x1F) != RenderBlend::INV_SRC_ALPHA)
        return;
    HostImage& ms = s_r.hdrGlowMs;
    if (ms.width != targets.color->width || ms.height != targets.color->height || ms.fresh)
        return;
    // (Read by its samples, as the compose pass reads it: no resolve.)
    const uint32_t from = SamplesDescriptor(ms);
    HostImage& layer = s_r.hdrGlow;
    EnsureImage(layer, targets.color->width, targets.color->height, RenderFormat::R16G16B16A16_FLOAT, false, 1, false, true);
    Transition(cl, ms, RenderTextureLayout::SHADER_READ);
    Transition(cl, layer, RenderTextureLayout::COLOR_WRITE);
    if (g_video.GpuPassTiming())
        g_video.MarkGpuPass("hdr glow under clouds");
    cl.setFramebuffer(GetFramebuffer(&layer, nullptr));
    s_r.markedFramebuffer = nullptr;
    if (s_r.hdrGlowFrame != s_r.frame || layer.fresh)
    {
        cl.clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
        s_r.hdrGlowFrame = s_r.frame;
        layer.fresh = false;
    }
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(layer.width), float(layer.height)));
    SetScissor(cl, RenderRect(0, 0, int32_t(layer.width), int32_t(layer.height)));
    SetPipeline(cl, s_r.glowMergePipeline.get());
    BindCommon(cl);
    const uint32_t push[7] = { from, 0, 0, 0, 0, 0, 0 };
    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    cl.drawInstanced(3, 1, 0, 0);
    layer.singleCurrent = false;
    s_r.hdrGlowMsFrame = ~0ull;
    cl.setFramebuffer(targets.motion ? GetMotionFramebuffer(targets.color, targets.depth) : GetFramebuffer(targets.color, targets.depth));
}

// A draw's part in the cloud gate (RunCloudGate), with its targets bound, before it is drawn: the game's scene copy
// (ReductionBufferHalfRes) gives the effects pass's targets. The first background cloud sprite draw keeps a copy of
// the target as it is before it. The gate runs before the first draw after the sprites that isn't one.
void CloudGateAtDraw(RenderCommandList& cl, const Targets& targets, uint64_t psHash)
{
    constexpr uint64_t kHalfResPs = 0x1F54F7BFE2B54D54ull, kBgCloudPs = 0xF9E3608D860E2165ull, kCompositePs = 0x8E36B699AF21554Aull;
    if (s_r.cloudEarlyFrame != s_r.frame)
        return;
    if (psHash == kHalfResPs)
    {
        s_r.fxArmedFrame = s_r.frame;
        s_r.fxTargetFrame = s_r.frame;
        s_r.fxColor = targets.color;
        s_r.fxDepth = targets.depth;
        s_r.fxWidth = targets.width;
        s_r.fxHeight = targets.height;
        s_r.fxSamples = targets.samples;
        s_r.fxSprites = false;
        return;
    }
    if (s_r.fxArmedFrame != s_r.frame || s_r.fxGateFrame == s_r.frame || !targets.color || targets.color != s_r.fxColor)
        return;
    if (psHash == kCompositePs)
    {
        s_r.fxGateFrame = s_r.frame;   // the pass is over
        return;
    }
    if (psHash == kBgCloudPs)
    {
        if (!s_r.fxSprites && targets.samples == 1 && targets.color->texture)
        {
            // The target before the sprites: a copy.
            HostImage& color = *targets.color;
            cl.setFramebuffer(nullptr);
            s_r.markedFramebuffer = nullptr;
            EnsureImage(s_r.fxBackup, color.width, color.height, color.format, false, 1, false, true);
            Transition(cl, color, RenderTextureLayout::COPY_SOURCE);
            Transition(cl, s_r.fxBackup, RenderTextureLayout::COPY_DEST);
            cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(s_r.fxBackup.texture.get()), RenderTextureCopyLocation::Subresource(color.texture.get()));
            Transition(cl, s_r.fxBackup, RenderTextureLayout::SHADER_READ);
            Transition(cl, color, RenderTextureLayout::COLOR_WRITE);
            cl.setFramebuffer(GetFramebuffer(targets.color, targets.depth));
            s_r.fxSprites = true;
        }
        return;
    }
    RunCloudGate(cl, &targets);
}

// After the effects composite, into its target: a quad per record, ray marched (volumetrics.hlsl).
void RunVolumetrics(RenderCommandList& cl, const Targets& targets)
{
    static uint32_t logged = 0;
    if (logged++ < 3)
    {
        auto depthIt = s_r.resolved.find(0x05F83000);
        log::Info("Volumetric smoke frame {}: {} particle(s), camera {}, noise {}, depth {} (layout {}), target {}x{} x{} format {}", s_r.frame,
            s_r.volCount, s_r.volHaveVp, s_r.volNoise != nullptr, depthIt != s_r.resolved.end(), depthIt != s_r.resolved.end() ? int(depthIt->second.layout) : -1,
            targets.width, targets.height, targets.samples, targets.color ? int(targets.color->format) : -1);
    }
    if (!targets.color || !s_r.volNoise || !s_r.volHaveVp)
        return;
    double inverse[16];
    if (!Invert4(s_r.volVp, inverse))
        return;
    // The pipelines: the particles into the frame or into the reduced image (its alpha accumulating too), and the
    // reduced image onto the frame. Premultiplied "over", the frame's alpha kept.
    auto volumePipeline = [&](bool composite, RenderFormat format, uint32_t samples, bool depth) -> RenderPipeline* {
        const uint64_t key = uint64_t(format) | (uint64_t(samples) << 16) | (uint64_t(depth) << 24) | (uint64_t(composite) << 25);
        auto& pipeline = s_r.volPipelines[key];
        if (!pipeline)
        {
            const bool offscreen = format == RenderFormat::R16G16B16A16_FLOAT && !composite;
            RenderGraphicsPipelineDesc desc;
            desc.pipelineLayout = s_r.layout.get();
            desc.vertexShader = composite ? s_r.copyVs.get() : s_r.volVs.get();
            desc.pixelShader = composite ? s_r.volComposePs.get() : s_r.volPs.get();
            desc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
            desc.cullMode = RenderCullMode::NONE;
            desc.renderTargetFormat[0] = format;
            desc.renderTargetCount = 1;
            RenderBlendDesc blend;
            blend.blendEnabled = true;
            blend.srcBlend = RenderBlend::ONE;
            blend.dstBlend = RenderBlend::INV_SRC_ALPHA;
            blend.blendOp = RenderBlendOperation::ADD;
            blend.srcBlendAlpha = offscreen ? RenderBlend::ONE : RenderBlend::ZERO;
            blend.dstBlendAlpha = offscreen ? RenderBlend::INV_SRC_ALPHA : RenderBlend::ONE;
            blend.blendOpAlpha = RenderBlendOperation::ADD;
            blend.renderTargetWriteMask = 0xF;
            desc.renderTargetBlend[0] = blend;
            if (depth)
            {
                desc.depthTargetFormat = kDepthFormat;
                desc.depthEnabled = false;
                desc.depthWriteEnabled = false;
            }
            desc.multisampling.sampleCount = samples;
            pipeline = g_video.device->createGraphicsPipeline(desc);
        }
        return pipeline.get();
    };
    // AC6_VOL_RES=<divisor>: the volumes at 1/2 of the frame's resolution on each axis by default (1: full).
    static const uint32_t divisor = uint32_t(std::clamp(VolSetting("AC6_VOL_RES", 2.0f), 1.0f, 4.0f));
    const uint32_t width = std::max(1u, targets.width / divisor), height = std::max(1u, targets.height / divisor);
    RenderPipeline* pipeline = divisor > 1 ? volumePipeline(false, RenderFormat::R16G16B16A16_FLOAT, 1, false)
                                           : volumePipeline(false, targets.color->format, targets.samples, targets.depth != nullptr);
    RenderPipeline* composite = divisor > 1 ? volumePipeline(true, targets.color->format, targets.samples, targets.depth != nullptr) : nullptr;
    if (!pipeline || (divisor > 1 && !composite))
        return;
    const auto params = VolUploadParams(width, height);
    if (!params.memory)
        return;
    auto depthIt = s_r.resolved.find(0x05F83000);
    const uint32_t depthDescriptor = depthIt != s_r.resolved.end() && depthIt->second.texture ? depthIt->second.descriptor : kNullTexture2D;
    if (depthIt != s_r.resolved.end() && depthIt->second.texture && depthIt->second.layout != RenderTextureLayout::SHADER_READ)
        return;   // not readable inside this render pass
    const uint64_t particles = VolParticles()->getDeviceAddress();
    // AC6_VOL_DUMP=<frame>,... (diagnostics): the records as the volume pass reads them, to vol_<frame>.bin.
    static const std::set<uint64_t> dumpFrames = [] {
        std::set<uint64_t> f;
        if (const char* v = std::getenv("AC6_VOL_DUMP"))
            for (std::string item; const char c : std::string(v) + ",")
            {
                if (c == ',') { if (!item.empty()) f.insert(std::stoull(item)); item.clear(); }
                else item += c;
            }
        return f;
    }();
    if (dumpFrames.contains(s_r.frame) && s_r.volCount)
    {
        cl.setFramebuffer(nullptr);
        s_r.markedFramebuffer = nullptr;
        s_r.volDump = g_video.device->createBuffer(RenderBufferDesc::ReadbackBuffer(uint64_t(s_r.volCount) * kVolRecord));
        cl.barriers(RenderBarrierStage::COPY, RenderBufferBarrier(VolParticles(), RenderBufferAccess::READ));
        cl.copyBufferRegion(RenderBufferReference(s_r.volDump.get()), RenderBufferReference(VolParticles()), uint64_t(s_r.volCount) * kVolRecord);
        cl.barriers(RenderBarrierStage::GRAPHICS, RenderBufferBarrier(VolParticles(), RenderBufferAccess::READ));
        s_r.volDumpCount = s_r.volCount;
        s_r.volDumpFrame = s_r.frame;
        {
            std::string line;
            for (int i = 0; i < 16; i++)
                line += std::format(" {:.9g}", s_r.volVp[i]);
            log::Info("Volumetric dump frame {}: view-projection{}", s_r.frame, line);
            line.clear();
            for (uint32_t r = 0; r < 19; r++)
                line += std::format(" c{}=({:.6g} {:.6g} {:.6g} {:.6g})", 80 + r, s_r.volCloud[r][0], s_r.volCloud[r][1], s_r.volCloud[r][2], s_r.volCloud[r][3]);
            log::Info("Volumetric dump frame {}: cloud constants{}", s_r.frame, line);
            log::Info("Volumetric dump frame {}: c50 = {:.6g} {:.6g} {:.6g} {:.6g}", s_r.frame, s_r.volCloudFade[0], s_r.volCloudFade[1],
                s_r.volCloudFade[2], s_r.volCloudFade[3]);
        }
        cl.setFramebuffer(GetFramebuffer(targets.color, targets.depth));
    }
    MarkPass([] { return std::string("volumetric smoke"); });
    if (divisor > 1)
    {
        EnsureImage(s_r.volImage, width, height, RenderFormat::R16G16B16A16_FLOAT, false);
        Transition(cl, s_r.volImage, RenderTextureLayout::COLOR_WRITE);
        cl.setFramebuffer(GetFramebuffer(&s_r.volImage, nullptr));
        s_r.markedFramebuffer = nullptr;
        cl.clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
    }
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(width), float(height)));
    SetScissor(cl, RenderRect(0, 0, int32_t(width), int32_t(height)));
    SetPipeline(cl, pipeline);
    BindCommon(cl);
    const uint32_t push[6] = { uint32_t(params.deviceAddress), uint32_t(params.deviceAddress >> 32), uint32_t(particles), uint32_t(particles >> 32),
        depthDescriptor, s_r.volNoiseDescriptor };
    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    cl.drawInstanced(6, s_r.volCount, 0, 0);
    if (divisor > 1)
    {
        // Back into the effects composite's target: the reduced image over it.
        Transition(cl, s_r.volImage, RenderTextureLayout::SHADER_READ);
        Transition(cl, *targets.color, RenderTextureLayout::COLOR_WRITE);
        if (targets.depth)
            Transition(cl, *targets.depth, RenderTextureLayout::DEPTH_WRITE);
        cl.setFramebuffer(GetFramebuffer(targets.color, targets.depth));
        s_r.markedFramebuffer = nullptr;
        SetViewport(cl, RenderViewport(0.0f, 0.0f, float(targets.width), float(targets.height)));
        SetScissor(cl, RenderRect(0, 0, int32_t(targets.width), int32_t(targets.height)));
        SetPipeline(cl, composite);
        BindCommon(cl);
        const uint32_t composePush[4] = { s_r.volImage.descriptor, 0, targets.color->format == RenderFormat::R16G16B16A16_FLOAT ? 0u : 1u, 0 };
        cl.setGraphicsPushConstants(0, composePush, 0, sizeof(composePush));
        cl.drawInstanced(3, 1, 0, 0);
    }
}

// A line's width in host pixels: the game's (PA_SU_LINE_CNTL bits 0-15, the half width in 12.4 fixed point: 1 pixel for
// the HUD's) times the target's host pixels per guest pixel. Vulkan's lines stay 1 pixel otherwise, so at a higher
// resolution the radar's and the PiP's outlines and the gun's range arc were thin lines that no longer covered the
// edges they cover at 720p. Whole pixels, up to the device's widest (1 without wide lines).
float LineWidth(uint32_t device, uint32_t targetScale)
{
    const uint32_t half = U32(device + kDevPointSize + 8) & 0xFFFF;
    const float guest = half ? float(half) / 8.0f : 1.0f;
    const float host = std::round(guest * float(targetScale) * std::min(s_r.resX, s_r.resY));
    return std::clamp(host, 1.0f, g_video.device->getCapabilities().maxLineWidth);
}

// The UI at the frame's size. UiScaled: whether the 3D is rendered at another size than the frame's (AC6_RENDER_SCALE).
// BeginUiImage, on the frame's first 2D layer draw into its target, scales that target into the UI image, and the
// UI's draws then go there. False if it can't (another format than the output's).
bool UiScaled()
{
    return s_r.sceneWidth != s_r.targetWidth || s_r.sceneHeight != s_r.targetHeight;
}

bool BeginUiImage(RenderCommandList& cl, HostImage& frame)
{
    if (s_r.uiImageFrame == s_r.frame)
        return true;
    // The frame's format: the output's, or the HDR canvas's float.
    RenderPipeline* const copy = frame.format == kTargetFormat ? s_r.copyPipeline.get()
        : frame.format == RenderFormat::R16G16B16A16_FLOAT ? s_r.copyHdrPipeline.get() : nullptr;
    if (!copy)
    {
        static bool warned = false;
        if (!std::exchange(warned, true))
            log::Warn("UI at the frame's size: the frame's target is format {}, not the output's; the UI at the render scale", int(frame.format));
        return false;
    }
    const float resX = s_r.resX, resY = s_r.resY;
    s_r.resX = s_r.uiResX;
    s_r.resY = s_r.uiResY;
    EnsureImage(s_r.uiImage, s_r.targetWidth, s_r.targetHeight, frame.format, false);
    s_r.uiImage.bgraOrder = frame.bgraOrder;
    s_r.resX = resX;
    s_r.resY = resY;
    cl.setFramebuffer(nullptr);
    s_r.markedFramebuffer = nullptr;
    HostImage& source = Readable(cl, frame);
    Transition(cl, source, RenderTextureLayout::SHADER_READ);
    Transition(cl, s_r.uiImage, RenderTextureLayout::COLOR_WRITE);
    // AC6_FSR_LATE: FSR upscales the finished 3D frame into the UI's image, in a frame whose world pass it saw (else the
    // stretch below).
    const bool upscaled = s_r.lateUpscale && s_r.taaSurfaceFrame == s_r.frame && RunFsr(cl, frame, &source, &s_r.uiImage, true);
    if (!upscaled)
    {
        cl.setFramebuffer(GetFramebuffer(&s_r.uiImage, nullptr));
        const float kx = float(s_r.targetWidth) / float(s_r.sceneWidth), ky = float(s_r.targetHeight) / float(s_r.sceneHeight);
        SetViewport(cl, RenderViewport(0.0f, 0.0f, float(source.width) * kx, float(source.height) * ky));
        SetScissor(cl, RenderRect(0, 0, int32_t(s_r.targetWidth), int32_t(s_r.targetHeight)));
        SetPipeline(cl, copy);
        BindCommon(cl);
        const uint32_t push[2] = { source.descriptor, 0 };   // linear, clamped
        cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
        cl.drawInstanced(3, 1, 0, 0);
    }
    cl.setFramebuffer(nullptr);
    // The HDR output's UI split: the image as it is now is the frame before its UI.
    if (UiSplit())
    {
        if (s_r.sceneDrawFrame != s_r.frame)
            s_r.allUiFrame = s_r.frame;
        else
        {
            EnsureImage(s_r.preUi, s_r.targetWidth, s_r.targetHeight, frame.format, false);
            Transition(cl, s_r.uiImage, RenderTextureLayout::COPY_SOURCE);
            Transition(cl, s_r.preUi, RenderTextureLayout::COPY_DEST);
            cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(s_r.preUi.texture.get()),
                RenderTextureCopyLocation::Subresource(s_r.uiImage.texture.get()));
            Transition(cl, s_r.preUi, RenderTextureLayout::SHADER_READ);
            s_r.preUiFrame = s_r.frame;
        }
    }
    s_r.uiImageFrame = s_r.frame;
    return true;
}

// AC6_GPU_BREADCRUMBS=sync: whether `bytes` from GPU address `address` lie in memory a computed-index fetch may read this
// frame: the upload ring's part for this frame, or a kept stream's buffer for this frame's slot.
bool StreamRangeLive(uint64_t address, uint64_t bytes)
{
    const UploadRing& u = s_r.upload;
    const uint64_t ringStart = u.deviceAddress + u.base;
    if (address >= ringStart && address + bytes <= ringStart + u.size)
        return true;
    const uint32_t slot = g_video.slot;
    for (const auto& [end, p] : s_r.persistentStreams)
        if (p.buffers[slot] && address >= p.address[slot] && address + bytes <= p.address[slot] + (p.end - p.start))
            return true;
    return false;
}

// AC6_GPU_BREADCRUMBS=sync: a draw's description for its breadcrumb, and an error at once for a computed-index stream outside
// the memory it may read (a GPU fault waiting to happen: robust buffer access doesn't cover raw addresses).
std::string DescribeDrawForBreadcrumb(const SharedConstants& sc, const HostShader& vs, const HostShader* ps, const Targets& targets,
    uint32_t primType, uint32_t firstVertex, uint32_t vertexCount, uint32_t indexCount)
{
    std::string d = std::format("primitive {}, first vertex {}, {} vertices, {} indices", primType, firstVertex, vertexCount, indexCount);
    if (targets.color)
        d += std::format(", colour {}x{} f{} x{}", targets.color->width, targets.color->height, int(targets.color->format), targets.color->samples);
    if (targets.depth)
        d += std::format(", depth {}x{} f{} x{}", targets.depth->width, targets.depth->height, int(targets.depth->format), targets.depth->samples);
    for (uint32_t s = 0; s < 16; s++)
    {
        if (!(vs.indexedFetchStreams & (1u << s)))
            continue;
        const uint64_t address = sc.streamAddress[s];
        const uint32_t dwords = sc.streamDwords[s];
        const bool live = StreamRangeLive(address, uint64_t(dwords) * 4);
        d += std::format(", stream {} at {:X} ({} dwords){}", s, address, dwords, live ? "" : " OUTSIDE LIVE MEMORY");
        if (!live)
        {
            static uint32_t reported = 0;
            if (reported++ < 32)
                log::Error("GPU breadcrumbs: vs {:016X} stream {} reads {:X}-{:X} ({} dwords), outside this frame's memory (frame {})",
                    vs.hash, s, address, address + uint64_t(dwords) * 4, dwords, s_r.frame);
        }
    }
    if (ps)
        for (uint32_t s = 0; s < kFetchSlots; s++)
            if (ps->textureSlots & (1u << s))
                d += std::format(", t{}={}", s, sc.texture2DIndices[s]);
    return d;
}

void Draw(uint32_t device, uint32_t primType, uint32_t firstVertex, uint32_t vertexCount, IndexList& indices, HostShader* vs, HostShader* ps,
    uint32_t indexBias = 0)
{
    if (!s_r.ready || vertexCount == 0)
        return;
    {
        static const uint64_t traceFrame = [] { const char* v = std::getenv("AC6_DRAW_TRACE"); return v ? std::stoull(v) : UINT64_MAX; }();
        if (s_r.frame == traceFrame && !indices.empty())
        {
            std::string first;
            for (size_t i = 0; i < std::min<size_t>(indices.size(), 24); i++)
                first += indices[i] == 0xFFFFFFFFu ? std::string(" R") : std::format(" {}", indices[i]);
            log::Info("  indices (first vertex {}):{}", firstVertex, first);
        }
    }
    if (s_inPresent.load())
        s_overlaps++;
    s_inDraw++;
    struct Leave { ~Leave() { s_inDraw--; } } leave;
    s_r.device = device;
    ScopedNs timer(s_stats.drawNs);
    // FSR upscaling: the scene's draws at the render scale, from here (the shared constants' scale,
    // viewports, scissors) to the end.
    const uint32_t drawTarget = U32(device + kDevRenderTargets) ? U32(device + kDevRenderTargets) : U32(device + kDevDepthStencil);
    const bool worldDraw = WorldSurface(drawTarget);
    ScopedScale worldScale(worldDraw, s_r.worldResX, s_r.worldResY);
    ScopedScale nativeScale(!worldDraw && NativeSurface(drawTarget), 1.0f, 1.0f);   // the tree impostors
    // AC6_VP_TRACE=1 (diagnostics): each frame's distinct c116-c119 matrices (perspective ones), with
    // their draw counts, in order of first use. Only draws whose vertex shader reads those registers (the others see what
    // the frame before left there).
    {
        static const bool vpTrace = std::getenv("AC6_VP_TRACE") != nullptr;
        if (vpTrace)
        {
            static uint64_t traceFrame = ~0ull;
            static std::vector<std::pair<std::array<float, 16>, uint32_t>> seen;
            auto flush = [] {
                for (auto& [m, n] : seen)
                {
                    std::string row;
                    for (float f : m)
                        row += std::format(" {:.6g}", f);
                    log::Info("VPTRACE {} {}{}", traceFrame, n, row);
                }
                seen.clear();
            };
            if (s_r.frame != traceFrame)
            {
                flush();
                traceFrame = s_r.frame;
            }
            std::array<float, 16> m;
            for (uint32_t i = 0; i < 16; i++)
                m[i] = std::bit_cast<float>(U32(device + kDevVsConstants + 116 * 16 + i * 4));
            // Only draws whose vertex shader reads c116-c119 (the registers' range).
            if (std::fabs(m[3]) + std::fabs(m[7]) + std::fabs(m[11]) > 1e-6 && vs && vs->constantFirst <= 116 && vs->constantEnd >= 120)
            {
                bool found = false;
                for (auto& [k, n] : seen)
                    if (k == m)
                    {
                        n++;
                        found = true;
                        break;
                    }
                if (!found && seen.size() < 8)
                    seen.push_back({ m, 1u });
            }
        }
    }
    s_stats.draws++;
    // Line widths (PA_SU_LINE_CNTL, 0x2282: bits 0-15 the half width in 12.4 fixed point) as the game sets them.
    if (primType == xenos::kPrimLineList || primType == xenos::kPrimLineStrip)
    {
        static std::set<uint32_t> seenLines;
        if (const uint32_t control = U32(device + kDevPointSize + 8); seenLines.size() < 8 && seenLines.insert(control).second)
            log::Info("Line draw (vs {:016X}): PA_SU_LINE_CNTL {:08X}", vs ? vs->hash : 0, control);
    }
    // Draws dropped for a reason that isn't a guest error are logged once per vertex shader (and every
    // time during AC6_DRAW_TRACE), so missing geometry leaves a trace.
    auto skipped = [&](const char* why) {
        static std::set<std::pair<uint64_t, std::string>> seen;
        static const uint64_t traceFrame = [] { const char* v = std::getenv("AC6_DRAW_TRACE"); return v ? std::stoull(v) : UINT64_MAX; }();
        const uint64_t hash = vs ? vs->hash : 0;
        if (seen.insert({ hash, why }).second || s_r.frame == traceFrame)
            log::Warn("Draw skipped (vs {:016X} ps {:016X}, prim {}, {} vertices): {}", hash, ps ? ps->hash : 0, primType, vertexCount, why);
    };
    if (!vs || !ps || !vs->ok || !ps->ok)
        return skipped("shader unavailable");
    const VertexLayout* cachedLayout = CachedVertexLayout(*vs);
    if (!cachedLayout)
        return skipped("no vertex layout");
    const VertexLayout& layout = *cachedLayout;

    const bool indexed = !indices.empty();
    RenderPrimitiveTopology topology;
    uint32_t drawCount = vertexCount;
    bool pointSprites = false;
    switch (primType)
    {
    case xenos::kPrimPointList:
        // Point sprites: each point is two triangles, six copies of its vertex that the
        // vertex shader moves to the corners of a square of the point's size. Vulkan's points are a pixel
        // unless a point size is written, and have no point coordinates: the trees would be dots.
        if (indexed)
        {
            static uint32_t warned = 0;
            if (warned++ < 4)
                log::Warn("Indexed point lists drawn as points");
            topology = RenderPrimitiveTopology::POINT_LIST;
            break;
        }
        topology = RenderPrimitiveTopology::TRIANGLE_LIST;
        pointSprites = true;
        drawCount = vertexCount * 6;
        break;
    case xenos::kPrimLineList: topology = RenderPrimitiveTopology::LINE_LIST; break;
    case xenos::kPrimLineStrip: topology = RenderPrimitiveTopology::LINE_STRIP; break;
    case xenos::kPrimTriangleList: topology = RenderPrimitiveTopology::TRIANGLE_LIST; break;
    case xenos::kPrimTriangleStrip: topology = RenderPrimitiveTopology::TRIANGLE_STRIP; break;
    case xenos::kPrimTriangleFan:
    {
        // Expanded to a list: the terrain draws hundreds of small fans in one call, separated by restart
        // indices, and Vulkan pipelines (plume) only restart strips. Each fan is (first, v[i], v[i + 1]).
        topology = RenderPrimitiveTopology::TRIANGLE_LIST;
        thread_local IndexList list;
        const uint32_t n = indexed ? uint32_t(indices.size()) : vertexCount;
        list.resize(size_t(n) * 3);   // at most three per index, written through a pointer, trimmed below
        uint32_t* out = list.data();
        if (!indexed)
            for (uint32_t i = 1; i + 1 < n; i++, out += 3)
            {
                out[0] = 0;
                out[1] = i;
                out[2] = i + 1;
            }
        else
        {
            const uint32_t* in = indices.data();
            for (uint32_t start = 0; start < n;)
            {
                const uint32_t end = NextRestart(in, start, n);
                const uint32_t first = in[start];
                for (uint32_t i = start + 1; i + 1 < end; i++, out += 3)
                {
                    out[0] = first;
                    out[1] = in[i];
                    out[2] = in[i + 1];
                }
                start = end + 1;
            }
        }
        list.resize(size_t(out - list.data()));
        indices.swap(list);
        if (indices.empty())
            return;
        break;
    }
    case xenos::kPrimQuadList:
    {
        topology = RenderPrimitiveTopology::TRIANGLE_LIST;
        thread_local IndexList quads;
        const uint32_t n = indexed ? uint32_t(indices.size()) : vertexCount;
        quads.resize(size_t(n / 4) * 6);
        uint32_t* out = quads.data();
        for (uint32_t q = 0; q + 4 <= n; q += 4, out += 6)
            for (uint32_t k = 0; const uint32_t i : { 0u, 1u, 2u, 0u, 2u, 3u })
                out[k++] = indexed ? indices[q + i] : q + i;
        indices.swap(quads);
        if (indices.empty())
            return;
        break;
    }
    case xenos::kPrimRectList:
        // Each rectangle is three corners. The fourth (v0 + v2 - v1) is synthesised in RunVertexCopy.
        topology = RenderPrimitiveTopology::TRIANGLE_LIST;
        if (indexed)
        {
            static uint32_t warned = 0;
            if (warned++ < 4)
                log::Warn("Indexed rect lists not supported");
            return;
        }
        break;
    default:
    {
        static uint32_t warned = 0;
        if (warned++ < 4)
            log::Warn("Unsupported primitive type {}", primType);
        return;
    }
    }

    RenderCommandList& cl = g_video.Commands();

    // The optional lens effects on the finished 3D picture: before the frame's first 2D layer draw
    // (the HUD, a menu) into it, in a frame whose scene composite drew.
    // AC6_FSR_LATE at the native ratio: FSR on the finished 3D frame (its effects included), in place, before
    // its first 2D layer.
    if (t_uiDraw && s_r.lateNative && s_r.lateFsrFrame != s_r.frame && s_r.taaSurfaceFrame == s_r.frame &&
        s_r.sceneCompositeFrame == s_r.frame)
    {
        s_r.lateFsrFrame = s_r.frame;
        if (HostImage* rt = GetSurface(U32(device + kDevRenderTargets), false);
            rt && rt->samples == 1 && rt->width == s_r.sceneWidth && rt->height == s_r.sceneHeight)
            RunFsr(cl, *rt, nullptr, nullptr, true);
    }
    if (t_uiDraw && s_r.lensFxFrame != s_r.frame && s_r.sceneCompositeFrame == s_r.frame)
    {
        s_r.lensFxFrame = s_r.frame;
        if (const LensFx fx = LensFxSettings(); fx.Any())
            if (HostImage* rt = GetSurface(U32(device + kDevRenderTargets), false);
                rt && rt->samples == 1 && rt->width == s_r.sceneWidth && rt->height == s_r.sceneHeight)
                RunLensFx(cl, *rt, fx);
    }
    // A 2D layer draw into the frame's target while the 3D is rendered at another size: into the UI
    // image, at the frame's scale (restored when the draw is done).
    bool uiRedirect = false;
    if (t_uiDraw && UiScaled())
        if (HostImage* rt = GetSurface(U32(device + kDevRenderTargets), false);
            rt && rt->samples == 1 && rt->width == s_r.sceneWidth && rt->height == s_r.sceneHeight)
            uiRedirect = BeginUiImage(cl, *rt);
    struct UiScale
    {
        bool on;
        float x, y;
        ~UiScale()
        {
            if (on)
                s_r.resX = x, s_r.resY = y;
        }
    } uiScale{ uiRedirect, s_r.resX, s_r.resY };
    if (uiRedirect)
    {
        s_r.resX = s_r.uiResX;
        s_r.resY = s_r.uiResY;
    }

    // Vertex streams: the guest data (32-bit words swapped to little-endian) goes into the ring. The
    // copy itself runs when the frame is submitted, which is when the console GPU would read it: the
    // game fills some buffers after issuing the draw (CPU-skinned characters on worker threads).
    // AC6_IMMEDIATE_VERTICES=1 copies at draw time instead.
    static const bool immediate = [] { const char* v = std::getenv("AC6_IMMEDIATE_VERTICES"); return v && *v && *v != '0'; }();
    RenderVertexBufferView views[16];
    uint32_t viewCount = 0;
    const bool rect = primType == xenos::kPrimRectList;
    if (rect)
        drawCount = vertexCount / 3 * 6;
    for (auto& slot : layout.slots)
    {
        const uint32_t s = slot.index;
        xenos::VertexFetch vf{ { U32(device + kDevVertexFetch + (17 - s) * 8), U32(device + kDevVertexFetch + (17 - s) * 8 + 4) } };
        const uint32_t stride = layout.strides[s];
        if (!stride || !vf.Address())
            return skipped(stride ? "stream without an address" : "stream with zero stride");
        const uint8_t* src = g_memory.As<uint8_t>(PhysicalToGuest(vf.Address())) + uint64_t(firstVertex) * stride;
        const uint32_t outVertices = rect ? vertexCount / 3 * 6 : pointSprites ? vertexCount * 6 : vertexCount;
        // A stream drawn from every frame: its kept copy, brought up to date at submission.
        // Not for draws reading past the fetch constant's size (zeros there) or expanded (rects, sprites).
        if (!rect && !pointSprites && !immediate && uint64_t(firstVertex + vertexCount) * stride <= uint64_t(vf.SizeDwords()) * 4)
            if (Renderer::PersistentStream* p = UseVertexStream(vf.Address(), vf.Address() + vf.SizeDwords() * 4))
            {
                const uint64_t offset = uint64_t(vf.Address()) + uint64_t(firstVertex) * stride - p->start;
                views[viewCount++] = RenderVertexBufferView(RenderBufferReference(p->buffers[g_video.slot].get(), offset), outVertices * stride);
                if (StatsEnabled())
                    s_stats.cachedBytes += uint64_t(vertexCount) * stride;
                continue;
            }
        auto a = s_r.upload.Allocate(uint64_t(outVertices) * std::max(stride, 4u), 16);
        if (!a.memory)
            return;
        uint8_t* const dst = a.memory;
        // The fetch constant's size bounds the buffer: the console's vertex fetch reads zeros past it. The
        // sun flare draws two quads from a four-vertex buffer. Its second quad, read from whatever follows
        // (the flare's element table), would be a large bright square in the sky.
        const uint64_t sizeBytes = uint64_t(vf.SizeDwords()) * 4, skipped = uint64_t(firstVertex) * stride;
        const uint64_t validBytes = sizeBytes > skipped ? sizeBytes - skipped : 0;
        static std::set<uint32_t> reported;
        if (validBytes < uint64_t(vertexCount) * stride && reported.size() < 16 && reported.insert(vf.Address()).second)
            log::Info("Vertex stream {:08X}: draw reads {} vertices past its {}-dword buffer (vs {:016X}): zeros", vf.Address(),
                (uint64_t(vertexCount) * stride - validBytes + stride - 1) / stride, vf.SizeDwords(), vs->hash);
        const Renderer::VertexCopy copy{ dst, src, vertexCount, stride, s, validBytes, rect, &layout.elements, pointSprites };
        static const bool earlyCopies = [] { const char* v = std::getenv("AC6_EARLY_VERTEX_COPIES"); return !v || *v != '0'; }();
        const uint32_t guestSrc = PhysicalToGuest(vf.Address()) + firstVertex * stride;
        const uint32_t readBytes = uint32_t(std::min<uint64_t>(validBytes, uint64_t(vertexCount) * stride));
        if (immediate)
            RunVertexCopy(copy);
        else if (earlyCopies && readBytes && g_renderQueue->Threaded())
        {
            Renderer::SourceWatch& source = s_r.vertexSources[uint64_t(guestSrc) << 32 | readBytes];
            if (source.staticAgainFrame > s_r.frame)
                s_r.vertexCopies.push_back(copy);
            else
            {
                // Watched before it is read: a write during or after the copy is seen at submission.
                const uint64_t generation = g_memory.Watch(guestSrc, readBytes);
                if (AsyncCopier::Enabled())
                    s_asyncCopier.Push(copy);
                else
                    RunVertexCopy(copy);
                s_r.earlyCopies.push_back({ copy, guestSrc, readBytes, generation, &source });
                s_stats.earlyBytes += readBytes;
            }
        }
        else
            s_r.vertexCopies.push_back(copy);
        // Traced frame: check what the GPU will read for stream 0 (after the copy).
        static const uint64_t checkFrame = [] { const char* v = std::getenv("AC6_DRAW_TRACE"); return v ? std::stoull(v) : UINT64_MAX; }();
        if (s_r.frame == checkFrame && s == 0 && stride >= 12)
        {
            const uint64_t vsHash = vs->hash;
            auto check = [dst, stride, outVertices, vsHash, indices, indexBias] {
                uint32_t zero = 0, bad = 0, oob = 0;
                for (uint32_t v = 0; v < outVertices; v++)
                {
                    float p[3];
                    std::memcpy(p, dst + uint64_t(v) * stride, 12);
                    if (p[0] == 0.0f && p[1] == 0.0f && p[2] == 0.0f)
                        zero++;
                    if (!(std::fabs(p[0]) < 1e5f && std::fabs(p[1]) < 1e5f && std::fabs(p[2]) < 1e5f))
                        bad++;
                }
                for (uint32_t i : indices)
                    if (i != 0xFFFFFFFFu && i - indexBias >= outVertices)
                        oob++;
                log::Info("Uploaded stream 0 vs {:016X}: {} vertices, {} at origin, {} invalid, {} indices out of range", vsHash, outVertices,
                    zero, bad, oob);
            };
            if (immediate)
                check();
            else
                s_r.pendingCopies.push_back(std::move(check));
            // Which vertices change after draw time (at submission, and one frame later)?
            {
                const uint8_t* guestSrc = src;
                auto snapshot = std::make_shared<std::vector<uint32_t>>(size_t(outVertices) * 3);
                for (uint32_t v = 0; v < outVertices; v++)
                    std::memcpy(snapshot->data() + v * 3, guestSrc + uint64_t(v) * stride, 12);
                auto compare = [guestSrc, stride, outVertices, vsHash, snapshot](const char* when) {
                    uint32_t changed = 0;
                    for (uint32_t v = 0; v < outVertices; v++)
                        if (std::memcmp(snapshot->data() + v * 3, guestSrc + uint64_t(v) * stride, 12) != 0)
                            changed++;
                    log::Info("Guest stream 0 {} vs {:016X}: {} of {} vertices changed since the draw", when, vsHash, changed, outVertices);
                };
                s_r.pendingCopies.push_back([compare] { compare("at submit"); });
                s_r.nextFrameChecks.push_back([compare] { compare("one frame later"); });
            }
            // The same guest buffer one frame later (is it filled after this frame's submission?).
            const uint8_t* guestSrc = src;
            auto later = [guestSrc, stride, outVertices, vsHash] {
                uint32_t zero = 0, bad = 0;
                for (uint32_t v = 0; v < outVertices; v++)
                {
                    uint32_t w[3];
                    std::memcpy(w, guestSrc + uint64_t(v) * stride, 12);
                    float p[3];
                    for (int c = 0; c < 3; c++)
                        p[c] = std::bit_cast<float>(__builtin_bswap32(w[c]));
                    if (p[0] == 0.0f && p[1] == 0.0f && p[2] == 0.0f)
                        zero++;
                    if (!(std::fabs(p[0]) < 1e5f && std::fabs(p[1]) < 1e5f && std::fabs(p[2]) < 1e5f))
                        bad++;
                }
                log::Info("Guest stream 0 one frame later, vs {:016X}: {} vertices, {} at origin, {} invalid", vsHash, outVertices, zero, bad);
            };
            s_r.nextFrameChecks.push_back(std::move(later));
        }
        views[viewCount++] = RenderVertexBufferView(RenderBufferReference(s_r.upload.buffer.get(), a.offset), outVertices * stride);
    }

    // Constants: VS/PS float constants straight from the guest device (byte-swapped), plus the shared
    // block with the texture/sampler descriptor indices. A constant file no setter has touched since the draw
    // before (its dirty mask clear) is that draw's upload again: most draws change a few
    // VS registers (the object's matrices) and no PS ones. Only the registers the shader reads are written
    // (the rest of its block is never read): whole files, a mission frame's ~950 VS and ~280 PS uploads are 5 MB of
    // byte swaps into write-combined memory, the shaders reading a few registers of each. A
    // block is reused only if it holds this shader's registers. AC6_CONSTANT_RANGES=0: whole files.
    static const bool constantRanges = [] { const char* v = std::getenv("AC6_CONSTANT_RANGES"); return !(v && *v == '0'); }();
    struct ConstantUpload { uint64_t frame = UINT64_MAX; uint64_t deviceAddress = 0, offset = 0; uint32_t first = 0, end = 0; };
    static ConstantUpload lastVs, lastPs;
    const bool noted = std::exchange(t_dirtyNoted, false);
    // (The relighting factors are not applied to the uploaded light registers, which would give every lit draw its own
    // block: 6.4 ms of draw CPU a mission frame. The shaders scale them by the
    // shared block's factors.)
    auto uploadConstants = [&](ConstantUpload& last, uint64_t& dirty, uint32_t offset, uint32_t registers, const HostShader& shader) {
        const uint32_t end = constantRanges ? std::min(shader.constantEnd, registers) : registers;
        const uint32_t first = constantRanges ? std::min(shader.constantFirst, end) : 0;
        const bool band = registers == 256 && shader.idHash == 0x6B6FBBD8930EFA4Dull && ExposureBand() > 0.0f && HdrSceneOn() &&
            first <= 106 && 106 < end;
        const bool scaled = band;
        if (!scaled && noted && !dirty && last.frame == s_r.frame && last.first <= first && end <= last.end)
            return true;
        auto a = s_r.upload.Allocate(registers * 16, 256);
        if (!a.memory)
            return false;
        simd::ByteSwap32(reinterpret_cast<uint32_t*>(a.memory) + first * 4, g_memory.As<uint32_t>(device + offset) + first * 4, (end - first) * 4);
        s_stats.constantBlocks[registers == 256 ? 0 : 1]++;
        s_stats.constantBytes += (end - first) * 16;
        last = { s_r.frame, a.deviceAddress, a.offset, first, end };
        if (scaled)
        {
            float* f = reinterpret_cast<float*>(a.memory);
            if (band)
            {
                // (The band closed from below, its lower end towards its upper.)
                float& lo = f[106 * 4], &hi = f[106 * 4 + 1];
                static bool logged = false;
                if (!std::exchange(logged, true))
                    log::Info("Exposure band: lo {} hi {} (AC6_HDR_EXPOSURE_BAND {})", lo, hi, ExposureBand());
                lo += (hi - lo) * ExposureBand();
            }
            last.frame = UINT64_MAX;   // (not reused by the next draw)
        }
        dirty = 0;
        return true;
    };
    if (!uploadConstants(lastVs, t_vsDirty, kDevVsConstants, 256, *vs) || !uploadConstants(lastPs, t_psDirty, kDevPsConstants, 224, *ps))
        return;

    auto shared = s_r.upload.Allocate(kConstantBlockBytes[2], 256);
    if (!shared.memory)
        return;
    SharedConstants sc{};
    static_assert(offsetof(SharedConstants, booleans) == 512);
    for (uint32_t s = 0; s < kFetchSlots; s++)
    {
        sc.texture2DIndices[s] = kNullTexture2D;
        sc.texture3DIndices[s] = kNullTexture3D;
        sc.textureCubeIndices[s] = kNullTextureCube;
    }
    // AC6_MAX_ANISO=<n> caps every draw's anisotropy (0 or 1 = off).
    static const int32_t globalMaxAniso = [] { const char* v = std::getenv("AC6_MAX_ANISO"); return v && *v ? std::atoi(v) : -1; }();
    const uint32_t maxAniso = globalMaxAniso >= 0 ? uint32_t(globalMaxAniso) : 16u;
    // FSR upscaling: the scene's draws sample their textures with a finer mip bias (cached apart).
    const bool upscaledSamplers = worldDraw;
    // AC6_ANISO_OVERRIDE applies to every slot (GetSampler leaves alone a fetch whose instruction turns anisotropy
    // off).
    const bool overridable = true;
    const uint32_t samplerKey = maxAniso | (upscaledSamplers ? 0x100u : 0u) | (overridable ? 0x200u : 0u);
    // Each slot's texture and sampler are kept from the draw before when its fetch constant is the same:
    // a texture's content is checked once a frame anyway, and resolves, which can change
    // what a resolve target's fetch constant samples, bump the texture epoch. A mission frame's ~1100 draws
    // use ~10 textures each, most of them the draw before's.
    struct SlotCache { uint32_t words[6]; uint64_t frame = UINT64_MAX, epoch = 0; uint32_t maxAniso = 0, texture = 0, sampler = 0; bool cube = false; };
    static SlotCache slotCache[kUsedFetchSlots];
    static const bool slotCaching = [] { const char* v = std::getenv("AC6_TEXTURE_SLOT_CACHE"); return !v || *v != '0'; }() &&
        dumpTexture.frames.empty();
    const uint8_t* fetchWords = g_memory.As<uint8_t>(device + kDevTextureFetch);
    // All the slots as the draw before had them (the same material's next draw): its results.
    struct BlockCache
    {
        uint8_t words[kUsedFetchSlots * 24];
        uint64_t frame = UINT64_MAX, epoch = 0;
        uint32_t maxAniso = 0;
        uint32_t texture2D[kUsedFetchSlots], textureCube[kUsedFetchSlots], samplers[kUsedFetchSlots];
    };
    static BlockCache block;
    const bool sameBlock = slotCaching && block.frame == s_r.frame && block.epoch == s_r.textureEpoch && block.maxAniso == samplerKey &&
        std::memcmp(block.words, fetchWords, sizeof(block.words)) == 0;
    if (sameBlock)
    {
        std::memcpy(sc.texture2DIndices, block.texture2D, sizeof(block.texture2D));
        std::memcpy(sc.textureCubeIndices, block.textureCube, sizeof(block.textureCube));
        std::memcpy(sc.samplerIndices, block.samplers, sizeof(block.samplers));
    }
    for (uint32_t s = 0; s < kUsedFetchSlots && !sameBlock; s++)
    {
        SlotCache& c = slotCache[s];
        if (slotCaching && c.frame == s_r.frame && c.epoch == s_r.textureEpoch && c.maxAniso == samplerKey &&
            std::memcmp(c.words, fetchWords + s * 24, 24) == 0)
        {
            if (c.texture)
            {
                (c.cube ? sc.textureCubeIndices[s] : sc.texture2DIndices[s]) = c.texture;
                sc.samplerIndices[s] = c.sampler;
            }
            continue;
        }
        auto fetch = xenos::ReadTextureFetch(fetchWords + s * 24);
        uint32_t texture = 0, sampler = 0;
        const bool cube = fetch.Dimension() == 3;
        if (fetch.Type() == 2 && fetch.BaseAddress())
        {
            texture = GetTexture(cl, fetch);
            // A mod's replacement with mips: sampled through them (linear between levels: gpu/texture_mods.h).
            const uint32_t filters = s_r.mippedReplacements.contains(texture) ? (0x1FFu & ~0x30u) | 0x10u : 0x1FFu;
            sampler = GetSampler(fetch, filters, maxAniso, upscaledSamplers, overridable || s != 0);
            (cube ? sc.textureCubeIndices[s] : sc.texture2DIndices[s]) = texture;
            sc.samplerIndices[s] = sampler;
        }
        std::memcpy(c.words, fetchWords + s * 24, 24);
        c.cube = cube;
        c.frame = s_r.frame;
        c.epoch = s_r.textureEpoch;
        c.maxAniso = samplerKey;
        c.texture = texture;
        c.sampler = sampler;
    }
    if (slotCaching && !sameBlock)
    {
        std::memcpy(block.words, fetchWords, sizeof(block.words));
        block.frame = s_r.frame;
        block.epoch = s_r.textureEpoch;
        block.maxAniso = samplerKey;
        std::memcpy(block.texture2D, sc.texture2DIndices, sizeof(block.texture2D));
        std::memcpy(block.textureCube, sc.textureCubeIndices, sizeof(block.textureCube));
        std::memcpy(block.samplers, sc.samplerIndices, sizeof(block.samplers));
    }
    // The map's tile filter: with the Anisotropic filtering setting (AC6_ANISO_OVERRIDE of 2 or
    // more) the map's pixel shaders filter their atlas page themselves, as many taps along the footprint as the setting
    // (under AC6_MAX_ANISO), each kept in the pixel's tile. AC6_TERRAIN_TILE_FILTER=0: off (the game's two fetches),
    // =<n>: that many taps whatever the setting.
    sc.tileTaps = 0;
    if (ps->idHash == 0xCDB30131360ACAAAull || ps->idHash == 0x3ADEF3EC4BA4640Dull)
    {
        static const int tileFilter = [] { const char* v = std::getenv("AC6_TERRAIN_TILE_FILTER"); return v && *v ? std::atoi(v) : -1; }();
        const uint32_t taps = tileFilter >= 0 ? uint32_t(tileFilter) : AnisoOverride();
        sc.tileTaps = std::min(std::min(taps, maxAniso), 16u) >= 2 ? std::min(std::min(taps, maxAniso), 16u) : 0u;
    }
    // Fetches whose instruction sets its own filters sample through their own sampler.
    for (const auto& [shader, first] : { std::pair{ ps, 0u }, std::pair{ vs, 8u } })
        for (size_t k = 0; k < shader->samplerOverrides.size() && k < 8; k++)
        {
            const uint32_t slot = shader->samplerOverrides[k] & 0xFF, fields = (shader->samplerOverrides[k] >> 8) & 0x1FF;
            const bool levelQuery = (shader->samplerOverrides[k] >> 24) & 1;
            sc.overrideSamplers[first + k] = slot < kFetchSlots ? sc.samplerIndices[slot] : 0;
            if (slot >= kUsedFetchSlots)
                continue;
            auto fetch = xenos::ReadTextureFetch(g_memory.As<uint8_t>(device + kDevTextureFetch + slot * 24));
            if (fetch.Type() == 2 && fetch.BaseAddress())
                sc.overrideSamplers[first + k] = GetSampler(fetch, fields, maxAniso, upscaledSamplers, overridable || slot != 0, levelQuery);
        }

    // Alpha test (XenosRecomp implements it as a spec constant + threshold).
    uint32_t specConstants = (vs->specConstantsMask | ps->specConstantsMask) & ~uint32_t(SPEC_CONSTANT_R11G11B10_NORMAL);
    if (pointSprites)
    {
        specConstants |= SPEC_CONSTANT_POINT_LIST;
        // PA_SU_POINT_MINMAX clamps the per-vertex diameter, PA_SU_POINT_SIZE is the diameter of shaders
        // without a size export. Both 12.4 fixed-point radii (halves of the diameter). The NDC scale is set
        // with the viewport below.
        const uint32_t minMax = U32(device + kDevPointMinMax), size = U32(device + kDevPointSize);
        sc.pointDiameterMin = float(minMax & 0xFFFF) * (2.0f / 16.0f);
        sc.pointDiameterMax = float(minMax >> 16) * (2.0f / 16.0f);
        sc.pointConstantDiameterX = float(size >> 16) * (2.0f / 16.0f);
        sc.pointConstantDiameterY = float(size & 0xFFFF) * (2.0f / 16.0f);
    }
    const uint32_t colorControl = U32(device + kDevColorControl);
    if (colorControl & 8)
    {
        specConstants |= SPEC_CONSTANT_ALPHA_TEST;
        sc.alphaThreshold = std::bit_cast<float>(U32(device + kDevAlphaRef));
    }
    sc.drawFlags = specConstants;
    // The relighting factors, the same for every draw of a frame (the shared block still reused).
    {
        const bool relight = HdrSceneOn() && Relight().on;
        for (int i = 0; i < 10; i++)
            sc.relight[i] = relight ? Relight().k[i] : 1.0f;
        sc.cloudRim = relight ? Relight().cloudRim : 1.0f;
    }
    // The constant polygon offset: PA_SU_POLY_OFFSET's front (or back) offset, in depth units, added
    // to the depth by the vertex shaders. Vulkan's constant depth bias on a float depth buffer is in units of each depth's
    // own exponent: the shadow maps' 0.00075 would come out 4-8x too small at their low depths (shadow acne).
    if (const uint32_t scMode = U32(device + kDevScModeControl); DepthBiasInShader() && (scMode & 0x1800) && U32(device + kDevDepthStencil))
        sc.depthBias = std::bit_cast<float>(U32(device + kDevPolyOffset + ((scMode & 0x800) ? 4 : 12)));

    // Streams the vertex shader reads at indices it computes: the whole guest buffer (fetch constant
    // size), words swapped like the input assembler's copies, at submission like those.
    sc.vertexIndexBase = firstVertex;
    {
        // As GetSurface promotes it: 4x MSAA targets (not tiled) are drawn at twice the size.
        const uint32_t rt = U32(device + kDevRenderTargets), ds = U32(device + kDevDepthStencil);
        const uint32_t surface = rt ? rt : ds;
        sc.pixelPosScale = surface && !s_r.tiling && ((U32(surface + kSurfaceInfo) >> 16) & 3) == 2 ? 0.5f : 1.0f;
        // The shaders snap the host pixel to the guest's grid first.
        sc.resolutionScale = s_r.resX;
        sc.resolutionScaleY = s_r.resY;
        sc.fallbackAddress = s_r.upload.deviceAddress;
    }
    for (uint32_t s = 0; s < 16; s++)
    {
        if (!(vs->indexedFetchStreams & (1u << s)))
            continue;
        xenos::VertexFetch vf{ { U32(device + kDevVertexFetch + (17 - s) * 8), U32(device + kDevVertexFetch + (17 - s) * 8 + 4) } };
        const uint32_t dwords = std::min<uint32_t>(vf.SizeDwords(), 4u << 20);
        if (!vf.Address() || !dwords)
            continue;
        // One copy per frame serves every slice of a pool that starts at or after the uploaded start.
        const uint32_t start = vf.Address(), end = start + dwords * 4;
        auto it = s_r.indexedStreams.find(end);
        static const bool streamCache = [] { const char* v = std::getenv("AC6_STREAM_CACHE"); return !v || *v != '0'; }();
        // Streams of at least AC6_STREAM_KEEP_MIN dwords (1024 by default: one page) are kept from frame to frame:
        // without that a mission frame swaps ~2 MB of small computed-index streams whole, every frame, though the game
        // hasn't written them since.
        static const uint32_t keepMin = [] { const char* v = std::getenv("AC6_STREAM_KEEP_MIN"); return v && *v ? uint32_t(std::atoi(v)) : 1024u; }();
        if (streamCache && !immediate && dwords >= keepMin && (it == s_r.indexedStreams.end() || start < it->second.first))
            if (Renderer::PersistentStream* p = UsePersistentStream(start, end))
                it = s_r.indexedStreams.insert_or_assign(end, std::pair{ p->start, p->address[g_video.slot] }).first;
        if (it == s_r.indexedStreams.end() || start < it->second.first)
        {
            auto a = s_r.upload.Allocate(uint64_t(dwords) * 4, 16);
            if (!a.memory)
                return;
            static std::set<uint32_t> large;
            if (dwords >= (256u << 10) && large.size() < 16 && large.insert(vf.Address()).second)
                log::Info("Indexed vertex stream {:08X}: {} KB per frame (vs {:016X} stream {})", vf.Address(), dwords / 256, vs->hash, s);
            const auto* src = g_memory.As<uint32_t>(PhysicalToGuest(vf.Address()));
            auto* dst = reinterpret_cast<uint32_t*>(a.memory);
            if (immediate)
                simd::ByteSwap32(dst, src, dwords);
            else
                s_r.swapFills.push_back({ dst, src, dwords });
            (dwords >= (64u << 10) ? s_stats.swapLargeBytes : s_stats.swapSmallBytes) += uint64_t(dwords) * 4;
            s_r.wholeStreams.push_back({ PhysicalToGuest(vf.Address()), dwords * 4 });
            it = s_r.indexedStreams.insert_or_assign(end, std::pair{ start, a.deviceAddress }).first;
        }
        sc.streamAddress[s] = it->second.second + (start - it->second.first);
        sc.streamDwords[s] = dwords;
        sc.streamAddress[s] = FaultTestAddress(sc.streamAddress[s]);
        static const uint64_t traceFrame = [] { const char* v = std::getenv("AC6_DRAW_TRACE"); return v ? std::stoull(v) : UINT64_MAX; }();
        if (s_r.frame == traceFrame)
        {
            std::string words;
            for (uint32_t i = 0; i < std::min<uint32_t>(dwords, dwords <= 64 ? 64 : 8); i++)
                words += std::format(" {:08X}", U32(PhysicalToGuest(start) + i * 4));
            log::Info("  indexed stream {}: fetch {:08X} {:08X} ({} dwords, endian {}):{}", s, vf.dw[0], vf.dw[1], dwords, vf.Endian(), words);
        }
    }
    // Traced frame: capture draws with indexed fetches for offline evaluation (cap_<n>_<vs>.bin: first vertex, index
    // count, primitive and the indexed-stream mask, then 256 VS constants, the indexed streams (stream, address,
    // dwords, words swapped), the indices, 224 PS constants and the input-assembler streams).
    if (vs->indexedFetchStreams)
    {
        static const uint64_t traceFrame = [] { const char* v = std::getenv("AC6_DRAW_TRACE"); return v ? std::stoull(v) : UINT64_MAX; }();
        static uint32_t captured = 0;
        if (s_r.frame == traceFrame && captured < 64)
        {
            std::ofstream f(std::filesystem::path(log::Directory()) / std::format("cap_{:03}_{:016X}.bin", captured++, vs->hash), std::ios::binary);
            const uint32_t header[4] = { firstVertex, uint32_t(indices.size()), primType, vs->indexedFetchStreams };
            f.write(reinterpret_cast<const char*>(header), sizeof(header));
            for (uint32_t i = 0; i < 256 * 4; i++)
            {
                const uint32_t w = U32(device + kDevVsConstants + i * 4);
                f.write(reinterpret_cast<const char*>(&w), 4);
            }
            for (uint32_t st = 0; st < 16; st++)
            {
                if (!(vs->indexedFetchStreams & (1u << st)))
                    continue;
                xenos::VertexFetch vf{ { U32(device + kDevVertexFetch + (17 - st) * 8), U32(device + kDevVertexFetch + (17 - st) * 8 + 4) } };
                const uint32_t dwords = std::min<uint32_t>(vf.SizeDwords(), 4u << 20);
                const uint32_t fields[3] = { st, vf.Address(), dwords };
                f.write(reinterpret_cast<const char*>(fields), sizeof(fields));
                for (uint32_t i = 0; i < dwords; i++)
                {
                    const uint32_t w = U32(PhysicalToGuest(vf.Address()) + i * 4);
                    f.write(reinterpret_cast<const char*>(&w), 4);
                }
            }
            f.write(reinterpret_cast<const char*>(indices.data()), indices.size() * 4);
            // Pixel shader constants (224 registers).
            for (uint32_t i = 0; i < 224 * 4; i++)
            {
                const uint32_t w = U32(device + kDevPsConstants + i * 4);
                f.write(reinterpret_cast<const char*>(&w), 4);
            }
            // Then the input-assembler streams: (stream, address, dwords, words) each, up to 4 MB.
            for (auto& slot : layout.slots)
            {
                xenos::VertexFetch vf{ { U32(device + kDevVertexFetch + (17 - slot.index) * 8), U32(device + kDevVertexFetch + (17 - slot.index) * 8 + 4) } };
                const uint32_t dwords = std::min<uint32_t>(vf.SizeDwords(), 1u << 20);
                const uint32_t fields[3] = { slot.index, vf.Address(), dwords };
                f.write(reinterpret_cast<const char*>(fields), sizeof(fields));
                for (uint32_t i = 0; i < dwords; i++)
                {
                    const uint32_t w = U32(PhysicalToGuest(vf.Address()) + i * 4);
                    f.write(reinterpret_cast<const char*>(&w), 4);
                }
            }
        }
    }
    // The slots sampling images the game rendered, which are at the internal resolution: the
    // shaders take their guest size for texel offsets and size queries. Images rendered at another scale than this draw's
    // (FSR upscaling: the scene's at the render scale, the rest at the frame's) at their own.
    const bool shadowDetail = U32(device + kDevRenderTargets) != 0 && ShadowDetail();
    for (uint32_t s = 0; s < kFetchSlots; s++)
    {
        const std::array<float, 2> scale = s_r.scaledDescriptors[sc.texture2DIndices[s]];
        if (scale[0] == 0.0f || (s_r.resX == 1.0f && s_r.resY == 1.0f && scale[0] == 1.0f && scale[1] == 1.0f))
            continue;
        // A shadow map read by a draw into a colour target (not the passes that copy it back into its depth surface,
        // which work in the game's texels): its own texels, with ShadowDetail.
        if (shadowDetail && s_r.shadowDescriptors[sc.texture2DIndices[s]])
            continue;
        sc.scaledTextures |= 1u << s;
        if (scale[0] != s_r.resX || scale[1] != s_r.resY)
        {
            sc.altScaledTextures |= 1u << s;
            sc.altScaleX = scale[0];
            sc.altScaleY = scale[1];
        }
    }
    if (SwizzleTrace())
        for (uint32_t s = 0; s < kFetchSlots; s++)
            if (ps && (ps->textureSlots & (1u << s)) && s_r.swizzleTraceDescriptors.contains(sc.texture2DIndices[s]))
            {
                static std::set<std::pair<uint64_t, uint32_t>> seen;
                if (seen.insert({ ps ? ps->hash : 0, s }).second)
                    log::Info("Swizzle trace: ps {:016X} (vs {:016X}) samples an image from an EDRAM swizzle in slot {} (frame {})",
                        ps ? ps->hash : 0, vs->hash, s, s_r.frame);
            }
    std::memcpy(shared.memory, &sc, sizeof(sc));

    uint32_t indexOffset = 0;
    if (!indices.empty())
    {
        auto ia = s_r.upload.Allocate(indices.size() * 4, 16);
        if (!ia.memory)
            return;
        UploadIndices(reinterpret_cast<uint32_t*>(ia.memory), indices.data(), indices.size(), indexBias);
        indexOffset = uint32_t(ia.offset);
    }

    // Targets last: texture uploads above end the render pass with their barriers.
    Targets targets;
    const bool overwritesColor = OverwritesColor(device, primType, firstVertex, vertexCount, !indices.empty(), *vs, *ps);
    t_overwritesColor = overwritesColor;
    // Volumetric smoke: after the effects composite (one full-screen draw a frame).
    constexpr uint64_t kEffectsCompositePs = 0x8E36B699AF21554Aull;
    uint32_t lateCaptureBase = 0, lateCaptureEnd = 0;   // a late mesh's capture slots
    const bool volComposite = s_r.volVs && ps->idHash == kEffectsCompositePs && s_r.volFrame == s_r.frame && s_r.volCount && !s_r.volComposited;
    // HDR from the scene: the scene composite (decoded scene x exposure + bloom into
    // the canvas, where the game clips at 1) and everything after it this frame draw into the canvas's float twin.
    constexpr uint64_t kSceneCompositePs = 0xD02C9AE3A213CABBull;
    if (ps->idHash == kSceneCompositePs && HdrSceneOn())
        s_r.hdrCanvasFrame = s_r.frame;
    if (ps->idHash == kSceneCompositePs)
        s_r.sceneCompositeFrame = s_r.frame;   // (a frame with a 3D picture: the lens effects)
    if (volComposite)
    {
        VolBeforeComposite(cl);
        RunCloudscape(cl, device);   // (if it didn't run at the scene's resolve)
        if (s_r.volCloudFrame == s_r.frame)
        {
            s_r.cloudRecFrame = s_r.frame;
            s_r.cloudRecCount = s_r.volCount;
            std::memcpy(s_r.cloudRecVp, s_r.volVp, sizeof(s_r.cloudRecVp));
        }
    }
    const bool bound = BindTargets(cl, device, targets, 0, false, !uiRedirect);
    t_overwritesColor = false;
    if (!bound)
        return;
    if (s_r.cloudEarlyFrame == s_r.frame && !uiRedirect)
        CloudGateAtDraw(cl, targets, ps->idHash);
    if (!uiRedirect && (vs->cloud || ps->cloud))
        GlowUnderClouds(cl, device, targets);
    if (uiRedirect)
    {
        targets.color = &s_r.uiImage;
        targets.depth = nullptr;
        targets.width = s_r.uiImage.width;
        targets.height = s_r.uiImage.height;
        targets.scale = 1;
        targets.samples = 1;
        Transition(cl, s_r.uiImage, RenderTextureLayout::COLOR_WRITE);
        cl.setFramebuffer(GetFramebuffer(&s_r.uiImage, nullptr));
        s_r.markedFramebuffer = nullptr;
    }
    // The late 3D pass after the effects composite (the canopy glass), multisampled.
    else if (ps->idHash == kEffectsCompositePs)
    {
        s_r.compositeTarget = targets.color;
        s_r.compositeFrame = s_r.frame;
    }
    else if (ps->idHash != kEffectsCompositePs && targets.color == s_r.compositeTarget && s_r.compositeFrame == s_r.frame && LateRectWanted())
    {
        // The late pass's rectangle: the draw's mesh bounded on screen, or captured this time.
        const uint32_t stream = layout.slots.empty() ? 0 : U32(device + kDevVertexFetch + (17 - layout.slots[0].index) * 8);
        const uint64_t key = HashBytes(&stream, sizeof(stream), vs->hash ^ (uint64_t(firstVertex) << 32 | vertexCount) ^
            (uint64_t(indices.size()) << 40) ^ (uint64_t(primType) << 56));
        // Slots: three a triangle (the capture stage sees triangles: a strip's n - 2).
        const uint32_t emitted = !indices.empty() ? uint32_t(indices.size()) : drawCount;
        const uint32_t triangleVertices = topology == RenderPrimitiveTopology::TRIANGLE_LIST ? emitted
            : topology == RenderPrimitiveTopology::TRIANGLE_STRIP && emitted >= 3 ? 3 * (emitted - 2) : 0;
        RenderRect rect;
        if (vs->positionMatrix != UINT32_MAX)
            LateReadCaptures();
        const bool known = vs->positionMatrix != UINT32_MAX && LateDrawRect(device, *vs, targets, key, triangleVertices, rect, lateCaptureBase, lateCaptureEnd);
        if (BeginLateDraw(cl, targets, vs->positionMatrix != UINT32_MAX, known ? &rect : nullptr) && lateCaptureEnd)
        {
            // Its capture's slots zeroed (a strip's restarts make fewer triangles than slots: the rest stay empty), then
            // written by the GPU (the buffer last read by a copy).
            cl.setFramebuffer(nullptr);
            const uint64_t bytes = uint64_t(lateCaptureEnd - lateCaptureBase) * 16;
            auto zeros = s_r.upload.Allocate(bytes, 16);
            if (zeros.memory)
            {
                std::memset(zeros.memory, 0, bytes);
                cl.barriers(RenderBarrierStage::COPY, RenderBufferBarrier(s_r.lateCapture.get(), RenderBufferAccess::WRITE));
                cl.copyBufferRegion(RenderBufferReference(s_r.lateCapture.get(), uint64_t(lateCaptureBase) * 16),
                    RenderBufferReference(s_r.upload.buffer.get(), zeros.offset), bytes);
            }
            cl.barriers(RenderBarrierStage::GRAPHICS, RenderBufferBarrier(s_r.lateCapture.get(), RenderBufferAccess::WRITE));
            cl.setFramebuffer(GetFramebuffer(targets.color, targets.depth));
            s_r.markedFramebuffer = nullptr;
        }
        else
            lateCaptureEnd = 0;
    }
    else
        BeginLateDraw(cl, targets, vs->positionMatrix != UINT32_MAX, nullptr);
    // The HDR output's UI: the frame-sized target as it was before the first 2D layer draw. A frame
    // with nothing else drawn into it before (menus, the briefing: the game's Flash UI layer, swg, draws the whole
    // screen) is all UI.
    const bool frameSized = targets.color && targets.color->samples == 1 && targets.color->width == s_r.sceneWidth &&
        targets.color->height == s_r.sceneHeight;
    if (!t_uiDraw && frameSized)
        s_r.sceneDrawFrame = s_r.frame;
    if (t_uiDraw && frameSized && s_r.preUiFrame != s_r.frame && s_r.allUiFrame != s_r.frame && UiSplit() && s_r.sceneDrawFrame != s_r.frame)
        s_r.allUiFrame = s_r.frame;
    else if (t_uiDraw && frameSized && s_r.preUiFrame != s_r.frame && s_r.allUiFrame != s_r.frame && UiSplit())
    {
        cl.setFramebuffer(nullptr);
        s_r.markedFramebuffer = nullptr;
        EnsureImage(s_r.preUi, targets.color->width, targets.color->height, targets.color->format, false);
        Transition(cl, *targets.color, RenderTextureLayout::COPY_SOURCE);
        Transition(cl, s_r.preUi, RenderTextureLayout::COPY_DEST);
        cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(s_r.preUi.texture.get()),
            RenderTextureCopyLocation::Subresource(targets.color->texture.get()));
        Transition(cl, s_r.preUi, RenderTextureLayout::SHADER_READ);
        Transition(cl, *targets.color, RenderTextureLayout::COLOR_WRITE);
        if (targets.depth)
            Transition(cl, *targets.depth, RenderTextureLayout::DEPTH_WRITE);
        cl.setFramebuffer(GetFramebuffer(targets.color, targets.depth));
        s_r.preUiFrame = s_r.frame;
    }
    g_video.CountGpuPassDraw();
    breadcrumbs::NoteDraw(cl, s_r.frame, vs->hash, ps ? ps->hash : 0);
    if (breadcrumbs::Detailed())
        breadcrumbs::Detail(DescribeDrawForBreadcrumb(sc, *vs, ps, targets, primType, firstVertex, vertexCount, uint32_t(indices.size())));
    // AC6_SKIP_VS=<hex prefix>[,<prefix>...]@<first>-<last>[;...]: skip draws whose vertex shader hash
    // starts with a prefix during those frames (bisecting which draws produce something on screen).
    // <prefix>#<n0>-<n1> skips only the n0-th to n1-th matching draws of each frame (counted from 0).
    static const auto skipGroups = [] {
        struct Group { std::vector<std::string> prefixes; uint64_t first, last; uint32_t nthFirst = 0, nthLast = UINT32_MAX; };
        std::vector<Group> groups;
        const char* v = std::getenv("AC6_SKIP_VS");
        std::stringstream all(v ? v : "");
        for (std::string group; std::getline(all, group, ';');)
        {
            const size_t at = group.find('@'), dash = group.find('-', at);
            if (at == std::string::npos || dash == std::string::npos)
                continue;
            Group g{ {}, std::stoull(group.substr(at + 1, dash - at - 1)), std::stoull(group.substr(dash + 1)) };
            std::stringstream list(group.substr(0, at));
            for (std::string prefix; std::getline(list, prefix, ',');)
            {
                if (const size_t hashAt = prefix.find('#'), nthDash = prefix.find('-', hashAt); hashAt != std::string::npos && nthDash != std::string::npos)
                {
                    g.nthFirst = uint32_t(std::stoul(prefix.substr(hashAt + 1, nthDash - hashAt - 1)));
                    g.nthLast = uint32_t(std::stoul(prefix.substr(nthDash + 1)));
                    prefix.resize(hashAt);
                }
                g.prefixes.push_back(prefix);
            }
            groups.push_back(std::move(g));
        }
        return groups;
    }();
    if (!skipGroups.empty())
    {
        static std::vector<uint32_t> matched(skipGroups.size());
        static uint64_t matchedFrame = UINT64_MAX;
        if (matchedFrame != s_r.frame)
        {
            std::fill(matched.begin(), matched.end(), 0);
            matchedFrame = s_r.frame;
        }
        const std::string hash = std::format("{:016X}", vs->hash);
        for (size_t i = 0; i < skipGroups.size(); i++)
        {
            const auto& g = skipGroups[i];
            if (s_r.frame >= g.first && s_r.frame <= g.last)
                for (auto& prefix : g.prefixes)
                    if (hash.starts_with(prefix))
                    {
                        const uint32_t n = matched[i]++;
                        if (n >= g.nthFirst && n <= g.nthLast)
                            return;
                        break;
                    }
        }
    }

    // AC6_DRAW_TRACE=<frame>: log every draw of that frame (finding a broken draw's shaders).
    static const uint64_t traceFrame = [] { const char* v = std::getenv("AC6_DRAW_TRACE"); return v ? std::stoull(v) : UINT64_MAX; }();
    if (s_r.frame == traceFrame)
    {
        std::string elements;
        for (size_t e = 0; e < layout.elements.size(); e++)
            elements += std::format(" {}:{}@{}+{}", layout.elements[e].location, int(layout.elements[e].format), layout.elements[e].slotIndex,
                layout.elements[e].alignedByteOffset);
        // The world-view-projection constants most AC6 vertex shaders use (c218-c221).
        {
            std::string m;
            for (uint32_t r = 218; r < 222; r++)
                for (uint32_t c = 0; c < 4; c++)
                    m += std::format(" {:.3g}", std::bit_cast<float>(U32(device + kDevVsConstants + r * 16 + c * 4)));
            log::Info("  c218-221:{}", m);
            // Non-finite or huge vertex constants (a CPU-side bug in the transforms hides geometry).
            std::string wild;
            for (uint32_t r = 0; r < 256; r++)
                for (uint32_t c = 0; c < 4; c++)
                    if (const float x = std::bit_cast<float>(U32(device + kDevVsConstants + r * 16 + c * 4)); !(std::fabs(x) < 1.0e7f))
                    {
                        wild += std::format(" c{}.{}={:.3g}", r, "xyzw"[c], x);
                        break;
                    }
            if (!wild.empty())
                log::Info("  wild vs constants:{}", wild);
        }
        // Dump stream 0 positions and the indices of each traced draw (offline mesh checks).
        if (!layout.slots.empty() && layout.strides[0] >= 12)
        {
            static uint32_t dumped = 0;
            const uint32_t fetch0 = device + kDevVertexFetch + 17 * 8;
            const uint32_t base = PhysicalToGuest(xenos::VertexFetch{ { U32(fetch0), U32(fetch0 + 4) } }.Address()) + firstVertex * layout.strides[0];
            std::ofstream f(std::filesystem::path(log::Directory()) / std::format("draw_{:03}_{:016X}.bin", dumped++, vs->hash), std::ios::binary);
            const uint32_t header[4] = { vertexCount, uint32_t(indices.size()), primType, layout.strides[0] };
            f.write(reinterpret_cast<const char*>(header), sizeof(header));
            for (uint32_t v = 0; v < vertexCount; v++)
                for (uint32_t c = 0; c < 3; c++)
                {
                    const float x = std::bit_cast<float>(U32(base + v * layout.strides[0] + c * 4));
                    f.write(reinterpret_cast<const char*>(&x), 4);
                }
            f.write(reinterpret_cast<const char*>(indices.data()), indices.size() * 4);
            for (uint32_t r = 218; r < 222; r++)
                for (uint32_t c = 0; c < 4; c++)
                {
                    const float x = std::bit_cast<float>(U32(device + kDevVsConstants + r * 16 + c * 4));
                    f.write(reinterpret_cast<const char*>(&x), 4);
                }
        }
        // Garbage-vertex census of stream 0 (position at +0): at draw time and at submission.
        if (!layout.slots.empty() && layout.strides[0] >= 12)
        {
            const uint32_t fetch0 = device + kDevVertexFetch + 17 * 8;
            const uint32_t base = PhysicalToGuest(xenos::VertexFetch{ { U32(fetch0), U32(fetch0 + 4) } }.Address()) + firstVertex * layout.strides[0];
            const uint32_t stride0 = layout.strides[0], count0 = vertexCount;
            auto census = [base, stride0, count0](const char* when, uint64_t vsHash) {
                uint32_t bad = 0, fill = 0;
                for (uint32_t v = 0; v < count0; v++)
                    for (uint32_t c = 0; c < 3; c++)
                    {
                        const uint32_t w = U32(base + v * stride0 + c * 4);
                        const float f = std::bit_cast<float>(w);
                        if (w == 0xFEFEFEFE)
                            fill++;
                        else if (!(std::fabs(f) < 1.0e5f))
                            bad++;
                    }
                log::Info("Vertex census {} vs {:016X}: {} vertices, {} fill words, {} wild words", when, vsHash, count0, fill, bad);
            };
            census("at draw", vs->hash);
            const uint64_t vsHash = vs->hash;
            s_r.pendingCopies.push_back([census, vsHash] { census("at submit", vsHash); });
        }
        for (auto& slot : layout.slots)
        {
            const uint32_t s = slot.index, fetch = device + kDevVertexFetch + (17 - s) * 8;
            xenos::VertexFetch vf{ { U32(fetch), U32(fetch + 4) } };
            std::string words;
            for (uint32_t i = 0; i < std::min<uint32_t>(layout.strides[s] / 4, 16); i++)
                words += std::format(" {:08X}", U32(PhysicalToGuest(vf.Address()) + firstVertex * layout.strides[s] + i * 4));
            elements += std::format(" [stream {} fetch {:08X} {:08X} stride {}:{}]", s, vf.dw[0], vf.dw[1], layout.strides[s], words);
        }
        std::string textures;
        for (uint32_t s = 0; s < kUsedFetchSlots; s++)
        {
            const auto f = xenos::ReadTextureFetch(g_memory.As<uint8_t>(device + kDevTextureFetch + s * 24));
            if (f.Type() == 2 && f.BaseAddress())
                textures += std::format(" t{}={}@{:08X}/f{}d{}{}", s, f.Dimension() == 3 ? sc.textureCubeIndices[s] : sc.texture2DIndices[s],
                    f.BaseAddress(), f.Format(), f.Dimension(), (f.Dimension() == 3 ? sc.textureCubeIndices[s] == kNullTextureCube : sc.texture2DIndices[s] == kNullTexture2D) ? "NULL" : "");
        }
        // Each distinct texture once: its fetch constant, and its guest data (first 1 MB) for offline decoding.
        {
            static std::set<uint32_t> dumpedTextures;
            std::string details;
            for (uint32_t s = 0; s < kUsedFetchSlots; s++)
            {
                const auto f = xenos::ReadTextureFetch(g_memory.As<uint8_t>(device + kDevTextureFetch + s * 24));
                if (f.Type() != 2 || !f.BaseAddress())
                    continue;
                details += std::format(" t{}={}x{} pitch {} tiled {} endian {} maxmip {} mips@{:08X} sw {:03X} [{:08X} {:08X} {:08X} {:08X} {:08X} {:08X}]", s,
                    f.Width(), f.Height(), f.Pitch(), int(f.Tiled()), f.Endian(), f.MaxMip(), f.MipAddress(), (f.dw[3] >> 1) & 0xFFF,
                    f.dw[0], f.dw[1], f.dw[2], f.dw[3], f.dw[4], f.dw[5]);
                if (dumpedTextures.insert(f.BaseAddress()).second && f.Dimension() == 1)
                {
                    std::ofstream out(std::filesystem::path(log::Directory()) / std::format("tex_{:08X}.bin", f.BaseAddress()), std::ios::binary);
                    out.write(reinterpret_cast<const char*>(f.dw), sizeof(f.dw));
                    out.write(reinterpret_cast<const char*>(g_memory.As<uint8_t>(PhysicalToGuest(f.BaseAddress()))), 1 << 20);
                }
            }
            log::Info("  textures:{}", details);
        }
        log::Info("  state: colorcontrol {:08X} alpharef {} depth {:08X} stencil {:08X}/{:08X} blend {:08X} ({} {} {} {}) mask {:08X} scmode {:08X} clip {:08X} polyoffset {} {} {} {}",
            U32(device + kDevColorControl), std::bit_cast<float>(U32(device + kDevAlphaRef)), U32(device + kDevDepthControl),
            U32(device + kDevStencilRefMask), U32(device + kDevStencilRefMaskBack),
            U32(device + kDevBlendControl), std::bit_cast<float>(U32(device + kDevBlendConstant)), std::bit_cast<float>(U32(device + kDevBlendConstant + 4)),
            std::bit_cast<float>(U32(device + kDevBlendConstant + 8)), std::bit_cast<float>(U32(device + kDevBlendConstant + 12)),
            U32(device + kDevColorMask), U32(device + kDevScModeControl), U32(device + kDevClipControl),
            std::bit_cast<float>(U32(device + kDevPolyOffset)), std::bit_cast<float>(U32(device + kDevPolyOffset + 4)),
            std::bit_cast<float>(U32(device + kDevPolyOffset + 8)), std::bit_cast<float>(U32(device + kDevPolyOffset + 12)));

        log::Info("Draw: vs {:016X} (object {:08X}) ps {:016X} prim {} first {} count {} indices {} rt {:08X} ds {:08X} spec {:X} |{} |{}", vs->hash,
            U32(device + kDevVertexShader), ps->hash, primType, firstVertex, vertexCount, indices.size(), U32(device + kDevRenderTargets), U32(device + kDevDepthStencil), specConstants,
            elements, textures);
    }
    // Rectangles and point sprites are expanded with their own winding: never culled.
    s_r.rectList = primType == xenos::kPrimRectList || pointSprites;
    // AC6_SKIP_POLY_OFFSET=1 / =2: skip draws with / without polygon offset (diagnostics).
    static const int skipPolyOffset = [] { const char* v = std::getenv("AC6_SKIP_POLY_OFFSET"); return v ? std::atoi(v) : 0; }();
    if (skipPolyOffset && ((U32(device + kDevScModeControl) & 0x1800) != 0) == (skipPolyOffset == 1))
        return;
    // Volumetric smoke: a smoke draw's particles (two triangles each) get record slots and its
    // pipeline the harvesting geometry shader.
    int volLayout = s_r.volHarvestGs && topology == RenderPrimitiveTopology::TRIANGLE_LIST ? VolumetricLayout(ps->idHash) : -1;
    uint32_t volBase = 0, volEnd = 0;
    if (volLayout >= 0)
    {
        VolBeginFrame();
        // As the draw goes to the GPU: quad lists come as indices whether the game's draw was indexed or not (counting
        // their vertices would reserve 4 slots for 6, and the rest of a draw's records would overwrite the next draws'
        // slots and run past the buffer).
        const uint32_t particles = (!indices.empty() ? uint32_t(indices.size()) : drawCount) / 6;
        if (!s_r.volTarget)
            s_r.volTarget = targets.color;
        if (particles && s_r.volCount + particles <= kVolCapacity && !s_r.volComposited && targets.color == s_r.volTarget)
        {
            volBase = s_r.volCount;
            s_r.volCount += particles;
            volEnd = s_r.volCount;
            VolNoteConstants(device, volLayout);
        }
        else
            volLayout = -1;
    }
    // TAA's motion vectors: a draw into the scene writes its motion where it covers what is behind
    // (it writes depth, or colour unblended). Translucent ones (smoke, glass, glows) leave the motion of what they're over.
    if (targets.motion && ps->motionOutput && volLayout < 0)
    {
        const bool depthWrite = targets.depth && (U32(device + kDevDepthControl) & 6) == 6;
        const bool opaque = U32(device + kDevBlendControl) == kBlendDisabled && (U32(device + kDevColorMask) & 0xF);
        if (depthWrite || opaque)
        {
            specConstants |= SPEC_CONSTANT_MOTION;
            sc.drawFlags = specConstants;
            std::memcpy(shared.memory + offsetof(SharedConstants, drawFlags), &sc.drawFlags, sizeof(sc.drawFlags));
        }
        else if (targets.motion->format == RenderFormat::R16G16B16A16_FLOAT && (U32(device + kDevColorMask) & 0xF) &&
            vs->effect && primType != xenos::kPrimRectList)
        {
            // FSR's reactive mask: an effect's coverage (its vertex shader from the game's Effect folder, HostShader::effect:
            // in the scene pass the afterburners, bullets and tracers, vapour, searchlights and light sprites. The smoke,
            // fire and explosions come later, over FSR's result). Not translucent parts of models and the map, or the
            // clouds, which move with the scene. Additive ones (the destination kept whole) by their brightest channel,
            // the others by their alpha. Not a full-screen pass (a rectangle list).
            const uint32_t control = U32(device + kDevBlendControl);
            specConstants |= SPEC_CONSTANT_REACTIVE | (((control >> 8) & 0x1F) == 1 ? SPEC_CONSTANT_REACTIVE_LUMA : 0);
            // AC6_FSR_REACTIVE_LOG=1: each pixel shader the mask takes, once.
            static const bool logReactive = [] { const char* v = std::getenv("AC6_FSR_REACTIVE_LOG"); return v && *v == '1'; }();
            static std::unordered_set<uint64_t> logged;
            if (logReactive && logged.size() < 400 && logged.insert(ps->idHash).second)
                log::Info("FSR reactive: ps {:016X} {} vs {:016X} {} (blend {:08X}, primitive {}, {} vertices{})", ps->idHash, ps->source,
                    vs->hash, vs->source, control, primType, drawCount, (specConstants & SPEC_CONSTANT_REACTIVE_LUMA) ? ", additive" : "");
            sc.drawFlags = specConstants;
            std::memcpy(shared.memory + offsetof(SharedConstants, drawFlags), &sc.drawFlags, sizeof(sc.drawFlags));
        }
    }
    // The sun's disc on screen: its quad's centre c100.xy and half sizes c101.xy in clip space (the vertex shader
    // D66D6627B10F1081, the half sizes already made round on a wide output), 0-1 across the scene.
    if (targets.color && ps->idHash == 0xBE0A9DB65D5CB28Cull && s_r.hdrCanvasFrame != s_r.frame && SceneSized(*targets.color) && HdrSceneOn())
    {
        auto vsc = [&](uint32_t reg, uint32_t i) { return std::bit_cast<float>(U32(device + kDevVsConstants + reg * 16 + i * 4)); };
        const float cx = vsc(100, 0) * 0.5f + 0.5f, cy = 0.5f - vsc(100, 1) * 0.5f;
        const float hx = std::fabs(vsc(101, 0)) * 0.5f, hy = std::fabs(vsc(101, 1)) * 0.5f;
        if (std::isfinite(cx) && std::isfinite(cy) && std::isfinite(hx) && std::isfinite(hy) && hx < 1.0f && hy < 1.0f)
        {
            const float box[4] = { cx - hx, cy - hy, cx + hx, cy + hy };
            if (s_r.sunFrame != s_r.frame)
                std::copy(box, box + 4, s_r.sunBox);
            else
                s_r.sunBox[0] = std::min(s_r.sunBox[0], box[0]), s_r.sunBox[1] = std::min(s_r.sunBox[1], box[1]),
                s_r.sunBox[2] = std::max(s_r.sunBox[2], box[2]), s_r.sunBox[3] = std::max(s_r.sunBox[3], box[3]);
            s_r.sunFrame = s_r.frame;
        }
    }
    // The game's lighting in float: the sun's shaders (its disc BE0A9DB65D5CB28C, its glow
    // 80390F4960C3D4A9: they write the game's emitter codes 241-255, its 1.1-100) drawing into a float scene surface before
    // the composite write their colour decoded, as the game's decoders would read it (the shared draw flags only).
    if (targets.color && (ps->idHash == 0xBE0A9DB65D5CB28Cull || ps->idHash == 0x80390F4960C3D4A9ull) && s_r.hdrCanvasFrame != s_r.frame &&
        targets.color->format == RenderFormat::R16G16B16A16_FLOAT && targets.color->bgraOrder && SceneSized(*targets.color) && SceneFloatOn())
    {
        sc.drawFlags = specConstants | SPEC_CONSTANT_EMITTER_DECODE;
        std::memcpy(shared.memory + offsetof(SharedConstants, drawFlags), &sc.drawFlags, sizeof(sc.drawFlags));
    }
    // The 2D's lines (the 2D layers' and those without the depth test, 1 guest pixel wide, single-sample targets) as
    // Bresenham parallelograms on the guest's grid (line2dGs), at every size: a strip's corners always drawn (at 720p
    // the diamond rule left one to neither segment when the HUD swayed between pixels), whole guest pixels above it.
    // The grid comes from the vertex shader (the line snap above the guest's size, LINE_GRID at it).
    const bool lineTopology = topology == RenderPrimitiveTopology::LINE_LIST || topology == RenderPrimitiveTopology::LINE_STRIP;
    const uint32_t lineHalfWidth = U32(device + kDevPointSize + 8) & 0xFFFF;
    const bool lineRects = s_r.line2dGs && lineTopology && (t_uiDraw || !(U32(device + kDevDepthControl) & 2)) && targets.samples == 1 &&
        (lineHalfWidth == 0 || lineHalfWidth == 8) && volLayout < 0 && !lateCaptureEnd;
    const PendingPipeline* pipeline = GetPipeline(device, *vs, *ps, layout, topology,
        specConstants | (volLayout >= 0 ? kSpecVolHarvest : 0) | (lateCaptureEnd ? kSpecLateBounds : 0) | (lineRects ? kSpecLine2D : 0), targets);
    if (pipeline && targets.depth && (U32(device + kDevDepthControl) & 6) == 6)
        NoteDepthWrite(targets.depth);
    if (pipeline && targets.color && (U32(device + kDevColorMask) & 0xF))
        NoteColorWrite(targets.color);
    if (!pipeline)
        return;

    RenderViewport viewport = ReadViewport(device, targets);
    // Half-pixel offset, opt-in with AC6_HALF_PIXEL_OFFSET=1: the console rasterises with
    // pixel centres at integer window positions (Direct3D 9: AC6's command buffer sets PA_SU_VTX_CNTL to 4,
    // pix_center 0), Vulkan at .5. Applied to every draw it breaks AC6's depth of field. On multisampled targets
    // alone it only moves the dashes along the horizon (which centroid interpolation fixes). Off by default: no pass
    // is known to need it.
    static const bool halfPixelOffset = [] { const char* v = std::getenv("AC6_HALF_PIXEL_OFFSET"); return v && std::strcmp(v, "1") == 0; }();
    // The 2D (the 2D layers' draws: HUD, menus, text; and lines without the depth test) gets it where the draw asks
    // for Direct3D's pixel centres (PA_SU_VTX_CNTL's pix_center 0), as on the console and in Xenia (its
    // half_pixel_offset, on by default: +0.5 across and down for those draws). AC6 sets them for the HUD's lines (4) and
    // OpenGL's for its sprites, text and fills (5): the lines are placed for the offset (without it every frame line sat
    // a pixel up and left of its fill's edge, the radar's, the PiP's and the orders bar's: a gap on the top and left, an
    // overlap on the bottom and right; a recording of the original and rexglue show neither), the sprites without it
    // (with it their texels were sampled half a texel off: blurred text and icons). AC6_HALF_PIXEL_2D=0: never (the
    // 3D never has it).
    static const bool halfPixel2D = [] { const char* v = std::getenv("AC6_HALF_PIXEL_2D"); return !(v && *v == '0'); }();
    const bool line2D = (topology == RenderPrimitiveTopology::LINE_LIST || topology == RenderPrimitiveTopology::LINE_STRIP) &&
        (t_uiDraw || !(U32(device + kDevDepthControl) & 2));
    const bool offset2D = halfPixel2D && (t_uiDraw || line2D) && !(U32(device + kDevVtxControl) & 1);
    if (((halfPixelOffset && targets.samples > 1) || offset2D) && viewport.width > 0.0f && viewport.height > 0.0f)
    {
        const float offset[2] = { 1.0f / viewport.width, -1.0f / viewport.height };
        std::memcpy(shared.memory + offsetof(SharedConstants, halfPixelOffsetX), offset, sizeof(offset));
        std::memcpy(&sc.halfPixelOffsetX, offset, sizeof(offset));
    }
    // 2D above the guest's size, after the half-pixel offset (above), on the game's pixel grid. Lines: a 1-pixel line
    // covers the pixel its vertices are in, or, on the edge between two pixels, the one left of and above it; a wider
    // line (LineWidth) is centred on its vertices, so on an edge it would straddle it. The vertex shader moves each
    // vertex on the half-pixel grid onto the centre of that pixel (SPEC_CONSTANT_LINE_SNAP; the grid below); others
    // (circles, moving lines) stay. Not the scene's lines (the contrails): they would move in whole guest pixels. The 2D
    // layers' untextured triangles: an edge exactly through pixel centres covers those pixels at the guest's size (the
    // left / top tie rule) but would cut them in half above it; a vertex on a pixel centre moves onto the pixel's edge
    // (SPEC_CONSTANT_EDGE_SNAP), others stay. Not textured ones: their texture would move with the vertex. At the
    // guest's size nothing changes.
    const bool lineDraw = topology == RenderPrimitiveTopology::LINE_LIST || topology == RenderPrimitiveTopology::LINE_STRIP;
    // (Lines: the 2D layers' and any without the depth test, not the scene's depth-tested contrails, also on a
    // single-sample scene surface.)
    const bool snap2D = (lineDraw && (t_uiDraw || !(U32(device + kDevDepthControl) & 2))) ||
        (!lineDraw && t_uiDraw && !pointSprites && ps && ps->ok && ps->textureSlots == 0);
    const bool snapScaled = snap2D && targets.samples == 1 && float(targets.scale) * std::min(s_r.resX, s_r.resY) > 1.001f &&
        viewport.width > 0.0f && viewport.height > 0.0f;
    // The rest of the 2D layers' draws above the guest's size (sprites, text) mapped the same way, not snapped
    // (LINE_GRID): a layer's contents all at the frame's scale. (The line parallelograms need the grid at the guest's
    // size too: written, the vertices left as they are.)
    const bool remap2D = !snapScaled && t_uiDraw && !pointSprites && !lineDraw && targets.samples == 1 &&
        float(targets.scale) * std::min(s_r.resX, s_r.resY) > 1.001f && viewport.width > 0.0f && viewport.height > 0.0f;
    const bool snapGrid = snapScaled || remap2D || (lineRects && viewport.width > 0.0f && viewport.height > 0.0f);
    // The line parallelograms put a frame's side on the inside of its box, the right of the line's way, the boxes' way
    // round (the radar's, the PiP's, the score and time boxes'). A strip the other way round (the orders bar's outline, an
    // open U) has its segments marked for the geometry shader by their primitive IDs (the first 64 a draw makes): the
    // inside on their left. (Reversed, an open strip would lose its first pixel: Bresenham's last pixel is left out.)
    if (lineRects)
    {
        uint32_t flip[2] = { 0, 0 };
        if (snapScaled && topology == RenderPrimitiveTopology::LINE_STRIP && !layout.slots.empty() && !layout.elements.empty())
        {
            const uint32_t fetch0 = device + kDevVertexFetch + (17 - layout.slots[0].index) * 8;
            const uint32_t stride = layout.strides[layout.slots[0].index];
            const uint32_t base = PhysicalToGuest(xenos::VertexFetch{ { U32(fetch0), U32(fetch0 + 4) } }.Address()) +
                layout.elements[0].alignedByteOffset;
            auto at = [&](size_t i) {
                const uint32_t index = indices.empty() ? firstVertex + uint32_t(i) : indices[i];
                return std::pair<double, double>{ std::bit_cast<float>(U32(base + index * stride)), std::bit_cast<float>(U32(base + index * stride + 4)) };
            };
            const size_t count = indices.empty() ? vertexCount : indices.size();
            uint32_t segment = 0;
            for (size_t first = 0; first < count && segment < 64;)
            {
                size_t last = first;
                while (last < count && (indices.empty() || indices[last] != 0xFFFFFFFFu))
                    last++;
                if (last - first >= 2)
                {
                    double area = 0.0;   // twice the signed area, y down: positive clockwise
                    for (size_t i = first; i < last; i++)
                    {
                        const auto [x0, y0] = at(i);
                        const auto [x1, y1] = at(i + 1 < last ? i + 1 : first);
                        area += x0 * y1 - x1 * y0;
                    }
                    for (size_t i = first; i + 1 < last; i++, segment++)
                        if (area < 0.0 && segment < 64)
                            flip[segment / 32] |= 1u << (segment % 32);
                }
                first = last + 1;
            }
        }
        std::memcpy(shared.memory + offsetof(SharedConstants, lineFlip), flip, sizeof(flip));
        std::memcpy(sc.lineFlip, flip, sizeof(flip));
    }
    if (snapGrid)
    {
        // (Textured 2D snapped too where a guest pixel isn't a whole number of host pixels: a sprite's edge on a host
        // pixel's centre missed the row the fill or the line beside it starts at, the ALL highlight a row short of the
        // orders bar's outline at 1080p. At whole scales its edges are on host pixel edges already: it stays.)
        const float frameScale = float(targets.scale) * std::min(s_r.resX, s_r.resY);
        const bool wholeScale = std::fabs(frameScale - std::round(frameScale)) < 1e-3f;
        sc.drawFlags |= (remap2D && !wholeScale) ? SPEC_CONSTANT_EDGE_SNAP : !snapScaled ? SPEC_CONSTANT_LINE_GRID :
            lineDraw ? SPEC_CONSTANT_LINE_SNAP : SPEC_CONSTANT_EDGE_SNAP;
        std::memcpy(shared.memory + offsetof(SharedConstants, drawFlags), &sc.drawFlags, sizeof(sc.drawFlags));
    }
    // Centroid interpolation: on multisampled targets the console interpolates at the
    // centroid of the covered samples unless SQ_CONTEXT_MISC's sample control asks for centres (all, or the
    // interpolators SQ_INTERPOLATOR_CNTL's sampling pattern marks). At the pixel centre a partly covered
    // pixel extrapolates: the sky dome's horizon edge would read its texture outside the triangle, a dark line.
    if (targets.samples > 1)
    {
        uint32_t centroid = 0;
        switch ((U32(device + kDevContextMisc) >> 2) & 3)
        {
        case 0: centroid = 0xFFFF; break;
        case 2: centroid = ~(U32(device + kDevInterpolatorControl) >> 16) & 0xFFFF; break;
        }
        // Every input at the centroid: the inputs a draw asks centres for still extrapolate on the
        // stair steps of an edge (black dots along the F-16C's wing leading edges and fin). AC6_CENTROID=0: as asked.
        static std::atomic<uint64_t> centroidGeneration{ UINT64_MAX };   // (live)
        static std::atomic<bool> centroidValue{ true };
        if (const uint64_t now = settings::Generation(); centroidGeneration.load(std::memory_order_acquire) != now)
        {
            const std::string v = settings::Get("AC6_CENTROID");
            centroidValue.store(v != "0" && v != "false", std::memory_order_relaxed);
            centroidGeneration.store(now, std::memory_order_release);
        }
        const bool allCentroid = centroidValue.load(std::memory_order_relaxed);
        if (allCentroid)
            centroid = 0xFFFF;
        std::memcpy(shared.memory + offsetof(SharedConstants, interpolatorCentroid), &centroid, sizeof(centroid));
        sc.interpolatorCentroid = centroid;
    }
    // Point sizes from guest pixels to NDC: a diameter of d guest pixels is d / (viewport extent) in NDC radius units.
    // On a wide output a guest pixel is wider than tall on screen, so the width is measured in the vertical's pixels
    // (x scaled by resY / resX) and points stay round.
    if (pointSprites && viewport.width > 0.0f && viewport.height > 0.0f)
    {
        const float scale[2] = { s_r.resY / (s_r.resX * viewport.width), 1.0f / viewport.height };
        std::memcpy(shared.memory + offsetof(SharedConstants, pointNdcPerPixelX), scale, sizeof(scale));
        std::memcpy(&sc.pointNdcPerPixelX, scale, sizeof(scale));
    }
    RenderViewport exactViewport;   // (the guest viewport's exact place in host pixels, below)
    bool exactRounded = false;
    // Guest pixels to the target's host pixels: its 4x MSAA promotion and the internal resolution. The edges
    // go through ToHost as the image sizes do: at a fractional scale a 1x1 target is 2x2 host pixels, and a
    // viewport of 1.5 would cover one of them (the exposure's 1x1 average then reads three unwritten texels: white).
    if ((targets.scale != 1 || s_r.resX != 1.0f || s_r.resY != 1.0f) && (U32(device + kDevVteControl) & 0x3F) == 0x3F)
    {
        if (s_r.resX == 1.0f && s_r.resY == 1.0f)
        {
            const float scale = float(targets.scale);
            viewport.x *= scale;
            viewport.y *= scale;
            viewport.width *= scale;
            viewport.height *= scale;
        }
        else
        {
            // A 2D layer's clip rect on a wide or narrow output (NoteLayerClip): its canvas pixels on the screen, the
            // canvas in the middle at 16:9, its pixel the shorter axis's host pixels. Through the rounded guest
            // viewport its edges fell up to half a guest pixel off the canvas's pixels (the radar's clip, and its
            // contents scaled to fit).
            std::array<int32_t, 4> canvas;
            if (std::fabs(s_r.resX - s_r.resY) > 1e-4f && targets.width == s_r.targetWidth && targets.height == s_r.targetHeight &&
                LayerClip(viewport, canvas))
            {
                const float p = float(targets.scale) * std::min(s_r.resX, s_r.resY);
                viewport.x = 0.5f * float(targets.width) + (float(canvas[0]) - 0.5f * float(kGuestWidth)) * p;
                viewport.y = 0.5f * float(targets.height) + (float(canvas[1]) - 0.5f * float(kGuestHeight)) * p;
                viewport.width = float(canvas[2]) * p;
                viewport.height = float(canvas[3]) * p;
            }
            else
            {
                const double k = double(targets.scale);
                const float x0 = float(ToHostX(viewport.x * k)), y0 = float(ToHostY(viewport.y * k));
                const float x1 = float(ToHostX((double(viewport.x) + viewport.width) * k)), y1 = float(ToHostY((double(viewport.y) + viewport.height) * k));
                // (Its exact place for the 2D's grid: a 2D layer's, the radar's 191 guest pixels at 94, 458, is 238.75
                // host pixels at 117.5, 572.5 at 900p, drawn in 238 at 118, 573. The 2D's vertex shaders map their
                // contents from the exact place onto the drawn one, below.)
                exactViewport = RenderViewport(float(double(viewport.x) * k * s_r.resX), float(double(viewport.y) * k * s_r.resY),
                    float(double(viewport.width) * k * s_r.resX), float(double(viewport.height) * k * s_r.resY));
                viewport.x = x0;
                viewport.y = y0;
                viewport.width = x1 - x0;
                viewport.height = y1 - y0;
                exactRounded = true;
            }
        }
    }
    if (!exactRounded)
        exactViewport = viewport;
    // The snaps' grid: the game's pixels in host pixels, as the shader finds a vertex's place on it, g = ndc x A + B
    // (y down). On a wide or narrow output the frame's 2D is the 1280x720 canvas in the middle at 16:9 (the 2D layers'
    // ortho), its pixel the shorter axis's host pixels across both ways, from the frame's centre: there the guest
    // viewport's pixels are wider (or taller) than the HUD's, and a snap on them left the radar's gaps uneven (1 and 3
    // host pixels left and right at 3440x1440). Other targets: their guest pixels.
    if (snapGrid)
    {
        float px = float(targets.scale) * s_r.resX, py = float(targets.scale) * s_r.resY, ox = 0.0f, oy = 0.0f;
        if (std::fabs(s_r.resX - s_r.resY) > 1e-4f && targets.width == s_r.targetWidth && targets.height == s_r.targetHeight)
        {
            px = py = float(targets.scale) * std::min(s_r.resX, s_r.resY);
            ox = 0.5f * float(targets.width) - 0.5f * float(kGuestWidth) * px;
            oy = 0.5f * float(targets.height) - 0.5f * float(kGuestHeight) * py;
        }
        const RenderViewport& e = exactViewport;
        const float grid[4] = { 0.5f * e.width / px, -0.5f * e.height / py, (0.5f * e.width + e.x - ox) / px, (0.5f * e.height + e.y - oy) / py };
        std::memcpy(shared.memory + offsetof(SharedConstants, pointDiameterMin), grid, sizeof(grid));
        std::memcpy(&sc.pointDiameterMin, grid, sizeof(grid));
        const float host[8] = { px, py, ox, oy, 0.5f * viewport.width, -0.5f * viewport.height, viewport.x + 0.5f * viewport.width,
            viewport.y + 0.5f * viewport.height };
        std::memcpy(shared.memory + offsetof(SharedConstants, hostGridX), host, sizeof(host));
        std::memcpy(&sc.hostGridX, host, sizeof(host));
    }
    const RenderViewport steadyViewport = viewport;   // (the planar reflection's: without the jitter)
    // Temporal AA: the scene surface's draws move by this frame's sub-pixel jitter.
    // Not the 2D layers' draws: in the hangar they go into the scene's surface after the scene's
    // resolve, so nothing resolves their jitter, and the Flash UI layer's (swg's) clip (whole pixels) would not move
    // with its contents:
    // a pixel column at the 16:9 box's edge would be covered on some frames by neither the menu's bands nor their wrapped
    // copies (faint light lines at the box's edges where the hangar's top and bottom bands darken it).
    // Moved in clip space (the half-pixel offset's oPos.xy += offset x oPos.w, before the y flip), not by moving the
    // viewport: Vulkan clips to the viewport, so a viewport moved up or left by up to half a pixel left the samples past
    // its far edges uncovered on those frames.
    if ((TaaOn() || FsrRunning()) && targets.color && targets.color == s_r.taaSurface && TaaSurfaceLive() && !t_uiDraw &&
        viewport.width > 0.0f && viewport.height > 0.0f)
    {
        // (FSR's jitter only while it runs; the camera below is gathered either way, FSR needs it to start.)
        const auto [jx, jy] = TaaOn() || FsrJitters() ? TaaJitter(SeedFrame()) : std::pair<float, float>{ 0.0f, 0.0f };
        const float offset[2] = { sc.halfPixelOffsetX + 2.0f * jx / viewport.width, sc.halfPixelOffsetY - 2.0f * jy / viewport.height };
        std::memcpy(shared.memory + offsetof(SharedConstants, halfPixelOffsetX), offset, sizeof(offset));
        std::memcpy(&sc.halfPixelOffsetX, offset, sizeof(offset));
        // The world's view-projection (c116-c119) as a perspective draw into the scene has it: its z column depends on the
        // position (the sky's and the clouds' pin z to the far plane), and so does w. Not every such draw holds it there:
        // ~60 of a frame's ~670 (some object shaders) put their own object's world x view-projection in c116-c119, and
        // taking the frame's last perspective draw, often one of those, would make the reprojection follow that object
        // (a moving one would smear the whole frame). The matrix most draws use is the world's.
        double m[16];
        for (uint32_t i = 0; i < 16; i++)
            m[i] = double(std::bit_cast<float>(U32(device + kDevVsConstants + 116 * 16 + i * 4)));
        const bool perspective = std::fabs(m[2]) + std::fabs(m[6]) + std::fabs(m[10]) > 1e-6 && std::fabs(m[3]) + std::fabs(m[7]) + std::fabs(m[11]) > 1e-6;
        // (The draw's own position matrix too, for a scene without the world camera: the hangar.)
        if (!perspective && vs->positionMatrix != UINT32_MAX && vs->constantFirst <= vs->positionMatrix &&
            vs->positionMatrix + 4 <= vs->constantEnd)
        {
            double o[16];
            for (uint32_t i = 0; i < 16; i++)
                o[i] = double(std::bit_cast<float>(U32(device + kDevVsConstants + vs->positionMatrix * 16 + i * 4)));
            if (std::fabs(o[3]) + std::fabs(o[7]) + std::fabs(o[11]) > 1e-6)
            {
                if (s_r.taaObjectFrame != s_r.frame)
                    s_r.taaObjectCandidates.clear();
                s_r.taaObjectFrame = s_r.frame;
                auto& objects = s_r.taaObjectCandidates;
                auto same = [&](const Renderer::TaaVpCandidate& c) { return std::memcmp(c.m, o, sizeof(o)) == 0; };
                if (auto it = std::find_if(objects.begin(), objects.end(), same); it != objects.end())
                    it->draws++;
                else if (objects.size() < 256)
                {
                    objects.push_back({ {}, 1 });
                    std::memcpy(objects.back().m, o, sizeof(o));
                }
            }
        }
        // The draw's motion: the matrix its vertex shader transforms the position with (by name:
        // HostShader::positionMatrix) against the same object's last frame, when it holds a perspective transform.
        if (specConstants & SPEC_CONSTANT_MOTION)
        {
            const uint32_t reg = vs->positionMatrix;
            const bool reads = reg != UINT32_MAX && vs->constantFirst <= reg && reg + 4 <= vs->constantEnd;
            double p[16] = {};
            if (reads)
                for (uint32_t i = 0; i < 16; i++)
                    p[i] = double(std::bit_cast<float>(U32(device + kDevVsConstants + reg * 16 + i * 4)));
            const bool transform = reads && std::fabs(p[3]) + std::fabs(p[7]) + std::fabs(p[11]) > 1e-6;
            const uint32_t stream = layout.slots.empty() ? 0 : U32(device + kDevVertexFetch + (17 - layout.slots[0].index) * 8);
            const uint64_t object = HashBytes(&stream, sizeof(stream), vs->hash ^ (uint64_t(firstVertex) << 32 | vertexCount) ^
                (uint64_t(indices.size()) << 40) ^ (uint64_t(primType) << 56));
            MotionTransform(object, p, transform, sc.motion);
            std::memcpy(shared.memory + offsetof(SharedConstants, motion), sc.motion, sizeof(sc.motion));
            // AC6_TAA_DEBUG=8: every 120th frame's motion draws.
            static const bool traceMotion = [] { const char* v = std::getenv("AC6_TAA_DEBUG"); return v && std::atoi(v) == 8; }();
            if (traceMotion && s_r.frame % 120 == 0)
                log::Info("TAA motion draw: vs {:016X} ps {:016X} prim {} first {} count {} indices {} stream {:08X} object {:016X}: c{} {} {}",
                    vs->hash, ps->hash, primType, firstVertex, vertexCount, indices.size(), stream, object, reg == UINT32_MAX ? -1 : int(reg),
                    transform ? "perspective" : "no transform", sc.motion[15] != 0.0f || sc.motion[0] != 0.0f ? "paired" : "none");
        }
        if (perspective)
        {
            // AC6_TAA_DEBUG=6: every view-projection change between the frame's scene draws (which draws use which).
            static const bool traceVp = [] { const char* v = std::getenv("AC6_TAA_DEBUG"); return v && std::atoi(v) == 6; }();
            static uint64_t traceFrame = ~0ull;
            static uint32_t traceDraw = 0;
            if (traceVp && traceFrame != s_r.frame)
            {
                traceFrame = s_r.frame;
                traceDraw = 0;
            }
            traceDraw++;
            if (traceVp && (s_r.taaVpFrame != s_r.frame || std::memcmp(s_r.taaVp, m, sizeof(m)) != 0))
                log::Info("TAA vp frame {} draw {}: {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} vs {:016X}",
                    s_r.frame, traceDraw, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15],
                    vs ? vs->hash : 0);
            if (s_r.taaVpFrame != s_r.frame)
                s_r.taaVpCandidates.clear();
            s_r.taaVpFrame = s_r.frame;
            auto& candidates = s_r.taaVpCandidates;
            auto same = [&](const Renderer::TaaVpCandidate& c) { return std::memcmp(c.m, m, sizeof(m)) == 0; };
            if (!candidates.empty() && same(candidates.back()))
                candidates.back().draws++;
            else if (auto it = std::find_if(candidates.begin(), candidates.end(), same); it != candidates.end())
                it->draws++;
            else if (candidates.size() < 256)
            {
                candidates.push_back({ {}, 1 });
                std::memcpy(candidates.back().m, m, sizeof(m));
            }
            std::memcpy(s_r.taaVp, m, sizeof(m));
        }
    }
    else if (targets.color && targets.color == s_r.cloudSceneSurface && CloudscapeEarly())
    {
        // (The early cloudscape's camera: as above.)
        double m[16];
        for (uint32_t i = 0; i < 16; i++)
            m[i] = double(std::bit_cast<float>(U32(device + kDevVsConstants + 116 * 16 + i * 4)));
        if (std::fabs(m[2]) + std::fabs(m[6]) + std::fabs(m[10]) > 1e-6 && std::fabs(m[3]) + std::fabs(m[7]) + std::fabs(m[11]) > 1e-6)
            NoteSceneVp(m);
    }
    SetViewport(cl, viewport);
    SetScissor(cl, RenderRect(0, 0, int32_t(targets.width), int32_t(targets.height)));
    SetPipeline(cl, pipeline);
    const bool lines = lineDraw;
    if (lines)
        cl.setLineWidth(LineWidth(device, targets.scale));
    if (UsesBlendConstant(U32(device + kDevBlendControl)))
    {
        float constants[4];
        for (uint32_t i = 0; i < 4; i++)
            constants[i] = std::bit_cast<float>(U32(device + kDevBlendConstant + i * 4));
        cl.setBlendConstants(constants);
        // The game's own shadow (the aircraft's shadows, a stencilled full-screen pass that leaves
        // constant colour + destination x (1 - constant alpha), (9, 11, 12, 76) / 255 in mission 01): what the ambient
        // occlusion's darkest is.
        if (ps->idHash == 0x3D40F4A3B370D071ull)
            std::memcpy(s_r.aoShadow, constants, sizeof(constants));
    }
    SeaReflectionAtDraw(device, *vs, *ps, targets, sc);
    if (ps->idHash == kSceneCompositePs && PlanarSea())
    {
        // (The adapted luminance: the composite's vertex shader's fetch slot 19, the band its c106.xy.)
        s_r.planarLuminance = sc.texture2DIndices[19] != kNullTexture2D ? sc.texture2DIndices[19] : 0;
        s_r.planarBand[0] = std::bit_cast<float>(U32(device + kDevVsConstants + 106 * 16));
        s_r.planarBand[1] = std::bit_cast<float>(U32(device + kDevVsConstants + 106 * 16 + 4));
    }
    NoteAoProjection(device, *vs, *ps, targets);
    std::memcpy(shared.memory + offsetof(SharedConstants, seaImage), &sc.seaImage, sizeof(sc.seaImage) + sizeof(sc.sea));
    BindCommon(cl);
    // The shared block of the draw before when nothing in it changed (most draws of a batch), so that with the float
    // constants unchanged too the constants' set isn't bound again: each bind is a driver call on the command
    // replay worker, the critical path at 720p.
    struct SharedUpload { uint64_t frame = UINT64_MAX, offset = 0, deviceAddress = 0; std::array<uint8_t, sizeof(SharedConstants)> bytes; };
    static SharedUpload lastShared;
    // (Compared on the CPU-side copy: the ring is write-combined memory, slow to read.)
    if (lastShared.frame != s_r.frame || std::memcmp(lastShared.bytes.data(), &sc, sizeof(SharedConstants)) != 0)
    {
        std::memcpy(lastShared.bytes.data(), &sc, sizeof(SharedConstants));
        lastShared.frame = s_r.frame;
        lastShared.offset = shared.offset;
        lastShared.deviceAddress = shared.deviceAddress;
        s_stats.constantBlocks[2]++;
    }
    else
        s_stats.constantSharedReused++;
    // The blocks' addresses for shaders that read them through buffer addresses (the HLSL backend, or
    // AC6_UNIFORM_CONSTANTS=0). The others read the uniform buffers only.
    if (!UniformConstants())
    {
        const uint64_t push[3] = { lastVs.deviceAddress, lastPs.deviceAddress, lastShared.deviceAddress };
        cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    }
    if (volLayout >= 0)
    {
        const uint64_t address = VolParticles()->getDeviceAddress();
        const uint32_t flags = (VolumetricsMode() == 2 ? 1u : 0u) | (uint32_t(volLayout) << 1) | (uint32_t(s_r.frame & 0xFFFFFF) << 8);
        const uint32_t push[5] = { uint32_t(address), uint32_t(address >> 32), volBase, flags, volEnd };
        cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    }
    else if (lateCaptureEnd)
    {
        const uint64_t address = s_r.lateCapture->getDeviceAddress();
        const uint32_t push[4] = { uint32_t(address), uint32_t(address >> 32), lateCaptureBase, lateCaptureEnd };
        cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    }
    const uint32_t constantOffsets[3] = { uint32_t(lastVs.offset), uint32_t(lastPs.offset), uint32_t(lastShared.offset) };
    if (UniformConstants() && (!s_bound.hasConstants || std::memcmp(s_bound.constantOffsets, constantOffsets, sizeof(constantOffsets)) != 0))
    {
        cl.setGraphicsDescriptorSetDynamic(s_r.constantSet.get(), 4, constantOffsets, 3);
        std::memcpy(s_bound.constantOffsets, constantOffsets, sizeof(constantOffsets));
        s_bound.hasConstants = true;
    }
    // Each stream binds at its own binding number (the pipeline's bindings are the stream numbers). One
    // call for all of them would bind 0..n-1 instead, leaving streams that don't start at 0 or have gaps
    // reading whatever an earlier draw bound there (the terrain's colours, alone in stream 3, would come out
    // a stale dark grey).
    for (uint32_t i = 0; i < viewCount; i++)
        cl.setVertexBuffers(layout.slots[i].index, &views[i], 1, &layout.slots[i]);
    // AC6_GPU_PASSES=2: each draw timed, by its shaders and target size.
    if (g_video.GpuDrawTiming())
        g_video.MarkGpuPass(std::format("draw vs {:016X} ps {:016X} {}x{}{}", vs->hash, ps->hash, targets.width, targets.height,
            targets.samples > 1 ? std::format(" x{}", targets.samples) : ""));
    if (!indices.empty())
    {
        RenderIndexBufferView ib(RenderBufferReference(s_r.upload.buffer.get(), indexOffset), uint32_t(indices.size() * 4), RenderFormat::R32_UINT);
        cl.setIndexBuffer(&ib);
        cl.drawIndexedInstanced(uint32_t(indices.size()), 1, 0, 0, 0);
    }
    else
    {
        cl.drawInstanced(drawCount, 1, 0, 0);
    }
    // The terrain's depth alone, for the ambient occlusion: the scene has nothing that says what a
    // pixel is (its alpha and its stencil are the game's), so the draws of the map and of the trees on it (sprites: creases
    // between them and the ground mean nothing) into the scene that write depth are drawn a second time, without a colour
    // target, into a depth image of their own. Where the scene's depth ends up equal to it the terrain is the surface
    // seen. Noted here and issued together when the occlusion runs (RunAmbientOcclusion): drawn in place they would split
    // the scene's pass into as many passes as there are such draws (89 in mission 01).
    // The hangar, a scene without terrain: the same for the aircraft's draws, those into the scene's depth after
    // the game put its kept surroundings back into it this frame (Renderer::aoAliasFrame). The occlusion is then made
    // from that depth alone and lands on the aircraft alone (RunAmbientOcclusion). AC6_AO_HANGAR=0: none there.
    static const bool hangarOcclusion = [] { const char* v = std::getenv("AC6_AO_HANGAR"); return !(v && *v == '0'); }();
    const bool terrainNote = AoTerrainMode() != 2 && ps->aoTerrain;
    const bool subjectNote = hangarOcclusion && !ps->aoTerrain && (s_r.aoMapFrame == ~0ull || s_r.aoMapFrame + 2 < s_r.frame) && s_r.aoAliasFrame == s_r.frame &&
        s_r.aoAliasDepth == targets.depth;
    if (AoMode() && (terrainNote || subjectNote) && targets.color && targets.depth && SceneSized(*targets.color) &&
        (U32(device + kDevDepthControl) & 0x4) &&
        (s_r.aoTerrainFrame != s_r.frame || (s_r.aoTerrainFor == targets.depth && s_r.aoSubject == subjectNote)))
    {
        HostImage& own = s_r.aoTerrainDepth;
        if (s_r.aoTerrainFrame != s_r.frame)
        {
            s_r.aoSubject = subjectNote;
            EnsureImage(own, targets.depth->width, targets.depth->height, targets.depth->format, true, targets.depth->samples, false, true);
            own.resX = targets.depth->resX;
            own.resY = targets.depth->resY;
            own.scale = targets.depth->scale;
            s_aoTerrainDraws.clear();
            s_r.aoTerrainFrame = s_r.frame;
            s_r.aoTerrainFor = targets.depth;
        }
        Targets depthTargets = targets;
        depthTargets.color = nullptr;
        depthTargets.motion = nullptr;
        depthTargets.depth = &own;
        t_noStencil = subjectNote;
        const PendingPipeline* depthPipeline = GetPipeline(device, *vs, *ps, layout, topology, specConstants & ~SPEC_CONSTANT_MOTION, depthTargets);
        t_noStencil = false;
        if (depthPipeline && s_aoTerrainDraws.size() < 4096)
        {
            AoTerrainDraw& note = s_aoTerrainDraws.emplace_back();
            note.pipeline = depthPipeline;
            note.viewport = viewport;
            note.width = targets.width;
            note.height = targets.height;
            std::memcpy(note.constantOffsets, constantOffsets, sizeof(constantOffsets));
            note.addresses[0] = lastVs.deviceAddress;
            note.addresses[1] = lastPs.deviceAddress;
            note.addresses[2] = lastShared.deviceAddress;
            note.layout = &layout;
            note.viewCount = viewCount;
            for (uint32_t i = 0; i < viewCount; i++)
                note.views[i] = views[i];
            note.indexOffset = indexOffset;
            note.indexCount = uint32_t(indices.size());
            note.drawCount = drawCount;
        }
    }
    // Light from the emitters: the draws the HDR glow takes below (additive, a plain-colour shader:
    // fire, explosions, flares, the afterburners), into the scene before its composite, noted to be issued again when
    // the scene is resolved (RunEmitterLight): into a float light image, single-sampled, with a depth of their own that
    // each writes whatever is there (the light's own distance, no stencil).
    if (EmitterLight() > 0.0f && s_r.lightApplyPs && targets.color && targets.depth && SceneSized(*targets.color) &&
        s_r.sceneCompositeFrame != s_r.frame && (U32(device + kDevColorMask) & 0x7) && ps->codeScale && !ps->litEncode && !ps->decoder)
    {
        const uint32_t control = U32(device + kDevBlendControl);
        const bool adds = control != kBlendDisabled && Blend((control >> 8) & 0x1F) == RenderBlend::ONE &&
            BlendOp((control >> 5) & 7) == RenderBlendOperation::ADD && Blend(control & 0x1F) != RenderBlend::ZERO;
        if (adds)
        {
            if (s_r.lightNotesFrame != s_r.frame)
            {
                s_r.lightNotesFrame = s_r.frame;
                s_lightDraws.clear();
                EnsureImage(s_r.lightColor, targets.color->width, targets.color->height, RenderFormat::R16G16B16A16_FLOAT, false, 1, false, true);
                EnsureImage(s_r.lightDepth, targets.color->width, targets.color->height, kDepthFormat, true, 1, false, true);
            }
            if (s_r.lightColor.width == targets.color->width && s_r.lightColor.height == targets.color->height && s_lightDraws.size() < 4096)
            {
                Targets lightTargets = targets;
                lightTargets.color = &s_r.lightColor;
                lightTargets.depth = &s_r.lightDepth;
                lightTargets.motion = nullptr;
                lightTargets.samples = 1;
                t_noStencil = true;
                t_depthAlwaysWrite = true;
                const PendingPipeline* lightPipeline = GetPipeline(device, *vs, *ps, layout, topology, specConstants & ~SPEC_CONSTANT_MOTION, lightTargets);
                t_noStencil = false;
                t_depthAlwaysWrite = false;
                if (lightPipeline)
                {
                    AoTerrainDraw& note = s_lightDraws.emplace_back();
                    note.pipeline = lightPipeline;
                    note.viewport = viewport;
                    note.width = targets.width;
                    note.height = targets.height;
                    std::memcpy(note.constantOffsets, constantOffsets, sizeof(constantOffsets));
                    note.addresses[0] = lastVs.deviceAddress;
                    note.addresses[1] = lastPs.deviceAddress;
                    note.addresses[2] = lastShared.deviceAddress;
                    note.layout = &layout;
                    note.viewCount = viewCount;
                    for (uint32_t i = 0; i < viewCount; i++)
                        note.views[i] = views[i];
                    note.indexOffset = indexOffset;
                    note.indexCount = uint32_t(indices.size());
                    note.drawCount = drawCount;
                    note.blendConstant = UsesBlendConstant(control);
                    for (uint32_t i = 0; i < 4; i++)
                        note.blend[i] = std::bit_cast<float>(U32(device + kDevBlendConstant + i * 4));
                }
            }
        }
    }
    // The planar reflection: the scene's draws with a perspective position matrix, noted to be issued again mirrored
    // in the sea's plane when the effects composite runs (RunPlanarReflection), into a colour and depth of their own at
    // a fraction of the scene's size. Each gets its own copy of the vertex constants, whose position matrix is
    // rewritten then (the frame's camera is the sea draws').
    // AC6_PLANAR_TRACE=<frame>: that frame's draws, where they go and whether the reflection takes them.
    {
        static const uint64_t traceFrame = [] { const char* v = std::getenv("AC6_PLANAR_TRACE"); return v ? std::stoull(v) : UINT64_MAX; }();
        if (s_r.frame == traceFrame)
        {
            const char* where = !targets.color ? "none" : targets.color == s_r.seaSceneSurface ? "scene" :
                targets.depth && targets.depth == s_r.planarSceneDepth ? "scenedepth" : "other";
            log::Info("PT {} | {} | {} {}x{} depth {} | pts {} rect {} topo {} | reg {} | taken {} | after-reflection {} after-composite {} | vs {:016X} ps {:016X}",
                ps->source, vs->source, where, targets.color ? targets.color->width : 0, targets.color ? targets.color->height : 0,
                U32(device + kDevDepthControl) & 0xFF, pointSprites, s_r.rectList, int(topology), int(vs->positionMatrix),
                s_r.planarNotesFrame == s_r.frame ? s_planarDraws.size() : 0, s_r.seaReflectionFrame == s_r.frame, s_r.sceneCompositeFrame == s_r.frame,
                vs->hash, ps->hash);
        }
    }
    // (Into the scene, or into the effects' buffer, the scene's size with a depth of its own: explosions, smoke, missile
    // trails, glows and cloud sprites, drawn apart from the scene and composited over it before the reflection is made.
    // Points and lines too: the city's lights are point sprites.)
    if (targets.color && targets.color == s_r.seaSceneSurface)
        s_r.planarSceneDepth = targets.depth;
    static const bool planarEffects = [] { const char* v = std::getenv("AC6_PLANAR_EFFECTS"); return !(v && *v == '0'); }();
    // AC6_PLANAR_SKIP=<vertex shader hash>[,...] (debugging): those draws left out of the reflection.
    static const std::string planarSkip = [] { const char* v = std::getenv("AC6_PLANAR_SKIP"); return std::string(v ? v : ""); }();
    const bool planarSkipped = !planarSkip.empty() && planarSkip.find(std::format("{:016X}", vs->hash)) != std::string::npos;
    if (PlanarSea() && s_r.planarBlurPs && targets.color && targets.depth && !t_uiDraw && !planarSkipped &&
        (targets.color == s_r.seaSceneSurface || (planarEffects && SceneSized(*targets.color))) &&
        volLayout < 0 && !lateCaptureEnd &&
        s_r.sceneCompositeFrame != s_r.frame && vs->positionMatrix != UINT32_MAX && vs->constantFirst <= vs->positionMatrix &&
        vs->positionMatrix + 4 <= vs->constantEnd && vs->constantEnd <= 256)
    {
        const uint32_t reg = vs->positionMatrix;
        double m[16];
        for (uint32_t i = 0; i < 16; i++)
            m[i] = double(std::bit_cast<float>(U32(device + kDevVsConstants + reg * 16 + i * 4)));
        if (std::fabs(m[3]) + std::fabs(m[7]) + std::fabs(m[11]) > 1e-6)
        {
            static const uint32_t divisor = [] { const char* v = std::getenv("AC6_PLANAR_RES"); return v && *v ? std::clamp(uint32_t(std::atoi(v)), 1u, 8u) : 2u; }();
            if (s_r.planarNotesFrame != s_r.frame)
            {
                s_r.planarNotesFrame = s_r.frame;
                s_planarDraws.clear();
                const uint32_t w = std::max(1u, targets.color->width / divisor), h = std::max(1u, targets.color->height / divisor);
                EnsureImage(s_r.planarColor, w, h, RenderFormat::R16G16B16A16_FLOAT, false, 1, false, true);
                EnsureImage(s_r.planarDepth, w, h, kDepthFormat, true, 1, false, true);
            }
            auto block = s_r.upload.Allocate(256 * 16, 256);
            if (block.memory && s_planarDraws.size() < 4096)
            {
                simd::ByteSwap32(reinterpret_cast<uint32_t*>(block.memory) + vs->constantFirst * 4,
                    g_memory.As<uint32_t>(device + kDevVsConstants) + vs->constantFirst * 4, (vs->constantEnd - vs->constantFirst) * 4);
                Targets mirrored = targets;
                mirrored.color = &s_r.planarColor;
                mirrored.depth = &s_r.planarDepth;
                mirrored.motion = nullptr;
                mirrored.width = s_r.planarColor.width;
                mirrored.height = s_r.planarColor.height;
                mirrored.scale = 1;
                mirrored.samples = 1;
                t_noStencil = true;
                t_mirror = true;
                const PendingPipeline* mirrorPipeline = GetPipeline(device, *vs, *ps, layout, topology,
                    (specConstants & ~(SPEC_CONSTANT_MOTION | SPEC_CONSTANT_REACTIVE | SPEC_CONSTANT_REACTIVE_LUMA)) | SPEC_CONSTANT_PLANAR_CLIP, mirrored);
                t_noStencil = false;
                t_mirror = false;
                if (mirrorPipeline)
                {
                    PlanarDraw& note = s_planarDraws.emplace_back();
                    AoTerrainDraw& d = note.draw;
                    d.pipeline = mirrorPipeline;
                    d.width = targets.width;
                    d.height = targets.height;
                    d.constantOffsets[0] = uint32_t(block.offset);
                    d.constantOffsets[1] = constantOffsets[1];
                    d.constantOffsets[2] = constantOffsets[2];
                    d.addresses[0] = block.deviceAddress;
                    d.addresses[1] = lastPs.deviceAddress;
                    d.addresses[2] = lastShared.deviceAddress;
                    // Its own shared block with SPEC_CONSTANT_PLANAR_CLIP (the vertex shader clips at the plane in
                    // constant 255, written when the pass runs).
                    if (auto shared2 = s_r.upload.Allocate(kConstantBlockBytes[2], 256); shared2.memory)
                    {
                        SharedConstants copy;
                        std::memcpy(&copy, lastShared.bytes.data(), sizeof(copy));
                        copy.drawFlags |= SPEC_CONSTANT_PLANAR_CLIP;
                        std::memcpy(shared2.memory, &copy, sizeof(copy));
                        d.constantOffsets[2] = uint32_t(shared2.offset);
                        d.addresses[2] = shared2.deviceAddress;
                    }
                    note.plane = reinterpret_cast<float*>(block.memory) + 255 * 4;
                    d.layout = &layout;
                    d.viewCount = viewCount;
                    for (uint32_t i = 0; i < viewCount; i++)
                        d.views[i] = views[i];
                    d.indexOffset = indexOffset;
                    d.indexCount = uint32_t(indices.size());
                    d.drawCount = drawCount;
                    const uint32_t control = U32(device + kDevBlendControl);
                    d.blendConstant = UsesBlendConstant(control);
                    for (uint32_t i = 0; i < 4; i++)
                        d.blend[i] = std::bit_cast<float>(U32(device + kDevBlendConstant + i * 4));
                    note.matrix = reinterpret_cast<float*>(block.memory) + reg * 4;
                    std::memcpy(note.m, m, sizeof(m));
                    auto cameraRegisters = [&](uint32_t r, uint32_t count, float*& place, double* value) {
                        if (r == UINT32_MAX || r < vs->constantFirst || r + count > vs->constantEnd)
                            return;
                        place = reinterpret_cast<float*>(block.memory) + r * 4;
                        for (uint32_t i = 0; i < count * 4; i++)
                            value[i] = double(std::bit_cast<float>(U32(device + kDevVsConstants + r * 16 + i * 4)));
                    };
                    cameraRegisters(vs->inverseViewProjectionMatrix, 4, note.inverseVp, note.inverseVpValue);
                    cameraRegisters(vs->inverseViewMatrix, 2, note.inverseView, note.inverseViewValue);
                    cameraRegisters(vs->viewMatrix, 4, note.view, note.viewValue);
                    cameraRegisters(vs->worldInverseViewMatrix, 2, note.worldInverseView, note.worldInverseViewValue);
                    note.viewport = steadyViewport;
                    const uint32_t depthControl = U32(device + kDevDepthControl);
                    note.depthWrite = (depthControl & 2) && (depthControl & 4);
                }
            }
        }
    }
    // The effects in float: every additive draw of a plain-colour shader (the
    // game's code scale, neither its lit encode nor a decoder: fire, explosions, flares, glows, the afterburners) into the
    // scene or the effects surface before the scene composite is light the 8-bit targets clip at white (and, summed over
    // the nozzle's encoded values, read as emitter sparkles). With HDR from the scene it is drawn again into a float layer
    // of the target's size and samples, against the target's depth. The 8-bit draw stays (the bloom and the exposure read
    // it, as on the console). Alpha-blended effects (smoke) are not light: left alone.
    const uint32_t blendControl = U32(device + kDevBlendControl);
    const bool additive = blendControl != kBlendDisabled && Blend((blendControl >> 8) & 0x1F) == RenderBlend::ONE &&
        BlendOp((blendControl >> 5) & 7) == RenderBlendOperation::ADD && Blend(blendControl & 0x1F) != RenderBlend::ZERO;
    auto drawLayer = [&](HostImage& layer, uint64_t& layerFrame, uint32_t blendOverride = 0, float blendConstant = -1.0f) {
        // (Exactly the target's size: the compose pass takes a layer of another size as one at another scale.)
        EnsureImage(layer, targets.color->width, targets.color->height, RenderFormat::R16G16B16A16_FLOAT, false, targets.color->samples, false,
            true);
        Targets glowTargets = targets;
        glowTargets.color = &layer;
        glowTargets.motion = nullptr;
        t_blendOverride = blendOverride;
        const PendingPipeline* glowPipeline = GetPipeline(device, *vs, *ps, layout, topology, specConstants & ~SPEC_CONSTANT_MOTION, glowTargets);
        t_blendOverride = 0;
        if (glowPipeline)
        {
            Transition(cl, layer, RenderTextureLayout::COLOR_WRITE);
            cl.setFramebuffer(GetFramebuffer(&layer, targets.depth));
            s_r.markedFramebuffer = nullptr;
            if (layerFrame != s_r.frame || layer.fresh)
            {
                cl.clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
                layerFrame = s_r.frame;
                layer.fresh = false;
            }
            SetPipeline(cl, glowPipeline);
            if (lines)
                cl.setLineWidth(LineWidth(device, glowTargets.scale));
            if (blendConstant >= 0.0f)
            {
                const float constants[4] = { blendConstant, blendConstant, blendConstant, blendConstant };
                cl.setBlendConstants(constants);
            }
            if (!indices.empty())
                cl.drawIndexedInstanced(uint32_t(indices.size()), 1, 0, 0, 0);
            else
                cl.drawInstanced(drawCount, 1, 0, 0);
            layer.singleCurrent = false;
            cl.setFramebuffer(targets.motion ? GetMotionFramebuffer(targets.color, targets.depth) : GetFramebuffer(targets.color, targets.depth));
        }
    };
    // (The world's surfaces with FSR upscaling too: their layer is at the render size, the compose pass samples it.)
    const bool beforeComposite = targets.color && s_r.hdrCanvasFrame != s_r.frame && SceneSized(*targets.color) && HdrSceneOn() &&
        s_r.hdrComposePipeline;
    if (beforeComposite && additive && ps->codeScale && !ps->litEncode && !ps->decoder && (U32(device + kDevColorMask) & 0x7) &&
        EffectsGlow() > 0.0f)
    {
        const bool ms = targets.color->samples > 1;
        // The afterburners: their own weight, the blend's source factor the constant (the game's ONE, ONE).
        constexpr uint64_t kAfterburnerVs = 0x7186B5F32D48D10Bull;
        if (vs->hash == kAfterburnerVs && (blendControl & 0x1F) == 1)
        {
            if (AfterburnerGlow() > 0.0f)
                drawLayer(ms ? s_r.hdrGlowMs : s_r.hdrGlow, ms ? s_r.hdrGlowMsFrame : s_r.hdrGlowFrame, (blendControl & ~0x1Fu) | 12u,
                    AfterburnerGlow());
        }
        else
            drawLayer(ms ? s_r.hdrGlowMs : s_r.hdrGlow, ms ? s_r.hdrGlowMsFrame : s_r.hdrGlowFrame);
    }
    // A cloud drawn after the effects' light hides it (the layers' light is added after the scene composite, over
    // everything, where in the game's own picture a cloud drawn later covers the effect under it). A cloud's draw (a
    // shader that reads a cloud constant, alpha blended) into a target whose layer holds light this frame is drawn into
    // the layer too with its colour taken out (ZERO, INV_SRC_ALPHA: the layer x (1 - alpha)), as the cloud layer's
    // below. Clouds only: an explosion's own smoke and fire sprites cover its light the same way in the game's
    // picture, but dimming the layer by every alpha-blended draw would take most of the explosions' and the
    // afterburners' HDR light with it.
    // The scene's own lights (the multisampled layer) lie under the whole effects pass, another surface: they are
    // added into the effects pass's layer before its first cloud (GlowUnderClouds, before the draw).
    if (GlowOcclusion() && beforeComposite && EffectsGlow() > 0.0f && (vs->cloud || ps->cloud) && (U32(device + kDevColorMask) & 0x7) &&
        Blend(blendControl & 0x1F) == RenderBlend::SRC_ALPHA && Blend((blendControl >> 8) & 0x1F) == RenderBlend::INV_SRC_ALPHA &&
        BlendOp((blendControl >> 5) & 7) == RenderBlendOperation::ADD)
    {
        const bool ms = targets.color->samples > 1;
        HostImage& layer = ms ? s_r.hdrGlowMs : s_r.hdrGlow;
        if ((ms ? s_r.hdrGlowMsFrame : s_r.hdrGlowFrame) == s_r.frame && layer.width == targets.color->width && layer.height == targets.color->height)
            drawLayer(layer, ms ? s_r.hdrGlowMsFrame : s_r.hdrGlowFrame, 0x01000700u);
    }
    // The clouds: the cloud particles (pixel shader 5776EB8756E408E9, `saturate(lit + CloudFogColor)`, alpha blended
    // into the effects surface) are the cumulus banks. The sky panorama
    // E425A78B5955E8F7 holds only the sky's gradient. The particles are drawn again into a float layer with their own
    // blending over nothing: their contribution, which the compose pass lifts where it is bright (sunlit cloud, not
    // shade or sky).
    constexpr uint64_t kCloudPs = 0x5776EB8756E408E9ull;
    if (beforeComposite && ps->idHash == kCloudPs && CloudLift() > 1.0f)
    {
        const bool ms = targets.color->samples > 1;
        drawLayer(ms ? s_r.hdrSkyMs : s_r.hdrSky, ms ? s_r.hdrSkyMsFrame : s_r.hdrSkyFrame);
    }
    // What covers them after (smoke, other alpha-blended effects over the clouds already drawn, into the same surface):
    // drawn into the cloud layer too with their colour taken out (ZERO, INV_SRC_ALPHA: the layer x (1 - alpha)), so the
    // lift stays off what covers a cloud (a black smoke puff in front of a cloud bank would wash out).
    else if (beforeComposite && CloudLift() > 1.0f && (U32(device + kDevColorMask) & 0x7) && Blend(blendControl & 0x1F) == RenderBlend::SRC_ALPHA &&
        Blend((blendControl >> 8) & 0x1F) == RenderBlend::INV_SRC_ALPHA && BlendOp((blendControl >> 5) & 7) == RenderBlendOperation::ADD)
    {
        const bool ms = targets.color->samples > 1;
        if ((ms ? s_r.hdrSkyMsFrame : s_r.hdrSkyFrame) == s_r.frame)
            drawLayer(ms ? s_r.hdrSkyMs : s_r.hdrSky, ms ? s_r.hdrSkyMsFrame : s_r.hdrSkyFrame, 0x01000700u);
    }
    // The enhanced bloom, added to the canvas right after the composite drew it without its own (this block). Then the
    // canvas is composed (the next block): its range above white compressed by the brightest channel (the lit
    // highlights: AC6_HDR_LIT_HIGHLIGHTS), then the effects' layers added x 255/240 (their codes as values) and rolled
    // off towards a ceiling of 2 x AC6_HDR_EFFECTS, before the chromatic aberration and the grade, so they take the
    // grade's colour as the 8-bit scene does. The output pass's stray-emitter cap still removes the 8-bit sums'
    // sparkles under them.
    if (ps->idHash == kSceneCompositePs && s_r.bloomScalesFrame == s_r.frame && targets.color && targets.color->samples == 1)
    {
        RunEnhancedBloom(cl, device, targets, sc);
        Transition(cl, *targets.color, RenderTextureLayout::COLOR_WRITE);
        if (targets.depth)
            Transition(cl, *targets.depth, RenderTextureLayout::DEPTH_WRITE);
        cl.setFramebuffer(targets.motion ? GetMotionFramebuffer(targets.color, targets.depth) : GetFramebuffer(targets.color, targets.depth));
        s_r.markedFramebuffer = nullptr;
    }
    if (ps->idHash == kSceneCompositePs && HdrSceneOn() && s_r.hdrComposePipeline && targets.color &&
        targets.color->format == RenderFormat::R16G16B16A16_FLOAT && targets.color->samples == 1)
    {
        const bool glow = s_r.hdrGlowFrame == s_r.frame && EffectsGlow() > 0.0f, glowMs = s_r.hdrGlowMsFrame == s_r.frame && EffectsGlow() > 0.0f;
        const bool skyMs = s_r.hdrSkyMsFrame == s_r.frame && CloudLift() > 1.0f;
        const bool sky = !skyMs && s_r.hdrSkyFrame == s_r.frame && CloudLift() > 1.0f;
        if (glow || glowMs || sky || skyMs || LitHighlights() < 1.0f || SceneCeiling() > 0.0f)
        {
            HostImage& canvas = *targets.color;
            cl.setFramebuffer(nullptr);
            s_r.markedFramebuffer = nullptr;
            // The pass reads the canvas and writes every pixel of it, so it draws into a second image of the canvas's
            // kind and the two images change places after (as the resolve swaps do: the canvas's image, view and
            // descriptor are the second image's from then, and the second image keeps the frame as it was before the
            // pass). AC6_HDR_COMPOSE_COPY=1: the pass reads a full-frame copy of the canvas instead.
            static const bool composeCopy = [] { const char* v = std::getenv("AC6_HDR_COMPOSE_COPY"); return v && *v == '1'; }();
            HostImage& second = s_r.hdrComposeSource;
            EnsureImage(second, canvas.width, canvas.height, canvas.format, false);
            const bool composeSwap = !composeCopy && !canvas.fresh && !canvas.mipChain && !canvas.swizzleTrace;
            if (composeSwap)
            {
                Transition(cl, canvas, RenderTextureLayout::SHADER_READ);
                Transition(cl, second, RenderTextureLayout::COLOR_WRITE);
            }
            else
            {
                Transition(cl, canvas, RenderTextureLayout::COPY_SOURCE);
                Transition(cl, second, RenderTextureLayout::COPY_DEST);
                cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(second.texture.get()),
                    RenderTextureCopyLocation::Subresource(canvas.texture.get()));
                Transition(cl, second, RenderTextureLayout::SHADER_READ);
            }
            HostImage& composeFrom = composeSwap ? canvas : second;
            HostImage& composeTo = composeSwap ? second : canvas;
            // The multisampled layers are read as drawn, their samples averaged by the pass. A layer of another size
            // (FSR upscaling) is resolved first, and the pass samples the resolved copy. AC6_HDR_LAYER_RESOLVE=1: every
            // layer resolved first.
            static const bool layerResolve = [] { const char* v = std::getenv("AC6_HDR_LAYER_RESOLVE"); return v && *v == '1'; }();
            uint32_t samplesRead = 0;   // bit 0: the glow's multisampled layer, bit 1: the clouds'
            auto layerDescriptor = [&](HostImage& layer, uint32_t bit) {
                if (layer.samples > 1 && !layerResolve && !layer.fresh && layer.width == canvas.width && layer.height == canvas.height)
                {
                    const uint32_t descriptor = SamplesDescriptor(layer);
                    Transition(cl, layer, RenderTextureLayout::SHADER_READ);
                    samplesRead |= bit;
                    return descriptor;
                }
                HostImage& g = Readable(cl, layer);
                Transition(cl, g, RenderTextureLayout::SHADER_READ);
                return g.descriptor;
            };
            uint32_t layers = 0;
            if (glow)
                layers |= layerDescriptor(s_r.hdrGlow, 0) & 0xFFFF;
            if (glowMs)
                layers |= (layerDescriptor(s_r.hdrGlowMs, 1) & 0xFFFF) << 16;
            uint32_t skyLayer = 0;
            if (sky || skyMs)
                skyLayer = layerDescriptor(skyMs ? s_r.hdrSkyMs : s_r.hdrSky, 2);
            Transition(cl, composeTo, RenderTextureLayout::COLOR_WRITE);
            if (g_video.GpuPassTiming())
                g_video.MarkGpuPass("hdr compose");
            cl.setFramebuffer(GetFramebuffer(&composeTo, nullptr));
            SetViewport(cl, RenderViewport(0.0f, 0.0f, float(canvas.width), float(canvas.height)));
            SetScissor(cl, RenderRect(0, 0, int32_t(canvas.width), int32_t(canvas.height)));
            SetPipeline(cl, s_r.hdrComposePipeline.get());
            BindCommon(cl);
            // The effects' ceiling: 2 x the setting in the canvas's units (2.2-encoded: base white + 2 is about 11x white).
            const float ceiling = 2.0f * EffectsGlow();
            uint32_t push[7] = { composeFrom.descriptor, layers, std::bit_cast<uint32_t>(ceiling), std::bit_cast<uint32_t>(LitHighlights()),
                skyLayer, (std::bit_cast<uint32_t>(CloudLift()) & ~3u) | samplesRead, std::bit_cast<uint32_t>(SceneCeiling()) };
            cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
            cl.drawInstanced(3, 1, 0, 0);
            // The sun's disc softened: the same pass again over its box and the spread's reach (3 sigma, sigma 8
            // pixels at 1440 lines), the ceiling negated (the shader's spread, too costly to run over the whole frame).
            if (SceneCeiling() > 0.0f && s_r.sunFrame == s_r.frame)
            {
                const float reach = 3.0f * std::max(1.0f, float(canvas.height) / 1440.0f * 8.0f) + 2.0f;
                const float cw = float(canvas.width), ch = float(canvas.height);
                const int32_t x0 = std::clamp(int32_t(std::floor(s_r.sunBox[0] * cw - reach)), 0, int32_t(canvas.width));
                const int32_t y0 = std::clamp(int32_t(std::floor(s_r.sunBox[1] * ch - reach)), 0, int32_t(canvas.height));
                const int32_t x1 = std::clamp(int32_t(std::ceil(s_r.sunBox[2] * cw + reach)), 0, int32_t(canvas.width));
                const int32_t y1 = std::clamp(int32_t(std::ceil(s_r.sunBox[3] * ch + reach)), 0, int32_t(canvas.height));
                if (x1 > x0 && y1 > y0)
                {
                    SetScissor(cl, RenderRect(x0, y0, x1, y1));
                    // (Bit 31 the spread, bits 0-15 the frame's ceiling x 256, 16-30 the sun's x 128.)
                    const float sunTop = std::max(SunCeiling(), SceneCeiling());
                    push[6] = 0x80000000u | uint32_t(std::clamp(SceneCeiling() * 256.0f, 0.0f, 65535.0f)) |
                        uint32_t(std::clamp(sunTop * 128.0f, 0.0f, 32767.0f)) << 16;
                    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
                    cl.drawInstanced(3, 1, 0, 0);
                }
            }
            if (composeSwap)
            {
                cl.setFramebuffer(nullptr);
                second.fresh = false;
                std::swap(canvas.texture, second.texture);
                std::swap(canvas.view, second.view);
                std::swap(canvas.descriptor, second.descriptor);
                std::swap(canvas.layout, second.layout);
                std::swap(canvas.swizzled, second.swizzled);
            }
        }
    }
    if (overwritesColor && targets.color)
        UnlinkStale(*targets.color);
    if (volComposite)
    {
        RunVolumetrics(cl, targets);
        s_r.volComposited = true;
    }
    // The sea's reflection of the scene, for the next frame's sea: from the scene as the effects
    // composite leaves it (with the clouds, the smoke and the explosions).
    if (ps->idHash == kEffectsCompositePs && WaterSsr() && targets.color && s_r.seaSceneSurface)
    {
        if (PlanarSea())
            RunPlanarReflection(cl, *targets.color, s_r.seaSceneSurface);
        else
            RunSeaReflection(cl, *targets.color, s_r.seaSceneSurface);
        // (The composite's target bound again, as the draws after it expect.)
        Transition(cl, *targets.color, RenderTextureLayout::COLOR_WRITE);
        if (targets.depth)
            Transition(cl, *targets.depth, RenderTextureLayout::DEPTH_WRITE);
        cl.setFramebuffer(GetFramebuffer(targets.color, targets.depth));
        s_r.markedFramebuffer = nullptr;
    }
}

} // namespace

void DrawImpl(uint32_t device, uint32_t primType, uint32_t startVertex, uint32_t vertexCount, HostShader* vs, HostShader* ps)
{
    thread_local IndexList none;
    none.clear();
    Draw(device, primType, startVertex, vertexCount, none, vs, ps);
}

// The cockpit view's exterior body (game/cockpit.cpp): the cockpit groups' draws leave their
// model-space boxes (the vertex positions, 32-bit float xyz at offset 0, before any transform). The body's draws keep
// only the triangles that aren't wholly inside one of them, filtered once per index list and box set. Returns true when
// the draw was handled (drawn filtered, or dropped).
namespace {

thread_local int t_cockpitGroup = -1;   // the queued draw's (RunDraw)
struct CockpitBox { float lo[3], hi[3]; uint64_t frame; };
std::map<std::tuple<uint32_t, uint32_t, uint32_t>, CockpitBox> s_cockpitBoxes;   // by vertex buffer and range
std::map<std::tuple<uint32_t, uint32_t, uint32_t, size_t, uint64_t>, IndexList> s_cockpitFiltered;

bool CockpitDraw(uint32_t device, int group, uint32_t primType, int32_t baseVertex, IndexList& indices, uint32_t lo, uint32_t hi, HostShader* vs,
    HostShader* ps)
{
    const VertexLayout* layout = CachedVertexLayout(*vs);
    const RenderInputElement* position = nullptr;
    if (layout)
        for (const RenderInputElement& e : layout->elements)
            if (e.format == RenderFormat::R32G32B32_FLOAT && e.alignedByteOffset == 0)
            {
                position = &e;
                break;
            }
    if (!position)
    {
        static bool logged = false;
        if (!logged && layout)
        {
            std::string list;
            for (const RenderInputElement& e : layout->elements)
                list += std::format(" loc {} format {} slot {} offset {};", e.location, int(e.format), e.slotIndex, e.alignedByteOffset);
            log::Info("Cockpit view: no float3 position in vs {:016X}:{}", vs->hash, list);
        }
        logged = true;
        return group == 0;   // a body draw that can't be filtered isn't drawn
    }
    const uint32_t stream = position->slotIndex, stride = layout->strides[stream];
    const xenos::VertexFetch vf{ { U32(device + kDevVertexFetch + (17 - stream) * 8), U32(device + kDevVertexFetch + (17 - stream) * 8 + 4) } };
    if (!vf.Address() || stride < 12 || (primType != xenos::kPrimTriangleList && primType != xenos::kPrimTriangleStrip))
        return group == 0;
    const uint32_t vb = PhysicalToGuest(vf.Address());
    auto point = [&](uint32_t index, float p[3]) {
        const uint32_t at = vb + uint32_t(int32_t(index) + baseVertex) * stride;
        for (int k = 0; k < 3; k++)
            p[k] = std::bit_cast<float>(U32(at + k * 4));
    };
    if (group != 0)
    {
        auto [entry, fresh] = s_cockpitBoxes.try_emplace({ vb, lo, hi });
        entry->second.frame = s_r.frame;
        if (fresh)
        {
            CockpitBox& box = entry->second;
            std::fill(std::begin(box.lo), std::end(box.lo), 1e30f);
            std::fill(std::begin(box.hi), std::end(box.hi), -1e30f);
            for (uint32_t i : indices)
                if (i != 0xFFFFFFFFu)
                {
                    float p[3];
                    point(i, p);
                    for (int k = 0; k < 3; k++)
                    {
                        box.lo[k] = std::min(box.lo[k], p[k] - 0.05f);
                        box.hi[k] = std::max(box.hi[k], p[k] + 0.05f);
                    }
                }
        }
        return false;
    }
    // The boxes of the cockpit drawn lately (this aircraft's: a cockpit is drawn every frame of the view).
    std::vector<const CockpitBox*> boxes;
    uint64_t signature = 0xCBF29CE484222325ull;
    for (const auto& [where, box] : s_cockpitBoxes)
        if (s_r.frame - box.frame <= 3)
        {
            boxes.push_back(&box);
            signature = (signature ^ std::get<0>(where) ^ (uint64_t(std::get<1>(where)) << 32)) * 0x100000001B3ull;
        }
    if (boxes.empty())
        return true;
    if (s_cockpitFiltered.size() > 64)
        s_cockpitFiltered.clear();
    const auto key = std::make_tuple(vb, primType, uint32_t(baseVertex), indices.size(), signature);
    auto [it, fresh] = s_cockpitFiltered.try_emplace(key);
    IndexList& kept = it->second;
    if (fresh)
    {
        auto inside = [&](const float p[3], const CockpitBox& b) {
            return p[0] >= b.lo[0] && p[0] <= b.hi[0] && p[1] >= b.lo[1] && p[1] <= b.hi[1] && p[2] >= b.lo[2] && p[2] <= b.hi[2];
        };
        auto triangle = [&](uint32_t a, uint32_t b, uint32_t c) {
            if (a == b || b == c || a == c)
                return;
            float pa[3], pb[3], pc[3];
            point(a, pa);
            point(b, pb);
            point(c, pc);
            for (const CockpitBox* box : boxes)
                if (inside(pa, *box) && inside(pb, *box) && inside(pc, *box))
                    return;
            kept.insert(kept.end(), { a, b, c });
        };
        if (primType == xenos::kPrimTriangleList)
        {
            for (size_t i = 0; i + 2 < indices.size(); i += 3)
                triangle(indices[i], indices[i + 1], indices[i + 2]);
        }
        else
        {
            // Strips with restarts. Every other triangle's first two corners swap (the winding kept).
            size_t start = 0;
            for (size_t i = 0; i <= indices.size(); i++)
                if (i == indices.size() || indices[i] == 0xFFFFFFFFu)
                {
                    for (size_t k = start; k + 2 < i; k++)
                        if ((k - start) & 1)
                            triangle(indices[k + 1], indices[k], indices[k + 2]);
                        else
                            triangle(indices[k], indices[k + 1], indices[k + 2]);
                    start = i + 1;
                }
        }
        static uint32_t logged = 0;
        if (logged++ < 8)
            log::Info("Cockpit view: the exterior body keeps {} triangle(s) of a draw of {} indices (outside the cockpit's {} box(es))",
                kept.size() / 3, indices.size(), boxes.size());
    }
    if (kept.empty())
        return true;
    thread_local IndexList list;
    list.assign(kept.begin(), kept.end());
    Draw(device, xenos::kPrimTriangleList, uint32_t(int32_t(lo) + baseVertex), hi - lo + 1, list, vs, ps, lo);
    return true;
}

} // namespace

void DrawIndexedImpl(uint32_t device, uint32_t primType, int32_t baseVertex, uint32_t startIndex, uint32_t indexCount, HostShader* vs,
    HostShader* ps)
{
    if (!s_r.ready || indexCount == 0)
        return;
    const uint32_t ib = U32(device + kDevIndexBuffer);
    static uint32_t noBuffer = 0, allRestart = 0;
    if (!ib)
    {
        if (noBuffer++ < 4)
            log::Warn("Indexed draw without an index buffer ({} indices): skipped", indexCount);
        return;
    }
    // The GPU reads 32-bit words, applies the buffer's endian swap (common bits 29-30, as
    // DrawIndexedVertices programs VGT_DMA_SIZE) and takes 16-bit indices low half first.
    // 0xFFFF / 0xFFFFFFFF restart strips.
    const uint32_t common = U32(ib);
    const bool wide = (common & 0x80000000) != 0;
    const uint32_t endian = (common >> 29) & 3;
    static std::set<uint32_t> seenCommon;
    if (seenCommon.size() < 16 && seenCommon.insert(common).second)
        log::Info("Index buffer header {:08X}: {}-bit, endian {}", common, wide ? 32 : 16, endian);
    auto swap = [endian](uint32_t w) {
        switch (endian)
        {
        case 1: return ((w & 0x00FF00FFu) << 8) | ((w >> 8) & 0x00FF00FFu);   // 8in16
        case 2: return __builtin_bswap32(w);                                   // 8in32
        case 3: return std::rotl(w, 16);                                       // 16in32
        default: return w;
        }
    };
    const uint32_t restart = wide ? 0xFFFFFFu : 0xFFFFu;
    const uint32_t base = PhysicalToGuest(U32(ib + kIndexBufferAddress)) + startIndex * (wide ? 4 : 2);
    thread_local IndexList indices;
    indices.resize(indexCount);
    uint32_t lo = UINT32_MAX, hi = 0;
    uint32_t i = 0;
#if defined(__AVX2__)
    // The game's two layouts, 16 or 8 indices at a time: 16-bit indices swapped per
    // index (endian 1: each index big-endian where it lies) and 32-bit ones swapped per word (endian 2).
    if ((!wide && endian == 1) || (wide && endian == 2))
    {
        const uint8_t* src = g_memory.As<uint8_t>(base);
        uint32_t* out = indices.data();
        __m256i vmin = _mm256_set1_epi32(-1), vmax = _mm256_setzero_si256();
        const __m256i all = _mm256_set1_epi32(-1);
        if (!wide)
        {
            const __m256i swap16 = _mm256_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
                1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
            const __m256i restartValue = _mm256_set1_epi32(0xFFFF);
            for (; i + 16 <= indexCount; i += 16)
            {
                const __m256i raw = _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i * 2)), swap16);
                for (int half = 0; half < 2; half++)
                {
                    const __m256i v = _mm256_cvtepu16_epi32(half ? _mm256_extracti128_si256(raw, 1) : _mm256_castsi256_si128(raw));
                    const __m256i isRestart = _mm256_cmpeq_epi32(v, restartValue);
                    const __m256i value = _mm256_blendv_epi8(v, all, isRestart);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + half * 8), value);
                    vmin = _mm256_min_epu32(vmin, value);
                    vmax = _mm256_max_epu32(vmax, _mm256_andnot_si256(isRestart, v));
                }
            }
        }
        else
        {
            const __m256i swap32 = _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
                3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
            const __m256i mask24 = _mm256_set1_epi32(0xFFFFFF);
            for (; i + 8 <= indexCount; i += 8)
            {
                const __m256i v = _mm256_and_si256(
                    _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i * 4)), swap32), mask24);
                const __m256i isRestart = _mm256_cmpeq_epi32(v, mask24);
                const __m256i value = _mm256_blendv_epi8(v, all, isRestart);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), value);
                vmin = _mm256_min_epu32(vmin, value);
                vmax = _mm256_max_epu32(vmax, _mm256_andnot_si256(isRestart, v));
            }
        }
        alignas(32) uint32_t mins[8], maxs[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(mins), vmin);
        _mm256_store_si256(reinterpret_cast<__m256i*>(maxs), vmax);
        // Lanes that saw only restarts keep min 0xFFFFFFFF and max 0, as the scalar loop leaves lo/hi.
        for (int k = 0; k < 8; k++)
        {
            if (mins[k] != 0xFFFFFFFFu)
                lo = std::min(lo, mins[k]);
            hi = std::max(hi, maxs[k]);
        }
    }
#endif
    for (; i < indexCount; i++)
    {
        uint32_t v;
        if (wide)
        {
            // The GPU uses 24 bits of an index: 0xFFFFFFFF arrives as the 0xFFFFFF restart value.
            std::memcpy(&v, g_memory.Translate(base + i * 4), 4);
            v = swap(v) & 0xFFFFFF;
        }
        else
        {
            const uint32_t at = base + i * 2;
            uint32_t word;
            std::memcpy(&word, g_memory.Translate(at & ~3u), 4);
            word = swap(word);
            v = (at & 2) ? word >> 16 : word & 0xFFFF;
        }
        if (v == restart)
        {
            indices[i] = 0xFFFFFFFFu;
            continue;
        }
        indices[i] = v;
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    if (lo > hi)
    {
        if (allRestart++ < 4)
            log::Warn("Indexed draw of {} indices at {:08X} has only restart indices: skipped", indexCount, base);
        return;
    }
    if (hi - lo >= (1u << 22))
    {
        static uint32_t warned = 0;
        if (warned++ < 4)
            log::Warn("Indexed draw spans {} vertices: skipped", hi - lo + 1);
        return;
    }
    {
        static const uint64_t traceFrame = [] { const char* v = std::getenv("AC6_DRAW_TRACE"); return v ? std::stoull(v) : UINT64_MAX; }();
        if (s_r.frame == traceFrame)
        {
            std::string first;
            for (uint32_t i = 0; i < std::min<uint32_t>(indexCount, 12); i++)
                first += indices[i] == 0xFFFFFFFFu ? std::string(" R") : std::format(" {}", indices[i]);
            log::Info("Indexed draw: {} {}-bit indices from {:08X} (start {}), base vertex {}, range {}-{}, raw:{}", indexCount, wide ? 32 : 16,
                base, startIndex, baseVertex, lo, hi, first);
        }
    }
    // The cockpit view's exterior body.
    if (const int group = t_cockpitGroup; group >= 0 && vs && ps &&
        CockpitDraw(device, group, primType, baseVertex, indices, lo, hi, vs, ps))
        return;
    // Relative to the lowest vertex: taken off as the indices are uploaded.
    Draw(device, primType, uint32_t(int32_t(lo) + baseVertex), hi - lo + 1, indices, vs, ps, lo);
}

// SMAA 1x on a resolved colour image: edges (RG8, cleared: no edge is discarded), blending weights
// with SMAA's precomputed area and search textures (RGBA8, cleared), then the neighbourhood blend into a copy that goes back
// into the image. The lookup textures are uploaded the first time.
struct SmaaTargets { HostImage edges, weights, output; };
SmaaTargets s_smaa;

// (The scene image is B8G8R8A8, or with HDR from the scene R16G16B16A16_FLOAT: the same passes, the blend into
// a float copy.)
void RunSmaa(RenderCommandList& cl, HostImage& image)
{
    const bool floatImage = image.format == RenderFormat::R16G16B16A16_FLOAT && s_r.smaaBlendPipelineFloat;
    if (!SmaaOn() || !image.texture || (image.format != RenderFormat::B8G8R8A8_UNORM && !floatImage))
        return;
    if (!s_r.smaaArea)
    {
        auto upload = [&](std::unique_ptr<RenderTexture>& texture, std::unique_ptr<RenderTextureView>& view, uint32_t& descriptor,
                          RenderFormat format, uint32_t width, uint32_t height, uint32_t bytesPerTexel, const unsigned char* bytes) {
            texture = MakeTexture(RenderTextureDesc::Texture2D(width, height, 1, format), "an SMAA lookup texture");
            view = texture->createTextureView(RenderTextureViewDesc::Texture2D(format));
            descriptor = AllocateDescriptor();
            SetDescriptor(descriptor, texture.get(), view.get());
            const uint64_t size = uint64_t(width) * height * bytesPerTexel;
            auto a = s_r.upload.Allocate(size, 512);
            if (!a.memory)
                return false;
            std::memcpy(a.memory, bytes, size);
            cl.barriers(RenderBarrierStage::COPY, RenderTextureBarrier(texture.get(), RenderTextureLayout::COPY_DEST));
            cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(texture.get()),
                RenderTextureCopyLocation::PlacedFootprint(s_r.upload.buffer.get(), format, width, height, 1, width, a.offset));
            cl.barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(texture.get(), RenderTextureLayout::SHADER_READ));
            return true;
        };
        if (!upload(s_r.smaaArea, s_r.smaaAreaView, s_r.smaaAreaDescriptor, RenderFormat::R8G8_UNORM, AREATEX_WIDTH, AREATEX_HEIGHT, 2, areaTexBytes) ||
            !upload(s_r.smaaSearch, s_r.smaaSearchView, s_r.smaaSearchDescriptor, RenderFormat::R8_UNORM, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, 1, searchTexBytes))
        {
            s_r.smaaArea.reset();   // the upload ring was full: next time
            return;
        }
    }
    const uint32_t w = image.width, h = image.height;
    EnsureImage(s_smaa.edges, w, h, RenderFormat::R8G8_UNORM, false);
    EnsureImage(s_smaa.weights, w, h, RenderFormat::R8G8B8A8_UNORM, false);
    EnsureImage(s_smaa.output, w, h, image.format, false);
    MarkPass([] { return std::string("SMAA"); });
    auto pass = [&](HostImage& target, RenderPipeline* pipeline, bool clear, uint32_t texture, uint32_t extra) {
        Transition(cl, target, RenderTextureLayout::COLOR_WRITE);
        cl.setFramebuffer(GetFramebuffer(&target, nullptr));
        s_r.markedFramebuffer = nullptr;
        if (clear)
            cl.clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
        SetViewport(cl, RenderViewport(0.0f, 0.0f, float(w), float(h)));
        SetScissor(cl, RenderRect(0, 0, int32_t(w), int32_t(h)));
        SetPipeline(cl, pipeline);
        BindCommon(cl);
        const uint32_t push[4] = { texture, extra, w, h };
        cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
        cl.drawInstanced(3, 1, 0, 0);
    };
    Transition(cl, image, RenderTextureLayout::SHADER_READ);
    pass(s_smaa.edges, s_r.smaaEdgePipeline.get(), true, image.descriptor, 0);
    Transition(cl, s_smaa.edges, RenderTextureLayout::SHADER_READ);
    pass(s_smaa.weights, s_r.smaaWeightPipeline.get(), true, s_smaa.edges.descriptor, s_r.smaaAreaDescriptor | (s_r.smaaSearchDescriptor << 16));
    Transition(cl, s_smaa.weights, RenderTextureLayout::SHADER_READ);
    pass(s_smaa.output, floatImage ? s_r.smaaBlendPipelineFloat.get() : s_r.smaaBlendPipeline.get(), false, image.descriptor,
        s_smaa.weights.descriptor);
    cl.setFramebuffer(nullptr);
    Transition(cl, s_smaa.output, RenderTextureLayout::COPY_SOURCE);
    Transition(cl, image, RenderTextureLayout::COPY_DEST);
    cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(image.texture.get()), RenderTextureCopyLocation::Subresource(s_smaa.output.texture.get()));
}

// The camera's reprojection for TAA and FSR: this frame's world view-projection, the c116-c119 matrix most scene draws
// use (rows: clip = x c116 + y c117 + z c118 + c119), into `vp`, and R = inverse(it) x last frame's into `r` (this
// frame's clip position to last frame's). False when the frame had none.
bool SceneReprojection(double vp[16], double r[16])
{
    const bool worldCamera = s_r.taaVpFrame == s_r.frame && !s_r.taaVpCandidates.empty();
    // (No world camera this frame: the position matrix most of the scene's draws use, the hangar's building.)
    const bool objectCamera = !worldCamera && s_r.taaObjectFrame == s_r.frame && !s_r.taaObjectCandidates.empty();
    if (!worldCamera && !objectCamera)
        return false;
    const auto& candidates = worldCamera ? s_r.taaVpCandidates : s_r.taaObjectCandidates;
    if (objectCamera)
    {
        static bool logged = false;
        if (!std::exchange(logged, true))
            log::Info("TAA / FSR: no world camera on frame {} (the hangar): the position matrix of the most draws is the camera", s_r.frame);
    }
    const auto world = std::max_element(candidates.begin(), candidates.end(),
        [](const Renderer::TaaVpCandidate& a, const Renderer::TaaVpCandidate& b) { return a.draws < b.draws; });
    std::memcpy(vp, world->m, 16 * sizeof(double));
    double inverse[16];
    if (!Invert4(vp, inverse))
        return false;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < 4; k++)
                sum += inverse[i * 4 + k] * s_r.taaPrevVp[k * 4 + j];
            r[i * 4 + j] = sum;
        }
    return true;
}

// A camera cut between last frame's view-projection and this one (`vp`, SceneReprojection's): the camera moved more than
// an aircraft does in a frame (400 m, the clouds' rule) or turned by more than 40 degrees. TAA and FSR then start
// without history, so the last shot doesn't ghost into the new one. Asked once a frame. AC6_CUT_RESET=0: off.
bool CameraCut(const double vp[16])
{
    static const bool on = [] { const char* v = std::getenv("AC6_CUT_RESET"); return !v || *v != '0'; }();
    static uint64_t askedFrame = ~0ull;
    static bool cut = false;
    if (!on || !s_r.taaHavePrev)
        return false;
    if (askedFrame == s_r.frame)
        return cut;
    askedFrame = s_r.frame;
    cut = false;
    // The camera (clip 0, 0, 1, 0: the eye with the game's reversed, infinite projection) and the direction to the point
    // halfway into the depth at the picture's centre, from each frame's inverse.
    auto view = [](const double m[16], double eye[3], double ahead[3]) {
        double inv[16];
        if (!Invert4(m, inv) || inv[11] == 0.0)
            return false;
        double p[4];
        for (int c = 0; c < 4; c++)
            p[c] = 0.5 * inv[8 + c] + inv[12 + c];
        if (p[3] == 0.0)
            return false;
        double length = 0.0;
        for (int c = 0; c < 3; c++)
        {
            eye[c] = inv[8 + c] / inv[11];
            ahead[c] = p[c] / p[3] - eye[c];
            length += ahead[c] * ahead[c];
        }
        length = std::sqrt(length);
        if (length <= 0.0)
            return false;
        for (int c = 0; c < 3; c++)
            ahead[c] /= length;
        return true;
    };
    double eye[3], ahead[3], lastEye[3], lastAhead[3];
    if (!view(vp, eye, ahead) || !view(s_r.taaPrevVp, lastEye, lastAhead))
        return false;
    double moved = 0.0, cosine = 0.0;
    for (int c = 0; c < 3; c++)
    {
        moved += (eye[c] - lastEye[c]) * (eye[c] - lastEye[c]);
        cosine += ahead[c] * lastAhead[c];
    }
    cut = moved > 400.0 * 400.0 || cosine < 0.766;   // cos 40 degrees
    static uint32_t logged = 0;
    if (cut && logged++ < 200)
        log::Info("Camera cut at frame {}: moved {:.0f} m, turned {:.0f} degrees (TAA and FSR history dropped)", s_r.frame,
            std::sqrt(moved), std::acos(std::clamp(cosine, -1.0, 1.0)) * 180.0 / 3.14159265358979);
    return cut;
}

// The motion the scene's draws wrote this frame, single-sampled (the samples averaged: where one is
// unknown the average is out of range, unknown too), in the shader-read layout: its descriptor, 0 when there is none.
uint32_t DrawnMotion(RenderCommandList& cl)
{
    if (!MotionVectors() || !s_r.motion.texture || s_r.motionClearedFrame != s_r.frame)
        return 0;
    HostImage* motion = &s_r.motion;
    cl.setFramebuffer(nullptr);
    s_r.markedFramebuffer = nullptr;
    if (s_r.motion.samples > 1)
    {
        EnsureImage(s_r.motionSingle, s_r.motion.width, s_r.motion.height, s_r.motion.format, false);
        Transition(cl, s_r.motion, RenderTextureLayout::RESOLVE_SOURCE);
        Transition(cl, s_r.motionSingle, RenderTextureLayout::RESOLVE_DEST);
        const RenderRect rect(0, 0, int32_t(s_r.motion.width), int32_t(s_r.motion.height));
        cl.resolveTextureRegion(s_r.motionSingle.texture.get(), 0, 0, s_r.motion.texture.get(), &rect);
        motion = &s_r.motionSingle;
    }
    Transition(cl, *motion, RenderTextureLayout::SHADER_READ);
    return motion->descriptor;
}

// XeGTAO's defaults (thirdparty/XeGTAO/XeGTAO.h: XE_GTAO_DEFAULT_*).
constexpr float kGtaoRadiusMultiplier = 1.457f, kGtaoFalloffRange = 0.615f, kGtaoSampleDistributionPower = 2.0f,
    kGtaoThinOccluderCompensation = 0.0f, kGtaoFinalValuePower = 2.2f, kGtaoDepthMipSamplingOffset = 3.30f;

// Ambient occlusion on the scene (AC6_AO=1, off by default): the visibility of each pixel from the
// scene's depth, multiplied into the scene's resolved colour before the anti-aliasing (so the temporal ones take its
// grain), by XeGTAO (Intel's ground truth ambient occlusion, as the renderer's own passes: ao.hlsl), then one apply pass.
// The camera: the projection most of the scene's draws have (NoteAoProjection). AC6_AO_DEBUG=1: the visibility alone.
void RunAmbientOcclusion(RenderCommandList& cl, HostImage& image)
{
    const int mode = AoMode();
    if (!mode || !s_r.aoApplyPs || !image.texture ||
        (image.format != RenderFormat::B8G8R8A8_UNORM && image.format != RenderFormat::R16G16B16A16_FLOAT))
        return;
    auto depthIt = s_r.resolved.find(0x05F83000);
    if (depthIt == s_r.resolved.end() || !depthIt->second.texture)
        return;
    HostImage& depth = depthIt->second;
    if (s_r.aoProjectionsFrame != s_r.frame || s_r.aoProjections.empty())
        return;
    // Only in scenes with terrain (the map's pixel shaders drew this frame): the hangar's depth holds its surroundings
    // in the right half of the picture only (every frame, seen with AC6_AO_DEBUG=2), and occlusion there would end at a
    // line down the middle. AC6_AO_ANYWHERE=1: everywhere (to look into the hangar's).
    static const bool anywhere = [] { const char* v = std::getenv("AC6_AO_ANYWHERE"); return v && *v == '1'; }();
    // The hangar: the aircraft alone, by its own depth (below).
    const bool subject = s_r.aoSubject && s_r.aoTerrainFrame == s_r.frame && !s_aoTerrainDraws.empty() && s_r.aoSubjectPs;
    if (s_r.aoMapFrame != s_r.frame && !anywhere && !subject)
        return;
    // (Of the draws into targets of this image's size first.)
    const auto projection = std::max_element(s_r.aoProjections.begin(), s_r.aoProjections.end(),
        [&](const Renderer::AoProjection& a, const Renderer::AoProjection& b) {
            const bool sizeA = a.width == image.width && a.height == image.height, sizeB = b.width == image.width && b.height == image.height;
            return sizeA != sizeB ? sizeB : a.draws < b.draws;
        });
    const float tanX = projection->tanX, tanY = projection->tanY, clipZ = projection->clipZ;

    auto envFloat = [](const char* name, float fallback) { const char* v = std::getenv(name); return v && *v ? float(std::atof(v)) : fallback; };
    static const float radius = envFloat("AC6_AO_RADIUS", 0.5f), slope = envFloat("AC6_AO_SLOPE", 0.1f);
    static const float power = envFloat("AC6_AO_POWER", kGtaoFinalValuePower), debug = envFloat("AC6_AO_DEBUG", 0.0f);
    // Fading out by the distance: gone by 6 km, and before the nearest distance the map's buildings are drawn from
    // (game/draw_distance.cpp: 10 km in mission 01, 3 km a map's default), so that a building never appears with its
    // occlusion on it.
    static const float fadeFromSet = envFloat("AC6_AO_FADE_FROM", 3000.0f), fadeToSet = envFloat("AC6_AO_FADE_TO", 6000.0f);
    const float nearestParts = envFloat("AC6_AO_DEBUG", 0.0f) == 5.0f ? 0.0f : game::MapPartsNearestRange();
    const float fadeTo = nearestParts > 0.0f ? std::min(fadeToSet, nearestParts * 0.9f) : fadeToSet;
    const float fadeFrom = std::min(fadeFromSet, fadeTo * 0.5f);
    // The radius by the distance: the base (0.5) out to 40 units (the cockpit, the player's aircraft in the chase
    // view), then growing by the slope (0.1 per unit) to the cap (AC6_AO_RADIUS_MAX, 12), reached 155 units out: from
    // there on the radius is the same length in the world whatever the distance, so a building's occlusion is the same
    // size on the building however far it is.
    static const float radiusMax = envFloat("AC6_AO_RADIUS_MAX", 12.0f);
    constexpr float kRadiusNear = 40.0f;
    // AC6_AO_SHADOW=<k>: the game's shadow blend taken k times as deep (1: the game's own, default 1.25, 0: black).
    static const float shadowDepth = envFloat("AC6_AO_SHADOW", 1.25f);
    static const bool gameShadow = shadowDepth != 0.0f;
    // The occlusion's gain (AC6_AO_STRENGTH): with the game's shadow as its darkest, 2.5, so an occlusion of 0.4
    // already reaches that shadow (at 1 a wall's foot comes only a tenth under its colour). To black: 1.
    static const float strength = envFloat("AC6_AO_STRENGTH", gameShadow ? 2.5f : 1.0f);
    static const int quality = std::clamp(int(envFloat("AC6_AO_QUALITY", 2.0f)), 0, 3);
    // XeGTAO's denoiser: one pass where a temporal filter follows (TAA, FSR: its grain is theirs to take), two without
    // (XeGTAO's "sharp" and "medium"), AC6_AO_DENOISE=<0..3> sets it.
    static const int denoiseSet = std::clamp(int(envFloat("AC6_AO_DENOISE", -1.0f)), -1, 3);
    const int denoises = denoiseSet >= 0 ? denoiseSet : (TaaOn() || FsrRunning()) ? 1 : 2;
    // AC6_AO_RES=<divisor>: XeGTAO's visibility at 1 / divisor of the scene's size on each axis (High 1, Low 2).
    static const float divisorSet = envFloat("AC6_AO_RES", 0.0f);
    const uint32_t divisor = divisorSet >= 1.0f ? uint32_t(std::min(divisorSet, 4.0f)) : mode == 2 ? 2u : 1u;
    const uint32_t w = std::max(1u, image.width / divisor), h = std::max(1u, image.height / divisor);
    constexpr float kFarthest = 1.0e6f;

    auto pipelineFor = [&](RenderShader* shader, RenderFormat format) -> RenderPipeline* {
        auto& pipeline = s_r.aoPipelines[{ shader, uint32_t(format) }];
        if (!pipeline && shader)
        {
            RenderGraphicsPipelineDesc desc;
            desc.pipelineLayout = s_r.layout.get();
            desc.vertexShader = s_r.copyVs.get();
            desc.pixelShader = shader;
            desc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
            desc.cullMode = RenderCullMode::NONE;
            desc.renderTargetFormat[0] = format;
            desc.renderTargetBlend[0] = RenderBlendDesc::Copy();
            desc.renderTargetCount = 1;
            pipeline = g_video.device->createGraphicsPipeline(desc);
        }
        return pipeline.get();
    };
    // The parameters (ao.hlsl LoadParams): XeGTAO's constants, then the renderer's.
    float values[48] = {};
    values[0] = float(w);
    values[1] = float(h);
    values[2] = 1.0f / float(w);
    values[3] = 1.0f / float(h);
    values[4] = tanX * 2.0f;             // NDCToViewMul
    values[5] = tanY * -2.0f;
    values[6] = tanX * -1.0f;            // NDCToViewAdd
    values[7] = tanY;
    values[8] = values[4] * values[2];   // NDCToViewMul_x_PixelSize
    values[9] = values[5] * values[3];
    values[10] = radius;
    values[11] = slope;
    values[12] = kGtaoFalloffRange;
    values[13] = kGtaoRadiusMultiplier;
    values[14] = power;
    values[15] = denoises == 0 ? 1e4f : 1.2f;   // (XeGTAO: a high value disables the denoise)
    values[16] = kGtaoSampleDistributionPower;
    values[17] = kGtaoThinOccluderCompensation;
    values[18] = kGtaoDepthMipSamplingOffset;
    // (XeGTAO: the noise changes with the frame only where a temporal filter follows.)
    values[19] = denoises > 0 && (TaaOn() || FsrRunning()) ? float(s_r.frame % 64) : 0.0f;
    constexpr float kSlices[4] = { 1.0f, 2.0f, 3.0f, 9.0f }, kSteps[4] = { 2.0f, 2.0f, 3.0f, 3.0f };   // XeGTAO's low, medium, high, ultra
    values[20] = kSlices[quality];
    values[21] = kSteps[quality];
    values[22] = clipZ;
    values[23] = kFarthest;
    values[29] = strength;
    values[30] = fadeFrom;
    values[31] = fadeTo;
    values[32] = image.format == RenderFormat::B8G8R8A8_UNORM ? 240.5f / 255.0f : 1.0e9f;
    values[33] = debug;
    values[34] = 0.0f;
    // The terrain: its own depth, as a single-sampled image, where the frame drew it for this scene, what it
    // takes of the occlusion (AoTerrainMode), and whether the apply pass leaves it alone.
    HostImage* terrain = nullptr;
    int terrainMode = AoTerrainMode();
    bool terrainSamples = false;
    if ((terrainMode != 2 || subject) && s_r.aoSubject == subject && s_r.aoTerrainFrame == s_r.frame && s_r.aoTerrainDepth.texture &&
        !s_aoTerrainDraws.empty())
    {
        // The terrain's draws the frame noted (Draw), or the hangar's aircraft's, in one pass.
        HostImage& own = s_r.aoTerrainDepth;
        MarkPass([subject] { return std::string(subject ? "ambient occlusion: the aircraft's depth" : "ambient occlusion: the terrain's depth"); });
        Transition(cl, own, RenderTextureLayout::DEPTH_WRITE);
        cl.setFramebuffer(GetFramebuffer(nullptr, &own));
        s_r.markedFramebuffer = nullptr;
        SetViewport(cl, RenderViewport(0.0f, 0.0f, float(own.width), float(own.height)));
        SetScissor(cl, RenderRect(0, 0, int32_t(own.width), int32_t(own.height)));
        cl.clearDepthStencil(true, true, 0.0f, 0);
        own.fresh = false;
        BindCommon(cl);
        for (const AoTerrainDraw& note : s_aoTerrainDraws)
        {
            if (breadcrumbs::Detailed())
            {
                breadcrumbs::NoteDraw(cl, s_r.frame, 0, 0);
                breadcrumbs::Detail(std::format("ambient occlusion: a terrain draw again, {} indices", note.indexCount));
            }
            SetViewport(cl, note.viewport);
            SetScissor(cl, RenderRect(0, 0, int32_t(note.width), int32_t(note.height)));
            SetPipeline(cl, note.pipeline);
            if (UniformConstants())
                cl.setGraphicsDescriptorSetDynamic(s_r.constantSet.get(), 4, note.constantOffsets, 3);
            else
                cl.setGraphicsPushConstants(0, note.addresses, 0, sizeof(note.addresses));
            for (uint32_t i = 0; i < note.viewCount; i++)
                cl.setVertexBuffers(note.layout->slots[i].index, &note.views[i], 1, &note.layout->slots[i]);
            if (note.indexCount)
            {
                RenderIndexBufferView ib(RenderBufferReference(s_r.upload.buffer.get(), note.indexOffset), note.indexCount * 4, RenderFormat::R32_UINT);
                cl.setIndexBuffer(&ib);
                cl.drawIndexedInstanced(note.indexCount, 1, 0, 0, 0);
            }
            else
            {
                cl.drawInstanced(note.drawCount, 1, 0, 0);
            }
        }
        s_aoTerrainDraws.clear();
        own.singleCurrent = false;
        cl.setFramebuffer(nullptr);
        s_bound = BoundState{ g_video.recordingEpoch };   // (the constants, the vertex buffers: the next draw binds its own)
        // Read as drawn: a multisampled one by its first sample (what the scene's depth resolve takes), without the
        // pass a single-sampled copy costs.
        if (own.samples > 1 && own.msDepthDescriptor)
        {
            terrain = &own;
            terrainSamples = true;
        }
        else
        {
            terrain = &Readable(cl, own);
        }
        Transition(cl, *terrain, RenderTextureLayout::SHADER_READ);
    }
    if (!terrain)
        terrainMode = 2;
    if (subject && !terrain)
        return;
    values[35] = terrain ? std::bit_cast<float>(terrainSamples ? terrain->msDepthDescriptor : terrain->descriptor) : 0.0f;
    values[36] = float(terrainMode);
    values[37] = terrainMode == 0 ? 1.0f : 0.0f;
    values[38] = terrainSamples ? 1.0f : 0.0f;
    values[39] = radiusMax;
    values[44] = kRadiusNear;
    // (Deeper than the game's: more of the colour taken away, and the colour added kept in its proportion to what is left.)
    const float shadowAlpha = gameShadow ? std::clamp(s_r.aoShadow[3] * shadowDepth, 0.0f, 1.0f) : 1.0f;
    const float shadowAdd = gameShadow && s_r.aoShadow[3] < 1.0f ? (1.0f - shadowAlpha) / (1.0f - std::clamp(s_r.aoShadow[3], 0.0f, 0.999f)) : 0.0f;
    values[40] = s_r.aoShadow[0] * shadowAdd;
    values[41] = s_r.aoShadow[1] * shadowAdd;
    values[42] = s_r.aoShadow[2] * shadowAdd;
    values[43] = shadowAlpha;
    auto upload = [&]() {
        auto params = s_r.upload.Allocate(sizeof(values), 16);
        if (params.memory)
            std::memcpy(params.memory, values, sizeof(values));
        return params;
    };
    auto draw = [&](RenderPipeline* pipeline, HostImage& target, uint32_t tw, uint32_t th, uint32_t texture, uint32_t second, uint64_t address) {
        Transition(cl, target, RenderTextureLayout::COLOR_WRITE);
        cl.setFramebuffer(GetFramebuffer(&target, nullptr));
        s_r.markedFramebuffer = nullptr;
        SetViewport(cl, RenderViewport(0.0f, 0.0f, float(tw), float(th)));
        SetScissor(cl, RenderRect(0, 0, int32_t(tw), int32_t(th)));
        SetPipeline(cl, pipeline);
        BindCommon(cl);
        const uint32_t push[4] = { texture, second, uint32_t(address), uint32_t(address >> 32) };
        cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
        cl.drawInstanced(3, 1, 0, 0);
        cl.setFramebuffer(nullptr);
    };
    static uint32_t logged = 0;
    if (logged++ < 2)
        log::Info("Ambient occlusion frame {} ({}): {}x{} for the scene {}x{} format {}, half field of view {:.1f} x {:.1f} degrees, clip z {:.3f}; "
            "depth {}x{}, the terrain's {}x{} (mode {})",
            s_r.frame, "XeGTAO", w, h, image.width, image.height, int(image.format), std::atan(tanX) * 57.29578,
            std::atan(tanY) * 57.29578, clipZ, depth.width, depth.height, terrain ? terrain->width : 0, terrain ? terrain->height : 0, terrainMode);
    cl.setFramebuffer(nullptr);
    s_r.markedFramebuffer = nullptr;
    Transition(cl, depth, RenderTextureLayout::SHADER_READ);
    HostImage* sceneDepth = &depth;
    if (subject)
    {
        // The hangar: the depth everything below reads is the aircraft's where the aircraft is the surface seen, and
        // nothing (the sky's 0) elsewhere. From there on as a scene with no terrain to leave out.
        RenderPipeline* subjectPipeline = pipelineFor(s_r.aoSubjectPs.get(), RenderFormat::R32_FLOAT);
        const auto params = upload();
        if (!subjectPipeline || !params.memory)
            return;
        EnsureImage(s_r.aoSubjectDepth, depth.width, depth.height, RenderFormat::R32_FLOAT, false, 1, false, true);
        MarkPass([] { return std::string("ambient occlusion: the aircraft alone"); });
        draw(subjectPipeline, s_r.aoSubjectDepth, depth.width, depth.height, depth.descriptor, 0, params.deviceAddress);
        Transition(cl, s_r.aoSubjectDepth, RenderTextureLayout::SHADER_READ);
        sceneDepth = &s_r.aoSubjectDepth;
        values[36] = 2.0f;
        values[37] = 0.0f;
    }
    HostImage* visibility = nullptr;
    {
        RenderPipeline* depthPipeline = pipelineFor(s_r.aoDepthPs.get(), RenderFormat::R32_FLOAT);
        RenderPipeline* mipPipeline = pipelineFor(s_r.aoDepthMipPs.get(), RenderFormat::R32_FLOAT);
        RenderPipeline* mainPipeline = pipelineFor(s_r.aoMainPs.get(), RenderFormat::R8G8_UNORM);
        RenderPipeline* denoisePipeline = pipelineFor(s_r.aoDenoisePs.get(), RenderFormat::R8G8_UNORM);
        if (!depthPipeline || !mipPipeline || !mainPipeline || !denoisePipeline)
            return;
        for (uint32_t level = 0; level < 5; level++)
            EnsureImage(s_r.aoDepth[level], std::max(1u, w >> level), std::max(1u, h >> level), RenderFormat::R32_FLOAT, false, 1, false, true);
        for (HostImage& work : s_r.aoWork)
            EnsureImage(work, w, h, RenderFormat::R8G8_UNORM, false, 1, false, true);
        for (uint32_t level = 0; level < 5; level++)
            values[24 + level] = std::bit_cast<float>(s_r.aoDepth[level].descriptor);
        const auto params = upload();
        if (!params.memory)
            return;
        MarkPass([] { return std::string("ambient occlusion: depth"); });
        draw(depthPipeline, s_r.aoDepth[0], w, h, sceneDepth->descriptor, 0, params.deviceAddress);
        for (uint32_t level = 1; level < 5; level++)
        {
            Transition(cl, s_r.aoDepth[level - 1], RenderTextureLayout::SHADER_READ);
            draw(mipPipeline, s_r.aoDepth[level], std::max(1u, w >> level), std::max(1u, h >> level), s_r.aoDepth[level - 1].descriptor, 0,
                params.deviceAddress);
        }
        Transition(cl, s_r.aoDepth[4], RenderTextureLayout::SHADER_READ);
        MarkPass([] { return std::string("ambient occlusion: XeGTAO"); });
        draw(mainPipeline, s_r.aoWork[0], w, h, 0, 0, params.deviceAddress);
        // XeGTAO's denoiser: its passes, the last one scaling the visibility back (one even when it is off).
        MarkPass([] { return std::string("ambient occlusion: denoise"); });
        const int passes = std::max(1, denoises);
        uint32_t from = 0;
        for (int pass = 0; pass < passes; pass++)
        {
            uint64_t address = params.deviceAddress;
            if (pass == passes - 1)
            {
                values[34] = 1.0f;
                const auto last = upload();
                if (!last.memory)
                    return;
                address = last.deviceAddress;
            }
            Transition(cl, s_r.aoWork[from], RenderTextureLayout::SHADER_READ);
            draw(denoisePipeline, s_r.aoWork[from ^ 1], w, h, s_r.aoWork[from].descriptor, 0, address);
            from ^= 1;
        }
        visibility = &s_r.aoWork[from];
    }
    // Into the scene: a copy of it x the visibility.
    RenderPipeline* applyPipeline = pipelineFor(s_r.aoApplyPs.get(), image.format);
    if (!applyPipeline)
        return;
    MarkPass([] { return std::string("ambient occlusion: apply"); });
    values[34] = 0.0f;
    const auto params = upload();
    if (!params.memory)
        return;
    EnsureImage(s_r.aoCopy, image.width, image.height, image.format, false, 1, false, true);
    Transition(cl, image, RenderTextureLayout::COPY_SOURCE);
    Transition(cl, s_r.aoCopy, RenderTextureLayout::COPY_DEST);
    cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(s_r.aoCopy.texture.get()), RenderTextureCopyLocation::Subresource(image.texture.get()));
    Transition(cl, s_r.aoCopy, RenderTextureLayout::SHADER_READ);
    Transition(cl, *visibility, RenderTextureLayout::SHADER_READ);
    draw(applyPipeline, image, image.width, image.height, s_r.aoCopy.descriptor, visibility->descriptor | (sceneDepth->descriptor << 16), params.deviceAddress);
}

// The enhanced bloom (bloom.hlsl): the bright image from the scene at half the canvas's size, six levels
// below it, the results added back up from the coarsest with each level's weight, and the result with the star added into
// the canvas. AC6_BLOOM_DEBUG=1 / 2 / 3: the game's bloom / this bloom / the star alone. AC6_BLOOM_GAIN: a scale on this
// bloom (1).
void RunEnhancedBloom(RenderCommandList& cl, uint32_t device, const Targets& targets, const SharedConstants& sc)
{
    HostImage& canvas = *targets.color;
    if (!s_r.bloomBrightPs || !s_r.bloomDownPs || !s_r.bloomUpPs || !s_r.bloomApplyPs || !canvas.texture ||
        (canvas.format != RenderFormat::B8G8R8A8_UNORM && canvas.format != RenderFormat::R16G16B16A16_FLOAT))
        return;
    const uint32_t scene = sc.texture2DIndices[0], gameBloom = sc.texture2DIndices[1], star = sc.texture2DIndices[2];
    // (The adapted luminance: the composite's vertex shader's TEXTURE_SCENE3, fetch slot 19: the 1 x 1 32-bit float.)
    const uint32_t luminance = sc.texture2DIndices[19];
    if (scene == kNullTexture2D || star == kNullTexture2D || luminance == kNullTexture2D)
        return;
    auto pipelineFor = [&](RenderShader* shader, RenderFormat format, bool add) -> RenderPipeline* {
        auto& pipeline = s_r.bloomPipelines[{ shader, uint32_t(format), add }];
        if (!pipeline && shader)
        {
            RenderGraphicsPipelineDesc desc;
            desc.pipelineLayout = s_r.layout.get();
            desc.vertexShader = s_r.copyVs.get();
            desc.pixelShader = shader;
            desc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
            desc.cullMode = RenderCullMode::NONE;
            desc.renderTargetFormat[0] = format;
            RenderBlendDesc blend = RenderBlendDesc::Copy();
            if (add)
            {
                blend.blendEnabled = true;
                blend.srcBlend = RenderBlend::ONE;
                blend.dstBlend = RenderBlend::ONE;
                blend.blendOp = RenderBlendOperation::ADD;
                blend.srcBlendAlpha = RenderBlend::ZERO;
                blend.dstBlendAlpha = RenderBlend::ONE;
                blend.blendOpAlpha = RenderBlendOperation::ADD;
            }
            desc.renderTargetBlend[0] = blend;
            desc.renderTargetCount = 1;
            pipeline = g_video.device->createGraphicsPipeline(desc);
        }
        return pipeline.get();
    };
    constexpr RenderFormat kLevelFormat = RenderFormat::R16G16B16A16_FLOAT;
    RenderPipeline* bright = pipelineFor(s_r.bloomBrightPs.get(), kLevelFormat, false);
    RenderPipeline* down = pipelineFor(s_r.bloomDownPs.get(), kLevelFormat, false);
    RenderPipeline* up = pipelineFor(s_r.bloomUpPs.get(), kLevelFormat, false);
    static const float debug = [] { const char* v = std::getenv("AC6_BLOOM_DEBUG"); return v && *v ? float(std::atof(v)) : 0.0f; }();
    RenderPipeline* apply = pipelineFor(s_r.bloomApplyPs.get(), canvas.format, debug == 0.0f);
    if (!bright || !down || !up || !apply)
        return;
    auto draw = [&](RenderPipeline* pipeline, HostImage& target, const uint32_t* push, uint32_t words) {
        Transition(cl, target, RenderTextureLayout::COLOR_WRITE);
        cl.setFramebuffer(GetFramebuffer(&target, nullptr));
        s_r.markedFramebuffer = nullptr;
        SetViewport(cl, RenderViewport(0.0f, 0.0f, float(target.width), float(target.height)));
        SetScissor(cl, RenderRect(0, 0, int32_t(target.width), int32_t(target.height)));
        SetPipeline(cl, pipeline);
        BindCommon(cl);
        cl.setGraphicsPushConstants(0, push, 0, words * 4);
        cl.drawInstanced(3, 1, 0, 0);
        cl.setFramebuffer(nullptr);
        s_r.markedFramebuffer = nullptr;
    };
    cl.setFramebuffer(nullptr);
    s_r.markedFramebuffer = nullptr;
    MarkPass([] { return std::string("enhanced bloom"); });
    constexpr uint32_t kLevels = 7;
    for (uint32_t n = 0; n < kLevels; n++)
        for (HostImage* image : { &s_r.bloomLevels[n], &s_r.bloomResults[n] })
            EnsureImage(*image, std::max(1u, canvas.width >> (n + 1)), std::max(1u, canvas.height >> (n + 1)), kLevelFormat, false, 1, false, true);
    // The bright image: the exposure band from the composite's vertex constants (c106.xy), the game's bright pass's
    // threshold and offset.
    const be<float>* vsRegs = reinterpret_cast<const be<float>*>(g_memory.As<uint8_t>(device) + kDevVsConstants);
    {
        const uint32_t push[6] = { scene, luminance, std::bit_cast<uint32_t>(float(vsRegs[106 * 4 + 0])),
            std::bit_cast<uint32_t>(float(vsRegs[106 * 4 + 1])), std::bit_cast<uint32_t>(s_r.bloomBright[0]),
            std::bit_cast<uint32_t>(s_r.bloomBright[1]) };
        draw(bright, s_r.bloomLevels[0], push, 6);
    }
    for (uint32_t n = 1; n < kLevels; n++)
    {
        Transition(cl, s_r.bloomLevels[n - 1], RenderTextureLayout::SHADER_READ);
        const uint32_t push[2] = { s_r.bloomLevels[n - 1].descriptor, 0 };
        draw(down, s_r.bloomLevels[n], push, 2);
    }
    Transition(cl, s_r.bloomLevels[kLevels - 1], RenderTextureLayout::SHADER_READ);
    // Back up: each level's share of the light, the shares summing to 1, so the bloom's light is the bright pass's
    // spread out. The widest level is as wide as the game's own bloom, whose Gaussian spans about 3% of the picture's
    // height.
    constexpr float kWeights[kLevels] = { 0.04f, 0.06f, 0.10f, 0.15f, 0.25f, 0.25f, 0.15f };
    for (int n = int(kLevels) - 1; n >= 0; n--)
    {
        const uint32_t coarser = n + 1 < int(kLevels) ? s_r.bloomResults[n + 1].descriptor : 0u;
        if (n + 1 < int(kLevels))
            Transition(cl, s_r.bloomResults[n + 1], RenderTextureLayout::SHADER_READ);
        const uint32_t push[3] = { s_r.bloomLevels[n].descriptor, coarser, std::bit_cast<uint32_t>(kWeights[n]) };
        draw(up, s_r.bloomResults[n], push, 3);
    }
    Transition(cl, s_r.bloomResults[0], RenderTextureLayout::SHADER_READ);
    // Into the canvas.
    static const float gain = [] { const char* v = std::getenv("AC6_BLOOM_GAIN"); return v && *v ? float(std::atof(v)) : 1.0f; }();
    constexpr float kBloomEnergy = 1.0f;   // this bloom's light against the game's at the same scale (measured)
    const uint32_t push[6] = { s_r.bloomResults[0].descriptor, star, std::bit_cast<uint32_t>(s_r.bloomScales[0] * kBloomEnergy * gain),
        std::bit_cast<uint32_t>(s_r.bloomScales[1]), std::bit_cast<uint32_t>(debug), gameBloom };
    draw(apply, canvas, push, 6);
}

// Light from the game's emitters on what is near them (emitter_light.hlsl, AC6_EMITTER_LIGHT): at the
// scene's resolve, the frame's noted emitter draws issued into the light image and its depth, ten levels made of it,
// and the light added into the scene's colour. AC6_EMITTER_LIGHT_DEBUG=1: the light alone.
void RunEmitterLight(RenderCommandList& cl, HostImage& image)
{
    if (s_r.lightNotesFrame != s_r.frame || s_lightDraws.empty() || !s_r.lightApplyPs || !s_r.lightDownPs || !s_r.lightDown0Ps ||
        !s_r.lightDown0bPs ||
        !image.texture || (image.format != RenderFormat::B8G8R8A8_UNORM && image.format != RenderFormat::R16G16B16A16_FLOAT))
        return;
    auto depthIt = s_r.resolved.find(0x05F83000);
    if (depthIt == s_r.resolved.end() || !depthIt->second.texture || s_r.aoProjectionsFrame != s_r.frame || s_r.aoProjections.empty())
        return;
    HostImage& depth = depthIt->second;
    const auto projection = std::max_element(s_r.aoProjections.begin(), s_r.aoProjections.end(),
        [&](const Renderer::AoProjection& a, const Renderer::AoProjection& b) {
            const bool sizeA = a.width == image.width && a.height == image.height, sizeB = b.width == image.width && b.height == image.height;
            return sizeA != sizeB ? sizeB : a.draws < b.draws;
        });
    const float tanX = projection->tanX, tanY = projection->tanY, clipZ = projection->clipZ;
    auto pipelineFor = [&](RenderShader* shader, RenderFormat format) -> RenderPipeline* {
        auto& pipeline = s_r.lightPipelines[{ shader, uint32_t(format) }];
        if (!pipeline && shader)
        {
            RenderGraphicsPipelineDesc desc;
            desc.pipelineLayout = s_r.layout.get();
            desc.vertexShader = s_r.copyVs.get();
            desc.pixelShader = shader;
            desc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
            desc.cullMode = RenderCullMode::NONE;
            desc.renderTargetFormat[0] = format;
            desc.renderTargetBlend[0] = RenderBlendDesc::Copy();
            desc.renderTargetCount = 1;
            pipeline = g_video.device->createGraphicsPipeline(desc);
        }
        return pipeline.get();
    };
    RenderPipeline* down0 = pipelineFor(s_r.lightDown0Ps.get(), RenderFormat::R16G16B16A16_FLOAT);
    RenderPipeline* down0b = pipelineFor(s_r.lightDown0bPs.get(), RenderFormat::R16G16B16A16_FLOAT);
    RenderPipeline* down = pipelineFor(s_r.lightDownPs.get(), RenderFormat::R16G16B16A16_FLOAT);
    RenderPipeline* apply = pipelineFor(s_r.lightApplyPs.get(), image.format);
    if (!down0 || !down0b || !down || !apply)
        return;
    // The emitters' draws, in one pass.
    HostImage& light = s_r.lightColor;
    HostImage& lightDepth = s_r.lightDepth;
    MarkPass([] { return std::string("emitter light: the emitters"); });
    Transition(cl, light, RenderTextureLayout::COLOR_WRITE);
    Transition(cl, lightDepth, RenderTextureLayout::DEPTH_WRITE);
    cl.setFramebuffer(GetFramebuffer(&light, &lightDepth));
    s_r.markedFramebuffer = nullptr;
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(light.width), float(light.height)));
    SetScissor(cl, RenderRect(0, 0, int32_t(light.width), int32_t(light.height)));
    cl.clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
    cl.clearDepthStencil(true, true, 0.0f, 0);
    light.fresh = lightDepth.fresh = false;
    BindCommon(cl);
    for (const AoTerrainDraw& note : s_lightDraws)
    {
        if (breadcrumbs::Detailed())
        {
            breadcrumbs::NoteDraw(cl, s_r.frame, 0, 0);
            breadcrumbs::Detail(std::format("emitter light: a draw again, {} indices", note.indexCount));
        }
        SetViewport(cl, note.viewport);
        SetScissor(cl, RenderRect(0, 0, int32_t(note.width), int32_t(note.height)));
        SetPipeline(cl, note.pipeline);
        if (note.blendConstant)
            cl.setBlendConstants(note.blend);
        if (UniformConstants())
            cl.setGraphicsDescriptorSetDynamic(s_r.constantSet.get(), 4, note.constantOffsets, 3);
        else
            cl.setGraphicsPushConstants(0, note.addresses, 0, sizeof(note.addresses));
        for (uint32_t i = 0; i < note.viewCount; i++)
            cl.setVertexBuffers(note.layout->slots[i].index, &note.views[i], 1, &note.layout->slots[i]);
        if (note.indexCount)
        {
            RenderIndexBufferView ib(RenderBufferReference(s_r.upload.buffer.get(), note.indexOffset), note.indexCount * 4, RenderFormat::R32_UINT);
            cl.setIndexBuffer(&ib);
            cl.drawIndexedInstanced(note.indexCount, 1, 0, 0, 0);
        }
        else
        {
            cl.drawInstanced(note.drawCount, 1, 0, 0);
        }
    }
    static uint32_t logged = 0;
    if (logged++ < 3)
        log::Info("Emitter light frame {}: {} emitter draws into {}x{}, the scene {}x{}", s_r.frame, s_lightDraws.size(), light.width, light.height,
            image.width, image.height);
    s_lightDraws.clear();
    cl.setFramebuffer(nullptr);
    s_r.markedFramebuffer = nullptr;
    s_bound = BoundState{ g_video.recordingEpoch };

    auto draw = [&](RenderPipeline* pipeline, HostImage& target, const uint32_t* push, uint32_t words) {
        Transition(cl, target, RenderTextureLayout::COLOR_WRITE);
        cl.setFramebuffer(GetFramebuffer(&target, nullptr));
        s_r.markedFramebuffer = nullptr;
        SetViewport(cl, RenderViewport(0.0f, 0.0f, float(target.width), float(target.height)));
        SetScissor(cl, RenderRect(0, 0, int32_t(target.width), int32_t(target.height)));
        SetPipeline(cl, pipeline);
        BindCommon(cl);
        cl.setGraphicsPushConstants(0, push, 0, words * 4);
        cl.drawInstanced(3, 1, 0, 0);
        cl.setFramebuffer(nullptr);
        s_r.markedFramebuffer = nullptr;
    };
    // The levels.
    MarkPass([] { return std::string("emitter light: levels"); });
    Transition(cl, light, RenderTextureLayout::SHADER_READ);
    Transition(cl, lightDepth, RenderTextureLayout::SHADER_READ);
    constexpr uint32_t kLevels = 10;
    for (uint32_t n = 0; n < kLevels; n++)
        for (HostImage* level : { &s_r.lightLevels[n], &s_r.lightPlaces[n] })
            EnsureImage(*level, std::max(1u, light.width >> (n + 1)), std::max(1u, light.height >> (n + 1)), RenderFormat::R16G16B16A16_FLOAT,
                false, 1, false, true);
    {
        const uint32_t push[3] = { light.descriptor, lightDepth.descriptor, std::bit_cast<uint32_t>(clipZ) };
        draw(down0, s_r.lightLevels[0], push, 3);
        draw(down0b, s_r.lightPlaces[0], push, 3);
    }
    for (uint32_t n = 1; n < kLevels; n++)
        for (HostImage* levels : { s_r.lightLevels, s_r.lightPlaces })
        {
            Transition(cl, levels[n - 1], RenderTextureLayout::SHADER_READ);
            const uint32_t push[2] = { levels[n - 1].descriptor, 0 };
            draw(down, levels[n], push, 2);
        }
    Transition(cl, s_r.lightLevels[kLevels - 1], RenderTextureLayout::SHADER_READ);
    Transition(cl, s_r.lightPlaces[kLevels - 1], RenderTextureLayout::SHADER_READ);
    // Into the scene.
    static const float debug = [] { const char* v = std::getenv("AC6_EMITTER_LIGHT_DEBUG"); return v && *v ? float(std::atof(v)) : 0.0f; }();
    float values[28] = {};
    for (uint32_t n = 0; n < kLevels; n++)
    {
        values[n] = std::bit_cast<float>(s_r.lightLevels[n].descriptor);
        values[10 + n] = std::bit_cast<float>(s_r.lightPlaces[n].descriptor);
    }
    values[20] = clipZ;
    values[21] = 2.0f * tanY / float(std::max(light.height, 1u));
    // The emitters' intensity: a sprite's colour is what the display can show of a flame, not its light: x 8 at 1
    // (at lower gains the lights barely show on their surroundings at night), what reaches a pixel then
    // eased toward a ceiling in the shader.
    constexpr float kEmitterLightGain = 8.0f;
    values[22] = EmitterLight() * kEmitterLightGain;
    values[23] = image.format == RenderFormat::B8G8R8A8_UNORM ? 240.5f / 255.0f : 1.0e9f;
    values[24] = debug;
    values[25] = tanX;
    values[26] = tanY;
    auto params = s_r.upload.Allocate(sizeof(values), 16);
    if (!params.memory)
        return;
    std::memcpy(params.memory, values, sizeof(values));
    MarkPass([] { return std::string("emitter light: apply"); });
    EnsureImage(s_r.lightCopy, image.width, image.height, image.format, false, 1, false, true);
    Transition(cl, image, RenderTextureLayout::COPY_SOURCE);
    Transition(cl, s_r.lightCopy, RenderTextureLayout::COPY_DEST);
    cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(s_r.lightCopy.texture.get()), RenderTextureCopyLocation::Subresource(image.texture.get()));
    Transition(cl, s_r.lightCopy, RenderTextureLayout::SHADER_READ);
    Transition(cl, depth, RenderTextureLayout::SHADER_READ);
    const uint32_t push[4] = { s_r.lightCopy.descriptor, depth.descriptor, uint32_t(params.deviceAddress), uint32_t(params.deviceAddress >> 32) };
    draw(apply, image, push, 4);
}

// Object motion blur. Every pixel's motion for it: the motion completion pass (the draws'
// motion where they wrote it, the camera's reprojection of the depth elsewhere. This frame's position to last frame's,
// in texture coordinates), at the scene depth's size. Null when the frame has no camera or depth.
HostImage* MotionForBlur(RenderCommandList& cl)
{
    if (!s_r.motionCompletePs || !s_r.motionBlurPs || !s_r.taaHavePrev || s_r.frame != s_r.taaLastFrame + 1)
        return nullptr;
    auto depthIt = s_r.resolved.find(0x05F83000);
    if (depthIt == s_r.resolved.end() || !depthIt->second.texture)
        return nullptr;
    HostImage& depth = depthIt->second;
    double vp[16], r[16];
    if (!SceneReprojection(vp, r))
        return nullptr;
    if (!s_r.blurMotionPipeline)
    {
        RenderGraphicsPipelineDesc desc;
        desc.pipelineLayout = s_r.layout.get();
        desc.vertexShader = s_r.copyVs.get();
        desc.pixelShader = s_r.motionCompletePs.get();
        desc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
        desc.cullMode = RenderCullMode::NONE;
        desc.renderTargetFormat[0] = RenderFormat::R16G16_FLOAT;
        desc.renderTargetBlend[0] = RenderBlendDesc::Copy();
        desc.renderTargetCount = 1;
        s_r.blurMotionPipeline = g_video.device->createGraphicsPipeline(desc);
        if (!s_r.blurMotionPipeline)
            return nullptr;
    }
    const uint32_t w = depth.width, h = depth.height;
    EnsureImage(s_r.blurMotion, w, h, RenderFormat::R16G16_FLOAT, false, 1, false, true);
    const auto [jx, jy] = TaaJitter(SeedFrame());
    // (The draws' motion image is read texel for texel from its top left, as FSR's pass reads it: with FSR upscaling
    // it is the larger image, the scene in its top left. Taken for another size's, the player's aircraft would have
    // only the camera's motion and be blurred with the ground.)
    const uint32_t drawn = DrawnMotion(cl);
    auto params = s_r.upload.Allocate(80, 16);
    if (!params.memory)
        return nullptr;
    float* f = reinterpret_cast<float*>(params.memory);
    for (int i = 0; i < 16; i++)
        f[i] = float(r[i]);
    f[16] = 2.0f * jx / float(w);
    f[17] = -2.0f * jy / float(h);
    f[18] = f[19] = 0.0f;
    MarkPass([] { return std::string("motion blur: motion"); });
    Transition(cl, depth, RenderTextureLayout::SHADER_READ);
    Transition(cl, s_r.blurMotion, RenderTextureLayout::COLOR_WRITE);
    cl.setFramebuffer(GetFramebuffer(&s_r.blurMotion, nullptr));
    s_r.markedFramebuffer = nullptr;
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(w), float(h)));
    SetScissor(cl, RenderRect(0, 0, int32_t(w), int32_t(h)));
    SetPipeline(cl, s_r.blurMotionPipeline.get());
    BindCommon(cl);
    const uint32_t push[4] = { drawn, depth.descriptor, uint32_t(params.deviceAddress), uint32_t(params.deviceAddress >> 32) };
    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    cl.drawInstanced(3, 1, 0, 0);
    cl.setFramebuffer(nullptr);
    s_r.markedFramebuffer = nullptr;
    return &s_r.blurMotion;
}

// The blur itself, on the scene after the temporal anti-aliasing (at the frame's size with FSR upscaling: the motion and
// the depth are read by position, at their own size).
void RunMotionBlur(RenderCommandList& cl, HostImage& image, HostImage& motion)
{
    const float strength = MotionBlurStrength();
    if (strength <= 0.0f || !s_r.motionBlurPs || !image.texture ||
        (image.format != RenderFormat::B8G8R8A8_UNORM && image.format != RenderFormat::R16G16B16A16_FLOAT))
        return;
    auto depthIt = s_r.resolved.find(0x05F83000);
    if (depthIt == s_r.resolved.end() || !depthIt->second.texture)
        return;
    HostImage& depth = depthIt->second;
    auto& pipeline = s_r.motionBlurPipelines[uint32_t(image.format)];
    if (!pipeline)
    {
        RenderGraphicsPipelineDesc desc;
        desc.pipelineLayout = s_r.layout.get();
        desc.vertexShader = s_r.copyVs.get();
        desc.pixelShader = s_r.motionBlurPs.get();
        desc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
        desc.cullMode = RenderCullMode::NONE;
        desc.renderTargetFormat[0] = image.format;
        desc.renderTargetBlend[0] = RenderBlendDesc::Copy();
        desc.renderTargetCount = 1;
        pipeline = g_video.device->createGraphicsPipeline(desc);
    }
    if (!pipeline)
        return;
    // The exposure: 1 / 120 s at 1 (a half-open shutter at 60 fps) whatever the frame rate, as a share of the frame's own
    // time (the motion is a frame's): the frame time smoothed, AC6_MOTION_BLUR_FPS=<n> a fixed rate in its place (tests:
    // a replay's frames come as fast as they can).
    static auto last = std::chrono::steady_clock::now();
    static float smoothed = 1.0f / 60.0f;
    const auto now = std::chrono::steady_clock::now();
    const float dt = std::clamp(std::chrono::duration<float>(now - last).count(), 1.0f / 1000.0f, 1.0f / 15.0f);
    last = now;
    smoothed += (dt - smoothed) * 0.1f;
    static const float fixedFps = [] { const char* v = std::getenv("AC6_MOTION_BLUR_FPS"); return v && *v ? float(std::atof(v)) : 0.0f; }();
    const float frameTime = fixedFps > 0.0f ? 1.0f / fixedFps : smoothed;
    const float exposure = std::min(strength * (1.0f / 120.0f) / frameTime, 2.0f);
    constexpr float kLongest = 0.05f;   // of the picture's height
    MarkPass([] { return std::string("motion blur"); });
    cl.setFramebuffer(nullptr);
    s_r.markedFramebuffer = nullptr;
    EnsureImage(s_r.motionBlurCopy, image.width, image.height, image.format, false, 1, false, true);
    Transition(cl, image, RenderTextureLayout::COPY_SOURCE);
    Transition(cl, s_r.motionBlurCopy, RenderTextureLayout::COPY_DEST);
    cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(s_r.motionBlurCopy.texture.get()), RenderTextureCopyLocation::Subresource(image.texture.get()));
    Transition(cl, s_r.motionBlurCopy, RenderTextureLayout::SHADER_READ);
    Transition(cl, motion, RenderTextureLayout::SHADER_READ);
    Transition(cl, depth, RenderTextureLayout::SHADER_READ);
    Transition(cl, image, RenderTextureLayout::COLOR_WRITE);
    cl.setFramebuffer(GetFramebuffer(&image, nullptr));
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(image.width), float(image.height)));
    SetScissor(cl, RenderRect(0, 0, int32_t(image.width), int32_t(image.height)));
    SetPipeline(cl, pipeline.get());
    BindCommon(cl);
    const uint32_t push[7] = { s_r.motionBlurCopy.descriptor, motion.descriptor | (depth.descriptor << 16), std::bit_cast<uint32_t>(exposure),
        std::bit_cast<uint32_t>(kLongest), std::bit_cast<uint32_t>(float(image.width) / float(std::max(image.height, 1u))),
        uint32_t(s_r.frame & 0xFFFFFFu), 0u };
    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    cl.drawInstanced(3, 1, 0, 0);
    cl.setFramebuffer(nullptr);
    s_r.markedFramebuffer = nullptr;
}

// The optional lens effects (lens_fx.hlsl): the picture copied, then drawn back through the pass.
void RunLensFx(RenderCommandList& cl, HostImage& image, const LensFx& fx)
{
    if (!s_r.lensFxPs || !image.texture || (image.format != RenderFormat::B8G8R8A8_UNORM && image.format != RenderFormat::R16G16B16A16_FLOAT))
        return;
    auto& pipeline = s_r.lensFxPipelines[uint32_t(image.format)];
    if (!pipeline)
    {
        RenderGraphicsPipelineDesc desc;
        desc.pipelineLayout = s_r.layout.get();
        desc.vertexShader = s_r.copyVs.get();
        desc.pixelShader = s_r.lensFxPs.get();
        desc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
        desc.cullMode = RenderCullMode::NONE;
        desc.renderTargetFormat[0] = image.format;
        desc.renderTargetBlend[0] = RenderBlendDesc::Copy();
        desc.renderTargetCount = 1;
        pipeline = g_video.device->createGraphicsPipeline(desc);
    }
    if (!pipeline)
        return;
    MarkPass([] { return std::string("lens effects"); });
    cl.setFramebuffer(nullptr);
    s_r.markedFramebuffer = nullptr;
    EnsureImage(s_r.lensFxCopy, image.width, image.height, image.format, false, 1, false, true);
    Transition(cl, image, RenderTextureLayout::COPY_SOURCE);
    Transition(cl, s_r.lensFxCopy, RenderTextureLayout::COPY_DEST);
    cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(s_r.lensFxCopy.texture.get()), RenderTextureCopyLocation::Subresource(image.texture.get()));
    Transition(cl, s_r.lensFxCopy, RenderTextureLayout::SHADER_READ);
    Transition(cl, image, RenderTextureLayout::COLOR_WRITE);
    cl.setFramebuffer(GetFramebuffer(&image, nullptr));
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(image.width), float(image.height)));
    SetScissor(cl, RenderRect(0, 0, int32_t(image.width), int32_t(image.height)));
    SetPipeline(cl, pipeline.get());
    BindCommon(cl);
    const uint32_t push[7] = { s_r.lensFxCopy.descriptor, uint32_t(s_r.frame & 0xFFFFFFu), std::bit_cast<uint32_t>(fx.grain),
        std::bit_cast<uint32_t>(fx.vignette), std::bit_cast<uint32_t>(fx.aberration), 0u,
        std::bit_cast<uint32_t>(float(image.width) / float(std::max(image.height, 1u))) };
    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    cl.drawInstanced(3, 1, 0, 0);
    cl.setFramebuffer(nullptr);
    s_r.markedFramebuffer = nullptr;
}

// The sea's reflection of the scene (AC6_SSR=1, internal_shaders.hlsl WATER_SSR): one pass from the
// scene as the effects composite leaves it and the scene's depth into an image the sea's shaders read in the next frame (SeaReflectionAtDraw), with
// the camera the frame's sea was drawn with. AC6_SSR_DEBUG=1: what each ray found, as colours on the sea.
void RunSeaReflection(RenderCommandList& cl, HostImage& image, const HostImage* surface)
{
    if (!s_r.waterSsrPs || !image.texture || (image.format != RenderFormat::B8G8R8A8_UNORM && image.format != RenderFormat::R16G16B16A16_FLOAT))
        return;
    auto depthIt = s_r.resolved.find(0x05F83000);
    if (depthIt == s_r.resolved.end() || !depthIt->second.texture || s_r.seaViewsFrame != s_r.frame)
        return;
    HostImage& depth = depthIt->second;
    const auto view = std::find_if(s_r.seaViews.begin(), s_r.seaViews.end(), [&](const Renderer::SeaView& v) { return v.target == surface; });
    if (view == s_r.seaViews.end())
        return;   // no sea drawn into the scene this frame
    double inverse[16];
    if (!Invert4(view->vp, inverse) || std::fabs(inverse[11]) < 1e-12)
        return;
    auto& pipeline = s_r.waterSsrPipelines[uint32_t(image.format)];
    if (!pipeline)
    {
        RenderGraphicsPipelineDesc desc;
        desc.pipelineLayout = s_r.layout.get();
        desc.vertexShader = s_r.copyVs.get();
        desc.pixelShader = s_r.waterSsrPs.get();
        desc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
        desc.cullMode = RenderCullMode::NONE;
        desc.renderTargetFormat[0] = image.format;
        desc.renderTargetBlend[0] = RenderBlendDesc::Copy();
        desc.renderTargetCount = 1;
        pipeline = g_video.device->createGraphicsPipeline(desc);
        if (!pipeline)
            return;
    }
    auto envFloat = [](const char* name, float fallback) { const char* v = std::getenv(name); return v && *v ? float(std::atof(v)) : fallback; };
    static const float sea = envFloat("AC6_SSR_SEA", 0.0f), stride = envFloat("AC6_SSR_STRIDE", 10.0f), debug = envFloat("AC6_SSR_DEBUG", 0.0f);
    static const float waves = envFloat("AC6_SSR_WAVES", 0.03f), thickness = envFloat("AC6_SSR_THICKNESS", 0.2f);
    // AC6_SSR_RES=<divisor>: the reflection at 1 / divisor of the scene's size on each axis (2: every pixel is marched,
    // and half the size takes about half the time of the full size).
    static const uint32_t divisor = uint32_t(std::clamp(envFloat("AC6_SSR_RES", 2.0f), 1.0f, 4.0f));
    const uint32_t w = std::max(1u, image.width / divisor), h = std::max(1u, image.height / divisor);
    EnsureImage(s_r.seaReflection, w, h, image.format, false);
    auto params = s_r.upload.Allocate(192, 16);
    if (!params.memory)
        return;
    float* f = reinterpret_cast<float*>(params.memory);
    for (int i = 0; i < 16; i++)
    {
        f[i] = float(view->vp[i]);
        f[16 + i] = float(inverse[i]);
    }
    // The camera: the point every clip position (0, 0, z, 0) comes from.
    f[32] = float(inverse[8] / inverse[11]);
    f[33] = float(inverse[9] / inverse[11]);
    f[34] = float(inverse[10] / inverse[11]);
    f[35] = sea;
    const auto [jx, jy] = SceneJitter();
    f[36] = 2.0f * jx / float(image.width);
    f[37] = -2.0f * jy / float(image.height);
    f[38] = float(std::fabs(view->vp[14]));
    f[39] = 0.0f;
    f[40] = 0.0f;
    f[41] = stride * std::max(1.0f, float(image.height) / 1080.0f);
    f[42] = debug;
    f[43] = waves;
    // (The 8-bit scene's values above 240 / 255 are the game's emitter codes: a reflection never holds one.)
    f[44] = image.format == RenderFormat::B8G8R8A8_UNORM ? 240.0f / 255.0f : 64.0f;
    f[45] = float(double(s_r.frame % 1000000) / 60.0);
    f[46] = thickness;
    f[47] = 0.0f;
    // AC6_SSR_LOG=1: the camera every 30 frames (its height above the sea and its pitch: the view's slope).
    static const bool trace = [] { const char* v = std::getenv("AC6_SSR_LOG"); return v && *v == '1'; }();
    if (trace && s_r.frame % 30 == 0)
    {
        // The view's direction: the unprojection of the screen's centre, far from the camera.
        const double* I = inverse;
        const double ahead[4] = { 1e-6 * I[8] + I[12], 1e-6 * I[9] + I[13], 1e-6 * I[10] + I[14], 1e-6 * I[11] + I[15] };
        const double d[3] = { ahead[0] / ahead[3] - f[32], ahead[1] / ahead[3] - f[33], ahead[2] / ahead[3] - f[34] };
        const double length = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        log::Info("Sea reflection frame {}: camera height {:.0f}, pitch {:.1f} degrees", s_r.frame, f[33] - sea,
            length > 0.0 ? std::asin(d[1] / length) * 57.29578 : 0.0);
    }
    static uint32_t logged = 0;
    if (logged++ < 3)
        log::Info("Sea reflection frame {}: {}x{} from the scene {}x{} format {}, depth {}x{}, camera {:.1f} {:.1f} {:.1f}, clip z {:.3f}, jitter {:.3f} {:.3f}",
            s_r.frame, w, h, image.width, image.height, int(image.format), depth.width, depth.height, f[32], f[33], f[34], f[38], jx, jy);
    MarkPass([] { return std::string("sea reflection"); });
    cl.setFramebuffer(nullptr);
    s_r.markedFramebuffer = nullptr;
    Transition(cl, image, RenderTextureLayout::SHADER_READ);
    Transition(cl, depth, RenderTextureLayout::SHADER_READ);
    Transition(cl, s_r.seaReflection, RenderTextureLayout::COLOR_WRITE);
    cl.setFramebuffer(GetFramebuffer(&s_r.seaReflection, nullptr));
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(w), float(h)));
    SetScissor(cl, RenderRect(0, 0, int32_t(w), int32_t(h)));
    SetPipeline(cl, pipeline.get());
    BindCommon(cl);
    const uint32_t push[4] = { image.descriptor, depth.descriptor, uint32_t(params.deviceAddress), uint32_t(params.deviceAddress >> 32) };
    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    cl.drawInstanced(3, 1, 0, 0);
    cl.setFramebuffer(nullptr);
    Transition(cl, s_r.seaReflection, RenderTextureLayout::SHADER_READ);
    s_r.seaReflectionFrame = s_r.frame;
    s_r.seaReflectionSurface = surface;
    std::memcpy(s_r.seaReflectionVp, view->vp, sizeof(s_r.seaReflectionVp));
    // (The image is made on the steady picture: no jitter of its own.)
    s_r.seaReflectionJitter[0] = s_r.seaReflectionJitter[1] = 0.0f;
}

// The sea's planar reflection (AC6_SSR=planar): the scene's draws noted in Draw are issued again with the world
// mirrored in the sea's plane, into a colour and depth of their own at a fraction of the scene's size, then blurred into
// the image the sea's shaders read in the next frame (SeaReflectionAtDraw, as the screen-space one's).
// The mirror is a change of each draw's position matrix M (row vectors, clip = p x M): with VP the frame's camera,
// M x inverse(VP) x R x VP draws the object mirrored by R, and a point of the sea's plane lands where the camera sees it,
// so the image lines up with the sea as the screen-space one does. What lies under the sea's plane is clipped by the
// vertex shaders (SPEC_CONSTANT_PLANAR_CLIP: the clip position against the plane, in constant 255 of the draw's own
// copy of the constants), and the depth stays the game's.
// What moves with the camera (the sky's dome) is mirrored in the camera's own height instead: it stands for what is
// infinitely far. The coverage: 1 where something wrote depth, elsewhere what the sky's layers (the draws without depth
// writes) cover, from those draws issued a second time over white. Where nothing covers, the sea keeps its own sky cube.
// AC6_PLANAR_RES=<divisor> (2: half the scene's size on each axis), AC6_PLANAR_BLUR=<sigma in its pixels> (0.75, at the
// water's line: the blur grows with the height above the water, AC6_PLANAR_ROUGHNESS and AC6_PLANAR_BLUR_MAX below),
// AC6_PLANAR_WAVES=<slope> (0.01: the waves' distortion), AC6_PLANAR_CLIP=<metres above the sea> (1: the sea itself and
// the flat ground at its height clipped), AC6_PLANAR_SKY=0 (the sky's layers not taken: the sea's sky cube there),
// AC6_PLANAR_DEBUG=1 (the reflection alone on the sea, magenta where nothing covers), AC6_PLANAR_EFFECTS=0 (not the
// effects' image's draws), AC6_PLANAR_STRETCH and AC6_PLANAR_SPECKS (the blur), AC6_PLANAR_GLOW and
// AC6_PLANAR_GLOW_BLUR (the lights' bloom, below). AC6_DUMP_RESOLVES also writes
// the mirrored scene and the sea's image.
void RunPlanarReflection(RenderCommandList& cl, HostImage& image, const HostImage* surface)
{
    struct Clear { ~Clear() { s_planarDraws.clear(); } } clear;
    if (!s_r.planarBlurPs || s_r.planarNotesFrame != s_r.frame || s_planarDraws.empty() || s_r.seaViewsFrame != s_r.frame ||
        (image.format != RenderFormat::B8G8R8A8_UNORM && image.format != RenderFormat::R16G16B16A16_FLOAT))
        return;
    const auto view = std::find_if(s_r.seaViews.begin(), s_r.seaViews.end(), [&](const Renderer::SeaView& v) { return v.target == surface; });
    if (view == s_r.seaViews.end())
        return;
    auto envFloat = [](const char* name, float fallback) { const char* v = std::getenv(name); return v && *v ? float(std::atof(v)) : fallback; };
    static const float sigma = envFloat("AC6_PLANAR_BLUR", 0.75f), waves = envFloat("AC6_PLANAR_WAVES", 0.01f);
    static const double sea = envFloat("AC6_SSR_SEA", 0.0f), lift = envFloat("AC6_PLANAR_CLIP", 1.0f);
    const double* vp = view->vp;
    double inverse[16];
    if (!Invert4(vp, inverse) || std::fabs(inverse[11]) < 1e-12)
        return;
    const double camera[3] = { inverse[8] / inverse[11], inverse[9] / inverse[11], inverse[10] / inverse[11] };
    if (camera[1] <= sea + lift)
        return;   // the camera at or under the sea
    auto multiply = [](const double* a, const double* b, double* out) {
        double r[16];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                r[i * 4 + j] = a[i * 4] * b[j] + a[i * 4 + 1] * b[4 + j] + a[i * 4 + 2] * b[8 + j] + a[i * 4 + 3] * b[12 + j];
        std::memcpy(out, r, sizeof(r));
    };
    // R: y -> 2 x sea - y. K = inverse(VP) x R x VP.
    const double mirror[16] = { 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 2.0 * sea, 0, 1 };
    double k[16];
    multiply(inverse, mirror, k);
    multiply(k, vp, k);
    // The kept side, in the mirrored picture's clip space: d = clip . c > 0 for what stands above the plane (sea + lift)
    // in the world (c = inverse(VP) x R x (0, 1, 0, -(sea + lift))).
    const double rn[4] = { 0.0, -1.0, 0.0, 2.0 * sea - (sea + lift) };
    double c[4];
    for (int r = 0; r < 4; r++)
        c[r] = inverse[r * 4] * rn[0] + inverse[r * 4 + 1] * rn[1] + inverse[r * 4 + 2] * rn[2] + inverse[r * 4 + 3] * rn[3];
    // (Nothing above the plane in view: d / w at its largest, at infinity, NDC x and y in the frustum and z 0, not
    // positive.)
    if (!(std::fabs(c[0]) + std::fabs(c[1]) + c[3] > 1e-12))
        return;
    const double mirrorSky[16] = { 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 2.0 * camera[1], 0, 1 };
    double kSky[16];
    multiply(inverse, mirrorSky, kSky);
    multiply(kSky, vp, kSky);
    uint32_t issued = 0, skipped = 0, skies = 0;
    for (PlanarDraw& note : s_planarDraws)
    {
        // Only a world transform before the frame's camera (M x inverse(VP) affine): not the cockpit's own projection.
        double affine[16];
        multiply(note.m, inverse, affine);
        if (std::fabs(affine[3]) + std::fabs(affine[7]) + std::fabs(affine[11]) > 1e-3 || std::fabs(affine[15] - 1.0) > 1e-2)
        {
            note.draw.pipeline = nullptr;
            skipped++;
            continue;
        }
        {
            static const uint64_t traceFrame = [] { const char* v = std::getenv("AC6_PLANAR_TRACE"); return v ? std::stoull(v) : UINT64_MAX; }();
            if (s_r.frame == traceFrame)
                log::Info("PN draw at {:.1f} {:.1f} {:.1f} (camera {:.1f} {:.1f} {:.1f}), scale {:.3f}", affine[12], affine[13], affine[14],
                    camera[0], camera[1], camera[2], std::sqrt(affine[0] * affine[0] + affine[1] * affine[1] + affine[2] * affine[2]));
        }
        double t[16];
        // AC6_PLANAR_IDENTITY=1 (debugging): the draws issued again as they were, unmirrored.
        static const bool identity = [] { const char* v = std::getenv("AC6_PLANAR_IDENTITY"); return v && *v == '1'; }();
        if (identity)
        {
            for (int i = 0; i < 16; i++)
                note.matrix[i] = float(note.m[i]);
            issued++;
            continue;
        }
        // What moves with the camera (the sky's dome, its world placed at the camera's x and z) stands for what is
        // infinitely far: mirrored in the camera's own height, so the reflection shows the sky as far above the
        // horizon as it is below, and kept at its own depth.
        const bool sky = std::fabs(affine[12] - camera[0]) < 1.0 && std::fabs(affine[14] - camera[2]) < 1.0;
        // The clip: the vertex shader keeps what lies above the sea's plane (its clip position . the plane, kept where
        // positive). The depth stays the game's: shaders that size their sprites by it (the lights') see what they
        // expect. The sky's dome isn't clipped (the plane then w: always positive).
        if (sky)
        {
            multiply(note.m, kSky, t);
            skies++;
        }
        else
            multiply(note.m, k, t);
        if (note.plane)
            for (int i = 0; i < 4; i++)
                note.plane[i] = sky ? (i == 3 ? 1.0f : 0.0f) : float(c[i]);
        for (int i = 0; i < 16; i++)
            note.matrix[i] = float(t[i]);
        // The other camera matrices, mirrored to match, else a sprite built from them lands where the real camera
        // would put it (the lights' sprites unproject their corners through the inverse of the view-projection, and
        // face the camera along its right and up). The inverse: of the new transform, when the game's was the inverse
        // of its. The view: the mirror before it. The camera's right and up: mirrored (y negated), in the world or,
        // for WorldInverseViewMatrix, in the object's space (through the object's 3 x 3).
        const double* r = sky ? mirrorSky : mirror;
        if (note.inverseVp)
        {
            double product[16];
            multiply(note.m, note.inverseVpValue, product);
            double off = 0.0;
            for (int i = 0; i < 16; i++)
                off = std::max(off, std::fabs(product[i] - (i % 5 == 0 ? 1.0 : 0.0)));
            double back[16];
            static const uint64_t traceInverse = [] { const char* v = std::getenv("AC6_PLANAR_TRACE"); return v ? std::stoull(v) : UINT64_MAX; }();
            if (s_r.frame == traceInverse)
                log::Info("PI inverse check {:.4g}: M row0 {:.4g} {:.4g} {:.4g} {:.4g}, inverse row0 {:.4g} {:.4g} {:.4g} {:.4g}", off, note.m[0], note.m[1],
                    note.m[2], note.m[3], note.inverseVpValue[0], note.inverseVpValue[1], note.inverseVpValue[2], note.inverseVpValue[3]);
            if (off < 1e-2 && Invert4(t, back))
                for (int i = 0; i < 16; i++)
                    note.inverseVp[i] = float(back[i]);
        }
        if (note.view)
        {
            double v[16];
            multiply(r, note.viewValue, v);
            for (int i = 0; i < 16; i++)
                note.view[i] = float(v[i]);
        }
        if (note.inverseView)
            for (int row = 0; row < 2; row++)
                for (int i = 0; i < 4; i++)
                    note.inverseView[row * 4 + i] = float(i == 1 ? -note.inverseViewValue[row * 4 + i] : note.inverseViewValue[row * 4 + i]);
        if (note.worldInverseView)
        {
            // (Object -> world: the affine part of M x inverse(VP), its 3 x 3.)
            const double w3[9] = { affine[0], affine[1], affine[2], affine[4], affine[5], affine[6], affine[8], affine[9], affine[10] };
            const double det = w3[0] * (w3[4] * w3[8] - w3[5] * w3[7]) - w3[1] * (w3[3] * w3[8] - w3[5] * w3[6]) + w3[2] * (w3[3] * w3[7] - w3[4] * w3[6]);
            if (std::fabs(det) > 1e-12)
            {
                const double inv3[9] = { (w3[4] * w3[8] - w3[5] * w3[7]) / det, (w3[2] * w3[7] - w3[1] * w3[8]) / det, (w3[1] * w3[5] - w3[2] * w3[4]) / det,
                    (w3[5] * w3[6] - w3[3] * w3[8]) / det, (w3[0] * w3[8] - w3[2] * w3[6]) / det, (w3[2] * w3[3] - w3[0] * w3[5]) / det,
                    (w3[3] * w3[7] - w3[4] * w3[6]) / det, (w3[1] * w3[6] - w3[0] * w3[7]) / det, (w3[0] * w3[4] - w3[1] * w3[3]) / det };
                for (int row = 0; row < 2; row++)
                {
                    const double* o = note.worldInverseViewValue + row * 4;
                    double world[3];
                    for (int j = 0; j < 3; j++)
                        world[j] = o[0] * w3[j] + o[1] * w3[3 + j] + o[2] * w3[6 + j];
                    world[1] = -world[1];
                    for (int j = 0; j < 3; j++)
                        note.worldInverseView[row * 4 + j] = float(world[0] * inv3[j] + world[1] * inv3[3 + j] + world[2] * inv3[6 + j]);
                    note.worldInverseView[row * 4 + 3] = float(o[3]);
                }
            }
        }
        issued++;
    }
    if (!issued)
        return;
    auto pipelineFor = [&](RenderFormat format) -> RenderPipeline* {
        auto& pipeline = s_r.planarBlurPipelines[uint32_t(format)];
        if (!pipeline)
        {
            RenderGraphicsPipelineDesc desc;
            desc.pipelineLayout = s_r.layout.get();
            desc.vertexShader = s_r.copyVs.get();
            desc.pixelShader = s_r.planarBlurPs.get();
            desc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
            desc.cullMode = RenderCullMode::NONE;
            desc.renderTargetFormat[0] = format;
            desc.renderTargetBlend[0] = RenderBlendDesc::Copy();
            desc.renderTargetCount = 1;
            pipeline = g_video.device->createGraphicsPipeline(desc);
        }
        return pipeline.get();
    };
    // (Every pass in float: the sea's image holds the lights' brightness above white too.)
    RenderPipeline* blur = pipelineFor(RenderFormat::R16G16B16A16_FLOAT);
    if (!blur)
        return;
    HostImage& colour = s_r.planarColor;
    HostImage& depth = s_r.planarDepth;
    const uint32_t w = colour.width, h = colour.height;
    EnsureImage(s_r.planarBlur, w, h, RenderFormat::R16G16B16A16_FLOAT, false, 1, false, true);
    // The glow at half the size (AC6_PLANAR_GLOW_RES=1: the full size): a wide, soft blur.
    static const uint32_t glowStep = envFloat("AC6_PLANAR_GLOW_RES", 2.0f) >= 1.5f ? 2u : 1u;
    const uint32_t gw = (w + glowStep - 1) / glowStep, gh = (h + glowStep - 1) / glowStep;
    EnsureImage(s_r.planarGlowAcross, gw, gh, RenderFormat::R16G16B16A16_FLOAT, false, 1, false, true);
    EnsureImage(s_r.planarGlow, gw, gh, RenderFormat::R16G16B16A16_FLOAT, false, 1, false, true);
    EnsureImage(s_r.planarLights, w, h, RenderFormat::R16G16B16A16_FLOAT, false, 1, false, true);
    EnsureImage(s_r.seaReflection, w, h, RenderFormat::R16G16B16A16_FLOAT, false, 1, false, true);

    // The mirrored scene.
    MarkPass([] { return std::string("planar reflection: the mirrored scene"); });
    cl.setFramebuffer(nullptr);
    Transition(cl, colour, RenderTextureLayout::COLOR_WRITE);
    Transition(cl, depth, RenderTextureLayout::DEPTH_WRITE);
    cl.setFramebuffer(GetFramebuffer(&colour, &depth));
    s_r.markedFramebuffer = nullptr;
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(w), float(h)));
    SetScissor(cl, RenderRect(0, 0, int32_t(w), int32_t(h)));
    cl.clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
    cl.clearDepthStencil(true, true, 0.0f, 0);
    colour.fresh = depth.fresh = false;
    BindCommon(cl);
    auto issue = [&](bool layersOnly) {
    for (const PlanarDraw& note : s_planarDraws)
    {
        const AoTerrainDraw& d = note.draw;
        if (!d.pipeline || (layersOnly && note.depthWrite))
            continue;
        if (breadcrumbs::Detailed())
        {
            breadcrumbs::NoteDraw(cl, s_r.frame, 0, 0);
            breadcrumbs::Detail(std::format("planar reflection: a scene draw again, {} indices, {} vertex streams, depth write {}",
                d.indexCount, d.viewCount, note.depthWrite));
        }
        const float sx = float(w) / float(d.width), sy = float(h) / float(d.height);
        SetViewport(cl, RenderViewport(note.viewport.x * sx, note.viewport.y * sy, note.viewport.width * sx, note.viewport.height * sy,
            note.viewport.minDepth, note.viewport.maxDepth));
        SetScissor(cl, RenderRect(0, 0, int32_t(w), int32_t(h)));
        SetPipeline(cl, d.pipeline);
        if (d.blendConstant)
            cl.setBlendConstants(d.blend);
        if (UniformConstants())
            cl.setGraphicsDescriptorSetDynamic(s_r.constantSet.get(), 4, d.constantOffsets, 3);
        else
            cl.setGraphicsPushConstants(0, d.addresses, 0, sizeof(d.addresses));
        for (uint32_t i = 0; i < d.viewCount; i++)
            cl.setVertexBuffers(d.layout->slots[i].index, &d.views[i], 1, &d.layout->slots[i]);
        if (d.indexCount)
        {
            RenderIndexBufferView ib(RenderBufferReference(s_r.upload.buffer.get(), d.indexOffset), d.indexCount * 4, RenderFormat::R32_UINT);
            cl.setIndexBuffer(&ib);
            cl.drawIndexedInstanced(d.indexCount, 1, 0, 0, 0);
        }
        else
        {
            cl.drawInstanced(d.drawCount, 1, 0, 0);
        }
    }
    };
    issue(false);
    cl.setFramebuffer(nullptr);
    s_r.markedFramebuffer = nullptr;
    // What the sky's layers let through: the draws that write no depth (the sky's layers, the clouds, glows) again,
    // over white, against the mirrored scene's depth. Their blending is linear in what is under them, so where
    // nothing wrote depth the scene's colour is C (over black) and this is C + T (over white): T is what shows
    // through them, the sea's own sky cube there (coverage 1 - T).
    static const bool skyLayers = [] { const char* v = std::getenv("AC6_PLANAR_SKY"); return !(v && *v == '0'); }();
    if (skyLayers)
    {
        EnsureImage(s_r.planarSky, w, h, RenderFormat::R16G16B16A16_FLOAT, false, 1, false, true);
        MarkPass([] { return std::string("planar reflection: the sky's layers over white"); });
        Transition(cl, s_r.planarSky, RenderTextureLayout::COLOR_WRITE);
        Transition(cl, depth, RenderTextureLayout::DEPTH_WRITE);
        cl.setFramebuffer(GetFramebuffer(&s_r.planarSky, &depth));
        s_r.markedFramebuffer = nullptr;
        SetViewport(cl, RenderViewport(0.0f, 0.0f, float(w), float(h)));
        SetScissor(cl, RenderRect(0, 0, int32_t(w), int32_t(h)));
        cl.clearColor(0, RenderColor(1.0f, 1.0f, 1.0f, 1.0f));
        s_r.planarSky.fresh = false;
        BindCommon(cl);
        issue(true);
        cl.setFramebuffer(nullptr);
        s_r.markedFramebuffer = nullptr;
    }
    s_bound = BoundState{ g_video.recordingEpoch };

    // The blur: across from the mirrored scene (its coverage from the depth), then down into the sea's image with the
    // waves' distortion. The glow: what is brighter than white (the lights, the game's emitter codes decoded) blurred
    // wider, across then down, and added to the sea's image: the reflected lights' bloom.
    constexpr uint32_t kParamBytes = 320;
    auto params = s_r.upload.Allocate(kParamBytes, 16);
    if (!params.memory)
        return;
    float* f = reinterpret_cast<float*>(params.memory);
    for (int i = 0; i < 16; i++)
    {
        f[i] = float(vp[i]);
        f[16 + i] = float(inverse[i]);
    }
    f[32] = float(camera[0]);
    f[33] = float(camera[1]);
    f[34] = float(camera[2]);
    f[35] = float(sea);
    f[36] = 0.0f;   // the pass: across
    f[37] = sigma;
    f[38] = 1.0f;   // (the blurred colour's ceiling: above white is the glow's)
    f[39] = waves;
    f[40] = float(double(s_r.frame % 1000000) / 60.0);
    static const float debug = envFloat("AC6_PLANAR_DEBUG", 0.0f);
    f[41] = debug;
    // AC6_PLANAR_SKY=0: the mirrored sky's layers and clouds not taken (the sea keeps the game's sky cube there).
    f[42] = skyLayers ? 1.0f : 0.0f;
    f[43] = std::bit_cast<float>(skyLayers ? s_r.planarSky.descriptor : 0u);
    // AC6_PLANAR_STRETCH=<factor>: the downward blur this many times the across one (rippled water stretches
    // reflections into streaks). AC6_PLANAR_SPECKS=<weight>: how much less a bright texel weighs (0: plain average).
    static const float stretch = envFloat("AC6_PLANAR_STRETCH", 2.5f), specks = envFloat("AC6_PLANAR_SPECKS", 0.25f);
    f[44] = stretch;
    f[45] = specks;
    // The scene's values: the 8-bit scene's codes (decoded: above 240 / 255 they stand for 1.1-100), or the float
    // scene's (x 255 / 240). The glow: what the game's bright pass takes of them, with its exposure, threshold and offset
    // (the bloom the reflected lights would have), back in the scene's values.
    f[46] = image.format == RenderFormat::R16G16B16A16_FLOAT ? 1.0f : 0.0f;
    // AC6_PLANAR_GLOW=<strength> (2, 0: none), AC6_PLANAR_GLOW_BLUR=<sigma in its pixels> (4, x the stretch downward).
    static const float glow = envFloat("AC6_PLANAR_GLOW", 2.0f), glowSigma = envFloat("AC6_PLANAR_GLOW_BLUR", 4.0f);
    f[47] = glow;
    f[48] = glowSigma;
    f[49] = s_r.planarBand[0];
    f[50] = s_r.planarBand[1];
    f[51] = s_r.bloomBright[0];
    f[52] = s_r.bloomBright[1];
    f[53] = std::bit_cast<float>(s_r.planarLuminance);
    f[54] = 0.0f;
    f[55] = float(glowStep);   // (the mirrored scene's pixels to one of the glow's)
    // The blur by height (AC6_PLANAR_ROUGHNESS=<the water's slope>, 0.04, 0: AC6_PLANAR_BLUR everywhere, and AC6_PLANAR_BLUR is
    // then the blur at the water's line, AC6_PLANAR_BLUR_MAX the most, 6): the mirrored scene's depth, and the way back
    // from its clip space to the world (the mirrored camera with the oblique near plane, inverted).
    static const float roughness = envFloat("AC6_PLANAR_ROUGHNESS", 0.04f), blurMost = envFloat("AC6_PLANAR_BLUR_MAX", 6.0f);
    f[56] = std::bit_cast<float>(depth.descriptor);
    f[57] = roughness;
    f[58] = std::max(blurMost, sigma);
    f[59] = 0.0f;
    {
        double world[16], back[16];
        multiply(vp, k, world);
        if (!Invert4(world, back))
            f[57] = 0.0f;
        for (int i = 0; i < 16; i++)
            f[60 + i] = float(back[i]);
    }
    // The lights' streaks (AC6_PLANAR_LIGHTS=<strength>, 0: none, the default, and AC6_PLANAR_STREAK=<length in its pixels>, 10): lamps
    // and glints, too small and too dim for the blur and the bloom, kept at their brightness down the water. Off by
    // default: at this size a city's lit windows pass the test in some frames and not in others, so their streaks flicker.
    static const float lights = envFloat("AC6_PLANAR_LIGHTS", 0.0f), streakLength = envFloat("AC6_PLANAR_STREAK", 10.0f);
    f[76] = lights;
    f[77] = streakLength;
    f[78] = std::bit_cast<float>(lights > 0.0f ? s_r.planarLights.descriptor : 0u);
    f[79] = 0.0f;
    auto paramsFor = [&](float mode) {
        auto a = s_r.upload.Allocate(kParamBytes, 16);
        if (a.memory)
        {
            std::memcpy(a.memory, params.memory, kParamBytes);
            reinterpret_cast<float*>(a.memory)[36] = mode;
        }
        return a;
    };
    auto paramsDown = paramsFor(1.0f), paramsGlowAcross = paramsFor(2.0f), paramsGlowDown = paramsFor(3.0f), paramsLights = paramsFor(4.0f);
    if (!paramsDown.memory || !paramsGlowAcross.memory || !paramsGlowDown.memory || !paramsLights.memory)
        return;
    auto pass = [&](RenderPipeline* pipeline, HostImage& target, uint32_t texture, uint32_t second, uint64_t address) {
        Transition(cl, target, RenderTextureLayout::COLOR_WRITE);
        cl.setFramebuffer(GetFramebuffer(&target, nullptr));
        s_r.markedFramebuffer = nullptr;
        SetViewport(cl, RenderViewport(0.0f, 0.0f, float(target.width), float(target.height)));
        SetScissor(cl, RenderRect(0, 0, int32_t(target.width), int32_t(target.height)));
        SetPipeline(cl, pipeline);
        BindCommon(cl);
        const uint32_t push[4] = { texture, second, uint32_t(address), uint32_t(address >> 32) };
        cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
        cl.drawInstanced(3, 1, 0, 0);
        cl.setFramebuffer(nullptr);
        s_r.markedFramebuffer = nullptr;
    };
    MarkPass([] { return std::string("planar reflection: blur"); });
    Transition(cl, colour, RenderTextureLayout::SHADER_READ);
    Transition(cl, depth, RenderTextureLayout::SHADER_READ);
    if (skyLayers)
        Transition(cl, s_r.planarSky, RenderTextureLayout::SHADER_READ);
    pass(blur, s_r.planarBlur, colour.descriptor, depth.descriptor, params.deviceAddress);
    Transition(cl, s_r.planarBlur, RenderTextureLayout::SHADER_READ);
    uint32_t glowImage = 0;
    if (glow > 0.0f || lights > 0.0f)
    {
        MarkPass([] { return std::string("planar reflection: glow and lights"); });
        if (lights > 0.0f)
        {
            pass(blur, s_r.planarLights, colour.descriptor, 0, paramsLights.deviceAddress);
            Transition(cl, s_r.planarLights, RenderTextureLayout::SHADER_READ);
        }
        pass(blur, s_r.planarGlowAcross, colour.descriptor, 0, paramsGlowAcross.deviceAddress);
        Transition(cl, s_r.planarGlowAcross, RenderTextureLayout::SHADER_READ);
        pass(blur, s_r.planarGlow, s_r.planarGlowAcross.descriptor, 0, paramsGlowDown.deviceAddress);
        Transition(cl, s_r.planarGlow, RenderTextureLayout::SHADER_READ);
        glowImage = s_r.planarGlow.descriptor;
    }
    pass(blur, s_r.seaReflection, s_r.planarBlur.descriptor, glowImage, paramsDown.deviceAddress);
    Transition(cl, s_r.seaReflection, RenderTextureLayout::SHADER_READ);
    s_bound = BoundState{ g_video.recordingEpoch };
    static uint32_t logged = 0;
    static const bool trace = [] { const char* v = std::getenv("AC6_SSR_LOG"); return v && *v == '1'; }();
    if (logged++ < 3 || (trace && s_r.frame % 60 == 0))
        log::Info("Planar reflection frame {}: {} draws mirrored ({} with the camera: the sky, {} skipped: not the world's camera) into {}x{}, camera height {:.0f}",
            s_r.frame, issued, skies, skipped, w, h, camera[1] - sea);
    s_r.seaReflectionFrame = s_r.frame;
    s_r.seaReflectionSurface = surface;
    std::memcpy(s_r.seaReflectionVp, vp, sizeof(s_r.seaReflectionVp));
    s_r.seaReflectionJitter[0] = s_r.seaReflectionJitter[1] = 0.0f;
}

void RunTaa(RenderCommandList& cl, HostImage& image)
{
    // (HDR from the scene's float image too: the frame's copy and the sharpened one in its format.)
    const bool floatImage = image.format == RenderFormat::R16G16B16A16_FLOAT && s_r.taaPipelineFloat;
    if (!TaaOn() || !image.texture || (image.format != RenderFormat::B8G8R8A8_UNORM && !floatImage) || !s_r.device)
        return;
    auto depthIt = s_r.resolved.find(0x05F83000);
    if (depthIt == s_r.resolved.end() || !depthIt->second.texture)
        return;
    HostImage& depth = depthIt->second;
    double vp[16], r[16];
    if (!SceneReprojection(vp, r))
        return;
    // AC6_TAA_DEBUG=1: the matrices to the log, 2: the motion on screen (red / green, x 40) and the near pixels (blue),
    // 3: the reprojected history's difference from the frame (x 4: a correct reprojection leaves static scenery dark),
    // 4: the depth the reprojection reads, as distance (log: near white, 16 km black), 5: that in odd columns, the frame in
    // even ones (with AC6_PRESENT=05BEB000, for offline checks).
    static const int debug = [] { const char* v = std::getenv("AC6_TAA_DEBUG"); return v && *v ? std::atoi(v) : 0; }();
    if (debug == 1 || debug == 5)
        ac6::log::Info("TAA frame {} vp rows: {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f}",
            s_r.frame, vp[0], vp[1], vp[2], vp[3], vp[4], vp[5], vp[6], vp[7], vp[8], vp[9], vp[10], vp[11], vp[12], vp[13], vp[14], vp[15]);
    const uint32_t w = image.width, h = image.height;
    const bool resized = !s_r.taaHistory[0].texture || s_r.taaHistory[0].width != w || s_r.taaHistory[0].height != h;
    for (HostImage& history : s_r.taaHistory)
        EnsureImage(history, w, h, RenderFormat::R16G16B16A16_FLOAT, false);
    EnsureImage(s_r.taaColor, w, h, image.format, false);
    const auto [jx, jy] = TaaJitter(SeedFrame());
    const bool noHistory = resized || !s_r.taaHavePrev || s_r.frame != s_r.taaLastFrame + 1 || CameraCut(vp);
    const uint32_t motionDescriptor = DrawnMotion(cl);
    auto params = s_r.upload.Allocate(128, 16);
    if (!params.memory)
        return;
    float* f = reinterpret_cast<float*>(params.memory);
    for (int i = 0; i < 16; i++)
        f[i] = float(r[i]);
    f[16] = 2.0f * jx / float(w);
    f[17] = -2.0f * jy / float(h);
    static const float blend = [] { const char* v = std::getenv("AC6_TAA_BLEND"); return v && *v ? float(std::atof(v)) : 0.1f; }();
    f[18] = blend;
    f[19] = noHistory && debug < 4 ? 1.0f : float((debug >= 2 && debug <= 5) || debug == 7 ? debug : 0);
    // AC6_TAA_DEBUG=7: the motion image as written (x 40, magenta: unknown), and its counts to the log.
    if (debug == 7 && s_r.frame % 60 == 0)
        log::Info("TAA motion frame {}: image {} (cleared frame {}), {} draws wrote motion, {} with a transform, {} paired with last frame's",
            s_r.frame, motionDescriptor, s_r.motionClearedFrame, s_r.motionStats[0], s_r.motionStats[1], s_r.motionStats[2]);
    // Nearer than AC6_TAA_NEAR world units (60): moving with the camera (the game's depth is clip z / w with a constant
    // clip z, c119.z: reversed, infinite far plane).
    static const double nearUnits = [] { const char* v = std::getenv("AC6_TAA_NEAR"); return v && *v ? std::atof(v) : 60.0; }();
    f[20] = float(std::fabs(vp[14]) / nearUnits);
    f[21] = float(std::fabs(vp[14]));   // clip z: depth x distance (for the depth view)
    f[22] = f[23] = 0.0f;
    // AC6_TAA_MOTION_BLEND (0.5): the new frame's weight at fast motion, reached at AC6_TAA_MOTION_PX host pixels a
    // frame (24). AC6_TAA_NEAR_BLEND (0.4): the near pixels' weight (the player's own aircraft, which has nothing to
    // reproject it with). AC6_TAA_CLIP (1.25): the variance clip's width in standard deviations.
    auto envFloat = [](const char* name, float fallback) { const char* v = std::getenv(name); return v && *v ? float(std::atof(v)) : fallback; };
    static const float motionBlend = envFloat("AC6_TAA_MOTION_BLEND", 0.5f), motionPx = std::max(1.0f, envFloat("AC6_TAA_MOTION_PX", 24.0f));
    static const float nearBlend = envFloat("AC6_TAA_NEAR_BLEND", 0.4f), clipWidth = envFloat("AC6_TAA_CLIP", 1.25f);
    f[24] = motionBlend;
    f[25] = motionPx;
    f[26] = nearBlend;
    f[27] = clipWidth;
    f[28] = std::bit_cast<float>(motionDescriptor);
    f[29] = f[30] = f[31] = 0.0f;
    HostImage& previous = s_r.taaHistory[s_r.taaHistoryIndex];
    HostImage& next = s_r.taaHistory[s_r.taaHistoryIndex ^ 1];
    MarkPass([] { return std::string("TAA"); });
    Transition(cl, image, RenderTextureLayout::SHADER_READ);
    Transition(cl, depth, RenderTextureLayout::SHADER_READ);
    Transition(cl, previous, RenderTextureLayout::SHADER_READ);
    Transition(cl, s_r.taaColor, RenderTextureLayout::COLOR_WRITE);
    Transition(cl, next, RenderTextureLayout::COLOR_WRITE);
    auto& framebuffer = s_r.depthResolveFramebuffers[{ s_r.taaColor.texture.get(), next.texture.get() }];
    if (!framebuffer)
    {
        const RenderTexture* attachments[2] = { s_r.taaColor.texture.get(), next.texture.get() };
        RenderFramebufferDesc desc;
        desc.colorAttachments = attachments;
        desc.colorAttachmentsCount = 2;
        framebuffer = g_video.device->createFramebuffer(desc);
    }
    cl.setFramebuffer(framebuffer.get());
    s_r.markedFramebuffer = nullptr;
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(w), float(h)));
    SetScissor(cl, RenderRect(0, 0, int32_t(w), int32_t(h)));
    SetPipeline(cl, floatImage ? s_r.taaPipelineFloat.get() : s_r.taaPipeline.get());
    BindCommon(cl);
    const uint32_t push[4] = { image.descriptor, depth.descriptor | (previous.descriptor << 16), uint32_t(params.deviceAddress),
        uint32_t(params.deviceAddress >> 32) };
    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    cl.drawInstanced(3, 1, 0, 0);
    cl.setFramebuffer(nullptr);
    // Sharpening (AC6_TAA_SHARPEN=<0..1>, 0.5, 0 off) on what is shown, the history left as it is.
    static const float sharpen = envFloat("AC6_TAA_SHARPEN", 0.5f);
    HostImage* shown = &s_r.taaColor;
    RenderPipeline* const sharpenPipeline = floatImage ? s_r.sharpenPipelineFloat.get() : s_r.sharpenPipeline.get();
    if (sharpenPipeline && sharpen > 0.0f && debug == 0)
    {
        EnsureImage(s_r.taaSharpened, w, h, image.format, false);
        Transition(cl, s_r.taaColor, RenderTextureLayout::SHADER_READ);
        Transition(cl, s_r.taaSharpened, RenderTextureLayout::COLOR_WRITE);
        auto& sharpenFramebuffer = s_r.depthResolveFramebuffers[{ s_r.taaSharpened.texture.get(), nullptr }];
        if (!sharpenFramebuffer)
        {
            const RenderTexture* attachment = s_r.taaSharpened.texture.get();
            RenderFramebufferDesc desc;
            desc.colorAttachments = &attachment;
            desc.colorAttachmentsCount = 1;
            sharpenFramebuffer = g_video.device->createFramebuffer(desc);
        }
        cl.setFramebuffer(sharpenFramebuffer.get());
        s_r.markedFramebuffer = nullptr;
        SetPipeline(cl, sharpenPipeline);
        BindCommon(cl);
        const uint32_t sharpenPush[4] = { s_r.taaColor.descriptor, 0, std::bit_cast<uint32_t>(sharpen), 0 };
        cl.setGraphicsPushConstants(0, sharpenPush, 0, sizeof(sharpenPush));
        cl.drawInstanced(3, 1, 0, 0);
        cl.setFramebuffer(nullptr);
        shown = &s_r.taaSharpened;
    }
    Transition(cl, *shown, RenderTextureLayout::COPY_SOURCE);
    Transition(cl, image, RenderTextureLayout::COPY_DEST);
    cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(image.texture.get()), RenderTextureCopyLocation::Subresource(shown->texture.get()));
    s_r.taaHistoryIndex ^= 1;
    std::memcpy(s_r.taaPrevVp, vp, sizeof(vp));
    s_r.taaHavePrev = true;
    s_r.taaLastFrame = s_r.frame;
}

// `src` drawn into the whole of `dst`, made `width` x `height` at the frame's scale (bilinear, FSR upscaling's
// fallback).
void StretchInto(RenderCommandList& cl, HostImage& src, HostImage& dst, uint32_t width, uint32_t height)
{
    EnsureImage(dst, width, height, src.format, false, 1, false, true);
    dst.bgraOrder = src.bgraOrder;
    dst.resX = s_r.sceneResX;
    dst.resY = s_r.sceneResY;
    dst.fresh = false;
    cl.setFramebuffer(nullptr);
    Transition(cl, src, RenderTextureLayout::SHADER_READ);
    Transition(cl, dst, RenderTextureLayout::COLOR_WRITE);
    cl.setFramebuffer(GetFramebuffer(&dst, nullptr));
    s_r.markedFramebuffer = nullptr;
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(width), float(height)));
    SetScissor(cl, RenderRect(0, 0, int32_t(width), int32_t(height)));
    // (A float image, HDR from the scene: the copy's float pipeline.)
    SetPipeline(cl, src.format == RenderFormat::R16G16B16A16_FLOAT && s_r.copyHdrPipeline ? s_r.copyHdrPipeline.get() : s_r.copyPipeline.get());
    BindCommon(cl);
    const uint32_t push[2] = { src.descriptor, 0 };   // linear, clamped
    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    cl.drawInstanced(3, 1, 0, 0);
    cl.setFramebuffer(nullptr);
}

// FSR 3.1 in TAA's place: the scene image, its depth and every pixel's motion (the draws' where known,
// the camera's elsewhere: the motion completion pass) to FSR, its result into the image. `input`: with FSR upscaling
// the scene's resolve at the render size, from which FSR makes `image` at the frame's size, null: `image` is both (native
// anti-aliasing). False when FSR couldn't run.
bool RunFsr(RenderCommandList& cl, HostImage& image, HostImage* input = nullptr, HostImage* lateOut = nullptr, bool late = false)
{
    late = late || lateOut;
    HostImage& source = input ? *input : image;
    // (HDR from the scene's float image too: FSR told the colour is high dynamic range, the copy back in float.)
    const bool floatImage = source.format == RenderFormat::R16G16B16A16_FLOAT && s_r.copyHdrPipeline;
    // (Why it didn't run, once a reason.)
    static uint32_t reported = 0;
    auto notRun = [](uint32_t reason, const char* why) {
        if (!(reported & reason))
        {
            reported |= reason;
            log::Info("FSR not run on frame {}: {}", s_r.frame, why);
        }
        return false;
    };
    if (!s_r.fsr || !s_r.motionCompletePipeline || !s_r.device)
        return notRun(1, "no FSR pipelines");
    if (!source.texture || (source.format != RenderFormat::B8G8R8A8_UNORM && !floatImage))
        return notRun(2, "the scene image's format");
    auto depthIt = s_r.resolved.find(0x05F83000);
    if (depthIt == s_r.resolved.end() || !depthIt->second.texture)
        return notRun(4, "no scene depth (0x05F83000) resolved");
    HostImage& depth = depthIt->second;
    double vp[16], r[16];
    if (!SceneReprojection(vp, r))
        return notRun(8, "no camera this frame");
    // The render size (the source's) and the output's (the frame's scale).
    const uint32_t w = source.width, h = source.height;
    // (AC6_FSR_LATE: into `lateOut`, the output-size frame its 2D layers are drawn over.)
    const uint32_t outW = lateOut ? lateOut->width : input ? uint32_t(std::lround(double(GuestWidth(source)) * s_r.sceneResX)) : w;
    const uint32_t outH = lateOut ? lateOut->height : input ? uint32_t(std::lround(double(GuestHeight(source)) * s_r.sceneResY)) : h;
    const bool resized = !s_r.fsrOutput.texture || s_r.fsrOutput.width != outW || s_r.fsrOutput.height != outH ||
        !s_r.fsrMotion.texture || s_r.fsrMotion.width != w || s_r.fsrMotion.height != h;
    EnsureImage(s_r.fsrMotion, w, h, RenderFormat::R16G16_FLOAT, false, 1, false, true);
    EnsureImage(s_r.fsrReactive, w, h, RenderFormat::R8_UNORM, false, 1, false, true);
    if (s_r.fsrMotionFramebufferOf[0] != s_r.fsrMotion.texture.get() || s_r.fsrMotionFramebufferOf[1] != s_r.fsrReactive.texture.get())
    {
        if (s_r.fsrMotionFramebuffer)
            s_r.retiredFramebuffers.push_back(std::move(s_r.fsrMotionFramebuffer));
        s_r.fsrMotionFramebufferOf[0] = s_r.fsrMotion.texture.get();
        s_r.fsrMotionFramebufferOf[1] = s_r.fsrReactive.texture.get();
        s_r.fsrMotionFramebuffer = g_video.device->createFramebuffer(RenderFramebufferDesc(s_r.fsrMotionFramebufferOf, 2));
    }
    EnsureImage(s_r.fsrOutput, outW, outH, RenderFormat::R16G16B16A16_FLOAT, false, 1, true, true);
    // (The jitter the draws had: none on the first frame it runs again.)
    const auto [jx, jy] = FsrJitters() ? TaaJitter(SeedFrame()) : std::pair<float, float>{ 0.0f, 0.0f };
    const bool reset = resized || !s_r.taaHavePrev || s_r.frame != s_r.taaLastFrame + 1 || CameraCut(vp);
    const uint32_t motionDescriptor = DrawnMotion(cl);

    // Every pixel's motion, as FSR wants it.
    auto params = s_r.upload.Allocate(80, 16);
    if (!params.memory)
        return false;
    float* f = reinterpret_cast<float*>(params.memory);
    for (int i = 0; i < 16; i++)
        f[i] = float(r[i]);
    f[16] = 2.0f * jx / float(w);
    f[17] = -2.0f * jy / float(h);
    f[18] = f[19] = 0.0f;
    MarkPass([] { return std::string("FSR motion"); });
    Transition(cl, depth, RenderTextureLayout::SHADER_READ);
    Transition(cl, s_r.fsrMotion, RenderTextureLayout::COLOR_WRITE);
    Transition(cl, s_r.fsrReactive, RenderTextureLayout::COLOR_WRITE);
    cl.setFramebuffer(s_r.fsrMotionFramebuffer.get());
    s_r.markedFramebuffer = nullptr;
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(w), float(h)));
    SetScissor(cl, RenderRect(0, 0, int32_t(w), int32_t(h)));
    SetPipeline(cl, s_r.motionCompletePipeline.get());
    BindCommon(cl);
    const uint32_t push[4] = { motionDescriptor, depth.descriptor, uint32_t(params.deviceAddress), uint32_t(params.deviceAddress >> 32) };
    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    cl.drawInstanced(3, 1, 0, 0);
    cl.setFramebuffer(nullptr);

    // FSR: AC6_FSR_SHARPNESS=<0..1> (0.5, 0 off) the sharpening, AC6_FSR_DEBUG=1 its debug view. The sharpening is done
    // in the copy back (one full-screen pass less than FSR's own RCAS),
    // AC6_FSR_RCAS=1 FSR's RCAS instead.
    MarkPass([] { return std::string("FSR"); });
    Transition(cl, source, RenderTextureLayout::SHADER_READ);
    Transition(cl, s_r.fsrMotion, RenderTextureLayout::SHADER_READ);
    Transition(cl, s_r.fsrReactive, RenderTextureLayout::SHADER_READ);
    Transition(cl, s_r.fsrOutput, RenderTextureLayout::GENERAL);
    auto envFloat = [](const char* name, float fallback) { const char* v = std::getenv(name); return v && *v ? float(std::atof(v)) : fallback; };
    static const float sharpness = std::clamp(envFloat("AC6_FSR_SHARPNESS", 0.5f), 0.0f, 1.0f);
    static const bool rcas = [] { const char* v = std::getenv("AC6_FSR_RCAS"); return v && *v == '1'; }();
    RenderPipeline* const fsrSharpen = floatImage ? s_r.sharpenPipelineFloat.get() : s_r.sharpenPipeline.get();
    const bool fused = !rcas && sharpness > 0.0f && fsrSharpen;
    static const bool debugView = [] { const char* v = std::getenv("AC6_FSR_DEBUG"); return v && *v == '1'; }();
    static auto lastTime = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const float frameMs = std::clamp(std::chrono::duration<float, std::milli>(now - lastTime).count(), 1.0f, 100.0f);
    lastTime = now;
    fsr::Frame frame;
    frame.color = { source.texture.get(), source.format, w, h };
    frame.depth = { depth.texture.get(), depth.format, depth.width, depth.height };
    frame.motion = { s_r.fsrMotion.texture.get(), s_r.fsrMotion.format, w, h };
    // The reactive mask, where the draws wrote one (a four-channel motion image this frame).
    if (motionDescriptor && s_r.motion.format == RenderFormat::R16G16B16A16_FLOAT)
        frame.reactive = { s_r.fsrReactive.texture.get(), s_r.fsrReactive.format, w, h };
    frame.output = { s_r.fsrOutput.texture.get(), s_r.fsrOutput.format, outW, outH };
    // AC6_FSR_LATE: the effects layer reactive, by FSR's own mask from the world pass's resolve (the scene before the
    // effects buffer, the compositor and the post chain) and the finished 3D frame.
    if (late)
        if (auto opaque = s_r.resolved.find(0x05BEB000); opaque != s_r.resolved.end() && opaque->second.texture &&
            opaque->second.width == w && opaque->second.height == h && opaque->second.format == source.format)
        {
            EnsureImage(s_r.fsrAutoReactive, w, h, RenderFormat::R8_UNORM, false, 1, true, true);
            Transition(cl, opaque->second, RenderTextureLayout::SHADER_READ);
            Transition(cl, s_r.fsrAutoReactive, RenderTextureLayout::GENERAL);
            frame.opaqueOnly = { opaque->second.texture.get(), opaque->second.format, w, h };
            frame.autoReactive = { s_r.fsrAutoReactive.texture.get(), s_r.fsrAutoReactive.format, w, h };
        }
    if (late)
    {
        static bool logged = false;
        if (!std::exchange(logged, true))
        {
            auto opaque = s_r.resolved.find(0x05BEB000);
            log::Info("FSR at the end of the 3D frame: {}x{} format {} -> {}x{}, the scene before its effects {}", w, h, int(source.format),
                outW, outH, opaque == s_r.resolved.end() ? std::string("missing") : std::format("{}x{} format {}{}", opaque->second.width,
                opaque->second.height, int(opaque->second.format), frame.autoReactive.texture ? " (FSR's own reactive mask)" : ""));
        }
    }
    frame.jitterX = jx;
    frame.jitterY = jy;
    frame.motionScaleX = float(w);
    frame.motionScaleY = float(h);
    frame.frameTimeMs = frameMs;
    // The game's depth is the near plane's distance over the distance (reversed, infinite far plane: c119.z). The
    // view-projection's y column has the projection's vertical scale (the view's rotation keeps its length).
    frame.cameraNear = float(std::fabs(vp[14]));
    const double scaleY = std::sqrt(vp[1] * vp[1] + vp[5] * vp[5] + vp[9] * vp[9]);
    frame.fovY = scaleY > 0.0 ? float(2.0 * std::atan(1.0 / scaleY)) : 1.0f;
    frame.sharpness = fused ? 0.0f : sharpness;
    frame.reset = reset;
    frame.debugView = debugView;
    DeferredCommandList* deferred = g_video.deferred && &cl == g_video.deferred.get() ? g_video.deferred.get() : nullptr;
    if (!fsr::Dispatch(*g_video.device, cl, deferred, frame))
        return notRun(16, "FSR's dispatch failed");
    s_bound = BoundState{ g_video.recordingEpoch };   // FSR bound its own pipelines and descriptor sets

    // Its result into the scene image (upscaling: made at the output's size, at the frame's scale).
    if (input && !lateOut)
    {
        EnsureImage(image, outW, outH, source.format, false, 1, false, true);
        image.bgraOrder = source.bgraOrder;   // (a float scene's image reads as the 8-bit one's channels)
        image.resX = s_r.sceneResX;
        image.resY = s_r.sceneResY;
        image.fresh = false;
    }
    MarkPass([] { return std::string("FSR copy back"); });
    HostImage& dest = lateOut ? *lateOut : image;
    Transition(cl, s_r.fsrOutput, RenderTextureLayout::SHADER_READ);
    Transition(cl, dest, RenderTextureLayout::COLOR_WRITE);
    cl.setFramebuffer(GetFramebuffer(&dest, nullptr));
    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(outW), float(outH)));
    SetScissor(cl, RenderRect(0, 0, int32_t(outW), int32_t(outH)));
    // (Same size: its samples are the texels.) Sharpened as TAA's output is, x 0.6 to match RCAS at the same setting
    // (by the Laplacian variance of a city scene).
    static const float fusedScale = envFloat("AC6_FSR_SHARPEN_SCALE", 0.6f);
    SetPipeline(cl, fused ? fsrSharpen : floatImage ? s_r.copyHdrPipeline.get() : s_r.copyPipeline.get());
    BindCommon(cl);
    const uint32_t copyPush[4] = { s_r.fsrOutput.descriptor, 0, std::bit_cast<uint32_t>(sharpness * fusedScale), 0 };
    cl.setGraphicsPushConstants(0, copyPush, 0, sizeof(copyPush));
    cl.drawInstanced(3, 1, 0, 0);
    // AC6_FSR_DEBUG=2: the reactive mask instead of the picture (red: reactive, black: not).
    static const bool reactiveView = [] { const char* v = std::getenv("AC6_FSR_DEBUG"); return v && *v == '2'; }();
    if (reactiveView && frame.reactive.texture && !floatImage)
    {
        cl.setFramebuffer(nullptr);
        // (AC6_FSR_LATE: FSR's own mask, the one it used.)
        HostImage& mask = frame.autoReactive.texture ? s_r.fsrAutoReactive : s_r.fsrReactive;
        Transition(cl, mask, RenderTextureLayout::SHADER_READ);
        Transition(cl, dest, RenderTextureLayout::COLOR_WRITE);
        cl.setFramebuffer(GetFramebuffer(&dest, nullptr));
        SetPipeline(cl, s_r.copyPipeline.get());
        BindCommon(cl);
        const uint32_t maskPush[4] = { mask.descriptor, 0, 0, 0 };
        cl.setGraphicsPushConstants(0, maskPush, 0, sizeof(maskPush));
        cl.drawInstanced(3, 1, 0, 0);
    }
    cl.setFramebuffer(nullptr);
    std::memcpy(s_r.taaPrevVp, vp, sizeof(vp));
    s_r.taaHavePrev = true;
    s_r.taaLastFrame = s_r.frame;
    s_r.fsrRanFrame = s_r.frame;
    return true;
}

void ResolveImpl(uint32_t device, uint32_t flags, const ResolveRect* srcRect, uint32_t destTexture, const ResolvePoint* destPoint,
    uint32_t destLevel, const float* clearColor, float clearZ, uint32_t clearStencil)
{
    if (!s_r.ready)
        return;
    s_r.textureEpoch++;
    s_r.device = device;
    RenderCommandList& cl = g_video.Commands();
    EndLatePass(cl);
    const bool depth = (flags & 7) == kResolveDepth;
    if ((flags & 7) > kResolveDepth || ((flags & 7) != 0 && !depth))
    {
        static uint32_t warned = 0;
        if (warned++ < 4)
            log::Warn("Resolve from colour target {} not supported", flags & 7);
        return;
    }
    // FSR upscaling: the scene's resolves at the render scale, into images of that size.
    const uint32_t sourceSurface = U32(device + (depth ? kDevDepthStencil : kDevRenderTargets));
    const bool worldResolve = WorldSurface(sourceSurface);
    ScopedScale worldScale(worldResolve, s_r.worldResX, s_r.worldResY);
    const bool nativeResolve = !worldResolve && NativeSurface(sourceSurface);   // the tree impostors
    ScopedScale nativeScale(nativeResolve, 1.0f, 1.0f);
    HostImage* src = GetSurface(sourceSurface, depth);
    MarkPass([&] { return std::format("resolve {} {}{}", depth ? "depth" : "colour", PassImage(src), destTexture ? "" : " (clear only)"); });
    // A swapped surface resolved again gets its contents back first (a clear alone is left to the
    // clear below).
    if (src && src->staleHolder && destTexture)
        RestoreStale(cl, *src);
    HostImage* const surface = src;
    // The cloudscape's gate, if the effects pass drew nothing after the background cloud sprites.
    if (!depth && src && src == s_r.fxColor && s_r.fxTargetFrame == s_r.frame && s_r.fxGateFrame != s_r.frame && s_r.fxSprites &&
        s_r.cloudEarlyFrame == s_r.frame)
    {
        RunCloudGate(cl, nullptr);
        cl.setFramebuffer(nullptr);
        s_r.markedFramebuffer = nullptr;
    }

    if (src && destTexture && destLevel == 0)
    {
        const auto fetch = xenos::ReadTextureFetch(g_memory.As<uint8_t>(destTexture + kTextureObjectFetch));
        if (fetch.Type() == 2 && fetch.BaseAddress())
        {
            const uint32_t dstW = fetch.Width(), dstH = fetch.Height();
            static const uint64_t traceFrame = [] { const char* v = std::getenv("AC6_DRAW_TRACE"); return v ? std::stoull(v) : UINT64_MAX; }();
            if (s_r.frame == traceFrame)
                log::Info("Resolve: {} into {:08X} ({}x{}, format {}, flags {:X}) from {:08X} {}x{} samples {} scale {} rect {} {} {} {} dest {} {}", depth ? "depth" : "colour", fetch.BaseAddress(), dstW, dstH,
                    fetch.Format(), flags, U32(device + (depth ? kDevDepthStencil : kDevRenderTargets)), src->width, src->height, src->samples, src->scale,
                    srcRect ? srcRect->x1 : -1, srcRect ? srcRect->y1 : -1, srcRect ? srcRect->x2 : -1, srcRect ? srcRect->y2 : -1,
                    destPoint ? destPoint->x : -1, destPoint ? destPoint->y : -1);
            // FSR upscaling: the scene's colour resolve into FSR's render-size input. FSR fills the
            // target, at the frame's size, from it.
            const bool sceneColor = !depth && (fetch.BaseAddress() & 0x1FFFFFFF) == 0x05BEB000;
            // The cloudscape into the scene before it is resolved, once a frame (the scene is resolved in two halves).
            if (sceneColor && s_r.volCloudVs)
            {
                s_r.cloudSceneSurface = src;
                RunCloudscape(cl, device, src);
            }
            // HDR from the scene: the float canvas's resolves into the target's float twin.
            const uint32_t address = fetch.BaseAddress() & 0x1FFFFFFF;
            const bool hdrDst = !depth && src->bgraOrder && src->format == RenderFormat::R16G16B16A16_FLOAT && !(worldResolve && sceneColor);
            if (hdrDst || !s_r.hdrResolved.empty())
                s_r.hdrResolvedCurrent[address] = hdrDst;
            HostImage& dst = worldResolve && sceneColor ? s_r.fsrColor : hdrDst ? s_r.hdrResolved[address] : s_r.resolved[address];
            // A surface whose contents this target's image holds gets them back before it changes.
            if (dst.staleOf)
                RestoreStale(cl, *dst.staleOf);
            const RenderFormat format = depth ? RenderFormat::R32_FLOAT : src->format;
            // At the internal resolution: dstW x dstH guest pixels.
            const uint32_t hostW = uint32_t(ToHostX(dstW)), hostH = uint32_t(ToHostY(dstH));
            if (!dst.texture || dst.format != format || dst.width < hostW || dst.height < hostH)
                log::Info("Resolve target {:08X}: {}x{} format {} from {} (frame {})", fetch.BaseAddress(), dstW, dstH, int(format), depth ? "depth" : "colour", s_r.frame);
            EnsureImage(dst, hostW, hostH, format, false, 1, false, worldResolve);
            dst.bgraOrder = !depth && src->bgraOrder;   // the HDR canvas's resolves
            if (s_r.upscaling || nativeResolve)
            {
                dst.resX = s_r.resX;   // the scale its contents are at (sampled at it)
                dst.resY = s_r.resY;
            }
            if (dst.fresh)
            {
                // A new image's contents are undefined (zeros on some drivers, garbage on others). The
                // parts this resolve doesn't cover must read as cleared memory.
                Transition(cl, dst, RenderTextureLayout::COLOR_WRITE);
                cl.setFramebuffer(GetFramebuffer(&dst, nullptr));
                cl.clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
                dst.fresh = false;
            }

            // Source rectangle in guest pixels (a promoted 4x MSAA source has `scale` host pixels per guest pixel).
            if (!depth)
                SyncAliasedColor(cl, *src, false);
            // A multisampled surface resolves through its single-sampled copy: the samples averaged, as the
            // console's default (all fragments). Depth and stencil from sample 0, which depth resolves select.
            if (src->samples > 1 && !depth && ((flags >> 4) & 7) != 0)
            {
                static uint32_t warned = 0;
                if (warned++ < 4)
                    log::Warn("Colour resolve of fragment selection {} from a multisampled surface: averaged instead", (flags >> 4) & 7);
            }
            if (src->swizzleTrace)
                dst.swizzleTrace = true;
            // A multisampled depth surface whose single-sampled copy is out of date resolves from its samples in one
            // pass, depth and stencil: making the copy first costs 0.3 ms a frame at 4K. Not while
            // resolve statistics read the source back.
            HostImage* const msDepth = depth && src->samples > 1 && src->msDepthDescriptor && !src->fresh && !src->singleCurrent &&
                    s_r.frame - s_r.resolveStatsFrame >= s_r.resolveStatsFrames ? src : nullptr;
            if (!msDepth)
                src = &Readable(cl, *src);
            const int32_t scale = int32_t(src->scale);
            const int32_t srcW = int32_t(GuestWidth(*src)), srcH = int32_t(GuestHeight(*src));
            int32_t sx0 = 0, sy0 = 0, sx1 = srcW, sy1 = srcH;
            if (srcRect)
            {
                sx0 = srcRect->x1; sy0 = srcRect->y1; sx1 = srcRect->x2; sy1 = srcRect->y2;
            }
            const int32_t dx = destPoint ? destPoint->x : 0, dy = destPoint ? destPoint->y : 0;
            const int32_t w = std::min({ sx1 - sx0, srcW - sx0, int32_t(dstW) - dx });
            const int32_t h = std::min({ sy1 - sy0, srcH - sy0, int32_t(dstH) - dy });
            // The rectangles in host pixels: the source's edges and the destination's, the copy
            // the smaller of the two (they can differ by a pixel at a fractional scale).
            const int32_t hsx = ToHostX(sx0), hsy = ToHostY(sy0), hdx = ToHostX(dx), hdy = ToHostY(dy);
            const int32_t hw = std::min({ ToHostX(sx0 + w) - hsx, ToHostX(dx + w) - hdx, int32_t(dst.width) - hdx }),
                          hh = std::min({ ToHostY(sy0 + h) - hsy, ToHostY(dy + h) - hdy, int32_t(dst.height) - hdy });
            if (w > 0 && h > 0 && sx0 >= 0 && sy0 >= 0 && dx >= 0 && dy >= 0 && hw > 0 && hh > 0)
            {
                RenderPipeline* const texelCopy = !depth && scale == 1 && DrawCopies() ? TexelCopyPipeline(dst.format) : nullptr;
                // The whole of a single-sampled surface into a target of its size and format: the images swap instead
                // of copying. The surface holds the target's old image until RestoreStale or a whole
                // overwrite. In the post chain the next use of the surface overwrites it 4 times in 6.
                // AC6_RESOLVE_SWAP_FROM=<frame>: none before that frame (a check against copies from the same start).
                static const uint64_t swapFrom = [] { const char* v = std::getenv("AC6_RESOLVE_SWAP_FROM"); return v && *v ? std::stoull(v) : 0ull; }();
                const bool swap = !depth && scale == 1 && ResolveSwaps() && s_r.frame >= swapFrom && src == surface && src->samples == 1 && !src->fresh &&
                    !src->staleHolder && !dst.staleOf && !dst.mipChain && !src->swizzleTrace && !dst.swizzleTrace &&
                    src->format == dst.format && src->width == dst.width && src->height == dst.height &&
                    hsx == 0 && hsy == 0 && hdx == 0 && hdy == 0 && hw == int32_t(dst.width) && hh == int32_t(dst.height);
                if (swap)
                {
                    std::swap(src->texture, dst.texture);
                    std::swap(src->view, dst.view);
                    std::swap(src->descriptor, dst.descriptor);
                    std::swap(src->layout, dst.layout);
                    std::swap(src->fresh, dst.fresh);
                    std::swap(src->swizzled, dst.swizzled);
                    src->staleHolder = &dst;
                    dst.staleOf = src;
                    s_stats.resolveSwaps++;
                }
                else if (texelCopy)
                {
                    // Drawn, not copied (AC6_DRAW_COPIES=1): the texels through a shader into the
                    // destination as a render target.
                    Transition(cl, *src, RenderTextureLayout::SHADER_READ);
                    Transition(cl, dst, RenderTextureLayout::COLOR_WRITE);
                    cl.setFramebuffer(GetFramebuffer(&dst, nullptr));
                    s_r.markedFramebuffer = nullptr;
                    SetViewport(cl, RenderViewport(float(hdx), float(hdy), float(hw), float(hh)));
                    SetScissor(cl, RenderRect(hdx, hdy, hdx + hw, hdy + hh));
                    SetPipeline(cl, texelCopy);
                    BindCommon(cl);
                    const int32_t push[4] = { int32_t(src->descriptor), 0, hsx - hdx, hsy - hdy };
                    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
                    cl.drawInstanced(3, 1, 0, 0);
                }
                else if (!depth && scale == 1)
                {
                    Transition(cl, *src, RenderTextureLayout::COPY_SOURCE, dst, RenderTextureLayout::COPY_DEST);
                    const RenderBox box(hsx, hsy, hsx + hw, hsy + hh);
                    cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(dst.texture.get()),
                        RenderTextureCopyLocation::Subresource(src->texture.get()), uint32_t(hdx), uint32_t(hdy), 0, &box);
                }
                else if (!depth)
                {
                    // A promoted 4x MSAA surface resolves like the console's MSAA resolve: the samples of each
                    // destination pixel averaged, by a filtered blit of the source rectangle.
                    if (dst.format != RenderFormat::B8G8R8A8_UNORM)
                    {
                        static uint32_t warned = 0;
                        if (warned++ < 4)
                            log::Warn("Resolve of a 4x MSAA surface into format {} not supported", int(dst.format));
                    }
                    else
                    {
                        Transition(cl, *src, RenderTextureLayout::SHADER_READ);
                        Transition(cl, dst, RenderTextureLayout::COLOR_WRITE);
                        cl.setFramebuffer(GetFramebuffer(&dst, nullptr));
                        // The copy shader's triangle spans uv 0..1 over the viewport: place the whole source
                        // (in guest pixels) so that its rectangle lands on the destination, and scissor to it.
                        SetViewport(cl, RenderViewport(float(dx - sx0) * s_r.resX, float(dy - sy0) * s_r.resY, float(srcW) * s_r.resX,
                            float(srcH) * s_r.resY));
                        SetScissor(cl, RenderRect(hdx, hdy, hdx + hw, hdy + hh));
                        SetPipeline(cl, s_r.copyPipeline.get());
                        BindCommon(cl);
                        const uint32_t push[4] = { src->descriptor, 0, 0, 0 };
                        cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
                        cl.drawInstanced(3, 1, 0, 0);
                    }
                }
                else
                {
                    // Resolve statistics: the source depth surface too (is it empty, or is the copy lossy?).
                    if (s_r.frame - s_r.resolveStatsFrame < s_r.resolveStatsFrames && src->format == kDepthFormat)
                    {
                        Transition(cl, *src, RenderTextureLayout::COPY_SOURCE);
                        auto buffer = g_video.device->createBuffer(RenderBufferDesc::ReadbackBuffer(uint64_t(src->width) * src->height * 4));
                        cl.copyTextureRegion(RenderTextureCopyLocation::PlacedFootprint(buffer.get(), RenderFormat::R32_FLOAT, src->width, src->height, 1, src->width),
                            RenderTextureCopyLocation::Subresource(src->texture.get()));
                        s_r.resolveCaptures.push_back({ s_r.resolveIndex, std::format("source depth {:08X} before resolve", U32(device + kDevDepthStencil)),
                            std::move(buffer), src->width, src->height, RenderFormat::R32_FLOAT });
                    }
                    // One pass, depth and stencil: from a multisampled surface's samples (msDepth), or from a single-sampled
                    // image 1:1 (its stencil went through a buffer copy before).
                    HostImage* oneStencil = nullptr;
                    if (msDepth || (scale == 1 && src->samples == 1 && src->format == kDepthFormat && src->stencilDescriptor))
                    {
                        oneStencil = &s_r.resolvedStencil[fetch.BaseAddress() & 0x1FFFFFFF];
                        EnsureResolvedStencil(*oneStencil, hostW, hostH, worldResolve, nativeResolve);
                        if (oneStencil->width != dst.width || oneStencil->height != dst.height)
                        {
                            oneStencil = nullptr;   // one framebuffer needs them the same size: the passes below
                            if (msDepth)
                                src = &Readable(cl, *src);
                        }
                    }
                    if (oneStencil)
                    {
                        HostImage& stencilImage = *oneStencil;
                        if (stencilImage.fresh)
                        {
                            Transition(cl, stencilImage, RenderTextureLayout::COLOR_WRITE);
                            cl.setFramebuffer(GetFramebuffer(&stencilImage, nullptr));
                            cl.clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
                            stencilImage.fresh = false;
                        }
                        Transition(cl, *src, RenderTextureLayout::SHADER_READ);
                        Transition(cl, dst, RenderTextureLayout::COLOR_WRITE);
                        Transition(cl, stencilImage, RenderTextureLayout::COLOR_WRITE);
                        auto& framebuffer = s_r.depthResolveFramebuffers[{ dst.texture.get(), stencilImage.texture.get() }];
                        if (!framebuffer)
                        {
                            const RenderTexture* attachments[2] = { dst.texture.get(), stencilImage.texture.get() };
                            RenderFramebufferDesc desc;
                            desc.colorAttachments = attachments;
                            desc.colorAttachmentsCount = 2;
                            framebuffer = g_video.device->createFramebuffer(desc);
                        }
                        cl.setFramebuffer(framebuffer.get());
                        s_r.markedFramebuffer = nullptr;
                        SetViewport(cl, RenderViewport(float(hdx), float(hdy), float(hw), float(hh)));
                        SetScissor(cl, RenderRect(hdx, hdy, hdx + hw, hdy + hh));
                        SetPipeline(cl, msDepth ? s_r.depthCopyMsPipeline.get() : s_r.depthCopySsPipeline.get());
                        BindCommon(cl);
                        const int32_t push[4] = { int32_t(msDepth ? src->msDepthDescriptor : src->descriptor), int32_t(src->stencilDescriptor),
                            ToHostX(double(sx0 - dx)), ToHostY(double(sy0 - dy)) };
                        cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
                        cl.drawInstanced(3, 1, 0, 0);
                        Transition(cl, stencilImage, RenderTextureLayout::SHADER_READ);
                    }
                    else
                    {
                        // Depth can't be copied into a colour image: draw it (depth texel -> R32F).
                        Transition(cl, *src, RenderTextureLayout::SHADER_READ);
                        Transition(cl, dst, RenderTextureLayout::COLOR_WRITE);
                        cl.setFramebuffer(GetFramebuffer(&dst, nullptr));
                        SetViewport(cl, RenderViewport(float(hdx), float(hdy), float(hw), float(hh)));
                        SetScissor(cl, RenderRect(hdx, hdy, hdx + hw, hdy + hh));
                        SetPipeline(cl, s_r.depthCopyPipeline.get());
                        BindCommon(cl);
                        // Source host pixel = destination host pixel * scale + offset (host units).
                        const int32_t push[4] = { int32_t(src->descriptor), scale, ToHostX(double(sx0 - dx) * scale), ToHostY(double(sy0 - dy) * scale) };
                        cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
                        cl.drawInstanced(3, 1, 0, 0);
                        // The stencil goes along into an R8 image beside the depth (1:1 sources: buffer copies of the
                        // stencil aspect are tightly packed bytes).
                        if (scale == 1 && src->format == kDepthFormat)
                        {
                            HostImage& stencilImage = s_r.resolvedStencil[fetch.BaseAddress() & 0x1FFFFFFF];
                            EnsureResolvedStencil(stencilImage, hostW, hostH, worldResolve, nativeResolve);
                            if (stencilImage.fresh)
                            {
                                Transition(cl, stencilImage, RenderTextureLayout::COLOR_WRITE);
                                cl.setFramebuffer(GetFramebuffer(&stencilImage, nullptr));
                                cl.clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
                                stencilImage.fresh = false;
                            }
                            CopyStencil(cl, *src, hsx, hsy, hw, hh, stencilImage, hdx, hdy);
                        }
                    }
                }
                if (s_r.frame - s_r.resolveStatsFrame < s_r.resolveStatsFrames &&
                    (dst.format == RenderFormat::R32_FLOAT || dst.format == RenderFormat::R16G16B16A16_FLOAT || dst.format == RenderFormat::B8G8R8A8_UNORM))
                {
                    const uint32_t bpp = dst.format == RenderFormat::R16G16B16A16_FLOAT ? 8 : 4;
                    Transition(cl, dst, RenderTextureLayout::COPY_SOURCE);
                    auto buffer = g_video.device->createBuffer(RenderBufferDesc::ReadbackBuffer(uint64_t(dst.width) * dst.height * bpp));
                    cl.copyTextureRegion(RenderTextureCopyLocation::PlacedFootprint(buffer.get(), dst.format, dst.width, dst.height, 1, dst.width),
                        RenderTextureCopyLocation::Subresource(dst.texture.get()));
                    s_r.resolveCaptures.push_back({ s_r.resolveIndex, std::format("{:08X} from {:08X} rect {},{} {}x{}", fetch.BaseAddress(),
                        U32(device + (depth ? kDevDepthStencil : kDevRenderTargets)), sx0, sy0, w, h), std::move(buffer), dst.width, dst.height, dst.format });
                }
                // A mip chain gets the new level 0 too.
                if (dst.mipChain && dst.mipChain->texture && dst.mipChain->format == dst.format && dst.mipWritten)
                {
                    HostImage& chain = *dst.mipChain;
                    const int32_t cw = std::min(hw, int32_t(chain.width) - hdx), ch = std::min(hh, int32_t(chain.height) - hdy);
                    if (cw > 0 && ch > 0)
                    {
                        Transition(cl, dst, RenderTextureLayout::COPY_SOURCE);
                        Transition(cl, chain, RenderTextureLayout::COPY_DEST);
                        const RenderBox box(hdx, hdy, hdx + cw, hdy + ch);
                        cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(chain.texture.get()),
                            RenderTextureCopyLocation::Subresource(dst.texture.get()), uint32_t(hdx), uint32_t(hdy), 0, &box);
                        Transition(cl, chain, RenderTextureLayout::SHADER_READ);
                    }
                }
                // The scene image (0xA5BEB000, the world pass's 1280x720 resolve, which the post chain, the distortion
                // and the composite read), once its last part is in (the game resolves it in halves): ambient
                // occlusion, emitter light, SMAA, then TAA or FSR and the object motion blur. Once a frame: as a mission is
                // paused the game resolves the finished frame (its HUD on it) into the same image again, as the pause
                // menu's background. Taken for the scene, that copy had the ambient occlusion over it a second time,
                // without the terrain's depth (the frame's terrain draws already used) and so on the terrain too, held
                // for as long as the game stays paused.
                if (!depth && (fetch.BaseAddress() & 0x1FFFFFFF) == 0x05BEB000 && dx + w >= int32_t(dstW) && dy + h >= int32_t(dstH) &&
                    s_r.scenePostFrame != s_r.frame)
                {
                    s_r.scenePostFrame = s_r.frame;
                    // (The sea's reflection of the scene: the scene's surface, whose sea draws give the camera.)
                    s_r.seaSceneSurface = surface;
                    // Ambient occlusion into the scene's colour, before the anti-aliasing.
                    if (AoMode())
                        RunAmbientOcclusion(cl, dst);
                    if (EmitterLight() > 0.0f)
                        RunEmitterLight(cl, dst);
                    if (SmaaOn())
                        RunSmaa(cl, dst);
                    // Temporal AA or FSR: the surface the scene is drawn in, jittered from the next frame on.
                    // With FSR upscaling the scene's surfaces are at the render scale from the next frame on, when FSR
                    // ran on this one: FSR makes the target, at the frame's size, from the render-size resolve. Where
                    // it can't run (no world camera among the draws: the hangar's aircraft select), the surfaces stay at
                    // the frame's size. Some screens draw their menus into the surface the scene is resolved from: at
                    // the render scale with nothing to scale it up, the whole frame showed small in a corner.
                    if (TaaOn() || s_r.fsr)
                    {
                        s_r.taaSurface = surface;
                        s_r.taaSurfaceFrame = s_r.frame;
                        // (Without upscaling FSR works on the resolve's own image: with HDR from the scene that is the
                        // target's float twin, not this map's.)
                        HostImage& target = worldResolve ? s_r.resolved[0x05BEB000] : dst;
                        // (Object motion blur: its motion is made before the anti-aliasing, which stores this frame's
                        // camera as the last one, and the blur is applied after it.)
                        HostImage* const blurMotion = MotionBlurStrength() > 0.0f ? MotionForBlur(cl) : nullptr;
                        // (AC6_FSR_LATE: FSR runs before the frame's first 2D layer instead, BeginUiImage.)
                        const bool fsrRan = s_r.fsr && (s_r.lateUpscale || s_r.lateNative || RunFsr(cl, target, worldResolve ? &s_r.fsrColor : nullptr));
                        if (s_r.upscaling && fsrRan)
                        {
                            s_r.worldColorSurface = sourceSurface;
                            s_r.worldDepthSurface = U32(device + kDevDepthStencil);
                            s_r.worldFrame = s_r.frame;
                        }
                        if (!fsrRan)
                        {
                            if (worldResolve)
                                StretchInto(cl, s_r.fsrColor, target, uint32_t(ToHostX(dstW) * s_r.sceneResX / s_r.resX + 0.5),
                                    uint32_t(ToHostY(dstH) * s_r.sceneResY / s_r.resY + 0.5));   // FSR failed: scaled up as it is
                            else
                                RunTaa(cl, dst);
                        }
                        if (blurMotion)
                            RunMotionBlur(cl, target, *blurMotion);
                    }
                }
                s_r.resolveIndex++;
                Transition(cl, dst, RenderTextureLayout::SHADER_READ);
                HostImage& shown = worldResolve && sceneColor ? s_r.resolved[0x05BEB000] : dst;
                if (!depth && dstW == kTargetWidth && dstH == kTargetHeight)
                {
                    s_r.frontBuffer = &shown;
                    s_r.frontBufferFrame = s_r.frame;
                }
            }
        }
    }
    else if (src && destTexture && destLevel != 0)
    {
        // Resolve into a mip level: AC6 renders the trees' sprite textures every frame
        // and draws each smaller level from the one above, at the side of the render target, then resolves
        // it into that level. Without the levels the distant trees would sample the full-size image: noise that
        // shimmers as they move. Colour resolves of a single-sampled copy into the level-0 image's
        // format, after level 0 (which makes the image).
        const auto fetch = xenos::ReadTextureFetch(g_memory.As<uint8_t>(destTexture + kTextureObjectFetch));
        auto it = fetch.Type() == 2 && fetch.BaseAddress() ? s_r.resolved.find(fetch.BaseAddress() & 0x1FFFFFFF) : s_r.resolved.end();
        bool done = false;
        if (!depth && it != s_r.resolved.end() && it->second.texture && destLevel <= fetch.MaxMip() &&
            std::has_single_bit(fetch.Width()) && std::has_single_bit(fetch.Height()))
        {
            HostImage& dst = it->second;
            SyncAliasedColor(cl, *src, false);
            HostImage& from = Readable(cl, *src);
            if (from.scale == 1 && from.format == dst.format)
            {
                HostImage& chain = EnsureMipChain(cl, dst, uint32_t(ToHostX(fetch.Width())), uint32_t(ToHostY(fetch.Height())), fetch.MaxMip() + 1);
                // Guest pixels, then host ones.
                const int32_t lw = int32_t(std::max(1u, fetch.Width() >> destLevel)), lh = int32_t(std::max(1u, fetch.Height() >> destLevel));
                const int32_t fromW = int32_t(GuestWidth(from)), fromH = int32_t(GuestHeight(from));
                int32_t sx0 = 0, sy0 = 0, sx1 = fromW, sy1 = fromH;
                if (srcRect)
                {
                    sx0 = srcRect->x1; sy0 = srcRect->y1; sx1 = srcRect->x2; sy1 = srcRect->y2;
                }
                const int32_t dx = destPoint ? destPoint->x : 0, dy = destPoint ? destPoint->y : 0;
                const int32_t w = std::min({ sx1 - sx0, fromW - sx0, lw - dx });
                const int32_t h = std::min({ sy1 - sy0, fromH - sy0, lh - dy });
                const int32_t hsx = ToHostX(sx0), hsy = ToHostY(sy0), hdx = ToHostX(dx), hdy = ToHostY(dy);
                const int32_t levelW = int32_t(std::max(1u, chain.width >> destLevel)), levelH = int32_t(std::max(1u, chain.height >> destLevel));
                const int32_t hw = std::min({ ToHostX(sx0 + w) - hsx, ToHostX(dx + w) - hdx, levelW - hdx }),
                              hh = std::min({ ToHostY(sy0 + h) - hsy, ToHostY(dy + h) - hdy, levelH - hdy });
                if (w > 0 && h > 0 && sx0 >= 0 && sy0 >= 0 && dx >= 0 && dy >= 0 && hw > 0 && hh > 0 && destLevel < chain.mipLevels)
                {
                    Transition(cl, from, RenderTextureLayout::COPY_SOURCE);
                    Transition(cl, chain, RenderTextureLayout::COPY_DEST);
                    const RenderBox box(hsx, hsy, hsx + hw, hsy + hh);
                    cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(chain.texture.get(), destLevel),
                        RenderTextureCopyLocation::Subresource(from.texture.get()), uint32_t(hdx), uint32_t(hdy), 0, &box);
                    Transition(cl, chain, RenderTextureLayout::SHADER_READ);
                    dst.mipWritten |= 1u << destLevel;
                }
                done = true;
            }
        }
        static uint32_t warned = 0;
        if (!done && warned++ < 4)
            log::Warn("Resolve into mip level {} not supported ({})", destLevel, depth ? "depth" : "no level-0 image or other format");
    }

    // The clear covers the resolved rectangle only: with tiling, each tile resolves and clears its own
    // part while the whole frame lives in one host surface.
    if (flags & (kResolveClearColor | kResolveClearDepth))
    {
        Targets t;
        // A colour clear over the whole surface overwrites a swapped one.
        bool clearsColor = false;
        if (flags & kResolveClearColor)
            if (HostImage* color = GetSurface(U32(device + kDevRenderTargets), false))
                clearsColor = !srcRect || (srcRect->x1 <= 0 && srcRect->y1 <= 0 && srcRect->x2 >= int32_t(GuestWidth(*color)) &&
                    srcRect->y2 >= int32_t(GuestHeight(*color)));
        t_overwritesColor = clearsColor;
        const bool bound = BindTargets(cl, device, t, 0, true);
        t_overwritesColor = false;
        if (bound)
        {
            RenderRect area(0, 0, int32_t(t.width), int32_t(t.height));
            // The guest rectangle in the bound surface's host pixels (its 4x MSAA promotion and the internal
            // resolution).
            if (srcRect)
                area = RenderRect(std::max(0, ToHostX(double(srcRect->x1) * t.scale)), std::max(0, ToHostY(double(srcRect->y1) * t.scale)),
                    std::min(int32_t(t.width), ToHostX(double(srcRect->x2) * t.scale)), std::min(int32_t(t.height), ToHostY(double(srcRect->y2) * t.scale)));
            if (area.right > area.left && area.bottom > area.top)
            {
                MarkPass([&] { return std::format("resolve clear{}{} {} + {}", (flags & kResolveClearColor) ? " colour" : "",
                    (flags & kResolveClearDepth) ? " depth" : "", PassImage(t.color), PassImage(t.depth)); });
                if ((flags & kResolveClearColor) && t.color)
                {
                    cl.clearColor(0, clearColor ? RenderColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]) : RenderColor(),
                        &area, 1);
                    if (clearsColor)
                        UnlinkStale(*t.color);
                    if (t.color == s_r.taaSurface)
                        s_r.motionClearPending = true;
                }
                if ((flags & kResolveClearDepth) && t.depth)
                {
                    cl.clearDepthStencil(true, true, clearZ, clearStencil, &area, 1);
                    NoteDepthWrite(t.depth);
                }
                // A multisampled surface's single-sampled copy that was up to date gets the same clear, so it stays
                // so: the scene resolves in two halves, each cleared after, and the second half
                // does not resolve the whole surface again. A cleared area's samples are equal, so its average
                // (and sample 0) is the clear value.
                auto clearSingle = [&](HostImage* image, bool cleared) {
                    if (!image || image->samples == 1 || !cleared || !image->singleCurrent || !image->single || !image->single->texture)
                        return;
                    HostImage& single = *image->single;
                    if (image->depth)
                    {
                        Transition(cl, single, RenderTextureLayout::DEPTH_WRITE);
                        cl.setFramebuffer(GetFramebuffer(nullptr, &single));
                        cl.clearDepthStencil(true, true, clearZ, clearStencil, &area, 1);
                    }
                    else
                    {
                        Transition(cl, single, RenderTextureLayout::COLOR_WRITE);
                        cl.setFramebuffer(GetFramebuffer(&single, nullptr));
                        cl.clearColor(0, clearColor ? RenderColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]) : RenderColor(),
                            &area, 1);
                    }
                    s_r.markedFramebuffer = nullptr;
                };
                clearSingle(t.color, (flags & kResolveClearColor) != 0);
                clearSingle(t.depth, (flags & kResolveClearDepth) != 0);
            }
        }
    }
}

void BeginTilingImpl(const ResolveRect* rects, uint32_t count)
{
    uint32_t w = 0, h = 0;
    for (uint32_t i = 0; rects && i < count && i < 16; i++)
    {
        w = std::max<uint32_t>(w, uint32_t(std::max(0, int32_t(rects[i].x2))));
        h = std::max<uint32_t>(h, uint32_t(std::max(0, int32_t(rects[i].y2))));
    }
    s_r.tiling = w && h;
    s_r.tilingWidth = w;
    s_r.tilingHeight = h;
}

void EndTilingImpl()
{
    s_r.tiling = false;
}

uint32_t SnapshotBytesPerTexel(RenderFormat f)
{
    switch (f)
    {
    case RenderFormat::R32_FLOAT: case RenderFormat::B8G8R8A8_UNORM: case RenderFormat::R16G16_FLOAT: case RenderFormat::D32_FLOAT:
    case RenderFormat::D32_FLOAT_S8_UINT: return 4;   // depth only (buffer copies take the depth aspect)
    case RenderFormat::R16G16B16A16_FLOAT: case RenderFormat::R32G32_FLOAT: return 8;
    default: return 0;
    }
}

// Frame capture: copies every resolve target and render surface (they carry state across frames: exposure
// adaptation, last frame's scene, shadow maps) at the start of this frame. The data reach the capture at
// the frame's present.
void SnapshotResolvedImpl(uint64_t maxTexels)
{
    if (!s_r.ready)
        return;
    RenderCommandList& cl = g_video.Commands();
    // The ring's (only the small images) are collected when this frame's slot comes round again.
    const bool ring = maxTexels != UINT64_MAX;
    auto& snapshots = ring ? s_r.ringSnapshots[g_video.slot] : s_r.resolvedSnapshots;
    if (ring)
    {
        snapshots.clear();
        s_r.ringSnapshotFrame[g_video.slot] = s_r.frame;
    }
    auto snapshot = [&cl, &snapshots, maxTexels](uint32_t kind, uint32_t address, HostImage& image) {
        const uint32_t bpp = SnapshotBytesPerTexel(image.format);
        if (!image.texture || !bpp || uint64_t(image.width) * image.height > maxTexels)
            return;
        const RenderTextureLayout after = image.layout;
        Transition(cl, image, RenderTextureLayout::COPY_SOURCE);
        auto buffer = g_video.device->createBuffer(RenderBufferDesc::ReadbackBuffer(uint64_t(image.width) * image.height * bpp));
        cl.copyTextureRegion(RenderTextureCopyLocation::PlacedFootprint(buffer.get(), image.format, image.width, image.height, 1, image.width),
            RenderTextureCopyLocation::Subresource(image.texture.get()));
        if (after != RenderTextureLayout::UNKNOWN)
            Transition(cl, image, after);
        snapshots.push_back({ kind, address, image.width, image.height, image.format, std::move(buffer) });
    };
    for (auto& [address, image] : s_r.resolved)
        snapshot(0, address, image);
    for (auto& [address, image] : s_r.surfaces)
        if (SurfaceOwner(address) == address)   // not an image left behind by a surface that now shares another's
            snapshot(image.depth ? 2 : 1, address, Readable(cl, image));
}

// Replay: recreates the resolve targets and surfaces from a capture's snapshot (u32 count, then per image
// u32 kind (0 resolve target, 1 colour surface, 2 depth surface), address, width, height, format, then the
// texels). Surfaces are made from their guest objects, so the memory snapshot must be restored first.
void RestoreResolvedImpl(const uint8_t* data, size_t size)
{
    if (!s_r.ready)
        return;
    s_r.textureEpoch++;
    RenderCommandList& cl = g_video.Commands();
    size_t at = 0;
    auto get = [&]() { uint32_t v = 0; if (at + 4 <= size) std::memcpy(&v, data + at, 4); at += 4; return v; };
    for (uint32_t n = get(), i = 0; i < n && at < size; i++)
    {
        const uint32_t kind = get(), address = get(), width = get(), height = get();
        RenderFormat format = RenderFormat(get());
        if (kind == 2 && format == RenderFormat::D32_FLOAT)
            format = kDepthFormat;   // an older capture's depth, saved without stencil
        const uint32_t bpp = SnapshotBytesPerTexel(format);
        const uint64_t bytes = uint64_t(width) * height * bpp;
        if (!bpp || at + bytes > size)
            break;
        HostImage* target = nullptr;
        if (kind == 0)
        {
            target = &s_r.resolved[address];
            EnsureImage(*target, width, height, format, false);
        }
        else
            target = GetSurface(address, kind == 2);
        if (!target || !target->texture || target->format != format || target->width != width || target->height != height)
        {
            at += bytes;
            continue;
        }
        // A multisampled surface is restored into its single-sampled copy, then into its samples.
        HostImage& surface = *target;
        HostImage& image = surface.samples > 1 ? Readable(cl, surface) : surface;
        image.fresh = false;
        const uint32_t rowPitch = (width * bpp + 255) & ~255u;
        // Large snapshots (frame-sized ones at high resolutions, which together exceed the ring) get
        // a staging buffer of their own, retired with the frame like the large textures'.
        const uint64_t imageBytes = uint64_t(rowPitch) * height;
        RenderBuffer* source = s_r.upload.buffer.get();
        uint8_t* memory = nullptr;
        uint64_t offset = 0;
        std::unique_ptr<RenderBuffer> staging;
        if (imageBytes > (4ull << 20))
        {
            staging = g_video.device->createBuffer(RenderBufferDesc::UploadBuffer(imageBytes));
            memory = static_cast<uint8_t*>(staging->map());
            source = staging.get();
        }
        else
        {
            auto a = s_r.upload.Allocate(imageBytes, 512);
            if (!a.memory)
                break;
            memory = a.memory;
            offset = a.offset;
        }
        for (uint32_t y = 0; y < height; y++)
            std::memcpy(memory + uint64_t(y) * rowPitch, data + at + uint64_t(y) * width * bpp, width * bpp);
        at += bytes;
        Transition(cl, image, RenderTextureLayout::COPY_DEST);
        cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(image.texture.get()),
            RenderTextureCopyLocation::PlacedFootprint(source, format, width, height, 1, rowPitch / bpp, offset));
        if (staging)
        {
            staging->unmap();
            s_r.retiredStaging.push_back(std::move(staging));
        }
        Transition(cl, image, kind == 2 ? RenderTextureLayout::DEPTH_WRITE : kind == 1 ? RenderTextureLayout::COLOR_WRITE
                                                                                         : RenderTextureLayout::SHADER_READ);
        if (surface.samples > 1)
            WriteSamples(cl, surface);
    }
}

// Replay diagnostics: reads back the device's current depth surface. The statistics are logged at present.
void ProbeDepthImpl(uint32_t device, const char* label)
{
    if (!s_r.ready)
        return;
    HostImage* src = GetSurface(U32(device + kDevDepthStencil), true);
    if (!src || !src->texture || src->format != kDepthFormat)
        return;
    RenderCommandList& cl = g_video.Commands();
    src = &Readable(cl, *src);
    const RenderTextureLayout after = src->layout;
    Transition(cl, *src, RenderTextureLayout::COPY_SOURCE);
    auto buffer = g_video.device->createBuffer(RenderBufferDesc::ReadbackBuffer(uint64_t(src->width) * src->height * 4));
    cl.copyTextureRegion(RenderTextureCopyLocation::PlacedFootprint(buffer.get(), RenderFormat::R32_FLOAT, src->width, src->height, 1, src->width),
        RenderTextureCopyLocation::Subresource(src->texture.get()));
    if (after != RenderTextureLayout::UNKNOWN)
        Transition(cl, *src, after);
    s_r.resolveCaptures.push_back({ 9999, std::format("depth probe {} ({:08X})", label, U32(device + kDevDepthStencil)), std::move(buffer),
        src->width, src->height, RenderFormat::R32_FLOAT });
}

// --- Display gamma ramp --------------------------------------------------------------
// The display controller maps each scanned-out 10-bit level through a gamma ramp the game sets (corrected by
// D3D for the display type). Both of its forms are kept as one 1024-entry table, applied at present.
// AC6_GAMMA_RAMP=0 presents the image as rendered.
struct GammaRamp
{
    std::mutex mutex;
    uint32_t lut[1024];   // red << 20 | green << 10 | blue
    bool set = false;
} s_gamma;

// AC6_DIRECT_PRESENT=0: always draw the frame into the output target and copy that to the window.
bool DirectPresent()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_DIRECT_PRESENT"); return !(v && *v == '0'); }();
    return enabled;
}

bool GammaRampEnabled()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_GAMMA_RAMP"); return !v || std::strcmp(v, "0") != 0; }();
    return enabled;
}

void StoreGammaRamp(const uint32_t (&lut)[1024], const char* kind)
{
    std::lock_guard lock(s_gamma.mutex);
    if (s_gamma.set && std::memcmp(s_gamma.lut, lut, sizeof(lut)) == 0)
        return;
    std::memcpy(s_gamma.lut, lut, sizeof(lut));
    s_gamma.set = true;
    auto level = [&](uint32_t i) { return (lut[i] >> 10) & 1023; };   // green
    log::Info("Gamma ramp ({}): 10-bit levels 64/256/512/768/1023 -> {}/{}/{}/{}/{}", kind, level(64), level(256), level(512), level(768), level(1023));
}

bool RendererGammaRamp(uint32_t lut[1024])
{
    std::lock_guard lock(s_gamma.mutex);
    if (s_gamma.set)
        std::memcpy(lut, s_gamma.lut, sizeof(s_gamma.lut));
    return s_gamma.set;
}

void RendererRestoreGammaRamp(const uint32_t lut[1024])
{
    uint32_t copy[1024];
    std::memcpy(copy, lut, sizeof(copy));
    StoreGammaRamp(copy, "replay");
}

bool RendererPipelineListWants(uint64_t shaderHash)
{
    return s_pipelineList.Wants(shaderHash);
}

void RendererShaderReady(const HostShader& shader)
{
    if (shader.ok && shader.shader)
        s_pipelineList.ShaderReady(shader);
}

bool RendererPrepareShaderLibrary(HostShader& shader)
{
    if (!s_r.ready || !g_video.device->usesShaderLibraries() || !shader.ok || shader.spirv.empty())
        return false;
    if (!shader.shader)
        MakeShaderModule(shader);
    // As GetPipeline specialises shaders that read the draw flags.
    RenderSpecConstant spec(0, 0);
    g_video.device->prepareShaderLibrary(s_r.layout.get(), shader.shader.get(), shader.pixel, &spec, 1);
    return true;
}

void RendererSetGammaTable(const uint16_t ramp[3][256])
{
    // 256 entries, indexed by the 8-bit level, 10-bit outputs in the top bits.
    uint32_t lut[1024];
    for (uint32_t v = 0; v < 1024; v++)
    {
        const uint32_t index = (v * 255 + 511) / 1023;
        lut[v] = uint32_t(ramp[0][index] >> 6) << 20 | uint32_t(ramp[1][index] >> 6) << 10 | uint32_t(ramp[2][index] >> 6);
    }
    StoreGammaRamp(lut, "table");
}

void RendererSetGammaPwl(const uint16_t ramp[3][128][2])
{
    // 128 segments of 8 levels: base + delta * (position in the segment) / 8, 16-bit (bits 0-5 are zero).
    uint32_t lut[1024];
    for (uint32_t v = 0; v < 1024; v++)
    {
        uint32_t out[3];
        for (uint32_t c = 0; c < 3; c++)
        {
            const float base = float(ramp[c][v >> 3][0] & ~0x3Fu), delta = float(ramp[c][v >> 3][1] & ~0x3Fu);
            const float value = std::min((base + delta * float(v & 7) * 0.125f) / 65536.0f, 1.0f);
            out[c] = uint32_t(value * 1023.0f + 0.5f);
        }
        lut[v] = out[0] << 20 | out[1] << 10 | out[2];
    }
    StoreGammaRamp(lut, "piecewise linear");
}

// The settings overlay (src/ui/overlay.cpp): the frame's ImGui draw lists, taken once a presented frame, and its texture
// changes made then (copies: outside any pass, the caller binds its framebuffer again after).
ui::DrawFrame s_overlay;
uint64_t s_overlayTaken = UINT64_MAX;
bool s_overlayDraw = false;

bool TakeOverlay(RenderCommandList& cl)
{
    if (s_overlayTaken == g_video.frameCount)
        return s_overlayDraw;
    ServiceDumpRequests();   // the texture browser's, while the game is held (gpu/texture_mods.h)
    s_overlayTaken = g_video.frameCount;
    s_overlayDraw = ui::TakeFrame(s_overlay) && s_r.imguiPipeline && !s_overlay.commands.empty() && s_overlay.width > 0.0f &&
        s_overlay.height > 0.0f;
    if (s_overlay.textures.empty())
        return s_overlayDraw;
    cl.setFramebuffer(nullptr);
    for (ui::TextureUpdate& u : s_overlay.textures)
    {
        if (u.destroy)
        {
            if (auto it = s_r.overlayTextures.find(u.id); it != s_r.overlayTextures.end())
            {
                s_r.retiredTextures.push_back(std::move(it->second.texture));
                s_r.overlayTextures.erase(it);
            }
            continue;
        }
        Renderer::OverlayTexture& t = s_r.overlayTextures[u.id];
        if (!t.texture || t.width != u.width || t.height != u.height)
        {
            t.width = u.width;
            t.height = u.height;
            if (t.texture)
                s_r.retiredTextures.push_back(std::move(t.texture));
            t.texture = MakeTexture(RenderTextureDesc::Texture2D(u.width, u.height, 1, RenderFormat::R8G8B8A8_UNORM), "an uploaded texture");
            t.view = t.texture->createTextureView(RenderTextureViewDesc::Texture2D(RenderFormat::R8G8B8A8_UNORM));
            if (!t.descriptor)
                t.descriptor = AllocateDescriptor();
            SetDescriptor(t.descriptor, t.texture.get(), t.view.get());
        }
        // Rows padded to 256 bytes, as copies from buffers want on some drivers.
        const uint32_t pitch = (u.width + 63) & ~63u;
        auto a = s_r.upload.Allocate(uint64_t(pitch) * u.height * 4, 512);
        if (!a.memory)
            continue;
        for (uint32_t y = 0; y < u.height; y++)
            std::memcpy(a.memory + uint64_t(y) * pitch * 4, u.rgba.data() + size_t(y) * u.width * 4, size_t(u.width) * 4);
        cl.barriers(RenderBarrierStage::COPY, RenderTextureBarrier(t.texture.get(), RenderTextureLayout::COPY_DEST));
        cl.copyTextureRegion(RenderTextureCopyLocation::Subresource(t.texture.get()),
            RenderTextureCopyLocation::PlacedFootprint(s_r.upload.buffer.get(), RenderFormat::R8G8B8A8_UNORM, u.width, u.height, 1, pitch, a.offset));
        cl.barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(t.texture.get(), RenderTextureLayout::SHADER_READ));
    }
    s_overlay.textures.clear();
    return s_overlayDraw;
}

// The overlay's triangles into the bound framebuffer (w x h pixels), clipped per command. HDR windows (scRGB) get it at
// paper white (paper > 0), as the game's UI.
void DrawOverlayInto(RenderCommandList& cl, const RenderPipeline* pipeline, float w, float h, float paper, bool frameReadable)
{
    const ui::DrawFrame& frame = s_overlay;
    const uint64_t vertexBytes = frame.vertices.size() * sizeof(ui::DrawVertex), indexBytes = frame.indices.size() * sizeof(uint16_t);
    auto v = s_r.upload.Allocate(vertexBytes, 16);
    auto i = s_r.upload.Allocate(indexBytes, 16);
    if (!v.memory || !i.memory)
        return;
    std::memcpy(v.memory, frame.vertices.data(), vertexBytes);
    std::memcpy(i.memory, frame.indices.data(), indexBytes);
    const float sx = w / frame.width, sy = h / frame.height;   // ImGui's units to the image's pixels
    SetPipeline(cl, pipeline);
    BindCommon(cl);
    SetViewport(cl, RenderViewport(0.0f, 0.0f, w, h));
    const RenderVertexBufferView vb(RenderBufferReference(s_r.upload.buffer.get(), v.offset), uint32_t(vertexBytes));
    const RenderInputSlot slot(0, sizeof(ui::DrawVertex));
    cl.setVertexBuffers(0, &vb, 1, &slot);
    const RenderIndexBufferView ib(RenderBufferReference(s_r.upload.buffer.get(), i.offset), uint32_t(indexBytes), RenderFormat::R16_UINT);
    cl.setIndexBuffer(&ib);
    // Pixels (ImGui's units) to clip space, D3D style (the vertex shader is compiled with the y flip).
    struct { float scale[2], translate[2]; uint32_t texture; float paper; } push{ { 2.0f / frame.width, -2.0f / frame.height }, { -1.0f, 1.0f }, 0, paper };
    static_assert(sizeof(push) == 24);
    for (const ui::DrawCommand& c : frame.commands)
    {
        const int32_t x0 = std::clamp(int32_t(c.clip[0] * sx), 0, int32_t(w)), y0 = std::clamp(int32_t(c.clip[1] * sy), 0, int32_t(h));
        const int32_t x1 = std::clamp(int32_t(std::ceil(c.clip[2] * sx)), 0, int32_t(w)), y1 = std::clamp(int32_t(std::ceil(c.clip[3] * sy)), 0, int32_t(h));
        if (x1 <= x0 || y1 <= y0)
            continue;
        if (c.texture == ui::kHeldFrameTexture)
            push.texture = frameReadable ? s_r.targetDescriptor : kNullTexture2D;
        else if ((c.texture & 0xFF000000u) == mods::kGameTextureFlag)
            push.texture = c.texture & 0x00FFFFFFu;   // a game texture by its descriptor (the texture browser)
        else
        {
            const auto it = s_r.overlayTextures.find(c.texture);
            push.texture = it != s_r.overlayTextures.end() ? it->second.descriptor : kNullTexture2D;
        }
        cl.setGraphicsPushConstants(0, &push, 0, sizeof(push));
        SetScissor(cl, RenderRect(x0, y0, x1, y1));
        cl.drawIndexedInstanced(c.indexCount, 1, c.firstIndex, int32_t(c.vertexOffset), 0);
    }
}

// Over the window's image, which is bound for drawing (bound again after the texture copies). frameReadable: the output
// target holds the frame and can be sampled (the overlay shows it).
void DrawOverlay(RenderCommandList& cl, bool frameReadable)
{
    const bool draw = TakeOverlay(cl);
    cl.setFramebuffer(g_video.BackBufferFramebuffer());
    if (!draw)
        return;
    MarkPass([] { return std::string("present: the settings overlay"); });
    DrawOverlayInto(cl, s_r.imguiPipeline.get(), float(g_video.BackBufferWidth()), float(g_video.BackBufferHeight()),
        g_video.OutputMode() == Video::Output::Hdr ? OutputConfig().paperScale : 0.0f, frameReadable);
}

// The output target into the window, fitted (letterboxed or pillarboxed where its shape isn't the window's), through
// the output pass: the bound framebuffer is the window's after.
void TargetToWindow(RenderCommandList& cl)
{
    TransitionTarget(cl, RenderTextureLayout::SHADER_READ);
    cl.barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(g_video.BackBuffer(), RenderTextureLayout::COLOR_WRITE));
    cl.setFramebuffer(g_video.BackBufferFramebuffer());
    // Fit the target into the window, keeping its shape (letterbox/pillarbox).
    const float ww = float(g_video.BackBufferWidth()), wh = float(g_video.BackBufferHeight());
    const float scale = std::min(ww / float(s_r.targetWidth), wh / float(s_r.targetHeight));
    const float w = float(s_r.targetWidth) * scale, h = float(s_r.targetHeight) * scale;
    // Bars only where the picture doesn't reach.
    if (w < ww || h < wh)
        cl.clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 1.0f));
    SetViewport(cl, RenderViewport((ww - w) / 2, (wh - h) / 2, w, h));
    SetScissor(cl, RenderRect(0, 0, int32_t(ww), int32_t(wh)));
    if (s_r.outputPipeline)
    {
        // Debanding, dithering, the HDR encode.
        SetPipeline(cl, s_r.outputPipeline.get());
        BindCommon(cl);
        const OutputSettings& o = OutputConfig();
        // HDR from the scene: the float frame and its frame before the UI instead, grade1 bit 24.
        uint32_t frame = s_r.targetDescriptor | (PreUiDescriptor() << 16), grade1 = o.grade1;
        // (The fourth word: the frame number, the dither's. With HDR, which doesn't dither, the ramp word.)
        uint32_t seed = (o.flags & 4) ? 0u : uint32_t(SeedFrame());
        if ((o.flags & 4) && HdrSceneOn() && s_r.hdrDirectFrame == s_r.frame)
        {
            // The frames as the game drew them, the gamma ramp in the pass.
            frame = s_r.hdrDirectTextures;
            seed = s_r.hdrDirectRamp;
            grade1 |= 1u << 24;
        }
        else if ((o.flags & 4) && HdrSceneOn() && s_r.hdrFrameFrame == s_r.frame && s_r.hdrFrame.descriptor < 65536)
        {
            const bool preUi = s_r.hdrPreUiFrame == s_r.frame && s_r.hdrPreUi.descriptor < 65536;
            frame = s_r.hdrFrame.descriptor | (preUi ? s_r.hdrPreUi.descriptor << 16 : 0u);
            grade1 |= 1u << 24;
        }
        const uint32_t push[7] = { frame, o.flags | AllUiFlag(), std::bit_cast<uint32_t>(o.paperScale), seed, o.grade0, grade1,
            o.stripes };
        // AC6_HDR_SIM: on a screenshot frame, the HDR output pass into the float image too, read back.
        if (s_r.outputSimPipeline && s_r.screenshotFrames.contains(s_r.frame) && s_r.simDumpFrame != s_r.frame)
        {
            EnsureImage(s_r.hdrSimImage, s_r.targetWidth, s_r.targetHeight, RenderFormat::R16G16B16A16_FLOAT, false);
            Transition(cl, s_r.hdrSimImage, RenderTextureLayout::COLOR_WRITE);
            cl.setFramebuffer(GetFramebuffer(&s_r.hdrSimImage, nullptr));
            s_r.markedFramebuffer = nullptr;
            SetViewport(cl, RenderViewport(0.0f, 0.0f, float(s_r.targetWidth), float(s_r.targetHeight)));
            SetScissor(cl, RenderRect(0, 0, int32_t(s_r.targetWidth), int32_t(s_r.targetHeight)));
            SetPipeline(cl, s_r.outputSimPipeline.get());
            BindCommon(cl);
            cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
            cl.drawInstanced(3, 1, 0, 0);
            cl.setFramebuffer(nullptr);
            if (!s_r.simReadback)
                s_r.simReadback = g_video.device->createBuffer(RenderBufferDesc::ReadbackBuffer(uint64_t(s_r.targetWidth) * s_r.targetHeight * 8));
            Transition(cl, s_r.hdrSimImage, RenderTextureLayout::COPY_SOURCE);
            cl.copyTextureRegion(RenderTextureCopyLocation::PlacedFootprint(s_r.simReadback.get(), s_r.hdrSimImage.format,
                                     s_r.hdrSimImage.width, s_r.hdrSimImage.height, 1, s_r.hdrSimImage.width),
                RenderTextureCopyLocation::Subresource(s_r.hdrSimImage.texture.get()));
            s_r.simDumpFrame = s_r.frame;
            // The window's framebuffer bound again for its draw.
            cl.barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(g_video.BackBuffer(), RenderTextureLayout::COLOR_WRITE));
            cl.setFramebuffer(g_video.BackBufferFramebuffer());
            SetViewport(cl, RenderViewport((ww - w) / 2, (wh - h) / 2, w, h));
            SetScissor(cl, RenderRect(0, 0, int32_t(ww), int32_t(wh)));
            SetPipeline(cl, s_r.outputPipeline.get());
            BindCommon(cl);
        }
        // The window's draw's constants, set after the block above (which sets its own): without them the HDR output
        // would draw with an earlier pass's constants.
        cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    }
    else
    {
        SetPipeline(cl, s_r.copyPipeline.get());
        BindCommon(cl);
        const uint32_t push[2] = { s_r.targetDescriptor, 0 };
        cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
    }
    cl.drawInstanced(3, 1, 0, 0);
}

// Frosted glass at the sides of a 16:9 picture on a wide screen (pillarboxing, instead of e.g. the mission
// title card's smoke mirrored past its box). The game's side says
// where the box is (SetPillarFrost: game/widescreen.cpp, the screens whose pictures end at the box). The frame about to
// be presented is copied with its sides replaced: the box's picture continued past its edges (mirrored there, so the
// glass has the colours of what is next to it), reduced to 180 rows, blurred (a Gaussian of AC6_PILLAR_FROST_BLUR % of
// the height, 3) and shaded (x AC6_PILLAR_FROST_DIM, 0.85). The frame itself is not touched. internal_shaders.hlsl
// PILLAR_FROST.
std::atomic<uint64_t> s_pillarFrost{ 0 };

HostImage& PillarFrost(RenderCommandList& cl, HostImage& source)
{
    const uint64_t packed = s_pillarFrost.load(std::memory_order_relaxed);
    const float left = std::bit_cast<float>(uint32_t(packed)), right = std::bit_cast<float>(uint32_t(packed >> 32));
    if (!(right > left) || !s_r.pillarFrostPs || !source.texture || source.depth || source.descriptor >= 65536)
        return source;
    static const float dim = VolSetting("AC6_PILLAR_FROST_DIM", 0.85f), blur = VolSetting("AC6_PILLAR_FROST_BLUR", 3.0f);
    const uint32_t x0 = uint32_t(std::lround(left * float(source.width))), x1 = uint32_t(std::lround(right * float(source.width)));
    if (x0 == 0 || x1 >= source.width || x1 <= x0 + 16)
        return source;
    const uint32_t fh = std::min(180u, source.height);
    const uint32_t fw = std::max(8u, uint32_t(std::lround(double(source.width) * fh / source.height)));
    EnsureImage(s_r.frostA, fw, fh, RenderFormat::R16G16B16A16_FLOAT, false, 1, false, true);
    EnsureImage(s_r.frostB, fw, fh, RenderFormat::R16G16B16A16_FLOAT, false, 1, false, true);
    EnsureImage(s_r.frostFrame, source.width, source.height, source.format, false, 1, false, true);
    const float l = float(x0) / float(source.width), r = float(x1) / float(source.width), sigma = blur * 0.01f * float(fh);
    auto pass = [&](HostImage& from, HostImage& into, uint32_t mode, uint32_t extra) {
        auto& pipeline = s_r.pillarFrostPipelines[uint32_t(into.format)];
        if (!pipeline)
        {
            RenderGraphicsPipelineDesc desc;
            desc.pipelineLayout = s_r.layout.get();
            desc.vertexShader = s_r.copyVs.get();
            desc.pixelShader = s_r.pillarFrostPs.get();
            desc.renderTargetFormat[0] = into.format;
            desc.renderTargetBlend[0] = RenderBlendDesc::Copy();
            desc.renderTargetCount = 1;
            pipeline = g_video.device->createGraphicsPipeline(desc);
        }
        Transition(cl, from, RenderTextureLayout::SHADER_READ);
        Transition(cl, into, RenderTextureLayout::COLOR_WRITE);
        cl.setFramebuffer(GetFramebuffer(&into, nullptr));
        s_r.markedFramebuffer = nullptr;
        SetViewport(cl, RenderViewport(0.0f, 0.0f, float(into.width), float(into.height)));
        SetScissor(cl, RenderRect(0, 0, int32_t(into.width), int32_t(into.height)));
        SetPipeline(cl, pipeline.get());
        BindCommon(cl);
        const uint32_t push[7] = { from.descriptor, mode, std::bit_cast<uint32_t>(l), std::bit_cast<uint32_t>(r), extra,
            std::bit_cast<uint32_t>(dim), std::bit_cast<uint32_t>(sigma) };
        cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
        cl.drawInstanced(3, 1, 0, 0);
    };
    static bool logged = false;
    if (!std::exchange(logged, true))
        log::Info("Frosted pillars: frame {}, a {}x{} frame, the box {} to {}, the glass from {}x{} (sigma {:.1f}, x {})", s_r.frame,
            source.width, source.height, x0, x1, fw, fh, sigma, dim);
    MarkPass([] { return std::string("present: frosted pillars"); });
    pass(source, s_r.frostA, 0, 0);
    pass(s_r.frostA, s_r.frostB, 1, 0);
    Transition(cl, source, RenderTextureLayout::SHADER_READ);
    pass(s_r.frostB, s_r.frostFrame, 2, source.descriptor | (source.format == RenderFormat::B8G8R8A8_UNORM ? 0x10000u : 0u));
    cl.setFramebuffer(nullptr);
    s_r.markedFramebuffer = nullptr;
    Transition(cl, s_r.frostFrame, RenderTextureLayout::SHADER_READ);
    return s_r.frostFrame;
}

void PresentImpl(uint32_t vblanks)
{
    if (s_inDraw.load())
        s_overlaps++;
    s_inPresent++;
    struct Leave { ~Leave() { s_inPresent--; } } leave;
    if (s_r.lateActive)
        EndLatePass(g_video.Commands());
    // A live output change (the settings overlay: HDR on / off, 10-bit SDR): the swap chain built
    // again, and the pipelines that draw into it for its format and the output pass's settings.
    if (s_r.ready && (g_video.ApplyOutputChange() || g_video.BackBufferFormat() != s_r.windowFormat ||
                         OutputConfig().active != s_r.windowOutputActive))
    {
        g_video.WaitGpuIdle();   // (the old pipelines may be in frames in flight)
        CreateWindowPipelines();
    }
    MarkPass([] { return std::string("present"); });
    // AC6_WATCH=<hex guest address>:<hex size> with AC6_WATCH_FRAME=<n>: arm a write watch at frame n.
    static const uint64_t watchFrame = [] { const char* v = std::getenv("AC6_WATCH_FRAME"); return v ? std::stoull(v) : UINT64_MAX; }();
    if (s_r.frame == watchFrame)
        if (const char* v = std::getenv("AC6_WATCH"))
        {
            const std::string text(v);
            const size_t colon = text.find(':');
            if (colon != std::string::npos)
                crash::ArmWriteWatch(uint32_t(std::stoul(text.substr(0, colon), nullptr, 16)), uint32_t(std::stoul(text.substr(colon + 1), nullptr, 16)));
        }
    static std::atomic<uint32_t> reported{ 0 };
    if (s_overlaps.load() && reported.exchange(1) == 0)
        log::Warn("Present and draws overlap on different threads ({} so far)", s_overlaps.load());
    if (s_r.ready)
    {
        RenderCommandList& cl = g_video.Commands();

        // What the console would scan out: this frame's display-sized resolve (the front buffer), or
        // the guest back buffer (render target 0) when the frame didn't resolve one.
        HostImage* source = s_r.frontBufferFrame == s_r.frame ? s_r.frontBuffer : nullptr;
        // AC6_PRESENT=<hex physical address>: show that resolve target instead (debugging passes).
        static const uint32_t presentOverride = [] { const char* v = std::getenv("AC6_PRESENT"); return v ? uint32_t(std::stoul(v, nullptr, 16)) : 0u; }();
        if (presentOverride)
            if (auto it = s_r.resolved.find(presentOverride & 0x1FFFFFFF); it != s_r.resolved.end() && it->second.texture)
                source = &it->second;
        if (!source && s_r.device)
        {
            auto& images = s_r.hdrCanvasFrame == s_r.frame && !s_r.hdrSurfaces.empty() ? s_r.hdrSurfaces : s_r.surfaces;
            if (auto it = images.find(SurfaceOwner(U32(s_r.device + kDevRenderTargets))); it != images.end() && it->second.texture)
                source = &it->second;
            else if (auto plain = s_r.surfaces.find(SurfaceOwner(U32(s_r.device + kDevRenderTargets))); plain != s_r.surfaces.end() && plain->second.texture)
                source = &plain->second;
        }
        // The optional lens effects where no 2D layer drew this frame (an in-engine cutscene has no
        // HUD, a replay's draws don't say which are the HUD's, so there they go over it): on the frame itself.
        if (source && !presentOverride && source->samples == 1 && s_r.lensFxFrame != s_r.frame && s_r.sceneCompositeFrame == s_r.frame)
        {
            s_r.lensFxFrame = s_r.frame;
            if (const LensFx fx = LensFxSettings(); fx.Any())
                RunLensFx(cl, *source, fx);
        }
        // AC6_DUMP_RESOLVES=<frame>: from that frame on, show and capture one resolve target or colour
        // surface per frame (resolve_<frame>_<address>.bmp), to compare passes between machines.
        if (s_r.frame == s_r.dumpFrame)
        {
            // (The planar reflection's mirrored scene and the sea's image first.)
            if (s_r.planarColor.texture)
                s_r.dumpQueue.emplace_back(std::format("planar_{}_scene", s_r.frame), &s_r.planarColor);
            if (s_r.seaReflection.texture)
                s_r.dumpQueue.emplace_back(std::format("planar_{}_sea", s_r.frame), &s_r.seaReflection);
            if (s_r.planarGlow.texture)
                s_r.dumpQueue.emplace_back(std::format("planar_{}_glow", s_r.frame), &s_r.planarGlow);
            for (auto& [address, image] : s_r.resolved)
                if (image.texture)
                    s_r.dumpQueue.emplace_back(std::format("resolve_{}_{:08X}", s_r.frame, address), &image);
            for (auto& [surface, image] : s_r.surfaces)
                if (image.texture && !image.depth)
                    s_r.dumpQueue.emplace_back(std::format("surface_{}_{:08X}", s_r.frame, surface), &image);
            log::Info("Dumping {} resolve targets and surfaces, one per frame", s_r.dumpQueue.size());
        }
        bool dumping = false;
        if (!s_r.dumpQueue.empty())
        {
            source = &Readable(cl, *s_r.dumpQueue.front().second);
            s_r.screenshotName = s_r.dumpQueue.front().first;
            s_r.dumpQueue.erase(s_r.dumpQueue.begin());
            dumping = true;
            // Float targets are also read back raw: the picture can't show alpha, values outside 0..1,
            // NaN or infinity.
            if (uint64_t(source->width) * source->height * 8 <= uint64_t(s_r.targetWidth) * s_r.targetHeight * 16 &&
                (source->format == RenderFormat::R32_FLOAT || source->format == RenderFormat::R16G16B16A16_FLOAT ||
                 source->format == RenderFormat::B8G8R8A8_UNORM))
            {
                Transition(cl, *source, RenderTextureLayout::COPY_SOURCE);
                cl.copyTextureRegion(RenderTextureCopyLocation::PlacedFootprint(s_r.rawReadback.get(), source->format, source->width,
                                         source->height, 1, source->width),
                    RenderTextureCopyLocation::Subresource(source->texture.get()));
                s_r.rawName = s_r.screenshotName;
                s_r.rawWidth = source->width;
                s_r.rawHeight = source->height;
                s_r.rawFormat = source->format;
            }
        }
        // The UI drawn at the frame's size this frame: that image, with the frame's 3D in it, is the
        // frame. Otherwise the frame's target, rendered at AC6_RENDER_SCALE times the output target, is scaled into it.
        float sourceScaleX = 1.0f, sourceScaleY = 1.0f;
        if (!dumping && s_r.uiImageFrame == s_r.frame && s_r.uiImage.texture)
            source = &s_r.uiImage;
        else if (!dumping && source && UiScaled())
        {
            sourceScaleX = float(s_r.targetWidth) / float(s_r.sceneWidth);
            sourceScaleY = float(s_r.targetHeight) / float(s_r.sceneHeight);
        }
        if (source)
            source = &Readable(cl, *source);
        // AC6_GPU_FAULT_TEST: the gamma ramp table's pass into its image, reading the table at an unmapped address.
        if (FaultTestDue() && s_r.gammaLutPipeline)
        {
            MarkPass([] { return std::string("AC6_GPU_FAULT_TEST: a read of an unmapped address"); });
            EnsureImage(s_r.rampImage, 1026, 1, RenderFormat::R16G16B16A16_FLOAT, false, 1, false, true);
            Transition(cl, s_r.rampImage, RenderTextureLayout::COLOR_WRITE);
            cl.setFramebuffer(GetFramebuffer(&s_r.rampImage, nullptr));
            SetViewport(cl, RenderViewport(0.0f, 0.0f, 1026.0f, 1.0f));
            SetScissor(cl, RenderRect(0, 0, 1026, 1));
            SetPipeline(cl, s_r.gammaLutPipeline.get());
            BindCommon(cl);
            const uint32_t push[4] = { 0, 0, 0x1000, 0 };
            cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
            cl.drawInstanced(3, 1, 0, 0);
            cl.setFramebuffer(nullptr);
            Transition(cl, s_r.rampImage, RenderTextureLayout::SHADER_READ);
        }
        // Frosted glass at the sides of a 16:9 picture, where the frame is the whole picture.
        if (source && !dumping && !presentOverride && sourceScaleX == 1.0f && sourceScaleY == 1.0f &&
            source->width == s_r.targetWidth && source->height == s_r.targetHeight)
            source = &PillarFrost(cl, *source);
        // The scanned-out image goes through the display's gamma ramp (not resolve dumps: those show what was rendered).
        // Binds the pipeline and push constants that draw `source` 1:1 into the bound framebuffer.
        auto bindFrame = [&] {
            uint32_t push[4] = { source->descriptor, 0, 0, 0 };
            RenderPipeline* pipeline = s_r.copyPipeline.get();
            if (!dumping && GammaRampEnabled())
            {
                std::lock_guard lock(s_gamma.mutex);
                if (s_gamma.set)
                    if (auto lut = s_r.upload.Allocate(sizeof(s_gamma.lut), 16); lut.memory)
                    {
                        std::memcpy(lut.memory, s_gamma.lut, sizeof(s_gamma.lut));
                        push[2] = uint32_t(lut.deviceAddress);
                        push[3] = uint32_t(lut.deviceAddress >> 32);
                        pipeline = s_r.gammaPipeline.get();
                    }
            }
            SetPipeline(cl, pipeline);
            BindCommon(cl);
            cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
        };
        // The source covers the output target: what a clear would leave shows nowhere.
        const bool covers = source && float(source->width) * sourceScaleX >= float(s_r.targetWidth) - 0.5f &&
            float(source->height) * sourceScaleY >= float(s_r.targetHeight) - 0.5f;
        const bool screenshot = s_r.screenshotFrames.contains(s_r.frame) || dumping || (s_shotRequest.exchange(false) && s_r.readback);
        // A window the output target's size takes the frame straight: one full-frame pass instead
        // of four (clear the target, draw the frame into it, clear the window, copy the target across), the same
        // pixels, as that copy is 1:1. On an integrated GPU each such pass costs a share of its shared memory's
        // bandwidth. Screenshots still go through the target, which they read back.
        // The frame ring reads the target, so it takes the target path too.
        // While the settings overlay is open, frames go through the target too: holding the game re-presents it.
        if (covers && !screenshot && DirectPresent() && !s_r.outputPipeline && RingSeconds() <= 0.0 && !ui::IsOpen() &&
            g_video.BackBufferWidth() == s_r.targetWidth &&
            g_video.BackBufferHeight() == s_r.targetHeight)
        {
            MarkPass([] { return std::string("present: frame into the window (gamma ramp)"); });
            Transition(cl, *source, RenderTextureLayout::SHADER_READ);
            cl.barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(g_video.BackBuffer(), RenderTextureLayout::COLOR_WRITE));
            cl.setFramebuffer(g_video.BackBufferFramebuffer());
            SetViewport(cl, RenderViewport(0.0f, 0.0f, float(source->width) * sourceScaleX, float(source->height) * sourceScaleY));
            SetScissor(cl, RenderRect(0, 0, int32_t(s_r.targetWidth), int32_t(s_r.targetHeight)));
            bindFrame();
            cl.drawInstanced(3, 1, 0, 0);
            DrawOverlay(cl, false);
            cl.setFramebuffer(nullptr);
            MarkPass([] { return std::string("present: end of frame (submission)"); });
        }
        else
        {
            MarkPass([] { return std::string("present: frame into the output target (gamma ramp)"); });
            TransitionTarget(cl, RenderTextureLayout::COLOR_WRITE);
            cl.setFramebuffer(s_r.targetFramebuffer.get());
            if (!covers)
                cl.clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 1.0f));
            if (source)
            {
                Transition(cl, *source, RenderTextureLayout::SHADER_READ);
                TransitionTarget(cl, RenderTextureLayout::COLOR_WRITE);
                cl.setFramebuffer(s_r.targetFramebuffer.get());
                // HDR from the scene: the output pass takes the float frames made below, and the 8-bit target and
                // its frame before the UI are read by no one unless a screenshot, the frame ring or the settings overlay
                // (its held frame) wants them: not drawn then (two full-frame passes a frame). The float frame's
                // image from an earlier frame says the output pass can take it (a descriptor in range). The first frame
                // draws both. AC6_HDR_TARGET_COPY=1: always.
                const bool hdrFrames = !dumping && HdrSceneOn() && s_r.gammaHdrPipeline && s_r.copyHdrPipeline &&
                    source->format == RenderFormat::R16G16B16A16_FLOAT;
                static const bool targetAlways = [] { const char* v = std::getenv("AC6_HDR_TARGET_COPY"); return v && *v == '1'; }();
                // And the output pass takes the float frames as the game drew them where they are the picture at the
                // output's size, applying the gamma ramp itself from the ramp's image (saving the two float copies
                // through the ramp). Not with debanding (its taps would each need the ramp) or the
                // overlay open. AC6_HDR_DIRECT=0: the copies.
                static const bool directOff = [] { const char* v = std::getenv("AC6_HDR_DIRECT"); return v && *v == '0'; }();
                const bool preUiFloat = s_r.preUiFrame == s_r.frame && s_r.preUi.texture && s_r.preUi.format == RenderFormat::R16G16B16A16_FLOAT;
                bool direct = hdrFrames && !directOff && s_r.outputPipeline && (OutputConfig().flags & 5) == 4 && !ui::IsOpen() &&
                    sourceScaleX == 1.0f && sourceScaleY == 1.0f && source->width == s_r.targetWidth && source->height == s_r.targetHeight &&
                    source->descriptor < 65536 &&
                    (!preUiFloat || (s_r.preUi.width == s_r.targetWidth && s_r.preUi.height == s_r.targetHeight && s_r.preUi.descriptor < 65536));
                uint32_t rampWord = 0x80000000u;
                if (direct && GammaRampEnabled())
                {
                    std::lock_guard lock(s_gamma.mutex);
                    if (s_gamma.set)
                    {
                        auto lut = s_r.gammaLutPipeline ? s_r.upload.Allocate(sizeof(s_gamma.lut), 16) : decltype(s_r.upload.Allocate(0, 16)){};
                        if (lut.memory)
                        {
                            std::memcpy(lut.memory, s_gamma.lut, sizeof(s_gamma.lut));
                            EnsureImage(s_r.rampImage, 1026, 1, RenderFormat::R16G16B16A16_FLOAT, false, 1, false, true);
                            Transition(cl, s_r.rampImage, RenderTextureLayout::COLOR_WRITE);
                            cl.setFramebuffer(GetFramebuffer(&s_r.rampImage, nullptr));
                            SetViewport(cl, RenderViewport(0.0f, 0.0f, 1026.0f, 1.0f));
                            SetScissor(cl, RenderRect(0, 0, 1026, 1));
                            SetPipeline(cl, s_r.gammaLutPipeline.get());
                            BindCommon(cl);
                            const uint32_t push[4] = { 0, 0, uint32_t(lut.deviceAddress), uint32_t(lut.deviceAddress >> 32) };
                            cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
                            cl.drawInstanced(3, 1, 0, 0);
                            Transition(cl, s_r.rampImage, RenderTextureLayout::SHADER_READ);
                            TransitionTarget(cl, RenderTextureLayout::COLOR_WRITE);
                            cl.setFramebuffer(s_r.targetFramebuffer.get());
                            rampWord |= s_r.rampImage.descriptor;
                        }
                        if (!lut.memory || s_r.rampImage.descriptor == 0 || s_r.rampImage.descriptor >= 65536)
                            direct = false;
                    }
                }
                if (direct)
                {
                    Transition(cl, *source, RenderTextureLayout::SHADER_READ);
                    if (preUiFloat)
                        Transition(cl, s_r.preUi, RenderTextureLayout::SHADER_READ);
                    s_r.hdrDirectFrame = s_r.frame;
                    s_r.hdrDirectTextures = source->descriptor | (preUiFloat ? s_r.preUi.descriptor << 16 : 0u);
                    s_r.hdrDirectRamp = rampWord;
                    static bool logged = false;
                    if (!std::exchange(logged, true))
                        log::Info("HDR: the output pass takes the float frames as drawn (ramp {})", (rampWord & 0xFFFF) ? "from its image" : "none");
                }
                const bool targetUnread = hdrFrames && !targetAlways && !screenshot && RingSeconds() <= 0.0 && !ui::IsOpen() &&
                    s_r.outputPipeline && (OutputConfig().flags & 4) && (direct || (s_r.hdrFrame.texture && s_r.hdrFrame.descriptor < 65536));
                // The source may be larger than the guest screen (tiling surfaces): map it 1:1 and keep the
                // top-left 1280x720 guest pixels (the scissor).
                SetViewport(cl, RenderViewport(0.0f, 0.0f, float(source->width) * sourceScaleX, float(source->height) * sourceScaleY));
                SetScissor(cl, RenderRect(0, 0, int32_t(s_r.targetWidth), int32_t(s_r.targetHeight)));
                if (!targetUnread)
                {
                    bindFrame();
                    cl.drawInstanced(3, 1, 0, 0);
                }
                // The frame before its UI through the same gamma ramp: the output pass compares it
                // with the target, like with like.
                if (s_r.preUiFrame == s_r.frame && s_r.preUi.texture && !targetUnread)
                {
                    EnsureImage(s_r.preUiOut, s_r.targetWidth, s_r.targetHeight, kTargetFormat, false);
                    Transition(cl, s_r.preUi, RenderTextureLayout::SHADER_READ);
                    Transition(cl, s_r.preUiOut, RenderTextureLayout::COLOR_WRITE);
                    cl.setFramebuffer(GetFramebuffer(&s_r.preUiOut, nullptr));
                    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(s_r.preUi.width), float(s_r.preUi.height)));
                    HostImage* const frame = source;
                    source = &s_r.preUi;
                    bindFrame();
                    cl.drawInstanced(3, 1, 0, 0);
                    source = frame;
                    Transition(cl, s_r.preUiOut, RenderTextureLayout::SHADER_READ);
                    s_r.preUiOutFrame = s_r.frame;
                    TransitionTarget(cl, RenderTextureLayout::COLOR_WRITE);
                    cl.setFramebuffer(s_r.targetFramebuffer.get());
                }
                // HDR from the scene: the frame (and the frame before its UI) also into float images
                // through the ramp continued past 1, for the output pass. The 8-bit target stays the SDR picture
                // (screenshots, the frame ring).
                // (The float copies: without the direct route, and on a screenshot's frame for its statistics.)
                if (hdrFrames && (!direct || screenshot))
                {
                    auto drawHdr = [&](HostImage& from, HostImage& into, float width, float height) {
                        EnsureImage(into, s_r.targetWidth, s_r.targetHeight, RenderFormat::R16G16B16A16_FLOAT, false);
                        Transition(cl, from, RenderTextureLayout::SHADER_READ);
                        Transition(cl, into, RenderTextureLayout::COLOR_WRITE);
                        cl.setFramebuffer(GetFramebuffer(&into, nullptr));
                        SetViewport(cl, RenderViewport(0.0f, 0.0f, width, height));
                        SetScissor(cl, RenderRect(0, 0, int32_t(s_r.targetWidth), int32_t(s_r.targetHeight)));
                        uint32_t push[4] = { from.descriptor, 0, 0, 0 };
                        RenderPipeline* pipeline = s_r.copyHdrPipeline.get();
                        if (GammaRampEnabled())
                        {
                            std::lock_guard lock(s_gamma.mutex);
                            if (s_gamma.set)
                                if (auto lut = s_r.upload.Allocate(sizeof(s_gamma.lut), 16); lut.memory)
                                {
                                    std::memcpy(lut.memory, s_gamma.lut, sizeof(s_gamma.lut));
                                    push[2] = uint32_t(lut.deviceAddress);
                                    push[3] = uint32_t(lut.deviceAddress >> 32);
                                    pipeline = s_r.gammaHdrPipeline.get();
                                }
                        }
                        SetPipeline(cl, pipeline);
                        BindCommon(cl);
                        cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
                        cl.drawInstanced(3, 1, 0, 0);
                        Transition(cl, into, RenderTextureLayout::SHADER_READ);
                    };
                    drawHdr(*source, s_r.hdrFrame, float(source->width) * sourceScaleX, float(source->height) * sourceScaleY);
                    s_r.hdrFrameFrame = s_r.frame;
                    // A screenshot's frame: its float frame's texel statistics in the log too ("Raw hdr_frame_<n>").
                    if (screenshot && s_r.rawReadback && s_r.rawName.empty())
                    {
                        Transition(cl, s_r.hdrFrame, RenderTextureLayout::COPY_SOURCE);
                        cl.copyTextureRegion(RenderTextureCopyLocation::PlacedFootprint(s_r.rawReadback.get(), s_r.hdrFrame.format,
                                                 s_r.hdrFrame.width, s_r.hdrFrame.height, 1, s_r.hdrFrame.width),
                            RenderTextureCopyLocation::Subresource(s_r.hdrFrame.texture.get()));
                        Transition(cl, s_r.hdrFrame, RenderTextureLayout::SHADER_READ);
                        s_r.rawName = std::format("hdr_frame_{}", s_r.frame);
                        s_r.rawWidth = s_r.hdrFrame.width;
                        s_r.rawHeight = s_r.hdrFrame.height;
                        s_r.rawFormat = s_r.hdrFrame.format;
                    }
                    if (s_r.preUiFrame == s_r.frame && s_r.preUi.texture && s_r.preUi.format == RenderFormat::R16G16B16A16_FLOAT)
                    {
                        drawHdr(s_r.preUi, s_r.hdrPreUi, float(s_r.preUi.width), float(s_r.preUi.height));
                        s_r.hdrPreUiFrame = s_r.frame;
                    }
                    TransitionTarget(cl, RenderTextureLayout::COLOR_WRITE);
                    cl.setFramebuffer(s_r.targetFramebuffer.get());
                }
            }

            if (screenshot)
            {
                if (!dumping)
                    s_r.screenshotName = std::format("frame_{}", s_r.frame);
                // The settings overlay, when open, is in the screenshot too (the window can't be read back).
                if (!dumping && s_r.imguiTargetPipeline && TakeOverlay(cl))
                {
                    TransitionTarget(cl, RenderTextureLayout::COLOR_WRITE);
                    cl.setFramebuffer(s_r.targetFramebuffer.get());
                    DrawOverlayInto(cl, s_r.imguiTargetPipeline.get(), float(s_r.targetWidth), float(s_r.targetHeight), 0.0f, false);
                    cl.setFramebuffer(nullptr);
                }
                // With the output pass (SDR), the screenshot shows its debanding and dithering: the target through it.
                RenderTexture* shot = s_r.target.get();
                if (s_r.outputTargetPipeline && !dumping && g_video.OutputMode() != Video::Output::Hdr)
                {
                    EnsureImage(s_r.outputShot, s_r.targetWidth, s_r.targetHeight, kTargetFormat, false);
                    TransitionTarget(cl, RenderTextureLayout::SHADER_READ);
                    Transition(cl, s_r.outputShot, RenderTextureLayout::COLOR_WRITE);
                    cl.setFramebuffer(GetFramebuffer(&s_r.outputShot, nullptr));
                    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(s_r.targetWidth), float(s_r.targetHeight)));
                    SetScissor(cl, RenderRect(0, 0, int32_t(s_r.targetWidth), int32_t(s_r.targetHeight)));
                    SetPipeline(cl, s_r.outputTargetPipeline.get());
                    BindCommon(cl);
                    // AC6_HDR_UI_DEBUG=1: the screenshot shows what the HDR output takes as UI, in red (bit 6).
                    static const bool uiDebug = [] { const char* v = std::getenv("AC6_HDR_UI_DEBUG"); return v && *v == '1'; }();
                    const uint32_t push[4] = { s_r.targetDescriptor | (PreUiDescriptor() << 16), (OutputConfig().flags & ~(4u | 8u)) | (uiDebug ? 64u : 0u) | AllUiFlag(), 0,
                        uint32_t(SeedFrame()) };
                    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
                    cl.drawInstanced(3, 1, 0, 0);
                    cl.setFramebuffer(nullptr);
                    Transition(cl, s_r.outputShot, RenderTextureLayout::COPY_SOURCE);
                    shot = s_r.outputShot.texture.get();
                }
                else
                    TransitionTarget(cl, RenderTextureLayout::COPY_SOURCE);
                cl.copyTextureRegion(RenderTextureCopyLocation::PlacedFootprint(s_r.readback.get(), kTargetFormat, s_r.targetWidth, s_r.targetHeight, 1,
                    s_r.targetWidth),
                    RenderTextureCopyLocation::Subresource(shot));
                s_r.screenshotPending = true;
            }
            // The frame ring: at most 60 frames a second, scaled, read back without waiting.
            if (RingSeconds() > 0.0)
            {
                const auto now = std::chrono::steady_clock::now();
                const uint32_t slot = g_video.slot;
                if (now - s_r.ringLast >= std::chrono::microseconds(16500) && !s_r.ringPending[slot])
                {
                    s_r.ringLast = now;
                    static const uint32_t wantHeight = [] { const char* v = std::getenv("AC6_RING_HEIGHT"); return v && *v ? uint32_t(std::atoi(v)) : 720u; }();
                    const uint32_t h = std::clamp(wantHeight, 90u, s_r.targetHeight);
                    const uint32_t w = std::max(2u, uint32_t(std::lround(double(s_r.targetWidth) * h / s_r.targetHeight)) & ~1u);
                    if (w != s_r.ringWidth || h != s_r.ringHeight)
                    {
                        s_r.ringWidth = w;
                        s_r.ringHeight = h;
                        s_r.ringFrames.clear();
                        s_r.ringSpare.clear();
                        for (auto& buffer : s_r.ringReadback)
                            buffer = g_video.device->createBuffer(RenderBufferDesc::ReadbackBuffer(uint64_t(w) * h * 4));
                        log::Info("Frame ring: the last {:.1f} s at {}x{} (F11 writes them out)", RingSeconds(), w, h);
                    }
                    EnsureImage(s_r.ringImage, w, h, kTargetFormat, false);
                    TransitionTarget(cl, RenderTextureLayout::SHADER_READ);
                    Transition(cl, s_r.ringImage, RenderTextureLayout::COLOR_WRITE);
                    cl.setFramebuffer(GetFramebuffer(&s_r.ringImage, nullptr));
                    SetViewport(cl, RenderViewport(0.0f, 0.0f, float(w), float(h)));
                    SetScissor(cl, RenderRect(0, 0, int32_t(w), int32_t(h)));
                    SetPipeline(cl, s_r.copyPipeline.get());
                    BindCommon(cl);
                    const uint32_t push[2] = { s_r.targetDescriptor, 0 };   // linear, clamped
                    cl.setGraphicsPushConstants(0, push, 0, sizeof(push));
                    cl.drawInstanced(3, 1, 0, 0);
                    cl.setFramebuffer(nullptr);
                    Transition(cl, s_r.ringImage, RenderTextureLayout::COPY_SOURCE);
                    cl.copyTextureRegion(RenderTextureCopyLocation::PlacedFootprint(s_r.ringReadback[slot].get(), kTargetFormat, w, h, 1, w),
                        RenderTextureCopyLocation::Subresource(s_r.ringImage.texture.get()));
                    s_r.ringPending[slot] = true;
                    s_r.ringPendingFrame[slot] = s_r.frame;
                }
            }
            MarkPass([] { return std::string("present: output target to the window"); });
            TargetToWindow(cl);
            s_r.targetHeldFrame = s_r.frame;
            DrawOverlay(cl, true);
            cl.setFramebuffer(nullptr);
            MarkPass([] { return std::string("present: end of frame (submission)"); });
        }
    }
    {
        for (auto& check : s_r.dueChecks)
            check();
        s_r.dueChecks = std::move(s_r.nextFrameChecks);
        s_r.nextFrameChecks.clear();
        {
            if (StatsEnabled())
            {
                thread_local std::set<std::tuple<const uint8_t*, uint32_t, uint32_t, bool, bool>> seen;
                seen.clear();
                for (const auto& copy : s_r.vertexCopies)
                {
                    const uint64_t bytes = uint64_t(copy.vertexCount) * copy.stride;
                    s_stats.vertexCopies++;
                    s_stats.vertexBytes += bytes;
                    if (!seen.insert({ copy.src, copy.vertexCount, copy.stride, copy.rect, copy.point }).second)
                        s_stats.repeatedVertexBytes += bytes;
                }
            }
            ScopedNs vertexTimer(s_stats.vertexNs);
            if (AsyncCopier::Enabled())
                s_asyncCopier.Wait();
            for (const Renderer::EarlyCopy& e : s_r.earlyCopies)
            {
                if (g_memory.Unchanged(e.guest, e.bytes, e.generation))
                {
                    s_stats.earlyKeptBytes += e.bytes;
                    e.source->streak = 0;
                    continue;
                }
                s_r.vertexCopies.push_back(e.copy);
                if (++e.source->streak >= 2)
                    e.source->staticAgainFrame = s_r.frame + 120;
            }
            s_r.earlyCopies.clear();
            // The pipeline ring: what these copies read, unwatched.
            if (capture::RingEnabled())
                for (const auto& copy : s_r.vertexCopies)
                    capture::OnRendererRead(g_memory.MapVirtual(copy.src),
                        uint32_t(std::min<uint64_t>(copy.validBytes, uint64_t(copy.vertexCount) * copy.stride)));
            UpdatePersistentStreams();
            // Every copy writes its own part of the ring: they run on several threads. Work
            // items: each vertex copy, and the whole-stream swaps in 128 KB pieces.
            constexpr uint32_t kPiece = 32u << 10;   // dwords
            static std::vector<std::pair<uint32_t, uint32_t>> pieces;   // (fill, first dword), read by the pool threads
            pieces.clear();
            uint64_t bytes = 0;
            for (uint32_t f = 0; f < s_r.swapFills.size(); f++)
            {
                for (uint32_t at = 0; at < s_r.swapFills[f].dwords; at += kPiece)
                    pieces.push_back({ f, at });
                bytes += uint64_t(s_r.swapFills[f].dwords) * 4;
            }
            s_stats.swapBytes += bytes;
            for (const auto& copy : s_r.vertexCopies)
                bytes += uint64_t(copy.vertexCount) * copy.stride;
            const uint32_t copies = uint32_t(s_r.vertexCopies.size()), items = copies + uint32_t(pieces.size());
            auto work = [&](uint32_t i) {
                if (i < copies)
                {
                    RunVertexCopy(s_r.vertexCopies[i]);
                    return;
                }
                const auto [f, at] = pieces[i - copies];
                const Renderer::SwapFill& fill = s_r.swapFills[f];
                simd::ByteSwap32(fill.dst + at, fill.src + at, std::min(kPiece, fill.dwords - at));
            };
            if (bytes >= (512u << 10))
                s_copyPool.Run(items, work);
            else
                for (uint32_t i = 0; i < items; i++)
                    work(i);
            s_r.vertexCopies.clear();
            s_r.swapFills.clear();
        }
        for (auto& copy : s_r.pendingCopies)
            copy();
        s_r.pendingCopies.clear();
        static const bool poolHazard = [] { const char* v = std::getenv("AC6_POOL_HAZARD"); return v && *v == '1'; }();
        if (poolHazard && g_video.HasFrameCounter())
            for (auto [guest, bytes] : s_r.wholeStreams)
                g_memory.MarkGpuRead(guest, bytes, g_video.FramesIssued() + 1);
        s_r.wholeStreams.clear();
        // The frame's guest data is read: an unpaced present lets the game go on from here (RendererPresent).
        s_dataReleased.fetch_add(1, std::memory_order_release);
        s_dataReleased.notify_all();
        s_r.indexedStreams.clear();
        EvictTextures();
        // The frame's retirees live until its GPU work is done.
        auto& bucket = s_r.inFlight[g_video.slot];
        auto move = [](auto& from, auto& to) {
            for (auto& r : from)
                to.push_back(std::move(r));
            from.clear();
        };
        move(s_r.retiredTextures, bucket.textures);
        move(s_r.retiredStaging, bucket.staging);
        move(s_r.retiredViews, bucket.views);
        move(s_r.retiredFramebuffers, bucket.framebuffers);
        bucket.descriptors.insert(bucket.descriptors.end(), s_r.retiredDescriptors.begin(), s_r.retiredDescriptors.end());
        s_r.retiredDescriptors.clear();
        // Readbacks below need this frame's results: then the GPU is waited for completely.
        const bool readback = !s_r.rawName.empty() || !s_r.resolveCaptures.empty() || !s_r.resolvedSnapshots.empty() ||
            s_r.screenshotPending || !s_r.dueChecks.empty() || !s_r.nextFrameChecks.empty() || s_r.volDump;
        ScopedNs timer(s_stats.presentNs);
        // Events are pumped by the thread that presented (the window's), after the queue drained.
        g_video.Present(vblanks, readback, !g_renderQueue->Threaded());
        // g_video.slot is now the next frame's, and the GPU is done with what it used before.
        auto& done = s_r.inFlight[g_video.slot];
        // The frame ring: this slot's readback is complete.
        if (const uint32_t slot = g_video.slot; s_r.ringPending[slot] && s_r.ringReadback[slot])
        {
            const size_t bytes = size_t(s_r.ringWidth) * s_r.ringHeight * 4;
            std::vector<uint8_t> pixels;
            if (!s_r.ringSpare.empty())
            {
                pixels = std::move(s_r.ringSpare.back());
                s_r.ringSpare.pop_back();
            }
            pixels.resize(bytes);
            const auto* p = static_cast<const uint8_t*>(s_r.ringReadback[slot]->map());
            std::memcpy(pixels.data(), p, bytes);
            s_r.ringReadback[slot]->unmap();
            for (size_t i = 3; i < bytes; i += 4)
                pixels[i] = 255;   // opaque in viewers
            s_r.ringFrames.emplace_back(s_r.ringPendingFrame[slot], std::move(pixels));
            const size_t keep = size_t(std::ceil(RingSeconds() * 60.0));
            while (s_r.ringFrames.size() > keep)
            {
                s_r.ringSpare.push_back(std::move(s_r.ringFrames.front().second));
                s_r.ringFrames.pop_front();
            }
            s_r.ringPending[slot] = false;
        }
        // AC6_RING_DUMP_SECONDS=<s>: one dump that long after start, as F11 would (tests).
        static const double dumpAt = [] { const char* v = std::getenv("AC6_RING_DUMP_SECONDS"); return v && *v ? std::atof(v) : 0.0; }();
        static const auto started = std::chrono::steady_clock::now();
        static bool dumped = false;
        if (dumpAt > 0.0 && !dumped && std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= dumpAt)
        {
            dumped = true;
            RequestRingDump();
            capture::RequestRingCapture();   // and the pipeline ring's, at the same moment
        }
        if (s_r.ringDump.exchange(false))
            DumpRing();
        done.textures.clear();
        done.staging.clear();
        done.views.clear();
        done.framebuffers.clear();
        s_r.freeDescriptors.insert(s_r.freeDescriptors.end(), done.descriptors.begin(), done.descriptors.end());
        done.descriptors.clear();
    }
    LogHitch();
    if (StatsEnabled() && s_r.frame % 120 == 119)
    {
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - s_stats.windowStart).count();
        log::Info("Render stats: {} draw/present overlaps so far, {} texture descriptors ({} free), {} textures ({} MB)", s_overlaps.load(),
            s_r.nextDescriptor, s_r.freeDescriptors.size(), s_r.textures.size(), s_r.textureBytes >> 20);
        // AC6_GPU_PASSES: every 10 stats windows, per frame: time, share, passes, draws, pass.
        if (s_r.frame % 1200 == 1199)
            for (const std::string& line : g_video.TakeGpuPasses(30))
                log::Info("GPU pass: {}", line);
        log::Info("Render stats: {:.1f} fps, {} draws/frame, CPU per frame: draws {:.1f} ms (textures {:.1f} ms, {:.1f} MB hashed, "
            "{:.1f} uploads {:.1f} MB, {:.1f} deferred, {:.1f} staged), present wait {:.1f} ms; GPU per frame {:.2f} ms",
            120 / seconds, s_stats.draws / 120, s_stats.drawNs / 120e6, s_stats.textureNs / 120e6, s_stats.hashedBytes / 120.0 / 1048576,
            s_stats.uploads / 120.0, s_stats.uploadedBytes / 120.0 / 1048576, s_stats.deferredUploads / 120.0, s_stats.stagedUploads / 120.0,
            s_stats.presentNs / 120e6, g_video.TakeGpuFrameMs());
        log::Info("Render stats: vertex copies {:.0f}/frame, {:.2f} MB/frame ({:.2f} MB repeats within the frame), {:.2f} ms/frame; "
            "early copies {:.2f} MB/frame ({:.2f} MB unchanged at submission), stream swaps {:.2f} MB/frame, kept streams drawn {:.2f} MB/frame",
            s_stats.vertexCopies / 120.0, s_stats.vertexBytes / 120.0 / 1048576, s_stats.repeatedVertexBytes / 120.0 / 1048576,
            s_stats.vertexNs / 120e6, s_stats.earlyBytes / 120.0 / 1048576, s_stats.earlyKeptBytes / 120.0 / 1048576,
            s_stats.swapBytes / 120.0 / 1048576, s_stats.cachedBytes / 120.0 / 1048576);
        log::Info("Render stats: stream swaps {:.2f} MB small whole streams, {:.2f} MB large (rewritten) whole streams, {:.2f} MB "
            "kept streams' written pages per frame", s_stats.swapSmallBytes / 120.0 / 1048576, s_stats.swapLargeBytes / 120.0 / 1048576,
            s_stats.swapPageBytes / 120.0 / 1048576);
        log::Info("Render stats: resolve swaps {:.1f} per frame, {:.1f} of them copied back", s_stats.resolveSwaps / 120.0,
            s_stats.resolveRestores / 120.0);
        log::Info("Render stats: constant blocks uploaded {:.0f} VS, {:.0f} PS ({:.2f} MB of registers), {:.0f} shared ({:.0f} shared "
            "reused) per frame", s_stats.constantBlocks[0] / 120.0, s_stats.constantBlocks[1] / 120.0, s_stats.constantBytes / 120.0 / 1048576,
            s_stats.constantBlocks[2] / 120.0, s_stats.constantSharedReused / 120.0);
        {
            uint64_t kept = 0, keptVertex = 0;
            uint32_t count = 0;
            for (const auto& [end, p] : s_r.persistentStreams)
                if (p.buffers[0] && !p.dynamic)
                {
                    count++;
                    kept += p.end - p.start;
                    keptVertex += p.vertex ? p.end - p.start : 0;
                }
            log::Info("Render stats: {} kept streams, {:.1f} MB each frame in flight ({:.1f} MB vertex streams)", count, kept / 1048576.0,
                keptVertex / 1048576.0);
        }
        if (auto* vd = dynamic_cast<plume::VulkanDevice*>(g_video.device.get()); vd && vd->pipelineLibrariesSupported)
        {
            std::lock_guard lock(vd->pipelineLibraryMutex);
            log::Info("Render stats: {} pipeline libraries built so far, {} pipelines", vd->pipelineLibrariesBuilt, s_r.pipelines.size());
        }
        if (const std::string list = s_pipelineList.Stats(); !list.empty())
            log::Info("Render stats: {}, {} pipelines in all", list, s_r.pipelines.size());
        if (const auto w = g_memory.TakeGpuWriteStats(); w.early || w.late)
            log::Info("Render stats: writes to whole-uploaded streams {} before the GPU finished the frame reading them, {} after "
                "(GPU {} frames behind)", w.early, w.late, g_video.FramesIssued() - g_video.FramesCompleted());
        s_stats = Stats{};
    }
    if (s_r.volDump)   // AC6_VOL_DUMP
    {
        const auto* p = static_cast<const uint8_t*>(s_r.volDump->map());
        std::ofstream(std::filesystem::path(log::Directory()) / std::format("vol_{}.bin", s_r.volDumpFrame), std::ios::binary)
            .write(reinterpret_cast<const char*>(p), std::streamsize(uint64_t(s_r.volDumpCount) * kVolRecord));
        s_r.volDump->unmap();
        s_r.volDump.reset();
    }
    if (s_r.simReadback && s_r.simDumpFrame != ~0ull)
    {
        // AC6_HDR_SIM: the HDR output, scRGB (1 = 80 nits), width, height, format as uint32 then the rows.
        const auto* p = static_cast<const uint8_t*>(s_r.simReadback->map());
        std::ofstream file(std::filesystem::path(log::Directory()) / std::format("raw_hdr_out_{}.bin", s_r.simDumpFrame), std::ios::binary);
        const uint32_t header[3] = { s_r.hdrSimImage.width, s_r.hdrSimImage.height, uint32_t(s_r.hdrSimImage.format) };
        file.write(reinterpret_cast<const char*>(header), sizeof(header));
        file.write(reinterpret_cast<const char*>(p), std::streamsize(uint64_t(s_r.hdrSimImage.width) * s_r.hdrSimImage.height * 8));
        s_r.simReadback->unmap();
        log::Info("HDR output dumped: raw_hdr_out_{}.bin ({}x{})", s_r.simDumpFrame, s_r.hdrSimImage.width, s_r.hdrSimImage.height);
        s_r.simDumpFrame = ~0ull;
    }
    if (!s_r.rawName.empty())
    {
        // Present waited for the GPU: the raw copy is complete.
        const auto* p = static_cast<const uint8_t*>(s_r.rawReadback->map());
        log::Info("Raw {} ({}x{}):{}", s_r.rawName, s_r.rawWidth, s_r.rawHeight, TexelStats(p, s_r.rawFormat, s_r.rawWidth, s_r.rawHeight));
        // AC6_RAW_DUMP=1: also the texels, as raw_<name>.bin (width, height, format as uint32, then the rows, packed).
        static const bool dump = [] { const char* v = std::getenv("AC6_RAW_DUMP"); return v && *v == '1'; }();
        if (dump)
        {
            std::ofstream file(std::filesystem::path(log::Directory()) / std::format("raw_{}.bin", s_r.rawName), std::ios::binary);
            const uint32_t header[3] = { s_r.rawWidth, s_r.rawHeight, uint32_t(s_r.rawFormat) };
            file.write(reinterpret_cast<const char*>(header), sizeof(header));
            file.write(reinterpret_cast<const char*>(p), std::streamsize(uint64_t(s_r.rawWidth) * s_r.rawHeight * 8));
        }
        s_r.rawReadback->unmap();
        s_r.rawName.clear();
    }
    for (auto& capture : s_r.resolveCaptures)
    {
        const auto* p = static_cast<const uint8_t*>(capture.buffer->map());
        log::Info("Resolve capture {}.{}: {} ({}x{}):{}", s_r.frame, capture.index, capture.name, capture.width, capture.height,
            TexelStats(p, capture.format, capture.width, capture.height));
        // AC6_RESOLVE_STATS_BMP=1: also the images, as resolve_<frame>_<index>.bmp.
        static const bool bmp = [] { const char* v = std::getenv("AC6_RESOLVE_STATS_BMP"); return v && *v == '1'; }();
        if (bmp)
            WriteTexelsBmp(std::filesystem::path(log::Directory()) / std::format("resolve_{}_{:02}.bmp", s_r.frame, capture.index), p,
                capture.format, capture.width, capture.height);
        capture.buffer->unmap();
    }
    s_r.resolveCaptures.clear();
    // The snapshots as a capture's blob (RestoreResolvedImpl's layout).
    auto blobOf = [](std::vector<Renderer::ResolvedSnapshot>& snapshots) {
        std::vector<uint8_t> blob;
        auto put = [&blob](uint32_t v) { blob.insert(blob.end(), reinterpret_cast<uint8_t*>(&v), reinterpret_cast<uint8_t*>(&v) + 4); };
        put(uint32_t(snapshots.size()));
        for (auto& snap : snapshots)
        {
            put(snap.kind);
            put(snap.address);
            put(snap.width);
            put(snap.height);
            put(uint32_t(snap.format));
            const auto* p = static_cast<const uint8_t*>(snap.buffer->map());
            blob.insert(blob.end(), p, p + uint64_t(snap.width) * snap.height * SnapshotBytesPerTexel(snap.format));
            snap.buffer->unmap();
        }
        snapshots.clear();
        return blob;
    };
    if (!s_r.resolvedSnapshots.empty())
        capture::OnResolvedImages(blobOf(s_r.resolvedSnapshots));
    // The pipeline ring's, of the frame that last used the slot now starting: its GPU work is done.
    if (auto& ring = s_r.ringSnapshots[g_video.slot]; !ring.empty())
        capture::OnRingResolvedImages(s_r.ringSnapshotFrame[g_video.slot], blobOf(ring));
    capture::OnPresent();
    if (s_r.screenshotPending)
    {
        WriteScreenshot(s_r.screenshotName);
        s_r.screenshotPending = false;
    }
    s_r.upload.StartFrame(g_video.slot);   // that part of the ring is free: Present waited for its last frame
    s_r.resolveIndex = 0;
    // The texture browser's list, dumps asked for, and the mods read again when the settings changed (every texture
    // checked again if replacements came or went: gpu/texture_mods.h).
    mods::PublishFrame(std::move(s_r.frameTextures));
    s_r.frameTextures.clear();
    ServiceDumpRequests();
    if (mods::RefreshIndex())
        for (auto& [key, t] : s_r.textures)
        {
            t.contentHash = 0;
            t.watchGeneration = 0;
        }
    s_r.frame++;
    capture::OnFrameStart(s_r.frame);
    crash::NoteFrame();
}

// --- Renderer calls queued to the worker -----------------------------------------------

namespace {

// The parts of the guest device the calls read, copied with each call into the worker's device image: the
// fetch constants (textures, vertex streams), the register shadow's render state, and the binding slots
// (index buffer, render targets, depth, shaders). The float constants travel as the 64-byte runs their dirty
// masks mark (a few a draw). Objects the slots point to (surfaces, index buffers, textures) and the data the
// GPU reads are read by the worker, as the console GPU reads them after the call.
struct DeviceRegion { uint32_t begin, end; };
constexpr DeviceRegion kSnapshotRegions[] = { { 0x480, 0x780 }, { 0x28C0, 0x2A70 }, { 0x3088, 0x3198 } };
constexpr uint32_t kSnapshotBytes = (0x780 - 0x480) + (0x2A70 - 0x28C0) + (0x3198 - 0x3088);
constexpr uint32_t kConstantRun = 64;   // bytes per dirty bit: four float4 registers

void TakeSnapshot(uint8_t* out, uint32_t device)
{
    for (const auto& r : kSnapshotRegions)
    {
        std::memcpy(out, g_memory.As<uint8_t>(device + r.begin), r.end - r.begin);
        out += r.end - r.begin;
    }
}

// Applies a snapshot to the device image. Returns the image's guest address, the device the call then reads.
uint32_t ApplySnapshot(const uint8_t* in)
{
    uint8_t* image = g_memory.As<uint8_t>(s_deviceImage);
    for (const auto& r : kSnapshotRegions)
    {
        std::memcpy(image + r.begin, in, r.end - r.begin);
        in += r.end - r.begin;
    }
    return s_deviceImage;
}

// The constant runs a mask marks (bit 63 = registers 0-3), from `from` to `to`.
uint8_t* CopyConstantRuns(uint8_t* to, const uint8_t* from, uint64_t mask)
{
    for (; mask; mask &= mask - 1)
    {
        const uint32_t run = 63 - uint32_t(std::countr_zero(mask));
        std::memcpy(to, from + run * kConstantRun, kConstantRun);
        to += kConstantRun;
    }
    return to;
}

const uint8_t* ApplyConstantRuns(uint8_t* to, const uint8_t* from, uint64_t mask)
{
    for (; mask; mask &= mask - 1)
    {
        const uint32_t run = 63 - uint32_t(std::countr_zero(mask));
        std::memcpy(to + run * kConstantRun, from, kConstantRun);
        from += kConstantRun;
    }
    return from;
}

// The issuing side's constant masks since the last queued draw (NoteDrawDirty), and the frame of its last
// full copy: every frame starts with one, so a run a mask missed would be stale for a frame at most.
thread_local uint64_t t_pendingVs = ~0ull, t_pendingPs = ~0ull;
thread_local bool t_pendingNoted = false;
thread_local uint64_t t_fullCopyFrame = UINT64_MAX;

struct DrawCall
{
    uint32_t primType, start, count;
    int32_t baseVertex;
    bool indexed, noted;
    int8_t cockpitGroup;   // the cockpit view's part group of the request issuing it, or -1
    bool ui;               // issued by the game's 2D layers
    HostShader* vs;
    HostShader* ps;
    uint64_t vsMask, psMask;
    // Followed by the device snapshot and the constant runs (VS, then PS).
};

void RunDraw(void* payload)
{
    const auto& c = *static_cast<const DrawCall*>(payload);
    const uint8_t* data = reinterpret_cast<const uint8_t*>(&c + 1);
    const uint32_t device = ApplySnapshot(data);
    data += kSnapshotBytes;
    uint8_t* image = g_memory.As<uint8_t>(device);
    data = ApplyConstantRuns(image + kDevVsConstants, data, c.vsMask);
    ApplyConstantRuns(image + kDevPsConstants, data, c.psMask);
    t_vsDirty |= c.vsMask;
    t_psDirty |= c.psMask;
    t_dirtyNoted = c.noted;
    // Outputs not 16:9: shader constants the game works out in 16:9 screen units, scaled
    // for that draw along the axis the screen extends (x on a wide output, hor+, y on a narrow one, vert+) by the smaller
    // scale over that axis's (1 / k wide, k narrow: k = output aspect / 16:9), so the shapes come out round and keep their
    // size along the other axis:
    //  - the sun lens flare (VS 2F771DF29242C99C): its quads' sizes in clip space, c105 (x, y, x2, y2),
    //  - the depth of field's 13-tap gathers: the horizontal one's (VS 9EBCAF1EAAEC959C) UV offsets c101.xyzw and c102.xy,
    //    the vertical one's (VS 2337072EA2F30C33) c103.xyzw and c102.zw (a guest texel is k times wider than tall on screen),
    //  - the bloom's star streaks (PS 9825C7C374A8B834, 8 taps along a streak): the taps' UV offsets c50-c57 (x, y), so
    //    the streaks keep their angles and lengths,
    //  - the heat haze (PS 5A9903B0E66580C3): its offset's scale to UV, c95 (x, y),
    //  - the sun (an ellipse k times wide on a 21:9 output): its disc's quad (VS D66D6627B10F1081,
    //    centre c100.xy, half sizes c101.xy in clip space) and its glow's (VS DFC1E74274A5F143, half sizes c101.zw),
    //  - the moon (mission 14: stretched on a wide output): its disc's quad (VS
    //    B061B43670950F03, vsMapMoon: the moon's direction through the view-projection c100-c103, then the corner,
    //    turned by vMoonRadiusRotZ's z, x its half sizes in clip space, c105.xy). Its glow is the sun's glow shader.
    struct WideConstant { bool pixel; uint64_t hash; uint32_t reg, count, maskX, maskY; };
    static constexpr WideConstant kWideConstants[] = {
        { false, 0x2F771DF29242C99Cull, 105, 1, 0b0101, 0b1010 },
        { false, 0x9EBCAF1EAAEC959Cull, 101, 1, 0b1111, 0 },
        { false, 0x9EBCAF1EAAEC959Cull, 102, 1, 0b0011, 0 },
        { false, 0x2337072EA2F30C33ull, 103, 1, 0, 0b1111 },
        { false, 0x2337072EA2F30C33ull, 102, 1, 0, 0b1100 },
        { true, 0x9825C7C374A8B834ull, 50, 8, 0b0001, 0b0010 },
        { true, 0x5A9903B0E66580C3ull, 95, 1, 0b0001, 0b0010 },
        { false, 0xD66D6627B10F1081ull, 101, 1, 0b0001, 0b0010 },
        { false, 0xDFC1E74274A5F143ull, 101, 1, 0b0100, 0b1000 },
        { false, 0xB061B43670950F03ull, 105, 1, 0b0001, 0b0010 },
    };
    const float minScale = std::min(s_r.resX, s_r.resY), wideX = minScale / s_r.resX, wideY = minScale / s_r.resY;
    constexpr uint32_t kMaxWideRegs = 8;
    float wideSaved[std::size(kWideConstants)][kMaxWideRegs][4] = {};
    bool wideHit[std::size(kWideConstants)] = {};
    bool wide = false;
    auto wideRegs = [&](const WideConstant& w) {
        return reinterpret_cast<be<float>*>(image + (w.pixel ? kDevPsConstants : kDevVsConstants) + w.reg * 16);
    };
    auto markWide = [&](const WideConstant& w) {
        for (uint32_t i = 0; i < w.count; i++)
            (w.pixel ? t_psDirty : t_vsDirty) |= 1ull << (63 - (w.reg + i) / 4);
    };
    if (std::fabs(s_r.resX - s_r.resY) > minScale * 1e-4f)
        for (uint32_t w = 0; w < std::size(kWideConstants); w++)
        {
            const WideConstant& entry = kWideConstants[w];
            const HostShader* shader = entry.pixel ? c.ps : c.vs;
            if (!shader || shader->hash != entry.hash)
                continue;
            be<float>* r = wideRegs(entry);
            for (uint32_t i = 0; i < entry.count; i++)
                for (uint32_t x = 0; x < 4; x++)
                    if ((entry.maskX | entry.maskY) & (1u << x))
                    {
                        wideSaved[w][i][x] = r[i * 4 + x];
                        r[i * 4 + x] = wideSaved[w][i][x] * ((entry.maskX & (1u << x)) ? wideX : wideY);
                    }
            markWide(entry);
            wideHit[w] = wide = true;
        }
    // The enhanced bloom: the composite drawn without its bloom and star (c100.zw at 0, kept for the
    // passes after it: Draw). The game's bright pass's threshold and offset (c104.yz) noted as it draws.
    be<float>* const psRegs = reinterpret_cast<be<float>*>(image + kDevPsConstants);
    float bloomSaved[2] = {};
    bool bloomZeroed = false;
    if (c.ps && (c.ps->idHash & kOptionFree) == kBloomBrightPassPs)
    {
        s_r.bloomBright[0] = psRegs[104 * 4 + 1];
        s_r.bloomBright[1] = psRegs[104 * 4 + 2];
    }
    if (c.ps && (c.ps->idHash & kOptionFree) == kBloomCompositePs && BloomEnhanced() && s_r.bloomApplyPs)
    {
        bloomSaved[0] = psRegs[100 * 4 + 2];
        bloomSaved[1] = psRegs[100 * 4 + 3];
        psRegs[100 * 4 + 2] = 0.0f;
        psRegs[100 * 4 + 3] = 0.0f;
        t_psDirty |= 1ull << (63 - 100 / 4);
        s_r.bloomScales[0] = bloomSaved[0];
        s_r.bloomScales[1] = bloomSaved[1];
        s_r.bloomScalesFrame = s_r.frame;
        bloomZeroed = true;
    }
    t_cockpitGroup = c.cockpitGroup;
    t_uiDraw = c.ui;
    if (c.indexed)
        DrawIndexedImpl(device, c.primType, c.baseVertex, c.start, c.count, c.vs, c.ps);
    else
        DrawImpl(device, c.primType, c.start, c.count, c.vs, c.ps);
    t_cockpitGroup = -1;
    if (bloomZeroed)
    {
        psRegs[100 * 4 + 2] = bloomSaved[0];
        psRegs[100 * 4 + 3] = bloomSaved[1];
        t_psDirty |= 1ull << (63 - 100 / 4);
    }
    if (wide)
        for (uint32_t w = 0; w < std::size(kWideConstants); w++)
            if (wideHit[w])
            {
                const WideConstant& entry = kWideConstants[w];
                be<float>* r = wideRegs(entry);
                for (uint32_t i = 0; i < entry.count; i++)
                    for (uint32_t x = 0; x < 4; x++)
                        if ((entry.maskX | entry.maskY) & (1u << x))
                            r[i * 4 + x] = wideSaved[w][i][x];
                markWide(entry);
            }
}

void QueueDraw(uint32_t device, bool indexed, uint32_t primType, int32_t baseVertex, uint32_t start, uint32_t count)
{
    if (!s_r.ready || count == 0)
        return;
    // The shaders as they are now: D3D patches a vertex shader's microcode in place for each draw's declaration.
    HostShader* vs = GetShader(U32(device + kDevVertexShader), false);
    const uint32_t psObject = U32(device + kDevPixelShader);
    HostShader* ps = psObject ? GetShader(psObject, true) : &s_r.nullPixelShader;
    uint64_t vsMask = t_pendingVs, psMask = t_pendingPs;
    if (const uint64_t frame = RendererFrame(); !t_pendingNoted || t_fullCopyFrame != frame)
    {
        vsMask = psMask = ~0ull;
        t_fullCopyFrame = frame;
    }
    const uint32_t runs = uint32_t(std::popcount(vsMask) + std::popcount(psMask));
    auto* c = static_cast<DrawCall*>(g_renderQueue->Begin(uint32_t(sizeof(DrawCall) + kSnapshotBytes + runs * kConstantRun), RunDraw));
    *c = { primType, start, count, baseVertex, indexed, t_pendingNoted, int8_t(game::CockpitDrawGroup()), game::InUiLayers(), vs, ps, vsMask, psMask };
    uint8_t* data = reinterpret_cast<uint8_t*>(c + 1);
    TakeSnapshot(data, device);
    data += kSnapshotBytes;
    data = CopyConstantRuns(data, g_memory.As<uint8_t>(device + kDevVsConstants), vsMask);
    CopyConstantRuns(data, g_memory.As<uint8_t>(device + kDevPsConstants), psMask);
    g_renderQueue->Commit();
    t_pendingVs = t_pendingPs = 0;
    t_pendingNoted = false;
}

struct ClearCall
{
    uint32_t flags, stencil;
    float color[4], z;
};

struct ResolveCall
{
    uint32_t flags, destTexture, destLevel, clearStencil;
    float clearZ, clearColor[4];
    bool hasRect, hasPoint, hasColor;
    ResolveRect rect;
    ResolvePoint point;
};

struct TilingCall
{
    uint32_t count;
    bool hasRects;
    ResolveRect rects[16];
};

} // namespace

void NoteDrawDirty(uint32_t device)
{
    t_pendingVs |= *g_memory.As<be<uint64_t>>(device);
    t_pendingPs |= *g_memory.As<be<uint64_t>>(device + 8);
    t_pendingNoted = true;
}

void RendererDraw(uint32_t device, uint32_t primType, uint32_t startVertex, uint32_t vertexCount)
{
    capture::OnDraw(device, primType, startVertex, vertexCount);
    QueueDraw(device, false, primType, 0, startVertex, vertexCount);
}

void RendererDrawIndexed(uint32_t device, uint32_t primType, int32_t baseVertex, uint32_t startIndex, uint32_t indexCount)
{
    capture::OnDrawIndexed(device, primType, baseVertex, startIndex, indexCount);
    QueueDraw(device, true, primType, baseVertex, startIndex, indexCount);
}

// D3D's inline quad draw (sub_821DEED0): `count` vertices of 20 bytes at `src` as a quad list.
// The console's draw core copies them into the command ring and points vertex fetch constant 95 (stream 0 here) at
// that copy with fetch constants written into the ring, not into the device's fetch constant file. The renderer
// reads the file, so the vertices go to a ring of the runtime's own guest memory (kept for the frame's deferred
// vertex copies and for captures) and stream 0 points there for this draw only, as the console's dirty bit restores
// the file's constant for the next draw. The shadow-map filter passes draw this way (the hangar's floor shadows).
void RendererDrawQuadsUP(uint32_t device, uint32_t count, uint32_t src)
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_QUADS_UP"); return !(v && *v == '0'); }();
    constexpr uint32_t kStride = 20, kRingBytes = 0x40000;
    // The first draw from each vertex source (the two callers each draw from their own table).
    static std::set<uint32_t> sources;
    if (sources.size() < 8 && sources.insert(src).second)
        log::Info("Inline quad draws: first from {:08X}, {} vertices (frame {})", src, count, s_r.frame);
    if (!enabled || count == 0 || count * kStride > kRingBytes)
        return;
    static uint32_t ring = 0, head = 0;
    if (!ring)
    {
        void* host = g_heap.AllocPhysical(kRingBytes, 4096);
        if (!host)
        {
            static bool warned = false;
            if (!std::exchange(warned, true))
                log::Warn("Inline quad draws: no memory for the vertex ring");
            return;
        }
        ring = g_memory.MapVirtual(host);
        log::Info("Inline quad draws: vertex ring at {:08X}", ring);
    }
    const uint32_t bytes = count * kStride;
    if (head + bytes > kRingBytes)
        head = 0;
    const uint32_t at = ring + head;
    head = (head + bytes + 15) & ~15u;
    std::memcpy(g_memory.Translate(at), g_memory.Translate(src), bytes);
    auto* stream0 = g_memory.As<be<uint32_t>>(device + kDevVertexFetch + 17 * 8);
    const uint32_t saved[2] = { stream0[0], stream0[1] };
    // As the draw core builds them: the physical address with type 3 (vertex), then endian 8in32 and the size in dwords.
    stream0[0] = (at & 0x1FFFFFFF) | 3u;
    stream0[1] = 0x10000002u | ((bytes / 4) << 2 & 0x03FFFFFCu);
    RendererDraw(device, xenos::kPrimQuadList, 0, count);
    stream0[0] = saved[0];
    stream0[1] = saved[1];
}

// Without a Vulkan device (headless: no renderer, no queue) the calls below only reach the capture
// and count the frames.
void RendererClear(uint32_t device, uint32_t flags, const float color[4], float z, uint32_t stencil)
{
    capture::OnClear(device, flags, color, z, stencil);
    if (!g_renderQueue)
        return;
    auto* c = static_cast<ClearCall*>(g_renderQueue->Begin(sizeof(ClearCall) + kSnapshotBytes, [](void* p) {
        const auto& c = *static_cast<const ClearCall*>(p);
        ClearImpl(ApplySnapshot(reinterpret_cast<const uint8_t*>(&c + 1)), c.flags, c.color, c.z, c.stencil);
    }));
    *c = { flags, stencil, { color[0], color[1], color[2], color[3] }, z };
    TakeSnapshot(reinterpret_cast<uint8_t*>(c + 1), device);
    g_renderQueue->Commit();
}

void RendererResolve(uint32_t device, uint32_t flags, const ResolveRect* srcRect, uint32_t destTexture, const ResolvePoint* destPoint,
    uint32_t destLevel, const float* clearColor, float clearZ, uint32_t clearStencil)
{
    capture::OnResolve(device, flags, srcRect, destTexture, destPoint, destLevel, clearColor, clearZ, clearStencil);
    if (!g_renderQueue)
        return;
    auto* c = static_cast<ResolveCall*>(g_renderQueue->Begin(sizeof(ResolveCall) + kSnapshotBytes, [](void* p) {
        const auto& c = *static_cast<const ResolveCall*>(p);
        ResolveImpl(ApplySnapshot(reinterpret_cast<const uint8_t*>(&c + 1)), c.flags, c.hasRect ? &c.rect : nullptr, c.destTexture,
            c.hasPoint ? &c.point : nullptr, c.destLevel, c.hasColor ? c.clearColor : nullptr, c.clearZ, c.clearStencil);
    }));
    *c = ResolveCall{ flags, destTexture, destLevel, clearStencil, clearZ, {}, srcRect != nullptr, destPoint != nullptr, clearColor != nullptr,
        srcRect ? *srcRect : ResolveRect{}, destPoint ? *destPoint : ResolvePoint{} };
    if (clearColor)
        std::memcpy(c->clearColor, clearColor, sizeof(c->clearColor));
    TakeSnapshot(reinterpret_cast<uint8_t*>(c + 1), device);
    g_renderQueue->Commit();
}

void RendererBeginTiling(const ResolveRect* rects, uint32_t count)
{
    capture::OnBeginTiling(rects, count);
    if (!g_renderQueue)
        return;
    auto* c = static_cast<TilingCall*>(g_renderQueue->Begin(sizeof(TilingCall), [](void* p) {
        const auto& c = *static_cast<const TilingCall*>(p);
        BeginTilingImpl(c.hasRects ? c.rects : nullptr, c.count);
    }));
    c->count = count;
    c->hasRects = rects != nullptr;
    for (uint32_t i = 0; rects && i < count && i < 16; i++)
        c->rects[i] = rects[i];
    g_renderQueue->Commit();
}

void RendererEndTiling()
{
    capture::OnEndTiling();
    if (!g_renderQueue)
        return;
    g_renderQueue->Begin(0, [](void*) { EndTilingImpl(); });
    g_renderQueue->Commit();
}

// The frame's end, queued. A paced frame waits until it is presented, so the present paces the game, and so does every
// frame during a capture (its recording starts and stops at presents). An unpaced frame lets the game go on once its
// guest data (vertices, textures) is read: the worker then submits it while the game issues the next frame's calls
// (queued behind). Events are pumped here, on the window's thread. AC6_PRESENT_RELEASE=0 waits for every present.
void RendererPresent(uint32_t vblanks)
{
    const uint64_t frame = s_issuedFrames.load(std::memory_order_relaxed);
    if (!g_renderQueue)
    {
        s_issuedFrames.store(frame + 1, std::memory_order_release);
        g_video.PumpEvents();
        return;
    }
    *static_cast<uint32_t*>(g_renderQueue->Begin(sizeof(uint32_t), [](void* p) { PresentImpl(*static_cast<const uint32_t*>(p)); })) = vblanks;
    g_renderQueue->Commit();
    static const bool release = [] { const char* v = std::getenv("AC6_PRESENT_RELEASE"); return !v || *v != '0'; }();
    if (release && g_renderQueue->Threaded() && !(g_video.framePacing && vblanks) && !capture::Active())
    {
        for (uint64_t released; (released = s_dataReleased.load(std::memory_order_acquire)) <= frame;)
            s_dataReleased.wait(released, std::memory_order_acquire);
    }
    else
        g_renderQueue->WaitIdle();
    s_issuedFrames.store(frame + 1, std::memory_order_release);
    if (g_renderQueue->Threaded())
        g_video.PumpEvents();
}

// The settings overlay's hold (RendererHoldForOverlay): the last frame, which is in the output target, again, with the
// overlay over it.
void RepresentImpl()
{
    if (s_r.ready)
    {
        RenderCommandList& cl = g_video.Commands();
        if (s_r.targetHeldFrame + 1 == s_r.frame)
        {
            MarkPass([] { return std::string("present: the held frame again (settings overlay)"); });
            // As that frame was presented: the output pass's UI mask and dither seed are the frame's.
            const uint64_t frame = s_r.frame;
            s_r.frame = s_r.targetHeldFrame;
            TargetToWindow(cl);
            s_r.frame = frame;
        }
        else
        {
            cl.barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(g_video.BackBuffer(), RenderTextureLayout::COLOR_WRITE));
            cl.setFramebuffer(g_video.BackBufferFramebuffer());
            cl.clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 1.0f));
        }
        DrawOverlay(cl, s_r.targetHeldFrame + 1 == s_r.frame);
        cl.setFramebuffer(nullptr);
        auto& bucket = s_r.inFlight[g_video.slot];
        for (auto& t : s_r.retiredTextures)
            bucket.textures.push_back(std::move(t));
        s_r.retiredTextures.clear();
    }
    g_video.Present(1, false, !g_renderQueue->Threaded());
    if (s_r.ready)
        s_r.inFlight[g_video.slot].textures.clear();   // (Present waited for the slot's previous frame)
    // The upload ring moves on with the slot, as after a frame: the overlay's vertices go there (without this the ring
    // would fill up while held and the overlay's draws would be dropped).
    s_r.upload.StartFrame(g_video.slot);
}

uint64_t RendererHoldForOverlay()
{
    // AC6_UI_PAUSE=1: the game is held under the overlay (by default it keeps running).
    static const bool hold = [] { const char* v = std::getenv("AC6_UI_PAUSE"); return v && *v == '1'; }();
    if (!hold || !g_renderQueue || !ui::IsOpen())
        return 0;
    g_renderQueue->WaitIdle();
    // A frame that went straight to the window isn't in the target: the next one will be (PresentImpl).
    if (s_r.ready && s_r.targetHeldFrame + 1 != s_r.frame)
        return 0;
    using Clock = std::chrono::steady_clock;
    const uint64_t tsc0 = __rdtsc();
    const auto t0 = Clock::now();
    constexpr auto kStep = std::chrono::nanoseconds(1'000'000'000 / 60);
    auto next = t0;
    uint64_t presents = 0;
    while (ui::IsOpen())
    {
        *static_cast<uint32_t*>(g_renderQueue->Begin(sizeof(uint32_t), [](void*) { RepresentImpl(); })) = 0;
        g_renderQueue->Commit();
        g_renderQueue->WaitIdle();
        if (g_renderQueue->Threaded())
            g_video.PumpEvents();
        presents++;
        // At most 60 a second (an unpaced, unsynced window would spin).
        next += kStep;
        if (const auto now = Clock::now(); now > next + kStep)
            next = now;
        std::this_thread::sleep_until(next);
    }
    s_heldForOverlay = true;
    log::Info("Settings overlay: the game was held {:.1f} s ({} presents)", std::chrono::duration<double>(Clock::now() - t0).count(), presents);
    return __rdtsc() - tsc0;
}

void RendererWaitIdle()
{
    if (g_renderQueue && !g_renderQueue->OnWorker())
        g_renderQueue->WaitIdle();
}

void RendererSnapshotResolved(uint64_t maxTexels)
{
    RendererWaitIdle();
    SnapshotResolvedImpl(maxTexels);
}

void RendererRestoreResolved(const uint8_t* data, size_t size)
{
    RendererWaitIdle();
    RestoreResolvedImpl(data, size);
}

void RendererProbeDepth(uint32_t device, const char* label)
{
    RendererWaitIdle();
    ProbeDepthImpl(device, label);
}

uint64_t RendererFrame() { return s_issuedFrames.load(std::memory_order_acquire); }

void SetPillarFrost(float left, float right)
{
    s_pillarFrost.store(right > left ? uint64_t(std::bit_cast<uint32_t>(left)) | (uint64_t(std::bit_cast<uint32_t>(right)) << 32) : 0,
        std::memory_order_relaxed);
}

bool HdrSceneActive() { return HdrSceneOn(); }
bool SceneFloatActive() { return SceneFloatOn(); }

void RendererSetSeedFrame(uint64_t liveFrame)
{
    RendererWaitIdle();
    s_frameSeedBase = liveFrame - s_r.frame;
}

} // namespace ac6::gpu
