#pragma once
// The host renderer. Draws are translated at the guest D3D API level: the
// guest D3D library keeps maintaining its device state, and at each draw the renderer reads what it
// needs (shaders, constants, fetch constants, vertex declaration, render targets, render states)
// straight from the guest device and records the equivalent Vulkan work through plume.
//
// Render targets are native: one host image per guest D3DSurface (no EDRAM), depth included. A resolve
// copies the source surface into a host image that stands in for the destination texture. Textures
// pointing at that address sample it directly. Predicated tiling draws the frame once at full size.
// Present shows the frame's display-sized resolve (the front buffer), scaled to the window.

#include <cstddef>
#include <cstdint>

namespace ac6::gpu {

// Host-endian copies of the guest D3DRECT / D3DPOINT arguments.
struct ResolveRect { int32_t x1, y1, x2, y2; };
struct ResolvePoint { int32_t x, y; };

bool InitRenderer();

void RendererClear(uint32_t device, uint32_t flags, const float color[4], float z, uint32_t stencil);
void RendererDraw(uint32_t device, uint32_t primType, uint32_t startVertex, uint32_t vertexCount);
// The device's VS/PS constant dirty masks (device +0 / +8) as the draw core about to run found them, before it
// flushes and clears them: the next RendererDraw* reuses the constants of the draw before when a mask is clear.
void NoteDrawDirty(uint32_t device);
void RendererDrawIndexed(uint32_t device, uint32_t primType, int32_t baseVertex, uint32_t startIndex, uint32_t indexCount);
// D3D's inline quad-list draw (sub_821DEED0): `count` 20-byte vertices copied from guest `src`.
void RendererDrawQuadsUP(uint32_t device, uint32_t count, uint32_t src);
void RendererResolve(uint32_t device, uint32_t flags, const ResolveRect* srcRect, uint32_t destTexture, const ResolvePoint* destPoint,
    uint32_t destLevel, const float* clearColor, float clearZ, uint32_t clearStencil);
void RendererBeginTiling(const ResolveRect* rects, uint32_t count);
void RendererEndTiling();
// vblanks: the game's present interval in 60 Hz vblanks (Video::Present paces to it), 0 = unpaced.
void RendererPresent(uint32_t vblanks = 0);
// After a present: while the settings overlay is open, holds the game (the caller's thread) and shows the last frame with
// the overlay over it. Returns the time held in timebase (TSC) ticks, 0 when it didn't hold.
uint64_t RendererHoldForOverlay();
// The frame ring (AC6_RING=<seconds>, only with AC6_CAPTURES=1): F11 writes the frames it holds (the last seconds, at
// 60 a second) as BMPs into a ring_<date>_<time> folder next to the log.
void RequestRingDump();
// Builds the shader's pipeline library ahead of time where plume links pipelines from per-shader libraries.
// False when it doesn't (the shader module is then left for the first draw).
struct HostShader;
bool RendererPrepareShaderLibrary(HostShader& shader);
// Whether the sea reflects the scene (the planar reflection or AC6_SSR, both read at start): the map's pixel shaders
// carry the reflection's code only then.
bool RendererSeaReflects();
// The pipeline list (pipelines known from earlier runs, built ahead of their draws). RendererPipelineListWants: whether
// any of them is made of the shader with this hash (its module is then made as soon as it is translated).
// RendererShaderReady: the shader has its module (a listed pipeline is built once both its shaders have one).
bool RendererPipelineListWants(uint64_t shaderHash);
void RendererShaderReady(const HostShader& shader);

void RendererSetGammaTable(const uint16_t ramp[3][256]);
void RendererSetGammaPwl(const uint16_t ramp[3][128][2]);
// Capture and replay of the ramp, as 1024 packed entries, RendererGammaRamp returns false if none was set.
bool RendererGammaRamp(uint32_t lut[1024]);
void RendererRestoreGammaRamp(const uint32_t lut[1024]);
// Frame capture / replay (gpu/capture.h): the resolve targets carried from frame to frame.
void RendererSnapshotResolved(uint64_t maxTexels = UINT64_MAX);   // only images up to that size (the pipeline ring)
void RendererProbeDepth(uint32_t device, const char* label);
void RendererRestoreResolved(const uint8_t* data, size_t size);
// Replay: the frame starting now is the live game's frame `liveFrame` (the per-frame noise is seeded alike).
void RendererSetSeedFrame(uint64_t liveFrame);
// Waits until the queued renderer calls have run: where the game waits for the GPU.
void RendererWaitIdle();
// Frames presented so far (diagnostics).
uint64_t RendererFrame();
// Frosted glass at the sides of the frames presented from now on: outside the box from `left` to
// `right` (fractions of the frame's width) the frame is replaced by the box's picture continued, blurred and shaded.
// left >= right: off. From the game's thread (game/widescreen.cpp).
void SetPillarFrost(float left, float right);
// HDR from the game's scene: with an HDR output (or AC6_HDR_SCENE=force), the
// frame-sized canvas keeps values above 1 (a float image) and the game's grade pass doesn't clip (gpu/shaders.cpp).
bool HdrSceneActive();
// The game's lighting in float with HDR from the scene: AC6_HDR_SCENE_FLOAT.
bool SceneFloatActive();

} // namespace ac6::gpu
