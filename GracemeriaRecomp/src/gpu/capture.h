#pragma once
// Frame capture and replay (renderer development): AC6_CAPTURE=<frame>[:<frames>] records every renderer
// call of those frames with the guest state it reads, then a snapshot of guest memory, to
// capture_<frame>.ac6cap next to the log. `ac6.exe --replay <file> [--repeat N]` restores the snapshot
// and re-issues the calls against the renderer without running the game, so renderer changes can be
// checked in seconds. Diagnostics keyed by frame (AC6_DRAW_TRACE, AC6_SCREENSHOT, ...) count replayed
// frames from 0.
#include <cstdint>
#include <vector>

namespace ac6::gpu {

struct ResolveRect;
struct ResolvePoint;

namespace capture {

// Recording (no-ops unless a capture is in progress). Called at the top of the renderer entry points.
bool Recording();
// Recording, or about to (a capture requested or set by AC6_CAPTURE, or the pipeline ring on): the renderer then keeps
// its calls in step with the game.
bool Active();
void OnClear(uint32_t device, uint32_t flags, const float color[4], float z, uint32_t stencil);
void OnDraw(uint32_t device, uint32_t primType, uint32_t startVertex, uint32_t vertexCount);
void OnDrawIndexed(uint32_t device, uint32_t primType, int32_t baseVertex, uint32_t startIndex, uint32_t indexCount);
void OnResolve(uint32_t device, uint32_t flags, const ResolveRect* srcRect, uint32_t destTexture, const ResolvePoint* destPoint,
    uint32_t destLevel, const float* clearColor, float clearZ, uint32_t clearStencil);
void OnBeginTiling(const ResolveRect* rects, uint32_t count);
void OnEndTiling();
void OnVertexShaderVariant(uint32_t object, uint32_t variant);
void OnVertexShaderLoad(uint32_t object, uint32_t offset, const std::vector<uint8_t>& bytes);
void OnVertexShaderPatchedInPlace(uint32_t object, uint32_t variant);
// At the end of RendererPresent (records the present, writes the file after the last frame) and after
// the frame counter advanced (starts a requested capture).
void OnPresent();
// The resolve targets as the captured frames started (read back by the renderer at the first present).
void OnResolvedImages(std::vector<uint8_t> blob);
// The pipeline ring's small resolve targets as `frame` started (read back a few frames later).
void OnRingResolvedImages(uint64_t frame, std::vector<uint8_t> blob);
void OnFrameStart(uint64_t frame);

// F12 in the game window: capture the next `frames` frames (a player can grab a frame showing a bug).
void RequestCapture(uint32_t frames);

// The pipeline ring (AC6_PIPELINE_RING=<seconds>, only with AC6_CAPTURES=1): every frame recorded into memory, the
// last seconds kept. F12 writes them as a capture (pipeline_<frame>.ac6cap). Recording() is then always true.
bool RingEnabled();
void RequestRingCapture();
// Guest data the renderer reads without watching it (vertex copies at submission, textures hashed every frame):
// recorded with the frame (the ring only, watched data is recorded as its pages start being watched).
void OnRendererRead(uint32_t guest, uint32_t size);

// Replays a capture file. Returns the process exit code.
int Replay(const char* path, uint32_t repeat);

} // namespace capture
} // namespace ac6::gpu
