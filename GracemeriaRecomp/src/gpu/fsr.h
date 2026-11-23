#pragma once
// AMD FSR 3.1 (the FidelityFX SDK v1.1.4's upscaler, MIT, thirdparty/FidelityFX-SDK) in place of the renderer's temporal
// AA: AMD's Vulkan backend, recording its compute passes into the frame's command buffer.

#include <plume_render_interface.h>

#include <cstdint>

namespace ac6::gpu {

class DeferredCommandList;

namespace fsr {

// AC6_AA=fsr.
bool Requested();

// The render size over the output's with FSR upscaling: AC6_FSR_QUALITY = native (1, the default),
// quality (1 / 1.5), balanced (1 / 1.7), performance (1 / 2), ultra (1 / 3), or AC6_FSR_SCALE=<0.33..1>.
float RenderRatio();
// FSR at the end of the 3D frame (the whole 3D frame at the render size), the default. AC6_FSR_LATE=0: right after the
// world pass instead (only the world's surfaces at the render size).
bool Late();

// FSR's sub-pixel jitter for a frame (render pixels, -0.5..0.5): its Halton sequence, as many phases as the upscale needs.
void Jitter(uint64_t frame, int32_t renderWidth, int32_t outputWidth, float& x, float& y);

struct Image
{
    const plume::RenderTexture* texture = nullptr;
    plume::RenderFormat format = plume::RenderFormat::UNKNOWN;
    uint32_t width = 0, height = 0;
};

struct Frame
{
    Image color;      // this frame's scene image (render size), in the shader-read layout
    Image depth;      // its depth (R32 float, reversed, infinite far plane), shader-read
    Image motion;     // motion vectors (render size): from this frame's unjittered position to last frame's, shader-read
    Image reactive;   // the reactive mask (render size, R8: 0 none), shader-read, or none
    Image output;     // the result (upscale size), a storage image in the general layout
    // With both: FSR makes the reactive mask itself (into autoReactive) from the scene before its translucent layers
    // (opaqueOnly, render size, shader-read) and `color`, and `reactive` is taken as the transparency and composition mask.
    Image opaqueOnly;
    Image autoReactive;   // render size, R8, a storage image in the general layout
    float jitterX = 0.0f, jitterY = 0.0f;              // the scene draws' jitter, render pixels
    float motionScaleX = 1.0f, motionScaleY = 1.0f;    // motion vectors to render pixels
    float frameTimeMs = 16.7f;
    float cameraNear = 1.0f, fovY = 1.0f;              // the near plane's distance, the vertical field of view (radians)
    float sharpness = 0.0f;                            // RCAS, 0 off
    bool reset = false;                                // no usable history
    bool debugView = false;                            // FSR's debug view in the output
};

// Records FSR's passes for `frame` (outside a render pass), through `deferred` when the frame's commands are recorded
// there (DeferredCommandList::custom), else into `cl` directly. Makes or remakes the context as the sizes need. False
// when FSR can't run (the context failed). The caller falls back.
bool Dispatch(plume::RenderDevice& device, plume::RenderCommandList& cl, DeferredCommandList* deferred, const Frame& frame);
// FSR couldn't start (its backend, context or images): the scene is scaled up as it is, with no jitter.
bool Failed();

} // namespace fsr
} // namespace ac6::gpu
