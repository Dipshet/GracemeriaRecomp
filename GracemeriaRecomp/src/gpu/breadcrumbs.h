#pragma once

#include <cstdint>
#include <source_location>

namespace plume {
struct RenderCommandList;
struct RenderDevice;
}

// GPU crash breadcrumbs: when the driver loses the device (a GPU fault or hang), the log says how far the GPU got.
// Each pass the renderer marks (MarkPass, a change of render target) leaves a numbered breadcrumb: the GPU writes the
// number into a small buffer the CPU can read, and the CPU keeps where in the code it was made, the frame, the draws
// made after it and their last shaders. With VK_AMD_buffer_marker (NVIDIA, AMD, Mesa) the GPU writes two: one when it
// reaches the breadcrumb, one when the work before it has finished. Without (Intel's Windows driver) one transfer fill,
// made where it is outside a render pass and at the end of the render pass otherwise. A few bytes and one GPU write a
// pass: on in every build. AC6_GPU_BREADCRUMBS=0: off; =fill: the fills even where markers exist (for testing).
//
// With the markers a watch also reads them (10 times a second): when submitted work hasn't finished and the GPU's
// breadcrumbs haven't moved for a second (AC6_GPU_WATCH_MS, 0: no watch), the GPU is stuck, and the log says where at
// once, onto the disk. A hang the driver can't recover from can freeze the whole machine before a lost device is ever
// reported; Windows waits 2 seconds before resetting the GPU.
namespace ac6::gpu::breadcrumbs {

void Init(plume::RenderDevice& device);
// A breadcrumb in `cl` (the renderer's command list, frame `frame`), at the caller's place in the code. width / height:
// the pass's target, where it has one.
void Mark(plume::RenderCommandList& cl, uint64_t frame, uint32_t width = 0, uint32_t height = 0,
    std::source_location where = std::source_location::current());
// A draw after the last breadcrumb, with its guest shaders' hashes.
void NoteDraw(uint64_t vs, uint64_t ps);
// The breadcrumbs made so far (at a frame's end: that frame's and the ones before), and that they were submitted.
uint32_t Made();
void Submitted(uint32_t made);
// After a lost device: the breadcrumbs around where the GPU stopped, and the driver's own fault report where it has one,
// to the log.
void Report(plume::RenderDevice* device);

} // namespace ac6::gpu::breadcrumbs
