#pragma once
// The display's vertical-blank interrupt. The Xbox D3D library registers a graphics interrupt
// callback (VdSetGraphicsInterruptCallback). On VBlank it bumps the device's VBlank counter, flips
// queued swaps and calls the game's VBlank callback, which paces the game's frames.
//
// VBlanks come from a 60 Hz host timer on a runtime-owned guest thread (faster with AC6_VBLANK_HZ, 1 kHz through the
// boot logos with AC6_FAST_BOOT).

#include <cstdint>

namespace ac6::gpu {

// Registers the kernel-level graphics interrupt callback. The first registration starts VBlanks.
void SetInterruptCallback(uint32_t callback, uint32_t data);

uint64_t VBlankCount();

} // namespace ac6::gpu
