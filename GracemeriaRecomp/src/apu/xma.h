#pragma once
// XMA decoder hardware: the contexts the game's audio engine programs and kicks.

#include <cstdint>

namespace ac6::apu {

// Allocates the context array and publishes its physical address in the XMA register block (register
// 0x600 at 0x7FEA1800, read once by the game's audio engine). Call after the heap and the GPU register
// range exist.
void XmaInit();

// XMACreateContext / XMAReleaseContext: one 64-byte context from the array (guest virtual address), 0 when
// all are taken.
uint32_t XmaAllocateContext();
void XmaReleaseContext(uint32_t context);

// How many voices of several contexts (multichannel streams: music, a cutscene's 5.1 soundtrack) have started
// so far, and the rendered sample count (RenderedSamples) at the latest start.
struct MultichannelStart
{
    uint32_t count;
    uint64_t samples;
};
MultichannelStart LastMultichannelStart();

} // namespace ac6::apu
