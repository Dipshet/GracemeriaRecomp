#pragma once
// The XAudio render driver (apu/audio_driver.cpp).

#include <cstdint>

namespace ac6::apu {

constexpr uint32_t kAudioSampleRate = 48000;

// Samples per channel the game's mixer has rendered so far: the soundtrack's own timeline, which keeps
// moving in real time while the render thread stalls.
uint64_t RenderedSamples();

} // namespace ac6::apu
