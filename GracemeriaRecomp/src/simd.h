#pragma once
// Bulk byte-order conversion for guest data: the guest is big-endian, so vertex
// streams, shader constants, index buffers and textures are swapped on their way to the host GPU. These
// run 32 bytes at a time with AVX2 (SSSE3 in a build without it). A plain
// loop compiles to one 4-byte movbe per word (the compiler can't prove the buffers apart).

#include <cstddef>
#include <cstdint>

namespace ac6::simd {

// dst[i] = byteswap(src[i]). The ranges must not overlap, no alignment needed.
void ByteSwap32(uint32_t* dst, const uint32_t* src, size_t count);
void ByteSwap16(uint16_t* dst, const uint16_t* src, size_t count);

// A plain copy with the same wide sequential stores, for write-combined GPU memory (the C runtime's
// large-copy path is several times slower there).
void CopyToGpu(void* dst, const void* src, size_t bytes);

// The vector path in use ("AVX2", "SSSE3" or "scalar"), for the log.
const char* PathName();

} // namespace ac6::simd
