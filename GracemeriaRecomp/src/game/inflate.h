#pragma once
// Raw deflate (RFC 1951) decoding: the game's data packages after their XOR (game/packages.cpp).
#include <cstddef>
#include <cstdint>

namespace ac6::game {

// `in` (raw deflate, no zlib header) into `out`. Returns the bytes made, or SIZE_MAX for a damaged stream or one that
// makes more than `outSize` bytes.
size_t Inflate(const uint8_t* in, size_t inSize, uint8_t* out, size_t outSize);

} // namespace ac6::game
