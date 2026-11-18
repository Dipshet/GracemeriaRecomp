#include "simd.h"

#include <cstdlib>
#include <cstring>
#include <immintrin.h>

namespace ac6::simd {

namespace {

// Shuffle masks: reverse the bytes of each 32-bit / 16-bit lane.
alignas(32) constexpr uint8_t kSwap32[32] = {
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12 };
alignas(32) constexpr uint8_t kIdentity[32] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
alignas(32) constexpr uint8_t kSwap16[32] = {
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14 };

// Swaps `bytes` (a multiple of the lane size) with the shuffle `mask`. Returns the bytes done.
#if defined(__AVX2__)
size_t SwapAvx2(uint8_t* dst, const uint8_t* src, size_t bytes, const uint8_t* mask)
{
    const __m256i m = _mm256_load_si256(reinterpret_cast<const __m256i*>(mask));
    size_t i = 0;
    for (; i + 128 <= bytes; i += 128)
    {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 32));
        const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 64));
        const __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 96));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_shuffle_epi8(a, m));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i + 32), _mm256_shuffle_epi8(b, m));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i + 64), _mm256_shuffle_epi8(c, m));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i + 96), _mm256_shuffle_epi8(d, m));
    }
    for (; i + 32 <= bytes; i += 32)
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
            _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i)), m));
    return i;
}
#endif

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("ssse3")))
size_t SwapSsse3(uint8_t* dst, const uint8_t* src, size_t bytes, const uint8_t* mask)
{
    const __m128i m = _mm_load_si128(reinterpret_cast<const __m128i*>(mask));
    size_t i = 0;
    for (; i + 16 <= bytes; i += 16)
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
            _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i)), m));
    return i;
}
#endif

// No AVX-512 path: it measured no faster than AVX2 (the swaps are bound by memory).
enum class Path { Scalar, Ssse3, Avx2 };

Path Detect()
{
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__)
    return Path::Avx2;
#else
    __builtin_cpu_init();
    if (__builtin_cpu_supports("ssse3"))
        return Path::Ssse3;
#endif
#endif
    return Path::Scalar;
}

const Path s_path = Detect();

// Vector part of a swap. The caller finishes the tail.
size_t SwapVector(uint8_t* dst, const uint8_t* src, size_t bytes, const uint8_t* mask)
{
    switch (s_path)
    {
#if defined(__x86_64__) || defined(_M_X64)
    case Path::Ssse3:
        return SwapSsse3(dst, src, bytes, mask);
#endif
#if defined(__AVX2__)
    case Path::Avx2:
        return SwapAvx2(dst, src, bytes, mask);
#endif
    default:
        return 0;
    }
}

} // namespace

void ByteSwap32(uint32_t* dst, const uint32_t* src, size_t count)
{
    size_t i = SwapVector(reinterpret_cast<uint8_t*>(dst), reinterpret_cast<const uint8_t*>(src), count * 4, kSwap32) / 4;
    for (; i < count; i++)
        dst[i] = __builtin_bswap32(src[i]);
}

void ByteSwap16(uint16_t* dst, const uint16_t* src, size_t count)
{
    size_t i = SwapVector(reinterpret_cast<uint8_t*>(dst), reinterpret_cast<const uint8_t*>(src), count * 2, kSwap16) / 2;
    for (; i < count; i++)
        dst[i] = __builtin_bswap16(src[i]);
}

void CopyToGpu(void* dst, const void* src, size_t bytes)
{
    // The shuffle with the identity mask costs nothing next to the stores. It keeps one code path.
    const size_t done = SwapVector(static_cast<uint8_t*>(dst), static_cast<const uint8_t*>(src), bytes & ~size_t(3), kIdentity);
    std::memcpy(static_cast<uint8_t*>(dst) + done, static_cast<const uint8_t*>(src) + done, bytes - done);
}

const char* PathName()
{
    switch (s_path)
    {
    case Path::Avx2: return "AVX2";
    case Path::Ssse3: return "SSSE3";
    default: return "scalar";
    }
}

} // namespace ac6::simd
