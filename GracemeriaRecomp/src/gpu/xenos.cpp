#include "gpu/xenos.h"

#include <cstring>

#include <bit>
#include <utility>

namespace ac6::gpu::xenos {

namespace {

// Second byte of each entry of the game's per-format table at 0x82067E48 (bits per pixel).
constexpr uint8_t kBitsPerPixel[64] = {
    1, 1, 8, 16, 16, 16, 32, 32, 8, 8, 16, 16, 16, 32, 32, 16,
    32, 32, 4, 8, 8, 64, 32, 32, 16, 32, 64, 16, 32, 64, 16, 32,
    64, 32, 64, 128, 32, 64, 128, 8, 16, 16, 32, 8, 8, 16, 16, 16,
    32, 8, 32, 4, 8, 8, 32, 32, 32, 96, 4, 4, 4, 4, 32, 32,
};

} // namespace

uint32_t BitsPerPixel(uint32_t format) { return kBitsPerPixel[format & 0x3F]; }

uint32_t BlockWidth(uint32_t format)
{
    // From 0x821DF758: 4x4 blocks for the DXT/DXN/CTX family, 2x1 / 4x1 for packed formats.
    switch (format & 0x3F)
    {
    case 18: case 19: case 20: case 49: case 51: case 52: case 53:
    case 58: case 59: case 60: case 61:
        return 4;
    case 11: case 12: case 40:
        return 2;
    case 39: case 44:
        return 4;
    default:
        return 1;
    }
}

uint32_t BlockHeight(uint32_t format)
{
    switch (format & 0x3F)
    {
    case 18: case 19: case 20: case 49: case 51: case 52: case 53:
    case 58: case 59: case 60: case 61:
        return 4;
    default:
        return 1;
    }
}

void SwapEndian(uint8_t* data, size_t size, uint32_t mode)
{
    // A word at a time (the loops vectorise). Of a trailing partial word only a whole 8-in-16 pair is swapped, the rest
    // is left as it is.
    const size_t words = size / 4;
    auto each = [&](auto f) {
        for (size_t i = 0; i < words; i++)
        {
            uint32_t w;
            std::memcpy(&w, data + i * 4, 4);
            w = f(w);
            std::memcpy(data + i * 4, &w, 4);
        }
    };
    switch (mode)
    {
    case 1:   // 8 in 16
        each([](uint32_t w) { return ((w & 0x00FF00FFu) << 8) | ((w >> 8) & 0x00FF00FFu); });
        if (size % 4 >= 2)
            std::swap(data[words * 4], data[words * 4 + 1]);
        break;
    case 2:   // 8 in 32
        each([](uint32_t w) { return __builtin_bswap32(w); });
        break;
    case 3:   // 16 in 32
        each([](uint32_t w) { return (w << 16) | (w >> 16); });
        break;
    default:
        break;
    }
}

} // namespace ac6::gpu::xenos
